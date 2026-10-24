"""Forecast-quality and trading-performance statistics."""

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """HAC standard error of the mean of x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    d = x - x.mean()
    s = d @ d / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return np.sqrt(max(s, 0) / n)


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, horizon: int = 1, dates=None) -> tuple:
    """DM test on squared errors. Negative stat = model has lower loss. Returns (stat, two-sided p).

    With `dates`, the loss differential is averaged per date first: stocks on the same day are strongly
    correlated, so treating them as independent observations would overstate significance.
    """
    d = np.asarray(e_model) ** 2 - np.asarray(e_bench) ** 2
    if dates is not None:
        d = pd.Series(d).groupby(np.asarray(dates)).mean().to_numpy()
    se = newey_west_se(d, max(horizon - 1, 0) + 5)
    stat = d.mean() / se if se > 0 else 0.0
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    n_pos, n_neg = y_true.sum(), (~y_true).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(score)
    return (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ic_series(df: pd.DataFrame, col: str = "pred_ret") -> pd.Series:
    """Daily cross-sectional Spearman correlation between prediction and realised return."""
    def one(g):
        if len(g) < 3 or g[col].nunique() < 2:
            return np.nan
        return stats.spearmanr(g[col], g["y"]).statistic
    return df.groupby("date")[[col, "y"]].apply(one).dropna()


def prediction_metrics(df: pd.DataFrame, horizon: int = 1) -> dict:
    """df has columns date, ticker, pred_ret, prob_up, y (realised forward log return)."""
    df = df.dropna(subset=["pred_ret", "y"])
    y, p, prob = df["y"].to_numpy(), df["pred_ret"].to_numpy(), df["prob_up"].to_numpy()
    up = y > 0
    ic = ic_series(df)
    ts_corr = df.groupby("ticker")[["pred_ret", "y"]].apply(
        lambda g: g["pred_ret"].corr(g["y"]) if g["pred_ret"].std() > 0 else np.nan)
    dm, dm_p = diebold_mariano(y - p, y, horizon, df["date"].to_numpy())
    prob_c = np.clip(prob, 1e-6, 1 - 1e-6)
    return {
        "n": len(df),
        "mse_e4": np.mean((y - p) ** 2) * 1e4,
        "r2_oos": 1 - np.sum((y - p) ** 2) / np.sum(y ** 2),     # vs predicting zero
        "hit_rate": np.mean(np.sign(p) == np.sign(y)) if np.any(p != 0) else np.nan,
        "auc": auc(up, prob),
        "brier": np.mean((prob - up) ** 2),
        "logloss": -np.mean(up * np.log(prob_c) + (~up) * np.log(1 - prob_c)),
        "ic_mean": ic.mean(),
        "ic_t": ic.mean() / newey_west_se(ic.to_numpy(), max(5, horizon)) if len(ic) > 10 else np.nan,
        "ts_corr": ts_corr.mean(),
        "dm_vs_zero": dm,
        "dm_p": dm_p,
    }


# ---- trading performance -------------------------------------------------------------------

def max_drawdown(equity) -> float:
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def sharpe(r: np.ndarray, ann: int = ANN) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return r.mean() / sd * np.sqrt(ann) if sd > 0 else 0.0


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0, ann: int = ANN) -> float:
    """P(true Sharpe > benchmark) accounting for sample length, skew and kurtosis (Bailey & López de Prado)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 30 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    sr_b = sr_benchmark / np.sqrt(ann)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def bootstrap_sharpe_ci(r: np.ndarray, n_boot: int = 1000, block: int = 20, seed: int = 0, ann: int = ANN) -> tuple:
    """95% CI of the annualised Sharpe ratio via a moving-block bootstrap."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 2 * block:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        out[b] = sharpe(sample, ann)
    return tuple(np.percentile(out, [2.5, 97.5]))


def performance(r: pd.Series, turnover: pd.Series | None = None, costs: pd.Series | None = None,
                exposure: pd.Series | None = None, ann: int = ANN) -> dict:
    """Annualised statistics of a per-period return series with `ann` periods per year."""
    r = r.dropna()
    x = r.to_numpy()
    eq = np.cumprod(1 + x)
    years = len(x) / ann
    downside = x[x < 0].std(ddof=1) if (x < 0).sum() > 1 else np.nan
    lo, hi = bootstrap_sharpe_ci(x, ann=ann)
    mdd = max_drawdown(np.concatenate([[1.0], eq]))
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 else np.nan
    out = {
        "cagr": cagr,
        "ann_vol": x.std(ddof=1) * np.sqrt(ann),
        "sharpe": sharpe(x, ann),
        "sharpe_lo95": lo,
        "sharpe_hi95": hi,
        "psr": probabilistic_sharpe(x, ann=ann),
        "sortino": x.mean() / downside * np.sqrt(ann) if downside and downside > 0 else np.nan,
        "max_dd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "worst_day": x.min(),
        "hit_days": np.mean(x[x != 0] > 0) if (x != 0).any() else np.nan,
    }
    if turnover is not None:
        out["turnover_ann"] = turnover.mean() * ann
    if costs is not None:
        out["cost_drag_ann"] = costs.mean() * ann
    if exposure is not None:
        out["avg_gross"] = exposure.mean()
    return out
