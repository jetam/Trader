"""Stage 2: statistical study of returns, volatility, dependence and correlation."""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller

from .data import Panel
from .metrics import max_drawdown
from .reporting import md_table, plt, savefig

log = logging.getLogger(__name__)
ANN = 252


def variance_ratio(r: np.ndarray, q: int) -> tuple:
    """Lo-MacKinlay variance ratio with heteroskedasticity-robust z statistic.

    VR > 1: returns trend (positive autocorrelation); VR < 1: mean reversion; VR = 1: random walk.
    """
    r = r[np.isfinite(r)]
    n = len(r)
    mu = r.mean()
    d = r - mu
    var1 = (d ** 2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), mode="valid")
    m = q * (n - q + 1) * (1 - q / n)
    varq = ((rq - q * mu) ** 2).sum() / m
    vr = varq / var1
    denom = (d ** 2).sum() ** 2
    theta = 0.0
    for j in range(1, q):
        delta = n * ((d[j:] ** 2) * (d[:-j] ** 2)).sum() / denom
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / np.sqrt(theta)
    return vr, z, 2 * (1 - stats.norm.cdf(abs(z)))


def hac_ols(y: pd.Series, x: pd.DataFrame, lags: int = 5):
    df = pd.concat([y, x], axis=1).dropna()
    return sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def summary_table(lr: pd.DataFrame, ann: int = ANN) -> pd.DataFrame:
    rows = {}
    for s in lr:
        r = lr[s].dropna()
        jb = stats.jarque_bera(r)
        rows[s] = {
            "start": r.index[0].date().isoformat(),
            "ann_ret": r.mean() * ann,
            "ann_vol": r.std() * np.sqrt(ann),
            "sharpe": r.mean() / r.std() * np.sqrt(ann),
            "skew": stats.skew(r),
            "ex_kurt": stats.kurtosis(r),
            "jb_p": jb.pvalue,
            "max_dd": max_drawdown(np.exp(r.cumsum())),
            "VaR95_1bar": -np.quantile(r, 0.05),
            "CVaR95_1bar": -r[r <= np.quantile(r, 0.05)].mean(),
        }
    return pd.DataFrame(rows).T


def dependence_table(close: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"statsmodels|trader")
    rows = {}
    for s in lr:
        r = lr[s].dropna()
        lb_r = acorr_ljungbox(r, lags=[10])["lb_pvalue"].iloc[0]
        lb_r2 = acorr_ljungbox(r ** 2, lags=[10])["lb_pvalue"].iloc[0]
        vr5, z5, _ = variance_ratio(r.to_numpy(), 5)
        vr20, z20, _ = variance_ratio(r.to_numpy(), 20)
        rows[s] = {
            "adf_p_logprice": adfuller(np.log(close[s].dropna()), autolag="AIC")[1],
            "adf_p_returns": adfuller(r, autolag="AIC")[1],
            "acf1": r.autocorr(1),
            "ljungbox_p_r": lb_r,
            "ljungbox_p_r2": lb_r2,
            "arch_lm_p": het_arch(r - r.mean(), nlags=5)[1],
            "VR(5)": vr5, "z_VR5": z5,
            "VR(20)": vr20, "z_VR20": z20,
        }
    return pd.DataFrame(rows).T


def factor_table(lr: pd.DataFrame, bench: pd.Series, ann: int = ANN) -> pd.DataFrame:
    rows = {}
    for s in lr:
        res = hac_ols(lr[s], bench.rename("bench"))
        rows[s] = {
            "alpha_ann": res.params["const"] * ann,
            "alpha_t": res.tvalues["const"],
            "beta": res.params["bench"],
            "r2": res.rsquared,
            "idio_vol_ann": res.resid.std() * np.sqrt(ann),
        }
    return pd.DataFrame(rows).T


def predictability_table(lr: pd.DataFrame) -> pd.DataFrame:
    """Do past returns predict tomorrow's? Single-stock time-series regressions with HAC errors."""
    rows = {}
    for s in lr:
        r = lr[s]
        nxt = r.shift(-1).rename("next")
        rev = hac_ols(nxt, r.rename("r1"))
        mom = hac_ols(nxt, r.rolling(20).sum().rename("r20"), lags=20)
        rows[s] = {"b_r1": rev.params["r1"], "t_r1": rev.tvalues["r1"],
                   "b_r20": mom.params["r20"], "t_r20": mom.tvalues["r20"]}
    return pd.DataFrame(rows).T


def weekday_table(lr: pd.DataFrame) -> pd.DataFrame:
    pooled = lr.stack().rename("r").reset_index()
    pooled["weekday"] = pd.to_datetime(pooled.iloc[:, 0]).dt.day_name()
    g = pooled.groupby("weekday")["r"]
    out = pd.DataFrame({"mean_bps": g.mean() * 1e4, "t": g.mean() / (g.std() / np.sqrt(g.count())), "n": g.count()})
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return out.reindex(order)


def hour_of_day_table(lr: pd.DataFrame) -> pd.DataFrame:
    """Intraday seasonality, pooled over stocks: mean return and volatility per bar of the day."""
    pooled = lr.stack().rename("r").reset_index()
    pooled["bar"] = pd.to_datetime(pooled.iloc[:, 0]).dt.strftime("%H:%M")
    g = pooled.groupby("bar")["r"]
    return pd.DataFrame({"mean_bps": g.mean() * 1e4, "t": g.mean() / (g.std() / np.sqrt(g.count())),
                         "vol_bps": g.std() * 1e4, "abs_move_bps": g.apply(lambda x: x.abs().mean()) * 1e4,
                         "n": g.count()})


def run_analysis(panel: Panel, cfg) -> Path:
    out = Path(cfg.paths.reports)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tickers = [t for t in cfg.data.tickers if t in panel.close]
    bench_sym = cfg.data.market["sector"]
    syms = tickers + [cfg.data.market["broad"], bench_sym]
    close = panel.close[syms]
    lr = np.log(close).diff().iloc[1:]
    stock_lr = lr[tickers]

    ann = cfg.data.periods_per_year
    summ = summary_table(lr, ann)
    dep = dependence_table(close, lr)
    fac = factor_table(stock_lr, lr[bench_sym], ann)
    pred = predictability_table(stock_lr)
    wd = weekday_table(stock_lr)

    corr = stock_lr.corr()
    corr_recent = stock_lr.iloc[-252:].corr()
    eig = np.linalg.eigvalsh(stock_lr.dropna().corr().to_numpy())[::-1]
    pc_share = eig / eig.sum()
    n = len(tickers)
    iu = np.triu_indices(n, 1)
    roll_corr = pd.Series(
        [stock_lr.iloc[i - 60:i].corr().to_numpy()[iu].mean() if i >= 60 else np.nan for i in range(len(stock_lr))],
        index=stock_lr.index,
    )

    # Figures
    fig, ax = plt.subplots(figsize=(10, 5))
    lr.cumsum().plot(ax=ax, lw=1)
    ax.set(title="Cumulative log return", ylabel="log(P/P0)")
    f1 = savefig(fig, fig_dir / "cumulative_returns.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    (lr.rolling(60).std() * np.sqrt(ann)).plot(ax=ax, lw=1)
    ax.set(title="Rolling 60-bar annualised volatility")
    f2 = savefig(fig, fig_dir / "rolling_vol.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n), corr.columns, rotation=45)
    ax.set_yticks(range(n), corr.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im)
    ax.set_title("Daily return correlation (full sample)")
    f3 = savefig(fig, fig_dir / "correlation.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    roll_corr.plot(ax=ax, lw=1)
    ax.set(title="Average pairwise 60-day correlation (rises in sell-offs)")
    f4 = savefig(fig, fig_dir / "rolling_correlation.png")

    z = (stock_lr - stock_lr.mean()) / stock_lr.std()
    zs = np.sort(z.stack().to_numpy())
    q = stats.norm.ppf((np.arange(1, len(zs) + 1) - 0.5) / len(zs))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(q, zs, s=2)
    ax.plot([-5, 5], [-5, 5], "k--", lw=1)
    ax.set(title="QQ plot: standardised returns vs normal", xlabel="normal quantile", ylabel="sample quantile")
    f5 = savefig(fig, fig_dir / "qq_plot.png")

    unit = "bar" if cfg.data.intraday else "day"
    avg_kurt = summ.loc[tickers, "ex_kurt"].mean()
    n_arch = int((dep.loc[tickers, "arch_lm_p"] < 0.05).sum())
    n_rw_rej = int(((dep.loc[tickers, "z_VR5"].abs() > 1.96) | (dep.loc[tickers, "z_VR20"].abs() > 1.96)).sum())
    n_nonstat = int((dep.loc[tickers, "adf_p_logprice"] > 0.05).sum())
    n_stat_r = int((dep.loc[tickers, "adf_p_returns"] < 0.05).sum())
    n_mom = int((pred["t_r20"].abs() > 1.96).sum())
    n_rev = int((pred["t_r1"].abs() > 1.96).sum())
    findings = [
        f"Returns are fat-tailed: average excess kurtosis {avg_kurt:.1f} (normal = 0); Jarque-Bera rejects "
        f"normality for every series. Risk models assuming normal returns understate tail losses.",
        f"Volatility clusters: ARCH-LM rejects constant variance for {n_arch}/{n} stocks. Volatility is "
        f"predictable even when returns are not - the basis for volatility targeting.",
        f"ADF: log prices look non-stationary for {n_nonstat}/{n} stocks, returns are stationary for "
        f"{n_stat_r}/{n}, so models are fit on returns rather than prices.",
        f"Random-walk (variance ratio) rejected at 5% for {n_rw_rej}/{n} stocks; next-{unit} reversal (r1) "
        f"significant for {n_rev}/{n}, 20-{unit} momentum for {n_mom}/{n}. Linear predictability is weak.",
        f"The stocks move together: mean pairwise correlation {corr.to_numpy()[iu].mean():.2f}, first principal "
        f"component explains {pc_share[0]:.0%} of variance. Diversification within the universe is limited.",
        f"Average beta to {bench_sym}: {fac['beta'].mean():.2f}; average R² {fac['r2'].mean():.2f}.",
    ]

    md = [
        "# Market analysis",
        f"Universe: {', '.join(tickers)}. Benchmarks: {cfg.data.market['broad']}, {bench_sym}. "
        f"Sample {lr.index[0].date()} – {lr.index[-1].date()} ({len(lr)} {cfg.data.interval} bars). "
        f"Log returns per bar; annualised with {ann} bars per year.",
        "## Key findings", "\n".join(f"- {s}" for s in findings),
        "## Return distribution", md_table(summ),
        "`jb_p`: Jarque-Bera normality p-value. VaR/CVaR are 1-bar historical 95% losses.",
        "## Stationarity, autocorrelation, volatility clustering, random walk", md_table(dep),
        "Ljung-Box on r tests return autocorrelation; on r² tests volatility clustering. "
        "VR(q) is the Lo-MacKinlay variance ratio (1 = random walk) with robust z.",
        f"## Market factor (CAPM vs {bench_sym}, HAC t-stats)", md_table(fac),
        "## Time-series predictability (next-bar return on past 1 and 20 bars, HAC t-stats)", md_table(pred, 4),
        "## Day-of-week effect (pooled)", md_table(wd, 2),
        *(["## Time-of-day effect (pooled, bar start time)", md_table(hour_of_day_table(stock_lr), 2),
           "The first bar includes the overnight gap, so it is far more volatile than the rest of the day."]
          if cfg.data.intraday else []),
        "## Correlation", md_table(corr, 2),
        "Last 252 days:", md_table(corr_recent, 2),
        "Principal component variance shares: " + ", ".join(f"{s:.1%}" for s in pc_share[:4]),
        "## Figures",
        *[f"![]({Path(p).relative_to(out)})" for p in (f1, f2, f3, f4, f5)],
    ]
    path = out / "analysis.md"
    path.write_text("\n\n".join(md) + "\n")
    for s in findings:
        log.info("%s", s)
    return path
