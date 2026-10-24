"""Stage 5c: compare all models out of sample, backtest them, and answer the project question."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import equal_weight_returns, run_backtest
from .data import Panel
from .features import FeatureSet, execution_returns
from .metrics import diebold_mariano, ic_series, performance, prediction_metrics
from .reporting import md_table, plt, savefig
from .optimal import make_trader
from .walkforward import schedule

log = logging.getLogger(__name__)


def verdict(pm: pd.DataFrame, bt: pd.DataFrame, bench_sharpe: float, focus: str = "transformer",
            focus_bt: str | None = None) -> list:
    """Plain-language conclusions. Deliberately strict: one backtest is weak evidence."""
    lines = []
    if focus not in pm.index:
        return ["No transformer predictions found - run `python -m trader walkforward --models transformer`."]
    p, b = pm.loc[focus], bt.loc[focus_bt or focus]
    baselines = pm.drop(index=[focus])
    best_base = baselines["ic_mean"].idxmax()
    lines.append(f"Forecast skill: transformer IC {p['ic_mean']:.4f} (t = {p['ic_t']:.2f}), hit rate "
                 f"{p['hit_rate']:.3f}, AUC {p['auc']:.3f}, out-of-sample R² {p['r2_oos']:.4f}. "
                 f"Best baseline by IC: {best_base} ({baselines.loc[best_base, 'ic_mean']:.4f}).")
    skill = p["ic_t"] > 2 or (p["dm_vs_zero"] < 0 and p["dm_p"] < 0.05)
    lines.append("The transformer's forecasts are statistically distinguishable from noise."
                 if skill else
                 "The transformer's forecasts are NOT statistically distinguishable from noise "
                 "(IC t-stat below 2 and no significant MSE improvement over predicting zero).")
    beats = p["ic_mean"] > baselines["ic_mean"].max()
    lines.append(f"It {'beats' if beats else 'does not beat'} every baseline on rank correlation (IC).")
    lines.append(f"Trading after costs ({focus_bt or focus}): Sharpe {b['sharpe']:.2f} (95% CI {b['sharpe_lo95']:.2f} to "
                 f"{b['sharpe_hi95']:.2f}, P(Sharpe>0) = {b['psr']:.2f}), CAGR {b['cagr']:.1%}, max drawdown "
                 f"{b['max_dd']:.1%}, cost drag {b['cost_drag_ann']:.2%}/yr. Equal-weight buy & hold Sharpe: "
                 f"{bench_sharpe:.2f}.")
    reliable = skill and b["psr"] > 0.95 and b["sharpe_lo95"] > 0 and b["sharpe"] > bench_sharpe
    if reliable:
        lines.append("VERDICT: evidence of a tradable signal after costs. Next step is paper trading for several "
                     "months to confirm before any real money - backtests overstate live results.")
    elif b["sharpe"] > 0 and b["psr"] > 0.95:
        lines.append("VERDICT: the strategy is profitable in backtest but does not clearly beat simply holding "
                     "the stocks risk-adjusted, or the forecast edge is not significant. Profit likely comes "
                     "from market exposure (beta) rather than prediction. Not a reliable signal.")
    else:
        lines.append("VERDICT: no reliable, cost-surviving signal. Do not trade real money on this model.")
    lines.append("Caveat: every extra model or setting tried raises the chance that a good backtest is luck "
                 "(multiple testing). Treat a single favourable result with suspicion.")
    return lines


def run_evaluation(all_preds: dict, fs: FeatureSet, panel: Panel, cfg) -> Path:
    out = Path(cfg.paths.reports)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Score every model on the same set of (date, ticker) samples with a realised target.
    keys = None
    for df in all_preds.values():
        k = set(zip(df["date"], df["ticker"]))
        keys = k if keys is None else keys & k
    common = {}
    for name, df in all_preds.items():
        m = pd.Series([(d, t) in keys for d, t in zip(df["date"], df["ticker"])], index=df.index)
        common[name] = df[m]

    pm = pd.DataFrame({name: prediction_metrics(df, fs.horizon) for name, df in common.items()}).T
    if "zero" in common and "hist_mean" in common:
        ref = common["hist_mean"].dropna(subset=["y"]).sort_values(["date", "ticker"])
        dm = {}
        for name, df in common.items():
            df = df.dropna(subset=["y"]).sort_values(["date", "ticker"])
            dm[name] = diebold_mariano(df["y"] - df["pred_ret"], ref["y"] - ref["pred_ret"], fs.horizon,
                                       df["date"].to_numpy())[0]
        pm["dm_vs_histmean"] = pd.Series(dm)

    ann = cfg.data.periods_per_year
    results = {}
    for name, df in all_preds.items():
        results[name] = run_backtest(df, fs, make_trader(cfg, "rules"))
        results[f"{name} (optimal)"] = run_backtest(df, fs, make_trader(cfg, "optimal"))
    bt = pd.DataFrame({name: r.stats() for name, r in results.items()}).T

    dates = next(iter(results.values())).returns.index
    ew = equal_weight_returns(fs, dates)
    sector = cfg.data.market["sector"]
    qqq = execution_returns(panel, [sector], fs.execution)[sector].reindex(dates).fillna(0.0)
    bench = pd.DataFrame({"equal_weight_hold": performance(ew, ann=ann),
                          f"{sector}_hold": performance(qqq, ann=ann)}).T
    bt_all = pd.concat([bt, bench])

    # Per-year Sharpe of each model's strategy: is performance stable or one lucky year?
    yearly = pd.DataFrame({name: r.returns.groupby(r.returns.index.year).apply(
        lambda x: x.mean() / x.std() * np.sqrt(ann) if x.std() > 0 else 0.0) for name, r in results.items()})
    yearly["equal_weight_hold"] = ew.groupby(ew.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(ann))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, r in results.items():
        ax.plot(np.log((1 + r.returns).cumprod()), lw=1.6 if name.startswith("transformer") else 1, label=name,
                ls="--" if name.endswith("(optimal)") else "-")
    ax.plot(np.log((1 + ew).cumprod()), "k--", lw=1, label="equal-weight hold")
    ax.plot(np.log((1 + qqq).cumprod()), ":", color="gray", lw=1, label=f"{sector} hold")
    ax.set(title="Out-of-sample strategy equity after costs (log scale)", ylabel="log equity")
    ax.legend(fontsize=8)
    f_eq = savefig(fig, fig_dir / "equity_curves.png")

    if "transformer" in common:
        ic = ic_series(common["transformer"])
        fig, ax = plt.subplots(figsize=(11, 4))
        ic.rolling(126 * (7 if cfg.data.intraday else 1)).mean().plot(ax=ax)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title="Transformer: rolling 6-month mean information coefficient")
        f_ic = savefig(fig, fig_dir / "transformer_ic.png")
    else:
        f_ic = None

    first, months = schedule(fs, cfg.walkforward)
    focus_bt = "transformer (optimal)" if cfg.optimizer.trader == "optimal" else "transformer"
    findings = verdict(pm, bt, bench.loc["equal_weight_hold", "sharpe"], "transformer", focus_bt)
    s, c = cfg.strategy, cfg.costs
    md = [
        "# Out-of-sample evaluation",
        f"{cfg.data.interval} bars. Walk-forward from {first.date()}, retraining every {months} month(s). "
        f"Target: {fs.horizon}-bar forward log return, execution at `{fs.execution}`. "
        f"{pm['n'].iloc[0]:.0f} common out-of-sample predictions per model.",
        "## Conclusion", "\n".join(f"- {x}" for x in findings),
        "## Forecast quality", md_table(pm, 4),
        "`r2_oos` vs predicting 0; `ic_mean` = mean daily cross-sectional Spearman correlation, `ic_t` its "
        "Newey-West t-stat; `ts_corr` = average per-stock time-series correlation; `dm_*` = Diebold-Mariano "
        "statistic on squared errors (negative = better than the reference).",
        "## Trading simulation (after costs)", md_table(bt_all, 3),
        f"Strategy: long{'/short' if s.allow_short else '-only'}, P(up) ≥ {s.prob_threshold}, "
        f"target vol {s.target_vol:.0%}, max {s.max_position:.0%} per stock, gross ≤ {s.max_gross:.0%}, "
        f"daily loss limit {s.daily_loss_limit:.0%}, drawdown stop {s.max_drawdown:.0%}. "
        f"Costs {c.commission_bps + c.slippage_bps:.1f} bps per side. Benchmarks ignore costs. "
        f"Rows marked (optimal) use the OptimalTrader (risk aversion {cfg.optimizer.risk_aversion}, holding "
        f"period {cfg.optimizer.holding_periods}, calibrated on the last {cfg.optimizer.calib_window} periods "
        f"of each model's own track record) under the same limits.",
        "## Sharpe ratio by year", md_table(yearly, 2),
        "## Figures", f"![](figures/{Path(f_eq).name})",
    ]
    if f_ic:
        md.append(f"![](figures/{Path(f_ic).name})")
    path = out / "evaluation.md"
    path.write_text("\n\n".join(md) + "\n")
    pm.to_csv(Path(cfg.paths.artifacts) / "prediction_metrics.csv")
    bt_all.to_csv(Path(cfg.paths.artifacts) / "backtest_metrics.csv")
    print("\nForecast quality\n" + pm[["n", "r2_oos", "hit_rate", "auc", "ic_mean", "ic_t", "dm_vs_zero"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nTrading (after costs)\n" + bt_all[["cagr", "ann_vol", "sharpe", "psr", "max_dd", "turnover_ann"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n" + "\n".join(f"* {x}" for x in findings))
    return path
