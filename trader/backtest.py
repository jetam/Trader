"""Stage 5b: event-driven daily backtest of a Strategy on out-of-sample predictions.

Timeline for decision row t (data up to the close of day t):
    execution='close'      trade at close t, hold to close t+1
    execution='next_open'  trade at open t+1, hold to open t+2
Costs are charged on every change of weight (commission + slippage per side), plus a borrow fee on
shorts. Risk limits only see equity that would actually be known at decision time.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FeatureSet
from .metrics import performance
from .strategy import RiskState, Strategy, covariance


@dataclass
class BacktestResult:
    returns: pd.Series        # net daily return
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    exposure: pd.Series       # sum of |weights|
    weights: pd.DataFrame
    halted: pd.Series         # reason the strategy was forced flat, or ""
    periods_per_year: int = 252

    def stats(self) -> dict:
        s = performance(self.returns, self.turnover, self.costs, self.exposure, self.periods_per_year)
        s["days_halted"] = int((self.halted != "").sum())
        return s


def run_backtest(preds: pd.DataFrame, fs: FeatureSet, strategy: Strategy,
                 start: pd.Timestamp | None = None) -> BacktestResult:
    """Trade on `preds` from `start` (default: the first prediction). Earlier predictions are still used
    for calibration by traders that learn from the model's track record."""
    cfg, costs = strategy.cfg, strategy.costs
    pred_w = preds.pivot(index="date", columns="ticker", values="pred_ret").reindex(columns=fs.tickers)
    prob_w = preds.pivot(index="date", columns="ticker", values="prob_up").reindex(columns=fs.tickers)
    row_of = {d: i for i, d in enumerate(fs.dates)}
    rows = [row_of[d] for d in pred_w.index if d in row_of]
    rows = [t for t in rows if np.isfinite(fs.exec_ret[t]).any()]   # drop rows with no realised return yet
    if start is not None:
        rows = [t for t in rows if fs.dates[t] >= start]
    lag = 2 if fs.execution == "next_open" else 1   # how stale the known equity is at decision time

    # The OptimalTrader calibrates on the model's track record. A prediction made at row r is only
    # scored once its target is realised, i.e. at row r + horizon + 1 at the earliest.
    calibrating = hasattr(strategy, "calibrate")
    if calibrating:
        hist = preds.dropna(subset=["pred_ret", "y"])
        hist_rows = hist["date"].map(row_of).to_numpy()
        hist_pred, hist_y = hist["pred_ret"].to_numpy(), hist["y"].to_numpy()
        opt = strategy.opt

    N = len(fs.tickers)
    w_cur = np.zeros(N)
    eq = [1.0]                                    # eq[k] = equity after k holding periods
    state = RiskState()
    out = {k: [] for k in ("net", "gross", "cost", "turn", "expo", "halt")}
    weights = []
    for k, t in enumerate(rows):
        if calibrating and k % opt.recalibrate_every == 0:
            known_hi = t - fs.horizon - 1
            m = (hist_rows <= known_hi) & (hist_rows > known_hi - opt.calib_window)
            strategy.calibrate(hist_pred[m], hist_y[m])
        known = eq[max(k + 1 - lag, 0)]
        prev = eq[max(k - lag, 0)]
        reason = state.update(known, prev, cfg)
        cov = covariance(fs.past_ret[max(t - cfg.cov_lookback + 1, 0):t + 1])
        d = fs.dates[t]
        target = strategy.target_weights(pred_w.loc[d].to_numpy(), prob_w.loc[d].to_numpy(), cov, w_cur, reason)

        r = np.nan_to_num(fs.exec_ret[t])
        turn = np.abs(target - w_cur).sum()
        cost = turn * costs.one_way + np.abs(np.minimum(target, 0)).sum() * costs.borrow_bps_annual / 1e4 / strategy.ann
        gross = float(target @ r)
        net = gross - cost
        eq.append(eq[-1] * (1 + net))
        w_cur = target * (1 + r) / (1 + gross) if 1 + gross > 1e-9 else np.zeros(N)

        for key, v in zip(out, (net, gross, cost, turn, np.abs(target).sum(), reason or "")):
            out[key].append(v)
        weights.append(target)

    idx = pd.DatetimeIndex([fs.dates[t] for t in rows], name="date")
    s = lambda key: pd.Series(out[key], index=idx)  # noqa: E731
    return BacktestResult(s("net"), s("gross"), s("cost"), s("turn"), s("expo"),
                          pd.DataFrame(weights, index=idx, columns=fs.tickers), s("halt"), strategy.ann)


def equal_weight_returns(fs: FeatureSet, dates: pd.DatetimeIndex) -> pd.Series:
    """Buy-and-hold proxy: equal weight in every available stock, rebalanced daily, no costs."""
    r = pd.DataFrame(fs.exec_ret, index=fs.dates, columns=fs.tickers).reindex(dates)
    return r.mean(axis=1, skipna=True).fillna(0.0)
