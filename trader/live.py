"""Live (paper-first) trading loop and ongoing monitoring of model quality.

Daily model (config.yaml), once after the US close, e.g. 17:00 New York time:
    python -m trader trade            # dry run: show the plan, place nothing
    python -m trader trade --execute  # send market-on-open orders to the IB paper account
    python -m trader monitor          # compare logged forecasts with what actually happened

Hourly model (-c config_hourly.yaml), a minute after each bar closes: 10:31, 11:31, ... 15:31 send
market orders that fill right away; the 16:01 run (after the last bar) sends market-on-open orders.
"""

import dataclasses
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .broker import FILL_COLUMNS, Broker, SafetyError, plan_orders
from .data import NY, Panel, download_yahoo, next_bar
from .features import build_features
from .metrics import ic_series, prediction_metrics, sharpe
from .optimal import OptimalTrader, make_trader
from .strategy import RiskState, covariance
from .transformer import TransformerPredictor

log = logging.getLogger(__name__)


def model_path(cfg) -> Path:
    return Path(cfg.paths.models) / "transformer.pt"


def recent_panel(cfg, broker: Broker | None = None) -> Panel:
    """Enough recent history to compute every feature: 2 years of daily bars or 120 days of hourly bars."""
    symbols = cfg.data.symbols
    days = 120 if cfg.data.intraday else 730
    if cfg.broker.data_source == "ib":
        if broker is None:
            raise ValueError("data_source 'ib' needs a broker connection")
        long = broker.history(symbols, duration=f"{days} D", interval=cfg.data.interval)
    else:
        long = download_yahoo(symbols, start=(date.today() - timedelta(days=days)).isoformat(),
                              interval=cfg.data.interval)
    m = cfg.data.market
    return Panel.from_long(long, calendar_symbol=m["sector"], ffill_symbols=(m["vol"], m["rates"]))


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _append_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def track_record(cfg, fs) -> pd.DataFrame:
    """Past transformer predictions with realised returns: walk-forward test plus live forecasts."""
    parts = []
    wf = Path(cfg.paths.artifacts) / "predictions" / "transformer.parquet"
    if wf.exists():
        parts.append(pd.read_parquet(wf)[["date", "ticker", "pred_ret", "y"]])
    live = Path(cfg.paths.logs) / "predictions.csv"
    if live.exists():
        logged = pd.read_csv(live, parse_dates=["date"])[["date", "ticker", "pred_ret"]]
        y = pd.DataFrame(fs.y, index=fs.dates, columns=fs.tickers).stack().rename("y").reset_index()
        y.columns = ["date", "ticker", "y"]
        parts.append(logged.merge(y, on=["date", "ticker"], how="inner"))
    if not parts:
        return pd.DataFrame(columns=["date", "ticker", "pred_ret", "y"])
    hist = pd.concat(parts).assign(date=lambda d: pd.to_datetime(d["date"]))
    hist = hist.dropna(subset=["pred_ret", "y"]).drop_duplicates(["date", "ticker"], keep="last")
    recent_dates = np.sort(hist["date"].unique())[-cfg.optimizer.calib_window:]
    return hist[hist["date"].isin(recent_dates)]


def fills_path(cfg) -> Path:
    return Path(cfg.paths.logs) / "fills.csv"


def load_fills(cfg) -> pd.DataFrame:
    path = fills_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=FILL_COLUMNS)
    return pd.read_csv(path, parse_dates=["time"])


def sync_fills(cfg, broker: Broker) -> pd.DataFrame:
    """Add this trader's executions (its client id) to logs/fills.csv and return the full history.

    IB only reports recent executions, so this runs on every `trade` run and GUI refresh.
    """
    new = broker.fills()
    new = new[new["client_id"] == cfg.broker.client_id]
    old = load_fills(cfg)
    if new.empty:
        return old
    allf = pd.concat([df for df in (old, new) if not df.empty], ignore_index=True)
    allf = allf.drop_duplicates("exec_id", keep="last").sort_values("time")   # later copies carry commissions
    path = fills_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    allf.to_csv(path, index=False)
    return allf


def account_snapshot(cfg) -> dict:
    """Read-only look at the IB account: value, positions, open orders, and the synced fill history."""
    gui = dataclasses.replace(cfg.broker, client_id=cfg.broker.client_id + 100)   # never clash with `trade`
    with Broker(gui, read_only=True) as broker:
        fills = sync_fills(cfg, broker)
        return {"time": datetime.now(), "account": broker.account, "paper": broker.is_paper,
                "nlv": broker.net_liquidation(), "portfolio": broker.portfolio(),
                "open_orders": broker.open_orders(), "fills": fills}


def run_trade(cfg, execute: bool = False) -> list:
    predictor = TransformerPredictor.load(model_path(cfg))
    meta = predictor.meta
    logs = Path(cfg.paths.logs)
    logs.mkdir(parents=True, exist_ok=True)
    kill = logs / "KILL"
    if execute and kill.exists():
        raise SafetyError(f"kill switch {kill} exists - no orders will be sent. Delete the file to resume.")
    strategy = make_trader(cfg)
    intraday = meta.get("interval", "1d") != "1d"
    if meta.get("interval", "1d") != cfg.data.interval:
        raise ValueError(f"model was trained on {meta.get('interval')} bars but config uses {cfg.data.interval}")

    with Broker(cfg.broker, read_only=not execute) as broker:
        try:
            sync_fills(cfg, broker)
        except Exception as e:   # the fill log is for reporting only; never block trading on it
            log.warning("could not sync fills: %s", e)
        panel = recent_panel(cfg, broker)
        fs = build_features(panel, predictor.tickers, cfg.data.market, predictor.lookback, meta["horizon"],
                            meta["execution"], intraday)
        t = len(fs.dates) - 1
        asof = fs.dates[t]
        # The forecast is for the bar after `asof`; once that bar is well under way the data is stale.
        nxt = next_bar(asof, intraday) + (pd.Timedelta(0) if intraday else pd.Timedelta(hours=9, minutes=30))
        if pd.Timestamp(datetime.now(NY).replace(tzinfo=None)) > nxt + pd.Timedelta(minutes=30):
            raise RuntimeError(f"latest bar is {asof}, but the bar it forecasts ({nxt}) started over 30 minutes ago. "
                               "Refusing to trade on stale data (data source behind, or a market holiday).")
        n_idx = np.nonzero(fs.window_ok()[t])[0]
        pred = np.full(len(fs.tickers), np.nan)
        prob = np.full(len(fs.tickers), np.nan)
        pred[n_idx], prob[n_idx] = predictor.predict(fs, np.full(len(n_idx), t), n_idx)
        _append_csv(logs / "predictions.csv", pd.DataFrame({
            "date": asof, "ticker": fs.tickers, "pred_ret": pred, "prob_up": prob,
            "model_trained_through": meta.get("trained_through")}))

        if isinstance(strategy, OptimalTrader):
            hist = track_record(cfg, fs)
            cal = strategy.calibrate(hist["pred_ret"].to_numpy(), hist["y"].to_numpy())
            print(f"OptimalTrader calibration on {cal.n} past forecasts: slope {cal.raw_slope:.3f} "
                  f"(t = {cal.t_stat:.2f}) -> {cal.slope:.3f} after shrinkage")
            if cal.slope == 0:
                print("  The model has no reliable track record, so the OptimalTrader will not take positions.")

        nlv = broker.net_liquidation()
        # Real-money guard: size positions on at most `max_capital`, even if the account holds more.
        capital = min(nlv, cfg.broker.max_capital) if cfg.broker.max_capital else nlv
        shares = broker.positions()
        prices = {s: float(panel.close[s].iloc[-1]) for s in fs.tickers}
        current = np.array([shares.get(s, 0.0) * prices[s] / capital for s in fs.tickers])

        # Risk state advances once per bar, on the account's own equity.
        state_path = logs / "risk_state.json"
        saved = _load_state(state_path)
        state = RiskState(peak=saved.get("peak", nlv), halted_days=saved.get("halted_days", 0))
        if saved.get("date") == str(asof):
            reason = saved.get("reason")
        else:
            reason = state.update(nlv, saved.get("equity"), cfg.strategy)
            state_path.write_text(json.dumps({**state.to_dict(), "date": str(asof), "equity": nlv,
                                              "reason": reason}, indent=2))
            _append_csv(logs / "equity.csv", pd.DataFrame({"date": [asof], "nlv": [nlv]}))

        cov = covariance(fs.past_ret[max(t - cfg.strategy.cov_lookback + 1, 0):t + 1])
        target = strategy.target_weights(pred, prob, cov, current, reason)
        plans = plan_orders(fs.tickers, target, shares, prices, capital, cfg.broker.max_order_value)

        table = pd.DataFrame({"pred_ret_%": pred * 100, "p_up": prob, "cur_w": current, "target_w": target,
                              "order_qty": [p.quantity for p in plans]}, index=fs.tickers)
        if not broker.is_paper:
            print("\n" + "*" * 70 + "\n*** LIVE ACCOUNT - THESE ORDERS USE REAL MONEY ***\n" + "*" * 70)
        print(f"\nAccount {broker.account} ({'paper' if broker.is_paper else 'LIVE'})  NLV {nlv:,.0f}  "
              f"trading capital {capital:,.0f}  trader: {type(strategy).__name__}  "
              f"data as of {asof}  model trained through {meta.get('trained_through')}")
        if reason:
            print(f"RISK LIMIT ACTIVE: {reason} -> target is flat")
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))

        orders = [p for p in plans if p.quantity != 0]
        if not orders:
            print("No orders needed.")
        elif execute:
            pending = broker.open_orders()
            pending = pending[pending["symbol"].isin(fs.tickers)]
            if not pending.empty:
                raise SafetyError(f"{len(pending)} order(s) for these stocks are still waiting at IB "
                                  f"({', '.join(pending['symbol'])}). Sending more would double up; wait for "
                                  "them to fill or cancel them in TWS, then run again.")
            broker.place(orders, meta["execution"], intraday)
            _append_csv(logs / "orders.csv", pd.DataFrame([{**vars(p), "date": asof} for p in orders]))
        else:
            print(f"Dry run: {len(orders)} order(s) NOT sent. Re-run with --execute to submit to IB.")
        return plans


def run_monitor(cfg) -> None:
    """Score logged live forecasts against realised returns and flag degradation."""
    logs = Path(cfg.paths.logs)
    path = logs / "predictions.csv"
    if not path.exists():
        print("No live predictions logged yet.")
        return
    preds = pd.read_csv(path, parse_dates=["date"]).drop_duplicates(["date", "ticker"], keep="last")
    predictor = TransformerPredictor.load(model_path(cfg))
    panel = recent_panel(cfg)
    fs = build_features(panel, predictor.tickers, cfg.data.market, predictor.lookback,
                        predictor.meta["horizon"], predictor.meta["execution"], cfg.data.intraday)
    y = pd.DataFrame(fs.y, index=fs.dates, columns=fs.tickers).stack().rename("y").reset_index()
    y.columns = ["date", "ticker", "y"]
    df = preds.merge(y, on=["date", "ticker"], how="inner").dropna(subset=["pred_ret", "y"])
    if len(df) < 20:
        print(f"Only {len(df)} scored live predictions so far - too few to judge.")
        return
    m = prediction_metrics(df, predictor.meta["horizon"])
    print(f"Live forecasts scored: {m['n']}  hit rate {m['hit_rate']:.3f}  AUC {m['auc']:.3f}  "
          f"IC {m['ic_mean']:.4f} (t {m['ic_t']:.2f})")

    ref_path = Path(cfg.paths.artifacts) / "prediction_metrics.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path, index_col=0)
        if "transformer" in ref.index:
            print(f"Backtest reference: hit rate {ref.loc['transformer', 'hit_rate']:.3f}  "
                  f"IC {ref.loc['transformer', 'ic_mean']:.4f}")
    recent_ic = ic_series(df).tail(60).mean()
    if recent_ic < 0:
        print(f"WARNING: mean IC over the last 60 days is negative ({recent_ic:.4f}). Consider retraining "
              f"(`python -m trader train`) or pausing trading.")

    eq_path = logs / "equity.csv"
    if eq_path.exists():
        eq = pd.read_csv(eq_path, parse_dates=["date"]).drop_duplicates("date", keep="last").set_index("date")["nlv"]
        r = eq.pct_change().dropna()
        if len(r) > 5:
            print(f"Paper account: {len(r)} bars, return {eq.iloc[-1] / eq.iloc[0] - 1:.2%}, "
                  f"Sharpe {sharpe(r.to_numpy(), cfg.data.periods_per_year):.2f}, max drawdown {(eq / eq.cummax() - 1).min():.2%}")
