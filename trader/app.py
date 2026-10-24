"""GUI: "what if I had followed the model" simulations and the model's forecast for the next bar.

Launch with:  python -m trader gui
"""

import argparse
import asyncio
import dataclasses
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trader.backtest import equal_weight_returns, run_backtest
from trader.broker import plan_orders
from trader.config import load_config
from trader.data import NY, fetch_prices, load_panel, prices_path
from trader.features import execution_returns, features_from_config
from trader.forecast import latest_forecast, next_bar
from trader.live import account_snapshot, load_fills, model_path
from trader.metrics import performance
from trader.optimal import OptimalTrader, make_trader
from trader.strategy import covariance
from trader.walkforward import load_all_predictions

# Fixed entity -> colour so a series never changes colour when others are toggled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ENTITIES = ["transformer", "Equal-weight hold", "QQQ hold", "ridge", "gbm", "momentum", "hist_mean", "zero"]
COLOR = dict(zip(ENTITIES, PALETTE))
BENCHMARKS = ["Equal-weight hold", "QQQ hold"]
GREY = "#8a8984"

st.set_page_config(page_title="Trader", page_icon="📈", layout="wide")


TIMEFRAMES = {"Daily": "config.yaml", "Hourly": "config_hourly.yaml"}


def config_path() -> str:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_known_args(sys.argv[1:])[0].config


@st.cache_resource(show_spinner="Loading data and building features…")
def load_state(path: str):
    cfg = load_config(path)
    panel = load_panel(cfg)
    fs = features_from_config(panel, cfg)
    preds = {k: df.assign(date=pd.to_datetime(df["date"])) for k, df in load_all_predictions(cfg).items()}
    return cfg, panel, fs, preds


available = {name: path for name, path in TIMEFRAMES.items() if Path(path).exists()}
default = config_path()
if default not in available.values():
    available = {Path(default).stem: default, **available}
with st.sidebar:
    timeframe = st.radio("Timeframe", list(available), horizontal=True,
                         index=list(available.values()).index(default))
cfg_path = available[timeframe]
try:
    cfg, panel, fs, preds = load_state(cfg_path)
except FileNotFoundError:
    st.error(f"No data for the {timeframe.lower()} system yet. Run `python -m trader -c {cfg_path} fetch`.")
    st.stop()
has_transformer = model_path(cfg).exists()
if not preds and not has_transformer:
    st.error(f"No models found. Run `python -m trader -c {cfg_path} walkforward` first.")
    st.stop()
ann = cfg.data.periods_per_year
intraday = cfg.data.intraday
bar = "bar" if intraday else "day"

models = sorted(set(preds) | ({"transformer"} if has_transformer else set()),
                key=lambda m: ENTITIES.index(m) if m in ENTITIES else 99)

# ---- sidebar ---------------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", models, index=0,
                         help="Past simulation uses out-of-sample predictions: each period was predicted by a "
                              "model trained only on earlier data.")
    capital = st.number_input("Capital ($)", min_value=100.0, value=10_000.0, step=1_000.0)
    trader_kind = st.radio("Trader", ["rules", "optimal"], horizontal=True,
                           index=["rules", "optimal"].index(cfg.optimizer.trader),
                           format_func={"rules": "Rule-based", "optimal": "OptimalTrader"}.get,
                           help="Rule-based: buy when P(up) and the expected return clear fixed thresholds. "
                                "OptimalTrader: calibrates the model on its own track record, then solves for "
                                "the weights with the best return/risk/cost trade-off.")
    o = cfg.optimizer
    if trader_kind == "optimal":
        with st.expander("OptimalTrader settings", expanded=True):
            risk_aversion = st.slider("Risk aversion", 0.1, 10.0, float(o.risk_aversion), 0.1,
                                      help="Higher = smaller positions for the same expected return.")
            holding_periods = st.slider(f"Expected holding period ({bar}s)", 1.0, 40.0, float(o.holding_periods), 1.0,
                                        help="Trading cost is spread over this many periods.")
            use_drift = st.checkbox("Also bet on average market drift", o.use_drift,
                                    help="Adds the average realised return to every forecast: more market "
                                         "exposure, not more skill.")
    else:
        risk_aversion, holding_periods, use_drift = o.risk_aversion, o.holding_periods, o.use_drift

    s, c = cfg.strategy, cfg.costs
    with st.expander("Strategy & risk limits"):
        allow_short = st.checkbox("Allow short selling", s.allow_short)
        prob_threshold = st.slider("Min P(up) to buy", 0.50, 0.65, s.prob_threshold, 0.005)
        target_vol = st.slider("Target annual volatility", 0.05, 0.40, s.target_vol, 0.01)
        max_position = st.slider("Max weight per stock", 0.05, 1.0, s.max_position, 0.05)
        max_gross = st.slider("Max total exposure", 0.2, 2.0, s.max_gross, 0.1)
        min_trade = st.slider("Min weight change to trade", 0.0, 0.10, s.min_trade, 0.005)
        daily_loss_limit = st.slider(f"Loss limit per {bar}", 0.01, 0.20, s.daily_loss_limit, 0.01)
        max_drawdown = st.slider("Drawdown stop", 0.05, 0.60, s.max_drawdown, 0.01)
        cooldown_days = st.slider(f"Cooldown after drawdown stop ({bar}s)", 0, 250 if intraday else 60,
                                  s.cooldown_days)
    with st.expander("Trading costs"):
        commission_bps = st.number_input("Commission (bps per side)", 0.0, 50.0, c.commission_bps, 0.5)
        slippage_bps = st.number_input("Slippage (bps per side)", 0.0, 50.0, c.slippage_bps, 0.5)

strategy_params = dict(allow_short=allow_short, prob_threshold=prob_threshold, target_vol=target_vol,
                       max_position=max_position, max_gross=max_gross, min_trade=min_trade,
                       daily_loss_limit=daily_loss_limit, max_drawdown=max_drawdown, cooldown_days=cooldown_days)
cost_params = dict(commission_bps=commission_bps, slippage_bps=slippage_bps)
opt_params = dict(risk_aversion=risk_aversion, holding_periods=holding_periods, use_drift=use_drift)


def make_sim_trader(strategy_params: dict, cost_params: dict, opt_params: dict, trader_kind: str):
    sim_cfg = dataclasses.replace(cfg, strategy=dataclasses.replace(cfg.strategy, **strategy_params),
                                  costs=dataclasses.replace(cfg.costs, **cost_params),
                                  optimizer=dataclasses.replace(cfg.optimizer, **opt_params))
    return make_trader(sim_cfg, trader_kind)


@st.cache_data(show_spinner="Simulating…")
def simulate(cfg_path: str, model: str, start: date, end: date, strategy_params: dict, cost_params: dict,
             trader_kind: str, opt_params: dict):
    end_ts = pd.Timestamp(end + timedelta(days=1))    # end date is inclusive
    df = preds[model]
    df = df[df["date"] < end_ts]
    # Trading starts at `start`; the OptimalTrader also learns from predictions made before it.
    res = run_backtest(df, fs, make_sim_trader(strategy_params, cost_params, opt_params, trader_kind),
                       pd.Timestamp(start))
    return {"returns": res.returns, "weights": res.weights, "costs": res.costs, "turnover": res.turnover,
            "exposure": res.exposure, "halted": res.halted, "stats": res.stats()}


@st.cache_data(show_spinner="Fitting the model on all data up to now…")
def forecast(cfg_path: str, model: str, last_bar: str, model_mtime: float | None) -> pd.DataFrame:
    return latest_forecast(panel, fs, model, cfg)


def benchmarks(dates: pd.DatetimeIndex) -> pd.DataFrame:
    sector = cfg.data.market["sector"]
    qqq = execution_returns(panel, [sector], fs.execution)[sector].reindex(dates).fillna(0.0)
    return pd.DataFrame({"Equal-weight hold": equal_weight_returns(fs, dates), "QQQ hold": qqq})


XFMT = "%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"
WHEN = "%a %Y-%m-%d %H:%M" if intraday else "%a %Y-%m-%d"


def compress_time(fig: go.Figure) -> go.Figure:
    """Hide nights and weekends on hourly charts so the x-axis has no empty gaps."""
    if intraday:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
    return fig


def line_chart(df: pd.DataFrame, title: str, yfmt: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_scatter(x=df.index, y=df[col], name=col, mode="lines",
                        line=dict(width=2.5 if col == model else 2, color=COLOR.get(col),
                                  dash="dot" if col in BENCHMARKS else None),
                        hovertemplate=f"%{{x|{XFMT}}}<br>{col}: %{{y:{yfmt}}}<extra></extra>")
    fig.update_layout(title=title, height=height, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    fig.update_yaxes(tickformat=yfmt)
    return compress_time(fig)


# ---- Past simulation -------------------------------------------------------------------------

def past_simulation():
    if model not in preds:
        st.info(f"No walk-forward predictions for {model} yet, so there is nothing to simulate. Run "
                f"`python -m trader -c {cfg_path} walkforward --models {model}`.")
        return
    first_pred = preds[model]["date"].min().date()
    last_pred = preds[model]["date"].max().date()
    cols = st.columns([1, 1, 2])
    start = cols[0].date_input("Start date", value=max(date(2020, 1, 2), first_pred), min_value=first_pred,
                               max_value=last_pred,
                               help=f"Model predictions exist from {first_pred} (start of the walk-forward test).")
    end = cols[1].date_input("End date", value=last_pred, min_value=first_pred, max_value=last_pred)
    with cols[2]:
        st.write("")
        compare_all = st.checkbox("Also simulate every other model", value=False)
    if start >= end:
        st.error("Start date must be before the end date.")
        return

    res = simulate(cfg_path, model, start, end, strategy_params, cost_params, trader_kind, opt_params)
    r = res["returns"]
    if r.empty:
        st.warning("No trading days in the selected range.")
        return
    bench = benchmarks(r.index)
    rets = pd.concat({model: r, **{b: bench[b] for b in BENCHMARKS}}, axis=1)
    if compare_all:
        for m in preds:
            if m != model:
                rets[m] = simulate(cfg_path, m, start, end, strategy_params, cost_params, trader_kind, opt_params)["returns"]
    money = capital * (1 + rets).cumprod()
    final = money.iloc[-1]

    real_start, real_end = r.index[0].date(), r.index[-1].date()
    st.subheader(f"Following **{model}** from {real_start} to {real_end}")
    cols = st.columns(4)
    cols[0].metric(f"Your ${capital:,.0f} became", f"${final[model]:,.0f}",
                   f"{final[model] / capital - 1:+.1%} total return")
    for col, b in zip(cols[1:3], BENCHMARKS):
        more = final[model] / final[b] - 1
        col.metric(f"More money than {b}", f"{more:+.1%}",
                   f"{final[model] - final[b]:+,.0f} $ (holding gave ${final[b]:,.0f})")
    worst = (money / money.cummax() - 1).min()
    cols[3].metric("Worst fall along the way", f"{worst[model]:.1%}",
                   f"{worst[model] - worst['Equal-weight hold']:+.1%} vs equal-weight hold")
    better = final[model] >= final["Equal-weight hold"]
    st.caption(
        f"{'Following the model beat' if better else 'Following the model trailed'} simply buying all "
        f"{len(fs.tickers)} stocks in equal amounts on {real_start}. The model account pays "
        f"{commission_bps + slippage_bps:.1f} bps per trade side in costs; the hold benchmarks are cost-free. "
        f"Trades are assumed to happen at the next {bar}'s open after each signal.")

    st.plotly_chart(line_chart(money, "Account value ($)", "$,.0f"), width="stretch")

    dd = money / money.cummax() - 1
    st.plotly_chart(line_chart(dd, "Drawdown from previous peak", ".0%", 260), width="stretch")

    left, right = st.columns(2)
    with left:
        period = rets.index.strftime("%Y-%m") if intraday else rets.index.year
        yearly = (1 + rets).groupby(period).prod() - 1
        yl = yearly.reset_index(names="period").melt("period", var_name="strategy", value_name="return")
        fig = px.bar(yl, x="period", y="return", color="strategy", barmode="group", color_discrete_map=COLOR,
                     title=f"Return by calendar {'month' if intraday else 'year'}")
        fig.update_xaxes(type="category", title="")
        fig.update_traces(hovertemplate="%{x}: %{y:.1%}<extra>%{fullData.name}</extra>")
        fig.update_layout(yaxis_tickformat=".0%", height=380, bargap=0.25, bargroupgap=0.08,
                          margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    with right:
        w = res["weights"]
        held = w.loc[:, (w != 0).any()]
        wl = held.reset_index().melt("date", var_name="stock", value_name="weight")
        fig = px.area(wl, x="date", y="weight", color="stock", title=f"What {model} held (fraction of account)",
                      color_discrete_sequence=PALETTE)
        fig.update_traces(line=dict(width=0.5),
                          hovertemplate=f"%{{x|{XFMT}}}: %{{y:.1%}}<extra>%{{fullData.name}}</extra>")
        fig.update_layout(yaxis_tickformat=".0%", height=380, margin=dict(l=10, r=10, t=50, b=10),
                          legend_title_text="")
        st.plotly_chart(compress_time(fig), width="stretch")

    st.markdown("#### Risk and performance")
    table = {name: performance(rets[name], ann=ann) for name in rets}
    table[model] = res["stats"]
    perf = pd.DataFrame(table).T
    perf.insert(0, "final_value", final.reindex(perf.index))
    show = {"final_value": "Final $", "cagr": "Return / yr", "ann_vol": "Volatility / yr", "sharpe": "Sharpe",
            "sharpe_lo95": "Sharpe 95% low", "sharpe_hi95": "Sharpe 95% high", "max_dd": "Max drawdown",
            "worst_day": "Worst day", "turnover_ann": "Turnover / yr", "cost_drag_ann": "Costs / yr",
            "avg_gross": "Avg exposure", "days_halted": f"{bar.capitalize()}s halted by risk limits"}
    perf = perf[[k for k in show if k in perf]].rename(columns=show)
    pct = ["Return / yr", "Volatility / yr", "Max drawdown", "Worst day", "Costs / yr", "Avg exposure"]
    st.dataframe(perf.style.format({"Final $": "${:,.0f}", **{k: "{:.1%}" for k in pct}}, na_rep="–",
                                   precision=2), width="stretch")

    with st.expander("Recent target positions"):
        st.dataframe(w.tail(15).iloc[::-1].style.format("{:.1%}"), width="stretch")
    with st.expander("Daily account values (table)"):
        st.dataframe(money.iloc[::-1].style.format("${:,.0f}"), width="stretch")


# ---- Future prediction -----------------------------------------------------------------------

def rule_reason(pred: float, prob: float, held: bool, hurdle: float) -> str:
    """Why the rule-based trader wants (or does not want) a stock; mirrors Strategy.target_weights."""
    if not (np.isfinite(pred) and np.isfinite(prob)):
        return "not enough recent data"
    if held and prob > 0.5 and pred > 0:
        return "already owned and forecast still positive → keep"
    if pred <= 0:
        return "predicted to fall"
    if prob < prob_threshold:
        return f"not confident enough: P(up) {prob:.1%} < {prob_threshold:.1%}"
    if pred <= hurdle:
        return f"expected gain {pred:+.3%} does not cover the {hurdle:.3%} trading cost"
    return "passes both tests → buy"


def action(qty: int, own: float, target: float) -> str:
    if qty > 0:
        return "BUY" if own <= 0 else "BUY MORE"
    if qty < 0:
        return "SELL ALL" if target <= 0 else "REDUCE"
    return "HOLD" if own > 0 else "—"


def make_plan(fc: pd.DataFrame, shares: dict, capital: float) -> dict:
    """Target weights and whole-share orders for a forecast, computed the way `python -m trader trade` does."""
    tickers = fs.tickers
    prices = panel.close[tickers].iloc[-1]
    current = np.array([shares.get(t, 0.0) * prices[t] / capital for t in tickers])
    trader = make_sim_trader(strategy_params, cost_params, opt_params, trader_kind)
    pred, prob = fc["pred_ret"].to_numpy(), fc["prob_up"].to_numpy()
    cal = None
    if isinstance(trader, OptimalTrader):
        hist = preds.get(model, pd.DataFrame(columns=["date", "pred_ret", "y"])).dropna(subset=["pred_ret", "y"])
        hist = hist[hist["date"].isin(np.sort(hist["date"].unique())[-cfg.optimizer.calib_window:])]
        cal = trader.calibrate(hist["pred_ret"].to_numpy(), hist["y"].to_numpy())
    t = len(fs.dates) - 1
    cov = covariance(fs.past_ret[max(t - cfg.strategy.cov_lookback + 1, 0):t + 1])
    target = trader.target_weights(pred, prob, cov, current)
    plans = plan_orders(tickers, target, shares, prices.to_dict(), capital, 2 * capital)
    qty = np.array([p.quantity for p in plans])
    return {"trader": trader, "calibration": cal, "prices": prices, "current": current, "target": target,
            "qty": qty, "order_value": qty * prices.to_numpy()}


def model_forecast() -> pd.DataFrame | None:
    """The selected model's forecast for the next bar, or None if the transformer is not trained yet."""
    if model == "transformer" and not has_transformer:
        return None
    mtime = model_path(cfg).stat().st_mtime if model == "transformer" else None
    return forecast(cfg_path, model, str(fs.dates[-1]), mtime).reindex(fs.tickers)


def future_prediction():
    last = fs.dates[-1]
    nxt = next_bar(last, intraday)
    downloaded = datetime.fromtimestamp(prices_path(cfg.data.dir).stat().st_mtime)
    bar_len = pd.Timedelta(hours=1) if intraday else pd.Timedelta(hours=16, minutes=30)
    stale = pd.Timestamp(datetime.now(NY).replace(tzinfo=None)) > nxt + bar_len

    left, right = st.columns([3, 1])
    left.markdown(f"Latest data: the {bar} of **{last:{WHEN}}** · prices downloaded {downloaded:%Y-%m-%d %H:%M}")
    if right.button("🔄 Get latest prices", width="stretch",
                    help="Download fresh prices from Yahoo Finance and recompute the forecast."):
        with st.spinner("Downloading prices…"):
            fetch_prices(cfg)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    if stale:
        st.warning(f"Newer {bar}s exist than the data shown here, so this forecast is out of date. "
                   "Click **Get latest prices**.")

    fc = model_forecast()
    if fc is None:
        st.info("The transformer has not been trained yet. Run `python -m trader train` (cloud GPU), copy "
                "`models/` back here, and its forecast appears in this tab.")
        return

    if fs.execution == "next_open":
        timing = (f"Decided after the {bar} of {last:{WHEN}} closes. Trades happen at the open of the "
                  f"{bar} starting **{nxt:{WHEN}}**; the forecast is the return from that open to the open of the "
                  f"{bar} after it.")
    else:
        timing = f"Trades happen at the close of {last:{WHEN}}; the forecast is the return to the next close."
    st.subheader(f"{model}: forecast for the {bar} starting {nxt:{WHEN}}")
    st.caption(timing)

    tickers = fs.tickers
    with st.expander("Your current holdings (optional)"):
        st.caption(f"By default you start from cash with ${capital:,.0f}. Enter shares you already own and the "
                   "plan shows what to change.")
        holdings = st.data_editor(pd.DataFrame({"Shares you own": 0}, index=pd.Index(tickers, name="Stock")),
                                  column_config={"Shares you own": st.column_config.NumberColumn(min_value=0, step=1)},
                                  key=f"holdings_{cfg_path}")
    shares = holdings["Shares you own"].fillna(0).astype(float).to_dict()
    p = make_plan(fc, shares, capital)
    trader, prices, current, target, qty, order_value = (p["trader"], p["prices"], p["current"], p["target"],
                                                         p["qty"], p["order_value"])
    pred, prob = fc["pred_ret"].to_numpy(), fc["prob_up"].to_numpy()
    if p["calibration"] is not None and p["calibration"].slope == 0:
        st.info(f"OptimalTrader: {model}'s past forecasts ({p['calibration'].n} checked) showed no reliable skill, "
                "so it does not take positions.")

    one_way = (commission_bps + slippage_bps) / 1e4
    invested = float(np.sum(np.floor(np.abs(target) * capital / prices.to_numpy()) * prices.to_numpy()))
    m = st.columns(4)
    m[0].metric("Stocks to own", f"{int((target > 0).sum())} of {len(tickers)}")
    m[1].metric("Invested", f"${invested:,.0f}", f"{invested / capital:.0%} of capital", delta_color="off")
    m[2].metric("Kept in cash", f"${capital - invested:,.0f}")
    m[3].metric("Orders", f"{int((qty != 0).sum())}",
                f"~${np.abs(order_value).sum() * one_way:,.2f} in costs", delta_color="off")

    table = pd.DataFrame({
        "Price": prices.to_numpy(),
        "Predicted return": pred,
        "P(up)": prob,
        "Now": current,
        "Target": target,
        "Action": [action(q, shares[tk], w) for q, tk, w in zip(qty, tickers, target)],
        "Shares": np.abs(qty),
        "Order $": np.abs(order_value),
    }, index=pd.Index(tickers, name="Stock"))
    if isinstance(trader, OptimalTrader):
        table.insert(3, "Expected after calibration", trader.expected_returns(pred))
    else:
        hurdle = 2 * one_way + cfg.strategy.min_edge
        why = [rule_reason(p, q, shares[tk] > 0, hurdle) for p, q, tk in zip(pred, prob, tickers)]
        table["Why"] = [f"wants {w:.1%} of capital, less than one share (${px:,.0f})" if w > 0 and q == 0 and
                        shares[tk] == 0 else r for r, w, q, tk, px in zip(why, target, qty, tickers, prices)]
    table = table.sort_values("Predicted return", ascending=False)
    st.dataframe(table.style.format({"Price": "${:,.2f}", "Predicted return": "{:+.2%}", "P(up)": "{:.1%}",
                                     "Expected after calibration": "{:+.3%}", "Now": "{:.0%}", "Target": "{:.1%}",
                                     "Shares": "{:,.0f}", "Order $": "${:,.0f}"}, na_rep="–"),
                 width="stretch")
    st.caption("Whole shares only, so small targets on expensive stocks can round down to zero. The loss and "
               "drawdown limits need your account history and are applied by `python -m trader trade`, not here.")

    chart = table.reset_index()
    chart["Decision"] = np.where(chart["Target"] > 0, "In the portfolio", "Not bought")
    fig = px.bar(chart, x="Stock", y="Predicted return", color="Decision", custom_data=["P(up)"],
                 color_discrete_map={"In the portfolio": PALETTE[0], "Not bought": GREY},
                 title=f"Predicted return for the {bar} starting {nxt:{WHEN}}")
    fig.update_traces(hovertemplate="%{x}: predicted %{y:+.2%} · P(up) %{customdata[0]:.0%}<extra></extra>")
    if not isinstance(trader, OptimalTrader):
        fig.add_hline(y=2 * one_way, line_width=1, line_dash="dot", line_color=GREY,
                      annotation_text=f"trading cost {2 * one_way:.2%}", annotation_position="top right")
    fig.add_hline(y=0, line_width=1, line_color=GREY)
    fig.update_layout(yaxis_tickformat=".2%", height=380, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    if model in preds:
        scored = preds[model].dropna(subset=["y"])
        recent = scored[scored["date"].isin(np.sort(scored["date"].unique())[-60:])]
        hit = (np.sign(recent["pred_ret"]) == np.sign(recent["y"])).mean()
        st.caption(f"Track record: over its last 60 checked {bar}s, {model} called the direction right "
                   f"{hit:.1%} of the time (a coin flip is 50%). Forecasts are noisy; treat them as small tilts, "
                   "not certainties.")


# ---- Account ---------------------------------------------------------------------------------

def read_log(name: str, **kwargs) -> pd.DataFrame | None:
    path = Path(cfg.paths.logs) / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


def account():
    b = cfg.broker
    key = f"snapshot_{cfg_path}"
    left, right = st.columns([3, 1])
    if right.button("🔌 Refresh from IB", width="stretch",
                    help="Read-only connection to TWS / IB Gateway: reads the account, can never place orders."):
        asyncio.set_event_loop(asyncio.new_event_loop())    # ib_async needs an event loop in this thread
        try:
            with st.spinner(f"Connecting to TWS / IB Gateway at {b.host}:{b.port}…"):
                st.session_state[key] = account_snapshot(cfg)
        except Exception as e:
            st.error(f"Could not read the account from {b.host}:{b.port} ({type(e).__name__}: {e}). Is TWS or "
                     "IB Gateway running and logged in, with the API enabled on that port?")
    snap = st.session_state.get(key)
    if snap:
        kind = "paper" if snap["paper"] else "LIVE, real money"
        left.markdown(f"Account **{snap['account']}** ({kind}) · read from IB at {snap['time']:%Y-%m-%d %H:%M:%S}")
    else:
        left.markdown(f"Not connected to IB: showing what the trader logged in `{cfg.paths.logs}/`. Click "
                      "**Refresh from IB** while TWS or IB Gateway is running for live values.")

    equity = read_log("equity.csv", parse_dates=["date"])
    orders = read_log("orders.csv", parse_dates=["date"])
    fills = snap["fills"] if snap else load_fills(cfg)
    if equity is None and orders is None and fills.empty and not snap:
        st.info(f"The trader has not traded yet. To start on the paper account: log in to TWS (paper), enable the "
                f"API on port {b.port}, then run `python -m trader -c {cfg_path} trade --execute` "
                f"{'after each hourly bar' if intraday else 'once a day after the close'}. Its trades and "
                "earnings appear here.")
        return

    # -- money earned --
    curve = pd.Series(dtype=float)
    if equity is not None:
        curve = equity.drop_duplicates("date", keep="last").set_index("date")["nlv"].astype(float)
    if snap:
        curve.loc[pd.Timestamp(snap["time"])] = snap["nlv"]
    m = st.columns(4)
    if len(curve):
        m[0].metric("Account value", f"${curve.iloc[-1]:,.0f}")
    if len(curve) > 1:
        earned = curve.iloc[-1] - curve.iloc[0]
        m[1].metric(f"Earned since {curve.index[0]:%Y-%m-%d}", f"${earned:+,.0f}",
                    f"{curve.iloc[-1] / curve.iloc[0] - 1:+.2%}")
    if not fills.empty:
        m[2].metric("Profit from closed trades", f"${fills['realized_pnl'].sum():+,.0f}",
                    help="IB's realised P&L on sells (after commissions).")
        m[3].metric("Commissions paid", f"${fills['commission'].sum():,.2f}")
    if len(curve) > 1:
        fig = go.Figure()
        fig.add_scatter(x=curve.index, y=curve, mode="lines+markers", name="Account value",
                        line=dict(width=2, color=PALETTE[0]), marker=dict(size=8),
                        hovertemplate="%{x|%Y-%m-%d %H:%M}: $%{y:,.0f}<extra></extra>")
        fig.update_layout(title="Account value at each trader run", yaxis_tickprefix="$", height=340,
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    st.caption("Earned = change in the whole IB account's value, so deposits and withdrawals count too, and "
               "both the daily and the hourly trader share it if both run on one account.")

    # -- trades made --
    st.markdown("#### Trades made")
    if not fills.empty:
        f = fills.sort_values("time", ascending=False)
        f = f[["time", "symbol", "side", "shares", "price", "value", "commission", "realized_pnl"]]
        f.columns = ["Time", "Stock", "Side", "Shares", "Price", "Amount", "Commission", "Profit (closed)"]
        st.dataframe(f.style.format({"Time": lambda t: f"{t:%Y-%m-%d %H:%M}", "Shares": "{:,.0f}",
                                     "Price": "${:,.2f}", "Amount": "${:,.0f}", "Commission": "${:,.2f}",
                                     "Profit (closed)": "${:+,.2f}"}, na_rep="–"), width="stretch", hide_index=True)
    else:
        st.write("No fills recorded yet. IB reports executions for about a day, so they are saved on every "
                 "`trade` run and every **Refresh from IB**.")
    if orders is not None:
        with st.expander(f"Orders the trader sent ({len(orders)})"):
            o = orders.sort_values("date", ascending=False)[["date", "symbol", "quantity", "price", "target_weight"]]
            o.columns = ["Decided on", "Stock", "Shares (+buy / −sell)", "Price then", "Target weight"]
            st.dataframe(o.style.format({"Price then": "${:,.2f}", "Target weight": "{:.1%}"}),
                         width="stretch", hide_index=True)

    # -- trades that will be made --
    st.markdown("#### Trades that will be made")
    if not snap:
        st.write("Click **Refresh from IB** to see orders waiting at IB and the next run's plan for your real "
                 "positions. The **Future prediction** tab shows the plan starting from cash.")
    else:
        oo = snap["open_orders"]
        st.markdown("**Waiting at IB** (sent, not filled yet)")
        if oo.empty:
            st.write("No orders waiting.")
        else:
            oo = oo[["symbol", "action", "quantity", "filled", "type", "tif", "limit", "status"]]
            oo.columns = ["Stock", "Side", "Shares", "Filled", "Order type", "Valid", "Limit", "Status"]
            st.dataframe(oo.style.format({"Shares": "{:,.0f}", "Filled": "{:,.0f}", "Limit": "${:,.2f}"},
                                         na_rep="–"), width="stretch", hide_index=True)
            st.caption("While these wait, `trade --execute` refuses to send more orders for the same stocks, so "
                       "nothing is bought twice.")
        fc = model_forecast()
        st.markdown(f"**Next trader run** for the {bar} starting {next_bar(fs.dates[-1], intraday):{WHEN}}")
        if fc is None:
            st.write("The transformer is not trained yet, so there is no plan. Pick a baseline model to preview one.")
        else:
            shares = {r.symbol: float(r.shares) for r in snap["portfolio"].itertuples()}
            capital_now = min(snap["nlv"], b.max_capital) if b.max_capital else snap["nlv"]
            p = make_plan(fc, shares, capital_now)
            nxt = pd.DataFrame({"Stock": fs.tickers, "Side": np.where(p["qty"] > 0, "BUY", "SELL"),
                                "Shares": np.abs(p["qty"]), "Price now": p["prices"].to_numpy(),
                                "Amount": np.abs(p["order_value"]), "Weight now": p["current"],
                                "Target weight": p["target"]})[p["qty"] != 0]
            if nxt.empty:
                st.write("No changes: the current positions already match the plan.")
            else:
                st.dataframe(nxt.style.format({"Shares": "{:,.0f}", "Price now": "${:,.2f}", "Amount": "${:,.0f}",
                                               "Weight now": "{:.1%}", "Target weight": "{:.1%}"}),
                             width="stretch", hide_index=True)
            st.caption(f"Preview with **{model}** and the sidebar settings on trading capital ${capital_now:,.0f}. "
                       "`python -m trader trade` uses the transformer and the settings in the config file, so its "
                       "orders can differ.")

    # -- positions --
    if snap:
        st.markdown("#### Positions now")
        pf = snap["portfolio"]
        if pf.empty:
            st.write("No stock positions.")
        else:
            pf = pf.copy()
            pf.columns = ["Stock", "Shares", "Avg cost", "Price", "Value", "Open profit", "Closed profit"]
            st.dataframe(pf.style.format({"Shares": "{:,.0f}", "Avg cost": "${:,.2f}", "Price": "${:,.2f}",
                                          "Value": "${:,.0f}", "Open profit": "${:+,.0f}",
                                          "Closed profit": "${:+,.0f}"}, na_rep="–"),
                         width="stretch", hide_index=True)


tab_past, tab_future, tab_account = st.tabs(["💰 Past simulation", "🔮 Future prediction", "📒 Account & trades"])
with tab_past:
    past_simulation()
with tab_future:
    future_prediction()
with tab_account:
    account()
