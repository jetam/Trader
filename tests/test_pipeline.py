import numpy as np
import pandas as pd
import pytest

from trader.backtest import run_backtest
from trader.broker import Broker, SafetyError, plan_orders
from trader.config import Config, CostConfig, StrategyConfig
from trader.data import FIELDS, Panel
from trader.features import build_features
from trader.strategy import RiskState, Strategy
from trader.walkforward import folds

TICKERS = ["AAA", "BBB", "CCC"]
MARKET = {"broad": "SPY", "sector": "QQQ", "vol": "^VIX", "rates": "^TNX"}


def synthetic_panel(T=700, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)
    syms = TICKERS + list(MARKET.values())
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.015, (T, len(syms))), axis=0)), dates, syms)
    open_ = close.shift(1).fillna(close) * np.exp(rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    vol = pd.DataFrame(rng.integers(1e5, 1e6, close.shape).astype(float), dates, syms)
    return Panel(open=open_, high=high, low=low, close=close, volume=vol)


def features(panel, execution="next_open", lookback=20):
    return build_features(panel, TICKERS, MARKET, lookback=lookback, horizon=1, execution=execution)


@pytest.mark.parametrize("execution", ["next_open", "close"])
def test_features_have_no_lookahead(execution):
    panel = synthetic_panel()
    fs = features(panel, execution)
    k = 400
    shocked = Panel(**{f: getattr(panel, f).copy() for f in FIELDS})
    factors = np.linspace(2.0, 3.0, len(panel.dates) - k - 1)[:, None]
    for f in FIELDS:
        getattr(shocked, f).iloc[k + 1:] *= factors   # change everything after day k
    fs2 = features(shocked, execution)
    np.testing.assert_array_equal(fs.X[:k + 1], fs2.X[:k + 1])
    assert not np.allclose(fs.y[k], fs2.y[k])         # the target is about the future, so it must change


def test_target_alignment():
    panel = synthetic_panel()
    fs = features(panel, "next_open")
    t = 300
    o = panel.open["AAA"]
    assert fs.y[t, 0] == pytest.approx(np.log(o.iloc[t + 2] / o.iloc[t + 1]))
    assert fs.exec_ret[t, 0] == pytest.approx(o.iloc[t + 2] / o.iloc[t + 1] - 1)


def test_walkforward_folds_are_purged():
    fs = features(synthetic_panel(T=900))
    for train_hi, test_lo, test_hi in folds(fs, "2016-01-01", 12, embargo=5):
        # The last training target (row train_hi-1, spanning up to train_hi+horizon) ends before the test block.
        assert train_hi - 1 + fs.horizon + 1 < test_lo
        assert test_lo < test_hi


def test_strategy_respects_limits():
    cfg = StrategyConfig(max_position=0.2, max_gross=0.5, target_vol=1.0)
    s = Strategy(cfg, CostConfig())
    rng = np.random.default_rng(1)
    cov = np.diag(np.full(5, 0.02 ** 2))
    for _ in range(200):
        w = s.target_weights(rng.normal(0, 0.01, 5), rng.uniform(0, 1, 5), cov, rng.uniform(-0.3, 0.3, 5))
        assert (w >= 0).all()                       # long-only by default
        assert w.max() <= 0.2 + 1e-12
        assert np.abs(w).sum() <= 0.5 + 1e-12
    assert not s.target_weights(np.full(5, 0.05), np.full(5, 0.9), cov, np.full(5, 0.1), "max_drawdown").any()


def test_strategy_needs_edge_above_costs():
    s = Strategy(StrategyConfig(), CostConfig(commission_bps=5, slippage_bps=5))
    cov = np.diag([0.02 ** 2])
    assert s.target_weights(np.array([0.001]), np.array([0.9]), cov, np.zeros(1))[0] == 0   # below 20bp hurdle
    assert s.target_weights(np.array([0.01]), np.array([0.9]), cov, np.zeros(1))[0] > 0


def test_risk_state_drawdown_and_cooldown():
    cfg = StrategyConfig(max_drawdown=0.1, cooldown_days=3, daily_loss_limit=0.05)
    st = RiskState()
    assert st.update(1.05, 1.0, cfg) is None
    assert st.update(0.94, 1.05, cfg) == "max_drawdown"
    assert [st.update(0.94, 0.94, cfg) for _ in range(3)] == ["cooldown"] * 3
    assert st.update(0.95, 0.94, cfg) is None and st.peak == pytest.approx(0.95)
    assert st.update(0.90, 0.95, cfg) == "daily_loss"


def test_backtest_charges_costs():
    fs = features(synthetic_panel())
    dates = fs.dates[100:110]
    preds = pd.DataFrame([(d, t, 0.05 if t == "AAA" else -0.05, 0.9 if t == "AAA" else 0.1)
                          for d in dates for t in TICKERS], columns=["date", "ticker", "pred_ret", "prob_up"])
    costs = CostConfig(commission_bps=10, slippage_bps=0)
    res = run_backtest(preds, fs, Strategy(StrategyConfig(min_trade=0.0), costs))
    w0 = res.weights.iloc[0]
    assert w0["AAA"] > 0 and w0[["BBB", "CCC"]].eq(0).all()
    assert res.costs.iloc[0] == pytest.approx(w0["AAA"] * 10 / 1e4)
    assert res.returns.iloc[0] == pytest.approx(w0["AAA"] * fs.exec_ret[100, 0] - res.costs.iloc[0])


def test_plan_orders_caps_notional():
    plans = plan_orders(["AAA", "BBB"], [0.5, 0.0], {"BBB": 10}, {"AAA": 100.0, "BBB": 50.0}, 100_000, 20_000)
    assert plans[0].quantity == 200                  # 50k wanted, capped to 20k / $100
    assert plans[1].quantity == -10                  # close BBB


def test_broker_refuses_live_account(monkeypatch):
    b = Broker(Config().broker, read_only=False)
    b.account = "U1234567"
    monkeypatch.delenv("TRADER_ALLOW_LIVE", raising=False)
    with pytest.raises(SafetyError):
        b.check_can_trade()
    b.account = "DU1234567"
    b.check_can_trade()
    b.read_only = True
    with pytest.raises(SafetyError):
        b.check_can_trade()


def test_transformer_learns_a_planted_signal():
    torch = pytest.importorskip("torch")
    from trader.transformer import TransformerPredictor

    fs = features(synthetic_panel(T=900, seed=3), lookback=10)
    # Plant a signal: tomorrow's return has the sign of today's (standardised) RSI feature.
    k = fs.names.index("rsi14")
    sig = np.nan_to_num(fs.X[:, :, k])
    fs.y = np.where(np.isfinite(fs.y), 0.02 * np.sign(sig) + 0.002 * np.random.default_rng(0).normal(size=fs.y.shape),
                    np.nan)
    cfg = Config()
    cfg.training.epochs, cfg.training.patience, cfg.training.threads = 15, 15, 2
    cfg.model.d_model, cfg.model.n_layers = 32, 1
    t_idx, n_idx = fs.samples(0, 650)
    model = TransformerPredictor(cfg.model, cfg.training).fit(fs, t_idx, n_idx)
    te_t, te_n = fs.samples(660)
    pred, prob = model.predict(fs, te_t, te_n)
    y = fs.y[te_t, te_n]
    assert np.mean(np.sign(pred) == np.sign(y)) > 0.7
    assert np.mean((prob > 0.5) == (y > 0)) > 0.7
    assert torch.isfinite(torch.as_tensor(pred)).all()


def synthetic_hourly_long(days=60, seed=0) -> pd.DataFrame:
    """Long-format hourly bars: stocks at 9:30..15:30, VIX with extended hours, TNX on :20 offsets."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2025-01-06", periods=days):
        for sym in TICKERS + ["SPY", "QQQ", "^VIX", "^TNX"]:
            if sym == "^VIX":
                times = [d + pd.Timedelta(hours=h, minutes=15) for h in range(3, 16)]
            elif sym == "^TNX":
                times = [d + pd.Timedelta(hours=h, minutes=20) for h in range(8, 15)]
            else:
                times = [d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h) for h in range(7)]
            for t in times:
                c = 100 * np.exp(rng.normal(0, 0.01))
                rows.append((t, sym, c, c * 1.002, c * 0.998, c, float(rng.integers(1e4, 1e5))))
    df = pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    return df.set_index(["date", "symbol"]).sort_index()


def test_hourly_index_series_are_aligned_without_lookahead():
    long = synthetic_hourly_long(days=5)
    panel = Panel.from_long(long, calendar_symbol="QQQ", ffill_symbols=("^VIX", "^TNX"))
    assert sorted(set(panel.dates.strftime("%H:%M"))) == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30",
                                                         "15:30"]
    for sym in ("^VIX", "^TNX"):
        own = long["close"].xs(sym, level="symbol")
        for t, v in panel.close[sym].dropna().items():
            source = own[own == v].index
            assert (source <= t).any(), f"{sym} value at {t} came from a later bar"


def test_hourly_features_and_folds():
    from trader.walkforward import schedule
    from types import SimpleNamespace

    panel = Panel.from_long(synthetic_hourly_long(days=120), "QQQ", ("^VIX", "^TNX"))
    fs = build_features(panel, TICKERS, MARKET, lookback=14, horizon=1, execution="next_open", intraday=True)
    assert "tod" in fs.names
    tod = fs.X[:, 0, fs.names.index("tod")]
    assert tod.min() == pytest.approx(-0.5) and tod.max() < 0.5
    wf = SimpleNamespace(min_train_months=3, step_months=1, first_test_year=None, step_years=None, embargo_days=7)
    first, months = schedule(fs, wf)
    blocks = list(folds(fs, first, months, wf.embargo_days))
    assert len(blocks) >= 2
    for train_hi, test_lo, test_hi in blocks:
        assert train_hi - 1 + fs.horizon + 1 < test_lo


def test_incomplete_bars_are_dropped(monkeypatch):
    import trader.data as data

    class FakeNow:
        @staticmethod
        def now(tz=None):
            return pd.Timestamp("2025-01-06 11:45", tz="America/New_York").to_pydatetime()

    monkeypatch.setattr(data, "datetime", FakeNow)
    long = synthetic_hourly_long(days=1)
    kept = data._drop_incomplete_bars(long.xs("QQQ", level="symbol", drop_level=False), "1h")
    times = sorted(set(kept.index.get_level_values("date").strftime("%H:%M")))
    assert times == ["09:30", "10:30"]  # the 11:30 bar is still in progress until 12:30


# ---- OptimalTrader ---------------------------------------------------------------------------

def _optimal(**opt):
    from trader.config import OptimizerConfig
    from trader.optimal import OptimalTrader

    return OptimalTrader(StrategyConfig(max_position=0.2, max_gross=1.0, target_vol=0.15, min_trade=0.0),
                         CostConfig(), OptimizerConfig(**opt), 252)


def test_calibration_rewards_skill_and_ignores_noise():
    from trader.optimal import calibrate

    rng = np.random.default_rng(0)
    pred = rng.normal(0, 0.002, 5000)
    skilled = calibrate(pred, 0.5 * pred + rng.normal(0, 0.002, 5000))
    assert skilled.t_stat > 5 and 0.4 < skilled.slope < 0.6
    assert calibrate(pred, rng.normal(0, 0.02, 5000)).slope < 0.05 * 0.5    # noise: heavily shrunk
    assert calibrate(pred, -pred + rng.normal(0, 0.002, 5000)).slope == 0   # wrong-way: never inverted
    assert calibrate(pred[:50], pred[:50]).slope == 0                      # too little history


def test_optimal_trader_respects_limits_and_costs():
    trader = _optimal(risk_aversion=0.1)
    trader.calibration.slope = 1.0
    cov = np.diag(np.full(4, 0.02 ** 2)) + 0.0001
    w = trader.target_weights(np.array([0.01, 0.01, 0.01, -0.01]), np.full(4, 0.6), cov, np.zeros(4))
    assert (w >= 0).all() and w.max() <= 0.2 + 1e-9 and w.sum() <= 1.0 + 1e-9
    assert np.sqrt(w @ cov @ w * 252) <= 0.15 + 1e-4
    assert w[3] == 0 and w[0] > 0
    # An edge far below the trading cost is not worth moving out of cash for.
    tiny = trader.target_weights(np.full(4, 1e-6), np.full(4, 0.6), cov, np.zeros(4))
    assert not tiny.any()
    # No calibrated skill -> no positions.
    trader.calibration.slope = 0.0
    assert not trader.target_weights(np.full(4, 0.01), np.full(4, 0.9), cov, np.zeros(4)).any()


def test_optimal_backtest_calibrates_only_on_realised_outcomes():
    from trader.optimal import OptimalTrader

    fs = features(synthetic_panel())
    dates = fs.dates[200:400]
    rng = np.random.default_rng(0)
    preds = pd.DataFrame([(d, t, rng.normal(0, 0.01), 0.5) for d in dates for t in TICKERS],
                         columns=["date", "ticker", "pred_ret", "prob_up"])
    row = {d: i for i, d in enumerate(fs.dates)}
    preds["y"] = [fs.y[row[d], TICKERS.index(t)] for d, t in zip(preds["date"], preds["ticker"])]
    trader = _optimal(recalibrate_every=1)
    seen = []
    original = OptimalTrader.calibrate

    def spy(self, pred, realised):
        seen.append(len(pred))
        return original(self, pred, realised)

    trader.calibrate = spy.__get__(trader)
    run_backtest(preds, fs, trader)
    # First decision: nothing realised yet. Each later decision sees at most the rows realised by then.
    assert seen[0] == 0
    assert all(n <= max(k - fs.horizon, 0) * len(TICKERS) for k, n in enumerate(seen))


def test_next_bar_skips_nights_and_weekends():
    from trader.forecast import next_bar

    assert next_bar(pd.Timestamp("2026-10-23"), intraday=False) == pd.Timestamp("2026-10-26")       # Fri -> Mon
    assert next_bar(pd.Timestamp("2026-10-22 11:30"), intraday=True) == pd.Timestamp("2026-10-22 12:30")
    assert next_bar(pd.Timestamp("2026-10-23 15:30"), intraday=True) == pd.Timestamp("2026-10-26 09:30")


def test_latest_forecast_covers_the_last_row():
    from trader.config import Config
    from trader.forecast import latest_forecast

    fs = features(synthetic_panel())
    fc = latest_forecast(None, fs, "ridge", Config())
    assert list(fc.index) == TICKERS
    assert np.isfinite(fc["pred_ret"]).all() and fc["prob_up"].between(0, 1).all()


def test_fill_log_dedupes_and_keeps_only_this_trader(tmp_path):
    from trader.broker import FILL_COLUMNS
    from trader.config import Config
    from trader.live import load_fills, sync_fills

    cfg = Config()
    cfg.paths.logs = str(tmp_path)
    cfg.broker.client_id = 17

    def fill(exec_id, client, commission):
        return dict(zip(FILL_COLUMNS, [pd.Timestamp("2026-10-26 09:30"), exec_id, "AAA", "BUY", 10, 100.0, 1000.0,
                                       commission, np.nan, client]))

    class FakeBroker:
        def __init__(self, rows):
            self.rows = rows

        def fills(self):
            return pd.DataFrame(self.rows, columns=FILL_COLUMNS)

    sync_fills(cfg, FakeBroker([fill("e1", 17, np.nan), fill("e2", 99, 1.0)]))
    out = sync_fills(cfg, FakeBroker([fill("e1", 17, 1.0)]))     # same fill again, now with its commission
    assert list(out["exec_id"]) == ["e1"]
    assert load_fills(cfg)["commission"].tolist() == [1.0]


def test_diebold_mariano_counts_each_date_once():
    from trader.metrics import diebold_mariano

    rng = np.random.default_rng(0)
    day = np.repeat(rng.normal(0, 1, 300), 8)            # 8 stocks share one day's shock
    e_model = 1 + 0.2 * day + rng.normal(0, 0.05, day.size)
    dates = np.repeat(np.arange(300), 8)
    pooled, _ = diebold_mariano(e_model, np.ones_like(e_model))
    by_date, _ = diebold_mariano(e_model, np.ones_like(e_model), dates=dates)
    assert abs(by_date) < abs(pooled)                     # correlated same-day samples are not 8 observations
