"""Feature engineering and targets.

Every feature at row t uses only data available at the close of day t. The target at row t is
the forward log return over `horizon` days measured at the execution price:

    execution='close':     log(C[t+h] / C[t])
    execution='next_open': log(O[t+1+h] / O[t+1])   (decide after close t, trade at the next open)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Panel

TICKER_FEATURES = [
    "r1", "r5", "r20", "r60",           # trailing log returns
    "vol20", "vol60", "vol_ratio",       # realised volatility and its term structure
    "hl", "oc", "gap",                   # intraday range, body, overnight gap
    "volume_z",                          # abnormal volume
    "rsi14", "ma50", "ma200",            # oscillator and trend distance
    "rel_r1", "rel_r20", "beta60",       # behaviour relative to the sector ETF
]
MARKET_FEATURES = [
    "spy_r1", "qqq_r1", "qqq_r20", "qqq_vol20",
    "vix_log", "vix_chg", "rates_chg", "dow",
]
INTRADAY_FEATURES = ["tod"]          # time of day: intraday volume and volatility are U-shaped
FEATURE_NAMES = TICKER_FEATURES + MARKET_FEATURES


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down)


def execution_prices(panel: Panel, symbols: list, execution: str) -> pd.DataFrame:
    """Price at which a decision taken at row t is executed, aligned to row t."""
    if execution == "close":
        return panel.close[symbols]
    return panel.open[symbols].shift(-1)


def execution_returns(panel: Panel, symbols: list, execution: str) -> pd.DataFrame:
    """Simple return earned by a position entered for decision row t and exited at row t+1."""
    px = execution_prices(panel, symbols, execution)
    return px.shift(-1) / px - 1


@dataclass
class FeatureSet:
    dates: pd.DatetimeIndex
    tickers: list
    names: list
    X: np.ndarray          # (T, N, F) float32, NaN where unavailable
    y: np.ndarray          # (T, N) forward log return over the horizon (NaN at the end)
    vol: np.ndarray        # (T, N) trailing 20d daily vol of log returns, used to scale targets
    exec_ret: np.ndarray   # (T, N) simple 1-period return at execution prices (for backtests)
    past_ret: np.ndarray   # (T, N) simple close-to-close return known at t (for covariances)
    lookback: int
    horizon: int
    execution: str

    def window_ok(self) -> np.ndarray:
        """(T, N) True where the trailing `lookback` rows are all fully populated."""
        row_ok = np.isfinite(self.X).all(-1) & np.isfinite(self.vol) & (self.vol > 0)
        cs = np.concatenate([np.zeros((1, row_ok.shape[1])), np.cumsum(row_ok, axis=0)])
        ok = np.zeros_like(row_ok)
        L = self.lookback
        ok[L - 1:] = (cs[L:] - cs[:-L]) == L
        return ok

    def samples(self, t_lo: int = 0, t_hi: int | None = None, require_target: bool = True):
        """(t_idx, n_idx) of usable samples with t in [t_lo, t_hi)."""
        ok = self.window_ok()
        if require_target:
            ok &= np.isfinite(self.y)
        ok[:t_lo] = False
        if t_hi is not None:
            ok[t_hi:] = False
        t_idx, n_idx = np.nonzero(ok)
        return t_idx, n_idx

    def last_features(self, t_idx, n_idx) -> np.ndarray:
        return self.X[t_idx, n_idx]

    def to_frame(self, t_idx, n_idx, **cols) -> pd.DataFrame:
        df = pd.DataFrame({"date": self.dates[t_idx], "ticker": np.asarray(self.tickers)[n_idx], **cols})
        return df


def build_features(panel: Panel, tickers: list, market: dict, lookback: int, horizon: int,
                   execution: str, intraday: bool = False) -> FeatureSet:
    """Rolling windows are counted in bars, so on hourly data `r20` is a 20-hour return, etc."""
    tickers = [t for t in tickers if t in panel.close.columns]
    C, O, H, L, V = (getattr(panel, f)[tickers] for f in ("close", "open", "high", "low", "volume"))
    lr = np.log(C).diff()
    qqq_px = panel.close[market["sector"]]
    qqq = np.log(qqq_px).diff()
    spy = np.log(panel.close[market["broad"]]).diff()
    vix = np.log(panel.close[market["vol"]])
    rates = panel.close[market["rates"]]

    vol20 = lr.rolling(20).std()
    vol60 = lr.rolling(60).std()
    lv = np.log(V.where(V > 0))
    if intraday:
        # Compare volume with the same hour on previous days, not with the (different) hours just before.
        slot = pd.Series(panel.dates.hour * 60 + panel.dates.minute, index=panel.dates)
        by_slot = lv.groupby(slot.to_numpy())
        lv_mean = by_slot.transform(lambda x: x.rolling(20, min_periods=10).mean())
        lv_std = by_slot.transform(lambda x: x.rolling(20, min_periods=10).std())
    else:
        lv_mean, lv_std = lv.rolling(20).mean(), lv.rolling(20).std()
    f = {
        "r1": lr,
        "r5": lr.rolling(5).sum(),
        "r20": lr.rolling(20).sum(),
        "r60": lr.rolling(60).sum(),
        "vol20": vol20,
        "vol60": vol60,
        "vol_ratio": np.log(vol20 / vol60),
        "hl": np.log(H / L),
        "oc": np.log(C / O),
        "gap": np.log(O / C.shift(1)),
        "volume_z": (lv - lv_mean) / lv_std,
        "rsi14": rsi(C) / 100 - 0.5,
        "ma50": C / C.rolling(50).mean() - 1,
        "ma200": C / C.rolling(200).mean() - 1,
        "rel_r1": lr.sub(qqq, axis=0),
        "rel_r20": lr.rolling(20).sum().sub(qqq.rolling(20).sum(), axis=0),
        "beta60": pd.DataFrame({t: lr[t].rolling(60).cov(qqq) for t in tickers}).div(qqq.rolling(60).var(), axis=0),
    }
    m = {
        "spy_r1": spy,
        "qqq_r1": qqq,
        "qqq_r20": qqq.rolling(20).sum(),
        "qqq_vol20": qqq.rolling(20).std(),
        "vix_log": vix,
        "vix_chg": vix.diff(),
        "rates_chg": rates.diff(),
        "dow": pd.Series(panel.dates.dayofweek / 4 - 0.5, index=panel.dates),
        "tod": pd.Series((panel.dates.hour * 60 + panel.dates.minute - 570) / 390 - 0.5, index=panel.dates),
    }
    market_names = MARKET_FEATURES + (INTRADAY_FEATURES if intraday else [])
    T, N = C.shape
    per_ticker = np.stack([f[k].to_numpy(dtype=float) for k in TICKER_FEATURES], axis=-1)
    mkt = np.stack([m[k].to_numpy(dtype=float) for k in market_names], axis=-1)
    X = np.concatenate([per_ticker, np.broadcast_to(mkt[:, None, :], (T, N, mkt.shape[-1]))], axis=-1)
    X[~np.isfinite(X)] = np.nan

    px = execution_prices(panel, tickers, execution)
    y = np.log(px.shift(-horizon) / px).to_numpy(dtype=float)
    return FeatureSet(
        dates=panel.dates,
        tickers=tickers,
        names=TICKER_FEATURES + market_names,
        X=X.astype(np.float32),
        y=y,
        vol=vol20.to_numpy(dtype=float),
        exec_ret=execution_returns(panel, tickers, execution).to_numpy(dtype=float),
        past_ret=C.pct_change(fill_method=None).to_numpy(dtype=float),
        lookback=lookback,
        horizon=horizon,
        execution=execution,
    )


def features_from_config(panel: Panel, cfg) -> FeatureSet:
    fc = cfg.features
    return build_features(panel, cfg.data.tickers, cfg.data.market, fc.lookback, fc.horizon, fc.execution,
                          cfg.data.intraday)
