"""Forecast for the next bar, without a broker: the latest data and a model fitted on every known outcome.

Baselines are refitted on all rows whose target is already realised (cheap). The transformer is the
production model saved by `python -m trader train`.
"""

import numpy as np
import pandas as pd

from .data import Panel, next_bar  # noqa: F401  (re-exported for the GUI)
from .features import FeatureSet, build_features
from .live import model_path
from .transformer import TransformerPredictor
from .walkforward import make_model


def latest_forecast(panel: Panel, fs: FeatureSet, model_name: str, cfg) -> pd.DataFrame:
    """Predicted return and P(up) per ticker for the bar after the last one in `fs`."""
    if model_name == "transformer":
        model = TransformerPredictor.load(model_path(cfg))
        meta = model.meta
        if meta.get("interval", "1d") != cfg.data.interval:
            raise ValueError(f"transformer was trained on {meta.get('interval')} bars, config uses {cfg.data.interval}")
        if model.lookback != fs.lookback or list(model.tickers) != list(fs.tickers):
            fs = build_features(panel, model.tickers, cfg.data.market, model.lookback, meta["horizon"],
                                meta["execution"], cfg.data.intraday)
    else:
        t_idx, n_idx = fs.samples()
        model = make_model(model_name, cfg).fit(fs, t_idx, n_idx)
    t = len(fs.dates) - 1
    n_idx = np.nonzero(fs.window_ok()[t])[0]
    pred = np.full(len(fs.tickers), np.nan)
    prob = np.full(len(fs.tickers), np.nan)
    if len(n_idx):
        pred[n_idx], prob[n_idx] = model.predict(fs, np.full(len(n_idx), t), n_idx)
    return pd.DataFrame({"pred_ret": pred, "prob_up": prob}, index=pd.Index(fs.tickers, name="ticker"))
