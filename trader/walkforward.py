"""Stage 5a: walk-forward out-of-sample predictions.

For each test block (calendar years), every model is trained only on data that ends
`horizon + 1 + embargo_days` rows before the block starts, then predicts the block. Stitching the
blocks gives a fully out-of-sample prediction history that all models share.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import make_baseline
from .features import FeatureSet
from .transformer import TransformerPredictor

log = logging.getLogger(__name__)


def make_model(name: str, cfg):
    if name == "transformer":
        return TransformerPredictor(cfg.model, cfg.training)
    return make_baseline(name, seed=cfg.training.seed)


def folds(fs: FeatureSet, first_test: str, step_months: int, embargo: int):
    """Yield (train_hi, test_lo, test_hi) row indices; train uses rows < train_hi."""
    dates = fs.dates
    purge = fs.horizon + 1 + embargo
    lo = pd.Timestamp(first_test)
    while lo <= dates[-1]:
        hi = lo + pd.DateOffset(months=step_months)
        in_block = np.nonzero((dates >= lo) & (dates < hi))[0]
        if len(in_block):
            yield max(in_block[0] - purge, 0), in_block[0], in_block[-1] + 1
        lo = hi


def schedule(fs: FeatureSet, wf) -> tuple:
    """(first test date, block length in months) from the walk-forward config."""
    if wf.min_train_months is not None:
        first = (fs.dates[0] + pd.DateOffset(months=wf.min_train_months)).normalize()
        return first, wf.step_months or 1
    return pd.Timestamp(f"{wf.first_test_year}-01-01"), wf.step_months or 12 * wf.step_years


def walk_forward(fs: FeatureSet, model_name: str, cfg) -> pd.DataFrame:
    wf = cfg.walkforward
    parts = []
    first, months = schedule(fs, wf)
    for train_hi, test_lo, test_hi in folds(fs, first, months, wf.embargo_days):
        tr_t, tr_n = fs.samples(0, train_hi)
        te_t, te_n = fs.samples(test_lo, test_hi, require_target=False)
        if len(tr_t) == 0 or len(te_t) == 0:
            continue
        t0 = time.time()
        model = make_model(model_name, cfg).fit(fs, tr_t, tr_n)
        pred, prob = model.predict(fs, te_t, te_n)
        part = fs.to_frame(te_t, te_n, pred_ret=pred, prob_up=prob, y=fs.y[te_t, te_n],
                           fold=str(fs.dates[test_lo].date()))
        parts.append(part)
        log.info("%-11s fold %s: train %6d  test %5d  (%.1fs)", model_name, fs.dates[test_lo].date(),
                 len(tr_t), len(te_t), time.time() - t0)
    return pd.concat(parts, ignore_index=True)


def predictions_path(cfg, model_name: str) -> Path:
    return Path(cfg.paths.artifacts) / "predictions" / f"{model_name}.parquet"


def save_predictions(df: pd.DataFrame, cfg, model_name: str) -> Path:
    path = predictions_path(cfg, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_all_predictions(cfg, fs: FeatureSet | None = None) -> dict:
    """Saved predictions per model. With `fs`, realised targets are re-read from it, so models whose
    predictions were made at different times (e.g. on a cloud machine) are scored on the same outcomes."""
    d = Path(cfg.paths.artifacts) / "predictions"
    preds = {p.stem: pd.read_parquet(p).assign(date=lambda df: pd.to_datetime(df["date"]))
             for p in sorted(d.glob("*.parquet"))}
    if fs is not None:
        y = pd.DataFrame(fs.y, index=fs.dates, columns=fs.tickers).stack(future_stack=True).rename("y")
        y.index.names = ["date", "ticker"]
        preds = {k: df.drop(columns="y").merge(y.reset_index(), on=["date", "ticker"], how="inner")
                 for k, df in preds.items()}
    return preds
