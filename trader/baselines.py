"""Stage 3: statistical baseline models.

All predictors share one interface so they can be compared fairly in the same walk-forward:

    fit(fs, t_idx, n_idx)               train on samples (t_idx[i], n_idx[i]) of a FeatureSet
    predict(fs, t_idx, n_idx) -> (pred_ret, prob_up)

Regression targets are volatility-normalised (y / vol20) so that calm and turbulent periods and
stocks carry equal weight; predictions are scaled back to return units.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureSet

CLIP = 5.0


def normalised_target(fs: FeatureSet, t_idx, n_idx) -> np.ndarray:
    return np.clip(fs.y[t_idx, n_idx] / fs.vol[t_idx, n_idx], -CLIP, CLIP)


class ZeroModel:
    """Random walk: expected return zero, P(up) = training frequency of up days."""

    name = "zero"

    def fit(self, fs, t_idx, n_idx):
        self.p_up = float(np.mean(fs.y[t_idx, n_idx] > 0))
        return self

    def predict(self, fs, t_idx, n_idx):
        return np.zeros(len(t_idx)), np.full(len(t_idx), self.p_up)


class HistMeanModel:
    """Each stock's historical mean return (the classic hard-to-beat benchmark)."""

    name = "hist_mean"

    def fit(self, fs, t_idx, n_idx):
        N = len(fs.tickers)
        y = fs.y[t_idx, n_idx]
        self.mean = np.array([y[n_idx == n].mean() if (n_idx == n).any() else 0.0 for n in range(N)])
        self.p_up = np.array([(y[n_idx == n] > 0).mean() if (n_idx == n).any() else 0.5 for n in range(N)])
        return self

    def predict(self, fs, t_idx, n_idx):
        return self.mean[n_idx], self.p_up[n_idx]


class TabularModel:
    """Any sklearn regressor + classifier on the last day's features."""

    def __init__(self, name, regressor, classifier, features=None):
        self.name = name
        self.reg = regressor
        self.clf = classifier
        self.features = features

    def _x(self, fs, t_idx, n_idx):
        X = fs.last_features(t_idx, n_idx)
        if self.features is not None:
            X = X[:, [fs.names.index(f) for f in self.features]]
        return X

    def fit(self, fs, t_idx, n_idx):
        X = self._x(fs, t_idx, n_idx)
        yn = normalised_target(fs, t_idx, n_idx)
        self.reg.fit(X, yn)
        self.clf.fit(X, fs.y[t_idx, n_idx] > 0)
        return self

    def predict(self, fs, t_idx, n_idx):
        X = self._x(fs, t_idx, n_idx)
        pred = self.reg.predict(X) * fs.vol[t_idx, n_idx]
        prob = self.clf.predict_proba(X)[:, 1]
        return pred, prob


def make_baseline(name: str, seed: int = 0):
    if name == "zero":
        return ZeroModel()
    if name == "hist_mean":
        return HistMeanModel()
    if name == "momentum":
        # Linear model on trailing returns only: tests reversal / momentum effects.
        feats = ["r1", "r5", "r20", "r60"]
        return TabularModel(name, make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
                            make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000)), feats)
    if name == "ridge":
        return TabularModel(name, make_pipeline(StandardScaler(), Ridge(alpha=1000.0)),
                            make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000)))
    if name == "gbm":
        return TabularModel(
            name,
            HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_leaf_nodes=15,
                                          min_samples_leaf=200, l2_regularization=1.0, loss="absolute_error",
                                          random_state=seed),
            HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_leaf_nodes=15,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=seed),
        )
    raise ValueError(f"unknown baseline {name!r}")


BASELINES = ["zero", "hist_mean", "momentum", "ridge", "gbm"]
