"""Stage 6: turn predictions into target portfolio weights under hard risk limits.

The same Strategy object is used by the backtester and by live trading, so what is tested is what
is traded.

Decision rule for each stock:
    long  if P(up) >= prob_threshold and expected return > round-trip cost + min_edge
    short if shorting allowed and the mirror conditions hold
    keep an existing position while the forecast still points the same way (hysteresis)
Sizing: signal strength / volatility, scaled so the portfolio hits `target_vol` using the trailing
covariance matrix, then capped per stock (`max_position`) and in total (`max_gross`).
Loss limits: a day losing more than `daily_loss_limit` -> flat for the next decision; a drawdown
beyond `max_drawdown` -> flat for `cooldown_days`, after which the high-water mark is reset.
"""

from dataclasses import asdict, dataclass

import numpy as np

ANN = 252


@dataclass
class RiskState:
    peak: float = 1.0
    halted_days: int = 0

    def update(self, equity: float, prev_equity: float | None, cfg) -> str | None:
        """Feed the latest known equity, once per bar. The 'daily' loss limit and cooldown therefore
        apply per bar on intraday data. Returns a reason to be flat, or None if trading is allowed."""
        if self.halted_days > 0:
            self.halted_days -= 1
            if self.halted_days == 0:
                self.peak = equity
            return "cooldown"
        self.peak = max(self.peak, equity)
        if equity / self.peak - 1 <= -cfg.max_drawdown:
            self.halted_days = cfg.cooldown_days
            return "max_drawdown"
        if prev_equity and equity / prev_equity - 1 <= -cfg.daily_loss_limit:
            return "daily_loss"
        return None

    def to_dict(self) -> dict:
        return asdict(self)


def covariance(past_ret: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Daily covariance of a (days, N) return window, NaN-robust; unknown entries get a safe value."""
    N = past_ret.shape[1]
    cov = np.full((N, N), np.nan)
    ok = np.isfinite(past_ret)
    for i in range(N):
        for j in range(i, N):
            m = ok[:, i] & ok[:, j]
            if m.sum() >= min_periods:
                a, b = past_ret[m, i], past_ret[m, j]
                cov[i, j] = cov[j, i] = np.mean((a - a.mean()) * (b - b.mean()))
    diag = np.diag(cov).copy()
    fallback = np.nanmedian(diag) if np.isfinite(diag).any() else (0.02 ** 2)
    diag[~np.isfinite(diag)] = fallback
    cov[~np.isfinite(cov)] = 0.0
    np.fill_diagonal(cov, diag)
    return cov


class Strategy:
    def __init__(self, cfg, costs, periods_per_year: int = ANN):
        self.cfg = cfg
        self.costs = costs
        self.ann = periods_per_year

    def _cap(self, w: np.ndarray) -> np.ndarray:
        c = self.cfg
        w = np.clip(w, -c.max_position, c.max_position)
        if not c.allow_short:
            w = np.maximum(w, 0.0)
        gross = np.abs(w).sum()
        if gross > c.max_gross:
            w = w * c.max_gross / gross
        return w

    def target_weights(self, pred: np.ndarray, prob: np.ndarray, cov: np.ndarray,
                       current: np.ndarray, halt_reason: str | None = None) -> np.ndarray:
        """All arrays are aligned to the same ticker order. `cov` is the daily covariance."""
        c = self.cfg
        current = np.nan_to_num(np.asarray(current, dtype=float))
        if halt_reason:
            return np.zeros_like(current)
        known = np.isfinite(pred) & np.isfinite(prob)
        pred = np.where(known, pred, 0.0)
        prob = np.where(known, prob, 0.5)
        vol = np.sqrt(np.diag(cov))
        hurdle = self.costs.round_trip + c.min_edge

        go_long = known & (prob >= c.prob_threshold) & (pred > hurdle)
        go_long |= known & (current > 0) & (prob > 0.5) & (pred > 0)
        go_short = np.zeros_like(go_long)
        if c.allow_short:
            go_short = known & (prob <= 1 - c.prob_threshold) & (pred < -hurdle)
            go_short |= known & (current < 0) & (prob < 0.5) & (pred < 0)

        strength = np.abs(pred) / vol
        raw = np.where(go_long, strength, 0.0) - np.where(go_short, strength, 0.0)
        if not raw.any():
            w = np.zeros_like(current)
        else:
            port_vol = np.sqrt(max(raw @ cov @ raw, 1e-18) * self.ann)
            w = self._cap(raw * c.target_vol / port_vol)

        # Turnover control: skip small adjustments, but always allow fully closing a position.
        small = (np.abs(w - current) < c.min_trade) & (w != 0)
        w = np.where(small, current, w)
        return self._cap(w)
