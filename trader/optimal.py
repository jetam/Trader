"""OptimalTrader: chooses the trades that are optimal given calibrated forecasts, risk and costs.

Instead of fixed buy rules it solves, every period,

    maximise   mu' w  -  (risk_aversion / 2) w' Sigma w  -  cost' |w - w_current|
    subject to 0 <= w_i <= max_position         (or -max_position.. if shorting is allowed)
               sum |w_i| <= max_gross
               annualised volatility of w <= target_vol

where `mu` are expected returns. Raw model outputs are not trusted as-is: they are calibrated on the
model's own track record (past predictions vs realised returns, using only outcomes already known).
If the past predictions carried no reliable information, mu shrinks to zero and the trader stays in
cash rather than acting on noise.

It has the same interface as Strategy, so the backtester, the GUI and live trading can use either.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .strategy import ANN, Strategy


@dataclass
class Calibration:
    intercept: float = 0.0
    slope: float = 0.0      # expected return per unit of predicted return, after shrinkage
    raw_slope: float = 0.0
    t_stat: float = 0.0
    n: int = 0


def calibrate(pred: np.ndarray, realised: np.ndarray, shrink_k: float = 4.0, use_drift: bool = False,
              min_obs: int = 100) -> Calibration:
    """Regress realised returns on predictions; shrink the slope by its statistical reliability.

    shrink = t^2 / (t^2 + k): t = 1 keeps 20% of the slope (k=4), t = 2 keeps 50%, t = 4 keeps 80%.
    A negative slope (the model is wrong-way) is set to zero rather than traded in reverse.
    """
    m = np.isfinite(pred) & np.isfinite(realised)
    x, y = pred[m], realised[m]
    n = len(x)
    if n < min_obs or x.std() == 0:
        return Calibration(n=n)
    xc = x - x.mean()
    b = (xc @ (y - y.mean())) / (xc @ xc)
    a = y.mean() - b * x.mean()
    resid = y - a - b * x
    se = np.sqrt(resid @ resid / (n - 2) / (xc @ xc))
    t = b / se if se > 0 else 0.0
    shrink = t * t / (t * t + shrink_k) if b > 0 else 0.0
    slope = b * shrink
    intercept = (y.mean() - slope * x.mean()) if use_drift else 0.0
    return Calibration(intercept, slope, b, t, n)


class OptimalTrader(Strategy):
    def __init__(self, cfg, costs, opt_cfg, periods_per_year: int = ANN):
        super().__init__(cfg, costs, periods_per_year)
        self.opt = opt_cfg
        self.calibration = Calibration()

    def calibrate(self, pred: np.ndarray, realised: np.ndarray) -> Calibration:
        self.calibration = calibrate(pred, realised, self.opt.shrink_k, self.opt.use_drift)
        return self.calibration

    def expected_returns(self, pred: np.ndarray) -> np.ndarray:
        c = self.calibration
        return np.where(np.isfinite(pred), c.intercept + c.slope * pred, 0.0)

    def target_weights(self, pred, prob, cov, current, halt_reason=None) -> np.ndarray:
        current = np.nan_to_num(np.asarray(current, dtype=float))
        if halt_reason:
            return np.zeros_like(current)
        mu = self.expected_returns(np.asarray(pred, dtype=float))
        known = np.isfinite(pred)
        w = self.optimize(mu, cov, current, known)
        small = (np.abs(w - current) < self.cfg.min_trade) & (w != 0)
        return self._cap(np.where(small, current, w))

    def optimize(self, mu: np.ndarray, cov: np.ndarray, w0: np.ndarray, known: np.ndarray) -> np.ndarray:
        c, o = self.cfg, self.opt
        N = len(mu)
        # A position is held for several periods on average, so its cost is spread over them.
        tc = self.costs.one_way / max(o.holding_periods, 1)
        lam = o.risk_aversion
        scale = 1e4                                   # work in basis points for solver accuracy
        upper = np.where(known, c.max_position, 0.0)
        short_upper = upper if c.allow_short else np.zeros(N)
        w0c = np.clip(w0, -c.max_position, c.max_position)
        if not c.allow_short:
            w0c = np.maximum(w0c, 0.0)

        # Variables z = [p, n, b, s]: w = p - n (long/short parts), w - w0 = b - s (buys/sells).
        def split(z):
            return z[:N], z[N:2 * N], z[2 * N:3 * N], z[3 * N:]

        def objective(z):
            p, n, b, s = split(z)
            w = p - n
            f = -mu @ w + 0.5 * lam * w @ cov @ w + tc * (b.sum() + s.sum()) + 1e-7 * (p.sum() + n.sum())
            gw = -mu + lam * cov @ w
            grad = np.concatenate([gw + 1e-7, -gw + 1e-7, np.full(N, tc), np.full(N, tc)])
            return f * scale, grad * scale

        eye = np.eye(N)
        A_eq = np.hstack([eye, -eye, -eye, eye])
        cons = [
            {"type": "eq", "fun": lambda z: A_eq @ z - w0, "jac": lambda z: A_eq},
            {"type": "ineq", "fun": lambda z: c.max_gross - z[:2 * N].sum(),
             "jac": lambda z: np.concatenate([-np.ones(2 * N), np.zeros(2 * N)])},
            {"type": "ineq",
             "fun": lambda z: scale * (c.target_vol ** 2 / self.ann - (z[:N] - z[N:2 * N]) @ cov @ (z[:N] - z[N:2 * N])),
             "jac": lambda z: scale * np.concatenate([-2 * cov @ (z[:N] - z[N:2 * N]), 2 * cov @ (z[:N] - z[N:2 * N]),
                                                       np.zeros(2 * N)])},
        ]
        trade_cap = 2 * c.max_position + np.abs(w0)
        bounds = ([(0, u) for u in upper] + [(0, u) for u in short_upper]
                  + [(0, t) for t in trade_cap] + [(0, t) for t in trade_cap])
        x0 = np.concatenate([np.maximum(w0c, 0), np.maximum(-w0c, 0), np.maximum(w0c - w0, 0),
                             np.maximum(w0 - w0c, 0)])
        res = minimize(objective, x0, jac=True, bounds=bounds, constraints=cons, method="SLSQP",
                       options={"maxiter": 200, "ftol": 1e-9})
        if not res.success:
            # Fall back to the feasible starting point (current positions, clipped to the limits).
            return self._cap(w0c)
        p, n, _, _ = split(res.x)
        w = p - n
        w[np.abs(w) < 1e-4] = 0.0
        return self._cap(w)


def make_trader(cfg, kind: str | None = None):
    """The trading policy from config: 'optimal' -> OptimalTrader, 'rules' -> rule-based Strategy."""
    kind = kind or cfg.optimizer.trader
    if kind == "optimal":
        return OptimalTrader(cfg.strategy, cfg.costs, cfg.optimizer, cfg.data.periods_per_year)
    if kind == "rules":
        return Strategy(cfg.strategy, cfg.costs, cfg.data.periods_per_year)
    raise ValueError(f"optimizer.trader must be 'rules' or 'optimal', got {kind!r}")
