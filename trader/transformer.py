"""Stage 4: transformer that reads a window of daily market features and predicts the next return.

Input per sample: the last `lookback` days of features for one stock (its own features plus market
context), shape (L, F). A learned summary token, tagged with a stock embedding, attends over the
window; two heads read it out:

    return head  -> volatility-normalised forward return (Huber loss)
    up/down head -> logit of P(return > 0) (binary cross-entropy)
"""

import copy
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .features import FeatureSet

log = logging.getLogger(__name__)
CLIP = 5.0


class MarketTransformer(nn.Module):
    def __init__(self, n_features: int, n_tickers: int, lookback: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, ff_mult: int = 2, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, d_model), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.randn(1, lookback + 1, d_model) * 0.02)
        self.summary = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ticker = nn.Embedding(n_tickers, d_model)
        nn.init.normal_(self.ticker.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * ff_mult, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head_ret = nn.Linear(d_model, 1)
        self.head_up = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, ticker: torch.Tensor):
        """x: (B, L, F) standardised features; ticker: (B,) index. Returns (ret_norm, up_logit)."""
        h = self.inp(x)
        s = self.summary.expand(x.shape[0], -1, -1) + self.ticker(ticker)[:, None, :]
        h = torch.cat([h, s], dim=1) + self.pos
        z = self.norm(self.encoder(h)[:, -1])
        return self.head_ret(z).squeeze(-1), self.head_up(z).squeeze(-1)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class TransformerPredictor:
    """Trains/predicts a (possibly ensembled) MarketTransformer with the shared predictor interface."""

    name = "transformer"

    def __init__(self, model_cfg, train_cfg, seed: int | None = None):
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.seed = train_cfg.seed if seed is None else seed
        self.device = pick_device(train_cfg.device)
        torch.set_num_threads(train_cfg.threads)
        self.models: list = []
        self.mu = self.sd = None
        self.history: list = []

    # -- data --------------------------------------------------------------------------------

    def _standardise(self, fs: FeatureSet) -> torch.Tensor:
        Xn = (fs.X - self.mu) / self.sd
        Xn = np.nan_to_num(np.clip(Xn, -CLIP, CLIP), nan=0.0)
        return torch.as_tensor(Xn, dtype=torch.float32, device=self.device)

    def _batch(self, Xn, t_idx, n_idx):
        L = self.lookback
        t = torch.as_tensor(t_idx, device=self.device)
        n = torch.as_tensor(n_idx, device=self.device)
        rows = t[:, None] - L + 1 + torch.arange(L, device=self.device)[None, :]
        return Xn[rows, n[:, None]], n

    # -- training ----------------------------------------------------------------------------

    def fit(self, fs: FeatureSet, t_idx: np.ndarray, n_idx: np.ndarray):
        self.lookback = fs.lookback
        self.n_tickers = len(fs.tickers)
        self.tickers = list(fs.tickers)
        self.names = list(fs.names)
        rows = fs.X[np.unique(t_idx)]
        self.mu = np.nanmean(rows, axis=(0, 1))
        self.sd = np.nanstd(rows, axis=(0, 1)) + 1e-8
        Xn = self._standardise(fs)

        # Chronological validation split with a purge gap so no target overlaps the validation period.
        dates = np.unique(t_idx)
        cut = dates[int(len(dates) * (1 - self.tcfg.val_fraction))]
        gap = fs.horizon + 1
        tr = t_idx < cut - gap
        va = t_idx >= cut
        y = fs.y[t_idx, n_idx]
        yn = np.clip(y / fs.vol[t_idx, n_idx], -CLIP, CLIP)
        data = {
            "tr": (t_idx[tr], n_idx[tr], yn[tr], y[tr] > 0),
            "va": (t_idx[va], n_idx[va], yn[va], y[va] > 0),
        }
        self.models = []
        for k in range(self.tcfg.ensemble):
            self.models.append(self._fit_one(Xn, data, self.seed + k))
        return self

    def _fit_one(self, Xn, data, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        m, t = self.mcfg, self.tcfg
        model = MarketTransformer(Xn.shape[-1], self.n_tickers, self.lookback, m.d_model, m.n_heads,
                                  m.n_layers, m.ff_mult, m.dropout).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
        t_tr, n_tr, yn_tr, up_tr = data["tr"]
        steps_per_epoch = math.ceil(len(t_tr) / t.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t.lr, total_steps=t.epochs * steps_per_epoch,
                                                    pct_start=0.1)
        best, best_state, bad = math.inf, None, 0
        for epoch in range(t.epochs):
            model.train()
            perm = rng.permutation(len(t_tr))
            total = 0.0
            for i in range(steps_per_epoch):
                b = perm[i * t.batch_size:(i + 1) * t.batch_size]
                loss = self._loss(model, Xn, t_tr[b], n_tr[b], yn_tr[b], up_tr[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(b)
            val = self._evaluate(model, Xn, *data["va"])
            self.history.append({"seed": seed, "epoch": epoch, "train": total / len(t_tr), "val": val})
            log.debug("seed %d epoch %d train %.4f val %.4f", seed, epoch, total / len(t_tr), val)
            if val < best - 1e-5:
                best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= t.patience:
                    break
        if best_state is None:
            raise RuntimeError("transformer training diverged: the validation loss was never finite")
        model.load_state_dict(best_state)
        model.eval()
        log.info("transformer seed %d: stopped after %d epochs, best val loss %.4f", seed, epoch + 1, best)
        return model

    def _loss(self, model, Xn, t_idx, n_idx, yn, up):
        x, n = self._batch(Xn, t_idx, n_idx)
        ret, logit = model(x, n)
        yn = torch.as_tensor(yn, dtype=torch.float32, device=self.device)
        up = torch.as_tensor(up, dtype=torch.float32, device=self.device)
        return F.huber_loss(ret, yn, delta=1.0) + self.tcfg.cls_weight * F.binary_cross_entropy_with_logits(logit, up)

    @torch.no_grad()
    def _evaluate(self, model, Xn, t_idx, n_idx, yn, up, batch=4096):
        model.eval()
        total = 0.0
        for i in range(0, len(t_idx), batch):
            s = slice(i, i + batch)
            total += self._loss(model, Xn, t_idx[s], n_idx[s], yn[s], up[s]).item() * len(t_idx[s])
        return total / max(len(t_idx), 1)

    # -- inference ---------------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, fs: FeatureSet, t_idx: np.ndarray, n_idx: np.ndarray, batch: int = 4096):
        if list(fs.tickers) != self.tickers or list(fs.names) != self.names:
            raise ValueError("FeatureSet tickers/features differ from those the model was trained on")
        Xn = self._standardise(fs)
        ret = np.zeros(len(t_idx))
        prob = np.zeros(len(t_idx))
        for model in self.models:
            model.eval()
            for i in range(0, len(t_idx), batch):
                s = slice(i, i + batch)
                x, n = self._batch(Xn, t_idx[s], n_idx[s])
                r, logit = model(x, n)
                ret[s] += r.cpu().numpy()
                prob[s] += torch.sigmoid(logit).cpu().numpy()
        k = len(self.models)
        return ret / k * fs.vol[t_idx, n_idx], prob / k

    # -- persistence -------------------------------------------------------------------------

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "states": [m.state_dict() for m in self.models],
            "model_cfg": vars(self.mcfg), "train_cfg": vars(self.tcfg),
            "mu": self.mu, "sd": self.sd, "lookback": self.lookback,
            "tickers": self.tickers, "names": self.names, "meta": meta or {},
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TransformerPredictor":
        from types import SimpleNamespace

        ck = torch.load(path, map_location=device, weights_only=False)
        tcfg = SimpleNamespace(**{**ck["train_cfg"], "device": device})
        mcfg = SimpleNamespace(**ck["model_cfg"])
        p = cls(mcfg, tcfg)
        p.mu, p.sd, p.lookback = ck["mu"], ck["sd"], ck["lookback"]
        p.tickers, p.names, p.n_tickers = ck["tickers"], ck["names"], len(ck["tickers"])
        p.meta = ck["meta"]
        for state in ck["states"]:
            m = MarketTransformer(len(p.names), p.n_tickers, p.lookback, mcfg.d_model, mcfg.n_heads,
                                  mcfg.n_layers, mcfg.ff_mult, mcfg.dropout).to(p.device)
            m.load_state_dict(state)
            m.eval()
            p.models.append(m)
        return p
