"""Typed configuration loaded from config.yaml."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# Bars per year: 252 trading days; regular hours 9:30-16:00 give 7 hourly bars (the last is 30 min).
PERIODS_PER_YEAR = {"1d": 252, "1h": 252 * 7}


@dataclass
class DataConfig:
    tickers: list = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "AVGO"])
    market: dict = field(default_factory=lambda: {"broad": "SPY", "sector": "QQQ", "vol": "^VIX", "rates": "^TNX"})
    start: str = "2010-01-01"
    end: str | None = None
    dir: str = "data/daily"
    interval: str = "1d"            # '1d' or '1h'
    history_days: int = 730         # intraday only: how far back to download (Yahoo caps 1h at ~730 days)

    def __post_init__(self):
        if self.interval not in PERIODS_PER_YEAR:
            raise ValueError(f"data.interval must be one of {sorted(PERIODS_PER_YEAR)}, got {self.interval!r}")

    @property
    def intraday(self) -> bool:
        return self.interval != "1d"

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR[self.interval]

    @property
    def symbols(self) -> list:
        return list(dict.fromkeys(self.tickers + list(self.market.values())))


@dataclass
class FeatureConfig:
    lookback: int = 60
    horizon: int = 1
    execution: str = "next_open"

    def __post_init__(self):
        if self.execution not in ("next_open", "close"):
            raise ValueError(f"features.execution must be 'next_open' or 'close', got {self.execution!r}")


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 2
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 512
    lr: float = 5e-4
    weight_decay: float = 0.01
    patience: int = 4
    val_fraction: float = 0.15
    cls_weight: float = 0.5
    ensemble: int = 1
    seed: int = 7
    device: str = "auto"
    threads: int = 8


@dataclass
class WalkForwardConfig:
    first_test_year: int = 2016
    step_years: int = 1
    embargo_days: int = 5           # in bars
    # Alternative schedule for short histories (e.g. hourly data): the first test block starts
    # `min_train_months` after the first bar and blocks are `step_months` long.
    min_train_months: int | None = None
    step_months: int | None = None


@dataclass
class StrategyConfig:
    allow_short: bool = False
    prob_threshold: float = 0.52
    min_edge: float = 0.0
    target_vol: float = 0.15
    max_position: float = 0.20
    max_gross: float = 1.0
    min_trade: float = 0.02
    cov_lookback: int = 60
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.15
    cooldown_days: int = 10


@dataclass
class OptimizerConfig:
    trader: str = "rules"           # 'rules' (Strategy) or 'optimal' (OptimalTrader)
    risk_aversion: float = 2.0      # penalty on variance in the objective
    holding_periods: float = 5.0    # expected holding period; trading cost is spread over it
    calib_window: int = 500         # past periods of predictions used to calibrate the model
    recalibrate_every: int = 20     # periods between recalibrations in backtests
    shrink_k: float = 4.0           # slope shrinkage: keep t^2 / (t^2 + k) of the fitted slope
    use_drift: bool = False         # also trade the average realised return (market drift)


@dataclass
class CostConfig:
    commission_bps: float = 1.0
    slippage_bps: float = 4.0
    borrow_bps_annual: float = 50.0

    @property
    def one_way(self) -> float:
        """Cost per unit of traded notional."""
        return (self.commission_bps + self.slippage_bps) / 1e4

    @property
    def round_trip(self) -> float:
        return 2 * self.one_way


@dataclass
class BrokerConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 17
    mode: str = "paper"
    max_order_value: float = 25_000.0
    data_source: str = "yfinance"
    max_capital: float | None = None    # size positions on at most this much equity (None = whole account)
    limit_offset_bps: float | None = None  # if set, use limit orders this far through the last price


@dataclass
class PathConfig:
    artifacts: str = "artifacts/daily"
    reports: str = "reports/daily"
    models: str = "models/daily"
    logs: str = "logs/daily"

    def ensure(self) -> None:
        for p in (self.artifacts, self.reports, self.models, self.logs):
            Path(p).mkdir(parents=True, exist_ok=True)


SECTIONS = {
    "data": DataConfig,
    "features": FeatureConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "walkforward": WalkForwardConfig,
    "strategy": StrategyConfig,
    "optimizer": OptimizerConfig,
    "costs": CostConfig,
    "broker": BrokerConfig,
    "paths": PathConfig,
}


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = "config.yaml") -> Config:
    raw = {}
    if path is not None and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return Config(**{name: cls(**(raw.get(name) or {})) for name, cls in SECTIONS.items()})
