"""Price data: download from Yahoo Finance, cache as parquet, load as aligned wide panels."""

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.offsets import BDay

log = logging.getLogger(__name__)

FIELDS = ["open", "high", "low", "close", "volume"]
NY = ZoneInfo("America/New_York")


def _drop_incomplete_bars(long: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Yahoo returns a partial bar for the period in progress; drop it.

    Timestamps are bar starts in New York time. Daily bars end at 16:00 (plus a margin for the
    final print); hourly bars end an hour later or at 16:00, whichever comes first.
    """
    now = pd.Timestamp(datetime.now(NY).replace(tzinfo=None))
    start = long.index.get_level_values("date")
    if interval == "1d":
        end = start + pd.Timedelta(hours=16, minutes=30)
    else:
        end = (start + pd.Timedelta(interval)).where(
            start + pd.Timedelta(interval) <= start.normalize() + pd.Timedelta(hours=16),
            start.normalize() + pd.Timedelta(hours=16))
    return long[end <= now]


def download_yahoo(symbols: list, start: str | None, end: str | None = None, interval: str = "1d",
                   history_days: int = 730) -> pd.DataFrame:
    """OHLCV in long format indexed by (date, symbol); timestamps are naive New York time.

    Daily bars are split/dividend adjusted. Intraday bars cover at most `history_days` days.
    """
    import yfinance as yf

    kwargs = dict(start=start, end=end)
    if interval != "1d":
        earliest = (datetime.now(NY) - timedelta(days=history_days - 1)).date().isoformat()
        kwargs = dict(start=max(start or earliest, earliest), end=end)
    raw = yf.download(list(symbols), interval=interval, auto_adjust=True, progress=False,
                      group_by="column", threads=True, **kwargs)
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {symbols}")
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY).tz_localize(None)
    raw.index = idx
    long = raw.stack(level=1, future_stack=True)
    long.index.names = ["date", "symbol"]
    long.columns = [str(c).lower() for c in long.columns]
    long = long[FIELDS].dropna(subset=["close"])
    missing = set(symbols) - set(long.index.get_level_values("symbol"))
    if missing:
        log.warning("No data returned for %s", sorted(missing))
    return _drop_incomplete_bars(long, interval).sort_index()


def next_bar(last: pd.Timestamp, intraday: bool) -> pd.Timestamp:
    """Start of the bar after `last` (weekends skipped, exchange holidays not)."""
    if intraday:
        nxt = last + pd.Timedelta(hours=1)
        if nxt.time() < time(16, 0):
            return nxt
        return last.normalize() + BDay(1) + pd.Timedelta(hours=9, minutes=30)
    return last.normalize() + BDay(1)


def prices_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "prices.parquet"


def save_prices(long: pd.DataFrame, data_dir: str | Path) -> Path:
    path = prices_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(path)
    return path


def load_prices(data_dir: str | Path) -> pd.DataFrame:
    path = prices_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m trader fetch` first")
    return pd.read_parquet(path)


@dataclass
class Panel:
    """Wide (date x symbol) frames on a common trading calendar."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @classmethod
    def from_long(cls, long: pd.DataFrame, calendar_symbol: str, ffill_symbols: tuple = ()) -> "Panel":
        """Pivot to wide frames on the calendar of `calendar_symbol` (an exchange-traded series).

        Index series such as ^VIX/^TNX trade on their own schedule (extended hours, bars at other
        minutes, holidays). Each `ffill_symbols` series is aligned as-of: a calendar bar gets the
        latest bar that *started* no later than it, which also ended no later than it, so no future
        data leaks in. Stale values are dropped after 3 of their own bars. Stocks are never filled.
        """
        wide = {f: long[f].unstack("symbol").sort_index() for f in FIELDS}
        calendar = wide["close"][calendar_symbol].dropna().index
        out = {}
        for f, df in wide.items():
            aligned = df.reindex(calendar)
            for s in ffill_symbols:
                if s in df:
                    own = df[s].dropna()
                    aligned[s] = own.reindex(own.index.union(calendar)).ffill(limit=3).reindex(calendar)
            out[f] = aligned
        return cls(**out)

    def tail(self, n: int) -> "Panel":
        return Panel(**{f: getattr(self, f).iloc[-n:] for f in FIELDS})


def fetch_prices(cfg) -> pd.DataFrame:
    """Download the full configured history and replace the parquet cache."""
    d = cfg.data
    long = download_yahoo(d.symbols, None if d.intraday else d.start, d.end, d.interval, d.history_days)
    save_prices(long, d.dir)
    return long


def load_panel(cfg) -> Panel:
    """Panel from the parquet cache for a Config."""
    m = cfg.data.market
    return Panel.from_long(load_prices(cfg.data.dir), calendar_symbol=m["sector"],
                           ffill_symbols=(m["vol"], m["rates"]))
