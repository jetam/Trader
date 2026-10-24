"""Stage 7: Interactive Brokers connection (TWS or IB Gateway) via ib_async.

Safety rules enforced here, not left to the caller:
  * Orders are only sent to paper accounts (IDs starting with DU/DF) unless broker.mode is 'live'
    AND the environment variable TRADER_ALLOW_LIVE=yes is set.
  * A single order may not exceed broker.max_order_value in notional.
  * In dry-run mode the connection is opened read-only, so the API cannot place orders at all.
"""

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

PAPER_PREFIXES = ("DU", "DF")
# Yahoo symbols of market series -> IB index contracts.
INDEX_MAP = {"^VIX": ("VIX", "CBOE"), "^TNX": ("TNX", "CBOE")}


PORTFOLIO_COLUMNS = ["symbol", "shares", "avg_cost", "price", "value", "unrealized_pnl", "realized_pnl"]
OPEN_ORDER_COLUMNS = ["symbol", "action", "quantity", "filled", "type", "tif", "limit", "status", "client_id"]
FILL_COLUMNS = ["time", "exec_id", "symbol", "side", "shares", "price", "value", "commission", "realized_pnl",
                "client_id"]


class SafetyError(RuntimeError):
    pass


def _num(v) -> float:
    """IB marks missing numbers with the largest float; turn those into NaN."""
    v = float(v)
    return math.nan if not math.isfinite(v) or abs(v) > 1e300 else v


def market_open(now: datetime | None = None) -> bool:
    """True during US regular trading hours (weekdays 9:30-16:00 New York; holidays not handled)."""
    now = now or datetime.now(ZoneInfo("America/New_York"))
    return now.weekday() < 5 and time(9, 30) <= now.time() < time(16, 0)


@dataclass
class OrderPlan:
    symbol: str
    quantity: int           # signed: + buy, - sell
    price: float
    current_weight: float
    target_weight: float

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price


class Broker:
    def __init__(self, cfg, read_only: bool = True):
        self.cfg = cfg
        self.read_only = read_only
        self.ib = None
        self.account = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    def connect(self):
        from ib_async import IB

        self.ib = IB()
        self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id, readonly=self.read_only,
                        timeout=10)
        accounts = self.ib.managedAccounts()
        if not accounts:
            raise RuntimeError("Connected to IB but no managed accounts were reported")
        self.account = accounts[0]
        log.info("connected to IB %s:%s account %s (%s)", self.cfg.host, self.cfg.port, self.account,
                 "paper" if self.is_paper else "LIVE")

    def disconnect(self):
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()

    @property
    def is_paper(self) -> bool:
        return self.account.startswith(PAPER_PREFIXES)

    def check_can_trade(self):
        if self.read_only:
            raise SafetyError("connection is read-only (dry run)")
        if self.is_paper:
            return
        if self.cfg.mode != "live" or os.environ.get("TRADER_ALLOW_LIVE") != "yes":
            raise SafetyError(f"account {self.account} is a LIVE account. Refusing to trade: set broker.mode: live "
                              f"in config.yaml AND export TRADER_ALLOW_LIVE=yes to allow real-money orders.")
        log.warning("REAL-MONEY TRADING ENABLED on account %s", self.account)

    # -- account -----------------------------------------------------------------------------

    def net_liquidation(self) -> float:
        for v in self.ib.accountSummary(self.account):
            if v.tag == "NetLiquidation" and v.currency in ("USD", "BASE", ""):
                return float(v.value)
        raise RuntimeError("NetLiquidation not found in account summary")

    def positions(self) -> dict:
        return {p.contract.symbol: float(p.position) for p in self.ib.positions(self.account)
                if p.contract.secType == "STK"}

    def portfolio(self) -> pd.DataFrame:
        """Stock positions with IB's own valuation and profit/loss."""
        rows = [{"symbol": p.contract.symbol, "shares": p.position, "avg_cost": p.averageCost,
                 "price": p.marketPrice, "value": p.marketValue, "unrealized_pnl": _num(p.unrealizedPNL),
                 "realized_pnl": _num(p.realizedPNL)}
                for p in self.ib.portfolio(self.account) if p.contract.secType == "STK"]
        return pd.DataFrame(rows, columns=PORTFOLIO_COLUMNS)

    def open_orders(self) -> pd.DataFrame:
        """Orders sent but not yet (fully) filled, from every API client and TWS."""
        rows = [{"symbol": t.contract.symbol, "action": t.order.action, "quantity": t.order.totalQuantity,
                 "filled": t.orderStatus.filled, "type": t.order.orderType, "tif": t.order.tif,
                 "limit": _num(t.order.lmtPrice) if t.order.orderType == "LMT" else math.nan,
                 "status": t.orderStatus.status, "client_id": t.order.clientId}
                for t in self.ib.reqAllOpenOrders()
                if t.order.account in ("", self.account) and not t.isDone()]
        return pd.DataFrame(rows, columns=OPEN_ORDER_COLUMNS)

    def fills(self) -> pd.DataFrame:
        """Executions IB still reports (by default those since the last server reset, about one day)."""
        self.ib.reqExecutions()
        self.ib.sleep(1)                      # commission reports arrive just after the executions
        rows = []
        for f in self.ib.fills():
            e = f.execution
            if e.acctNumber != self.account or f.contract.secType != "STK":
                continue
            t = pd.Timestamp(f.time)
            t = t.tz_convert("America/New_York").tz_localize(None) if t.tz is not None else t
            rows.append({"time": t, "exec_id": e.execId, "symbol": f.contract.symbol,
                         "side": "BUY" if e.side == "BOT" else "SELL", "shares": e.shares, "price": e.price,
                         "value": e.shares * e.price, "commission": _num(f.commissionReport.commission),
                         "realized_pnl": _num(f.commissionReport.realizedPNL), "client_id": e.clientId})
        return pd.DataFrame(rows, columns=FILL_COLUMNS)

    # -- market data -------------------------------------------------------------------------

    def contract(self, symbol: str):
        from ib_async import Index, Stock

        if symbol in INDEX_MAP:
            name, exchange = INDEX_MAP[symbol]
            return Index(name, exchange, "USD")
        return Stock(symbol, "SMART", "USD")

    def bars(self, symbol: str, duration: str = "730 D", interval: str = "1d") -> pd.DataFrame:
        """Daily or hourly bars in the same long format (naive New York time) as data.download_yahoo."""
        from ib_async import util

        c = self.contract(symbol)
        self.ib.qualifyContracts(c)
        show = "TRADES" if symbol in INDEX_MAP or interval != "1d" else "ADJUSTED_LAST"
        bar_size = {"1d": "1 day", "1h": "1 hour"}[interval]
        bars = self.ib.reqHistoricalData(c, endDateTime="", durationStr=duration, barSizeSetting=bar_size,
                                         whatToShow=show, useRTH=True, formatDate=2)
        df = util.df(bars)
        if df is None or df.empty:
            raise RuntimeError(f"IB returned no bars for {symbol}")
        dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        if dates.tz is not None:
            dates = dates.tz_convert("America/New_York").tz_localize(None)
        df["date"] = dates
        df["symbol"] = symbol
        return df.set_index(["date", "symbol"])[["open", "high", "low", "close", "volume"]]

    def history(self, symbols: list, duration: str = "730 D", interval: str = "1d") -> pd.DataFrame:
        return pd.concat([self.bars(s, duration, interval) for s in symbols]).sort_index()

    # -- orders ------------------------------------------------------------------------------

    def place(self, plans: list, execution: str, intraday: bool = False) -> list:
        """Submit orders.

        daily:   'next_open' -> market-on-open (MKT, tif OPG); 'close' -> market-on-close (MOC)
        hourly:  market order filling now during regular hours; market-on-open outside them
        """
        from ib_async import LimitOrder, MarketOrder, Order

        self.check_can_trade()
        trades = []
        for p in plans:
            if p.quantity == 0:
                continue
            if p.notional > self.cfg.max_order_value + 1e-6:
                raise SafetyError(f"{p.symbol}: order notional {p.notional:,.0f} exceeds max_order_value")
            action = "BUY" if p.quantity > 0 else "SELL"
            qty = abs(p.quantity)
            tif = "DAY" if intraday and market_open() else "OPG"
            if self.cfg.limit_offset_bps is not None and not (execution == "close" and not intraday):
                # Marketable limit: fills like a market order unless the price runs away from us.
                off = self.cfg.limit_offset_bps / 1e4
                limit = round(p.price * (1 + off if action == "BUY" else 1 - off), 2)
                order = LimitOrder(action, qty, limit, tif=tif, account=self.account)
            elif execution == "close" and not intraday:
                order = Order(action=action, totalQuantity=qty, orderType="MOC", account=self.account)
            else:
                order = MarketOrder(action, qty, tif=tif, account=self.account)
            c = self.contract(p.symbol)
            self.ib.qualifyContracts(c)
            trade = self.ib.placeOrder(c, order)
            log.info("placed %s %d %s (%s %s) -> %s", action, qty, p.symbol, order.orderType, order.tif,
                     trade.orderStatus.status)
            trades.append(trade)
        self.ib.sleep(2)
        return trades


def plan_orders(symbols: list, target_w, current_shares: dict, prices: dict, nlv: float,
                max_order_value: float) -> list:
    """Convert target weights into whole-share orders, clipped to the per-order notional cap."""
    plans = []
    for sym, w in zip(symbols, target_w):
        px = prices[sym]
        if not (px and math.isfinite(px)):
            continue
        cur = current_shares.get(sym, 0.0)
        target_shares = math.floor(abs(w) * nlv / px) * (1 if w >= 0 else -1)
        qty = int(target_shares - cur)
        max_qty = int(max_order_value // px)
        qty = max(-max_qty, min(max_qty, qty))
        plans.append(OrderPlan(sym, qty, px, cur * px / nlv, w))
    return plans
