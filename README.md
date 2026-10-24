# Trader

Quantitative analysis and transformer-based prediction for large technology stocks (AAPL, MSFT,
NVDA, AMD, GOOGL, META, AMZN, AVGO), with paper trading through Interactive Brokers.

The question this project answers: **can mathematical analysis and transformer predictions find
market signals that stay profitable after costs and risk?** It answers it honestly. Every model is
judged out of sample against simple baselines and against just holding the stocks.

## How it works

1. **Collect data.** Download daily split- and dividend-adjusted prices and volumes for the stocks,
   plus market context: SPY, QQQ, the VIX and the 10-year yield (`trader/data.py`).
2. **Analyze the market.** Measure return distributions (fat tails), stationarity, autocorrelation,
   volatility clustering, random-walk tests, beta to the market, correlations and principal
   components (`trader/analysis.py` → `reports/daily/analysis.md`).
3. **Build features.** Turn prices into 25 daily features: trailing returns, volatility, ranges,
   volume shocks, trend and RSI, behaviour relative to QQQ, and market state. A feature for day t
   only uses information available at the close of day t. Tests check this
   (`trader/features.py`).
4. **Baselines.** Zero return (random walk), historical mean, a momentum/reversal regression, ridge
   regression and gradient boosting (`trader/baselines.py`).
5. **Transformer.** For each stock and day, the model reads the last 60 days of features. A
   summary token tagged with the stock's identity attends over that window. Two heads then predict
   the next return (scaled by volatility) and the probability that it is positive
   (`trader/transformer.py`).
6. **Walk-forward testing.** Train on everything before a year, predict that year, then roll
   forward and retrain. A purge gap between training and test data stops any leakage. All models
   are tested on the same samples (`trader/walkforward.py`).
7. **Evaluation and simulated trading.** Score forecasts with out-of-sample R², hit rate, AUC,
   information coefficient (IC) and Diebold-Mariano tests. Then backtest the trading strategy
   with commissions, slippage and borrow costs. Report Sharpe confidence intervals and the
   probabilistic Sharpe ratio, and compare against buy-and-hold. The report ends with a verdict
   (`trader/evaluate.py` → `reports/daily/evaluation.md`).
8. **Strategy and risk.** Go long only when P(up) and the expected return clear the round-trip
   cost. Size positions by signal divided by volatility, scaled to a portfolio volatility target
   using the covariance matrix. Hard limits apply: per-stock cap, gross exposure cap, a daily loss
   limit, and a drawdown stop with a cooldown (`trader/strategy.py`). The backtest and live
   trading use exactly the same strategy code.
9. **Interactive Brokers.** Read account equity and positions, optionally pull daily bars, and
   send market-on-open orders. Orders go to paper accounts only unless you explicitly unlock
   real-money trading (`trader/broker.py`, `trader/live.py`).
10. **Continuous evaluation.** Every live forecast is logged. `monitor` scores the logged
    forecasts against what actually happened and warns when the edge disappears. `train`
    retrains the production model on the latest data.

## Setup

```bash
python3 -m venv --system-site-packages .venv      # reuses an existing torch install
.venv/bin/pip install -r requirements.txt          # or: pip --python .venv/bin/python install -r requirements.txt
```

## Research workflow

```bash
.venv/bin/python -m trader fetch            # download prices -> data/daily/prices.parquet
.venv/bin/python -m trader analyze          # -> reports/daily/analysis.md (+ figures)
.venv/bin/python -m trader walkforward      # all models; or --models ridge transformer
.venv/bin/python -m trader evaluate         # -> reports/daily/evaluation.md, verdict printed
.venv/bin/python -m trader research         # all of the above in one go
.venv/bin/python -m pytest                  # tests (synthetic data, no network)
```

### GUI

```bash
.venv/bin/python -m trader gui        # opens on http://localhost:8501
```

Three tabs. The model, capital, trader and limits are set in the sidebar.

- **Past simulation**: pick a start and end date. It shows what your money would have become by
  following the model (after costs), and how much more or less that is than buying the stocks
  equally or holding QQQ. Also: the account value curve, drawdowns, yearly returns, what the model
  held, and a risk table.
- **Future prediction**: the model's forecast for the next day (or hour, on the hourly timeframe):
  predicted return and P(up) per stock, the target portfolio, the buy/sell orders in whole shares
  for your capital, and why each stock was or wasn't bought. **Get latest prices** downloads fresh
  data first. Baselines are refitted on all known outcomes; the transformer uses `models/<daily|hourly>/transformer.pt`.
  Optionally enter the shares you already own. Needs no broker connection and places no orders.
- **Account & trades**: what the live/paper trader actually did. **Refresh from IB** opens a read-only
  connection to TWS / IB Gateway (it can never place orders) and shows the account value, money earned
  since the first trade run, the fills (trades made) with commissions and profit, orders still waiting
  at IB, the next run's planned orders for your real positions, and current positions with profit.
  Without a connection it shows the trader's logs (`equity.csv`, `orders.csv`, `fills.csv` in `logs/daily/` or `logs/hourly/`).

Simulations can start at the first walk-forward test year (2016 by default). Earlier dates have no
out-of-sample predictions.

All settings are in `config.yaml`: universe, lookback, horizon, model size, walk-forward schedule,
strategy limits, costs and broker connection.

## Hourly system

A second, independent system trades on **hourly bars**: its own transformer, data, reports, models
and logs, all driven by `config_hourly.yaml`. Every command takes `-c config_hourly.yaml`:

```bash
python -m trader -c config_hourly.yaml fetch          # last 2 years of hourly bars -> data/hourly/
python -m trader -c config_hourly.yaml analyze        # -> reports/hourly/analysis.md (incl. time-of-day effects)
python -m trader -c config_hourly.yaml walkforward    # monthly retraining after a 12-month warm-up
python -m trader -c config_hourly.yaml evaluate       # -> reports/hourly/evaluation.md
python -m trader -c config_hourly.yaml train          # -> models/hourly/transformer.pt
```

Differences from the daily system:
- **Bars:** 7 per session (9:30 … 15:30 New York). Feature windows count bars, so `r20` is a 20-hour
  return. Extra features: time of day, and volume compared with the same hour on previous days.
- **Market data:** VIX and 10-year yield bars run on their own schedule. They are aligned to each
  stock bar using only bars that had already closed.
- **Walk-forward:** with only 2 years of data, the first year is training only, then the model is
  retrained every month and predicts the next month.
- **Annualisation:** returns, volatility and Sharpe use 1,764 bars per year. The loss limit and
  cooldown apply per bar.
- **Live orders:** run a minute after each bar closes (10:31 … 15:31) to send market orders that fill
  immediately. The 16:01 run sends market-on-open orders. Yahoo data is about 15 minutes delayed,
  so for live hourly trading set `broker.data_source: ib`.
- **GUI:** switch between Daily and Hourly at the top of the sidebar.

```
31 10-15 * * 1-5  cd ~/code/Trader && .venv/bin/python -m trader -c config_hourly.yaml trade --execute
1  16    * * 1-5  cd ~/code/Trader && .venv/bin/python -m trader -c config_hourly.yaml trade --execute
```

## Paper trading with Interactive Brokers

1. Run TWS or IB Gateway logged into a **paper** account. Enable the API (Configure → API →
   Settings) on port 7497 (TWS paper) or 4002 (Gateway paper), and set `broker.port` to match.
2. Train the production model: `python -m trader train`.
3. After the US close each trading day:

```bash
python -m trader trade              # dry run: read-only connection, prints the plan
python -m trader trade --execute    # submits market-on-open orders to the paper account
python -m trader monitor            # live forecast quality and paper P&L vs the backtest
```

Example cron schedule (server in New York time):

```
15 17 * * 1-5  cd ~/code/Trader && .venv/bin/python -m trader trade --execute >> logs/daily/trade.log 2>&1
30 17 * * 1-5  cd ~/code/Trader && .venv/bin/python -m trader monitor >> logs/daily/monitor.log 2>&1
0  18 * * 6    cd ~/code/Trader && .venv/bin/python -m trader fetch && .venv/bin/python -m trader train
```

## OptimalTrader

`trader/optimal.py` is an alternative to the threshold rules. Turn it on with `optimizer.trader: optimal`,
or pick it in the GUI. Each period it:

1. **Calibrates** the model on its own track record. It regresses realised returns on the model's past
   predictions, using only outcomes already known, and shrinks the slope by its t-statistic. A
   model with no reliable record gets expected returns of about zero, so it stays in cash.
2. **Optimises** the weights: maximise expected return − (risk aversion / 2) × variance − trading
   cost, subject to the same per-stock cap, gross exposure, volatility target, and loss and
   drawdown stops as the rule-based strategy.

`evaluate` backtests every model with both traders: the rows marked "(optimal)".

## Trading real money

Real-money orders are blocked unless **all** of these hold:

1. TWS / IB Gateway is logged into a **live** account (live API port 7496 or 4001).
2. `broker.mode: live` is set in the config.
3. `export TRADER_ALLOW_LIVE=yes` is set in the shell that runs the command.
4. The command is run with `--execute`.

Further safeguards:

- `broker.max_capital` sizes positions on at most this much, even if the account holds more.
  Start small.
- `broker.max_order_value` caps every single order.
- `broker.limit_offset_bps` uses limit orders instead of market orders, so a bad opening print
  cannot fill you far from the last price.
- **Kill switch:** `touch logs/daily/KILL` (or `logs/hourly/KILL`) blocks order sending for that trader; delete the file to resume.
- The risk limits (per-stock cap, loss limit, drawdown stop) behave exactly as in the backtests.

Suggested path: paper trade until the `monitor` numbers match the backtest for several months.
Then go live with a small `max_capital` that you can afford to lose entirely, and raise it only
on live evidence.

## Outputs

| Path | Content |
|---|---|
| `reports/{daily,hourly}/analysis.md` | statistical study of the market |
| `reports/{daily,hourly}/evaluation.md` | model comparison, backtests, yearly Sharpe, verdict |
| `artifacts/{daily,hourly}/predictions/*.parquet` | out-of-sample predictions per model |
| `models/{daily,hourly}/transformer.pt` | production model with its normalisation statistics |
| `logs/{daily,hourly}/` | live predictions, orders sent, fills, equity, risk state |

## Caveats

- Daily returns of large stocks are close to unpredictable. An IC of 0.02–0.03 is a realistic
  "good" result, and costs can easily erase it.
- Each model or setting you try is another draw from luck (multiple testing). Change the config,
  re-run and pick the best, and the backtest will look better than the future will be.
- Yahoo data is survivorship-biased: this universe consists of today's winners. That flatters
  buy-and-hold more than it flatters the models.
- The backtest assumes you trade at the next open with the configured slippage. Large orders or
  thin opening auctions can be worse.
