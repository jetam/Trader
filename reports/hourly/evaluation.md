# Out-of-sample evaluation

1h bars. Walk-forward from 2025-10-24, retraining every 1 month(s). Target: 1-bar forward log return, execution at `next_open`. 12008 common out-of-sample predictions per model.

## Conclusion

- No transformer predictions found - run `python -m trader walkforward --models transformer`.

## Forecast quality

|  | n | mse_e4 | r2_oos | hit_rate | auc | brier | logloss | ic_mean | ic_t | ts_corr | dm_vs_zero | dm_p | dm_vs_histmean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gbm | 12008.0000 | 1.0259 | -0.0114 | 0.5051 | 0.4975 | 0.2530 | 0.6992 | 0.0098 | 0.9197 | 0.0020 | 2.6538 | 0.0080 | 2.6968 |
| hist_mean | 12008.0000 | 1.0146 | -0.0003 | 0.5002 | 0.5015 | 0.2501 | 0.6933 | -0.0030 | -0.2727 | -0.0416 | 0.6024 | 0.5469 | 0.0000 |
| momentum | 12008.0000 | 1.0153 | -0.0009 | 0.5049 | 0.5056 | 0.2501 | 0.6934 | 0.0156 | 1.4767 | 0.0108 | 0.5405 | 0.5888 | 0.2820 |
| ridge | 12008.0000 | 1.0174 | -0.0030 | 0.5045 | 0.5099 | 0.2504 | 0.6939 | 0.0282 | 2.7324 | 0.0029 | 1.1279 | 0.2594 | 1.0299 |
| zero | 12008.0000 | 1.0143 | 0.0000 | – | 0.4846 | 0.2501 | 0.6932 | – | – | – | 0.0000 | 1.0000 | -0.6024 |

`r2_oos` vs predicting 0; `ic_mean` = mean daily cross-sectional Spearman correlation, `ic_t` its Newey-West t-stat; `ts_corr` = average per-stock time-series correlation; `dm_*` = Diebold-Mariano statistic on squared errors (negative = better than the reference).

## Trading simulation (after costs)

|  | cagr | ann_vol | sharpe | sharpe_lo95 | sharpe_hi95 | psr | sortino | max_dd | calmar | worst_day | hit_days | turnover_ann | cost_drag_ann | avg_gross | days_halted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gbm | -0.051 | 0.097 | -0.493 | -2.354 | 1.416 | 0.314 | -0.455 | -0.105 | -0.489 | -0.017 | 0.374 | 265.797 | 0.133 | 0.144 | 0.000 |
| gbm (optimal) | 0.017 | 0.116 | 0.202 | -2.058 | 2.495 | 0.578 | 0.197 | -0.125 | 0.134 | -0.032 | 0.490 | 91.434 | 0.046 | 0.370 | 2.000 |
| hist_mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – | 0.000 | – | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.000 |
| hist_mean (optimal) | 0.003 | 0.056 | 0.089 | -1.469 | 1.518 | 0.535 | 0.066 | -0.047 | 0.072 | -0.012 | 0.511 | 1.070 | 5.35e-04 | 0.079 | 0.000 |
| momentum | 0.027 | 0.063 | 0.462 | -1.251 | 2.165 | 0.674 | 0.322 | -0.039 | 0.705 | -0.015 | 0.461 | 22.121 | 0.011 | 0.054 | 0.000 |
| momentum (optimal) | 0.033 | 0.067 | 0.524 | -1.119 | 2.516 | 0.694 | 0.338 | -0.065 | 0.516 | -0.020 | 0.510 | 20.507 | 0.010 | 0.131 | 0.000 |
| ridge | 0.096 | 0.088 | 1.087 | -0.594 | 2.958 | 0.861 | 0.920 | -0.055 | 1.743 | -0.021 | 0.439 | 63.627 | 0.032 | 0.080 | 1.000 |
| ridge (optimal) | 0.081 | 0.085 | 0.960 | -1.167 | 3.183 | 0.828 | 0.918 | -0.067 | 1.217 | -0.022 | 0.512 | 23.352 | 0.012 | 0.144 | 1.000 |
| zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – | 0.000 | – | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.000 |
| zero (optimal) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – | 0.000 | – | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.000 |
| equal_weight_hold | 0.309 | 0.239 | 1.247 | -0.830 | 3.455 | 0.889 | 1.671 | -0.130 | 2.374 | -0.025 | 0.530 | – | – | – | – |
| QQQ_hold | 0.202 | 0.197 | 1.031 | -0.820 | 2.982 | 0.844 | 1.336 | -0.108 | 1.870 | -0.027 | 0.534 | – | – | – | – |

Strategy: long-only, P(up) ≥ 0.52, target vol 15%, max 20% per stock, gross ≤ 100%, daily loss limit 2%, drawdown stop 15%. Costs 5.0 bps per side. Benchmarks ignore costs. Rows marked (optimal) use the OptimalTrader (risk aversion 2.0, holding period 5, calibrated on the last 500 periods of each model's own track record) under the same limits.

## Sharpe ratio by year

| date | gbm | gbm (optimal) | hist_mean | hist_mean (optimal) | momentum | momentum (optimal) | ridge | ridge (optimal) | zero | zero (optimal) | equal_weight_hold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | -1.02 | -3.38 | 0.00 | 0.28 | -1.41 | 3.95 | 0.51 | 4.90 | 0.00 | 0.00 | 0.20 |
| 2026 | -0.36 | 0.68 | 0.00 | -0.05 | 0.87 | -0.33 | 1.23 | 0.67 | 0.00 | 0.00 | 1.51 |

## Figures

![](figures/equity_curves.png)
