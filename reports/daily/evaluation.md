# Out-of-sample evaluation

1d bars. Walk-forward from 2016-01-01, retraining every 12 month(s). Target: 1-bar forward log return, execution at `next_open`. 21728 common out-of-sample predictions per model.

## Conclusion

- Forecast skill: transformer IC -0.0100 (t = -1.22), hit rate 0.525, AUC 0.502, out-of-sample R² -0.0040. Best baseline by IC: ridge (0.0263).
- The transformer's forecasts are NOT statistically distinguishable from noise (IC t-stat below 2 and no significant MSE improvement over predicting zero).
- It does not beat every baseline on rank correlation (IC).
- Trading after costs (transformer): Sharpe 1.03 (95% CI 0.45 to 1.71, P(Sharpe>0) = 1.00), CAGR 13.8%, max drawdown -25.7%, cost drag 0.32%/yr. Equal-weight buy & hold Sharpe: 1.28.
- VERDICT: the strategy is profitable in backtest but does not clearly beat simply holding the stocks risk-adjusted, or the forecast edge is not significant. Profit likely comes from market exposure (beta) rather than prediction. Not a reliable signal.
- Caveat: every extra model or setting tried raises the chance that a good backtest is luck (multiple testing). Treat a single favourable result with suspicion.

## Forecast quality

|  | n | mse_e4 | r2_oos | hit_rate | auc | brier | logloss | ic_mean | ic_t | ts_corr | dm_vs_zero | dm_p | dm_vs_histmean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gbm | 21728.0000 | 6.3060 | -0.0106 | 0.5165 | 0.4986 | 0.2523 | 0.6979 | 0.0221 | 2.7704 | 0.0262 | 1.5259 | 0.1270 | 1.7885 |
| hist_mean | 21728.0000 | 6.2320 | 0.0012 | 0.5364 | 0.4943 | 0.2491 | 0.6913 | -0.0031 | -0.4041 | -0.0299 | -1.1768 | 0.2393 | 0.0000 |
| momentum | 21728.0000 | 6.2252 | 0.0023 | 0.5380 | 0.5098 | 0.2487 | 0.6906 | 0.0171 | 2.1952 | 0.0270 | -0.9176 | 0.3588 | -0.5717 |
| ridge | 21728.0000 | 6.2495 | -0.0016 | 0.5270 | 0.5041 | 0.2497 | 0.6926 | 0.0263 | 3.2620 | 0.0263 | 0.4366 | 0.6624 | 0.8362 |
| transformer | 21728.0000 | 6.2648 | -0.0040 | 0.5252 | 0.5021 | 0.2495 | 0.6921 | -0.0100 | -1.2202 | 0.0039 | 1.5346 | 0.1249 | 2.2792 |
| zero | 21728.0000 | 6.2396 | 0.0000 | – | 0.4803 | 0.2489 | 0.6909 | – | – | – | 0.0000 | 1.0000 | 1.1768 |

`r2_oos` vs predicting 0; `ic_mean` = mean daily cross-sectional Spearman correlation, `ic_t` its Newey-West t-stat; `ts_corr` = average per-stock time-series correlation; `dm_*` = Diebold-Mariano statistic on squared errors (negative = better than the reference).

## Trading simulation (after costs)

|  | cagr | ann_vol | sharpe | sharpe_lo95 | sharpe_hi95 | psr | sortino | max_dd | calmar | worst_day | hit_days | turnover_ann | cost_drag_ann | avg_gross | days_halted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gbm | 0.069 | 0.130 | 0.582 | 0.105 | 1.234 | 0.970 | 0.680 | -0.222 | 0.312 | -0.076 | 0.477 | 86.458 | 0.043 | 0.397 | 47.000 |
| gbm (optimal) | 0.020 | 0.111 | 0.234 | -0.268 | 0.887 | 0.777 | 0.226 | -0.212 | 0.094 | -0.079 | 0.510 | 49.937 | 0.025 | 0.290 | 41.000 |
| hist_mean | 0.144 | 0.143 | 1.010 | 0.477 | 1.645 | 0.999 | 1.347 | -0.232 | 0.618 | -0.051 | 0.555 | 2.016 | 0.001 | 0.488 | 45.000 |
| hist_mean (optimal) | 0.009 | 0.066 | 0.164 | -0.431 | 0.741 | 0.703 | 0.158 | -0.182 | 0.048 | -0.044 | 0.537 | 1.143 | 5.72e-04 | 0.158 | 27.000 |
| momentum | 0.178 | 0.155 | 1.137 | 0.636 | 1.815 | 1.000 | 1.482 | -0.255 | 0.699 | -0.059 | 0.556 | 31.482 | 0.016 | 0.528 | 63.000 |
| momentum (optimal) | 0.084 | 0.119 | 0.743 | 0.260 | 1.408 | 0.992 | 0.848 | -0.252 | 0.335 | -0.060 | 0.550 | 19.458 | 0.010 | 0.316 | 45.000 |
| ridge | 0.124 | 0.137 | 0.926 | 0.423 | 1.630 | 0.999 | 1.193 | -0.199 | 0.626 | -0.054 | 0.516 | 64.216 | 0.032 | 0.456 | 32.000 |
| ridge (optimal) | 0.079 | 0.104 | 0.787 | 0.210 | 1.489 | 0.994 | 0.891 | -0.190 | 0.418 | -0.063 | 0.552 | 32.793 | 0.016 | 0.326 | 19.000 |
| transformer | 0.138 | 0.135 | 1.029 | 0.447 | 1.712 | 1.000 | 1.260 | -0.257 | 0.538 | -0.050 | 0.564 | 6.406 | 0.003 | 0.531 | 48.000 |
| transformer (optimal) | 0.050 | 0.085 | 0.616 | 0.044 | 1.240 | 0.977 | 0.671 | -0.175 | 0.287 | -0.056 | 0.558 | 2.149 | 0.001 | 0.297 | 25.000 |
| zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – | 0.000 | – | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.000 |
| zero (optimal) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – | 0.000 | – | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.000 |
| equal_weight_hold | 0.390 | 0.289 | 1.283 | 0.773 | 2.004 | 1.000 | 1.724 | -0.473 | 0.825 | -0.089 | 0.567 | – | – | – | – |
| QQQ_hold | 0.196 | 0.217 | 0.934 | 0.433 | 1.627 | 0.999 | 1.195 | -0.367 | 0.534 | -0.087 | 0.565 | – | – | – | – |

Strategy: long-only, P(up) ≥ 0.52, target vol 15%, max 20% per stock, gross ≤ 100%, daily loss limit 3%, drawdown stop 15%. Costs 5.0 bps per side. Benchmarks ignore costs. Rows marked (optimal) use the OptimalTrader (risk aversion 2.0, holding period 5, calibrated on the last 500 periods of each model's own track record) under the same limits.

## Sharpe ratio by year

| date | gbm | gbm (optimal) | hist_mean | hist_mean (optimal) | momentum | momentum (optimal) | ridge | ridge (optimal) | transformer | transformer (optimal) | zero | zero (optimal) | equal_weight_hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 0.23 | -0.65 | 0.85 | 0.52 | 1.84 | 1.39 | 0.53 | 0.82 | 0.98 | 1.01 | 0.00 | 0.00 | 2.19 |
| 2017 | 2.28 | 0.00 | 2.43 | 2.07 | 1.80 | 2.38 | 1.38 | 2.83 | 2.76 | 2.66 | 0.00 | 0.00 | 2.25 |
| 2018 | 0.86 | -0.39 | 0.10 | -0.47 | 0.32 | 0.63 | 0.46 | -0.28 | -0.05 | -0.23 | 0.00 | 0.00 | 0.28 |
| 2019 | 1.03 | 1.43 | 1.46 | 1.62 | 2.21 | 2.35 | 2.66 | 2.34 | 2.67 | 2.74 | 0.00 | 0.00 | 2.51 |
| 2020 | 0.87 | 0.27 | 0.73 | -0.65 | 0.75 | 1.29 | 1.13 | 0.92 | 2.06 | -0.02 | 0.00 | 0.00 | 1.48 |
| 2021 | 1.50 | 1.56 | 1.86 | 0.81 | 1.85 | 1.35 | 1.69 | 2.29 | 2.46 | 2.22 | 0.00 | 0.00 | 2.15 |
| 2022 | -1.49 | -1.21 | -1.28 | -1.68 | -1.63 | -1.31 | -1.31 | -1.12 | -1.52 | -1.71 | 0.00 | 0.00 | -1.10 |
| 2023 | 1.63 | 1.45 | 2.59 | 2.37 | 2.62 | 2.59 | 2.40 | 2.64 | 1.93 | 2.02 | 0.00 | 0.00 | 2.80 |
| 2024 | 2.10 | 1.15 | 2.12 | 1.12 | 1.92 | 1.71 | 1.62 | -0.61 | 1.69 | 1.69 | 0.00 | 0.00 | 1.91 |
| 2025 | -0.46 | -0.36 | 0.81 | 0.24 | 1.06 | 1.30 | 0.73 | 1.74 | 0.31 | 0.99 | 0.00 | 0.00 | 1.14 |
| 2026 | -1.00 | -0.29 | 0.32 | -1.56 | 0.69 | -0.23 | -0.31 | -1.11 | 0.18 | -1.12 | 0.00 | 0.00 | 1.18 |

## Figures

![](figures/equity_curves.png)

![](figures/transformer_ic.png)
