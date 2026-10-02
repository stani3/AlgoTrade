# i003 v1 - Weekly-horizon time-series momentum

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.475 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.475 | >= 70% | FAIL |
| core system: median Sharpe across symbols | 0.59 | >= 0.3 | PASS |
| core system: share of symbols with Sharpe > 0 | 0.8 | >= 60% | PASS |
| core system: median closed trades per symbol | 150 | >= 30 | PASS |
| monkey test: share of random monkeys beaten | 1 | >= 90% | PASS |
| limited optimisation: share of grid combinations profitable | 1 | >= 70% | PASS |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | 0.59 |
| plateau score of the chosen cell | 0.575 |
| monkey median Sharpe (median run) | -0.125 |
| median CAGR at the chosen parameters | 0.133 |
| worst drawdown at the chosen parameters | -0.734 |

Entry test by exit:

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.5 | 823 | 0.484 | 0.007 |
| 10 bars | 0.6 | 568 | 0.537 | 0.021 |
| 20 bars | 0.3 | 363 | 0.584 | 0.032 |
| 2/4 ATR bracket | 0.5 | 326 | 0.388 | 0.014 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 1515.0 | 0.333 | 0.005 | 7.337 | 3.0 |
| long | 756.0 | 0.347 | 0.013 | 9.827 | 3.0 |
| short | 759.0 | 0.32 | -0.003 | -2.49 | 4.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.206 | -0.227 | -0.0 |
| trend | ranging (ADX<=20) | 0.272 | -0.288 | -0.0 |
| trend | trending (ADX>=25) | 0.508 | 1.191 | 0.001 |
| trend | warm-up | 0.014 | 0.0 | 0.0 |
| volatility | high vol | 0.331 | 0.347 | 0.0 |
| volatility | low vol | 0.331 | 1.067 | 0.001 |
| volatility | mid vol | 0.331 | 0.394 | 0.0 |
| volatility | warm-up | 0.007 | 0.0 | 0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.019 | 0.042 | 0.179 | 0.408 | 0.305 | 2.594 |
| losers | 0.062 | 0.109 | 0.026 | 0.062 | 1.004 | 0.418 |

Report: `reports/research/i003/v1/feasibility/report.html`

Configurations evaluated in this stage: 5

Commit at run time: `690593e0f164b4e91d323545fd92d4bbd1c8c21b` | criteria `2f728c62a5eb5dab` | card `a2579f8eb1b0ef24`
