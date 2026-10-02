# i002 v1 - Liquidation shock fade

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.025 (needs >= 70%)
- core system: median Sharpe across symbols -0.292 (needs >= 0.3)
- core system: share of symbols with Sharpe > 0 0.1 (needs >= 60%)
- monkey test: share of random monkeys beaten 0.337 (needs >= 90%)
- limited optimisation: share of grid combinations profitable 0 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.025 | >= 70% | FAIL |
| core system: median Sharpe across symbols | -0.292 | >= 0.3 | FAIL |
| core system: share of symbols with Sharpe > 0 | 0.1 | >= 60% | FAIL |
| core system: median closed trades per symbol | 160 | >= 30 | PASS |
| monkey test: share of random monkeys beaten | 0.337 | >= 90% | FAIL |
| limited optimisation: share of grid combinations profitable | 0 | >= 70% | FAIL |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | -0.394 |
| plateau score of the chosen cell | -0.357 |
| monkey median Sharpe (median run) | -0.208 |
| median CAGR at the chosen parameters | -0.22 |
| worst drawdown at the chosen parameters | -1 |

Entry test by exit:

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.0 | 1377 | 0.486 | -0.008 |
| 10 bars | 0.0 | 1250 | 0.5 | -0.008 |
| 20 bars | 0.0 | 1008 | 0.48 | -0.015 |
| 2/4 ATR bracket | 0.1 | 1290 | 0.277 | -0.012 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 896.0 | 0.511 | -0.011 | -10.263 | 12.0 |
| long | 426.0 | 0.585 | 0.007 | 3.145 | 12.0 |
| short | 470.0 | 0.445 | -0.029 | -13.409 | 12.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.195 | 0.262 | 0.0 |
| trend | ranging (ADX<=20) | 0.298 | 0.551 | 0.0 |
| trend | trending (ADX>=25) | 0.504 | -0.777 | -0.0 |
| trend | warm-up | 0.002 | 0.0 | 0.0 |
| volatility | high vol | 0.333 | -0.416 | -0.0 |
| volatility | low vol | 0.333 | -0.737 | -0.0 |
| volatility | mid vol | 0.333 | -0.741 | -0.0 |
| volatility | warm-up | 0.001 | 0.0 | 0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.028 | 0.052 | 0.064 | 0.111 | 0.905 | 2.147 |
| losers | 0.09 | 0.16 | 0.023 | 0.047 | 3.06 | 0.737 |

Report: `reports/research/i002/v1/feasibility/report.html`

Configurations evaluated in this stage: 9

Commit at run time: `4bea7cf7e6fa22a969231cd87a81855c6fb60c51` | criteria `2f728c62a5eb5dab` | card `56640820ca066041`
