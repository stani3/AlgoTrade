# i009 v1 - Weekend move reversal

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.05 (needs >= 70%)
- core system: median Sharpe across symbols -0.232 (needs >= 0.3)
- core system: share of symbols with Sharpe > 0 0.3 (needs >= 60%)
- monkey test: share of random monkeys beaten 0.297 (needs >= 90%)
- limited optimisation: share of grid combinations profitable 0 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.05 | >= 70% | FAIL |
| core system: median Sharpe across symbols | -0.232 | >= 0.3 | FAIL |
| core system: share of symbols with Sharpe > 0 | 0.3 | >= 60% | FAIL |
| core system: median closed trades per symbol | 94.5 | >= 30 | PASS |
| monkey test: share of random monkeys beaten | 0.297 | >= 90% | FAIL |
| limited optimisation: share of grid combinations profitable | 0 | >= 70% | FAIL |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | -0.26 |
| plateau score of the chosen cell | -0.241 |
| monkey median Sharpe (median run) | -0.155 |
| median CAGR at the chosen parameters | -0.142 |
| worst drawdown at the chosen parameters | -0.841 |

Entry test by exit (fixed size per trade, summed (Davey)):

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.0 | 977 | 0.412 | -0.006 |
| 10 bars | 0.0 | 977 | 0.444 | -0.004 |
| 20 bars | 0.0 | 977 | 0.453 | -0.006 |
| 2/4 ATR bracket | 0.2 | 977 | 0.282 | -0.003 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 376.0 | 0.441 | -0.012 | -4.667 | 48.0 |
| long | 141.0 | 0.426 | -0.008 | -1.184 | 48.0 |
| short | 235.0 | 0.451 | -0.015 | -3.483 | 48.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.192 | -0.374 | -0.0 |
| trend | ranging (ADX<=20) | 0.308 | -0.471 | -0.0 |
| trend | trending (ADX>=25) | 0.5 | -0.164 | -0.0 |
| trend | warm-up | 0.001 | 0.0 | 0.0 |
| volatility | high vol | 0.333 | -0.269 | -0.0 |
| volatility | low vol | 0.333 | -0.468 | -0.0 |
| volatility | mid vol | 0.333 | -0.427 | -0.0 |
| volatility | warm-up | 0.0 | 0.0 | 0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.034 | 0.064 | 0.082 | 0.13 | 1.811 | 4.001 |
| losers | 0.11 | 0.187 | 0.021 | 0.044 | 6.725 | 1.422 |

Report: `reports/research/i009/v1/feasibility/report.html`

Configurations evaluated in this stage: 9

Commit at run time: `9a9b3ad4c8d03ec44642bf6934a82a032f2b54a5` | criteria `5f452e63b833a30c` | card `6723a059916ed99b`
