# i005 v1 - Funding-settlement rebound

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.15 (needs >= 70%)
- core system: median Sharpe across symbols -0.0971 (needs >= 0.3)
- core system: share of symbols with Sharpe > 0 0.4 (needs >= 60%)
- limited optimisation: share of grid combinations profitable 0.111 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.15 | >= 70% | FAIL |
| core system: median Sharpe across symbols | -0.0971 | >= 0.3 | FAIL |
| core system: share of symbols with Sharpe > 0 | 0.4 | >= 60% | FAIL |
| core system: median closed trades per symbol | 464 | >= 30 | PASS |
| monkey test: share of random monkeys beaten | 1 | >= 90% | PASS |
| limited optimisation: share of grid combinations profitable | 0.111 | >= 70% | FAIL |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | 0.139 |
| plateau score of the chosen cell | -0.0873 |
| monkey median Sharpe (median run) | -1.11 |
| median CAGR at the chosen parameters | 0.00956 |
| worst drawdown at the chosen parameters | -0.341 |

Entry test by exit (fixed size per trade, summed (Davey)):

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.1 | 4988 | 0.473 | -0.002 |
| 10 bars | 0.3 | 3003 | 0.495 | -0.001 |
| 20 bars | 0.1 | 2308 | 0.46 | -0.004 |
| 2/4 ATR bracket | 0.1 | 2017 | 0.319 | -0.004 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 1814.0 | 0.48 | 0.0 | 0.367 | 1.0 |
| long | 1431.0 | 0.483 | 0.0 | 0.254 | 1.0 |
| short | 383.0 | 0.47 | 0.0 | 0.114 | 1.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.192 | -0.025 | -0.0 |
| trend | ranging (ADX<=20) | 0.308 | -0.264 | 0.0 |
| trend | trending (ADX>=25) | 0.5 | 0.064 | 0.0 |
| trend | warm-up | 0.001 | 0.0 | -0.0 |
| volatility | high vol | 0.333 | 0.261 | 0.0 |
| volatility | low vol | 0.333 | -0.724 | -0.0 |
| volatility | mid vol | 0.333 | -0.653 | -0.0 |
| volatility | warm-up | 0.0 | 0.0 | -0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.004 | 0.009 | 0.019 | 0.032 | 0.177 | 0.838 |
| losers | 0.013 | 0.024 | 0.008 | 0.013 | 0.651 | 0.364 |

Report: `reports/research/i005/v1/feasibility/report.html`

Configurations evaluated in this stage: 9

Commit at run time: `21f4f923ad840685ed1728f90631e3b0f2de9c0c` | criteria `5f452e63b833a30c` | card `04233f291b728929`
