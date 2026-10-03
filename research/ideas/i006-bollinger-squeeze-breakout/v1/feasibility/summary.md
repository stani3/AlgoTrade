# i006 v1 - Bollinger squeeze breakout

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.325 (needs >= 70%)
- core system: median Sharpe across symbols -0.0783 (needs >= 0.3)
- core system: share of symbols with Sharpe > 0 0.5 (needs >= 60%)
- limited optimisation: share of grid combinations profitable 0.222 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.325 | >= 70% | FAIL |
| core system: median Sharpe across symbols | -0.0783 | >= 0.3 | FAIL |
| core system: share of symbols with Sharpe > 0 | 0.5 | >= 60% | FAIL |
| core system: median closed trades per symbol | 482 | >= 30 | PASS |
| monkey test: share of random monkeys beaten | 0.999 | >= 90% | PASS |
| limited optimisation: share of grid combinations profitable | 0.222 | >= 70% | FAIL |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | 0.0727 |
| plateau score of the chosen cell | 0.0745 |
| monkey median Sharpe (median run) | -0.522 |
| median CAGR at the chosen parameters | -0.0257 |
| worst drawdown at the chosen parameters | -0.799 |

Entry test by exit (fixed size per trade, summed (Davey)):

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.2 | 6087 | 0.405 | -0.001 |
| 10 bars | 0.2 | 4728 | 0.427 | -0.001 |
| 20 bars | 0.4 | 3457 | 0.438 | -0.001 |
| 2/4 ATR bracket | 0.5 | 4695 | 0.352 | -0.0 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 2302.0 | 0.406 | 0.001 | 3.316 | 23.0 |
| long | 1091.0 | 0.366 | -0.001 | -0.66 | 21.0 |
| short | 1211.0 | 0.443 | 0.003 | 3.976 | 25.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.192 | -0.1 | 0.0 |
| trend | ranging (ADX<=20) | 0.308 | -1.557 | -0.0 |
| trend | trending (ADX>=25) | 0.5 | 0.791 | 0.0 |
| trend | warm-up | 0.001 | 0.0 | 0.0 |
| volatility | high vol | 0.333 | 0.471 | 0.0 |
| volatility | low vol | 0.333 | -0.762 | -0.0 |
| volatility | mid vol | 0.333 | -0.089 | 0.0 |
| volatility | warm-up | 0.0 | 0.0 | 0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.01 | 0.018 | 0.057 | 0.078 | 1.066 | 6.194 |
| losers | 0.032 | 0.046 | 0.012 | 0.024 | 3.398 | 1.313 |

Report: `reports/research/i006/v1/feasibility/report.html`

Configurations evaluated in this stage: 27

Commit at run time: `468d18002c9cd0eaa7261fc687a51594d13b9b9f` | criteria `5f452e63b833a30c` | card `21b28885975e3b40`
