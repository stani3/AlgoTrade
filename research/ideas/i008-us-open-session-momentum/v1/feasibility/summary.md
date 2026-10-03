# i008 v1 - US-open session momentum

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.35 (needs >= 70%)
- core system: median Sharpe across symbols -0.799 (needs >= 0.3)
- core system: share of symbols with Sharpe > 0 0.1 (needs >= 60%)
- monkey test: share of random monkeys beaten 0.034 (needs >= 90%)
- limited optimisation: share of grid combinations profitable 0 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.35 | >= 70% | FAIL |
| core system: median Sharpe across symbols | -0.799 | >= 0.3 | FAIL |
| core system: share of symbols with Sharpe > 0 | 0.1 | >= 60% | FAIL |
| core system: median closed trades per symbol | 378 | >= 30 | PASS |
| monkey test: share of random monkeys beaten | 0.034 | >= 90% | FAIL |
| limited optimisation: share of grid combinations profitable | 0 | >= 70% | FAIL |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | -0.458 |
| plateau score of the chosen cell | -0.641 |
| monkey median Sharpe (median run) | -0.529 |
| median CAGR at the chosen parameters | -0.138 |
| worst drawdown at the chosen parameters | -0.851 |

Entry test by exit (fixed size per trade, summed (Davey)):

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.1 | 3846 | 0.431 | -0.002 |
| 10 bars | 0.3 | 3846 | 0.45 | -0.002 |
| 20 bars | 0.6 | 3836 | 0.466 | 0.0 |
| 2/4 ATR bracket | 0.4 | 3393 | 0.348 | -0.001 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 1963.0 | 0.443 | -0.002 | -4.323 | 8.0 |
| long | 1058.0 | 0.486 | 0.0 | 0.263 | 8.0 |
| short | 905.0 | 0.393 | -0.005 | -4.586 | 8.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.192 | -0.902 | -0.0 |
| trend | ranging (ADX<=20) | 0.308 | -1.292 | -0.0 |
| trend | trending (ADX>=25) | 0.5 | -0.039 | -0.0 |
| trend | warm-up | 0.001 | 0.0 | 0.0 |
| volatility | high vol | 0.333 | -0.314 | -0.0 |
| volatility | low vol | 0.333 | -0.054 | -0.0 |
| volatility | mid vol | 0.333 | -0.828 | -0.0 |
| volatility | warm-up | 0.0 | 0.0 | 0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.009 | 0.017 | 0.035 | 0.063 | 0.574 | 2.266 |
| losers | 0.034 | 0.057 | 0.009 | 0.017 | 2.019 | 0.598 |

Report: `reports/research/i008/v1/feasibility/report.html`

Configurations evaluated in this stage: 9

Commit at run time: `9dff4b7886ac21011261d1b5078ac2c9b479df7f` | criteria `5f452e63b833a30c` | card `e9fb3b116d66f978`
