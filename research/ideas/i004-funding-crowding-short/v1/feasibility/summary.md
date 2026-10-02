# i004 v1 - Funding-crowding short

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.075 (needs >= 70%)
- core system: median Sharpe across symbols -0.564 (needs >= 0.3)
- core system: share of symbols with Sharpe > 0 0.1 (needs >= 60%)
- core system: median closed trades per symbol 22.5 (needs >= 30)
- monkey test: share of random monkeys beaten 0.009 (needs >= 90%)
- limited optimisation: share of grid combinations profitable 0 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.075 | >= 70% | FAIL |
| core system: median Sharpe across symbols | -0.564 | >= 0.3 | FAIL |
| core system: share of symbols with Sharpe > 0 | 0.1 | >= 60% | FAIL |
| core system: median closed trades per symbol | 22.5 | >= 30 | FAIL |
| monkey test: share of random monkeys beaten | 0.009 | >= 90% | FAIL |
| limited optimisation: share of grid combinations profitable | 0 | >= 70% | FAIL |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | -0.223 |
| plateau score of the chosen cell | -0.341 |
| monkey median Sharpe (median run) | -0.223 |
| median CAGR at the chosen parameters | -0.0369 |
| worst drawdown at the chosen parameters | -0.622 |

Entry test by exit (fixed size per trade, summed (Davey)):

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.1 | 180 | 0.4 | -0.029 |
| 10 bars | 0.0 | 113 | 0.333 | -0.134 |
| 20 bars | 0.0 | 102 | 0.292 | -0.172 |
| 2/4 ATR bracket | 0.2 | 169 | 0.244 | -0.033 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 135.0 | 0.474 | -0.013 | -1.706 | 3.0 |
| long | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| short | 135.0 | 0.474 | -0.013 | -1.706 | 3.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.206 | -0.191 | 0.0 |
| trend | ranging (ADX<=20) | 0.272 | 0.414 | -0.0 |
| trend | trending (ADX>=25) | 0.508 | -0.357 | -0.0 |
| trend | warm-up | 0.014 | 0.0 | 0.0 |
| volatility | high vol | 0.331 | 0.472 | 0.0 |
| volatility | low vol | 0.331 | -0.31 | -0.0 |
| volatility | mid vol | 0.331 | -0.706 | -0.0 |
| volatility | warm-up | 0.007 | 0.0 | 0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.024 | 0.044 | 0.075 | 0.121 | 0.395 | 1.112 |
| losers | 0.083 | 0.143 | 0.026 | 0.044 | 1.283 | 0.491 |

Report: `reports/research/i004/v1/feasibility/report.html`

Configurations evaluated in this stage: 27

Commit at run time: `a6552cbc12a752a3e561e45599f4da2466f43072` | criteria `05ee199a428ad254` | card `1242c7e9a2436fc1`
