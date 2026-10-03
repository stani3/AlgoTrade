# i007 v1 - Dual Thrust intraday breakout

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.15 (needs >= 70%)
- core system: median Sharpe across symbols -0.232 (needs >= 0.3)
- core system: share of symbols with Sharpe > 0 0.3 (needs >= 60%)
- limited optimisation: share of grid combinations profitable 0 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.15 | >= 70% | FAIL |
| core system: median Sharpe across symbols | -0.232 | >= 0.3 | FAIL |
| core system: share of symbols with Sharpe > 0 | 0.3 | >= 60% | FAIL |
| core system: median closed trades per symbol | 1.61e+03 | >= 30 | PASS |
| monkey test: share of random monkeys beaten | 1 | >= 90% | PASS |
| limited optimisation: share of grid combinations profitable | 0 | >= 70% | FAIL |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | 0.0484 |
| plateau score of the chosen cell | 0.0511 |
| monkey median Sharpe (median run) | -0.768 |
| median CAGR at the chosen parameters | -0.0407 |
| worst drawdown at the chosen parameters | -0.796 |

Entry test by exit (fixed size per trade, summed (Davey)):

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.0 | 15414 | 0.406 | -0.001 |
| 10 bars | 0.0 | 14446 | 0.443 | -0.001 |
| 20 bars | 0.2 | 11596 | 0.463 | -0.002 |
| 2/4 ATR bracket | 0.4 | 10824 | 0.352 | -0.001 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 2711.0 | 0.457 | 0.001 | 2.599 | 8.0 |
| long | 1437.0 | 0.483 | 0.004 | 6.061 | 8.0 |
| short | 1274.0 | 0.428 | -0.003 | -3.462 | 8.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.192 | -0.072 | -0.0 |
| trend | ranging (ADX<=20) | 0.308 | -0.89 | -0.0 |
| trend | trending (ADX>=25) | 0.5 | 0.312 | 0.0 |
| trend | warm-up | 0.001 | 0.0 | 0.0 |
| volatility | high vol | 0.333 | 0.168 | 0.0 |
| volatility | low vol | 0.333 | -0.62 | -0.0 |
| volatility | mid vol | 0.333 | -0.175 | 0.0 |
| volatility | warm-up | 0.0 | 0.0 | 0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.008 | 0.018 | 0.036 | 0.062 | 0.647 | 2.711 |
| losers | 0.024 | 0.044 | 0.009 | 0.02 | 1.747 | 0.675 |

Report: `reports/research/i007/v1/feasibility/report.html`

Configurations evaluated in this stage: 9

Commit at run time: `c4beddd11d63a446111e24b0512d6d38e8508cdc` | criteria `5f452e63b833a30c` | card `6da0c641ecf59899`
