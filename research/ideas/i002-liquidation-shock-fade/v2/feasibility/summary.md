# i002 v2 - Liquidation shock fade

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.6 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.6 | >= 70% | FAIL |
| core system: median Sharpe across symbols | 0.387 | >= 0.3 | PASS |
| core system: share of symbols with Sharpe > 0 | 0.9 | >= 60% | PASS |
| core system: median closed trades per symbol | 73.5 | >= 30 | PASS |
| monkey test: share of random monkeys beaten | 0.98 | >= 90% | PASS |
| limited optimisation: share of grid combinations profitable | 0.778 | >= 70% | PASS |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | 0.141 |
| plateau score of the chosen cell | 0.404 |
| monkey median Sharpe (median run) | 0.0484 |
| median CAGR at the chosen parameters | -0.00127 |
| worst drawdown at the chosen parameters | -0.683 |

Entry test by exit:

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.6 | 727 | 0.568 | 0.007 |
| 10 bars | 0.7 | 706 | 0.594 | 0.008 |
| 20 bars | 0.8 | 662 | 0.562 | 0.015 |
| 2/4 ATR bracket | 0.3 | 680 | 0.339 | 0.006 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 768.0 | 0.548 | 0.003 | 2.187 | 3.0 |
| long | 768.0 | 0.548 | 0.003 | 2.187 | 3.0 |
| short | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.195 | -0.303 | -0.0 |
| trend | ranging (ADX<=20) | 0.298 | 1.014 | 0.0 |
| trend | trending (ADX>=25) | 0.504 | 0.078 | 0.0 |
| trend | warm-up | 0.002 | 0.0 | 0.0 |
| volatility | high vol | 0.333 | 0.207 | 0.0 |
| volatility | low vol | 0.333 | 0.135 | 0.0 |
| volatility | mid vol | 0.333 | -0.212 | -0.0 |
| volatility | warm-up | 0.001 | 0.0 | 0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.018 | 0.034 | 0.039 | 0.067 | 0.515 | 1.138 |
| losers | 0.049 | 0.091 | 0.011 | 0.021 | 1.616 | 0.38 |

Report: `reports/research/i002/v2/feasibility/report.html`

Configurations evaluated in this stage: 9

Commit at run time: `56d7848bf6a88f6ab30b87fe7309040af6626cbc` | criteria `2f728c62a5eb5dab` | card `f11ff0512a243dfc`
