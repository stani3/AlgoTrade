# i003 v2 - Weekly-horizon time-series momentum

**Feasibility: FAIL**

Why it failed:

- entry test: share of (exit, symbol) cells profitable 0.45 (needs >= 70%)

| Check | Value | Needs | Result |
|---|---|---|---|
| entry test: share of (exit, symbol) cells profitable | 0.45 | >= 70% | FAIL |
| core system: median Sharpe across symbols | 0.947 | >= 0.3 | PASS |
| core system: share of symbols with Sharpe > 0 | 0.9 | >= 60% | PASS |
| core system: median closed trades per symbol | 75 | >= 30 | PASS |
| monkey test: share of random monkeys beaten | 1 | >= 90% | PASS |
| limited optimisation: share of grid combinations profitable | 1 | >= 70% | PASS |

| Metric | Value |
|---|---|
| median Sharpe at the chosen parameters | 0.947 |
| plateau score of the chosen cell | 0.999 |
| monkey median Sharpe (median run) | 0.48 |
| median CAGR at the chosen parameters | 0.178 |
| worst drawdown at the chosen parameters | -0.488 |

Entry test by exit:

| exit | profitable | trades | win_rate | avg_trade |
|---|---|---|---|---|
| 5 bars | 0.2 | 619 | 0.445 | 0.002 |
| 10 bars | 0.5 | 493 | 0.436 | 0.008 |
| 20 bars | 0.7 | 384 | 0.5 | 0.047 |
| 2/4 ATR bracket | 0.4 | 332 | 0.394 | 0.008 |

Long against short (chosen parameters):

| index | trades | win_rate | avg_return | sum_return | median_bars |
|---|---|---|---|---|---|
| all | 756.0 | 0.347 | 0.013 | 9.827 | 3.0 |
| long | 756.0 | 0.347 | 0.013 | 9.827 | 3.0 |
| short | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Regimes (chosen parameters):

| kind | regime | share_of_time | median_sharpe | mean_net |
|---|---|---|---|---|
| trend | neither | 0.206 | 0.117 | 0.0 |
| trend | ranging (ADX<=20) | 0.272 | 0.394 | 0.0 |
| trend | trending (ADX>=25) | 0.508 | 1.614 | 0.001 |
| trend | warm-up | 0.014 | 0.0 | 0.0 |
| volatility | high vol | 0.331 | -0.88 | -0.001 |
| volatility | low vol | 0.331 | 2.741 | 0.002 |
| volatility | mid vol | 0.331 | 1.001 | 0.001 |
| volatility | warm-up | 0.007 | 0.0 | 0.0 |

Excursions (MAE/MFE):

| index | median_mae | p75_mae | median_mfe | p75_mfe | median_mae_atr | median_mfe_atr |
|---|---|---|---|---|---|---|
| winners | 0.02 | 0.044 | 0.207 | 0.566 | 0.327 | 2.918 |
| losers | 0.061 | 0.107 | 0.023 | 0.055 | 1.025 | 0.377 |

Report: `reports/research/i003/v2/feasibility/report.html`

Configurations evaluated in this stage: 5

Commit at run time: `ec9474071ee82485dbbff97046729d45170fa61c` | criteria `2f728c62a5eb5dab` | card `178a7b2614a0a56c`
