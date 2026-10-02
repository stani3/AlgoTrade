# i001 v1 - Book breakout + RSI + ATR bracket

**Feasibility: FAIL**

Why it failed:

- best median Sharpe in the grid 0.248 (needs >= 0.3)

| Check | Value | Needs | Result |
|---|---|---|---|
| best median Sharpe in the grid | 0.248 | >= 0.3 | FAIL |

| Metric | Value |
|---|---|
| combinations | 100 |
| share of combinations with median Sharpe > 0 | 0.22 |
| share of symbols where the kill switch fired | 0.998 |

Notes:

- Pre-journal grid search on the full history 2019-2026, holdout included (pre-journal).

Configurations evaluated in this stage: 100

Commit at run time: `b0da49e08f2d3ce15f4c5d189ff26e38220ff656` | criteria `2f728c62a5eb5dab` | card `205b9e9606442071`
