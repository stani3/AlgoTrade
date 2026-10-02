# i001 v2 - Book breakout + RSI + ATR bracket

**Feasibility: FAIL**

Why it failed:

- best median Sharpe in the grid 0.228 (needs >= 0.3)

| Check | Value | Needs | Result |
|---|---|---|---|
| best median Sharpe in the grid | 0.228 | >= 0.3 | FAIL |

| Metric | Value |
|---|---|
| combinations | 50 |
| share of combinations with median Sharpe > 0 | 0.32 |
| share of symbols where the kill switch fired | 0.974 |

Notes:

- Pre-journal grid search on the full history 2019-2026, holdout included (pre-journal).

Configurations evaluated in this stage: 50

Commit at run time: `b0da49e08f2d3ce15f4c5d189ff26e38220ff656` | criteria `2f728c62a5eb5dab` | card `868d65c0d35c55a7`
