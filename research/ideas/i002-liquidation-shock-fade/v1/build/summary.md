# i002 v1 - Liquidation shock fade

**Build: PASS**

| Check | Value | Needs | Result |
|---|---|---|---|
| ruff problems | 0 | <= 0 | PASS |
| tests pass (i002_shock_fade) | 1 | >= 1 | PASS |
| line coverage (i002_shock_fade) | 1 | >= 100% | PASS |
| branch coverage (i002_shock_fade) | 1 | >= 100% | PASS |

Notes:

- nearest earlier strategy: screened bollinger_reversion (4h, vol target 25%): correlation 0.30, exposure difference 0.77

Configurations evaluated in this stage: 0

Commit at run time: `06c794627aff2afcb2d2bdc8968363568de2a374` | criteria `2f728c62a5eb5dab` | card `56640820ca066041`
