# i002 v2 - Liquidation shock fade

**Build: PASS**

| Check | Value | Needs | Result |
|---|---|---|---|
| ruff problems | 0 | <= 0 | PASS |
| tests pass (i002_shock_fade) | 1 | >= 1 | PASS |
| line coverage (i002_shock_fade) | 1 | >= 100% | PASS |
| branch coverage (i002_shock_fade) | 1 | >= 100% | PASS |

Notes:

- nearest earlier strategy: screened bollinger_reversion (4h, vol target 25%): correlation 0.22, exposure difference 0.87

Configurations evaluated in this stage: 0

Commit at run time: `7935e263aa550ac7877bc2b29b3cbf14bd84a03d` | criteria `2f728c62a5eb5dab` | card `f11ff0512a243dfc`
