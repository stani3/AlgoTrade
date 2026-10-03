---
name: strategy-validate
description: Validate an AlgoTrade idea that passed feasibility - walk-forward analysis, deflated Sharpe ratio against every trial so far, Davey's Monte Carlo stake, then the single holdout look and freeze - and present the approval package (Davey reports plus decision summary) before asking the user whether to incubate. Use after strategy-feasibility passed, or when the user asks to validate idea iNNN.
---

# Strategy validate

Three gates in a row; any failure ends the idea (no revisions after validation starts). You run
them, explain the results, and, only if everything passed, put the decision in front of the
user. You never decide to incubate.

## 1. Walk-forward, deflated Sharpe, stake

```bash
python -m scripts.research validate <id>
```

On development data it re-optimises the card's pre-registered grid every 6 months on the
previous 2 years and trades the next 6 months, stitching the out-of-sample pieces. It checks
walk-forward efficiency, out-of-sample median Sharpe and share of profitable windows; the
deflated Sharpe ratio of the median symbol against every configuration in
`research/trials.csv`; and runs Davey's Monte Carlo on the out-of-sample trades to set the
stake (largest size with risk of ruin < 10%, median drawdown < 40%, return/drawdown > 2).

If it fails, report the failed checks and stop: the idea is closed.

## 2. The holdout (one look)

```bash
python -m scripts.research holdout <id>
```

Only after validation passed. It journals the look first, then trades the chosen parameters on
the data after `data.dev_end` (about 17 months). It passes if the median symbol keeps a positive
Sharpe and its drawdown stays inside the 95th percentile of bootstrapped paths of the same
length. Never run it twice and never pass `--force`: a second look is the user's decision alone.

If it fails, report and stop.

## 3. Freeze

```bash
python -m scripts.research freeze <id>
```

Writes `frozen.json` (the exact spec, timeframe, stake and symbols incubation will trade;
for ideas about other asset classes also their `universe`),
`decision.md`, commits and tags `strategy/<id>-v<n>`.

## 4. The approval package (always before asking)

1. Send the reports with SendUserFile:
   `reports/research/<id>/v<n>/validation/report.html` (walk-forward out-of-sample) and
   `reports/research/<id>/v<n>/holdout/report.html`. If one is missing (for example after a
   fresh clone), rebuild it with `python -m scripts.research report <id> <stage>`.
2. Open them in the browser pane: start the "reports" server from `.claude/launch.json` and
   navigate to `http://127.0.0.1:8765/research/<id>/v<n>/validation/report.html`.
3. Summarise `research/ideas/<id>-<slug>/v<n>/decision.md`: every gate with value and limit, the
   chosen parameters and why (plateau centre), the stake and its Monte Carlo ruin and drawdown,
   the deflated Sharpe with the trial count, the version history, and the known weaknesses.
4. Then ask the user whether to start incubation (paper trading on testnet). Make clear that
   incubation is paper trading and that real money is never part of this pipeline. A strategy
   whose card names other asset classes than crypto cannot be paper traded yet (the testnets
   are Binance and Bybit): present the package, say so, and ask the user what they want to do
   instead of asking about incubation.

## Never

Re-run a stage that already has a result, edit `research/criteria.yaml` or the card, change the
frozen spec, start incubation without an explicit yes, or ask for approval without the reports.

## Output

Verdict per stage with the failed checks, the stake, the deflated Sharpe and trial count, and,
for a frozen strategy, the approval question.
