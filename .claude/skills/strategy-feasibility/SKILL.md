---
name: strategy-feasibility
description: Run Davey's limited feasibility testing on a built AlgoTrade idea (entry test, core system, monkey test, limited optimisation over the pre-registered grid, diagnostics) with `research feasibility`, read the evidence for patterns, and decide whether one hypothesis-driven revision is justified. Use after strategy-build passed, or when the user asks to test idea iNNN for feasibility.
---

# Strategy feasibility

The gate decides PASS or FAIL; you read the evidence, explain it, and may propose at most one
revision. All of it runs on development data only (before `data.dev_end` in
`research/criteria.yaml`); you never touch the holdout here.

## 1. Run the gate

```bash
python -m scripts.research feasibility <id>
```

It refuses to run unless the build passed and the card, code and tests are committed. It runs:

1. **Entry test**: the strategy's entries with neutral exits (5/10/20-bar time exits and a
   2/4 ATR bracket) on every symbol; a good entry is profitable in most cells.
2. **Core system** at the card's own parameters: median Sharpe across symbols, share of symbols
   positive, enough trades.
3. **Monkey test**: share of random strategies with the same habits (signal frequency and
   exits, or the same exposure series shifted in time) that it beats.
4. **Limited optimisation** over the card's pre-registered grid only: the share of combinations
   that make money, and the parameters at the centre of the best plateau (not the best cell).
5. **Diagnostics** at the chosen parameters, saved next to the result.

Every grid combination is added to `research/trials.csv`, and the result is committed whatever
the verdict.

## 2. Read the evidence

Open `research/ideas/<id>-<slug>/v<n>/feasibility/summary.md`, the `diagnostics_*.csv` files,
and the Davey report (`reports/research/<id>/v<n>/feasibility/report.html`, served by the
"reports" entry in `.claude/launch.json`). Look for patterns, and say what each one means:

- **Long vs short** (`diagnostics_sides.csv`): does one side carry the result while the other
  bleeds?
- **Regimes** (`diagnostics_regimes.csv`): does it only work when ADX says the market trends, or
  in one volatility tercile?
- **Symbols and years**: is it one coin or one year (2021's bull run, say) doing all the work?
- **Holding times and excursions** (MAE/MFE): do winners first go against the trade by more than
  the stop allows? Do losers ever go into profit first, so a target would have saved them?
- **Entry test by exit**: if the entry only works with one exit, the edge may be in the exit.
- **Grid** (`grid.csv`): a broad plateau, or one bright cell surrounded by losses?

## 3. Decide

- **PASS**: report it. The next stage is `strategy-validate`, with the chosen parameters
  recorded in the result (`chosen_spec`).
- **FAIL with no clear reason**: the idea is done. Report why, in one or two sentences.
- **FAIL, or PASS with a clear structural weakness**: you may propose ONE revision, and only if:
  - it follows from a pattern above, not from wanting a better number ("shorts lose in every
    regime and every year, so long-only"; "MAE shows 80% of winners dip more than 1.5 ATR, so
    the 1 ATR stop cuts them");
  - it is a rule change or a regime filter, not a parameter nudge to the best cell;
  - the card still has revision budget (`budgets.revisions_per_idea`; feasibility only; never
    after validation started).

  Write the revised card (copy the stored one, change the rules, `optimise` grid if the change
  needs one, and the Rules section), then:

  ```bash
  python -m scripts.research revise <id> <draft.md> --reason "<the pattern that justifies it>"
  ```

  and hand it back to `strategy-build` (new code, or a defaulted parameter on the existing
  class; never change the behaviour of the tested version) and then feasibility again.

Never: run extra grids or backtests to "see if it helps" before revising (every look counts and
the revision must be decided from the evidence already produced), edit `criteria.yaml`, or
re-run feasibility on a version that already has a result.

## Output

Verdict, the failed checks with values, the chosen parameters, the two or three patterns that
matter most, and either "done" or the revision you registered with its reason.
