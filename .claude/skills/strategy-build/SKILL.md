---
name: strategy-build
description: Implement a registered AlgoTrade idea card as a strategy spec, writing new strategy code plus a full test suite (100% line and branch coverage, hand-computed long/short cases, lookahead checks, reference implementation for loops) when the catalogue cannot express it, then pass `research build-check`. Use after strategy-ideas registered an idea, or when the user asks to build/implement idea iNNN.
---

# Strategy build

Turn one registered idea into code that does exactly what its card says, proven by tests.
You do not tune, you do not look at performance, and you never edit the card.

## 1. Read the card

`research/ideas/<id>-<slug>/v<n>/idea.md`. The `spec` and `optimise` grid are fixed. If the card
is wrong or impossible to implement as written, stop: either the user revises it
(`python -m scripts.research revise ...`, which uses the revision budget) or you close it with
`python -m scripts.research abandon <id> --reason "..."`.

## 2. Catalogue composition or new code?

- If every `type` in the spec already exists (catalogue rules and wrappers), there is no code to
  write: go straight to step 4.
- Otherwise each new type (`iNNN_something`) gets one module and one test file:
  - `src/algotrade/strategies/ideas/iNNN_something.py`
  - `tests/ideas/test_iNNN_something.py`

  Classes in `strategies/ideas/` are registered automatically. Never edit or delete another
  idea's module: tested versions must keep their behaviour (a regression test recomputes every
  stored fingerprint).

## 3. Write the strategy and its tests

Strategy code, following the existing rules in `src/algotrade/strategies/`:

- A `@dataclass(frozen=True)` subclass of `Strategy` (position rules: `target_position(bars)`
  returning exposure in [-1, 1] decided at each bar's close) or of `BracketStrategy`
  (`signals(bars)` returning `BracketSignals`, plus `cooldown_win`, `cooldown_loss`,
  `kill_drawdown` fields), with `name = "iNNN_something"` matching the spec type.
- Parameter names exactly as in the card's spec, defaults equal to the card's values.
- Validate parameters in `__post_init__` (raise `ValueError` on nonsense).
- Only data up to each bar's close; no `shift(-1)`, no centred windows, no full-sample scaling.
- Import from submodules (`algotrade.strategies.base`, `algotrade.indicators`,
  `algotrade.backtest.bracket`), never from `algotrade.strategies` itself.
- New indicators go in the module (or `indicators.py` with their own tests).

Tests, to the same standard as `tests/test_bracket_*.py`. Every item applies:

1. **Hand-computed cases** on tiny bar sets where you work out the expected targets/signals by
   hand. Run every case for long and short with `pytest.mark.parametrize` and mirrored prices
   (see `directional()` in `tests/bracket_helpers.py`).
2. **Every boundary**: ties, `>` vs `>=`, the first bar after warm-up, NaN indicators during
   warm-up, `allow_short=False`, zero/negative funding where funding matters.
3. **Parameter validation**: each `ValueError` path.
4. **No lookahead**: the strategy on `bars.iloc[:cut]` equals the first `cut` rows on the full
   bars, for several cuts (for brackets compare the `simulate(...)` ledgers).
5. **Spec round trip** (`from_spec(to_spec(s)) == s`) and one run through
   `algotrade.backtest.runner.backtest`.
6. **Reference implementation** for any bar-by-bar loop you write: a deliberately plain,
   separately written Python version in the test file, compared on many random markets
   (`random_bars` in `tests/bracket_helpers.py`), long and short, with and without costs.
7. **Smoke run on real BTC data** (`algotrade.backtest.runner.load_bars`), skipped with
   `pytest.skip` when the file is missing; assert only that it runs and trades, never on
   performance.
8. **Session markets**, when the card's universe is not only crypto: hand-computed cases on
   bars with overnight and weekend gaps (US sessions 09:30-16:00 New York, the forex week
   Sunday to Friday 17:00; `session_index` and `random_walk` in `tests/research_helpers.py`
   build them), showing that rules which count bars or days (`bars_per_day`, lookbacks in
   days) mean sessions there and that nothing assumes 24 bars a day.

The `bars` fixture (`tests/conftest.py`) and `bracket_helpers` are importable from
`tests/ideas/`. The generic checks in `tests/ideas/test_ideas_generic.py` and
`tests/test_strategies.py` also run on your class automatically.

Never use `# pragma: no cover`, `skip` to dodge a failing case, or loosen an assertion to make
coverage or tests pass.

## 4. Pass the gate

```bash
python -m scripts.research build-check <id>
```

It runs ruff, your tests under branch coverage (Numba JIT off) and requires 100% line and branch
coverage of your module, builds every grid combination, and checks the idea is not a duplicate
of an earlier one by configuration or by behaviour (position fingerprint on BTC/ETH/SOL).

- "BUILD NOT READY": code-quality problems, nothing recorded. Fix the code or tests (never the
  card) and run it again.
- "BUILD FAIL" with a duplicate: the idea is closed and committed as failed. Do not rework the
  code to dodge the fingerprint; report it.
- "BUILD PASS": the code, tests, fingerprint and result are committed. Feasibility can run.

Also run the whole suite once (`pytest -q`) to be sure nothing else broke.

## Output

The build verdict, the files written, the coverage line, and the nearest earlier strategy the
fingerprint check reported.
