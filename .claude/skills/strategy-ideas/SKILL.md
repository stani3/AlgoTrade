---
name: strategy-ideas
description: Propose 1-3 new trading strategy ideas for the AlgoTrade research pipeline and pre-register them as idea cards (research/ideas/), drawing on trading books (Davey, Carver), classic indicator strategies and crypto-perpetual effects, without repeating anything already tried. Use when the user asks for new strategy ideas, or when strategy-pipeline needs its next idea.
---

# Strategy ideas

You propose and pre-register ideas. You never test them: no `scripts.run_backtest`, `scan`,
`grid` or notebooks here. Choosing an idea because you peeked at how it performs is the
overfitting this pipeline exists to prevent.

## 1. Read what has been done

1. Run `python -m scripts.research status`, then read `research/index.md`: every idea so far,
   its family/inputs/horizon, how far it got and why it stopped, the screened catalogue rules,
   and the sources already used.
2. Read [sources.md](sources.md) (idea catalogue with source ids) and skim
   `src/algotrade/indicators.py` and the "Strategy catalogue" section of `README.md` for what
   already exists.
3. Data available: 4h and 1d OHLCV plus funding rates for 10 USDT perpetuals
   (`research/criteria.yaml` > `data`). No order book, open interest, liquidations or on-chain
   data, and strategies trade one instrument at a time. Ideas that need anything else are out.

## 2. Choose ideas worth testing

Prefer, in this order:

- **A reason to work.** Name the behavioural or structural effect and who is on the other side
  (forced sellers, funding-paying longs, slow-to-react holders...). "The indicator crossed" is
  not a reason.
- **Different from what failed.** Read the failure reasons. A new idea should attack a different
  effect, horizon or input, not re-skin a failed rule.
- **Simple.** At most 3 optimisable parameters, defaults taken from the source (book, paper,
  common convention), never from data.
- **Enough trades.** Feasibility needs at least 30 closed trades per symbol over the development
  period (about 5 years for most symbols), so roughly 6+ trades per year.

## 3. Write the card

Copy [card_template.md](card_template.md) into the scratchpad and fill every field:

- `spec`: either a composition of existing catalogue types (see `specs/*.json`), or a new
  snake_case type name without any id prefix (`funding_fade`): `research new` adds the idea id,
  and `strategy-build` writes the code.
- `optimise`: the pre-registered grid. Coarse (3-5 values per parameter) and centred on the
  source defaults; at most 3 parameters and 100 combinations. Dotted paths for wrapped
  parameters (`strategy.fast`). This is the only grid feasibility and walk-forward will ever
  run, so choose it now, before seeing any result.
- `taxonomy`: family, inputs, horizon. Be honest; these drive the duplicate checks.
- `differs_from`: for every earlier idea with the same family and horizon and an input in
  common, one or two sentences on what is genuinely different (at least 30 characters).
- Body sections: Hypothesis, Why it should work, Rules (precise enough to code without
  guessing), Falsified if.

## 4. Register

```bash
python -m scripts.research new <path-to-draft.md>
```

It validates the card, refuses duplicates (exact configuration, overlapping parameter region,
same rules on another timeframe, unexplained neighbours), assigns the id, stores the card under
`research/ideas/<id>-<slug>/v1/idea.md` and commits it. If it is refused as a duplicate, write a
genuinely different idea; do not nudge parameters to slip past the check. If it is refused for a
missing `differs_from`, add honest notes or drop the idea.

Never pass `--retest`: re-testing a failed idea is the user's decision alone.

## Output

Report each registered id, title and one-line hypothesis, plus any idea you dropped and why.
