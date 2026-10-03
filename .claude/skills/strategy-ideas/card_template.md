---
title: Short descriptive name
source: "Book, paper or convention the rules come from (author, title)"
source_id: S00            # id from sources.md, or omit for an original idea
taxonomy:
  family: trend           # trend | breakout | mean_reversion | carry | volatility | seasonality | other
  inputs: [price]         # any of: price, volume, funding, calendar
  horizon: days           # typical holding time: hours | days | weeks | months
timeframe: 4h             # 1h | 4h | 1d
universe: [crypto]        # [crypto] (default), asset classes from data.universes, or all
spec:                     # existing catalogue type(s), or a new snake_case type for new code
  type: new_rule_name
  lookback: 20
optimise:                 # pre-registered grid: <= 3 parameters, <= 100 combinations
  lookback: [10, 20, 40]
expected_trades_per_year: 12
differs_from:             # one entry per earlier neighbour that `research new` names
  i000: "What is genuinely different about this idea, in a sentence or two."
---
## Hypothesis
One or two sentences: the effect this idea claims exists.

## Why it should work
The behavioural or structural reason, and who is on the other side of the trade.

## Rules
Entry, exit and sizing, precise enough to code without guessing. State which bar's data each
decision uses (the close of the signal bar; fills at the next open or the close, per engine).

## Falsified if
The concrete result that would show the idea does not work (for example: median Sharpe across
symbols below 0.3, or no better than random entries).
