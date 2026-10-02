---
id: i001
version: 2
registered: '2026-10-02T14:01:50Z'
title: Book breakout + RSI + ATR bracket
source: User's book example strategy (breakout, RSI filter, ATR stop and target)
taxonomy:
  family: breakout
  inputs:
  - price
  horizon: days
timeframe: 1d
spec:
  type: breakout_bracket
  lookback: 48
  rsi_length: 30
  rsi_level: 50.0
  atr_length: 14
  stop_atr: 2.0
  target_atr: 4.0
  cooldown_win: 20
  cooldown_loss: 5
  kill_drawdown: 0.5
  allow_short: true
optimise:
  stop_atr:
  - 1.0
  - 1.5
  - 2.0
  - 3.0
  - 4.0
  target_atr:
  - 2
  - 3
  - 4
  - 6
  - 8
  atr_length:
  - 14
  - 30
expected_trades_per_year: 10
historical: true
parent: 1
revision_reason: same rules on daily bars (pre-journal)
---
## Hypothesis
A close at the highest (lowest) close of the last 48 four-hour bars, confirmed by RSI(30) above
(below) 50, starts a move that is worth more than a fixed ATR bracket risks.

## Why it should work
Breakouts with momentum confirmation catch the start of trends; a stop two ATRs away keeps the
loss small while a target four ATRs away lets winners pay for several losers.

## Rules
Enter at the next bar's open. Stop `stop_atr` x ATR and target `target_atr` x ATR from the fill,
ATR read on the signal bar. Wait 5 bars after a loss and 20 after a win; stop trading for good at
a 50% drawdown from the equity peak.

## Falsified if
The median Sharpe ratio across the ten-symbol universe stays near zero across the stop/target
grid, or the kill switch trips on most symbols.

## Outcome (recorded by `research seed`)
Tested before the research journal existed, on the full history: the best grid cell reached a
median Sharpe of about 0.25 (4h) and the kill switch fired on most symbols. Failed.
