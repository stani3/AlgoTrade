---
id: i002
version: 2
registered: '2026-10-02T15:43:52Z'
parent: 1
revision_reason: 'long only: shorts (fading up-shocks) lost on all 10 symbols and in 5 of 7 years, winning
  only in the 2022 and early-2025 down markets, while longs (fading down-shocks) won 58% of 426 trades
  with a positive median trade on all 10 symbols - the forced-side asymmetry the card predicted (v1 diagnostics
  at z_entry 3.5, hold_bars 12)'
title: Liquidation shock fade
source: 'Crypto-perpetual liquidation overshoot (sources.md S24); reversal after forced trading as pay
  for liquidity provision: Coval & Stafford, ''Asset Fire Sales (and Purchases) in Equity Markets'' (JFE
  2007); Nagel, ''Evaporating Liquidity'' (RFS 2012); Brunnermeier & Pedersen, ''Market Liquidity and
  Funding Liquidity'' (RFS 2009)'
source_id: S24
taxonomy:
  family: mean_reversion
  inputs:
  - price
  horizon: days
timeframe: 4h
spec:
  type: i002_shock_fade
  z_entry: 3.0
  hold_bars: 6
  vol_days: 25.0
  allow_short: false
optimise:
  z_entry:
  - 2.5
  - 3.0
  - 3.5
  hold_bars:
  - 3
  - 6
  - 12
expected_trades_per_year: 14
differs_from:
  i001: i001 bets on continuation (a new 48-bar closing high or low with RSI confirmation, ATR stop and
    target); this fades single-bar moves scaled by prior volatility, the opposite effect, with a fixed
    time exit and no stop or target.
---
## Hypothesis
On 4h bars of liquid crypto perpetuals, a close-to-close fall of at least three times the
volatility known before it is partly a forced-flow overshoot (long liquidations and stop
cascades) that reverses over the following day or so. Buying at the close of the down-shock bar
and holding for a fixed number of bars makes money after costs and funding. Up-shocks are not
traded: they are not the forced side, and in v1 they kept going.

## Why it should work
Perpetuals are traded with high leverage, and an exchange closes an under-margined position with
a market order whatever the price; clustered stop-losses do the same. A wave of such sell orders
(a long liquidation cascade) pushes the price further down than unforced traders would take it,
because the sellers are not choosing the price. When the forced flow is exhausted, part of the
move comes back.

The other side of the trade is the liquidated or stopped-out trader who pays for immediacy; the
fade is paid for supplying liquidity at the extreme. This is the price pressure seen after fund
fire sales (Coval & Stafford 2007), short-term reversal as compensation for liquidity provision
that grows with volatility (Nagel 2012), and the margin spirals of Brunnermeier & Pedersen (2009),
in a market where leverage and automatic liquidation make forced selling unusually common.

Scoring each move against the volatility known at the previous close picks out bars where
volatility jumps, the signature of a cascade, and ignores ordinary large bars in an already
volatile market.

The rule is now one-sided. v1 already argued the asymmetry: funding is positive most of the time,
so longs are usually the more leveraged side and down-shocks the more forced; it traded both
sides anyway and had the diagnostics report them separately. They split cleanly (v1 feasibility
diagnostics, at the chosen z_entry 3.5 and hold_bars 12, net of costs and funding). Fading
up-shocks lost on all ten symbols (470 trades, 44% winners, -2.9% a trade, -13.4 summed; still
-10.2 without DOGE, whose January 2021 squeeze ruined that account) and in five of seven years,
winning only in the falling markets of 2022 and early 2025; its worst months were rallies
(January and February 2021, November 2020, November 2024). An up-shock in this market tends to
start or extend a rally rather than mark forced buying. Fading down-shocks won 58% of 426 trades
at +0.7% a trade, with a positive median trade on all ten symbols and a positive sum on eight.
v2 keeps only the side the mechanism names.

What can still sink it: v1's long trades lost heavily in systemic crashes, when the cascade kept
going (March 2020, May 2021, January and November 2022); their profit came mostly from 2021 and
2024-25, with 2020 and 2022 losing; and a long-only rule in a market that rose over most of the
period earns some drift that has nothing to do with reversal. The monkey test (random timing
with the same exposure) separates reversal from drift.

## Rules
Position strategy (`target_position` returns 0 or +1) on 4h bars: the existing `i002_shock_fade`
code, built and tested for v1, with `allow_short` false; no new code. Every value uses only bars
up to and including the bar it is computed on. Rules 1-5, 7 and 8 are v1's; rule 6 is now in
force.

1. Return: `r_t = close_t / close_{t-1} - 1` (undefined on the first bar).
2. Volatility: `sigma_t = indicators.ewm_vol(r, span)` with `span = vol_days x bars per day`
   (25 x 6 = 150 bars on 4h; `ewm_vol`'s own `min_periods`, span // 2, is the warm-up).
3. Shock score: `z_t = r_t / sigma_{t-1}`, measured against the volatility known at the
   previous close so the shock does not dilute its own yardstick. No signal while
   `sigma_{t-1}` is undefined or not positive, or `r_t` is undefined.
4. Bar t is a shock when `|z_t| >= z_entry`. Its trade direction is `-sign(r_t)`: long after a
   down-shock, short after an up-shock (the short is removed by rule 6).
5. Target at the close of bar t: the trade direction of the most recent shock bar s with
   `t - hold_bars < s <= t`, or 0 if there is none. A position therefore opens at the close of
   the shock bar, is held for exactly `hold_bars` bars (it earns the returns of bars s+1 to
   s+hold_bars) and is closed at the close of bar s+hold_bars. A new down-shock during a trade
   restarts the clock.
6. `allow_short` false (the v2 change): short targets become 0. An up-shock opens nothing; if it
   comes while a long is open, it closes the long at that bar's close (the most recent shock
   decides the target), and the position stays flat until the next down-shock.
7. Size: full 1x exposure, no stop, no target, no volatility scaling. Execution, costs and
   funding are the engine's: the target set at a bar's close is held through the next bar,
   every change in exposure pays fee plus slippage, and funding is charged on open positions.
8. Validation: `z_entry > 0`, `hold_bars` an integer >= 1, `vol_days > 0`; every grid cell is
   valid. Defaults as in v1: `z_entry` 3.0 (the conventional three-sigma outlier line),
   `hold_bars` 6 (one day of 4h bars), `vol_days` 25 (Carver's volatility lookback), now with
   `allow_short` false. The optimisation grid is v1's, unchanged. About 14 trades a year per
   symbol are expected, roughly half of v1's 30.

## Falsified if
- The core system at the registered parameters fails feasibility: median Sharpe across the ten
  symbols below 0.3, Sharpe positive on fewer than 60% of symbols, fewer than 30 closed trades
  per symbol, or random timing with the same exposure (monkey test) beaten less than 90% of the
  time. Failing the monkey test would mean v1's long-side gain was the market's upward drift,
  not reversal.
- The entry test (down-shock long entries with 5, 10 and 20-bar time exits and the 2/4 ATR
  bracket) is profitable in fewer than 70% of (exit, symbol) cells. If prices keep falling after
  down-shocks, even the forced side carries information rather than forced flow, and the idea is
  wrong.
- Fewer than 70% of the nine grid cells make money: the effect would hinge on one exact
  threshold or holding time rather than being a broad property of down-shocks.
