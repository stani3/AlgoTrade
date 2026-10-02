---
id: i003
version: 2
registered: '2026-10-02T16:12:19Z'
parent: 1
revision_reason: 'long only: v1 shorts lost -2.49 over 759 trades (-0.3% a trade) on 7 of 10 coins, lost
  in 2020, 2023, 2024 and early 2025, broke even in 2021 and won only in the 2022 bear market (+0.30 over
  168 trades), with an inverted payoff (best +26%, worst -49%), while longs made +9.83 over 756 trades
  on 9 of 10 coins (best +91%, worst -12%); the entry test''s four -100% cells (ETH/ADA 20-day, DOGE 10/20-day)
  can only be 1x shorts caught in a doubling (v1 diagnostics, lookback 28)'
title: Weekly-horizon time-series momentum
source: 'Moskowitz, Ooi & Pedersen, ''Time Series Momentum'' (JFE 2012): hold the sign of the past return,
  position scaled by ex-ante volatility (sources.md S02, sizing S41); Liu & Tsyvinski, ''Risks and Returns
  of Cryptocurrency'' (RFS 2021): significant crypto time-series momentum at daily and weekly frequency,
  past-week winners beat losers over the next 1-4 weeks; Schmeling, Schrimpf & Todorov, ''Crypto Carry''
  (BIS WP 1087): leveraged demand from trend-chasing investors; Carver, Systematic Trading: 25% volatility
  target, 25-day volatility lookback'
source_id: S02
taxonomy:
  family: trend
  inputs:
  - price
  horizon: weeks
timeframe: 1d
spec:
  type: vol_target
  annual_vol: 0.25
  vol_days: 25.0
  max_leverage: 1.0
  strategy:
    type: momentum
    lookback: 28
    allow_short: false
optimise:
  strategy.lookback:
  - 7
  - 14
  - 28
  - 56
  - 112
expected_trades_per_year: 14
differs_from:
  i001: i001 enters on a new lookback-bar closing high or low confirmed by RSI and exits on a 2x ATR stop
    or 4x ATR profit target, so its trades last days and its winners are capped; this holds a volatility-sized
    long while the four-week return is positive and is flat otherwise, with no stop or target until the
    return turns, betting on multi-week continuation of rallies whose payoff sits in the uncapped large
    trends.
---
## Hypothesis
On daily bars of liquid crypto perpetuals, a positive past four-week return predicts further
gains over the following weeks: rallies continue (time-series momentum on the long side).
Holding a volatility-sized long while the four-week return is positive, and staying flat while
it is not, makes money after fees, slippage and funding on most of the ten coins, across
lookbacks from one to sixteen weeks rather than at one lucky horizon, and it does so through its
timing rather than by holding the market's upward drift. Declines are no longer traded: in v1 the
short side lost.

## Why it should work
The v1 mechanism, unchanged for rallies: news and flows in crypto are priced over weeks rather
than at once (no cash flows to anchor value, a fragmented, retail-heavy investor base,
attention-driven flows), and Liu & Tsyvinski (2021) find significant crypto time-series momentum
at daily and weekly frequency, the underreaction-then-herding pattern Moskowitz, Ooi & Pedersen
(2012) document in other futures markets. Leverage adds positive feedback: Schmeling, Schrimpf &
Todorov trace the large crypto carry to demand for leveraged exposure from trend-chasing
investors, and the forced buying of liquidated shorts carries rallies further.

That feedback is one-sided in this market. The leveraged trend-chasers are mostly long (funding
is positive most of the time; it is the price of that demand), so rallies are fed by new
leveraged buying and by short squeezes. Declines are different: long-liquidation cascades are
fast and over in days, and dip buyers and squeezes against the new shorts then take part of the
fall back, so a four-week decline is a weaker signal than a four-week rise, and a short that
enters after one carries the squeeze risk.

v1 split cleanly by side (feasibility diagnostics at the chosen lookback 28, net of costs and
funding):

- Shorts: 759 trades, 32% winners, -0.3% a trade, -2.49 summed; negative on 7 of 10 coins (only
  BTC, DOGE and SOL slightly positive). They lost in 2020 (-0.30), 2023 (-0.69), 2024 (-1.48)
  and early 2025 (-0.21), broke even in 2021 (+0.02), and won only in the 2022 bear market, and
  then only +0.30 over 168 trades (+0.2% a trade). Their payoff was inverted for a trend
  follower: best short +26%, worst -49% (an XRP short caught in the July 2023 jump); the top 5%
  of short trades (+4.5) did not pay for the rest (-7.0).
- Longs: 756 trades, 35% winners, +1.3% a trade, +9.83 summed; positive on 9 of 10 coins (LTC
  -0.16), with the convex payoff trend following needs: best +91%, worst -12%, the top 5% of long
  trades +11.3 against -1.5 for the rest.
- Entry test: 4 of the 21 losing (exit, symbol) cells ended at -100% (ETH and ADA with the
  20-day exit, DOGE with the 10- and 20-day exits). A 1x position can only lose all its equity in
  one trade if it is short and the price roughly doubles while it is held: the squeeze risk of
  the short side, the very mechanism v1 named for rallies.

v2 keeps only the side the evidence and the mechanism support.

What can still sink it. v1's card pre-registered this reading: "if only the long side makes money
and shorts lose on most coins, the profit is the development period's bull drift rather than
momentum", and v1 showed exactly that pattern. v1 beat 100% of its monkeys (its own exposure
shifted in time; median monkey Sharpe -0.13 against 0.59), so its timing added something, but
those monkeys were half short and held little of the drift. v2's monkeys are its own
long-or-flat exposure shifted in time, so they hold the same share of the drift; v2 has to beat
them to show that the timing of the long side, not the rising market, made the money. The longs
also lost in falling markets (2022 -1.22, early 2025 -0.71): whipsaws cost money even
long-only. And v2's long trades are v1's long trades (same entries, exits and sizes), so its
core-system figures are largely known in advance; the new evidence is the entry test on long
entries alone, the monkey test, the other four lookbacks, and validation.

## Rules
Position strategy on 1d bars, composed from catalogue types (no new code):
`vol_target(momentum)` with `allow_short` false. Every value uses data up to and including the
close of the day it is computed on. Rules 1, 3, 6 and 7 are v1's; rules 2, 4, 5 and 8 change
only by removing the short side.

1. Momentum: `m_t = close_t / close_{t-L} - 1` with `L = lookback` = 28 daily bars (four
   weeks). Undefined for the first L bars.
2. Direction (the v2 change): `d_t = +1` if `m_t > 0`, otherwise 0 (when `m_t < 0`, when
   `m_t = 0`, and while `m_t` is undefined). `allow_short` is false: where v1 was short, v2 is
   flat.
3. Volatility: `sigma_t` = exponentially weighted standard deviation of daily close-to-close
   returns (`indicators.ewm_vol`, span 25 days, its own warm-up of 12 bars), annualised with
   sqrt(365.25).
4. Target exposure at the close of day t: `d_t * 0.25 / sigma_t`, clipped to [0, 1]; 0 while
   `sigma_t` is undefined. The size is reset every day as `sigma_t` changes; the position opens
   when `d_t` turns to 1 and closes when it turns to 0.
5. Exit: only `d_t` turning to 0 (the four-week return falls to zero or below). The long is sold
   at that close and the position stays flat until `m_t` is positive again; there is no stop, no
   profit target and no time exit.
6. Execution, costs and funding are the engine's: the target set at the close of day t is held
   through day t+1, every change in exposure (daily re-sizing included) pays fee plus slippage
   (5 + 3 bps on Binance), and funding settled during a held day is paid by the long when
   positive and received when negative. Flat days cost nothing.
7. Grid: `strategy.lookback` in {7, 14, 28, 56, 112} days (1, 2, 4, 8 and 16 weeks), unchanged
   from v1. Every value is a positive integer, so every cell builds. `annual_vol` 0.25,
   `vol_days` 25 and `max_leverage` 1.0 are fixed: they are sizing, not the hypothesis.
8. A trade is a run of long days. At L = 28 v1 had 756 long trades (a median of 75 per symbol,
   about 15 per symbol per year); v2's trades are those same runs, so expect about 14 per symbol
   per year and roughly 75 closed trades per symbol over the development period.

## Falsified if
- The core system at L = 28 fails feasibility: median Sharpe across the ten symbols below 0.3,
  Sharpe positive on fewer than 60% of symbols, or fewer than 30 closed trades per symbol.
- The monkey test: v2's own long-or-flat exposure, shifted in time, is not beaten at least 90% of
  the time. Then the profit is the development period's upward drift, as v1's card warned, not
  momentum timing.
- The entry test (long entries when the four-week return turns positive, with 5, 10 and 20-day
  time exits and the 2/4 ATR bracket) is profitable in fewer than 70% of (exit, symbol) cells:
  prices do not keep rising after the signal turns, and the long side's profit is in holding the
  trend, not in the entry.
- Fewer than 70% of the five lookbacks have a positive median CAGR: long-side momentum would be
  a single-horizon accident rather than the broad weekly effect the literature describes.
- Read alongside (diagnostic, not a gate): if the profit comes from the 2020-21 bull run alone,
  with 2023-24 adding nothing, it is one episode rather than an effect.
