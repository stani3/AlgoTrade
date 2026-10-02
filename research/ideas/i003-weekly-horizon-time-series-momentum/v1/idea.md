---
id: i003
version: 1
registered: '2026-10-02T16:02:11Z'
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
    allow_short: true
optimise:
  strategy.lookback:
  - 7
  - 14
  - 28
  - 56
  - 112
expected_trades_per_year: 25
differs_from:
  i001: i001 enters on a new lookback-bar closing high or low confirmed by RSI and exits on a 2x ATR stop
    or 4x ATR profit target, so its trades last days and its winners are capped; this holds the sign of
    the four-week return, always long or short and volatility-sized, with no stop or target until the
    sign flips, betting on multi-week continuation whose payoff sits in the uncapped large trends.
---
## Hypothesis
On daily bars of liquid crypto perpetuals, the sign of the past four-week return predicts the
direction of returns over the following weeks (time-series momentum). Holding that side (long
after a rise, short after a fall), sized to a constant volatility and kept until the sign flips,
makes money after fees, slippage and funding on most of the ten coins, and does so across
lookbacks from one to sixteen weeks rather than at one lucky horizon.

## Why it should work
Crypto has no cash flows to anchor a value, a fragmented, retail-heavy investor base and
attention-driven flows, so news and flows are priced over weeks rather than at once. Liu &
Tsyvinski (2021) find significant time-series momentum in crypto at daily and weekly frequency:
coins in the top quintile of past-week returns beat the bottom quintile over the next one to four
weeks, and investor-attention proxies forecast returns. This is the underreaction-then-herding
pattern behind time-series momentum in other futures markets (Moskowitz, Ooi & Pedersen 2012
find it for lookbacks of one to twelve months in 58 futures and forwards).

Leverage adds positive feedback. Schmeling, Schrimpf & Todorov trace the large, volatile crypto
carry to demand for leveraged exposure from smaller, trend-chasing investors, with little
arbitrage capital to lean against it. Trend chasers buy after rises and sell after falls, and
the wrong-side leveraged traders are liquidated with market orders in the direction of the move
(short squeezes in rallies, long liquidation cascades in declines), which carries prices further
the way they were already going.

The other side of the trade: early profit-takers and contrarians (long-term holders and miners
selling into rallies, dip buyers in declines), short-horizon mean-reversion liquidity providers,
and the leveraged traders forced out in the trend's direction.

The rule takes Moskowitz, Ooi & Pedersen's form: the sign of the past return, with the position
scaled by recent volatility so every coin and every regime carries similar risk (crypto
volatility varies several-fold between coins and over time). There is no profit target and no
stop: a position ends only when the signal flips, so the few large trends that pay for trend
following are kept whole instead of being cut at a fixed multiple of ATR. Shorts usually receive
funding and longs pay it; the engine charges it, so crowded rallies cost the long side carry.

The catalogue `momentum` rule was screened at its default 20 bars (not gated). This card
pre-registers it at the crypto literature's horizons with volatility targeting and puts it
through the gates.

## Rules
Position strategy on 1d bars, composed from catalogue types (no new code):
`vol_target(momentum)`. Every value uses data up to and including the close of the day it is
computed on.

1. Momentum: `m_t = close_t / close_{t-L} - 1` with `L = lookback` = 28 daily bars (four
   weeks). Undefined for the first L bars.
2. Direction: `d_t = +1` if `m_t > 0`, `-1` if `m_t < 0`, otherwise 0 (also 0 while `m_t` is
   undefined). `allow_short` is true: the rule is symmetric.
3. Volatility: `sigma_t` = exponentially weighted standard deviation of daily close-to-close
   returns (`indicators.ewm_vol`, span 25 days, its own warm-up of 12 bars), annualised with
   sqrt(365.25).
4. Target exposure at the close of day t: `d_t * 0.25 / sigma_t`, clipped to [-1, 1]; 0 while
   `sigma_t` is undefined. The size is reset every day as `sigma_t` changes; the side changes
   only when `d_t` does.
5. Exit: only a change of `d_t`. A flip from long to short (or back) reverses at that close;
   there is no stop, no profit target and no time exit.
6. Execution, costs and funding are the engine's: the target set at the close of day t is held
   through day t+1, every change in exposure (daily re-sizing included) pays fee plus slippage
   (5 + 3 bps on Binance), and funding settled during a held day is paid by longs and received by
   shorts when positive.
7. Grid: `strategy.lookback` in {7, 14, 28, 56, 112} days, i.e. 1, 2, 4, 8 and 16 weeks,
   doubling either side of the four-week default (the top of Liu & Tsyvinski's one-to-four-week
   range and the shortest Moskowitz, Ooi & Pedersen lookback, in crypto's 7-day weeks). Every
   value is a positive integer, so every cell builds. `annual_vol` 0.25, `vol_days` 25 and
   `max_leverage` 1.0 are fixed: they are sizing, not the hypothesis.
8. A trade is a run of days on the same side. At L = 28 expect about 25 per symbol per year (a
   28-day sum of independent daily returns changes sign about 31 times a year; persistence in
   trends lowers that), so roughly 100+ closed trades per symbol over the development period.

## Falsified if
- The core system at L = 28 fails feasibility: median Sharpe across the ten symbols below 0.3,
  Sharpe positive on fewer than 60% of symbols, or fewer than 30 closed trades per symbol.
- The monkey test: the same exposure series shifted in time is not beaten at least 90% of the
  time. Then the timing adds nothing beyond the period's drift and the volatility sizing.
- The entry test (sign-flip entries with 5, 10 and 20-day time exits and the 2/4 ATR bracket) is
  profitable in fewer than 70% of (exit, symbol) cells: prices do not keep moving the way the
  signal has just turned.
- Fewer than 70% of the five lookbacks have a positive median CAGR: momentum would be a
  single-horizon accident rather than the broad weekly effect the literature describes.
- Read alongside (diagnostic, not a gate): if only the long side makes money and shorts lose on
  most coins, the profit is the development period's bull drift rather than momentum.
