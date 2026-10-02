---
id: i006
version: 1
registered: '2026-10-02T20:39:06Z'
title: Bollinger squeeze breakout
source: 'Bollinger squeeze (sources.md S15): John Bollinger, ''Bollinger on Bollinger Bands'' (2001),
  Method I (volatility breakout): BandWidth at its lowest in about six months (125 daily bars) flags a
  squeeze, and the next close outside a band gives the direction; 20-bar bands at 2 standard deviations.
  Volatility contraction before expansion: Toby Crabel, ''Day Trading with Short Term Price Patterns and
  Opening Range Breakout'' (1990), NR4/NR7 (S14). Volatility clusters and mean-reverts: Engle, ''Autoregressive
  Conditional Heteroscedasticity'' (Econometrica 1982); Bollerslev, ''Generalized Autoregressive Conditional
  Heteroskedasticity'' (J. Econometrics 1986). Stop orders cluster just beyond recent ranges and extend
  moves once triggered: Osler, ''Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictive
  Success of Technical Analysis'' (JF 2003). Wilder''s ATR(14); 2 ATR stop and 4 ATR target as in Davey''s
  entry-test bracket.'
source_id: S15
taxonomy:
  family: breakout
  inputs:
  - price
  horizon: hours
timeframe: 1h
spec:
  type: i006_squeeze_breakout
  length: 20
  mult: 2.0
  squeeze_bars: 125
  armed_bars: 20
  atr_length: 14
  stop_atr: 2.0
  target_atr: 4.0
  max_bars: 48
  cooldown_win: 0
  cooldown_loss: 0
  kill_drawdown: 1.0
  allow_short: true
optimise:
  squeeze_bars:
  - 60
  - 125
  - 250
  stop_atr:
  - 1.5
  - 2.0
  - 3.0
  target_atr:
  - 3.0
  - 4.0
  - 6.0
expected_trades_per_year: 50
differs_from:
  i001: 'i001 enters on a new 48-bar highest or lowest close confirmed by RSI(30) on 4h or daily bars,
    with cooldowns and a 50% kill switch, and has no volatility condition: it trades any breakout. This
    trades only within 20 hourly bars of Bollinger bandwidth hitting its lowest of 125 bars (a volatility
    contraction), takes a close outside the 20-bar bands as the trigger, has no RSI filter, cooldown or
    kill switch, and caps every trade at 48 hours; it bets on the expansion that follows a squeeze, not
    on trend continuation after a new high.'
---
## Hypothesis
On 1h bars of the ten perpetuals, once Bollinger bandwidth has fallen to its lowest of the last
125 hours (a squeeze), the first close outside a band within the next 20 hours gives the
direction of the volatility expansion that follows. Entering that way at the next open, with a
stop 2 ATR and a target 4 ATR away (ATR measured while volatility is still compressed) and a
48-hour time limit, hits the target more often than the one time in three a random direction
would, by enough to beat costs, on both sides and on most of the ten coins.

## Why it should work
Volatility clusters and mean-reverts (Engle 1982; Bollerslev 1986). An unusually quiet stretch
therefore tends to end in expansion. That expansion alone pays nothing: with a stop at 2 ATR and a
target at 4 ATR, a driftless price hits the target first a third of the time, which only breaks
even before costs. The edge has to come from the direction of the break.

Why the break should point the right way more often than not: a tight range collects orders just
outside it. Range traders sell its top and buy its bottom with stops just beyond; breakout traders
park buy-stops above and sell-stops below; and on leveraged perpetuals the liquidation prices of
positions opened inside a narrow range sit close to its edges. A close outside the band means the
first of these orders have fired, and each stop or liquidation that fires is a market order in the
direction of the break that pushes the price toward the next one. Osler (2003) shows that stop
orders cluster just beyond such levels and make trends run on once triggered. The other side of
the trade is the range trader and the leveraged position stopped or liquidated out of the range,
who pays for immediacy. This is Bollinger's own use of the squeeze (Method I) and the pattern
behind Crabel's narrow-range breakouts.

Why the edge should beat the 0.16% round trip: because ATR is compressed at a squeeze, the 2 ATR
stop is tight in price (roughly 0.5-1.5% on these coins on 1h bars), and the 4 ATR target is a
1-3% move that an expansion reaches in hours. Costs are 0.16% a round trip (0.13% when the target
fills, as limit fills pay no slippage), about 10-25% of the risk taken, so the rule breaks even at a
hit rate of roughly 37-42% (`(1 + cost / risk) / 3`) instead of 33%. The claim is that the squeeze-and-break direction lifts
the hit rate above that; Bollinger's head-fake warning (a first break that reverses) is the risk.

How it differs from rules already screened: the catalogue's `bollinger_breakout` (screened on 4h
and 1d) enters on any close outside the bands, with no squeeze condition, and exits on a close
back through the middle band. This one trades only after a contraction, on 1h bars, with a fixed
bracket and time limit.

## Rules
Bracket strategy (subclass of `BracketStrategy`, run on the bar-by-bar simulator) on 1h bars;
new code of type `squeeze_breakout` (stored as `i0NN_squeeze_breakout`): the catalogue cannot
express it (`breakout_bracket` has no squeeze condition and its entry is an N-bar closing high
with an RSI filter). Every signal uses only bars up to and including the signal bar.

1. Bands: `bands = indicators.bollinger(close, length=20, mult=2.0)` (SMA20 +- 2 population
   standard deviations). Bandwidth `BW_t = (upper_t - lower_t) / mid_t`; undefined for the first 19
   bars.
2. Squeeze bar: `BW_t` equals the lowest bandwidth of the last `squeeze_bars` bars including t,
   i.e. `BW_t <= BW.rolling(squeeze_bars, min_periods=squeeze_bars).min()_t`. Not a squeeze while
   that minimum is undefined (warm-up of `length - 1 + squeeze_bars - 1` bars).
3. Armed: at least one squeeze bar among the last `armed_bars` bars including t (bars
   `t - armed_bars + 1 .. t`), with `armed_bars` = 20, one band length: after that the bands have
   re-estimated on post-squeeze data and the setup is stale.
4. Signals at the close of bar t: long when armed and `close_t > upper_t`; short when armed and
   `close_t < lower_t` (both are never true together).
5. Trade management, as the bracket simulator does it: entry at the open of bar t + 1; stop
   `stop_atr x ATR_t` and target `target_atr x ATR_t` from the fill, where
   `ATR_t = indicators.atr(high, low, close, atr_length=14)` read on the signal bar; levels never
   move. A bar opening beyond a level exits at the open; if one bar touches both, the stop is
   assumed first; target fills pay no slippage. Time exit: a trade still open at the close of its
   `max_bars`-th bar (48, the entry bar counting as one) exits at that close with slippage.
   Signals while in a trade are ignored (no reversal, no pyramiding); `cooldown_win` and
   `cooldown_loss` are 0 and `kill_drawdown` 1.0 (no kill switch), so the next trade can start at
   the open after any exit when a fresh signal bar appears while still armed. No signal while ATR
   is undefined.
6. Size: full 1x of equity at entry (simulator `leverage` 1), no volatility scaling; costs 5 bps
   fee plus 3 bps slippage per side; funding charged on the position held at each settlement.
7. Parameters. Fixed: `length` 20 and `mult` 2.0 (Bollinger's defaults), `armed_bars` 20,
   `atr_length` 14 (Wilder), `max_bars` 48 (two days: an expansion that has not come in two days
   means the setup failed), `allow_short` true. Optimised: `squeeze_bars` 125 (Bollinger's
   six-month low of daily bars, applied here as a bar count), `stop_atr` 2.0 and `target_atr` 4.0
   (the classic 1:2 bracket, the same distances as Davey's entry-test bracket). Grid:
   `squeeze_bars` {60, 125, 250} x `stop_atr` {1.5, 2.0, 3.0} x `target_atr` {3.0, 4.0, 6.0}, 27
   combinations, all valid. Validation: integers `length >= 2`, `squeeze_bars >= 2`,
   `armed_bars >= 1`, `atr_length >= 1`, `max_bars >= 1`; `mult`, `stop_atr`, `target_atr` > 0;
   cooldowns >= 0; `0 < kill_drawdown <= 1`.
8. Trades and holding time. On 1h bars bandwidth sets a new 125-bar low in runs of a few bars,
   roughly one squeeze episode every few days, and most episodes see a close outside a band within
   20 bars. Expect about 50 trades per symbol per year at the defaults (fewer with
   `squeeze_bars` 250, more with 60), over 200 per symbol over the development period. A trade
   lasts from an hour (a stop hit early) to the 48-hour limit; a typical one should resolve in
   roughly 5-20 hours.

## Falsified if
- The core system at the registered parameters fails feasibility: median Sharpe across the ten
  symbols below 0.3, Sharpe positive on fewer than 60% of symbols, or a median of fewer than 30
  closed trades per symbol.
- Random entries with the same frequency, side mix, bracket and time limit (the monkey test for
  bracket strategies) are beaten less than 90% of the time: the squeeze-and-break timing adds
  nothing to the bracket.
- The entry test (these entries with 5, 10 and 20-bar time exits and the 2/4 ATR bracket) is
  profitable in fewer than 70% of (exit, symbol) cells: breaks out of a squeeze do not run on.
- Fewer than 70% of the 27 grid cells make money: the effect would depend on one squeeze length
  or one bracket.
- Read alongside (diagnostic, not a gate): if only longs make money, and mostly in bull years, it
  is drift; if the stop is hit on most trades within the first hours (head fakes), the break
  direction carries no information.
