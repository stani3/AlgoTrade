---
id: i007
version: 1
registered: '2026-10-02T20:39:58Z'
title: Dual Thrust intraday breakout
source: 'Dual Thrust (sources.md S16): Michael Chalek''s intraday range-breakout system. Range = max(HH
  - LC, HC - LL) over the last N days (HH highest high, LL lowest low, HC highest close, LC lowest close);
  buy line = today''s open + K1 x Range, sell line = today''s open - K2 x Range; stop-and-reverse between
  the lines; flat at the end of the session; usual settings N = 1, K1 = K2 = 0.5. Close relative: Larry
  Williams'' volatility breakout (today''s open plus a fraction of the previous day''s range), ''Long-Term
  Secrets to Short-Term Trading'' (1999); Crabel (1990), opening range breakout. Intraday momentum: Gao,
  Han, Li & Zhou, ''Market Intraday Momentum'' (JFE 2018). Stop orders clustered beyond levels extend
  moves once triggered: Osler (JF 2003).'
source_id: S16
taxonomy:
  family: breakout
  inputs:
  - price
  - calendar
  horizon: hours
timeframe: 1h
spec:
  type: i007_dual_thrust
  k: 0.5
  range_days: 1
optimise:
  k:
  - 0.3
  - 0.5
  - 0.7
  range_days:
  - 1
  - 2
  - 4
expected_trades_per_year: 200
differs_from:
  i006: 'i006 trades only within 20 hours after Bollinger bandwidth hit its lowest of 125 hourly bars,
    on a close outside the 20-bar bands at any hour, with a fixed 2/4 ATR bracket filled at the next open
    and a 48-hour limit, and holds across days. This needs no volatility contraction: it measures the
    break from the UTC day''s own open against k times the previous day''s range, stops on a close back
    through that open, reverses once on the opposite line, allows one long and one short a day, and is
    always flat at 24:00 UTC.'
  i001: i001 enters on a 48-bar closing high or low with an RSI filter on 4h or daily bars and holds for
    days until a 2/4 ATR stop or target. This is intraday on 1h bars, anchored to the UTC day's open and
    the previous day's range, exits at the open or at the end of the day, and never holds a position overnight
    in UTC terms.
---
## Hypothesis
On 1h bars of the ten perpetuals, once the price closes an hour more than half the previous UTC
day's range away from today's UTC open, the move carries on to the end of the UTC day more often
and further than it reverses. Going with it on either side, with a stop if the price closes back
through today's open (or a reversal if it reaches the opposite line) and a forced exit at the end
of the UTC day, makes money after fees, slippage and funding on most of the ten coins.

## Why it should work
A move of half a normal day's range from the open, within the day, is unusual. It usually means
news or a large order being worked, and in this market it also means levels that many traders
watch have broken: the UTC daily open is the reference of the daily candle almost every crypto
chart and bot uses, and stops and liquidation prices sit beyond the previous day's extremes. Once
the move starts, stop-losses and liquidations fire as market orders in its direction, and
leveraged trend-chasers pile in, which carries it further (Osler 2003 on stop-order cascades).
Gao, Han, Li & Zhou (2018) find the same intraday momentum in the S&P 500, with late-informed
traders and infrequent rebalancers trading in the direction already taken. Scaling the threshold
by the previous day's range adapts it to the volatility regime, which is why Dual Thrust and
Williams' volatility breakout use it. The other side of the trade is the intraday mean-reversion
trader or market maker who sells into the move and the stopped-out position on the wrong side.

Why the edge should beat the 0.16% round trip: a trade risks about k x Range from its entry line to
the open, roughly 1-3% of price at k = 0.5 on these coins, and winners run to the end of the day,
so trade outcomes are percent-sized and the round trip is 5-15% of the risk taken. The rule needs
trend days that run to the close to outweigh stopped trades by that margin. With up to two trades a
day it turns over a lot, about 30% of equity a year in costs at 1x, so a weak or noisy continuation
will not pay for itself: that is the main risk, together with whipsaw days that hit both lines.

## Rules
Position strategy (`target_position` returns -1, 0 or +1) on 1h bars; new code of type
`dual_thrust` (stored as `i0NN_dual_thrust`): the catalogue has no session-anchored rule (its
channel breakouts use rolling windows and hold across days). Every value uses only bars up to and
including the bar it is computed on; days are UTC calendar days of the bar open times.

1. Day d = the bars opening 00:00 .. 23:00 UTC on date d. Today's open `O_d` = the open of the bar
   opening at 00:00 UTC on d. If that bar is missing, no trading on d.
2. Range from the previous `range_days` (N) complete UTC days d-N .. d-1 (all 24 hourly bars present
   in each, otherwise no trading on d): HH = highest high and LL = lowest low of their bars; each
   day's close is the close of its 23:00 bar; HC = highest and LC = lowest of those N daily closes.
   `R_d = max(HH - LC, HC - LL)`; with N = 1 that is `max(H - C, C - L)` of yesterday. No trading
   on d if `R_d <= 0`.
3. Lines: `Buy_d = O_d + k x R_d`, `Sell_d = O_d - k x R_d`.
4. State machine on the close of each bar t of day d (`long_used` and `short_used` reset to false
   at the start of each day, and the day always starts flat):
   - Bar opening 23:00 UTC: target 0 (flat by the end of the day); no entries.
   - Bars opening 00:00 .. 22:00 UTC:
     - Flat: if `close_t > Buy_d` and not `long_used`, go long (+1) and set `long_used`; else if
       `close_t < Sell_d` and not `short_used`, go short (-1) and set `short_used`; else stay flat.
     - Long: if `close_t < Sell_d` and not `short_used`, reverse to short (-1) and set
       `short_used` (Chalek's stop-and-reverse); else if `close_t < O_d`, go flat (stop: the move
       from the open has fully failed); else stay long.
     - Short: the mirror image (`close_t > Buy_d` and not `long_used` reverses to long; else
       `close_t > O_d` goes flat; else stay short).
   So there is at most one long and one short entry per day, a stopped trade is not re-entered on
   the same side that day, and `Buy_d > O_d > Sell_d` always.
5. Execution: the target set at a bar's close is held through the next bar (the engine's
   convention), so entries, stops and reversals fill at the close of the hourly bar that crossed
   the line, not intrabar at the line itself as in the original stop-order version. Size: full 1x,
   no volatility scaling. Costs 5 bps fee plus 3 bps slippage per change in exposure (a reversal
   pays twice). Funding at the 08:00 and 16:00 settlements is paid or received by the position
   held. The 00:00 settlement falls at the forced exit; the engine charges it to the position held
   in the 23:00 bar only when its timestamp is not delayed (about 57% of settlements, see the timing
   note in i005's card), a negligible difference.
6. Parameters. `k` 0.5 (the usual Dual Thrust setting, K1 = K2), `range_days` 1 (the previous UTC
   day, as the user specified and the usual Dual Thrust setting; it cannot go lower). Grid: `k`
   {0.3, 0.5, 0.7} x `range_days` {1, 2, 4}, 9 combinations, all valid. Validation: `k > 0` and
   finite; `range_days` an integer >= 1.
7. Warm-up: the first `range_days` complete UTC days of data.
8. Trades and holding time. At k = 0.5 an hourly close beyond one of the lines happens on a
   majority of days, sometimes on both sides: expect about 200 trades per symbol per year (at most
   two a day), about 1,000 over the development period. A trade lasts from one hour (stopped at
   the next close) to at most 23 hours (an entry at the 00:00 bar's close held to the end of the
   day); typically a few hours to about 15.

## Falsified if
- The core system at the registered parameters fails feasibility: median Sharpe across the ten
  symbols below 0.3, Sharpe positive on fewer than 60% of symbols, a median of fewer than 30
  closed trades per symbol, or its own exposure shifted in time (monkey test) beaten less than 90%
  of the time.
- The entry test (these entries with 5, 10 and 20-bar time exits and the 2/4 ATR bracket) is
  profitable in fewer than 70% of (exit, symbol) cells: moves of half a day's range from the open
  do not carry on.
- Fewer than 70% of the 9 grid cells make money: the effect would depend on one threshold or one
  range length.
- Read alongside (diagnostic, not a gate): if only the long side makes money and only in bull
  years, it is drift; if most trades end on the open-stop or reverse on the same day (whipsaw),
  intraday breaks in these coins mean-revert rather than trend.
