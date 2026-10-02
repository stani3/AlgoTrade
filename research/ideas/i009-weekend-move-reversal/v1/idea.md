---
id: i009
version: 1
registered: '2026-10-02T20:41:57Z'
title: Weekend move reversal
source: 'Day-of-week effect (sources.md S32): thin weekend liquidity and the Monday reopening of traditional
  markets. Grossman & Miller, ''Liquidity and Market Structure'' (JF 1988): the price concession paid
  for immediacy grows when fewer liquidity providers are present, and is recovered as they return; Nagel,
  ''Evaporating Liquidity'' (RFS 2012): short-term reversal is the return to liquidity provision and is
  larger when liquidity is scarce; French, ''Stock Returns and the Weekend Effect'' (JFE 1980): returns
  across the market closure differ from weekday returns. Schedules: CME bitcoin and ether futures closed
  from Friday 16:00 to Sunday 17:00 Chicago time; US equities and bank transfers closed at weekends.'
source_id: S32
taxonomy:
  family: seasonality
  inputs:
  - price
  - calendar
  horizon: days
timeframe: 1h
spec:
  type: i009_weekend_reversal
  min_move: 0.5
  hold_bars: 24
  vol_bars: 720
optimise:
  min_move:
  - 0.25
  - 0.5
  - 1.0
  hold_bars:
  - 12
  - 24
  - 48
expected_trades_per_year: 24
differs_from:
  i002: i002 buys after any single 4h bar falls three or more volatilities, at any time of the week, and
    holds 3-12 bars (long only in v2), betting on a liquidation overshoot. This fades the whole 48-hour
    weekend move in both directions, once a week at a fixed time (Sunday 22:00 UTC), with a much lower
    bar (half a 48-hour standard deviation), and bets on liquidity returning with the traditional markets
    on Monday.
  i004: i004 shorts crowded funding on daily bars for up to a week and never goes long. This uses no funding,
    trades both directions, enters only on Sunday 22:00 UTC against the weekend's price move and holds
    for a day.
  i008: i008 follows the 08:00-10:00 New York move on weekdays for six hours (momentum inside the US session).
    This fades the 48-hour weekend move, enters on Sunday evening UTC before the traditional markets reopen
    and holds through Monday.
---
## Hypothesis
On the ten perpetuals, a large price move over the weekend (Friday 22:00 to Sunday 22:00 UTC,
while CME crypto futures, US equities and the banking system are closed) partly reverses once
those markets reopen. Taking the opposite side of the weekend move at Sunday 22:00 UTC, when it
exceeds half a 48-hour standard deviation, and holding for 24 hours (through Monday's Asian,
European and US sessions) makes money after fees, slippage and funding, on both sides and on most
of the ten coins. The direction is fixed in advance: reversal, not continuation.

## Why it should work
At weekends crypto keeps trading but much of its liquidity does not. CME futures (the venue of the
basis trade), US equity and ETF desks and the traditional market makers are closed, and bank
transfers stop, so moving capital between exchanges and between spot and futures is slower and
dearer. Order flow meets fewer liquidity providers, and each provider demands a larger price
concession for taking the other side (Grossman & Miller 1988). Part of a weekend move is therefore
that concession rather than news. When the traditional markets come back (CME on Sunday evening,
Asia on Monday morning, then Europe and the US) the missing liquidity providers return, and the
concession is competed away: the price drifts back. Nagel (2012) shows that reversal returns are
the reward for providing liquidity and are larger when liquidity is scarce. The other side of the
trade is the weekend trader who demanded immediacy in a thin market; the rule is paid for standing
in as a liquidity provider until the weekday ones return. Traders call the pattern "CME gaps get
filled": that is consistent with it, not evidence for it.

Why reversal rather than a weekend drift: a fixed weekend position would need a weekend drift above
0.16% every week with a sign no theory fixes. The reversal has a reason, and trading only weekends
with a sizeable move keeps the trade count low and the expected move per trade large.

Why the edge should beat the 0.16% round trip: the rule trades only when the weekend move is at
least half a 48-hour standard deviation (measured from all hours of the last 30 days), roughly 2-3%
on BTC and more on the other coins. If a fifth of such a move comes back by Monday evening, a trade
earns about 0.4-0.7% gross, against 0.16% of costs plus about 0.03% of funding (three settlements
at the neutral rate).

What can sink it: weekend moves are not all thin-market noise. Exchange hacks, regulatory news and
macro shocks also land at weekends, and an informational move continues instead of reversing. A
1x position held through a continuing move has no stop. The falsification tests below separate
the two.

## Rules
Position strategy (`target_position` returns -1, 0 or +1) on 1h bars; new code of type
`weekend_reversal` (stored as `i0NN_weekend_reversal`): the catalogue has no calendar input.
Every value uses only bars up to and including the bar it is computed on. All times are UTC.

1. Weekend window: from Friday 22:00 to Sunday 22:00 UTC. In both daylight-saving regimes this lies
   inside the closure of the traditional markets: US equities close at 20:00 or 21:00 UTC on
   Friday, CME crypto futures close at 21:00 or 22:00 UTC on Friday and reopen at 22:00 or 23:00
   UTC on Sunday.
2. Weekend move: `W = close(bar opening Sunday 21:00) / close(bar opening Friday 21:00) - 1`, the
   price change from Friday 22:00 to Sunday 22:00. If either bar is missing, no trade that week.
3. Scale: `sigma_t` = standard deviation (pandas default, ddof 1) of hourly close-to-close returns
   over the `vol_bars` = 720 bars (30 days) ending at the decision bar, undefined until 720 returns
   exist; the 48-hour scale is `sigma_48 = sigma_t x sqrt(48)`.
4. Decision at the close of the bar opening Sunday 21:00 (22:00 UTC): target -1 if
   `W >= min_move x sigma_48`, +1 if `W <= -min_move x sigma_48`, otherwise 0 for the week (also 0
   while `sigma_t` is undefined).
5. Exit after `hold_bars` bars: the target stays at the entry sign at the closes of the bars
   opening Sunday 21:00 .. Sunday 21:00 + (hold_bars - 1) hours and is 0 at the close of the bar
   opening Sunday 21:00 + hold_bars hours. The trade earns the bars opening Sunday 22:00 onwards
   for `hold_bars` hours; at the default 24 it is closed at Monday 22:00 UTC, after Monday's US
   equity close in both daylight-saving regimes. No stop, no target.
6. Size: full 1x, no volatility scaling; costs 5 bps fee plus 3 bps slippage per side; funding at
   the settlements inside the holding window (00:00, 08:00 and 16:00 on Monday at the default) is
   paid or received as the engine books it. For every grid value of `hold_bars` each of those
   settlements falls between two bars that are both held, so their timestamp delay does not
   matter.
7. Parameters. `min_move` 0.5 (half a 48-hour standard deviation: large enough that a partial
   reversal pays the costs, small enough to trade about half the weekends), `hold_bars` 24 (one
   day: Monday's three sessions, by which the weekday liquidity providers are all back). Fixed:
   `vol_bars` 720 (30 days of hourly returns). Grid: `min_move` {0.25, 0.5, 1.0} x `hold_bars`
   {12, 24, 48}, 9 combinations, all valid. Validation: `min_move > 0` and finite; `hold_bars` an
   integer in 1..120 (a trade always ends before the next Sunday's decision); `vol_bars` an
   integer >= 2.
8. Trades and holding time: at most one trade a week, held exactly `hold_bars` hours (24 at the
   default). With the scale taken from all hours (weekdays are more volatile than weekends), about
   half the weekends qualify: expect about 24 trades per symbol per year, roughly 130 over the
   development period (about 110 for SOL and AVAX, listed in September 2020), and still about 50
   at `min_move` 1.0.

## Falsified if
- The core system at the registered parameters fails feasibility: median Sharpe across the ten
  symbols below 0.3, Sharpe positive on fewer than 60% of symbols, a median of fewer than 30
  closed trades per symbol, or its own exposure shifted in time (monkey test) beaten less than 90%
  of the time: fading a 48-hour move at a random time would then work as well as fading the
  weekend's.
- The entry test (these entries with 5, 10 and 20-bar time exits and the 2/4 ATR bracket) is
  profitable in fewer than 70% of (exit, symbol) cells: weekend moves do not reverse on Monday,
  and the idea is wrong. If such entries lose consistently, weekend moves continue (they carry
  information), which is the opposite of this card's claim and not a licence to flip it.
- Fewer than 70% of the 9 grid cells make money: the effect would hinge on one threshold or one
  holding time.
- Read alongside (diagnostic, not a gate): if only the longs (fading weekend falls) make money and
  only in bull years, it is drift; if the profit comes from two or three extreme weekends, it is
  luck rather than a weekly effect.
