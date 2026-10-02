---
id: i008
version: 1
registered: '2026-10-02T20:40:56Z'
title: US-open session momentum
source: 'Time-of-day sessions (sources.md S33), applied to the US open on 1h bars. Gao, Han, Li & Zhou,
  ''Market Intraday Momentum'' (JFE 2018): the first half-hour''s return predicts the last half-hour''s,
  through infrequent rebalancers and late-informed traders; Heston, Korajczyk & Sadka, ''Intraday Patterns
  in the Cross-section of Stock Returns'' (JF 2010): returns persist at the same time of day, consistent
  with institutions trading at regular times; Chordia & Subrahmanyam, ''Order Imbalance and Individual
  Stock Returns: Theory and Evidence'' (JFE 2004): large orders are split over time, so order imbalance
  persists and predicts returns; Crabel (1990), opening range breakout (S14). US schedule: macro data
  releases at 08:30, NYSE open 09:30 and close 16:00 New York time.'
source_id: S33
taxonomy:
  family: seasonality
  inputs:
  - price
  - calendar
  horizon: hours
timeframe: 1h
spec:
  type: i008_us_open_momentum
  k: 1.0
  hold_bars: 6
  vol_bars: 720
optimise:
  k:
  - 0.5
  - 1.0
  - 1.5
  hold_bars:
  - 4
  - 6
  - 8
expected_trades_per_year: 100
differs_from:
  i005: i005 trades for one to four hours right after each 8-hourly funding settlement (00/08/16 UTC)
    when the latest settled funding rate is extreme, buying after high positive funding, and uses no price
    signal. This trades at most once per weekday, at 10:00 New York time, in the direction of the 08:00-10:00
    New York price move when that move exceeds one standard deviation, holds to the US equity close, and
    uses no funding.
  i007: i007 enters at any hour of the UTC day when the price breaks today's UTC open by k times the previous
    day's range, stops at that open, may reverse once and exits at 24:00 UTC. This enters only at 10:00
    New York time on weekdays, on the size of the two-hour move around the US open measured against hourly
    volatility, holds a fixed six hours to the 16:00 close with no stop, and never trades in Asian or
    European hours.
---
## Hypothesis
On 1h bars of the ten perpetuals, a large move in the two hours around the US open (08:00 to 10:00
New York time, which contain the 08:30 macro data releases and the 09:30 equity open) carries on
through the rest of the US cash session. On weekdays when that move exceeds one standard deviation
of a two-hour return, holding a position in its direction from 10:00 to 16:00 New York time makes
money after fees, slippage and funding, on both sides and on most of the ten coins.

## Why it should work
Who trades then: the US session is when US-based participants are at their desks: macro and
multi-asset funds, proprietary desks, CME futures traders and, since 2024, the market makers and
authorised participants of the US spot bitcoin and ether ETFs. Crypto moves with US equities, so
the equity open and the 08:30 data releases are the busiest moment of the US day. These traders
deal in size and work their orders over hours (VWAP and TWAP execution) to limit market impact, so
an imbalance that shows up in the opening hours keeps arriving through the session (Chordia &
Subrahmanyam 2004). Late-informed traders and infrequent rebalancers add to it later in the day:
that is the mechanism Gao, Han, Li & Zhou (2018) give for intraday momentum in the S&P 500, and
Heston, Korajczyk & Sadka (2010) find that institutional trading at regular times of day leaves
return patterns that persist. The other side is the liquidity provider and the intraday
mean-reversion trader who sells into the flow.

Why it should persist: splitting large orders is the optimal response to impact costs, so the flow
stays predictable, while trading against it means holding six hours of directional risk in a
volatile market, which arbitrage capital is reluctant to do.

Why this variant and not a pure clock rule: a fixed position during fixed hours (say, long every US
session) would need a drift above 0.16% in those hours every day just to pay its costs, and no
account of who trades then implies a drift that large or of a fixed sign. Conditioning on the
opening move trades only on days with a strong flow signal, roughly a third of weekdays, and takes
the direction from the flow itself. The window is set in New York time because the US open moves
between 13:30 UTC (summer) and 14:30 UTC (winter); a fixed UTC hour would mix two different
moments of the US day. It is the one variant registered; no other session or hour is tried.

Why the edge should beat the 0.16% round trip: a two-hour standard deviation is roughly 0.7-2% on
these coins (lowest on BTC), so a trade needs an opening move at least that large. If a fifth to a
quarter of the opening move carries on to the close, a trade earns about 0.15-0.5% gross. The margin
is thin on BTC and wider on the more volatile coins; a weaker continuation will not pay the costs.

## Rules
Position strategy (`target_position` returns -1, 0 or +1) on 1h bars; new code of type
`us_open_momentum` (stored as `i0NN_us_open_momentum`): the catalogue has no calendar input.
Every value uses only bars up to and including the bar it is computed on.

1. Clock: bar open times converted from UTC to America/New_York (daylight saving handled by the
   time-zone database; New York offsets are whole hours, so 1h bars stay aligned). A trading day is
   a New York weekday, Monday to Friday. US market holidays are not excluded (no holiday calendar
   in the data); they add noise but no lookahead.
2. Opening move on New York date d: `r_d = close(bar opening 09:00 NY) / close(bar opening 07:00 NY)
   - 1`, the move from 08:00 to 10:00 New York time. If any of the bars opening 07:00, 08:00 or
   09:00 NY is missing, no trade that day.
3. Scale: `sigma_t` = standard deviation (pandas default, ddof 1) of hourly close-to-close returns
   over the `vol_bars` = 720 bars (30 days) ending at the decision bar, undefined until 720 returns
   exist; the two-hour scale is `sigma_open = sigma_t x sqrt(2)`.
4. Decision at the close of the bar opening 09:00 NY (10:00 New York time): target +1 if
   `r_d >= k x sigma_open`, -1 if `r_d <= -k x sigma_open`, otherwise 0 for the day (also 0 while
   `sigma_t` is undefined).
5. Exit after `hold_bars` bars: the target stays at the entry sign at the closes of the bars
   opening 09:00 .. (09 + hold_bars - 1):00 NY and is 0 at the close of the bar opening
   (09 + hold_bars):00 NY. The trade earns the bars opening 10:00 .. (10 + hold_bars - 1):00 NY; at
   the default 6 it is closed at 16:00 New York time, the equity close. No stop, no target.
6. Size: full 1x, no volatility scaling; costs 5 bps fee plus 3 bps slippage per side; funding as
   the engine books it. For every grid value of `hold_bars` the 16:00 UTC settlement falls inside
   the holding window, in summer and in winter, and the position holds both bars it can be booked
   to, so its funding is charged correctly whatever its timestamp delay.
7. Parameters. `k` 1.0 (one standard deviation: a clearly directional opening, met on roughly a
   third of days), `hold_bars` 6 (to the 16:00 close, where the session's flow ends). Fixed:
   `vol_bars` 720 (30 days of hourly returns). Grid: `k` {0.5, 1.0, 1.5} x `hold_bars` {4, 6, 8}
   (exit at 14:00, 16:00 or 18:00 New York time), 9 combinations, all valid. Validation: `k > 0` and
   finite; `hold_bars` an integer in 1..12; `vol_bars` an integer >= 2.
8. Trades and holding time: at most one trade per weekday, held exactly `hold_bars` hours (6 at the
   default). The opening window is the most volatile part of the US day, so moves beyond one
   average-hour-based standard deviation are common: expect about 100 trades per symbol per year,
   roughly 500 over the development period (SOL and AVAX, listed in September 2020, about 450).

## Falsified if
- The core system at the registered parameters fails feasibility: median Sharpe across the ten
  symbols below 0.3, Sharpe positive on fewer than 60% of symbols, a median of fewer than 30
  closed trades per symbol, or its own exposure shifted to random times (monkey test) beaten less
  than 90% of the time: the US open would be no different from any other hour.
- The entry test (these entries with 5, 10 and 20-bar time exits and the 2/4 ATR bracket) is
  profitable in fewer than 70% of (exit, symbol) cells: moves at the US open do not carry on.
- Fewer than 70% of the 9 grid cells make money: the effect would hinge on one threshold or one
  exit hour.
- Read alongside (diagnostic, not a gate): if only longs make money and only in bull years, it is
  drift; if the profit comes only from 2024-25 (the ETF era), the effect may be new and the
  earlier years say nothing about it, which validation's walk-forward will show.
