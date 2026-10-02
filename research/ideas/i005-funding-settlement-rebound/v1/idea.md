---
id: i005
version: 1
registered: '2026-10-02T20:38:19Z'
title: Funding-settlement rebound
source: 'Funding-settlement timing (sources.md S34). Binance USDT-M perpetuals settle funding every 8
  hours (00:00, 08:00, 16:00 UTC) between the positions open at that instant; the rate is the premium
  index plus clamp(interest - premium, -0.05%, +0.05%) with an interest component of 0.01% per 8h (Binance
  funding-rate documentation); base-tier fees 0.02% maker / 0.05% taker. He, Manela, Ross & von Wachter,
  ''Fundamentals of Perpetual Futures'' (2022): funding is the price of leverage that ties the perpetual
  to spot. Temporary price pressure from predictable, non-informational trades that reverses afterwards:
  Harris & Gurel, ''Price and Volume Effects Associated with Changes in the S&P 500 List'' (JF 1986);
  Coval & Stafford, ''Asset Fire Sales (and Purchases) in Equity Markets'' (JFE 2007).'
source_id: S34
taxonomy:
  family: seasonality
  inputs:
  - funding
  - calendar
  horizon: hours
timeframe: 1h
spec:
  type: i005_settlement_rebound
  threshold: 0.0005
  hold_bars: 2
optimise:
  threshold:
  - 0.0003
  - 0.0005
  - 0.001
  hold_bars:
  - 1
  - 2
  - 4
expected_trades_per_year: 60
differs_from:
  i004: i004 shorts on daily bars at the onset of a week of crowded funding (a percentile rank in the
    coin's trailing year plus a floor at the neutral rate) and holds up to a week, betting that a leveraged
    crowd unwinds over days. This trades both directions on 1h bars, only in the one to four hours right
    after each 8-hourly settlement, keyed on the single latest settled rate against a fixed level, and
    bets on a temporary dip-and-recovery caused by traders dodging (or collecting) that one payment. After
    high positive funding it goes long, the opposite of i004's short, and it is flat more than 90% of
    the time.
---
## Hypothesis
On 1h bars of the ten perpetuals, when the most recently settled funding rate is far from zero
(at least 0.05% per 8 hours in absolute value, five times Binance's neutral rate), the price is
pushed against the paying side into each 8-hourly settlement and recovers in the hours right
after it. Buying at the settlement that follows a high positive rate (and selling short at the
one that follows a strongly negative rate), and holding for two hours, makes money after fees,
slippage and funding on most of the ten coins.

## Why it should work
Funding is paid only by positions open at the settlement instant. When longs pay a lot, a long
can skip the payment by closing just before the snapshot and reopening just after it; this pays
when the payment is larger than the round trip, which for a base-tier trader with maker orders
(0.02% a side, 0.04% round trip) holds from about 0.05% per 8h. Funding collectors do the mirror
trade: they open a short perpetual (often against spot) before the snapshot to receive the
payment and close it afterwards. Both flows sell the perpetual before the settlement and buy it
back after it; both are predictable in time and carry no information about value. Predictable,
non-informational demand moves prices temporarily and the move reverses once the flow is done
(index-inclusion price pressure, Harris & Gurel 1986; fire-sale pressure, Coval & Stafford 2007).
With negative funding everything flips: shorts pay, they cover before the snapshot and re-short
after it, so the price is lifted into the settlement and eases afterwards.

The rule takes the rebound leg: it buys at the settlement after the pre-settlement selling and
sells into the avoiders' and collectors' buy-back. The other side of the trade is the trader who
pays a round trip of spread and impact to save one funding payment; that round trip is what the
rule earns. The pre-settlement leg (short into the settlement) is not traded: Davey's entry test
holds every entry for 5, 10 and 20 bars, so a short entered two hours before the settlement would
be judged mostly on the rebound that follows it, and its funding would be booked on the wrong side
of the settlement at random (see the timing facts below). The rebound leg is judged fairly by the
same test.

Why the edge should beat the 0.16% round trip: the rule trades only when the payment at stake is
at least 0.05% of notional per settlement, which happens in the hottest, most leveraged stretches
(2021, early and late 2024 on these coins), when open interest is large, hourly ranges are 1-3%
and a large share of positions has a reason to step aside. A pressure-and-release worth a tenth of
an hourly range clears the cost. That is the claim under test; with costs fixed at 0.16% plus
about 0.02% of funding booked by the engine (rule 7), a gross rebound of roughly 0.2% a trade is
needed, and a smaller effect makes this idea a cost donation.

What can still sink it: the realised rate of the previous settlement is only a proxy for the rate
traders see coming (the live predicted rate is not in our data). Funding is persistent, so a high
last rate usually means a high next rate, but the proxy will be wrong around turning points. In
hot markets the rule trades three times a day, so its costs pile up exactly when it is most
active. Long trades after positive funding fall in bull phases and collect some upward drift; the
monkey test (the same exposure shifted to random hours) separates that from settlement timing.

## Rules
Position strategy (`target_position` returns -1, 0 or +1) on 1h bars; new code of type
`settlement_rebound` (stored as `i0NN_settlement_rebound`): the catalogue has no calendar or
settlement-timed rule. Every value uses only bars up to and including the bar it is computed on.

Data timing facts (structure only, inspected on development data before writing this card; no
returns were looked at):

- All ten symbols settle at 00:00, 08:00 and 16:00 UTC throughout the development period. SOL also
  had one short spell of 2-hourly settlements (98 two-hour intervals, settlements at even hours)
  and three 4-hour intervals; the rule ignores those extra settlements as events.
- Recorded settlement times are 0-47 ms after the hour. `align_funding` charges a settlement to the
  bar whose interval `(open, open + 1h]` contains `timestamp - 1 ms`. A settlement stamped 0 or 1 ms
  after the hour therefore lands on the bar that closes at that hour, and one stamped 2-47 ms after
  lands on the bar that opens then. For BTC's 08:00 settlements that is 1,213 on the 07:00 bar and
  847 on the 08:00 bar, and the other symbols are similar (about 57% / 43%). Which of the two bars
  holds a settlement is not known in advance.
- Many settlements are exactly zero (BNB 2,676 of 5,720; LINK 344; LTC 312), so a nonzero
  `funding_rate` cannot be used to detect settlements. The rule finds settlement hours from bar
  timestamps only.

1. Events. A settlement hour H is any 00:00, 08:00 or 16:00 UTC. Its decision bar is the 1h bar
   opening at H - 1h (open hour 23, 07 or 15 UTC); the decision is taken at that bar's close,
   which is time H.
2. Funding proxy, looked up by timestamp (not by position): `f_H = funding_rate(bar opening
   H - 9h) + funding_rate(bar opening H - 8h)`. These two bars are the only possible owners of the
   previous scheduled settlement at H - 8h, wherever its timestamp delay put it, and no other
   scheduled settlement can fall in them (the extra SOL settlements at even hours fall on other
   bars). Both bars have closed by H - 7h, seven hours before the decision, so `f_H` is the
   realised rate of the previous settlement and never the one at H. If either bar is missing,
   `f_H` is undefined and there is no trade. A settlement of exactly zero gives `f_H = 0`: no
   trade, correctly.
3. Entry at the close of the decision bar: target +1 if `f_H >= threshold`, -1 if
   `f_H <= -threshold`, otherwise 0.
4. Exit: the position is held for `hold_bars` bars. The target stays at the entry sign at the
   closes of the bars opening H - 1h, H, ..., H + (hold_bars - 2)h and is 0 at the close of the
   bar opening H + (hold_bars - 1)h. The trade earns the bars opening H .. H + (hold_bars - 1)h and
   is closed at H + hold_bars hours. No stop, no target.
5. With `hold_bars <= 7` a trade always ends before the next decision bar closes, so trades never
   overlap. Consecutive events are separate trades with at least `8 - hold_bars` flat hours
   between them (4 at the default).
6. Size: full 1x, no volatility scaling. Execution, costs and funding are the engine's: the target
   set at a bar's close is held through the next bar, every change in exposure pays 5 bps fee plus
   3 bps slippage (0.16% a round trip), and funding settled while in a position is paid by longs
   when positive.
7. Funding booked by the engine: the position opened at H holds the bar opening at H, so the
   engine charges it the H settlement whenever that settlement is stamped 2 ms or more after the
   hour (about 43% of settlements). A long after positive funding then pays the H rate. In live
   trading the order is sent after the bar has closed, so it fills after the snapshot and pays
   nothing. The backtest is therefore slightly pessimistic, by about 0.43 x the rate per trade
   (about 0.02% at the default threshold). Exits come at least an hour before the next
   settlement.
8. Parameters. `threshold` 0.0005 (0.05% per 8h): five times Binance's neutral rate (the 0.01%
   interest component), about 55% a year annualised, and the level above which a base-tier trader
   closing and reopening with maker orders (0.04% round trip) saves more than they pay.
   `hold_bars` 2: the avoiders reopen right after the snapshot and the collectors unwind soon
   after; two hours covers the reopening hour plus a margin for slower traders. Grid: `threshold`
   {0.0003, 0.0005, 0.001} x `hold_bars` {1, 2, 4}, 9 combinations, all valid. Validation:
   `threshold > 0`; `hold_bars` an integer in 1..7.
9. Trades and holding time. Trades last exactly `hold_bars` hours (2 at the default) and the rule
   is flat otherwise. Events cluster: in hot phases most settlements qualify (three trades a day),
   while in quiet and bear stretches (much of 2022-2023) funding sits near 0.01% and the rule
   barely trades; strongly negative funding (squeezes, crashes) adds short trades. The estimate is
   about 60 trades per symbol per year at the default (BNB, whose funding is often exactly zero,
   fewer): several hundred closed trades per symbol over the development period (Sept 2019, or
   listing, to 2025-04-30), far above the 30 the gate needs.

## Falsified if
- The core system at the registered parameters fails feasibility: median Sharpe across the ten
  symbols below 0.3, Sharpe positive on fewer than 60% of symbols, a median of fewer than 30
  closed trades per symbol, or its own exposure shifted to random times (monkey test) beaten less
  than 90% of the time. Losing to the monkeys means the hour after a settlement is no different
  from any other hour.
- The entry test (these entries with 5, 10 and 20-bar time exits and the 2/4 ATR bracket) is
  profitable in fewer than 70% of (exit, symbol) cells: prices do not recover after settlements
  with extreme funding, and the idea is wrong.
- Fewer than 70% of the 9 grid cells make money: the effect would hinge on one threshold or one
  holding time.
- Read alongside (diagnostic, not a gate): if only the longs after positive funding make money and
  only in bull years, while the shorts after negative funding lose, it is drift rather than
  settlement timing; if the average gross trade is positive but smaller than the 0.16% cost, the
  effect may exist but cannot be traded at taker costs.
