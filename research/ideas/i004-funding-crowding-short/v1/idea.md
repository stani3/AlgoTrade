---
id: i004
version: 1
registered: '2026-10-02T18:58:09Z'
title: Funding-crowding short
source: 'Fade extreme funding (sources.md S25). Schmeling, Schrimpf & Todorov, ''Crypto Carry'' (BIS WP
  1087): crypto carry reflects demand for leveraged long exposure from trend-chasing investors meeting
  limited arbitrage capital, and high carry goes with later liquidations of long futures positions (crash
  risk); Brunnermeier, Nagel & Pedersen, ''Carry Trades and Currency Crashes'' (NBER Macroeconomics Annual
  2008): crowded speculative positions unwind suddenly, and speculators'' futures positioning predicts
  crash risk; Brunnermeier & Pedersen, ''Market Liquidity and Funding Liquidity'' (RFS 2009): margin spirals;
  He, Manela, Ross & von Wachter, ''Fundamentals of Perpetual Futures'' (2022): funding ties the perpetual
  to spot, and perpetual premia persist because arbitrage is limited. Binance''s neutral funding rate
  (interest component 0.01% per 8h). Liu, Tsyvinski & Wu, ''Common Risk Factors in Cryptocurrency'' (JF
  2022): quintile sorts. Wilder''s ATR(14); LeBeau''s 3-ATR chandelier exit multiple.'
source_id: S25
taxonomy:
  family: mean_reversion
  inputs:
  - funding
  - price
  horizon: days
timeframe: 1d
spec:
  type: i004_funding_crowding_short
  funding_days: 7
  quantile: 0.8
  hold_bars: 7
  history_days: 365
  exit_quantile: 0.5
  neutral_daily_funding: 0.0003
  stop_atr: 3.0
optimise:
  quantile:
  - 0.7
  - 0.8
  - 0.9
  funding_days:
  - 3
  - 7
  - 14
  hold_bars:
  - 3
  - 7
  - 14
expected_trades_per_year: 8
differs_from:
  i002: i002 fades a single 4h bar's price move of three or more prior volatilities (price only; symmetric
    in v1, long only in v2), buying after a liquidation cascade has already happened and holding 3-12
    bars. This sells short on daily bars when perpetual funding, a positioning measure rather than a price
    move, shows longs crowded and paying a premium, before any cascade; it never goes long and exits on
    funding normalising, a time limit or an ATR stop.
  i003: i003 holds the sign of the past 28-day return (trend following; v1 also shorted after falls, v2
    is long only). This ignores the price trend at entry and shorts on extreme funding, which mostly comes
    after rises, so it leans counter-trend. It was suggested by a split of i003 v1's short trades by funding
    (see the disclosure under Hypothesis), but it shares none of i003's entry or exit rules.
---
## Hypothesis
On daily bars of liquid crypto perpetuals, the onset of crowded long positioning, seen as a
week of funding that is both high for the coin's own past year and above Binance's neutral
rate, is followed over the next days to two weeks by a pullback as the crowd unwinds (long
liquidations, margin calls, longs giving up the carry they pay). Shorting at that onset, short
only, and covering when funding normalises, after a week, or on a 3-ATR stop, makes money after
fees, slippage and funding on most of the ten coins, through its timing rather than through the
market's drift.

Disclosure: this idea is partly informed by development data. It was chosen AFTER the user and
the agent looked at i003 v1's short trades split a dozen ways on development data. The only
positive split was shorts entered when the 7-day summed funding was in its top third (253
trades, about +0.1% a trade), against losses in the other two thirds. That is weak evidence:
one positive cell out of about twelve looked at, a small edge per trade, and trades taken on
i003's entries (shorts after a negative four-week return), not on this rule's. The default
7-day funding window is the window that split used (it is also one natural week of 21 Binance
settlements); the default threshold (top quintile of the coin's own trailing year) is not the
split's tercile, and the floor, the exits and the stop were not looked at in any data. Passing
feasibility here would therefore count for less than for an idea drawn from theory alone: the
walk-forward analysis, the deflated Sharpe ratio (counting every trial, 249 so far plus this
grid) and the single locked-holdout look after 2025-05-01 are the real tests.

## Why it should work
Funding is the price of leverage on a perpetual. When the perpetual trades above spot, longs pay
shorts every eight hours until arbitrageurs (short perp, long spot) close the gap. Binance sets
funding as `P + clamp(I - P, -0.05%, +0.05%)`, with P the premium index and I a fixed interest
component of 0.01% per 8h (0.03% a day, about 11% a year): while the premium is small the rate
sits exactly at that neutral level, and it rises above it only when the perpetual trades at a
real premium to spot. Funding
well above neutral for a week therefore means sustained demand for leveraged long exposure that
arbitrage capital is not absorbing (He, Manela, Ross & von Wachter; Schmeling, Schrimpf &
Todorov trace this demand to smaller, trend-chasing investors).

Such a crowd is fragile. Its margin buffers are thin, it bleeds funding every eight hours, and
it is all on one side. A modest decline puts the most leveraged longs under water; their
liquidations are market sells that push the price further and reach the next layer (the margin
spiral of Brunnermeier & Pedersen 2009). Brunnermeier, Nagel & Pedersen (2008) find the same in
currency carry trades: crowded speculative positions build up slowly and unwind in sudden
crashes, and speculators' futures positioning predicts that crash risk. Schmeling, Schrimpf &
Todorov link high crypto carry to later liquidations of long futures positions.

The other side of the trade is the leveraged long paying a premium for leverage. The short takes
the arbitrageur's side without the spot hedge: it collects the funding the crowd pays while it
waits, which partly pays for being early, and it bets on the unwind.

Two conditions define crowding. Relative to the coin's own past year (a percentile rank) makes
the rule comparable across coins whose normal funding differs, and across regimes. Above
neutral makes sure longs are actually paying a premium: in quiet or bear stretches funding sits
at or below neutral, and a "high for the year" reading there is not crowding, and shorting it
would only collect the bear market's drift. The entry is the onset of crowding (the first day
both conditions hold), not every crowded day, so a short squeezed out of a still-crowded market
is not re-opened into the same squeeze.

Timeframe 1d: the signal is a multi-day sum of funding that settles three times a day and moves
slowly, and the claimed unwind plays out over days to two weeks. Daily bars match both; 4h bars
would only add intraday noise, more stop hits and costs, and Davey's entry-test exits (5, 10 and
20 bars) would become 20-80 hours, too short to judge a weekly effect. On daily bars each bar
holds the three settlements of its day, all known at its close.

What can still sink it. The development period is mostly a bull market, crowding happens in
rallies, and shorting rallies fights the drift and risks squeezes (DOGE in January 2021 rose
several-fold in days). Funding can stay extreme for weeks in a strong bull run (early 2021), so
the unwind's timing is uncertain: the rule bets that it usually comes within about a week. The
stop is checked on daily closes only, so a gap day can lose far more than 3 ATR on a 1x short.
Part of any profit is funding received, which is carry (the catalogue's always-on
`funding_carry` was screened before the journal, not gated) rather than the unwind this card
claims. The monkey test shifts this rule's exposure in time, so it keeps the drift exposure but
loses both the timing of the price moves and the extra funding collected at crowded times.

## Rules
Position strategy (`target_position` returns -1 or 0, never long) on 1d bars, new code of type
`funding_crowding_short`: the catalogue cannot express it (`funding_carry` is an always-on
continuous forecast with no thresholds, exits or stop). Every value uses only bars up to and
including the bar it is computed on. `funding_rate` on a bar is the sum of the funding settled
during that bar (0 when none), known at its close.

1. Window lengths in bars: `nW = round(funding_days x bars_per_day)` (7 on 1d),
   `nY = round(history_days x bars_per_day)` (365 on 1d).
2. Funding sum: `F_t` = sum of `funding_rate` over bars `t-nW+1 .. t` (pandas rolling sum with
   `min_periods = nW`); undefined for the first `nW - 1` bars.
3. Rank: `rank_t` = (number of defined F values in bars `t-nY+1 .. t` that are `<= F_t`) /
   (number of defined F values in that window), i.e. `F.rolling(nY, min_periods=nY // 4)
   .rank(method="max", pct=True)`. Undefined until `nY // 4` (91 on 1d) F values exist, so a
   symbol trades from about its fourth month instead of waiting a full year.
4. Crowded: `crowded_t` is true when `rank_t >= quantile` and
   `F_t > neutral_daily_funding x funding_days + 1e-9` (the window's average funding is above
   Binance's neutral 0.03% a day; the 1e-9 keeps floating-point sums of exactly-neutral
   settlements from counting). False whenever `rank_t` or `F_t` is undefined.
5. Onset: `onset_t = crowded_t and not crowded_{t-1}` (the bar before the first bar counts as
   not crowded).
6. ATR: `atr_t = indicators.atr(high, low, close, 14)` (Wilder).
7. State machine, bar by bar; the target at the close of bar t:
   - Flat at the close of bar t-1: open a short (target -1) when `onset_t` and `atr_t` is
     defined. Record the entry bar `s = t` and the stop level
     `stop = close_s + stop_atr x atr_s`.
   - Short since the close of bar s < t: cover (target 0) when any of
     a. time: `t - s >= hold_bars` (the trade earns bars s+1 .. s+hold_bars at most);
     b. funding normalised: `rank_t < exit_quantile` (the weekly funding is back below its
        trailing-year median);
     c. stop on close: `close_t >= stop`.
     Otherwise stay short. Onsets while short are ignored and do not restart the clock.
   - A bar that covers does not open a new short (entries need flat at the previous close), so
     there is at least one flat bar between trades. After a time or stop exit with funding still
     crowded, a new short needs a fresh onset (crowding must lapse and return).
8. Size: full 1x short (-1), no volatility scaling. Execution, costs and funding are the
   engine's: the target set at a bar's close is held through the next bar, every change in
   exposure pays fee plus slippage (5 + 3 bps on Binance), and funding settled while short is
   received when positive and paid when negative.
9. Defaults and grid. `funding_days` 7: one week, 21 Binance settlements (also the split's
   window, see the disclosure). `quantile` 0.8: the top quintile of the coin's own trailing
   year, the extreme bucket of the quintile sorts used in the crypto factor literature (Liu,
   Tsyvinski & Wu, 'Common Risk Factors in Cryptocurrency', JF 2022). `hold_bars` 7: a crowd
   measured over a week is given a week to unwind. Fixed, not optimised: `history_days` 365
   (the coin's own past year), `exit_quantile` 0.5 (the median, "normal" funding),
   `neutral_daily_funding` 0.0003 (Binance's interest component, 0.01% per 8h = 0.03% a day),
   `stop_atr` 3.0 (LeBeau's chandelier-exit multiple; wider than the entry test's 2 ATR because
   crowded markets are volatile) and ATR length 14.
   Grid: `quantile` {0.7, 0.8, 0.9} x `funding_days` {3, 7, 14} x `hold_bars` {3, 7, 14}, 27
   combinations. Validation: `funding_days > 0` with `nW >= 1`, `history_days > funding_days`,
   `0 < exit_quantile < quantile <= 1`, `hold_bars` an integer >= 1, `stop_atr > 0`,
   `neutral_daily_funding >= 0`; every grid cell is valid.
10. Expected trades: about 20% of days rank in the top quintile of their trailing year; the
    neutral floor removes most of them in quiet and bear stretches (much of 2022-2023), and the
    rest come in bursts of days, clustered in rallies. With trades of at most a week that ignore
    onsets while open, expect roughly 6-10 trades per symbol per year (8): about 40 closed trades
    per symbol over the development period (2019-09 or the symbol's listing to 2025-05, minus
    about three months of warm-up; SOL and AVAX start in 2020-09). The trade count is the
    tightest feasibility gate for this rule.

## Falsified if
- The core system at the registered parameters fails feasibility: median Sharpe across the ten
  symbols below 0.3, Sharpe positive on fewer than 60% of symbols, a median of fewer than 30
  closed trades per symbol, or its own exposure shifted in time (monkey test) beaten less than
  90% of the time.
- The entry test (crowding-onset short entries with 5, 10 and 20-day time exits and the 2/4 ATR
  bracket) is profitable in fewer than 70% of (exit, symbol) cells: prices do not fall after
  longs become crowded, and the idea is wrong.
- Fewer than 70% of the 27 grid cells make money: the effect would hinge on one threshold,
  window or holding time rather than being a broad property of funding crowding.
- Later, as pre-registered by the pipeline: the walk-forward out-of-sample results, the deflated
  Sharpe ratio counting every trial, or the single holdout look after 2025-05-01 fail. Given the
  disclosure above, a pass in feasibility alone is not evidence that the effect is real.
- Read alongside (diagnostic, not a gate): if the profit is mostly funding received while
  prices did not fall (price P&L near zero or negative), it is carry, not the unwind this card
  claims; if it comes only from the 2022 bear market, or from one or two coins or episodes, it
  is drift or luck rather than crowding.
