---
id: i002
version: 1
registered: '2026-10-02T15:16:47Z'
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
  allow_short: true
optimise:
  z_entry:
  - 2.5
  - 3.0
  - 3.5
  hold_bars:
  - 3
  - 6
  - 12
expected_trades_per_year: 20
differs_from:
  i001: i001 bets on continuation (a new 48-bar closing high or low with RSI confirmation, ATR stop and
    target); this fades single-bar moves scaled by prior volatility, the opposite effect, with a fixed
    time exit and no stop or target.
---
## Hypothesis
On 4h bars of liquid crypto perpetuals, a close-to-close move of at least three times the
volatility known before it is partly a forced-flow overshoot (liquidations and stop cascades)
that reverses over the following day or so. Fading it at the shock bar's close and holding for a
fixed number of bars makes money after costs and funding.

## Why it should work
Perpetuals are traded with high leverage, and an exchange closes an under-margined position with
a market order whatever the price; clustered stop-losses do the same. A wave of such orders (a
long liquidation cascade on the way down, a short squeeze on the way up) pushes the price further
than unforced traders would take it, because the sellers or buyers are not choosing the price.
When the forced flow is exhausted, part of the move comes back.

The other side of the trade is the liquidated or stopped-out trader who pays for immediacy; the
fade is paid for supplying liquidity at the extreme. This is the price pressure seen after fund
fire sales (Coval & Stafford 2007), short-term reversal as compensation for liquidity provision
that grows with volatility (Nagel 2012), and the margin spirals of Brunnermeier & Pedersen (2009),
in a market where leverage and automatic liquidation make forced trading unusually common.

Scoring each move against the volatility known at the previous close picks out bars where
volatility jumps, the signature of a cascade, and ignores ordinary large bars in an already
volatile market. Funding is positive most of the time, so longs are usually the more leveraged
side and down-shocks the more forced; the rule is symmetric anyway, and the feasibility
diagnostics report long and short trades separately.

## Rules
Position strategy (`target_position` returns -1, 0 or +1) on 4h bars, new code of type
`shock_fade`. Every value uses only bars up to and including the bar it is computed on.

1. Return: `r_t = close_t / close_{t-1} - 1` (undefined on the first bar).
2. Volatility: `sigma_t = indicators.ewm_vol(r, span)` with `span = vol_days x bars per day`
   (25 x 6 = 150 bars on 4h; `ewm_vol`'s own `min_periods`, span // 2, is the warm-up).
3. Shock score: `z_t = r_t / sigma_{t-1}`, measured against the volatility known at the
   previous close so the shock does not dilute its own yardstick. No signal while
   `sigma_{t-1}` is undefined or not positive, or `r_t` is undefined.
4. Bar t is a shock when `|z_t| >= z_entry`. Its trade direction is `-sign(r_t)`: long after a
   down-shock, short after an up-shock.
5. Target at the close of bar t: the trade direction of the most recent shock bar s with
   `t - hold_bars < s <= t`, or 0 if there is none. A position therefore opens at the close of
   the shock bar, is held for exactly `hold_bars` bars (it earns the returns of bars s+1 to
   s+hold_bars) and is closed at the close of bar s+hold_bars. A new shock during a trade
   restarts the clock in its own direction: the same direction extends the trade, the opposite
   one reverses it.
6. `allow_short` false: short targets become 0 (an up-shock closes an open long, then flat).
7. Size: full 1x exposure, no stop, no target, no volatility scaling. Execution, costs and
   funding are the engine's: the target set at a bar's close is held through the next bar,
   every change in exposure pays fee plus slippage, and funding is charged on open positions.
8. Validation: `z_entry > 0`, `hold_bars` an integer >= 1, `vol_days > 0`; every grid cell is
   valid. Defaults: `z_entry` 3.0 (the conventional three-sigma outlier line), `hold_bars` 6
   (one day of 4h bars), `vol_days` 25 (Carver's volatility lookback, as in the codebase's
   `ewmac` and `vol_target`).

## Falsified if
- The core system at the registered parameters fails feasibility: median Sharpe across the ten
  symbols below 0.3, Sharpe positive on fewer than 60% of symbols, fewer than 30 closed trades
  per symbol, or random timing with the same exposure (monkey test) beaten less than 90% of the
  time.
- The entry test (shock entries with 5, 10 and 20-bar time exits and the 2/4 ATR bracket) is
  profitable in fewer than 70% of (exit, symbol) cells. If prices keep moving in the shock's
  direction, shocks carry information rather than forced flow, and the idea is wrong.
- Fewer than 70% of the nine grid cells make money: the effect would hinge on one exact
  threshold or holding time rather than being a broad property of shocks.
