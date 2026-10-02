# Idea sources

A starting catalogue, not a to-do list. Use the id as `source_id` on the card;
`research/index.md` lists which ids have been used. "Catalogue" names an existing rule (it can
be registered directly as a spec); "new code" means `strategy-build` writes it.

Data limits: OHLCV (4h, 1d) and funding only, one instrument at a time.

## Trend and momentum

| Id | Idea | Where it comes from | Implementation |
|---|---|---|---|
| S01 | Moving-average crossover | Common practice; Carver discusses it as the simplest trend rule | Catalogue `ma_crossover` |
| S02 | Time-series momentum: sign of the past N-bar return | Moskowitz, Ooi & Pedersen, "Time series momentum" (2012) | Catalogue `momentum` |
| S03 | EWMAC: difference of two EMAs scaled by volatility, sized by forecast strength | Carver, *Systematic Trading* | Catalogue `ewmac` |
| S04 | Carver breakout: position of price inside its N-bar range as a forecast | Carver, *Systematic Trading* | Catalogue `carver_breakout` |
| S05 | MACD line versus signal line | Gerald Appel | Catalogue `macd` |
| S06 | ADX/DMI: trade direction of +DI/-DI only when ADX shows a trend | J. Welles Wilder, *New Concepts in Technical Trading Systems* | Catalogue `adx_trend` |
| S07 | Parabolic SAR stop-and-reverse | Wilder | Catalogue `parabolic_sar` |
| S08 | SuperTrend: ATR band trailing the price, flip on a close through it | Common indicator | Catalogue `supertrend` |
| S09 | Ichimoku cloud: price above/below the cloud with Tenkan/Kijun cross | Goichi Hosoda | New code |
| S10 | Trend with a funding filter: skip longs when funding is extreme (crowded) | Crypto-perpetual specific | Composition or new code |

## Breakout and volatility

| Id | Idea | Where it comes from | Implementation |
|---|---|---|---|
| S11 | Donchian channel breakout with a shorter exit channel (Turtle system) | Richard Dennis and William Eckhardt's Turtles; Curtis Faith, *Way of the Turtle* | Catalogue `donchian_breakout`, `specs/turtle_55_20.json` |
| S12 | Bollinger band breakout | John Bollinger, *Bollinger on Bollinger Bands* | Catalogue `bollinger_breakout` |
| S13 | Keltner channel (EMA +- ATR) breakout | Chester Keltner; ATR version popularised by Linda Raschke | Catalogue `keltner_breakout` |
| S14 | Volatility contraction then expansion: narrowest range in N bars (NR4/NR7), trade the break of that bar | Toby Crabel, *Day Trading with Short Term Price Patterns and Opening Range Breakout* | New code (bracket) |
| S15 | Bollinger squeeze: bandwidth at an N-bar low, then trade the breakout direction | Bollinger | New code |
| S16 | Range breakout of the previous bar's range times k ("dual thrust" style) | Michael Chalek's Dual Thrust | New code (bracket) |
| S17 | Volume-confirmed breakout: channel break only on above-average volume | Common practice | New code |
| S18 | Breakout with RSI filter and ATR bracket | The user's book example | Catalogue `breakout_bracket`; tested as i001, failed |

## Mean reversion

| Id | Idea | Where it comes from | Implementation |
|---|---|---|---|
| S21 | Short-term RSI(2) reversion with a long-term trend filter | Larry Connors & Cesar Alvarez, *Short Term Trading Strategies That Work* | Catalogue `rsi_reversion`, `specs/connors_rsi2.json` |
| S22 | Bollinger band reversion to the mean | Bollinger | Catalogue `bollinger_reversion` |
| S23 | Williams %R or Stochastic oversold/overbought reversion | Larry Williams; George Lane | New code |
| S24 | Fade an outsized single bar (range z-score extreme), exit after N bars: forced liquidations overshoot | Crypto-perpetual specific | New code (bracket with time exit) |
| S25 | Fade extreme funding: when longs pay a lot, positioning is crowded and tends to unwind | Crypto-perpetual specific | New code |

## Carry and seasonality

| Id | Idea | Where it comes from | Implementation |
|---|---|---|---|
| S31 | Funding carry: hold the side that receives funding, sized by carry / volatility | Carver's carry rule adapted to perpetuals | Catalogue `funding_carry` |
| S32 | Day-of-week effect (weekend liquidity, Monday reopening of traditional markets) | Calendar studies in crypto | New code (calendar input) |
| S33 | Time-of-day sessions on 4h bars (Asia / Europe / US opens, UTC) | Calendar studies in crypto | New code (calendar input) |
| S34 | Funding-settlement timing: positioning before and after the 8-hourly settlement | Crypto-perpetual specific | New code (calendar + funding) |

## Sizing and filters (combine with a rule above)

| Id | Idea | Where it comes from | Implementation |
|---|---|---|---|
| S41 | Volatility targeting of any rule | Carver, *Systematic Trading* | Wrapper `vol_target` |
| S42 | Long-term trend filter (only trade with the 200-bar trend) | Common practice; used by Connors | Wrapper `trend_filter` |
| S43 | Forecast combination of uncorrelated rules | Carver, *Systematic Trading* | Wrapper `combine` |
| S44 | Regime filter: trade trend rules only when ADX or realised volatility says the market trends | Common practice | Composition or new code |

## Not possible with the current data or engine

Order-book imbalance, open interest, liquidation feeds, on-chain flows, cross-sectional
(multi-coin) momentum and pairs trading, trailing-stop exits (the bracket simulator only has
fixed stops, targets and, after phase 2, time exits).
