# AlgoTrade

Research stack for systematic trading of crypto perpetual futures (Binance USDⓈ-M, Bybit) on 4h and
daily bars. This repo is the **research layer**: data, a fast vectorized backtester and, next, the
validation pipeline and AI strategy loop. Live execution will run on NautilusTrader.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Data

```powershell
# 10 liquid perps, 4h + 1d bars + funding history, from listing date. Re-run to update incrementally.
python -m scripts.download_data --exchange binanceusdm
python -m scripts.download_data --exchange bybit --symbols BTC,ETH --timeframes 1d
```

Files land in `data/raw/<exchange>/<SYMBOL>_<tf>.parquet` and `<SYMBOL>_funding.parquet`.
`algotrade.data.market.load_market()` returns bars indexed by bar-open time (UTC) with each funding
settlement summed into the bar during which it is charged.

## Backtesting

```powershell
python -m scripts.run_backtest --strategy ma_crossover --param fast=20 --param slow=100 --symbols BTC,ETH,SOL --timeframe 1d --plot reports/ma.png
python -m scripts.run_backtest --strategy parabolic_sar --timeframe 4h --start 2022-01-01
```

Each run prints the strategy next to buy-and-hold for every symbol, net of all costs.

Engine rules (`src/algotrade/backtest/engine.py`):

- A strategy returns a **target exposure** per bar (signed fraction of equity) computed from data up
  to that bar's close. It is held through the next bar, so there is no lookahead in the engine.
- Every change in exposure pays fee + slippage (defaults: 5 bps taker + 3 bps slippage on Binance,
  5.5 + 3 on Bybit; override with `--fee-bps` / `--slippage-bps`).
- Funding is paid on the position held when each settlement happens (longs pay positive funding).
- Metrics annualize over 365.25 days, because crypto trades every day.

## Strategy catalogue

| Family | Rules (`--strategy` name) |
|---|---|
| Trend | `ma_crossover`, `momentum`, `macd`, `supertrend`, `adx_trend`, `parabolic_sar` |
| Breakout (entry/exit channels) | `donchian_breakout` (Turtle), `bollinger_breakout`, `keltner_breakout` |
| Mean reversion | `rsi_reversion` (incl. Connors RSI-2 via params), `bollinger_reversion` |
| Carver forecasts (sized by conviction) | `ewmac`, `carver_breakout`, `funding_carry` |
| Wrappers | `vol_target` (size by risk), `trend_filter` (SMA gate), `combine` (weighted blend) |

Every rule takes `allow_short`. Indicators live in `src/algotrade/indicators.py` (SMA, EMA, Wilder,
ATR, RSI, Bollinger, Donchian, Keltner, MACD, ADX/DI, z-score, Parabolic SAR, SuperTrend).

### Strategies as JSON specs

Any strategy, including nested wrappers, can be written as JSON and run without code:

```json
{"type": "vol_target", "annual_vol": 0.25,
 "strategy": {"type": "combine", "multiplier": 1.2,
   "strategies": [{"type": "ewmac", "fast": 16, "slow": 64},
                  {"type": "carver_breakout", "lookback": 80}]}}
```

```powershell
python -m scripts.run_backtest --spec specs/carver_trend.json --timeframe 4h --symbols BTC,ETH,SOL
python -m scripts.run_backtest --strategy donchian_breakout --param entry=55 --param exit=20 --vol-target 0.25
```

Examples are in `specs/`. The backtest prints each run's spec, so a variant you try is easy to save.

### Scanning the catalogue

```powershell
python -m scripts.scan --timeframe 4h --vol-target 0.25 --specs specs
```

Runs every rule at its default parameters (plus every spec file) over all 10 symbols and ranks by
median Sharpe across symbols, so one lucky coin cannot top the table. Per-symbol detail goes to
`reports/scan.csv`.

### Adding a rule

Subclass `Strategy` as a frozen dataclass in `src/algotrade/strategies/`, implement
`target_position(bars) -> pd.Series`, and add the class to `RULES` in `strategies/catalog.py`. The
tests then automatically check it for lookahead, bounds, spec round-trip, and that it trades.

## Tests

```powershell
pytest
```

## Roadmap

1. ~~Real data, cost-aware engine, correct metrics~~
2. Validation pipeline: walk-forward, parameter-neighbourhood and cross-symbol checks, Monte Carlo on
   trades, deflated Sharpe over the total trial count, locked holdout
3. Strategy spec format + AI generation loop (every trial logged and counted)
4. Carver-style portfolio with volatility targeting across survivors
5. Spec → NautilusTrader strategy, parity backtest, testnet paper trading
