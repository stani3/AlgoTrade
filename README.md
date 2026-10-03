# AlgoTrade

Research stack for systematic trading of crypto perpetual futures (Binance USDⓈ-M, Bybit), US
stocks, bond and commodity ETFs, and forex, on 1h, 4h and daily bars. This repo is the **research
layer**: data, a fast vectorized backtester, the validation pipeline and the AI strategy loop.
Live execution will run on NautilusTrader.

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

USDC-margined perpetuals sit next to the USDT ones (`BTCUSDC` beside `BTCUSDT`); `--symbols ALL`
takes every active perpetual in the chosen quote currency. On Binance they only exist since
January 2024, so most of their history falls in the research holdout period:

```powershell
python -m scripts.download_data --quote USDC --symbols ALL
python -m scripts.run_backtest --strategy ewmac --quote USDC --symbols BTC,ETH --timeframe 4h
```

Files land in `data/raw/<exchange>/<SYMBOL>_<tf>.parquet` and `<SYMBOL>_funding.parquet`.
`algotrade.data.market.load_market()` returns bars indexed by bar-open time (UTC) with each funding
settlement summed into the bar during which it is charged. Binance stamps settlements 0-47 ms
after the hour: `funding_alignment="nearest_second"` rounds each stamp to the second, so every
settlement lands on the bar that closes at its settlement hour. The default, `raw_timestamp`,
takes the stamp as recorded and so charges the ~45% stamped late one bar late; it stays the
default because recorded research results were computed with it (see `data.funding_alignment`
under Research pipeline). The command-line scripts use the default.

### Stocks, ETFs and forex

```powershell
python -m scripts.download_data --asset-class stocks --check       # 10 stocks (Alpaca)
python -m scripts.download_data --asset-class bonds --check        # 10 bond ETFs
python -m scripts.download_data --asset-class commodities --check  # 10 commodity ETFs
python -m scripts.download_data --asset-class fx --check           # 4 forex pairs (Dukascopy)
```

- **US stocks and ETFs** come from [Alpaca](https://alpaca.markets)'s free market-data plan:
  consolidated-tape bars since 2016, adjusted for splits, dividends and spin-offs (a buy-and-hold
  series is a total-return series). Put `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`. Only
  regular hours are kept, on the NYSE calendar: 1h bars open at 09:30, 10:30 ... 15:30 (the last
  one half an hour), 4h bars at 09:30 and 13:30, daily bars cover the session. Updates append to
  the stored 30-minute base without rewriting it (adjusted history is re-based on every
  dividend), scaling the new bars to meet the stored ones.
- **Forex** comes from Dukascopy's public datafeed (no account): hourly bid and ask candles, mid
  prices, inside the Sunday-to-Friday week; 4h and daily bars start at 17:00 New York.
- **Financing** (`<SYMBOL>_funding.parquet`, from FRED): a stock or ETF position pays the fed
  funds rate overnight, a forex position pays or earns the interest-rate difference at the
  17:00 New York rollover (Wednesdays count three days). Longs pay, shorts receive, so returns are
  in excess of cash, as with futures and perpetuals.
- `--check` reports duplicate bars, inconsistent prices, jumps (25% crypto, 15% stocks, 5% forex,
  e.g. GBPUSD on 24 June 2016), bars without trades and missing session bars.

The instruments research may use are listed by asset class in `research/criteria.yaml`
(`data.exchange`/`data.symbols` for crypto, `data.universes` for the others, with their costs and
leverage caps). A second checkout (a git worktree) can read another one's data with
`ALGOTRADE_DATA_ROOT=<path to data/raw>`.

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
  Financing of other instruments is paid by the position held after the charge.
- Metrics annualize with each market's own calendar (`src/algotrade/calendars.py`, read off the
  bars): 365.25 days for crypto, 252 sessions of 6.5 hours for US stocks and ETFs, 260 days for
  forex. Counting a year of 1h stock bars as 8,766 would overstate a Sharpe ratio 2.2 times.

### Performance report

Add `--report` to write a performance report in the style of Davey's *Building Winning
Algorithmic Trading Systems* to `reports/<strategy>_<timeframe>/` (or `--report <folder>`):

```powershell
python -m scripts.run_backtest --spec specs/breakout_bracket.json --timeframe 4h --symbols BTC,ETH,SOL --report
```

`report.html` is self-contained (charts embedded). It has one section per market:

- an equity curve against buy & hold, with drawdowns underneath;
- the TradeStation-style trade analysis for all, long and short trades: net profit, profit
  factor, percent profitable, average and largest winners/losers, consecutive wins/losses, bars
  held;
- account and drawdown statistics: CAGR, Sharpe, max drawdown in % and $, longest time
  underwater, return on account, costs and funding paid;
- annual and monthly returns;
- Davey's Monte Carlo test: one year of trades resampled 2,500 times at 0.5x, 1x and 2x size, with
  his goals (risk of ruin < 10%, median max drawdown < 40%, return / drawdown > 2) marked
  pass/fail;
- the full list of trades.

`summary.csv` and `trades_<market>.csv` hold the same numbers for further analysis. Options:
`--capital` (default 10,000), `--mc-runs`, `--mc-ruin` (default: losing 50% of the start).
Trades are sized from current equity, so per-trade figures are a percentage of the equity at
entry; money figures follow the compounded account and add up exactly to its net profit.

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
Parameters given in days mean sessions in session markets (`bars_per_day`).

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

### Bracket strategies (stops and profit targets)

`breakout_bracket` (spec: `specs/breakout_bracket.json`) runs on a separate bar-by-bar simulator,
`src/algotrade/backtest/bracket.py`, because its exits happen *inside* a bar and its rules depend on
earlier trades:

- Long when the close is the highest close of `lookback` bars and RSI(`rsi_length`) > `rsi_level`;
  short on the mirror image. Entry at the **next bar's open**; opposite signals during a trade are
  ignored.
- Stop `stop_atr` x ATR and target `target_atr` x ATR from the fill (ATR read on the signal bar).
  A stop or target touched intrabar fills at that level; a gap fills at the open; if one bar
  touches both, the stop is assumed first. Targets are limit orders and pay no slippage.
- After a winning trade wait `cooldown_win` bars, after a loser (after costs) `cooldown_loss` bars.
- Kill switch: once equity is `kill_drawdown` below its peak the strategy stops trading for good.
- Optional time exit (`max_bars`, used by the entry test): a trade still open after that many bars
  exits at the close with slippage. Infinite stop/target distances mean time exits only.

Explore its parameters with the grid script. It prints the total number of trials and saves a CSV
plus median-Sharpe heatmaps (look for a broad plateau, not a single bright cell):

```powershell
python -m scripts.grid --strategy breakout_bracket --grid stop_atr=1,1.5,2,3,4 --grid target_atr=2,3,4,6,8 --grid atr_length=14,30 --timeframe 4h
```

The simulator is tested against an independent reference implementation on random markets and
has 100% line and branch coverage (`NUMBA_DISABLE_JIT=1 coverage run --branch -m pytest`).

### Adding a rule

Subclass `Strategy` as a frozen dataclass in `src/algotrade/strategies/`, implement
`target_position(bars) -> pd.Series`, and add the class to `RULES` in `strategies/catalog.py`. The
tests then automatically check it for lookahead, bounds, spec round-trip, and that it trades.

## Research pipeline

Strategies are developed the way Davey describes: idea, limited feasibility testing, validation,
then incubation. The work is split into Claude Code skills (`.claude/skills/`) for the parts
that need judgement, and deterministic gates in `src/algotrade/research/` (CLI:
`scripts/research.py`) that decide PASS/FAIL against thresholds fixed in advance in
`research/criteria.yaml`. That file is yours: `.claude/settings.json` makes Claude ask before
every edit of it, Claude changes it only when you ask for a specific change, and the pipeline
stages never touch it.

| Skill | Does |
|---|---|
| `strategy-ideas` | Proposes ideas from books, classic indicators and crypto effects and pre-registers them as idea cards |
| `strategy-build` | Implements a card as a spec, writing new strategy code with a full test suite when needed |
| `strategy-feasibility` | Runs Davey's limited testing, reads the diagnostics for patterns, may propose one revision |
| `strategy-validate` | Walk-forward, deflated Sharpe, Monte Carlo stake, the one holdout look, freeze, and the approval package |
| `strategy-pipeline` | The loop: ideas through validation within a budget, moving on after each failure |
| `strategy-incubate` | After the user approves: parity check, testnet paper trading set-up, periodic incubation reports |

```powershell
python -m scripts.research seed                  # once: import pre-journal experiments
python -m scripts.research new draft_card.md     # pre-register an idea (assigns i002, i003...)
python -m scripts.research build-check i002      # ruff, tests, 100% coverage, duplicate checks
python -m scripts.research feasibility i002      # limited testing on development data
python -m scripts.research validate i002         # walk-forward, deflated Sharpe, stake
python -m scripts.research holdout i002          # the single look at data after 2025-05-01
python -m scripts.research freeze i002           # frozen.json, decision.md, git tag
python -m scripts.research report i002 validation   # rebuild an HTML report from the record
python -m scripts.research incubate i002         # start incubation (only after the user approved)
python -m scripts.research incubation-report i002   # shadow forward test + testnet slippage
python -m scripts.research status                # refresh research/index.md
python -m scripts.research revise i002 v2.md --reason "..."   # one revision per idea
python -m scripts.research abandon i002 --reason "..."
```

Rules the code enforces:

- **Pre-registration.** A card fixes the rules, the parameters and the only grid that will ever
  be optimised before any test runs. The journal stores the card's hash; an edited card is
  refused, and changes go through `revise` (budget: one revision per idea).
- **No duplicates.** A new card is refused if it repeats a registered configuration, overlaps a
  registered idea's parameter region on the same timeframe (fixed values cover +-20%), is the
  same rules on another timeframe (that is a revision), or does not explain how it differs from
  earlier ideas with the same family, horizon and inputs. When the strategy is built, its
  position fingerprint (daily exposure on BTC/ETH/SOL, positions only) must not correlate 0.9+
  with any earlier idea or with buy & hold; new strategy code must also not re-create a
  catalogue rule the scan already screened (use the catalogue rule instead). Failed ideas stay
  failed unless the user passes `--retest`.
- **Every trial counts.** `research/trials.csv` records every configuration ever evaluated; the
  count feeds the deflated Sharpe ratio, so testing more ideas raises the bar.
- **Holdout.** Development code can only load bars before `data.dev_end` (2025-05-01). The
  holdout is looked at once per idea version, and every look is journaled.
- **Pre-registered universe.** A card names the asset classes it is about (`universe: [fx,
  commodities]` or `all`; none means crypto) and every gate is judged on exactly those
  instruments, with a per-class breakdown for the agent to read. A strategy may work on some
  markets and not others, but only the markets named before testing count: choosing the
  winners afterwards is cherry-picking. The same rules on another universe are a revision and
  another trial (`trials.csv` keys trials by universe; an older ledger gains the column with
  the one-time `python -m scripts.research migrate-trials`). Funding inputs need `[crypto]`.
- **Leverage.** Position strategies may go up to each instrument's `max_leverage` (crypto 1x,
  bond and commodity ETFs as proxies for futures 10x, single stocks and ETFs without a futures
  counterpart 2x), and the validation stake stays within it. Strategies on other asset classes
  stop at the freeze: paper trading runs on the Binance and Bybit testnets only.
- **Coverage.** New strategy code needs 100% line and branch coverage, measured with Numba's
  JIT off, before it can be tested on data.
- **Versioned record.** Every idea (failed ones included), its code, tests, results and
  fingerprint are committed: each stage makes a local commit (never a push). Tested strategy
  versions never change behaviour; `tests/ideas/test_fingerprints.py` recomputes every stored
  fingerprint to prove it.
- **Funding alignment.** `data.funding_alignment` in `criteria.yaml` says how funding
  settlements are put on bars: `raw_timestamp` (what a missing key means, and what every result
  before the key existed used) or `nearest_second` (see Data). Results, fingerprints and
  `frozen.json` record the alignment they used. An idea version keeps the alignment of its first
  result through every later stage, `research report` reproduces with the recorded one, and the
  fingerprint test recomputes each fingerprint with its own, so changing the key only affects
  versions that have no result yet.

Feasibility (development data only) runs Davey's limited testing: the entry test (the
strategy's entries with 5/10/20-bar time exits and a 2/4 ATR bracket; `feasibility.entry_scoring`
in `criteria.yaml` scores each cell as `fixed`, Davey's same-size trades summed, or
`compounded`, the whole equity in every trade, which is harsher and is the default when the key
is missing), the core system across
all symbols, a monkey test against 1,000 random strategies with the same habits, limited
optimisation over the card's pre-registered grid (most combinations must make money; parameters
come from the centre of the best plateau, not the best cell), and diagnostics for the agent to
read: long vs short, symbols, years, ADX and volatility regimes, holding times, MAE/MFE. It
writes the Davey report for the chosen parameters to `reports/research/<id>/v<n>/feasibility/`.

Validation (still development data) runs a rolling walk-forward: every 6 months the card's grid
is re-optimised on the previous 2 years and traded on the next 6, and the out-of-sample pieces
are stitched into one record per symbol. It must keep at least half the in-sample return
(walk-forward efficiency), a median Sharpe of 0.3 and 60% profitable windows; its deflated Sharpe
ratio (median symbol, against every configuration in the trial ledger) must reach 0.95; and
Davey's Monte Carlo on the out-of-sample trades sets the stake, the largest size with risk of
ruin under 10%, median drawdown under 40% and return/drawdown above 2. Then the holdout: one
journaled look at the data after the cutoff, judged against paths bootstrapped from the
out-of-sample bar returns. A strategy that passes is frozen (`frozen.json`, `decision.md`, tag
`strategy/<id>-v<n>`) and shown to the user with its reports before anyone decides to incubate.

Incubation watches the frozen strategy, unchanged, on new data. Its quality is judged by a
shadow forward test (our engine on real mainnet bars since the start) against the 5th percentile
return and 95th percentile drawdown of bootstrapped paths of the same length; testnet paper
trading (NautilusTrader, started by the user) is used to measure execution, its slippage
compared with the cost model. It needs `incubation.min_days` and `min_trades` before it can pass.
Going live with real money is never part of the pipeline.

Paper trading runs our own strategy code inside NautilusTrader (`src/algotrade/live/`): the
adapter reproduces our engine and bracket simulator trade for trade (`tests/test_live_parity.py`),
and the node can only be configured for the Binance or Bybit testnet:

```powershell
pip install -e .[live]
python -m scripts.paper_trade i002 --parity    # adapter vs our engine on recent real bars
python -m scripts.paper_trade i002 --dry-run   # what would run; connects nothing
python -m scripts.paper_trade i002             # you start it; testnet keys from .env
```

`research/index.md` is the readable list of every idea and why it stopped.

## Tests

```powershell
pytest                                        # about 4 minutes
pytest -m slow tests/test_regression_feasibility.py   # recomputes committed crypto results
```

Research stages run one job per symbol in worker processes: `--workers N` on
`scripts.research`, or `ALGOTRADE_WORKERS` (default: CPUs minus two; tests run inline). The
results are identical with any number of workers.

## Roadmap

1. ~~Real data, cost-aware engine, correct metrics~~
2. ~~Validation pipeline: walk-forward, parameter-neighbourhood and cross-symbol checks, Monte Carlo
   on trades, deflated Sharpe over the total trial count, locked holdout~~
3. ~~Strategy spec format + AI generation loop (every trial logged and counted)~~
4. Carver-style portfolio with volatility targeting across survivors
5. ~~Spec → NautilusTrader strategy, parity backtest, testnet paper trading~~ (crypto)
6. ~~Stocks, ETFs and forex next to crypto, parallel research stages~~; a paper venue for them
