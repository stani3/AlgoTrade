---
name: strategy-incubate
description: Incubate a frozen AlgoTrade strategy after the user approved it - check NautilusTrader parity, prepare testnet paper trading for the user to start, and run periodic incubation reports (shadow forward test on real bars against bootstrap bands, testnet slippage) ending in continue, pass or retire. Use only after the user explicitly said yes to incubating a frozen strategy, or when they ask how an incubating strategy is doing.
---

# Strategy incubate

Davey's incubation: watch the frozen strategy, unchanged, on data that did not exist when it
was built. Paper trading only. Going live with real money is never part of this skill: it is
always the user's own step, outside the pipeline.

## 0. Preconditions

- The user explicitly approved incubating this frozen strategy (`research/ideas/<id>-<slug>/v<n>/
  frozen.json` exists, tag `strategy/<id>-v<n>`). Without that yes, stop.
- NautilusTrader is installed (`pip install -e .[live]`).

## 1. Parity

```bash
python -m pytest -q tests/test_live_parity.py
```

The Nautilus adapter must reproduce our engine's and bracket simulator's trades on historical
bars before anything runs on testnet. If parity fails, stop and report: never paper-trade a
strategy whose live code disagrees with the code that was validated.

## 2. Start the record

```bash
python -m scripts.research incubate <id>
```

Records the start time in the journal (from now on, only new data counts) and commits it.

## 3. Paper trading (the user starts it)

Testnet API keys are the user's: they put them in `.env` themselves (see `scripts/paper_trade.py`
for the variable names) and start the long-running node:

```bash
python -m scripts.paper_trade <id>
```

It refuses to run against anything but testnet. Never ask for, read, write or echo API keys,
and never start the node yourself. Fills are logged to
`research/ideas/<id>-<slug>/v<n>/incubation/fills.csv`.

## 4. Check-ins

Periodically (after downloading new bars with `python -m scripts.download_data`):

```bash
python -m scripts.research incubation-report <id>
```

- **Strategy quality**: the frozen spec run by our engine on real (mainnet) bars since the
  start, against the 5th percentile return and 95th percentile drawdown of same-length paths
  bootstrapped from the walk-forward's out-of-sample returns.
- **Execution**: median testnet slippage per fill against the cost model.

Verdicts: CONTINUE (not yet `incubation.min_days` days and `min_trades` trades, nothing broken),
FAIL (a band or the slippage limit broken: retire it, the record stays), PASS (long enough and
inside the bands).

Send the incubation report (`reports/research/<id>/v<n>/incubation/report.html`) next to the
validation report so the user can compare live behaviour with the backtest.

## Never

Change the frozen spec or stake, restart incubation to reset a bad start, touch API keys, start
or stop the node, or suggest going live as an automatic next step. A PASS means "behaved as
expected on paper"; whether to trade real money is the user's decision alone.
