"""Deflated Sharpe of i006 on SOL alone: as if SOL had been chosen in advance, and counting the
choice of the best of ten coins (dev data only, the gate's chosen setting)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.metrics import periods_per_year
from algotrade.backtest.runner import backtest
from algotrade.research.criteria import load_criteria
from algotrade.research.journal import TrialLedger
from algotrade.research.split import dev_universe
from algotrade.research.workspace import Workspace
from algotrade.strategies import from_spec
from algotrade.validation.overfitting import deflated_sharpe, expected_max_sharpe, moments

ws = Workspace(Path.cwd())
criteria = load_criteria(ws.criteria_path)
result = ws.research / "ideas/i006-bollinger-squeeze-breakout/v1/feasibility/result.json"
spec = json.loads(result.read_text())["chosen_spec"]
bars = dev_universe(ws, criteria, "1h", ["SOL"])["SOL"]
run = backtest(from_spec(spec), bars, EXCHANGE_COSTS[criteria.get("data.exchange")])
ppy = periods_per_year(run.ledger.index)
sr, n, skew, kurt = moments(run.returns.to_numpy())
trials = TrialLedger(ws.trials_path).distinct()
metric = pd.to_numeric(trials["metric"], errors="coerce").dropna()
count = len(trials)
CROSS_COIN_SD = 0.68  # spread of the ten coins' Sharpe ratios at this setting

print(f"SOL annualised Sharpe {sr * np.sqrt(ppy):.2f}, skew {skew:.2f}, kurtosis {kurt:.1f}")
for label, trials_n, var_annual in [
    ("as if SOL were chosen in advance", count, metric.var(ddof=1)),
    ("counting the pick of the best of 10 coins", count * 10, CROSS_COIN_SD**2),
]:
    variance = var_annual / ppy
    bar = expected_max_sharpe(trials_n, variance) * np.sqrt(ppy)
    dsr = deflated_sharpe(sr, n, skew, kurt, trials_n, variance)
    print(f"{label}: {trials_n} trials, luck bar {bar:.2f} a year, deflated Sharpe {dsr:.3f}")
