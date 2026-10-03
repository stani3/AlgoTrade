"""i003 v2 against the same vol-target wrapper around always-long (dev data only)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.runner import backtest
from algotrade.research.criteria import load_criteria
from algotrade.research.split import dev_universe
from algotrade.research.workspace import Workspace
from algotrade.strategies import from_spec

ws = Workspace(Path.cwd())
criteria = load_criteria(ws.criteria_path)
spec = json.loads((ws.research / "ideas/i003-weekly-horizon-time-series-momentum/v2/feasibility/result.json").read_text())["chosen_spec"]
held = {**spec, "strategy": {"type": "buy_and_hold"}}
costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
rows, years = [], []
for symbol, bars in dev_universe(ws, criteria, "1d").items():
    s = backtest(from_spec(spec), bars, costs).ledger["net"]
    v = backtest(from_spec(held), bars, costs).ledger["net"]
    sm, vm = (1 + s).resample("ME").prod() - 1, (1 + v).resample("ME").prod() - 1
    up, down = vm > 0, vm < 0
    beta, alpha = np.polyfit(v, s, 1)
    resid = s - beta * v
    rows.append({"symbol": symbol, "sharpe": s.mean() / s.std() * np.sqrt(365),
                 "vt_hold_sharpe": v.mean() / v.std() * np.sqrt(365),
                 "up_capture_m": sm[up].mean() / vm[up].mean(), "down_capture_m": sm[down].mean() / vm[down].mean(),
                 "beta": beta, "alpha_ann": alpha * 365, "alpha_t": resid.mean() / resid.std() * np.sqrt(len(resid))})
    for year, idx in s.groupby(s.index.year).groups.items():
        years.append({"year": year, "strategy": (1 + s[idx]).prod() - 1, "vt_hold": (1 + v[idx]).prod() - 1})
t = pd.DataFrame(rows)
print(t.round(2).to_string(index=False))
print("\nMedians:", t.drop(columns="symbol").median().round(2).to_dict())
print("\nMedian by year:\n", pd.DataFrame(years).groupby("year").median().round(3).to_string())
