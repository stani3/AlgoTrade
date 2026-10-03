"""i003 v2 at the chosen parameters against buy & hold, by year and by market direction (dev data only)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.runner import backtest, run_backtest
from algotrade.research.criteria import load_criteria
from algotrade.research.split import dev_universe
from algotrade.research.workspace import Workspace
from algotrade.strategies import from_spec

ws = Workspace(Path.cwd())
criteria = load_criteria(ws.criteria_path)
result = json.loads(
    (ws.research / "ideas/i003-weekly-horizon-time-series-momentum/v2/feasibility/result.json").read_text()
)
spec = result["chosen_spec"]
costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
universe = dev_universe(ws, criteria, "1d")

yearly, overall = [], []
for symbol, bars in universe.items():
    strat = backtest(from_spec(spec), bars, costs)
    hold = run_backtest(bars, pd.Series(1.0, index=bars.index), costs)
    s, h = strat.ledger["net"], hold.ledger["net"]
    pos = strat.ledger["position"]
    for year, idx in s.groupby(s.index.year).groups.items():
        yearly.append({"symbol": symbol, "year": year,
                       "strategy": (1 + s[idx]).prod() - 1, "hold": (1 + h[idx]).prod() - 1,
                       "exposure": pos[idx].mean()})
    # up/down capture on weekly returns, and beta/alpha on daily returns
    sw = (1 + s).resample("W").prod() - 1
    hw = (1 + h).resample("W").prod() - 1
    up, down = hw > 0, hw < 0
    beta, alpha = np.polyfit(h, s, 1)
    resid = s - beta * h
    overall.append({"symbol": symbol, "exposure": pos.mean(),
                    "up_capture": sw[up].mean() / hw[up].mean(),
                    "down_capture": sw[down].mean() / hw[down].mean(),
                    "beta": beta, "alpha_ann": alpha * 365,
                    "alpha_t": resid.mean() / resid.std() * np.sqrt(len(resid)),
                    "sharpe": s.mean() / s.std() * np.sqrt(365),
                    "hold_sharpe": h.mean() / h.std() * np.sqrt(365)})

yearly = pd.DataFrame(yearly)
print("BTC by year:")
print(yearly[yearly.symbol == "BTC"].round(3).to_string(index=False))
print("\nMedian across the 10 coins by year:")
print(yearly.groupby("year")[["strategy", "hold", "exposure"]].median().round(3).to_string())
print("\nBy coin:")
print(pd.DataFrame(overall).round(2).to_string(index=False))
print("\nMedians:", pd.DataFrame(overall).drop(columns="symbol").median().round(2).to_dict())
