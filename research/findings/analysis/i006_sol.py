"""Is i006's SOL result skill or luck? Dev data only; only settings the gate already ran."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.metrics import periods_per_year, sharpe_ratio
from algotrade.backtest.runner import backtest
from algotrade.research.cards import read_card
from algotrade.research.criteria import load_criteria
from algotrade.research.split import dev_universe
from algotrade.research.workspace import Workspace
from algotrade.strategies import from_spec
from algotrade.validation.monkey import monkey_test

ws = Workspace(Path.cwd())
criteria = load_criteria(ws.criteria_path)
costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
folder = ws.research / "ideas/i006-bollinger-squeeze-breakout/v1"
card = read_card(folder / "idea.md")
chosen = json.loads((folder / "feasibility/result.json").read_text())["chosen_spec"]
universe = dev_universe(ws, criteria, "1h")


def sharpe(result) -> float:
    return sharpe_ratio(result.returns, periods_per_year(result.ledger.index))


def closed(result) -> pd.DataFrame:
    t = result.trades[~result.trades["open"].astype(bool)].copy()
    t["year"] = pd.to_datetime(t["entry"]).dt.year
    return t


for label, spec in [("card 125/2/4", card.spec), ("chosen 250/3/6", chosen)]:
    print(f"\n=== {label} ===")
    rows = []
    for symbol, bars in universe.items():
        r = backtest(from_spec(spec), bars, costs)
        t = closed(r)
        by_year = t.groupby("year")["return"].sum()
        rows.append({"symbol": symbol, "sharpe": sharpe(r), "sum": t["return"].sum(),
                     "sum_ex_2021": t.loc[t["year"] != 2021, "return"].sum(),
                     "long": t.loc[t["direction"] > 0, "return"].sum(),
                     "short": t.loc[t["direction"] < 0, "return"].sum(), "trades": len(t)})
        if symbol == "SOL":
            sol_year = by_year
            sol_sides = t.groupby(["year", "direction"])["return"].sum().unstack()
            half = r.ledger.index[len(r.ledger) // 2]
            first = r.returns[r.returns.index < half]
            second = r.returns[r.returns.index >= half]
            ppy = periods_per_year(r.ledger.index)
            print(f"SOL first half Sharpe {sharpe_ratio(first, ppy):.2f}, second half "
                  f"{sharpe_ratio(second, ppy):.2f} (split at {half:%Y-%m-%d})")
    table = pd.DataFrame(rows).set_index("symbol").round(2)
    print(table.to_string())
    print("SOL fixed-size trade sum by year:", sol_year.round(2).to_dict())
    print("SOL by year and side (-1 short, 1 long):\n", sol_sides.round(2).to_string())

print("\n=== SOL across the whole pre-registered grid (27 cells the gate already ran) ===")
cells = []
for values in itertools.product(*card.optimise.values()):
    spec = dict(card.spec)
    for key, value in zip(card.optimise, values):
        spec[key.split(".")[-1]] = value
    r = backtest(from_spec(spec), universe["SOL"], costs)
    t = closed(r)
    cells.append({**dict(zip(card.optimise, values)), "sharpe": sharpe(r),
                  "ex_2021": t.loc[t["year"] != 2021, "return"].sum()})
cells = pd.DataFrame(cells)
print(cells.round(2).to_string(index=False))
print(f"SOL positive in {(cells['sharpe'] > 0).mean():.0%} of cells; "
      f"positive without 2021 in {(cells['ex_2021'] > 0).mean():.0%}")

print("\n=== Random entries on SOL alone (1000 monkeys, same trade count and exits) ===")
for label, spec in [("card", card.spec), ("chosen", chosen)]:
    m = monkey_test(from_spec(spec), {"SOL": universe["SOL"]}, costs, 1000, 0)
    print(f"{label}: strategy beats {m.percentile:.1%} of random-entry monkeys on SOL")

print("\n=== Luck check: best of 10 coins when every coin has the same true Sharpe ===")
sharpes = pd.DataFrame(rows)["sharpe"]
years = 4.6
rng = np.random.default_rng(0)
draws = rng.normal(sharpes.median(), 1 / np.sqrt(years), size=(100_000, 10)).max(axis=1)
print(f"cross-coin median {sharpes.median():.2f}, spread (sd) {sharpes.std():.2f}, "
      f"pure-noise sd {1 / np.sqrt(years):.2f}")
print(f"P(best of 10 >= SOL's {sharpes.max():.2f} by noise alone) = {(draws >= sharpes.max()).mean():.2f}")
