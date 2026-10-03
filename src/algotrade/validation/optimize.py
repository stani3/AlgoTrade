"""Limited optimisation: run the pre-registered grid, then pick the middle of a plateau.

Davey's limited optimisation asks two things of a parameter grid: most of it should make money
(robust rules are not one lucky cell), and the parameters to carry forward should sit in a broad
region of good results. The chosen cell is the one whose neighbourhood (itself and every cell
one grid step away in each parameter) has the best average median Sharpe, not the single best.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from algotrade.backtest.costs import CostModel, costs_for
from algotrade.backtest.engine import BacktestResult
from algotrade.backtest.metrics import aggregate, summarize
from algotrade.backtest.runner import backtest
from algotrade.parallel import Pool, run
from algotrade.research.dedup import grid_specs, spec_hash
from algotrade.strategies import from_spec

Costs = CostModel | Mapping[str, CostModel]


def evaluate(
    spec: dict, universe: dict[str, pd.DataFrame], costs: Costs, max_leverage: float = 1.0
) -> tuple[dict[str, BacktestResult], pd.DataFrame]:
    """Backtest one configuration on every symbol: results and per-symbol statistics."""

    strategy = from_spec(spec)
    results = {
        s: backtest(strategy, bars, costs_for(costs, s), max_leverage)
        for s, bars in universe.items()
    }
    return results, pd.DataFrame({s: summarize(r) for s, r in results.items()}).T


def grid_symbol(
    specs: list[dict], bars: pd.DataFrame, costs: CostModel, max_leverage: float = 1.0
) -> list[tuple[dict, bool]]:
    """Every grid configuration on one symbol: its statistics and whether it was killed."""

    out = []
    for spec in specs:
        result = backtest(from_spec(spec), bars, costs, max_leverage)
        out.append((summarize(result), "killed_at" in result.meta))
    return out


def _grid_job(args: tuple) -> list[tuple[dict, bool]]:
    return grid_symbol(*args)


def assemble_board(
    grid: dict[str, list], specs: list[dict], by_symbol: dict[str, list[tuple[dict, bool]]]
) -> pd.DataFrame:
    """The grid table from per-symbol results (``by_symbol`` in universe order)."""

    rows = []
    combos = list(itertools.product(*grid.values())) if grid else [()]
    for number, (combo, spec) in enumerate(zip(combos, specs, strict=True)):
        per_symbol = pd.DataFrame({s: cells[number][0] for s, cells in by_symbol.items()}).T
        killed = [cells[number][1] for cells in by_symbol.values()]
        rows.append(
            {
                **dict(zip(grid, combo, strict=True)),
                **aggregate(per_symbol),
                "median_trades": float(per_symbol["trades"].median()),
                "killed": float(np.mean(killed)),
                "spec_hash": spec_hash(spec),
                "spec": json.dumps(spec, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def run_grid(
    base: dict,
    grid: dict[str, list],
    universe: dict[str, pd.DataFrame],
    costs: Costs,
    max_leverage: float = 1.0,
    pool: Pool | None = None,
) -> pd.DataFrame:
    """One row per grid combination: its parameters, cross-symbol statistics and spec hash."""

    specs = grid_specs(base, grid)
    jobs = [(specs, bars, costs_for(costs, s), max_leverage) for s, bars in universe.items()]
    cells = run(_grid_job, jobs, pool)
    return assemble_board(grid, specs, dict(zip(universe, cells, strict=True)))


def profitable_share(board: pd.DataFrame) -> float:
    """Share of combinations whose median compounded growth across symbols is positive."""

    return float((board["median_cagr"] > 0).mean()) if len(board) else 0.0


def neighbourhood_scores(
    board: pd.DataFrame, keys: list[str], metric: str = "median_sharpe"
) -> pd.Series:
    """Mean ``metric`` over each cell and its neighbours one grid step away in every key."""

    if not keys:
        return board[metric].astype(float).copy()
    coords = np.column_stack(
        [board[key].map({v: i for i, v in enumerate(sorted(board[key].unique()))}) for key in keys]
    )
    values = board[metric].to_numpy(dtype=float)
    scores = [
        float(np.mean(values[np.all(np.abs(coords - coords[row]) <= 1, axis=1)]))
        for row in range(len(board))
    ]
    return pd.Series(scores, index=board.index, name=f"{metric}_plateau")


def choose(board: pd.DataFrame, keys: list[str], metric: str = "median_sharpe") -> pd.Series:
    """The row at the centre of the best plateau (ties go to the better raw score)."""

    scored = board.assign(plateau=neighbourhood_scores(board, keys, metric))
    best = scored.sort_values(["plateau", metric], ascending=False, kind="stable")
    return best.iloc[0]
