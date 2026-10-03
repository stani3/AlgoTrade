"""Davey's limited testing, one symbol per job, so symbols can run in parallel.

:func:`run_symbol` does everything feasibility needs from one symbol before the parameters are
chosen: the entry test's cells, the core system at the card's parameters, the monkeys and every
grid configuration. Once the parent has picked the parameters from the whole universe,
:func:`chosen_symbol` backtests them and, if asked, builds that symbol's performance report and
its HTML section (the charts are the slowest part of a report).

Both return exactly what the serial code computed; :func:`run_limited` puts the pieces back
together in universe order.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from algotrade.backtest.costs import CostModel, cap_for, costs_for
from algotrade.backtest.engine import BacktestResult, run_backtest
from algotrade.backtest.metrics import summarize
from algotrade.backtest.report import PerformanceReport, build_report
from algotrade.backtest.report_html import symbol_section
from algotrade.backtest.runner import backtest
from algotrade.parallel import Pool, run
from algotrade.research.dedup import grid_specs
from algotrade.strategies import BracketStrategy, from_spec

from .entry_test import EntryTest, check_scoring, entry_rows, entry_signals
from .monkey import MonkeyDraws, MonkeyTest, combine, monkey_symbol, plan_draws
from .optimize import assemble_board, grid_symbol


@dataclass(frozen=True)
class EntrySettings:
    exit_bars: list[int]
    stop_atr: float
    target_atr: float
    atr_length: int
    scoring: str


@dataclass(frozen=True)
class SymbolTask:
    symbol: str
    bars: pd.DataFrame
    costs: CostModel
    spec: dict
    grid_specs: list[dict]
    entry: EntrySettings
    monkey_runs: int
    draws: MonkeyDraws
    max_leverage: float = 1.0


@dataclass
class SymbolOutcome:
    entry_rows: list[dict]
    core: dict
    monkey: tuple[float, np.ndarray]
    grid: list[tuple[dict, bool]]


def _same(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def run_symbol(task: SymbolTask) -> SymbolOutcome:
    strategy = from_spec(task.spec)
    signals = strategy.signals(task.bars) if isinstance(strategy, BracketStrategy) else None
    if signals is not None:
        sides = (signals.long.fillna(False).astype(bool), signals.short.fillna(False).astype(bool))
    else:
        sides = entry_signals(strategy, task.bars)
    e = task.entry
    rows = entry_rows(
        strategy, task.symbol, task.bars, task.costs, e.exit_bars, e.stop_atr, e.target_atr,
        e.atr_length, e.scoring, sides,
    )  # fmt: skip
    grid = grid_symbol(task.grid_specs, task.bars, task.costs, task.max_leverage)
    # The card's own parameters are often a grid cell: same spec, same backtest.
    same = [n for n, spec in enumerate(task.grid_specs) if _same(spec, task.spec)]
    if same:
        core = grid[same[0]][0]
    else:
        core = summarize(backtest(strategy, task.bars, task.costs, task.max_leverage))
    monkey = monkey_symbol(
        strategy, task.bars, task.costs, task.monkey_runs, task.draws, signals, task.max_leverage
    )
    return SymbolOutcome(entry_rows=rows, core=core, monkey=monkey, grid=grid)


@dataclass(frozen=True)
class ChosenTask:
    symbol: str
    bars: pd.DataFrame
    costs: CostModel
    spec: dict
    report: bool
    seed: int
    anchor: str
    is_open: bool
    max_leverage: float = 1.0


def chosen_symbol(task: ChosenTask) -> tuple[BacktestResult, PerformanceReport | None, str | None]:
    result = backtest(from_spec(task.spec), task.bars, task.costs, task.max_leverage)
    if not task.report:
        return result, None, None
    hold = run_backtest(task.bars, pd.Series(1.0, index=task.bars.index), task.costs)
    report = build_report(result, benchmark=hold, seed=task.seed)
    return result, report, symbol_section(report, task.anchor, task.is_open)


@dataclass
class LimitedTest:
    entry: EntryTest
    per_symbol: pd.DataFrame  # the core system at the card's parameters
    monkeys: MonkeyTest
    board: pd.DataFrame


@dataclass
class ChosenRun:
    results: dict[str, BacktestResult]
    per_symbol: pd.DataFrame
    reports: list[PerformanceReport] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)


def run_limited(
    spec: dict,
    grid: dict[str, list],
    universe: dict[str, pd.DataFrame],
    costs: CostModel | Mapping[str, CostModel],
    entry: EntrySettings,
    monkey_runs: int,
    seed: int = 0,
    pool: Pool | None = None,
    max_leverage: float | Mapping[str, float] = 1.0,
) -> LimitedTest:
    """Entry test, core system, monkey test and grid over the universe; ``max_leverage`` caps
    each symbol's exposure (bracket strategies: their position size)."""

    check_scoring(entry.scoring)
    bracket = isinstance(from_spec(spec), BracketStrategy)
    draws = plan_draws(bracket, [len(bars) for bars in universe.values()], monkey_runs, seed)
    specs = grid_specs(spec, grid)
    tasks = [
        SymbolTask(symbol, bars, costs_for(costs, symbol), spec, specs, entry, monkey_runs, plan,
                   cap_for(max_leverage, symbol))
        for (symbol, bars), plan in zip(universe.items(), draws, strict=True)
    ]  # fmt: skip
    outcomes = dict(zip(universe, run(run_symbol, tasks, pool), strict=True))
    return LimitedTest(
        entry=EntryTest(pd.DataFrame([row for o in outcomes.values() for row in o.entry_rows])),
        per_symbol=pd.DataFrame({s: o.core for s, o in outcomes.items()}).T,
        monkeys=combine([o.monkey for o in outcomes.values()], monkey_runs),
        board=assemble_board(grid, specs, {s: o.grid for s, o in outcomes.items()}),
    )


def run_chosen(
    spec: dict,
    universe: dict[str, pd.DataFrame],
    costs: CostModel | Mapping[str, CostModel],
    report: bool = True,
    seed: int = 0,
    pool: Pool | None = None,
    max_leverage: float | Mapping[str, float] = 1.0,
) -> ChosenRun:
    """The chosen parameters on every symbol, with the performance reports if ``report``."""

    tasks = [
        ChosenTask(symbol, bars, costs_for(costs, symbol), spec, report, seed, f"s{n}", n == 0,
                   cap_for(max_leverage, symbol))
        for n, (symbol, bars) in enumerate(universe.items())
    ]  # fmt: skip
    found = run(chosen_symbol, tasks, pool)
    results = {symbol: result for symbol, (result, _, _) in zip(universe, found, strict=True)}
    return ChosenRun(
        results=results,
        per_symbol=pd.DataFrame({s: summarize(r) for s, r in results.items()}).T,
        reports=[r for _, r, _ in found if r is not None],
        sections=[html for _, _, html in found if html is not None],
    )
