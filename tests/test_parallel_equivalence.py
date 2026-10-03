"""The per-symbol, parallel research code gives exactly the numbers of the serial code it
replaced (tests/legacy_serial.py), with any number of workers."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import random_bars
from legacy_serial import (
    legacy_entry_test,
    legacy_feasibility,
    legacy_monkey_test,
    legacy_run_grid,
    legacy_walk_forward,
)

from algotrade.backtest.costs import CostModel
from algotrade.parallel import Pool
from algotrade.research.criteria import Criteria, load_criteria
from algotrade.research.feasibility import run_feasibility
from algotrade.strategies import from_spec
from algotrade.validation.entry_test import run_entry_test
from algotrade.validation.monkey import monkey_test
from algotrade.validation.optimize import run_grid
from algotrade.validation.walkforward import walk_forward

COSTS = CostModel(fee_bps=5.0, slippage_bps=3.0)
OTHER = CostModel(fee_bps=1.0, slippage_bps=2.0)
CRITERIA = Path(__file__).resolve().parent.parent / "research" / "criteria.yaml"

POSITION = ({"type": "ma_crossover", "fast": 8, "slow": 40}, {"fast": [5, 10], "slow": [40, 60]})
BRACKET = (
    {"type": "breakout_bracket", "lookback": 24, "rsi_length": 14, "rsi_level": 50,
     "atr_length": 14, "stop_atr": 2.0, "target_atr": 4.0, "cooldown_win": 5,
     "cooldown_loss": 2, "kill_drawdown": 0.5},
    {"stop_atr": [1.5, 2.0], "target_atr": [3.0, 4.0]},
)  # fmt: skip
CASES = {"position": POSITION, "bracket": BRACKET}


def universe() -> dict[str, pd.DataFrame]:
    """Three markets of different lengths that start on different days."""

    markets = {}
    for number, (size, delay) in enumerate([(1800, 0), (1500, 80), (1700, 11)]):
        bars = random_bars(40 + number, n=size)
        bars.index = bars.index + pd.Timedelta(days=delay)
        markets[f"S{number}"] = bars
    return markets


@pytest.fixture(scope="module")
def pool():
    with Pool(2) as shared:
        yield shared


@pytest.fixture(scope="module")
def markets() -> dict[str, pd.DataFrame]:
    return universe()


def criteria(monkeys: int = 25) -> Criteria:
    values = copy.deepcopy(load_criteria(CRITERIA).values)
    values["feasibility"]["monkey_runs"] = monkeys
    return Criteria(values=values, hash="test")


@pytest.mark.parametrize("case", CASES)
def test_entry_test(case: str, markets, pool) -> None:
    strategy = from_spec(CASES[case][0])
    old = legacy_entry_test(strategy, markets, COSTS, [5, 10], 2.0, 4.0, 14, "fixed")
    new = run_entry_test(strategy, markets, COSTS, [5, 10], 2.0, 4.0, 14, "fixed")
    pd.testing.assert_frame_equal(new.table, old.table, check_exact=True)
    with pytest.raises(ValueError, match="scoring"):
        run_entry_test(strategy, markets, COSTS, [5], 2.0, 4.0, 14, "average")


@pytest.mark.parametrize("case", CASES)
def test_monkeys(case: str, markets, pool) -> None:
    strategy = from_spec(CASES[case][0])
    old = legacy_monkey_test(strategy, markets, COSTS, runs=30, seed=5)
    for used in (None, pool):
        new = monkey_test(strategy, markets, COSTS, runs=30, seed=5, pool=used)
        assert new.real == old.real
        np.testing.assert_array_equal(new.monkeys, old.monkeys)


@pytest.mark.parametrize("case", CASES)
def test_grid(case: str, markets, pool) -> None:
    spec, grid = CASES[case]
    old = legacy_run_grid(spec, grid, markets, COSTS)
    for used in (None, pool):
        pd.testing.assert_frame_equal(
            run_grid(spec, grid, markets, COSTS, pool=used), old, check_exact=True
        )
    pd.testing.assert_frame_equal(
        run_grid(spec, {}, markets, COSTS), legacy_run_grid(spec, {}, markets, COSTS)
    )


def test_costs_per_symbol(markets) -> None:
    spec, grid = POSITION
    mixed = {"S0": COSTS, "S1": OTHER, "S2": COSTS}
    board = run_grid(spec, grid, markets, mixed)
    alone = legacy_run_grid(spec, grid, {"S1": markets["S1"]}, OTHER)
    together = legacy_run_grid(spec, grid, markets, COSTS)
    assert not board["median_sharpe"].equals(together["median_sharpe"])
    assert run_grid(spec, grid, {"S1": markets["S1"]}, mixed).equals(alone)


@pytest.mark.parametrize("case", CASES)
def test_walk_forward(case: str, markets, pool) -> None:
    spec, grid = CASES[case]
    end = max(bars.index[-1] for bars in markets.values())
    old = legacy_walk_forward(spec, grid, markets, COSTS, end, 0.25, 1)
    for used in (None, pool):
        new = walk_forward(spec, grid, markets, COSTS, end, 0.25, 1, pool=used)
        pd.testing.assert_frame_equal(new.windows, old.windows, check_exact=True)
        assert list(new.results) == list(old.results)
        for symbol, result in old.results.items():
            pd.testing.assert_frame_equal(new.results[symbol].ledger, result.ledger)
            pd.testing.assert_frame_equal(new.results[symbol].trades, result.trades)
    assert len(old.windows) > 5
    assert old.windows["is_symbols"].min() < 3  # the late starters miss early windows


def test_walk_forward_skips_windows_without_in_sample_data(markets) -> None:
    spec, grid = POSITION
    end = max(bars.index[-1] for bars in markets.values())
    old = legacy_walk_forward(spec, grid, markets, COSTS, end, 0.25, 1, min_in_sample_share=2.0)
    new = walk_forward(spec, grid, markets, COSTS, end, 0.25, 1, min_in_sample_share=2.0)
    assert new.windows.empty and old.windows.empty
    assert new.results == old.results == {}


@pytest.mark.parametrize("case", CASES)
def test_full_feasibility(case: str, markets) -> None:
    spec, grid = CASES[case]
    rules = criteria()
    old = legacy_feasibility(spec, grid, markets, COSTS, rules)
    for workers in (1, 2):
        new = run_feasibility(spec, grid, markets, COSTS, rules, report=workers == 2,
                              workers=workers)  # fmt: skip
        assert [c.to_dict() for c in new.checks] == old["checks"]
        assert new.metrics == old["metrics"]
        assert new.notes == old["notes"]
        assert new.chosen_params == old["chosen"]
        assert new.spec == old["chosen_spec"]
        assert new.csvs == old["csvs"]
        assert len(new.chosen.sections) == (len(markets) if workers == 2 else 0)
