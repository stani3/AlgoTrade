import math

import numpy as np
import pandas as pd
import pytest

from algotrade.backtest.engine import BacktestResult
from algotrade.backtest.report import DAVEY_CRITERIA, monte_carlo
from algotrade.validation.bands import block_bootstrap
from algotrade.validation.overfitting import (
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    probabilistic_sharpe,
)
from algotrade.validation.sizing import choose_stake, closed_trade_returns, trades_per_year

# --- deflated Sharpe ratio -------------------------------------------------------------------


def test_deflated_sharpe_matches_the_published_example() -> None:
    """Bailey & Lopez de Prado (2014): SR 2.5 a year over 5 years of daily data (T = 1250),
    skew -3, kurtosis 10, 100 trials whose annualised Sharpe ratios have variance 0.5."""

    variance = 0.5 / 250
    assert expected_max_sharpe(100, variance) == pytest.approx(0.1132, abs=1e-4)
    dsr = deflated_sharpe(2.5 / math.sqrt(250), 1250, -3, 10, 100, variance)
    assert dsr == pytest.approx(0.9004, abs=1e-4)


def test_more_trials_raise_the_bar() -> None:
    sr = 1.0 / math.sqrt(365)
    few = deflated_sharpe(sr, 2000, 0, 3, 10, 0.25 / 365)
    many = deflated_sharpe(sr, 2000, 0, 3, 1000, 0.25 / 365)
    assert few > many
    assert expected_max_sharpe(1, 0.01) == 0.0 and expected_max_sharpe(50, 0.0) == 0.0


def test_probabilistic_sharpe_edge_cases() -> None:
    assert probabilistic_sharpe(0.1, 0.1, 500, 0, 3) == pytest.approx(0.5)
    assert probabilistic_sharpe(0.1, 0.0, 1, 0, 3) == 0.0
    # Extreme negative skew with a large Sharpe makes the variance term negative.
    assert probabilistic_sharpe(2.0, 0.0, 500, 5.0, 3) == 0.0


def test_moments() -> None:
    rng = np.random.default_rng(0)
    normal = rng.normal(0.001, 0.01, 200_000)
    sharpe, count, skew, kurtosis = moments(normal)
    assert count == 200_000 and sharpe == pytest.approx(0.1, abs=0.01)
    assert skew == pytest.approx(0.0, abs=0.02) and kurtosis == pytest.approx(3.0, abs=0.05)
    assert moments(np.full(10, 0.01)) == (0.0, 10, 0.0, 3.0)
    assert moments(np.array([0.5])) == (0.0, 1, 0.0, 3.0)
    lopsided = moments(np.array([0.0] * 9 + [1.0]))
    assert lopsided[2] > 2


# --- Monte Carlo sizing --------------------------------------------------------------------------


LIMITS = {"risk_of_ruin": 0.10, "median_max_dd": 0.40, "return_dd": 2.0}


def test_stake_is_the_largest_size_meeting_daveys_goals() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.01, 0.04, 400)
    sizing = choose_stake(returns, 40, [0.25, 0.5, 1.0, 2.0, 4.0], 800, 0.5, LIMITS)
    passing = [s for s in sizing.table.index if sizing.passing(s)]
    assert sizing.stake == max(passing)
    assert sizing.stake < 4.0 and not sizing.passing(4.0)
    assert sizing.limits == LIMITS


def test_stake_never_goes_past_the_peak_median_return() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(0.01, 0.1, 400)  # growth-optimal size about mean / variance = 1
    anything = {"risk_of_ruin": 1.01, "median_max_dd": 1.01, "return_dd": -1e9}
    sizing = choose_stake(returns, 40, [0.5, 1.0, 1.5, 2.0, 3.0], 800, 0.99, anything)
    peak = sizing.table["median_return"].idxmax()
    assert peak < 3.0 and sizing.passing(3.0)  # the limits would allow more size
    assert sizing.stake == peak


def test_no_stake_when_nothing_passes() -> None:
    losers = np.array([-0.05, -0.04, 0.01, -0.03] * 10)
    sizing = choose_stake(losers, 30, [0.5, 1.0], 300, 0.5, LIMITS)
    assert sizing.stake is None


def test_monte_carlo_checks_with_custom_limits_and_size() -> None:
    mc = monte_carlo(np.array([0.02, -0.01] * 20), 30, runs=300, multipliers=(0.5, 2.0))
    default = mc.checks()
    assert [name for name, *_ in default] == [
        "Risk of ruin",
        "Median max drawdown",
        "Return / drawdown",
    ]
    strict = mc.checks({**DAVEY_CRITERIA, "return_dd": 1e9}, size=2.0)
    assert strict[2][3] is False and strict[2][1] == mc.table.loc[2.0, "return_dd"]


def result_with(trades: pd.DataFrame, days: int) -> BacktestResult:
    index = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
    ledger = pd.DataFrame({"net": 0.0, "equity": 1.0, "position": 0.0}, index=index)
    return BacktestResult(ledger=ledger, trades=trades, costs=None)


def test_trade_pool_and_rate() -> None:
    one = pd.DataFrame({"return": [0.1, -0.05, 0.2], "open": [False, False, True]})
    two = pd.DataFrame({"return": [0.03] * 20, "open": [False] * 20})
    results = {"A": result_with(one, 366), "B": result_with(two, 731)}
    np.testing.assert_allclose(closed_trade_returns(results), [0.1, -0.05] + [0.03] * 20)
    # A: 2 closed trades in a year, B: 20 in two years -> median of 2 and 10 is 6.
    assert trades_per_year(results) == 6
    assert trades_per_year({}) == 1
    assert closed_trade_returns({}).size == 0


# --- bootstrap bands -------------------------------------------------------------------------------


def test_block_bootstrap_paths() -> None:
    rising = np.full(500, 0.01)
    bands = block_bootstrap([rising], length=10, runs=50, block=5)
    np.testing.assert_allclose(bands.final_returns, 1.01**10 - 1)
    assert (bands.max_drawdowns == 0).all()
    assert bands.drawdown(0.95) == 0.0 and bands.total_return(0.05) == pytest.approx(1.01**10 - 1)


def test_block_bootstrap_keeps_blocks_and_varies() -> None:
    rng = np.random.default_rng(5)
    noisy = [rng.normal(0, 0.02, 300), rng.normal(0, 0.03, 400)]
    bands = block_bootstrap(noisy, length=120, runs=400, block=10, seed=2)
    assert len(bands.max_drawdowns) == 400 and bands.max_drawdowns.std() > 0
    assert bands.drawdown(0.95) > bands.drawdown(0.5)
    again = block_bootstrap(noisy, length=120, runs=400, block=10, seed=2)
    np.testing.assert_array_equal(bands.max_drawdowns, again.max_drawdowns)


def test_block_bootstrap_needs_data() -> None:
    with pytest.raises(ValueError, match="at least one series"):
        block_bootstrap([np.zeros(3)], length=10, runs=5, block=5)
    with pytest.raises(ValueError):
        block_bootstrap([np.zeros(30)], length=0, runs=5, block=5)
