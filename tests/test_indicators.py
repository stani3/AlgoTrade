import numpy as np
import pandas as pd
import pytest

from algotrade.indicators import atr, donchian, entry_exit_state, rsi, supertrend
from algotrade.strategies import DonchianBreakout


def series(values) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def test_rsi_extremes() -> None:
    assert rsi(series(range(1, 40)), 14).iloc[-1] == 100.0
    assert rsi(series(range(40, 1, -1)), 14).iloc[-1] == pytest.approx(0.0)


def test_atr_of_constant_range() -> None:
    close = series([100.0] * 30)
    assert atr(close + 1, close - 1, close, 14).iloc[-1] == pytest.approx(2.0)


def test_supertrend_flips_with_the_trend() -> None:
    close = series(list(np.linspace(100, 150, 40)) + list(np.linspace(150, 90, 40)))
    st = supertrend(close + 1, close - 1, close, length=10, mult=3.0)
    assert st["direction"].iloc[35] == 1.0
    assert st["direction"].iloc[-1] == -1.0


def test_donchian_breakout_needs_a_new_high_versus_prior_bars() -> None:
    close = series([10, 11, 12, 11, 10, 11, 13, 12, 12, 12])
    bars = pd.DataFrame({"high": close, "low": close, "close": close})
    channel = donchian(bars["high"], bars["low"], 3).shift()
    target = DonchianBreakout(entry=3, exit=2, allow_short=False).target_position(bars)
    # bar 6 closes at 13, above the prior 3-bar high of 11 -> long; stays long until the
    # close drops below the prior 2-bar low.
    assert channel["upper"].iloc[6] == 11
    assert target.iloc[5] == 0 and target.iloc[6] == 1.0 and target.iloc[-1] == 1.0


def test_entry_exit_state_machine() -> None:
    el = np.array([1, 0, 0, 0, 0, 0, 0], dtype=bool)
    xl = np.array([0, 0, 1, 0, 0, 0, 0], dtype=bool)
    es = np.array([0, 0, 0, 1, 0, 0, 0], dtype=bool)
    xs = np.array([0, 0, 0, 0, 0, 1, 0], dtype=bool)
    assert entry_exit_state(el, xl, es, xs).tolist() == [1, 1, 0, -1, -1, 0, 0]


def test_opposite_entry_reverses_in_one_bar() -> None:
    el = np.array([1, 0, 0], dtype=bool)
    es = np.array([0, 1, 0], dtype=bool)
    none = np.zeros(3, dtype=bool)
    assert entry_exit_state(el, none, es, none).tolist() == [1, -1, -1]
