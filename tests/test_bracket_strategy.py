"""BreakoutBracket signals, causality and integration with the rest of the stack."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import bars_from, random_bars

from algotrade.backtest.bracket import EXIT_REASONS
from algotrade.backtest.costs import CostModel
from algotrade.backtest.metrics import summarize
from algotrade.backtest.runner import backtest
from algotrade.config import load_settings
from algotrade.data.exchange import MarketId, ohlcv_path
from algotrade.data.market import load_market
from algotrade.strategies import BreakoutBracket, Combine, VolTarget, from_spec, to_spec

SPEC = Path(__file__).parent.parent / "specs" / "breakout_bracket.json"


def closes(values) -> pd.DataFrame:
    return bars_from([(c, c + 0.5, c - 0.5, c) for c in values])


SMALL = {"lookback": 3, "rsi_length": 5, "atr_length": 3}


def test_long_signal_on_highest_close_with_rsi_above_level() -> None:
    bars = closes([10, 11, 12, 13, 14, 15, 16, 17])
    sig = BreakoutBracket(**SMALL).signals(bars)
    # RSI(5) needs 5 changes, so the first signal is bar 5; every later bar is a new high.
    assert sig.long.tolist() == [False] * 5 + [True] * 3
    assert not sig.short.any()


def test_short_signal_is_the_mirror_image() -> None:
    bars = closes([17, 16, 15, 14, 13, 12, 11, 10])
    sig = BreakoutBracket(**SMALL).signals(bars)
    assert sig.short.tolist() == [False] * 5 + [True] * 3
    assert not sig.long.any()


def test_new_high_with_weak_rsi_is_not_a_signal() -> None:
    falling = list(np.linspace(100, 60, 30))
    bars = closes([*falling, 61, 62, 63])  # 3-bar high closes, but RSI(14) still below 50
    sig = BreakoutBracket(lookback=3, rsi_length=14, atr_length=3).signals(bars)
    close = bars["close"]
    assert close.iloc[-1] == close.iloc[-3:].max()
    assert not sig.long.iloc[-3:].any()


def test_strong_rsi_without_a_new_high_is_not_a_signal() -> None:
    bars = closes([10, 11, 12, 13, 14, 15, 16, 15.5])  # last close below the 3-bar high
    sig = BreakoutBracket(**SMALL).signals(bars)
    assert not sig.long.iloc[-1]


def test_a_tie_with_the_highest_close_counts_as_a_new_high() -> None:
    bars = closes([10, 11, 12, 13, 14, 15, 16, 16])
    assert BreakoutBracket(**SMALL).signals(bars).long.iloc[-1]


def test_rsi_level_is_a_strict_threshold() -> None:
    bars = closes([10, 11, 12, 13, 14, 15, 16, 17])
    strong = BreakoutBracket(**SMALL, rsi_level=99.0).signals(bars)
    assert strong.long.iloc[-1]  # all gains -> RSI is 100, above 99
    flat = closes([10.0] * 8)  # no movement: RSI 100 by convention, but ties both ways
    sig = BreakoutBracket(**SMALL).signals(flat)
    assert sig.long.iloc[-1] and not sig.short.iloc[-1]


def test_no_signals_during_warm_up(bars) -> None:
    sig = BreakoutBracket().signals(bars)
    assert not sig.long.iloc[:47].any() and not sig.short.iloc[:47].any()


def test_bracket_distances_are_atr_multiples(bars) -> None:
    strategy = BreakoutBracket(stop_atr=1.5, target_atr=4.5)
    sig = strategy.signals(bars)
    valid = sig.stop_dist.notna()
    np.testing.assert_allclose(sig.target_dist[valid], 3 * sig.stop_dist[valid])
    assert (sig.stop_dist[valid] > 0).all()


def test_allow_short_false_never_shorts() -> None:
    bars = random_bars(3, n=600)
    strategy = BreakoutBracket(**SMALL, allow_short=False)
    assert not strategy.signals(bars).short.any()
    trades = strategy.simulate(bars).trades
    assert len(trades) > 0 and (trades["direction"] == 1).all()


@pytest.mark.parametrize("seed", range(5))
def test_no_lookahead(seed) -> None:
    """Appending future bars must not change anything already decided."""
    bars = random_bars(seed, n=500)
    strategy = BreakoutBracket(lookback=10, rsi_length=10, atr_length=10, kill_drawdown=0.5)
    costs = CostModel(fee_bps=5.0, slippage_bps=3.0)
    full = strategy.simulate(bars, costs)
    full_sig = strategy.signals(bars)
    for cut in (120, 250, 377):
        part = strategy.simulate(bars.iloc[:cut], costs)
        pd.testing.assert_frame_equal(part.ledger, full.ledger.iloc[:cut])
        part_sig = strategy.signals(bars.iloc[:cut])
        pd.testing.assert_series_equal(part_sig.long, full_sig.long.iloc[:cut])
        pd.testing.assert_series_equal(part_sig.stop_dist, full_sig.stop_dist.iloc[:cut])
        closed = part.trades[~part.trades["open"]]
        pd.testing.assert_frame_equal(closed, full.trades.iloc[: len(closed)])


def test_spec_round_trip_and_file() -> None:
    strategy = BreakoutBracket(stop_atr=1.5, target_atr=6.0, atr_length=30)
    assert from_spec(to_spec(strategy)) == strategy
    book = from_spec(SPEC)
    assert book == BreakoutBracket()  # the spec file holds the book defaults


def test_runner_dispatches_to_the_intrabar_simulator() -> None:
    bars = random_bars(1)
    strategy = BreakoutBracket(**SMALL)
    costs = CostModel()
    via_runner = backtest(strategy, bars, costs, max_leverage=2.0)
    direct = strategy.simulate(bars, costs, leverage=2.0)
    pd.testing.assert_frame_equal(via_runner.ledger, direct.ledger)
    assert "exit_reason" in via_runner.trades


def test_cannot_be_used_as_a_target_position_strategy(bars) -> None:
    with pytest.raises(TypeError, match="intrabar"):
        BreakoutBracket().target_position(bars)
    with pytest.raises(TypeError, match="intrabar"):
        VolTarget(BreakoutBracket()).target_position(bars)
    with pytest.raises(TypeError, match="intrabar"):
        Combine((BreakoutBracket(),)).target_position(bars)


REAL = ohlcv_path(load_settings().data_paths.raw, MarketId("binanceusdm", "BTC"), "4h")


@pytest.mark.skipif(not REAL.exists(), reason="BTC 4h data not downloaded")
def test_smoke_on_real_btc_data() -> None:
    bars = load_market(load_settings().data_paths.raw, "binanceusdm", "BTC", "4h")
    result = BreakoutBracket().simulate(bars, CostModel())
    assert len(result.trades) > 10
    assert set(result.trades["exit_reason"]) <= set(EXIT_REASONS)
    stats = summarize(result)
    assert all(math.isfinite(v) for k, v in stats.items() if k != "profit_factor")
    assert stats["max_drawdown"] >= -1.0
