"""Fill rules of the bracket simulator, each checked for a long and the mirrored short."""

import numpy as np
import pytest
from bracket_helpers import BASE, ZERO, directional, entry, flat, run

from algotrade.backtest.costs import CostModel
from algotrade.backtest.metrics import summarize

SLIP = CostModel(fee_bps=0.0, slippage_bps=10.0, include_funding=False)
DIRECTIONS = pytest.mark.parametrize("d", [1, -1], ids=["long", "short"])


def only_trade(result):
    assert len(result.trades) == 1
    return result.trades.iloc[0]


@DIRECTIONS
def test_entry_fills_at_next_open_not_signal_close(d) -> None:
    bars = directional(d, [(0, 0, 0, 3), (2, 3, 1, 2), (2, 2, 2, 2)])
    trade = only_trade(run(bars, **entry(d, 0)))
    assert trade["entry_price"] == BASE + 2 * d  # bar 1 open, not bar 0's close of +3
    assert trade["entry"] == bars.index[1]


@DIRECTIONS
def test_entry_pays_slippage_against_the_trade(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (2, 3, 1, 2), (2, 2, 2, 2)])
    trade = only_trade(run(bars, costs=SLIP, **entry(d, 0)))
    assert trade["entry_price"] == pytest.approx((BASE + 2 * d) * (1 + d * 0.001))


@DIRECTIONS
def test_signal_on_last_bar_never_fills(d) -> None:
    bars = directional(d, flat(4))
    result = run(bars, costs=CostModel(fee_bps=5, slippage_bps=5), **entry(d, 3))
    assert result.trades.empty
    assert result.ledger["equity"].eq(1.0).all()
    assert result.ledger["trading_cost"].eq(0.0).all()


@DIRECTIONS
def test_levels_use_fill_price_and_signal_bar_distance(d) -> None:
    # Distances change after the signal bar; the trade must keep the signal-bar ones (5 / 10).
    bars = directional(d, [(0, 0, 0, 0), (1, 1, 1, 1), (1, 2, 4.5, 1), (1, 1, 4, 1)])
    stop = [5.0, 50.0, 50.0, 50.0]
    result = run(bars, stop=stop, target=[10.0, 1.0, 1.0, 1.0], **entry(d, 0))
    trade = only_trade(result)
    # Fill at +1 puts the stop at +1 - 5 = -4, which bar 2 (adverse 4.5) reaches. Measured from
    # the signal close (-5) or with the later distance (50) it would not trigger at all.
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == BASE + d * (1 - 5)
    assert trade["exit"] == bars.index[2]


@DIRECTIONS
def test_stop_fills_exactly_at_stop_minus_slippage(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 1, 1, 0), (0, 1, 6, -2), (0, 0, 0, 0)])
    trade = only_trade(run(bars, costs=SLIP, **entry(d, 0)))
    fill = BASE * (1 + d * 0.001)
    stop = fill - d * 5
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == pytest.approx(stop * (1 - d * 0.001))


@DIRECTIONS
def test_target_fills_exactly_at_target_without_slippage(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 1, 1, 0), (0, 11, 1, 9), (0, 0, 0, 0)])
    trade = only_trade(run(bars, costs=SLIP, **entry(d, 0)))
    fill = BASE * (1 + d * 0.001)
    assert trade["exit_reason"] == "target"
    assert trade["exit_price"] == pytest.approx(fill + d * 10)


@DIRECTIONS
def test_gap_through_stop_fills_at_the_worse_open(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 1, 1, 0), (-8, 0, 9, -8), (0, 0, 0, 0)])
    trade = only_trade(run(bars, **entry(d, 0)))
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == BASE - d * 8  # open, 3 points worse than the -5 stop


@DIRECTIONS
def test_gap_through_target_fills_at_the_better_open(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 1, 1, 0), (12, 13, 0, 12), (0, 0, 0, 0)])
    trade = only_trade(run(bars, **entry(d, 0)))
    assert trade["exit_reason"] == "target"
    assert trade["exit_price"] == BASE + d * 12


@DIRECTIONS
def test_stop_and_target_in_same_bar_assumes_stop(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 1, 1, 0), (0, 11, 6, 0), (0, 0, 0, 0)])
    trade = only_trade(run(bars, **entry(d, 0)))
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == BASE - d * 5


@DIRECTIONS
@pytest.mark.parametrize(
    ("row", "reason", "offset"),
    [((0, 1, 6, 0), "stop", -5), ((0, 11, 1, 0), "target", 10)],
    ids=["stop", "target"],
)
def test_exit_can_happen_on_the_entry_bar(d, row, reason, offset) -> None:
    bars = directional(d, [(0, 0, 0, 0), row, (0, 0, 0, 0)])
    trade = only_trade(run(bars, **entry(d, 0)))
    assert trade["exit_reason"] == reason
    assert trade["exit_price"] == BASE + d * offset
    assert trade["entry"] == trade["exit"] == bars.index[1]
    assert trade["bars"] == 1


@DIRECTIONS
@pytest.mark.parametrize(
    ("row", "reason"), [((0, 1, 5, 0), "stop"), ((0, 10, 1, 0), "target")], ids=["stop", "target"]
)
def test_touching_a_level_exactly_triggers_it(d, row, reason) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 1, 1, 0), row, (0, 0, 0, 0)])
    assert only_trade(run(bars, **entry(d, 0)))["exit_reason"] == reason


@DIRECTIONS
def test_coming_close_to_a_level_does_not_trigger_it(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 9.9, 4.9, 0), (0, 9.9, 4.9, 3)])
    trade = only_trade(run(bars, **entry(d, 0)))
    assert trade["exit_reason"] == "open"


@DIRECTIONS
def test_trade_open_at_end_is_marked_and_left_out_of_stats(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 1, 1, 0), (0, 3, 1, 2)])
    result = run(bars, **entry(d, 0))
    trade = only_trade(result)
    assert trade["open"] and trade["exit_reason"] == "open"
    assert trade["exit_price"] == BASE + d * 2
    assert trade["pnl"] == pytest.approx(2 / BASE)  # marked to the last close
    stats = summarize(result)
    assert stats["trades"] == 0
    assert result.ledger["equity"].iloc[-1] == pytest.approx(1 + 2 / BASE)


@DIRECTIONS
@pytest.mark.parametrize("missing", ["stop", "target"])
def test_missing_bracket_distance_means_no_trade(d, missing) -> None:
    bars = directional(d, flat(5))
    distances = {"stop": 5.0, "target": 10.0, missing: np.nan}
    assert run(bars, **distances, **entry(d, 0, 1, 2)).trades.empty


@DIRECTIONS
def test_position_column_tracks_exposure_while_held(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 1, 1, 0), (0, 1, 1, 0), (0, 1, 6, 0), (0, 0, 0, 0)])
    position = run(bars, **entry(d, 0)).ledger["position"]
    assert position.iloc[0] == 0
    assert position.iloc[1] == pytest.approx(d) and position.iloc[2] == pytest.approx(d)
    assert position.iloc[3] == 0 and position.iloc[4] == 0  # stopped out inside bar 3


def test_no_signals_means_untouched_equity() -> None:
    result = run(directional(1, flat(5)))
    assert result.trades.empty
    assert result.ledger["equity"].eq(1.0).all()
    assert summarize(result)["trades"] == 0
    assert run(directional(1, flat(5)), costs=ZERO).ledger["net"].eq(0).all()
