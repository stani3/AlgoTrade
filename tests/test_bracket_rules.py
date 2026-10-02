"""State-machine rules of the bracket simulator: cooldowns, signal handling, kill switch."""

import pytest
from bracket_helpers import BASE, bar_numbers, directional, entry, flat, run

from algotrade.backtest.costs import CostModel
from algotrade.strategies import BreakoutBracket

DIRECTIONS = pytest.mark.parametrize("d", [1, -1], ids=["long", "short"])
LOSER = (0, 1, 6, 0)  # stopped out (stop 5)
WINNER = (0, 11, 1, 0)  # target hit (target 10)


def market(n: int, exit_row: tuple, exit_bar: int = 2) -> list:
    """Signal at 0, entry at 1, the given exit at ``exit_bar``, flat bars elsewhere."""
    rows = flat(n)
    rows[exit_bar] = exit_row
    return rows


def entry_bars(result, bars) -> list[int]:
    return list(bar_numbers(bars, result.trades["entry"]))


# --- cooldowns ---------------------------------------------------------------------------


@DIRECTIONS
@pytest.mark.parametrize(
    ("exit_row", "cooldown"), [(LOSER, 5), (WINNER, 20)], ids=["after_loss", "after_win"]
)
def test_cooldown_boundary(d, exit_row, cooldown) -> None:
    n = 30
    bars = directional(d, market(n, exit_row))
    rules = {"cooldown_loss": 5, "cooldown_win": 20}
    exit_bar = 2
    too_early = run(bars, **entry(d, 0, exit_bar + cooldown - 1), **rules)
    assert entry_bars(too_early, bars) == [1]
    on_time = run(bars, **entry(d, 0, exit_bar + cooldown), **rules)
    assert entry_bars(on_time, bars) == [1, exit_bar + cooldown + 1]


@DIRECTIONS
def test_every_signal_inside_the_cooldown_is_ignored(d) -> None:
    bars = directional(d, market(12, LOSER))
    result = run(bars, cooldown_loss=5, **entry(d, 0, 2, 3, 4, 5, 6, 7))
    assert entry_bars(result, bars) == [1, 8]  # signal at 7 = exit bar 2 + 5


@DIRECTIONS
def test_win_or_loss_is_judged_after_costs(d) -> None:
    # Target 0.1 on a 128 price earns 0.078% before costs; 5 bps fees each way cost ~0.1%.
    bars = directional(d, market(12, (0, 0.2, 0.0, 0)))
    costs = CostModel(fee_bps=5.0, slippage_bps=0.0, include_funding=False)
    result = run(bars, target=0.1, costs=costs, cooldown_win=20, cooldown_loss=5, **entry(d, 0, 7))
    first = result.trades.iloc[0]
    assert first["exit_reason"] == "target"
    assert d * (first["exit_price"] - first["entry_price"]) > 0  # won before costs
    assert first["pnl"] < 0  # lost after costs
    assert entry_bars(result, bars) == [1, 8]  # so the 5-bar loser cooldown applied


@DIRECTIONS
def test_breakeven_counts_as_a_loser(d) -> None:
    # A 1e-15 target rounds to the entry price itself (128 + 1e-15 == 128 in floating point),
    # so the trade exits on its entry bar for exactly zero P&L.
    bars = directional(d, flat(12))
    result = run(bars, target=1e-15, cooldown_win=20, cooldown_loss=5, **entry(d, 0, 6))
    first = result.trades.iloc[0]
    assert first["pnl"] == 0.0 and first["exit"] == bars.index[1]
    assert entry_bars(result, bars) == [1, 7]


@DIRECTIONS
def test_cooldown_of_one_ignores_a_signal_on_the_exit_bar(d) -> None:
    bars = directional(d, market(8, LOSER))
    result = run(bars, cooldown_loss=1, **entry(d, 0, 2, 3))
    assert entry_bars(result, bars) == [1, 4]


@DIRECTIONS
def test_cooldown_of_zero_allows_a_signal_on_the_exit_bar(d) -> None:
    bars = directional(d, market(8, LOSER))
    result = run(bars, cooldown_loss=0, **entry(d, 0, 2))
    assert entry_bars(result, bars) == [1, 3]


# --- signals while in a trade ---------------------------------------------------------------


@DIRECTIONS
def test_opposite_and_repeat_signals_are_ignored_in_a_trade(d) -> None:
    bars = directional(d, flat(8))
    same = list(range(6))
    opposite = list(range(1, 6))
    kwargs = {"long": same, "short": opposite} if d > 0 else {"short": same, "long": opposite}
    result = run(bars, **kwargs)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == d and trade["open"]


def test_long_wins_when_both_signals_fire_on_the_same_bar() -> None:
    bars = directional(1, flat(4))
    assert run(bars, long=[0], short=[0]).trades["direction"].tolist() == [1]


# --- kill switch -----------------------------------------------------------------------------


@DIRECTIONS
def test_kill_switch_trips_exactly_at_the_threshold(d) -> None:
    # Half the base price away: equity 1 -> 0.5, exactly 50% below the peak of 1.
    rows = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 64, -64), (-64, 0, 0, -64)] + flat(4)
    bars = directional(d, rows)
    result = run(bars, stop=100.0, target=100.0, kill_drawdown=0.5, **entry(d, 0, 5))
    assert result.meta["killed_at"] == bars.index[2]
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "kill"
    assert trade["exit"] == bars.index[3] and trade["exit_price"] == BASE - d * 64
    assert trade["bars"] == 2  # held through bars 1 and 2, sold at bar 3's open
    assert len(result.trades) == 1  # the signal at bar 5 is ignored for good
    assert result.ledger["equity"].iloc[-1] == pytest.approx(0.5)
    assert result.ledger["position"].iloc[3:].eq(0).all()


@DIRECTIONS
def test_kill_switch_does_not_trip_just_above_the_threshold(d) -> None:
    rows = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 63.9, -63.9), (-63.9, 0, 0, -63.9)]
    result = run(directional(d, rows), stop=100.0, target=100.0, **entry(d, 0))
    assert "killed_at" not in result.meta
    assert result.trades.iloc[0]["exit_reason"] == "open"


@DIRECTIONS
def test_kill_switch_measures_from_the_equity_peak(d) -> None:
    # Up to +64 (equity 1.5), then down to -32 (equity 0.75 = half the peak).
    rows = [(0, 0, 0, 0), (0, 64, 0, 64), (64, 64, 32, -32), (-32, 0, 0, -32)]
    result = run(directional(d, rows), stop=200.0, target=200.0, kill_drawdown=0.5, **entry(d, 0))
    assert result.trades.iloc[0]["exit_reason"] == "kill"
    assert result.ledger["equity"].max() == pytest.approx(1.5)


@DIRECTIONS
def test_kill_switch_can_trip_on_the_exit_bar_while_flat(d) -> None:
    rows = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 64, 0)] + flat(4)
    bars = directional(d, rows)
    result = run(
        bars, stop=64.0, target=100.0, cooldown_loss=0, kill_drawdown=0.5, **entry(d, 0, 3, 4)
    )
    assert result.meta["killed_at"] == bars.index[2]
    assert result.trades["exit_reason"].tolist() == ["stop"]  # no kill trade, no new trades


@DIRECTIONS
def test_kill_drawdown_of_one_disables_the_switch(d) -> None:
    rows = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 64, -64), (-64, 0, 0, -64)]
    result = run(directional(d, rows), stop=100.0, target=100.0, kill_drawdown=1.0, **entry(d, 0))
    assert "killed_at" not in result.meta
    assert result.trades.iloc[0]["exit_reason"] == "open"


# --- liquidation -----------------------------------------------------------------------------


@DIRECTIONS
def test_gap_through_stop_with_leverage_liquidates(d) -> None:
    # 3x leverage, gap of -48 on 128 (-37.5%) loses 112.5% of equity.
    rows = [(0, 0, 0, 0), (0, 0, 0, 0), (-48, 0, 50, -48)] + flat(3)
    bars = directional(d, rows)
    result = run(bars, leverage=3.0, kill_drawdown=1.0, cooldown_loss=0, **entry(d, 0, 3))
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "liquidated"
    assert trade["pnl"] == pytest.approx(-1.0)
    assert len(result.trades) == 1
    assert result.ledger["equity"].iloc[2:].eq(0).all()
    assert result.meta["killed_at"] == bars.index[2]


@DIRECTIONS
def test_mark_to_market_loss_beyond_equity_liquidates_at_the_close(d) -> None:
    rows = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 50, -50)] + flat(2)
    result = run(directional(d, rows), stop=100.0, target=100.0, leverage=3.0, **entry(d, 0))
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "liquidated"
    assert trade["pnl"] == pytest.approx(-1.0)
    assert result.ledger["equity"].min() == 0.0
    assert (result.ledger["equity"] >= 0).all()


# --- parameter validation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"lookback": 1},
        {"rsi_length": 0},
        {"atr_length": 0},
        {"rsi_level": 0},
        {"rsi_level": 100},
        {"stop_atr": 0},
        {"target_atr": -1},
        {"cooldown_win": -1},
        {"cooldown_loss": -1},
        {"kill_drawdown": 0},
        {"kill_drawdown": 1.5},
    ],
    ids=str,
)
def test_invalid_parameters_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        BreakoutBracket(**bad)
