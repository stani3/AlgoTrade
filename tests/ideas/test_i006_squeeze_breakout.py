"""i006_squeeze_breakout: the first closes outside the Bollinger bands after a squeeze.

Hand-computed cases are 1h markets written as close offsets from 100: each bar opens at the
previous close and its high and low sit one point beyond its body unless a case moves them. Every
case runs as written (``d = +1``) and as its mirror image (``d = -1``: prices reflected about
100), where the rule must take the opposite side on the same bars. Bands are 20 bars at 2
standard deviations unless a case says otherwise.

The step market carries most cases: closes flat at 0, then 5 from bar ``s`` on.

* A flat window has a bandwidth of exactly 0, so every flat bar whose rolling minimum is
  defined is a squeeze bar (a tie at 0).
* With ``k`` of the 20 closes at 5 (``p = k / 20``) the window's mean is ``5p`` and its
  standard deviation ``5 sqrt(p (1 - p))``, so the close of 5 lies ``5 (1 - p - 2 sqrt(p (1 -
  p)))`` above the upper band: 2.57 for k = 1, 1.5 for k = 2, 0.68 for k = 3, exactly 0 for
  k = 4 (on the band) and below it from k = 5. Armed, a step at bar s signals on s, s + 1, s + 2.
* The bandwidth ``20 sqrt(p (1 - p)) / (100 + 5pd)`` rises up to k = 10, falls back and is 0
  again from k = 20.
* ATR: every flat bar has a true range of 2; the step bar's is 7 (high 6, low -1).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import ZERO, random_bars

from algotrade.backtest.bracket import EXIT_REASONS, BracketSignals, simulate_bracket
from algotrade.backtest.costs import EXCHANGE_COSTS, ZERO_COSTS
from algotrade.backtest.runner import backtest, load_bars
from algotrade.research.cards import read_card
from algotrade.research.criteria import load_criteria
from algotrade.research.workspace import Workspace
from algotrade.strategies import Combine, VolTarget, from_spec, to_spec
from algotrade.strategies.ideas.i006_squeeze_breakout import SqueezeBreakout

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "research" / "ideas" / "i006-bollinger-squeeze-breakout" / "v1" / "idea.md"
SIDES = pytest.mark.parametrize("d", [1, -1], ids=["long", "short"])
CARD_SPEC = {
    "type": "i006_squeeze_breakout",
    "length": 20,
    "mult": 2.0,
    "squeeze_bars": 125,
    "armed_bars": 20,
    "atr_length": 14,
    "stop_atr": 2.0,
    "target_atr": 4.0,
    "max_bars": 48,
    "cooldown_win": 0,
    "cooldown_loss": 0,
    "kill_drawdown": 1.0,
    "allow_short": True,
}
GRID = [
    {"squeeze_bars": n, "stop_atr": s, "target_atr": t}
    for n in (60, 125, 250)
    for s in (1.5, 2.0, 3.0)
    for t in (3.0, 4.0, 6.0)
]

BASE = 100.0
START = pd.Timestamp("2024-01-01", tz="UTC")


def market(
    d: int,
    closes: list[float],
    tops: dict[int, float] | None = None,
    bottoms: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Bars from close offsets about ``BASE``, reflected for ``d = -1``.

    ``tops`` and ``bottoms`` move a bar's extremes in the trade's favour and against it (as
    offsets); by default they are one point beyond the bar's body.
    """
    offsets = np.asarray(closes, dtype="float64")
    before = np.concatenate((offsets[:1], offsets[:-1]))
    top = np.maximum(before, offsets) + 1.0
    bottom = np.minimum(before, offsets) - 1.0
    for bar, value in (tops or {}).items():
        top[bar] = value
    for bar, value in (bottoms or {}).items():
        bottom[bar] = value
    favourable, adverse = BASE + d * top, BASE + d * bottom
    return pd.DataFrame(
        {
            "open": BASE + d * before,
            "high": np.maximum(favourable, adverse),
            "low": np.minimum(favourable, adverse),
            "close": BASE + d * offsets,
            "volume": 1.0,
            "funding_rate": 0.0,
        },
        index=pd.date_range(START, periods=len(offsets), freq="1h"),
    )


def step(at: int, n: int, jump: float = 5.0) -> list[float]:
    """Closes flat at 0, then ``jump`` from bar ``at`` on."""
    return [0.0] * at + [jump] * (n - at)


def alternating(amplitude: float, n: int) -> list[float]:
    """+amplitude, -amplitude, ... : every full 20-bar window has the same bandwidth."""
    return [amplitude if i % 2 == 0 else -amplitude for i in range(n)]


def positions(flags: pd.Series) -> list[int]:
    return np.flatnonzero(flags.to_numpy(dtype=bool)).tolist()


def signal_bars(bars: pd.DataFrame, **params) -> tuple[list[int], list[int]]:
    """(long signal bars, short signal bars)."""
    signals = SqueezeBreakout(**params).signals(bars)
    for flags in (signals.long, signals.short):
        assert flags.index.equals(bars.index) and flags.dtype == bool
    return positions(signals.long), positions(signals.short)


def expect(d: int, bars_at: list[int]) -> tuple[list[int], list[int]]:
    return (bars_at, []) if d > 0 else ([], bars_at)


def trade_bars(bars: pd.DataFrame, trades: pd.DataFrame) -> list[tuple[int, int]]:
    entries = bars.index.get_indexer(pd.DatetimeIndex(trades["entry"]))
    exits = bars.index.get_indexer(pd.DatetimeIndex(trades["exit"]))
    return list(zip(entries.tolist(), exits.tolist(), strict=True))


# --- bands, squeeze and arming, worked by hand ----------------------------------------------


@SIDES
def test_the_setup_of_a_step_worked_by_hand(d) -> None:
    bars = market(d, step(30, 60))
    state = SqueezeBreakout(squeeze_bars=2, atr_length=3).setup(bars)
    assert state.index.equals(bars.index)
    # Bands need 20 closes; the minimum of two bandwidths needs bars 19 and 20.
    for column in ("upper", "lower", "bandwidth"):
        assert state[column].isna().tolist() == [True] * 19 + [False] * 41
    assert state["lowest"].isna().tolist() == [True] * 20 + [False] * 40
    assert state["atr"].isna().tolist() == [True] * 2 + [False] * 58
    width = state["bandwidth"].to_numpy()
    assert (width[19:30] == 0).all() and (width[49:] == 0).all()
    for k in range(1, 20):  # bar 29 + k has k of its 20 closes at 5
        p = k / 20
        assert width[29 + k] == pytest.approx(20 * math.sqrt(p * (1 - p)) / (BASE + 5 * p * d))
    assert (np.diff(width[29:40]) > 0).all() and (np.diff(width[39:50]) < 0).all()
    # Squeeze bars: the flat bars from 20 (ties at 0), then the falling bandwidth (bars 40-48,
    # each below the bar before) and the new flat stretch (49 on).
    assert positions(state["squeeze"]) == list(range(20, 30)) + list(range(40, 60))
    assert positions(state["armed"]) == list(range(20, 60))
    atr = state["atr"].to_numpy()
    assert atr[29] == pytest.approx(2.0) and atr[30] == pytest.approx(2.0 + (7.0 - 2.0) / 3)
    assert signal_bars(bars, squeeze_bars=2, atr_length=3) == expect(d, [30, 31, 32])


@SIDES
def test_the_close_on_the_band_is_not_a_signal(d) -> None:
    # k = 4 at bar 33: the close is exactly on the band (p = 1/5: 1 - p = 2 sqrt(p (1 - p))).
    bars = market(d, step(30, 60))
    state = SqueezeBreakout(squeeze_bars=5).setup(bars)
    band = state["upper"] if d > 0 else state["lower"]
    close = bars["close"]
    assert band.iloc[33] == close.iloc[33]
    assert (d * (close - band)).iloc[30:33].gt(0).all() and state["armed"].iloc[33]
    assert signal_bars(bars, squeeze_bars=5) == expect(d, [30, 31, 32])
    # Slightly narrower bands put bar 33 outside; bar 34 (k = 5) stays inside.
    assert signal_bars(bars, squeeze_bars=5, mult=1.99) == expect(d, [30, 31, 32, 33])
    assert signal_bars(bars, squeeze_bars=5, mult=2.01) == expect(d, [30, 31, 32])


@SIDES
@pytest.mark.parametrize("squeeze_bars", [2, 5, 125])
@pytest.mark.parametrize(
    ("offset", "expected"),
    [(1, [0, 1, 2]), (0, [])],
    ids=["step-after-the-first-squeeze", "step-on-the-first-defined-minimum"],
)
def test_no_squeeze_before_the_minimum_is_defined(d, squeeze_bars, offset, expected) -> None:
    # The minimum is first defined on bar 20 + squeeze_bars - 2. A step one bar after it is
    # armed by that bar; a step on it finds no squeeze bar before it, and the bars after are
    # above the minimum of 0 until the closes are back on the bands.
    first = 20 + squeeze_bars - 2
    at = first + offset
    bars = market(d, step(at, at + 40))
    state = SqueezeBreakout(squeeze_bars=squeeze_bars).setup(bars)
    assert state["lowest"].isna().tolist() == [True] * first + [False] * (at + 40 - first)
    assert not state["squeeze"].iloc[:first].any()
    assert signal_bars(bars, squeeze_bars=squeeze_bars) == expect(d, [at + k for k in expected])


@SIDES
@pytest.mark.parametrize(
    ("armed_bars", "expected"),
    [(1, []), (2, [30]), (3, [30, 31]), (4, [30, 31, 32]), (20, [30, 31, 32])],
)
def test_the_oldest_bar_of_the_arming_window(d, armed_bars, expected) -> None:
    # The last squeeze bar is 29: bar t is armed while t - 29 < armed_bars.
    bars = market(d, step(30, 60))
    state = SqueezeBreakout(squeeze_bars=5, armed_bars=armed_bars).setup(bars)
    assert positions(state["squeeze"].iloc[:35]) == list(range(23, 30))
    assert signal_bars(bars, squeeze_bars=5, armed_bars=armed_bars) == expect(d, expected)


# Volatile closes (+-10) on bars 0-29, flat on 30-48, then 5 from bar 49. Each flat bar's window
# holds one volatile close fewer, so its bandwidth is a new low; bar 49's window (19 zeros and
# one 5: bandwidth 0.043) is lower still than bar 48's (19 zeros and one 10: 0.087). Bar 49 is
# a squeeze bar and its close is far above its band (mean 0.25, deviation 1.09).
DECAY = alternating(10.0, 30) + [0.0] * 19 + [5.0] * 7


@SIDES
@pytest.mark.parametrize(
    ("squeeze_bars", "armed_bars", "expected"),
    [
        (5, 1, [49]),  # the breakout bar arms itself
        (5, 2, [49, 50]),
        (5, 3, [49, 50, 51]),
        (31, 1, [49]),  # bar 49 is the first bar whose minimum is defined
        (32, 1, []),  # bar 18's bandwidth is undefined: no minimum on bar 49
        (32, 20, []),  # and none on bars 50-55 is a squeeze (bar 49 is below them)
    ],
)
def test_the_newest_bar_of_the_arming_window(d, squeeze_bars, armed_bars, expected) -> None:
    bars = market(d, DECAY)
    rule = {"squeeze_bars": squeeze_bars, "armed_bars": armed_bars, "atr_length": 3}
    width = SqueezeBreakout(**rule).setup(bars)["bandwidth"].to_numpy()
    assert width[49] < width[48] and (np.diff(width[29:49]) < 0).all()
    assert signal_bars(bars, **rule) == expect(d, expected)


# Closes alternate +-3 on bars 0-39 and +-b on bars 40-59; bar 60 closes at +m and bar 61 at 10.
# Bar 59 is the first window of the quiet stretch and a squeeze bar. With squeeze_bars 2 and
# armed_bars 2, bar 61's close (far above its band) signals only if bar 60 is a squeeze bar:
# its bandwidth must not exceed bar 59's, so a tie counts.
@SIDES
@pytest.mark.parametrize(
    ("b", "m", "armed_bars", "relation", "expected"),
    [
        (0.0, 0.0, 2, "tie", [61]),  # a tie at a bandwidth of 0
        (0.0, 0.5, 2, "above", [60]),  # bar 60 breaks out itself, armed by bar 59
        (1.0, 1.0, 2, "tie", [61]),  # a tie at a positive bandwidth
        (1.0, 0.5, 2, "below", [61]),
        (1.0, 1.5, 2, "above", []),
        (1.0, 1.5, 3, "above", [61]),  # bar 59 is in reach again
    ],
    ids=["zero-tie", "zero-above", "tie", "below", "above", "above-armed-3"],
)
def test_bandwidth_ties_count_as_a_squeeze(d, b, m, armed_bars, relation, expected) -> None:
    bars = market(d, alternating(3.0, 40) + alternating(b, 20) + [m, 10.0])
    rule = {"squeeze_bars": 2, "armed_bars": armed_bars, "atr_length": 3}
    state = SqueezeBreakout(**rule).setup(bars)
    width = state["bandwidth"].to_numpy()
    assert width[59] < width[58] and state["squeeze"].iloc[59]
    if b == 1.0:  # the quiet windows: 10 closes at 101 and 10 at 99
        assert width[59] == pytest.approx(0.04)
    sign = {"below": -1, "tie": 0, "above": 1}[relation]
    assert np.sign(width[60] - width[59]) == sign
    assert state["squeeze"].iloc[60] == (sign <= 0)
    assert not state["squeeze"].iloc[61]
    assert signal_bars(bars, **rule) == expect(d, expected)


@SIDES
@pytest.mark.parametrize(
    ("atr_length", "expected"),
    [(26, [25, 26, 27]), (27, [26, 27]), (28, [27]), (29, [])],
)
def test_no_signal_while_atr_is_undefined(d, atr_length, expected) -> None:
    # Step at 25, squeeze bars 20-24; ATR(n) is defined from bar n - 1.
    bars = market(d, step(25, 40))
    rule = {"squeeze_bars": 2, "atr_length": atr_length}
    signals = SqueezeBreakout(**rule).signals(bars)
    assert signals.stop_dist.isna().tolist() == [True] * (atr_length - 1) + [False] * (
        41 - atr_length
    )
    assert signal_bars(bars, **rule) == expect(d, expected)


@SIDES
def test_allow_short_false_drops_only_the_shorts(d) -> None:
    bars = market(d, step(30, 60))
    expected = ([30, 31, 32], []) if d > 0 else ([], [])
    assert signal_bars(bars, squeeze_bars=5, allow_short=False) == expected
    trades = SqueezeBreakout(squeeze_bars=5, allow_short=False).simulate(bars, ZERO).trades
    assert len(trades) == (1 if d > 0 else 0)


@SIDES
def test_bracket_distances_are_atr_multiples_on_the_signal_bar(d) -> None:
    bars = market(d, step(30, 60))
    signals = SqueezeBreakout(squeeze_bars=5, atr_length=3, stop_atr=1.5, target_atr=6.0).signals(
        bars
    )
    atr = 2.0 + (7.0 - 2.0) / 3  # ATR(3) on the step bar
    assert signals.stop_dist.iloc[30] == pytest.approx(1.5 * atr)
    assert signals.target_dist.iloc[30] == pytest.approx(6.0 * atr)
    valid = signals.stop_dist.notna()
    np.testing.assert_allclose(signals.target_dist[valid], 4 * signals.stop_dist[valid])


# --- trades, worked by hand -----------------------------------------------------------------


@SIDES
def test_the_card_defaults_worked_by_hand(d) -> None:
    # Flat to bar 149, 5 from 150 on. The minimum of 125 bandwidths is defined from bar 143, so
    # bars 143-149 are squeeze bars and bars 150-152 signal. The trade enters at bar 151's open
    # (105) with ATR(14) of the step bar, 2 + (7 - 2) / 14; neither level is touched by bars
    # ranging 104-106, and it exits at the close of its 48th bar, 198. Nothing signals after.
    bars = market(d, step(150, 220))
    rule = SqueezeBreakout()
    assert signal_bars(bars) == expect(d, [150, 151, 152])
    atr = 2.0 + (7.0 - 2.0) / 14
    signals = rule.signals(bars)
    assert signals.stop_dist.iloc[150] == pytest.approx(2 * atr)
    assert signals.target_dist.iloc[150] == pytest.approx(4 * atr)
    trades = rule.simulate(bars, ZERO).trades
    assert trade_bars(bars, trades) == [(151, 198)]
    trade = trades.iloc[0]
    assert (trade["direction"], trade["bars"], trade["exit_reason"]) == (d, 48, "time")
    assert trade["entry_price"] == trade["exit_price"] == BASE + 5 * d
    assert trade["pnl"] == 0.0


@SIDES
@pytest.mark.parametrize(
    ("max_bars", "expected"),
    [(1, [(31, 31), (32, 32), (33, 33)]), (2, [(31, 32), (33, 34)]), (3, [(31, 33)])],
)
def test_signals_in_a_trade_are_ignored_and_the_next_one_needs_no_cooldown(
    d, max_bars, expected
) -> None:
    # Signals on 30, 31 and 32. A signal on the bar a trade exits enters at the next open; one
    # during a trade is dropped; bar 33 (on the band) never signals.
    bars = market(d, step(30, 60))
    trades = SqueezeBreakout(squeeze_bars=5, atr_length=3, max_bars=max_bars).simulate(bars, ZERO)
    assert trade_bars(bars, trades.trades) == expected
    assert (trades.trades["direction"] == d).all()
    assert (trades.trades["exit_reason"] == "time").all()


@SIDES
@pytest.mark.parametrize(
    ("tops", "bottoms", "reason", "level"),
    [
        ({34: 20.0}, {}, "target", 4.0),
        ({}, {34: -3.0}, "stop", -2.0),
        ({34: 20.0}, {34: -3.0}, "stop", -2.0),  # both touched: the stop is assumed first
    ],
    ids=["target", "stop", "both"],
)
def test_stop_and_target_sit_at_atr_multiples_of_the_signal_bar(
    d, tops, bottoms, reason, level
) -> None:
    # Entry at bar 31's open (105 for the long) after the signal on 30, whose ATR(3) is 11/3:
    # target at 105 + 44/3 = 119.67, stop at 105 - 22/3 = 97.67. Bar 34 reaches 120 or 97.
    bars = market(d, step(30, 45), tops=tops, bottoms=bottoms)
    result = SqueezeBreakout(squeeze_bars=5, atr_length=3).simulate(bars, ZERO)
    assert trade_bars(bars, result.trades) == [(31, 34)]
    trade = result.trades.iloc[0]
    assert (trade["direction"], trade["bars"], trade["exit_reason"]) == (d, 4, reason)
    exit_price = BASE + d * (5.0 + level * 11.0 / 3)
    assert trade["exit_price"] == pytest.approx(exit_price)
    # Equity of 1 buys 1 / entry price of the coin (no costs, leverage 1).
    assert trade["entry_price"] == BASE + 5 * d
    assert trade["pnl"] == pytest.approx(
        d * (exit_price - trade["entry_price"]) / trade["entry_price"]
    )


# --- parameters and specs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"length": 1}, "^length"),
        ({"length": 20.0}, "^length"),
        ({"length": True}, "^length"),
        ({"length": "20"}, "^length"),
        ({"squeeze_bars": 1}, "^squeeze_bars"),
        ({"squeeze_bars": 125.5}, "^squeeze_bars"),
        ({"armed_bars": 0}, "^armed_bars"),
        ({"atr_length": 0}, "^atr_length"),
        ({"max_bars": 0}, "^max_bars"),
        ({"max_bars": -48}, "^max_bars"),
        ({"cooldown_win": -1}, "^cooldown_win"),
        ({"cooldown_loss": -1}, "^cooldown_loss"),
        ({"cooldown_loss": 2.5}, "^cooldown_loss"),
        ({"mult": 0.0}, "^mult"),
        ({"mult": -2.0}, "^mult"),
        ({"mult": math.nan}, "^mult"),
        ({"mult": math.inf}, "^mult"),
        ({"mult": "2"}, "^mult"),
        ({"stop_atr": 0.0}, "^stop_atr"),
        ({"stop_atr": True}, "^stop_atr"),
        ({"target_atr": -4.0}, "^target_atr"),
        ({"target_atr": math.nan}, "^target_atr"),
        ({"kill_drawdown": 0.0}, "^kill_drawdown"),
        ({"kill_drawdown": 1.5}, "^kill_drawdown"),
        ({"kill_drawdown": math.nan}, "^kill_drawdown"),
        ({"kill_drawdown": True}, "^kill_drawdown"),
        ({"kill_drawdown": "1"}, "^kill_drawdown"),
    ],
)
def test_invalid_parameters_are_refused(params, message) -> None:
    with pytest.raises(ValueError, match=message):
        SqueezeBreakout(**params)


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_allow_short_must_be_a_boolean(value) -> None:
    with pytest.raises(TypeError, match="^allow_short"):
        SqueezeBreakout(allow_short=value)


def test_the_edges_of_the_valid_ranges_build() -> None:
    SqueezeBreakout(
        length=2,
        mult=1e-9,
        squeeze_bars=2,
        armed_bars=1,
        atr_length=1,
        stop_atr=1e-9,
        target_atr=1e-9,
        max_bars=1,
        cooldown_win=0,
        cooldown_loss=0,
        kill_drawdown=1e-9,
        allow_short=False,
    )
    # Grid values read back from a results table arrive as numpy numbers.
    built = SqueezeBreakout(
        squeeze_bars=np.int64(250), stop_atr=np.float64(3.0), target_atr=np.float64(6.0)
    )
    assert built == SqueezeBreakout(squeeze_bars=250, stop_atr=3.0, target_atr=6.0)


def test_the_card_spec_and_grid_are_the_class_defaults_and_build() -> None:
    card = read_card(CARD)
    assert card.spec == CARD_SPEC
    assert card.timeframe == "1h"
    assert card.combinations() == GRID
    for cell in GRID:
        assert from_spec({**CARD_SPEC, **cell}) == SqueezeBreakout(**cell)


def test_spec_round_trip_and_card_defaults() -> None:
    assert to_spec(SqueezeBreakout()) == CARD_SPEC
    assert from_spec(CARD_SPEC) == SqueezeBreakout()
    custom = SqueezeBreakout(squeeze_bars=60, stop_atr=1.5, target_atr=6.0, allow_short=False)
    assert from_spec(to_spec(custom)) == custom
    assert from_spec(json.dumps(to_spec(custom))) == custom
    assert str(SqueezeBreakout()).startswith("i006_squeeze_breakout(length=20, mult=2.0, ")


# --- engine ---------------------------------------------------------------------------------


def test_runner_backtest_runs_the_bracket_simulator_with_the_trade_rules() -> None:
    bars = random_bars(2, n=1200, freq="1h")
    rule = SqueezeBreakout(squeeze_bars=60, cooldown_win=6, cooldown_loss=3, max_bars=12)
    costs = EXCHANGE_COSTS["binanceusdm"]
    result = backtest(rule, bars, costs, max_leverage=2.0)
    direct = simulate_bracket(
        bars,
        rule.signals(bars),
        costs,
        leverage=2.0,
        cooldown_win=6,
        cooldown_loss=3,
        kill_drawdown=1.0,
        max_bars=12,
    )
    pd.testing.assert_frame_equal(result.ledger, direct.ledger)
    pd.testing.assert_frame_equal(result.trades, direct.trades)
    assert len(result.trades) > 3 and (result.trades["bars"] <= 12).all()
    assert "time" in set(result.trades["exit_reason"])


def test_cannot_be_used_as_a_target_position_strategy() -> None:
    bars = random_bars(0, n=300, freq="1h")
    for strategy in (
        SqueezeBreakout(),
        VolTarget(SqueezeBreakout()),
        Combine((SqueezeBreakout(),)),
    ):
        with pytest.raises(TypeError, match="intrabar"):
            strategy.target_position(bars)


def test_empty_and_short_markets_have_no_signals() -> None:
    bars = random_bars(0, n=300, freq="1h")
    for part in (bars.iloc[:0], bars.iloc[:1], bars.iloc[:143]):
        assert signal_bars(part) == ([], [])
        assert SqueezeBreakout().simulate(part, ZERO).trades.empty


# --- no lookahead ---------------------------------------------------------------------------

LOOKAHEAD_CASES = [
    SqueezeBreakout(),
    SqueezeBreakout(length=10, mult=1.5, squeeze_bars=30, armed_bars=5, atr_length=5, max_bars=6),
    SqueezeBreakout(squeeze_bars=60, cooldown_win=3, cooldown_loss=7, kill_drawdown=0.7),
]


def closed_trades(trades: pd.DataFrame) -> list[dict]:
    return trades[~trades["open"].astype(bool)].to_dict("records")


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("rule", LOOKAHEAD_CASES, ids=str)
def test_no_lookahead(rule, seed) -> None:
    """Appending bars changes nothing already decided: signals, ledger and closed trades."""
    bars = random_bars(seed, n=900, freq="1h")
    costs = EXCHANGE_COSTS["binanceusdm"]
    full = rule.simulate(bars, costs)
    signals = rule.signals(bars)
    assert len(full.trades) >= 2
    warm = rule.length + rule.squeeze_bars - 2  # first bar with a defined minimum
    entries = bars.index.get_indexer(pd.DatetimeIndex(full.trades["entry"]))[:4]
    exits = bars.index.get_indexer(pd.DatetimeIndex(full.trades["exit"]))[:4]
    # Cut through the warm-up, right after each signal bar, entry bar and exit bar, and on them.
    cuts = {0, 1, rule.length - 1, rule.length, warm, warm + 1, 500}
    cuts |= {int(c) for c in (*entries, *(entries + 1), *exits, *(exits + 1))}
    for cut in sorted(cuts):
        part_signals = rule.signals(bars.iloc[:cut])
        for name in ("long", "short", "stop_dist", "target_dist"):
            pd.testing.assert_series_equal(
                getattr(part_signals, name), getattr(signals, name).iloc[:cut]
            )
        part = rule.simulate(bars.iloc[:cut], costs)
        pd.testing.assert_frame_equal(part.ledger, full.ledger.iloc[:cut])
        closed = closed_trades(part.trades)
        assert closed == full.trades.iloc[: len(closed)].to_dict("records")


# --- plain-Python reference -----------------------------------------------------------------


def reference_signals(
    bars: pd.DataFrame,
    length: int = 20,
    mult: float = 2.0,
    squeeze_bars: int = 125,
    armed_bars: int = 20,
    atr_length: int = 14,
    stop_atr: float = 2.0,
    target_atr: float = 4.0,
    allow_short: bool = True,
    **trade_rules,
) -> BracketSignals:
    """The card's signal rules in plain Python, written separately from the module."""
    close = bars["close"].tolist()
    high = bars["high"].tolist()
    low = bars["low"].tolist()
    n = len(close)
    upper, lower, width = [math.nan] * n, [math.nan] * n, [math.nan] * n
    for t in range(length - 1, n):
        window = close[t - length + 1 : t + 1]
        mid = sum(window) / length
        deviation = math.sqrt(sum((x - mid) ** 2 for x in window) / length)
        upper[t] = mid + mult * deviation
        lower[t] = mid - mult * deviation
        width[t] = (upper[t] - lower[t]) / mid
    squeeze = [False] * n
    for t in range(length + squeeze_bars - 2, n):  # every bandwidth in the window is defined
        squeeze[t] = width[t] <= min(width[t - squeeze_bars + 1 : t + 1])
    atr = [math.nan] * n
    smoothed = high[0] - low[0]
    for t in range(n):
        if t > 0:
            true_range = max(
                high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1])
            )
            smoothed += (true_range - smoothed) / atr_length
        if t >= atr_length - 1:
            atr[t] = smoothed
    long, short = [], []
    for t in range(n):
        armed = any(squeeze[max(0, t - armed_bars + 1) : t + 1]) and not math.isnan(atr[t])
        long.append(armed and close[t] > upper[t])
        short.append(armed and allow_short and close[t] < lower[t])
    index = bars.index
    return BracketSignals(
        long=pd.Series(long, index=index, dtype=bool),
        short=pd.Series(short, index=index, dtype=bool),
        stop_dist=pd.Series([stop_atr * a for a in atr], index=index),
        target_dist=pd.Series([target_atr * a for a in atr], index=index),
    )


REFERENCE_CASES = [
    {},
    {"squeeze_bars": 60, "stop_atr": 1.5, "target_atr": 3.0},
    {"squeeze_bars": 250, "stop_atr": 3.0, "target_atr": 6.0},
    {"length": 10, "mult": 1.5, "squeeze_bars": 30, "armed_bars": 5, "atr_length": 5,
     "stop_atr": 1.0, "target_atr": 2.5, "max_bars": 6},
    {"length": 30, "mult": 2.5, "squeeze_bars": 40, "armed_bars": 40, "atr_length": 30,
     "max_bars": 100, "cooldown_win": 4, "cooldown_loss": 2, "kill_drawdown": 0.6},
    {"squeeze_bars": 40, "armed_bars": 1, "allow_short": False},
]  # fmt: skip


def mirrored(bars: pd.DataFrame) -> pd.DataFrame:
    """Prices turned upside down (reciprocals): rallies become sell-offs."""
    return bars.assign(
        open=1e4 / bars["open"],
        high=1e4 / bars["low"],
        low=1e4 / bars["high"],
        close=1e4 / bars["close"],
        funding_rate=-bars["funding_rate"],
    )


@pytest.mark.parametrize("mirror", [False, True], ids=["as-drawn", "mirrored"])
@pytest.mark.parametrize("seed", range(12))
def test_signals_match_the_plain_python_reference(seed, mirror) -> None:
    bars = random_bars(seed, n=1200, freq="1h")
    bars = mirrored(bars) if mirror else bars
    sides = {"long": 0, "short": 0}
    for params in REFERENCE_CASES:
        signals = SqueezeBreakout(**params).signals(bars)
        expected = reference_signals(bars, **params)
        for name in sides:
            pd.testing.assert_series_equal(getattr(signals, name), getattr(expected, name))
            sides[name] += int(getattr(signals, name).sum())
        for name in ("stop_dist", "target_dist"):
            np.testing.assert_allclose(getattr(signals, name), getattr(expected, name), rtol=1e-9)
    assert sides["long"] > 0 and sides["short"] > 0


@pytest.mark.parametrize(
    "costs", [ZERO_COSTS, EXCHANGE_COSTS["binanceusdm"]], ids=["free", "costs"]
)
@pytest.mark.parametrize("mirror", [False, True], ids=["as-drawn", "mirrored"])
@pytest.mark.parametrize("seed", range(4))
def test_simulations_match_the_reference(seed, mirror, costs) -> None:
    bars = random_bars(100 + seed, n=1500, freq="1h")
    bars = mirrored(bars) if mirror else bars
    directions: set[int] = set()
    for params in REFERENCE_CASES:
        rule = SqueezeBreakout(**params)
        reference = simulate_bracket(
            bars,
            reference_signals(bars, **params),
            costs,
            cooldown_win=rule.cooldown_win,
            cooldown_loss=rule.cooldown_loss,
            kill_drawdown=rule.kill_drawdown,
            max_bars=rule.max_bars,
        )
        result = rule.simulate(bars, costs)
        pd.testing.assert_frame_equal(result.ledger, reference.ledger, rtol=1e-9, atol=1e-12)
        pd.testing.assert_frame_equal(result.trades, reference.trades, rtol=1e-9, atol=1e-12)
        directions |= set(result.trades["direction"])
    assert directions == {-1, 1}


# --- real data ------------------------------------------------------------------------------


def test_smoke_on_real_btc_data() -> None:
    criteria = load_criteria(Workspace(ROOT).criteria_path)
    exchange = criteria.get("data.exchange")
    dev_end = criteria.get("data.dev_end")
    try:  # development bars only: they end where the holdout starts
        bars = load_bars(exchange, "BTC", "1h", end=dev_end)
    except FileNotFoundError:
        pytest.skip("BTC 1h data not downloaded")
    assert bars.index[-1] < pd.Timestamp(dev_end, tz="UTC")
    rule = SqueezeBreakout()
    result = backtest(rule, bars, EXCHANGE_COSTS[exchange])
    trades = result.trades
    assert len(trades) > 0
    assert set(trades["direction"]) <= {-1, 1}
    assert set(trades["exit_reason"]) <= set(EXIT_REASONS) - {"kill", "liquidated"}
    assert (trades["bars"] <= rule.max_bars).all()
    # Every trade enters at the open after a signal bar of its own side.
    signals = rule.signals(bars)
    before = bars.index.get_indexer(pd.DatetimeIndex(trades["entry"])) - 1
    side = np.where(
        trades["direction"] > 0, signals.long.to_numpy()[before], signals.short.to_numpy()[before]
    )
    assert side.all()
    assert np.isfinite(result.equity).all()
