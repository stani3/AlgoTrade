"""i009_weekend_reversal: the Friday 22:00 to Sunday 22:00 UTC move, faded from Sunday 22:00.

Hand-computed cases are 1h markets that start on Monday 2024-01-01 00:00 UTC, so bar ``t``
opens at hour ``t`` and the bar opening at hour ``h`` of day ``D`` (day 0 is Monday 1 January)
is bar ``at(D, h) = 24 D + h``. Week ``w`` has its Friday 21:00 bar at ``friday(w) = 168 w +
117`` and its decision bar, Sunday 21:00, at ``sunday(w) = 168 w + 165``, 48 bars later.

Closes are offsets from 100. Where a case sets nothing they alternate -0.5 and +0.5 (99.5 on
even bars, 100.5 on odd ones), so every hourly return is about +-1%: sigma_t is about 1% and the
48-hour scale ``sigma_t x sqrt(48)`` about 6.9%, and 8.35-10.0% once a case's moves are in the
window, which puts the threshold ``min_move x scale`` at 2.09-2.50%, 4.18-5.00% and 8.35-10.0%
for min_move = 0.25, 0.5 and 1. Both bars of a weekend are odd, so a weekend without a move has
W = 0 exactly. A weekend move of x sets the Friday 21:00 close to 100 and the Sunday 21:00
close to 100 + x, so W = x% exactly.

The first 720 bars (to Wednesday 31 January 00:00) are the warm-up of the scale; the first
decision bar with a scale is week 4's, ``sunday(4) = 837`` (Sunday 4 February). Targets are
written as {bar: target} for the bars that are not flat.

Every case runs as written (``d = +1``) and as its mirror image (``d = -1``: offsets reflected
about 100, so W = -x% exactly), where the rule must take the opposite side on the same bars. As
written, a rise is sold (-1) and a fall bought (+1).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import bars_from, random_bars

from algotrade.backtest.costs import EXCHANGE_COSTS, ZERO_COSTS, CostModel
from algotrade.backtest.engine import run_backtest
from algotrade.backtest.runner import backtest, load_bars
from algotrade.research.cards import read_card
from algotrade.research.criteria import load_criteria
from algotrade.research.workspace import Workspace
from algotrade.strategies import from_spec, to_spec
from algotrade.strategies.ideas.i009_weekend_reversal import WeekendReversal

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "research" / "ideas" / "i009-weekend-move-reversal" / "v1" / "idea.md"
SIDES = pytest.mark.parametrize("d", [1, -1], ids=["as-written", "mirrored"])
CARD_SPEC = {"type": "i009_weekend_reversal", "min_move": 0.5, "hold_bars": 24, "vol_bars": 720}
GRID = [{"min_move": m, "hold_bars": h} for m in (0.25, 0.5, 1.0) for h in (12, 24, 48)]

BASE = 100.0
WIGGLE = (-0.5, 0.5)  # the offset of even and odd bars where a case sets nothing
START = pd.Timestamp("2024-01-01", tz="UTC")  # a Monday
HOUR = pd.Timedelta(hours=1)
WEEK = 168
N = WEEK * 12  # to Sunday 24 March 23:00
WEEKS = range(12)
PRICES = ["open", "high", "low", "close"]


def at(day: int, hour: int) -> int:
    """The bar opening at ``hour`` UTC on day ``day`` of the market."""
    return 24 * day + hour


def friday(week: int) -> int:
    """The bar opening Friday 21:00 of ``week``: the move starts at its close (22:00)."""
    return at(7 * week + 4, 21)


def sunday(week: int) -> int:
    """The decision bar of ``week``, opening Sunday 21:00: it closes at 22:00."""
    return at(7 * week + 6, 21)


def weekend_move(week: int, x: float) -> dict[int, float]:
    """Closes 100 at Friday 21:00 and 100 + x at Sunday 21:00: W = x% exactly."""
    return {friday(week): 0.0, sunday(week): x}


def trade(week: int, side: int, hold: int = 24) -> dict[int, int]:
    """``side`` on the bars opening Sunday 21:00 .. Sunday 21:00 + (hold - 1) hours."""
    return {sunday(week) + h: side for h in range(hold)}


def market(
    d: int,
    closes: dict[int, float] | None = None,
    n: int = N,
    drop: tuple[int, ...] = (),
    start: int = 0,
    unit: str = "us",
    begin: pd.Timestamp = START,
    funding: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Bars ``start .. n - 1`` (bar ``t`` opens at ``begin + t`` hours) without those in
    ``drop``. ``closes`` maps bars to offsets about 100 (the wiggle elsewhere), reflected for
    ``d = -1``; ``funding`` is not reflected."""
    offsets = np.array([(closes or {}).get(t, WIGGLE[t % 2]) for t in range(n)])
    close = BASE + d * offsets
    before = np.concatenate((close[:1], close[:-1]))
    frame = pd.DataFrame(
        {
            "open": before,
            "high": np.maximum(before, close) + 0.25,
            "low": np.minimum(before, close) - 0.25,
            "close": close,
            "volume": 1.0,
            "funding_rate": [(funding or {}).get(t, 0.0) for t in range(n)],
        },
        index=pd.date_range(begin, periods=n, freq="h").as_unit(unit),
    )
    return frame.iloc[[t for t in range(start, n) if t not in drop]]


def held(bars: pd.DataFrame, begin: pd.Timestamp = START, **params) -> dict[int, float]:
    """The targets that are not flat, by bar number."""
    target = WeekendReversal(**params).target_position(bars)
    assert target.index.equals(bars.index)
    assert set(target.unique()) <= {-1.0, 0.0, 1.0}
    numbers = (bars.index - begin) // HOUR
    return {int(t): float(v) for t, v in zip(numbers, target, strict=True) if v != 0}


def expect(d: int, values: dict[int, int]) -> dict[int, float]:
    return {t: float(d * v) for t, v in values.items()}


def merge(*parts: dict[int, int]) -> dict[int, int]:
    merged: dict[int, int] = {}
    for part in parts:
        merged |= part
    return merged


# --- worked by hand -------------------------------------------------------------------------


def test_the_bar_numbers_are_the_card_s_hours() -> None:
    for week in WEEKS:
        fri, sun = START + friday(week) * HOUR, START + sunday(week) * HOUR
        assert (fri.day_name(), fri.hour, sun.day_name(), sun.hour) == ("Friday", 21, "Sunday", 21)
        assert sun - fri == 48 * HOUR
    assert START + sunday(4) * HOUR == pd.Timestamp("2024-02-04 21:00", tz="UTC")


@SIDES
def test_one_weekend_worked_by_hand(d) -> None:
    # 73 bars from Friday 5 January 21:00 with vol_bars = 2: closes of 100 to the bar opening
    # Sunday 19:00, then 100 + 4d (Sunday 20:00) and 100 + 5d from Sunday 21:00 on.
    index = pd.date_range("2024-01-05 21:00", periods=73, freq="h", tz="UTC")
    close = [BASE] * 47 + [BASE + 4 * d] + [BASE + 5 * d] * 25
    bars = pd.DataFrame({"close": close}, index=index)
    frame = WeekendReversal(vol_bars=2).weekend(bars)
    assert np.flatnonzero(frame["decision"]).tolist() == [48]
    # W = 105 / 100 - 1 = 5% (95 / 100 - 1 = -5% mirrored).
    assert frame["move"].iloc[48] == pytest.approx(0.05 * d, rel=1e-12)
    # sigma_t of the two returns r1 = 4d% and r2 = (100 + 5d) / (100 + 4d) - 1 is |r1 - r2| /
    # sqrt(2), so the scale is sqrt(24) |r1 - r2|: 14.89% (14.49% mirrored), W / scale 0.336
    # (0.345): beyond 0.25 scales, short of 0.5.
    r1, r2 = 0.04 * d, (BASE + 5 * d) / (BASE + 4 * d) - 1
    scale = math.sqrt(24) * abs(r1 - r2)
    assert frame["scale"].iloc[48] == pytest.approx(scale, rel=1e-12)
    assert 0.25 < 0.05 / scale < 0.5
    assert np.isnan(frame["scale"].iloc[:2]).all() and (frame["scale"].iloc[2:47] == 0).all()
    assert held(bars, begin=index[0], min_move=0.25, vol_bars=2) == expect(
        d, dict.fromkeys(range(48, 72), -1)
    )
    assert held(bars, begin=index[0], min_move=0.5, vol_bars=2) == {}


# Weekend moves (x, in %) by week. Week 3's comes before the scale exists (sunday(3) = 669 <
# 720); week 7's is too small for any threshold.
WORKED = {3: 12.0, 4: 12.0, 5: -6.0, 6: 3.0, 7: -1.0, 8: -12.0, 9: 6.0, 10: -3.0}
WORKED_CLOSES = merge(*(weekend_move(week, x) for week, x in WORKED.items()))
# With a scale of 8.35-10.0%: 12% clears all three thresholds, 6% those of 0.25 and 0.5, 3% only
# that of 0.25 and 1% none. Rises are sold, falls bought.
SIDES_BY_MIN_MOVE = {
    0.25: {4: -1, 5: 1, 6: -1, 8: 1, 9: -1, 10: 1},
    0.5: {4: -1, 5: 1, 8: 1, 9: -1},
    1.0: {4: -1, 8: 1},
}


@SIDES
def test_the_weekend_worked_by_hand(d) -> None:
    bars = market(d, WORKED_CLOSES)
    frame = WeekendReversal().weekend(bars)
    assert frame.index.equals(bars.index)
    assert list(frame.columns) == ["decision", "move", "scale", "side"]
    # Decision bars open at Sunday 21:00 UTC, once a week.
    decisions = [sunday(week) for week in WEEKS]
    assert np.flatnonzero(frame["decision"]).tolist() == decisions
    move = frame["move"]
    assert move.drop(bars.index[decisions]).isna().all()
    for week in WEEKS:  # W = x% exactly, and 0 on weekends without a move (warm-up included)
        assert move.iloc[sunday(week)] == (BASE + d * WORKED.get(week, 0.0)) / BASE - 1
    # The scale: the sample standard deviation of the 720 returns of bars t - 719 .. t, times
    # sqrt(48), undefined on the first 720 bars.
    scale = frame["scale"]
    assert scale.iloc[:720].isna().all() and scale.iloc[720:].notna().all()
    close = bars["close"].to_numpy()
    returns = close[1:] / close[:-1] - 1  # returns[t - 1] is the return of bar t
    for t in [720, 721, 1000, N - 1, *decisions[4:]]:
        window = returns[t - 720 : t]
        assert scale.iloc[t] == pytest.approx(np.std(window, ddof=1) * math.sqrt(48), rel=1e-9)
    wiggle = WeekendReversal().weekend(market(d))["scale"].iloc[720:]
    assert ((0.069 < wiggle) & (wiggle < 0.0695)).all()  # the wiggle alone: 1.0% x sqrt(48)
    for t in decisions[4:]:  # 8.35-10.0% with the moves in the window
        assert 0.0835 < scale.iloc[t] < 0.1001
    sides = {t: s for t, s in enumerate(frame["side"]) if s}
    assert sides == expect(d, {sunday(week): side for week, side in SIDES_BY_MIN_MOVE[0.5].items()})


@SIDES
@pytest.mark.parametrize("hold_bars", [12, 24, 48])
@pytest.mark.parametrize("min_move", [0.25, 0.5, 1.0])
def test_the_targets_worked_by_hand(d, min_move, hold_bars) -> None:
    trades = SIDES_BY_MIN_MOVE[min_move].items()
    expected = merge(*(trade(week, side, hold_bars) for week, side in trades))
    bars = market(d, WORKED_CLOSES)
    assert held(bars, min_move=min_move, hold_bars=hold_bars) == expect(d, expected)


def test_the_scale_worked_by_hand() -> None:
    # Closes 100, 110, 99, 108.9: returns +a, -a, +a with a = 10%.
    index = pd.date_range(START, periods=4, freq="h")
    bars = pd.DataFrame({"close": [100.0, 110.0, 99.0, 108.9]}, index=index)
    two = WeekendReversal(vol_bars=2).weekend(bars)["scale"].to_numpy()
    # Two returns +a, -a: standard deviation a sqrt(2), times sqrt(48) = a sqrt(96).
    assert np.isnan(two[:2]).all()
    assert two[2:] == pytest.approx([0.1 * math.sqrt(96)] * 2, rel=1e-12)
    three = WeekendReversal(vol_bars=3).weekend(bars)["scale"].to_numpy()
    # +a, -a, +a: mean a/3, squared deviations (4 + 16 + 4) a^2 / 9, over 2: 4a^2/3, so the
    # standard deviation is 2a / sqrt(3) and the scale 2a sqrt(16) = 8a.
    assert np.isnan(three[:3]).all()
    assert three[3] == pytest.approx(0.8, rel=1e-12)


@pytest.mark.parametrize("vol_bars", [2, 3, 24, 720])
def test_the_scale_is_the_sample_deviation_of_the_last_vol_bars_returns(vol_bars) -> None:
    bars = hourly(1)
    scale = WeekendReversal(vol_bars=vol_bars).weekend(bars)["scale"].to_numpy()
    assert np.isnan(scale[:vol_bars]).all() and not np.isnan(scale[vol_bars:]).any()
    close = bars["close"].to_numpy()
    returns = close[1:] / close[:-1] - 1
    for t in range(vol_bars, len(bars), 37):
        expected = np.std(returns[t - vol_bars : t], ddof=1) * math.sqrt(48)
        assert scale[t] == pytest.approx(expected, rel=1e-9)


# --- the threshold --------------------------------------------------------------------------


def on_the_line(move: float, scale: float) -> float | None:
    """The min_move whose threshold ``min_move * scale`` is exactly ``|move|`` in floating
    point, or None when no float does."""
    k, target = abs(move) / scale, abs(move)
    while k * scale < target:
        k = float(np.nextafter(k, math.inf))
    while k * scale > target:
        k = float(np.nextafter(k, 0.0))
    return k if k * scale == target else None


def step(k: float, scale: float, target: float, up: bool) -> float:
    """The nearest min_move whose threshold is strictly beyond (``up``) or short of
    ``target``."""
    while (k * scale <= target) if up else (k * scale >= target):
        k = float(np.nextafter(k, math.inf if up else 0.0))
    return k


def exactly_on_the_line(
    d: int, vol_bars: int, x: float
) -> tuple[pd.DataFrame, float, float, float]:
    """A week-4 move of x% nudged by a few 1e-8 points until some float min_move puts the
    threshold exactly on W: the bars, W, the scale and that min_move."""
    for nudge in range(100):
        bars = market(d, weekend_move(4, x + nudge * 1e-8))
        frame = WeekendReversal(vol_bars=vol_bars).weekend(bars)
        move, scale = frame["move"].iloc[sunday(4)], frame["scale"].iloc[sunday(4)]
        assert move == (BASE + d * (x + nudge * 1e-8)) / BASE - 1
        k = on_the_line(move, scale)
        if k is not None:
            return bars, move, scale, k
    raise AssertionError("no float min_move puts the threshold exactly on the move")


@SIDES
@pytest.mark.parametrize("x", [3.0, 6.0, 12.0])
@pytest.mark.parametrize("vol_bars", [2, 24, 720])
def test_a_move_exactly_on_the_threshold_trades_and_one_just_short_does_not(d, vol_bars, x) -> None:
    bars, move, scale, k = exactly_on_the_line(d, vol_bars, x)
    assert k * scale == abs(move)
    # W >= min_move sigma_48 as written, W <= -min_move sigma_48 mirrored: equality trades.
    assert held(bars, min_move=k, vol_bars=vol_bars) == expect(d, trade(4, -1))
    below = step(k, scale, abs(move), up=False)
    assert held(bars, min_move=below, vol_bars=vol_bars) == expect(d, trade(4, -1))
    above = step(k, scale, abs(move), up=True)
    assert held(bars, min_move=above, vol_bars=vol_bars) == {}


@SIDES
@pytest.mark.parametrize("min_move", [0.25, 0.5, 1.0])
def test_a_move_is_always_faded(d, min_move) -> None:
    # A rise is never bought and a fall never sold, whatever its size.
    for x in (0.05, 1.0, 3.0, 6.0, 12.0, 25.0):
        sides = set(held(market(d, weekend_move(4, x)), min_move=min_move).values())
        assert sides <= {float(-d)}
    assert held(market(d, weekend_move(4, 25.0)), min_move=min_move) == expect(d, trade(4, -1))


# The 72 hours of Friday, Saturday and Sunday of week 4, but for the two bars that make W.
OTHER_HOURS = [t for t in range(at(32, 0), at(35, 0)) if t not in (friday(4), sunday(4))]


@SIDES
@pytest.mark.parametrize("bar", OTHER_HOURS)
def test_only_the_friday_and_sunday_21_00_closes_make_the_move(d, bar) -> None:
    # A 6% move clears min_move = 0.5 but not 1. An 8% spike on any other bar of the weekend
    # (the Friday 22:00 and Sunday 20:00 and 22:00 closes included) does not change W; it raises
    # the scale a little, which leaves the two thresholds either side of 6%.
    bars = market(d, weekend_move(4, 6.0) | {bar: 8.0})
    assert WeekendReversal().weekend(bars)["move"].iloc[sunday(4)] == (BASE + d * 6.0) / BASE - 1
    assert held(bars) == expect(d, trade(4, -1))
    assert held(bars, min_move=1.0) == {}


def test_without_any_move_the_rule_goes_short_as_the_card_lists_the_short_test_first() -> None:
    # Closes of 100 throughout: every return is 0, so sigma_t is exactly 0 and W = 0 meets both
    # W >= min_move sigma_48 and W <= -min_move sigma_48. The card lists the short test first.
    # (This is the one case without a mirror image, and it cannot happen on a traded market.)
    bars = market(1, dict.fromkeys(range(N), 0.0))
    frame = WeekendReversal().weekend(bars)
    assert (frame["scale"].iloc[720:] == 0).all()
    expected = merge(*(trade(week, -1) for week in WEEKS if sunday(week) >= 720))
    assert held(bars) == {t: s for t, s in expected.items() if t < N}


# --- holding ---------------------------------------------------------------------------------


@SIDES
@pytest.mark.parametrize("hold_bars", range(1, 121))
def test_a_trade_is_held_for_hold_bars_hours(d, hold_bars) -> None:
    # Short in week 4, long in week 5: the target keeps the side on the bars opening Sunday
    # 21:00 .. Sunday 21:00 + (hold_bars - 1) hours and is 0 from Sunday 21:00 + hold_bars.
    bars = market(d, weekend_move(4, 12.0) | weekend_move(5, -6.0))
    expected = trade(4, -1, hold_bars) | trade(5, 1, hold_bars)
    targets = held(bars, hold_bars=hold_bars)
    assert targets == expect(d, expected)
    assert sunday(4) + hold_bars not in targets


@SIDES
def test_the_longest_trade_runs_into_the_next_weekend_and_ends_where_its_move_starts(d) -> None:
    # Four weeks in a row with alternating moves and hold_bars = 120: each trade keeps its side
    # on the bars opening Sunday 21:00 .. Friday 20:00 and is 0 from the bar opening Friday
    # 21:00, whose close is where the next weekend's move starts; that bar earns the trade's
    # last hour, so the trade never earns an hour of the next weekend (Friday 22:00 on).
    moves = {4: 12.0, 5: -12.0, 6: 12.0, 7: -12.0}
    bars = market(d, merge(*(weekend_move(week, x) for week, x in moves.items())))
    target = WeekendReversal(hold_bars=120).target_position(bars).tolist()
    pattern = [0] * sunday(4) + ([-1] * 120 + [0] * 48 + [1] * 120 + [0] * 48) * 2
    assert target[: len(pattern)] == [d * p for p in pattern]
    assert not any(target[len(pattern) :])
    assert sunday(4) + 120 == friday(5)
    result = backtest(WeekendReversal(hold_bars=120), bars, ZERO_COSTS)
    entry = pd.DatetimeIndex(result.trades["entry"])
    exit_ = pd.DatetimeIndex(result.trades["exit"])
    assert entry.day_name().tolist() == ["Sunday"] * 4 and (entry.hour == 22).all()
    assert exit_.day_name().tolist() == ["Friday"] * 4 and (exit_.hour == 22).all()
    assert result.trades["direction"].tolist() == [-d, d, -d, d]


@SIDES
def test_consecutive_weeks_trade_separately(d) -> None:
    # Every week from 4 to 10 with a move: each is traded on its own, whatever the one before.
    moves = {4: 12.0, 5: 12.0, 6: -12.0, 7: -12.0, 8: 12.0, 9: -12.0, 10: 12.0}
    bars = market(d, merge(*(weekend_move(week, x) for week, x in moves.items())))
    for hold_bars in (12, 24, 48, 120):
        expected = merge(*(trade(w, -1 if x > 0 else 1, hold_bars) for w, x in moves.items()))
        assert held(bars, hold_bars=hold_bars) == expect(d, expected)


# --- the warm-up and missing bars -------------------------------------------------------------

TWO_WEEKS = weekend_move(4, 12.0) | weekend_move(5, -6.0)
WEEK_4 = trade(4, -1)
WEEK_5 = trade(5, 1)


@SIDES
@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (0, WEEK_4 | WEEK_5),
        (sunday(4) - 720, WEEK_4 | WEEK_5),  # week 4's decision bar is the 721st: 720 returns
        (sunday(4) - 719, WEEK_5),  # it is the 720th: 719 returns, no scale
        (sunday(5) - 720, WEEK_5),  # week 5's decision bar is the 721st
        (sunday(5) - 719, {}),
    ],
    ids=["all", "first-with-a-scale", "last-without", "week-5-first", "week-5-last-without"],
)
def test_no_trade_until_vol_bars_returns_exist(d, start, expected) -> None:
    bars = market(d, TWO_WEEKS, start=start)
    frame = WeekendReversal().weekend(bars)
    assert frame["scale"].iloc[:720].isna().all() and frame["scale"].iloc[720:].notna().all()
    assert held(bars) == expect(d, expected)


@pytest.mark.parametrize("vol_bars", [2, 24, 100, 720, 1000])
def test_the_warm_up_is_vol_bars(vol_bars) -> None:
    # The first decision bar with a scale is the first one at position vol_bars or later.
    bars = market(1, merge(*(weekend_move(week, 12.0) for week in WEEKS)))
    frame = WeekendReversal(min_move=0.1, vol_bars=vol_bars).weekend(bars)
    traded = np.flatnonzero(frame["side"]).tolist()
    assert traded == [sunday(week) for week in WEEKS if sunday(week) >= vol_bars]


def without(targets: dict[int, int], *bars: int) -> dict[int, int]:
    return {t: s for t, s in targets.items() if t not in bars}


@SIDES
@pytest.mark.parametrize(
    ("drop", "expected"),
    [
        ((), WEEK_4 | WEEK_5),
        ((friday(4),), WEEK_5),  # no Friday 21:00 bar: no move, no trade that week
        ((sunday(4),), WEEK_5),  # no decision bar: no trade, the hours after stay flat
        ((sunday(4) + 1,), without(WEEK_4, sunday(4) + 1) | WEEK_5),  # Sunday 22:00
        ((sunday(4) + 23,), without(WEEK_4, sunday(4) + 23) | WEEK_5),  # the last held bar
        ((sunday(4) + 24,), WEEK_4 | WEEK_5),  # the exit bar: Monday 21:00 is flat
        ((friday(4) - 1, friday(4) + 1), WEEK_4 | WEEK_5),  # either side of Friday 21:00
        ((sunday(4) - 1,), WEEK_4 | WEEK_5),  # Sunday 20:00: W reads only the two closes
        (tuple(range(at(33, 0), at(34, 0))), WEEK_4 | WEEK_5),  # all of Saturday
        (tuple(range(100, 110)), WEEK_4 | WEEK_5),  # warm-up bars: the scale counts bars
        ((friday(5),), WEEK_4),
        ((sunday(5),), WEEK_4),
    ],
    ids=[
        "complete",
        "no-friday-21:00-bar",
        "no-decision-bar",
        "no-sunday-22:00-bar",
        "no-last-held-bar",
        "no-exit-bar",
        "no-friday-20:00-and-22:00-bars",
        "no-sunday-20:00-bar",
        "no-saturday",
        "no-warm-up-bars",
        "no-week-5-friday-bar",
        "no-week-5-decision-bar",
    ],
)
def test_missing_bars(d, drop, expected) -> None:
    assert held(market(d, TWO_WEEKS, drop=drop)) == expect(d, expected)


@SIDES
def test_a_missing_friday_21_00_bar_leaves_the_move_undefined(d) -> None:
    bars = market(d, TWO_WEEKS, drop=(friday(4),))
    row = WeekendReversal().weekend(bars).loc[START + sunday(4) * HOUR]
    assert row["decision"] and math.isnan(row["move"]) and row["side"] == 0
    assert row["scale"] > 0


@SIDES
def test_a_missing_exit_bar_leaves_the_position_on_until_the_next_close(d) -> None:
    # Without the bar opening Monday 21:00, the engine holds the target set at the close of the
    # bar opening Monday 20:00 through the next bar there is, the one opening Monday 22:00, and
    # that bar's close (target 0) ends the trade.
    bars = market(d, TWO_WEEKS, drop=(sunday(4) + 24,))
    result = backtest(WeekendReversal(), bars, ZERO_COSTS)
    numbers = ((bars.index - START) // HOUR).tolist()
    position = result.ledger["position"]
    on = {t: p for t, p in zip(numbers, position, strict=True) if p and t < friday(5)}
    held_on = [*range(sunday(4) + 1, sunday(4) + 24), sunday(4) + 25]
    assert on == expect(d, dict.fromkeys(held_on, -1))


# --- the clock ---------------------------------------------------------------------------------


def hourly(seed: int, n: int = 24 * 60) -> pd.DataFrame:
    """``random_bars`` on 1h bars from a start (day of the week and hour) that varies with the
    seed."""
    bars = random_bars(seed, n=n, freq="1h")
    bars.index = pd.date_range(BEGINS[seed % len(BEGINS)], periods=n, freq="h", tz="UTC")
    return bars


BEGINS = [
    "2024-02-05",
    "2023-10-06 21:00",
    "2022-02-13 22:00",
    "2021-10-02 07:00",
    "2020-02-14 05:00",
    "2024-06-09 21:00",
    "2019-10-04 22:00",
    "2023-01-31 13:00",
    "2022-09-25",
    "2021-02-19 20:00",
]


def test_the_decision_bar_is_sunday_21_00_utc_in_every_week() -> None:
    # Every hour of 2019 to the end of the development data, checked against Python's own
    # calendar: one decision bar a week, and the Friday 21:00 bar 48 hours before it.
    hours = pd.date_range("2019-01-01", "2025-05-01", freq="h", tz="UTC", inclusive="left")
    bars = pd.DataFrame({"close": 100.0}, index=hours)
    decision = WeekendReversal().weekend(bars)["decision"].tolist()
    stamps = hours.to_pydatetime()
    assert decision == [s.weekday() == 6 and s.hour == 21 for s in stamps]
    decided = hours[np.flatnonzero(decision)]
    assert (np.diff(decided.asi8) == np.diff(decided[:2].asi8)[0]).all()  # every 168 hours
    assert decided[1] - decided[0] == WEEK * HOUR
    for stamp in decided.to_pydatetime():
        before = stamp - dt.timedelta(hours=48)
        assert (before.weekday(), before.hour) == (4, 21)


@pytest.mark.parametrize("zone", ["Asia/Kolkata", "America/New_York", "Europe/London", "naive"])
def test_hours_are_utc_whatever_the_index_timezone(zone) -> None:
    # The market runs from 5 February to 5 April 2024, across the clock changes of New York
    # (10 March) and London (31 March), which must not move the decision bar.
    utc = hourly(0)
    other = utc.copy()
    other.index = utc.index.tz_localize(None) if zone == "naive" else utc.index.tz_convert(zone)
    for params in ({}, {"min_move": 0.25, "hold_bars": 48}):
        rule = WeekendReversal(**params)
        expected = rule.target_position(utc)
        assert (expected > 0).sum() > 10 and (expected < 0).sum() > 10
        target = rule.target_position(other)
        assert target.index.equals(other.index)
        assert target.tolist() == expected.tolist()
        pd.testing.assert_frame_equal(
            rule.weekend(other).reset_index(drop=True), rule.weekend(utc).reset_index(drop=True)
        )


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_any_timestamp_resolution(unit) -> None:
    # The stored data is indexed in milliseconds, the synthetic bars in microseconds.
    bars = market(1, WORKED_CLOSES, unit=unit)
    assert bars.index.unit == unit
    expected = merge(*(trade(week, side) for week, side in SIDES_BY_MIN_MOVE[0.5].items()))
    assert held(bars) == expected


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2024-01-01 00:30", periods=40, freq="1h", tz="UTC"),
        pd.date_range("2024-01-01", periods=80, freq="30min", tz="UTC"),
        pd.date_range("2024-01-01 00:00:00.002", periods=40, freq="1h", tz="UTC"),
        pd.date_range("2024-01-01", periods=39, freq="1h", tz="UTC").append(
            pd.DatetimeIndex([START + 39.5 * HOUR])
        ),
        pd.date_range("2024-01-01 05:00", periods=40, freq="1h", tz="Asia/Kolkata"),
    ],
    ids=["half-past", "30-minute-bars", "2-ms-late", "one-bar-off", "whole-kolkata-hours"],
)
def test_bars_off_the_whole_utc_hour_are_refused(index) -> None:
    bars = pd.DataFrame({"close": 100.0}, index=index)
    with pytest.raises(ValueError, match="whole UTC hours"):
        WeekendReversal().target_position(bars)
    with pytest.raises(ValueError, match="whole UTC hours"):
        WeekendReversal().weekend(bars)


@pytest.mark.parametrize("freq", ["2h", "4h", "8h", "12h", "1D"])
def test_the_exchange_s_coarser_bars_have_no_sunday_21_00_bar(freq) -> None:
    # Binance's 2h-1d bars open on multiples of their length from 00:00 UTC, never at 21:00:
    # with any move enough (min_move tiny) and a scale from two returns, the rule is flat there.
    rows = random_bars(1, n=4400).to_numpy()[:, :4]
    bars = pd.DataFrame(rows, columns=PRICES, index=pd.date_range(START, periods=4400, freq=freq))
    rule = WeekendReversal(min_move=1e-9, vol_bars=2)
    frame = rule.weekend(bars)
    assert not frame["decision"].any()
    assert frame["move"].isna().all() and (frame["side"] == 0).all()
    assert (rule.target_position(bars) == 0).all()


def test_empty_and_short_markets_are_flat() -> None:
    bars = market(1, TWO_WEEKS)
    weekend = slice(friday(4), sunday(4) + 1)  # the weekend's bars without any history
    for part in (bars.iloc[:0], bars.iloc[:1], bars.iloc[sunday(4) : sunday(4) + 1],
                 bars.iloc[weekend], bars.iloc[:720]):  # fmt: skip
        target = WeekendReversal().target_position(part)
        assert target.index.equals(part.index)
        assert (target == 0).all()
    assert WeekendReversal().weekend(bars.iloc[:0]).empty
    assert WeekendReversal().target_position(bars.iloc[:0].tz_localize(None)).empty


# --- parameters and specs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"min_move": 0.0}, "^min_move must"),
        ({"min_move": -0.5}, "^min_move must"),
        ({"min_move": math.nan}, "^min_move must"),
        ({"min_move": math.inf}, "^min_move must"),
        ({"min_move": -math.inf}, "^min_move must"),
        ({"min_move": True}, "^min_move must"),
        ({"min_move": "0.5"}, "^min_move must"),
        ({"min_move": None}, "^min_move must"),
        ({"hold_bars": 0}, "^hold_bars"),
        ({"hold_bars": -24}, "^hold_bars"),
        ({"hold_bars": 121}, "^hold_bars"),
        ({"hold_bars": 168}, "^hold_bars"),
        ({"hold_bars": 24.0}, "^hold_bars"),
        ({"hold_bars": 24.5}, "^hold_bars"),
        ({"hold_bars": True}, "^hold_bars"),
        ({"hold_bars": "24"}, "^hold_bars"),
        ({"hold_bars": None}, "^hold_bars"),
        ({"vol_bars": 1}, "^vol_bars"),
        ({"vol_bars": 0}, "^vol_bars"),
        ({"vol_bars": -720}, "^vol_bars"),
        ({"vol_bars": 720.0}, "^vol_bars"),
        ({"vol_bars": True}, "^vol_bars"),
        ({"vol_bars": "720"}, "^vol_bars"),
        ({"vol_bars": None}, "^vol_bars"),
    ],
)
def test_invalid_parameters_are_refused(params, message) -> None:
    with pytest.raises(ValueError, match=message):
        WeekendReversal(**params)


def test_the_edges_of_the_valid_ranges_build() -> None:
    WeekendReversal(min_move=1e-12, hold_bars=1, vol_bars=2)
    WeekendReversal(min_move=100.0, hold_bars=120, vol_bars=100_000)
    WeekendReversal(min_move=1)  # an integer min_move is a number like any other
    # Grid values read back from a results table arrive as numpy numbers.
    built = WeekendReversal(
        min_move=np.float64(0.25), hold_bars=np.int64(48), vol_bars=np.int64(720)
    )
    assert built == WeekendReversal(min_move=0.25, hold_bars=48)


def test_the_card_spec_and_grid_are_the_class_defaults_and_build() -> None:
    card = read_card(CARD)
    assert card.spec == CARD_SPEC
    assert card.timeframe == "1h"
    assert card.combinations() == GRID
    for cell in GRID:
        assert from_spec({**CARD_SPEC, **cell}) == WeekendReversal(**cell)


def test_spec_round_trip_and_card_defaults() -> None:
    assert to_spec(WeekendReversal()) == CARD_SPEC
    assert from_spec(CARD_SPEC) == WeekendReversal()
    custom = WeekendReversal(min_move=0.75, hold_bars=100, vol_bars=48)
    assert from_spec(to_spec(custom)) == custom
    assert from_spec(json.dumps(to_spec(custom))) == custom
    assert (
        str(WeekendReversal()) == "i009_weekend_reversal(min_move=0.5, hold_bars=24, vol_bars=720)"
    )


# --- engine ---------------------------------------------------------------------------------


@SIDES
@pytest.mark.parametrize(
    "booked", [-1, 0], ids=["settled-on-the-hour-before", "settled-on-the-hour"]
)
@pytest.mark.parametrize(
    ("hold_bars", "settlements"), [(12, 2), (24, 3), (48, 6), (120, 15)], ids=str
)
def test_backtest_earns_from_sunday_22_00_and_holds_both_bars_of_each_settlement(
    d, hold_bars, settlements, booked
) -> None:
    # Funding of 0.04% at every 00:00, 08:00 and 16:00 UTC settlement, booked on the bar
    # opening at the settlement or on the one before. The trade earns the bars opening Sunday
    # 22:00 .. Sunday 22:00 + (hold_bars - 1) hours, so it holds both bars of the settlements
    # at Monday 00:00 and 08:00 (hold 12), and 16:00 (24), all of Monday and Tuesday's (48),
    # and Monday 00:00 to Friday 16:00 (120), and neither bar of any other.
    funding = {s + booked: 0.0004 for s in range(8, N, 8)}
    bars = market(d, weekend_move(4, 12.0), funding=funding)
    rule = WeekendReversal(hold_bars=hold_bars)
    rate = 0.0008  # 5 bps fee + 3 bps slippage
    result = backtest(rule, bars, CostModel(fee_bps=5.0, slippage_bps=3.0))
    ledger = result.ledger
    pd.testing.assert_series_equal(ledger["target"], rule.target_position(bars), check_names=False)
    side = -d  # the rise is sold (the fall bought, mirrored)
    first = sunday(4) + 1
    earning = range(first, first + hold_bars)
    assert {t: p for t, p in enumerate(ledger["position"]) if p} == dict.fromkeys(earning, side)
    assert {t: c for t, c in enumerate(ledger["trading_cost"]) if c} == pytest.approx(
        {first: rate, first + hold_bars: rate}
    )
    paid = {t: f for t, f in enumerate(ledger["funding_cost"]) if f}
    assert paid == {t: side * 0.0004 for t in funding if t in earning}
    assert len(paid) == settlements
    trades = result.trades
    assert len(trades) == 1
    row = trades.iloc[0]
    assert (row["direction"], row["bars"], row["open"]) == (side, hold_bars, False)
    assert (row["entry"], row["exit"]) == (START + first * HOUR, START + (first + hold_bars) * HOUR)
    assert (row["entry"].day_name(), row["entry"].hour) == ("Sunday", 22)
    if hold_bars == 24:  # the card: closed at Monday 22:00 UTC at the default
        assert (row["exit"].day_name(), row["exit"].hour) == ("Monday", 22)
    close = bars["close"].to_numpy()
    price = sum(side * (close[t] / close[t - 1] - 1) for t in earning)
    assert row["return"] == pytest.approx(price - 2 * rate - side * 0.0004 * settlements, abs=1e-12)


# --- no lookahead ---------------------------------------------------------------------------

LOOKAHEAD_CASES = [
    WeekendReversal(),
    WeekendReversal(min_move=0.25, hold_bars=48),
    WeekendReversal(min_move=1.0, hold_bars=120, vol_bars=24),
]


def lookahead_cuts(rule: WeekendReversal, bars: pd.DataFrame) -> list[int]:
    """Fixed cuts, plus cuts ending around the Friday 21:00 bar, on the Sunday 20:00, 21:00 and
    22:00 bars of several weeks, on the last held bar, on the exit bar and just after it."""
    frame = rule.weekend(bars)
    ready = np.flatnonzero(frame["decision"] & frame["scale"].notna())
    traded = [p for p in ready if frame["side"].iloc[p]][:4]
    quiet = [p for p in ready if not frame["side"].iloc[p]][:2]
    assert len(traded) >= 2 and len(quiet) >= 1
    hold = rule.hold_bars
    cuts = {0, 1, 2, rule.vol_bars, rule.vol_bars + 1, len(bars) // 2, len(bars)}
    for p in traded + quiet:  # bars[:cut] ends on bar cut - 1
        cuts |= {p - 1, p, p + 1, p + 2, p + hold, p + hold + 1, p + hold + 2}
        q = bars.index.get_indexer([bars.index[p] - 48 * HOUR])[0]  # the Friday 21:00 bar
        if q >= 0:
            cuts |= {q, q + 1, q + 2}
    return sorted(c for c in cuts if c <= len(bars))


@pytest.mark.parametrize("gaps", [0.0, 0.02], ids=["complete", "gaps"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("rule", LOOKAHEAD_CASES, ids=str)
def test_no_lookahead(rule, seed, gaps) -> None:
    bars = hourly(seed, 24 * 150)
    if gaps:
        bars = bars[np.random.default_rng(seed).random(len(bars)) >= gaps]
    full = rule.target_position(bars)
    weekend = rule.weekend(bars)
    assert (full > 0).any() and (full < 0).any()
    for cut in lookahead_cuts(rule, bars):
        pd.testing.assert_series_equal(rule.target_position(bars.iloc[:cut]), full.iloc[:cut])
        pd.testing.assert_frame_equal(rule.weekend(bars.iloc[:cut]), weekend.iloc[:cut])


# --- plain-Python reference -----------------------------------------------------------------


def reference_targets(
    bars: pd.DataFrame, min_move: float = 0.5, hold_bars: int = 24, vol_bars: int = 720
) -> list[float]:
    """The card's rules in plain Python, written separately from the module.

    Walk the bars in UTC; on each Sunday 21:00 bar find the Friday 21:00 bar two days earlier
    by its timestamp, take the standard deviation of the last ``vol_bars`` returns the long
    way, then mark every bar that opens within ``hold_bars`` hours of the decision bar.
    """
    index = bars.index
    utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    stamps = utc.to_pydatetime().tolist()
    closes = bars["close"].tolist()
    row_at = {when: row for row, when in enumerate(stamps)}
    targets = [0.0] * len(stamps)
    for row, when in enumerate(stamps):
        if when.weekday() != 6 or when.hour != 21 or row < vol_bars:
            continue
        before = row_at.get(when - dt.timedelta(days=2))
        if before is None:
            continue
        window = [closes[j] / closes[j - 1] - 1 for j in range(row - vol_bars + 1, row + 1)]
        mean = sum(window) / vol_bars
        sigma = math.sqrt(sum((x - mean) ** 2 for x in window) / (vol_bars - 1))
        threshold = min_move * (sigma * math.sqrt(48))
        move = closes[row] / closes[before] - 1
        if move >= threshold:
            side = -1.0
        elif move <= -threshold:
            side = 1.0
        else:
            continue
        end = when + dt.timedelta(hours=hold_bars)
        j = row
        while j < len(stamps) and stamps[j] < end:
            targets[j] = side
            j += 1
    return targets


def mirrored(bars: pd.DataFrame) -> pd.DataFrame:
    """Prices turned upside down (reciprocals): rallies become sell-offs."""
    return bars.assign(
        open=1e4 / bars["open"],
        high=1e4 / bars["low"],
        low=1e4 / bars["high"],
        close=1e4 / bars["close"],
        funding_rate=-bars["funding_rate"],
    )


# (day of the week, hour) of the Friday 21:00 and 22:00, Sunday 20:00-22:00 and Monday 21:00
# and 22:00 bars: the start of the move, the decision, the first earning bar and the exit at
# the default hold.
KEY_HOURS = [(4, 21), (4, 22), (6, 20), (6, 21), (6, 22), (0, 21), (0, 22)]


def gappy(seed: int, n: int = 24 * 120, share: float = 0.02) -> pd.DataFrame:
    """Random bars missing ``share`` of the hours and, in some weeks, one of the key hours."""
    bars = hourly(seed, n)
    rng = np.random.default_rng(50_000 + seed)
    keep = rng.random(n) >= share
    for day, hour in KEY_HOURS:
        key = (bars.index.dayofweek == day) & (bars.index.hour == hour)
        keep &= ~(key & (rng.random(n) < 0.15))
    return bars[keep]


def tied(seed: int, n: int = 24 * 120) -> pd.DataFrame:
    """Whole-number prices about 1000, so the weekend move is often exactly zero."""
    rng = np.random.default_rng(70_000 + seed)
    close = 1000.0 + np.cumsum(rng.choice([-9.0, -3.0, -1.0, 0.0, 0.0, 1.0, 3.0, 9.0], n))
    open_ = np.concatenate(([1000.0], close[:-1]))
    high = np.maximum(open_, close) + rng.integers(0, 3, n)
    low = np.minimum(open_, close) - rng.integers(0, 3, n)
    bars = bars_from(list(zip(open_, high, low, close, strict=True)), funding=0.0, freq="1h")
    bars.index = hourly(seed, n).index
    return bars


MARKETS = {
    "random": lambda seed: hourly(seed, 24 * 120),
    "mirrored": lambda seed: mirrored(hourly(seed, 24 * 120)),
    "gaps": gappy,
    "ties": tied,
    "kolkata-index": lambda seed: hourly(seed, 24 * 120).tz_convert("Asia/Kolkata"),
    "naive-index": lambda seed: hourly(seed, 24 * 120).tz_localize(None),
}
EXTRA_CASES = [
    {"min_move": 0.1, "hold_bars": 1},
    {"min_move": 2.0, "hold_bars": 120},
    {"min_move": 0.3, "hold_bars": 7, "vol_bars": 24},
    {"min_move": 0.75, "hold_bars": 100, "vol_bars": 2},
    {"min_move": 1.5, "hold_bars": 60, "vol_bars": 300},
]


@pytest.mark.parametrize("kind", MARKETS)
@pytest.mark.parametrize("seed", range(10))
def test_matches_the_plain_python_reference(kind, seed) -> None:
    bars = MARKETS[kind](seed)
    sides: set[float] = set()
    for params in GRID + EXTRA_CASES:
        expected = reference_targets(bars, **params)
        assert WeekendReversal(**params).target_position(bars).tolist() == expected, params
        sides |= set(expected)
    assert sides == {-1.0, 0.0, 1.0}


@pytest.mark.parametrize("kind", ["random", "ties"])
@pytest.mark.parametrize("seed", range(5))
def test_the_rules_hold_on_random_markets(kind, seed) -> None:
    """Every trade fades its weekend's move from Sunday 22:00 UTC and lasts hold_bars hours."""
    bars = (hourly if kind == "random" else tied)(seed, 24 * 180)
    for params in GRID:
        rule = WeekendReversal(**params)
        trades = backtest(rule, bars, ZERO_COSTS).trades
        assert len(trades) > 0
        entry = pd.DatetimeIndex(trades["entry"])
        assert (entry.dayofweek == 6).all() and (entry.hour == 22).all()
        assert entry.is_unique  # at most one trade a week
        assert (trades["bars"] == rule.hold_bars).all()
        move = rule.weekend(bars)["move"].reindex(entry - HOUR).to_numpy()
        assert (trades["direction"].to_numpy() == -np.sign(move)).all()


@pytest.mark.parametrize(
    "costs", [ZERO_COSTS, EXCHANGE_COSTS["binanceusdm"]], ids=["free", "costs"]
)
@pytest.mark.parametrize("mirror", [False, True], ids=["as-drawn", "mirrored"])
@pytest.mark.parametrize("seed", range(5))
def test_backtests_match_the_reference(seed, mirror, costs) -> None:
    bars = gappy(100 + seed, n=24 * 180, share=0.005)
    bars = mirrored(bars) if mirror else bars
    directions: list[int] = []
    for params in ({}, {"min_move": 0.25, "hold_bars": 12}, {"min_move": 1.0, "hold_bars": 48}):
        expected = pd.Series(reference_targets(bars, **params), index=bars.index)
        reference = run_backtest(bars, expected, costs)
        result = backtest(WeekendReversal(**params), bars, costs)
        pd.testing.assert_frame_equal(result.ledger, reference.ledger)
        pd.testing.assert_frame_equal(result.trades, reference.trades)
        assert len(result.trades) > 0
        directions += result.trades["direction"].tolist()
    assert len(directions) > 10 and set(directions) == {-1, 1}


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
    rule = WeekendReversal()
    result = backtest(rule, bars, EXCHANGE_COSTS[exchange])
    assert set(result.ledger["target"].unique()) <= {-1.0, 0.0, 1.0}
    trades = result.trades
    assert len(trades) > 0 and set(trades["direction"]) == {-1, 1}
    # Every trade earns from Sunday 22:00 UTC, at most once a week; the development bars have
    # no gaps, so each lasts exactly hold_bars bars.
    entry = pd.DatetimeIndex(trades["entry"]).tz_convert("UTC")
    assert (entry.dayofweek == 6).all() and (entry.hour == 22).all()
    assert entry.is_unique
    assert (trades["bars"] == rule.hold_bars).all()
    assert np.isfinite(result.equity).all()
