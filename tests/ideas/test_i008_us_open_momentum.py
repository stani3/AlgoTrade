"""i008_us_open_momentum: a large 08:00-10:00 New York move, followed from 10:00 New York time.

Hand-computed cases are 1h markets that start on Monday 2024-01-01 00:00 UTC, when New York is
on winter time (UTC-5), so bar ``t`` opens at hour ``t`` and the bar opening at hour ``h`` New
York time on New York day ``D`` (day 0 is Monday 1 January) is bar ``at(D, h) = 24 D + h + 5``.
Cases in summer time or across a clock change name their UTC hours explicitly.

Closes are offsets from 100. Where a case sets nothing they alternate -0.5 and +0.5 (99.5 on
even bars, 100.5 on odd ones), so every hourly return is about +-1%: sigma_t is about 1% and the
two-hour scale ``sigma_t x sqrt(2)`` about 1.41% (up to 1.52% once a case's moves are in the
window), which puts the threshold ``k x scale`` at about 0.71-0.76%, 1.41-1.52% and 2.12-2.28%
for k = 0.5, 1 and 1.5. The 07:00 and 09:00 bars of a day have the same
parity, so a day without a move has r = 0 exactly. An opening move of x sets the 07:00 close to
100, the 08:00 close to 100 + x/2 and the 09:00 close to 100 + x, so r = x% exactly.

The first 720 bars (to Wednesday 31 January 00:00 UTC) are the warm-up of the scale; the first
decision bar with a scale is Wednesday's, ``at(30, 9) = 734``. Targets are written as
{bar: target} for the bars that are not flat.

Every case runs as written (``d = +1``) and as its mirror image (``d = -1``: offsets reflected
about 100, so r = -x% exactly), where the rule must take the opposite side on the same bars.
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
from algotrade.strategies.ideas.i008_us_open_momentum import USOpenMomentum

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "research" / "ideas" / "i008-us-open-session-momentum" / "v1" / "idea.md"
SIDES = pytest.mark.parametrize("d", [1, -1], ids=["long", "short"])
CARD_SPEC = {"type": "i008_us_open_momentum", "k": 1.0, "hold_bars": 6, "vol_bars": 720}
GRID = [{"k": k, "hold_bars": h} for k in (0.5, 1.0, 1.5) for h in (4, 6, 8)]

NEW_YORK = "America/New_York"
BASE = 100.0
WIGGLE = (-0.5, 0.5)  # the offset of even and odd bars where a case sets nothing
START = pd.Timestamp("2024-01-01", tz="UTC")  # a Monday, New York on winter time
SUMMER = pd.Timestamp("2024-06-03", tz="UTC")  # a Monday, New York on summer time (UTC-4)
HOUR = pd.Timedelta(hours=1)
WINTER_LAG, SUMMER_LAG = 5, 4  # hours New York is behind UTC
N = 24 * 44  # to Tuesday 13 February
WEEKDAYS = [day for day in range(44) if day % 7 < 5]
PRICES = ["open", "high", "low", "close"]


def at(day: int, hour: int, lag: int = WINTER_LAG) -> int:
    """The bar opening at ``hour`` New York time on New York day ``day`` of the market."""
    return 24 * day + hour + lag


def opening_move(day: int, x: float, lag: int = WINTER_LAG) -> dict[int, float]:
    """Closes 100, 100 + x/2 and 100 + x on the bars opening 07:00, 08:00 and 09:00 New York."""
    return {at(day, 7, lag): 0.0, at(day, 8, lag): x / 2, at(day, 9, lag): x}


def trade(day: int, side: int = 1, hold: int = 6, lag: int = WINTER_LAG) -> dict[int, int]:
    """``side`` on the bars opening 09:00 .. (09 + hold - 1):00 New York time on ``day``."""
    return {at(day, hour, lag): side for hour in range(9, 9 + hold)}


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
    target = USOpenMomentum(**params).target_position(bars)
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

# Opening moves (x, in %) by New York day. Days 33 and 34 are a Saturday and a Sunday; day 37
# (Wednesday) has no move, r = 0.
WORKED = {30: 3.5, 31: -1.0, 32: 1.8, 33: 5.0, 34: -5.0, 35: 0.3, 36: -3.5, 38: -1.8, 39: 1.0,
          42: 3.5}  # fmt: skip
WORKED_CLOSES = {
    bar: offset for day, x in WORKED.items() for bar, offset in opening_move(day, x).items()
}
# With a scale of 1.42-1.51%: 3.5% clears all three thresholds, 1.8% those of k = 0.5 and 1,
# 1.0% only that of k = 0.5 and 0.3% none; the weekend moves are not traded.
SIDES_BY_K = {
    0.5: {30: 1, 31: -1, 32: 1, 36: -1, 38: -1, 39: 1, 42: 1},
    1.0: {30: 1, 32: 1, 36: -1, 38: -1, 42: 1},
    1.5: {30: 1, 36: -1, 42: 1},
}


@SIDES
def test_the_opening_worked_by_hand(d) -> None:
    bars = market(d, WORKED_CLOSES)
    frame = USOpenMomentum().opening(bars)
    assert frame.index.equals(bars.index)
    assert list(frame.columns) == ["decision", "move", "scale", "side"]
    # Decision bars open at 09:00 New York (14:00 UTC) on weekdays only.
    decisions = [at(day, 9) for day in WEEKDAYS]
    assert np.flatnonzero(frame["decision"]).tolist() == decisions
    move = frame["move"]
    assert move.drop(bars.index[decisions]).isna().all()  # Saturday and Sunday too
    for day in WEEKDAYS:  # r = x% exactly, and 0 on days without a move (warm-up included)
        assert move.iloc[at(day, 9)] == (BASE + d * WORKED.get(day, 0.0)) / BASE - 1
    # The scale: the sample standard deviation of the 720 returns of bars t - 719 .. t, times
    # sqrt(2), undefined on the first 720 bars.
    scale = frame["scale"]
    assert scale.iloc[:720].isna().all() and scale.iloc[720:].notna().all()
    close = bars["close"].to_numpy()
    returns = close[1:] / close[:-1] - 1  # returns[t - 1] is the return of bar t
    for t in [720, 721, *decisions[-10:], N - 1]:
        window = returns[t - 720 : t]
        assert scale.iloc[t] == pytest.approx(np.std(window, ddof=1) * math.sqrt(2), rel=1e-9)
        # 1.41% on the wiggle alone (bar 720), 1.42-1.52% with the moves in the window.
        assert 0.0141 < scale.iloc[t] < 0.0152
    sides = {t: s for t, s in enumerate(frame["side"]) if s}
    assert sides == expect(d, {at(day, 9): side for day, side in SIDES_BY_K[1.0].items()})


@SIDES
@pytest.mark.parametrize("hold_bars", [4, 6, 8])
@pytest.mark.parametrize("k", [0.5, 1.0, 1.5])
def test_the_targets_worked_by_hand(d, k, hold_bars) -> None:
    expected = merge(*(trade(day, side, hold_bars) for day, side in SIDES_BY_K[k].items()))
    assert held(market(d, WORKED_CLOSES), k=k, hold_bars=hold_bars) == expect(d, expected)


def test_the_scale_worked_by_hand() -> None:
    # Closes 100, 110, 99, 108.9: returns +a, -a, +a with a = 10%.
    index = pd.date_range(START, periods=4, freq="h")
    bars = pd.DataFrame({"close": [100.0, 110.0, 99.0, 108.9]}, index=index)
    two = USOpenMomentum(vol_bars=2).opening(bars)["scale"].to_numpy()
    # Two returns +a, -a: standard deviation a sqrt(2), times sqrt(2) = 2a.
    assert np.isnan(two[:2]).all()
    assert two[2:] == pytest.approx([0.2, 0.2], rel=1e-12)
    three = USOpenMomentum(vol_bars=3).opening(bars)["scale"].to_numpy()
    # +a, -a, +a: mean a/3, squared deviations (4 + 16 + 4) a^2 / 9, over 2: 4a^2/3, so the
    # standard deviation is 2a / sqrt(3) and the scale 2a sqrt(2/3).
    assert np.isnan(three[:3]).all()
    assert three[3] == pytest.approx(0.2 * math.sqrt(2 / 3), rel=1e-12)


@pytest.mark.parametrize("vol_bars", [2, 3, 24, 720])
def test_the_scale_is_the_sample_deviation_of_the_last_vol_bars_returns(vol_bars) -> None:
    bars = hourly(1)
    scale = USOpenMomentum(vol_bars=vol_bars).opening(bars)["scale"].to_numpy()
    assert np.isnan(scale[:vol_bars]).all() and not np.isnan(scale[vol_bars:]).any()
    close = bars["close"].to_numpy()
    returns = close[1:] / close[:-1] - 1
    for t in range(vol_bars, len(bars), 37):
        expected = np.std(returns[t - vol_bars : t], ddof=1) * math.sqrt(2)
        assert scale[t] == pytest.approx(expected, rel=1e-9)


# --- the threshold --------------------------------------------------------------------------


def on_the_line(move: float, scale: float) -> float:
    """The k whose threshold ``k * scale`` is exactly ``|move|`` in floating point."""
    k, target = abs(move) / scale, abs(move)
    while k * scale < target:
        k = float(np.nextafter(k, math.inf))
    while k * scale > target:
        k = float(np.nextafter(k, 0.0))
    assert k * scale == target, "no float k puts the threshold exactly on the move"
    return k


def step(k: float, scale: float, target: float, up: bool) -> float:
    """The nearest k whose threshold is strictly beyond (``up``) or short of ``target``."""
    while (k * scale <= target) if up else (k * scale >= target):
        k = float(np.nextafter(k, math.inf if up else 0.0))
    return k


@SIDES
@pytest.mark.parametrize("x", [1.0, 1.8, 3.5])
@pytest.mark.parametrize("vol_bars", [2, 24, 720])
def test_a_move_exactly_on_the_threshold_trades_and_one_just_short_does_not(d, vol_bars, x) -> None:
    bars = market(d, opening_move(30, x))
    frame = USOpenMomentum(vol_bars=vol_bars).opening(bars)
    move, scale = frame["move"].iloc[at(30, 9)], frame["scale"].iloc[at(30, 9)]
    assert move == (BASE + d * x) / BASE - 1
    k = on_the_line(move, scale)
    # r >= k sigma_open as written, r <= -k sigma_open mirrored: equality trades.
    assert held(bars, k=k, vol_bars=vol_bars) == expect(d, trade(30))
    below = step(k, scale, abs(move), up=False)
    assert held(bars, k=below, vol_bars=vol_bars) == expect(d, trade(30))
    above = step(k, scale, abs(move), up=True)
    assert held(bars, k=above, vol_bars=vol_bars) == {}


@SIDES
@pytest.mark.parametrize("k", [0.5, 1.0, 1.5])
def test_a_move_never_trades_against_its_sign(d, k) -> None:
    # A rise is never shorted and a fall never bought, whatever its size.
    for x in (0.05, 0.3, 1.0, 1.8, 3.5, 8.0):
        sides = set(held(market(d, opening_move(30, x)), k=k).values())
        assert sides <= {float(d)}


@SIDES
@pytest.mark.parametrize("hour", [h for h in range(24) if h not in (7, 9)])
def test_only_the_07_00_and_09_00_closes_make_the_move(d, hour) -> None:
    # A 1.8% move clears k = 1 but not 1.5. A 5% spike on any other bar of Wednesday or
    # Tuesday (the 08:00 close included) does not change r; it raises the scale by about 3%,
    # which leaves the two thresholds either side of 1.8%.
    for day in (29, 30):
        closes = opening_move(30, 1.8) | {at(day, hour): 5.0}
        bars = market(d, closes)
        assert USOpenMomentum().opening(bars)["move"].iloc[at(30, 9)] == (BASE + d * 1.8) / BASE - 1
        assert held(bars) == expect(d, trade(30))
        assert held(bars, k=1.5) == {}


def test_without_any_move_the_rule_goes_long_as_the_card_lists_the_long_test_first() -> None:
    # Closes of 100 throughout: every return is 0, so sigma_t is exactly 0 and r = 0 meets both
    # r >= k sigma_open and r <= -k sigma_open. The card lists the long test first. (This is
    # the one case without a mirror image, and it cannot happen on a traded market.)
    bars = market(1, dict.fromkeys(range(N), 0.0))
    frame = USOpenMomentum().opening(bars)
    assert (frame["scale"].iloc[720:] == 0).all()
    expected = merge(*(trade(day) for day in WEEKDAYS if day >= 30))
    assert held(bars) == expected


# --- holding ---------------------------------------------------------------------------------


@SIDES
@pytest.mark.parametrize("hold_bars", range(1, 13))
def test_a_trade_is_held_for_hold_bars_hours(d, hold_bars) -> None:
    # Long on Wednesday, short on Thursday: the target keeps the side on the bars opening
    # 09:00 .. (09 + hold_bars - 1):00 New York and is 0 from (09 + hold_bars):00, so the
    # longest trade (12) is flat from 21:00 and never meets the next day's.
    bars = market(d, opening_move(30, 3.5) | opening_move(31, -3.5))
    expected = trade(30, 1, hold_bars) | trade(31, -1, hold_bars)
    assert held(bars, hold_bars=hold_bars) == expect(d, expected)
    assert at(30, 9 + hold_bars) not in held(bars, hold_bars=hold_bars)


@SIDES
def test_consecutive_days_trade_separately(d) -> None:
    # Wednesday to Friday with alternating moves, then the weekend: Monday trades again.
    moves = {30: 3.5, 31: -3.5, 32: 3.5, 35: -3.5}
    bars = market(d, merge(*(opening_move(day, x) for day, x in moves.items())))
    pattern = [0] * at(30, 9) + ([1] * 12 + [0] * 12 + [-1] * 12 + [0] * 12 + [1] * 12)
    target = USOpenMomentum(hold_bars=12).target_position(bars).tolist()
    assert target[: len(pattern)] == [d * p for p in pattern]
    assert {t: v for t, v in enumerate(target) if v and t >= len(pattern)} == expect(
        d, trade(35, -1, 12)
    )


# --- the warm-up and missing bars -------------------------------------------------------------

TWO_DAYS = opening_move(30, 3.5) | opening_move(31, -3.5)
WEDNESDAY = trade(30)
THURSDAY = trade(31, -1)


@SIDES
@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (0, WEDNESDAY | THURSDAY),
        (14, WEDNESDAY | THURSDAY),  # Wednesday's decision bar is the 721st bar: 720 returns
        (15, THURSDAY),  # it is the 720th: 719 returns, no scale; Thursday's has 743
        (at(31, 9) - 720, THURSDAY),  # Thursday's decision bar is the 721st
        (at(31, 9) - 719, {}),
    ],
    ids=["all", "first-with-a-scale", "last-without", "thursday-first", "thursday-last-without"],
)
def test_no_trade_until_vol_bars_returns_exist(d, start, expected) -> None:
    bars = market(d, TWO_DAYS, start=start)
    frame = USOpenMomentum().opening(bars)
    assert frame["scale"].iloc[:720].isna().all() and frame["scale"].iloc[720:].notna().all()
    assert held(bars) == expect(d, expected)


@pytest.mark.parametrize("vol_bars", [2, 24, 100, 720, 1000])
def test_the_warm_up_is_vol_bars(vol_bars) -> None:
    # The first decision bar with a scale is the first one at position vol_bars or later.
    bars = market(1, merge(*(opening_move(day, 3.5) for day in WEEKDAYS)))
    frame = USOpenMomentum(vol_bars=vol_bars).opening(bars)
    decisions = [at(day, 9) for day in WEEKDAYS]
    traded = np.flatnonzero(frame["side"]).tolist()
    assert traded == [t for t in decisions if t >= vol_bars]


@SIDES
@pytest.mark.parametrize(
    ("drop", "expected"),
    [
        ((), WEDNESDAY | THURSDAY),
        ((at(30, 7),), THURSDAY),  # no 07:00 bar: no move
        ((at(30, 8),), THURSDAY),  # no 08:00 bar, although r reads only 07:00 and 09:00
        ((at(30, 9),), THURSDAY),  # no decision bar: no trade, the hours after stay flat
        ((at(30, 10),), {t: s for t, s in (WEDNESDAY | THURSDAY).items() if t != at(30, 10)}),
        ((at(30, 14),), {t: s for t, s in (WEDNESDAY | THURSDAY).items() if t != at(30, 14)}),
        ((at(30, 15),), WEDNESDAY | THURSDAY),  # the exit bar: 16:00 is flat
        ((at(30, 6), at(30, 5)), WEDNESDAY | THURSDAY),  # bars before the window
        ((at(29, 9), at(29, 8)), WEDNESDAY | THURSDAY),  # Tuesday's opening
        (tuple(range(100, 110)), WEDNESDAY | THURSDAY),  # warm-up bars: the scale counts bars
        ((at(31, 7),), WEDNESDAY),
        ((at(31, 8),), WEDNESDAY),
    ],
    ids=[
        "complete",
        "no-07:00-bar",
        "no-08:00-bar",
        "no-decision-bar",
        "no-10:00-bar",
        "no-last-held-bar",
        "no-exit-bar",
        "no-bars-before",
        "no-tuesday-opening",
        "no-warm-up-bars",
        "no-thursday-07:00-bar",
        "no-thursday-08:00-bar",
    ],
)
def test_missing_bars(d, drop, expected) -> None:
    assert held(market(d, TWO_DAYS, drop=drop)) == expect(d, expected)


@SIDES
@pytest.mark.parametrize("hour", [7, 8])
def test_a_missing_07_00_or_08_00_bar_leaves_the_move_undefined(d, hour) -> None:
    bars = market(d, TWO_DAYS, drop=(at(30, hour),))
    row = USOpenMomentum().opening(bars).loc[START + at(30, 9) * HOUR]
    assert row["decision"] and math.isnan(row["move"]) and row["side"] == 0
    assert row["scale"] > 0


@SIDES
def test_a_missing_exit_bar_leaves_the_position_on_until_the_next_close(d) -> None:
    # The engine holds the target set at the 14:00 close through the next bar there is, the
    # 16:00 bar, and the 16:00 close (target 0) ends the trade.
    bars = market(d, TWO_DAYS, drop=(at(30, 15),))
    result = backtest(USOpenMomentum(), bars, ZERO_COSTS)
    numbers = ((bars.index - START) // HOUR).tolist()
    position = result.ledger["position"]
    on = {t: p for t, p in zip(numbers, position, strict=True) if p and t < at(31, 0)}
    held_on = [at(30, h) for h in range(10, 15)] + [at(30, 16)]
    assert on == expect(d, dict.fromkeys(held_on, 1))


# --- the New York clock -----------------------------------------------------------------------


def bar_of(begin: pd.Timestamp, stamp: str) -> int:
    return int((pd.Timestamp(stamp, tz="UTC") - begin) // HOUR)


def hours_from(begin: pd.Timestamp, stamp: str, side: int, count: int = 6) -> dict[int, int]:
    first = bar_of(begin, stamp)
    return dict.fromkeys(range(first, first + count), side)


SPRING = pd.Timestamp("2024-02-05", tz="UTC")  # New York moves to summer time on 10 March
AUTUMN = pd.Timestamp("2024-09-30", tz="UTC")  # and back to winter time on 3 November
CLOCK_CHANGES = {
    "march": (
        SPRING,
        {
            # Friday 8 March, winter time: 07:00-09:00 New York is 12:00-14:00 UTC. A summer
            # clock would read 11:00 -> 13:00 UTC: 101.75 / 103.5 - 1 = -1.7%, a short.
            "2024-03-08 11:00": 3.5,
            "2024-03-08 12:00": 0.0,
            "2024-03-08 13:00": 1.75,
            "2024-03-08 14:00": 3.5,
            # Sunday 10 March, the day of the change: no trade at the weekend.
            "2024-03-10 11:00": 0.0,
            "2024-03-10 12:00": 2.5,
            "2024-03-10 13:00": 5.0,
            # Monday 11 March, summer time: 11:00-13:00 UTC. A winter clock would read 12:00 ->
            # 14:00 UTC: 98.25 / 101.75 - 1 = -3.4%, a short.
            "2024-03-11 11:00": 0.0,
            "2024-03-11 12:00": 1.75,
            "2024-03-11 13:00": 3.5,
            "2024-03-11 14:00": -1.75,
            # Tuesday 12 March: a fall.
            "2024-03-12 11:00": 0.0,
            "2024-03-12 12:00": -1.75,
            "2024-03-12 13:00": -3.5,
        },
        {"2024-03-08 14:00": 1, "2024-03-11 13:00": 1, "2024-03-12 13:00": -1},
        {"2024-03-08 13:00", "2024-03-11 14:00", "2024-03-12 14:00"},
    ),
    "november": (
        AUTUMN,
        {
            # Friday 1 November, summer time: 11:00-13:00 UTC. A winter clock would read 12:00
            # -> 14:00 UTC: -3.4%.
            "2024-11-01 11:00": 0.0,
            "2024-11-01 12:00": 1.75,
            "2024-11-01 13:00": 3.5,
            "2024-11-01 14:00": -1.75,
            # Sunday 3 November, the day of the change.
            "2024-11-03 12:00": 0.0,
            "2024-11-03 13:00": -2.5,
            "2024-11-03 14:00": -5.0,
            # Monday 4 November, winter time: 12:00-14:00 UTC. A summer clock would read 11:00
            # -> 13:00 UTC: -1.7%.
            "2024-11-04 11:00": 3.5,
            "2024-11-04 12:00": 0.0,
            "2024-11-04 13:00": 1.75,
            "2024-11-04 14:00": 3.5,
            # Tuesday 5 November: a fall.
            "2024-11-05 12:00": 0.0,
            "2024-11-05 13:00": -1.75,
            "2024-11-05 14:00": -3.5,
        },
        {"2024-11-01 13:00": 1, "2024-11-04 14:00": 1, "2024-11-05 14:00": -1},
        {"2024-11-01 14:00", "2024-11-04 13:00", "2024-11-05 13:00"},
    ),
}


@SIDES
@pytest.mark.parametrize("change", CLOCK_CHANGES)
def test_the_opening_follows_new_york_across_the_clock_changes(d, change) -> None:
    begin, moves, trades, other_clock = CLOCK_CHANGES[change]
    closes = {bar_of(begin, stamp): x for stamp, x in moves.items()}
    bars = market(d, closes, n=24 * 40, begin=begin)
    frame = USOpenMomentum().opening(bars)
    decision = frame["decision"]
    for stamp in trades:
        assert decision.loc[pd.Timestamp(stamp, tz="UTC")]
    for stamp in other_clock:
        assert not decision.loc[pd.Timestamp(stamp, tz="UTC")]
    expected = merge(*(hours_from(begin, stamp, side) for stamp, side in trades.items()))
    assert held(bars, begin=begin) == expect(d, expected)


@SIDES
@pytest.mark.parametrize(
    ("begin", "holiday", "decision"),
    [
        ("2024-02-26", "2024-03-29", "2024-03-29 13:00"),  # Good Friday, summer time
        ("2024-06-03", "2024-07-04", "2024-07-04 13:00"),  # Independence Day
        ("2024-11-25", "2024-12-25", "2024-12-25 14:00"),  # Christmas Day, winter time
    ],
)
def test_us_market_holidays_are_traded_like_any_weekday(d, begin, holiday, decision) -> None:
    # The card keeps holidays (there is no holiday calendar in the data). The Saturday after
    # each holiday has the same move and is not traded.
    start = pd.Timestamp(begin, tz="UTC")
    when = pd.Timestamp(decision, tz="UTC")
    saturday = when + pd.Timedelta(days=(5 - when.dayofweek) % 7)
    closes = {}
    for day in (when, saturday):
        bar = int((day - start) // HOUR)
        closes |= {bar - 2: 0.0, bar - 1: 1.75, bar: 3.5}
    bars = market(d, closes, n=24 * 36, begin=start)
    assert when.tz_convert(NEW_YORK).hour == 9 and str(when.date()) == holiday
    expected = hours_from(start, decision, 1)
    assert held(bars, begin=start) == expect(d, expected)


def hourly(seed: int, n: int = 24 * 60) -> pd.DataFrame:
    """``random_bars`` on 1h bars from a start that varies with the seed; all but one cross a
    clock change."""
    bars = random_bars(seed, n=n, freq="1h")
    bars.index = pd.date_range(BEGINS[seed % len(BEGINS)], periods=n, freq="h", tz="UTC")
    return bars


BEGINS = [
    "2024-02-05",
    "2023-10-02 07:00",
    "2022-02-14 13:00",
    "2021-10-04",
    "2020-02-10 05:00",
    "2024-06-03",
    "2019-10-07 21:00",
    "2023-01-31",
    "2022-09-26",
    "2021-02-15",
]


@pytest.mark.parametrize("zone", [NEW_YORK, "Asia/Kolkata", "Europe/London", "naive"])
def test_the_clock_is_new_york_whatever_the_index_timezone(zone) -> None:
    # The market runs from 5 February to 5 April 2024, across New York's change on 10 March.
    utc = hourly(0)
    other = utc.copy()
    other.index = utc.index.tz_localize(None) if zone == "naive" else utc.index.tz_convert(zone)
    for params in ({}, {"k": 0.5, "hold_bars": 12}):
        rule = USOpenMomentum(**params)
        expected = rule.target_position(utc)
        assert (expected > 0).sum() > 10 and (expected < 0).sum() > 10
        target = rule.target_position(other)
        assert target.index.equals(other.index)
        assert target.tolist() == expected.tolist()
        pd.testing.assert_frame_equal(
            rule.opening(other).reset_index(drop=True), rule.opening(utc).reset_index(drop=True)
        )


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_any_timestamp_resolution(unit) -> None:
    # The stored data is indexed in milliseconds, the synthetic bars in microseconds.
    bars = market(1, WORKED_CLOSES, unit=unit)
    assert bars.index.unit == unit
    expected = merge(*(trade(day, side) for day, side in SIDES_BY_K[1.0].items()))
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
        USOpenMomentum().target_position(bars)
    with pytest.raises(ValueError, match="whole UTC hours"):
        USOpenMomentum().opening(bars)


@pytest.mark.parametrize(
    ("freq", "offset", "decisions"),
    [
        ("2h", 0, True),  # 14:00 UTC is 09:00 in winter, but 13:00 UTC (08:00) is missing
        ("2h", 1, True),  # 13:00 UTC is 09:00 in summer, but 12:00 UTC (08:00) is missing
        ("4h", 0, False),
        ("4h", 1, True),
        ("4h", 2, True),
        ("4h", 3, False),
        ("1D", 0, False),
        ("1D", 13, True),
        ("1D", 14, True),
    ],
)
def test_bars_two_or_more_hours_apart_never_trade(freq, offset, decisions) -> None:
    # Over a year or more of summer and winter time, with any move enough (k tiny) and a scale
    # from two returns: the 08:00 bar an hour before the decision bar never exists.
    rows = random_bars(1, n=4400).to_numpy()[:, :4]
    index = pd.date_range(START + offset * HOUR, periods=4400, freq=freq)
    bars = pd.DataFrame(rows, columns=PRICES, index=index)
    rule = USOpenMomentum(k=1e-9, vol_bars=2)
    frame = rule.opening(bars)
    assert frame["decision"].any() == decisions
    assert frame["move"].isna().all() and (frame["side"] == 0).all()
    assert (rule.target_position(bars) == 0).all()


def test_empty_and_short_markets_are_flat() -> None:
    bars = market(1, TWO_DAYS)
    wednesday = slice(at(30, 7), at(30, 9) + 1)  # the opening bars without any history
    for part in (bars.iloc[:0], bars.iloc[:1], bars.iloc[at(30, 9) : at(30, 9) + 1],
                 bars.iloc[wednesday], bars.iloc[:720]):  # fmt: skip
        target = USOpenMomentum().target_position(part)
        assert target.index.equals(part.index)
        assert (target == 0).all()
    assert USOpenMomentum().opening(bars.iloc[:0]).empty
    assert USOpenMomentum().target_position(bars.iloc[:0].tz_localize(None)).empty


# --- parameters and specs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"k": 0.0}, "^k must"),
        ({"k": -1.0}, "^k must"),
        ({"k": math.nan}, "^k must"),
        ({"k": math.inf}, "^k must"),
        ({"k": -math.inf}, "^k must"),
        ({"k": True}, "^k must"),
        ({"k": "1.0"}, "^k must"),
        ({"k": None}, "^k must"),
        ({"hold_bars": 0}, "^hold_bars"),
        ({"hold_bars": -6}, "^hold_bars"),
        ({"hold_bars": 13}, "^hold_bars"),
        ({"hold_bars": 6.0}, "^hold_bars"),
        ({"hold_bars": 6.5}, "^hold_bars"),
        ({"hold_bars": True}, "^hold_bars"),
        ({"hold_bars": "6"}, "^hold_bars"),
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
        USOpenMomentum(**params)


def test_the_edges_of_the_valid_ranges_build() -> None:
    USOpenMomentum(k=1e-12, hold_bars=1, vol_bars=2)
    USOpenMomentum(k=100.0, hold_bars=12, vol_bars=100_000)
    USOpenMomentum(k=1)  # an integer k is a number like any other
    # Grid values read back from a results table arrive as numpy numbers.
    built = USOpenMomentum(k=np.float64(1.5), hold_bars=np.int64(8), vol_bars=np.int64(720))
    assert built == USOpenMomentum(k=1.5, hold_bars=8)


def test_the_card_spec_and_grid_are_the_class_defaults_and_build() -> None:
    card = read_card(CARD)
    assert card.spec == CARD_SPEC
    assert card.timeframe == "1h"
    assert card.combinations() == GRID
    for cell in GRID:
        assert from_spec({**CARD_SPEC, **cell}) == USOpenMomentum(**cell)


def test_spec_round_trip_and_card_defaults() -> None:
    assert to_spec(USOpenMomentum()) == CARD_SPEC
    assert from_spec(CARD_SPEC) == USOpenMomentum()
    custom = USOpenMomentum(k=1.25, hold_bars=11, vol_bars=48)
    assert from_spec(to_spec(custom)) == custom
    assert from_spec(json.dumps(to_spec(custom))) == custom
    assert str(USOpenMomentum()) == "i008_us_open_momentum(k=1.0, hold_bars=6, vol_bars=720)"


# --- engine ---------------------------------------------------------------------------------

SEASONS = {"winter": (START, WINTER_LAG), "summer": (SUMMER, SUMMER_LAG)}


@SIDES
@pytest.mark.parametrize("booked", [15, 16], ids=["settled-on-15:00", "settled-on-16:00"])
@pytest.mark.parametrize("hold_bars", [4, 6, 8])
@pytest.mark.parametrize("season", SEASONS)
def test_backtest_earns_from_10_00_and_holds_both_bars_of_the_16_00_utc_settlement(
    d, season, hold_bars, booked
) -> None:
    # The 16:00 UTC settlement is booked on the bar opening at 15:00 or 16:00 UTC. A trade
    # earns the bars opening 10:00 .. (10 + hold_bars - 1):00 New York time: 15:00-18:00 UTC
    # and later in winter, 14:00-17:00 UTC and later in summer, so it holds both.
    begin, lag = SEASONS[season]
    settlement = 24 * 30 + booked
    bars = market(d, opening_move(30, 3.5, lag), begin=begin, funding={settlement: 0.0004})
    rule = USOpenMomentum(hold_bars=hold_bars)
    rate = 0.0008  # 5 bps fee + 3 bps slippage
    result = backtest(rule, bars, CostModel(fee_bps=5.0, slippage_bps=3.0))
    ledger = result.ledger
    pd.testing.assert_series_equal(ledger["target"], rule.target_position(bars), check_names=False)
    first = at(30, 10, lag)
    earning = range(first, first + hold_bars)
    assert {t: p for t, p in enumerate(ledger["position"]) if p} == expect(
        d, dict.fromkeys(earning, 1)
    )
    assert {24 * 30 + 15, 24 * 30 + 16} <= set(earning)
    assert {t: c for t, c in enumerate(ledger["trading_cost"]) if c} == pytest.approx(
        {first: rate, first + hold_bars: rate}
    )
    assert {t: f for t, f in enumerate(ledger["funding_cost"]) if f} == {settlement: d * 0.0004}
    trades = result.trades
    assert len(trades) == 1
    row = trades.iloc[0]
    assert (row["direction"], row["bars"], row["open"]) == (d, hold_bars, False)
    assert (row["entry"], row["exit"]) == (begin + first * HOUR, begin + (first + hold_bars) * HOUR)
    assert row["entry"].tz_convert(NEW_YORK).hour == 10
    assert row["exit"].tz_convert(NEW_YORK).hour == 10 + hold_bars
    close = bars["close"].to_numpy()
    price = sum(d * (close[t] / close[t - 1] - 1) for t in earning)
    assert row["return"] == pytest.approx(price - 2 * rate - d * 0.0004, abs=1e-12)


# --- no lookahead ---------------------------------------------------------------------------

LOOKAHEAD_CASES = [
    USOpenMomentum(),
    USOpenMomentum(k=0.5, hold_bars=8),
    USOpenMomentum(k=1.2, hold_bars=12, vol_bars=48),
]


def lookahead_cuts(rule: USOpenMomentum, bars: pd.DataFrame) -> list[int]:
    """Fixed cuts, plus cuts ending on the 07:00, 08:00, 09:00 and 10:00 bars of several days,
    on the last held bar, on the exit bar and just after it."""
    frame = rule.opening(bars)
    ready = np.flatnonzero(frame["decision"] & frame["scale"].notna())
    traded = [p for p in ready if frame["side"].iloc[p]][:4]
    quiet = [p for p in ready if not frame["side"].iloc[p]][:2]
    assert len(traded) >= 2 and len(quiet) == 2
    hold = rule.hold_bars
    cuts = {0, 1, 2, rule.vol_bars, rule.vol_bars + 1, len(bars) // 2, len(bars)}
    for p in traded + quiet:  # bars[:cut] ends on bar cut - 1
        cuts |= {p - 1, p, p + 1, p + 2, p + hold, p + hold + 1, p + hold + 2}
    return sorted(c for c in cuts if c <= len(bars))


@pytest.mark.parametrize("gaps", [0.0, 0.02], ids=["complete", "gaps"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("rule", LOOKAHEAD_CASES, ids=str)
def test_no_lookahead(rule, seed, gaps) -> None:
    bars = hourly(seed, 24 * 90)
    if gaps:
        bars = bars[np.random.default_rng(seed).random(len(bars)) >= gaps]
    full = rule.target_position(bars)
    opening = rule.opening(bars)
    assert (full > 0).any() and (full < 0).any()
    for cut in lookahead_cuts(rule, bars):
        pd.testing.assert_series_equal(rule.target_position(bars.iloc[:cut]), full.iloc[:cut])
        pd.testing.assert_frame_equal(rule.opening(bars.iloc[:cut]), opening.iloc[:cut])


# --- plain-Python reference -----------------------------------------------------------------


def first_sunday(year: int, month: int) -> int:
    return 1 + (6 - dt.date(year, month, 1).weekday()) % 7


def new_york_time(utc: dt.datetime) -> dt.datetime:
    """The New York wall-clock time (naive) of a UTC time, by the US rules in force since 2007:
    summer time (UTC-4) from 02:00 on the second Sunday of March to 02:00 on the first Sunday
    of November, winter time (UTC-5) otherwise."""
    year = utc.year
    summer_from = dt.datetime(year, 3, first_sunday(year, 3) + 7, 7, tzinfo=dt.UTC)  # 02:00 EST
    summer_to = dt.datetime(year, 11, first_sunday(year, 11), 6, tzinfo=dt.UTC)  # 02:00 EDT
    behind = 4 if summer_from <= utc < summer_to else 5
    return (utc - dt.timedelta(hours=behind)).replace(tzinfo=None)


def reference_targets(
    bars: pd.DataFrame, k: float = 1.0, hold_bars: int = 6, vol_bars: int = 720
) -> list[float]:
    """The card's rules in plain Python, written separately from the module.

    Put every bar on the New York clock by hand; on each weekday's 09:00 bar find the 07:00
    and 08:00 bars of the same day by their wall-clock time, take the standard deviation of the
    last ``vol_bars`` returns the long way, and mark the hours each trade is held.
    """
    index = bars.index
    utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    clock = [new_york_time(stamp) for stamp in utc.to_pydatetime()]
    closes = bars["close"].tolist()
    row_at = {when: row for row, when in enumerate(clock)}
    sides: dict[dt.date, float] = {}
    for row, when in enumerate(clock):
        if when.hour != 9 or when.weekday() >= 5 or row < vol_bars:
            continue
        seven = row_at.get(when - dt.timedelta(hours=2))
        if seven is None or when - dt.timedelta(hours=1) not in row_at:
            continue
        window = [closes[j] / closes[j - 1] - 1 for j in range(row - vol_bars + 1, row + 1)]
        mean = sum(window) / vol_bars
        sigma = math.sqrt(sum((x - mean) ** 2 for x in window) / (vol_bars - 1))
        threshold = k * (sigma * math.sqrt(2))
        move = closes[row] / closes[seven] - 1
        if move >= threshold:
            sides[when.date()] = 1.0
        elif move <= -threshold:
            sides[when.date()] = -1.0
    return [
        sides.get(when.date(), 0.0) if 9 <= when.hour < 9 + hold_bars else 0.0 for when in clock
    ]


def test_the_reference_clock_matches_the_time_zone_database() -> None:
    # Every hour of 2019-2025, compared with pandas' zone data: the two clocks are written
    # independently, so the reference does not inherit a time-zone mistake from the module.
    hours = pd.date_range("2019-01-01", "2025-05-01", freq="h", tz="UTC")
    mine = [new_york_time(stamp) for stamp in hours.to_pydatetime()]
    theirs = hours.tz_convert(NEW_YORK).tz_localize(None).to_pydatetime().tolist()
    assert mine == theirs


def mirrored(bars: pd.DataFrame) -> pd.DataFrame:
    """Prices turned upside down (reciprocals): rallies become sell-offs."""
    return bars.assign(
        open=1e4 / bars["open"],
        high=1e4 / bars["low"],
        low=1e4 / bars["high"],
        close=1e4 / bars["close"],
        funding_rate=-bars["funding_rate"],
    )


def gappy(seed: int, n: int = 24 * 60, share: float = 0.02) -> pd.DataFrame:
    """Random bars missing ``share`` of the hours and, on some days, the 07:00, 08:00, 09:00 or
    11:00 New York bar."""
    bars = hourly(seed, n)
    rng = np.random.default_rng(50_000 + seed)
    keep = rng.random(n) >= share
    local = bars.index.tz_convert(NEW_YORK)
    for hour in (7, 8, 9, 11):
        keep &= ~((local.hour == hour) & (rng.random(n) < 0.15))
    return bars[keep]


def tied(seed: int, n: int = 24 * 60) -> pd.DataFrame:
    """Whole-number prices about 1000, so the move is often exactly zero."""
    rng = np.random.default_rng(70_000 + seed)
    close = 1000.0 + np.cumsum(rng.choice([-3.0, -1.0, 0.0, 0.0, 1.0, 3.0], n))
    open_ = np.concatenate(([1000.0], close[:-1]))
    high = np.maximum(open_, close) + rng.integers(0, 3, n)
    low = np.minimum(open_, close) - rng.integers(0, 3, n)
    bars = bars_from(list(zip(open_, high, low, close, strict=True)), funding=0.0, freq="1h")
    bars.index = hourly(seed, n).index
    return bars


MARKETS = {
    "random": hourly,
    "mirrored": lambda seed: mirrored(hourly(seed)),
    "gaps": gappy,
    "ties": tied,
    "new-york-index": lambda seed: hourly(seed).tz_convert(NEW_YORK),
    "naive-index": lambda seed: hourly(seed).tz_localize(None),
}
EXTRA_CASES = [
    {"k": 0.25, "hold_bars": 1},
    {"k": 2.0, "hold_bars": 12},
    {"k": 0.8, "hold_bars": 3, "vol_bars": 24},
    {"k": 1.2, "hold_bars": 10, "vol_bars": 2},
    {"k": 3.0, "hold_bars": 5, "vol_bars": 100},
]


@pytest.mark.parametrize("kind", MARKETS)
@pytest.mark.parametrize("seed", range(10))
def test_matches_the_plain_python_reference(kind, seed) -> None:
    bars = MARKETS[kind](seed)
    sides: set[float] = set()
    for params in GRID + EXTRA_CASES:
        expected = reference_targets(bars, **params)
        assert USOpenMomentum(**params).target_position(bars).tolist() == expected, params
        sides |= set(expected)
    assert sides == {-1.0, 0.0, 1.0}


@pytest.mark.parametrize("kind", ["random", "ties"])
@pytest.mark.parametrize("seed", range(5))
def test_the_rules_hold_on_random_markets(kind, seed) -> None:
    """Every trade starts earning at 10:00 New York on a weekday and lasts hold_bars hours."""
    bars = (hourly if kind == "random" else tied)(seed, 24 * 120)
    for params in GRID:
        rule = USOpenMomentum(**params)
        trades = backtest(rule, bars, ZERO_COSTS).trades
        assert len(trades) > 0
        entry = pd.DatetimeIndex(trades["entry"]).tz_convert(NEW_YORK)
        assert (entry.hour == 10).all() and (entry.dayofweek <= 4).all()
        assert entry.normalize().is_unique  # at most one trade a day
        assert (trades["bars"] == rule.hold_bars).all()


@pytest.mark.parametrize(
    "costs", [ZERO_COSTS, EXCHANGE_COSTS["binanceusdm"]], ids=["free", "costs"]
)
@pytest.mark.parametrize("mirror", [False, True], ids=["as-drawn", "mirrored"])
@pytest.mark.parametrize("seed", range(5))
def test_backtests_match_the_reference(seed, mirror, costs) -> None:
    bars = gappy(100 + seed, n=24 * 120, share=0.005)
    bars = mirrored(bars) if mirror else bars
    directions: list[int] = []
    for params in ({}, {"k": 0.5, "hold_bars": 8}, {"k": 1.5, "hold_bars": 4}):
        expected = pd.Series(reference_targets(bars, **params), index=bars.index)
        reference = run_backtest(bars, expected, costs)
        result = backtest(USOpenMomentum(**params), bars, costs)
        pd.testing.assert_frame_equal(result.ledger, reference.ledger)
        pd.testing.assert_frame_equal(result.trades, reference.trades)
        assert len(result.trades) > 0
        directions += result.trades["direction"].tolist()
    assert len(directions) > 20 and set(directions) == {-1, 1}


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
    rule = USOpenMomentum()
    result = backtest(rule, bars, EXCHANGE_COSTS[exchange])
    assert set(result.ledger["target"].unique()) <= {-1.0, 0.0, 1.0}
    trades = result.trades
    assert len(trades) > 0 and set(trades["direction"]) == {-1, 1}
    # Every trade earns from 10:00 New York time on a weekday, summer and winter; the
    # development bars have no gaps, so each lasts exactly hold_bars bars.
    entry = pd.DatetimeIndex(trades["entry"]).tz_convert(NEW_YORK)
    assert (entry.hour == 10).all() and (entry.dayofweek <= 4).all()
    assert set(pd.DatetimeIndex(trades["entry"]).hour) == {14, 15}
    assert (trades["bars"] == rule.hold_bars).all()
    assert np.isfinite(result.equity).all()
