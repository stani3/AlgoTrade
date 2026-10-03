"""i007_dual_thrust: Dual Thrust on 1h bars, anchored to the UTC day.

Hand-computed cases are 1h markets that start at 2024-01-01 00:00 UTC, so bar ``t`` opens at
hour ``t``: day ``t // 24``, UTC hour ``t % 24``. Prices are offsets from 100: a close listed for
a bar holds until the next listed one, each bar opens at the previous close and its high and low
are the ends of its body unless a case moves them. Targets are written as {bar: target} for the
bars that are not flat.

The canonical market: day 0 (bars 0-23) closes at 100 throughout, with a high of 110 on bar 5
and a low of 95 on bar 17. Its range is ``max(110 - 100, 100 - 95) = 10``; day 1 opens at 100,
so with k = 0.5 its lines are Buy 105 and Sell 95. Day 1's cases list closes by UTC hour.

Every case runs as written (``d = +1``) and as its mirror image (``d = -1``: prices reflected
about 100, highs and lows swapped), where the rule must take the opposite side on the same bars.
The mirror keeps the range (``HH - LC`` and ``HC - LL`` swap) and the open, so the lines are
the same: 105 and 95.
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
from algotrade.strategies.ideas.i007_dual_thrust import DualThrust

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "research" / "ideas" / "i007-dual-thrust-intraday-breakout" / "v1" / "idea.md"
SIDES = pytest.mark.parametrize("d", [1, -1], ids=["long", "short"])
CARD_SPEC = {"type": "i007_dual_thrust", "k": 0.5, "range_days": 1}
GRID = [{"k": k, "range_days": n} for k in (0.3, 0.5, 0.7) for n in (1, 2, 4)]

BASE = 100.0
START = pd.Timestamp("2024-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)
RANGE_DAY = {"tops": {5: 10.0}, "bottoms": {17: -5.0}}  # day 0: H 110, L 95, C 100 (R = 10)


def market(
    d: int,
    closes: dict[int, float] | None = None,
    n: int = 48,
    tops: dict[int, float] | None = None,
    bottoms: dict[int, float] | None = None,
    opens: dict[int, float] | None = None,
    drop: tuple[int, ...] = (),
    start: int = 0,
    unit: str = "us",
    funding: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Bars ``start .. n - 1`` without those in ``drop``, from offsets about ``BASE``.

    ``closes`` lists the bars where the close moves (it holds in between, 0 before the first);
    ``opens`` moves a bar's open off the previous close; ``tops`` and ``bottoms`` set a bar's
    high and low. Offsets are reflected for ``d = -1``; ``funding`` is not.
    """
    level, path = 0.0, []
    for t in range(n):
        level = (closes or {}).get(t, level)
        path.append(level)
    close = np.array(path)
    before = np.concatenate(([0.0], close[:-1]))
    for t, value in (opens or {}).items():
        before[t] = value
    top, bottom = np.maximum(before, close), np.minimum(before, close)
    for t, value in (tops or {}).items():
        top[t] = value
    for t, value in (bottoms or {}).items():
        bottom[t] = value
    favourable, adverse = BASE + d * top, BASE + d * bottom
    frame = pd.DataFrame(
        {
            "open": BASE + d * before,
            "high": np.maximum(favourable, adverse),
            "low": np.minimum(favourable, adverse),
            "close": BASE + d * close,
            "volume": 1.0,
            "funding_rate": [(funding or {}).get(t, 0.0) for t in range(n)],
        },
        index=pd.DatetimeIndex([START + t * HOUR for t in range(n)]).as_unit(unit),
    )
    return frame.iloc[[t for t in range(start, n) if t not in drop]]


def day_one(d: int, moves: dict[int, float], n: int = 48, **kwargs) -> pd.DataFrame:
    """The canonical market with day 1's closes given by UTC hour."""
    closes = {24 + hour: value for hour, value in moves.items()}
    return market(d, closes, n, **(RANGE_DAY | kwargs))


def held(bars: pd.DataFrame, **params) -> dict[int, float]:
    """The targets that are not flat, by bar number."""
    target = DualThrust(**params).target_position(bars)
    assert target.index.equals(bars.index)
    assert set(target.unique()) <= {-1.0, 0.0, 1.0}
    numbers = (bars.index - START) // HOUR
    return {int(t): float(v) for t, v in zip(numbers, target, strict=True) if v != 0}


def expect(d: int, values: dict[int, int]) -> dict[int, float]:
    return {t: float(d * v) for t, v in values.items()}


def side(value: int, first: int, last: int, day: int = 1) -> dict[int, int]:
    """``value`` on the bars opening at UTC hours ``first .. last`` of ``day``."""
    return {24 * day + hour: value for hour in range(first, last + 1)}


def long(first: int, last: int, day: int = 1) -> dict[int, int]:
    return side(1, first, last, day)


def short(first: int, last: int, day: int = 1) -> dict[int, int]:
    return side(-1, first, last, day)


# --- the range and the lines, worked by hand ------------------------------------------------


@SIDES
@pytest.mark.parametrize("k", [0.3, 0.5, 0.7])
def test_the_lines_of_the_canonical_market(d, k) -> None:
    bars = day_one(d, {})
    lines = DualThrust(k=k).levels(bars)
    assert lines.index.equals(bars.index)
    assert list(lines.columns) == ["open", "range", "buy", "sell", "trading"]
    # Day 0 has its 00:00 bar but no range history: no trading, and no lines.
    first = lines.iloc[:24]
    assert (first["open"] == BASE).all() and first["range"].isna().all()
    assert first[["buy", "sell"]].isna().all().all() and not first["trading"].any()
    # Day 1: O = 100, R = 10, lines 100 +- 10k (exactly 103/97, 105/95 and 107/93).
    second = lines.iloc[24:]
    assert (second["open"] == BASE).all() and (second["range"] == 10.0).all()
    assert (second["buy"] == BASE + 10 * k).all() and (second["sell"] == BASE - 10 * k).all()
    assert second["buy"].iloc[0] == {0.3: 103.0, 0.5: 105.0, 0.7: 107.0}[k]
    assert second["trading"].all()


@SIDES
def test_the_range_takes_the_larger_of_its_two_spans(d) -> None:
    # As written, H - C = 10 beats C - L = 5; mirrored, C - L = 10 beats H - C = 5.
    bars = day_one(d, {})
    day0 = bars.iloc[:24]
    high, low, close = day0["high"].max(), day0["low"].min(), day0["close"].iloc[-1]
    assert (high - close, close - low) == ((10.0, 5.0) if d > 0 else (5.0, 10.0))
    assert DualThrust().levels(bars)["range"].iloc[24] == 10.0


# Day 0: closes 100, a high of 103 on bar 5 and a low of 98 on bar 10. Day 1: the close moves
# to 104 on bar 27 and stays there, with a high of 112 on bar 32 (its low is 100 on bars 24-27).
# As written:
#   day 1, N = 1: max(103 - 100, 100 - 98) = 3, open 100: lines 101.5 / 98.5
#   day 2, N = 1: max(112 - 104, 104 - 100) = 8, open 104: lines 108 / 100
#   day 2, N = 2: HH 112 (day 1), LL 98 (day 0), HC 104 (day 1), LC 100 (day 0):
#                 max(112 - 100, 104 - 98) = 12: lines 110 / 98
# Mirrored the highs and lows swap and every range is the same.
TWO_DAYS = {
    "closes": {27: 4.0, 49: 9.0, 52: 10.5},
    "n": 72,
    "tops": {5: 3.0, 32: 12.0},
    "bottoms": {10: -2.0},
}


@SIDES
@pytest.mark.parametrize(
    ("range_days", "ranges", "lines", "expected"),
    [
        (1, (3.0, 8.0), (1.5, 8.0, 0.0), long(3, 22) | long(1, 22, day=2)),
        (2, (math.nan, 12.0), (math.nan, 10.0, -2.0), long(4, 22, day=2)),
        (3, (math.nan, math.nan), (math.nan, math.nan, math.nan), {}),
    ],
    ids=["one-day", "two-days", "three-days"],
)
def test_the_range_spans_range_days_days(d, range_days, ranges, lines, expected) -> None:
    bars = market(d, **TWO_DAYS)
    levels = DualThrust(range_days=range_days).levels(bars)
    np.testing.assert_array_equal(levels["range"].iloc[[24, 48]], ranges)
    # (Buy on day 1, Buy and Sell on day 2) as offsets for the trade's side.
    offsets = (
        levels["buy" if d > 0 else "sell"].iloc[24],
        levels["buy" if d > 0 else "sell"].iloc[48],
        levels["sell" if d > 0 else "buy"].iloc[48],
    )
    np.testing.assert_array_equal(d * (np.array(offsets) - BASE), lines)
    # Day 2 opens at 104: a close of 109 on its 01:00 bar is above the one-day line (108) but
    # not the two-day line (110); 110.5 on 04:00 is above both. With one day, day 1 trades too:
    # 104 on 03:00 is above 101.5.
    assert held(bars, range_days=range_days) == expect(d, expected)


@SIDES
@pytest.mark.parametrize(
    ("tops", "bottoms", "expected"),
    [
        ({}, {}, {}),  # a flat day: R = 0, no trading
        ({5: 0.5}, {}, long(2, 22)),  # R = 0.5 as written (H - C), Buy 100.25
        ({}, {5: -0.5}, long(2, 22)),  # R = 0.5 as written (C - L)
    ],
    ids=["zero", "high-only", "low-only"],
)
def test_no_trading_after_a_range_of_zero(d, tops, bottoms, expected) -> None:
    bars = market(d, {26: 0.5, 30: 1.0}, tops=tops, bottoms=bottoms)
    levels = DualThrust().levels(bars)
    assert levels["range"].iloc[24] == (0.0 if not expected else 0.5)
    assert levels["trading"].iloc[24:].all() == bool(expected)
    assert held(bars) == expect(d, expected)


# --- the state machine on day 1 -------------------------------------------------------------

# Day 1 closes by UTC hour, and the expected targets: lines 105 / 95, open 100.
DAY_ONE_CASES = {
    "held-to-the-end-of-the-day": ({2: 6}, long(2, 22)),
    "on-the-buy-line-is-not-an-entry": ({2: 5, 4: 5.5}, long(4, 22)),
    "on-the-sell-line-is-not-an-entry": ({2: -5, 6: -5.25}, short(6, 22)),
    "short-held-to-the-end-of-the-day": ({3: -6}, short(3, 22)),
    "entry-on-the-00:00-close": ({0: 6}, long(0, 22)),
    "entry-on-the-22:00-close": ({22: 6}, long(22, 22)),
    "no-entry-on-the-23:00-close": ({23: 6}, {}),
    "stop-below-the-open-and-no-second-long": ({2: 6, 6: -1, 8: 6}, long(2, 5)),
    "close-at-the-open-is-not-a-stop": ({2: 6, 6: 0, 9: -0.5}, long(2, 8)),
    "stop-on-the-sell-line-then-a-short": ({2: 6, 5: -5, 7: -6}, long(2, 4) | short(7, 22)),
    "reversal-then-stop-and-nothing-more": (
        {2: 6, 5: -6, 9: 1, 12: 6, 15: -6},
        long(2, 4) | short(5, 8),
    ),
    "short-stop-at-the-open-is-not-a-stop": ({2: -6, 5: 0, 7: 0.5}, short(2, 6)),
    "short-reverses-to-long": ({1: -6, 3: 6}, short(1, 2) | long(3, 22)),
    "no-second-reversal-the-long-is-stopped": (
        {1: -6, 3: 6, 6: -6, 10: 6, 14: -7},
        short(1, 2) | long(3, 5),
    ),
    "no-second-reversal-the-short-is-stopped": ({1: 6, 3: -6, 6: 6}, long(1, 2) | short(3, 5)),
    "short-stop-on-the-buy-line-then-a-long": ({1: -6, 4: 5, 6: 6}, short(1, 3) | long(6, 22)),
    "after-a-stopped-long-only-a-short": ({1: 6, 3: -1, 5: 7, 8: -6}, long(1, 2) | short(8, 22)),
    "after-a-stopped-short-only-a-long": ({1: -6, 3: 1, 5: -7, 8: 6}, short(1, 2) | long(8, 22)),
    "a-reversal-straight-through-the-open": ({2: 6, 3: -6}, long(2, 2) | short(3, 22)),
}


@SIDES
@pytest.mark.parametrize("case", DAY_ONE_CASES)
def test_the_state_machine_worked_by_hand(d, case) -> None:
    moves, expected = DAY_ONE_CASES[case]
    assert held(day_one(d, moves)) == expect(d, expected)


@SIDES
@pytest.mark.parametrize("k", [0.3, 0.5, 0.7])
def test_a_close_exactly_on_the_line_waits_for_the_next(d, k) -> None:
    # Lines at 100 +- 10k: the close on the line (bar 26) does not enter, the one a quarter
    # point beyond (bar 28) does; the close back at the open (bar 31) holds; below it stops.
    moves = {2: 10 * k, 4: 10 * k + 0.25, 7: 0.0, 9: -0.25}
    assert held(day_one(d, moves), k=k) == expect(d, long(4, 8))


@SIDES
def test_each_day_starts_flat_with_both_sides_unused(d) -> None:
    # Day 1: long at 02:00, held to 22:00, flat at 23:00. Day 1's bars (H 106, L 100, C 106)
    # give day 2 a range of 6 and its open is 106: lines 109 / 103. Day 2: long again at 01:00
    # (109.5), stop at 05:00 (105.5 < 106), short at 07:00 (102.5), held to 22:00.
    closes = {26: 6.0, 49: 9.5, 53: 5.5, 55: 2.5}
    bars = market(d, closes, n=72, **RANGE_DAY)
    levels = DualThrust().levels(bars)
    assert levels["range"].iloc[48] == 6.0 and levels["open"].iloc[48] == BASE + 6 * d
    expected = long(2, 22) | long(1, 4, day=2) | short(7, 22, day=2)
    assert held(bars) == expect(d, expected)


@SIDES
def test_a_gap_at_the_open_moves_the_lines(d) -> None:
    # Day 1 opens at 103, not at day 0's close: lines 108 / 98. 107 on 02:00 is not an entry,
    # 108.5 on 04:00 is; 102.5 on 06:00 is below the open: stop.
    bars = day_one(d, {0: 3, 2: 7, 4: 8.5, 6: 2.5}, opens={24: 3.0})
    levels = DualThrust().levels(bars)
    assert levels["open"].iloc[24] == BASE + 3 * d
    assert held(bars) == expect(d, long(4, 5))


# --- missing bars and the first days --------------------------------------------------------

# Three days: the canonical range day, day 1 long from 02:00 (106), day 2 (range 6 from day 1,
# open 106, Buy 109) long from 02:00 (110).
THREE_DAYS = {26: 6.0, 50: 10.0}
DAY_1 = long(2, 22)
DAY_2 = long(2, 22, day=2)


@SIDES
@pytest.mark.parametrize(
    ("drop", "expected"),
    [
        ((), DAY_1 | DAY_2),
        ((0,), DAY_2),  # day 0 incomplete: no range for day 1
        ((17,), DAY_2),  # the bar with day 0's low is missing: still incomplete
        ((23,), DAY_2),  # day 0's 23:00 bar (its close) is missing
        ((24,), {}),  # no 00:00 bar on day 1: no trading, and day 2 has no range
        ((25,), DAY_1),  # a missing bar before the entry; day 2 has no range
        ((26,), long(3, 22)),  # the crossing bar is missing: the next close enters
        ((30,), {t: 1 for t in DAY_1 if t != 30}),  # a gap inside the trade
        ((47,), DAY_1),  # no 23:00 bar: nothing to be flat at (see the engine test)
        ((48,), DAY_1),  # no 00:00 bar on day 2
        ((71,), DAY_1 | DAY_2),  # day 2's own 23:00 bar does not matter to day 2
        (tuple(range(24, 48)), {}),  # day 1 missing altogether
    ],
    ids=[
        "complete",
        "no-first-bar-of-day-0",
        "no-low-of-day-0",
        "no-close-of-day-0",
        "no-open-of-day-1",
        "gap-before-the-entry",
        "no-crossing-bar",
        "gap-in-the-trade",
        "no-23:00-bar-of-day-1",
        "no-open-of-day-2",
        "no-23:00-bar-of-day-2",
        "no-day-1",
    ],
)
def test_missing_bars(d, drop, expected) -> None:
    bars = market(d, THREE_DAYS, n=72, drop=drop, **RANGE_DAY)
    assert held(bars) == expect(d, expected)


@SIDES
def test_a_day_without_its_00_00_bar_has_a_range_but_no_lines(d) -> None:
    bars = market(d, THREE_DAYS, n=72, drop=(24,), **RANGE_DAY)
    levels = DualThrust().levels(bars)
    day1 = levels.iloc[24:47]  # bars 25-47 (bar 24 is missing)
    assert (day1["range"] == 10.0).all() and day1["open"].isna().all()
    assert day1[["buy", "sell"]].isna().all().all() and not day1["trading"].any()
    assert levels["range"].iloc[47:].isna().all()  # day 1 is incomplete


@SIDES
def test_a_position_left_by_a_missing_23_00_bar_is_closed_on_the_next_close(d) -> None:
    # Day 1 has no 23:00 bar, so day 2 does not trade: its 00:00 close sets the target to 0.
    # The long set at 22:00 is held through the 00:00 bar by the engine (there was no 23:00
    # close to be flat at) and is gone from the 01:00 bar on.
    bars = market(d, THREE_DAYS, n=72, drop=(47,), **RANGE_DAY)
    result = backtest(DualThrust(), bars, ZERO_COSTS)
    position = result.ledger["position"]
    numbers = ((bars.index - START) // HOUR).tolist()
    held_on = {t: p for t, p in zip(numbers, position, strict=True) if p}
    assert held_on == expect(d, {t: 1 for t in range(27, 47)} | {48: 1})


@SIDES
@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (0, DAY_1 | DAY_2),
        (1, DAY_2),  # day 0 starts at 01:00: incomplete
        (17, DAY_2),  # like the real data, which starts in the afternoon
        (24, DAY_2),  # no day 0 at all
        (25, {}),  # day 1 starts at 01:00: no open, and no range for day 2
    ],
)
def test_the_first_day_with_a_full_history_is_the_first_to_trade(d, start, expected) -> None:
    assert held(market(d, THREE_DAYS, n=72, start=start, **RANGE_DAY)) == expect(d, expected)


@pytest.mark.parametrize("range_days", [1, 2, 4, 7])
@pytest.mark.parametrize("start", [0, 17])
def test_the_warm_up_is_range_days_complete_days(range_days, start) -> None:
    bars = random_bars(4, n=24 * 10, freq="1h").iloc[start:]
    trading = DualThrust(range_days=range_days).levels(bars)["trading"]
    days = trading.groupby(bars.index.floor("D")).agg(["all", "any"])
    assert (days["all"] == days["any"]).all()  # a whole day trades or none of it
    first = range_days + (start > 0)  # a partial first day does not count
    assert days["all"].tolist() == [False] * first + [True] * (10 - first)


# --- clocks, time zones and bar sizes -------------------------------------------------------


@pytest.mark.parametrize("zone", ["America/New_York", "Asia/Kolkata", "naive"])
def test_days_are_utc_whatever_the_index_timezone(zone) -> None:
    # Two weeks of bars from 2024-03-03 cross New York's switch to summer time (2024-03-10).
    utc = random_bars(3, n=24 * 14, freq="1h")
    utc.index = utc.index + pd.Timedelta(days=62)
    other = utc.copy()
    other.index = utc.index.tz_localize(None) if zone == "naive" else utc.index.tz_convert(zone)
    for params in ({}, {"k": 0.3, "range_days": 4}):
        expected = DualThrust(**params).target_position(utc)
        assert (expected != 0).sum() > 10
        target = DualThrust(**params).target_position(other)
        assert target.index.equals(other.index)
        assert target.tolist() == expected.tolist()


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_any_timestamp_resolution(unit) -> None:
    # The stored data is indexed in milliseconds, the synthetic bars in microseconds.
    bars = market(1, THREE_DAYS, n=72, unit=unit, **RANGE_DAY)
    assert bars.index.unit == unit
    assert held(bars) == DAY_1 | DAY_2


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2024-01-01 00:30", periods=72, freq="1h", tz="UTC"),
        pd.date_range("2024-01-01", periods=144, freq="30min", tz="UTC"),
        pd.date_range("2024-01-01 00:00:00.002", periods=72, freq="1h", tz="UTC"),
        market(1, n=72).index[:-1].append(pd.DatetimeIndex([START + 71.5 * HOUR])),
    ],
    ids=["half-past", "30-minute-bars", "2-ms-late", "one-bar-off"],
)
def test_bars_off_the_whole_hour_are_refused(index) -> None:
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index)
    with pytest.raises(ValueError, match="whole UTC hours"):
        DualThrust().target_position(bars)
    with pytest.raises(ValueError, match="whole UTC hours"):
        DualThrust().levels(bars)


@pytest.mark.parametrize(("freq", "offset"), [("2h", 1), ("4h", 0), ("4h", 3), ("1D", 0)])
def test_coarser_bars_never_make_a_complete_day(freq, offset) -> None:
    rows = random_bars(1, n=300).to_numpy()[:, :4]
    index = pd.date_range(START + offset * HOUR, periods=300, freq=freq)
    bars = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    levels = DualThrust(k=0.01).levels(bars)
    assert levels["range"].isna().all() and not levels["trading"].any()
    assert (DualThrust(k=0.01).target_position(bars) == 0).all()


def test_empty_and_short_markets_are_flat() -> None:
    bars = market(1, THREE_DAYS, n=72, **RANGE_DAY)
    for part in (bars.iloc[:0], bars.iloc[:1], bars.iloc[:24], bars.iloc[24:25], bars.iloc[25:48]):
        target = DualThrust().target_position(part)
        assert target.index.equals(part.index)
        assert (target == 0).all()
    assert DualThrust().levels(bars.iloc[:0]).empty


# --- parameters and specs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"k": 0.0}, "^k must"),
        ({"k": -0.5}, "^k must"),
        ({"k": math.nan}, "^k must"),
        ({"k": math.inf}, "^k must"),
        ({"k": -math.inf}, "^k must"),
        ({"k": True}, "^k must"),
        ({"k": "0.5"}, "^k must"),
        ({"k": None}, "^k must"),
        ({"range_days": 0}, "^range_days"),
        ({"range_days": -1}, "^range_days"),
        ({"range_days": 1.0}, "^range_days"),
        ({"range_days": 1.5}, "^range_days"),
        ({"range_days": True}, "^range_days"),
        ({"range_days": "1"}, "^range_days"),
        ({"range_days": None}, "^range_days"),
    ],
)
def test_invalid_parameters_are_refused(params, message) -> None:
    with pytest.raises(ValueError, match=message):
        DualThrust(**params)


def test_the_edges_of_the_valid_ranges_build() -> None:
    DualThrust(k=1e-12, range_days=1)
    DualThrust(k=50.0, range_days=365)
    DualThrust(k=1)  # an integer k is a number like any other
    # Grid values read back from a results table arrive as numpy numbers.
    built = DualThrust(k=np.float64(0.7), range_days=np.int64(4))
    assert built == DualThrust(k=0.7, range_days=4)


def test_the_card_spec_and_grid_are_the_class_defaults_and_build() -> None:
    card = read_card(CARD)
    assert card.spec == CARD_SPEC
    assert card.timeframe == "1h"
    assert card.combinations() == GRID
    for cell in GRID:
        assert from_spec({**CARD_SPEC, **cell}) == DualThrust(**cell)


def test_spec_round_trip_and_card_defaults() -> None:
    assert to_spec(DualThrust()) == CARD_SPEC
    assert from_spec(CARD_SPEC) == DualThrust()
    custom = DualThrust(k=0.3, range_days=4)
    assert from_spec(to_spec(custom)) == custom
    assert from_spec(json.dumps(to_spec(custom))) == custom
    assert str(DualThrust()) == "i007_dual_thrust(k=0.5, range_days=1)"


# --- engine ---------------------------------------------------------------------------------


@SIDES
def test_backtest_fills_at_the_close_and_a_reversal_pays_twice(d) -> None:
    # Long at bar 26's close (02:00), reversed at bar 29's (05:00), flat at bar 47's (23:00).
    # The engine holds each target through the next bar. Funding settles on bars 32 (08:00)
    # and 40 (16:00), both in the short; day 2's lines (open 94, R 12) are never crossed.
    bars = day_one(d, {2: 6, 5: -6}, n=50, funding={32: 0.0001, 40: 0.0001})
    rule = DualThrust()
    rate = 0.0008  # 5 bps fee + 3 bps slippage
    result = backtest(rule, bars, CostModel(fee_bps=5.0, slippage_bps=3.0))
    ledger = result.ledger
    pd.testing.assert_series_equal(ledger["target"], rule.target_position(bars), check_names=False)
    held_on = {t: p for t, p in enumerate(ledger["position"]) if p}
    assert held_on == expect(d, {27: 1, 28: 1, 29: 1} | dict.fromkeys(range(30, 48), -1))
    cost = {t: c for t, c in enumerate(ledger["trading_cost"]) if c}
    assert cost == pytest.approx({27: rate, 30: 2 * rate, 48: rate})
    funding = {t: f for t, f in enumerate(ledger["funding_cost"]) if f}
    assert funding == {32: -d * 0.0001, 40: -d * 0.0001}
    trades = result.trades
    assert len(trades) == 2 and not trades["open"].any()
    assert trades["direction"].tolist() == [d, -d]
    assert trades["bars"].tolist() == [3, 18]
    assert trades["entry"].tolist() == [START + 27 * HOUR, START + 30 * HOUR]
    assert trades["exit"].tolist() == [START + 30 * HOUR, START + 48 * HOUR]
    close = bars["close"]
    first = d * (close.iloc[29] / close.iloc[28] - 1)
    assert trades["return"].iloc[0] == pytest.approx(first - 2 * rate, abs=1e-12)
    # The short earns nothing on price and receives (as written) or pays the funding.
    assert trades["return"].iloc[1] == pytest.approx(2 * d * 0.0001 - 2 * rate, abs=1e-12)


@SIDES
def test_the_longest_trade_is_23_hours(d) -> None:
    result = backtest(DualThrust(), day_one(d, {0: 6}, n=50), ZERO_COSTS)
    trades = result.trades
    assert len(trades) == 1
    assert (trades["bars"].iloc[0], trades["direction"].iloc[0]) == (23, d)
    assert trades["exit"].iloc[0] == START + 48 * HOUR  # flat from the next day's 00:00 bar


# --- no lookahead ---------------------------------------------------------------------------

# Bar t opens at hour t of a market starting at 00:00: cuts at 24k end on a day's 23:00 bar,
# 24k + 1 on its 00:00 bar, 24k + 23 just before its 23:00 bar, the rest inside a day.
CUTS = (0, 1, 2, 23, 24, 25, 26, 47, 48, 49, 60, 71, 72, 73, 96, 97, 108, 119, 120, 121, 240,
        241, 263, 264, 413, 455, 456, 457, 600, 719, 720)  # fmt: skip
LOOKAHEAD_CASES = [
    DualThrust(),
    DualThrust(k=0.3, range_days=4),
    DualThrust(k=0.7, range_days=2),
]


@pytest.mark.parametrize("gaps", [0.0, 0.01], ids=["complete", "gaps"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("rule", LOOKAHEAD_CASES, ids=str)
def test_no_lookahead(rule, seed, gaps) -> None:
    bars = random_bars(seed, n=24 * 30 + 12, freq="1h")
    if gaps:
        bars = bars[np.random.default_rng(seed).random(len(bars)) >= gaps]
    full = rule.target_position(bars)
    levels = rule.levels(bars)
    assert (full != 0).sum() > 10
    for cut in CUTS:
        pd.testing.assert_series_equal(rule.target_position(bars.iloc[:cut]), full.iloc[:cut])
        pd.testing.assert_frame_equal(rule.levels(bars.iloc[:cut]), levels.iloc[:cut])


# --- plain-Python reference -----------------------------------------------------------------


def reference_targets(bars: pd.DataFrame, k: float = 0.5, range_days: int = 1) -> list[float]:
    """The card's rules in plain Python, written separately from the module.

    File every bar under its UTC date and hour; work out each date's lines from the dates
    before it; then walk the bars with a side and the set of sides used today.
    """
    days: dict[dt.date, dict[int, tuple[float, float, float, float]]] = {}
    walk = []
    rows = zip(bars.index, bars["open"], bars["high"], bars["low"], bars["close"], strict=True)
    for stamp, o, h, low, c in rows:
        when = stamp.tz_convert("UTC") if stamp.tzinfo else stamp
        days.setdefault(when.date(), {})[when.hour] = (o, h, low, c)
        walk.append((when.date(), when.hour, c))

    lines = {}
    for date, hours in days.items():
        before = [date - dt.timedelta(days=j) for j in range(1, range_days + 1)]
        if 0 not in hours or any(len(days.get(b, {})) != 24 for b in before):
            continue
        highs = [bar[1] for b in before for bar in days[b].values()]
        lows = [bar[2] for b in before for bar in days[b].values()]
        closes = [days[b][23][3] for b in before]
        spread = max(max(highs) - min(closes), max(closes) - min(lows))
        if spread > 0:
            o = hours[0][0]
            lines[date] = (o, o + k * spread, o - k * spread)

    targets, current, held_side, used = [], None, 0, set()
    for date, hour, c in walk:
        if date != current:
            current, held_side, used = date, 0, set()
        if date not in lines or hour == 23:
            held_side = 0
        else:
            o, buy, sell = lines[date]
            up = c > buy and "long" not in used
            down = c < sell and "short" not in used
            if held_side == 0 and up:
                held_side = 1
            elif held_side == 0 and down:
                held_side = -1
            elif held_side == 1:
                held_side = -1 if down else (0 if c < o else 1)
            elif held_side == -1:
                held_side = 1 if up else (0 if c > o else -1)
            if held_side == 1:
                used.add("long")
            elif held_side == -1:
                used.add("short")
        targets.append(float(held_side))
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


def gappy(seed: int, n: int = 24 * 40, share: float = 0.02) -> pd.DataFrame:
    """Random bars missing ``share`` of the hours, a 00:00 bar, a 23:00 bar and a whole day."""
    bars = random_bars(seed, n=n, freq="1h")
    keep = np.random.default_rng(50_000 + seed).random(n) >= share
    keep[[24 * 5, 24 * 9 + 23, 24 * 12 + 7]] = False
    keep[24 * 20 : 24 * 21] = False
    return bars[keep]


def tied(seed: int, n: int = 24 * 40) -> pd.DataFrame:
    """Whole-number prices, so closes land exactly on the open and on the lines."""
    rng = np.random.default_rng(70_000 + seed)
    close = 1000.0 + np.cumsum(rng.choice([-3.0, -1.0, 0.0, 0.0, 1.0, 3.0], n))
    open_ = np.concatenate(([1000.0], close[:-1])) + rng.choice([0.0, 0.0, 0.0, 1.0, -1.0], n)
    high = np.maximum(open_, close) + rng.integers(0, 3, n)
    low = np.minimum(open_, close) - rng.integers(0, 3, n)
    return bars_from(list(zip(open_, high, low, close, strict=True)), funding=0.0, freq="1h")


MARKETS = {
    "random": lambda seed: random_bars(seed, n=24 * 40, freq="1h"),
    "mirrored": lambda seed: mirrored(random_bars(seed, n=24 * 40, freq="1h")),
    "mid-day-start": lambda seed: random_bars(seed, n=24 * 40, freq="1h").iloc[7 + seed :],
    "gaps": gappy,
    "ties": tied,
}
EXTRA_CASES = [
    {"k": 0.1, "range_days": 3},
    {"k": 0.25, "range_days": 2},
    {"k": 1.0, "range_days": 1},
    {"k": 2.0, "range_days": 7},
]


@pytest.mark.parametrize("kind", MARKETS)
@pytest.mark.parametrize("seed", range(10))
def test_matches_the_plain_python_reference(kind, seed) -> None:
    bars = MARKETS[kind](seed)
    sides: set[float] = set()
    for params in GRID + EXTRA_CASES:
        expected = reference_targets(bars, **params)
        assert DualThrust(**params).target_position(bars).tolist() == expected, params
        sides |= set(expected)
    assert sides == {-1.0, 0.0, 1.0}


def test_the_tied_markets_put_closes_on_the_open_and_on_the_lines() -> None:
    # Without such closes the comparison above would not test ">" against ">=".
    ties = {"open": 0, "line": 0}
    for seed in range(10):
        bars = tied(seed)
        for k in (0.5, 1.0):
            levels = DualThrust(k=k).levels(bars)
            close = bars["close"]
            on = levels["trading"] & (bars.index.hour < 23)
            ties["open"] += int((on & (close == levels["open"])).sum())
            ties["line"] += int((on & ((close == levels["buy"]) | (close == levels["sell"]))).sum())
    assert ties["open"] > 50 and ties["line"] > 50


@pytest.mark.parametrize("kind", ["random", "gaps", "ties"])
@pytest.mark.parametrize("seed", range(5))
def test_the_rules_hold_on_random_markets(kind, seed) -> None:
    """Flat on 23:00 bars and on days that do not trade; one long and one short a day."""
    bars = MARKETS[kind](seed)
    for params in GRID:
        rule = DualThrust(**params)
        target = rule.target_position(bars)
        levels = rule.levels(bars)
        assert (target[bars.index.hour == 23] == 0).all()
        assert (target[~levels["trading"]] == 0).all()
        frame = pd.DataFrame({"target": target, "day": bars.index.floor("D")})
        previous = frame.groupby("day")["target"].shift(fill_value=0.0)
        entries = frame[(frame["target"] != previous) & (frame["target"] != 0)]
        # Neither side is ever entered twice in a day, so at most two entries a day.
        assert (entries.groupby(["day", "target"]).size() == 1).all()
        assert (entries.groupby("day").size() <= 2).all()


@pytest.mark.parametrize(
    "costs", [ZERO_COSTS, EXCHANGE_COSTS["binanceusdm"]], ids=["free", "costs"]
)
@pytest.mark.parametrize("mirror", [False, True], ids=["as-drawn", "mirrored"])
@pytest.mark.parametrize("seed", range(5))
def test_backtests_match_the_reference(seed, mirror, costs) -> None:
    bars = gappy(100 + seed, n=24 * 60, share=0.002)
    bars = mirrored(bars) if mirror else bars
    directions: list[int] = []
    for params in ({}, {"k": 0.3, "range_days": 2}, {"k": 0.7, "range_days": 4}):
        expected = pd.Series(reference_targets(bars, **params), index=bars.index)
        reference = run_backtest(bars, expected, costs)
        result = backtest(DualThrust(**params), bars, costs)
        pd.testing.assert_frame_equal(result.ledger, reference.ledger)
        pd.testing.assert_frame_equal(result.trades, reference.trades)
        assert len(result.trades) > 0
        directions += result.trades["direction"].tolist()
    assert len(directions) > 50 and set(directions) == {-1, 1}


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
    rule = DualThrust()
    result = backtest(rule, bars, EXCHANGE_COSTS[exchange])
    target = result.ledger["target"]
    assert set(target.unique()) <= {-1.0, 0.0, 1.0}
    trades = result.trades
    assert len(trades) > 0 and set(trades["direction"]) == {-1, 1}
    # Flat at every 23:00 close, so nothing is held on the bar after it and no trade lasts
    # longer than 23 hours.
    assert (target[bars.index.hour == 23] == 0).all()
    after_23 = np.flatnonzero(bars.index.hour == 23) + 1
    assert (result.ledger["position"].iloc[after_23[after_23 < len(bars)]] == 0).all()
    assert (trades["bars"] <= 23).all()
    assert np.isfinite(result.equity).all()
