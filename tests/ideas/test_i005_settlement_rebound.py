"""i005_settlement_rebound: the side of the previous funding settlement, held after the next.

Hand-computed cases are 1h markets that start at 2024-01-01 00:00 UTC, so bar ``t`` opens at
hour ``t`` (UTC hour ``t % 24``). Settlements fall at t = 0, 8, 16, 24, ...; the decision bar of
the settlement at H is bar H - 1, and its funding proxy is the funding of bars H - 9 and H - 8
(the two bars that can hold the settlement at H - 8). Prices are a flat 100 unless a case says
otherwise, and targets are written as {bar: target} for the bars that are not flat.

Every case runs as written (``d = +1``) and as its mirror image (``d = -1``: funding negated,
prices reflected about 100), where the rule must take the opposite side on the same bars.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import random_bars

from algotrade.backtest.costs import EXCHANGE_COSTS, ZERO_COSTS, CostModel
from algotrade.backtest.engine import run_backtest
from algotrade.backtest.runner import backtest, load_bars
from algotrade.research.cards import read_card
from algotrade.research.criteria import load_criteria
from algotrade.research.workspace import Workspace
from algotrade.strategies import from_spec, to_spec
from algotrade.strategies.ideas.i005_settlement_rebound import SettlementRebound

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "research" / "ideas" / "i005-funding-settlement-rebound" / "v1" / "idea.md"
SIDES = pytest.mark.parametrize("d", [1, -1], ids=["as-written", "mirrored"])
CARD_SPEC = {"type": "i005_settlement_rebound", "threshold": 0.0005, "hold_bars": 2}
GRID = [{"threshold": t, "hold_bars": h} for t in (0.0003, 0.0005, 0.001) for h in (1, 2, 4)]

START = pd.Timestamp("2024-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)


def market(
    d: int,
    funding: dict[int, float],
    n: int,
    closes: dict[int, float] | None = None,
    drop: tuple[int, ...] = (),
    start: int = 0,
    unit: str = "us",
) -> pd.DataFrame:
    """Bars ``start .. n - 1`` without those in ``drop``; ``funding`` and ``closes`` map bar
    numbers to values (0 and 100 elsewhere). ``d = -1`` negates funding and reflects prices
    about 100."""
    hours = [t for t in range(start, n) if t not in drop]
    closes = closes or {}
    close = np.array([100.0 + d * (closes.get(t, 100.0) - 100.0) for t in hours])
    before = np.concatenate((close[:1], close[:-1]))
    index = pd.DatetimeIndex([START + t * HOUR for t in hours], tz="UTC").as_unit(unit)
    return pd.DataFrame(
        {
            "open": before,
            "high": np.maximum(before, close) + 1.0,
            "low": np.minimum(before, close) - 1.0,
            "close": close,
            "volume": 1.0,
            "funding_rate": [d * funding.get(t, 0.0) for t in hours],
        },
        index=index,
    )


def held(bars: pd.DataFrame, **params) -> dict[int, float]:
    """The targets that are not flat, by bar number."""
    target = SettlementRebound(**params).target_position(bars)
    assert target.index.equals(bars.index)
    assert set(target.unique()) <= {-1.0, 0.0, 1.0}
    numbers = (bars.index - START) // HOUR
    return {int(t): float(v) for t, v in zip(numbers, target, strict=True) if v != 0}


def expect(d: int, values: dict[int, int]) -> dict[int, float]:
    return {t: float(d * v) for t, v in values.items()}


# --- worked by hand -------------------------------------------------------------------------

# Funding as align_funding books it: a settlement stamped 0-1 ms after the hour sits on the bar
# closing at the hour, one stamped 2-47 ms after on the bar opening at it.
WORKED = {
    0: 0.002,  # 00:00 day 1 on the bar opening at it; its other possible bar (23:00) is missing
    7: 0.0005,  # 08:00 on the bar closing at 08:00
    15: -0.003,  # 16:00 on the decision bar of 16:00 itself
    24: 0.0004,  # 00:00 day 2 on the bar opening at it
    32: -0.0007,  # 08:00 day 2; the 16:00 settlement of day 2 is exactly zero
    42: 0.004,  # SOL-style extra settlements at 18:00 and 22:00, on the bars 42 and 45
    45: -0.004,
    48: 0.0011,  # 00:00 day 3
}
WORKED_N = 58


@SIDES
def test_the_settlements_worked_by_hand(d) -> None:
    signals = SettlementRebound().settlements(market(d, WORKED, WORKED_N))
    decision_bars = [7, 15, 23, 31, 39, 47, 55]  # opening at 07:00, 15:00 and 23:00 UTC
    assert np.flatnonzero(signals["decision"]).tolist() == decision_bars
    proxy = signals["funding"].iloc[decision_bars]
    # Bar 7: f_8 needs the missing bar -1, so it is undefined although bar 0 holds 0.002.
    # Bar 15: bars 7 + 8 = 0.0005 (the 16:00 settlement on bar 15 itself is not used).
    # Bar 23: bars 15 + 16 = -0.003. Bar 31: 0.0004. Bar 39: -0.0007. Bar 47: bars 39 + 40 = 0
    # (the extra settlements on 42 and 45 are never read). Bar 55: 0.0011.
    assert math.isnan(proxy.iloc[0])
    expected = [0.0005, -0.003, 0.0004, -0.0007, 0.0, 0.0011]
    assert proxy.iloc[1:].tolist() == [d * f for f in expected]
    assert signals["funding"].drop(signals.index[decision_bars]).isna().all()
    sides = {int(t): float(s) for t, s in enumerate(signals["side"]) if s != 0}
    assert sides == expect(d, {15: 1, 23: -1, 39: -1, 55: 1})


@SIDES
@pytest.mark.parametrize(
    ("hold_bars", "expected"),
    [
        (1, {15: 1, 23: -1, 39: -1, 55: 1}),
        (2, {15: 1, 16: 1, 23: -1, 24: -1, 39: -1, 40: -1, 55: 1, 56: 1}),
        (4, {15: 1, 16: 1, 17: 1, 18: 1, 23: -1, 24: -1, 25: -1, 26: -1, 39: -1, 40: -1,
             41: -1, 42: -1, 55: 1, 56: 1, 57: 1}),
    ],
    ids=["one-hour", "card-default", "four-hours"],
)  # fmt: skip
def test_the_targets_worked_by_hand(d, hold_bars, expected) -> None:
    # Long at 15 (f = 0.0005, exactly at the threshold), short at 23 (-0.003), flat at 31
    # (0.0004 is below), short at 39 (-0.0007), flat at 47 (zero), long at 55 (0.0011). A trade
    # is held on bars H - 1 .. H + hold_bars - 2; the market ends at bar 57.
    assert held(market(d, WORKED, WORKED_N), hold_bars=hold_bars) == expect(d, expected)


@SIDES
@pytest.mark.parametrize("hold_bars", range(1, 8))
def test_a_trade_is_held_for_hold_bars_hours_and_never_overlaps_the_next(d, hold_bars) -> None:
    # Long at decision bar 15, short at decision bar 23 (bar 16 holds the 16:00 settlement).
    bars = market(d, {7: 0.001, 16: -0.001}, 40)
    long = dict.fromkeys(range(15, 15 + hold_bars), 1)
    short = dict.fromkeys(range(23, 23 + hold_bars), -1)
    # 8 - hold_bars flat bars between the two trades: one with the longest hold.
    assert held(bars, hold_bars=hold_bars) == expect(d, long | short)


@SIDES
def test_consecutive_settlements_alternate_with_one_flat_hour_at_the_longest_hold(d) -> None:
    # The settlements at 08:00, 16:00, 00:00 and 08:00 alternate in sign; each one is booked on
    # the decision bar of its own hour and read by the next decision bar.
    funding = {7: 0.001, 15: -0.001, 23: 0.001, 31: -0.001}
    target = SettlementRebound(hold_bars=7).target_position(market(d, funding, 48))
    pattern = [0] * 15 + [1] * 7 + [0] + [-1] * 7 + [0] + [1] * 7 + [0] + [-1] * 7 + [0] * 2
    assert target.tolist() == [d * p for p in pattern]


# --- the threshold --------------------------------------------------------------------------


# Funding on the two bars of the previous settlement, as a function of the threshold.
THRESHOLD_CASES = {
    "at-early": (lambda th: th, lambda th: 0.0, True),  # on the bar closing at H - 8h
    "at-late": (lambda th: 0.0, lambda th: th, True),  # on the bar opening at H - 8h
    "split": (lambda th: th / 2, lambda th: th / 2, True),  # halves add up to it exactly
    "sum": (lambda th: 0.6 * th, lambda th: 0.6 * th, True),  # neither bar alone reaches it
    "just-below": (lambda th: float(np.nextafter(th, 0.0)), lambda th: 0.0, False),
    "zero": (lambda th: 0.0, lambda th: 0.0, False),  # a settlement of exactly zero
    "cancelling": (lambda th: 4 * th, lambda th: -4 * th, False),
}


@SIDES
@pytest.mark.parametrize("threshold", [0.0003, 0.0005, 0.001])
@pytest.mark.parametrize("case", THRESHOLD_CASES)
def test_the_threshold_is_inclusive_on_the_sum_of_both_bars(d, threshold, case) -> None:
    early, late, trades = THRESHOLD_CASES[case]
    bars = market(d, {7: early(threshold), 8: late(threshold)}, 20)
    proxy = SettlementRebound(threshold=threshold).settlements(bars)["funding"].iloc[15]
    assert proxy == d * (early(threshold) + late(threshold))
    if case == "split":
        assert proxy == d * threshold
    assert held(bars, threshold=threshold) == (expect(d, {15: 1, 16: 1}) if trades else {})


@SIDES
@pytest.mark.parametrize("bar", [b for b in range(18) if b not in (7, 8)])
def test_only_the_two_bars_of_the_previous_settlement_count(d, bar) -> None:
    # Bars 15 and 16 hold the 16:00 settlement itself, bars 9-14 SOL's extra 2-hourly ones,
    # bars 0-6 older settlements: none of them opens a trade at 16:00 (or at 08:00, whose
    # funding bar at 23:00 the day before is missing).
    assert held(market(d, {bar: 0.01}, 18)) == {}


@SIDES
@pytest.mark.parametrize("bar", [7, 8])
def test_nan_funding_leaves_the_proxy_undefined(d, bar) -> None:
    funding = {7: 0.001, 8: 0.001, bar: math.nan}
    bars = market(d, funding, 20)
    assert math.isnan(SettlementRebound().settlements(bars)["funding"].iloc[15])
    assert held(bars) == {}


# --- missing bars and the first settlement --------------------------------------------------


@SIDES
@pytest.mark.parametrize(
    ("funded", "drop", "hold_bars", "expected"),
    [
        (7, (), 2, {15: 1, 16: 1}),
        (8, (7,), 2, {}),  # H - 9h missing: undefined, though bar 8 holds the settlement
        (7, (8,), 2, {}),  # H - 8h missing
        (7, (15,), 2, {}),  # no decision bar: no trade, and bar 16 stays flat
        (7, (16,), 2, {15: 1}),  # the settlement bar is missing: bar 17 is already flat
        (7, (16,), 4, {15: 1, 17: 1, 18: 1}),  # the hours are counted by the clock
        (7, (17,), 2, {15: 1, 16: 1}),  # the exit bar is missing: bar 18 is flat
        (7, tuple(range(9, 15)), 2, {15: 1, 16: 1}),  # looked up by time, not by position
        (7, (6, 14), 2, {15: 1, 16: 1}),  # the bars just outside the funding window
    ],
    ids=[
        "complete",
        "no-early-bar",
        "no-late-bar",
        "no-decision-bar",
        "no-settlement-bar",
        "no-settlement-bar-hold-4",
        "no-exit-bar",
        "no-bars-in-between",
        "no-neighbours",
    ],
)
def test_missing_bars_around_a_settlement(d, funded, drop, hold_bars, expected) -> None:
    bars = market(d, {funded: 0.001}, 20, drop=drop)
    assert held(bars, hold_bars=hold_bars) == expect(d, expected)


@SIDES
@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (0, {15: 1, 16: 1, 23: -1, 24: -1}),
        (7, {15: 1, 16: 1, 23: -1, 24: -1}),  # the first bar is the early funding bar
        (8, {23: -1, 24: -1}),  # 16:00 has no previous settlement on record: 00:00 is first
        (15, {23: -1, 24: -1}),  # the first bar is a decision bar without funding history
        (16, {}),  # the funding bar 15 of 00:00 is missing too
        (23, {}),
    ],
)
def test_the_first_settlement_needs_both_bars_of_the_one_before(d, start, expected) -> None:
    bars = market(d, {8: 0.001, 16: -0.001}, 30, start=start)
    assert held(bars) == expect(d, expected)


# --- random markets and bar times -----------------------------------------------------------------


def settlement_market(
    seed: int, n: int = 600, gaps: float = 0.0, extra: bool = False
) -> pd.DataFrame:
    """``random_bars`` on 1h with funding booked the way ``align_funding`` books Binance's.

    Each 00:00, 08:00 and 16:00 UTC settlement goes on the bar closing at the hour (57%) or
    the one opening at it (43%). Rates are often far from zero, and 30% are exactly zero or
    exactly at a grid threshold. ``extra`` adds SOL-style settlements at the other even hours;
    ``gaps`` drops that share of the bars at random.
    """
    bars = random_bars(seed, n=n, freq="1h")
    rng = np.random.default_rng(30_000 + seed)
    funding = np.zeros(n)
    for t, opened in enumerate(bars.index):
        if opened.hour % (2 if extra else 8):
            continue
        if rng.random() < 0.3:
            rate = rng.choice([0.0, 0.0003, 0.0005, 0.001]) * rng.choice([-1.0, 1.0])
        else:
            rate = rng.normal(2e-4, 6e-4)
        funding[t - 1 if t and rng.random() < 0.57 else t] += rate
    bars["funding_rate"] = funding
    return bars[rng.random(n) >= gaps] if gaps else bars


@pytest.mark.parametrize("zone", ["America/New_York", "Asia/Kolkata", "naive"])
def test_hours_are_utc_whatever_the_index_timezone(zone) -> None:
    # Two weeks of bars from 2024-03-03 cross New York's switch to summer time (2024-03-10).
    utc = settlement_market(3, n=24 * 14)
    utc.index = utc.index + pd.Timedelta(days=62)
    other = utc.copy()
    other.index = utc.index.tz_localize(None) if zone == "naive" else utc.index.tz_convert(zone)
    for params in ({}, {"threshold": 0.0003, "hold_bars": 7}):
        expected = SettlementRebound(**params).target_position(utc)
        assert (expected != 0).sum() > 10
        target = SettlementRebound(**params).target_position(other)
        assert target.index.equals(other.index)
        assert target.tolist() == expected.tolist()


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_any_timestamp_resolution(unit) -> None:
    # The stored data is indexed in milliseconds, the synthetic bars in microseconds.
    bars = market(1, WORKED, WORKED_N, unit=unit)
    assert bars.index.unit == unit
    assert held(bars) == {15: 1, 16: 1, 23: -1, 24: -1, 39: -1, 40: -1, 55: 1, 56: 1}


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2024-01-01 00:30", periods=40, freq="1h", tz="UTC"),
        pd.date_range("2024-01-01", periods=80, freq="30min", tz="UTC"),
        pd.date_range("2024-01-01 00:00:00.002", periods=40, freq="1h", tz="UTC"),
        market(1, WORKED, 40).index[:-1].append(pd.DatetimeIndex([START + 39.5 * HOUR])),
    ],
    ids=["half-past", "30-minute-bars", "2-ms-late", "one-bar-off"],
)
def test_bars_off_the_whole_hour_are_refused(index) -> None:
    bars = pd.DataFrame({"close": 100.0, "funding_rate": 0.001}, index=index)
    with pytest.raises(ValueError, match="whole UTC hours"):
        SettlementRebound().target_position(bars)
    with pytest.raises(ValueError, match="whole UTC hours"):
        SettlementRebound().settlements(bars)


@pytest.mark.parametrize(
    ("freq", "offset"),
    [("4h", 0), ("4h", 3), ("2h", 1), ("1D", 0), ("1D", 23)],
)
def test_bars_two_or_more_hours_apart_never_trade(freq, offset) -> None:
    # Bars opening at 03:00, 07:00, 11:00, 15:00, 19:00 and 23:00 include decision bars, but the
    # two funding bars an hour apart never both exist.
    index = pd.date_range(START + offset * HOUR, periods=200, freq=freq)
    bars = pd.DataFrame({"close": 100.0, "funding_rate": 0.01}, index=index)
    signals = SettlementRebound().settlements(bars)
    assert signals["decision"].any() == (offset % 2 == 1)
    assert signals["funding"].isna().all()
    assert (SettlementRebound(threshold=1e-9).target_position(bars) == 0).all()


def test_empty_and_one_bar_markets_are_flat() -> None:
    bars = market(1, WORKED, WORKED_N)
    for part in (bars.iloc[:0], bars.iloc[15:16], bars.iloc[7:16:8]):
        target = SettlementRebound().target_position(part)
        assert target.index.equals(part.index)
        assert (target == 0).all()
    assert SettlementRebound().settlements(bars.iloc[:0]).empty


# --- parameters and specs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"threshold": 0.0}, "^threshold"),
        ({"threshold": -0.0005}, "^threshold"),
        ({"threshold": math.nan}, "^threshold"),
        ({"threshold": math.inf}, "^threshold"),
        ({"hold_bars": 0}, "^hold_bars"),
        ({"hold_bars": -2}, "^hold_bars"),
        ({"hold_bars": 8}, "^hold_bars"),
        ({"hold_bars": 2.0}, "^hold_bars"),
        ({"hold_bars": 2.5}, "^hold_bars"),
        ({"hold_bars": True}, "^hold_bars"),
        ({"hold_bars": "2"}, "^hold_bars"),
    ],
)
def test_invalid_parameters_are_refused(params, message) -> None:
    with pytest.raises(ValueError, match=message):
        SettlementRebound(**params)


def test_the_edges_of_the_valid_ranges_build() -> None:
    SettlementRebound(threshold=1e-12, hold_bars=1)
    SettlementRebound(threshold=5.0, hold_bars=7)
    # Grid values read back from a results table arrive as numpy numbers.
    built = SettlementRebound(threshold=np.float64(0.001), hold_bars=np.int64(4))
    assert built.hold_bars == 4 and built.threshold == 0.001


def test_the_card_spec_and_grid_are_the_class_defaults_and_build() -> None:
    card = read_card(CARD)
    assert card.spec == CARD_SPEC
    assert card.timeframe == "1h"
    assert card.combinations() == GRID
    for cell in GRID:
        assert from_spec({**CARD_SPEC, **cell}) == SettlementRebound(**cell)


def test_spec_round_trip_and_card_defaults() -> None:
    assert to_spec(SettlementRebound()) == CARD_SPEC
    assert from_spec(CARD_SPEC) == SettlementRebound()
    custom = SettlementRebound(threshold=0.00125, hold_bars=7)
    assert from_spec(to_spec(custom)) == custom
    assert from_spec(json.dumps(to_spec(custom))) == custom
    assert str(SettlementRebound()) == "i005_settlement_rebound(threshold=0.0005, hold_bars=2)"


# --- engine ---------------------------------------------------------------------------------


@SIDES
@pytest.mark.parametrize("booked", [15, 16], ids=["settled-on-the-decision-bar", "settled-late"])
def test_backtest_holds_the_trade_for_two_hours_and_pays_the_settlement_it_holds(d, booked) -> None:
    # The 08:00 rate (bar 7) opens a trade at bar 15's close. The 16:00 settlement of 0.0004
    # is charged to the trade only when it is booked on bar 16, which the trade holds.
    funding = {7: 0.0006, booked: 0.0004}
    bars = market(d, funding, 20, closes={16: 101.0, 17: 102.0, 18: 102.0, 19: 102.0})
    rule = SettlementRebound()
    result = backtest(rule, bars, CostModel(fee_bps=5.0, slippage_bps=3.0))
    pd.testing.assert_series_equal(
        result.ledger["target"], rule.target_position(bars), check_names=False
    )
    position = result.ledger["position"]
    assert {t: p for t, p in enumerate(position) if p} == expect(d, {16: 1, 17: 1})
    paid = result.ledger["funding_cost"]
    assert paid.iloc[16] == (0.0004 if booked == 16 else 0.0)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert (trade["direction"], trade["bars"], trade["open"]) == (d, 2, False)
    # It earns the bars opening at 16:00 and 17:00 and is closed at 18:00.
    assert (trade["entry"], trade["exit"]) == (START + 16 * HOUR, START + 18 * HOUR)
    close = bars["close"]
    price = d * (close.iloc[16] / close.iloc[15] - 1) + d * (close.iloc[17] / close.iloc[16] - 1)
    charged = 0.0004 if booked == 16 else 0.0
    assert trade["return"] == pytest.approx(price - 2 * 0.0008 - charged, abs=1e-12)


# --- no lookahead ---------------------------------------------------------------------------

# Bar t opens at hour t: cuts 8/16/232 end on a decision bar, 15/231 one hour before it, 17/233
# on the settlement bar, 18/234 on the exit bar, 9 on the late funding bar of 16:00.
CUTS = (0, 1, 2, 7, 8, 9, 15, 16, 17, 18, 23, 24, 25, 26, 100, 231, 232, 233, 234, 400, 599)
LOOKAHEAD_CASES = [
    SettlementRebound(),
    SettlementRebound(threshold=0.0003, hold_bars=7),
    SettlementRebound(threshold=0.001, hold_bars=1),
]


def assert_no_lookahead(rule: SettlementRebound, bars: pd.DataFrame) -> None:
    full = rule.target_position(bars)
    assert (full > 0).any() and (full < 0).any()
    for cut in CUTS:
        pd.testing.assert_series_equal(rule.target_position(bars.iloc[:cut]), full.iloc[:cut])


@pytest.mark.parametrize("gaps", [0.0, 0.05], ids=["complete", "gaps"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("rule", LOOKAHEAD_CASES, ids=str)
def test_no_lookahead_on_random_markets(rule, seed, gaps) -> None:
    assert_no_lookahead(rule, settlement_market(seed, gaps=gaps, extra=seed % 2 == 1))


# --- plain-Python reference -----------------------------------------------------------------


def reference_targets(
    bars: pd.DataFrame, threshold: float = 0.0005, hold_bars: int = 2
) -> list[float]:
    """The card's rules in plain Python, written separately from the module.

    Walk the bars; a bar whose close (open + 1h) is a settlement hour on the UTC clock is a
    decision bar; read the previous settlement off the two bars that can hold it; mark the
    hours the trade is held.
    """
    hour = pd.Timedelta(hours=1)
    utc = [t.tz_convert("UTC") for t in bars.index]
    funding = dict(zip(utc, bars["funding_rate"], strict=True))
    marked: dict[pd.Timestamp, float] = {}
    for opened in utc:
        settles = opened + hour
        if settles.hour not in (0, 8, 16):
            continue
        previous = settles - 8 * hour
        owners = (previous - hour, previous)  # the bar closing at it and the bar opening at it
        if owners[0] not in funding or owners[1] not in funding:
            continue
        rate = funding[owners[0]] + funding[owners[1]]
        if rate >= threshold:
            side = 1.0
        elif rate <= -threshold:
            side = -1.0
        else:
            continue
        for k in range(hold_bars):
            marked[opened + k * hour] = side
    return [marked.get(t, 0.0) for t in utc]


EXTRA_CASES = [
    {"threshold": 0.0001, "hold_bars": 7},
    {"threshold": 0.002, "hold_bars": 3},
    {"threshold": 0.00025, "hold_bars": 5},
    {"threshold": 0.0005, "hold_bars": 6},
]


@pytest.mark.parametrize("extra", [False, True], ids=["8-hourly", "with-2-hourly"])
@pytest.mark.parametrize("gaps", [0.0, 0.05], ids=["complete", "gaps"])
@pytest.mark.parametrize("seed", range(10))
def test_matches_the_plain_python_reference_on_random_markets(seed, gaps, extra) -> None:
    bars = settlement_market(seed, gaps=gaps, extra=extra)
    negated = bars.assign(funding_rate=-bars["funding_rate"])
    sides: set[float] = set()
    for params in GRID + EXTRA_CASES:
        rule = SettlementRebound(**params)
        expected = reference_targets(bars, **params)
        assert rule.target_position(bars).tolist() == expected, params
        # Negated funding flips every trade and keeps its hours.
        assert rule.target_position(negated).tolist() == [-x for x in expected], params
        sides |= set(expected)
    assert sides == {-1.0, 0.0, 1.0}


@pytest.mark.parametrize(
    "costs", [ZERO_COSTS, EXCHANGE_COSTS["binanceusdm"]], ids=["free", "costs"]
)
@pytest.mark.parametrize("mirror", [False, True], ids=["as-drawn", "mirrored"])
@pytest.mark.parametrize("seed", range(5))
def test_backtests_match_the_reference(seed, mirror, costs) -> None:
    bars = settlement_market(seed, n=1200, gaps=0.02)
    if mirror:  # prices reflected (reciprocals) and funding negated
        bars = bars.assign(
            open=1e4 / bars["open"],
            high=1e4 / bars["low"],
            low=1e4 / bars["high"],
            close=1e4 / bars["close"],
            funding_rate=-bars["funding_rate"],
        )
    for params in ({}, {"threshold": 0.0003, "hold_bars": 4}):
        expected = pd.Series(reference_targets(bars, **params), index=bars.index)
        reference = run_backtest(bars, expected, costs)
        result = backtest(SettlementRebound(**params), bars, costs)
        pd.testing.assert_frame_equal(result.ledger, reference.ledger)
        pd.testing.assert_frame_equal(result.trades, reference.trades)
        directions = set(result.trades["direction"])
        assert len(result.trades) > 10 and directions == {-1, 1}


# --- real data ------------------------------------------------------------------------------


def test_smoke_on_real_btc_data() -> None:
    criteria = load_criteria(Workspace(ROOT).criteria_path)
    exchange = criteria.get("data.exchange")
    try:  # development bars only: they end where the holdout starts
        bars = load_bars(exchange, "BTC", "1h", end=criteria.get("data.dev_end"))
    except FileNotFoundError:
        pytest.skip("BTC 1h data not downloaded")
    rule = SettlementRebound()
    result = backtest(rule, bars, EXCHANGE_COSTS[exchange])
    assert set(result.ledger["target"].unique()) <= {-1.0, 0.0, 1.0}
    trades = result.trades
    assert len(trades) > 0
    assert set(trades["direction"]) <= {-1, 1}
    # Every trade earns the hours from a settlement at 00:00, 08:00 or 16:00 UTC on; the
    # development bars have no gaps, so each lasts exactly hold_bars bars.
    assert (pd.DatetimeIndex(trades["entry"]).hour % 8 == 0).all()
    assert (trades["bars"] == rule.hold_bars).all()
    assert np.isfinite(result.equity).all()
