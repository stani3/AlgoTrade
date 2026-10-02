"""i004_funding_crowding_short: short the onset of funding crowding, never long.

Hand-computed cases are daily markets with a flat close of 100 whose bars span +-1, so ATR(14) is
exactly 2 from bar 13 on and a short's stop sits at 106. Funding is written in units of
``V = 2**-15`` (about 0.003%), which keeps every funding sum and percentile rank exact. Most cases
use ``funding_days=1`` (the funding sum is the bar's own funding) and ``history_days=8`` (ranks
over the last 8 bars, defined once 2 values exist); the neutral floor is then 0.0003 + 1e-9,
about 9.83 V, so 9 V and below is never crowded and 10 V and above can be.

The rule is short only, so the long/short mirror of every hand case checks that it never goes
long: ``d = +1`` runs a case as written and ``d = -1`` its mirror image (prices reflected about
100 and funding negated, so crowded longs become crowded shorts), which must stay flat.
"""

from __future__ import annotations

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
from algotrade.indicators import atr
from algotrade.research.cards import read_card
from algotrade.research.criteria import load_criteria
from algotrade.research.workspace import Workspace
from algotrade.strategies import from_spec, to_spec
from algotrade.strategies.ideas.i004_funding_crowding_short import FundingCrowdingShort

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "research" / "ideas" / "i004-funding-crowding-short" / "v1" / "idea.md"
SIDES = pytest.mark.parametrize("d", [1, -1], ids=["as-written", "mirrored"])
CARD_SPEC = {
    "type": "i004_funding_crowding_short",
    "funding_days": 7,
    "quantile": 0.8,
    "hold_bars": 7,
    "history_days": 365,
    "exit_quantile": 0.5,
    "neutral_daily_funding": 0.0003,
    "stop_atr": 3.0,
}
GRID = [
    {"quantile": q, "funding_days": f, "hold_bars": h}
    for q in (0.7, 0.8, 0.9)
    for f in (3, 7, 14)
    for h in (3, 7, 14)
]

V = 2.0**-15
SMALL = {"funding_days": 1, "history_days": 8, "hold_bars": 3}
# Bars 0..13, all below the floor; ATR(14) is first defined on bar 13.
PRELUDE = [t % 7 + 1 for t in range(14)]  # 1..7, 1..7 V


def market(
    d: int,
    funding: list[float],
    closes: list[float] | None = None,
    spreads: list[float] | None = None,
    freq: str = "1D",
) -> pd.DataFrame:
    """Bars with the given per-bar funding; each bar opens and closes at its close and spans
    close +- spread (default a flat 100 +- 1). ``d = -1`` reflects prices about 100 and negates
    funding."""
    n = len(funding)
    closes = [100.0] * n if closes is None else closes
    spreads = [1.0] * n if spreads is None else spreads

    def price(p: float) -> float:
        return 100.0 + d * (p - 100.0)

    rows = []
    for close, spread in zip(closes, spreads, strict=True):
        ends = (price(close + spread), price(close - spread))
        rows.append((price(close), max(ends), min(ends), price(close)))
    return bars_from(rows, funding=[d * f for f in funding], freq=freq)


def strategy(**params) -> FundingCrowdingShort:
    return FundingCrowdingShort(**{**SMALL, **params})


def after_prelude(
    d: int,
    funding: list[float],
    closes: list[float] | None = None,
    spreads: list[float] | None = None,
    spread: float = 1.0,
    **params,
) -> list[float]:
    """Targets on the bars of ``funding`` (in V) placed after PRELUDE, which must stay flat."""
    n, k = len(funding), len(PRELUDE)
    bars = market(
        d,
        [x * V for x in PRELUDE + funding],
        closes=[100.0] * k + ([100.0] * n if closes is None else closes),
        spreads=[spread] * k + ([spread] * n if spreads is None else spreads),
    )
    target = strategy(**params).target_position(bars).tolist()
    assert target[:k] == [0.0] * k
    return target[k:]


def expect(d: int, values: list[int]) -> list[int]:
    """The hand-worked targets as written; the mirror image never trades."""
    return values if d == 1 else [0] * len(values)


# --- crowding, worked by hand ---------------------------------------------------------------


def test_rank_counts_ties_and_crowding_starts_at_the_quantile() -> None:
    # history_days 5: ranks over the last 5 bars, defined from 5 // 4 = 1 value.
    funding = [x * V for x in [12, 10, 11, 11, 12, 13, 9, 12]]
    bars = market(1, funding)
    signals = strategy(history_days=5).crowding(bars)
    assert signals["funding_sum"].tolist() == funding
    # t3: 10, 11, 11 of [12, 10, 11, 11] are <= 11 (the tie counts): 3/4 < 0.8.
    # t7: 11, 12, 9, 12 of [11, 12, 13, 9, 12] are <= 12: exactly 4/5 = 0.8.
    assert signals["rank"].tolist() == [1.0, 1 / 2, 2 / 3, 3 / 4, 1.0, 1.0, 1 / 5, 4 / 5]
    # 9 V on t6 is also below the floor; the bar before the first counts as not crowded.
    assert signals["crowded"].tolist() == [True, False, False, False, True, True, False, True]
    assert signals["onset"].tolist() == [True, False, False, False, True, False, False, True]
    # Every onset comes before ATR(14) exists, so nothing trades.
    assert (strategy(history_days=5).target_position(bars) == 0).all()
    # The mirror image (funding negated) is never crowded.
    assert not strategy(history_days=5).crowding(market(-1, funding))["crowded"].any()


def test_funding_sum_and_rank_warm_up_by_hand() -> None:
    # funding_days 2: F is the sum of two bars; history_days 8 needs 8 // 4 = 2 F values.
    # In units of W = 2**-12 the floor is (0.0006 + 1e-9) / W, about 2.46 W.
    w = 2.0**-12
    bars = market(1, [x * w for x in [1, 1, 2, 0, 3]])
    signals = strategy(funding_days=2).crowding(bars)
    funding_sum = signals["funding_sum"]
    assert math.isnan(funding_sum.iloc[0])
    assert funding_sum.iloc[1:].tolist() == [2 * w, 3 * w, 2 * w, 3 * w]
    rank = signals["rank"]
    assert rank.iloc[:2].isna().all()  # one F value on bar 1 is not enough
    # t3: 2 of [2, 3, 2] are <= 2; t4: all 4 of [2, 3, 2, 3] are <= 3.
    assert rank.iloc[2:].tolist() == [1.0, 2 / 3, 1.0]
    assert signals["crowded"].tolist() == [False, False, True, False, True]
    assert signals["onset"].tolist() == [False, False, True, False, True]


# --- targets, worked by hand ----------------------------------------------------------------


@SIDES
def test_tiny_market_worked_by_hand(d) -> None:
    # Bar 14: 16 V tops [1..7, 16] -> rank 1, above the floor: an onset, short (s = 14).
    # Bars 15-16 stay crowded (16 V ties rank 1); bar 17 is s + 3: covered by time. Bar 18 is
    # still crowded but not an onset: flat. Bar 19: 1 V ranks 1/8, crowding lapses. Bar 20:
    # 16 V ranks 8/8 again: a fresh onset, short.
    funding = [16, 16, 16, 16, 16, 1, 16, 16]
    expected = [-1, -1, -1, 0, 0, 0, -1, -1]
    assert after_prelude(d, funding) == expect(d, expected)


@SIDES
def test_normalised_funding_covers_and_a_fresh_onset_shorts_after_one_flat_bar(d) -> None:
    # Bar 15: 1 V in [2..7, 16, 1] ranks 1/8 < 0.5: covered. Bar 16: 16 V in
    # [3..7, 16, 1, 16] ranks 8/8 and bar 15 was not crowded: short again (s = 16) until 19.
    funding = [16, 1, 16, 16, 16, 16]
    assert after_prelude(d, funding) == expect(d, [-1, 0, -1, -1, -1, 0])


@SIDES
@pytest.mark.parametrize(
    ("dip", "expected"),
    [(4.5, [-1, -1, -1, 0]), (3.5, [-1, 0, -1, -1])],
    ids=["rank-0.5-holds", "rank-0.375-covers"],
)
def test_the_funding_exit_needs_a_rank_strictly_below_exit_quantile(d, dip, expected) -> None:
    # Bar 15's window is [2, 3, 4, 5, 6, 7, 16, dip]: 4.5 V has 2, 3, 4 and itself <= it
    # (4/8 = 0.5, not below 0.5); 3.5 V has 2, 3 and itself (3/8). After the cover at 15,
    # bar 16 (16 V, rank 1) is a fresh onset.
    assert after_prelude(d, [16, dip, 16, 16]) == expect(d, expected)


@SIDES
def test_an_onset_while_short_is_ignored_and_does_not_restart_the_clock(d) -> None:
    # Bar 15: 9 V ranks 7/8 (no funding exit) but is below the floor, so crowding lapses; bar
    # 16 (16 V) is an onset while short: ignored, and the time exit stays at s + 3 = 17.
    funding = [16, 9, 16, 16, 16]
    assert after_prelude(d, funding) == expect(d, [-1, -1, -1, 0, 0])


@SIDES
def test_an_onset_on_the_covering_bar_is_lost(d) -> None:
    # Bars 15-16 (9 V, rank 7/8) are not crowded; bar 17 (16 V) is an onset on the time-exit
    # bar: it covers and does not re-short, and bars 18-19 are crowded without an onset. Bar 20
    # (1 V, rank 1/8) lapses and bar 21 (16 V, rank 8/8) is the next onset.
    funding = [16, 9, 9, 16, 16, 16, 1, 16]
    assert after_prelude(d, funding) == expect(d, [-1, -1, -1, 0, 0, 0, 0, -1])


@SIDES
@pytest.mark.parametrize(
    ("hold_bars", "expected"),
    [(1, [-1, 0, 0]), (7, [-1] * 7 + [0] * 3)],
    ids=["one-bar", "card-default"],
)
def test_a_trade_lasts_hold_bars_while_funding_stays_crowded(d, hold_bars, expected) -> None:
    funding = [16] * len(expected)
    assert after_prelude(d, funding, hold_bars=hold_bars) == expect(d, expected)


# --- the stop -------------------------------------------------------------------------------


@SIDES
@pytest.mark.parametrize(
    ("close", "expected"),
    [(106.0, [-1, 0, 0, 0]), (float(np.nextafter(106.0, 0.0)), [-1, -1, -1, 0])],
    ids=["at-the-stop-covers", "just-below-holds"],
)
def test_a_close_at_the_stop_covers(d, close, expected) -> None:
    bars = market(d, [x * V for x in PRELUDE + [16]])
    assert atr(bars["high"], bars["low"], bars["close"]).iloc[13:].tolist() == [2.0, 2.0]
    # Stop = 100 + 3 x 2 = 106; after a stop with funding still crowded there is no onset.
    closes = [100.0, close, 100.0, 100.0]
    assert after_prelude(d, [16] * 4, closes=closes) == expect(d, expected)


@SIDES
def test_the_stop_is_fixed_at_entry_and_checked_on_closes(d) -> None:
    # Bar 15 trades up to 115 but closes at 105, and its range lifts ATR to 2 + 18/14: the stop
    # stays at 106 (it would be about 114.9 if it moved), so bar 16's close of 106 covers.
    funding = [16, 16, 16, 16]
    closes = [100.0, 105.0, 106.0, 100.0]
    spreads = [1.0, 10.0, 1.0, 1.0]
    target = after_prelude(d, funding, closes=closes, spreads=spreads)
    assert target == expect(d, [-1, -1, 0, 0])


@SIDES
def test_zero_atr_puts_the_stop_at_the_entry_close(d) -> None:
    bars = market(d, [x * V for x in PRELUDE + [16]], spreads=[0.0] * 15)
    assert atr(bars["high"], bars["low"], bars["close"]).iloc[14] == 0.0
    # Stop = 100 + 3 x 0: the next close of 100 is at the stop.
    assert after_prelude(d, [16] * 4, spread=0.0) == expect(d, [-1, 0, 0, 0])


@SIDES
def test_after_a_stop_a_new_short_needs_crowding_to_lapse_and_return(d) -> None:
    # Stopped at 15 (close 106); bar 16 is still crowded, bar 17 (1 V, rank 1/8) lapses, bar 18
    # (16 V, rank 8/8) is an onset: short with a new stop above 106, so 106 closes hold it.
    funding = [16, 16, 16, 1, 16, 16]
    closes = [100.0] + [106.0] * 5
    assert after_prelude(d, funding, closes=closes) == expect(d, [-1, 0, 0, 0, -1, -1])


# --- warm-up --------------------------------------------------------------------------------


@SIDES
@pytest.mark.parametrize(
    ("start", "expected"),
    [(12, [0] * 6 + [-1, -1]), (13, [-1, -1, -1, 0, 0, 0, -1, -1])],
    ids=["onset-before-atr-is-lost", "onset-on-the-first-atr-bar-trades"],
)
def test_an_onset_needs_a_defined_atr(d, start, expected) -> None:
    # Crowding from bar `start`; 1 V lapses it 5 bars later and 16 V brings a fresh onset.
    # ATR(14) is undefined on bar 12 (13 bars), so that onset is never traded, and the rest
    # of the crowded stretch has no onset.
    funding = PRELUDE[:start] + [16] * 5 + [1, 16, 16]
    bars = market(d, [x * V for x in funding])
    assert math.isnan(atr(bars["high"], bars["low"], bars["close"]).iloc[12])
    target = strategy().target_position(bars).tolist()
    assert target[:start] == [0.0] * start
    assert target[start:] == expect(d, expected)


@SIDES
@pytest.mark.parametrize(
    ("freq", "funding_days", "history_days", "first"),
    [
        ("1D", 7, 365, 96),
        ("1D", 3, 365, 92),
        ("1D", 14, 365, 103),
        ("1D", 2, 60, 15),
        ("4h", 7, 365, 587),
    ],
)
def test_the_first_tradable_bar_follows_the_rank_warm_up(
    d, freq, funding_days, history_days, first
) -> None:
    # F is first defined on bar nW - 1 and the rank once nY // 4 F values exist, so on bar
    # (nW - 1) + (nY // 4 - 1): on 1d defaults 6 + 90 = 96, on 4h 41 + 546 = 587. A spike of
    # funding inside the window F covers on that bar is shorted there, even if the spike came
    # earlier (an undefined rank counts as not crowded); a window that ends before it never is.
    n_window = round(funding_days * {"1D": 1, "4h": 6}[freq])
    shorts = FundingCrowdingShort(funding_days=funding_days, history_days=history_days)
    spikes = {first - n_window: [], first - n_window + 1: [first], first: [first]}
    for spike, entries in spikes.items():
        funding = [0.0] * (first + 30)
        funding[spike] = 2.0**-6
        target = shorts.target_position(market(d, funding, freq=freq))
        starts = np.flatnonzero(target.diff().fillna(target) < 0).tolist()
        assert starts == (entries if d == 1 else [])
        assert target.iloc[first] == (-1.0 if entries and d == 1 else 0.0)


# --- the neutral floor ----------------------------------------------------------------------

FLOOR = 0.0003 * 1 + 1e-9  # funding_days 1


@SIDES
@pytest.mark.parametrize(
    ("funding", "expected"),
    [(FLOOR, [0, 0]), (float(np.nextafter(FLOOR, 1.0)), [-1, -1])],
    ids=["at-the-floor", "just-above"],
)
def test_funding_must_be_strictly_above_the_neutral_floor(d, funding, expected) -> None:
    bars = market(d, [x * V for x in PRELUDE] + [funding] * 2)
    target = strategy().target_position(bars).tolist()
    assert target == [0.0] * 14 + expect(d, expected)


@SIDES
@pytest.mark.parametrize(
    ("funding", "expected"),
    [(0.0, [0, 0]), (1e-9, [0, 0]), (float(np.nextafter(1e-9, 1.0)), [-1, -1])],
    ids=["zero", "1e-9", "just-above-1e-9"],
)
def test_a_zero_neutral_rate_still_needs_funding_above_1e9(d, funding, expected) -> None:
    bars = market(d, [0.0] * 14 + [funding] * 2)
    target = strategy(neutral_daily_funding=0.0).target_position(bars).tolist()
    assert target == [0.0] * 14 + expect(d, expected)


@pytest.mark.parametrize("neutral", [0.0003, 0.0], ids=["neutral", "zero-neutral"])
def test_negative_funding_is_never_crowded_however_high_it_ranks(neutral) -> None:
    funding = [-(30 - t) * V for t in range(30)]  # every bar a new high of its window
    bars = market(1, funding)
    shorts = strategy(neutral_daily_funding=neutral)
    assert (shorts.crowding(bars)["rank"].iloc[1:] == 1.0).all()
    assert (shorts.target_position(bars) == 0).all()


@SIDES
@pytest.mark.parametrize("excess", [0.0, 1e-9], ids=["exactly-neutral", "neutral-plus-1e-9"])
def test_the_tolerance_keeps_exactly_neutral_funding_out(d, excess) -> None:
    # Three settlements of 0.01% a day: their 7-day float sum lands just above 0.0003 x 7, and
    # only the 1e-9 keeps it out. 1e-9 more a day (7e-9 a week) is crowded on the first ranked
    # bar, 6 + (60 // 4 - 1) = 20, and held for 7 bars.
    daily = 0.0001 + 0.0001 + 0.0001 + excess
    bars = market(d, [daily] * 30)
    shorts = strategy(funding_days=7, history_days=60, hold_bars=7)
    funding_sum = shorts.crowding(market(1, [daily] * 30))["funding_sum"].dropna()
    assert (funding_sum > 0.0003 * 7).all()
    target = shorts.target_position(bars).tolist()
    expected = [0] * 30 if excess == 0.0 else [0] * 20 + [-1] * 7 + [0] * 3
    assert target == expect(d, expected)


# --- edges ----------------------------------------------------------------------------------


def test_one_bar_and_empty_markets_are_flat() -> None:
    one = market(1, [16 * V])
    assert FundingCrowdingShort().target_position(one).tolist() == [0.0]
    assert strategy().target_position(one).tolist() == [0.0]
    empty = FundingCrowdingShort().target_position(one.iloc[:0])
    assert empty.empty and empty.index.equals(one.index[:0])


def test_a_funding_window_shorter_than_one_bar_is_refused() -> None:
    # funding_days 0.4 rounds to 0 daily bars but 2 four-hour bars.
    shorts = FundingCrowdingShort(funding_days=0.4, history_days=30)
    daily = market(1, [0.0] * 20)
    with pytest.raises(ValueError, match="at least one bar"):
        shorts.target_position(daily)
    with pytest.raises(ValueError, match="at least one bar"):
        shorts.crowding(daily)
    assert (shorts.target_position(market(1, [0.0] * 20, freq="4h")) == 0).all()


# --- never long -----------------------------------------------------------------------------


def crowding_market(seed: int, n: int = 600, freq: str = "1D") -> pd.DataFrame:
    """``random_bars`` with funding that drifts in and out of crowding.

    Daily funding is the neutral 0.03% plus AR(1) noise, split over the day's bars and rounded
    to a dyadic grid, so every funding sum is exact and equal sums tie exactly.
    """
    bars = random_bars(seed, n=n, freq=freq)
    rng = np.random.default_rng(10_000 + seed)
    per_day, unit = {"1D": (1, 2.0**-14), "4h": (6, 2.0**-16)}[freq]
    noise, level = 0.0, []
    for _ in range(n):
        noise = 0.9 * noise + rng.normal()
        level.append((0.0003 + 0.0003 * noise) / per_day)
    bars["funding_rate"] = np.round(np.asarray(level) / unit) * unit
    return bars


def mirrored(bars: pd.DataFrame, prices: bool = True, funding: bool = True) -> pd.DataFrame:
    """Prices reflected (reciprocals, so a rally becomes a fall) and/or funding negated."""
    out = bars.copy()
    if prices:
        out["open"], out["close"] = 1e4 / bars["open"], 1e4 / bars["close"]
        out["high"], out["low"] = 1e4 / bars["low"], 1e4 / bars["high"]
    if funding:
        out["funding_rate"] = -bars["funding_rate"]
    return out


NEVER_LONG_CASES = [
    FundingCrowdingShort(),
    FundingCrowdingShort(funding_days=2, history_days=30, quantile=0.6, exit_quantile=0.3),
    FundingCrowdingShort(funding_days=3, history_days=60, neutral_daily_funding=0.0),
]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shorts", NEVER_LONG_CASES, ids=str)
def test_never_long_on_random_markets_or_their_mirror_images(shorts, seed) -> None:
    bars = crowding_market(seed)
    assert (shorts.target_position(bars) < 0).any()
    for prices, funding in ((False, False), (True, False), (False, True), (True, True)):
        target = shorts.target_position(mirrored(bars, prices, funding))
        assert set(target.unique()) <= {-1.0, 0.0}


# --- parameters and specs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"funding_days": 0}, "^funding_days"),
        ({"funding_days": -7}, "^funding_days"),
        ({"funding_days": math.nan}, "^funding_days"),
        ({"funding_days": math.inf}, "^funding_days"),
        ({"history_days": 7}, "^history_days"),
        ({"history_days": 3}, "^history_days"),
        ({"history_days": math.nan}, "^history_days"),
        ({"history_days": math.inf}, "^history_days"),
        ({"quantile": 0.0}, "^quantile"),
        ({"quantile": -0.8}, "^quantile"),
        ({"quantile": 1.01}, "^quantile"),
        ({"quantile": math.nan}, "^quantile"),
        ({"exit_quantile": 0.0}, "^exit_quantile"),
        ({"exit_quantile": -0.5}, "^exit_quantile"),
        ({"exit_quantile": 0.8}, "^exit_quantile"),
        ({"exit_quantile": 0.9}, "^exit_quantile"),
        ({"exit_quantile": math.nan}, "^exit_quantile"),
        ({"hold_bars": 0}, "^hold_bars"),
        ({"hold_bars": -7}, "^hold_bars"),
        ({"hold_bars": 7.0}, "^hold_bars"),
        ({"hold_bars": 2.5}, "^hold_bars"),
        ({"hold_bars": True}, "^hold_bars"),
        ({"stop_atr": 0.0}, "^stop_atr"),
        ({"stop_atr": -3.0}, "^stop_atr"),
        ({"stop_atr": math.nan}, "^stop_atr"),
        ({"stop_atr": math.inf}, "^stop_atr"),
        ({"neutral_daily_funding": -0.0001}, "^neutral_daily_funding"),
        ({"neutral_daily_funding": math.nan}, "^neutral_daily_funding"),
        ({"neutral_daily_funding": math.inf}, "^neutral_daily_funding"),
    ],
)
def test_invalid_parameters_are_refused(params, message) -> None:
    with pytest.raises(ValueError, match=message):
        FundingCrowdingShort(**params)


def test_the_edges_of_the_valid_ranges_build() -> None:
    FundingCrowdingShort(quantile=1.0, exit_quantile=0.999)
    FundingCrowdingShort(neutral_daily_funding=0.0)
    FundingCrowdingShort(funding_days=0.1, history_days=0.2, hold_bars=1, stop_atr=1e-6)
    # Grid values read back from a results table arrive as numpy integers.
    assert FundingCrowdingShort(hold_bars=np.int64(3), funding_days=np.int64(14)).hold_bars == 3


def test_the_card_spec_and_grid_are_the_class_defaults_and_build() -> None:
    card = read_card(CARD)
    assert card.spec == CARD_SPEC
    assert card.combinations() == GRID
    for cell in GRID:
        assert from_spec({**CARD_SPEC, **cell}) == FundingCrowdingShort(**cell)


def test_spec_round_trip_and_card_defaults() -> None:
    assert to_spec(FundingCrowdingShort()) == CARD_SPEC
    assert from_spec(CARD_SPEC) == FundingCrowdingShort()
    custom = FundingCrowdingShort(
        funding_days=3,
        quantile=0.9,
        hold_bars=14,
        history_days=180.5,
        exit_quantile=0.4,
        neutral_daily_funding=0.0,
        stop_atr=2.5,
    )
    assert from_spec(to_spec(custom)) == custom
    assert from_spec(json.dumps(to_spec(custom))) == custom
    assert str(FundingCrowdingShort()) == (
        "i004_funding_crowding_short(funding_days=7, quantile=0.8, hold_bars=7, "
        "history_days=365, exit_quantile=0.5, neutral_daily_funding=0.0003, stop_atr=3.0)"
    )


# --- engine ---------------------------------------------------------------------------------


@SIDES
def test_backtest_holds_the_short_and_receives_the_funding(d) -> None:
    funding = [16, 16, 16, 16, 16]
    closes = [100.0, 98.0, 97.0, 99.0, 99.0]
    k = len(PRELUDE)
    bars = market(
        d, [x * V for x in PRELUDE + funding], closes=[100.0] * k + closes, spreads=[1.0] * 19
    )
    shorts = strategy()
    result = backtest(shorts, bars, CostModel(fee_bps=5.0, slippage_bps=3.0))
    target = shorts.target_position(bars)
    pd.testing.assert_series_equal(result.ledger["target"], target, check_names=False)
    position = result.ledger["position"]
    if d == -1:
        assert (position == 0).all() and result.trades.empty
        return
    s = k  # the short opens at bar 14's close and is covered by time at bar 17's
    assert position.iloc[s + 1 : s + 4].tolist() == [-1.0, -1.0, -1.0]
    assert (position.iloc[: s + 1] == 0).all() and (position.iloc[s + 4 :] == 0).all()
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert (trade["direction"], trade["bars"], trade["open"]) == (-1, 3, False)
    assert (trade["entry"], trade["exit"]) == (bars.index[s + 1], bars.index[s + 4])
    # Short the moves 100 -> 98 -> 97 -> 99, pay 8 bp in and out, receive 16 V on each bar.
    price = -((98 / 100 - 1) + (97 / 98 - 1) + (99 / 97 - 1))
    assert trade["return"] == pytest.approx(price - 2 * 0.0008 + 3 * 16 * V, abs=1e-12)


# --- no lookahead ---------------------------------------------------------------------------

LOOKAHEAD_CASES = [
    FundingCrowdingShort(),
    FundingCrowdingShort(funding_days=3, history_days=60, hold_bars=5),
    FundingCrowdingShort(
        funding_days=2,
        quantile=0.6,
        hold_bars=14,
        history_days=30,
        exit_quantile=0.55,
        stop_atr=0.5,
    ),
]
CUTS = (0, 1, 2, 13, 14, 20, 97, 98, 250, 333, 499)


def assert_no_lookahead(shorts: FundingCrowdingShort, bars: pd.DataFrame) -> None:
    full = shorts.target_position(bars)
    assert (full < 0).any()
    for cut in CUTS:
        pd.testing.assert_series_equal(shorts.target_position(bars.iloc[:cut]), full.iloc[:cut])


def test_no_lookahead_on_the_fixture_bars(bars) -> None:
    # 4h bars: a one-day funding window (6 bars) ranked over 20 days (120 bars).
    assert_no_lookahead(FundingCrowdingShort(funding_days=1, history_days=20), bars)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shorts", LOOKAHEAD_CASES, ids=str)
def test_no_lookahead_on_random_markets(shorts, seed) -> None:
    assert_no_lookahead(shorts, crowding_market(seed, n=500))


# --- plain-Python reference -----------------------------------------------------------------


def reference_ranks(
    bars: pd.DataFrame, funding_days: float, history_days: float
) -> tuple[list[float | None], list[float | None]]:
    """The funding sums and their trailing percentile ranks, counted out by hand.

    Plain Python sums equal pandas' rolling sums exactly on ``crowding_market``'s dyadic
    funding.
    """
    per_day = pd.Timedelta(days=1) / (bars.index[1] - bars.index[0])
    n_window = round(funding_days * per_day)
    n_year = round(history_days * per_day)
    funding = bars["funding_rate"].tolist()
    sums: list[float | None] = [None] * len(funding)
    ranks: list[float | None] = [None] * len(funding)
    for t in range(n_window - 1, len(funding)):
        sums[t] = sum(funding[t - n_window + 1 : t + 1])
    for t in range(len(funding)):
        if sums[t] is None:
            continue
        defined = [s for s in sums[max(0, t - n_year + 1) : t + 1] if s is not None]
        if len(defined) >= n_year // 4:
            ranks[t] = sum(1 for s in defined if s <= sums[t]) / len(defined)
    return sums, ranks


def reference_targets(
    bars: pd.DataFrame,
    funding_days: float = 7,
    quantile: float = 0.8,
    hold_bars: int = 7,
    history_days: float = 365,
    exit_quantile: float = 0.5,
    neutral_daily_funding: float = 0.0003,
    stop_atr: float = 3.0,
    ranks: tuple[list, list] | None = None,
) -> tuple[list[float], list[str]]:
    """The card's rules bar by bar in plain Python, written separately from the module.

    Only ATR comes from the indicator the card names (``indicators.atr``). Returns the targets
    and why each trade was covered.
    """
    sums, rank = ranks or reference_ranks(bars, funding_days, history_days)
    floor = neutral_daily_funding * funding_days + 1e-9
    crowded = [
        rank[t] is not None and rank[t] >= quantile and sums[t] > floor for t in range(len(sums))
    ]
    close = bars["close"].tolist()
    wilder_atr = atr(bars["high"], bars["low"], bars["close"], 14).tolist()
    targets: list[float] = []
    exits: list[str] = []
    entry_bar = None
    stop = 0.0
    for t in range(len(close)):
        if entry_bar is not None:
            if t - entry_bar >= hold_bars:
                exits.append("time")
            elif rank[t] is not None and rank[t] < exit_quantile:
                exits.append("funding")
            elif close[t] >= stop:
                exits.append("stop")
            else:
                targets.append(-1.0)
                continue
            entry_bar = None
            targets.append(0.0)
            continue
        onset = crowded[t] and not (t > 0 and crowded[t - 1])
        if onset and not math.isnan(wilder_atr[t]):
            entry_bar = t
            stop = close[t] + stop_atr * wilder_atr[t]
            targets.append(-1.0)
        else:
            targets.append(0.0)
    return targets, exits


EXTRA_CASES = [
    {"funding_days": 2, "history_days": 30, "quantile": 0.6, "exit_quantile": 0.55,
     "hold_bars": 10, "stop_atr": 0.5},
    {"funding_days": 1, "history_days": 8, "hold_bars": 3},
    {"funding_days": 3, "history_days": 60, "quantile": 1.0, "hold_bars": 1},
    {"funding_days": 5, "history_days": 90, "neutral_daily_funding": 0.0, "stop_atr": 1.0,
     "hold_bars": 20, "exit_quantile": 0.7},
]  # fmt: skip


@pytest.mark.parametrize("seed", range(10))
def test_every_grid_cell_matches_the_reference_on_random_markets(seed) -> None:
    bars = crowding_market(seed)
    exits: list[str] = []
    for funding_days in (3, 7, 14):
        ranks = reference_ranks(bars, funding_days, 365)
        for cell in (c for c in GRID if c["funding_days"] == funding_days):
            expected, why = reference_targets(bars, **cell, ranks=ranks)
            assert FundingCrowdingShort(**cell).target_position(bars).tolist() == expected, cell
            exits += why
    assert {"time", "funding", "stop"} <= set(exits)


@pytest.mark.parametrize("seed", range(20))
def test_matches_the_plain_python_reference_on_random_markets(seed) -> None:
    bars = crowding_market(seed)
    exits: list[str] = []
    for market_ in (bars, mirrored(bars, funding=False), mirrored(bars)):
        for params in EXTRA_CASES:
            expected, why = reference_targets(market_, **params)
            assert FundingCrowdingShort(**params).target_position(market_).tolist() == expected
            exits += why
    # The busy settings trade often and cover for every reason.
    busy, _ = reference_targets(bars, **EXTRA_CASES[0])
    assert sum(1 for a, b in zip([0.0, *busy], busy, strict=False) if b < a) > 10
    assert {"time", "funding", "stop"} <= set(exits)


@pytest.mark.parametrize("seed", range(5))
def test_matches_the_reference_on_4h_markets(seed) -> None:
    bars = crowding_market(seed, n=900, freq="4h")
    for params in (
        {"funding_days": 2, "history_days": 30},
        {"funding_days": 1, "history_days": 10, "quantile": 0.7, "hold_bars": 12, "stop_atr": 1.0},
    ):
        expected, _ = reference_targets(bars, **params)
        assert -1.0 in expected
        assert FundingCrowdingShort(**params).target_position(bars).tolist() == expected


@pytest.mark.parametrize(
    "costs", [ZERO_COSTS, EXCHANGE_COSTS["binanceusdm"]], ids=["free", "costs"]
)
@pytest.mark.parametrize("mirror", [False, True], ids=["as-drawn", "mirrored-prices"])
@pytest.mark.parametrize("seed", range(5))
def test_backtests_match_the_reference(seed, mirror, costs) -> None:
    bars = crowding_market(seed)
    bars = mirrored(bars, funding=False) if mirror else bars
    params = EXTRA_CASES[0]
    expected, _ = reference_targets(bars, **params)
    reference = run_backtest(bars, pd.Series(expected, index=bars.index), costs)
    result = backtest(FundingCrowdingShort(**params), bars, costs)
    pd.testing.assert_frame_equal(result.ledger, reference.ledger)
    pd.testing.assert_frame_equal(result.trades, reference.trades)
    assert len(result.trades) > 10 and (result.trades["direction"] == -1).all()


# --- real data ------------------------------------------------------------------------------


def test_smoke_on_real_btc_data() -> None:
    criteria = load_criteria(Workspace(ROOT).criteria_path)
    exchange = criteria.get("data.exchange")
    try:  # development bars only: they end where the holdout starts
        bars = load_bars(exchange, "BTC", "1d", end=criteria.get("data.dev_end"))
    except FileNotFoundError:
        pytest.skip("BTC 1d data not downloaded")
    result = backtest(FundingCrowdingShort(), bars, EXCHANGE_COSTS[exchange])
    assert set(result.ledger["target"].unique()) <= {-1.0, 0.0}
    assert len(result.trades) > 0
    assert (result.trades["direction"] == -1).all()
    assert np.isfinite(result.equity).all()
