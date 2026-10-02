"""i002_shock_fade: fade a close-to-close move of at least ``z_entry`` prior volatilities.

Hand-computed cases are 4h markets written as close-to-close returns. ``d = +1`` runs a case as
written (down-shocks, faded long) and ``d = -1`` runs its mirror image (up-shocks, faded short),
so the expected targets are ``d`` times the hand-written long-side ones.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import bars_from, random_bars

from algotrade.backtest.costs import EXCHANGE_COSTS, ZERO_COSTS, CostModel
from algotrade.backtest.engine import run_backtest
from algotrade.backtest.runner import backtest, load_bars
from algotrade.indicators import ewm_vol
from algotrade.research.criteria import load_criteria
from algotrade.research.workspace import Workspace
from algotrade.strategies import from_spec, to_spec
from algotrade.strategies.ideas.i002_shock_fade import ShockFade

ROOT = Path(__file__).resolve().parents[2]
SIDES = pytest.mark.parametrize("d", [1, -1], ids=["long", "short"])
CARD_SPEC = {
    "type": "i002_shock_fade",
    "z_entry": 3.0,
    "hold_bars": 6,
    "vol_days": 25.0,
    "allow_short": True,
}
GRID = [(z_entry, hold_bars) for z_entry in (2.5, 3.0, 3.5) for hold_bars in (3, 6, 12)]

# 80 returns of +-1%: ewm_vol's warm-up (75 returns at span 150) is over and volatility is about
# 1%, so a 6% move scores about 6 (about 4.8 right after another one) and +-1% bars below 1.
QUIET = [0.01, -0.01] * 40
SHOCK_BAR = len(QUIET) + 1  # the first move after QUIET (bar 0 has no return)


def market(d: int, moves: list[float], funding: float = 0.0, freq: str = "4h") -> pd.DataFrame:
    """Bars from a close of 100 whose close-to-close returns are ``d * moves``."""
    returns = d * np.asarray(moves, dtype="float64")
    closes = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + returns)))
    return bars_from([(c, c, c, c) for c in closes], funding=funding, freq=freq)


def after_quiet(d: int, moves: list[float], **params) -> list[float]:
    """Targets on the bars of ``moves`` placed after QUIET; the quiet stretch must stay flat."""
    target = ShockFade(**params).target_position(market(d, QUIET + moves))
    assert (target.iloc[: -len(moves)] == 0).all()
    return target.iloc[-len(moves) :].tolist()


def long_side(d: int, values: list[int]) -> list[int]:
    return [d * value for value in values]


def scores(bars: pd.DataFrame, vol_days: float = 25.0) -> pd.Series:
    """The card's shock score z_t = r_t / sigma_{t-1}, written out for 4h bars (6 a day)."""
    returns = bars["close"] / bars["close"].shift() - 1
    return returns / ewm_vol(returns, vol_days * 6).shift()


# --- hand-computed cases --------------------------------------------------------------------


@SIDES
@pytest.mark.parametrize(
    ("move", "expected"),
    [(-0.04, [0, 0, 0, 0, 0, 0]), (-0.045, [0, 0, 0, 1, 1, 0])],
    ids=["4.0pct-is-not-a-shock", "4.5pct-is-a-shock"],
)
def test_tiny_market_worked_by_hand(d, move, expected) -> None:
    # vol_days 0.5 on 4h bars: span 3 (alpha 1/2) and a 2-return warm-up, so bar 3 is the first
    # bar with a score. Returns +1% and -1% weigh 1/2 each: mean 0, variance 1e-4, bias
    # correction 1 / (1 - 1/4 - 1/4) = 2, so sigma_2 = 1.414% and a shock needs |r_3| >= 4.24%.
    bars = market(d, [0.01, -0.01, move, 0.01, 0.01])
    assert scores(bars, 0.5).iloc[3] == pytest.approx(d * move / (math.sqrt(2) / 100))
    # Measured against its own bar's volatility, the move would score far below 3.
    returns = bars["close"] / bars["close"].shift() - 1
    assert abs(returns.iloc[3] / ewm_vol(returns, 3.0).iloc[3]) < 2
    target = ShockFade(vol_days=0.5, hold_bars=2).target_position(bars)
    assert target.tolist() == long_side(d, expected)


@SIDES
def test_a_shock_is_faded_for_exactly_hold_bars(d) -> None:
    # Opened at the shock bar's close, still held at bar s+2, flat again at the close of s+3.
    moves = [-0.06, 0.01, -0.01, 0.01, -0.01, 0.01]
    assert after_quiet(d, moves, hold_bars=3) == long_side(d, [1, 1, 1, 0, 0, 0])


@SIDES
def test_default_hold_is_one_day_of_4h_bars(d) -> None:
    moves = [-0.06] + [0.01, -0.01] * 4
    assert after_quiet(d, moves) == long_side(d, [1, 1, 1, 1, 1, 1, 0, 0, 0])


@SIDES
def test_hold_bars_one_holds_through_the_next_bar_only(d) -> None:
    assert after_quiet(d, [-0.06, 0.01, -0.01], hold_bars=1) == long_side(d, [1, 0, 0])


@SIDES
def test_large_moves_below_z_entry_are_ignored(d) -> None:
    # 2% and 2.5% moves score about 2.0 and 2.4 against the 1% volatility.
    assert after_quiet(d, [-0.02, 0.025, -0.01, 0.01]) == [0, 0, 0, 0]


RESTARTS = {
    "same-direction-extends": (
        [-0.06, 0.01, -0.06, 0.01, -0.01, 0.01, -0.01],
        [1, 1, 1, 1, 1, 0, 0],
    ),
    "opposite-direction-reverses": (
        [-0.06, 0.01, 0.06, -0.01, 0.01, -0.01, 0.01],
        [1, 1, -1, -1, -1, 0, 0],
    ),
    "reversal-on-the-next-bar": (
        [-0.06, 0.06, 0.01, -0.01, 0.01, -0.01],
        [1, -1, -1, -1, 0, 0],
    ),
    "new-shock-on-the-expiry-bar": (
        [-0.06, 0.01, -0.01, 0.06, -0.01, 0.01, -0.01, 0.01],
        [1, 1, 1, -1, -1, -1, 0, 0],
    ),
}


@SIDES
@pytest.mark.parametrize(("moves", "expected"), list(RESTARTS.values()), ids=list(RESTARTS))
def test_a_new_shock_restarts_the_clock_in_its_own_direction(d, moves, expected) -> None:
    assert after_quiet(d, moves, hold_bars=3) == long_side(d, expected)


@pytest.mark.parametrize(
    ("d", "expected"),
    [(1, [1, 1, 0, 0, 0, 0, 0]), (-1, [0, 0, 1, 1, 1, 0, 0])],
    ids=["up-shock-closes-the-long", "down-shock-in-an-up-shock-window-goes-long"],
)
def test_long_only_turns_short_targets_flat(d, expected) -> None:
    moves = [-0.06, 0.01, 0.06, -0.01, 0.01, -0.01, 0.01]
    assert after_quiet(d, moves, hold_bars=3, allow_short=False) == expected


@SIDES
@pytest.mark.parametrize(("moves", "expected"), list(RESTARTS.values()), ids=list(RESTARTS))
def test_long_only_is_the_long_short_target_without_shorts(d, moves, expected) -> None:
    long_only = after_quiet(d, moves, hold_bars=3, allow_short=False)
    assert long_only == [max(value, 0) for value in long_side(d, expected)]


# --- boundaries -----------------------------------------------------------------------------


@SIDES
def test_a_score_of_exactly_z_entry_is_a_shock(d) -> None:
    bars = market(d, QUIET + [-0.04, 0.01, -0.01])
    score = abs(scores(bars).iloc[SHOCK_BAR])
    assert 3.5 < score < 4.5
    at = ShockFade(z_entry=float(score), hold_bars=2).target_position(bars)
    assert (at.iloc[:SHOCK_BAR] == 0).all()
    assert at.iloc[SHOCK_BAR:].tolist() == long_side(d, [1, 1, 0])
    above = ShockFade(z_entry=float(np.nextafter(score, np.inf)), hold_bars=2)
    assert (above.target_position(bars) == 0).all()


@SIDES
@pytest.mark.parametrize(
    ("freq", "vol_days", "first"),
    [("4h", 25.0, 76), ("4h", 2.0, 7), ("4h", 0.5, 3), ("1D", 25.0, 13), ("1h", 25.0, 301)],
)
def test_the_first_scored_bar_follows_the_volatility_warm_up(d, freq, vol_days, first) -> None:
    # span = vol_days x bars per day (6 on 4h, 1 on 1d, 24 on 1h) and ewm_vol needs span // 2
    # returns (at least 2), so sigma is first known at bar span // 2 and the next bar is the
    # first that can be a shock. The same -10% move one bar earlier is never scored, and the
    # volatility after it includes it.
    strategy = ShockFade(vol_days=vol_days, hold_bars=2)
    for at, expected in ((first - 1, []), (first, [first, first + 1])):
        moves = ([0.01, -0.01] * 200)[: at - 1] + [-0.10, 0.01, -0.01, 0.01]
        target = strategy.target_position(market(d, moves, freq=freq))
        assert np.flatnonzero(target.to_numpy()).tolist() == expected
        assert (target.iloc[expected] == d).all()


@SIDES
def test_no_shock_while_the_previous_volatility_is_zero(d) -> None:
    # Flat closes make sigma exactly 0, so the first -5% move scores -inf and is skipped; the
    # second is measured against a positive volatility (about 8 sigma) and is faded.
    bars = market(d, [0.0] * 80 + [-0.05, -0.05, 0.0, 0.0])
    assert scores(bars).iloc[-4] == -d * math.inf
    target = ShockFade(hold_bars=2).target_position(bars)
    assert (target.iloc[:-3] == 0).all()
    assert target.iloc[-3:].tolist() == long_side(d, [1, 1, 0])


def test_one_bar_and_empty_markets_are_flat() -> None:
    one = market(1, [])
    assert ShockFade().target_position(one).tolist() == [0.0]
    assert ShockFade().target_position(one.iloc[:0]).empty


# --- parameters and specs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"z_entry": 0.0}, "z_entry"),
        ({"z_entry": -3.0}, "z_entry"),
        ({"z_entry": math.nan}, "z_entry"),
        ({"z_entry": math.inf}, "z_entry"),
        ({"hold_bars": 0}, "hold_bars"),
        ({"hold_bars": -6}, "hold_bars"),
        ({"hold_bars": 6.0}, "hold_bars"),
        ({"hold_bars": 2.5}, "hold_bars"),
        ({"hold_bars": True}, "hold_bars"),
        ({"vol_days": 0.0}, "vol_days"),
        ({"vol_days": -25.0}, "vol_days"),
        ({"vol_days": math.nan}, "vol_days"),
        ({"vol_days": math.inf}, "vol_days"),
    ],
)
def test_invalid_parameters_are_refused(params, message) -> None:
    with pytest.raises(ValueError, match=message):
        ShockFade(**params)


def test_every_grid_cell_builds_from_the_card_spec() -> None:
    for z_entry, hold_bars in GRID:
        strategy = from_spec({**CARD_SPEC, "z_entry": z_entry, "hold_bars": hold_bars})
        assert strategy == ShockFade(z_entry=z_entry, hold_bars=hold_bars)
    # Grid values read back from a results table arrive as numpy integers.
    assert ShockFade(hold_bars=np.int64(3)).hold_bars == 3


def test_spec_round_trip_and_card_defaults() -> None:
    assert to_spec(ShockFade()) == CARD_SPEC
    assert from_spec(CARD_SPEC) == ShockFade()
    custom = ShockFade(z_entry=2.5, hold_bars=12, vol_days=10.0, allow_short=False)
    assert from_spec(to_spec(custom)) == custom
    assert from_spec(json.dumps(to_spec(custom))) == custom
    assert str(custom) == (
        "i002_shock_fade(z_entry=2.5, hold_bars=12, vol_days=10.0, allow_short=False)"
    )


# --- engine ---------------------------------------------------------------------------------


@SIDES
@pytest.mark.parametrize("funding", [0.0, 1e-4, -1e-4], ids=["no-funding", "positive", "negative"])
def test_backtest_holds_the_fade_through_the_bars_after_the_shock(d, funding) -> None:
    moves = [-0.06, 0.01, -0.01, 0.01, -0.01, 0.01]
    bars = market(d, QUIET + moves, funding=funding)
    strategy = ShockFade(hold_bars=3)
    result = backtest(strategy, bars, CostModel(fee_bps=5.0, slippage_bps=3.0))
    s = SHOCK_BAR
    target = strategy.target_position(bars)
    pd.testing.assert_series_equal(result.ledger["target"], target, check_names=False)
    position = result.ledger["position"]
    assert position.iloc[s + 1 : s + 4].tolist() == [d, d, d]
    assert (position.iloc[: s + 1] == 0).all() and (position.iloc[s + 4 :] == 0).all()
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert (trade["direction"], trade["bars"], trade["open"]) == (d, 3, False)
    assert (trade["entry"], trade["exit"]) == (bars.index[s + 1], bars.index[s + 4])
    # It earns bars s+1..s+3 (+1%, -1%, +1% in its favour), pays 8 bp to enter and 8 bp to
    # exit, and funding on each held bar (longs pay positive funding, shorts receive it).
    assert trade["return"] == pytest.approx(0.01 - 2 * 0.0008 - 3 * d * funding, abs=1e-12)


# --- no lookahead ---------------------------------------------------------------------------

LOOKAHEAD_CASES = [
    ShockFade(),
    ShockFade(z_entry=1.0, hold_bars=4, vol_days=2.0),
    ShockFade(z_entry=1.5, hold_bars=12, vol_days=5.0, allow_short=False),
]
CUTS = (0, 1, 2, 3, 77, 151, 250, 333, 399)


def assert_no_lookahead(strategy: ShockFade, bars: pd.DataFrame) -> None:
    full = strategy.target_position(bars)
    for cut in CUTS:
        pd.testing.assert_series_equal(strategy.target_position(bars.iloc[:cut]), full.iloc[:cut])


@pytest.mark.parametrize("strategy", LOOKAHEAD_CASES, ids=str)
def test_no_lookahead_on_the_fixture_bars(strategy, bars) -> None:
    assert_no_lookahead(strategy, bars)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("strategy", LOOKAHEAD_CASES, ids=str)
def test_no_lookahead_on_random_markets(strategy, seed) -> None:
    assert_no_lookahead(strategy, random_bars(seed, n=400))


# --- plain-Python reference -----------------------------------------------------------------


def reference_targets(
    bars: pd.DataFrame, z_entry: float, hold_bars: int, vol_days: float, allow_short: bool
) -> list[float]:
    """The card's rules bar by bar in plain Python, written separately from the module.

    Only the volatility comes from the indicator the card names (``indicators.ewm_vol``).
    """
    close = bars["close"].tolist()
    returns = [math.nan] + [close[t] / close[t - 1] - 1 for t in range(1, len(close))]
    sigma = ewm_vol(pd.Series(returns), vol_days * 6).tolist()  # 4h bars: 6 a day
    shock_side = [0] * len(close)  # the trade direction of each shock bar, 0 elsewhere
    for t in range(1, len(close)):
        known = sigma[t - 1]
        if known > 0 and abs(returns[t] / known) >= z_entry:
            shock_side[t] = 1 if returns[t] < 0 else -1
    targets = []
    for t in range(len(close)):
        side = 0  # the most recent shock s with t - hold_bars < s <= t
        for s in range(t, t - hold_bars, -1):
            if s >= 0 and shock_side[s]:
                side = shock_side[s]
                break
        if side < 0 and not allow_short:
            side = 0
        targets.append(float(side))
    return targets


REFERENCE_CASES = [
    *({"z_entry": z, "hold_bars": h, "vol_days": 25.0, "allow_short": True} for z, h in GRID),
    {"z_entry": 1.0, "hold_bars": 4, "vol_days": 2.0, "allow_short": True},
    {"z_entry": 1.0, "hold_bars": 4, "vol_days": 2.0, "allow_short": False},
    {"z_entry": 1.5, "hold_bars": 1, "vol_days": 5.0, "allow_short": True},
    {"z_entry": 2.0, "hold_bars": 12, "vol_days": 25.0, "allow_short": False},
    {"z_entry": 0.5, "hold_bars": 2, "vol_days": 0.5, "allow_short": True},
]


@pytest.mark.parametrize("seed", range(20))
def test_matches_the_plain_python_reference_on_random_markets(seed) -> None:
    bars = random_bars(seed, n=400)
    for params in REFERENCE_CASES:
        expected = reference_targets(bars, **params)
        assert ShockFade(**params).target_position(bars).tolist() == expected, params
    # The busy settings make many overlapping trades in both directions, including reversals.
    busy = reference_targets(bars, z_entry=1.0, hold_bars=4, vol_days=2.0, allow_short=True)
    assert busy.count(1.0) > 50 and busy.count(-1.0) > 50
    assert any(a * b < 0 for a, b in pairwise(busy))


@pytest.mark.parametrize(
    "costs", [ZERO_COSTS, EXCHANGE_COSTS["binanceusdm"]], ids=["free", "costs"]
)
@pytest.mark.parametrize("allow_short", [True, False], ids=["long-short", "long-only"])
@pytest.mark.parametrize("seed", range(5))
def test_backtests_match_the_reference(seed, allow_short, costs) -> None:
    bars = random_bars(seed, n=400)
    params = {"z_entry": 1.5, "hold_bars": 5, "vol_days": 2.0, "allow_short": allow_short}
    reference = run_backtest(
        bars, pd.Series(reference_targets(bars, **params), index=bars.index), costs
    )
    result = backtest(ShockFade(**params), bars, costs)
    pd.testing.assert_frame_equal(result.ledger, reference.ledger)
    pd.testing.assert_frame_equal(result.trades, reference.trades)
    assert len(result.trades) > 10


# --- real data ------------------------------------------------------------------------------


def test_smoke_on_real_btc_data() -> None:
    criteria = load_criteria(Workspace(ROOT).criteria_path)
    exchange = criteria.get("data.exchange")
    try:  # development bars only: they end where the holdout starts
        bars = load_bars(exchange, "BTC", "4h", end=criteria.get("data.dev_end"))
    except FileNotFoundError:
        pytest.skip("BTC 4h data not downloaded")
    result = backtest(ShockFade(), bars, EXCHANGE_COSTS[exchange])
    assert set(result.ledger["target"].unique()) <= {-1.0, 0.0, 1.0}
    assert len(result.trades) > 0
    assert np.isfinite(result.equity).all()
