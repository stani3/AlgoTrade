from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars

from algotrade.backtest.metrics import periods_per_year, sharpe_ratio
from algotrade.data.exchange import TIMEFRAMES
from algotrade.research.cards import read_card
from algotrade.research.registry import IDEA_TYPE, strategy_types
from algotrade.strategies import (
    EWMAC,
    IDEAS,
    RULES,
    BracketStrategy,
    CarverBreakout,
    Combine,
    DonchianBreakout,
    FundingCarry,
    MovingAverageCrossover,
    RSIReversion,
    Strategy,
    TrendFilter,
    VolTarget,
    from_spec,
    to_spec,
)

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted(ROOT.joinpath("specs").glob("*.json"))

CASES = (
    [cls() for cls in RULES]
    + [
        MovingAverageCrossover(fast=10, slow=50, kind="ema", allow_short=False),
        RSIReversion(length=2, lower=10, upper=90, trend_filter=100, allow_short=False),
        EWMAC(fast=8, slow=32, allow_short=False),
        VolTarget(DonchianBreakout(), annual_vol=0.3, max_leverage=2.0),
        TrendFilter(RSIReversion(), length=100),
        Combine((EWMAC(), CarverBreakout(), FundingCarry()), weights=(2, 1, 1), multiplier=1.3),
    ]
    + [s for s in map(from_spec, SPECS) if not isinstance(s, BracketStrategy)]
    + [cls() for cls in IDEAS if not issubclass(cls, BracketStrategy)]
)
# Filtered/long-only specs may legitimately sit flat on the (down-trending) synthetic bars.
MUST_TRADE = [cls() for cls in RULES + IDEAS if not issubclass(cls, BracketStrategy)]


FIXTURE_BARS = pd.Timedelta("4h")


def bar_length(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(TIMEFRAMES[timeframe])


def idea_timeframes() -> dict[str, str]:
    """The shortest timeframe each idea's own strategy type is registered on (from its cards)."""

    found: dict[str, str] = {}
    for path in sorted(ROOT.glob("research/ideas/*/v*/idea.md")):
        card = read_card(path)
        for kind in filter(IDEA_TYPE.match, strategy_types(card.spec)):
            if kind not in found or bar_length(card.timeframe) < bar_length(found[kind]):
                found[kind] = card.timeframe
    return found


IDEA_TIMEFRAMES = idea_timeframes()


def finer_bars(strategy: Strategy) -> str | None:
    """The bar frequency of an idea registered on bars finer than the 4h fixtures, else None.

    Rules timed by the clock inside a 4h bar (i005 trades the hours right after the 00:00,
    08:00 and 16:00 UTC funding settlements) have no events on 4h bars: they would sit flat
    there, pass the lookahead and bounds checks without testing anything and fail to trade. They
    get the same random market on their own bar size instead.
    """

    timeframe = IDEA_TIMEFRAMES.get(strategy.name)
    if timeframe is None or bar_length(timeframe) >= FIXTURE_BARS:
        return None
    return TIMEFRAMES[timeframe]


@pytest.fixture
def market(strategy, bars) -> pd.DataFrame:
    """``bars``, or the same random market on the strategy's own finer bars."""
    freq = finer_bars(strategy)
    return bars if freq is None else make_bars(freq=freq)


@pytest.fixture
def long_market(strategy, long_bars) -> pd.DataFrame:
    """``long_bars``, or the same random market on the strategy's own finer bars."""
    freq = finer_bars(strategy)
    return long_bars if freq is None else make_bars(n=1000, freq=freq)


@dataclass(frozen=True)
class AlwaysLong(Strategy):
    name = "always_long"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


@pytest.mark.parametrize("strategy", CASES, ids=str)
def test_no_lookahead(strategy, market) -> None:
    """A target must not change when future bars are appended."""
    full = strategy.target_position(market)
    for cut in (60, 150, 333):
        partial = strategy.target_position(market.iloc[:cut])
        pd.testing.assert_series_equal(partial, full.iloc[:cut], check_names=False)


@pytest.mark.parametrize("strategy", CASES, ids=str)
def test_targets_are_aligned_and_bounded(strategy, market) -> None:
    target = strategy.target_position(market)
    assert target.index.equals(market.index)
    assert target.notna().all()
    limit = getattr(strategy, "max_leverage", 1.0)
    assert target.abs().max() <= limit + 1e-12


@pytest.mark.parametrize("strategy", MUST_TRADE, ids=str)
def test_every_rule_trades_with_defaults(strategy, long_market) -> None:
    assert (strategy.target_position(long_market) != 0).any()


def test_only_ideas_on_finer_bars_leave_the_4h_fixtures() -> None:
    assert IDEA_TIMEFRAMES["i004_funding_crowding_short"] == "1d"
    assert IDEA_TIMEFRAMES["i005_settlement_rebound"] == "1h"
    assert "vol_target" not in IDEA_TIMEFRAMES  # a catalogue type, even inside a card's spec
    ideas = {cls.name: cls() for cls in IDEAS}
    assert finer_bars(ideas["i004_funding_crowding_short"]) is None
    assert finer_bars(ideas["i005_settlement_rebound"]) == "1h"
    assert finer_bars(EWMAC()) is None


@pytest.mark.parametrize("strategy", CASES, ids=str)
def test_spec_round_trip(strategy) -> None:
    assert from_spec(to_spec(strategy)) == strategy


def test_long_only_never_shorts(bars) -> None:
    for strategy in (EWMAC(allow_short=False), DonchianBreakout(allow_short=False)):
        assert (strategy.target_position(bars) >= 0).all()


def test_trend_filter_blocks_counter_trend_positions(bars) -> None:
    target = TrendFilter(RSIReversion(), length=100).target_position(bars)
    average = bars["close"].rolling(100).mean()
    assert not ((target > 0) & (bars["close"] <= average)).any()
    assert not ((target < 0) & (bars["close"] >= average)).any()


def test_vol_target_sizes_to_requested_risk(bars) -> None:
    # Synthetic bars have 2% vol per 4h bar: ~94% a year, so 25% target -> ~0.27x exposure.
    target = VolTarget(AlwaysLong(), annual_vol=0.25).target_position(bars)
    exposure = target.iloc[200:].median()
    assert exposure == pytest.approx(0.25 / (0.02 * np.sqrt(365.25 * 6)), rel=0.25)


def test_combine_normalizes_weights(bars) -> None:
    single = EWMAC().target_position(bars)
    doubled = Combine((EWMAC(), EWMAC()), weights=(3, 1)).target_position(bars)
    pd.testing.assert_series_equal(doubled, single, check_names=False)


def test_annualization_uses_calendar_year() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    assert periods_per_year(index) == pytest.approx(365.25 * 6)
    returns = pd.Series([0.01, -0.005] * 50)
    daily = returns.mean() / returns.std()
    assert sharpe_ratio(returns, 365.25) == pytest.approx(daily * 365.25**0.5)
