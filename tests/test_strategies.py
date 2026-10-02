from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algotrade.backtest.metrics import periods_per_year, sharpe_ratio
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

SPECS = sorted(Path(__file__).parent.parent.joinpath("specs").glob("*.json"))

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


@dataclass(frozen=True)
class AlwaysLong(Strategy):
    name = "always_long"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


@pytest.mark.parametrize("strategy", CASES, ids=str)
def test_no_lookahead(strategy, bars) -> None:
    """A target must not change when future bars are appended."""
    full = strategy.target_position(bars)
    for cut in (60, 150, 333):
        partial = strategy.target_position(bars.iloc[:cut])
        pd.testing.assert_series_equal(partial, full.iloc[:cut], check_names=False)


@pytest.mark.parametrize("strategy", CASES, ids=str)
def test_targets_are_aligned_and_bounded(strategy, bars) -> None:
    target = strategy.target_position(bars)
    assert target.index.equals(bars.index)
    assert target.notna().all()
    limit = getattr(strategy, "max_leverage", 1.0)
    assert target.abs().max() <= limit + 1e-12


@pytest.mark.parametrize("strategy", MUST_TRADE, ids=str)
def test_every_rule_trades_with_defaults(strategy, bars) -> None:
    assert (strategy.target_position(bars) != 0).any()


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
