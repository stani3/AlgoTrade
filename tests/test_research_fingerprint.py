from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from research_helpers import make_workspace

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.research import fingerprint
from algotrade.research.split import dev_universe
from algotrade.strategies import STRATEGIES, Strategy

COSTS = EXCHANGE_COSTS["binanceusdm"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    ws, criteria = make_workspace(tmp_path_factory.mktemp("fp"), repo=False)
    universe = dev_universe(ws, criteria, "4h", ["BTC", "ETH", "SOL"])
    return ws, criteria, universe


def fp(spec, universe):
    return fingerprint.compute(spec, universe, COSTS)


def test_fingerprint_is_daily_mean_position(setup) -> None:
    _, _, universe = setup
    frame = fp({"type": "buy_and_hold"}, universe)
    assert list(frame.columns) == ["BTC", "ETH", "SOL"]
    assert frame.index.name == "day" and frame.dtypes.eq(np.float32).all()
    # Entering at the first close: day one averages 5 of 6 bars long, then fully long.
    assert frame["BTC"].iloc[0] == pytest.approx(5 / 6)
    assert (frame["BTC"].iloc[1:] == 1.0).all()


@dataclass(frozen=True)
class EmaCrossRewrite(Strategy):
    """An EMA crossover written from scratch, as a careless idea might re-invent it."""

    short: int = 20
    long: int = 100

    name = "test_ema_cross_rewrite"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        gap = (
            close.ewm(span=self.short, adjust=False).mean()
            - close.ewm(span=self.long, adjust=False).mean()
        )
        signal = np.sign(gap).where(close.expanding().count() >= self.long, 0.0)
        return signal.fillna(0.0)


@dataclass(frozen=True)
class AlwaysInRewrite(Strategy):
    """Buy and hold in disguise."""

    name = "test_always_in_rewrite"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        return (bars["close"] > 0).astype(float)


@pytest.fixture
def rewrites(monkeypatch):
    for cls in (EmaCrossRewrite, AlwaysInRewrite):
        monkeypatch.setitem(STRATEGIES, cls.name, cls)


def test_ema_crossover_written_as_a_new_class_is_flagged(setup, rewrites) -> None:
    _, criteria, universe = setup
    stored = fp({"type": "ma_crossover", "fast": 20, "slow": 100, "kind": "ema"}, universe)
    rewritten = fp({"type": "test_ema_cross_rewrite"}, universe)
    result = fingerprint.compare(rewritten, stored, criteria, "k", "ma_crossover")
    assert result.duplicate and result.correlation > 0.9


def test_combine_of_one_is_flagged_against_its_child(setup) -> None:
    _, criteria, universe = setup
    child = {"type": "ewmac", "fast": 16, "slow": 64}
    single = fp({"type": "combine", "strategies": [child], "weights": [1]}, universe)
    result = fingerprint.compare(single, fp(child, universe), criteria)
    assert result.duplicate and result.difference == pytest.approx(0.0)


def test_always_long_rule_is_flagged_against_buy_and_hold(setup, rewrites) -> None:
    _, criteria, universe = setup
    hold = fp({"type": "buy_and_hold"}, universe)
    result = fingerprint.compare(fp({"type": "test_always_in_rewrite"}, universe), hold, criteria)
    assert result.duplicate and result.difference == 0.0


def test_constant_exposure_has_no_correlation(setup) -> None:
    _, criteria, _ = setup
    days = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    long = pd.DataFrame({"BTC": 1.0}, index=days)
    result = fingerprint.compare(long, long, criteria)
    assert result.duplicate and np.isnan(result.correlation)
    assert "n/a" in result.describe()
    assert result.to_dict()["correlation"] is None


def test_trend_and_mean_reversion_are_not_flagged(setup) -> None:
    _, criteria, universe = setup
    trend = fp({"type": "donchian_breakout"}, universe)
    reversion = fp({"type": "rsi_reversion"}, universe)
    result = fingerprint.compare(reversion, trend, criteria)
    assert not result.duplicate
    assert "correlation" in result.describe()


def test_too_little_overlap_returns_none(setup) -> None:
    _, criteria, _ = setup
    days = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    a = pd.DataFrame({"BTC": 1.0}, index=days)
    assert fingerprint.compare(a, a, criteria) is None
    assert fingerprint.compare(a, a.rename(columns={"BTC": "ETH"}), criteria) is None


def test_two_flat_strategies_count_as_identical(setup) -> None:
    _, criteria, _ = setup
    days = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    flat = pd.DataFrame({"BTC": 0.0}, index=days)
    assert fingerprint.compare(flat, flat, criteria).duplicate


def test_store_round_trip_and_nearest(setup, tmp_path) -> None:
    _, criteria, universe = setup
    store = fingerprint.FingerprintStore(tmp_path / "fp")
    assert store.keys() == []
    trend = fp({"type": "ewmac"}, universe)
    hold = fp({"type": "buy_and_hold"}, universe)
    store.save("aaa_4h", trend, {"idea": "i001", "spec_hash": "aaa", "label": "ewmac"})
    store.save("bbb_4h", hold, {"source": "baseline", "spec_hash": "bbb", "label": "hold"})
    assert store.exists("aaa_4h") and store.keys() == ["aaa_4h", "bbb_4h"]
    loaded, meta = store.load("aaa_4h")
    pd.testing.assert_frame_equal(loaded, trend, check_freq=False)
    assert meta["key"] == "aaa_4h"

    nearest = store.nearest(trend, criteria)
    assert nearest[0].key == "aaa_4h" and nearest[0].duplicate
    assert [s.key for s in store.nearest(trend, criteria, exclude_idea="i001")] == ["bbb_4h"]
    assert [s.key for s in store.nearest(trend, criteria, exclude_hash="bbb")] == ["aaa_4h"]
