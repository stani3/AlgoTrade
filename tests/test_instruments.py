import copy
from pathlib import Path

import pandas as pd
import pytest

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.calendars import infer_calendar
from algotrade.config import data_root_override
from algotrade.instruments import (
    UniverseError,
    costs_for_symbols,
    describe,
    registry,
    resolve_universe,
    symbols_for,
    universe_key,
    universes,
)
from algotrade.research.criteria import Criteria, load_criteria

REPO = Path(__file__).resolve().parent.parent
FROZEN = load_criteria(REPO / "tests" / "fixtures" / "criteria_2026-10-03.yaml")
DATA = data_root_override() or REPO / "data" / "raw"  # read at import: tests clear it

MULTI = {
    "stocks": {"source": "alpaca", "start": "2016-01-01", "symbols": ["AAPL", "GE"],
               "costs": {"fee_bps": 0.5, "slippage_bps": 2.0}, "financing": "fed_funds",
               "max_leverage": 2.0},
    "bonds": {"source": "alpaca", "start": "2016-01-01", "symbols": ["TLT", "MUB"],
              "costs": {"fee_bps": 0.5, "slippage_bps": 2.0},
              "cost_overrides": {"MUB": {"slippage_bps": 5.0}}, "financing": "fed_funds",
              "max_leverage": 10.0, "leverage_overrides": {"MUB": 2.0}},
    "fx": {"source": "dukascopy", "start": "2016-01-01", "symbols": ["EURUSD"],
           "costs": {"fee_bps": 0.3, "slippage_bps": 1.0}, "financing": "rate_difference",
           "max_leverage": 10.0},
}  # fmt: skip


def with_universes(extra: dict) -> Criteria:
    values = copy.deepcopy(FROZEN.values)
    values["data"]["universes"] = copy.deepcopy(extra)
    return Criteria(values=values, hash="test")


def test_crypto_is_the_criteria_exchange_and_symbols_unchanged() -> None:
    found = registry(FROZEN)
    assert list(found) == FROZEN.get("data.symbols")
    btc = found["BTC"]
    assert btc.is_crypto and btc.calendar == "24/7" and btc.max_leverage == 1.0
    assert btc.costs is EXCHANGE_COSTS["binanceusdm"]
    assert btc.quote == "USDT" and btc.start is None
    assert universes(FROZEN) == {"crypto": {"source": "binanceusdm",
                                            "symbols": FROZEN.get("data.symbols")}}  # fmt: skip
    assert resolve_universe(None, FROZEN) == ("crypto",)
    assert resolve_universe("all", FROZEN) == ("crypto",)
    assert symbols_for(FROZEN, ("crypto",)) == FROZEN.get("data.symbols")
    assert describe(FROZEN, ["BTC", "ETH"], "4h") == "binanceusdm 4h"


def test_other_asset_classes() -> None:
    criteria = with_universes(MULTI)
    found = registry(criteria)
    mub, tlt, ge, fx = found["MUB"], found["TLT"], found["GE"], found["EURUSD"]
    assert (mub.costs.fee_bps, mub.costs.slippage_bps) == (0.5, 5.0)
    assert (tlt.costs.fee_bps, tlt.costs.slippage_bps) == (0.5, 2.0)
    assert tlt.costs.include_funding  # financing is charged through the funding column
    assert (mub.max_leverage, tlt.max_leverage, ge.max_leverage) == (2.0, 10.0, 2.0)
    assert tlt.calendar == "us_equity" and fx.calendar == "fx"
    assert tlt.financing == "fed_funds" and fx.financing == "rate_difference"
    assert tlt.start == pd.Timestamp("2016-01-01", tz="UTC") and not tlt.is_crypto
    assert resolve_universe("all", criteria) == ("crypto", "stocks", "bonds", "fx")
    assert resolve_universe(["fx", "bonds", "fx"], criteria) == ("bonds", "fx")
    assert universe_key(("bonds", "fx")) == "bonds+fx"
    assert symbols_for(criteria, ("fx", "stocks")) == ["AAPL", "GE", "EURUSD"]
    assert costs_for_symbols(criteria, ["BTC", "MUB"]) == {
        "BTC": EXCHANGE_COSTS["binanceusdm"],
        "MUB": mub.costs,
    }
    assert describe(criteria, ["BTC", "TLT", "MUB", "EURUSD"], "1d") == (
        "crypto (binanceusdm), bonds (alpaca), fx (dukascopy) 1d"
    )
    bare = with_universes({"fx": {"source": "dukascopy", "symbols": ["USDJPY"]}})
    plain = registry(bare)["USDJPY"]
    assert plain.costs.rate == 0.0 and plain.max_leverage == 1.0 and plain.start is None


@pytest.mark.parametrize(
    ("value", "message"),
    [("bonds", "non-empty list"), ([], "non-empty list"), (["metals"], "unknown asset class")],
)
def test_bad_universes(value, message: str) -> None:
    with pytest.raises(UniverseError, match=message):
        resolve_universe(value, with_universes(MULTI))


def test_bad_criteria() -> None:
    with pytest.raises(UniverseError, match="unknown asset class 'metals'"):
        registry(with_universes({"metals": MULTI["fx"]}))
    with pytest.raises(UniverseError, match="unknown asset class 'crypto'"):
        registry(with_universes({"crypto": MULTI["fx"]}))
    with pytest.raises(UniverseError, match="unknown data source"):
        registry(with_universes({"fx": {**MULTI["fx"], "source": "yahoo"}}))
    with pytest.raises(UniverseError, match="BTC is listed in both crypto and fx"):
        registry(with_universes({"fx": {**MULTI["fx"], "symbols": ["BTC"]}}))


@pytest.mark.parametrize("timeframe", ["1h", "4h", "1d"])
def test_downloaded_data_has_the_calendar_of_its_source(timeframe: str) -> None:
    """Every downloaded file is read as the calendar its source should have."""

    from algotrade.data.files import read_bars

    checked = 0
    for source, calendar in (("alpaca", "us_equity"), ("dukascopy", "fx")):
        for path in sorted((DATA / source).glob(f"*_{timeframe}.parquet")):
            bars = read_bars(path)
            assert bars is not None and infer_calendar(bars.index) == calendar, path.name
            checked += 1
    if not checked:
        pytest.skip("no stock, ETF or forex data downloaded")
