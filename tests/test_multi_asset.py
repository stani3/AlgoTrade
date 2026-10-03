"""Research on stocks, ETFs and forex next to crypto: loading, calendars, costs."""

import pandas as pd
import pytest
from research_helpers import UNIVERSES, make_workspace, write_session_market

from algotrade.backtest.costs import EXCHANGE_COSTS, CostModel
from algotrade.calendars import infer_calendar, periods_per_year
from algotrade.data.files import bars_path, funding_file, write_bars
from algotrade.data.market import align_to_bars, load_instrument
from algotrade.instruments import costs_text, registry
from algotrade.research.split import HoldoutViolation, dev_universe, load_dev_bars, load_full_bars


@pytest.fixture
def multi(tmp_path):
    ws, criteria = make_workspace(tmp_path, {"data.universes": UNIVERSES}, repo=False)
    write_session_market(ws.raw_data)
    return ws, criteria


def charges(*rows: tuple[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"timestamp": pd.to_datetime([t for t, _ in rows], utc=True),
         "funding_rate": [r for _, r in rows]}
    )  # fmt: skip


def test_financing_is_paid_by_the_bar_that_holds_the_position_after_the_charge() -> None:
    bars = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02 14:30", "2024-01-02 18:30", "2024-01-03 14:30",
                        "2024-01-03 18:30"], utc=True)
    )  # fmt: skip
    aligned = align_to_bars(bars, charges(
        ("2024-01-01 21:00", 9.0),   # before the data: no bar holds it
        ("2024-01-02 21:00", 1.0),   # overnight after the 2 Jan close: the 3 Jan open
        ("2024-01-03 14:30", 2.0),   # exactly at an open: that bar
        ("2024-01-04 21:00", 5.0),   # after the last open: not yet attributable
    ))  # fmt: skip
    assert aligned.tolist() == [0.0, 0.0, 3.0, 0.0]
    assert align_to_bars(bars, charges()).tolist() == [0.0] * 4
    assert align_to_bars(bars[:0], charges(("2024-01-02 21:00", 1.0))).empty


def test_crypto_loads_exactly_as_before(multi) -> None:
    ws, criteria = multi
    from algotrade.data.market import load_market

    btc = load_dev_bars(ws, criteria, "BTC", "4h")
    direct = load_market(ws.raw_data, "binanceusdm", "BTC", "4h")
    pd.testing.assert_frame_equal(btc, direct[direct.index < pd.Timestamp("2025-05-01", tz="UTC")])
    assert dev_universe(ws, criteria, "4h").keys() == {"BTC", "ETH", "SOL"}
    assert registry(criteria)["BTC"].costs is EXCHANGE_COSTS["binanceusdm"]


@pytest.mark.parametrize("timeframe", ["1h", "4h", "1d"])
def test_session_markets_load_from_their_source(multi, timeframe: str) -> None:
    ws, criteria = multi
    universe = dev_universe(ws, criteria, timeframe, ["TLT", "EURUSD"])
    tlt, eur = universe["TLT"], universe["EURUSD"]
    assert tlt.index[0] >= pd.Timestamp("2023-01-01", tz="UTC")  # the class's start date
    assert tlt.index[-1] < pd.Timestamp("2025-05-01", tz="UTC")  # development data only
    assert infer_calendar(tlt.index) == "us_equity"
    assert infer_calendar(eur.index) == "fx"
    assert periods_per_year(tlt.index) == {"1h": 1764, "4h": 504, "1d": 252}[timeframe]
    assert (tlt["funding_rate"] == 0).all()  # no financing downloaded yet
    assert tlt.attrs == {"exchange": "alpaca", "symbol": "TLT", "timeframe": timeframe}
    with pytest.raises(HoldoutViolation):
        load_dev_bars(ws, criteria, "TLT", timeframe, end="2026-01-01")
    full = load_full_bars(ws, criteria, "EURUSD", timeframe)
    assert full.index[-1] > pd.Timestamp("2025-05-01", tz="UTC")


def test_financing_files_are_aligned_and_missing_data_is_explained(multi) -> None:
    ws, criteria = multi
    write_bars(
        pd.DataFrame({"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0,
                      "funding_rate": 0.0},
                     index=pd.DatetimeIndex([], tz="UTC")),
        funding_file(ws.raw_data, "alpaca", "UNUSED"),
    )  # fmt: skip
    path = funding_file(ws.raw_data, "alpaca", "TLT")
    frame = charges(("2024-03-01 21:00", 2e-4), ("2024-03-04 21:00", 1e-4))
    frame.to_parquet(path, index=False)
    tlt = load_instrument(ws.raw_data, registry(criteria)["TLT"], "1d")
    paid = tlt.loc[tlt["funding_rate"] != 0, "funding_rate"]
    assert paid.tolist() == [2e-4, 1e-4]
    assert paid.index.tz_convert("America/New_York").strftime("%m-%d %H:%M").tolist() == [
        "03-04 09:30", "03-05 09:30"
    ]  # fmt: skip
    bars_path(ws.raw_data, "alpaca", "IEF", "1d").unlink()
    with pytest.raises(FileNotFoundError, match="--asset-class bonds"):
        load_instrument(ws.raw_data, registry(criteria)["IEF"], "1d")


def test_unregistered_symbols_load_as_perpetuals(multi) -> None:
    ws, criteria = multi
    values = dict(criteria.values)
    values["data"] = {**values["data"], "symbols": ["BTC"]}
    from algotrade.research.criteria import Criteria

    narrow = Criteria(values=values, hash="narrow")
    assert "ETH" not in registry(narrow)
    assert len(load_dev_bars(ws, narrow, "ETH", "4h")) > 0


def test_costs_text() -> None:
    same = {"BTC": EXCHANGE_COSTS["binanceusdm"], "ETH": EXCHANGE_COSTS["binanceusdm"]}
    assert costs_text(same) == "fee 5.0 bps + slippage 3.0 bps, funding on"
    mixed = {**same, "TLT": CostModel(0.5, 2.0), "IEF": CostModel(0.5, 2.0)}
    assert costs_text(mixed) == (
        "5+3 bps per side: BTC, ETH; 0.5+2 bps per side: TLT, IEF (funding or financing on)"
    )
