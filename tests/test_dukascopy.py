import lzma

import numpy as np
import pandas as pd
import pytest
import requests
from http_fakes import FakeResponse, FakeSession

from algotrade.calendars import infer_calendar
from algotrade.data import dukascopy
from algotrade.data.dukascopy import (
    RECORD,
    DukascopyClient,
    DukascopyError,
    complete_months,
    decode_hours,
    mid_bars,
    month_path,
    update_fx,
)
from algotrade.data.files import bars_path, read_bars


def candle_file(month: pd.Timestamp, base: float, point: float = 1e5, spread: float = 0.0) -> bytes:
    """A month of hourly candles in Dukascopy's format; flat and empty while forex is closed
    (Friday 17:00 to Sunday 17:00 New York)."""

    hours = pd.date_range(month, month + pd.offsets.MonthBegin(1), freq="1h", inclusive="left")
    records = np.zeros(len(hours), dtype=RECORD)
    records["t"] = np.arange(len(hours)) * 3600
    price = base + spread + np.arange(len(hours)) * 0.0001
    local = hours.tz_localize("UTC").tz_convert("America/New_York")
    weekend = np.asarray(
        (local.dayofweek == 5)
        | ((local.dayofweek == 4) & (local.hour >= 17))
        | ((local.dayofweek == 6) & (local.hour < 17))
    )
    price[weekend] = base
    as_points = np.round(price * point).astype(">i4")
    records["open"] = as_points
    records["close"] = as_points + np.where(weekend, 0, 2)
    records["low"] = as_points - np.where(weekend, 0, 3)
    records["high"] = as_points + np.where(weekend, 0, 5)
    records["volume"] = np.where(weekend, 0.0, 12.5)
    return lzma.compress(records.tobytes())


def datafeed(pair: str = "EURUSD", base: float = 1.1):
    def answer(url: str, params) -> FakeResponse:
        _, year, month, name = url.split("/datafeed/")[1].split("/")
        start = pd.Timestamp(int(year), int(month) + 1, 1)
        point = 1e3 if pair.endswith("JPY") else 1e5
        spread = 0.0001 if name.startswith("ASK") else 0.0
        return FakeResponse(200, candle_file(start, base, point, spread))

    return answer


def test_paths_and_months() -> None:
    assert month_path("EURUSD", pd.Timestamp("2023-03-01"), "BID") == (
        "/datafeed/EURUSD/2023/02/BID_candles_hour_1.bi5"
    )
    months = complete_months(pd.Timestamp("2023-11-20"), pd.Timestamp("2024-02-03"))
    assert [m.strftime("%Y-%m") for m in months] == ["2023-11", "2023-12", "2024-01"]
    assert complete_months(pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-28")) == []


def test_decode_and_mid() -> None:
    month = pd.Timestamp("2023-03-01")
    bid = decode_hours(candle_file(month, 1.1), month, 1e5)
    assert len(bid) == 31 * 24
    assert bid.index[0] == pd.Timestamp("2023-03-01", tz="UTC")
    assert bid["open"].iloc[0] == pytest.approx(1.1)
    assert bid["high"].iloc[0] == pytest.approx(1.10005)
    ask = decode_hours(candle_file(month, 1.1, spread=0.0001), month, 1e5)
    mid = mid_bars(bid, ask.iloc[:-5])
    assert len(mid) == len(ask) - 5
    assert mid["open"].iloc[0] == pytest.approx(1.10005)
    assert (mid["high"] >= mid[["open", "close"]].max(axis=1)).all()
    assert decode_hours(b"", month, 1e5).empty
    with pytest.raises(DukascopyError, match="undecodable"):
        decode_hours(b"not lzma", month, 1e5)
    aware = decode_hours(candle_file(month, 1.1), month.tz_localize("UTC"), 1e5)
    assert aware.index.equals(bid.index)


def test_client_falls_back_to_http_retries_and_reports_errors() -> None:
    waits: list[float] = []
    state = {"limited": 2}

    def answer(url: str, params):
        if url.startswith("https://"):
            return requests.ConnectTimeout("no route")
        if "missing" in url:
            return FakeResponse(404)
        if "broken" in url:
            return FakeResponse(500)
        if "busy" in url and state["limited"]:
            state["limited"] -= 1
            return FakeResponse(429, headers={"Retry-After": "3"} if state["limited"] else {})
        return FakeResponse(200, b"data")

    session = FakeSession(answer)
    client = DukascopyClient(session, pause=0.1, sleep=waits.append)
    assert client.get("/busy") == b"data"
    assert client.hosts == ["http://datafeed.dukascopy.com"]
    assert waits == [3.0, 4.0, 0.1]  # Retry-After, then a doubling wait, then the pause
    assert client.get("/missing") == b""
    with pytest.raises(DukascopyError, match="HTTP 500"):
        client.get("/broken")
    unreachable = DukascopyClient(FakeSession(lambda u, p: requests.ConnectionError("down")))
    with pytest.raises(DukascopyError, match="cannot reach"):
        unreachable.get("/x")
    assert DukascopyClient().session is not None


def test_update_fx_appends_whole_months_and_never_rewrites(tmp_path) -> None:
    client = DukascopyClient(FakeSession(datafeed("USDJPY", 130.0)), pause=0.0)
    counts = update_fx(tmp_path, "USDJPY", ["1h", "4h", "1d"], client, start="2023-02-01",
                       now=pd.Timestamp("2023-04-10", tz="UTC"))  # fmt: skip
    assert set(counts) == {"1h", "4h", "1d"}
    hourly = read_bars(bars_path(tmp_path, "dukascopy", "USDJPY", "1h"))
    assert hourly.index[-1] < pd.Timestamp("2023-04-01", tz="UTC")
    assert hourly["open"].iloc[0] == pytest.approx(130.00005)  # yen pairs have 3 decimals
    assert infer_calendar(hourly.index) == "fx"
    daily = read_bars(bars_path(tmp_path, "dukascopy", "USDJPY", "1d"))
    assert set(daily.index.tz_convert("America/New_York").hour) == {17}

    before = bars_path(tmp_path, "dukascopy", "USDJPY", "1h").read_bytes()
    assert update_fx(tmp_path, "USDJPY", ["1h"], client, now=pd.Timestamp("2023-04-30")) == {
        "1h": len(hourly)
    }
    assert bars_path(tmp_path, "dukascopy", "USDJPY", "1h").read_bytes() == before
    more = update_fx(tmp_path, "USDJPY", ["1h"], client, now=pd.Timestamp("2023-05-02", tz="UTC"))
    again = read_bars(bars_path(tmp_path, "dukascopy", "USDJPY", "1h"))
    assert more["1h"] > len(hourly)
    pd.testing.assert_frame_equal(again.loc[hourly.index], hourly)
    assert again.index[-1] >= pd.Timestamp("2023-04-28", tz="UTC")


def test_update_fx_with_nothing_to_fetch(tmp_path) -> None:
    client = DukascopyClient(FakeSession(datafeed()), pause=0.0)
    assert update_fx(tmp_path, "EURUSD", ["1h"], client, now=pd.Timestamp("2016-01-20")) == {}
    assert dukascopy.POINT["USDJPY"] == 1e3
