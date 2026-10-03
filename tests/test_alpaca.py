import numpy as np
import pandas as pd
import pytest
from http_fakes import FakeResponse, FakeSession

from algotrade.calendars import infer_calendar, periods_per_year
from algotrade.data import alpaca
from algotrade.data.alpaca import (
    KEY_VARS,
    AlpacaClient,
    AlpacaError,
    chain,
    missing_keys,
    regular_hours,
    update_equity,
)
from algotrade.data.files import bars_path, read_bars
from algotrade.data.sessions import us_equity_sessions

NY = "America/New_York"
SECRET = "very-secret-value"


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_VARS[0], "key-id")
    monkeypatch.setenv(KEY_VARS[1], SECRET)
    monkeypatch.setattr(alpaca.time, "sleep", lambda seconds: None)


class FakeAlpaca:
    """Answers the bars endpoint from a fixed tape (30-minute bars with pre- and post-market),
    ``per_page`` bars at a time; prices are multiplied by ``factor`` (a dividend re-basing)."""

    def __init__(self, start: str, end: str, per_page: int = 400) -> None:
        sessions = us_equity_sessions(start, end)
        stamps = []
        for open_, close in zip(sessions["open"], sessions["close"], strict=True):
            stamps += list(
                pd.date_range(open_ - pd.Timedelta("1h"), close + pd.Timedelta("1h"),
                              freq="30min", inclusive="left")
            )  # fmt: skip
        self.index = pd.DatetimeIndex(stamps)
        self.price = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.2, len(stamps)))
        self.factor = 1.0
        self.per_page = per_page
        self.requests: list[dict] = []

    def __call__(self, url: str, params: dict) -> FakeResponse:
        self.requests.append(dict(params))
        start = pd.Timestamp(params["start"])
        end = pd.Timestamp(params["end"])
        chosen = np.flatnonzero((self.index >= start) & (self.index < end))
        offset = int(params.get("page_token") or 0)
        page = chosen[offset : offset + self.per_page]
        bars = [
            {"t": self.index[i].strftime("%Y-%m-%dT%H:%M:%SZ"),
             "o": self.price[i] * self.factor, "h": (self.price[i] + 0.3) * self.factor,
             "l": (self.price[i] - 0.3) * self.factor, "c": (self.price[i] + 0.1) * self.factor,
             "v": 1000, "n": 10, "vw": self.price[i]}
            for i in page
        ]  # fmt: skip
        more = offset + self.per_page < len(chosen)
        return FakeResponse(200, payload={"bars": bars, "symbol": "TLT",
                                          "next_page_token": str(offset + self.per_page)
                                          if more else None})  # fmt: skip


def test_keys_are_required_and_never_shown(monkeypatch: pytest.MonkeyPatch, keys) -> None:
    assert missing_keys() == []
    client = AlpacaClient(FakeSession(lambda url, params: FakeResponse(403, b"forbidden")))
    with pytest.raises(AlpacaError, match="HTTP 403") as error:
        client.bars("TLT", pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-04"))
    assert SECRET not in str(error.value)
    monkeypatch.delenv(KEY_VARS[1])
    assert missing_keys() == [KEY_VARS[1]]
    with pytest.raises(AlpacaError, match=KEY_VARS[1]):
        AlpacaClient()


def test_paging_parameters_and_rate_limits(keys) -> None:
    tape = FakeAlpaca("2023-01-03", "2023-01-06", per_page=10)
    limited = {"left": 2}

    def answer(url, params):
        if limited["left"]:
            limited["left"] -= 1
            reset = {"X-RateLimit-Reset": "0"} if limited["left"] else {}
            return FakeResponse(429, headers=reset)
        return tape(url, params)

    session = FakeSession(answer)
    client = AlpacaClient(session)
    bars = client.bars("TLT", pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-06"))
    assert len(bars) == (tape.index < pd.Timestamp("2023-01-06", tz="UTC")).sum()  # end excluded
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.tz is not None and bars.index.is_monotonic_increasing
    first = tape.requests[0]
    assert first["adjustment"] == "all" and first["feed"] == "sip" and first["timeframe"] == "30Min"
    assert session.calls[0][0].endswith("/v2/stocks/TLT/bars")
    assert session.calls[0][2]["APCA-API-SECRET-KEY"] == SECRET
    assert len(tape.requests) == -(-len(bars) // 10)
    stuck = AlpacaClient(FakeSession(lambda url, params: FakeResponse(429)), max_retries=1)
    with pytest.raises(AlpacaError, match="rate limited"):
        stuck.bars("TLT", pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"))


def test_regular_hours_and_chaining() -> None:
    sessions = us_equity_sessions("2023-07-03", "2023-07-05")
    tape = FakeAlpaca("2023-07-03", "2023-07-05")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                        index=tape.index)  # fmt: skip
    kept = regular_hours(bars, sessions)
    assert len(kept) == 7 + 13  # the early close on 3 July, then a full session
    local = kept.index.tz_convert(NY)
    assert local.min().strftime("%H:%M") == "09:30" and local.max().strftime("%H:%M") == "15:30"

    index = pd.date_range("2023-01-03 14:30", periods=6, freq="30min", tz="UTC")
    frame = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0,
                          "close": [10.0, 10.5, 11.0, 12.0, 13.0, 14.0], "volume": 5.0},
                         index=index)  # fmt: skip
    stored = frame.iloc[:3]
    rebased = frame.iloc[2:].copy()
    rebased[["open", "high", "low", "close"]] *= 0.5  # a dividend re-based the history
    joined = chain(stored, rebased)
    pd.testing.assert_frame_equal(joined.iloc[:3], stored)
    assert joined["close"].tolist() == [10.0, 10.5, 11.0, 12.0, 13.0, 14.0]
    assert chain(None, frame) is frame
    assert chain(stored, frame.iloc[1:3]) is stored
    with pytest.raises(AlpacaError, match="cannot chain"):
        chain(stored, frame.iloc[3:])


def test_update_equity_writes_session_bars_and_appends_without_rewriting(tmp_path, keys) -> None:
    tape = FakeAlpaca("2023-01-03", "2023-03-31")
    client = AlpacaClient(FakeSession(tape))
    counts = update_equity(tmp_path, "TLT", ["1h", "4h", "1d"], client, start="2023-01-01",
                           now=pd.Timestamp("2023-02-15 03:00", tz="UTC"))  # fmt: skip
    base = read_bars(bars_path(tmp_path, "alpaca", "TLT", "30m"))
    hourly = read_bars(bars_path(tmp_path, "alpaca", "TLT", "1h"))
    daily = read_bars(bars_path(tmp_path, "alpaca", "TLT", "1d"))
    assert counts == {"30m": len(base), "1h": len(hourly), "4h": counts["4h"], "1d": len(daily)}
    assert base.index[-1] < pd.Timestamp("2023-02-15", tz="UTC")  # whole days only
    assert set(base.index.tz_convert(NY).strftime("%H:%M")) <= {
        f"{h:02d}:{m}" for h in range(9, 16) for m in ("00", "30")
    }
    assert infer_calendar(hourly.index) == "us_equity"
    assert periods_per_year(hourly.index) == 252 * 7
    assert len(daily) == len(us_equity_sessions("2023-01-03", "2023-02-14"))

    before = bars_path(tmp_path, "alpaca", "TLT", "30m").read_bytes()
    assert update_equity(tmp_path, "TLT", ["1h"], client,
                         now=pd.Timestamp("2023-02-14 23:00", tz="UTC")) == {}  # fmt: skip
    assert bars_path(tmp_path, "alpaca", "TLT", "30m").read_bytes() == before

    tape.factor = 0.97  # a dividend since the last update re-based every price
    update_equity(tmp_path, "TLT", ["1h", "1d"], client, now=pd.Timestamp("2023-03-10", tz="UTC"))
    longer = read_bars(bars_path(tmp_path, "alpaca", "TLT", "30m"))
    pd.testing.assert_frame_equal(longer.loc[base.index], base)
    seam = longer.index.get_loc(base.index[-1])
    step = longer["close"].iloc[seam + 1] / longer["close"].iloc[seam]
    raw = tape.price + 0.1
    at = tape.index.get_loc(longer.index[seam])
    after = tape.index.get_loc(longer.index[seam + 1])  # the next regular-hours bar
    assert step == pytest.approx(raw[after] / raw[at])  # no jump at the seam
    assert tape.requests[-1]["start"].startswith(base.index[-1].strftime("%Y-%m-%dT%H:%M"))


def test_default_session_and_a_symbol_without_bars(tmp_path, keys) -> None:
    assert AlpacaClient().session is not None
    tape = FakeAlpaca("2023-01-03", "2023-01-06")
    tape.index = tape.index[:0]
    client = AlpacaClient(FakeSession(tape))
    assert update_equity(tmp_path, "NONE", ["1d"], client,
                         now=pd.Timestamp("2023-01-10", tz="UTC")) == {}  # fmt: skip
    assert update_equity(tmp_path, "NONE", ["1d"], client, start="2023-01-10",
                         now=pd.Timestamp("2023-01-10", tz="UTC")) == {}  # fmt: skip
