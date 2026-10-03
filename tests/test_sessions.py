import numpy as np
import pandas as pd
import pytest

from algotrade.calendars import infer_calendar, periods_per_year
from algotrade.data.sessions import (
    fill_flat,
    fx_label,
    fx_trading_hours,
    resample_fx,
    resample_us_equity,
    session_buckets,
    us_equity_sessions,
)

NY = "America/New_York"


def ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def thirty_minute_bars(sessions: pd.DataFrame, extended: bool = True) -> pd.DataFrame:
    """Every 30-minute bar of each session (plus pre- and post-market bars), price = counter."""

    stamps = []
    for open_, close in zip(sessions["open"], sessions["close"], strict=True):
        start = open_ - pd.Timedelta(hours=1) if extended else open_
        end = close + pd.Timedelta(hours=1) if extended else close
        stamps += list(pd.date_range(start, end, freq="30min", inclusive="left"))
    index = pd.DatetimeIndex(stamps, name="timestamp")
    price = np.arange(1, len(index) + 1, dtype=float)
    return pd.DataFrame(
        {"open": price, "high": price + 0.5, "low": price - 0.5, "close": price + 0.25,
         "volume": 100.0},
        index=index,
    )  # fmt: skip


def test_sessions_follow_the_nyse_calendar() -> None:
    sessions = us_equity_sessions("2023-06-30", "2023-07-06")
    assert list(sessions.index.strftime("%m-%d")) == ["06-30", "07-03", "07-05", "07-06"]
    assert sessions.loc["2023-07-03", "close"] == ny("2023-07-03 13:00")  # early close
    assert sessions.loc["2023-06-30", "open"] == ny("2023-06-30 09:30")


@pytest.mark.parametrize(
    ("timeframe", "normal", "early"), [("1h", 7, 4), ("4h", 2, 1), ("1d", 1, 1)]
)
def test_session_buckets(timeframe: str, normal: int, early: int) -> None:
    sessions = us_equity_sessions("2023-06-30", "2023-07-03")
    buckets = session_buckets(sessions, timeframe)
    day = buckets["start"].dt.tz_convert(NY).dt.date.astype(str)
    assert (day == "2023-06-30").sum() == normal
    assert (day == "2023-07-03").sum() == early
    assert buckets["end"].max() == ny("2023-07-03 13:00")
    assert (buckets["end"] > buckets["start"]).all()
    assert session_buckets(sessions.iloc[:0], timeframe).empty


def test_us_equity_resampling_keeps_regular_hours_and_anchors_at_the_open() -> None:
    sessions = us_equity_sessions("2023-06-30", "2023-07-05")
    bars = thirty_minute_bars(sessions)
    hourly = resample_us_equity(bars, sessions, "1h")
    assert len(hourly) == 7 + 4 + 7
    assert hourly.index[0] == ny("2023-06-30 09:30")
    first = bars.loc[ny("2023-06-30 09:30") : ny("2023-06-30 10:00")]
    row = hourly.iloc[0]
    assert row["open"] == first["open"].iloc[0] and row["close"] == first["close"].iloc[-1]
    assert row["high"] == first["high"].max() and row["volume"] == 200.0
    last_bar = bars.loc[ny("2023-06-30 15:30")]
    assert hourly.loc[ny("2023-06-30 15:30"), "close"] == last_bar["close"]  # a half hour
    daily = resample_us_equity(bars, sessions, "1d")
    assert daily.loc[ny("2023-07-03 09:30"), "volume"] == 7 * 100.0  # 09:30 to 13:00
    four = resample_us_equity(bars, sessions, "4h")
    assert list(four.index.tz_convert(NY).strftime("%H:%M")) == ["09:30", "13:30", "09:30",
                                                                 "09:30", "13:30"]  # fmt: skip
    assert infer_calendar(hourly.index) == "24/7"  # three days are too short to tell


def test_missing_buckets_are_filled_flat_and_empty_input() -> None:
    sessions = us_equity_sessions("2023-03-01", "2023-03-31")
    bars = thirty_minute_bars(sessions, extended=False)
    gap = bars.index[(bars.index >= ny("2023-03-02 11:30")) & (bars.index < ny("2023-03-02 12:30"))]
    hourly = resample_us_equity(bars.drop(gap), sessions, "1h")
    filled = hourly.loc[ny("2023-03-02 11:30")]
    before = hourly.loc[ny("2023-03-02 10:30"), "close"]
    assert filled["volume"] == 0
    assert filled[["open", "high", "low", "close"]].tolist() == [before] * 4
    assert infer_calendar(hourly.index) == "us_equity"
    assert periods_per_year(hourly.index) == 252 * 7
    assert resample_us_equity(bars.iloc[:0], sessions, "1h").empty
    with pytest.raises(ValueError, match="no US equity bars"):
        resample_us_equity(bars, sessions, "2h")
    assert fill_flat(hourly.iloc[:0], hourly.index).empty


def fx_hours(start: str, end: str) -> pd.DataFrame:
    """Every UTC hour (weekends included, flat and without volume there), Dukascopy style."""

    index = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left", name="timestamp")
    price = 1.0 + np.arange(len(index)) * 1e-4
    frame = pd.DataFrame(
        {"open": price, "high": price + 5e-5, "low": price - 5e-5, "close": price + 2e-5,
         "volume": 10.0},
        index=index,
    )  # fmt: skip
    local = index.tz_convert(NY)
    closed = (
        (local.dayofweek == 5)
        | ((local.dayofweek == 4) & (local.hour >= 17))
        | ((local.dayofweek == 6) & (local.hour < 17))
    )
    frame.loc[closed, ["open", "high", "low", "close"]] = 1.5
    frame.loc[closed, "volume"] = 0.0
    return frame


def test_fx_trading_week() -> None:
    hours = fx_trading_hours(fx_hours("2023-03-06", "2023-03-20"))
    local = hours.index.tz_convert(NY)
    assert not (local.dayofweek == 5).any()
    assert local[0].strftime("%a %H:%M") == "Sun 19:00"  # the data starts 00:00 UTC
    assert local[-1].strftime("%a %H:%M") == "Sun 19:00"
    assert local[local.dayofweek == 4].hour.max() == 16  # Friday's last hour opens at 16:00
    # Sunday 19:00 to Friday 17:00 (118 hours), a whole week (120), Sunday 17:00 to 20:00 (3)
    assert len(hours) == 118 + 120 + 3
    # an hour inside the week without any ticks is dropped too
    quiet = fx_hours("2023-03-07", "2023-03-08")
    quiet.iloc[3, :4] = quiet.iloc[3, 0]
    quiet.iloc[3, 4] = 0.0
    assert len(fx_trading_hours(quiet)) == 23


def test_fx_bars_anchor_at_the_new_york_close_across_a_clock_change() -> None:
    hours = fx_trading_hours(fx_hours("2023-03-05", "2023-03-25"))
    daily = resample_fx(hours, "1d")
    local = daily.index.tz_convert(NY)
    assert set(local.strftime("%H:%M")) == {"17:00"}
    assert set(local.dayofweek) == {6, 0, 1, 2, 3}  # Sunday to Thursday evenings
    assert len(daily) == 15
    four = resample_fx(hours, "4h")
    assert set(four.index.tz_convert(NY).hour) == {17, 21, 1, 5, 9, 13}
    utc_hours = set(four.index.hour)
    assert {21, 22} <= utc_hours  # the same New York hour, before and after 12 March
    first = hours.loc[four.index[0] : four.index[1] - pd.Timedelta("1ns")]
    assert four.iloc[0]["open"] == first["open"].iloc[0]
    assert four.iloc[0]["close"] == first["close"].iloc[-1]
    assert four.iloc[0]["volume"] == first["volume"].sum()
    assert resample_fx(hours, "1h").equals(hours[["open", "high", "low", "close", "volume"]])
    assert infer_calendar(daily.index) == "fx" and periods_per_year(daily.index) == 260
    assert resample_fx(hours.iloc[:0], "4h").empty
    with pytest.raises(ValueError, match="no forex bars"):
        resample_fx(hours, "2h")


def test_fx_label_keeps_hourly_bars() -> None:
    index = pd.DatetimeIndex([ny("2023-03-08 10:00")])
    assert fx_label(index, "1h")[0] == index[0]
    assert fx_label(index, "4h")[0] == ny("2023-03-08 09:00")
    assert fx_label(index, "1d")[0] == ny("2023-03-07 17:00")
