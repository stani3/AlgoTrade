"""Session-anchored bars for markets that close: US stocks and ETFs, and forex.

Every bar is labelled with the UTC time it opens, as for crypto, but the bars follow the
market's own clock:

* US equities, regular hours only (09:30 to 16:00 New York, 13:00 on early-close days, from the
  NYSE calendar): 1h bars open at 09:30, 10:30, ..., 15:30 (the last one half an hour); 4h bars
  at 09:30 and 13:30 (the second one two and a half hours); daily bars cover the session.
* Forex trades from Sunday 17:00 to Friday 17:00 New York time. Hourly bars are kept as they
  are; 4h bars and daily bars start at 17:00 New York (the "New York close" convention), so a
  week has five daily bars, Monday's opening on Sunday evening.

A bar with no trades at all is filled flat at the previous close with zero volume, so every
session has the same bars; :mod:`algotrade.data.quality` reports how many were filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEW_YORK = "America/New_York"
OHLCV = ["open", "high", "low", "close", "volume"]
EQUITY_STEPS = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}
FX_STEPS = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}
FX_ANCHOR = pd.Timedelta(hours=17)  # the New York close


def us_equity_sessions(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """NYSE regular sessions between two dates: ``open`` and ``close`` in UTC, one row each."""

    import exchange_calendars as xcals

    first = pd.Timestamp(start).tz_localize(None).normalize()
    last = pd.Timestamp(end).tz_localize(None).normalize()
    calendar = xcals.get_calendar("XNYS", start=min(first, pd.Timestamp("2016-01-01")))
    schedule = calendar.schedule.loc[first:last, ["open", "close"]]
    schedule.index.name = "session"
    return schedule


def session_buckets(sessions: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """The bars of each session: ``start`` and ``end`` (UTC), in time order."""

    def naive(column: str) -> np.ndarray:  # UTC wall times as plain datetime64 values
        return pd.DatetimeIndex(sessions[column]).tz_convert("UTC").tz_localize(None).to_numpy()

    opens, closes = naive("open"), naive("close")
    if timeframe == "1d":
        starts, ends = opens, closes
    elif len(opens):
        step = EQUITY_STEPS[timeframe].to_timedelta64()
        count = int(np.ceil((closes - opens).max() / step))
        starts = (opens[:, None] + np.arange(count) * step).ravel()
        session_close = np.repeat(closes, count)
        ends = np.minimum(starts + step, session_close)
        keep = starts < session_close
        starts, ends = starts[keep], ends[keep]
    else:
        starts = ends = opens
    return pd.DataFrame(
        {"start": pd.DatetimeIndex(starts).tz_localize("UTC"),
         "end": pd.DatetimeIndex(ends).tz_localize("UTC")}
    )  # fmt: skip


def _aggregate(bars: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    grouped = bars.groupby(labels.to_numpy(), sort=True)
    out = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
        }
    )
    out.index = pd.DatetimeIndex(out.index, name="timestamp")
    return out


def fill_flat(bars: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """``bars`` on ``grid`` (from the first bar on); a missing bar is flat at the last close."""

    grid = grid[grid >= bars.index[0]] if len(bars) else grid[:0]
    out = bars.reindex(grid)
    missing = out["close"].isna()
    close = out["close"].ffill()
    for column in ("open", "high", "low", "close"):
        out[column] = out[column].where(~missing, close)
    out["volume"] = out["volume"].fillna(0.0)
    out.index.name = "timestamp"
    return out


def resample_us_equity(bars: pd.DataFrame, sessions: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Regular-hours bars of ``timeframe`` from finer bars (indexed by UTC open time).

    Bars outside a session's regular hours are dropped; buckets with no bars inside a session
    are filled flat.
    """

    if timeframe not in ("1h", "4h", "1d"):
        raise ValueError(f"no US equity bars of {timeframe!r}")
    buckets = session_buckets(sessions, timeframe)
    starts = pd.DatetimeIndex(buckets["start"])
    ends = pd.DatetimeIndex(buckets["end"])
    position = starts.searchsorted(bars.index, side="right") - 1
    inside = (position >= 0) & (bars.index < ends[np.clip(position, 0, None)])
    kept = bars[inside]
    labels = pd.Series(starts[position[inside]], index=kept.index)
    if kept.empty:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    out = _aggregate(kept, labels)
    return fill_flat(out, starts[starts <= out.index[-1]])


def fx_trading_hours(bars: pd.DataFrame) -> pd.DataFrame:
    """Hourly forex bars inside the trading week (Sunday 17:00 to Friday 17:00 New York),
    without hours that had no ticks at all (flat, zero volume)."""

    local = bars.index.tz_convert(NEW_YORK)
    weekday, hour = local.dayofweek, local.hour
    open_week = ((weekday == 6) & (hour >= 17)) | (weekday <= 3) | ((weekday == 4) & (hour < 17))
    flat = (
        (bars["volume"] <= 0)
        & (bars["open"] == bars["high"])
        & (bars["high"] == bars["low"])
        & (bars["low"] == bars["close"])
    )
    return bars[open_week & ~flat.to_numpy()]


def fx_label(index: pd.DatetimeIndex, timeframe: str) -> pd.DatetimeIndex:
    """The opening time (UTC) of the ``timeframe`` bar each hour belongs to, counted from
    17:00 New York on the wall clock, so the bars keep their local times across clock changes."""

    step = FX_STEPS[timeframe]
    wall = index.tz_convert(NEW_YORK).tz_localize(None)
    start = (wall - FX_ANCHOR).floor(step) + FX_ANCHOR if step > pd.Timedelta(hours=1) else wall
    return start.tz_localize(NEW_YORK).tz_convert("UTC")


def resample_fx(hourly: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Forex bars of ``timeframe`` from trading-week hourly bars."""

    if timeframe not in FX_STEPS:
        raise ValueError(f"no forex bars of {timeframe!r}")
    if hourly.empty:
        return hourly[OHLCV].copy()
    labels = pd.Series(fx_label(hourly.index, timeframe), index=hourly.index)
    return _aggregate(hourly[OHLCV], labels)
