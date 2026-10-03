"""Trading calendars, read off the bars themselves: how many bars make a year and a day.

Crypto trades around the clock, so a year of 4h bars is 365.25 x 6. Session markets do not:
US stocks and ETFs trade 6.5 hours on about 252 days a year, so a year of 1h bars is 252 x 7
(the last bar of a session is half an hour), not 8766, which would overstate a Sharpe ratio
2.2 times. Forex trades around the clock five days a week, about 260 days a year.

The calendar is inferred from the bar index, so nothing has to carry it along:

1. any bar opening on a Saturday (UTC): ``24/7``;
2. a median step longer than a day (weekly bars): ``24/7``;
3. bars on Sundays but none on Saturdays: ``fx`` (the forex week opens on Sunday evening, UTC);
4. otherwise, given at least a week of bars: ``us_equity``;
5. shorter indexes cannot tell weekends from gaps and count as ``24/7``.

For ``24/7`` every function here returns exactly what the code computed before session markets
were added, so crypto results do not change. Two conventions are kept from then: metrics use the
median bar spacing, strategies the first one.

Session markets count ``SESSIONS_PER_YEAR`` sessions a year (conventions, not this year's exact
count) of ``bars_per_session`` bars each, and a "day" in a strategy parameter is a session.
"""

from __future__ import annotations

import math

import pandas as pd

YEAR = pd.Timedelta(days=365.25)
DAY = pd.Timedelta(days=1)
MIN_SPAN = pd.Timedelta(days=7)

CALENDARS = ("24/7", "us_equity", "fx")
SESSIONS_PER_YEAR = {"us_equity": 252.0, "fx": 260.0}
SESSION_LENGTH = {"us_equity": pd.Timedelta(hours=6, minutes=30), "fx": DAY}

SATURDAY, SUNDAY = 5, 6


def _median_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    return pd.Series(index).diff().median()


def infer_calendar(index: pd.DatetimeIndex) -> str:
    """``24/7``, ``us_equity`` or ``fx``, from which weekdays the bars open on (UTC)."""

    if len(index) < 2 or index[-1] - index[0] < MIN_SPAN:
        return "24/7"
    weekdays = index.dayofweek
    if (weekdays == SATURDAY).any() or _median_step(index) > DAY:
        return "24/7"
    return "fx" if (weekdays == SUNDAY).any() else "us_equity"


def _check(calendar: str) -> str:
    if calendar not in CALENDARS:
        raise ValueError(f"calendar must be one of {CALENDARS}, not {calendar!r}")
    return calendar


def bars_per_session(index: pd.DatetimeIndex, calendar: str) -> float:
    """Bars in one session of a session market.

    US equities: the session length over the shortest bar spacing, rounded up (1h bars make 7,
    the last one half an hour; 4h bars make 2, alternately 4 and 20 hours apart, so the median
    spacing says nothing). Forex: a day over the median spacing, which one stray bar cannot
    move.
    """

    if _check(calendar) == "24/7":
        raise ValueError("a 24/7 market has no sessions")
    steps = pd.Series(index).diff()
    if calendar == "us_equity":
        return float(math.ceil(SESSION_LENGTH[calendar] / steps.min()))
    return float(max(round(SESSION_LENGTH[calendar] / steps.median()), 1))


def periods_per_year(index: pd.DatetimeIndex, calendar: str | None = None) -> float:
    """Bars per year, for annualising per-bar statistics (Sharpe, volatility, CAGR)."""

    if len(index) < 2:
        return float("nan")
    calendar = _check(calendar or infer_calendar(index))
    if calendar == "24/7":
        return YEAR / _median_step(index)
    return SESSIONS_PER_YEAR[calendar] * bars_per_session(index, calendar)


def bars_per_year(index: pd.DatetimeIndex, calendar: str | None = None) -> float:
    """Bars per year as strategies count them (24/7: from the first bar spacing)."""

    if len(index) < 2:
        return 365.25
    calendar = _check(calendar or infer_calendar(index))
    if calendar == "24/7":
        return YEAR / (index[1] - index[0])
    return SESSIONS_PER_YEAR[calendar] * bars_per_session(index, calendar)


def bars_per_day(index: pd.DatetimeIndex, calendar: str | None = None) -> float:
    """Bars per day for strategy parameters given in days; for session markets, per session."""

    calendar = _check(calendar or infer_calendar(index))
    if calendar == "24/7" or len(index) < 2:
        return bars_per_year(index, "24/7") / 365.25
    return bars_per_session(index, calendar)
