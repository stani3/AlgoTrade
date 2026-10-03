import pandas as pd
import pytest

from algotrade import calendars
from algotrade.backtest import metrics
from algotrade.calendars import (
    YEAR,
    bars_per_day,
    bars_per_session,
    bars_per_year,
    infer_calendar,
    periods_per_year,
)
from algotrade.strategies import base

NEW_YORK = "America/New_York"


def crypto(freq: str, n: int = 3000) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n, freq=freq, tz="UTC")


def us_equity(timeframe: str, half_day: str | None = "2023-07-03") -> pd.DatetimeIndex:
    """Regular sessions 09:30-16:00 New York (13:00 on ``half_day``), as bar-open times in UTC."""

    starts = {"1h": [f"{h}:30" for h in range(9, 16)], "4h": ["9:30", "13:30"], "1d": ["9:30"]}
    stamps = []
    for day in pd.bdate_range("2023-01-02", "2023-12-29"):
        close = pd.Timestamp(f"{day.date()} 13:00" if str(day.date()) == half_day else
                             f"{day.date()} 16:00", tz=NEW_YORK)  # fmt: skip
        for start in starts[timeframe]:
            stamp = pd.Timestamp(f"{day.date()} {start}", tz=NEW_YORK)
            if stamp < close:
                stamps.append(stamp)
    return pd.DatetimeIndex(stamps).tz_convert("UTC")


def fx(timeframe: str) -> pd.DatetimeIndex:
    """Forex weeks from Sunday 17:00 to Friday 17:00 New York, bars anchored at 17:00 New York,
    over February to April 2023 (the US clocks change on 12 March)."""

    hours = pd.date_range("2023-02-05 17:00", "2023-04-28 16:00", freq="1h", tz=NEW_YORK)
    local = hours.tz_localize(None)
    week_open = (local.dayofweek == 6) & (local.hour >= 17)
    weekday = local.dayofweek <= 3
    friday = (local.dayofweek == 4) & (local.hour < 17)
    hours = hours[week_open | weekday | friday]
    step = {"1h": 1, "4h": 4, "1d": 24}[timeframe]
    since_anchor = (hours.hour - 17) % 24
    return hours[since_anchor % step == 0].tz_convert("UTC")


@pytest.mark.parametrize("freq", ["1h", "4h", "1D", "15min"])
def test_crypto_is_exactly_the_old_formulas(freq: str) -> None:
    index = crypto(freq)
    assert infer_calendar(index) == "24/7"
    old_metric = YEAR / pd.Series(index).diff().median()
    old_strategy = YEAR / (index[1] - index[0])
    assert periods_per_year(index) == old_metric
    assert metrics.periods_per_year(index) == old_metric
    assert bars_per_year(index) == old_strategy
    assert base.bars_per_year(index) == old_strategy
    assert bars_per_day(index) == old_strategy / 365.25
    assert base.bars_per_day(index) == old_strategy / 365.25


def test_crypto_with_a_gap_keeps_both_old_conventions() -> None:
    index = crypto("4h").delete([1, 2, 3])  # the first spacing is 16h, the median 4h
    assert periods_per_year(index) == YEAR / pd.Timedelta(hours=4)
    assert bars_per_year(index) == YEAR / pd.Timedelta(hours=16)


@pytest.mark.parametrize(
    ("timeframe", "per_year", "per_session"), [("1h", 1764, 7), ("4h", 504, 2), ("1d", 252, 1)]
)
def test_us_equity_sessions(timeframe: str, per_year: float, per_session: float) -> None:
    index = us_equity(timeframe)
    assert infer_calendar(index) == "us_equity"
    assert bars_per_session(index, "us_equity") == per_session
    assert periods_per_year(index) == per_year
    assert bars_per_year(index) == per_year
    assert bars_per_day(index) == per_session
    assert metrics.periods_per_year(index) == per_year
    assert base.bars_per_day(index) == per_session


@pytest.mark.parametrize(
    ("timeframe", "per_year", "per_session"), [("1h", 6240, 24), ("4h", 1560, 6), ("1d", 260, 1)]
)
def test_fx_weeks_across_a_clock_change(
    timeframe: str, per_year: float, per_session: float
) -> None:
    index = fx(timeframe)
    # New York changes its clocks on a Sunday at 02:00, while forex is closed, so in UTC the
    # change shortens one weekend gap by an hour and leaves the weekday spacing alone.
    gaps = set(pd.Series(index).diff().dropna())
    weekend = {
        "1h": pd.Timedelta(hours=49),
        "4h": pd.Timedelta(hours=52),
        "1d": pd.Timedelta(days=3),
    }
    assert weekend[timeframe] in gaps
    assert weekend[timeframe] - pd.Timedelta(hours=1) in gaps
    assert infer_calendar(index) == "fx"
    assert bars_per_session(index, "fx") == per_session
    assert periods_per_year(index) == per_year
    assert bars_per_year(index) == per_year
    assert bars_per_day(index) == per_session


def test_short_and_tiny_indexes_count_as_24_7() -> None:
    week = us_equity("1h")[:30]  # about four sessions
    assert infer_calendar(week) == "24/7"
    assert periods_per_year(week) == YEAR / pd.Series(week).diff().median()
    assert periods_per_year(week, "us_equity") == 1764
    one = crypto("4h", 1)
    assert infer_calendar(one) == "24/7"
    assert pd.isna(periods_per_year(one))
    assert bars_per_year(one) == 365.25
    assert bars_per_day(one) == 1.0
    assert bars_per_day(one, "us_equity") == 1.0


def test_weekly_bars_without_weekends_count_as_24_7() -> None:
    weekly = pd.date_range("2022-01-03", periods=100, freq="7D", tz="UTC")  # Mondays only
    assert infer_calendar(weekly) == "24/7"


def test_explicit_calendar_overrides_inference() -> None:
    index = crypto("1h")
    assert periods_per_year(index, "fx") == 260 * 24
    assert bars_per_year(index, "us_equity") == 252 * 7


def test_unknown_calendar_and_sessions_of_a_24_7_market() -> None:
    with pytest.raises(ValueError, match="calendar must be one of"):
        periods_per_year(crypto("1h"), "lse")
    with pytest.raises(ValueError, match="no sessions"):
        bars_per_session(crypto("1h"), "24/7")


def test_calendar_constants() -> None:
    assert calendars.CALENDARS == ("24/7", "us_equity", "fx")
    assert set(calendars.SESSIONS_PER_YEAR) == {"us_equity", "fx"}
