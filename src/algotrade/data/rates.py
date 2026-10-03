"""Financing charges for instruments that are not perpetual futures, from FRED's free rates.

Research measures every instrument in excess of cash, as futures and perpetuals are: holding a
position costs the interest on the money behind it. The charges go into the ``funding_rate``
column (a fraction of the position; longs pay a positive charge, shorts receive it), so the
backtest engines need no change.

* US stocks and ETFs (``fed_funds``): the effective federal funds rate, ACT/360, charged at each
  session's close for the calendar days to the next session (a weekend counts three days).
* Forex (``rate_difference``): a long position in BASE/QUOTE earns the base currency's
  overnight rate and pays the quote currency's, so it is charged (quote - base) x days/360 at
  each 17:00 New York rollover, Monday to Friday, with Wednesday's counting three days for the
  weekend (spot settles two business days later). Brokers add a markup, left out here.

The charge is paid on the position held from its time on (:func:`algotrade.data.market.
align_to_bars`). Rates are overnight rates: fed funds, the ECB deposit rate, the Japanese call
rate, SONIA and the Australian interbank overnight rate (monthly series are held for the month).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .files import funding_file
from .sessions import NEW_YORK, us_equity_sessions

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
CURRENCY_SERIES = {
    "USD": "DFF",
    "EUR": "ECBDFR",
    "JPY": "IRSTCI01JPM156N",
    "GBP": "IUDSOIA",
    "AUD": "IRSTCI01AUM156N",
}
DAY_COUNT = 360.0
ROLLOVER = pd.Timedelta(hours=17)


class RatesError(RuntimeError):
    """A rate series could not be downloaded or read."""


def parse_fred(text: str) -> pd.Series:
    """FRED's CSV (date, value in percent; '.' for no value) as a daily-indexed fraction."""

    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    if frame.shape[1] != 2 or frame.empty:
        raise RatesError("unexpected FRED file")
    dates = pd.to_datetime(frame.iloc[:, 0])
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce") / 100.0
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=frame.columns[1])


def fetch_series(series: str, session=None) -> pd.Series:
    if session is None:
        import requests

        session = requests.Session()
    response = session.get(FRED.format(series=series), timeout=60)
    if response.status_code != 200:
        raise RatesError(f"HTTP {response.status_code} for FRED series {series}")
    return parse_fred(response.content.decode("utf-8"))


def daily(rates: pd.Series, until: pd.Timestamp) -> pd.Series:
    """Every calendar day up to ``until``, each with the latest published rate (monthly series
    hold for the month; days before the first value have none)."""

    rates = rates.dropna()
    days = pd.date_range(rates.index[0], max(pd.Timestamp(until), rates.index[-1]), freq="D")
    return rates.reindex(days).ffill()


def us_financing(sessions: pd.DataFrame, rates: pd.Series) -> pd.DataFrame:
    """One charge per session close, for the calendar days until the next session."""

    if len(sessions) < 2:
        return pd.DataFrame({"timestamp": pd.DatetimeIndex([], tz="UTC"), "funding_rate": []})
    dates = pd.DatetimeIndex(sessions.index)
    days = (dates[1:] - dates[:-1]).days.to_numpy()
    rate = daily(rates, dates[-1]).reindex(dates[:-1]).to_numpy()
    closes = pd.DatetimeIndex(sessions["close"]).tz_convert("UTC")[:-1]
    frame = pd.DataFrame({"timestamp": closes, "funding_rate": rate * days / DAY_COUNT})
    return frame.dropna().reset_index(drop=True)


def fx_rollovers(
    base: pd.Series, quote: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Rollover charges of a long BASE/QUOTE position at 17:00 New York (wall-clock times
    ``start`` to ``end``, inclusive)."""

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    days = days[(days.dayofweek <= 4) & (days + ROLLOVER >= start) & (days + ROLLOVER <= end)]
    if not len(days):
        return pd.DataFrame({"timestamp": pd.DatetimeIndex([], tz="UTC"), "funding_rate": []})
    gap = (daily(quote, days[-1]).reindex(days) - daily(base, days[-1]).reindex(days)).to_numpy()
    count = np.where(days.dayofweek == 2, 3.0, 1.0)
    stamps = (days + ROLLOVER).tz_localize(NEW_YORK).tz_convert("UTC")
    frame = pd.DataFrame({"timestamp": stamps, "funding_rate": gap * count / DAY_COUNT})
    return frame.dropna().reset_index(drop=True)


def _append(frame: pd.DataFrame, path: Path) -> int:
    """Add the charges after the last stored one; stored charges are never rewritten (FRED
    revises some history, and research has hashed the bars these charges belong to)."""

    if path.exists():
        stored = pd.read_parquet(path)
        if len(stored):
            frame = pd.concat([stored, frame[frame["timestamp"] > stored["timestamp"].max()]])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index(drop=True).to_parquet(path, index=False)
    return len(frame)


def update_us_financing(root: Path, symbol: str, rates: pd.Series, source: str = "alpaca") -> int:
    """Extend a US symbol's financing charges over its stored daily bars; returns the count."""

    from .files import bars_path, read_bars

    bars = read_bars(bars_path(root, source, symbol, "1d"))
    if bars is None:
        return 0
    sessions = us_equity_sessions(bars.index[0], bars.index[-1])
    return _append(us_financing(sessions, rates), funding_file(root, source, symbol))


def update_fx_rollovers(
    root: Path, pair: str, rates: dict[str, pd.Series], source: str = "dukascopy"
) -> int:
    """Extend a pair's rollover charges over its stored hourly bars; returns the count."""

    from .files import bars_path, read_bars

    bars = read_bars(bars_path(root, source, pair, "1h"))
    if bars is None:
        return 0
    base, quote = pair[:3], pair[3:]
    local = bars.index.tz_convert(NEW_YORK)
    charges = fx_rollovers(rates[base], rates[quote], local[0].tz_localize(None),
                           local[-1].tz_localize(None))  # fmt: skip
    return _append(charges, funding_file(root, source, pair))
