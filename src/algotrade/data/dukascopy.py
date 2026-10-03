"""Forex history from Dukascopy's public datafeed: hourly bid and ask candles, no account.

Each month and side is one LZMA-compressed file of 24-byte big-endian records: seconds from
the start of the month, then open, close, low and high in points (1e-5, or 1e-3 for yen
pairs) and volume. Months are numbered from 0 in the path. Weekend hours are in the files as
flat candles with no volume.

Bars are the mid of bid and ask. Only complete months are fetched, and stored hours are never
rewritten: an update appends the months after the last one stored. The datafeed is tried over
HTTPS first; some networks cannot reach it there, and then plain HTTP is used (a damaged file
fails to decompress or fails the checks in :mod:`algotrade.data.quality`).
"""

from __future__ import annotations

import lzma
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .files import bars_path, read_bars, write_bars
from .sessions import fx_trading_hours, resample_fx

SOURCE = "dukascopy"
HOSTS = ("https://datafeed.dukascopy.com", "http://datafeed.dukascopy.com")
START = "2016-01-01"
POINT = {"USDJPY": 1e3, "EURJPY": 1e3, "GBPJPY": 1e3, "AUDJPY": 1e3}
DEFAULT_POINT = 1e5
RECORD = np.dtype(
    [("t", ">i4"), ("open", ">i4"), ("close", ">i4"), ("low", ">i4"), ("high", ">i4"),
     ("volume", ">f4")]
)  # fmt: skip
PRICES = ["open", "high", "low", "close"]


class DukascopyError(RuntimeError):
    """The datafeed could not be reached or returned something unexpected."""


def month_path(pair: str, month: pd.Timestamp, side: str) -> str:
    return f"/datafeed/{pair}/{month.year}/{month.month - 1:02d}/{side}_candles_hour_1.bi5"


def decode_hours(blob: bytes, month: pd.Timestamp, point: float) -> pd.DataFrame:
    """One month's hourly candles of one side, indexed by open time (UTC)."""

    if not blob:
        return pd.DataFrame(columns=[*PRICES, "volume"], index=pd.DatetimeIndex([], tz="UTC"))
    try:
        records = np.frombuffer(lzma.decompress(blob), dtype=RECORD)
    except (lzma.LZMAError, ValueError) as error:
        raise DukascopyError(f"undecodable candle file for {month:%Y-%m}: {error}") from error
    start = pd.Timestamp(month).tz_convert("UTC") if month.tzinfo else month.tz_localize("UTC")
    index = start + pd.to_timedelta(records["t"].astype("int64"), unit="s")
    frame = pd.DataFrame(
        {column: records[column].astype("int64") / point for column in PRICES},
        index=pd.DatetimeIndex(index, name="timestamp"),
    )
    frame["volume"] = records["volume"].astype("float64")
    return frame


def mid_bars(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """Mid prices of the hours both sides have; volume is the average of the two sides'."""

    both = bid.index.intersection(ask.index)
    bid, ask = bid.loc[both], ask.loc[both]
    mid = (bid[PRICES] + ask[PRICES]) / 2
    mid["high"] = np.maximum(mid["high"], mid[["open", "close"]].max(axis=1))
    mid["low"] = np.minimum(mid["low"], mid[["open", "close"]].min(axis=1))
    mid["volume"] = (bid["volume"] + ask["volume"]) / 2
    return mid


class DukascopyClient:
    """Fetches candle files; ``session`` is anything with a requests-style ``get``."""

    def __init__(
        self,
        session=None,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        pause: float = 0.5,
        max_retries: int = 6,
        sleep=time.sleep,
    ) -> None:
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.hosts = list(HOSTS)
        self.timeout = (connect_timeout, read_timeout)
        self.pause = pause
        self.max_retries = max_retries
        self.sleep = sleep

    def get(self, path: str) -> bytes:
        """The file at ``path``; empty when the datafeed has none (404 or an empty file).

        Requests that are rate limited (429) or hit a busy server (503) are retried after a
        growing wait (``Retry-After`` if the server gives one).
        """

        import requests

        attempt = 0
        while True:
            host = self.hosts[0]
            try:
                response = self.session.get(host + path, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as error:
                if len(self.hosts) == 1:
                    raise DukascopyError(f"cannot reach the datafeed: {error}") from error
                self.hosts.pop(0)  # remember for the rest of the run
                continue
            if response.status_code in (429, 503) and attempt < self.max_retries:
                wait = response.headers.get("Retry-After")
                self.sleep(min(float(wait) if wait else 2.0 ** (attempt + 1), 120.0))
                attempt += 1
                continue
            if self.pause:
                self.sleep(self.pause)
            if response.status_code == 404:
                return b""
            if response.status_code != 200:
                raise DukascopyError(f"HTTP {response.status_code} for {path}")
            return response.content

    def month(self, pair: str, month: pd.Timestamp) -> pd.DataFrame:
        """One month of hourly mid bars (all hours, weekends included)."""

        point = POINT.get(pair, DEFAULT_POINT)
        sides = {
            side: decode_hours(self.get(month_path(pair, month, side)), month, point)
            for side in ("BID", "ASK")
        }
        return mid_bars(sides["BID"], sides["ASK"])


def complete_months(first: pd.Timestamp, now: pd.Timestamp) -> list[pd.Timestamp]:
    """Month starts from ``first``'s month up to the last month that has ended before ``now``."""

    first = pd.Timestamp(first.year, first.month, 1)
    last = pd.Timestamp(now.year, now.month, 1) - pd.offsets.MonthBegin(1)
    return list(pd.date_range(first, last, freq="MS")) if first <= last else []


def update_fx(
    root: Path,
    pair: str,
    timeframes: list[str],
    client: DukascopyClient | None = None,
    start: str = START,
    now: pd.Timestamp | None = None,
) -> dict[str, int]:
    """Download or extend a pair's hourly history, then rebuild its bar files.

    The hourly trading-week bars are the stored base (``<PAIR>_1h.parquet``); 4h and daily bars
    are rebuilt from them. Returns the row count of each file written.
    """

    client = client or DukascopyClient()
    now = pd.Timestamp.now("UTC") if now is None else pd.Timestamp(now)
    path = bars_path(root, SOURCE, pair, "1h")
    stored = read_bars(path)
    if stored is None:
        first = pd.Timestamp(start)
    else:
        last = stored.index[-1].tz_convert("UTC")
        first = pd.Timestamp(last.year, last.month, 1) + pd.offsets.MonthBegin(1)
    months = complete_months(first, now.tz_localize(None) if now.tzinfo else now)
    fresh = [fx_trading_hours(client.month(pair, month)) for month in months]
    parts = ([stored] if stored is not None else []) + [f for f in fresh if len(f)]
    if not parts:
        return {}
    hourly = pd.concat(parts)
    hourly = hourly[~hourly.index.duplicated(keep="first")].sort_index()
    counts = {"1h": write_bars(hourly, path)}
    for timeframe in timeframes:
        if timeframe != "1h":
            bars = resample_fx(hourly, timeframe)
            counts[timeframe] = write_bars(bars, bars_path(root, SOURCE, pair, timeframe))
    return counts
