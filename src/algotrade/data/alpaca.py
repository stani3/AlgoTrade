"""US stock and ETF history from Alpaca's market data API (free "Basic" plan, since 2016).

Bars come from the consolidated tape (``feed=sip``) and are adjusted for splits, dividends and
spin-offs (``adjustment=all``), so a buy-and-hold series is a total-return series. Symbols are
mapped through renames (``asof``): META returns Facebook's history as well.

The stored base is 30-minute regular-hours bars (``<SYMBOL>_30m.parquet``); the 1h, 4h and
daily files are rebuilt from it with session-anchored buckets (:mod:`algotrade.data.sessions`).

Adjusted prices are rebased every time a dividend is paid, so refetching history would change
bars that research has already used (and whose hashes it recorded). An update therefore only
appends: it refetches from the last stored bar, scales the new bars so that bar's close matches
the stored one, and keeps every stored row as it was. Volume is appended unscaled.

The API keys are read from ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` (e.g. in ``.env``) and
never printed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from .files import COLUMNS, bars_path, read_bars, write_bars
from .sessions import resample_us_equity, us_equity_sessions

SOURCE = "alpaca"
URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
KEY_VARS = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
START = "2016-01-01"
BASE = "30m"
PAGE_LIMIT = 10_000
FIELDS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}


class AlpacaError(RuntimeError):
    """Missing keys, a refused request, or bars that cannot be chained onto the stored ones."""


def missing_keys() -> list[str]:
    return [name for name in KEY_VARS if not os.environ.get(name, "").strip()]


class AlpacaClient:
    """Pages through bars; ``session`` is anything with a requests-style ``get``."""

    def __init__(self, session=None, max_retries: int = 6, pause: float = 0.3) -> None:
        missing = missing_keys()
        if missing:
            raise AlpacaError(
                f"set {' and '.join(missing)} (free Alpaca account, e.g. in .env) to download "
                "US stocks and ETFs"
            )
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.headers = {
            "APCA-API-KEY-ID": os.environ[KEY_VARS[0]].strip(),
            "APCA-API-SECRET-KEY": os.environ[KEY_VARS[1]].strip(),
        }
        self.max_retries = max_retries
        self.pause = pause

    def _page(self, symbol: str, params: dict) -> dict:
        attempt = 0
        while True:
            response = self.session.get(
                URL.format(symbol=symbol), params=params, headers=self.headers, timeout=60
            )
            if response.status_code != 429:
                break
            if attempt == self.max_retries:
                raise AlpacaError(f"rate limited for {symbol} after {attempt} retries")
            reset = response.headers.get("X-RateLimit-Reset")
            wait = max(float(reset) - time.time(), 1.0) if reset else 2.0**attempt
            time.sleep(min(wait, 60.0))
            attempt += 1
        if response.status_code != 200:
            # The message is Alpaca's; the request headers (the keys) are never shown.
            raise AlpacaError(f"HTTP {response.status_code} for {symbol}: {response.text[:200]}")
        return response.json()

    def bars(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe: str = "30Min",
        adjustment: str = "all",
    ) -> pd.DataFrame:
        """Bars opening in ``[start, end)``, indexed by open time (UTC)."""

        params = {
            "timeframe": timeframe,
            "start": _utc(start).isoformat(),
            "end": _utc(end).isoformat(),
            "adjustment": adjustment,
            "feed": "sip",
            "limit": PAGE_LIMIT,
            "sort": "asc",
        }
        rows: list[dict] = []
        while True:
            page = self._page(symbol, params)
            rows += page.get("bars") or []
            token = page.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
            time.sleep(self.pause)
        frame = pd.DataFrame(rows, columns=["t", *FIELDS]).rename(columns=FIELDS)
        index = pd.to_datetime(frame.pop("t"), utc=True)
        frame.index = pd.DatetimeIndex(index, name="timestamp")
        frame = frame.astype("float64")
        return frame[(frame.index >= _utc(start)) & (frame.index < _utc(end))]


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def regular_hours(bars: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """The bars that open inside a regular session (between its open and close)."""

    opens = pd.DatetimeIndex(sessions["open"]).tz_convert("UTC")
    closes = pd.DatetimeIndex(sessions["close"]).tz_convert("UTC")
    position = opens.searchsorted(bars.index, side="right") - 1
    valid = position >= 0
    clipped = position.clip(0)
    inside = valid & (bars.index < closes[clipped])
    return bars[inside]


def chain(stored: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """``fresh`` appended to ``stored`` without touching stored rows.

    ``fresh`` must start at (and contain) the last stored bar; the appended prices are scaled so
    that bar's close equals the stored close, which undoes any re-basing for dividends or
    splits since the last update.
    """

    if stored is None:
        return fresh
    last = stored.index[-1]
    newer = fresh[fresh.index > last]
    if newer.empty:
        return stored
    if last not in fresh.index:
        raise AlpacaError(
            f"cannot chain new bars: the last stored bar ({last}) was not returned again"
        )
    scale = float(stored.at[last, "close"]) / float(fresh.at[last, "close"])
    newer = newer.copy()
    for column in ("open", "high", "low", "close"):
        newer[column] = newer[column] * scale
    return pd.concat([stored, newer[COLUMNS]])


def update_equity(
    root: Path,
    symbol: str,
    timeframes: list[str],
    client: AlpacaClient | None = None,
    start: str = START,
    now: pd.Timestamp | None = None,
) -> dict[str, int]:
    """Download or extend one symbol's history up to the last complete day, then rebuild its
    bar files. Returns the row count of each file written."""

    client = client or AlpacaClient()
    now = pd.Timestamp.now("UTC") if now is None else _utc(now)
    end = now.normalize()  # whole days only: the newest session has closed
    base_path = bars_path(root, SOURCE, symbol, BASE)
    stored = read_bars(base_path)
    begin = _utc(start) if stored is None else stored.index[-1]
    if begin >= end:
        return {}
    sessions = us_equity_sessions(begin, end - pd.Timedelta(days=1))
    fresh = regular_hours(client.bars(symbol, begin, end), sessions)
    bars = chain(stored, fresh)
    if bars.empty:
        return {}
    counts = {BASE: write_bars(bars, base_path)}
    every = us_equity_sessions(bars.index[0], bars.index[-1])
    for timeframe in timeframes:
        resampled = resample_us_equity(bars, every, timeframe)
        counts[timeframe] = write_bars(resampled, bars_path(root, SOURCE, symbol, timeframe))
    return counts
