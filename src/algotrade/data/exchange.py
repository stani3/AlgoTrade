"""Download historical perpetual-futures data (OHLCV + funding) through CCXT."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import pandas as pd

SUPPORTED_EXCHANGES = ("binanceusdm", "bybit")
TIMEFRAMES = {"1h": "1h", "4h": "4h", "1d": "1D"}
EARLIEST = pd.Timestamp("2019-01-01", tz="UTC")

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FUNDING_COLUMNS = ["timestamp", "funding_rate"]


@dataclass(frozen=True)
class MarketId:
    """A USDT-margined linear perpetual on a given exchange."""

    exchange: str
    base: str

    @property
    def ccxt_symbol(self) -> str:
        return f"{self.base}/USDT:USDT"

    @property
    def name(self) -> str:
        return f"{self.base}USDT"


def make_exchange(exchange_id: str) -> ccxt.Exchange:
    if exchange_id not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Unsupported exchange '{exchange_id}', use one of {SUPPORTED_EXCHANGES}")
    options = {"defaultType": "swap"} if exchange_id == "bybit" else {}
    return getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": options})


def ohlcv_path(root: Path, market: MarketId, timeframe: str) -> Path:
    return root / market.exchange / f"{market.name}_{timeframe}.parquet"


def funding_path(root: Path, market: MarketId) -> Path:
    return root / market.exchange / f"{market.name}_funding.parquet"


def _timeframe_ms(timeframe: str) -> int:
    return int(pd.Timedelta(TIMEFRAMES[timeframe]).total_seconds() * 1000)


def fetch_ohlcv(
    exchange: ccxt.Exchange, market: MarketId, timeframe: str, since_ms: int
) -> pd.DataFrame:
    """Page through closed candles from ``since_ms`` until now."""

    step = _timeframe_ms(timeframe)
    limit = 1000
    now_ms = exchange.milliseconds()
    rows: list[list[float]] = []
    cursor = since_ms
    while cursor < now_ms:
        batch = exchange.fetch_ohlcv(market.ccxt_symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            # No data yet (e.g. before listing): skip ahead a full page.
            cursor += step * limit
            continue
        rows.extend(batch)
        next_cursor = batch[-1][0] + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    # Drop the still-forming candle so every stored bar is final.
    frame = frame[frame["timestamp"] + step <= now_ms]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_funding(exchange: ccxt.Exchange, market: MarketId, since_ms: int) -> pd.DataFrame:
    """Page through settled funding rates from ``since_ms`` until now."""

    limit = 1000 if exchange.id == "binanceusdm" else 200
    skip = int(pd.Timedelta("8h").total_seconds() * 1000) * limit
    now_ms = exchange.milliseconds()
    rows: list[tuple[int, float]] = []
    cursor = since_ms
    while cursor < now_ms:
        batch = exchange.fetch_funding_rate_history(market.ccxt_symbol, since=cursor, limit=limit)
        if not batch:
            cursor += skip
            continue
        rows.extend((item["timestamp"], item["fundingRate"]) for item in batch)
        next_cursor = batch[-1]["timestamp"] + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    frame = pd.DataFrame(rows, columns=FUNDING_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["funding_rate"] = frame["funding_rate"].astype("float64")
    return frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _merge_and_save(existing: pd.DataFrame | None, fresh: pd.DataFrame, path: Path) -> pd.DataFrame:
    frame = fresh if existing is None else pd.concat([existing, fresh], ignore_index=True)
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    frame = frame.reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def _resume_from(path: Path, step: pd.Timedelta) -> tuple[pd.DataFrame | None, int]:
    if not path.exists():
        return None, int(EARLIEST.timestamp() * 1000)
    existing = pd.read_parquet(path)
    if existing.empty:
        return None, int(EARLIEST.timestamp() * 1000)
    last = existing["timestamp"].max()
    return existing, int((last + step).timestamp() * 1000)


def update_market(
    root: Path,
    market: MarketId,
    timeframes: list[str],
    exchange: ccxt.Exchange | None = None,
    pause: float = 0.0,
) -> dict[str, int]:
    """Download or extend local history for one market. Returns stored row counts."""

    exchange = exchange or make_exchange(market.exchange)
    counts: dict[str, int] = {}
    for timeframe in timeframes:
        path = ohlcv_path(root, market, timeframe)
        existing, since = _resume_from(path, pd.Timedelta(TIMEFRAMES[timeframe]))
        fresh = fetch_ohlcv(exchange, market, timeframe, since)
        counts[timeframe] = len(_merge_and_save(existing, fresh, path))
        time.sleep(pause)

    path = funding_path(root, market)
    existing, since = _resume_from(path, pd.Timedelta("1ms"))
    fresh = fetch_funding(exchange, market, since)
    counts["funding"] = len(_merge_and_save(existing, fresh, path))
    return counts
