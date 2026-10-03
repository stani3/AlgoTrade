"""Where non-crypto market data lives: ``<root>/<source>/<SYMBOL>_<timeframe>.parquet``.

The layout matches the crypto files (:func:`algotrade.data.exchange.ohlcv_path`): a
``timestamp`` column with each bar's UTC open time, then open, high, low, close and volume.
Financing charges sit next to the bars in ``<SYMBOL>_funding.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def bars_path(root: Path, source: str, symbol: str, timeframe: str) -> Path:
    return Path(root) / source / f"{symbol}_{timeframe}.parquet"


def funding_file(root: Path, source: str, symbol: str) -> Path:
    return Path(root) / source / f"{symbol}_funding.parquet"


def read_bars(path: Path) -> pd.DataFrame | None:
    """Stored bars indexed by open time, or None if there are none yet."""

    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty:
        return None
    return frame.set_index("timestamp").sort_index()


def write_bars(bars: pd.DataFrame, path: Path) -> int:
    """Store bars (indexed by open time); returns the row count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = bars[COLUMNS].copy()
    frame.index.name = "timestamp"
    frame.reset_index().to_parquet(path, index=False)
    return len(frame)
