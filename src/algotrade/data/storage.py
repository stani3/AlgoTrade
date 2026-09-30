"""Utilities for managing on-disk datasets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from algotrade.config import DataPaths


class StorageManager:
    """Provides high-level helpers for storing and retrieving market data."""

    def __init__(self, paths: DataPaths) -> None:
        self._paths = paths

    @property
    def paths(self) -> DataPaths:
        return self._paths

    def ensure_structure(self) -> None:
        for path in (self._paths.raw, self._paths.processed):
            path.mkdir(parents=True, exist_ok=True)

    def save_processed_frame(self, frame: pd.DataFrame, filename: str) -> Path:
        """Persist a processed DataFrame to CSV or Parquet based on suffix."""

        target = self._paths.processed / filename
        target.parent.mkdir(parents=True, exist_ok=True)

        suffix = target.suffix.lower()
        if suffix == ".csv":
            frame.to_csv(target, index=False)
        elif suffix == ".parquet":
            frame.to_parquet(target, index=False)
        else:
            raise ValueError("Supported formats are CSV and Parquet")
        return target

    def resolve_processed(self, filename: str) -> Path:
        return self._paths.processed / filename

    def iter_processed(self) -> Iterable[Path]:
        if not self._paths.processed.exists():
            return ()
        return tuple(self._paths.processed.glob("*"))

    def read_frame(self, filename: str) -> pd.DataFrame:
        path = self.resolve_processed(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported format: {path}")
