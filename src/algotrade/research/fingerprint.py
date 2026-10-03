"""Behavioural fingerprints: how a strategy is positioned, never how it performed.

A fingerprint is the daily mean position (signed exposure) on a fixed reference set of symbols
over the development period. Two strategies whose fingerprints correlate strongly, or whose
exposures are nearly the same, trade the same way however differently they are written.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from algotrade.backtest.costs import CostModel
from algotrade.backtest.runner import backtest
from algotrade.strategies import from_spec

from .criteria import Criteria


def compute(spec: dict, universe: dict[str, pd.DataFrame], costs: CostModel) -> pd.DataFrame:
    """Daily mean position per symbol (columns) for ``spec`` on ``universe``."""

    strategy = from_spec(spec)
    columns = {}
    for symbol, bars in universe.items():
        position = backtest(strategy, bars, costs).ledger["position"]
        columns[symbol] = position.resample("1D").mean()
    frame = pd.DataFrame(columns).astype("float32")
    frame.index.name = "day"
    return frame


@dataclass(frozen=True)
class Similarity:
    key: str
    label: str
    correlation: float  # NaN when either side never changes exposure
    difference: float  # sum|a - b| / sum(|a| + |b|): 0 identical, 1 never overlapping
    days: int
    duplicate: bool
    source: str = ""  # idea | screen | baseline
    spec: dict | None = None

    def describe(self) -> str:
        corr = "n/a" if np.isnan(self.correlation) else f"{self.correlation:.2f}"
        return f"{self.label}: correlation {corr}, exposure difference {self.difference:.2f}"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["correlation"] = None if np.isnan(self.correlation) else self.correlation
        return out


def compare(
    a: pd.DataFrame, b: pd.DataFrame, criteria: Criteria, key: str = "", label: str = ""
) -> Similarity | None:
    """Similarity on the days and symbols both cover, or None if they barely overlap."""

    symbols = [s for s in a.columns if s in b.columns]
    days = a.index.intersection(b.index)
    if not symbols or len(days) < criteria.get("dedup.min_overlap_days"):
        return None
    x = a.loc[days, symbols].to_numpy(dtype="float64").ravel()
    y = b.loc[days, symbols].to_numpy(dtype="float64").ravel()
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    scale = np.abs(x).sum() + np.abs(y).sum()
    difference = float(np.abs(x - y).sum() / scale) if scale > 0 else 0.0
    if x.std() > 0 and y.std() > 0:
        correlation = float(np.corrcoef(x, y)[0, 1])
    else:
        correlation = float("nan")
    duplicate = bool(
        (correlation >= criteria.get("dedup.max_position_correlation"))
        or difference <= criteria.get("dedup.min_exposure_difference")
    )
    return Similarity(key, label, correlation, difference, len(days), duplicate)


class FingerprintStore:
    """Parquet fingerprints with a JSON sidecar each (spec, timeframe and origin)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key(spec_hash: str, timeframe: str, universe: str = "crypto") -> str:
        """``<spec hash>_<timeframe>``, plus ``_<universe>`` for ideas about other asset
        classes (the positions are always measured on the crypto reference symbols)."""

        suffix = "" if universe == "crypto" else f"_{universe}"
        return f"{spec_hash}_{timeframe}{suffix}"

    def paths(self, key: str) -> tuple[Path, Path]:
        return self.directory / f"{key}.parquet", self.directory / f"{key}.json"

    def save(self, key: str, frame: pd.DataFrame, meta: dict) -> list[Path]:
        self.directory.mkdir(parents=True, exist_ok=True)
        data, sidecar = self.paths(key)
        frame.to_parquet(data, compression="zstd")
        sidecar.write_text(
            json.dumps({"key": key, **meta}, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return [data, sidecar]

    def exists(self, key: str) -> bool:
        return self.paths(key)[0].exists()

    def load(self, key: str) -> tuple[pd.DataFrame, dict]:
        data, sidecar = self.paths(key)
        return pd.read_parquet(data), json.loads(sidecar.read_text(encoding="utf-8"))

    def keys(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(path.stem for path in self.directory.glob("*.parquet"))

    def nearest(
        self,
        frame: pd.DataFrame,
        criteria: Criteria,
        exclude_idea: str | None = None,
        exclude_hash: str | None = None,
    ) -> list[Similarity]:
        """Similarity to every stored fingerprint, most similar first.

        Fingerprints of the same idea (its other versions) and of the identical configuration
        are skipped: those cases are governed by the revision and exact-duplicate rules.
        """

        found = []
        for key in self.keys():
            stored, meta = self.load(key)
            if exclude_idea and meta.get("idea") == exclude_idea:
                continue
            if exclude_hash and meta.get("spec_hash") == exclude_hash:
                continue
            similarity = compare(frame, stored, criteria, key, meta.get("label", key))
            if similarity is not None:
                found.append(
                    replace(similarity, source=meta.get("source", ""), spec=meta.get("spec"))
                )
        return sorted(
            found,
            key=lambda s: (s.duplicate, np.nan_to_num(s.correlation, nan=-1.0), -s.difference),
            reverse=True,
        )
