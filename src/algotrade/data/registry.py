"""Lightweight dataset registry that keeps track of local files and loaders."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FrameLoader = Callable[[Path], pd.DataFrame]


@dataclass(slots=True)
class DatasetSpec:
    """Metadata describing a dataset and how to read it into memory."""

    name: str
    path: Path
    frequency: str
    loader: FrameLoader
    description: str = ""


_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> None:
    """Register a dataset spec, ensuring we do not accidentally overwrite entries."""

    if spec.name in _REGISTRY:
        raise ValueError(f"Dataset '{spec.name}' already registered")
    _REGISTRY[spec.name] = spec


def unregister_dataset(name: str) -> None:
    """Remove a dataset from the registry, mainly useful for tests."""

    _REGISTRY.pop(name, None)


def list_datasets() -> Iterable[DatasetSpec]:
    return tuple(_REGISTRY.values())


def load_dataset(name: str) -> pd.DataFrame:
    spec = _REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"Dataset '{name}' has not been registered")
    frame = spec.loader(spec.path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def default_frame_loader(path: Path) -> pd.DataFrame:
    """Inspect file suffix and load a DataFrame accordingly."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path}")
