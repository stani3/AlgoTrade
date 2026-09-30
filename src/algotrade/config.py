"""Centralized settings helpers for file-system based workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataPaths:
    """Directory layout for raw and processed datasets."""

    raw: Path
    processed: Path


@dataclass(frozen=True)
class Settings:
    """Container for all configurable runtime settings."""

    data_paths: DataPaths


DEFAULT_SETTINGS = Settings(
    data_paths=DataPaths(raw=Path("data/raw"), processed=Path("data/processed"))
)


def _coerce_path(value: str | Path) -> Path:
    path = Path(value)
    return path.expanduser().resolve() if not path.is_absolute() else path


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings from YAML if provided, else fall back to sane defaults."""

    if config_path is None:
        return DEFAULT_SETTINGS

    path = Path(config_path)
    if not path.exists():
        return DEFAULT_SETTINGS

    raw_data: Mapping[str, Any] = yaml.safe_load(path.read_text()) or {}
    paths_section = raw_data.get("data_paths", {})

    raw_path = _coerce_path(paths_section.get("raw", DEFAULT_SETTINGS.data_paths.raw))
    processed_path = _coerce_path(
        paths_section.get("processed", DEFAULT_SETTINGS.data_paths.processed)
    )

    return Settings(data_paths=DataPaths(raw=raw_path, processed=processed_path))


def summarize_settings(settings: Settings) -> str:
    """Return a concise, human-readable summary for logging/CLI output."""

    return (
        f"Raw data: {settings.data_paths.raw.as_posix()} | "
        f"Processed data: {settings.data_paths.processed.as_posix()}"
    )
