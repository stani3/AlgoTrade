from pathlib import Path

import pytest

from algotrade.config import DATA_ROOT_ENV, DEFAULT_SETTINGS, data_root_override, load_settings


def test_without_override_the_defaults_apply() -> None:
    assert data_root_override() is None
    assert load_settings() is DEFAULT_SETTINGS


def test_data_root_override_moves_only_the_raw_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert data_root_override() == tmp_path
    settings = load_settings()
    assert settings.data_paths.raw == tmp_path
    assert settings.data_paths.processed == DEFAULT_SETTINGS.data_paths.processed


def test_blank_override_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, "  ")
    assert data_root_override() is None


def test_yaml_settings(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(f"data_paths:\n  raw: {tmp_path.as_posix()}/r\n")
    assert load_settings(config).data_paths.raw == tmp_path / "r"
    assert load_settings(tmp_path / "missing.yaml") is DEFAULT_SETTINGS
