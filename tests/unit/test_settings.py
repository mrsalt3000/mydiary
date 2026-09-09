"""Unit tests for application settings (Task 03)."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from diary.core.settings import (
    PROJECT_ROOT,
    AppSettings,
    get_settings,
    reload_settings,
)

_APP_ENV_VARS = (
    "APP_TIMEZONE",
    "APP_DATA_ROOT",
    "APP_DB_DIR",
    "APP_AUDIO_DIR",
    "APP_EXPORTS_DIR",
    "APP_TEMP_DIR",
    "APP_BACKUPS_DIR",
)


@pytest.fixture(autouse=True)
def _isolate_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: ignore APP_* variables from the developer shell."""
    for name in _APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: object) -> AppSettings:
    # _env_file=None disables dotenv so tests never depend on a real .env.
    return AppSettings(_env_file=None, **overrides)


def test_default_values() -> None:
    settings = _settings()
    assert settings.timezone == "Europe/Moscow"
    assert settings.data_root == PROJECT_ROOT / "data"
    assert settings.db_dir == PROJECT_ROOT / "data" / "db"
    assert settings.audio_dir == PROJECT_ROOT / "data" / "audio"
    assert settings.exports_dir == PROJECT_ROOT / "data" / "exports"
    assert settings.temp_dir == PROJECT_ROOT / "data" / "temp"
    assert settings.backups_dir == PROJECT_ROOT / "data" / "backups"
    assert settings.tzinfo == ZoneInfo("Europe/Moscow")


def test_wellbeing_enabled_defaults_and_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wellbeing master switch (Task 15) is on by default and follows
    ``APP_WELLBEING_ENABLED``."""
    assert _settings().wellbeing_enabled is True
    monkeypatch.setenv("APP_WELLBEING_ENABLED", "false")
    assert AppSettings(_env_file=None).wellbeing_enabled is False


def test_valid_timezone_override() -> None:
    settings = _settings(timezone="Asia/Tokyo")
    assert settings.timezone == "Asia/Tokyo"
    assert settings.tzinfo.key == "Asia/Tokyo"


def test_invalid_timezone_raises_clear_error() -> None:
    with pytest.raises(ValidationError, match="not a valid IANA timezone name"):
        _settings(timezone="Mars/Olympus")
    with pytest.raises(ValidationError, match="must not be empty"):
        _settings(timezone="   ")


def test_env_override_reads_app_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_TIMEZONE", "Europe/London")
    settings = AppSettings(_env_file=None)
    assert settings.timezone == "Europe/London"


def test_env_override_for_individual_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_DB_DIR", "/srv/diary/db")
    settings = AppSettings(_env_file=None)
    assert settings.db_dir == Path("/srv/diary/db")
    # Non-overridden directories keep their defaults.
    assert settings.audio_dir == PROJECT_ROOT / "data" / "audio"


def test_data_root_rebases_default_subdirs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_DATA_ROOT", "/srv/diary")
    settings = AppSettings(_env_file=None)
    assert settings.data_root == Path("/srv/diary")
    assert settings.db_dir == Path("/srv/diary/db")
    assert settings.exports_dir == Path("/srv/diary/exports")
    assert settings.backups_dir == Path("/srv/diary/backups")


def test_relative_data_root_resolved_against_project() -> None:
    settings = _settings(data_root="data-rel")
    assert settings.data_root == PROJECT_ROOT / "data-rel"
    assert settings.db_dir == PROJECT_ROOT / "data-rel" / "db"


def test_get_settings_cache_and_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_TIMEZONE", "Pacific/Auckland")
    reload_settings()
    try:
        first = get_settings()
        second = get_settings()
        assert first is second
        assert first.timezone == "Pacific/Auckland"

        monkeypatch.setenv("APP_TIMEZONE", "Europe/Paris")
        reload_settings()
        third = get_settings()
        assert third is not first
        assert third.timezone == "Europe/Paris"
    finally:
        reload_settings()
