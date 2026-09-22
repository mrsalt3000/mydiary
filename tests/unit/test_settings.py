"""Unit tests for application settings (Task 03)."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from diary.core.settings import (
    _SUBDIR_FIELDS,
    DEFAULT_DATA_ROOT,
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
    "APP_NLI_MODEL_DIR",
)


@pytest.fixture(autouse=True)
def _isolate_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: ignore APP_* variables from the developer shell."""
    for name in _APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: object) -> AppSettings:
    # _env_file=None disables dotenv so tests never depend on a real .env.
    return AppSettings(_env_file=None, **overrides)


def _sibling_of(field_name: str) -> str:
    """Another directory field, to check the root still reaches unset ones."""
    return next(name for name in _SUBDIR_FIELDS if name != field_name)


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


@pytest.mark.parametrize(("field_name", "subdir"), sorted(_SUBDIR_FIELDS.items()))
def test_explicit_directory_equal_to_the_default_is_kept_kwarg(
    field_name: str, subdir: str
) -> None:
    """T38: a value a source sets is kept even when it equals the default.

    The sibling directory is the discriminating half: unset directories still
    follow the new root, so a passing assertion cannot come from "nothing was
    rebased at all".
    """
    explicit = DEFAULT_DATA_ROOT / subdir
    settings = _settings(data_root="/srv/diary", **{field_name: explicit})
    sibling = _sibling_of(field_name)
    assert getattr(settings, field_name) == explicit
    assert getattr(settings, sibling) == Path("/srv/diary") / _SUBDIR_FIELDS[sibling]


@pytest.mark.parametrize(("field_name", "subdir"), sorted(_SUBDIR_FIELDS.items()))
def test_explicit_directory_equal_to_the_default_is_kept_env(
    monkeypatch: pytest.MonkeyPatch, field_name: str, subdir: str
) -> None:
    """The same rule through the environment, which is how T27 hit it."""
    monkeypatch.setenv("APP_DATA_ROOT", "/srv/diary")
    monkeypatch.setenv(f"APP_{field_name.upper()}", str(DEFAULT_DATA_ROOT / subdir))
    settings = AppSettings(_env_file=None)
    sibling = _sibling_of(field_name)
    assert getattr(settings, field_name) == DEFAULT_DATA_ROOT / subdir
    assert getattr(settings, sibling) == Path("/srv/diary") / _SUBDIR_FIELDS[sibling]


def test_default_directories_are_the_data_root_subdirs() -> None:
    """Without any APP_* variable every directory is <repo>/data/<subdir>."""
    settings = _settings()
    assert settings.data_root == PROJECT_ROOT / "data"
    for field_name, subdir in _SUBDIR_FIELDS.items():
        assert getattr(settings, field_name) == DEFAULT_DATA_ROOT / subdir


def test_relative_explicit_directory_resolves_against_project_root() -> None:
    """An explicit relative directory keeps resolving against the repository root."""
    settings = _settings(data_root="/srv/diary", db_dir="my/db")
    assert settings.data_root == Path("/srv/diary")
    assert settings.db_dir == PROJECT_ROOT / "my" / "db"
    assert settings.audio_dir == Path("/srv/diary/audio")


def test_a_directory_that_is_not_a_path_is_a_validation_error() -> None:
    """A wrong type is reported by the field, not by a TypeError in the hook."""
    with pytest.raises(ValidationError, match="db_dir"):
        _settings(db_dir=123)
    with pytest.raises(ValidationError, match="data_root"):
        _settings(data_root=123)


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
