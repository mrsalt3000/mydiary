"""Application-level settings built on Pydantic Settings (Task 03).

Covers the application timezone and base storage paths. Values are read
from ``APP_*`` environment variables and the repository-root ``.env`` file.

Privacy: this module holds no secrets. Secrets are never part of
``AppSettings``; model profiles reference them by environment-variable name
only (see ``diary.core.models_config`` and ``diary.core.profile_status``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root, derived from the package location:
#: ``<root>/src/diary/core/settings.py`` -> parents[3].
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Default location of the private application-data directory.
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

#: Mapping of settings field -> subdirectory name below the data root.
_SUBDIR_FIELDS: dict[str, str] = {
    "db_dir": "db",
    "audio_dir": "audio",
    "exports_dir": "exports",
    "temp_dir": "temp",
    "backups_dir": "backups",
}


class AppSettings(BaseSettings):
    """Application-wide settings loaded from the environment and ``.env``.

    Every value maps to an ``APP_*`` environment variable (for example
    ``APP_TIMEZONE``, ``APP_DATA_ROOT``, ``APP_DB_DIR``). Storage directories
    default to subdirectories of ``APP_DATA_ROOT`` unless overridden
    individually.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: IANA timezone name used for all local-time display and week math.
    timezone: str = "Europe/Moscow"
    #: Maximum length (in characters) of a single text-message body. Applied
    #: at the API boundary (see ``diary.api.schemas.MessageCreate``).
    max_message_length: int = Field(default=20000, ge=1)
    #: Root directory for all private application data.
    data_root: Path = DEFAULT_DATA_ROOT
    #: SQLite database directory.
    db_dir: Path = DEFAULT_DATA_ROOT / "db"
    #: Voice-message audio directory.
    audio_dir: Path = DEFAULT_DATA_ROOT / "audio"
    #: Weekly export artifacts root (layers are nested by later tasks).
    exports_dir: Path = DEFAULT_DATA_ROOT / "exports"
    #: Scratch directory for atomic/temporary writes.
    temp_dir: Path = DEFAULT_DATA_ROOT / "temp"
    #: Backup archives directory.
    backups_dir: Path = DEFAULT_DATA_ROOT / "backups"

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("APP_TIMEZONE must not be empty.")
        try:
            ZoneInfo(name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"{value!r} is not a valid IANA timezone name "
                "(set APP_TIMEZONE, e.g. 'Europe/Moscow')."
            ) from exc
        return name

    @model_validator(mode="after")
    def _resolve_data_paths(self) -> Self:
        """Make data paths absolute and rebase per-directory defaults.

        When ``APP_DATA_ROOT`` is changed, subdirectories that were not
        overridden explicitly follow the new root. Relative overrides are
        resolved against the repository root.
        """
        root = self.data_root
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        self.data_root = root
        for field_name, subdir in _SUBDIR_FIELDS.items():
            current = getattr(self, field_name)
            if current == DEFAULT_DATA_ROOT / subdir:
                setattr(self, field_name, root / subdir)
            elif not current.is_absolute():
                setattr(self, field_name, PROJECT_ROOT / current)
        return self

    @property
    def tzinfo(self) -> ZoneInfo:
        """``ZoneInfo`` compiled from :attr:`timezone` (already validated)."""
        return ZoneInfo(self.timezone)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the cached application-settings singleton.

    Overridable in tests by setting ``APP_*`` environment variables (or the
    ``.env`` file) and calling :func:`reload_settings` to clear the cache.
    """
    return AppSettings()


def reload_settings() -> None:
    """Clear the :func:`get_settings` cache (used in tests)."""
    get_settings.cache_clear()
