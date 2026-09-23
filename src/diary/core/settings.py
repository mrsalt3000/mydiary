"""Application-level settings built on Pydantic Settings (Task 03).

Covers the application timezone and base storage paths. Values are read
from ``APP_*`` environment variables and the repository-root ``.env`` file.

Storage directories are subdirectories of ``APP_DATA_ROOT`` only while no
source sets them: a directory a source set explicitly keeps exactly the value
it was given, even when that value equals the compiled-in default (T38).

Privacy: this module holds no secrets. Secrets are never part of
``AppSettings``; model profiles reference them by environment-variable name
only (see ``diary.core.models_config`` and ``diary.core.profile_status``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root, derived from the package location:
#: ``<root>/src/diary/core/settings.py`` -> parents[3].
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Default location of the private application-data directory.
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

#: sha256 of the NLI cross-encoder weights the summary validation expects (T24).
#:
#: The same value is written into ``scripts/fetch-nli-model.sh`` (a shell script
#: cannot import Python) and a test pins the two against each other; the measured
#: values for both conversions are documented in
#: ``docs/nli-validation-analysis.md`` §3.
PINNED_NLI_MODEL_SHA256 = "ee870cefeddcc1b79ed1eaf8c9cf3ec331d92f0d984945ab6aa32d017aac6c2c"

#: Mapping of settings field -> subdirectory name below the data root.
_SUBDIR_FIELDS: dict[str, str] = {
    "db_dir": "db",
    "audio_dir": "audio",
    "exports_dir": "exports",
    "temp_dir": "temp",
    "backups_dir": "backups",
    "nli_model_dir": "models/nli",
}

#: Settings that hold machine paths outside the data root (they are not
#: subdirectories of it) and therefore only get the "absolute path" treatment:
#: a relative value resolves against the repository root, an absolute one is
#: kept exactly as given (T38).
_TOOL_PATH_FIELDS: tuple[str, ...] = ("gigaam_python", "gigaam_worker_script")


def _absolute_path(value: Any) -> Path | None:
    """Return ``value`` resolved to an absolute path, or ``None`` when it is
    not a path at all (the field itself then reports the validation error)."""
    try:
        path = Path(os.fspath(value))
    except TypeError:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


#: Default MIME types accepted for voice-message uploads (Task 07).
#: Mirrors what browsers emit for MediaRecorder recordings and plain file
#: uploads (architecture brief §8.1). Only types with a built-in storage
#: profile in ``diary.domain.audio.service`` can be uploaded; entries unknown
#: to that profile are rejected at upload time with a clear error.
DEFAULT_ALLOWED_AUDIO_MIME_TYPES: list[str] = [
    "audio/webm",  # MediaRecorder default in Chrome/Edge/Firefox (Opus).
    "audio/ogg",  # Opus in an Ogg container (e.g. Firefox recordings).
    "audio/wav",  # Uploaded PCM WAV.
    "audio/mpeg",  # Uploaded MP3.
    "audio/mp4",  # M4A / MediaRecorder output in Safari.
]


class AppSettings(BaseSettings):
    """Application-wide settings loaded from the environment and ``.env``.

    Every value maps to an ``APP_*`` environment variable (for example
    ``APP_TIMEZONE``, ``APP_DATA_ROOT``, ``APP_DB_DIR``). A storage directory
    that no source sets follows ``APP_DATA_ROOT``; a directory a source sets —
    environment variable, ``.env`` or constructor argument — keeps exactly the
    value it was given, even when that value equals the default.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    #: IANA timezone name used for all local-time display and week math.
    timezone: str = "Europe/Moscow"
    #: Maximum length (in characters) of a single text-message body. Applied
    #: at the API boundary (see ``diary.api.schemas.MessageCreate``).
    max_message_length: int = Field(default=20000, ge=1)
    #: Maximum accepted size of one voice-message upload in mebibytes.
    #: Read from ``APP_MAX_AUDIO_MB`` (the field name would otherwise map to
    #: ``APP_MAX_AUDIO_SIZE_MB``; ``populate_by_name`` keeps construction by
    #: field name working too).
    max_audio_size_mb: int = Field(default=25, ge=1, validation_alias="APP_MAX_AUDIO_MB")
    #: MIME types accepted for voice-message uploads (bare types, no
    #: ``;codecs=...`` parameters). Env value is parsed as JSON, e.g.
    #: ``APP_ALLOWED_AUDIO_MIME_TYPES='["audio/webm","audio/wav"]'``.
    allowed_audio_mime_types: list[str] = DEFAULT_ALLOWED_AUDIO_MIME_TYPES
    #: Retry policy for the durable SQLite job queue (Task 08, ADR-003):
    #: default maximum attempts per job, exponential backoff base (seconds),
    #: backoff multiplier, backoff cap (seconds) and the staleness threshold
    #: (seconds) after which a ``running`` job is considered crashed and may be
    #: reclaimed by a worker (see ``diary.workers.runner``).
    job_max_attempts: int = Field(default=3, ge=1)
    job_backoff_base_s: float = Field(default=2.0, gt=0)
    job_backoff_multiplier: float = Field(default=2.0, gt=0)
    job_max_backoff_s: float = Field(default=300.0, ge=0)
    job_stale_after_s: float = Field(default=300.0, gt=0)
    #: Normalization pipeline (Task 13): maximum number of original export
    #: entries sent to the model in one LLM request. Longer exports are split
    #: into deterministic entry-chunks of at most this size; the pipeline
    #: merges the per-chunk results back in the original order. Read from
    #: ``APP_NORMALIZE_CHUNK_ENTRIES``.
    normalize_chunk_entries: int = Field(default=25, ge=1)
    #: Summarization pipeline (Task 14): context threshold that switches the
    #: run from single-pass to map-reduce, measured in characters of the
    #: normalized entry text. A normalized document at most this long is sent
    #: to the model in one request; a longer one is chunked deterministically
    #: (whole entries only) and summarized per chunk. Read from
    #: ``APP_SUMMARIZE_MAPREDUCE_THRESHOLD``.
    summarize_mapreduce_threshold: int = Field(default=20000, ge=1)
    #: Summarization pipeline (Task 14): maximum size of one map-reduce input
    #: chunk in characters (entries are never split). Read from
    #: ``APP_SUMMARIZE_CHUNK_SIZE``.
    summarize_chunk_size: int = Field(default=10000, ge=1)
    #: Master switch of the wellbeing analysis (layer 4, Task 15). When
    #: ``False`` the app rejects new wellbeing runs with a clear error and the
    #: UI hides the run action; previously created reports stay readable.
    #: Read from ``APP_WELLBEING_ENABLED``.
    wellbeing_enabled: bool = True
    #: Retention threshold (seconds) for ``data/temp/`` cleanup (Task 18):
    #: ``python -m diary.cli backup --clean-temp`` removes temp files older
    #: than this many seconds. Only the temp directory is ever cleaned —
    #: audio, exports, backups and the database are never touched. Read from
    #: ``APP_TEMP_RETENTION_S``.
    temp_retention_s: float = Field(default=86400.0, ge=0)
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
    #: Directory holding the pinned NLI cross-encoder of the summary validation
    #: (T24) — ``model.onnx``, ``tokenizer.json``, ``config.json``. The weights
    #: are fetched once by ``scripts/fetch-nli-model.sh`` and never live in the
    #: repository. Read from ``APP_NLI_MODEL_DIR``.
    nli_model_dir: Path = DEFAULT_DATA_ROOT / "models" / "nli"
    #: sha256 the NLI weights must match before they are used (T24). An empty
    #: value disables the pin, which is only useful while measuring a different
    #: conversion; the shipped default is the value measured and documented in
    #: ``docs/nli-validation-analysis.md``.
    nli_model_sha256: str = PINNED_NLI_MODEL_SHA256
    #: Interpreter of the GigaAM sidecar's own virtualenv (T44), e.g.
    #: ``~/.local/share/mydiary/gigaam/venv/bin/python``. It lives in settings
    #: rather than in the model profile because it is a machine path, not a
    #: decoding parameter; ``None`` (the default) means "not provisioned yet",
    #: which the provider reports as a clear unavailability instead of guessing
    #: an interpreter. Read from ``APP_GIGAAM_PYTHON``; an empty value is
    #: treated as unset.
    gigaam_python: Path | None = None
    #: Worker script of the GigaAM sidecar (T44) — the process the application
    #: starts for one transcription. Relative values resolve against the
    #: repository root, like every other path here. Read from
    #: ``APP_GIGAAM_WORKER_SCRIPT``.
    gigaam_worker_script: Path = PROJECT_ROOT / "scripts" / "gigaam-stt-worker.py"
    #: How long one sidecar invocation may take before it is killed and the
    #: attempt fails as a transient error (T44). Generous by default. Basis —
    #: measured 2026-09-23 on the 332 s recording at 20 s chunks
    #: (``docs/stt-backends-analysis.md`` §6.5): model load **15.9–25.6 s**
    #: (4 runs) plus inference **5.997–9.300 s** (5 runs). T44 starts one
    #: sidecar process per file, so the load is paid on every invocation.
    #: Read from ``APP_GIGAAM_WORKER_TIMEOUT_S``.
    gigaam_worker_timeout_s: float = Field(default=1800.0, gt=0)

    @field_validator("gigaam_python", mode="before")
    @classmethod
    def _empty_interpreter_is_unset(cls, value: Any) -> Any:
        """An empty ``APP_GIGAAM_PYTHON`` means "not configured", not the cwd."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

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

    @field_validator("allowed_audio_mime_types")
    @classmethod
    def _validate_audio_mime_types(cls, value: list[str]) -> list[str]:
        """Normalize allowed MIME entries: lowercase, trimmed, de-duplicated."""
        normalized: list[str] = []
        for entry in value:
            cleaned = entry.strip().lower()
            if not cleaned:
                raise ValueError("Allowed audio MIME types must not contain empty entries.")
            if ";" in cleaned or "/" not in cleaned:
                raise ValueError(
                    f"Allowed audio MIME type {cleaned!r} must be a bare type "
                    "such as 'audio/webm' without parameters."
                )
            if cleaned not in normalized:
                normalized.append(cleaned)
        if not normalized:
            raise ValueError("At least one allowed audio MIME type is required.")
        return normalized

    @model_validator(mode="before")
    @classmethod
    def _resolve_data_paths(cls, data: Any) -> Any:
        """Make data paths absolute and inherit the root only where it applies.

        The hook runs *before* validation because a later one could no longer
        tell a set directory from an unset one: an unset field then simply holds
        its compiled-in default, which is exactly the value that must not be
        read as "explicitly set" (T38). ``model_fields_set`` is no substitute
        there, because assigning to a field marks it as set — the previous
        ``mode="after"`` hook added the six directories and ``data_root`` to
        that set through its own assignments. Here, by contrast, the dictionary
        holds only the keys env, ``.env`` and constructor arguments supplied —
        defaults are not applied yet.

        So a directory present here keeps its value — even one equal to the
        compiled-in default, which used to be silently replaced by
        ``APP_DATA_ROOT/<subdir>`` (T38). Only directories absent here follow
        the root, and relative values (of the root and of any directory) resolve
        against the repository root. Values that are not paths are passed
        through untouched for the field to reject.

        The tool paths (:data:`_TOOL_PATH_FIELDS`, T44) get the last two steps
        only: made absolute against the repository root, never defaulted and
        never rebased, because they point outside the data root.
        """
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "data_root" in values:
            root = _absolute_path(values["data_root"])
            if root is None:
                return values
            values["data_root"] = root
        else:
            root = DEFAULT_DATA_ROOT
        for field_name, subdir in _SUBDIR_FIELDS.items():
            if field_name not in values:
                values[field_name] = root / subdir
                continue
            explicit = _absolute_path(values[field_name])
            if explicit is not None:
                values[field_name] = explicit
        for field_name in _TOOL_PATH_FIELDS:
            # Tool paths are not subdirectories of the data root, so they are
            # neither rebased nor defaulted here — only made absolute. A source
            # that set one keeps its value (T38); an empty string for the
            # optional interpreter is passed on and read as "unset".
            if field_name not in values:
                continue
            raw = values[field_name]
            if isinstance(raw, str) and not raw.strip():
                continue
            resolved = _absolute_path(raw)
            if resolved is not None:
                values[field_name] = resolved
        return values

    @property
    def tzinfo(self) -> ZoneInfo:
        """``ZoneInfo`` compiled from :attr:`timezone` (already validated)."""
        return ZoneInfo(self.timezone)

    @property
    def max_audio_size_bytes(self) -> int:
        """Maximum accepted voice-message upload size in bytes."""
        return self.max_audio_size_mb * 1024 * 1024


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
