"""Backup archive manifest and reusable path-safety helpers (Task 18).

A backup archive is a ``.tar.gz`` whose entries are data-root-relative file
paths plus one top-level ``manifest.json``. The manifest is the source of
truth for restore: only files listed in it are ever overwritten, so a
restore can never delete or modify anything outside the archive.

Manifest layout (``format`` ``local-diary-backup``, ``format_version`` 1)::

    {
      "format": "local-diary-backup",
      "format_version": 1,
      "created_at": "<ISO-8601 UTC>",
      "mode": "text" | "full",
      "timezone": "Europe/Moscow",
      "db_included": true,
      "schema_version": "<alembic revision> | null",
      "files": [
        {"path": "db/diary.db", "sha256": "<hex>", "size": 1234,
         "mtime": "<ISO-8601 UTC>"},
        ...
      ]
    }

Path safety
===========
Manifest paths use forward slashes and are always relative to the data root.
This module owns the reusable path helpers (AGENTS.md §14,
docs/privacy-and-safety.md §5): :func:`is_safe_relative_path` applies the
lexical checks (no absolute paths, ``..``/``.``/empty segments, backslashes,
Windows drive prefixes, NUL bytes) and :func:`resolve_data_path` adds the
canonical check (``resolve()`` + ``is_relative_to``) that also rejects
symlinked ancestors escaping the root. Restore and any other consumer of
user-controlled paths use these helpers.

Privacy: manifest paths are server-generated (db file, artifact file names,
audio ``YYYY/MM/<uuid>`` paths); no diary content and no secrets are stored
in the manifest. Repository placeholder files (``.gitkeep``) are never
archived.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from pydantic import BaseModel, Field, field_validator

#: File name of the manifest at the top level of every archive.
MANIFEST_FILENAME = "manifest.json"
#: Value of the ``format`` field of every manifest written by this application.
MANIFEST_FORMAT = "local-diary-backup"
#: Current manifest structure version.
MANIFEST_FORMAT_VERSION = 1

#: Placeholder kept by Git so empty data directories survive in the repo; it
#: is never archived and never cleaned from ``data/temp/``.
GITKEEP_NAME = ".gitkeep"

#: Chunk size used for streaming SHA-256 computation.
_HASH_CHUNK_BYTES = 1024 * 1024

_HEX_DIGITS = frozenset("0123456789abcdef")


class UnsafePathError(ValueError):
    """A supplied path may escape its intended root; nothing was touched."""


def is_safe_relative_path(value: str) -> bool:
    """Return whether ``value`` is a safe data-root-relative file path.

    Lexical policy: non-empty, no NUL bytes, not absolute, forward slashes
    only (backslashes are rejected), no Windows drive prefix, and no empty,
    ``.`` or ``..`` path segments. Traversal via ``..`` is therefore
    impossible by construction.
    """
    if not value or "\x00" in value:
        return False
    if value.startswith("/") or posixpath.isabs(value):
        return False
    if "\\" in value:
        return False
    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        return False
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    return True


def resolve_data_path(data_root: Path, relative: str) -> Path:
    """Resolve ``relative`` to a canonical absolute path below ``data_root``.

    Combines the lexical checks with a canonical one: the resolved candidate
    must stay under the resolved ``data_root`` (this also rejects symlinked
    ancestors that point outside the root). Raises :class:`UnsafePathError`
    on any violation.
    """
    if not is_safe_relative_path(relative):
        raise UnsafePathError(
            "Unsafe path: expected a relative path inside the data root "
            "without '..', absolute paths or symlink escapes."
        )
    root = data_root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise UnsafePathError("Unsafe path: the resolved location escapes the data root.")
    return candidate


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 hex digest of a file (never read whole into memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream: IO[bytes]) -> str:
    """Streaming SHA-256 hex digest of a binary stream."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def iso_utc(value: datetime | None = None) -> str:
    """ISO-8601 UTC timestamp string (``value`` defaults to now)."""
    moment = value if value is not None else datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


class ManifestEntry(BaseModel):
    """One archived file: its data-root-relative path and content metadata."""

    path: str
    sha256: str
    size: int
    mtime: str

    @field_validator("path")
    @classmethod
    def _path_is_safe(cls, value: str) -> str:
        if not is_safe_relative_path(value):
            raise ValueError("manifest path must be relative and confined to the data root")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_is_hex(cls, value: str) -> str:
        lowered = value.lower()
        if len(lowered) != 64 or any(ch not in _HEX_DIGITS for ch in lowered):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        return lowered

    @field_validator("size")
    @classmethod
    def _size_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("size must not be negative")
        return value


class BackupManifest(BaseModel):
    """Validated manifest of one backup archive."""

    format: str
    format_version: int
    created_at: str
    mode: str
    timezone: str
    db_included: bool
    schema_version: str | None = None
    files: list[ManifestEntry] = Field(default_factory=list)

    @field_validator("format")
    @classmethod
    def _known_format(cls, value: str) -> str:
        if value != MANIFEST_FORMAT:
            raise ValueError(f"unsupported manifest format {value!r}")
        return value

    @field_validator("format_version")
    @classmethod
    def _known_format_version(cls, value: int) -> int:
        if value != MANIFEST_FORMAT_VERSION:
            raise ValueError(f"unsupported manifest format version {value!r}")
        return value

    @field_validator("mode")
    @classmethod
    def _known_mode(cls, value: str) -> str:
        if value not in {"text", "full"}:
            raise ValueError(f"unsupported backup mode {value!r}")
        return value

    @field_validator("created_at")
    @classmethod
    def _created_at_is_iso(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp") from exc
        return value


def manifest_to_bytes(manifest: BackupManifest) -> bytes:
    """Serialize a manifest to UTF-8 JSON bytes (one trailing newline)."""
    return manifest.model_dump_json(indent=2).encode("utf-8") + b"\n"


def parse_manifest(data: bytes) -> BackupManifest:
    """Parse and validate manifest bytes.

    Raises ``ValueError`` (including pydantic ``ValidationError``) on invalid
    JSON or invalid content; never touches the filesystem.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest is not valid UTF-8") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("manifest is not valid JSON") from exc
    return BackupManifest.model_validate(raw)


def manifest_problems(manifest: BackupManifest) -> list[str]:
    """Structural problems of a parsed manifest beyond per-field validation.

    Currently detects duplicate file paths. Per-entry lexical path safety is
    already enforced by :class:`ManifestEntry` validation during parsing.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(manifest.files):
        if entry.path in seen:
            problems.append(f"file {index}: duplicate path {entry.path!r}")
        seen.add(entry.path)
    return problems


__all__ = [
    "GITKEEP_NAME",
    "MANIFEST_FILENAME",
    "MANIFEST_FORMAT",
    "MANIFEST_FORMAT_VERSION",
    "BackupManifest",
    "ManifestEntry",
    "UnsafePathError",
    "iso_utc",
    "is_safe_relative_path",
    "manifest_problems",
    "manifest_to_bytes",
    "parse_manifest",
    "resolve_data_path",
    "sha256_file",
    "sha256_stream",
]
