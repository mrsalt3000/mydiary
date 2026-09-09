"""Backups, recovery and ``data/temp/`` cleanup (Task 18).

Public surface of the package:

- :mod:`diary.domain.backup.manifest` — the archive manifest model and the
  reusable data-root path-safety helpers (``is_safe_relative_path``,
  ``resolve_data_path``);
- :mod:`diary.domain.backup.service` — backup creation (``text``/``full``),
  safe archive verification, confirmed restore, and temp-dir cleanup.

Recovery is an explicit, potentially destructive operation: :func:`restore`
refuses to run without ``confirm=True``, validates the whole archive first
and only ever overwrites files listed in the verified manifest below the
data root (AGENTS.md §14).
"""

from diary.domain.backup.manifest import (
    GITKEEP_NAME,
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    BackupManifest,
    ManifestEntry,
    UnsafePathError,
    is_safe_relative_path,
    iso_utc,
    manifest_problems,
    manifest_to_bytes,
    parse_manifest,
    resolve_data_path,
    sha256_file,
    sha256_stream,
)
from diary.domain.backup.service import (
    BackupConfirmRequiredError,
    BackupCreationError,
    BackupError,
    BackupMode,
    BackupSourceError,
    BackupVerificationError,
    CleanupError,
    CleanupReport,
    RestoreError,
    RestoreReport,
    VerificationReport,
    cleanup_temp,
    create_backup,
    read_manifest,
    restore,
    verify_backup,
)

__all__ = [
    "GITKEEP_NAME",
    "MANIFEST_FILENAME",
    "MANIFEST_FORMAT",
    "MANIFEST_FORMAT_VERSION",
    "BackupConfirmRequiredError",
    "BackupCreationError",
    "BackupError",
    "BackupManifest",
    "BackupMode",
    "BackupSourceError",
    "BackupVerificationError",
    "CleanupError",
    "CleanupReport",
    "ManifestEntry",
    "RestoreError",
    "RestoreReport",
    "UnsafePathError",
    "VerificationReport",
    "cleanup_temp",
    "create_backup",
    "iso_utc",
    "is_safe_relative_path",
    "manifest_problems",
    "manifest_to_bytes",
    "parse_manifest",
    "read_manifest",
    "resolve_data_path",
    "restore",
    "sha256_file",
    "sha256_stream",
    "verify_backup",
]
