"""Backup and recovery use cases plus temp-dir cleanup (Task 18).

Responsibilities
================

* :func:`create_backup` captures the SQLite database and selected data
  directories into a timestamped ``.tar.gz`` under ``AppSettings.backups_dir``
  with a JSON manifest inside the archive. Modes:

  - ``text`` — SQLite plus the text artifacts (``exports/`` layers
    01_original … 04_wellbeing);
  - ``full`` — ``text`` plus ``audio/``.

  The SQLite snapshot is taken with the ``sqlite3`` online backup API
  (``Connection.backup``): it produces a consistent copy of a live
  WAL-mode database without stopping readers/writers and without writing to
  the source file. Everything is first staged under ``data/temp/``, hashed
  into the manifest and only then packed; source data is never modified
  (read-only copy).
* :func:`verify_backup` inspects an archive without extracting anything to
  disk: archive readability, manifest validity, safe paths and content
  hashes/sizes against the manifest.
* :func:`restore` refuses to run unless ``confirm=True`` and overwrites only
  the files listed in the verified manifest, each resolved canonically below
  the current data root (path traversal is rejected before any write).
* :func:`cleanup_temp` deletes files inside the application temp directory
  older than the retention threshold. It can never touch audio, exports,
  backups or the database.

Safety contract (docs/privacy-and-safety.md §5, AGENTS.md §14): errors are
static, contain no diary content and no secret values; the CLI and the tests
are the only intended callers.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tarfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import IO

from diary.core.settings import AppSettings, get_settings
from diary.db.session import DEFAULT_DB_FILENAME
from diary.domain.backup.manifest import (
    GITKEEP_NAME,
    MANIFEST_FILENAME,
    BackupManifest,
    ManifestEntry,
    UnsafePathError,
    iso_utc,
    manifest_problems,
    manifest_to_bytes,
    parse_manifest,
    resolve_data_path,
    sha256_file,
    sha256_stream,
)

#: Chunk size used while streaming archive members.
_STREAM_CHUNK_BYTES = 1024 * 1024

#: Default SQLite connection timeout (seconds) used for the online snapshot.
_SQLITE_TIMEOUT_S = 10.0


class BackupMode(str, Enum):
    """What a backup archive captures."""

    text = "text"
    full = "full"


# ---------------------------------------------------------------------------
# Domain errors (static messages; never diary content or secrets)
# ---------------------------------------------------------------------------


class BackupError(Exception):
    """Base class for backup/recovery domain errors."""


class BackupSourceError(BackupError):
    """Source data cannot be backed up safely."""


class BackupCreationError(BackupError):
    """The archive could not be written."""


class BackupVerificationError(BackupError):
    """An archive failed integrity/path verification; restore is refused."""


class BackupConfirmRequiredError(BackupError):
    """Restore was requested without the mandatory confirmation flag."""


class RestoreError(BackupError):
    """Restore failed or was refused."""


class CleanupError(BackupError):
    """Temp cleanup was refused or failed."""


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of a safe archive inspection."""

    ok: bool
    archive: str
    problems: tuple[str, ...] = ()
    manifest_ok: bool = False
    mode: str | None = None
    db_included: bool | None = None
    schema_version: str | None = None
    checked_files: int = 0


@dataclass(frozen=True)
class RestoreReport:
    """Outcome of a confirmed restore."""

    archive: str
    restored_files: int
    restored_bytes: int
    paths: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CleanupReport:
    """Outcome of a ``data/temp/`` cleanup."""

    temp_dir: str
    max_age_s: float
    removed_files: int
    freed_bytes: int
    kept_files: int


# ---------------------------------------------------------------------------
# Small filesystem helpers
# ---------------------------------------------------------------------------


def _unlink_safe(path: Path) -> None:
    """Remove a file, ignoring "already gone" errors."""
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort cleanup
        pass


def _iter_source_files(root_dir: Path, data_root: Path) -> list[Path]:
    """Regular files below ``root_dir`` in deterministic order.

    ``root_dir`` must lie inside ``data_root`` (a configured directory that
    escapes the data root makes data-root-relative archive paths impossible,
    so the backup fails with a clear error). Files are returned resolved, in
    sorted order; repository ``.gitkeep`` placeholders are skipped. A
    symlink that resolves outside the data root is refused rather than
    silently archived under a data-root name.
    """
    if not root_dir.is_dir():
        return []
    resolved_data = data_root.resolve()
    resolved_root = root_dir.resolve()
    if not resolved_root.is_relative_to(resolved_data):
        raise BackupSourceError(
            "a backup source directory lies outside the data root; keep db/, "
            "audio/ and exports/ inside APP_DATA_ROOT so archive paths stay "
            "data-root-relative"
        )
    files: list[Path] = []
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file() or path.name == GITKEEP_NAME:
            continue
        resolved_file = path.resolve()
        if not resolved_file.is_relative_to(resolved_data):
            raise BackupSourceError(
                f"refusing to back up {path.name!r}: it resolves outside the data root"
            )
        files.append(path)
    return files


def _snapshot_sqlite(source: Path, target: Path) -> None:
    """Consistent SQLite snapshot via the online backup API.

    ``source`` must exist. The copy is written to ``target`` (plain,
    non-WAL). Readers and writers of the live database are not blocked; the
    source file bytes are never modified. On failure a partial ``target`` is
    removed.
    """
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        source_connection = sqlite3.connect(str(source), timeout=_SQLITE_TIMEOUT_S)
    except sqlite3.Error as exc:
        raise BackupSourceError("Failed to open the diary database for backup.") from exc
    try:
        try:
            target_connection = sqlite3.connect(str(target), timeout=_SQLITE_TIMEOUT_S)
        except sqlite3.Error as exc:
            raise BackupSourceError("Failed to create the database snapshot.") from exc
        try:
            with target_connection:
                source_connection.backup(target_connection)
        except sqlite3.Error as exc:
            raise BackupSourceError("Failed to copy the diary database.") from exc
        finally:
            target_connection.close()
    finally:
        source_connection.close()
    if not target.is_file():
        raise BackupSourceError("The database snapshot was not produced.")


def _schema_version_of(db_path: Path) -> str | None:
    """Read the ``alembic_version`` from a closed snapshot file.

    Returns ``None`` when the table is absent or unreadable — informational
    only, never a restore condition.
    """
    if not db_path.is_file():
        return None
    try:
        connection = sqlite3.connect(db_path, timeout=_SQLITE_TIMEOUT_S)
    except sqlite3.Error:
        return None
    try:
        try:
            row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        except sqlite3.DatabaseError:
            return None
        return str(row[0]) if row else None
    finally:
        connection.close()


def _write_atomic_from_stream(source: IO[bytes], target: Path) -> int:
    """Write a binary stream to ``target`` via a same-directory temp + rename.

    Private-data files are created with mode 0o600 (directories are created
    by callers with 0o700), matching the atomic-write convention used by the
    rest of the application (AGENTS.md §12). Returns the number of bytes
    written.
    """
    temp_path = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(temp_path, flags, 0o600)
    except OSError as exc:
        raise RestoreError("Failed to create a temporary restore file.") from exc
    total = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            while True:
                chunk = source.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except BaseException:
        _unlink_safe(temp_path)
        raise
    return total


# ---------------------------------------------------------------------------
# Backup creation
# ---------------------------------------------------------------------------


def _staged_entries(staging: Path) -> list[ManifestEntry]:
    """One manifest entry per staged file (deterministic order)."""
    entries: list[ManifestEntry] = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name == GITKEEP_NAME:
            continue
        stat = path.stat()
        entries.append(
            ManifestEntry(
                path=path.relative_to(staging).as_posix(),
                sha256=sha256_file(path),
                size=stat.st_size,
                mtime=iso_utc(datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)),
            )
        )
    return entries


def _stage_data(
    staging: Path,
    mode: BackupMode,
    settings: AppSettings,
    data_root: Path,
) -> tuple[bool, str | None]:
    """Copy the requested sources into ``staging``; returns (db_included, db_rel).

    The SQLite snapshot is created directly inside the staging tree via the
    online backup API. Exports/audio files are ``copy2``-ed, preserving
    mtimes. The data-root-relative path of every staged file determines its
    archive path.
    """
    db_path = settings.db_dir / DEFAULT_DB_FILENAME
    db_included = False
    db_rel: str | None = None
    if db_path.is_file():
        resolved_db = db_path.resolve()
        if not resolved_db.is_relative_to(data_root.resolve()):
            raise BackupSourceError(
                "the database directory lies outside the data root; keep the "
                "database inside APP_DATA_ROOT so archive paths stay "
                "data-root-relative"
            )
        rel = resolved_db.relative_to(data_root.resolve())
        staged_db = staging / rel
        _snapshot_sqlite(resolved_db, staged_db)
        db_rel = rel.as_posix()
        db_included = True

    roots = [settings.exports_dir]
    if mode is BackupMode.full:
        roots.append(settings.audio_dir)
    for root_dir in roots:
        for source in _iter_source_files(root_dir, data_root):
            rel = source.relative_to(data_root.resolve())
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copy2(source, target)
    return db_included, db_rel


def _write_archive(staging: Path, final_path: Path) -> None:
    """Pack the staging tree into ``final_path`` (``.tar.gz``) atomically.

    Only regular files become archive members; arcnames are the staged
    (data-root-relative) paths, so ``manifest.json`` sits at the top level.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if final_path.exists():
        raise BackupCreationError("a backup with this timestamp already exists; retry in a second")
    temp_path = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tarfile.open(temp_path, "w:gz") as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file() and path.name != GITKEEP_NAME:
                    arcname = path.relative_to(staging).as_posix()
                    archive.add(path, arcname=arcname, recursive=False)
        os.replace(temp_path, final_path)
    except BaseException:
        _unlink_safe(temp_path)
        raise


def create_backup(
    *,
    mode: BackupMode | str,
    settings: AppSettings | None = None,
    now: datetime | None = None,
) -> Path:
    """Create a ``text`` or ``full`` backup archive; returns its path.

    The archive is named ``YYYYmmdd-HHMMSS_<mode>.tar.gz`` (timestamp in the
    application timezone) and lives in ``AppSettings.backups_dir``. Source
    data is only read. Raises :class:`BackupError` subclasses with static,
    privacy-safe messages on failure.
    """
    backup_mode = mode if isinstance(mode, BackupMode) else BackupMode(mode)
    app_settings = settings or get_settings()
    moment = now if now is not None else datetime.now(timezone.utc)
    stamp = moment.astimezone(app_settings.tzinfo).strftime("%Y%m%d-%H%M%S")
    archive_name = f"{stamp}_{backup_mode.value}.tar.gz"

    temp_dir = app_settings.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = temp_dir / f"backup-{backup_mode.value}-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise BackupCreationError("Failed to create the backup staging directory.") from exc
    try:
        data_root = app_settings.data_root
        db_included, db_rel = _stage_data(staging, backup_mode, app_settings, data_root)
        schema_version = None
        if db_rel is not None:
            schema_version = _schema_version_of(staging / db_rel)
        manifest = BackupManifest(
            format="local-diary-backup",
            format_version=1,
            created_at=iso_utc(moment),
            mode=backup_mode.value,
            timezone=app_settings.timezone,
            db_included=db_included,
            schema_version=schema_version,
            files=_staged_entries(staging),
        )
        (staging / MANIFEST_FILENAME).write_bytes(manifest_to_bytes(manifest))
        final_path = app_settings.backups_dir / archive_name
        _write_archive(staging, final_path)
        return final_path
    except BackupError:
        raise
    except OSError as exc:
        raise BackupCreationError("Failed to create the backup archive.") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Archive inspection and restore
# ---------------------------------------------------------------------------


def _read_manifest_from_tar(archive: tarfile.TarFile) -> BackupManifest:
    """Read and validate the single top-level manifest of an open archive."""
    found: list[BackupManifest] = []
    for member in archive.getmembers():
        if member.name != MANIFEST_FILENAME:
            continue
        if not member.isfile():
            raise ValueError(f"{MANIFEST_FILENAME} is not a regular file")
        source = archive.extractfile(member)
        raw = source.read() if source is not None else b""
        found.append(parse_manifest(raw))
    if not found:
        raise ValueError(f"no {MANIFEST_FILENAME} found in the archive")
    if len(found) > 1:
        raise ValueError(f"more than one {MANIFEST_FILENAME} in the archive")
    return found[0]


def _verification_report(
    archive: Path, problems: list[str], *, manifest: BackupManifest | None
) -> VerificationReport:
    """Fold collected problems (and an optional manifest) into a report."""
    if manifest is not None:
        return VerificationReport(
            ok=not problems,
            archive=str(archive),
            problems=tuple(problems),
            manifest_ok=True,
            mode=manifest.mode,
            db_included=manifest.db_included,
            schema_version=manifest.schema_version,
            checked_files=len(manifest.files),
        )
    return VerificationReport(
        ok=False,
        archive=str(archive),
        problems=tuple(problems),
        manifest_ok=False,
        checked_files=0,
    )


def verify_backup(archive_path: str | Path) -> VerificationReport:
    """Inspect an archive without extracting anything to disk.

    Checks: the archive opens and reads; exactly one valid ``manifest.json``
    exists; every manifest path is safe and unique; every listed file is a
    regular member whose size and SHA-256 match the manifest. Returns a
    :class:`VerificationReport`; never raises for a damaged archive (only
    ``OSError``/``tarfile.TarError`` when the file cannot even be opened —
    still reported as a problem, never raised).
    """
    archive = Path(archive_path)
    if not archive.is_file():
        return _verification_report(archive, ["archive file not found"], manifest=None)
    problems: list[str] = []
    manifest: BackupManifest | None = None
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            try:
                manifest = _read_manifest_from_tar(tar)
            except ValueError as exc:
                problems.append(f"manifest invalid: {exc}")
            except (OSError, tarfile.TarError) as exc:
                problems.append(f"manifest cannot be read: {type(exc).__name__}")
            if manifest is not None:
                problems.extend(manifest_problems(manifest))
                members: dict[str, tarfile.TarInfo] = {m.name: m for m in tar.getmembers()}
                for entry in manifest.files:
                    member = members.get(entry.path)
                    if member is None:
                        problems.append(f"missing archive member {entry.path!r}")
                        continue
                    if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                        problems.append(f"non-regular archive member {entry.path!r}")
                        continue
                    if member.size != entry.size:
                        problems.append(
                            f"size mismatch for {entry.path!r}: manifest "
                            f"{entry.size}, archive {member.size}"
                        )
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        problems.append(f"cannot read archive member {entry.path!r}")
                        continue
                    if sha256_stream(source) != entry.sha256:
                        problems.append(f"hash mismatch for {entry.path!r}")
    except (OSError, tarfile.TarError) as exc:
        problems.append(f"archive cannot be read: {type(exc).__name__}")
    return _verification_report(archive, problems, manifest=manifest)


def read_manifest(archive_path: str | Path) -> BackupManifest:
    """Read and validate the manifest of an archive (safe inspection)."""
    archive = Path(archive_path)
    if not archive.is_file():
        raise BackupVerificationError("Archive file not found.")
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            return _read_manifest_from_tar(tar)
    except ValueError as exc:
        raise BackupVerificationError(f"Manifest is invalid: {exc}") from exc
    except (OSError, tarfile.TarError) as exc:
        raise BackupVerificationError(f"Archive cannot be read: {type(exc).__name__}") from exc


def restore(
    archive_path: str | Path,
    *,
    confirm: bool = False,
    settings: AppSettings | None = None,
) -> RestoreReport:
    """Restore an archive into the current data root.

    Refuses to run unless ``confirm=True`` (the CLI equivalent is
    ``--confirm``). The archive must pass :func:`verify_backup` first, so
    every restored path is a regular file already matched against the
    manifest. Only files listed in the manifest are overwritten; each target
    is resolved canonically below the data root (path traversal via ``..``,
    absolute paths or symlinked ancestors is rejected before any write).
    """
    if not confirm:
        raise BackupConfirmRequiredError(
            "Refusing to restore without explicit confirmation. Restore "
            "overwrites current diary files with the backup contents and "
            "cannot be undone. Re-run with confirm=True (CLI: python -m "
            "diary.cli recover --archive <path> --confirm)."
        )
    app_settings = settings or get_settings()
    verification = verify_backup(archive_path)
    if not verification.ok:
        head = "\n- ".join(verification.problems[:5])
        raise BackupVerificationError(
            f"Archive failed verification; refusing to restore:\n- {head}"
        )

    data_root = app_settings.data_root.resolve()
    restored_paths: list[str] = []
    restored_bytes = 0
    try:
        with tarfile.open(str(archive_path), mode="r:gz") as tar:
            manifest = _read_manifest_from_tar(tar)
            members = {member.name: member for member in tar.getmembers()}
            plan: list[tuple[str, tarfile.TarInfo]] = []
            for entry in manifest.files:
                member = members.get(entry.path)
                if member is None or not member.isfile() or member.issym():
                    raise RestoreError("Archive content changed between verification and restore.")
                plan.append((entry.path, member))
            for rel_path, member in plan:
                try:
                    target = resolve_data_path(data_root, rel_path)
                except UnsafePathError as exc:
                    raise RestoreError(str(exc)) from exc
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = tar.extractfile(member)
                if source is None:
                    raise RestoreError("Failed to read an archive member.")
                restored_bytes += _write_atomic_from_stream(source, target)
                restored_paths.append(rel_path)
    except BackupError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise RestoreError(f"Restore failed: {type(exc).__name__}.") from exc
    return RestoreReport(
        archive=str(archive_path),
        restored_files=len(restored_paths),
        restored_bytes=restored_bytes,
        paths=tuple(restored_paths),
    )


# ---------------------------------------------------------------------------
# data/temp cleanup (Task 18 retention policy)
# ---------------------------------------------------------------------------


def cleanup_temp(
    temp_dir: str | Path,
    *,
    max_age_s: float,
    now: datetime | None = None,
) -> CleanupReport:
    """Delete files in the application temp directory older than ``max_age_s``.

    Policy: only ``AppSettings.temp_dir`` may be cleaned (the function
    refuses any other directory, so audio/exports/backups/the database can
    never be passed by mistake); only entries strictly inside that directory
    are unlinked; the ``.gitkeep`` placeholder is kept; directories emptied
    by the pass are pruned. The temp directory itself is never removed.
    """
    app_settings = get_settings()
    expected = app_settings.temp_dir.resolve()
    root = Path(temp_dir).resolve()
    if root != expected:
        raise CleanupError(
            "cleanup_temp may only be used on the application temp directory "
            "(AppSettings.temp_dir); refusing to clean other directories."
        )
    if not root.is_dir():
        return CleanupReport(
            temp_dir=str(root), max_age_s=max_age_s, removed_files=0, freed_bytes=0, kept_files=0
        )
    moment = now if now is not None else datetime.now(timezone.utc)
    cutoff = moment.timestamp() - max_age_s
    removed_files = 0
    freed_bytes = 0
    kept_files = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink()):
        if path.name == GITKEEP_NAME:
            kept_files += 1
            continue
        try:
            stat = path.lstat()
        except OSError:
            kept_files += 1
            continue
        if stat.st_mtime < cutoff:
            try:
                path.unlink()
            except OSError:
                kept_files += 1
            else:
                removed_files += 1
                freed_bytes += stat.st_size
        else:
            kept_files += 1
    for directory in sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    return CleanupReport(
        temp_dir=str(root),
        max_age_s=max_age_s,
        removed_files=removed_files,
        freed_bytes=freed_bytes,
        kept_files=kept_files,
    )


__all__ = [
    "BackupConfirmRequiredError",
    "BackupCreationError",
    "BackupError",
    "BackupMode",
    "BackupSourceError",
    "BackupVerificationError",
    "CleanupError",
    "CleanupReport",
    "RestoreError",
    "RestoreReport",
    "VerificationReport",
    "cleanup_temp",
    "create_backup",
    "read_manifest",
    "restore",
    "verify_backup",
]
