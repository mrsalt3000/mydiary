"""Integration tests for backup/recovery/temp-cleanup services (Task 18).

Every test runs against its own temporary data root with a migrated SQLite
database and synthetic files only (no real diary content). Covers: text/full
archive content, manifest correctness, source immutability during backup,
verify on valid/damaged/hostile archives, restore gating and behaviour,
path-traversal refusal, and the ``data/temp/`` retention policy.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _migrate import upgrade_database_file

from diary.core.settings import AppSettings, get_settings, reload_settings
from diary.domain.backup import (
    BackupConfirmRequiredError,
    BackupMode,
    BackupVerificationError,
    CleanupError,
    cleanup_temp,
    create_backup,
    read_manifest,
    restore,
    verify_backup,
)
from diary.domain.backup.manifest import MANIFEST_FILENAME, parse_manifest

#: Environment variables the module must not inherit from the developer shell.
_ENV_VARS = (
    "APP_TIMEZONE",
    "APP_DATA_ROOT",
    "APP_DB_DIR",
    "APP_AUDIO_DIR",
    "APP_EXPORTS_DIR",
    "APP_TEMP_DIR",
    "APP_BACKUPS_DIR",
    "APP_TEMP_RETENTION_S",
)

ORIGINAL_NAME = "2026-09-13_original_0001_r1.txt"
AUDIO_REL = "audio/2026/09/11111111-2222-3333-4444-555555555555.webm"


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the app data root at a per-test directory and isolate the env."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    reload_settings()


def _migrate_db(settings: AppSettings) -> Path:
    """Create a migrated SQLite database below the settings data root."""
    db_path = settings.db_dir / "diary.db"
    upgrade_database_file(db_path)
    return db_path


def _write_sources(settings: AppSettings) -> None:
    """Synthetic export artifact + audio file inside the data root."""
    original = settings.exports_dir / "01_original" / ORIGINAL_NAME
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text("Синтетическая строка выгрузки.\n", encoding="utf-8")
    audio = settings.data_root / AUDIO_REL
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"synthetic-audio-bytes")


def _tree_manifest(settings: AppSettings) -> dict[str, bytes]:
    """Path -> bytes snapshot of every regular file under the data root.

    The backups directory is excluded: it is the expected write target of the
    operation under test, not part of the source state.
    """
    snapshot: dict[str, bytes] = {}
    for path in sorted(settings.data_root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(settings.data_root))
        if rel.startswith("backups/"):
            continue
        snapshot[rel] = path.read_bytes()
    return snapshot


def _archive_names(archive_path: Path) -> list[str]:
    with tarfile.open(archive_path) as archive:
        return archive.getnames()


def _build_hostile_archive(target: Path, manifest_raw: dict[str, object]) -> None:
    """Write a .tar.gz whose manifest is the given (possibly unsafe) raw dict."""
    data = json.dumps(manifest_raw).encode("utf-8")
    with tarfile.open(target, "w:gz") as archive:
        info = tarfile.TarInfo(MANIFEST_FILENAME)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def _hostile_manifest(paths: list[str]) -> dict[str, object]:
    return {
        "format": "local-diary-backup",
        "format_version": 1,
        "created_at": "2026-09-09T10:00:00+00:00",
        "mode": "text",
        "timezone": "UTC",
        "db_included": False,
        "files": [
            {"path": path, "sha256": "a" * 64, "size": 1, "mtime": "2026-09-09T10:00:00+00:00"}
            for path in paths
        ],
    }


# ---------------------------------------------------------------------------
# backup creation
# ---------------------------------------------------------------------------


def test_text_backup_contains_db_and_exports_but_not_audio() -> None:
    settings = get_settings()
    db = _migrate_db(settings)
    _write_sources(settings)

    archive = create_backup(mode=BackupMode.text, settings=settings)

    assert archive.is_file()
    assert archive.parent == settings.backups_dir
    assert "_text.tar.gz" in archive.name
    names = _archive_names(archive)
    assert "manifest.json" in names
    assert db.relative_to(settings.data_root).as_posix() in names
    assert f"exports/01_original/{ORIGINAL_NAME}" in names
    assert not any(name.startswith("audio/") for name in names)
    assert verify_backup(archive).ok


def test_full_backup_includes_audio() -> None:
    settings = get_settings()
    _migrate_db(settings)
    _write_sources(settings)

    archive = create_backup(mode=BackupMode.full, settings=settings)

    assert "_full.tar.gz" in archive.name
    names = _archive_names(archive)
    assert AUDIO_REL in names
    assert f"exports/01_original/{ORIGINAL_NAME}" in names
    assert "db/diary.db" in names
    assert verify_backup(archive).ok


def test_backup_does_not_modify_sources() -> None:
    settings = get_settings()
    _migrate_db(settings)
    _write_sources(settings)
    before = _tree_manifest(settings)

    create_backup(mode=BackupMode.text, settings=settings)
    create_backup(mode=BackupMode.full, settings=settings)

    after = _tree_manifest(settings)
    # Only the backups directory gained files; every other path is unchanged
    # byte-for-byte (including the SQLite database).
    for rel, data in before.items():
        assert after[rel] == data, rel
    assert set(after) - set(before) == set()


def test_backup_manifest_matches_archive_content() -> None:
    settings = get_settings()
    _migrate_db(settings)
    _write_sources(settings)
    archive = create_backup(mode=BackupMode.full, settings=settings)

    manifest = read_manifest(archive)
    assert manifest.mode == "full"
    assert manifest.db_included is True
    assert manifest.format == "local-diary-backup"
    assert manifest.schema_version  # alembic revision was captured
    names = set(_archive_names(archive))
    assert names == {MANIFEST_FILENAME, *(entry.path for entry in manifest.files)}
    # Round-trip of the raw manifest bytes inside the archive.
    with tarfile.open(archive) as tar:
        raw = tar.extractfile(MANIFEST_FILENAME).read()  # type: ignore[union-attr]
    assert parse_manifest(raw) == manifest
    parsed = json.loads(raw)
    assert isinstance(parsed["files"], list)
    assert all(
        isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64 for entry in parsed["files"]
    )


def test_backup_without_database_still_succeeds() -> None:
    settings = get_settings()
    _write_sources(settings)
    archive = create_backup(mode=BackupMode.text, settings=settings)
    manifest = read_manifest(archive)
    assert manifest.db_included is False
    assert verify_backup(archive).ok
    assert f"exports/01_original/{ORIGINAL_NAME}" in _archive_names(archive)


# ---------------------------------------------------------------------------
# verify_backup
# ---------------------------------------------------------------------------


def test_verify_backup_rejects_corrupted_member() -> None:
    settings = get_settings()
    _migrate_db(settings)
    _write_sources(settings)
    archive = create_backup(mode=BackupMode.full, settings=settings)
    assert verify_backup(archive).ok

    # Rebuild the archive with one member's bytes altered (same length, so
    # the recorded size still matches), keeping the original manifest — the
    # content no longer matches the recorded hash.
    tampered = archive.parent / "tampered.tar.gz"
    member_path = f"exports/01_original/{ORIGINAL_NAME}"
    with tarfile.open(archive) as source, tarfile.open(tampered, "w:gz") as target:
        for member in source.getmembers():
            if member.name == member_path:
                raw = source.extractfile(member).read()  # type: ignore[union-attr]
                data = bytes(value ^ 0xFF for value in raw)  # same length
                member.size = len(data)
                target.addfile(member, io.BytesIO(data))
            elif member.isfile():
                raw = source.extractfile(member).read()  # type: ignore[union-attr]
                target.addfile(member, io.BytesIO(raw))

    report = verify_backup(tampered)
    assert report.ok is False
    assert any("hash mismatch" in problem for problem in report.problems)
    assert any(member_path in problem for problem in report.problems)
    with pytest.raises(BackupVerificationError):
        restore(tampered, confirm=True, settings=settings)


def test_verify_backup_missing_file_is_reported() -> None:
    report = verify_backup("/nonexistent/archive.tar.gz")
    assert report.ok is False
    assert report.manifest_ok is False
    assert any("not found" in problem for problem in report.problems)


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------


def test_restore_without_confirm_refuses_and_changes_nothing() -> None:
    settings = get_settings()
    _migrate_db(settings)
    _write_sources(settings)
    archive = create_backup(mode=BackupMode.full, settings=settings)
    before = _tree_manifest(settings)

    with pytest.raises(BackupConfirmRequiredError, match="Refusing to restore"):
        restore(archive, confirm=False, settings=settings)
    with pytest.raises(BackupConfirmRequiredError):
        restore(archive, settings=settings)  # default confirm=False

    assert _tree_manifest(settings) == before


def test_restore_with_confirm_overwrites_only_manifest_files() -> None:
    settings = get_settings()
    _migrate_db(settings)
    _write_sources(settings)
    original_bytes = (settings.exports_dir / "01_original" / ORIGINAL_NAME).read_bytes()
    db_path = settings.db_dir / "diary.db"
    archive = create_backup(mode=BackupMode.full, settings=settings)

    # Break the live data: overwrite the export, drop the audio file, corrupt
    # the database, and add an unrelated file that must survive untouched.
    (settings.exports_dir / "01_original" / ORIGINAL_NAME).write_text("mutated", encoding="utf-8")
    (settings.data_root / AUDIO_REL).unlink()
    db_path.write_bytes(b"this is no longer a sqlite database")
    unrelated = settings.data_root / "exports" / "unrelated.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    report = restore(archive, confirm=True, settings=settings)

    assert report.restored_files == len(read_manifest(archive).files)
    # Export restored byte-for-byte; audio file recreated; db is a valid
    # SQLite database again with the migrated schema.
    assert (settings.exports_dir / "01_original" / ORIGINAL_NAME).read_bytes() == original_bytes
    assert (settings.data_root / AUDIO_REL).read_bytes() == b"synthetic-audio-bytes"
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        connection.close()
    assert row is not None and row[0]


def test_restore_rejects_traversal_and_absolute_manifest_paths() -> None:
    """Manifests naming ../ or absolute paths are refused before any write."""
    settings = get_settings()
    _migrate_db(settings)
    hostile_dir = settings.data_root / "exports" / "hostile"
    hostile_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in (
        ("traversal.tar.gz", ["../escaped.txt"]),
        ("absolute.tar.gz", ["/tmp/escaped.txt"]),
    ):
        archive = hostile_dir / name
        _build_hostile_archive(archive, _hostile_manifest(paths))
        report = verify_backup(archive)
        assert report.ok is False
        assert report.manifest_ok is False  # rejected at manifest parsing
        with pytest.raises(BackupVerificationError):
            restore(archive, confirm=True, settings=settings)


def test_restore_rejects_symlink_members() -> None:
    """A symlink member pointing outside must never be extracted as a file."""
    settings = get_settings()
    _migrate_db(settings)
    hostile_dir = settings.data_root / "exports" / "hostile2"
    hostile_dir.mkdir(parents=True, exist_ok=True)
    archive = hostile_dir / "symlink.tar.gz"
    manifest_data = json.dumps(_hostile_manifest(["db/diary.db"])).encode("utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(MANIFEST_FILENAME)
        info.size = len(manifest_data)
        tar.addfile(info, io.BytesIO(manifest_data))
        # The member under the manifest's name is a symlink, not a file.
        link = tarfile.TarInfo("db/diary.db")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"  # outside the data root
        tar.addfile(link)

    report = verify_backup(archive)
    assert report.ok is False
    assert any("non-regular archive member" in problem for problem in report.problems)
    with pytest.raises(BackupVerificationError):
        restore(archive, confirm=True, settings=settings)


# ---------------------------------------------------------------------------
# data/temp cleanup
# ---------------------------------------------------------------------------


def _stamp(path: Path, at: datetime) -> None:
    timestamp = at.timestamp()
    os.utime(path, (timestamp, timestamp))


def test_cleanup_temp_removes_only_old_temp_files() -> None:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    temp = settings.temp_dir
    temp.mkdir(parents=True, exist_ok=True)
    _write_sources(settings)  # audio + exports inside the data root

    old_temp = temp / "artifact-dead.tmp"
    old_temp.write_bytes(b"old")
    new_temp = temp / "artifact-live.tmp"
    new_temp.write_bytes(b"new")
    placeholder = temp / ".gitkeep"
    placeholder.write_bytes(b"")
    nested = temp / "sub"
    nested.mkdir()
    (nested / "old-nested.tmp").write_bytes(b"old")
    _stamp(old_temp, now - timedelta(days=2))
    _stamp(placeholder, now - timedelta(days=2))
    _stamp(nested / "old-nested.tmp", now - timedelta(days=2))
    _stamp(new_temp, now)  # fresh → kept

    old_audio = settings.data_root / AUDIO_REL
    old_export = settings.exports_dir / "01_original" / ORIGINAL_NAME
    _stamp(old_audio, now - timedelta(days=30))
    _stamp(old_export, now - timedelta(days=30))

    report = cleanup_temp(temp, max_age_s=86400, now=now)

    assert report.removed_files == 2  # old top-level + old nested file
    assert report.freed_bytes == 6
    assert report.kept_files == 2  # fresh file + .gitkeep placeholder
    assert not old_temp.exists()
    assert not (nested / "old-nested.tmp").exists()
    assert not nested.exists()  # emptied directory was pruned
    assert new_temp.exists()
    assert placeholder.exists()  # .gitkeep survives
    assert old_audio.exists() and old_export.exists()  # never touched


def test_cleanup_temp_refuses_non_temp_directory() -> None:
    settings = get_settings()
    _write_sources(settings)
    with pytest.raises(CleanupError, match="only be used on the application temp directory"):
        cleanup_temp(settings.audio_dir, max_age_s=86400)
    assert (settings.data_root / AUDIO_REL).exists()


def test_cleanup_temp_reports_empty_when_dir_missing() -> None:
    settings = get_settings()
    # The configured temp dir does not exist yet → zero report, no error.
    report = cleanup_temp(settings.temp_dir, max_age_s=86400)
    assert report.removed_files == 0
    assert report.freed_bytes == 0
