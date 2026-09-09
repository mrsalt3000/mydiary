"""Integration tests for the backup/recovery CLI (Task 18).

The CLI reads the same ``APP_*`` settings as the application; every test
points the data root at a per-test temporary directory. Run in-process via
``diary.cli.main`` (no subprocess needed) and checked through capsys.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _migrate import upgrade_database_file

from diary.cli import main
from diary.core.settings import get_settings, reload_settings
from diary.domain.backup import read_manifest

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


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the app data root at a per-test directory and isolate the env."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    monkeypatch.setenv("APP_TEMP_RETENTION_S", "86400")
    reload_settings()


def _seed(settings) -> None:
    """Migrated db + one export file + one stale temp file."""
    upgrade_database_file(settings.db_dir / "diary.db")
    original = settings.exports_dir / "01_original"
    original.mkdir(parents=True, exist_ok=True)
    (original / "2026-09-13_original_0001_r1.txt").write_text(
        "Синтетическая выгрузка CLI.\n", encoding="utf-8"
    )
    temp = settings.temp_dir
    temp.mkdir(parents=True, exist_ok=True)
    stale = temp / "artifact-stale.tmp"
    stale.write_bytes(b"stale")
    old = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
    os.utime(stale, (old, old))


def test_cli_backup_text_creates_archive(capsys: pytest.CaptureFixture[str]) -> None:
    settings = get_settings()
    _seed(settings)

    exit_code = main(["backup", "--mode", "text"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "backup created:" in captured.out
    archives = list(settings.backups_dir.glob("*_text.tar.gz"))
    assert archives
    assert read_manifest(archives[0]).mode == "text"


def test_cli_backup_with_clean_temp_removes_only_stale_temp(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = get_settings()
    _seed(settings)
    stale = settings.temp_dir / "artifact-stale.tmp"
    assert stale.exists()

    exit_code = main(["backup", "--mode", "full", "--clean-temp"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "clean-temp:" in captured.out
    assert not stale.exists()
    assert (settings.exports_dir / "01_original" / "2026-09-13_original_0001_r1.txt").exists()
    assert list(settings.backups_dir.glob("*_full.tar.gz"))


def test_cli_backup_default_mode_is_text(capsys: pytest.CaptureFixture[str]) -> None:
    settings = get_settings()
    _seed(settings)
    assert main(["backup"]) == 0
    assert list(settings.backups_dir.glob("*_text.tar.gz"))


def test_cli_recover_without_confirm_refuses(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = get_settings()
    _seed(settings)
    archive = main(["backup", "--mode", "text"])
    assert archive == 0
    archive_path = next(settings.backups_dir.glob("*_text.tar.gz"))

    exit_code = main(["recover", "--archive", str(archive_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Refusing to restore without explicit confirmation" in captured.err
    # Source data is untouched.
    assert (settings.exports_dir / "01_original" / "2026-09-13_original_0001_r1.txt").exists()


def test_cli_recover_with_confirm_restores(capsys: pytest.CaptureFixture[str]) -> None:
    settings = get_settings()
    _seed(settings)
    target = settings.exports_dir / "01_original" / "2026-09-13_original_0001_r1.txt"
    assert main(["backup", "--mode", "text"]) == 0
    archive_path = next(settings.backups_dir.glob("*_text.tar.gz"))

    # Mutate the export file, then restore it via the CLI.
    target.write_text("mutated by the test", encoding="utf-8")
    exit_code = main(["recover", "--archive", str(archive_path), "--confirm"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "restore complete:" in captured.out
    assert target.read_text(encoding="utf-8") == "Синтетическая выгрузка CLI.\n"


def test_cli_recover_rejects_nonexistent_archive(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["recover", "--archive", "/no/such/archive.tar.gz", "--confirm"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_cli_rejects_unknown_mode(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["backup", "--mode", "everything"])
