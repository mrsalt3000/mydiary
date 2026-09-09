"""Command-line interface: backups, recovery and temp cleanup (Task 18).

Commands
========

``python -m diary.cli backup --mode text|full [--clean-temp]``
    Create a timestamped backup archive under ``AppSettings.backups_dir``.
    ``text`` captures the SQLite database and the text artifact layers
    (exports/01_original … 04_wellbeing); ``full`` additionally captures
    ``audio/``. With ``--clean-temp`` the CLI first removes files inside the
    application temp directory older than ``AppSettings.temp_retention_s`` —
    only ``data/temp/`` is ever cleaned.

``python -m diary.cli recover --archive <path> [--confirm]``
    Restore an archive into the current data root. Recovery overwrites
    current diary files and cannot be undone, therefore it requires the
    explicit ``--confirm`` flag; without it the command refuses with a clear
    message and exits non-zero.

Privacy: output and error messages never contain diary content or secrets;
paths, counts and sizes are safe metadata. Settings are read from the
environment / ``.env`` exactly as in the web application (``APP_*``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from diary.core.settings import get_settings
from diary.domain.backup import (
    BackupConfirmRequiredError,
    BackupError,
    BackupMode,
    cleanup_temp,
    create_backup,
    restore,
)

_EXIT_OK = 0
_EXIT_ERROR = 1


def build_argument_parser() -> argparse.ArgumentParser:
    """The ``python -m diary.cli`` argument parser (public for tests)."""
    parser = argparse.ArgumentParser(
        prog="python -m diary.cli",
        description=(
            "Local Diary backup and recovery (Task 18). Creates and verifies "
            "backup archives and restores them with an explicit confirmation "
            "flag. Never logs diary content."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    backup_parser = subparsers.add_parser(
        "backup",
        help="create a backup archive (text = database + text artifacts; full = text + audio)",
    )
    backup_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in BackupMode],
        default=BackupMode.text.value,
        help="what to capture: 'text' (database + text artifacts) or "
        "'full' (text + audio); default: %(default)s",
    )
    backup_parser.add_argument(
        "--clean-temp",
        action="store_true",
        help=(
            "before backing up, remove files inside the application temp "
            "directory older than APP_TEMP_RETENTION_S seconds (only "
            "data/temp is ever cleaned)"
        ),
    )

    recover_parser = subparsers.add_parser(
        "recover",
        help="restore an archive into the current data root (destructive; requires --confirm)",
    )
    recover_parser.add_argument(
        "--archive",
        required=True,
        metavar="PATH",
        help="path to the backup .tar.gz archive to restore",
    )
    recover_parser.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "acknowledge that recovery overwrites current diary files and "
            "cannot be undone; without this flag the command refuses to run"
        ),
    )
    return parser


def _run_backup(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.clean_temp:
        report = cleanup_temp(settings.temp_dir, max_age_s=settings.temp_retention_s)
        print(
            "clean-temp: removed %d file(s), freed %d byte(s), kept %d file(s) "
            "in %s" % (report.removed_files, report.freed_bytes, report.kept_files, report.temp_dir)
        )
    archive = create_backup(mode=BackupMode(args.mode), settings=settings)
    print(f"backup created: {archive} (mode={args.mode})")
    return _EXIT_OK


def _run_recover(args: argparse.Namespace) -> int:
    settings = get_settings()
    report = restore(Path(args.archive).expanduser(), confirm=args.confirm, settings=settings)
    print(
        "restore complete: %d file(s), %d byte(s) restored from %s"
        % (report.restored_files, report.restored_bytes, report.archive)
    )
    return _EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "backup":
            return _run_backup(args)
        if args.command == "recover":
            return _run_recover(args)
    except BackupConfirmRequiredError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_ERROR
    except BackupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_ERROR
    return _EXIT_ERROR  # pragma: no cover - both commands above return early


if __name__ == "__main__":
    raise SystemExit(main())
