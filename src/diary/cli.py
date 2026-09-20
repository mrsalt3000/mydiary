"""Command-line interface: backups, recovery, temp cleanup and the phase-2 pipeline.

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

``python -m diary.cli day --date YYYY-MM-DD [--tz TZ]``
    Run the phase-2 day chain (task T09/T11) for one local calendar day over
    the application database: every message of the day through ``stt_fix`` and
    then the day through ``sum_daily``. This is the single way to produce a day
    summary from this machine outside the worker.

``python -m diary.cli week --week YYYY-Www [--run-days] [--tz TZ]``
    Run the weekly rollup for one ISO week: the week's **stored** daily
    summaries merged by ``sum_weekly`` (never raw text). ``--run-days`` first
    refreshes the week's days with the day chain above, which is how the whole
    ``raw → stt_fix → sum_daily → sum_weekly`` path is produced in one command.

``python -m diary.cli coach --week YYYY-Www``
    Produce the coaching note (layer 5) of one ISO week. **This command is the
    only launcher of layer 5**: neither the day nor the week chain can reach it,
    and the worker deliberately has no ``coach_note`` job handler (task T07,
    decision 3), so the note is never produced behind the user's back.

Output is identifiers and counts only (run/artifact ids, layer names, profile
and model names, token usage, skip reasons) — never diary text, prompts or model
output. A database that is not migrated yet is an error: create it with
``alembic upgrade head`` under the same ``APP_DATA_ROOT``.

Privacy: output and error messages never contain diary content or secrets;
paths, counts and sizes are safe metadata. Settings are read from the
environment / ``.env`` exactly as in the web application (``APP_*``).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from diary.core.models_config import ModelsConfig, ModelsConfigError, load_models_config
from diary.core.settings import get_settings
from diary.db.session import create_engine, create_session_factory
from diary.domain.backup import (
    BackupConfirmRequiredError,
    BackupError,
    BackupMode,
    cleanup_temp,
    create_backup,
    restore,
)
from diary.domain.pipelines import pipeline as phase2_pipeline
from diary.domain.pipelines.base import PipelineError
from diary.domain.pipelines.pipeline import (
    SKIP_PROFILE_OFF,
    ChainStep,
    CoachStepOutcome,
    DayChainOutcome,
    SkippedStep,
    WeekChainOutcome,
)

_EXIT_OK = 0
_EXIT_ERROR = 1

#: Phase-2 layer → the configuration profile field a skip of it names.
_PROFILE_FIELD_BY_LAYER = {
    "stt_corrected": "diary_stt",
    "daily_summary": "diary_sum",
    "weekly_summary": "diary_week",
    "coach_note": "diary_coach",
}


def _register_phase2_layers() -> None:
    """Make the four phase-2 layers resolvable in this process.

    The orchestrator resolves layers through the second-family registry and
    fails loudly for an unregistered one, so every entry point registers them
    first (guarded and idempotent, exactly as the worker does).
    """
    from diary.domain.pipelines.coach.registration import (
        ensure_coach_pipeline_registered,
    )
    from diary.domain.pipelines.stt_fix.registration import (
        ensure_stt_fix_pipeline_registered,
    )
    from diary.domain.pipelines.sum_daily.registration import (
        ensure_sum_daily_pipeline_registered,
    )
    from diary.domain.pipelines.sum_weekly.registration import (
        ensure_sum_weekly_pipeline_registered,
    )

    ensure_stt_fix_pipeline_registered()
    ensure_sum_daily_pipeline_registered()
    ensure_sum_weekly_pipeline_registered()
    ensure_coach_pipeline_registered()


@contextlib.contextmanager
def _open_session() -> Iterator[Session]:
    """A session over the application database (the same one the web app uses)."""
    engine = create_engine(settings=get_settings())
    try:
        with create_session_factory(engine)() as session:
            yield session
    finally:
        engine.dispose()


def build_argument_parser() -> argparse.ArgumentParser:
    """The ``python -m diary.cli`` argument parser (public for tests)."""
    parser = argparse.ArgumentParser(
        prog="python -m diary.cli",
        description=(
            "Local Diary command line: backups and recovery (Task 18) plus the "
            "phase-2 pipeline (day/week/coach, task T11). Creates, verifies and "
            "restores backup archives with an explicit confirmation flag, and runs "
            "the phase-2 chain over the local database. Never logs diary content."
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

    day_parser = subparsers.add_parser(
        "day",
        help="run the phase-2 day chain (stt_fix per message, then sum_daily) for one local day",
    )
    day_parser.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="the local calendar day to run (the day chain covers exactly this day)",
    )
    day_parser.add_argument(
        "--tz",
        default=None,
        metavar="TZ",
        help="IANA timezone of the day (default: APP_TIMEZONE); stored with the day anchor",
    )
    _add_models_config_option(day_parser)

    week_parser = subparsers.add_parser(
        "week",
        help="run the weekly rollup (sum_weekly over the week's stored day summaries)",
    )
    week_parser.add_argument(
        "--week",
        required=True,
        metavar="YYYY-Www",
        help="the ISO week to run, e.g. 2026-W38",
    )
    week_parser.add_argument(
        "--run-days",
        action="store_true",
        help=(
            "first refresh the week's seven days with the day chain, so the whole "
            "raw → stt_fix → sum_daily → sum_weekly path runs in one command"
        ),
    )
    week_parser.add_argument(
        "--tz",
        default=None,
        metavar="TZ",
        help="IANA timezone of the week's days (default: APP_TIMEZONE)",
    )
    _add_models_config_option(week_parser)

    coach_parser = subparsers.add_parser(
        "coach",
        help="produce the coaching note (layer 5) of one ISO week — explicit launch only",
    )
    coach_parser.add_argument(
        "--week",
        required=True,
        metavar="YYYY-Www",
        help="the ISO week whose newest stored weekly summary is the note's input",
    )
    _add_models_config_option(coach_parser)
    return parser


def _add_models_config_option(parser: argparse.ArgumentParser) -> None:
    """Add ``--models-config`` to a phase-2 command (default: MODELS_CONFIG/env)."""
    parser.add_argument(
        "--models-config",
        default=None,
        metavar="PATH",
        help=(
            "model-configuration YAML to use (default: the MODELS_CONFIG "
            "environment variable, else configs/models.yaml)"
        ),
    )


def _models_config(args: argparse.Namespace) -> ModelsConfig:
    """The model configuration of a phase-2 command."""
    path = getattr(args, "models_config", None)
    return load_models_config(Path(path).expanduser() if path else None)


def _format_step(step: ChainStep) -> str:
    """One safe line per executed step: identifiers and provenance only."""
    return (
        f"step layer={step.layer.value} input={step.input_ref} run={step.run_id} "
        f"artifact={step.artifact_id} profile={step.profile_name} "
        f"model={step.model_name} prompt={step.prompt_version}"
    )


def _format_skip(skipped: SkippedStep) -> str:
    """One safe line per skipped step (reason + safe subject)."""
    return f"skip layer={skipped.layer.value} reason={skipped.reason} subject={skipped.subject}"


def _requested_layer_off(skipped: SkippedStep, *, layer: str) -> bool:
    """Whether the layer a command was asked for is switched off."""
    return skipped.layer.value == layer and skipped.reason == SKIP_PROFILE_OFF


def _report_requested_layer_off(*, layer: str) -> None:
    """The stderr message of "the profile this command needs is not configured"."""
    profile_field = _PROFILE_FIELD_BY_LAYER.get(layer, layer)
    print(
        f"error: the phase-2 layer {layer!r} is switched off — the model profile "
        f"{profile_field!r} is not configured (tasklist-phase2.md §9.8).",
        file=sys.stderr,
    )


def _print_steps(steps: tuple[ChainStep, ...], skipped: tuple[SkippedStep, ...]) -> int:
    """Print the steps/skips of a chain run; return the exit code contribution.

    Executed steps are always exit code 0; a step skipped because its layer is
    switched off is reported but not an error of this command (the command only
    fails when the layer it was asked for is off; the callers check that).
    """
    for step in steps:
        print(_format_step(step))
    for skipped_step in skipped:
        print(_format_skip(skipped_step))
    return _EXIT_OK


def _run_day(args: argparse.Namespace) -> int:
    _register_phase2_layers()
    try:
        local_date = date.fromisoformat(args.date)
    except ValueError as exc:
        print(f"error: --date must be an ISO date (YYYY-MM-DD): {exc}", file=sys.stderr)
        return _EXIT_ERROR
    config = _models_config(args)
    with _open_session() as session:
        outcome: DayChainOutcome = phase2_pipeline.run_day_chain(
            session,
            local_date=local_date,
            models_config=config,
            local_tz=args.tz,
        )
    print(f"day {outcome.local_date.isoformat()} tz={args.tz or get_settings().timezone}")
    exit_code = _print_steps(outcome.steps, outcome.skipped)
    if outcome.summary is None:
        for skipped_step in outcome.skipped:
            if _requested_layer_off(skipped_step, layer="daily_summary"):
                _report_requested_layer_off(layer="daily_summary")
                return _EXIT_ERROR
        if not outcome.steps:
            print("note: nothing to do for this day (no messages, or the layer is off)")
        return exit_code
    summary = outcome.summary
    print(
        f"summary layer=daily_summary artifact={outcome.summary_artifact_id} "
        f"facts={len(summary.facts)} themes={len(summary.themes)} "
        f"energy={'set' if summary.energy is not None else 'none'} "
        f"open_questions={len(summary.open_questions)}"
    )
    return exit_code


def _run_week(args: argparse.Namespace) -> int:
    _register_phase2_layers()
    week_id = args.week.strip()
    config = _models_config(args)
    with _open_session() as session:
        outcome: WeekChainOutcome = phase2_pipeline.run_week_chain(
            session,
            week_id=week_id,
            run_days=args.run_days,
            models_config=config,
            local_tz=args.tz,
        )
    print(f"week {outcome.week_id} run_days={bool(args.run_days)}")
    exit_code = _print_steps(outcome.steps, outcome.skipped)
    if outcome.summary is None:
        for skipped_step in outcome.skipped:
            if _requested_layer_off(skipped_step, layer="weekly_summary"):
                _report_requested_layer_off(layer="weekly_summary")
                return _EXIT_ERROR
        print("note: the week has no stored day summaries yet — run `day` first or pass --run-days")
        return exit_code
    print(
        f"summary layer=weekly_summary artifact={outcome.summary_artifact_id} "
        f"daily_ids={len(outcome.summary.daily_ids)}"
    )
    return exit_code


def _run_coach(args: argparse.Namespace) -> int:
    _register_phase2_layers()
    config = _models_config(args)
    with _open_session() as session:
        outcome: CoachStepOutcome = phase2_pipeline.run_coach_step(
            session,
            week_id=args.week.strip(),
            models_config=config,
        )
    print(f"coach {outcome.week_id}")
    if outcome.skipped is not None:
        print(_format_skip(outcome.skipped))
        if outcome.skipped.reason == SKIP_PROFILE_OFF:
            _report_requested_layer_off(layer="coach_note")
            return _EXIT_ERROR
        return _EXIT_OK
    step = outcome.step
    assert step is not None  # exactly one of step/skipped is set
    note = outcome.note
    print(_format_step(step))
    if note is not None:
        print(
            f"note layer=coach_note artifact={step.artifact_id} "
            f"observations={len(note.observations)} hypotheses={len(note.hypotheses)} "
            f"disclaimer=present experiment_days=7"
        )
    return _EXIT_OK


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
        if args.command == "day":
            return _run_day(args)
        if args.command == "week":
            return _run_week(args)
        if args.command == "coach":
            return _run_coach(args)
    except BackupConfirmRequiredError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_ERROR
    except BackupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_ERROR
    except ModelsConfigError as exc:
        # The configuration problem is the whole message; it never contains a
        # secret value (Task-03 validation resolves keys by name only).
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_ERROR
    except PipelineError as exc:
        # Domain failures already carry static, non-sensitive messages.
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_ERROR
    return _EXIT_ERROR  # pragma: no cover - every command above returns early


if __name__ == "__main__":
    raise SystemExit(main())
