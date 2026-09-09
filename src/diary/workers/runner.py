"""Worker runner for the durable SQLite-backed queue (Task 08, ADR-003).

A separate process that shares the SQLite database with the web application:

1. (optional, per round) resets ``running`` jobs abandoned by a crashed worker
   (``--reset-stale``) — ``updated_at`` older than ``job_stale_after_s``;
2. atomically claims one due job (``pending``, or ``retrying`` whose
   ``next_retry_at`` has passed);
3. executes the registered handler in ``try/except``:
   - success → ``succeeded``;
   - :class:`~diary.workers.handlers.PermanentJobError` (or a job type with no
     registered handler) → ``failed`` immediately;
   - any other exception → ``failed`` after ``max_attempts``, otherwise
     ``retrying`` with exponential backoff (``next_retry_at``).

Privacy (AGENTS.md §7–8): logs carry only job ids/types/statuses and, on
completion, safe operational metrics (duration, model/provider, token counts,
artifact/audio size) — never job payloads or diary content;
``jobs.error_summary`` stores only a short safe summary (exception class plus
a truncated message, no traceback).

Usage:

    python -m diary.workers.runner --help
    python -m diary.workers.runner --once                 # drain due jobs and exit
    python -m diary.workers.runner --reset-stale          # recover crashed jobs
    python -m diary.workers.runner --poll-interval 10     # continuous polling
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
import uuid
from datetime import timedelta

from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from diary.core.logging_config import configure_logging
from diary.core.request_context import attach_request_id_filter
from diary.core.settings import AppSettings, get_settings
from diary.db.base import utc_now
from diary.db.models import Job, JobType, RunStatus
from diary.db.repositories import (
    MAX_ERROR_SUMMARY_LENGTH,
    ArtifactRepository,
    AudioFileRepository,
    JobRepository,
    ProcessingRunRepository,
    TranscriptionRepository,
)
from diary.db.session import create_engine, create_session_factory
from diary.workers.handlers import PermanentJobError, get_handler

logger = logging.getLogger("diary.workers.runner")
attach_request_id_filter(logger)

#: ``AppSettings`` field consumed for the staleness threshold.
_STALE_FIELD = "job_stale_after_s"


def _safe_error_summary(exc: BaseException) -> str:
    """A short, non-sensitive summary of a handler exception.

    Class name plus a whitespace-collapsed message truncated to
    ``MAX_ERROR_SUMMARY_LENGTH`` characters. Never includes a traceback, a job
    payload or diary content (handlers follow the same contract).
    """
    message = " ".join(str(exc).split())
    if not message:
        return type(exc).__name__
    return f"{type(exc).__name__}: {message[:MAX_ERROR_SUMMARY_LENGTH]}"


# ---------------------------------------------------------------------------
# Safe completion metrics (Task 19)
# ---------------------------------------------------------------------------
# The completion log lines of a job carry safe operational metrics when they
# can be derived from the database: the model/provider that actually ran,
# token counts recorded in run metadata and the size of the produced artifact
# file. Only provenance columns/metadata and file sizes are read — never diary
# content, prompts, transcriptions or LLM text (AGENTS.md §8).

#: Value rendered when a metric is not available.
_METRIC_NONE = "-"
#: Maximum length of a single metric value in the log line.
_METRIC_VALUE_LIMIT = 128


def _metric_text(value: object) -> str:
    """Collapse/trim one metric value into a single-line string (or ``-``)."""
    if value is None:
        return _METRIC_NONE
    text = " ".join(str(value).split())
    if not text:
        return _METRIC_NONE
    return text[:_METRIC_VALUE_LIMIT]


def _claim_one(session_factory: sessionmaker[Session]) -> Job | None:
    """Claim one due job in its own transaction and return it detached.

    The claim commits immediately so the write lock is released before the
    (possibly long) handler runs. Job attributes stay readable afterwards
    (``expire_on_commit=False``).
    """
    with session_factory() as session:
        job = JobRepository(session).claim_due()
        if job is None:
            session.rollback()
            return None
        session.commit()
        return job


def _completion_metric_fields(session_factory: sessionmaker[Session], job: Job) -> dict[str, str]:
    """Content-free metric fields of a completed job for its log line.

    Best-effort by design: when nothing can be derived (a retrying attempt, a
    failure before any run row, an unexpected lookup error) the function
    returns an empty mapping and the caller logs the core fields alone. It
    never raises and never reads diary content.
    """
    try:
        if job.job_type in (JobType.normalize, JobType.summarize, JobType.wellbeing):
            return _pipeline_run_metrics(session_factory, job)
        if job.job_type is JobType.transcribe_audio:
            return _transcription_metrics(session_factory, job)
    except Exception:
        # Logging must never fail because metric collection did.
        logger.debug("worker failed to collect completion metrics for job=%s", job.id)
    return {}


def _pipeline_run_metrics(session_factory: sessionmaker[Session], job: Job) -> dict[str, str]:
    """Metrics of the processing run/artifact recorded for a pipeline job.

    Provenance columns (``model_name``/``prompt_version``) and the run's
    ``metadata_json`` (provider type, token usage) are safe; the artifact file
    contributes only its size in bytes.
    """
    with session_factory() as session:
        runs = ProcessingRunRepository(session).list_by_job_id(job.id)
        if not runs:
            return {}
        run = runs[-1]
        if run.status is not RunStatus.succeeded:
            return {}
        metadata = run.metadata_json if isinstance(run.metadata_json, dict) else {}
        fields: dict[str, str] = {
            "model": _metric_text(run.model_name or metadata.get("model")),
            "provider": _metric_text(metadata.get("provider_type")),
            "prompt_version": _metric_text(metadata.get("prompt_version") or run.prompt_version),
        }
        usage = metadata.get("usage")
        if isinstance(usage, dict):
            fields["tokens_input"] = _metric_text(usage.get("input_tokens"))
            fields["tokens_output"] = _metric_text(usage.get("output_tokens"))
        artifacts = ArtifactRepository(session).list_by_run(run.id)
        if artifacts:
            artifact_path = get_settings().data_root / artifacts[-1].path
            try:
                fields["artifact_bytes"] = str(artifact_path.stat().st_size)
            except OSError:
                pass
        return {key: value for key, value in fields.items() if value != _METRIC_NONE}


def _transcription_metrics(session_factory: sessionmaker[Session], job: Job) -> dict[str, str]:
    """Metrics of the transcription row/audio file recorded for an STT job.

    The audio file id is read from the job payload (an identifier, never
    content); the newest transcription row of that file contributes the model/
    provider/language provenance, the audio row its stored size.
    """
    payload = job.payload_json if isinstance(job.payload_json, dict) else {}
    raw_id = payload.get("audio_file_id")
    try:
        audio_file_id = uuid.UUID(str(raw_id))
    except (TypeError, ValueError):
        return {}
    with session_factory() as session:
        audio = AudioFileRepository(session).get(audio_file_id)
        if audio is None:
            return {}
        rows = TranscriptionRepository(session).list_by_audio_file(audio_file_id)
        if not rows:
            return {}
        latest = rows[-1]
        fields = {
            "model": _metric_text(latest.model_name),
            "provider": _metric_text(latest.provider_name),
            "language": _metric_text(latest.language),
            "artifact_bytes": _metric_text(audio.size_bytes),
        }
        return {key: value for key, value in fields.items() if value != _METRIC_NONE}


def _metrics_suffix(fields: dict[str, str]) -> str:
    """Render metric fields as a trailing `` key=value`` chunk of the log line."""
    if not fields:
        return ""
    return " " + " ".join(f"{key}={value}" for key, value in sorted(fields.items()))


def _mark_succeeded(session_factory: sessionmaker[Session], job: Job, *, duration_ms: int) -> None:
    with session_factory() as session:
        updated = JobRepository(session).mark_succeeded(job.id)
        session.commit()
    if updated is not None:
        metrics = _metrics_suffix(_completion_metric_fields(session_factory, job))
        logger.info(
            "worker job=%s type=%s status=succeeded attempts=%s duration_ms=%d%s",
            job.id,
            job.job_type.value,
            updated.attempts,
            duration_ms,
            metrics,
        )


def _mark_failed(
    session_factory: sessionmaker[Session],
    job: Job,
    *,
    error_summary: str,
    permanent: bool,
    duration_ms: int,
) -> None:
    with session_factory() as session:
        updated = JobRepository(session).mark_failed(
            job.id, error_summary=error_summary, permanent=permanent
        )
        session.commit()
    if updated is not None:
        metrics = _metrics_suffix(_completion_metric_fields(session_factory, job))
        logger.warning(
            "worker job=%s type=%s status=%s attempts=%s permanent=%s duration_ms=%d%s",
            job.id,
            job.job_type.value,
            updated.status.value,
            updated.attempts,
            permanent,
            duration_ms,
            metrics,
        )


def _process_one_job(
    session_factory: sessionmaker[Session],
    *,
    worker_id: str,
    job: Job,
) -> None:
    """Execute one claimed job and persist the resulting transition."""
    handler = get_handler(job.job_type)
    if handler is None:
        _mark_failed(
            session_factory,
            job,
            error_summary=(
                f"No job handler is registered for job type "
                f"{job.job_type.value!r}; the job cannot be processed."
            ),
            permanent=True,
            duration_ms=0,
        )
        return
    logger.info(
        "worker=%s job=%s type=%s status=running attempts=%s",
        worker_id,
        job.id,
        job.job_type.value,
        job.attempts,
    )
    started = time.monotonic()
    try:
        with session_factory() as session:
            handler(session, job)
            session.commit()
    except PermanentJobError as exc:
        _mark_failed(
            session_factory,
            job,
            error_summary=_safe_error_summary(exc),
            permanent=True,
            duration_ms=_elapsed_ms(started),
        )
    except Exception as exc:  # transient: retry policy applies
        _mark_failed(
            session_factory,
            job,
            error_summary=_safe_error_summary(exc),
            permanent=False,
            duration_ms=_elapsed_ms(started),
        )
    else:
        _mark_succeeded(session_factory, job, duration_ms=_elapsed_ms(started))


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since the ``time.monotonic()`` instant ``started``."""
    return int((time.monotonic() - started) * 1000.0)


def process_due_jobs(
    session_factory: sessionmaker[Session],
    *,
    worker_id: str = "test",
    reset_stale: bool = False,
) -> int:
    """Process every currently due job; return how many were handled.

    When ``reset_stale`` is set, abandoned ``running`` jobs older than
    ``job_stale_after_s`` are first returned to ``pending``. Each job runs in
    its own claim/execute/mark transactions (see module docstring).
    """
    if reset_stale:
        _reset_stale_running(session_factory)
    processed = 0
    while True:
        job = _claim_one(session_factory)
        if job is None:
            break
        _process_one_job(session_factory, worker_id=worker_id, job=job)
        processed += 1
    return processed


def _reset_stale_running(session_factory: sessionmaker[Session]) -> int:
    """Return abandoned ``running`` jobs (older than the staleness threshold)."""
    settings = get_settings()
    older_than = utc_now() - timedelta(seconds=float(getattr(settings, _STALE_FIELD)))
    with session_factory() as session:
        count = JobRepository(session).reset_stale_running(older_than=older_than)
        session.commit()
    if count:
        logger.warning("worker reset %s stale running job(s) to pending", count)
    return count


def run_forever(
    session_factory: sessionmaker[Session],
    *,
    worker_id: str,
    poll_interval_s: float,
    reset_stale: bool,
) -> int:
    """Polling loop until interrupted; returns the process exit code."""
    logger.info(
        "worker=%s started poll_interval=%ss reset_stale=%s",
        worker_id,
        poll_interval_s,
        reset_stale,
    )
    while True:
        try:
            processed = process_due_jobs(
                session_factory, worker_id=worker_id, reset_stale=reset_stale
            )
            if processed == 0:
                time.sleep(poll_interval_s)
        except KeyboardInterrupt:
            logger.info("worker=%s stopped (KeyboardInterrupt).", worker_id)
            return 0
        except OperationalError:
            # SQLite is briefly busy (another writer): log the class only and
            # back off instead of crashing the worker.
            logger.warning("worker=%s database busy; retrying in %ss", worker_id, poll_interval_s)
            time.sleep(poll_interval_s)


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def build_argument_parser() -> argparse.ArgumentParser:
    """The ``python -m diary.workers.runner`` CLI parser (public for tests)."""
    parser = argparse.ArgumentParser(
        prog="python -m diary.workers.runner",
        description=(
            "Process due jobs of the durable SQLite-backed queue (ADR-003). "
            "Logs never contain job payloads or diary content."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="process all currently due jobs (and --reset-stale, if given), then exit",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="seconds to wait between polls when no job is due (default: 5)",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        metavar="ID",
        help="worker identity used in logs (default: hostname-pid)",
    )
    parser.add_argument(
        "--reset-stale",
        action="store_true",
        help=(
            "each poll round, return 'running' jobs abandoned for more than "
            "job_stale_after_s to 'pending' so they can be reclaimed"
        ),
    )
    return parser


def _session_factory(engine: Engine) -> sessionmaker[Session]:
    return create_session_factory(engine)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be greater than 0")
    configure_logging()
    # Task 13-15: register the implemented pipelines (normalization,
    # summarization, wellbeing) before any job runs, so real
    # ``normalize``/``summarize``/``wellbeing`` jobs no longer fail
    # permanently with "pipeline not registered yet". The functions are
    # guarded and idempotent; importing them here (not at module import time)
    # keeps the registry free of import side effects for the web process and
    # the test suite.
    from diary.domain.pipelines.normalize.registration import (
        ensure_normalize_pipeline_registered,
    )
    from diary.domain.pipelines.summarize.registration import (
        ensure_summarize_pipeline_registered,
    )
    from diary.domain.pipelines.wellbeing.registration import (
        ensure_wellbeing_pipeline_registered,
    )

    ensure_normalize_pipeline_registered()
    ensure_summarize_pipeline_registered()
    ensure_wellbeing_pipeline_registered()
    worker_id = args.worker_id or _default_worker_id()
    settings: AppSettings = get_settings()
    engine = create_engine(settings=settings)
    session_factory = _session_factory(engine)
    logger.info("worker=%s using database %s", worker_id, engine.url)
    try:
        if args.once:
            processed = process_due_jobs(
                session_factory, worker_id=worker_id, reset_stale=args.reset_stale
            )
            logger.info("worker=%s processed %s due job(s); exiting (--once)", worker_id, processed)
            return 0
        return run_forever(
            session_factory,
            worker_id=worker_id,
            poll_interval_s=args.poll_interval,
            reset_stale=args.reset_stale,
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
