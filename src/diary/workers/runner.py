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

Privacy (AGENTS.md §7–8): logs carry only job ids/types/statuses — never job
payloads or diary content; ``jobs.error_summary`` stores only a short safe
summary (exception class plus a truncated message, no traceback).

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
from datetime import timedelta

from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from diary.core.settings import AppSettings, get_settings
from diary.db.base import utc_now
from diary.db.models import Job
from diary.db.repositories import MAX_ERROR_SUMMARY_LENGTH, JobRepository
from diary.db.session import create_engine, create_session_factory
from diary.workers.handlers import PermanentJobError, get_handler

logger = logging.getLogger("diary.workers.runner")

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


def _mark_succeeded(session_factory: sessionmaker[Session], job: Job) -> None:
    with session_factory() as session:
        updated = JobRepository(session).mark_succeeded(job.id)
        session.commit()
    if updated is not None:
        logger.info(
            "worker job=%s type=%s status=succeeded attempts=%s",
            job.id,
            job.job_type.value,
            updated.attempts,
        )


def _mark_failed(
    session_factory: sessionmaker[Session],
    job: Job,
    *,
    error_summary: str,
    permanent: bool,
) -> None:
    with session_factory() as session:
        updated = JobRepository(session).mark_failed(
            job.id, error_summary=error_summary, permanent=permanent
        )
        session.commit()
    if updated is not None:
        logger.warning(
            "worker job=%s type=%s status=%s attempts=%s permanent=%s",
            job.id,
            job.job_type.value,
            updated.status.value,
            updated.attempts,
            permanent,
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
        )
        return
    logger.info(
        "worker=%s job=%s type=%s status=running attempts=%s",
        worker_id,
        job.id,
        job.job_type.value,
        job.attempts,
    )
    try:
        with session_factory() as session:
            handler(session, job)
            session.commit()
    except PermanentJobError as exc:
        _mark_failed(session_factory, job, error_summary=_safe_error_summary(exc), permanent=True)
    except Exception as exc:  # transient: retry policy applies
        _mark_failed(session_factory, job, error_summary=_safe_error_summary(exc), permanent=False)
    else:
        _mark_succeeded(session_factory, job)


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Task 13: register the implemented pipelines (normalization) before any
    # job runs, so real ``normalize`` jobs no longer fail permanently with
    # "pipeline not registered yet". The function is guarded and idempotent;
    # importing it here (not at module import time) keeps the registry free of
    # import side effects for the web process and the test suite.
    from diary.domain.pipelines.normalize.registration import (
        ensure_normalize_pipeline_registered,
    )

    ensure_normalize_pipeline_registered()
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
