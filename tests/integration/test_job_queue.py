"""Integration tests for the durable job queue and worker (Task 08, ADR-003).

Runs against the isolated migrated test database from ``conftest.py`` and uses
only synthetic payloads. The worker under test is driven through its real
entry helpers (``diary.workers.runner.process_due_jobs``), which exercise the
claim → handler → transition path with real repository code; handlers are fake
module functions registered per test.

Covered: enqueue defaults, idempotency reuse, atomic claim semantics (two
claimers never receive the same job), transient retries with growing backoff,
permanent failure after ``max_attempts``, success, unknown job type, the
``transcribe_audio`` handler rejecting an invalid payload permanently,
cancellation, restart survival, and stale-``running`` recovery.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.core.settings import reload_settings
from diary.db.base import utc_now
from diary.db.models import Job, JobStatus, JobType
from diary.db.repositories import JobRepository
from diary.db.session import create_engine, create_session_factory
from diary.workers import handlers
from diary.workers.runner import process_due_jobs

_JOB_ENV_VARS = (
    "APP_TIMEZONE",
    "APP_JOB_MAX_ATTEMPTS",
    "APP_JOB_BACKOFF_BASE_S",
    "APP_JOB_BACKOFF_MULTIPLIER",
    "APP_JOB_MAX_BACKOFF_S",
    "APP_JOB_STALE_AFTER_S",
)


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use default queue-policy settings regardless of the developer env."""
    for name in _JOB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    reload_settings()


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    """Restore the baseline handler registry after every test."""
    snapshot = dict(handlers.REGISTRY)
    yield
    handlers.REGISTRY.clear()
    handlers.REGISTRY.update(snapshot)


def _job_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Job)) or 0


def _make_retry_due(session_factory, job_id) -> None:
    """Force a ``retrying`` job's ``next_retry_at`` into the past."""
    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None and job.status is JobStatus.retrying
        job.next_retry_at = utc_now() - timedelta(seconds=1)
        session.commit()


# ---------------------------------------------------------------------------
# enqueue and idempotency
# ---------------------------------------------------------------------------


def test_enqueue_creates_pending_job_with_max_attempts(session_factory) -> None:
    with session_factory() as session:
        job = JobRepository(session).enqueue(
            job_type=JobType.transcribe_audio,
            payload_json={"audio_file_id": "synthetic-uuid"},
            max_attempts=4,
        )
        session.commit()
        job_id = job.id
        assert job.status is JobStatus.pending
        assert job.attempts == 0

        stored = JobRepository(session).get(job_id)
        assert stored is not None
        assert stored.max_attempts == 4
        assert stored.payload_json == {"audio_file_id": "synthetic-uuid"}


def test_enqueue_same_idempotency_key_reuses_job(session: Session) -> None:
    repository = JobRepository(session)
    first = repository.enqueue(
        job_type=JobType.transcribe_audio,
        payload_json={"audio_file_id": "audio-1"},
        idempotency_key="transcribe-audio-1",
        max_attempts=3,
    )
    second = repository.enqueue(
        job_type=JobType.transcribe_audio,
        payload_json={"audio_file_id": "audio-1"},
        idempotency_key="transcribe-audio-1",
        max_attempts=3,
    )
    session.commit()
    assert second.id == first.id
    assert _job_count(session) == 1


def test_enqueue_reuse_survives_commit(session_factory) -> None:
    with session_factory() as session:
        first = JobRepository(session).enqueue(
            job_type=JobType.normalize, idempotency_key="same-key"
        )
        session.commit()
        first_id = first.id
    with session_factory() as session:
        second = JobRepository(session).enqueue(
            job_type=JobType.normalize, idempotency_key="same-key"
        )
        session.commit()
        assert second.id == first_id
        assert _job_count(session) == 1


def test_enqueue_reuses_succeeded_job(session_factory) -> None:
    with session_factory() as session:
        job = JobRepository(session).enqueue(
            job_type=JobType.normalize, idempotency_key="done-key", max_attempts=3
        )
        job_id = job.id
        session.commit()
    with session_factory() as session:
        claimed = JobRepository(session).claim_due()
        assert claimed is not None
        JobRepository(session).mark_succeeded(claimed.id)
        session.commit()
    with session_factory() as session:
        again = JobRepository(session).enqueue(
            job_type=JobType.normalize, idempotency_key="done-key"
        )
        session.commit()
        assert again.id == job_id
        assert _job_count(session) == 1


def test_enqueue_after_permanent_failure_creates_new_job(session_factory) -> None:
    with session_factory() as session:
        job = JobRepository(session).enqueue(
            job_type=JobType.normalize, idempotency_key="failed-key", max_attempts=3
        )
        job_id = job.id
        session.commit()
    with session_factory() as session:
        JobRepository(session).mark_failed(job_id, error_summary="ValueError: boom", permanent=True)
        session.commit()
    with session_factory() as session:
        retry = JobRepository(session).enqueue(
            job_type=JobType.normalize, idempotency_key="failed-key"
        )
        session.commit()
        assert retry.id != job_id
        assert _job_count(session) == 2


def test_enqueue_rejects_bad_arguments(session_factory) -> None:
    with session_factory() as session:
        repository = JobRepository(session)
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            repository.enqueue(job_type=JobType.normalize, max_attempts=0)
        with pytest.raises(ValueError, match="at most 255 characters"):
            repository.enqueue(job_type=JobType.normalize, idempotency_key="k" * 256)


# ---------------------------------------------------------------------------
# atomic claim
# ---------------------------------------------------------------------------


def test_two_claimers_cannot_claim_the_same_job(session_factory) -> None:
    with session_factory() as session:
        job = JobRepository(session).enqueue(job_type=JobType.normalize, max_attempts=3)
        job_id = job.id
        session.commit()
    with session_factory() as worker_a:
        claimed_a = JobRepository(worker_a).claim_due()
        worker_a.commit()
    assert claimed_a is not None and claimed_a.id == job_id
    assert claimed_a.status is JobStatus.running
    assert claimed_a.attempts == 1

    # A second, fully separate claimer must not receive the running job.
    with session_factory() as worker_b:
        claimed_b = JobRepository(worker_b).claim_due()
        worker_b.rollback()
    assert claimed_b is None

    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None
        assert stored.status is JobStatus.running
        assert stored.attempts == 1


def test_two_claimers_receive_distinct_jobs(session_factory) -> None:
    with session_factory() as session:
        first = JobRepository(session).enqueue(job_type=JobType.normalize)
        second = JobRepository(session).enqueue(job_type=JobType.normalize)
        session.commit()
    with session_factory() as worker_a:
        claimed_a = JobRepository(worker_a).claim_due()
        worker_a.commit()
    with session_factory() as worker_b:
        claimed_b = JobRepository(worker_b).claim_due()
        worker_b.commit()
    assert claimed_a is not None and claimed_b is not None
    assert {claimed_a.id, claimed_b.id} == {first.id, second.id}


# ---------------------------------------------------------------------------
# retries, success, permanent failures (through the real worker helpers)
# ---------------------------------------------------------------------------


def test_transient_failure_retries_with_backoff_then_fails(session_factory) -> None:
    calls: list[str] = []

    def flaky(session, job) -> None:
        calls.append(str(job.id))
        raise ValueError("synthetic transient failure")

    # 'normalize' now has a real baseline handler (Task 12); the fake overrides
    # the registry entry directly (``register`` forbids duplicates).
    handlers.REGISTRY[JobType.normalize.value] = flaky
    with session_factory() as session:
        job = JobRepository(session).enqueue(
            job_type=JobType.normalize,
            payload_json={"synthetic": "payload"},
            max_attempts=3,
        )
        job_id = job.id
        session.commit()

    # First run: one attempt -> retrying, next_retry_at in the future.
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    assert calls == [str(job_id)]
    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None
        assert stored.status is JobStatus.retrying
        assert stored.attempts == 1
        assert stored.next_retry_at is not None and stored.next_retry_at > utc_now()
        assert stored.error_summary == "ValueError: synthetic transient failure"
        assert "Traceback" not in stored.error_summary
        assert stored.payload_json == {"synthetic": "payload"}  # never mutated
        first_retry_at = stored.next_retry_at

    # Second run: attempts grow and the backoff window widens.
    _make_retry_due(session_factory, job_id)
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None
        assert stored.status is JobStatus.retrying
        assert stored.attempts == 2
        assert stored.next_retry_at is not None
        assert stored.next_retry_at > first_retry_at  # exponential growth

    # Third run: attempts are exhausted -> permanent failed state.
    _make_retry_due(session_factory, job_id)
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None
        assert stored.status is JobStatus.failed
        assert stored.attempts == 3
        assert stored.next_retry_at is None
        assert stored.error_summary == "ValueError: synthetic transient failure"
        assert stored.error_summary is not None and "Traceback" not in stored.error_summary

    # No due jobs remain.
    assert process_due_jobs(session_factory, worker_id="test-worker") == 0
    assert len(calls) == 3  # exactly three handler executions, no duplicates


def test_successful_job_becomes_succeeded(session_factory) -> None:
    def succeed(session, job) -> None:
        session.commit()  # a handler may commit its own side effects

    # Override the Task-12 baseline handler directly (register forbids dupes).
    handlers.REGISTRY[JobType.normalize.value] = succeed
    with session_factory() as session:
        job = JobRepository(session).enqueue(job_type=JobType.normalize, max_attempts=3)
        job_id = job.id
        session.commit()

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None
        assert stored.status is JobStatus.succeeded
        assert stored.attempts == 1
        assert stored.error_summary is None
        assert stored.next_retry_at is None


def test_job_type_without_handler_fails_permanently(session_factory) -> None:
    # Task 12 registered a handler for 'normalize'; removing it here simulates
    # a future job type that has no handler so the runner's no-handler path is
    # still exercised (permanent failure, no retries burned).
    handlers.REGISTRY.pop(JobType.normalize.value, None)
    with session_factory() as session:
        job = JobRepository(session).enqueue(
            job_type=JobType.normalize,
            max_attempts=5,  # plenty left, still fails
        )
        job_id = job.id
        session.commit()

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None
        assert stored.status is JobStatus.failed
        assert stored.attempts == 1
        assert stored.error_summary is not None
        assert "No job handler is registered for job type 'normalize'" in stored.error_summary


def test_transcribe_audio_with_invalid_payload_fails_permanently(session_factory) -> None:
    """The real (Task 09) handler rejects a transcribe job whose payload has no
    valid audio_file_id as a deterministic failure — no retries are burned."""
    with session_factory() as session:
        job = JobRepository(session).enqueue(
            job_type=JobType.transcribe_audio,
            payload_json={"audio_file_id": "synthetic-audio"},
            max_attempts=3,
        )
        job_id = job.id
        session.commit()

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None
        assert stored.status is JobStatus.failed
        assert stored.attempts == 1  # failed immediately, not after max_attempts
        assert stored.error_summary is not None
        assert "audio_file_id" in stored.error_summary
        assert "not implemented" not in stored.error_summary


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------


def test_cancel_pending_job(session_factory) -> None:
    with session_factory() as session:
        job = JobRepository(session).enqueue(job_type=JobType.normalize)
        job_id = job.id
        session.commit()
        repository = JobRepository(session)
        cancelled = repository.cancel(job_id)
        assert cancelled is not None and cancelled.status is JobStatus.cancelled
        # Cancelling again is a no-op.
        assert repository.cancel(job_id) is None
        session.commit()
        assert JobRepository(session).get(job_id).status is JobStatus.cancelled


def test_cancel_running_job_is_rejected(session_factory) -> None:
    with session_factory() as session:
        job = JobRepository(session).enqueue(job_type=JobType.normalize)
        job_id = job.id
        session.commit()
    with session_factory() as session:
        claimed = JobRepository(session).claim_due()
        assert claimed is not None and claimed.id == job_id
        assert JobRepository(session).cancel(job_id) is None
        session.commit()
        assert JobRepository(session).get(job_id).status is JobStatus.running


# ---------------------------------------------------------------------------
# restart survival and stale-running recovery
# ---------------------------------------------------------------------------


def test_pending_job_survives_engine_restart(run_upgrade, db_url: str) -> None:
    run_upgrade(db_url)
    first_engine = create_engine(db_url)
    try:
        first_factory = create_session_factory(first_engine)
        with first_factory() as session:
            job = JobRepository(session).enqueue(
                job_type=JobType.normalize, payload_json={"synthetic": True}, max_attempts=3
            )
            job_id = job.id
            session.commit()
    finally:
        first_engine.dispose()

    second_engine = create_engine(db_url)
    try:
        second_factory = create_session_factory(second_engine)
        with second_factory() as session:
            claimed = JobRepository(session).claim_due()
            assert claimed is not None and claimed.id == job_id
            JobRepository(session).mark_succeeded(claimed.id)
            session.commit()
        with second_factory() as session:
            stored = JobRepository(session).get(job_id)
            assert stored is not None
            assert stored.status is JobStatus.succeeded
            assert stored.payload_json == {"synthetic": True}
    finally:
        second_engine.dispose()


def test_reset_stale_running_requeues_abandoned_job(session_factory) -> None:
    with session_factory() as session:
        job = JobRepository(session).enqueue(job_type=JobType.normalize, max_attempts=3)
        job_id = job.id
        session.commit()
        # Pretend a worker claimed the job and then crashed.
        claimed = JobRepository(session).claim_due()
        assert claimed is not None and claimed.id == job_id
        session.commit()

    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None
        stored.updated_at = utc_now() - timedelta(hours=1)  # abandoned long ago
        session.commit()

    with session_factory() as session:
        repository = JobRepository(session)
        reset = repository.reset_stale_running(older_than=utc_now() - timedelta(seconds=60))
        session.commit()
        assert reset == 1
        stored = JobRepository(session).get(job_id)
        assert stored is not None
        assert stored.status is JobStatus.pending
        assert stored.attempts == 1  # crashed attempt is preserved

        # The reclaimed job is executable again.
        reclaimed = JobRepository(session).claim_due()
        assert reclaimed is not None and reclaimed.id == job_id
        assert reclaimed.attempts == 2
        session.commit()


def test_reset_stale_ignores_fresh_running_jobs(session_factory) -> None:
    with session_factory() as session:
        job = JobRepository(session).enqueue(job_type=JobType.normalize, max_attempts=3)
        job_id = job.id
        session.commit()
        claimed = JobRepository(session).claim_due()
        assert claimed is not None and claimed.id == job_id
        session.commit()
        repository = JobRepository(session)
        # A running job that was just claimed is not stale yet.
        reset = repository.reset_stale_running(older_than=utc_now() - timedelta(seconds=60))
        session.commit()
        assert reset == 0
        assert JobRepository(session).get(job_id).status is JobStatus.running
