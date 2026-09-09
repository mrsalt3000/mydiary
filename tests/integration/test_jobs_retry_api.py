"""Integration tests for the durable-job retry API (Task 16).

Uses an isolated, migrated SQLite test database (fixtures from
``conftest.py``) plus synthetic data only. Covers:

* ``POST /api/v1/jobs/{id}/retry`` — a ``failed`` or ``cancelled`` job is
  re-enqueued as a fresh ``pending`` job of the same type with the same
  identifier payload, a fresh attempt budget and no payload leakage in the
  response; the original terminal row is left untouched;
* a retry of a job that is still in progress or already succeeded is rejected
  with 409; unknown ids are 404; malformed ids are 422.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import reload_settings
from diary.db.models import Job, JobStatus, JobType
from diary.db.repositories import JobRepository
from diary.main import app

JOBS_URL = "/api/v1/jobs"


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with default application settings."""
    monkeypatch.delenv("APP_TIMEZONE", raising=False)
    monkeypatch.delenv("APP_JOB_MAX_ATTEMPTS", raising=False)
    reload_settings()


@contextmanager
def _client_bound_to(session_factory) -> Iterator[TestClient]:
    """A TestClient whose ``get_session`` dependency serves ``session_factory``."""

    def _override_session() -> Iterator[Session]:
        with session_factory() as current_session:
            yield current_session

    app.dependency_overrides[get_session] = _override_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_session, None)


@pytest.fixture
def api_client(session_factory) -> Iterator[TestClient]:
    """TestClient bound to the per-test migrated database."""
    with _client_bound_to(session_factory) as client:
        yield client


def _make_terminal(
    session: Session,
    *,
    job_type: JobType,
    status: JobStatus,
    marker: str,
) -> uuid.UUID:
    """Create a job and move it to a terminal ``status`` (failed/cancelled)."""
    repository = JobRepository(session)
    job = repository.enqueue(
        job_type=job_type,
        payload_json={"export_id": marker},
        max_attempts=3,
    )
    if status is JobStatus.failed:
        claimed = repository.claim_due()
        assert claimed is not None and claimed.id == job.id
        repository.mark_failed(job.id, error_summary="RuntimeError: synthetic boom", permanent=True)
    elif status is JobStatus.cancelled:
        assert repository.cancel(job.id) is not None
    else:
        raise AssertionError(f"Unsupported terminal status {status!r}")
    session.commit()
    return job.id


def _job_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Job)) or 0


# ---------------------------------------------------------------------------
# retry of failed / cancelled jobs
# ---------------------------------------------------------------------------


def test_retry_failed_job_creates_fresh_pending_job(
    api_client: TestClient,
    session: Session,
) -> None:
    job_id = _make_terminal(
        session, job_type=JobType.normalize, status=JobStatus.failed, marker="exp-1"
    )
    assert _job_count(session) == 1

    response = api_client.post(f"{JOBS_URL}/{job_id}/retry")
    assert response.status_code == 202
    payload = response.json()
    assert payload["job_type"] == "normalize"
    assert payload["status"] == "pending"
    assert payload["attempts"] == 0
    assert payload["max_attempts"] == 3  # the default configured budget
    assert payload["error_summary"] is None
    assert "exp-1" not in response.text  # payload content never leaks

    rows = list(session.scalars(select(Job)))
    assert len(rows) == 2
    original = session.get(Job, job_id)
    fresh = next(row for row in rows if row.id != job_id)
    assert original is not None
    assert original.status is JobStatus.failed  # history preserved
    assert original.payload_json == {"export_id": "exp-1"}
    assert fresh.status is JobStatus.pending
    assert fresh.payload_json == {"export_id": "exp-1"}
    assert fresh.attempts == 0
    assert fresh.idempotency_key is None  # retry is a fresh, independent job


def test_retry_cancelled_job_creates_fresh_pending_job(
    api_client: TestClient,
    session: Session,
) -> None:
    job_id = _make_terminal(
        session, job_type=JobType.summarize, status=JobStatus.cancelled, marker="exp-2"
    )
    response = api_client.post(f"{JOBS_URL}/{job_id}/retry")
    assert response.status_code == 202
    assert response.json()["job_type"] == "summarize"
    assert response.json()["status"] == "pending"
    assert _job_count(session) == 2
    original = session.get(Job, job_id)
    assert original is not None and original.status is JobStatus.cancelled


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [JobStatus.pending, JobStatus.running, JobStatus.succeeded],
    ids=["pending", "running", "succeeded"],
)
def test_retry_of_non_retryable_job_is_409(
    api_client: TestClient,
    session: Session,
    status: JobStatus,
) -> None:
    repository = JobRepository(session)
    job = repository.enqueue(
        job_type=JobType.normalize,
        payload_json={"export_id": "exp-x"},
        max_attempts=3,
    )
    if status is JobStatus.running:
        claimed = repository.claim_due()
        assert claimed is not None and claimed.id == job.id
    elif status is JobStatus.succeeded:
        claimed = repository.claim_due()
        assert claimed is not None and claimed.id == job.id
        repository.mark_succeeded(job.id)
    session.commit()

    response = api_client.post(f"{JOBS_URL}/{job.id}/retry")
    assert response.status_code == 409
    assert "retried" in response.json()["detail"]
    assert _job_count(session) == 1  # no extra job row


def test_retry_unknown_job_is_404(api_client: TestClient) -> None:
    response = api_client.post(f"{JOBS_URL}/{uuid.uuid4()}/retry")
    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found."}


def test_retry_malformed_id_is_422(api_client: TestClient) -> None:
    response = api_client.post(f"{JOBS_URL}/not-a-uuid/retry")
    assert response.status_code == 422
