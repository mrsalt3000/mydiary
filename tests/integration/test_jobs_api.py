"""Integration tests for the job status polling API (Task 08).

Uses an isolated, migrated SQLite test database (fixtures from
``conftest.py``) plus synthetic data only. Covers: job status surface with
timestamps rendered in the application timezone, no payload leakage in
responses, 404/422 handling, and the newest-first recent-jobs listing.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import reload_settings
from diary.db.models import JobType
from diary.db.repositories import JobRepository
from diary.main import app

JOBS_URL = "/api/v1/jobs"

JOB_RESPONSE_FIELDS = {
    "id",
    "job_type",
    "status",
    "attempts",
    "max_attempts",
    "error_summary",
    "created_at",
    "updated_at",
}


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


def _enqueue(session: Session, *, job_type: JobType, marker: str = "synthetic") -> uuid.UUID:
    job = JobRepository(session).enqueue(
        job_type=job_type,
        payload_json={"marker": marker},
        metadata_json={"note": marker},
        max_attempts=3,
    )
    return job.id


def _utc(hour: int) -> datetime:
    return datetime(2026, 9, 1, hour, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# single-job polling
# ---------------------------------------------------------------------------


def test_get_job_returns_status_surface_without_payload(
    api_client: TestClient, session: Session
) -> None:
    job_id = _enqueue(session, job_type=JobType.transcribe_audio, marker="payload-marker")
    session.commit()

    response = api_client.get(f"{JOBS_URL}/{job_id}")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == JOB_RESPONSE_FIELDS  # no payload/metadata keys
    assert payload["id"] == str(job_id)
    assert payload["job_type"] == "transcribe_audio"
    assert payload["status"] == "pending"
    assert payload["attempts"] == 0
    assert payload["max_attempts"] == 3
    assert payload["error_summary"] is None
    assert "payload-marker" not in response.text  # payload content never leaks
    # Default application timezone is Europe/Moscow (UTC+03:00).
    assert payload["created_at"].endswith("+03:00")
    datetime.fromisoformat(payload["created_at"])
    datetime.fromisoformat(payload["updated_at"])


def test_get_job_reports_retrying_state_and_safe_error(
    api_client: TestClient, session: Session
) -> None:
    repository = JobRepository(session)
    job_id = repository.enqueue(job_type=JobType.normalize, max_attempts=3).id
    session.commit()
    # Simulate a worker outcome: claim then transient failure -> retrying.
    claimed = repository.claim_due()
    assert claimed is not None and claimed.id == job_id
    repository.mark_failed(job_id, error_summary="RuntimeError: synthetic boom")
    session.commit()

    payload = api_client.get(f"{JOBS_URL}/{job_id}").json()
    assert payload["status"] == "retrying"
    assert payload["attempts"] == 1
    assert payload["error_summary"] == "RuntimeError: synthetic boom"


def test_get_unknown_job_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"{JOBS_URL}/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found."}


def test_get_malformed_job_id_returns_422(api_client: TestClient) -> None:
    response = api_client.get(f"{JOBS_URL}/not-a-uuid")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# recent-jobs listing
# ---------------------------------------------------------------------------


def test_list_jobs_is_newest_first_without_payload(
    api_client: TestClient, session: Session
) -> None:
    repository = JobRepository(session)
    ids = [_enqueue(session, job_type=JobType.normalize, marker=f"m{i}") for i in range(3)]
    # Force distinct creation times so the order is fully deterministic.
    rows = [repository.get(job_id) for job_id in ids]
    for row, hour in zip(rows, (8, 9, 10), strict=True):
        assert row is not None
        row.created_at = _utc(hour)
    session.commit()

    response = api_client.get(JOBS_URL)
    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 50
    items = payload["items"]
    assert len(items) == 3
    # Newest first: created at 10:00 UTC, then 09:00, then 08:00.
    assert [item["id"] for item in items] == [str(ids[2]), str(ids[1]), str(ids[0])]
    for item in items:
        assert set(item) == JOB_RESPONSE_FIELDS
        assert item["job_type"] == "normalize"
    # Timestamps rendered in the application timezone.
    assert items[0]["created_at"] == "2026-09-01T13:00:00+03:00"
    assert "m0" not in response.text  # payload/metadata content never leaks


def test_list_jobs_respects_limit(api_client: TestClient, session: Session) -> None:
    for index in range(4):
        _enqueue(session, job_type=JobType.normalize, marker=f"lim-{index}")
    session.commit()

    response = api_client.get(JOBS_URL, params={"limit": "2"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 2
    assert response.json()["limit"] == 2


def test_list_jobs_empty_returns_empty_page(api_client: TestClient) -> None:
    response = api_client.get(JOBS_URL)
    assert response.status_code == 200
    assert response.json() == {"items": [], "limit": 50}


def test_list_jobs_rejects_invalid_limit(api_client: TestClient) -> None:
    for bad_limit in ("0", "201"):
        response = api_client.get(JOBS_URL, params={"limit": bad_limit})
        assert response.status_code == 422
