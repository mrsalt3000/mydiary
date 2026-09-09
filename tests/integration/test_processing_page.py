"""Integration tests for the processing page and its polling fragment (Task 16).

Uses an isolated, migrated SQLite test database and a temporary data root
with synthetic data only.

Covers:

* ``GET /processing`` — the recent-jobs page: job type/status captions,
  attempts, safe error summaries, the export link for pipeline jobs, the
  retry action only for failed/cancelled rows, and no payload leakage;
* ``GET /processing/fragments/jobs`` — the htmx-polled fragment shows the
  same table and reflects changed job statuses on a later request;
* HTML escaping of stored ``error_summary`` values on both endpoints.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import reload_settings
from diary.db.models import JobType, MessageKind, WeeklyExport
from diary.db.repositories import JobRepository, MessageRepository
from diary.domain.exports.service import export_week
from diary.main import app

SUNDAY = date(2026, 9, 13)
SCRIPT_SUMMARY = '<script>alert("xss")</script>'
ENV_VARS = (
    "APP_TIMEZONE",
    "APP_MAX_MESSAGE_LENGTH",
    "APP_MAX_AUDIO_MB",
    "APP_ALLOWED_AUDIO_MIME_TYPES",
    "APP_DATA_ROOT",
    "APP_AUDIO_DIR",
    "APP_DB_DIR",
    "APP_TEMP_DIR",
)


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point the data root at a per-test directory and isolate the environment."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
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
    with _client_bound_to(session_factory) as client:
        yield client


def _create_export(session: Session) -> WeeklyExport:
    """Create one synthetic export in the default test week."""
    monday = SUNDAY - timedelta(days=6)
    MessageRepository(session).create(
        kind=MessageKind.text,
        body_text="Запись для страницы обработки.",
        created_at=datetime(monday.year, monday.month, monday.day, 9, 0, tzinfo=timezone.utc),
    )
    session.commit()
    return export_week(session, sunday_date=SUNDAY)


def _enqueue(
    session: Session,
    *,
    job_type: JobType,
    export_id: str | None = None,
    marker: str | None = None,
) -> str:
    """Enqueue a job whose payload is identifier-only plus a hidden ``marker``
    that must never appear in rendered pages."""
    payload: dict[str, str] = {}
    if export_id is not None:
        payload["export_id"] = export_id
    if marker is not None:
        payload["marker"] = marker
    job = JobRepository(session).enqueue(job_type=job_type, payload_json=payload, max_attempts=3)
    session.commit()
    return str(job.id)


def _fail(session: Session, job_id: str, error_summary: str) -> None:
    repository = JobRepository(session)
    claimed = repository.claim_due()
    assert claimed is not None and str(claimed.id) == job_id
    repository.mark_failed(claimed.id, error_summary=error_summary, permanent=True)
    session.commit()


# ---------------------------------------------------------------------------
# processing page
# ---------------------------------------------------------------------------


def test_processing_page_shows_empty_state(api_client: TestClient) -> None:
    html = api_client.get("/processing").text
    assert "Обработка" in html
    assert "Задач пока нет" in html
    assert 'id="jobs-panel"' in html
    # The panel polls the local fragment endpoint.
    assert 'hx-get="/processing/fragments/jobs"' in html
    assert 'hx-trigger="every 4s"' in html


def test_processing_page_lists_jobs_with_statuses_and_export_links(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session)
    normalize_job = _enqueue(session, job_type=JobType.normalize, export_id=str(export.id))
    transcribe_job = _enqueue(
        session, job_type=JobType.transcribe_audio, marker="secret-transcribe-marker"
    )

    html = api_client.get("/processing").text
    assert "Нормализация" in html
    assert "Расшифровка аудио" in html
    assert "В очереди" in html
    # The pipeline job links to the real export detail page (id from the DB).
    assert f'href="/exports/{export.id}"' in html
    assert "Детали выгрузки" in html
    # Job ids are never rendered as an identifier column, and payload content
    # (including the marker) never leaks into the page.
    assert normalize_job not in html
    assert transcribe_job not in html
    assert "secret-transcribe-marker" not in html
    assert "job-retry" not in html  # nothing is retryable while pending


def test_processing_page_offers_retry_only_for_failed_or_cancelled(
    api_client: TestClient,
    session: Session,
) -> None:
    repository = JobRepository(session)
    failed = repository.enqueue(
        job_type=JobType.normalize, payload_json={"export_id": "export-1"}, max_attempts=3
    )
    claimed = repository.claim_due()
    assert claimed is not None and claimed.id == failed.id
    repository.mark_failed(claimed.id, error_summary="RuntimeError: synthetic boom", permanent=True)
    cancelled = repository.enqueue(
        job_type=JobType.summarize, payload_json={"export_id": "export-2"}, max_attempts=3
    )
    assert repository.cancel(cancelled.id) is not None
    succeeded = repository.enqueue(
        job_type=JobType.summarize, payload_json={"export_id": "export-3"}, max_attempts=3
    )
    assert repository.mark_succeeded(succeeded.id) is not None
    session.commit()

    html = api_client.get("/processing").text
    assert html.count("job-retry") == 2  # failed + cancelled only
    assert "Ошибка" in html
    assert "Отменено" in html
    assert "Успешно" in html
    assert "RuntimeError: synthetic boom" in html  # safe stored summary is shown
    assert "export-1" not in html  # payload content never leaks


def test_processing_page_escapes_error_summary(
    api_client: TestClient,
    session: Session,
) -> None:
    job = _enqueue(session, job_type=JobType.normalize, marker="secret-payload-marker")
    _fail(session, job, SCRIPT_SUMMARY)

    html = api_client.get("/processing").text
    assert "&lt;script&gt;" in html
    assert "alert(&#34;xss&#34;)" in html
    assert SCRIPT_SUMMARY not in html
    assert "secret-payload-marker" not in html  # payload content never leaks


# ---------------------------------------------------------------------------
# polling fragment
# ---------------------------------------------------------------------------


def test_processing_fragment_renders_jobs_and_tracks_status_updates(
    api_client: TestClient,
    session: Session,
) -> None:
    job = _enqueue(session, job_type=JobType.normalize, marker="export-update")

    first = api_client.get("/processing/fragments/jobs")
    assert first.status_code == 200
    assert 'id="jobs-panel"' in first.text
    assert 'hx-get="/processing/fragments/jobs"' in first.text
    assert "В очереди" in first.text

    # A later poll reflects the new terminal state and offers the retry.
    _fail(session, job, "RuntimeError: update failed")
    second = api_client.get("/processing/fragments/jobs")
    assert second.status_code == 200
    assert "Ошибка" in second.text
    assert "job-retry" in second.text
    assert "RuntimeError: update failed" in second.text


def test_processing_fragment_escapes_stored_error_summary(
    api_client: TestClient,
    session: Session,
) -> None:
    job = _enqueue(session, job_type=JobType.summarize, marker="export-esc")
    _fail(session, job, SCRIPT_SUMMARY)

    html = api_client.get("/processing/fragments/jobs").text
    assert "&lt;script&gt;" in html
    assert SCRIPT_SUMMARY not in html


def test_processing_fragment_empty_state(api_client: TestClient) -> None:
    html = api_client.get("/processing/fragments/jobs").text
    assert "Задач пока нет" in html
