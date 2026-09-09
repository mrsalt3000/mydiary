"""Integration tests for the normalization batch API and compare UI (Task 13).

Runs against an isolated, migrated SQLite test database and a temporary data
root with synthetic data only.

Covers:

* ``POST /api/v1/pipelines/normalize/batch`` — enqueues jobs only for eligible
  exports (original file present + no normalized artifact yet), returns the
  same jobs on a repeat call while they are in flight (no duplicates), and
  never enqueues for an export that already has a normalized result;
* ``GET /exports/{export_id}/normalize`` — the comparison page renders the
  original and the newest normalized version with full HTML escaping (or the
  "not normalized yet" state), and the exports list links to it.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import get_settings, reload_settings
from diary.db.models import ArtifactLayer, Job, MessageKind, WeeklyExport
from diary.db.repositories import (
    ArtifactRepository,
    MessageRepository,
)
from diary.domain.exports.service import export_week
from diary.domain.pipelines.naming import artifact_data_root_relative_path, derived_filename
from diary.main import app

PIPELINES_URL = "/api/v1/pipelines"
EXPORTS_URL = "/api/v1/exports"
SUNDAY = date(2026, 9, 13)
SCRIPT_TEXT = '<script>alert("xss")</script>'
_ENV_VARS = (
    "APP_TIMEZONE",
    "APP_MAX_MESSAGE_LENGTH",
    "APP_MAX_AUDIO_MB",
    "APP_ALLOWED_AUDIO_MIME_TYPES",
    "APP_DATA_ROOT",
    "APP_AUDIO_DIR",
    "APP_DB_DIR",
    "APP_TEMP_DIR",
    "APP_NORMALIZE_CHUNK_ENTRIES",
)


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point the data root at a per-test directory and isolate the environment."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    monkeypatch.setenv("APP_NORMALIZE_CHUNK_ENTRIES", "25")
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


def _message_time(sunday: date) -> datetime:
    monday = sunday - timedelta(days=6)
    return datetime(monday.year, monday.month, monday.day, 9, 0, tzinfo=timezone.utc)


def _create_export(session: Session, sunday: date, body: str) -> WeeklyExport:
    MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=_message_time(sunday)
    )
    session.commit()
    return export_week(session, sunday_date=sunday)


def _add_normalized_artifact(session: Session, export: WeeklyExport, text: str) -> None:
    """Simulate a succeeded normalized artifact (row + layer-2 file)."""
    filename = derived_filename(
        sunday=export.week_end_date,
        layer=ArtifactLayer.normalized,
        seq_number=export.seq_number,
        revision=export.revision,
        version=1,
    )
    relative = artifact_data_root_relative_path(layer=ArtifactLayer.normalized, filename=filename)
    target = get_settings().data_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    ArtifactRepository(session).create(
        layer=ArtifactLayer.normalized,
        export_id=export.id,
        path=relative,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        version=1,
    )
    session.commit()


def _job_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Job)) or 0


def _active_job_targets(session: Session) -> set[str]:
    """Export ids referenced by in-flight ``normalize`` jobs."""
    return {
        str(job.payload_json.get("export_id"))
        for job in session.scalars(select(Job)).all()
        if job.payload_json
    }


# ---------------------------------------------------------------------------
# batch enqueueing
# ---------------------------------------------------------------------------


def test_batch_enqueues_only_eligible_exports_and_does_not_duplicate(
    api_client: TestClient,
    session: Session,
) -> None:
    export_a = _create_export(session, date(2026, 8, 30), "Запись A.")
    export_b = _create_export(session, date(2026, 9, 6), "Запись B.")
    normalized = _create_export(session, SUNDAY, "Запись C (уже нормализована).")
    _add_normalized_artifact(session, normalized, "Нормализованная C.")

    first = api_client.post(f"{PIPELINES_URL}/normalize/batch")
    assert first.status_code == 202
    first_jobs = first.json()
    assert len(first_jobs) == 2
    assert {job["job_type"] for job in first_jobs} == {"normalize"}
    assert {job["status"] for job in first_jobs} == {"pending"}
    assert _job_count(session) == 2
    # Only the two eligible exports got jobs (the normalized one is excluded).
    assert _active_job_targets(session) == {str(export_a.id), str(export_b.id)}

    # A repeat call while the jobs are in flight reuses them (no duplicates).
    second = api_client.post(f"{PIPELINES_URL}/normalize/batch")
    assert second.status_code == 202
    second_jobs = second.json()
    assert {job["id"] for job in second_jobs} == {job["id"] for job in first_jobs}
    assert _job_count(session) == 2


def test_batch_returns_empty_when_every_export_is_normalized(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись.")
    _add_normalized_artifact(session, export, "Нормализованная запись.")
    response = api_client.post(f"{PIPELINES_URL}/normalize/batch")
    assert response.status_code == 202
    assert response.json() == []


def test_batch_skips_export_with_missing_original_file(
    api_client: TestClient,
    session: Session,
) -> None:
    present = _create_export(session, date(2026, 8, 30), "Запись с файлом.")
    missing = _create_export(session, SUNDAY, "Запись без файла.")
    (get_settings().data_root / missing.path).unlink()

    response = api_client.post(f"{PIPELINES_URL}/normalize/batch")
    assert response.status_code == 202
    jobs = response.json()
    assert len(jobs) == 1
    assert _active_job_targets(session) == {str(present.id)}


# ---------------------------------------------------------------------------
# compare page
# ---------------------------------------------------------------------------


def test_compare_page_renders_original_and_normalized_escaped(
    api_client: TestClient,
    session: Session,
) -> None:
    body = f"Оригинал с разметкой: {SCRIPT_TEXT}"
    export = _create_export(session, SUNDAY, body)
    normalized_text = f"Нормализованная версия: {SCRIPT_TEXT}"
    _add_normalized_artifact(session, export, normalized_text)

    response = api_client.get(f"/exports/{export.id}/normalize")
    assert response.status_code == 200
    html = response.text
    assert "Сравнение: оригинал и нормализация" in html
    # Diary text is escaped, never injected as markup.
    assert "&lt;script&gt;alert(&#34;xss&#34;)&lt;/script&gt;" in html
    assert SCRIPT_TEXT not in html
    assert "Нормализованная версия:" in html
    # The original download link of the Task-10 API is present.
    assert f"{EXPORTS_URL}/{export.id}/file" in html
    assert "Скачать оригинал" in html
    assert "2026-09-13_normalized_0001_r1_v1.txt" in html


def test_compare_page_shows_empty_state_without_normalized_version(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Только оригинал.")
    response = api_client.get(f"/exports/{export.id}/normalize")
    assert response.status_code == 200
    html = response.text
    assert "Только оригинал." in html
    assert "ещё нет нормализованной версии" in html


def test_compare_page_unknown_export_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"/exports/{uuid.uuid4()}/normalize")
    assert response.status_code == 404


def test_exports_list_links_to_the_compare_page(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для ссылки.")
    html = api_client.get("/exports").text
    assert f'href="/exports/{export.id}/normalize"' in html
    assert "Сравнить с нормализованным" in html
