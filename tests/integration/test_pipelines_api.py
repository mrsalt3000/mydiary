"""Integration tests for the pipeline HTTP API (Task 12).

Runs against an isolated, migrated SQLite test database and a temporary data
root with synthetic data only. Covers: POST runs (202 + job status surface),
422 for unknown pipeline types / malformed bodies, 404 for unknown exports,
409 when the export's original file is gone, active-job deduplication, and the
export-scoped runs/artifacts listing (metadata only, never diary content,
never file bytes).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone

import pytest
from _pipeline_fakes import (  # noqa: E501
    FakeLLMProvider,
    install_fake_llm,
    json_llm_response,
    register_fake_normalize_pipeline,
)
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import get_settings, reload_settings
from diary.db.models import Job, JobStatus, MessageKind
from diary.db.repositories import JobRepository, MessageRepository
from diary.domain.exports.service import export_week
from diary.domain.pipelines import registry as pipeline_registry
from diary.main import app
from diary.workers.runner import process_due_jobs

SUNDAY = "2026-09-13"
PIPELINES_URL = "/api/v1/pipelines"
EXPORTS_URL = "/api/v1/exports"
DIARY_PHRASE = "Синтетика: приватный текст для API."
_ENV_VARS = (
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
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    reload_settings()


@pytest.fixture(autouse=True)
def _restore_pipeline_registry() -> None:
    snapshot = dict(pipeline_registry.REGISTRY)
    yield
    pipeline_registry.REGISTRY.clear()
    pipeline_registry.REGISTRY.update(snapshot)


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


def _add_week_message(session: Session, body: str) -> None:
    MessageRepository(session).create(
        kind=MessageKind.text,
        body_text=body,
        created_at=datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc),  # Mon 09:00 MSK
    )
    session.commit()


def _create_export(session: Session):
    _add_week_message(session, DIARY_PHRASE)
    return export_week(session, sunday_date=date(2026, 9, 13))


def _job_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Job)) or 0


# ---------------------------------------------------------------------------
# enqueue endpoint
# ---------------------------------------------------------------------------


def test_post_run_returns_202_with_job_status_surface(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session)
    response = api_client.post(
        f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export.id)}
    )
    assert response.status_code == 202
    body = response.json()
    assert set(body) == {
        "id",
        "job_type",
        "status",
        "attempts",
        "max_attempts",
        "error_summary",
        "created_at",
        "updated_at",
    }
    assert body["job_type"] == "normalize"
    assert body["status"] == "pending"
    assert body["error_summary"] is None
    assert DIARY_PHRASE not in response.text


def test_post_run_reuses_active_job_without_duplicate(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session)
    first = api_client.post(f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export.id)})
    second = api_client.post(f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export.id)})
    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    assert _job_count(session) == 1


def test_post_run_unknown_export_returns_404(api_client: TestClient) -> None:
    response = api_client.post(
        f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(uuid.uuid4())}
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Export not found."}


def test_post_run_unknown_pipeline_type_returns_422(api_client: TestClient) -> None:
    response = api_client.post(
        f"{PIPELINES_URL}/transcribe/runs", json={"export_id": str(uuid.uuid4())}
    )
    assert response.status_code == 422


def test_post_run_missing_body_returns_422(api_client: TestClient) -> None:
    response = api_client.post(f"{PIPELINES_URL}/normalize/runs", json={})
    assert response.status_code == 422


def test_post_run_missing_original_file_returns_409(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session)
    (get_settings().data_root / export.path).unlink()
    response = api_client.post(
        f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export.id)}
    )
    assert response.status_code == 409
    assert "missing" in response.json()["detail"]
    assert _job_count(session) == 0  # nothing was enqueued


# ---------------------------------------------------------------------------
# runs/artifacts listing
# ---------------------------------------------------------------------------


def test_runs_and_artifacts_listing_after_successful_run(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(
        response=json_llm_response(
            {"lines": ["# Обработанная выгрузка", "Синтетическая строка API."]}
        )
    )
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    export = _create_export(session)

    created = api_client.post(f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export.id)})
    assert created.status_code == 202
    job_id = uuid.UUID(created.json()["id"])
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as s:
        job = JobRepository(s).get(job_id)
        assert job is not None and job.status is JobStatus.succeeded

    runs = api_client.get(f"{EXPORTS_URL}/{export.id}/runs")
    assert runs.status_code == 200
    items = runs.json()
    assert len(items) == 1
    item = items[0]
    assert set(item) == {
        "id",
        "pipeline_type",
        "status",
        "profile_name",
        "model_name",
        "prompt_version",
        "error_summary",
        "created_at",
        "started_at",
        "finished_at",
    }
    assert item["pipeline_type"] == "normalize"
    assert item["status"] == "succeeded"
    assert item["profile_name"] == "normalization"
    assert item["model_name"] == "fake-norm-model"
    assert item["prompt_version"] == "v1"
    assert item["error_summary"] is None
    assert DIARY_PHRASE not in runs.text  # no diary content

    artifacts = api_client.get(f"{EXPORTS_URL}/{export.id}/artifacts")
    assert artifacts.status_code == 200
    artifact_items = artifacts.json()
    assert len(artifact_items) == 1
    artifact = artifact_items[0]
    assert set(artifact) == {
        "id",
        "layer",
        "version",
        "filename",
        "sha256",
        "processing_run_id",
        "created_at",
    }
    assert artifact["layer"] == "normalized"
    assert artifact["version"] == 1
    assert artifact["filename"] == "2026-09-13_normalized_0001_r1_v1.txt"
    assert artifact["processing_run_id"] == items[0]["id"]
    assert len(artifact["sha256"]) == 64
    assert DIARY_PHRASE not in artifacts.text


def test_runs_listing_is_empty_before_any_run(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session)
    response = api_client.get(f"{EXPORTS_URL}/{export.id}/runs")
    assert response.status_code == 200
    assert response.json() == []
    artifacts = api_client.get(f"{EXPORTS_URL}/{export.id}/artifacts")
    assert artifacts.status_code == 200
    assert artifacts.json() == []


def test_runs_listing_unknown_export_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"{EXPORTS_URL}/{uuid.uuid4()}/runs")
    assert response.status_code == 404
    assert response.json() == {"detail": "Export not found."}


def test_artifacts_listing_unknown_export_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"{EXPORTS_URL}/{uuid.uuid4()}/artifacts")
    assert response.status_code == 404
    assert response.json() == {"detail": "Export not found."}


def test_malformed_export_id_returns_422(api_client: TestClient) -> None:
    response = api_client.get(f"{EXPORTS_URL}/not-a-uuid/runs")
    assert response.status_code == 422
