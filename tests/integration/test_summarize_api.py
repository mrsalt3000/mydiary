"""Integration tests for the summarization API, download route and UI (Task 14).

Runs against an isolated, migrated SQLite test database and a temporary data
root with synthetic data only.

Covers:

* ``POST /api/v1/pipelines/summarize/runs`` — the generic single-run endpoint
  enqueues a ``summarize`` job and the worker produces a layer-3 artifact
  when a normalized input exists (202 → succeeded);
* ``POST /api/v1/pipelines/summarize/batch`` — enqueues jobs only for eligible
  exports and does not duplicate active jobs;
* ``GET /api/v1/exports/{export_id}/artifacts/{artifact_id}/file`` — generic
  derived-artifact download (Markdown media type, correct bytes, 404s);
* ``GET /exports/{export_id}/summary`` — the read-only summary page: escaped
  rendering, empty states, 404 for unknown exports;
* the exports list page links to the summary page.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import pytest
from _normalize_fakes import FakeNormalizeLLM
from _pipeline_fakes import install_fake_llm
from _summarize_fakes import FakeSummarizeLLM
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import get_settings, reload_settings
from diary.db.models import ArtifactLayer, Job, MessageKind, PipelineType
from diary.db.repositories import ArtifactRepository, MessageRepository
from diary.domain.exports.service import export_week
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines import service as pipeline_service
from diary.domain.pipelines.naming import artifact_data_root_relative_path, derived_filename
from diary.domain.pipelines.normalize.registration import (
    ensure_normalize_pipeline_registered,
)
from diary.domain.pipelines.summarize.registration import (
    ensure_summarize_pipeline_registered,
)
from diary.main import app
from diary.workers.runner import process_due_jobs

PIPELINES_URL = "/api/v1/pipelines"
EXPORTS_URL = "/api/v1/exports"
SUNDAY = date(2026, 9, 13)
SCRIPT_TEXT = '<script>alert("xss")</script>'
SUMMARY_TEXT = (
    "# Саммари недели\n"
    "Источник (нормализованный): 2026-09-13_normalized_0001_r1_v1.txt\n"
    "---\n"
    "## Ключевые события\n"
    f"- Событие: {SCRIPT_TEXT}\n"
)
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
    "APP_SUMMARIZE_MAPREDUCE_THRESHOLD",
    "APP_SUMMARIZE_CHUNK_SIZE",
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


@pytest.fixture(autouse=True)
def _restore_pipeline_registry() -> None:
    """Restore the baseline pipeline registry after every test."""
    snapshot = dict(pipeline_registry.REGISTRY)
    yield
    pipeline_registry.REGISTRY.clear()
    pipeline_registry.REGISTRY.update(snapshot)


def _message_time(sunday: date) -> datetime:
    monday = sunday - timedelta(days=6)
    return datetime(monday.year, monday.month, monday.day, 9, 0, tzinfo=timezone.utc)


def _create_export(session: Session, sunday: date, body: str):
    MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=_message_time(sunday)
    )
    session.commit()
    return export_week(session, sunday_date=sunday)


def _add_artifact(session: Session, export, layer: ArtifactLayer, text: str, version: int = 1):
    """Simulate a succeeded artifact (row + file) of ``layer``."""
    filename = derived_filename(
        sunday=export.week_end_date,
        layer=layer,
        seq_number=export.seq_number,
        revision=export.revision,
        version=version,
    )
    relative = artifact_data_root_relative_path(layer=layer, filename=filename)
    target = get_settings().data_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    ArtifactRepository(session).create(
        layer=layer,
        export_id=export.id,
        path=relative,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        version=version,
    )
    session.commit()


def _normalized_text() -> str:
    return (
        "# Дневник — нормализованный\n"
        "Дата воскресенья: 2026-09-13\n"
        "Номер выгрузки: 0001\n"
        "Ревизия: r1\n"
        "---\n"
        "[2026-09-07 09:00] TEXT\n"
        "Синтетическая запись для саммари.\n"
    )


def _job_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Job)) or 0


# ---------------------------------------------------------------------------
# single-run (generic endpoint) end-to-end
# ---------------------------------------------------------------------------


def test_single_run_endpoint_produces_a_summary(
    api_client: TestClient,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Export + normalized input through real database rows (no normalize job).
    with session_factory() as s:
        export = _create_export(s, SUNDAY, "Запись для саммари через API.")
        export_id = export.id
        _add_artifact(s, export, ArtifactLayer.normalized, _normalized_text())

    provider = FakeSummarizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True

    created = api_client.post(f"{PIPELINES_URL}/summarize/runs", json={"export_id": str(export_id)})
    assert created.status_code == 202
    body = created.json()
    assert body["job_type"] == "summarize"
    assert body["status"] == "pending"
    assert "Запись для саммари" not in created.text

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as s:
        summary = ArtifactRepository(s).list_by_export_and_layer(export_id, ArtifactLayer.summary)
        assert len(summary) == 1
        assert summary[0].version == 1
    assert provider.calls == 1


def test_full_layer_flow_serves_the_summary_page(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual-smoke path: export → normalize → summarize → summary page 200.

    Uses the real normalization and summarization pipelines with the fake LLM
    providers; afterwards the layer-3 Markdown exists under ``03_summaries``
    and the read-only page renders it (HTTP 200).
    """
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    export = _create_export(session, SUNDAY, "Смоук: запись для нормализации и саммари.")
    export_id = export.id

    assert process_due_jobs(session_factory, worker_id="test-worker") == 0
    with session_factory() as s:
        normalize_job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.normalize, export_id=export_id
        )
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    provider = FakeSummarizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True
    with session_factory() as s:
        summarize_job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.summarize, export_id=export_id
        )
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    assert normalize_job.id is not None and summarize_job.id is not None
    summaries_dir = get_settings().exports_dir / "03_summaries"
    assert [p.name for p in summaries_dir.rglob("*.md")] == ["2026-09-13_summary_0001_r1_v1.md"]
    normalized_dir = get_settings().exports_dir / "02_normalized"
    assert [p.name for p in normalized_dir.rglob("*.txt")] == [
        "2026-09-13_normalized_0001_r1_v1.txt"
    ]

    response = api_client.get(f"/exports/{export_id}/summary")
    assert response.status_code == 200
    assert "Саммари недели" in response.text
    assert "Синтетическое событие всей недели." in response.text
    assert "Смоук: запись для нормализации" not in response.text  # only derived text shown


def test_batch_endpoint_enqueues_only_eligible_and_does_not_duplicate(
    api_client: TestClient,
    session: Session,
) -> None:
    eligible = _create_export(session, date(2026, 8, 30), "Запись A.")
    summarized = _create_export(session, date(2026, 9, 6), "Запись B.")
    _add_artifact(session, eligible, ArtifactLayer.normalized, _normalized_text())
    _add_artifact(session, summarized, ArtifactLayer.normalized, _normalized_text())
    _add_artifact(session, summarized, ArtifactLayer.summary, SUMMARY_TEXT)

    first = api_client.post(f"{PIPELINES_URL}/summarize/batch")
    assert first.status_code == 202
    first_jobs = first.json()
    assert len(first_jobs) == 1
    assert first_jobs[0]["job_type"] == "summarize"
    assert _job_count(session) == 1

    second = api_client.post(f"{PIPELINES_URL}/summarize/batch")
    assert second.status_code == 202
    assert {job["id"] for job in second.json()} == {job["id"] for job in first_jobs}
    assert _job_count(session) == 1
    assert eligible.id is not None and summarized.id is not None


# ---------------------------------------------------------------------------
# generic derived-artifact download
# ---------------------------------------------------------------------------


def test_artifact_download_returns_markdown_file(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для скачивания.")
    _add_artifact(session, export, ArtifactLayer.summary, SUMMARY_TEXT)
    artifact = ArtifactRepository(session).list_by_export_and_layer(
        export.id, ArtifactLayer.summary
    )[0]

    response = api_client.get(f"{EXPORTS_URL}/{export.id}/artifacts/{artifact.id}/file")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text == SUMMARY_TEXT
    assert "Событие:" in response.text


def test_artifact_download_unknown_artifact_returns_404(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись.")
    response = api_client.get(f"{EXPORTS_URL}/{export.id}/artifacts/{uuid.uuid4()}/file")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# summary page
# ---------------------------------------------------------------------------


def test_summary_page_renders_markdown_escaped_and_lists_download(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для страницы.")
    _add_artifact(session, export, ArtifactLayer.normalized, _normalized_text())
    _add_artifact(session, export, ArtifactLayer.summary, SUMMARY_TEXT)

    response = api_client.get(f"/exports/{export.id}/summary")
    assert response.status_code == 200
    html = response.text
    assert "Саммари недели" in html
    # Diary-derived markdown is escaped, never injected as markup.
    assert "&lt;script&gt;alert(&#34;xss&#34;)&lt;/script&gt;" in html
    assert SCRIPT_TEXT not in html
    assert "2026-09-13_summary_0001_r1_v1.md" in html
    # The download link uses the generic derived-artifact route.
    artifact = ArtifactRepository(session).list_by_export_and_layer(
        export.id, ArtifactLayer.summary
    )[0]
    assert f"{EXPORTS_URL}/{export.id}/artifacts/{artifact.id}/file" in html
    assert "Скачать саммари" in html


def test_summary_page_empty_state_without_normalized_version(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Только оригинал.")
    response = api_client.get(f"/exports/{export.id}/summary")
    assert response.status_code == 200
    html = response.text
    assert "сначала нужна нормализация" in html


def test_summary_page_empty_state_with_normalized_but_no_summary(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Нормализовано, но нет саммари.")
    _add_artifact(session, export, ArtifactLayer.normalized, _normalized_text())
    response = api_client.get(f"/exports/{export.id}/summary")
    assert response.status_code == 200
    html = response.text
    assert "ещё нет саммари" in html


def test_summary_page_unknown_export_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"/exports/{uuid.uuid4()}/summary")
    assert response.status_code == 404


def test_exports_list_links_to_the_summary_page(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для ссылки.")
    html = api_client.get("/exports").text
    assert f'href="/exports/{export.id}/summary"' in html
    assert "Саммари недели" in html
