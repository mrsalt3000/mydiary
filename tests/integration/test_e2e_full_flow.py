"""End-to-end full-flow test of the MVP (Task 19).

Drives the whole four-layer workflow exactly as a user would in the browser —
through the real HTTP API, the real durable queue and the real worker helpers
— against an isolated migrated SQLite database and a temporary data root:

1. create a text message through ``POST /api/v1/messages``;
2. upload an audio voice message through ``POST /api/v1/audio``;
3. queue and complete a mocked transcription (fake STT provider);
4. export the week (original, layer 1);
5. normalize the export (layer 2);
6. summarize the normalized artifact (layer 3);
7. run the wellbeing reflection over the normalized artifact (layer 4);
8. verify the full artifact lineage through the database, the API and the
   pages (files exist, hashes match, runs/artifacts/relationships correct,
   the original export is byte-for-byte unchanged after every downstream run).

Only synthetic data is used; every LLM step runs against deterministic fake
providers (no network, model or API key — AGENTS.md §13).
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator

import pytest
from _e2e_helpers import (
    FakeNormalizeLLM,
    FakeSummarizeLLM,
    FakeWellbeingLLM,
    SyntheticSTTProvider,
    client_bound_to,
    install_fake_llm,
    install_fake_stt,
    run_due_jobs,
    synthetic_body,
    upload_synthetic_wav,
)
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from diary.core.settings import get_settings, reload_settings
from diary.db.models import ArtifactLayer, Message, PipelineType, RunStatus
from diary.db.repositories import (
    ArtifactRepository,
    MessageRepository,
    ProcessingRunRepository,
    TranscriptionRepository,
    WeeklyExportRepository,
)
from diary.domain.exports.service import export_filename
from diary.domain.exports.weeks import sunday_date_for
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.normalize.registration import (
    ensure_normalize_pipeline_registered,
)
from diary.domain.pipelines.summarize.registration import ensure_summarize_pipeline_registered
from diary.domain.pipelines.wellbeing.registration import (
    ensure_wellbeing_pipeline_registered,
)
from diary.workers import handlers

EXPORTS_URL = "/api/v1/exports"
PIPELINES_URL = "/api/v1/pipelines"
_ENV_VARS = (
    "APP_TIMEZONE",
    "APP_MAX_MESSAGE_LENGTH",
    "APP_MAX_AUDIO_MB",
    "APP_ALLOWED_AUDIO_MIME_TYPES",
    "APP_JOB_MAX_ATTEMPTS",
    "APP_JOB_BACKOFF_BASE_S",
    "APP_JOB_BACKOFF_MULTIPLIER",
    "APP_JOB_MAX_BACKOFF_S",
    "APP_JOB_STALE_AFTER_S",
    "APP_DATA_ROOT",
    "APP_WELLBEING_ENABLED",
)


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point the data root at a per-test directory and isolate the environment."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    monkeypatch.setenv("APP_WELLBEING_ENABLED", "true")
    reload_settings()


@pytest.fixture
def api_client(session_factory) -> Iterator[TestClient]:
    """TestClient bound to the per-test migrated database."""
    with client_bound_to(session_factory) as client:
        yield client


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    """Restore the pipeline registry and the handler registry after every test."""
    pipelines_snapshot = dict(pipeline_registry.REGISTRY)
    handlers_snapshot = dict(handlers.REGISTRY)
    yield
    pipeline_registry.REGISTRY.clear()
    pipeline_registry.REGISTRY.update(pipelines_snapshot)
    handlers.REGISTRY.clear()
    handlers.REGISTRY.update(handlers_snapshot)


def _message_sunday(session: Session, message: Message):
    """Sunday date (application timezone) of the week containing a message."""
    settings = get_settings()
    local_date = message.created_at.astimezone(settings.tzinfo).date()
    return sunday_date_for(local_date)


def _create_text_message(api_client: TestClient) -> tuple[uuid.UUID, str]:
    """Step 1: create a text message through the real API; return (id, body)."""
    body = synthetic_body("text")
    response = api_client.post("/api/v1/messages", json={"body": body})
    assert response.status_code == 201, response.text
    return uuid.UUID(response.json()["id"]), body


def _upload_audio_message(api_client: TestClient) -> tuple[uuid.UUID, uuid.UUID]:
    """Step 2: upload one generated WAV through the real API."""
    return upload_synthetic_wav(api_client)


def _run_mocked_transcription(
    api_client: TestClient,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
    audio_file_id: uuid.UUID,
) -> str:
    """Step 3: queue and complete a mocked transcription; return the raw text."""
    provider = SyntheticSTTProvider(
        outputs=["Синтетическая расшифровка голосового сообщения (e2e)."]
    )
    install_fake_stt(monkeypatch, provider)
    response = api_client.post(f"/api/v1/audio/{audio_file_id}/transcribe")
    assert response.status_code == 202, response.text
    run_due_jobs(session_factory, expected=1)
    assert provider.calls == 1
    return provider.outputs[0]


def _assert_lineage_consistent(
    session: Session,
    *,
    export_id: uuid.UUID,
    layer: ArtifactLayer,
    expected_run_pipeline: PipelineType,
) -> dict[str, object]:
    """Shared per-layer lineage check on the database + filesystem."""
    settings = get_settings()
    artifacts = ArtifactRepository(session).list_by_export_and_layer(export_id, layer)
    assert len(artifacts) == 1, f"expected exactly one {layer.value} artifact"
    artifact = artifacts[0]
    assert artifact.processing_run_id is not None
    run = ProcessingRunRepository(session).get(artifact.processing_run_id)
    assert run is not None
    assert run.pipeline_type is expected_run_pipeline
    assert run.status is RunStatus.succeeded
    # The artifact file exists on disk and matches the recorded hash.
    path = settings.data_root / artifact.path
    assert path.is_file(), f"artifact file missing: {path}"
    on_disk = path.read_bytes()
    assert hashlib.sha256(on_disk).hexdigest() == artifact.sha256
    return {"artifact": artifact, "run": run, "path": path, "content": on_disk}


def test_full_e2e_four_layer_flow_and_lineage(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Steps 1–8 of the Task-19 end-to-end scenario in one coherent run."""
    settings = get_settings()

    # -- Steps 1–2: text + audio messages through the real API -------------
    text_id, text_body = _create_text_message(api_client)
    audio_file_id, audio_message_id = _upload_audio_message(api_client)
    assert text_id != audio_message_id

    with session_factory() as check:
        text_message = MessageRepository(check).get(text_id)
        audio_message = MessageRepository(check).get(audio_message_id)
        assert text_message is not None and audio_message is not None
        # Both messages live in the same week (they were just created).
        assert _message_sunday(check, text_message) == _message_sunday(check, audio_message)
        week_sunday = _message_sunday(check, text_message)
    sunday = week_sunday.isoformat()

    # The chat page already shows both messages (readability check).
    chat_html = api_client.get("/").text
    assert text_body in chat_html
    assert f'id="message-{text_id}"' in chat_html
    assert f'id="message-{audio_message_id}"' in chat_html
    assert "Не расшифровано" in chat_html

    # -- Step 3: mocked transcription ---------------------------------------
    raw_text = _run_mocked_transcription(api_client, session_factory, monkeypatch, audio_file_id)

    with session_factory() as check:
        rows = TranscriptionRepository(check).list_by_audio_file(audio_file_id)
        succeeded = [row for row in rows if row.status.value == "succeeded"]
        assert len(succeeded) == 1 and succeeded[0].raw_text == raw_text

    # -- Step 4: export the week (original, layer 1) ------------------------
    export_response = api_client.post(f"{EXPORTS_URL}/weeks/{sunday}")
    assert export_response.status_code == 201, export_response.text
    export_payload = export_response.json()
    export_id = uuid.UUID(export_payload["id"])
    assert export_payload["sunday"] == sunday
    assert export_payload["seq_number"] == 1
    assert export_payload["revision"] == 1
    original_filename = export_payload["filename"]
    assert original_filename == export_filename(week_sunday, 1, 1)

    original_path = settings.exports_dir / "01_original" / original_filename
    original_bytes_before = original_path.read_bytes()
    original_text = original_bytes_before.decode("utf-8")
    assert original_text.startswith("# Дневник — оригинал")
    assert text_body in original_text  # exact original text of the message
    assert "TEXT" in original_text and "AUDIO" in original_text
    assert "STT raw:" in original_text and raw_text in original_text

    with session_factory() as check:
        record = WeeklyExportRepository(check).get(export_id)
        assert record is not None
        assert hashlib.sha256(original_bytes_before).hexdigest() == record.sha256

    # -- Step 5: normalization (layer 2) ------------------------------------
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export_id)}
    )
    assert response.status_code == 202, response.text
    run_due_jobs(session_factory, expected=1)

    normalized = _assert_lineage_consistent(
        session,
        export_id=export_id,
        layer=ArtifactLayer.normalized,
        expected_run_pipeline=PipelineType.normalize,
    )
    normalized_run = normalized["run"]
    assert normalized_run.model_name == "fake-norm-model"
    assert normalized_run.prompt_version == "v1"
    assert normalized_run.input_artifact_id is None
    # The layer-2 header names the source original revision (lineage in-file).
    assert "0001_r1" in (normalized["content"]).decode("utf-8")

    # -- Step 6: summarize the normalized artifact (layer 3) ----------------
    install_fake_llm(monkeypatch, FakeSummarizeLLM())
    assert ensure_summarize_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/summarize/runs", json={"export_id": str(export_id)}
    )
    assert response.status_code == 202, response.text
    run_due_jobs(session_factory, expected=1)

    summary = _assert_lineage_consistent(
        session,
        export_id=export_id,
        layer=ArtifactLayer.summary,
        expected_run_pipeline=PipelineType.summarize,
    )
    summary_run = summary["run"]
    assert summary_run.input_artifact_id == normalized["artifact"].id  # layer 2 input
    assert (summary["content"]).decode("utf-8").lstrip().startswith("#")

    # -- Step 7: wellbeing reflection over the normalized artifact ----------
    install_fake_llm(monkeypatch, FakeWellbeingLLM())
    assert ensure_wellbeing_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/wellbeing/runs",
        json={"export_id": str(export_id), "confirm": True},
    )
    assert response.status_code == 202, response.text
    run_due_jobs(session_factory, expected=1)

    wellbeing = _assert_lineage_consistent(
        session,
        export_id=export_id,
        layer=ArtifactLayer.wellbeing,
        expected_run_pipeline=PipelineType.wellbeing,
    )
    wellbeing_run = wellbeing["run"]
    assert wellbeing_run.input_artifact_id == normalized["artifact"].id
    wellbeing_text = wellbeing["content"].decode("utf-8")
    # Required non-diagnostic disclaimer is present in the rendered report.
    assert "Это не медицинский и не психологический диагноз" in wellbeing_text

    # -- Step 8: full lineage through DB / API / pages ----------------------
    with session_factory() as check:
        artifacts = ArtifactRepository(check).list_by_export(export_id)
        layers = [artifact.layer for artifact in artifacts]
        assert layers == [
            ArtifactLayer.normalized,
            ArtifactLayer.summary,
            ArtifactLayer.wellbeing,
        ]
        runs = ProcessingRunRepository(check).list_by_input_export(export_id)
        assert sorted(run.pipeline_type for run in runs) == [
            PipelineType.normalize,
            PipelineType.summarize,
            PipelineType.wellbeing,
        ]

    # Original export is byte-for-byte unchanged after every downstream run.
    assert original_path.read_bytes() == original_bytes_before

    # API lineage endpoints (Task 12): runs and artifacts of the export.
    runs_response = api_client.get(f"{EXPORTS_URL}/{export_id}/runs")
    assert runs_response.status_code == 200
    assert {item["pipeline_type"] for item in runs_response.json()} == {
        "normalize",
        "summarize",
        "wellbeing",
    }
    artifacts_response = api_client.get(f"{EXPORTS_URL}/{export_id}/artifacts")
    assert artifacts_response.status_code == 200
    artifact_items = artifacts_response.json()
    assert {item["layer"] for item in artifact_items} == {
        "normalized",
        "summary",
        "wellbeing",
    }
    for artifact in (normalized["artifact"], summary["artifact"], wellbeing["artifact"]):
        download = api_client.get(f"{EXPORTS_URL}/{export_id}/artifacts/{artifact.id}/file")
        assert download.status_code == 200
        assert hashlib.sha256(download.content).hexdigest() == artifact.sha256
    original_download = api_client.get(f"{EXPORTS_URL}/{export_id}/file")
    assert original_download.status_code == 200
    assert original_download.content == original_bytes_before

    # UI readability: exports list, the per-export detail page and the
    # processing page all reflect the produced lineage.
    exports_html = api_client.get("/exports").text
    assert original_filename in exports_html
    assert f'href="/exports/{export_id}"' in exports_html
    detail_html = api_client.get(f"/exports/{export_id}").text
    for label in (
        "Оригинал (слой 1)",
        "Нормализованный (слой 2)",
        "Саммари (слой 3)",
        "Wellbeing-рефлексия (слой 4)",
    ):
        assert label in detail_html
    for artifact in (normalized["artifact"], summary["artifact"], wellbeing["artifact"]):
        assert artifact.path.rsplit("/", 1)[-1] in detail_html
    assert "производный от оригинала 0001_r1" in detail_html
    processing_html = api_client.get("/processing").text
    for label in ("Расшифровка аудио", "Нормализация", "Саммаризация", "Wellbeing-рефлексия"):
        assert label in processing_html
    assert "Успешно" in processing_html
    # Transcription raw text is readable on the chat page, escaped.
    chat_after = api_client.get("/").text
    assert "Расшифровано" in chat_after
    assert raw_text in chat_after
