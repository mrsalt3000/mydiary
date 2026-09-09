"""Observability tests of the MVP (Task 19).

Covers the request-id plumbing, the ``/ready`` endpoint and the privacy of
application logs:

* ``GET /ready`` answers ``200 {"status":"ready"}`` against a reachable SQLite
  database and ``503`` when the database is unreachable;
* every HTTP request produces a fresh request id that is attached to the log
  records of the request (middleware start/finish lines share one id);
* worker completion lines carry safe operational metrics (duration, model,
  provider, token counts, artifact size) — never diary content;
* application logs never contain diary text, transcriptions or secrets
  (caplog scanners over the request path and the worker paths).

Only synthetic data is used; STT/LLM steps run against deterministic fakes.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterator
from datetime import date, datetime, timezone

import pytest
from _e2e_helpers import (
    FakeLLMProvider,
    SyntheticSTTProvider,
    client_bound_to,
    enqueue_job,
    install_fake_llm,
    install_fake_stt,
    run_due_jobs,
    synthetic_body,
    upload_synthetic_wav,
)
from _pipeline_fakes import json_llm_response, register_fake_normalize_pipeline
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import get_settings, reload_settings
from diary.db.base import utc_now
from diary.db.models import JobType, MessageKind
from diary.db.repositories import MessageRepository
from diary.db.session import create_engine, create_session_factory
from diary.domain.exports.service import export_week
from diary.domain.exports.weeks import sunday_date_for
from diary.domain.llm.models import LLMUsage
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.normalize.registration import (
    ensure_normalize_pipeline_registered,
)
from diary.main import app
from diary.workers import handlers

#: Fixed synthetic week for the export-related log-scanner scenarios.
SUNDAY = date(2026, 9, 13)
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
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")


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
    """Restore the pipeline and handler registries after every test."""
    pipelines_snapshot = dict(pipeline_registry.REGISTRY)
    handlers_snapshot = dict(handlers.REGISTRY)
    yield
    pipeline_registry.REGISTRY.clear()
    pipeline_registry.REGISTRY.update(pipelines_snapshot)
    handlers.REGISTRY.clear()
    handlers.REGISTRY.update(handlers_snapshot)


def _records_with_path(caplog, path: str):
    return [
        record
        for record in caplog.records
        if record.name == "diary.api" and path in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# /ready endpoint
# ---------------------------------------------------------------------------


def test_ready_returns_ready_when_database_is_reachable(
    api_client: TestClient,
) -> None:
    response = api_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_database_is_unreachable(tmp_path) -> None:
    """The lightweight ``SELECT 1`` fails on an unopenable database file."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    bad_url = f"sqlite:///{(tmp_path / 'not-a-directory' / 'diary.db').as_posix()}"
    bad_engine = create_engine(bad_url)
    bad_factory = create_session_factory(bad_engine)

    def _override_session() -> Iterator[Session]:
        with bad_factory() as current_session:
            yield current_session

    app.dependency_overrides[get_session] = _override_session
    try:
        response = TestClient(app).get("/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "unavailable"}
    finally:
        app.dependency_overrides.pop(get_session, None)
        bad_engine.dispose()


# ---------------------------------------------------------------------------
# request-id plumbing
# ---------------------------------------------------------------------------


def test_each_request_gets_one_request_id_in_request_logs(
    api_client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    assert api_client.get("/health").status_code == 200
    records = _records_with_path(caplog, "path=/health")
    # The middleware logs a start line and a finish line per request.
    assert len(records) >= 2
    request_ids = {getattr(record, "request_id", None) for record in records}
    assert len(request_ids) == 1
    request_id = request_ids.pop()
    assert request_id is not None and _REQUEST_ID_RE.match(request_id)
    messages = [record.getMessage() for record in records]
    assert any("method=GET" in message and "path=/health" in message for message in messages)
    assert any("status=200" in message for message in messages)

    # A second request receives a different request id.
    caplog.clear()
    assert api_client.get("/health").status_code == 200
    second_records = _records_with_path(caplog, "path=/health")
    second_ids = {getattr(record, "request_id", None) for record in second_records}
    assert second_ids and second_ids != {request_id}


def test_request_id_does_not_leak_across_requests(api_client: TestClient) -> None:
    """The contextvar is reset after each request: the next request must not
    reuse an id and non-request log records render ``-``."""
    from diary.core.request_context import NO_REQUEST_ID, current_request_id

    assert current_request_id() == ""
    api_client.get("/health")
    assert current_request_id() == ""
    # A record emitted outside any request carries the explicit "no id" marker.
    import logging as _logging

    logger = _logging.getLogger("diary.api")
    captured: list[logging.LogRecord] = []

    class _Capture(_logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    try:
        logger.info("outside request")
    finally:
        logger.removeHandler(handler)
    assert captured and getattr(captured[0], "request_id", None) == NO_REQUEST_ID


# ---------------------------------------------------------------------------
# worker completion metrics (safe, structured)
# ---------------------------------------------------------------------------


def test_worker_completion_log_carries_safe_metrics(
    session: Session,
    session_factory,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A succeeded pipeline job logs duration plus model/provider/tokens/bytes —
    derived from provenance, never from diary content."""
    MessageRepository(session).create(
        kind=MessageKind.text,
        body_text="Синтетика: метрики воркера.",
        created_at=datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc),
    )
    session.commit()
    export = export_week(session, sunday_date=SUNDAY, now=utc_now())

    provider = FakeLLMProvider(
        response=json_llm_response(
            {"lines": ["Синтетическая нормализованная строка."]},
            usage=LLMUsage(input_tokens=12, output_tokens=7),
        )
    )
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    job_id = enqueue_job(
        session_factory,
        job_type=JobType.normalize,
        payload_json={"export_id": str(export.id)},
    )

    caplog.set_level(logging.INFO)
    run_due_jobs(session_factory, expected=1, worker_id="metrics-worker")

    matches = [
        record.getMessage()
        for record in caplog.records
        if record.name == "diary.workers.runner"
        and f"job={job_id}" in record.getMessage()
        and "status=succeeded" in record.getMessage()
    ]
    assert len(matches) == 1
    line = matches[0]
    assert re.search(r"duration_ms=\d+", line)
    assert "model=fake-norm-model" in line
    assert "provider=openai_compatible" in line
    assert "prompt_version=v1" in line
    assert "tokens_input=12" in line and "tokens_output=7" in line
    assert "artifact_bytes=" in line
    # The message content never appears in the completion line.
    assert "Синтетика: метрики воркера." not in line


def test_transcription_completion_log_carries_provider_and_size(
    api_client: TestClient,
    session_factory,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file_id, _message_id = upload_synthetic_wav(api_client)
    raw = "Синтетический текст расшифровки для метрик."
    provider = SyntheticSTTProvider(outputs=[raw])
    install_fake_stt(monkeypatch, provider)
    response = api_client.post(f"/api/v1/audio/{audio_file_id}/transcribe")
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["id"])

    caplog.set_level(logging.INFO)
    run_due_jobs(session_factory, expected=1, worker_id="metrics-worker")

    matches = [
        record.getMessage()
        for record in caplog.records
        if record.name == "diary.workers.runner"
        and f"job={job_id}" in record.getMessage()
        and "status=succeeded" in record.getMessage()
    ]
    assert len(matches) == 1
    line = matches[0]
    assert re.search(r"duration_ms=\d+", line)
    assert "model=tiny" in line
    assert "provider=faster_whisper" in line
    assert "language=ru" in line
    assert "artifact_bytes=" in line
    assert raw not in line  # raw transcription text is never logged


# ---------------------------------------------------------------------------
# log-content scanners (no diary text / transcriptions / secrets in logs)
# ---------------------------------------------------------------------------


def test_request_and_worker_logs_never_contain_diary_content_or_secrets(
    api_client: TestClient,
    session: Session,
    session_factory,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.INFO)

    # Unique markers would appear in logs if any code path leaked content.
    text_marker = synthetic_body("запись-маркер")
    response = api_client.post("/api/v1/messages", json={"body": text_marker})
    assert response.status_code == 201
    text_message_id = uuid.UUID(response.json()["id"])

    audio_file_id, audio_message_id = upload_synthetic_wav(api_client)
    raw_marker = "СЕКРЕТНЫЙ-РАСШИФРОВАННЫЙ-ТЕКСТ-для-сканера"
    stt_provider = SyntheticSTTProvider(outputs=[raw_marker])
    install_fake_stt(monkeypatch, stt_provider)
    assert api_client.post(f"/api/v1/audio/{audio_file_id}/transcribe").status_code == 202
    run_due_jobs(session_factory, expected=1)

    # Export the current week (contains both markers in the artifact text).
    with session_factory() as check:
        settings = get_settings()
        text_message = MessageRepository(check).get(text_message_id)
        audio_message = MessageRepository(check).get(audio_message_id)
        assert text_message is not None and audio_message is not None
        sunday_text = text_message.created_at.astimezone(settings.tzinfo).date()
        sunday_audio = audio_message.created_at.astimezone(settings.tzinfo).date()

    sundays = {sunday_date_for(sunday_text), sunday_date_for(sunday_audio)}
    assert len(sundays) == 1
    export_response = api_client.post(f"/api/v1/exports/weeks/{sundays.pop().isoformat()}")
    assert export_response.status_code == 201

    install_fake_llm(monkeypatch, FakeLLMProvider())
    assert ensure_normalize_pipeline_registered() is True
    export_id = export_response.json()["id"]
    assert (
        api_client.post(
            "/api/v1/pipelines/normalize/runs", json={"export_id": export_id}
        ).status_code
        == 202
    )
    run_due_jobs(session_factory, expected=1)

    # Readability of chat and detail pages is out of scope here — the point is
    # that nothing the requests/jobs touched leaked into the logs.
    joined_logs = "\n".join(record.getMessage() for record in caplog.records)
    # The scanner only proves absence if it actually saw the worker/API logs.
    assert "worker job=" in joined_logs
    assert "http_request" in joined_logs
    for marker in (text_marker, raw_marker):
        assert marker not in joined_logs
    # No secret value or provider key name appears (config references only the
    # variable name in-memory; nothing here is ever logged).
    assert "FAKE_PIPELINE_API_KEY" not in joined_logs
