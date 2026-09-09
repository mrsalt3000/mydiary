"""Failure-safety tests of the MVP (Task 19).

Each scenario simulates a realistic pipeline failure and then verifies that
all *primary* data stays byte-for-byte intact and the system state stays
consistent:

* failed STT (transient provider failure after retries): audio bytes, the
  audio message and the audio-file metadata are untouched; exactly one
  terminal ``failed`` transcription row exists;
* failed LLM pipeline (transient, retried): the original weekly export is
  unchanged and no partial derived artifact appears;
* malformed LLM JSON (deterministic/permanent): the job fails on the first
  attempt, the run records a safe summary and the original export is
  unchanged;
* duplicate worker claims (two workers, one job): the job is executed exactly
  once;
* re-export revision: after a failed pipeline run, an explicit re-export
  creates ``r2`` while ``r1`` stays byte-identical.

All data is synthetic; providers are deterministic fakes (AGENTS.md §13).
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from _e2e_helpers import (
    FakeLLMProvider,
    SyntheticSTTProvider,
    client_bound_to,
    install_fake_llm,
    install_fake_stt,
    make_retry_due,
    run_due_jobs,
    upload_synthetic_wav,
)
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from diary.core.settings import get_settings, reload_settings
from diary.db.base import utc_now
from diary.db.models import (
    ArtifactLayer,
    Job,
    JobStatus,
    MessageKind,
    RunStatus,
    TranscriptionStatus,
    WeeklyExport,
)
from diary.db.repositories import (
    ArtifactRepository,
    AudioFileRepository,
    JobRepository,
    MessageRepository,
    ProcessingRunRepository,
    TranscriptionRepository,
    WeeklyExportRepository,
)
from diary.domain.exports.service import export_week
from diary.domain.llm.models import LLMResponse
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.normalize.registration import (
    ensure_normalize_pipeline_registered,
)
from diary.workers import handlers
from diary.workers.runner import process_due_jobs

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

#: Fixed synthetic week used by the export-related failure scenarios.
SUNDAY = date(2026, 9, 13)


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


def _audio_bytes(session: Session, audio_file_id: uuid.UUID) -> bytes:
    audio = AudioFileRepository(session).get(audio_file_id)
    assert audio is not None
    return (get_settings().audio_dir / audio.storage_path).read_bytes()


def _original_files() -> list[Path]:
    original_dir = get_settings().exports_dir / "01_original"
    if not original_dir.exists():
        return []
    return [path for path in original_dir.rglob("*.txt") if path.is_file()]


def _make_export(session: Session, body: str) -> WeeklyExport:
    """Insert one text message in the fixed week and export it (r1)."""
    MessageRepository(session).create(
        kind=MessageKind.text,
        body_text=body,
        created_at=datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc),
    )
    session.commit()
    return export_week(session, sunday_date=SUNDAY, now=utc_now())


def _assert_job_terminal(session_factory, job_id: uuid.UUID, *, status: JobStatus) -> Job:
    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None and job.status is status
        return job


class MalformedJSONProvider:
    """An LLM provider returning text that is not valid structured JSON."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text="это не JSON, а произвольный текст", model="fake-model")


# ---------------------------------------------------------------------------
# failed STT: primary data intact after retries are exhausted
# ---------------------------------------------------------------------------


def test_failed_stt_keeps_audio_and_message_byte_intact(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file_id, message_id = upload_synthetic_wav(api_client)
    audio_before = _audio_bytes(session, audio_file_id)
    audio_row_before = AudioFileRepository(session).get(audio_file_id)
    assert audio_row_before is not None
    message_before = MessageRepository(session).get(message_id)
    assert message_before is not None and message_before.kind is MessageKind.audio

    provider = SyntheticSTTProvider(failures=99)
    install_fake_stt(monkeypatch, provider)
    response = api_client.post(f"/api/v1/audio/{audio_file_id}/transcribe")
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["id"])

    # Exhaust the default retry budget (3 attempts).
    for attempt in range(3):
        run_due_jobs(session_factory, expected=1)
        if attempt < 2:
            make_retry_due(session_factory, job_id)

    job = _assert_job_terminal(session_factory, job_id, status=JobStatus.failed)
    assert job.error_summary and "synthetic STT failure" in job.error_summary
    assert provider.calls == 3

    # Exactly one terminal failed transcription row (plus the upload sentinel).
    runs = [
        row
        for row in TranscriptionRepository(session).list_by_audio_file(audio_file_id)
        if row.status is not TranscriptionStatus.not_requested
    ]
    assert len(runs) == 1 and runs[0].status is TranscriptionStatus.failed
    assert runs[0].raw_text == ""
    assert runs[0].metadata_json is not None and runs[0].metadata_json["terminal"] is True

    # Primary data is byte-for-byte intact and the rows are unchanged.
    assert _audio_bytes(session, audio_file_id) == audio_before
    audio_after = AudioFileRepository(session).get(audio_file_id)
    assert audio_after is not None
    assert (audio_after.sha256, audio_after.size_bytes, audio_after.storage_path) == (
        audio_row_before.sha256,
        audio_row_before.size_bytes,
        audio_row_before.storage_path,
    )
    message_after = MessageRepository(session).get(message_id)
    assert message_after is not None
    assert (message_after.kind, message_after.body_text, message_after.deleted_at) == (
        message_before.kind,
        message_before.body_text,
        message_before.deleted_at,
    )


# ---------------------------------------------------------------------------
# failed LLM pipeline (transient, retried): original export untouched
# ---------------------------------------------------------------------------


def test_failed_llm_pipeline_keeps_original_export_intact(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "Синтетика: запись недели для сбоя LLM."
    export = _make_export(session, body)
    export_id = export.id
    original_path = get_settings().data_root / export.path
    original_before = original_path.read_bytes()

    install_fake_llm(
        monkeypatch, FakeLLMProvider(failures=[RuntimeError("synthetic network error")] * 3)
    )
    assert ensure_normalize_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export_id)}
    )
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["id"])

    for attempt in range(3):
        run_due_jobs(session_factory, expected=1)
        if attempt < 2:
            make_retry_due(session_factory, job_id)

    job = _assert_job_terminal(session_factory, job_id, status=JobStatus.failed)
    assert job.error_summary and "synthetic network error" in job.error_summary

    # Exactly one failed processing run; no derived artifact of any kind.
    runs = ProcessingRunRepository(session).list_by_input_export(export_id)
    assert len(runs) == 1
    assert runs[0].status is RunStatus.failed
    assert ArtifactRepository(session).list_by_export(export_id) == []
    assert (
        ArtifactRepository(session).list_by_export_and_layer(export_id, ArtifactLayer.normalized)
        == []
    )
    assert _original_files() == [original_path]  # no partial artifacts anywhere

    # The original export and its row are byte-for-byte unchanged.
    assert original_path.read_bytes() == original_before
    record = WeeklyExportRepository(session).get(export_id)
    assert record is not None and record.sha256 == hashlib.sha256(original_before).hexdigest()


# ---------------------------------------------------------------------------
# malformed LLM JSON: deterministic permanent failure, no partial artifact
# ---------------------------------------------------------------------------


def test_malformed_llm_json_fails_permanently_and_keeps_original_intact(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "Синтетика: запись для невалидного JSON модели."
    export = _make_export(session, body)
    export_id = export.id
    original_path = get_settings().data_root / export.path
    original_before = original_path.read_bytes()

    provider = MalformedJSONProvider()
    install_fake_llm(monkeypatch, provider)
    assert ensure_normalize_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export_id)}
    )
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["id"])

    # Permanent failure: the worker handles the job once and does not retry.
    run_due_jobs(session_factory, expected=1)
    job = _assert_job_terminal(session_factory, job_id, status=JobStatus.failed)
    assert job.attempts == 1
    assert provider.calls == 1

    runs = ProcessingRunRepository(session).list_by_input_export(export_id)
    assert len(runs) == 1
    assert runs[0].status is RunStatus.failed
    assert runs[0].error_summary and runs[0].error_summary.startswith("PipelineOutputError")
    # No artifact was written; nothing new exists under the original layer dir.
    assert (
        ArtifactRepository(session).list_by_export_and_layer(export_id, ArtifactLayer.normalized)
        == []
    )
    assert _original_files() == [original_path]
    assert original_path.read_bytes() == original_before
    record = WeeklyExportRepository(session).get(export_id)
    assert record is not None and record.sha256 == hashlib.sha256(original_before).hexdigest()


# ---------------------------------------------------------------------------
# duplicate worker claim: two workers, one job, executed exactly once
# ---------------------------------------------------------------------------


def test_two_workers_never_execute_the_same_job_twice(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker A claims and blocks mid-transcription; worker B polls in the
    meantime and must not claim the running job. After A finishes, the job is
    ``succeeded`` with exactly one execution and one result row."""
    audio_file_id, _message_id = upload_synthetic_wav(api_client)
    audio_before = _audio_bytes(session, audio_file_id)

    entered = threading.Event()
    release = threading.Event()
    provider = SyntheticSTTProvider(
        outputs=["Единственная расшифровка."],
        before_transcribe=lambda: (entered.set(), release.wait()),
    )
    install_fake_stt(monkeypatch, provider)

    response = api_client.post(f"/api/v1/audio/{audio_file_id}/transcribe")
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["id"])

    worker_a_results: list[int] = []

    def worker_a() -> None:
        worker_a_results.append(process_due_jobs(session_factory, worker_id="worker-a"))

    thread = threading.Thread(target=worker_a)
    thread.start()
    try:
        # Wait until worker A is inside the provider (the job is "running").
        assert entered.wait(timeout=10), "worker A never started the job"
        # Worker B polls while the job is running: it must find nothing to do.
        run_due_jobs(session_factory, expected=0, worker_id="worker-b")
        with session_factory() as check:
            running = JobRepository(check).get(job_id)
            assert running is not None and running.status is JobStatus.running
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()

    assert worker_a_results == [1]
    job = _assert_job_terminal(session_factory, job_id, status=JobStatus.succeeded)
    assert job.attempts == 1
    assert provider.calls == 1  # the model ran exactly once

    runs = [
        row
        for row in TranscriptionRepository(session).list_by_audio_file(audio_file_id)
        if row.status is not TranscriptionStatus.not_requested
    ]
    assert len(runs) == 1 and runs[0].status is TranscriptionStatus.succeeded
    assert runs[0].raw_text == "Единственная расшифровка."
    assert _audio_bytes(session, audio_file_id) == audio_before


# ---------------------------------------------------------------------------
# re-export revision: r2 created, r1 immutable even after pipeline failures
# ---------------------------------------------------------------------------


def test_reexport_creates_r2_and_keeps_r1_byte_identical_after_failures(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "Синтетика: ревизии после сбоя конвейера."
    export = _make_export(session, body)
    r1_path = get_settings().data_root / export.path
    r1_before = r1_path.read_bytes()

    # A malformed-LLM failure runs against r1 first (no data may be harmed).
    install_fake_llm(monkeypatch, MalformedJSONProvider())
    assert ensure_normalize_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export.id)}
    )
    assert response.status_code == 202
    run_due_jobs(session_factory, expected=1)
    assert r1_path.read_bytes() == r1_before

    # Explicit re-export: r2 keeps the lineage sequence number.
    r2 = export_week(session, sunday_date=SUNDAY, re_export=True, now=utc_now())
    assert r2.revision == 2 and r2.seq_number == export.seq_number == 1
    r2_path = get_settings().data_root / r2.path
    r2_bytes = r2_path.read_bytes()
    assert r2_bytes != r1_before
    assert body.encode("utf-8") in r2_bytes  # the week's message is re-exported

    with session_factory() as check:
        rows = WeeklyExportRepository(check).list_by_week(SUNDAY)
        assert [row.revision for row in rows] == [1, 2]
        assert {row.seq_number for row in rows} == {1}

    # r1 is byte-for-byte identical and the failed run left it untouched.
    assert r1_path.read_bytes() == r1_before
    assert hashlib.sha256(r2_bytes).hexdigest() == r2.sha256
    failed_runs = [
        run
        for run in ProcessingRunRepository(session).list_by_input_export(export.id)
        if run.status is RunStatus.failed
    ]
    assert len(failed_runs) == 1
