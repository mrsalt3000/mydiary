"""Integration tests for the real normalization pipeline (Task 13).

End-to-end through the worker helpers and repository code against an
isolated, migrated SQLite test database and a temporary data root. The real
:class:`~diary.domain.pipelines.normalize.pipeline.NormalizePipeline` is
registered explicitly per test and driven with a fake LLM provider (see
``_normalize_fakes``), so no network, YAML file or real model is ever touched.

Covers: a successful single-request run (layer-2 artifact/file, lineage
header, unchanged original), chunked runs (multiple deterministic calls,
ordered merge, summed provenance), rerun → v2 with v1 intact, malformed and
schema-invalid output, dropped/changed entries, unparseable input, and the
guarded registration function.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from _normalize_fakes import NORM_PREFIX, FakeNormalizeLLM, echo_records, extract_input_records
from _pipeline_fakes import (
    FakeLLMProvider,
    install_fake_llm,
    json_llm_response,
    register_fake_normalize_pipeline,
)
from sqlalchemy.orm import Session

from diary.core.settings import get_settings, reload_settings
from diary.db.models import (
    JobStatus,
    MessageKind,
    PipelineType,
    RunStatus,
    TranscriptionStatus,
)
from diary.db.repositories import (
    ArtifactRepository,
    JobRepository,
    MessageRepository,
    ProcessingRunRepository,
    WeeklyExportRepository,
)
from diary.domain.exports.service import export_week
from diary.domain.llm.models import LLMResponse
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines import service as pipeline_service
from diary.domain.pipelines.normalize.pipeline import NormalizePipeline
from diary.domain.pipelines.normalize.registration import (
    ensure_normalize_pipeline_registered,
)
from diary.workers.runner import process_due_jobs

SUNDAY = date(2026, 9, 13)
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


@pytest.fixture(autouse=True)
def _restore_pipeline_registry() -> None:
    """Restore the baseline pipeline registry after every test."""
    snapshot = dict(pipeline_registry.REGISTRY)
    yield
    pipeline_registry.REGISTRY.clear()
    pipeline_registry.REGISTRY.update(snapshot)


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc)


def _add_text(session: Session, body: str, created_at: datetime) -> None:
    MessageRepository(session).create(kind=MessageKind.text, body_text=body, created_at=created_at)
    session.commit()


def _add_audio(session: Session, created_at: datetime, *, raw_stt: str | None) -> None:
    """Insert an audio message with its audio-file row and, when given, a
    succeeded transcription (mirrors the Task-10 test helper)."""
    from diary.db.repositories import AudioFileRepository, TranscriptionRepository

    message = MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=created_at
    )
    audio = AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/wav",
        storage_path="2026/09/voice.wav",
        size_bytes=123,
        sha256="0" * 64,
        created_at=created_at,
    )
    if raw_stt is not None:
        TranscriptionRepository(session).create(
            audio_file_id=audio.id,
            raw_text=raw_stt,
            status=TranscriptionStatus.succeeded,
            created_at=created_at,
        )
    session.commit()


def _create_export(
    session: Session,
    *,
    n_text: int = 1,
    transcribed_audio: bool = False,
    untranscribed_audio: bool = False,
) -> uuid.UUID:
    """Seed the week with synthetic messages and create the original export."""
    base_day = 7
    for position in range(n_text):
        _add_text(session, f"Синтетика: текстовая запись {position}.", _at(base_day, 6 + position))
    if transcribed_audio:
        _add_audio(session, _at(8, 10), raw_stt="Синтетика: расшифровка голоса.")
    if untranscribed_audio:
        _add_audio(session, _at(8, 12), raw_stt=None)
    record = export_week(session, sunday_date=SUNDAY)
    return record.id


def _export_file(export) -> Path:
    return get_settings().data_root / export.path


def _normalized_files() -> list[Path]:
    directory = get_settings().exports_dir / "02_normalized"
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _run_rows(session: Session, job_id: uuid.UUID):
    return ProcessingRunRepository(session).list_by_job_id(job_id)


def _run_successfully(session_factory, export_id: uuid.UUID):
    """Enqueue + process one normalize job; return the job id."""
    with session_factory() as s:
        job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.normalize, export_id=export_id
        )
        job_id = job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    return job_id


def _expect_failed_run(session: Session, job_id: uuid.UUID) -> str:
    stored = JobRepository(session).get(job_id)
    assert stored is not None and stored.status is JobStatus.failed
    assert stored.attempts == 1  # deterministic failure, no retries burned
    runs = _run_rows(session, job_id)
    assert len(runs) == 1 and runs[0].status is RunStatus.failed
    assert runs[0].error_summary is not None
    return runs[0].error_summary


# ---------------------------------------------------------------------------
# successful single-request run
# ---------------------------------------------------------------------------


def test_successful_run_writes_layer2_artifact_and_preserves_original(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeNormalizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_normalize_pipeline_registered() is True

    export_id = _create_export(
        session,
        n_text=2,
        transcribed_audio=True,
        untranscribed_audio=True,
    )
    export = WeeklyExportRepository(session).get(export_id)
    assert export is not None
    original_before = _export_file(export).read_bytes()

    job_id = _run_successfully(session_factory, export_id)

    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.succeeded
        runs = _run_rows(s, job_id)
        assert len(runs) == 1
        run = runs[0]
        assert run.status is RunStatus.succeeded
        assert run.pipeline_type is PipelineType.normalize
        assert run.input_export_id == export_id
        assert run.profile_name == "normalization"
        assert run.model_name == "fake-norm-model"
        assert run.prompt_version == "v1"
        metadata = run.metadata_json or {}
        assert "call_count" not in metadata
        assert isinstance(metadata["input_sha256"], str) and len(metadata["input_sha256"]) == 64

        artifacts = ArtifactRepository(s).list_by_export(export_id)
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.layer.value == "normalized"
        assert artifact.version == 1
        assert artifact.processing_run_id == run.id
        assert artifact.path == "exports/02_normalized/2026-09-13_normalized_0001_r1_v1.txt"

    files = _normalized_files()
    assert [path.name for path in files] == ["2026-09-13_normalized_0001_r1_v1.txt"]
    written = files[0].read_text(encoding="utf-8")
    assert "# Дневник — нормализованный" in written
    assert "Неделя: 2026-09-07 — 2026-09-13" in written
    assert "Номер выгрузки: 0001" in written
    assert "Ревизия: r1" in written
    assert "Версия: v1" in written
    assert "Оригинал: 2026-09-13_original_0001_r1.txt" in written
    assert "Timezone: Europe/Moscow" in written
    # Content entries keep the original order (two texts, then the
    # transcription of the audio entry).
    assert written.index(f"{NORM_PREFIX}Синтетика: текстовая запись 0.") < written.index(
        f"{NORM_PREFIX}Синтетика: текстовая запись 1."
    )
    assert written.index(f"{NORM_PREFIX}Синтетика: текстовая запись 1.") < written.index(
        f"{NORM_PREFIX}Синтетика: расшифровка голоса."
    )
    # ... and the untranscribed audio entry keeps its marker as TEXT content.
    assert "[2026-09-08 15:00] TEXT\n[не расшифровано]" in written
    # Layer-2 file has no layer-1 audio scaffolding.
    assert "STT raw:" not in written
    assert "Файл:" not in written
    assert written.count(NORM_PREFIX) == 3

    digest = hashlib.sha256(files[0].read_bytes()).hexdigest()
    with session_factory() as s:
        artifacts = ArtifactRepository(s).list_by_export(export_id)
        assert artifacts[0].sha256 == digest
    # The original export is byte-for-byte untouched.
    assert _export_file(export).read_bytes() == original_before


# ---------------------------------------------------------------------------
# rerun → v2
# ---------------------------------------------------------------------------


def test_rerun_creates_v2_and_keeps_v1_intact(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeNormalizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)

    first_job = _run_successfully(session_factory, export_id)
    files = _normalized_files()
    assert [p.name for p in files] == ["2026-09-13_normalized_0001_r1_v1.txt"]
    v1_bytes = files[0].read_bytes()

    second_job = _run_successfully(session_factory, export_id)
    assert second_job != first_job
    files = _normalized_files()
    assert [p.name for p in files] == [
        "2026-09-13_normalized_0001_r1_v1.txt",
        "2026-09-13_normalized_0001_r1_v2.txt",
    ]
    assert files[0].read_bytes() == v1_bytes
    assert "Версия: v2" in files[1].read_text(encoding="utf-8")
    with session_factory() as s:
        artifacts = ArtifactRepository(s).list_by_export(export_id)
        assert [artifact.version for artifact in artifacts] == [1, 2]


# ---------------------------------------------------------------------------
# chunked input
# ---------------------------------------------------------------------------


def test_chunked_export_makes_several_calls_and_merges_in_order(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_NORMALIZE_CHUNK_ENTRIES", "2")
    reload_settings()
    provider = FakeNormalizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=5)

    job_id = _run_successfully(session_factory, export_id)
    assert provider.calls == 3  # chunks of 2, 2, 1

    per_request = [len(extract_input_records(request)) for request in provider.requests]
    assert per_request == [2, 2, 1]
    # Order is preserved across chunks (the union of indices is 0..4 in order).
    all_indices = [
        record["index"]
        for request in provider.requests
        for record in extract_input_records(request)
    ]
    assert all_indices == [0, 1, 2, 3, 4]

    with session_factory() as s:
        runs = _run_rows(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.succeeded
        metadata = runs[0].metadata_json or {}
        assert metadata["call_count"] == 3
        assert isinstance(metadata["input_sha256"], list) and len(metadata["input_sha256"]) == 3

    files = _normalized_files()
    assert [p.name for p in files] == ["2026-09-13_normalized_0001_r1_v1.txt"]
    written = files[0].read_text(encoding="utf-8")
    assert "Версия: v1" in written
    for position in range(5):
        assert f"{NORM_PREFIX}Синтетика: текстовая запись {position}." in written
    positions = [
        written.index(f"{NORM_PREFIX}Синтетика: текстовая запись {position}.")
        for position in range(5)
    ]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# deterministic failures
# ---------------------------------------------------------------------------


def test_malformed_json_fails_without_writing_a_file(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(response=LLMResponse(text="это не JSON вовсе", model="fake"))
    install_fake_llm(monkeypatch, provider)
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)
    export = WeeklyExportRepository(session).get(export_id)
    assert export is not None
    original_before = _export_file(export).read_bytes()

    job_id = _run_successfully(session_factory, export_id)
    with session_factory() as s:
        summary = _expect_failed_run(s, job_id)
        assert "structured validation" in summary
        assert ArtifactRepository(s).list_by_export(export_id) == []
    assert _normalized_files() == []
    assert _export_file(export).read_bytes() == original_before


def test_schema_invalid_json_fails_without_writing_a_file(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(response=json_llm_response({"unexpected": "shape"}))
    install_fake_llm(monkeypatch, provider)
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)

    job_id = _run_successfully(session_factory, export_id)
    with session_factory() as s:
        summary = _expect_failed_run(s, job_id)
        assert "structured validation" in summary
        assert ArtifactRepository(s).list_by_export(export_id) == []
    assert _normalized_files() == []


def test_dropped_entry_fails_with_completeness_error(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeNormalizeLLM(transform=lambda records: echo_records(records)[:-1])
    install_fake_llm(monkeypatch, provider)
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=2)

    job_id = _run_successfully(session_factory, export_id)
    with session_factory() as s:
        summary = _expect_failed_run(s, job_id)
        assert "dropped or added entries" in summary
        assert ArtifactRepository(s).list_by_export(export_id) == []
    assert _normalized_files() == []


def test_changed_timestamp_fails_the_run(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _relabel(records):
        output = echo_records(records)
        if output:
            output[0]["timestamp"] = "1999-01-01 00:00"
        return output

    provider = FakeNormalizeLLM(transform=_relabel)
    install_fake_llm(monkeypatch, provider)
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)

    job_id = _run_successfully(session_factory, export_id)
    with session_factory() as s:
        summary = _expect_failed_run(s, job_id)
        assert "timestamp or source kind" in summary
        assert ArtifactRepository(s).list_by_export(export_id) == []
    assert _normalized_files() == []


def test_unparseable_original_fails_permanently(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeNormalizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)
    export = WeeklyExportRepository(session).get(export_id)
    assert export is not None
    # Corrupt the input file (simulates an on-disk inconsistency).
    _export_file(export).write_bytes("вообще не формат выгрузки".encode("utf-8"))

    job_id = _run_successfully(session_factory, export_id)
    with session_factory() as s:
        summary = _expect_failed_run(s, job_id)
        assert "cannot be parsed" in summary
        assert ArtifactRepository(s).list_by_export(export_id) == []
    assert _normalized_files() == []


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_ensure_registration_is_guarded_and_idempotent() -> None:
    assert pipeline_registry.get_pipeline(PipelineType.normalize) is None
    assert ensure_normalize_pipeline_registered() is True
    registered = pipeline_registry.get_pipeline(PipelineType.normalize)
    assert isinstance(registered, NormalizePipeline)
    # A second call is a no-op (idempotent startup call).
    assert ensure_normalize_pipeline_registered() is False
    assert pipeline_registry.get_pipeline(PipelineType.normalize) is registered


def test_ensure_registration_does_not_replace_a_registered_fake() -> None:
    register_fake_normalize_pipeline()
    fake = pipeline_registry.get_pipeline(PipelineType.normalize)
    assert fake is not None and not isinstance(fake, NormalizePipeline)
    # The guarded registration keeps the existing (fake) implementation.
    assert ensure_normalize_pipeline_registered() is False
    assert pipeline_registry.get_pipeline(PipelineType.normalize) is fake
