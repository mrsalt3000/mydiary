"""Integration tests for the pipeline framework core (Task 12).

End-to-end through the real worker helpers and repository code, against an
isolated migrated test database and a temporary data root. A fake LLM
provider and a fake ``normalize`` pipeline (see ``_pipeline_fakes``) replace
the real endpoint/config/model, so no network or YAML file is ever needed.
The ``wellbeing`` pipeline stays unregistered (as before Task 15) to exercise
the "pipeline not registered yet" permanent failure.

Covers: successful run (run/artifact/file/lineage + provenance without
content), rerun → v2 with v1 byte-identical, replay idempotency, invalid
structured output, missing export / missing original file, atomic-write
failure, unregistered pipeline, transient LLM retries, enqueue validation and
active-job deduplication.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from _pipeline_fakes import (
    INVALID_OUTPUT_MESSAGE,
    FakeLLMProvider,
    install_fake_llm,
    json_llm_response,
    register_fake_normalize_pipeline,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.core.settings import get_settings, reload_settings
from diary.db.base import utc_now
from diary.db.models import Job, JobStatus, JobType, MessageKind, PipelineType, RunStatus
from diary.db.repositories import (
    ArtifactRepository,
    JobRepository,
    MessageRepository,
    ProcessingRunRepository,
)
from diary.domain.exports.service import export_week
from diary.domain.llm.errors import LLMNetworkError, LLMRateLimitError
from diary.domain.llm.models import LLMUsage
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines import service as pipeline_service
from diary.domain.pipelines.base import (
    PipelineExportNotFoundError,
    PipelineOriginalFileMissingError,
)
from diary.workers import handlers
from diary.workers.runner import process_due_jobs

MOSCOW_SUNDAY = date(2026, 9, 13)
DIARY_PHRASE = "Синтетика: приватный текст для выгрузки."
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
    """Restore the baseline pipeline registry (empty) after every test."""
    snapshot = dict(pipeline_registry.REGISTRY)
    yield
    pipeline_registry.REGISTRY.clear()
    pipeline_registry.REGISTRY.update(snapshot)


def _add_text(session: Session, body: str, created_at: datetime) -> None:
    MessageRepository(session).create(kind=MessageKind.text, body_text=body, created_at=created_at)
    session.commit()


def _create_export(session: Session):
    """Create the r1 original export of the default week and return its row."""
    _add_text(session, DIARY_PHRASE, datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    return export_week(session, sunday_date=MOSCOW_SUNDAY)


def _original_bytes(export) -> bytes:
    return (get_settings().data_root / export.path).read_bytes()


def _layer_files(layer_dir: str) -> list[Path]:
    directory = get_settings().exports_dir / layer_dir
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _run_rows_for_job(session: Session, job_id: uuid.UUID):
    return ProcessingRunRepository(session).list_by_job_id(job_id)


def _normalized_response() -> object:
    return {"lines": ["# Обработанная выгрузка", "Синтетическая нормализованная строка."]}


def _fake_response(usage: LLMUsage | None = None):
    return json_llm_response(_normalized_response(), usage=usage)


def _make_retry_due(session_factory, job_id: uuid.UUID) -> None:
    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None and job.status is JobStatus.retrying
        job.next_retry_at = utc_now() - timedelta(seconds=1)
        session.commit()


# ---------------------------------------------------------------------------
# successful run
# ---------------------------------------------------------------------------


def test_successful_run_writes_artifact_and_run_with_provenance(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(
        response=_fake_response(usage=LLMUsage(input_tokens=11, output_tokens=3))
    )
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    export = _create_export(session)
    original_before = _original_bytes(export)

    with session_factory() as s:
        job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.normalize, export_id=export.id
        )
        job_id = job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as s:
        stored_job = JobRepository(s).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.succeeded
        assert stored_job.attempts == 1
        assert stored_job.error_summary is None

        runs = _run_rows_for_job(s, job_id)
        assert len(runs) == 1
        run = runs[0]
        assert run.status is RunStatus.succeeded
        assert run.pipeline_type is PipelineType.normalize
        assert run.input_export_id == export.id
        assert run.input_artifact_id is None
        assert run.profile_name == "normalization"
        assert run.model_name == "fake-norm-model"
        assert run.prompt_version == "v1"
        assert run.started_at is not None and run.finished_at is not None
        assert run.finished_at >= run.started_at
        assert run.error_summary is None
        metadata = run.metadata_json or {}
        assert metadata["job_id"] == str(job_id)
        assert metadata["provider_type"] == "openai_compatible"
        assert metadata["model"] == "fake-norm-model"
        assert metadata["prompt_version"] == "v1"
        assert metadata["usage"] == {"input_tokens": 11, "output_tokens": 3}
        assert len(metadata["input_sha256"]) == 64
        # Provenance never carries diary content.
        dumped = json.dumps(metadata)
        assert DIARY_PHRASE not in dumped
        assert "приватный текст" not in dumped

        artifacts = ArtifactRepository(s).list_by_export(export.id)
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.layer.value == "normalized"
        assert artifact.version == 1
        assert artifact.processing_run_id == run.id
        assert artifact.path == "exports/02_normalized/2026-09-13_normalized_0001_r1_v1.txt"

    # The rendered file matches the validated structured output and the hash.
    files = _layer_files("02_normalized")
    assert [path.name for path in files] == ["2026-09-13_normalized_0001_r1_v1.txt"]
    written = files[0].read_text(encoding="utf-8")
    assert written == "# Обработанная выгрузка\nСинтетическая нормализованная строка.\n"
    digest = hashlib.sha256(files[0].read_bytes()).hexdigest()
    with session_factory() as s:
        artifacts = ArtifactRepository(s).list_by_export(export.id)
        assert len(artifacts) == 1
        assert artifacts[0].sha256 == digest
    # The original export is byte-for-byte untouched.
    assert _original_bytes(export) == original_before


# ---------------------------------------------------------------------------
# rerun → v2
# ---------------------------------------------------------------------------


def test_rerun_creates_v2_and_keeps_v1_intact(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(response=_fake_response())
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    export = _create_export(session)

    with session_factory() as s:
        first_job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.normalize, export_id=export.id
        )
        first_job_id = first_job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    v1_files = _layer_files("02_normalized")
    assert [p.name for p in v1_files] == ["2026-09-13_normalized_0001_r1_v1.txt"]
    v1_bytes = v1_files[0].read_bytes()

    # An explicit repeat after a success starts a fresh job (rerun → v2).
    with session_factory() as s:
        second_job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.normalize, export_id=export.id
        )
        second_job_id = second_job.id
        assert second_job_id != first_job_id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    files = _layer_files("02_normalized")
    assert [p.name for p in files] == [
        "2026-09-13_normalized_0001_r1_v1.txt",
        "2026-09-13_normalized_0001_r1_v2.txt",
    ]
    # v1 is byte-identical after the rerun.
    assert files[0].read_bytes() == v1_bytes

    with session_factory() as s:
        artifacts = ArtifactRepository(s).list_by_export(export.id)
        assert [artifact.version for artifact in artifacts] == [1, 2]


# ---------------------------------------------------------------------------
# replay idempotency
# ---------------------------------------------------------------------------


def test_replaying_a_succeeded_job_is_a_noop(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker "crash" after the handler committed the run/artifact, then the
    job is executed again (stale-running reset): the second run_pipeline call
    must not call the provider again or create a second artifact."""
    provider = FakeLLMProvider(response=_fake_response())
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    export = _create_export(session)

    with session_factory() as s:
        job = JobRepository(s).enqueue(
            job_type=JobType.normalize,
            payload_json={"pipeline_type": "normalize", "export_id": str(export.id)},
            max_attempts=3,
        )
        job_id = job.id
        s.commit()
        claimed = JobRepository(s).claim_due()
        assert claimed is not None and claimed.id == job_id
        s.commit()

    # Execute the handler and commit; the job stays "running" (simulated crash
    # before the worker's mark_succeeded).
    handler = handlers.get_handler(JobType.normalize)
    assert handler is not None
    with session_factory() as s:
        handler(s, JobRepository(s).get(job_id))
        s.commit()

    with session_factory() as s:
        assert JobRepository(s).get(job_id).status is JobStatus.running

    # Replay the same job: run_pipeline detects the committed succeeded run.
    with session_factory() as s:
        pipeline_service.run_pipeline(s, JobRepository(s).get(job_id))

    with session_factory() as s:
        assert len(_run_rows_for_job(s, job_id)) == 1
        artifacts = ArtifactRepository(s).list_by_export(export.id)
        assert len(artifacts) == 1  # no duplicate artifact
    assert provider.calls == 1  # the model ran exactly once


# ---------------------------------------------------------------------------
# deterministic failures
# ---------------------------------------------------------------------------


def test_invalid_output_fails_run_without_writing_a_file(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(response=json_llm_response({"unexpected": "shape"}))
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    export = _create_export(session)
    original_before = _original_bytes(export)

    with session_factory() as s:
        job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.normalize, export_id=export.id
        )
        job_id = job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as s:
        stored_job = JobRepository(s).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.failed
        assert stored_job.attempts == 1
        assert stored_job.error_summary is not None
        assert INVALID_OUTPUT_MESSAGE in stored_job.error_summary

        runs = _run_rows_for_job(s, job_id)
        assert len(runs) == 1
        run = runs[0]
        assert run.status is RunStatus.failed
        assert run.error_summary is not None
        assert INVALID_OUTPUT_MESSAGE in run.error_summary
        assert DIARY_PHRASE not in run.error_summary
        assert run.metadata_json is not None and run.metadata_json["terminal"] is True

        assert ArtifactRepository(s).list_by_export(export.id) == []
    # No file was written and the original is untouched.
    assert _layer_files("02_normalized") == []
    assert _original_bytes(export) == original_before


def test_missing_export_fails_permanently(session_factory) -> None:
    register_fake_normalize_pipeline()
    with session_factory() as s:
        job = JobRepository(s).enqueue(
            job_type=JobType.normalize,
            payload_json={"pipeline_type": "normalize", "export_id": str(uuid.uuid4())},
            max_attempts=5,
        )
        job_id = job.id
        s.commit()

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as s:
        stored_job = JobRepository(s).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.failed
        assert stored_job.attempts == 1  # failed immediately, no retries burned
        assert (
            stored_job.error_summary is not None and "Export not found." in stored_job.error_summary
        )  # noqa: E501
        runs = _run_rows_for_job(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.failed
        assert runs[0].error_summary is not None and "Export not found." in runs[0].error_summary
    assert _layer_files("02_normalized") == []


def test_missing_original_file_fails_permanently(
    session: Session,
    session_factory,
) -> None:
    register_fake_normalize_pipeline()
    export = _create_export(session)
    (get_settings().data_root / export.path).unlink()

    with session_factory() as s:
        job = JobRepository(s).enqueue(
            job_type=JobType.normalize,
            payload_json={"pipeline_type": "normalize", "export_id": str(export.id)},
            max_attempts=5,
        )
        job_id = job.id
        s.commit()

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as s:
        stored_job = JobRepository(s).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.failed
        assert stored_job.error_summary is not None and "missing" in stored_job.error_summary
        runs = _run_rows_for_job(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.failed
        assert runs[0].input_export_id == export.id
        assert runs[0].error_summary is not None and "missing" in runs[0].error_summary
        assert ArtifactRepository(s).list_by_export(export.id) == []
    assert _layer_files("02_normalized") == []


def test_atomic_write_failure_fails_run_with_no_partial_artifact(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(response=_fake_response())
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    export = _create_export(session)
    original_before = _original_bytes(export)

    # Occupy the target layer-directory node with a regular file so the atomic
    # writer cannot create ``02_normalized`` (real storage failure, no
    # monkeypatched writer).
    blocked = get_settings().exports_dir / "02_normalized"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("occupied", encoding="utf-8")

    with session_factory() as s:
        job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.normalize, export_id=export.id
        )
        job_id = job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as s:
        stored_job = JobRepository(s).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.failed
        runs = _run_rows_for_job(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.failed
        assert (
            runs[0].error_summary is not None and "pipeline artifact file" in runs[0].error_summary
        )  # noqa: E501
        assert ArtifactRepository(s).list_by_export(export.id) == []
    # The blocking node is untouched and no scratch file was left behind.
    assert blocked.read_text(encoding="utf-8") == "occupied"
    temp_dir = get_settings().temp_dir
    leftovers = [p for p in temp_dir.rglob("*") if p.is_file()] if temp_dir.exists() else []
    assert leftovers == []
    assert _original_bytes(export) == original_before


def test_unregistered_pipeline_fails_permanently(
    session: Session,
    session_factory,
) -> None:
    """wellbeing is intentionally not registered before Task 15: its job must
    fail permanently with a clear summary instead of retrying."""
    export = _create_export(session)

    with session_factory() as s:
        job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.wellbeing, export_id=export.id
        )
        job_id = job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as s:
        stored_job = JobRepository(s).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.failed
        assert stored_job.attempts == 1
        assert (
            stored_job.error_summary is not None
            and "not registered yet" in stored_job.error_summary
        )  # noqa: E501
        runs = _run_rows_for_job(s, job_id)
        assert len(runs) == 1
        run = runs[0]
        assert run.status is RunStatus.failed
        assert run.input_export_id == export.id
        assert run.error_summary is not None and "not registered yet" in run.error_summary
        assert ArtifactRepository(s).list_by_export(export.id) == []
    assert _layer_files("04_wellbeing") == []


# ---------------------------------------------------------------------------
# transient LLM failures follow the retry policy
# ---------------------------------------------------------------------------


def test_transient_llm_failure_retries_then_succeeds(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(
        response=_fake_response(),
        failures=[
            LLMRateLimitError(
                "The LLM endpoint rate-limited the request (HTTP 429).", status_code=429
            )
        ],
    )
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    export = _create_export(session)

    with session_factory() as s:
        job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.normalize, export_id=export.id
        )
        job_id = job.id
        assert job.max_attempts == 3

    # First attempt: transient → retrying with backoff, no terminal run yet.
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as s:
        stored_job = JobRepository(s).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.retrying
        assert _run_rows_for_job(s, job_id) == []

    # Second attempt succeeds.
    _make_retry_due(session_factory, job_id)
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as s:
        stored_job = JobRepository(s).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.succeeded
        runs = _run_rows_for_job(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.succeeded
        assert len(ArtifactRepository(s).list_by_export(export.id)) == 1
    assert provider.calls == 2


def test_terminal_transient_failure_records_single_failed_run(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(
        response=_fake_response(),
        failures=[LLMNetworkError("The LLM endpoint is unreachable (network error).")],
    )
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    export = _create_export(session)

    with session_factory() as s:
        job = JobRepository(s).enqueue(
            job_type=JobType.normalize,
            payload_json={"pipeline_type": "normalize", "export_id": str(export.id)},
            max_attempts=1,
        )
        job_id = job.id
        s.commit()

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as s:
        stored_job = JobRepository(s).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.failed
        runs = _run_rows_for_job(s, job_id)
        assert len(runs) == 1
        run = runs[0]
        assert run.status is RunStatus.failed
        assert run.error_summary is not None and "LLMNetworkError" in run.error_summary
        assert ArtifactRepository(s).list_by_export(export.id) == []
    assert _layer_files("02_normalized") == []
    assert provider.calls == 1


# ---------------------------------------------------------------------------
# enqueueing rules
# ---------------------------------------------------------------------------


def test_enqueue_rejects_unknown_export(session: Session) -> None:
    with pytest.raises(PipelineExportNotFoundError):
        pipeline_service.enqueue_pipeline_run(
            session, pipeline_type=PipelineType.normalize, export_id=uuid.uuid4()
        )


def test_enqueue_rejects_export_with_missing_file(session: Session) -> None:
    export = _create_export(session)
    (get_settings().data_root / export.path).unlink()
    with pytest.raises(PipelineOriginalFileMissingError):
        pipeline_service.enqueue_pipeline_run(
            session, pipeline_type=PipelineType.normalize, export_id=export.id
        )


def test_enqueue_does_not_duplicate_an_active_job(
    session: Session,
    session_factory,
) -> None:
    export = _create_export(session)
    first = pipeline_service.enqueue_pipeline_run(
        session, pipeline_type=PipelineType.normalize, export_id=export.id
    )
    second = pipeline_service.enqueue_pipeline_run(
        session, pipeline_type=PipelineType.normalize, export_id=export.id
    )
    assert second.id == first.id
    with session_factory() as s:
        count = s.scalar(select(func.count()).select_from(Job)) or 0
        assert count == 1
        active = pipeline_service.find_active_pipeline_job(
            s, pipeline_type=PipelineType.normalize, export_id=export.id
        )
        assert active is not None and active.id == first.id


def test_enqueue_allows_a_fresh_job_after_terminal_outcome(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLLMProvider(response=_fake_response())
    install_fake_llm(monkeypatch, provider)
    register_fake_normalize_pipeline()
    export = _create_export(session)

    first = pipeline_service.enqueue_pipeline_run(
        session, pipeline_type=PipelineType.normalize, export_id=export.id
    )
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    second = pipeline_service.enqueue_pipeline_run(
        session, pipeline_type=PipelineType.normalize, export_id=export.id
    )
    assert second.id != first.id
