"""Integration tests for the real summarization pipeline (Task 14).

End-to-end through the worker helpers and repository code against an
isolated, migrated SQLite test database and a temporary data root. The real
normalization pipeline produces the layer-2 input with a fake LLM; the real
:class:`~diary.domain.pipelines.summarize.pipeline.SummarizePipeline` is
registered explicitly per test and driven with a fake LLM provider (see
``_summarize_fakes``), so no network, YAML file or real model is ever touched.

Covers: a successful single-pass run (layer-3 artifact/file, lineage header
referencing the exact normalized source, untouched normalized input), rerun →
v2 with v1 intact, map-reduce selection above the configured threshold
(deterministic per-chunk calls, ordered deterministic merge, wellbeing merge),
malformed and schema-invalid output, an input export without a normalized
artifact (permanent failure, provider never called), batch eligibility and the
guarded registration function.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from _normalize_fakes import FakeNormalizeLLM
from _pipeline_fakes import FakeLLMProvider, install_fake_llm, json_llm_response
from _summarize_fakes import (
    FakeSummarizeLLM,
    default_summary_payload,
    extract_mode,
    summary_payload,
)
from sqlalchemy.orm import Session

from diary.core.settings import get_settings, reload_settings
from diary.db.models import (
    ArtifactLayer,
    JobStatus,
    MessageKind,
    PipelineType,
    RunStatus,
)
from diary.db.repositories import (
    ArtifactRepository,
    JobRepository,
    MessageRepository,
    ProcessingRunRepository,
)
from diary.domain.exports.service import export_week
from diary.domain.llm.models import LLMResponse
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines import service as pipeline_service
from diary.domain.pipelines.normalize.registration import (
    ensure_normalize_pipeline_registered,
)
from diary.domain.pipelines.summarize.batch import (
    enqueue_summarize_batch,
    list_eligible_exports,
)
from diary.domain.pipelines.summarize.pipeline import SummarizePipeline
from diary.domain.pipelines.summarize.registration import (
    ensure_summarize_pipeline_registered,
)
from diary.workers.runner import process_due_jobs

SUNDAY = date(2026, 9, 13)
DIARY_PHRASE = "Синтетика: приватный текст для саммари."
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


@pytest.fixture(autouse=True)
def _restore_pipeline_registry() -> None:
    """Restore the baseline pipeline registry after every test."""
    snapshot = dict(pipeline_registry.REGISTRY)
    yield
    pipeline_registry.REGISTRY.clear()
    pipeline_registry.REGISTRY.update(snapshot)


def _message_time(sunday: date, position: int) -> datetime:
    """A UTC instant inside the export's week (Monday 09:00 + ``position`` h)."""
    monday = sunday - timedelta(days=6)
    return datetime(monday.year, monday.month, monday.day, 9, position, tzinfo=timezone.utc)


def _create_export(session: Session, *, sunday: date = SUNDAY, n_text: int = 1) -> uuid.UUID:
    """Seed a week with synthetic text messages and create the original."""
    for position in range(n_text):
        body = f"{DIARY_PHRASE} Запись {position}."
        MessageRepository(session).create(
            kind=MessageKind.text, body_text=body, created_at=_message_time(sunday, position)
        )
    session.commit()
    record = export_week(session, sunday_date=sunday)
    return record.id


def _summary_files() -> list[Path]:
    directory = get_settings().exports_dir / "03_summaries"
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _run_rows(session: Session, job_id: uuid.UUID):
    return ProcessingRunRepository(session).list_by_job_id(job_id)


def _run_normalization(session_factory, export_id: uuid.UUID) -> None:
    """Run the real normalization pipeline with the fake normalize LLM."""
    with session_factory() as s:
        job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.normalize, export_id=export_id
        )
        job_id = job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.succeeded


def _normalized_artifact_id(session_factory, export_id: uuid.UUID) -> uuid.UUID:
    with session_factory() as s:
        artifacts = ArtifactRepository(s).list_by_export_and_layer(
            export_id, ArtifactLayer.normalized
        )
        assert len(artifacts) == 1
        return artifacts[0].id


def _run_summarization(session_factory, export_id: uuid.UUID) -> uuid.UUID:
    """Enqueue + process one summarize job; return the job id."""
    with session_factory() as s:
        job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.summarize, export_id=export_id
        )
        job_id = job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    return job_id


# ---------------------------------------------------------------------------
# successful single-pass run
# ---------------------------------------------------------------------------


def test_single_pass_writes_summary_from_normalized_input(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=2)
    _run_normalization(session_factory, export_id)
    normalized_id = _normalized_artifact_id(session_factory, export_id)

    summarize_provider = FakeSummarizeLLM()
    install_fake_llm(monkeypatch, summarize_provider)
    assert ensure_summarize_pipeline_registered() is True
    job_id = _run_summarization(session_factory, export_id)

    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.succeeded
        runs = _run_rows(s, job_id)
        assert len(runs) == 1
        run = runs[0]
        assert run.status is RunStatus.succeeded
        assert run.pipeline_type is PipelineType.summarize
        assert run.input_export_id == export_id
        # The layer-3 run consumed the normalized artifact by database id.
        assert run.input_artifact_id == normalized_id
        assert run.profile_name == "summarization"
        assert run.model_name == "fake-summary-model"
        assert run.prompt_version == "v1"
        metadata = run.metadata_json or {}
        assert "call_count" not in metadata  # single-pass shape
        assert isinstance(metadata["input_sha256"], str) and len(metadata["input_sha256"]) == 64
        dumped = json.dumps(metadata)
        assert DIARY_PHRASE not in dumped
        assert "Синтетическое событие" not in dumped  # no partial/final content

        artifacts = ArtifactRepository(s).list_by_export(export_id)
        assert len(artifacts) == 2  # normalized v1 + summary v1
        summary_artifact = artifacts[-1]
        assert summary_artifact.layer.value == "summary"
        assert summary_artifact.version == 1
        assert summary_artifact.processing_run_id == run.id
        assert summary_artifact.path == "exports/03_summaries/2026-09-13_summary_0001_r1_v1.md"

    files = _summary_files()
    assert [path.name for path in files] == ["2026-09-13_summary_0001_r1_v1.md"]
    written = files[0].read_text(encoding="utf-8")
    assert "# Саммари недели" in written
    assert "Неделя: 2026-09-07 — 2026-09-13" in written
    assert "Номер выгрузки: 0001" in written
    assert "Ревизия: r1" in written
    assert "Версия: v1" in written
    # The exact normalized source artifact is identified.
    assert "Источник (нормализованный): 2026-09-13_normalized_0001_r1_v1.txt" in written
    assert "Timezone: Europe/Moscow" in written
    assert "## Ключевые события" in written
    assert "- Синтетическое событие всей недели." in written
    assert "## Работа и нагрузка" in written
    # Default wellbeing payload has no explicit source support → the rendered
    # no-data statement, never an invented reading.
    assert "## Самочувствие и эмоции" in written
    assert "Явных данных о самочувствии и эмоциях в источнике нет." in written
    assert "## Поддерживающие факторы" in written
    assert "## Открытые вопросы и неизвестное" in written

    digest = hashlib.sha256(files[0].read_bytes()).hexdigest()
    with session_factory() as s:
        artifacts = ArtifactRepository(s).list_by_export_and_layer(export_id, ArtifactLayer.summary)
        assert artifacts[0].sha256 == digest
    assert summarize_provider.calls == 1  # single-pass below the threshold
    # The layer-2 input file is untouched by the layer-3 run.
    normalized_dir = get_settings().exports_dir / "02_normalized"
    normalized_files = [p for p in normalized_dir.rglob("*.txt")]
    assert [p.name for p in normalized_files] == ["2026-09-13_normalized_0001_r1_v1.txt"]


def test_rerun_creates_v2_and_keeps_v1_intact(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)
    _run_normalization(session_factory, export_id)

    provider = FakeSummarizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True
    first_job = _run_summarization(session_factory, export_id)
    files = _summary_files()
    assert [p.name for p in files] == ["2026-09-13_summary_0001_r1_v1.md"]
    v1_bytes = files[0].read_bytes()

    second_job = _run_summarization(session_factory, export_id)
    assert second_job != first_job
    files = _summary_files()
    assert [p.name for p in files] == [
        "2026-09-13_summary_0001_r1_v1.md",
        "2026-09-13_summary_0001_r1_v2.md",
    ]
    assert files[0].read_bytes() == v1_bytes
    assert "Версия: v2" in files[1].read_text(encoding="utf-8")
    with session_factory() as s:
        artifacts = ArtifactRepository(s).list_by_export_and_layer(export_id, ArtifactLayer.summary)
        assert [artifact.version for artifact in artifacts] == [1, 2]


# ---------------------------------------------------------------------------
# map-reduce
# ---------------------------------------------------------------------------


def test_long_document_uses_map_reduce_and_merges_in_order(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    # Twelve entries make the normalized document clearly exceed the tiny
    # threshold, so the summarize run must switch to map-reduce.
    export_id = _create_export(session, n_text=12)
    _run_normalization(session_factory, export_id)

    monkeypatch.setenv("APP_SUMMARIZE_MAPREDUCE_THRESHOLD", "1")
    monkeypatch.setenv("APP_SUMMARIZE_CHUNK_SIZE", "150")
    reload_settings()
    provider = FakeSummarizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True
    job_id = _run_summarization(session_factory, export_id)

    assert provider.calls >= 2  # map-reduce: several deterministic calls
    blocks = [extract_mode(request.messages[-1].content) for request in provider.requests]
    assert all(mode == "partial" for mode, _index, _total in blocks)
    total = provider.calls
    assert [(_mode, index, seen_total) for _mode, index, seen_total in blocks] == [
        ("partial", position, total) for position in range(1, total + 1)
    ]

    with session_factory() as s:
        runs = _run_rows(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.succeeded
        metadata = runs[0].metadata_json or {}
        assert metadata["call_count"] == total
        assert isinstance(metadata["input_sha256"], list) and len(metadata["input_sha256"]) == total
        dumped = json.dumps(metadata)
        assert "Синтетическое событие" not in dumped  # partials never persisted

    files = _summary_files()
    assert [p.name for p in files] == ["2026-09-13_summary_0001_r1_v1.md"]
    written = files[0].read_text(encoding="utf-8")
    # Every partial's key event reached the final document, in block order.
    positions = [
        written.index(f"- Синтетическое событие блока {position}.")
        for position in range(1, total + 1)
    ]
    assert positions == sorted(positions)
    # The last partial turned the wellbeing section on; the merge kept it.
    assert "## Самочувствие и эмоции" in written
    assert f"Явная усталость в конце недели (блока {total})." in written
    assert "Версия: v1" in written
    assert "Источник (нормализованный): 2026-09-13_normalized_0001_r1_v1.txt" in written


def test_wellbeing_rendering_when_explicitly_supported(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)
    _run_normalization(session_factory, export_id)

    def _with_wellbeing(mode: str, index: int, total: int) -> dict[str, object]:
        payload = default_summary_payload(mode, index, total)
        payload["wellbeing"] = {
            "explicit_support": True,
            "points": ["В среду было хорошее настроение."],
        }
        return payload

    provider = FakeSummarizeLLM(responder=_with_wellbeing)
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True
    _run_summarization(session_factory, export_id)

    written = _summary_files()[0].read_text(encoding="utf-8")
    assert "- В среду было хорошее настроение." in written
    assert "Явных данных о самочувствии и эмоциях в источнике нет." not in written


# ---------------------------------------------------------------------------
# deterministic failures
# ---------------------------------------------------------------------------


def test_malformed_json_fails_without_writing_a_file(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)
    _run_normalization(session_factory, export_id)

    provider = FakeLLMProvider(response=LLMResponse(text="это не JSON вовсе", model="fake"))
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True
    job_id = _run_summarization(session_factory, export_id)
    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.failed
        assert stored.attempts == 1
        runs = _run_rows(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.failed
        assert "structured validation" in (runs[0].error_summary or "")
        assert (
            ArtifactRepository(s).list_by_export_and_layer(export_id, ArtifactLayer.summary) == []
        )
    assert _summary_files() == []


def test_schema_invalid_json_fails_without_writing_a_file(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)
    _run_normalization(session_factory, export_id)

    provider = FakeLLMProvider(response=json_llm_response({"unexpected": "shape"}))
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True
    job_id = _run_summarization(session_factory, export_id)
    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.failed
        runs = _run_rows(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.failed
        assert "structured validation" in (runs[0].error_summary or "")
        assert (
            ArtifactRepository(s).list_by_export_and_layer(export_id, ArtifactLayer.summary) == []
        )
    assert _summary_files() == []


def test_wellbeing_points_without_explicit_support_fail_the_run(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    export_id = _create_export(session, n_text=1)
    _run_normalization(session_factory, export_id)

    provider = FakeSummarizeLLM(
        responder=lambda _mode, _index, _total: summary_payload(
            key_events=["Событие."],
            wellbeing_explicit=False,
            wellbeing_points=["Выдуманная эмоция."],
        )
    )
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True
    job_id = _run_summarization(session_factory, export_id)
    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.failed
        runs = _run_rows(s, job_id)
        assert "structured validation" in (runs[0].error_summary or "")
    assert _summary_files() == []


def test_missing_normalized_input_fails_without_calling_the_provider(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer 3 requires layer 2: an export without a normalized artifact must
    fail permanently and the LLM provider must never be called (the original
    export file is never read for summarization)."""
    export_id = _create_export(session, n_text=1)
    provider = FakeSummarizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True

    job_id = _run_summarization(session_factory, export_id)
    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.failed
        assert stored.attempts == 1  # deterministic failure, no retries burned
        runs = _run_rows(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.failed
        assert "no normalized artifact yet" in (runs[0].error_summary or "")
        assert runs[0].input_artifact_id is None
        assert ArtifactRepository(s).list_by_export(export_id) == []
    assert provider.calls == 0  # input resolution failed before any request
    assert provider.requests == []
    assert _summary_files() == []


# ---------------------------------------------------------------------------
# batch eligibility
# ---------------------------------------------------------------------------


def test_batch_eligibility_and_enqueueing(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    eligible_id = _create_export(session, sunday=date(2026, 8, 30), n_text=1)
    already_summarized_id = _create_export(session, sunday=date(2026, 9, 6), n_text=2)
    _run_normalization(session_factory, eligible_id)
    _run_normalization(session_factory, already_summarized_id)

    provider = FakeSummarizeLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_summarize_pipeline_registered() is True
    # Give the second export a summary so only the first stays eligible.
    _run_summarization(session_factory, already_summarized_id)
    with session_factory() as s:
        eligible = list_eligible_exports(s)
        assert [export.id for export in eligible] == [eligible_id]
        jobs = enqueue_summarize_batch(s)
        assert len(jobs) == 1

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as s:
        summaries = ArtifactRepository(s).list_by_export_and_layer(
            eligible_id, ArtifactLayer.summary
        )
        assert len(summaries) == 1
        # The already-summarized export keeps exactly the single summary that
        # was created before the batch (the batch did not touch it again).
        already_summaries = ArtifactRepository(s).list_by_export_and_layer(
            already_summarized_id, ArtifactLayer.summary
        )
        assert len(already_summaries) == 1
        assert already_summaries[0].version == 1
        # After the succeeded run the export is no longer eligible.
        assert list_eligible_exports(s) == []


def test_batch_skips_exports_without_normalized_input(session: Session) -> None:
    without_normalized = _create_export(session, sunday=date(2026, 8, 30), n_text=1)
    assert list_eligible_exports(session) == []
    # Once a normalized artifact exists (row + file), the export is eligible.
    normalized_dir = get_settings().exports_dir / "02_normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    from diary.domain.pipelines.naming import artifact_data_root_relative_path, derived_filename

    filename = derived_filename(
        sunday=date(2026, 8, 30),
        layer=ArtifactLayer.normalized,
        seq_number=1,
        revision=1,
        version=1,
    )
    relative = artifact_data_root_relative_path(layer=ArtifactLayer.normalized, filename=filename)
    target = get_settings().data_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# Дневник — нормализованный\n---\n[2026-08-24 09:00] TEXT\nТекст.", "utf-8")
    ArtifactRepository(session).create(
        layer=ArtifactLayer.normalized,
        export_id=without_normalized,
        path=relative,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        version=1,
    )
    session.commit()
    eligible = list_eligible_exports(session)
    assert [export.id for export in eligible] == [without_normalized]


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_ensure_registration_is_guarded_and_idempotent() -> None:
    assert pipeline_registry.get_pipeline(PipelineType.summarize) is None
    assert ensure_summarize_pipeline_registered() is True
    registered = pipeline_registry.get_pipeline(PipelineType.summarize)
    assert isinstance(registered, SummarizePipeline)
    assert ensure_summarize_pipeline_registered() is False
    assert pipeline_registry.get_pipeline(PipelineType.summarize) is registered
