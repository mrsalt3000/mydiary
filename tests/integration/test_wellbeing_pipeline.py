"""Integration tests for the real wellbeing pipeline (Task 15).

End-to-end through the worker helpers and repository code against an
isolated, migrated SQLite test database and a temporary data root. The real
:class:`~diary.domain.pipelines.wellbeing.pipeline.WellbeingPipeline` is
registered explicitly per test and driven with a fake LLM provider (see
``_wellbeing_fakes``); normalized inputs are written as real artifact rows +
files, so no network, YAML file or real model is ever touched.

Covers: a successful single-pass run (layer-4 artifact/file, lineage header
referencing the exact normalized source, untouched input), rerun → v2 with v1
intact, deterministic safety levels and their rendering (normal /
needs_support with the support note / urgent with the fixed escalation block
that the LLM payload cannot remove), the mandatory disclaimer in every
report, malformed and schema-invalid output, an export without a normalized
artifact (permanent failure, provider never called), and the guarded
registration function.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from _pipeline_fakes import FakeLLMProvider, install_fake_llm
from _wellbeing_fakes import FakeWellbeingLLM, wellbeing_payload
from sqlalchemy.orm import Session

from diary.core.settings import get_settings, reload_settings
from diary.db.models import (
    ArtifactLayer,
    JobStatus,
    MessageKind,
    PipelineType,
    RunStatus,
    WeeklyExport,
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
from diary.domain.pipelines.naming import artifact_data_root_relative_path, derived_filename
from diary.domain.pipelines.wellbeing.pipeline import WellbeingPipeline
from diary.domain.pipelines.wellbeing.registration import (
    ensure_wellbeing_pipeline_registered,
)
from diary.domain.pipelines.wellbeing.safety import (
    DISCLAIMER,
    ESCALATION_TITLE,
    SUPPORT_NOTE,
)
from diary.domain.pipelines.wellbeing.service import enqueue_wellbeing_run
from diary.workers.runner import process_due_jobs

SUNDAY = date(2026, 9, 13)
DIARY_PHRASE = "Синтетика: приватный текст для wellbeing."
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


def _create_export(
    session: Session, *, sunday: date = SUNDAY, body: str | None = None
) -> WeeklyExport:
    """Seed a week with one synthetic text message and create the original."""
    body = body or f"{DIARY_PHRASE} Запись 0."
    MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=_message_time(sunday, 0)
    )
    session.commit()
    return export_week(session, sunday_date=sunday)


def _normalized_entry_text(body: str, *, stamp: str = "[2026-09-07 09:00]") -> str:
    """A parseable layer-2 document body for the helpers below."""
    return (
        "# Дневник — нормализованный\n"
        "Дата воскресенья: 2026-09-13\n"
        "Номер выгрузки: 0001\n"
        "---\n"
        f"{stamp} TEXT\n"
        f"{body}\n"
    )


def _add_normalized_artifact(
    session: Session,
    export: WeeklyExport,
    text: str,
) -> uuid.UUID:
    """Write a real normalized (layer 2) artifact row + file for ``export``."""
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
    artifact = ArtifactRepository(session).create(
        layer=ArtifactLayer.normalized,
        export_id=export.id,
        path=relative,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        version=1,
    )
    session.commit()
    return artifact.id


def _wellbeing_files() -> list[Path]:
    directory = get_settings().exports_dir / "04_wellbeing"
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _run_wellbeing(session_factory, export_id: uuid.UUID) -> uuid.UUID:
    """Enqueue (confirmed) + process one wellbeing job; return the job id."""
    with session_factory() as s:
        job = enqueue_wellbeing_run(s, export_id=export_id, confirmed=True)
        job_id = job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    return job_id


def _run_rows(session: Session, job_id: uuid.UUID):
    return ProcessingRunRepository(session).list_by_job_id(job_id)


# ---------------------------------------------------------------------------
# successful single-pass run
# ---------------------------------------------------------------------------


def test_single_pass_writes_report_from_normalized_input(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _create_export(session)
    export_id = export.id
    normalized_id = _add_normalized_artifact(
        session, export, _normalized_entry_text("Спокойная неделя, работал в обычном режиме.")
    )
    provider = FakeWellbeingLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True

    job_id = _run_wellbeing(session_factory, export_id)
    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.succeeded
        runs = _run_rows(s, job_id)
        assert len(runs) == 1
        run = runs[0]
        assert run.status is RunStatus.succeeded
        assert run.pipeline_type is PipelineType.wellbeing
        assert run.input_export_id == export_id
        # The layer-4 run consumed the normalized artifact by database id.
        assert run.input_artifact_id == normalized_id
        assert run.profile_name == "wellbeing_analysis"
        assert run.model_name == "fake-wellbeing-model"
        assert run.prompt_version == "v1"
        metadata = run.metadata_json or {}
        dumped = json.dumps(metadata)
        assert DIARY_PHRASE not in dumped  # no diary content in run metadata
        assert "Синтетический факт" not in dumped

        artifacts = ArtifactRepository(s).list_by_export(export_id)
        assert len(artifacts) == 2  # normalized v1 + wellbeing v1
        report_artifact = artifacts[-1]
        assert report_artifact.layer.value == "wellbeing"
        assert report_artifact.version == 1
        assert report_artifact.processing_run_id == run.id
        assert report_artifact.path == "exports/04_wellbeing/2026-09-13_wellbeing_0001_r1_v1.md"

    files = _wellbeing_files()
    assert [path.name for path in files] == ["2026-09-13_wellbeing_0001_r1_v1.md"]
    written = files[0].read_text(encoding="utf-8")
    assert "# Wellbeing-рефлексия" in written
    assert "Неделя: 2026-09-07 — 2026-09-13" in written
    assert "Номер выгрузки: 0001" in written
    assert "Ревизия: r1" in written
    assert "Версия: v1" in written
    assert "Источник (нормализованный): 2026-09-13_normalized_0001_r1_v1.txt" in written
    assert "Уровень: normal" in written
    assert DISCLAIMER in written
    assert "## Наблюдаемые факты" in written
    assert "- [2026-09-07] Синтетический факт:" in written
    assert "## Рекомендации" in written
    assert ESCALATION_TITLE not in written
    digest = hashlib.sha256(files[0].read_bytes()).hexdigest()
    with session_factory() as s:
        reports = ArtifactRepository(s).list_by_export_and_layer(export_id, ArtifactLayer.wellbeing)
        assert reports[0].sha256 == digest
    assert provider.calls == 1  # single-pass


def test_rerun_creates_v2_and_keeps_v1_intact(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _create_export(session)
    export_id = export.id
    _add_normalized_artifact(session, export, _normalized_entry_text("Обычная неделя."))
    provider = FakeWellbeingLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True

    first_job = _run_wellbeing(session_factory, export_id)
    files = _wellbeing_files()
    assert [path.name for path in files] == ["2026-09-13_wellbeing_0001_r1_v1.md"]
    v1_bytes = files[0].read_bytes()

    second_job = _run_wellbeing(session_factory, export_id)
    assert second_job != first_job
    files = _wellbeing_files()
    assert [path.name for path in files] == [
        "2026-09-13_wellbeing_0001_r1_v1.md",
        "2026-09-13_wellbeing_0001_r1_v2.md",
    ]
    assert files[0].read_bytes() == v1_bytes
    assert "Версия: v2" in files[1].read_text(encoding="utf-8")
    with session_factory() as s:
        reports = ArtifactRepository(s).list_by_export_and_layer(export_id, ArtifactLayer.wellbeing)
        assert [artifact.version for artifact in reports] == [1, 2]


# ---------------------------------------------------------------------------
# deterministic safety levels drive the render
# ---------------------------------------------------------------------------


def test_needs_support_level_renders_support_note(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _create_export(session)
    export_id = export.id
    _add_normalized_artifact(
        session, export, _normalized_entry_text("Не сплю, постоянная тревога и нет сил.")
    )
    provider = FakeWellbeingLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True

    _run_wellbeing(session_factory, export_id)
    written = _wellbeing_files()[0].read_text(encoding="utf-8")
    assert "Уровень: needs_support" in written
    assert SUPPORT_NOTE in written
    assert ESCALATION_TITLE not in written
    assert DISCLAIMER in written


def test_urgent_input_gets_fixed_escalation_regardless_of_llm_payload(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deterministic classifier (not the LLM) decides ``urgent``: even a
    fake LLM that would claim a perfectly calm week cannot remove the fixed
    escalation content from the report (ADR-006)."""
    export = _create_export(session)
    export_id = export.id
    _add_normalized_artifact(
        session,
        export,
        _normalized_entry_text("Сегодня снова думала, что не хочу больше жить."),
    )
    provider = FakeWellbeingLLM()  # default payload claims a neutral week
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True

    _run_wellbeing(session_factory, export_id)
    written = _wellbeing_files()[0].read_text(encoding="utf-8")
    assert "Уровень: urgent" in written
    assert ESCALATION_TITLE in written
    for line in (
        "не заменяет специалиста",
        "в службу экстренной помощи по месту жительства",
    ):
        assert line in written
    assert DISCLAIMER in written
    # The report still reflects the LLM data, but the crisis content is code.
    assert "- [2026-09-07] Синтетический факт:" in written


def test_every_report_contains_the_disclaimer(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mandatory disclaimer is code-rendered into EVERY report regardless
    of the input level (tasklist Task 15)."""
    export = _create_export(session)
    export_id = export.id
    _add_normalized_artifact(session, export, _normalized_entry_text("Спокойная неделя."))
    provider = FakeWellbeingLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True
    _run_wellbeing(session_factory, export_id)
    assert DISCLAIMER in _wellbeing_files()[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# deterministic failures
# ---------------------------------------------------------------------------


def test_malformed_json_fails_without_writing_a_file(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _create_export(session)
    export_id = export.id
    _add_normalized_artifact(session, export, _normalized_entry_text("Спокойная неделя."))
    provider = FakeLLMProvider(response=LLMResponse(text="это не JSON вовсе", model="fake"))
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True

    job_id = _run_wellbeing(session_factory, export_id)
    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.failed
        assert stored.attempts == 1
        runs = _run_rows(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.failed
        assert "structured validation" in (runs[0].error_summary or "")
        assert (
            ArtifactRepository(s).list_by_export_and_layer(export_id, ArtifactLayer.wellbeing) == []
        )
    assert _wellbeing_files() == []


def test_undated_observed_fact_fails_the_run(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fact the model cannot anchor to a dated record is not written."""
    export = _create_export(session)
    export_id = export.id
    _add_normalized_artifact(session, export, _normalized_entry_text("Спокойная неделя."))
    provider = FakeWellbeingLLM(
        responder=lambda _text: wellbeing_payload(observed_facts=["Автор всю неделю плохо спал."])
    )
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True

    job_id = _run_wellbeing(session_factory, export_id)
    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.failed
        runs = _run_rows(s, job_id)
        assert "structured validation" in (runs[0].error_summary or "")
    assert _wellbeing_files() == []


def test_missing_normalized_input_fails_without_calling_the_provider(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer 4 requires layer 2: an export without a normalized artifact must
    fail permanently and the LLM provider must never be called (the original
    export file is never read for wellbeing). Enqueued through the generic
    pipeline service to exercise the run-time guard."""
    export = _create_export(session)
    export_id = export.id
    original_path = get_settings().data_root / export.path
    original_bytes = original_path.read_bytes()
    provider = FakeWellbeingLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True

    with session_factory() as s:
        job = pipeline_service.enqueue_pipeline_run(
            s, pipeline_type=PipelineType.wellbeing, export_id=export_id
        )
        job_id = job.id
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as s:
        stored = JobRepository(s).get(job_id)
        assert stored is not None and stored.status is JobStatus.failed
        assert stored.attempts == 1  # deterministic failure, no retries burned
        runs = _run_rows(s, job_id)
        assert len(runs) == 1 and runs[0].status is RunStatus.failed
        assert "no normalized artifact yet" in (runs[0].error_summary or "")
        assert runs[0].input_artifact_id is None
        assert ArtifactRepository(s).list_by_export(export_id) == []
    assert provider.calls == 0
    assert _wellbeing_files() == []
    # The original export file is untouched by the failed attempt.
    assert original_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# input is layer 2, never the layer-1 original
# ---------------------------------------------------------------------------


def test_input_is_the_normalized_document_not_the_original(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request must carry only the normalized text; a token present only in
    the original export must never reach the model, and the original file must
    stay byte-for-byte unchanged."""
    original_token = "ОРИГИНАЛ-СЕКРЕТ-ТОЛЬКО-ЗДЕСЬ"
    normalized_token = "НОРМАЛИЗОВАННЫЙ-ТЕКСТ-ВИДИТ-МОДЕЛЬ"
    export = _create_export(session, body=f"Запись с {original_token}.")
    export_id = export.id
    original_path = get_settings().data_root / export.path
    original_bytes = original_path.read_bytes()
    assert original_token.encode() in original_bytes

    _add_normalized_artifact(
        session,
        export,
        _normalized_entry_text(f"Спокойная неделя. {normalized_token}"),
    )
    provider = FakeWellbeingLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True

    _run_wellbeing(session_factory, export_id)
    assert len(provider.requests) == 1
    request_content = provider.requests[0].messages[-1].content
    assert normalized_token in request_content
    assert original_token not in request_content
    assert original_path.read_bytes() == original_bytes
    assert _wellbeing_files()[0].exists()


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_ensure_registration_is_guarded_and_idempotent() -> None:
    assert pipeline_registry.get_pipeline(PipelineType.wellbeing) is None
    assert ensure_wellbeing_pipeline_registered() is True
    registered = pipeline_registry.get_pipeline(PipelineType.wellbeing)
    assert isinstance(registered, WellbeingPipeline)
    assert ensure_wellbeing_pipeline_registered() is False
    assert pipeline_registry.get_pipeline(PipelineType.wellbeing) is registered
