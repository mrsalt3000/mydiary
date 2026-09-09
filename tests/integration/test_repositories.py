"""Integration tests for repositories and schema constraints (Task 04).

Uses only synthetic data against an isolated, migrated test database.
Covers: create/read through repositories, UTC timestamp round-trips, unique
constraints (weekly export week+revision), foreign-key behaviour, and DB-level
CHECK constraints on enum columns.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from diary.db.models import (
    ArtifactLayer,
    ExportStatus,
    JobStatus,
    JobType,
    MessageKind,
    PipelineType,
    RunStatus,
    TranscriptionStatus,
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


def _iso(day: int, hour: int = 12) -> datetime:
    return datetime(2026, 9, day, hour, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


def test_message_roundtrip_preserves_values_and_utc(session: Session) -> None:
    repository = MessageRepository(session)
    message = repository.create(
        kind=MessageKind.text,
        body_text="Синтетическая запись для теста.",
        created_at=_iso(1),
    )
    session.commit()
    session.expunge_all()

    fetched = repository.get(message.id)
    assert fetched is not None
    assert fetched.kind is MessageKind.text
    assert fetched.body_text == "Синтетическая запись для теста."
    assert fetched.created_at == _iso(1)
    assert fetched.created_at.tzinfo is timezone.utc


def test_message_created_at_defaults_to_utc_now(session: Session) -> None:
    repository = MessageRepository(session)
    message = repository.create(kind=MessageKind.text, body_text="Синтетика")
    session.commit()
    assert message.created_at.tzinfo is timezone.utc


def test_message_listing_is_chronological(session: Session) -> None:
    repository = MessageRepository(session)
    first = repository.create(kind=MessageKind.text, body_text="Первая", created_at=_iso(1, 9))
    second = repository.create(kind=MessageKind.audio, body_text="", created_at=_iso(1, 11))
    third = repository.create(kind=MessageKind.text, body_text="Третья", created_at=_iso(2, 8))
    session.commit()

    assert [m.id for m in repository.list_messages()] == [first.id, second.id, third.id]
    text_only = repository.list_messages(kind=MessageKind.text)
    assert [m.id for m in text_only] == [first.id, third.id]
    before = repository.list_messages(created_before=_iso(2, 0))
    assert [m.id for m in before] == [first.id, second.id]
    limited = repository.list_messages(limit=2)
    assert [m.id for m in limited] == [first.id, second.id]


def test_message_kind_is_check_constrained(session: Session) -> None:
    statement = text(
        "INSERT INTO messages (id, kind, body_text, created_at) "
        "VALUES (:id, :kind, :body, :created)"
    )
    with pytest.raises(IntegrityError):
        session.execute(
            statement,
            {
                "id": uuid.uuid4().hex,
                "kind": "telegram",
                "body": "недопустимо",
                "created": "2026-09-01 00:00:00.000000",
            },
        )
    session.rollback()


# ---------------------------------------------------------------------------
# audio_files / transcriptions
# ---------------------------------------------------------------------------


def test_audio_and_transcription_lineage(session: Session) -> None:
    messages = MessageRepository(session)
    audio = AudioFileRepository(session)
    transcriptions = TranscriptionRepository(session)

    message = messages.create(kind=MessageKind.audio, body_text="")
    audio_file = audio.create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path="audio/2026/09/synthetic-id.webm",
        size_bytes=2048,
        sha256="a" * 64,
        duration_seconds=3.25,
    )
    first = transcriptions.create(
        audio_file_id=audio_file.id,
        raw_text="Синтетический распознанный текст.",
        status=TranscriptionStatus.succeeded,
        profile_name="transcription",
        provider_name="faster_whisper",
        model_name="small",
        language="ru",
        metadata_json={"device": "cpu", "compute_type": "int8"},
        created_at=_iso(1, 0),
    )
    second = transcriptions.create(
        audio_file_id=audio_file.id,
        raw_text="Повторная синтетическая расшифровка.",
        status=TranscriptionStatus.succeeded,
        created_at=_iso(2, 0),
    )
    session.commit()
    session.expunge_all()

    assert audio.get(audio_file.id) is not None
    fetched_audio = audio.get_by_message(message.id)
    assert [f.id for f in fetched_audio] == [audio_file.id]
    history = transcriptions.list_by_audio_file(audio_file.id)
    assert [t.id for t in history] == [first.id, second.id]
    assert history[0].metadata_json == {"device": "cpu", "compute_type": "int8"}


def test_audio_file_foreign_key_is_enforced(session: Session) -> None:
    repository = AudioFileRepository(session)
    missing_message_id = uuid.uuid4()
    with pytest.raises(IntegrityError):
        repository.create(
            message_id=missing_message_id,
            original_mime="audio/webm",
            storage_path="audio/2026/09/does-not-exist.webm",
            size_bytes=1,
            sha256="b" * 64,
        )
    session.rollback()


# ---------------------------------------------------------------------------
# weekly_exports
# ---------------------------------------------------------------------------


def test_weekly_export_unique_week_and_revision(session: Session) -> None:
    repository = WeeklyExportRepository(session)
    first = repository.create(
        week_end_date=date(2026, 9, 13),
        seq_number=1,
        revision=1,
        path="exports/01_original/2026-09-13_original_0001_r1.txt",
        sha256="c" * 64,
    )
    first_id = first.id
    session.commit()

    with pytest.raises(IntegrityError):
        repository.create(
            week_end_date=date(2026, 9, 13),
            seq_number=1,
            revision=1,
            path="exports/01_original/2026-09-13_original_0001_r1.txt",
            sha256="c" * 64,
        )
    session.rollback()

    # A new revision of the same week (explicit re-export) is allowed.
    revision_two = repository.create(
        week_end_date=date(2026, 9, 13),
        seq_number=1,
        revision=2,
        path="exports/01_original/2026-09-13_original_0001_r2.txt",
        sha256="d" * 64,
    )
    revision_two_id = revision_two.id
    session.commit()
    session.expunge_all()

    # Re-fetch by stable ids captured before the rollback expired old instances.
    assert repository.get(first_id) is not None
    revision_rows = repository.list_by_week(date(2026, 9, 13))
    assert [e.id for e in revision_rows] == [first_id, revision_two_id]
    assert repository.get_by_week_and_revision(date(2026, 9, 13), 2).id == revision_two_id


def test_weekly_export_sequence_ordering_and_max(session: Session) -> None:
    repository = WeeklyExportRepository(session)
    for seq, week_end in enumerate((date(2026, 9, 6), date(2026, 9, 13)), start=1):
        repository.create(
            week_end_date=week_end,
            seq_number=seq,
            revision=1,
            path=f"exports/01_original/export-{seq}.txt",
            sha256=f"{seq:x}" * 64,
        )
    session.commit()

    exports = repository.list_all_by_sequence()
    assert [e.seq_number for e in exports] == [1, 2]
    assert repository.max_sequence_number() == 2
    assert repository.get(exports[0].id).status is ExportStatus.created


# ---------------------------------------------------------------------------
# processing_runs / artifacts (lineage)
# ---------------------------------------------------------------------------


def test_processing_run_and_artifact_lineage(session: Session) -> None:
    exports = WeeklyExportRepository(session)
    runs = ProcessingRunRepository(session)
    artifacts = ArtifactRepository(session)

    export = exports.create(
        week_end_date=date(2026, 9, 13),
        seq_number=1,
        revision=1,
        path="exports/01_original/2026-09-13_original_0001_r1.txt",
        sha256="e" * 64,
    )
    run = runs.create(
        pipeline_type=PipelineType.normalize,
        status=RunStatus.succeeded,
        input_export_id=export.id,
        profile_name="normalization",
        model_name="local-normalizer",
        prompt_version="1",
        metadata_json={"temperature": 0.0},
    )
    normalized_v1 = artifacts.create(
        layer=ArtifactLayer.normalized,
        export_id=export.id,
        processing_run_id=run.id,
        version=1,
        path="exports/02_normalized/2026-09-13_normalized_0001_r1_v1.txt",
        sha256="f" * 64,
    )
    normalized_v2 = artifacts.create(
        layer=ArtifactLayer.normalized,
        export_id=export.id,
        processing_run_id=run.id,
        version=2,
        path="exports/02_normalized/2026-09-13_normalized_0001_r1_v2.txt",
        sha256="0a" * 32,
    )
    session.commit()
    session.expunge_all()

    assert runs.get(run.id).prompt_version == "1"
    runs_for_export = runs.list_by_input_export(export.id)
    assert [r.id for r in runs_for_export] == [run.id]
    assert [a.id for a in artifacts.list_by_export(export.id)] == [
        normalized_v1.id,
        normalized_v2.id,
    ]
    versions = artifacts.list_by_export_and_layer(export.id, ArtifactLayer.normalized)
    assert [a.version for a in versions] == [1, 2]
    assert [a.id for a in artifacts.list_by_run(run.id)] == [
        normalized_v1.id,
        normalized_v2.id,
    ]


def test_processing_run_status_is_check_constrained(session: Session) -> None:
    statement = text(
        "INSERT INTO processing_runs (id, pipeline_type, status, created_at) "
        "VALUES (:id, :pipeline_type, :status, :created)"
    )
    with pytest.raises(IntegrityError):
        session.execute(
            statement,
            {
                "id": uuid.uuid4().hex,
                "pipeline_type": "normalize",
                "status": "exploded",
                "created": "2026-09-01 00:00:00.000000",
            },
        )
    session.rollback()


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def test_job_repository_roundtrip_and_filters(session: Session) -> None:
    repository = JobRepository(session)
    first = repository.create(
        job_type=JobType.transcribe_audio,
        payload_json={"audio_file_id": uuid.uuid4().hex},
        idempotency_key="transcribe-synthetic-1",
        max_attempts=3,
    )
    second = repository.create(
        job_type=JobType.normalize,
        status=JobStatus.running,
        payload_json={"export_id": uuid.uuid4().hex},
    )
    session.commit()

    assert first.status is JobStatus.pending
    assert first.attempts == 0
    assert repository.get(first.id).id == first.id
    assert repository.get_by_idempotency_key("transcribe-synthetic-1").id == first.id
    pending = repository.list_by_status(JobStatus.pending)
    assert [j.id for j in pending] == [first.id]
    running = repository.list_by_status(JobStatus.running)
    assert [j.id for j in running] == [second.id]


def test_job_status_is_check_constrained(session: Session) -> None:
    statement = text(
        "INSERT INTO jobs "
        "(id, job_type, status, attempts, created_at, updated_at) "
        "VALUES (:id, :job_type, :status, :attempts, :created, :updated)"
    )
    with pytest.raises(IntegrityError):
        session.execute(
            statement,
            {
                "id": uuid.uuid4().hex,
                "job_type": "transcribe_audio",
                "status": "zombie",
                "attempts": 0,
                "created": "2026-09-01 00:00:00.000000",
                "updated": "2026-09-01 00:00:00.000000",
            },
        )
    session.rollback()
