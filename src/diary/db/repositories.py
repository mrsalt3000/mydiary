"""Repository layer for basic persistence of the core entities (Task 04).

Each repository is bound to a single SQLAlchemy ``Session`` and exposes plain
create/read operations plus small list queries by key filters — enough for
integration tests and for the service layer of later tasks. Repositories
contain no business logic: they flush on create (so constraint/FK errors
surface immediately) but never commit — the caller owns the transaction and
commits atomically at the end of its unit of work.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.db.base import utc_now
from diary.db.models import (
    Artifact,
    ArtifactLayer,
    AudioFile,
    ExportStatus,
    Job,
    JobStatus,
    JobType,
    Message,
    MessageKind,
    PipelineType,
    ProcessingRun,
    RunStatus,
    Transcription,
    TranscriptionStatus,
    WeeklyExport,
)

__all__ = [
    "ArtifactRepository",
    "AudioFileRepository",
    "JobRepository",
    "MessageRepository",
    "ProcessingRunRepository",
    "TranscriptionRepository",
    "WeeklyExportRepository",
]


class MessageRepository:
    """Persistence for diary :class:`Message` records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        kind: MessageKind,
        body_text: str = "",
        created_at: datetime | None = None,
    ) -> Message:
        message = Message(kind=kind, body_text=body_text, created_at=created_at or utc_now())
        self._session.add(message)
        self._session.flush()
        return message

    def get(self, message_id: uuid.UUID) -> Message | None:
        return self._session.get(Message, message_id)

    def list_messages(
        self,
        *,
        kind: MessageKind | None = None,
        created_before: datetime | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Messages in chronological order (oldest first, ties by id)."""
        statement = select(Message).order_by(Message.created_at.asc(), Message.id.asc())
        if kind is not None:
            statement = statement.where(Message.kind == kind)
        if created_before is not None:
            statement = statement.where(Message.created_at < created_before)
        if limit is not None:
            statement = statement.limit(limit)
        return list(self._session.scalars(statement))


class AudioFileRepository:
    """Persistence for :class:`AudioFile` metadata records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        message_id: uuid.UUID,
        original_mime: str,
        storage_path: str,
        size_bytes: int,
        sha256: str,
        duration_seconds: float | None = None,
        created_at: datetime | None = None,
    ) -> AudioFile:
        audio_file = AudioFile(
            message_id=message_id,
            original_mime=original_mime,
            storage_path=storage_path,
            size_bytes=size_bytes,
            sha256=sha256,
            duration_seconds=duration_seconds,
            created_at=created_at or utc_now(),
        )
        self._session.add(audio_file)
        self._session.flush()
        return audio_file

    def get(self, audio_file_id: uuid.UUID) -> AudioFile | None:
        return self._session.get(AudioFile, audio_file_id)

    def get_by_message(self, message_id: uuid.UUID) -> list[AudioFile]:
        statement = (
            select(AudioFile)
            .where(AudioFile.message_id == message_id)
            .order_by(AudioFile.created_at.asc(), AudioFile.id.asc())
        )
        return list(self._session.scalars(statement))


class TranscriptionRepository:
    """Persistence for raw STT results (:class:`Transcription`)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        audio_file_id: uuid.UUID,
        raw_text: str,
        status: TranscriptionStatus,
        profile_name: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        language: str | None = None,
        metadata_json: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> Transcription:
        transcription = Transcription(
            audio_file_id=audio_file_id,
            raw_text=raw_text,
            status=status,
            profile_name=profile_name,
            provider_name=provider_name,
            model_name=model_name,
            language=language,
            metadata_json=metadata_json,
            created_at=created_at or utc_now(),
        )
        self._session.add(transcription)
        self._session.flush()
        return transcription

    def get(self, transcription_id: uuid.UUID) -> Transcription | None:
        return self._session.get(Transcription, transcription_id)

    def list_by_audio_file(self, audio_file_id: uuid.UUID) -> list[Transcription]:
        """Transcription history for one audio file, oldest first."""
        statement = (
            select(Transcription)
            .where(Transcription.audio_file_id == audio_file_id)
            .order_by(Transcription.created_at.asc(), Transcription.id.asc())
        )
        return list(self._session.scalars(statement))


class WeeklyExportRepository:
    """Persistence for original weekly exports (:class:`WeeklyExport`).

    Sequence-number assignment (global, no gaps) is export business logic
    (Task 10); this repository only reads the current maximum, so callers can
    compute ``max + 1`` inside their own transaction.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        week_end_date: date,
        seq_number: int,
        revision: int,
        path: str,
        sha256: str,
        status: ExportStatus = ExportStatus.created,
        created_at: datetime | None = None,
    ) -> WeeklyExport:
        export = WeeklyExport(
            week_end_date=week_end_date,
            seq_number=seq_number,
            revision=revision,
            status=status,
            path=path,
            sha256=sha256,
            created_at=created_at or utc_now(),
        )
        self._session.add(export)
        self._session.flush()
        return export

    def get(self, export_id: uuid.UUID) -> WeeklyExport | None:
        return self._session.get(WeeklyExport, export_id)

    def get_by_week_and_revision(self, week_end_date: date, revision: int) -> WeeklyExport | None:
        statement = select(WeeklyExport).where(
            WeeklyExport.week_end_date == week_end_date,
            WeeklyExport.revision == revision,
        )
        return self._session.scalar(statement)

    def list_by_week(self, week_end_date: date) -> list[WeeklyExport]:
        statement = (
            select(WeeklyExport)
            .where(WeeklyExport.week_end_date == week_end_date)
            .order_by(WeeklyExport.revision.asc())
        )
        return list(self._session.scalars(statement))

    def list_all_by_sequence(self) -> list[WeeklyExport]:
        """All exports ordered by sequence number (stable tie-break by id)."""
        statement = select(WeeklyExport).order_by(
            WeeklyExport.seq_number.asc(), WeeklyExport.id.asc()
        )
        return list(self._session.scalars(statement))

    def max_sequence_number(self) -> int | None:
        """Highest assigned ``seq_number`` (``None`` when no export exists yet)."""
        return self._session.scalar(select(func.max(WeeklyExport.seq_number)))


class ProcessingRunRepository:
    """Persistence for pipeline :class:`ProcessingRun` records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        pipeline_type: PipelineType,
        status: RunStatus = RunStatus.queued,
        input_export_id: uuid.UUID | None = None,
        input_artifact_id: uuid.UUID | None = None,
        profile_name: str | None = None,
        model_name: str | None = None,
        prompt_version: str | None = None,
        error_summary: str | None = None,
        metadata_json: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> ProcessingRun:
        run = ProcessingRun(
            pipeline_type=pipeline_type,
            status=status,
            input_export_id=input_export_id,
            input_artifact_id=input_artifact_id,
            profile_name=profile_name,
            model_name=model_name,
            prompt_version=prompt_version,
            error_summary=error_summary,
            metadata_json=metadata_json,
            created_at=created_at or utc_now(),
        )
        self._session.add(run)
        self._session.flush()
        return run

    def get(self, run_id: uuid.UUID) -> ProcessingRun | None:
        return self._session.get(ProcessingRun, run_id)

    def list_by_status(self, status: RunStatus) -> list[ProcessingRun]:
        statement = (
            select(ProcessingRun)
            .where(ProcessingRun.status == status)
            .order_by(ProcessingRun.created_at.asc(), ProcessingRun.id.asc())
        )
        return list(self._session.scalars(statement))

    def list_by_input_export(self, export_id: uuid.UUID) -> list[ProcessingRun]:
        statement = (
            select(ProcessingRun)
            .where(ProcessingRun.input_export_id == export_id)
            .order_by(ProcessingRun.created_at.asc(), ProcessingRun.id.asc())
        )
        return list(self._session.scalars(statement))


class ArtifactRepository:
    """Persistence for derived/original :class:`Artifact` records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        layer: ArtifactLayer,
        export_id: uuid.UUID,
        path: str,
        sha256: str,
        processing_run_id: uuid.UUID | None = None,
        version: int | None = None,
        created_at: datetime | None = None,
    ) -> Artifact:
        artifact = Artifact(
            layer=layer,
            export_id=export_id,
            processing_run_id=processing_run_id,
            version=version,
            path=path,
            sha256=sha256,
            created_at=created_at or utc_now(),
        )
        self._session.add(artifact)
        self._session.flush()
        return artifact

    def get(self, artifact_id: uuid.UUID) -> Artifact | None:
        return self._session.get(Artifact, artifact_id)

    def list_by_export(self, export_id: uuid.UUID) -> list[Artifact]:
        """Every artifact of one export lineage, ordered by layer then version."""
        statement = (
            select(Artifact)
            .where(Artifact.export_id == export_id)
            .order_by(Artifact.layer.asc(), Artifact.version.asc().nullsfirst())
        )
        return list(self._session.scalars(statement))

    def list_by_export_and_layer(
        self, export_id: uuid.UUID, layer: ArtifactLayer
    ) -> list[Artifact]:
        """Artifacts of one layer for an export, ordered by version (v1, v2, ...)."""
        statement = (
            select(Artifact)
            .where(Artifact.export_id == export_id, Artifact.layer == layer)
            .order_by(Artifact.version.asc().nullsfirst())
        )
        return list(self._session.scalars(statement))

    def list_by_run(self, processing_run_id: uuid.UUID) -> list[Artifact]:
        statement = select(Artifact).where(Artifact.processing_run_id == processing_run_id)
        return list(self._session.scalars(statement))


class JobRepository:
    """Persistence for durable queue :class:`Job` records (ADR-003)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        job_type: JobType,
        status: JobStatus = JobStatus.pending,
        payload_json: dict[str, Any] | None = None,
        metadata_json: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        max_attempts: int | None = None,
        next_retry_at: datetime | None = None,
    ) -> Job:
        job = Job(
            job_type=job_type,
            status=status,
            payload_json=payload_json,
            metadata_json=metadata_json,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            next_retry_at=next_retry_at,
        )
        self._session.add(job)
        self._session.flush()
        return job

    def get(self, job_id: uuid.UUID) -> Job | None:
        return self._session.get(Job, job_id)

    def get_by_idempotency_key(self, idempotency_key: str) -> Job | None:
        statement = select(Job).where(Job.idempotency_key == idempotency_key)
        return self._session.scalar(statement)

    def list_by_status(self, status: JobStatus) -> list[Job]:
        statement = (
            select(Job).where(Job.status == status).order_by(Job.created_at.asc(), Job.id.asc())
        )
        return list(self._session.scalars(statement))
