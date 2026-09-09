"""Repository layer for basic persistence of the core entities (Task 04).

Each repository is bound to a single SQLAlchemy ``Session`` and exposes plain
create/read operations plus small list queries by key filters — enough for
integration tests and for the service layer of later tasks. Repositories
contain no business logic: they flush on create (so constraint/FK errors
surface immediately) but never commit — the caller owns the transaction and
commits atomically at the end of its unit of work.

The one deliberate exception is the durable job queue (Task 08, ADR-003): the
queue *mechanics* (atomic claim, retry/backoff transitions, staleness reset)
live on :class:`JobRepository` below, because they are purely persistence
operations over the ``jobs`` table — there is no domain service for them yet.
The callers of these methods (worker runner, API routes) still own all
transactions.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any, cast

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from diary.core.settings import get_settings
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
        include_deleted: bool = False,
    ) -> list[Message]:
        """Messages in chronological order (oldest first, ties by id).

        Soft-deleted rows are excluded by default (Task 05 policy); pass
        ``include_deleted=True`` to read every row regardless of ``deleted_at``.
        """
        statement = select(Message).order_by(Message.created_at.asc(), Message.id.asc())
        if not include_deleted:
            statement = statement.where(Message.deleted_at.is_(None))
        if kind is not None:
            statement = statement.where(Message.kind == kind)
        if created_before is not None:
            statement = statement.where(Message.created_at < created_before)
        if limit is not None:
            statement = statement.limit(limit)
        return list(self._session.scalars(statement))

    def list_active_window(
        self,
        *,
        limit: int,
        older_than: tuple[datetime, uuid.UUID] | None = None,
    ) -> list[Message]:
        """Up to ``limit`` active messages in a keyset "history" window.

        Returns the ``limit`` most recent active messages when ``older_than``
        is ``None``; otherwise the ``limit`` most recent active messages that
        are strictly older than the keyset position ``(created_at, id)``.
        The result is in chronological order (oldest first, ties by id), which
        is what chat-history rendering needs. Soft-deleted rows are excluded.
        """
        statement = select(Message).where(Message.deleted_at.is_(None))
        if older_than is not None:
            created_before, id_before = older_than
            statement = statement.where(
                or_(
                    Message.created_at < created_before,
                    and_(Message.created_at == created_before, Message.id < id_before),
                )
            )
        newest_first = self._session.scalars(
            statement.order_by(Message.created_at.desc(), Message.id.desc()).limit(limit)
        )
        return list(reversed(list(newest_first)))

    def has_active_older_than(self, older_than: tuple[datetime, uuid.UUID]) -> bool:
        """Whether any active message is strictly older than the keyset
        position ``(created_at, id)`` (drives the ``next_cursor`` decision)."""
        created_before, id_before = older_than
        statement = (
            select(Message.id)
            .where(
                Message.deleted_at.is_(None),
                or_(
                    Message.created_at < created_before,
                    and_(Message.created_at == created_before, Message.id < id_before),
                ),
            )
            .limit(1)
        )
        return self._session.scalar(statement) is not None


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
        id: uuid.UUID | None = None,
    ) -> AudioFile:
        """Insert one audio-file metadata row.

        ``id`` defaults to a fresh UUID; callers that already generated the
        server-side file name (``<id>.<ext>``) pass it explicitly so the row
        primary key and the on-disk name always match.
        """
        audio_file = AudioFile(
            message_id=message_id,
            original_mime=original_mime,
            storage_path=storage_path,
            size_bytes=size_bytes,
            sha256=sha256,
            duration_seconds=duration_seconds,
            created_at=created_at or utc_now(),
        )
        if id is not None:
            # Setting the attribute explicitly suppresses the column default,
            # so the primary key matches the already-generated file name.
            audio_file.id = id
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


# ---------------------------------------------------------------------------
# Job-queue policy helpers (Task 08, ADR-003, architecture brief §11)
# ---------------------------------------------------------------------------

#: Job statuses for which an ``idempotency_key`` replay returns the existing
#: job instead of creating a duplicate. A job that already succeeded counts as
#: "already handled"; an in-flight or scheduled one must not be duplicated.
#: Jobs that ended ``failed``/``cancelled`` do not block a fresh enqueue with
#: the same key (the caller explicitly retries).
_REUSABLE_IDEMPOTENCY_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.pending,
        JobStatus.running,
        JobStatus.retrying,
        JobStatus.succeeded,
    }
)

#: Upper bound for a stored ``error_summary`` / exception message fragment.
MAX_ERROR_SUMMARY_LENGTH = 300


def retry_delay_seconds(
    *,
    attempts: int,
    base_s: float,
    multiplier: float,
    max_backoff_s: float,
) -> float:
    """Exponential-backoff delay before the next run after ``attempts`` tries.

    The ``k``-th failed attempt (``attempts == k``) waits
    ``base_s * multiplier ** (k - 1)`` seconds, capped at ``max_backoff_s``.
    Deterministic (no jitter): the MVP runs one local worker, and deterministic
    delays keep retry scheduling testable.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}.")
    return min(max_backoff_s, base_s * (multiplier ** (attempts - 1)))


class JobRepository:
    """Persistence for durable queue :class:`Job` records (ADR-003).

    Task 04 fixed the schema; Task 08 adds the queue mechanics on top of it:

    - :meth:`enqueue` — create a ``pending`` job with idempotency-key reuse;
    - :meth:`claim_due` — atomically move one due job to ``running``
      (single conditional ``UPDATE``, so two workers can never claim the same
      job — SQLite serializes writers, WAL + ``busy_timeout`` are set up in
      ``diary.db.session``);
    - :meth:`mark_succeeded` / :meth:`mark_failed` / :meth:`cancel` —
      terminal/transition updates, retries with exponential backoff;
    - :meth:`reset_stale_running` — crash recovery of abandoned ``running``
      jobs after a worker restart;
    - :meth:`list_recent` — newest-first listing for status polling.

    The schema intentionally stores no ``worker_id``/``started_at`` columns
    (Task 04): a claim refreshes ``updated_at``, which therefore marks when the
    job entered ``running`` and is used as the staleness reference. No method
    commits — the caller owns the transaction and commits atomically.
    """

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
        """Return the most recent job recorded with ``idempotency_key``."""
        statement = (
            select(Job)
            .where(Job.idempotency_key == idempotency_key)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(1)
        )
        return self._session.scalar(statement)

    def list_by_status(self, status: JobStatus) -> list[Job]:
        statement = (
            select(Job).where(Job.status == status).order_by(Job.created_at.asc(), Job.id.asc())
        )
        return list(self._session.scalars(statement))

    def list_active_of_type(self, job_type: JobType) -> list[Job]:
        """Jobs of one type still in flight, oldest first.

        "In flight" means ``pending``, ``running`` or ``retrying``. Used by the
        transcription service (Task 09) to keep a single active job per audio
        file; terminal jobs are deliberately excluded so an explicit repeat
        request can start a fresh run.
        """
        statement = (
            select(Job)
            .where(
                Job.job_type == job_type,
                Job.status.in_((JobStatus.pending, JobStatus.running, JobStatus.retrying)),
            )
            .order_by(Job.created_at.asc(), Job.id.asc())
        )
        return list(self._session.scalars(statement))

    def list_recent(self, *, limit: int, status: JobStatus | None = None) -> list[Job]:
        """The ``limit`` most recent jobs, newest first (stable id tie-break).

        Optionally narrowed to one status. ``limit`` must be >= 1.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}.")
        statement = select(Job).order_by(Job.created_at.desc(), Job.id.desc())
        if status is not None:
            statement = statement.where(Job.status == status)
        return list(self._session.scalars(statement.limit(limit)))

    # ------------------------------------------------------------------
    # Queue mechanics (Task 08)
    # ------------------------------------------------------------------

    def enqueue(
        self,
        *,
        job_type: JobType,
        payload_json: dict[str, Any] | None = None,
        metadata_json: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        max_attempts: int | None = None,
    ) -> Job:
        """Create one ``pending`` job (or return an existing equivalent).

        With an ``idempotency_key`` and a matching job in a reusable state
        (pending/running/retrying/succeeded) no second row is created — the
        existing job is returned unchanged. ``max_attempts`` defaults to the
        configured ``job_max_attempts`` and must be >= 1. The payload holds
        only input identifiers/parameters, never diary content.
        """
        if max_attempts is None:
            max_attempts = get_settings().job_max_attempts
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")
        if idempotency_key is not None and len(idempotency_key) > 255:
            raise ValueError("idempotency_key must be at most 255 characters.")
        if idempotency_key is not None:
            existing = self.get_by_idempotency_key(idempotency_key)
            if existing is not None and existing.status in _REUSABLE_IDEMPOTENCY_STATUSES:
                return existing
        job = Job(
            job_type=job_type,
            status=JobStatus.pending,
            payload_json=payload_json,
            metadata_json=metadata_json,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
        )
        self._session.add(job)
        self._session.flush()
        return job

    def claim_due(self, *, now: datetime | None = None) -> Job | None:
        """Atomically claim one due job and return it as ``running``.

        A job is due when it is ``pending``, or ``retrying`` with
        ``next_retry_at`` already reached. The claim is a single conditional
        ``UPDATE ... RETURNING`` that increments ``attempts`` and refreshes
        ``updated_at`` (the staleness marker). Because the status check and the
        transition happen in one serialized write statement, a concurrent
        worker can never claim the same job (ADR-003).
        """
        claimed_at = now if now is not None else utc_now()
        due_job_ids = (
            select(Job.id)
            .where(
                or_(
                    Job.status == JobStatus.pending,
                    and_(
                        Job.status == JobStatus.retrying,
                        Job.next_retry_at <= claimed_at,
                    ),
                )
            )
            .order_by(Job.created_at.asc(), Job.id.asc())
            .limit(1)
        )
        statement = (
            update(Job)
            .where(Job.id.in_(due_job_ids))
            .values(
                status=JobStatus.running,
                attempts=Job.attempts + 1,
                next_retry_at=None,
                error_summary=None,
                updated_at=claimed_at,
            )
            .returning(Job.id)
        )
        claimed_id = self._session.scalar(statement)
        if claimed_id is None:
            return None
        return self._session.get(Job, claimed_id)

    def mark_succeeded(self, job_id: uuid.UUID) -> Job | None:
        """Mark a ``running`` job as ``succeeded`` (no commit; caller owns it)."""
        job = self._session.get(Job, job_id)
        if job is None:
            return None
        job.status = JobStatus.succeeded
        job.next_retry_at = None
        job.error_summary = None
        self._session.flush()
        return job

    def mark_failed(
        self,
        job_id: uuid.UUID,
        *,
        error_summary: str,
        permanent: bool = False,
    ) -> Job | None:
        """Apply the failure policy to a claimed job (no commit; caller owns it).

        A transient failure moves the job to ``retrying`` with
        ``next_retry_at`` computed by exponential backoff while attempts remain;
        after ``max_attempts`` failures it becomes ``failed``. A ``permanent``
        failure (unknown job type, explicit non-retryable handler error)
        becomes ``failed`` immediately. ``error_summary`` is stored verbatim and
        must already be a safe, non-sensitive summary.
        """
        job = self._session.get(Job, job_id)
        if job is None:
            return None
        if permanent:
            job.status = JobStatus.failed
            job.next_retry_at = None
        else:
            effective_max = (
                job.max_attempts
                if job.max_attempts is not None
                else get_settings().job_max_attempts
            )
            if job.attempts < effective_max:
                policy = get_settings()
                delay_s = retry_delay_seconds(
                    attempts=job.attempts,
                    base_s=policy.job_backoff_base_s,
                    multiplier=policy.job_backoff_multiplier,
                    max_backoff_s=policy.job_max_backoff_s,
                )
                job.status = JobStatus.retrying
                job.next_retry_at = utc_now() + timedelta(seconds=delay_s)
            else:
                job.status = JobStatus.failed
                job.next_retry_at = None
        job.error_summary = error_summary
        self._session.flush()
        return job

    def cancel(self, job_id: uuid.UUID) -> Job | None:
        """Cancel a job that has not started yet (pending/retrying only).

        ``running`` and terminal jobs cannot be cancelled: this returns
        ``None``. No commit — the caller owns the transaction.
        """
        job = self._session.get(Job, job_id)
        if job is None or job.status not in (JobStatus.pending, JobStatus.retrying):
            return None
        job.status = JobStatus.cancelled
        job.next_retry_at = None
        self._session.flush()
        return job

    def reset_stale_running(self, *, older_than: datetime) -> int:
        """Move abandoned ``running`` jobs older than ``older_than`` to pending.

        Crash recovery for workers: a claimed job whose ``updated_at`` is older
        than the threshold was left behind by a crashed/restarted worker. The
        job returns to ``pending`` (preserving its ``attempts``) so it is
        immediately claimable again. Returns the number of reset jobs.
        """
        result = self._session.execute(
            update(Job)
            .where(Job.status == JobStatus.running, Job.updated_at < older_than)
            .values(status=JobStatus.pending)
        )
        return int(cast("CursorResult[Any]", result).rowcount or 0)
