"""SQLAlchemy ORM models for the Local Diary persistence layer (Task 04).

Implements the seven core tables of the MVP data model (ADR-001, Task 04):

- ``messages``          — text and audio diary messages (source data);
- ``audio_files``       — metadata of voice messages; files live on the local
                          filesystem under server-generated names;
- ``transcriptions``    — raw STT results, one row per transcription run;
- ``weekly_exports``    — original weekly TXT exports (layer 1), revisions;
- ``processing_runs``   — normalize/summarize/wellbeing pipeline runs;
- ``artifacts``         — derived files of layers 2-4 (versioned ``vM``);
- ``jobs``              — durable SQLite-backed worker queue (ADR-003).

Schema conventions (see ``skills/sqlite-data-model`` and the architecture
brief §5):

- stable UUID primary keys (SQLAlchemy ``Uuid``, stored as text on SQLite);
- timezone-aware UTC timestamps through ``UTCDateTime`` (``db.base``);
- foreign keys with ``PRAGMA foreign_keys=ON`` enforcement (``db.session``);
- statuses/kind as SQLAlchemy enums with DB-level CHECK constraints;
- JSON columns only for genuinely flexible metadata, never for core fields;
- hash fields (SHA-256) for audio files and artifacts.

Deliberate decision (Task 04 scope, item 8): a ``model_configs`` table is NOT
introduced. Task 03 keeps per-task model profiles in the file-based
``configs/models.yaml`` configuration; it remains the source for runtime
profiles. A persisted copy would duplicate configuration state and risk
divergence, so no table is defined here.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    BigInteger,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column

from diary.db.base import Base, UTCDateTime, utc_now

# ---------------------------------------------------------------------------
# Enumerations (values are stored as their names, which equal the values here).
# ---------------------------------------------------------------------------


class MessageKind(str, Enum):
    """Diary message type: plain text or a voice message."""

    text = "text"
    audio = "audio"


class TranscriptionStatus(str, Enum):
    """Lifecycle of an STT transcription request (architecture brief §5)."""

    not_requested = "not_requested"
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ExportStatus(str, Enum):
    """Original weekly export state; a row is written only for a real file."""

    created = "created"
    failed = "failed"


class PipelineType(str, Enum):
    """Derived-artifact pipeline kind (layers 2-4; see also ``ArtifactLayer``)."""

    normalize = "normalize"
    summarize = "summarize"
    wellbeing = "wellbeing"


class RunStatus(str, Enum):
    """Processing-run lifecycle (queued → running → succeeded | failed)."""

    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ArtifactLayer(str, Enum):
    """Artifact layer; ``original`` is reserved for layer-1 export files."""

    original = "original"
    normalized = "normalized"
    summary = "summary"
    wellbeing = "wellbeing"


class JobType(str, Enum):
    """Durable queue job kinds (queue mechanics are implemented in Task 08)."""

    transcribe_audio = "transcribe_audio"
    normalize = "normalize"
    summarize = "summarize"
    wellbeing = "wellbeing"


class JobStatus(str, Enum):
    """Durable queue job lifecycle (ADR-003, architecture brief §11)."""

    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    retrying = "retrying"
    cancelled = "cancelled"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid_primary_key():
    """A UUID primary-key column with a client-side ``uuid.uuid4`` default."""
    return mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


_ENUM_OPTIONS = dict(native_enum=False, create_constraint=True, validate_strings=True)

_MIME_LEN = 100
_PATH_LEN = 512
_SHA256_LEN = 64
_IDEMPOTENCY_KEY_LEN = 255


def _enum(enum_cls: type[Enum], name: str, length: int) -> SAEnum:
    """SQLAlchemy enum column type for SQLite (VARCHAR + CHECK constraint)."""
    return SAEnum(enum_cls, name=name, length=length, **_ENUM_OPTIONS)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class Message(Base):
    """A single diary entry: original user text or a link to a voice message.

    Immutable source data. Deletion follows the soft-delete policy fixed in
    Task 05 (migration ``0002``): deleting sets ``deleted_at``, the row stays
    in the database for traceability, the body is never overwritten, and
    deleted messages are hidden from the history by default.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _uuid_primary_key()
    kind: Mapped[MessageKind] = mapped_column(_enum(MessageKind, "message_kind", 16))
    body_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, nullable=False, index=True
    )
    #: UTC moment of soft deletion; NULL while the message is active (Task 05).
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class AudioFile(Base):
    """Metadata of a voice message. The audio bytes live on the local
    filesystem under a server-generated name; user-provided file names are
    never stored (no path traversal surface, ADR-002)."""

    __tablename__ = "audio_files"

    id: Mapped[uuid.UUID] = _uuid_primary_key()
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id"), nullable=False, index=True
    )
    original_mime: Mapped[str] = mapped_column(String(_MIME_LEN), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(_PATH_LEN), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(_SHA256_LEN), nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)


class Transcription(Base):
    """Raw STT output for one audio file. Each transcription run inserts a new
    row; history is never erased (ADR-005). Text messages stay separate."""

    __tablename__ = "transcriptions"

    id: Mapped[uuid.UUID] = _uuid_primary_key()
    audio_file_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_files.id"), nullable=False, index=True
    )
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TranscriptionStatus] = mapped_column(
        _enum(TranscriptionStatus, "transcription_status", 16), nullable=False
    )
    # Provider provenance (snapshot of what actually ran; configuration files
    # may change later, so the used profile/model is kept on the record).
    profile_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Flexible STT-run details (device, compute type, beam size, ...).
    metadata_json: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)


class WeeklyExport(Base):
    """An original weekly TXT export (layer 1) and its revisions.

    ``week_end_date`` is the Sunday date of the week. ``seq_number`` is the
    global export sequence ``NNNN`` (no gaps, assigned by the export service in
    Task 10) and is shared by all revisions of the same week/export lineage;
    ``revision`` distinguishes ``r1``, ``r2``, ... for one week. The pair
    (week, revision) is unique — a silent duplicate export is impossible.
    """

    __tablename__ = "weekly_exports"
    __table_args__ = (
        UniqueConstraint(
            "week_end_date", "revision", name="uq_weekly_exports_week_end_date_revision"
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_primary_key()
    week_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    seq_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ExportStatus] = mapped_column(
        _enum(ExportStatus, "export_status", 16), nullable=False, default=ExportStatus.created
    )
    path: Mapped[str] = mapped_column(String(_PATH_LEN), nullable=False)
    sha256: Mapped[str] = mapped_column(String(_SHA256_LEN), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)


class ProcessingRun(Base):
    """One pipeline run over a weekly export (normalize/summarize/wellbeing).

    The input is recorded either as the source original export
    (``input_export_id``) or as a previously produced artifact
    (``input_artifact_id``). The produced file is linked back through the
    ``artifacts.processing_run_id`` foreign key, which closes the lineage loop:
    artifact → run → input → export.
    """

    __tablename__ = "processing_runs"

    id: Mapped[uuid.UUID] = _uuid_primary_key()
    pipeline_type: Mapped[PipelineType] = mapped_column(
        _enum(PipelineType, "pipeline_type", 16), nullable=False
    )
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status", 16), nullable=False, default=RunStatus.queued
    )
    input_export_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("weekly_exports.id"), nullable=True, index=True
    )
    input_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True, index=True
    )
    profile_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Safe error summary (never diary content or provider secrets).
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Flexible run metadata (provider details, token usage, chunking, ...).
    metadata_json: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class Artifact(Base):
    """A derived (or original) artifact file of an export lineage.

    Every derived artifact (layers 2-4) is traceable to its source export and
    to the exact processing run that produced it. ``version`` holds ``vM`` for
    derived artifacts (a re-run produces ``v+1`` instead of overwriting);
    ``version`` stays NULL for ``original`` layer rows, whose revision is
    recorded on ``weekly_exports``.
    """

    __tablename__ = "artifacts"
    __table_args__ = (Index("ix_artifacts_export_id_layer", "export_id", "layer"),)

    id: Mapped[uuid.UUID] = _uuid_primary_key()
    layer: Mapped[ArtifactLayer] = mapped_column(
        _enum(ArtifactLayer, "artifact_layer", 16), nullable=False
    )
    export_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("weekly_exports.id"), nullable=False)
    processing_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("processing_runs.id"), nullable=True, index=True
    )
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    path: Mapped[str] = mapped_column(String(_PATH_LEN), nullable=False)
    sha256: Mapped[str] = mapped_column(String(_SHA256_LEN), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)


class Job(Base):
    """A durable SQLite-backed queue task (ADR-003; worker in Task 08).

    The payload holds only input identifiers and parameters — never diary
    content. Status transitions and retry/claim mechanics belong to Task 08;
    this task fixes the schema (status index included) only.
    """

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_primary_key()
    job_type: Mapped[JobType] = mapped_column(_enum(JobType, "job_type", 24), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        _enum(JobStatus, "job_status", 16), nullable=False, default=JobStatus.pending, index=True
    )
    payload_json: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(_IDEMPOTENCY_KEY_LEN), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False
    )
