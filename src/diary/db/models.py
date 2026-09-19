"""SQLAlchemy ORM models for the Local Diary persistence layer (Task 04).

Implements the seven core tables of the MVP data model (ADR-001, Task 04):

- ``messages``          — text and audio diary messages (source data);
- ``audio_files``       — metadata of voice messages; files live on the local
                          filesystem under server-generated names;
- ``transcriptions``    — raw STT results, one row per transcription run;
- ``weekly_exports``    — original weekly TXT exports (layer 1), revisions;
- ``processing_runs``   — normalize/summarize/wellbeing pipeline runs;
- ``artifacts``         — derived artifacts of layers 2-4 (versioned ``vM``)
                          **and** the append-only results of the phase-2 second
                          anchor family (``stt_corrected``, ``daily_summary``,
                          ``weekly_summary``, ``coach_note``);
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
    CheckConstraint,
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
    """Derived-artifact pipeline kind (see also ``ArtifactLayer``).

    Values ``normalize``/``summarize``/``wellbeing`` are the three pipelines of
    the first (export-centric) anchor family (MVP layers 2-4). The four phase-2
    values (task T08, ADR-007/ADR-008) name the pipelines of the second anchor
    family and map 1:1 onto ``ArtifactLayer`` and ``JobType`` values, so a
    processing run always names the exact layer it produced.
    """

    # --- first anchor family: weekly export → normalized → summary/wellbeing ---
    normalize = "normalize"
    summarize = "summarize"
    wellbeing = "wellbeing"

    # --- second anchor family (phase 2, ADR-007): message/day/week/summary ---
    stt_corrected = "stt_corrected"
    daily_summary = "daily_summary"
    weekly_summary = "weekly_summary"
    coach_note = "coach_note"


class RunStatus(str, Enum):
    """Processing-run lifecycle (queued → running → succeeded | failed)."""

    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ArtifactLayer(str, Enum):
    """Artifact layer; ``original`` is reserved for layer-1 export files.

    The first four values belong to the first (export-centric) anchor family of
    the MVP (ADR-001/ADR-005): their anchor is ``artifacts.export_id``. The four
    phase-2 values are the layers of the **second anchor family** (ADR-007 §1):
    each is anchored to its own source — a message (``stt_corrected``), a local
    calendar day (``daily_summary``), an ISO week (``weekly_summary``) or an
    upstream weekly-summary artifact (``coach_note``) — never to a weekly
    export. ``wellbeing`` (layer 4) and ``coach_note`` (layer 5) are separate
    layers with separate disclaimers and are never merged (ADR-007 §8).
    """

    # --- first anchor family (weekly export lineages) ---
    original = "original"
    normalized = "normalized"
    summary = "summary"
    wellbeing = "wellbeing"

    # --- second anchor family (phase 2, ADR-007) ---
    stt_corrected = "stt_corrected"
    daily_summary = "daily_summary"
    weekly_summary = "weekly_summary"
    coach_note = "coach_note"


#: Layers of the first anchor family: anchored by ``artifacts.export_id``.
EXPORT_FAMILY_LAYERS: tuple[ArtifactLayer, ...] = (
    ArtifactLayer.original,
    ArtifactLayer.normalized,
    ArtifactLayer.summary,
    ArtifactLayer.wellbeing,
)

#: Layer of the second anchor family → the single anchor column that must be
#: set for it (ADR-007 §1). Order is stable: it is part of the schema CHECK
#: below and of the SQL frozen in migration ``0003``.
SECOND_FAMILY_LAYER_ANCHORS: dict[ArtifactLayer, str] = {
    ArtifactLayer.stt_corrected: "raw_id",
    ArtifactLayer.daily_summary: "local_date",
    ArtifactLayer.weekly_summary: "week_id",
    ArtifactLayer.coach_note: "summary_id",
}

#: Second-family anchor columns, in a stable order (schema CHECK below).
SECOND_FAMILY_ANCHORS: tuple[str, ...] = ("raw_id", "local_date", "week_id", "summary_id")

#: The second anchor family requires a version ``vM`` per pair (anchor, layer).
ANCHOR_FAMILY_CHECK = (
    "(export_id IS NOT NULL"
    " AND layer IN ('original', 'normalized', 'summary', 'wellbeing')"
    " AND raw_id IS NULL AND local_date IS NULL AND week_id IS NULL AND summary_id IS NULL)"
    " OR (export_id IS NULL AND version IS NOT NULL"
    " AND (CASE WHEN raw_id IS NOT NULL THEN 1 ELSE 0 END"
    " + CASE WHEN local_date IS NOT NULL THEN 1 ELSE 0 END"
    " + CASE WHEN week_id IS NOT NULL THEN 1 ELSE 0 END"
    " + CASE WHEN summary_id IS NOT NULL THEN 1 ELSE 0 END) = 1"
    " AND ((layer = 'stt_corrected' AND raw_id IS NOT NULL)"
    " OR (layer = 'daily_summary' AND local_date IS NOT NULL)"
    " OR (layer = 'weekly_summary' AND week_id IS NOT NULL)"
    " OR (layer = 'coach_note' AND summary_id IS NOT NULL)))"
)

#: A second-family day anchor always records the timezone it was computed in.
ANCHOR_LOCAL_TZ_CHECK = "(local_date IS NULL) = (local_tz IS NULL)"

#: A first-family artifact keeps exactly its previous shape (a rendered file and
#: its hash, never an inline result); a second-family result stores a rendered
#: file and/or an inline result payload — a row with neither would be a result
#: that stores nothing.
ANCHOR_PAYLOAD_CHECK = (
    "(export_id IS NOT NULL AND path IS NOT NULL AND sha256 IS NOT NULL AND result_json IS NULL)"
    " OR (export_id IS NULL AND (path IS NOT NULL OR result_json IS NOT NULL))"
)


class JobType(str, Enum):
    """Durable queue job kinds (queue mechanics are implemented in Task 08).

    The phase-2 values (task T08, ADR-007/ADR-008) are the job kinds of the
    second anchor family; each maps 1:1 onto a ``PipelineType``/``ArtifactLayer``
    value. A job type without a registered handler fails permanently with a
    clear summary (the Task-12 registry behaviour), so enqueuing a phase-2 job
    before its handler exists is safe.
    """

    # --- first anchor family (MVP) ---
    transcribe_audio = "transcribe_audio"
    normalize = "normalize"
    summarize = "summarize"
    wellbeing = "wellbeing"

    # --- second anchor family (phase 2, ADR-007) ---
    stt_corrected = "stt_corrected"
    daily_summary = "daily_summary"
    weekly_summary = "weekly_summary"
    coach_note = "coach_note"


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
    """A derived (or original) artifact of one anchor lineage.

    The table carries **two anchor families** (ADR-007 §1; the persistence
    shape is fixed in task T08 — see ADR-008):

    * **first family** (MVP layers 1-4): the row is anchored to a weekly export
      through ``export_id`` and is traceable to its source export and to the
      exact processing run that produced it. ``version`` holds ``vM`` for
      derived artifacts (a re-run produces ``v+1`` instead of overwriting);
      ``version`` stays NULL for ``original`` layer rows, whose revision is
      recorded on ``weekly_exports``.
    * **second family** (phase 2 layers ``stt_corrected`` / ``daily_summary`` /
      ``weekly_summary`` / ``coach_note``): the row is anchored **directly to
      its source** — the message (``raw_id``), the local calendar day
      (``local_date`` + the ``local_tz`` it was computed in), the ISO week
      (``week_id``) or the upstream weekly-summary artifact (``summary_id``) —
      with ``export_id`` NULL. No weekly export is required: the day summary
      must exist before the week is exported (ADR-007).

    Invariants enforced by the schema itself (not by the caller):

    * ``ck_artifacts_anchor_family`` — **exactly one** anchor family is set, the
      layer matches its anchor, and a second-family row always carries its
      ``vM`` version (``docs/data-lineage.md`` §3: the version belongs to the
      pair (input, layer); a re-run appends ``v+1`` and never overwrites);
    * ``ck_artifacts_anchor_local_tz`` — a day anchor always records its
      timezone (a local date without a timezone is ambiguous);
    * ``ck_artifacts_anchor_payload`` — a first-family artifact keeps exactly its
      previous shape (a rendered file plus its hash, never an inline result),
      while a second-family result stores a rendered file
      (``path``/``sha256``) and/or the validated structured result of the layer
      (``result_json`` — the AI output; the raw sources in
      ``messages``/``audio_files``/``transcriptions``/``weekly_exports`` are
      never written by the phase-2 layers);
    * one ``uq_artifacts_<anchor>_layer_version`` per anchor — a version of a
      (anchor, layer) pair can be written only once, so append-only versioning
      is a schema guarantee and not a convention.

    ``path``/``sha256`` are NULL for an inline result that has no rendered file;
    ``result_json`` is NULL for first-family rows (their content is the file).
    """

    __tablename__ = "artifacts"
    __table_args__ = (
        Index("ix_artifacts_export_id_layer", "export_id", "layer"),
        Index("ix_artifacts_raw_id_layer", "raw_id", "layer"),
        Index("ix_artifacts_local_date_layer", "local_date", "layer"),
        Index("ix_artifacts_week_id_layer", "week_id", "layer"),
        Index("ix_artifacts_summary_id_layer", "summary_id", "layer"),
        UniqueConstraint(
            "export_id",
            "layer",
            "version",
            name="uq_artifacts_export_id_layer_version",
        ),
        UniqueConstraint("raw_id", "layer", "version", name="uq_artifacts_raw_id_layer_version"),
        UniqueConstraint(
            "local_date",
            "local_tz",
            "layer",
            "version",
            name="uq_artifacts_local_date_layer_version",
        ),
        UniqueConstraint("week_id", "layer", "version", name="uq_artifacts_week_id_layer_version"),
        UniqueConstraint(
            "summary_id", "layer", "version", name="uq_artifacts_summary_id_layer_version"
        ),
        CheckConstraint(ANCHOR_FAMILY_CHECK, name="anchor_family"),
        CheckConstraint(ANCHOR_LOCAL_TZ_CHECK, name="anchor_local_tz"),
        CheckConstraint(ANCHOR_PAYLOAD_CHECK, name="anchor_payload"),
    )

    id: Mapped[uuid.UUID] = _uuid_primary_key()
    layer: Mapped[ArtifactLayer] = mapped_column(
        _enum(ArtifactLayer, "artifact_layer", 16), nullable=False
    )
    #: First-family anchor: the weekly export this artifact belongs to (NULL for
    #: a second-family artifact).
    export_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("weekly_exports.id"), nullable=True
    )
    processing_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("processing_runs.id"), nullable=True, index=True
    )
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Rendered artifact file, relative to the data root; NULL for an inline
    #: second-family result that was not rendered to a file.
    path: Mapped[str | None] = mapped_column(String(_PATH_LEN), nullable=True)
    #: SHA-256 of the rendered file; NULL when there is no file.
    sha256: Mapped[str | None] = mapped_column(String(_SHA256_LEN), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    # --- second-family anchors (ADR-007 §1; exactly one is set) ---
    #: ``stt_corrected``: the message whose text/raw STT was corrected.
    raw_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("messages.id"), nullable=True)
    #: ``daily_summary``: the local calendar day (``APP_TIMEZONE``) of the day.
    local_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    #: IANA timezone name the ``local_date`` day boundary was computed in.
    local_tz: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: ``weekly_summary``: the ISO week id ``YYYY-Www`` (ISO year, ``%G``).
    week_id: Mapped[str | None] = mapped_column(String(8), nullable=True)
    #: ``coach_note``: the upstream ``weekly_summary`` artifact it was built from.
    summary_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    #: Validated structured result payload of a phase-2 layer (the AI output),
    #: stored as JSON. Never a copy of raw source data.
    result_json: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)


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
