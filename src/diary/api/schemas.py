"""Pydantic v2 schemas of the messages API (Task 05).

Validation happens at the API boundary: ``MessageCreate.body`` must be a
non-empty string after stripping and no longer than
``AppSettings.max_message_length`` characters. Error messages are static and
never echo diary content.

Timestamps are timezone-aware: the database stores UTC
(``diary.db.base.UTCDateTime``) and responses render ``created_at`` in the
application timezone (``AppSettings.tzinfo``) as ISO-8601 with an offset.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import BaseModel, field_serializer, field_validator

from diary.core.settings import get_settings
from diary.db.models import TranscriptionStatus

if TYPE_CHECKING:
    from diary.db.models import Job, Message


class MessageCreate(BaseModel):
    """Request body of ``POST /api/v1/messages`` (text diary entry)."""

    body: str

    @field_validator("body")
    @classmethod
    def _validate_body(cls, value: str) -> str:
        """Reject empty (after strip) and oversized bodies.

        The stored value is kept exactly as submitted — original user text is
        immutable source data (ADR-005); only emptiness and size are checked.
        """
        if not value.strip():
            raise ValueError("Message body must not be empty.")
        limit = get_settings().max_message_length
        if len(value) > limit:
            raise ValueError(f"Message body must be no longer than {limit} characters.")
        return value


class MessageRead(BaseModel):
    """A diary message as returned by the API."""

    id: UUID
    kind: Literal["text", "audio"]
    body: str
    created_at: datetime

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        """ISO-8601 with offset (e.g. ``2026-01-01T05:45:00+05:45``)."""
        return value.isoformat()


class MessagePage(BaseModel):
    """Cursor page of the history endpoint ``GET /api/v1/messages``."""

    items: list[MessageRead]
    next_cursor: str | None
    limit: int


def message_read_from(record: Message) -> MessageRead:
    """Map an ORM message to ``MessageRead`` in the application timezone.

    The database stores UTC; display conversion happens here, when the
    response is assembled (see the chat service module docstring).
    """
    created_local = record.created_at.astimezone(get_settings().tzinfo)
    return MessageRead(
        id=record.id,
        kind=record.kind.value,
        body=record.body_text,
        created_at=created_local,
    )


class AudioRead(BaseModel):
    """A stored voice message as returned by ``POST /api/v1/audio``.

    ``id`` is the ``audio_files`` row id (the playback endpoint is
    ``GET /api/v1/audio/{id}/file``); ``message_id`` links it to the chat
    history. No user-supplied file name is ever echoed back.
    """

    id: UUID
    message_id: UUID
    mime: str
    size_bytes: int
    transcription_status: Literal["not_requested", "queued", "running", "succeeded", "failed"]
    created_at: datetime

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        """ISO-8601 with offset (e.g. ``2026-01-01T05:45:00+05:45``)."""
        return value.isoformat()


def audio_read_from(
    *,
    audio_file_id: UUID,
    message_id: UUID,
    mime: str,
    size_bytes: int,
    transcription_status: TranscriptionStatus,
    created_at: datetime,
) -> AudioRead:
    """Map a stored audio upload to ``AudioRead`` in the application timezone."""
    created_local = created_at.astimezone(get_settings().tzinfo)
    return AudioRead(
        id=audio_file_id,
        message_id=message_id,
        mime=mime,
        size_bytes=size_bytes,
        transcription_status=transcription_status.value,
        created_at=created_local,
    )


class JobRead(BaseModel):
    """Job status as returned by the polling endpoints (Task 08).

    Privacy rule: the response carries the status surface only — never the
    ``payload_json``/``metadata_json`` (they may hold identifiers of diary
    resources) and never diary content. ``error_summary`` is already a safe,
    non-sensitive summary written by the worker.
    """

    id: UUID
    job_type: str
    status: str
    attempts: int
    max_attempts: int | None
    error_summary: str | None
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def _serialize_timestamp(self, value: datetime) -> str:
        """ISO-8601 with offset (e.g. ``2026-01-01T05:45:00+05:45``)."""
        return value.isoformat()


class JobPage(BaseModel):
    """Recent jobs of the listing endpoint ``GET /api/v1/jobs``."""

    items: list[JobRead]
    limit: int


def job_read_from(record: Job) -> JobRead:
    """Map an ORM job to ``JobRead`` in the application timezone."""
    timezone_info = get_settings().tzinfo
    return JobRead(
        id=record.id,
        job_type=record.job_type.value,
        status=record.status.value,
        attempts=record.attempts,
        max_attempts=record.max_attempts,
        error_summary=record.error_summary,
        created_at=record.created_at.astimezone(timezone_info),
        updated_at=record.updated_at.astimezone(timezone_info),
    )
