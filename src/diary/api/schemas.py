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

if TYPE_CHECKING:
    from diary.db.models import Message


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
