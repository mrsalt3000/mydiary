"""Message use cases: create, list (cursor history), soft delete (Task 05).

Deletion policy — soft delete only
===================================

* ``DELETE /messages/{id}`` marks a message with ``deleted_at = utc_now()``.
  The row stays in the database for traceability and ``body_text`` is never
  overwritten (immutable source data, ADR-005).
* Soft-deleted messages are hidden from the history by default.
* A second delete of an already-deleted message is reported as not found
  (``MessageNotFoundError`` → HTTP 404): once hidden, the message is no longer
  an addressable chat resource.
* There is no restore and no hard delete in the MVP (Task 05 scope). Weekly
  exports (Task 10) decide separately how deleted entries are treated.

Timestamps
==========

The database stores UTC only (``diary.db.base.UTCDateTime``). Conversion to
the application timezone (``AppSettings.tzinfo``) happens when the API response
is built (``diary.api.schemas.message_read_from``), never at write time.

History and pagination
======================

History is deterministic chronological order (``created_at``, then ``id`` as
tie-breaker). Pages walk backwards in time: the first page (``cursor=None``)
contains the most recent messages; each next page returns messages strictly
older than the previous page's oldest message (keyset cursor, no gaps and no
duplicates across page boundaries).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from diary.db.base import utc_now
from diary.db.models import Message, MessageKind
from diary.db.repositories import MessageRepository

#: Keyset pagination position: ``(created_at, id)`` of the oldest message of
#: the page already returned; the next page consists of active messages
#: strictly older than this position.
MessageCursor = tuple[datetime, uuid.UUID]


class MessageNotFoundError(Exception):
    """The message id does not exist or the message is already deleted.

    Carries no diary content; maps to HTTP 404 at the API boundary.
    """


def create_text_message(session: Session, body: str) -> Message:
    """Create a ``kind=text`` diary message and commit the transaction."""
    repository = MessageRepository(session)
    message = repository.create(kind=MessageKind.text, body_text=body)
    session.commit()
    return message


def list_messages(
    session: Session,
    *,
    limit: int,
    cursor: MessageCursor | None = None,
) -> tuple[list[Message], MessageCursor | None]:
    """Return up to ``limit`` active messages plus the next-page cursor.

    The messages are in chronological order (oldest first, ``id`` as the
    tie-breaker). When ``cursor`` is ``None`` the most recent messages are
    returned; otherwise only messages strictly older than the cursor position
    are considered. The second element of the result is the position of the
    oldest returned message when older active messages exist, else ``None``
    (client stops paging).
    """
    repository = MessageRepository(session)
    messages = repository.list_active_window(limit=limit, older_than=cursor)
    next_cursor: MessageCursor | None = None
    if messages:
        oldest = messages[0]
        anchor = (oldest.created_at, oldest.id)
        if repository.has_active_older_than(anchor):
            next_cursor = anchor
    return messages, next_cursor


def soft_delete_message(session: Session, message_id: uuid.UUID) -> None:
    """Soft-delete a message by setting ``deleted_at`` to now (UTC).

    Commits the transaction. Raises :class:`MessageNotFoundError` when the id
    does not exist or the message is already deleted (both → HTTP 404).
    """
    repository = MessageRepository(session)
    message = repository.get(message_id)
    if message is None or message.deleted_at is not None:
        raise MessageNotFoundError("Message not found.")
    message.deleted_at = utc_now()
    session.commit()
