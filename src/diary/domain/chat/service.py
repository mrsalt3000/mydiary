"""Message use cases: create, list (cursor history), delete (Tasks 05/T15).

Deletion policy — mark, erase text, remove audio bytes
======================================================

Task 05 marked a message and left the row untouched; **ADR-011 (task T15)**
replaces that policy at the user's explicit request:

* ``DELETE /messages/{id}`` still sets ``deleted_at = utc_now()`` (the mark
  hides the message from the history and from every pipeline query), **and**
  additionally erases the source data itself: ``messages.body_text`` and the
  ``raw_text`` of every transcription of the message's audio files are
  emptied, and the stored audio bytes are unlinked from disk.
* Rows are kept (``messages``, ``audio_files``, ``transcriptions`` with their
  statuses, profile snapshots and timestamps): lineage — and therefore
  ``trace_layer_result`` — stays intact. Derived artifacts and weekly export
  files are never touched by a deletion.
* Order of operations: the database transition is one atomic commit, and only
  after it succeeds are the audio bytes removed (best effort). A failure
  during the file removal leaves the message deleted and the text erased —
  never a visible message with broken playback (ADR-011 §5).
* Idempotency: a second delete of an already-deleted message is reported as
  not found (``MessageNotFoundError`` → HTTP 404), changes nothing and does
  not fail; the client treats that 404 as "already deleted".
* There is no restore.

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
from pathlib import Path

from sqlalchemy.orm import Session

from diary.db.base import utc_now
from diary.db.models import Message, MessageKind
from diary.db.repositories import (
    AudioFileRepository,
    MessageRepository,
    TranscriptionRepository,
)
from diary.domain.audio.service import resolve_audio_file

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


def delete_message(session: Session, message_id: uuid.UUID) -> None:
    """Delete a message: mark it, erase its text and remove its audio bytes.

    Implements the ADR-011 policy (task T15). The database transition — the
    ``deleted_at`` mark, emptying ``body_text`` and the ``raw_text`` of every
    transcription of the message's audio files — is committed atomically
    first; only then are the stored audio bytes removed from disk (best
    effort). The reversed order would let a crash leave a message that is
    still visible but whose playback is already broken.

    Idempotent by outcome: deleting a message that is already deleted changes
    nothing and raises :class:`MessageNotFoundError` (→ HTTP 404), which the
    clients read as "already deleted". Raises the same error for an unknown id.
    """
    repository = MessageRepository(session)
    message = repository.get(message_id)
    if message is None or message.deleted_at is not None:
        raise MessageNotFoundError("Message not found.")

    audio_files = AudioFileRepository(session).get_by_message(message_id)
    # Resolved *before* the mark is written: playback resolution refuses a
    # deleted message's audio (ADR-011 §6). Errors here abort the deletion
    # with nothing changed, which is the safe direction.
    audio_paths: list[Path] = []
    for audio in audio_files:
        path, _mime = resolve_audio_file(session, audio.id)
        audio_paths.append(path)

    transcriptions = TranscriptionRepository(session)
    message.deleted_at = utc_now()
    message.body_text = ""
    for audio in audio_files:
        for transcription in transcriptions.list_by_audio_file(audio.id):
            transcription.raw_text = ""
    session.commit()

    for path in audio_paths:
        _unlink_safe(path)


def _unlink_safe(path: Path) -> None:
    """Remove a file, ignoring "already gone" and other filesystem errors.

    Mirrors the best-effort cleanup used across the codebase (for example
    ``diary.domain.audio.service._unlink_safe``): a file that is already
    absent — the normal case for a repeated delete — must never turn a
    completed deletion into an error.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort removal
        pass
