"""Day-scoped input selection for the daily-summary layer (task T05).

The daily-summary layer consumes **the messages of one local calendar day**
(the ``APP_TIMEZONE`` day, ADR-007 §1) mapped onto the ``raw_entry`` contract
(T00). That selection is a distinct concern from the LLM call, so it lives in
this small module inside the layer package:

* :func:`day_bounds` — the two UTC instants that bound a local calendar day
  (local midnight through the instant before the next local midnight);
* :func:`local_date_of` — which local day a moment belongs to;
* :func:`collect_day_entries` — the day's messages as ``raw_entry[]`` in
  chronological order.

Read-only by construction: the module only *reads* ``messages``,
``audio_files`` and ``transcriptions`` (through the Task-04 repositories), never
writes to them (AGENTS.md §6: the originals are immutable). A text message maps
to ``source="text"`` with its ``body_text``; a voice message maps to
``source="voice"`` with its ``audio_path`` and the ``raw_text`` of its newest
**succeeded** transcription — an empty string while the message has no
transcription yet, which is exactly what the ``raw_entry`` contract says. A
failed transcription row is not a transcription result and is never used.

Soft-deleted messages (Task 05 policy) are excluded: they are hidden from the
history, and a summary must not resurrect them.

A voice message whose ``audio_files`` row is missing cannot be mapped onto the
contract (``raw_entry`` requires ``audio_path`` for a voice entry) and no path
may be invented, so :class:`DayInputError` is raised — a loud, non-retryable
input failure rather than a silently shortened day.

Privacy: the selected texts stay in memory as contract objects and are never
logged (AGENTS.md §7-8).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone, tzinfo

from sqlalchemy.orm import Session

from diary.db.models import AudioFile, Message, MessageKind, Transcription, TranscriptionStatus
from diary.db.repositories import AudioFileRepository, MessageRepository, TranscriptionRepository
from diary.domain.contracts import RawEntry
from diary.domain.pipelines.base import PipelinePermanentError

__all__ = [
    "DayInputError",
    "collect_day_entries",
    "day_bounds",
    "local_date_of",
]


class DayInputError(PipelinePermanentError):
    """The day cannot be turned into a valid ``raw_entry[]`` input.

    Raised for an unusable day input — currently a voice message without its
    ``audio_files`` row, whose ``raw_entry`` cannot be completed without
    inventing an ``audio_path``. The message is a static, non-sensitive summary
    (AGENTS.md §8).
    """


def day_bounds(local_date: date, tz: tzinfo) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` — the boundary instants of the local day.

    ``start`` is local midnight converted to its UTC instant; ``end`` is one
    microsecond before the next local midnight, so the day spans exactly one
    local calendar day even across a DST transition (the same rule
    ``diary.domain.exports.weeks.week_bounds`` applies to a week). Both results
    are timezone-aware UTC datetimes, directly comparable with the
    UTC-stored ``messages.created_at`` (``diary.db.base.UTCDateTime``).
    """
    start_local = datetime.combine(local_date, time.min, tzinfo=tz)
    next_midnight_local = datetime.combine(local_date + timedelta(days=1), time.min, tzinfo=tz)
    end_local = next_midnight_local - timedelta(microseconds=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def local_date_of(ts: datetime, tz: tzinfo) -> date:
    """The local calendar day of the moment ``ts`` in timezone ``tz``.

    ``ts`` must be timezone-aware: without an offset the day boundary would be
    undefined (the same rule the ``raw_entry`` contract enforces).
    """
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("ts must be timezone-aware to resolve its local calendar day")
    return ts.astimezone(tz).date()


def _latest_transcription_text(session: Session, audio_file: AudioFile) -> str:
    """``raw_text`` of the newest succeeded transcription, or ``""``.

    A failed transcription row carries no text and is not a result: the newest
    *succeeded* row is the raw STT text of the voice message. Until one exists
    the voice message has simply not been transcribed yet (contract: empty
    text).
    """
    rows: Sequence[Transcription] = TranscriptionRepository(session).list_by_audio_file(
        audio_file.id
    )
    succeeded = [row for row in rows if row.status is TranscriptionStatus.succeeded]
    return succeeded[-1].raw_text if succeeded else ""


def _entry_for(session: Session, message: Message) -> RawEntry:
    """Map one source message onto the ``raw_entry`` contract."""
    if message.kind is MessageKind.text:
        return RawEntry(
            id=message.id,
            ts=message.created_at,
            source="text",
            text=message.body_text,
        )
    audio_rows = AudioFileRepository(session).get_by_message(message.id)
    if not audio_rows:
        raise DayInputError(
            "A voice message of this day has no stored audio file; its raw_entry cannot be "
            "built without inventing an audio path."
        )
    # Task 07 stores exactly one audio_files row per voice message; the newest
    # row wins if history ever holds more than one.
    audio_file = audio_rows[-1]
    return RawEntry(
        id=message.id,
        ts=message.created_at,
        source="voice",
        text=_latest_transcription_text(session, audio_file),
        audio_path=audio_file.storage_path,
    )


def collect_day_entries(
    session: Session,
    *,
    local_date: date,
    tz: tzinfo,
) -> list[RawEntry]:
    """The messages of the local day ``local_date`` as ``raw_entry[]``.

    Chronological order (oldest first, ties by id) as the repository returns
    it; soft-deleted messages are excluded. The returned contract objects are
    fresh copies of the source values — nothing here can write to the raw
    tables.
    """
    start, end = day_bounds(local_date, tz)
    messages = MessageRepository(session).list_active_between(start=start, end=end)
    return [_entry_for(session, message) for message in messages]
