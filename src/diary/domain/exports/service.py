"""Weekly original TXT export use cases (Task 10, architecture brief §7).

Responsibilities
================

* Collect a week's messages: every active (non-soft-deleted) ``messages`` row
  — text or audio — whose ``created_at`` falls inside the week's boundaries,
  in deterministic chronological order (``created_at`` then ``id``).
* For audio messages, resolve the linked ``audio_files`` row and the raw STT
  text of its newest *succeeded* transcription; audio with no successful
  transcription is explicitly marked ``[не расшифровано]``.
* Assign the global export sequence ``NNNN`` (no gaps) and revisions:
  the first export of a week is ``r1`` with ``max_sequence_number() + 1``; a
  re-export is an *explicit* action that keeps the lineage sequence number and
  takes the next revision of that week (``rN+1``). Empty weeks create no file,
  no row and consume no sequence number. A duplicate export of an already
  exported week (without re-export) is an error.
* Render the TXT byte-for-byte per the Task 10 template with the backend
  (never an LLM), timestamps converted to the application timezone.
* Write the artifact atomically: temporary file in ``data/temp/`` → flush →
  fsync → ``os.replace`` into ``data/exports/01_original/`` (AGENTS.md §12),
  then persist the ``weekly_exports`` row (path, sha256, week, sequence,
  revision) *after* the file write succeeded. On any persistence failure the
  freshly written file is removed again, so no orphaned "final" artifact and
  no row without a file survive (docs/data-lineage.md §5).

Path policy
===========

Paths are stored **relative to the data root**, with forward slashes:

- ``weekly_exports.path`` = ``exports/01_original/YYYY-MM-DD_original_NNNN_rN.txt``;
- the audio reference rendered in the TXT body = ``audio/<storage_path>``
  (the ``audio_files.storage_path`` column holds ``YYYY/MM/<uuid>.<ext>``
  relative to the audio directory, so ``audio/...`` is data-root relative
  too).

This mirrors the naming in ``docs/data-lineage.md`` §4 and lets the whole
``data/`` tree be relocated while every stored path stays resolvable by
joining it with the data root.

Privacy: no diary content is logged or placed in error messages; exception
messages are static (AGENTS.md §7–8).
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import date, datetime, tzinfo
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from diary.core.settings import get_settings
from diary.db.base import utc_now
from diary.db.models import Message, MessageKind, TranscriptionStatus, WeeklyExport
from diary.db.repositories import (
    AudioFileRepository,
    MessageRepository,
    TranscriptionRepository,
    WeeklyExportRepository,
)
from diary.domain.exports.weeks import is_sunday, week_bounds, week_monday

#: Layer-1 artifact directory below ``AppSettings.exports_dir``.
ORIGINAL_EXPORT_DIR = Path("01_original")
#: Artifact directory path relative to the data root (stored in the DB).
_ORIGINAL_DATA_ROOT_PREFIX = f"exports/{ORIGINAL_EXPORT_DIR}"

#: File suffix of original exports.
_EXPORT_SUFFIX = ".txt"


# ---------------------------------------------------------------------------
# Domain errors (static messages; mapped to HTTP responses at the API boundary)
# ---------------------------------------------------------------------------


class ExportError(Exception):
    """Base class for weekly-export domain errors (never diary content)."""


class InvalidSundayDateError(ExportError):
    """The week must be identified by an actual Sunday date."""


class EmptyWeekError(ExportError):
    """The week has no active messages; no export may be created."""


class WeekAlreadyExportedError(ExportError):
    """The week already has an export; a new revision needs an explicit re-export."""


class WeekNotExportedError(ExportError):
    """Re-export was requested for a week that has no export yet."""


class ExportNotFoundError(ExportError):
    """No weekly-export row exists for the given id."""


class ExportStorageError(ExportError):
    """The artifact file write or the persistence transaction failed."""


class ExportDataError(ExportError):
    """Internal inconsistency in the diary data (missing audio linkage)."""


# ---------------------------------------------------------------------------
# Rendered entry and pure helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportItem:
    """One diary entry to be rendered into the export text.

    For ``kind=audio`` entries ``body_text`` is empty (audio messages never
    carry text); the content comes from ``audio_path`` (data-root relative,
    e.g. ``audio/2026/09/<uuid>.wav``) and ``raw_stt`` (the newest succeeded
    raw transcription, or ``None`` when there is none → rendered as the
    ``[не расшифровано]`` marker).
    """

    created_at: datetime
    kind: MessageKind
    body_text: str
    audio_path: str | None = None
    raw_stt: str | None = None


def export_filename(sunday: date, seq_number: int, revision: int) -> str:
    """Original artifact file name per Task 10 naming rules.

    ``YYYY-MM-DD_original_NNNN_rN.txt`` where ``YYYY-MM-DD`` is the Sunday
    date, ``NNNN`` the zero-padded global sequence and ``rN`` the revision.
    """
    return f"{sunday.isoformat()}_original_{seq_number:04d}_r{revision}{_EXPORT_SUFFIX}"


def _text_lines(value: str) -> list[str]:
    """Split free text into rendered lines, dropping only trailing newlines.

    Interior line breaks (and blank lines inside the text) are preserved
    verbatim. A trailing line break would otherwise double as the blank
    separator between entries, so exactly the *trailing* breaks are dropped.
    """
    return value.rstrip("\n").split("\n")


def render_original_export(
    *,
    sunday: date,
    seq_number: int,
    revision: int,
    created_at: datetime,
    timezone_info: tzinfo,
    timezone_name: str,
    items: list[ExportItem],
) -> str:
    """Render the original weekly TXT byte-for-byte per the Task 10 template.

    ``created_at`` is the export creation instant (aware; converted to
    ``timezone_info`` for the ``Создано`` line). ``items`` are already in
    deterministic order and are emitted in the given order.
    """
    local_created = created_at.astimezone(timezone_info)
    lines: list[str] = [
        "# Дневник — оригинал",
        f"Неделя: {week_monday(sunday).isoformat()} — {sunday.isoformat()}",
        f"Дата воскресенья: {sunday.isoformat()}",
        f"Номер выгрузки: {seq_number:04d}",
        f"Ревизия: r{revision}",
        f"Создано: {local_created.isoformat(timespec='seconds')}",
        f"Timezone: {timezone_name}",
        "",
        "---",
        "",
    ]
    for index, item in enumerate(items):
        if index:
            lines.append("")  # exactly one blank line between entries.
        lines.extend(_item_lines(item, timezone_info))
    return "\n".join(lines) + "\n"


def _item_lines(item: ExportItem, timezone_info: tzinfo) -> list[str]:
    """Render one entry's content lines (no surrounding blank lines)."""
    stamp = item.created_at.astimezone(timezone_info).strftime("%Y-%m-%d %H:%M")
    if item.kind is MessageKind.audio:
        raw = item.raw_stt if item.raw_stt is not None else "[не расшифровано]"
        return [f"[{stamp}] AUDIO", f"Файл: {item.audio_path or ''}", "STT raw:"] + _text_lines(raw)
    return [f"[{stamp}] TEXT"] + _text_lines(item.body_text)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _unlink_safe(path: Path) -> None:
    """Remove a file, ignoring "already gone" errors."""
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort cleanup
        pass


def _write_export_atomically(data: bytes, target_path: Path, temp_dir: Path) -> None:
    """Write the export bytes atomically (AGENTS.md §12, data-lineage §5).

    A unique temporary file inside ``temp_dir`` is written, flushed and
    fsynced, then renamed over the final path. When the temporary directory
    and the target directory live on different filesystems (``rename`` raises
    ``EXDEV``) a copy+delete fallback is used — the payload is already fsynced,
    so durability does not depend on the rename being atomic. No partial file
    ever appears at the target path.
    """
    temp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp_path = temp_dir / f"export-{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(temp_path, flags, 0o600)
    except OSError:
        raise
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _unlink_safe(temp_path)
        raise
    try:
        os.replace(temp_path, target_path)
    except OSError as exc:
        _unlink_safe(temp_path)
        if exc.errno != errno.EXDEV:
            raise
        try:
            shutil.move(str(temp_path), str(target_path))
        except BaseException:
            _unlink_safe(target_path)
            _unlink_safe(temp_path)
            raise


def _remove_file_if_digest_matches(path: Path, expected_sha256: str) -> None:
    """Best-effort removal of a file only when its hash is still ours.

    After a failed DB commit the caller wants to clean up the file it just
    wrote; comparing the hash first keeps an identical-named file written by a
    concurrent request (same week/revision) from being deleted.
    """
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - file already gone
        return
    if digest == expected_sha256:
        _unlink_safe(path)


# ---------------------------------------------------------------------------
# Use cases
# ---------------------------------------------------------------------------


def export_week(
    session: Session,
    *,
    sunday_date: date,
    re_export: bool = False,
    now: datetime | None = None,
) -> WeeklyExport:
    """Create (or, with ``re_export``, revise) the original export of a week.

    The week is identified by its Sunday date ``sunday_date``. Without
    ``re_export`` the week must not have been exported yet and receives the
    next global sequence number (``max_sequence_number() + 1``, no gaps) at
    revision ``r1``. With ``re_export`` the week must already have an export:
    the new revision keeps the lineage sequence number and takes revision
    ``max(revisions) + 1``; previously written files are never modified.

    Raises :class:`InvalidSundayDateError` for a non-Sunday date,
    :class:`EmptyWeekError` when the week has no active messages (no file, no
    row and no sequence number are consumed), :class:`WeekAlreadyExportedError`
    / :class:`WeekNotExportedError` for numbering-state violations, and
    :class:`ExportStorageError` when the file write or the DB transaction
    fails. Commits the transaction (the file→row pair is the unit of work).
    """
    settings = get_settings()
    if not is_sunday(sunday_date):
        raise InvalidSundayDateError("The export week must be identified by its Sunday date.")
    timezone_info = settings.tzinfo
    if now is None:
        now = utc_now()
    if now.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime.")
    start, end = week_bounds(sunday_date, timezone_info)
    messages = MessageRepository(session).list_active_between(start=start, end=end)
    if not messages:
        raise EmptyWeekError("This week has no messages, so no export was created.")
    items = [_build_item(session, message) for message in messages]

    repository = WeeklyExportRepository(session)
    week_rows = repository.list_by_week(sunday_date)
    if re_export:
        if not week_rows:
            raise WeekNotExportedError(
                "This week has no export yet; create the first export first."
            )
        seq_number = week_rows[0].seq_number
        revision = max(row.revision for row in week_rows) + 1
    else:
        if week_rows:
            raise WeekAlreadyExportedError(
                "This week is already exported; use the re-export action to create a new revision."
            )
        maximum = repository.max_sequence_number()
        seq_number = 1 if maximum is None else maximum + 1
        revision = 1

    filename = export_filename(sunday_date, seq_number, revision)
    text = render_original_export(
        sunday=sunday_date,
        seq_number=seq_number,
        revision=revision,
        created_at=now,
        timezone_info=timezone_info,
        timezone_name=settings.timezone,
        items=items,
    )
    data = text.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    target_dir = settings.exports_dir / ORIGINAL_EXPORT_DIR
    target_path = target_dir / filename
    relative_path = f"{_ORIGINAL_DATA_ROOT_PREFIX}/{filename}"
    try:
        _write_export_atomically(data, target_path, settings.temp_dir)
    except OSError as exc:
        raise ExportStorageError("Failed to write the export file.") from exc

    try:
        record = repository.create(
            week_end_date=sunday_date,
            seq_number=seq_number,
            revision=revision,
            path=relative_path,
            sha256=digest,
        )
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        _remove_file_if_digest_matches(target_path, digest)
        raise WeekAlreadyExportedError("This week and revision are already exported.") from exc
    except Exception as exc:
        session.rollback()
        _remove_file_if_digest_matches(target_path, digest)
        raise ExportStorageError("Failed to persist the export metadata.") from exc
    return record


def _build_item(session: Session, message: Message) -> ExportItem:
    """Map one message to a renderable :class:`ExportItem`.

    Audio messages are resolved to their linked ``audio_files`` row
    (Task 07 stores exactly one per audio message) and to the newest
    *succeeded* transcription row. An audio message whose linkage is missing
    is an internal data inconsistency and fails the export loudly rather than
    producing a silently wrong entry.
    """
    if message.kind is not MessageKind.audio:
        return ExportItem(
            created_at=message.created_at, kind=message.kind, body_text=message.body_text
        )
    audio_files = AudioFileRepository(session).get_by_message(message.id)
    if not audio_files:
        raise ExportDataError("An audio message has no linked audio-file record.")
    audio = audio_files[-1]
    transcriptions = TranscriptionRepository(session).list_by_audio_file(audio.id)
    latest_success = next(
        (row for row in reversed(transcriptions) if row.status is TranscriptionStatus.succeeded),
        None,
    )
    return ExportItem(
        created_at=message.created_at,
        kind=message.kind,
        body_text=message.body_text,
        audio_path=f"audio/{audio.storage_path}",
        raw_stt=latest_success.raw_text if latest_success is not None else None,
    )


def list_exports(session: Session) -> list[WeeklyExport]:
    """Every weekly export, most recent week first, newest revision first.

    Ordering is deterministic: ``(week_end_date, revision)`` is unique per
    row, so no tie-breaker is needed. Used by the list API and the minimal
    UI page (Task 16 replaces the page with the full management UI).
    """
    rows = WeeklyExportRepository(session).list_all_by_sequence()
    return sorted(rows, key=lambda row: (row.week_end_date, row.revision), reverse=True)


def resolve_export_file(session: Session, export_id: uuid.UUID) -> tuple[Path, str]:
    """Resolve an export row to its absolute file path and download file name.

    The path always originates from the database row written by the server.
    As defense in depth the resolved path is checked to stay inside
    ``<exports_dir>/01_original``. Raises :class:`ExportNotFoundError` for an
    unknown id (or an out-of-tree stored path).
    """
    settings = get_settings()
    row = WeeklyExportRepository(session).get(export_id)
    if row is None:
        raise ExportNotFoundError("Export not found.")
    original_dir = (settings.exports_dir / ORIGINAL_EXPORT_DIR).resolve()
    candidate = (settings.data_root / row.path).resolve()
    if not candidate.is_relative_to(original_dir):
        # ``row.path`` is server-generated; this branch is purely defensive.
        raise ExportNotFoundError("Export not found.")
    return candidate, export_filename(row.week_end_date, row.seq_number, row.revision)


__all__ = [
    "ORIGINAL_EXPORT_DIR",
    "EmptyWeekError",
    "ExportDataError",
    "ExportError",
    "ExportItem",
    "ExportNotFoundError",
    "ExportStorageError",
    "InvalidSundayDateError",
    "WeekAlreadyExportedError",
    "WeekNotExportedError",
    "export_filename",
    "export_week",
    "list_exports",
    "render_original_export",
    "resolve_export_file",
]
