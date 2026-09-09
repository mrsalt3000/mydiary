"""Thin page routes for the chat UI (Task 06).

Mounted at the application root (no prefix; ``/api/v1`` stays the JSON API).

- ``GET /``                              — the chat page: first history window,
  the compose form and the "load older" control when older messages exist;
- ``GET /messages/fragments/older``      — HTML fragment of the next older
  window (htmx ``afterbegin`` into ``#message-list``) plus an out-of-band
  ``#older-control`` that carries the next cursor (or stops paging).

Rendering decisions
===================
* Timestamps are converted to the application timezone
  (``AppSettings.tzinfo``) and formatted as ``%d.%m.%Y %H:%M``; the ``<time
  datetime>`` attribute keeps the full ISO-8601 value for the page's own use.
* All user text is escaped by Jinja2 autoescape (default for ``.html``).
* The keyset cursor format is owned by the messages API
  (``diary.api.routes.messages``); this module imports the codec instead of
  duplicating it so page and API paging can never drift apart.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from diary.api.routes.messages import _decode_cursor, _encode_cursor, get_session
from diary.core.settings import get_settings
from diary.db.models import (
    ArtifactLayer,
    JobStatus,
    Message,
    MessageKind,
    TranscriptionStatus,
    WeeklyExport,
)
from diary.db.repositories import (
    ArtifactRepository,
    AudioFileRepository,
    TranscriptionRepository,
    WeeklyExportRepository,
)
from diary.domain.chat.service import list_messages
from diary.domain.exports.service import export_filename, list_exports
from diary.domain.transcription.service import find_active_transcription_job

#: This module lives at ``src/diary/api/routes/pages.py``.
_PACKAGE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = _PACKAGE_DIR / "templates"
STATIC_DIR = _PACKAGE_DIR / "static"

#: History window size: messages per page and per "older" fragment.
PAGE_LIMIT = 50

#: Wall-clock display format for message timestamps (application timezone).
#: Numeric-only codes: the rendering does not depend on the process locale.
TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M"

#: Human-readable transcription-state captions for audio messages.
TRANSCRIPTION_LABELS: dict[TranscriptionStatus, str] = {
    TranscriptionStatus.not_requested: "Не расшифровано",
    TranscriptionStatus.queued: "Ожидает расшифровки",
    TranscriptionStatus.running: "Расшифровывается…",
    TranscriptionStatus.succeeded: "Расшифровано",
    TranscriptionStatus.failed: "Ошибка расшифровки",
}

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

pages_router = APIRouter(tags=["pages"])


def _audio_view(session: Session, message_id: uuid.UUID) -> dict[str, Any] | None:
    """Template fields for the audio file of one ``kind=audio`` message.

    Task 07 stores exactly one ``audio_files`` row per audio message; the row
    is looked up by message id (never from any user-supplied path). Task 09
    adds the transcription panel state:

    - an in-flight ``transcribe_audio`` job (pending/retrying → shown as
      ``queued``, running → ``running``) overrides the row-derived status, so
      a page reload during processing still shows the busy state and the
      client can resume polling via ``active_job_id``;
    - otherwise the status comes from the newest ``transcriptions`` row
      (``not_requested`` before the first manual run, ``succeeded``/``failed``
      afterwards);
    - ``transcription_raw`` is the text of the newest succeeded row and is
      rendered (escaped) below the audio player; history beyond the latest
      version is served by ``GET /api/v1/audio/{id}/transcriptions``.

    Returns ``None`` for text messages or when the linkage is missing (the
    template then renders the message body only).
    """
    audio_files = AudioFileRepository(session).get_by_message(message_id)
    if not audio_files:
        return None
    audio = audio_files[-1]
    transcriptions = TranscriptionRepository(session).list_by_audio_file(audio.id)
    active_job = find_active_transcription_job(session, audio.id)
    if active_job is not None:
        # pending/retrying → queued; running → running.
        status = (
            TranscriptionStatus.running
            if active_job.status is JobStatus.running
            else TranscriptionStatus.queued
        )
    else:
        status = transcriptions[-1].status if transcriptions else TranscriptionStatus.not_requested
    latest_success = next(
        (row for row in reversed(transcriptions) if row.status is TranscriptionStatus.succeeded),
        None,
    )
    return {
        "audio_file_id": str(audio.id),
        "audio_url": f"/api/v1/audio/{audio.id}/file",
        "audio_mime": audio.original_mime,
        "transcription_status": status.value,
        "transcription_label": TRANSCRIPTION_LABELS.get(status, "Не расшифровано"),
        "transcription_raw": latest_success.raw_text if latest_success is not None else None,
        "transcription_button_label": (
            "Перерасшифровать" if latest_success is not None else "Расшифровать"
        ),
        "active_job_id": str(active_job.id) if active_job is not None else None,
    }


def _message_view(record: Message, session: Session) -> dict[str, Any]:
    """Map a message to template-ready fields in the application timezone.

    Audio messages additionally carry a server-generated ``audio_url`` and the
    transcription-state caption; all values originate from the database, never
    from user input, and are escaped by Jinja2 at render time.
    """
    settings = get_settings()
    created_local = record.created_at.astimezone(settings.tzinfo)
    view: dict[str, Any] = {
        "id": str(record.id),
        "kind": record.kind.value,
        "created_at_iso": created_local.isoformat(),
        "created_at_display": created_local.strftime(TIMESTAMP_FORMAT),
        "body": record.body_text,
    }
    if record.kind is MessageKind.audio:
        audio = _audio_view(session, record.id)
        if audio is not None:
            view.update(audio)
    return view


@pages_router.get("/", response_class=HTMLResponse)
def chat_page(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Render the chat page with the first (most recent) history window."""
    messages, next_cursor = list_messages(session, limit=PAGE_LIMIT)
    older_cursor = _encode_cursor(next_cursor) if next_cursor is not None else None
    settings = get_settings()
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "messages": [_message_view(message, session) for message in messages],
            "has_more": next_cursor is not None,
            # Pre-quoted for the hx-get URL; Jinja never sees raw "+" or "|".
            "older_cursor": quote(older_cursor, safe="") if older_cursor else None,
            "max_message_length": settings.max_message_length,
            "max_audio_size_mb": settings.max_audio_size_mb,
            # Client-side ``accept`` for the file-upload fallback.
            "audio_accept": ",".join(settings.allowed_audio_mime_types),
        },
    )


def _export_view(record: WeeklyExport) -> dict[str, Any]:
    """Template fields for one weekly export (minimal Task 10 UI page).

    The full exports-management page with actions/confirmations is Task 16;
    this page only lists created exports and offers a download link. ``sunday``
    is the week's Sunday date and ``filename`` is derived server-side.
    """
    settings = get_settings()
    created_local = record.created_at.astimezone(settings.tzinfo)
    return {
        "id": str(record.id),
        "sunday": record.week_end_date,
        "monday": record.week_end_date - timedelta(days=6),
        "seq_number": record.seq_number,
        "revision": record.revision,
        "filename": export_filename(record.week_end_date, record.seq_number, record.revision),
        "created_at_display": created_local.strftime(TIMESTAMP_FORMAT),
        "download_url": f"/api/v1/exports/{record.id}/file",
        "compare_url": f"/exports/{record.id}/normalize",
    }


@pages_router.get("/exports", response_class=HTMLResponse)
def exports_page(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Minimal exports list page with download links (Task 10/16 boundary).

    Lists every original weekly export (most recent week first). Full
    management UI (creation actions, re-export, lineage views) is Task 16.
    """
    exports = [_export_view(record) for record in list_exports(session)]
    return templates.TemplateResponse(request, "exports.html", {"exports": exports})


@pages_router.get("/exports/{export_id}/normalize", response_class=HTMLResponse)
def export_normalize_compare_page(
    export_id: uuid.UUID,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Minimal original↔normalized comparison page for one export (Task 13).

    Shows the original layer-1 text (read from the export row's file) and the
    newest normalized layer-2 version when one exists; diary text is escaped
    by Jinja2. The original download link is the existing Task-10 API route.
    The full exports-management page with actions is Task 16.
    """
    record = WeeklyExportRepository(session).get(export_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Export not found.")
    settings = get_settings()
    original_path = settings.data_root / record.path
    if not original_path.is_file():
        raise HTTPException(status_code=404, detail="Export file not found.")
    artifacts = ArtifactRepository(session).list_by_export_and_layer(
        export_id, ArtifactLayer.normalized
    )
    latest = artifacts[-1] if artifacts else None
    normalized_text: str | None = None
    normalized_filename: str | None = None
    if latest is not None:
        normalized_path = settings.data_root / latest.path
        if normalized_path.is_file():
            normalized_text = normalized_path.read_text(encoding="utf-8")
            normalized_filename = latest.path.rsplit("/", 1)[-1]
    return templates.TemplateResponse(
        request,
        "export_compare.html",
        {
            "export": _export_view(record),
            "original_text": original_path.read_text(encoding="utf-8"),
            "normalized_text": normalized_text,
            "normalized_filename": normalized_filename,
        },
    )


@pages_router.get("/messages/fragments/older", response_class=HTMLResponse)
def older_messages_fragment(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    cursor: Annotated[str, Query(max_length=256)],
    limit: Annotated[int, Query(ge=1, le=200)] = PAGE_LIMIT,
) -> HTMLResponse:
    """Return the next older messages window as an HTML fragment.

    The cursor points at the oldest message already shown; the response holds
    the messages strictly older than it in chronological order (they are
    prepended to ``#message-list``), followed by an out-of-band replacement of
    the ``#older-control`` element with the next cursor.
    """
    try:
        position = _decode_cursor(cursor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    messages, next_cursor = list_messages(session, limit=limit, cursor=position)
    older_cursor = _encode_cursor(next_cursor) if next_cursor is not None else None
    return templates.TemplateResponse(
        request,
        "_older_messages.html",
        {
            "messages": [_message_view(message, session) for message in messages],
            "has_more": next_cursor is not None,
            "older_cursor": quote(older_cursor, safe="") if older_cursor else None,
        },
    )
