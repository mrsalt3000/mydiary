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

from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from diary.api.routes.messages import _decode_cursor, _encode_cursor, get_session
from diary.core.settings import get_settings
from diary.db.models import Message
from diary.domain.chat.service import list_messages

#: This module lives at ``src/diary/api/routes/pages.py``.
_PACKAGE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = _PACKAGE_DIR / "templates"
STATIC_DIR = _PACKAGE_DIR / "static"

#: History window size: messages per page and per "older" fragment.
PAGE_LIMIT = 50

#: Wall-clock display format for message timestamps (application timezone).
#: Numeric-only codes: the rendering does not depend on the process locale.
TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

pages_router = APIRouter(tags=["pages"])


def _message_view(record: Message) -> dict[str, str]:
    """Map a message to template-ready fields in the application timezone."""
    settings = get_settings()
    created_local = record.created_at.astimezone(settings.tzinfo)
    return {
        "id": str(record.id),
        "created_at_iso": created_local.isoformat(),
        "created_at_display": created_local.strftime(TIMESTAMP_FORMAT),
        "body": record.body_text,
    }


@pages_router.get("/", response_class=HTMLResponse)
def chat_page(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Render the chat page with the first (most recent) history window."""
    messages, next_cursor = list_messages(session, limit=PAGE_LIMIT)
    older_cursor = _encode_cursor(next_cursor) if next_cursor is not None else None
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "messages": [_message_view(message) for message in messages],
            "has_more": next_cursor is not None,
            # Pre-quoted for the hx-get URL; Jinja never sees raw "+" or "|".
            "older_cursor": quote(older_cursor, safe="") if older_cursor else None,
            "max_message_length": get_settings().max_message_length,
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
            "messages": [_message_view(message) for message in messages],
            "has_more": next_cursor is not None,
            "older_cursor": quote(older_cursor, safe="") if older_cursor else None,
        },
    )
