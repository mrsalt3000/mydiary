"""Thin HTTP routes for diary messages (Task 05).

Each handler performs HTTP validation, delegates to
:mod:`diary.domain.chat.service` and maps the result (or a domain error) to an
HTTP response. No business logic and no direct session/model access here.

- ``POST /api/v1/messages``        — create a text message (201);
- ``GET /api/v1/messages``         — cursor history, newest page first (200);
- ``DELETE /api/v1/messages/{id}`` — soft delete (204; 404 when unknown or
  already deleted; 422 for a malformed UUID).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session, sessionmaker

from diary.api.schemas import MessageCreate, MessagePage, MessageRead, message_read_from
from diary.core.settings import get_settings
from diary.db.session import create_engine, create_session_factory
from diary.domain.chat.service import (
    MessageCursor,
    MessageNotFoundError,
    create_text_message,
    list_messages,
    soft_delete_message,
)

router = APIRouter(prefix="/messages", tags=["messages"])


# ---------------------------------------------------------------------------
# Session dependency
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _app_session_factory() -> sessionmaker[Session]:
    """Session factory over the application database (engine built lazily)."""
    return create_session_factory(create_engine(settings=get_settings()))


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one SQLAlchemy ``Session`` per request."""
    with _app_session_factory()() as session:
        yield session


# ---------------------------------------------------------------------------
# Cursor codec (representation detail)
# ---------------------------------------------------------------------------


def _encode_cursor(position: MessageCursor) -> str:
    created_at, message_id = position
    return f"{created_at.isoformat()}|{message_id}"


def _decode_cursor(value: str) -> MessageCursor:
    """Parse a cursor back into a keyset position; raises ``ValueError``."""
    try:
        created_at_raw, message_id_raw = value.split("|", 1)
        created_at = datetime.fromisoformat(created_at_raw)
        message_id = uuid.UUID(message_id_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid cursor.") from exc
    if created_at.tzinfo is None:
        raise ValueError("Invalid cursor.")
    return created_at, message_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=MessageRead, status_code=201)
def create_message(
    payload: MessageCreate,
    session: Annotated[Session, Depends(get_session)],
) -> MessageRead:
    """Create a new ``kind=text`` diary message."""
    message = create_text_message(session, payload.body)
    return message_read_from(message)


@router.get("", response_model=MessagePage)
def read_messages(
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(max_length=256)] = None,
) -> MessagePage:
    """Return the message history as a cursor page.

    Items are in deterministic chronological order. The first page holds the
    most recent messages; ``next_cursor`` walks towards older messages.
    """
    position: MessageCursor | None = None
    if cursor is not None:
        try:
            position = _decode_cursor(cursor)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    messages, next_position = list_messages(session, limit=limit, cursor=position)
    return MessagePage(
        items=[message_read_from(message) for message in messages],
        next_cursor=_encode_cursor(next_position) if next_position is not None else None,
        limit=limit,
    )


@router.delete("/{message_id}", status_code=204)
def delete_message(
    message_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> Response:
    """Soft-delete a message (deletion policy documented in the chat service)."""
    try:
        soft_delete_message(session, message_id)
    except MessageNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Message not found.") from exc
    return Response(status_code=204)
