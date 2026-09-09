"""Unit tests for the messages API schemas (Task 05).

Covers the validation contract of ``MessageCreate`` (empty after strip and
oversized bodies are rejected with static messages) and the ISO-8601
serialization of ``MessageRead`` timestamps. Synthetic data only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from diary.api.schemas import MessageCreate, MessageRead
from diary.core.settings import reload_settings


@pytest.fixture(autouse=True)
def _isolate_max_message_length(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ignore any ``APP_MAX_MESSAGE_LENGTH`` from the developer shell and
    rebuild the cached settings before and after each test."""
    monkeypatch.delenv("APP_MAX_MESSAGE_LENGTH", raising=False)
    reload_settings()
    yield
    reload_settings()


def _with_max_length(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    monkeypatch.setenv("APP_MAX_MESSAGE_LENGTH", str(value))
    reload_settings()


def test_empty_body_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MessageCreate(body="")


def test_whitespace_only_body_is_rejected() -> None:
    for body in ("   ", "\n\t", "  \n  "):
        with pytest.raises(ValidationError):
            MessageCreate(body=body)


def test_body_longer_than_max_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_max_length(monkeypatch, 10)
    with pytest.raises(ValidationError):
        MessageCreate(body="x" * 11)


def test_body_exactly_at_max_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_max_length(monkeypatch, 10)
    message = MessageCreate(body="x" * 10)
    assert message.body == "x" * 10


def test_valid_body_is_kept_as_submitted() -> None:
    body = "  Запись  без  изменений  "
    message = MessageCreate(body=body)
    # Original user text is stored exactly as entered (immutable source data).
    assert message.body == body


def test_error_messages_do_not_echo_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_max_length(monkeypatch, 10)
    with pytest.raises(ValidationError) as excinfo:
        MessageCreate(body="секретный текст, который не должен попасть в ошибку")
    error_text = str(excinfo.value)
    assert "секретный текст" not in error_text
    assert "no longer than 10 characters" in error_text


def test_message_read_serializes_created_at_with_offset() -> None:
    created_local = datetime(2026, 1, 1, 5, 45, tzinfo=ZoneInfo("Asia/Kathmandu"))
    message = MessageRead(
        id=uuid4(),
        kind="text",
        body="Синтетика",
        created_at=created_local,
    )
    payload = message.model_dump_json()
    assert '"created_at":"2026-01-01T05:45:00+05:45"' in payload
    assert message.created_at.tzinfo is not None
    assert message.created_at.utcoffset() is not None


def test_message_read_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        MessageRead(
            id=uuid4(),
            kind="telegram",
            body="Синтетика",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
