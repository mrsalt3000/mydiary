"""Integration tests for the messages API (Task 05).

Uses an isolated, migrated SQLite test database (fixtures from
``tests/integration/conftest.py``) plus synthetic data only. Covers: message
creation, persistence across an engine restart, deterministic history
ordering (including same-second tie-breaks by id), cursor pagination without
gaps or duplicates, application-timezone rendering of timestamps, the soft
delete policy, and validation errors that never echo submitted content.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import reload_settings
from diary.db.models import MessageKind
from diary.db.repositories import MessageRepository
from diary.db.session import create_engine, create_session_factory
from diary.main import app

MESSAGES_URL = "/api/v1/messages"


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with default application settings.

    Rebuilds the cached settings so tests never depend on environment or on
    the ordering of other test modules.
    """
    monkeypatch.delenv("APP_TIMEZONE", raising=False)
    monkeypatch.delenv("APP_MAX_MESSAGE_LENGTH", raising=False)
    reload_settings()


@contextmanager
def _client_bound_to(session_factory) -> Iterator[TestClient]:
    """A TestClient whose ``get_session`` dependency serves ``session_factory``."""

    def _override_session() -> Iterator[Session]:
        with session_factory() as current_session:
            yield current_session

    app.dependency_overrides[get_session] = _override_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_session, None)


@pytest.fixture
def api_client(session_factory) -> Iterator[TestClient]:
    """TestClient bound to the per-test migrated database."""
    with _client_bound_to(session_factory) as client:
        yield client


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 10, hour, minute, tzinfo=timezone.utc)


def _create_via_repository(session: Session, body: str, created_at: datetime):
    return MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=created_at
    )


# ---------------------------------------------------------------------------
# creation
# ---------------------------------------------------------------------------


def test_post_creates_text_message(api_client: TestClient, session: Session) -> None:
    response = api_client.post(MESSAGES_URL, json={"body": "Синтетическая запись."})
    assert response.status_code == 201
    payload = response.json()
    assert payload["kind"] == "text"
    assert payload["body"] == "Синтетическая запись."
    # Default application timezone is Europe/Moscow (UTC+03:00, no DST).
    assert payload["created_at"].endswith("+03:00")
    datetime.fromisoformat(payload["created_at"])

    message_id = uuid.UUID(payload["id"])
    stored = MessageRepository(session).get(message_id)
    assert stored is not None
    assert stored.kind is MessageKind.text
    assert stored.body_text == "Синтетическая запись."
    assert stored.deleted_at is None


def test_message_survives_engine_restart(run_upgrade, db_url: str) -> None:
    run_upgrade(db_url)

    first_engine = create_engine(db_url)
    try:
        first_factory = create_session_factory(first_engine)
        with _client_bound_to(first_factory) as client:
            response = client.post(MESSAGES_URL, json={"body": "Переживёт рестарт приложения."})
            assert response.status_code == 201
            message_id = response.json()["id"]
    finally:
        first_engine.dispose()

    # A brand-new engine/session over the same file must still see the message.
    second_engine = create_engine(db_url)
    try:
        second_factory = create_session_factory(second_engine)
        with _client_bound_to(second_factory) as client:
            history = client.get(MESSAGES_URL)
            assert history.status_code == 200
            items = history.json()["items"]
            assert [item["id"] for item in items] == [message_id]
            assert items[0]["body"] == "Переживёт рестарт приложения."
        with second_factory() as session:
            stored = MessageRepository(session).get(uuid.UUID(message_id))
            assert stored is not None
            assert stored.deleted_at is None
    finally:
        second_engine.dispose()


# ---------------------------------------------------------------------------
# history ordering
# ---------------------------------------------------------------------------


def test_history_is_deterministic_chronological_with_id_tiebreak(
    api_client: TestClient, session: Session
) -> None:
    created = {
        "m1": _at(8),
        "m2": _at(9, 30),
        "m3": _at(12),
    }
    created_at_of: dict[uuid.UUID, datetime] = {}
    for body, timestamp in created.items():
        row = _create_via_repository(session, body, timestamp)
        created_at_of[row.id] = timestamp
    # A same-second cluster: the deterministic tie-breaker is the id.
    cluster = [_create_via_repository(session, f"cluster-{i}", _at(13)) for i in range(3)]
    for row in cluster:
        created_at_of[row.id] = _at(13)
    session.commit()

    first = api_client.get(MESSAGES_URL)
    second = api_client.get(MESSAGES_URL)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()  # deterministic across requests

    items = first.json()["items"]
    returned_ids = [uuid.UUID(item["id"]) for item in items]
    assert len(returned_ids) == len(created_at_of)
    assert len(set(returned_ids)) == len(returned_ids)

    expected_order = sorted(returned_ids, key=lambda mid: (created_at_of[mid], mid))
    assert returned_ids == expected_order


def test_cursor_pagination_has_no_gaps_or_duplicates(
    api_client: TestClient, session: Session
) -> None:
    bodies = [f"запись-{index}" for index in range(1, 8)]
    rows = [
        _create_via_repository(session, body, _at(8 + index)) for index, body in enumerate(bodies)
    ]
    session.commit()
    all_ids = {str(row.id) for row in rows}

    collected: list[str] = []
    cursor: str | None = None
    page_sizes: list[int] = []
    for _ in range(10):  # hard guard against an infinite walk
        params = {"limit": "3"}
        if cursor is not None:
            params["cursor"] = cursor
        response = api_client.get(MESSAGES_URL, params=params)
        assert response.status_code == 200
        payload = response.json()
        page_sizes.append(len(payload["items"]))
        for item in payload["items"]:
            collected.append(item["id"])
        cursor = payload["next_cursor"]
        if cursor is None:
            break

    assert len(collected) == len(all_ids)
    assert set(collected) == all_ids  # no gaps
    assert len(set(collected)) == len(collected)  # no duplicates
    # Each page is internally chronological.
    for size in page_sizes:
        assert 0 < size <= 3


def test_pagination_stops_at_history_start(api_client: TestClient, session: Session) -> None:
    for index in range(6):
        _create_via_repository(session, f"запись-{index}", _at(8 + index))
    session.commit()

    first = api_client.get(MESSAGES_URL, params={"limit": "3"})
    assert first.status_code == 200
    assert len(first.json()["items"]) == 3
    assert first.json()["next_cursor"] is not None

    second = api_client.get(
        MESSAGES_URL, params={"limit": "3", "cursor": first.json()["next_cursor"]}
    )
    assert second.status_code == 200
    # A full final page must still report no next cursor.
    assert len(second.json()["items"]) == 3
    assert second.json()["next_cursor"] is None


def test_empty_history_returns_empty_page(api_client: TestClient) -> None:
    response = api_client.get(MESSAGES_URL)
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None, "limit": 50}


def test_invalid_cursor_returns_422(api_client: TestClient) -> None:
    response = api_client.get(MESSAGES_URL, params={"cursor": "not-a-cursor"})
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid cursor."}


# ---------------------------------------------------------------------------
# timezone rendering
# ---------------------------------------------------------------------------


def test_created_at_rendered_in_application_timezone(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Asia/Kathmandu is UTC+05:45: an offset that is not a whole hour, so a
    # naive (offset-less) rendering could not pass by accident.
    monkeypatch.setenv("APP_TIMEZONE", "Asia/Kathmandu")
    reload_settings()
    try:
        _create_via_repository(
            session, "Запись с известным UTC", datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        )
        session.commit()

        history = api_client.get(MESSAGES_URL)
        assert history.status_code == 200
        items = history.json()["items"]
        assert items[0]["created_at"] == "2026-01-01T05:45:00+05:45"
    finally:
        reload_settings()


# ---------------------------------------------------------------------------
# soft delete policy
# ---------------------------------------------------------------------------


def test_delete_hides_message_but_keeps_row(api_client: TestClient, session: Session) -> None:
    created = api_client.post(MESSAGES_URL, json={"body": "Удаляемая запись."}).json()
    message_id = created["id"]

    deleted = api_client.delete(f"{MESSAGES_URL}/{message_id}")
    assert deleted.status_code == 204
    assert deleted.content == b""

    history = api_client.get(MESSAGES_URL)
    assert history.status_code == 200
    assert history.json()["items"] == []

    row = MessageRepository(session).get(uuid.UUID(message_id))
    assert row is not None
    assert row.deleted_at is not None
    assert row.deleted_at.tzinfo is timezone.utc
    assert row.body_text == "Удаляемая запись."  # immutable source text intact


def test_repeat_delete_returns_404(api_client: TestClient) -> None:
    created = api_client.post(MESSAGES_URL, json={"body": "Удаляется дважды."}).json()
    message_id = created["id"]

    assert api_client.delete(f"{MESSAGES_URL}/{message_id}").status_code == 204
    repeated = api_client.delete(f"{MESSAGES_URL}/{message_id}")
    assert repeated.status_code == 404
    assert repeated.json() == {"detail": "Message not found."}


def test_delete_unknown_message_returns_404(api_client: TestClient) -> None:
    response = api_client.delete(f"{MESSAGES_URL}/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Message not found."}


def test_delete_malformed_message_id_returns_422(api_client: TestClient) -> None:
    response = api_client.delete(f"{MESSAGES_URL}/not-a-uuid")
    assert response.status_code == 422


def test_deleted_message_not_returned_by_cursor_walk(
    api_client: TestClient, session: Session
) -> None:
    keep = _create_via_repository(session, "останется", _at(8))
    remove = _create_via_repository(session, "удалится", _at(9))
    session.commit()

    response = api_client.delete(f"{MESSAGES_URL}/{remove.id}")
    assert response.status_code == 204

    page = api_client.get(MESSAGES_URL, params={"limit": "1"}).json()
    assert page["items"][0]["id"] == str(keep.id)
    assert page["next_cursor"] is None


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------


def test_post_rejects_empty_and_whitespace_body(api_client: TestClient) -> None:
    for body in ("", "   ", "\n\t  "):
        response = api_client.post(MESSAGES_URL, json={"body": body})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "body"]


def test_post_rejects_oversized_body_without_echoing_it(
    api_client: TestClient,
) -> None:
    oversized = "д" * 20001  # one character above the default maximum
    response = api_client.post(MESSAGES_URL, json={"body": oversized})
    assert response.status_code == 422
    body = response.text
    assert oversized[:64] not in body  # content is never echoed back
    assert "no longer than 20000 characters" in body
