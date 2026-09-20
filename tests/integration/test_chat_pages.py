"""Integration tests for the chat UI pages (Task 06).

Uses an isolated, migrated SQLite test database (fixtures from
``tests/integration/conftest.py``) plus synthetic data only. Covers: the chat
page render (form, empty state, created messages after a refresh), safe
escaping of user HTML, timestamps rendered in the application timezone,
progressive loading of older messages through the fragment endpoint (order,
next-cursor management, history end) and the no-external-resources / safe-DOM
rules of the client assets.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import diary.api.routes.pages as pages_module
from diary.api.routes.messages import get_session
from diary.core.settings import reload_settings
from diary.db.models import MessageKind
from diary.db.repositories import MessageRepository
from diary.main import app

#: Repository root: tests/integration/test_chat_pages.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]

OLDER_CURSOR_RE = re.compile(r'hx-get="/messages/fragments/older\?cursor=([^"]+)"')


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with default application settings."""
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


def _at(hour: int) -> datetime:
    return datetime(2026, 9, 10, hour, tzinfo=timezone.utc)


def _create(session: Session, body: str, created_at: datetime):
    return MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=created_at
    )


# ---------------------------------------------------------------------------
# chat page render
# ---------------------------------------------------------------------------


def test_chat_page_renders_form_and_empty_state(api_client: TestClient) -> None:
    response = api_client.get("/")
    assert response.status_code == 200
    html = response.text
    assert '<html lang="ru">' in html
    assert 'id="compose-form"' in html
    assert 'data-max-length="20000"' in html
    assert 'id="message-text"' in html
    assert "Пока нет записей" in html
    assert 'id="message-empty"' in html
    assert "Показать более старые записи" not in html
    assert 'class="message"' not in html


def test_chat_page_shows_messages_after_creation_and_refresh(
    api_client: TestClient,
) -> None:
    created = api_client.post("/api/v1/messages", json={"body": "Синтетическая запись."})
    assert created.status_code == 201

    for _ in range(2):  # refresh must keep showing the message
        response = api_client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "Синтетическая запись." in html
        assert 'class="message"' in html
        assert 'id="message-empty"' not in html
        assert re.search(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}", html) is not None


# ---------------------------------------------------------------------------
# escaping
# ---------------------------------------------------------------------------


def test_chat_page_escapes_user_html(api_client: TestClient, session: Session) -> None:
    payload = '<script>alert("xss")</script><b>bold</b>'
    _create(session, payload, _at(8))
    session.commit()

    html = api_client.get("/").text
    assert "&lt;script&gt;alert(&#34;xss&#34;)&lt;/script&gt;" in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html
    assert '<script>alert("xss")</script>' not in html
    assert "<b>bold</b>" not in html


# ---------------------------------------------------------------------------
# timestamps in the application timezone
# ---------------------------------------------------------------------------


def test_chat_page_renders_timestamps_in_application_timezone(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Asia/Kathmandu is UTC+05:45: a non-whole-hour offset, so naive rendering
    # could not pass by accident.
    monkeypatch.setenv("APP_TIMEZONE", "Asia/Kathmandu")
    reload_settings()
    try:
        _create(session, "Запись с известным UTC", datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
        session.commit()

        html = api_client.get("/").text
        assert 'datetime="2026-01-01T05:45:00+05:45"' in html
        assert "01.01.2026 05:45" in html
    finally:
        reload_settings()


# ---------------------------------------------------------------------------
# progressive loading of older messages
# ---------------------------------------------------------------------------


def test_chat_page_shows_older_control_only_when_older_messages_exist(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pages_module, "PAGE_LIMIT", 3)
    _create(session, "запись-1", _at(8))
    _create(session, "запись-2", _at(9))
    session.commit()

    html = api_client.get("/").text
    assert "запись-1" in html
    assert "запись-2" in html
    assert "Показать более старые записи" not in html


def test_older_fragment_walks_history_in_order_without_gaps(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pages_module, "PAGE_LIMIT", 3)
    for index in range(1, 9):
        _create(session, f"запись-{index}", _at(7 + index))
    session.commit()

    # First page: the most recent window (запись-6, -7, -8), oldest first.
    page = api_client.get("/")
    assert page.status_code == 200
    html = page.text
    assert "Показать более старые записи" in html
    positions = [html.index(f"запись-{index}") for index in (6, 7, 8)]
    assert positions == sorted(positions)

    match = OLDER_CURSOR_RE.search(html)
    assert match is not None
    cursor = urllib.parse.unquote(match.group(1))

    collected: list[str] = []
    seen_cursors: list[str] = []
    for _ in range(10):  # hard guard against an infinite walk
        fragment = api_client.get(
            "/messages/fragments/older", params={"cursor": cursor, "limit": "3"}
        )
        assert fragment.status_code == 200
        body = fragment.text
        seen_cursors.append(cursor)

        collected.extend(re.findall(r"запись-(\d)", body))
        assert 'hx-swap-oob="outerHTML"' in body
        assert 'id="older-control"' in body

        next_match = OLDER_CURSOR_RE.search(body)
        if next_match is None:
            assert "Начало истории" in body
            break
        cursor = urllib.parse.unquote(next_match.group(1))
    else:  # pragma: no cover - the guard must never trigger
        pytest.fail("older fragment walk did not terminate")

    # The whole history is returned exactly once, oldest window last.
    assert sorted(int(value) for value in collected) == list(range(1, 6))
    assert len(collected) == 5
    assert len(set(collected)) == len(collected)

    # Each cursor points strictly further into the past.
    def cursor_hour(value: str) -> int:
        return datetime.fromisoformat(value.split("|", 1)[0]).hour

    assert len(seen_cursors) >= 2
    hours = [cursor_hour(value) for value in seen_cursors]
    assert all(earlier > later for earlier, later in zip(hours, hours[1:]))


def test_older_fragment_with_exhausted_cursor_returns_empty_end(
    api_client: TestClient,
    session: Session,
) -> None:
    created = _create(session, "единственная", _at(8))
    session.commit()
    cursor = f"{_at(8).isoformat()}|{created.id}"

    fragment = api_client.get("/messages/fragments/older", params={"cursor": cursor, "limit": "3"})
    assert fragment.status_code == 200
    body = fragment.text
    assert "единственная" not in body
    assert "Начало истории" in body
    assert OLDER_CURSOR_RE.search(body) is None


def test_older_fragment_rejects_invalid_cursor(api_client: TestClient) -> None:
    response = api_client.get("/messages/fragments/older", params={"cursor": "not-a-cursor"})
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid cursor."}


def test_older_fragment_missing_cursor_returns_422(api_client: TestClient) -> None:
    response = api_client.get("/messages/fragments/older")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# form validation path and client safety rules
# ---------------------------------------------------------------------------


def test_form_validation_backstop_returns_422_without_echo(
    api_client: TestClient,
) -> None:
    """The API the compose form posts to rejects empty/oversized bodies with a
    clear status and never echoes submitted diary content."""
    for body in ("", "   "):
        response = api_client.post("/api/v1/messages", json={"body": body})
        assert response.status_code == 422
        assert response.json()["detail"]  # non-empty, static error payload

    oversized = "д" * 20001
    response = api_client.post("/api/v1/messages", json={"body": oversized})
    assert response.status_code == 422
    assert oversized[:64] not in response.text


def test_client_assets_follow_privacy_and_escaping_rules() -> None:
    """The chat page must not reference external hosts, and the client code
    must place diary text into the DOM only via ``textContent``."""
    template_root = REPO_ROOT / "src/diary/templates"
    for path in sorted(template_root.rglob("*.html")):
        assert re.search(r"https?://", path.read_text(encoding="utf-8")) is None
    css_root = REPO_ROOT / "src/diary/static"
    for path in sorted(css_root.rglob("*.css")):
        assert re.search(r"https?://", path.read_text(encoding="utf-8")) is None

    js_path = css_root / "js" / "app.js"
    script = js_path.read_text(encoding="utf-8")
    for sink in ("innerHTML", "insertAdjacentHTML", "outerHTML", "document.write"):
        assert sink not in script
    assert "textContent" in script
    assert "Запись не может быть пустой" in script
    assert "слишком длинная" in script


def test_recording_timer_keeps_the_indicator_accessible() -> None:
    """T14: the recording indicator may show a live duration, but the
    ``role="status"`` region must stay a one-time announcement.

    This guards the static contract only. Whether repeated record/stop cycles
    strand an interval — and whether the value actually ticks — are runtime
    properties, verified by driving the page in a browser, not by reading the
    source.
    """
    template = (REPO_ROOT / "src/diary/templates/chat.html").read_text(encoding="utf-8")
    indicator = re.search(r'<span id="voice-recording".*?</span>', template, re.DOTALL)
    assert indicator is not None, "the recording indicator must stay in the template"
    markup = indicator.group(0)
    assert 'role="status"' in markup
    assert "Идёт запись…" in markup
    elapsed = re.search(r'<span id="voice-elapsed"[^>]*>', markup)
    assert elapsed is not None, "the ticking duration must live in its own element"
    # role="status" is an implicit polite live region: a value left in the
    # accessibility tree would be announced on every tick.
    assert 'aria-hidden="true"' in elapsed.group(0)

    script = (REPO_ROOT / "src/diary/static/js/app.js").read_text(encoding="utf-8")
    assert "window.performance.now()" in script
    assert "window.clearInterval" in script
    assert "pagehide" in script

    css = (REPO_ROOT / "src/diary/static/css/app.css").read_text(encoding="utf-8")
    rule = re.search(r"\.voice-recording\s*\{[^}]*\}", css)
    assert rule is not None
    assert "font-variant-numeric: tabular-nums" in rule.group(0)


def test_app_serves_local_static_assets(api_client: TestClient) -> None:
    for url_path in (
        "/static/css/app.css",
        "/static/js/app.js",
        "/static/vendor/htmx.min.js",
    ):
        response = api_client.get(url_path)
        assert response.status_code == 200
        assert len(response.content) > 0
