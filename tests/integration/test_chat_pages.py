"""Integration tests for the diary day page (Task 06, day feed in T16b).

Uses an isolated, migrated SQLite test database (fixtures from
``tests/integration/conftest.py``) plus synthetic data only. Covers: the day
page render (form, empty state of a day, created messages after a refresh),
safe escaping of user HTML, timestamps rendered in the application timezone,
the date-as-unit behaviour of T16b (which day the page shows, counters, the
day picker, day navigation and the invalid-date fallback), progressive loading
of older entries *within the current day* through the fragment endpoint (order,
next-cursor management, history end) and the no-external-resources / safe-DOM
rules of the client assets.

Messages are seeded on a fixed day (2026-09-10) and the page is asked for that
day with ``?date=`` — the page shows one local calendar day, so a test that
wants a specific message on screen has to name the day it belongs to.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import diary.api.routes.pages as pages_module
from diary.api.routes.messages import get_session
from diary.core.settings import get_settings, reload_settings
from diary.db.models import Message, MessageKind, TranscriptionStatus
from diary.db.repositories import AudioFileRepository, MessageRepository
from diary.main import app

#: Repository root: tests/integration/test_chat_pages.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The fixed local day the seeded messages belong to (the default application
#: timezone is Europe/Moscow, so 2026-09-10T09:00Z is 12:00 local, same day).
DAY = date(2026, 9, 10)

OLDER_CURSOR_RE = re.compile(r'hx-get="/messages/fragments/older\?cursor=([^"]+)"')


def _day_url(day: date) -> str:
    return f"/?date={day.isoformat()}"


def _app_today() -> date:
    """Today as the application defines it (local calendar day, not UTC)."""
    return datetime.now(get_settings().tzinfo).date()


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Start every test with default application settings and a private data root."""
    monkeypatch.delenv("APP_TIMEZONE", raising=False)
    monkeypatch.delenv("APP_MAX_MESSAGE_LENGTH", raising=False)
    monkeypatch.delenv("APP_DATA_ROOT", raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
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


def _create_voice(session: Session, created_at: datetime) -> Message:
    """A voice message with its audio-file row (no bytes are written)."""
    message = MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=created_at
    )
    AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path="audio/2026/09/synthetic.webm",
        size_bytes=1024,
        sha256="0" * 64,
    )
    return message


def _card_of(html: str, message_id) -> str:
    """The rendered ``<article>`` card of one message."""
    match = re.search(
        rf'<article class="message[^"]*" id="message-{message_id}">.*?</article>',
        html,
        re.DOTALL,
    )
    assert match is not None, f"no card rendered for message {message_id}"
    return match.group(0)


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

    html = api_client.get(_day_url(DAY)).text
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

        # 2026-01-01T00:00Z is 05:45 on 2026-01-01 in Kathmandu, so that is the
        # local day the page has to be asked for (T16b: the page is one day).
        html = api_client.get(_day_url(date(2026, 1, 1))).text
        assert 'datetime="2026-01-01T05:45:00+05:45"' in html
        assert "01.01.2026 05:45" in html
    finally:
        reload_settings()


# ---------------------------------------------------------------------------
# the day as the unit (task T16b)
# ---------------------------------------------------------------------------


def test_day_page_shows_only_the_requested_day(api_client: TestClient, session: Session) -> None:
    _create(session, "запись десятого", _at(9))
    _create(session, "запись одиннадцатого", datetime(2026, 9, 11, 9, tzinfo=timezone.utc))
    session.commit()

    html = api_client.get(_day_url(DAY)).text
    assert "запись десятого" in html
    assert "запись одиннадцатого" not in html
    assert 'datetime="2026-09-10T12:00:00+03:00"' in html
    assert 'datetime="2026-09-11' not in html


def test_day_page_defaults_to_today_and_names_it(api_client: TestClient, session: Session) -> None:
    today = _app_today()
    MessageRepository(session).create(
        kind=MessageKind.text,
        body_text="сегодняшняя запись",
        created_at=datetime.now(timezone.utc),
    )
    session.commit()

    html = api_client.get("/").text
    assert "сегодняшняя запись" in html
    # The human-readable title of the day and its ISO date are both rendered.
    assert f'value="{today.isoformat()}"' in html
    assert "Сегодня" in html


def test_day_counters_count_records_and_voice_messages(
    api_client: TestClient, session: Session
) -> None:
    _create(session, "первая", _at(8))
    _create(session, "вторая", _at(9))
    _create_voice(session, _at(10))
    session.commit()

    html = api_client.get(_day_url(DAY)).text
    assert "3 записи · из них 1 голосовое сообщение" in html


def test_day_counters_use_the_right_plural_forms(api_client: TestClient, session: Session) -> None:
    """The counters agree with the number they count (no "1 записи")."""
    _create(session, "одна", _at(8))
    session.commit()
    assert "1 запись" in api_client.get(_day_url(DAY)).text

    for index in range(2):
        _create(session, f"ещё-{index}", _at(9 + index))
    session.commit()
    assert "3 записи" in api_client.get(_day_url(DAY)).text


def test_day_counters_stay_in_step_with_the_feed(api_client: TestClient) -> None:
    """The header counts real rows and says so in machine-readable form.

    The server renders the numbers it counted as ``data-total``/``data-voice``
    and app.js adjusts exactly those numbers when this page saves or deletes an
    entry (see the "day counters" module in app.js), so the sentence can never
    drift from the feed. Whether it really moves is a runtime property, checked
    by driving the page, not here.
    """
    html = api_client.get(_day_url(DAY)).text
    assert 'data-role="day-counters"' in html
    assert 'data-total="0"' in html
    assert 'data-voice="0"' in html

    script = (REPO_ROOT / "src/diary/static/js/app.js").read_text(encoding="utf-8")
    assert 'announceEntryChange("diary:entry-added", "text")' in script
    assert 'announceEntryChange("diary:entry-added", "audio")' in script
    assert '"diary:entry-removed"' in script
    assert 'document.addEventListener("diary:entry-added"' in script
    assert "counters.textContent = label(total, voice)" in script


def test_day_page_counts_the_whole_day_not_only_the_rendered_window(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pages_module, "PAGE_LIMIT", 1)
    _create(session, "старейшая", _at(8))
    _create(session, "средняя", _at(9))
    _create_voice(session, _at(10))
    session.commit()

    html = api_client.get(_day_url(DAY)).text
    # One entry is rendered (the newest), the counters still describe the day.
    assert "средняя" not in html
    assert "3 записи · из них 1 голосовое сообщение" in html


def test_empty_day_shows_a_hint_instead_of_entries(api_client: TestClient) -> None:
    html = api_client.get(_day_url(DAY)).text
    assert 'id="message-empty"' in html
    assert "Пока нет записей за этот день." in html
    assert "Напишите первую ниже" in html
    assert "Записей нет" in html
    assert 'class="message"' not in html


def test_invalid_date_falls_back_to_today_with_a_notice(
    api_client: TestClient,
) -> None:
    """A malformed ?date= is a readable page, never an error (and never a 500)."""
    today = _app_today()
    for value in ("not-a-date", "2026-13-45", "10.09.2026", "9999-12-31", "0001-01-01"):
        response = api_client.get("/", params={"date": value})
        assert response.status_code == 200, value
        html = response.text
        assert "Дата в адресе не распознана" in html, value
        assert f'value="{today.isoformat()}"' in html, value

    # An empty parameter means "no date asked for": today, without a notice.
    blank = api_client.get("/", params={"date": ""})
    assert blank.status_code == 200
    assert "Дата в адресе не распознана" not in blank.text


def test_date_longer_than_the_limit_is_rejected_with_422(api_client: TestClient) -> None:
    response = api_client.get("/", params={"date": "9" * 33})
    assert response.status_code == 422


def test_day_picker_and_day_navigation_links(api_client: TestClient) -> None:
    today = _app_today()
    html = api_client.get("/").text
    assert 'type="date"' in html
    assert 'id="day-picker"' in html
    assert 'name="date"' in html
    assert f'value="{today.isoformat()}"' in html
    # Previous/next are plain links to the neighbouring local days ...
    assert f'href="/?date={(today - timedelta(days=1)).isoformat()}"' in html
    assert f'href="/?date={(today + timedelta(days=1)).isoformat()}"' in html
    # ... and on today's page there is nothing to go "back to today" from.
    assert 'href="/">К сегодняшнему дню' not in html

    yesterday = today - timedelta(days=1)
    earlier = api_client.get(_day_url(yesterday)).text
    assert f'href="/?date={(yesterday - timedelta(days=1)).isoformat()}"' in earlier
    assert f'href="/?date={today.isoformat()}"' in earlier
    assert "К сегодняшнему дню" in earlier
    # Yesterday is named as such, without pretending to be today.
    assert "Вчера" in earlier


def test_day_page_uses_the_same_day_window_as_the_daily_summary_layer(
    api_client: TestClient, session: Session
) -> None:
    """Midnight belongs to the *local* day, exactly as ADR-007 §1 defines it.

    The default application timezone is Europe/Moscow (UTC+03:00, no DST), so
    20:59Z and 21:01Z on 2026-09-10 are 23:59 and 00:01 local — the two sides
    of midnight. Each lands on its own day page and on no other.
    """
    _create(session, "до полуночи", datetime(2026, 9, 10, 20, 59, tzinfo=timezone.utc))
    _create(session, "после полуночи", datetime(2026, 9, 10, 21, 1, tzinfo=timezone.utc))
    session.commit()

    html = api_client.get(_day_url(DAY)).text
    assert "до полуночи" in html
    assert "после полуночи" not in html

    tomorrow = api_client.get(_day_url(DAY + timedelta(days=1))).text
    assert "после полуночи" in tomorrow
    assert "до полуночи" not in tomorrow


# ---------------------------------------------------------------------------
# progressive loading of older entries inside the day
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

    html = api_client.get(_day_url(DAY)).text
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
    page = api_client.get(_day_url(DAY))
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


def test_older_fragment_stops_at_the_beginning_of_the_cursor_day(
    api_client: TestClient,
    session: Session,
) -> None:
    """Paging is bounded by the day (T16b): it never walks into the day before.

    A day with entries *and* older entries on the previous day: the fragment
    returns the rest of the cursor's day and then says "start of history",
    instead of offering the neighbouring day's entries (which are reached
    through the day navigation of the header).
    """
    older_day_message = _create(
        session, "запись предыдущего дня", datetime(2026, 9, 9, 9, tzinfo=timezone.utc)
    )
    day_messages = [_create(session, f"запись-{index}", _at(8 + index)) for index in range(3)]
    session.commit()

    cursor = f"{day_messages[-1].created_at.isoformat()}|{day_messages[-1].id}"
    fragment = api_client.get("/messages/fragments/older", params={"cursor": cursor, "limit": "5"})
    assert fragment.status_code == 200
    body = fragment.text
    # The rest of the same day is returned, the previous day is not.
    for message in day_messages[:-1]:
        assert f'id="message-{message.id}"' in body
    assert f'id="message-{older_day_message.id}"' not in body
    assert "запись предыдущего дня" not in body
    assert "Начало истории" in body
    assert OLDER_CURSOR_RE.search(body) is None


# ---------------------------------------------------------------------------
# the voice card: real duration only, decorative waveform (task T16b)
# ---------------------------------------------------------------------------


def test_voice_card_renders_no_duration_from_the_server(
    api_client: TestClient, session: Session
) -> None:
    """The duration is never rendered by the server.

    This guards the static contract: the card carries an empty, hidden duration
    element (filled in from the audio metadata by app.js) and no number that
    could have been invented server-side. Whether the browser really reports a
    duration is a runtime property, checked by driving the page, not here.
    """
    _create_voice(session, _at(10))
    session.commit()

    html = api_client.get(_day_url(DAY)).text
    wrapper = re.search(r'<p class="audio-duration"[^>]*>.*?</p>', html, re.DOTALL)
    assert wrapper is not None, "the voice card must carry the duration element"
    markup = wrapper.group(0)
    assert 'data-role="duration-wrap"' in markup
    assert "hidden" in markup.split(">", 1)[0]
    # No time-looking value anywhere inside the duration element.
    assert re.search(r"\d{1,2}:\d{2}", markup) is None
    # The element is empty until the browser provides a real value.
    value = re.search(r'<span class="audio-duration__value" data-role="duration"></span>', markup)
    assert value is not None

    # The rendered decorative waveform: bars, no text and no numbers at all.
    rendered_waveform = re.search(r'<span class="waveform"[^>]*>.*?</span>', html, re.DOTALL)
    assert rendered_waveform is not None
    bars = rendered_waveform.group(0)
    assert 'aria-hidden="true"' in bars
    assert bars.count("<i>") == 20
    assert re.search(r"\d", bars) is None, "the waveform must not carry numbers"


# ---------------------------------------------------------------------------
# a voice card whose audio row is missing (task T31)
# ---------------------------------------------------------------------------


def _create_voice_without_audio_row(session: Session, created_at: datetime):
    """A ``kind=audio`` message with no ``audio_files`` row at all.

    The state the T31 defect is about: the message row exists (and is
    deletable), the audio linkage does not — the upload never completed, or the
    row was lost. Nothing writes bytes in this test.
    """
    return MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=created_at
    )


def test_voice_card_without_an_audio_row_draws_no_dead_controls(
    api_client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T31: such a card states the fact instead of drawing what cannot work.

    Before T31 every audio key of this card was *undefined*, so Jinja2
    rendered ``id="transcription-"`` (an anchor nothing can reach),
    ``<audio src="">`` and a «Расшифровать» button with an empty caption: two
    dead controls and a broken jump target. The card now carries the caption
    the processing panel already uses for the very same state, and draws no
    player, no transcription block and no button — while keeping its own
    delete action (the record itself is a real, deletable message).

    Both render paths are checked: the day page (its newest window) and the
    "older entries" fragment, which renders the same card from the other
    endpoint.
    """
    monkeypatch.setattr(pages_module, "PAGE_LIMIT", 1)
    older = _create_voice_without_audio_row(session, _at(10))
    newest = _create_voice_without_audio_row(session, _at(11))
    session.commit()

    html = api_client.get(_day_url(DAY)).text
    newest_card = _card_of(html, newest.id)
    assert pages_module.PANEL_AUDIO_FILE_MISSING in newest_card
    # No player, no anchor, no trigger — anywhere on the page.
    assert "transcription-" not in html
    assert "<audio" not in html
    assert 'data-action="transcribe"' not in html
    # The card is still a card: its own actions are untouched.
    assert f'data-action="delete-message" data-message-id="{newest.id}"' in newest_card

    match = OLDER_CURSOR_RE.search(html)
    assert match is not None, "the older control must be on the page"
    fragment = api_client.get(
        "/messages/fragments/older",
        params={"cursor": urllib.parse.unquote(match.group(1)), "limit": "1"},
    )
    assert fragment.status_code == 200
    older_card = _card_of(fragment.text, older.id)
    assert pages_module.PANEL_AUDIO_FILE_MISSING in older_card
    assert "transcription-" not in fragment.text
    assert "<audio" not in fragment.text
    assert 'data-action="transcribe"' not in fragment.text
    assert f'data-action="delete-message" data-message-id="{older.id}"' in older_card


def test_voice_context_without_an_audio_row_is_explicit_not_undefined(
    session: Session,
) -> None:
    """T31: the missing state is carried as values, not as absent keys.

    The rendered page cannot tell the two apart (Jinja2's default ``Undefined``
    is falsy, so ``{% if message.audio_available %}`` would take the same
    branch), which is exactly why the defect could hide. This test reads the
    context the card is rendered from — the same mapping both render paths
    build from (:func:`_audio_view` here, the batch loader on a page) — and
    requires every audio key to be present with an explicit value.
    """
    message = _create_voice_without_audio_row(session, _at(10))
    session.commit()
    missing = pages_module._message_view(message, session)
    assert missing["audio_available"] is False
    assert missing["audio_file_id"] is None
    assert missing["audio_url"] is None
    assert missing["audio_mime"] is None
    assert missing["transcription_raw"] is None
    assert missing["transcription_button_label"] is None
    assert missing["active_job_id"] is None
    # The caption is the panel's, and the state is named (never a status value
    # of the transcription table — no such row exists).
    assert missing["transcription_label"] == pages_module.PANEL_AUDIO_FILE_MISSING
    assert missing["transcription_status"] == pages_module.AUDIO_LINK_MISSING_STATE
    assert missing["transcription_status"] not in {status.value for status in TranscriptionStatus}

    # The same keys, explicitly present, for a voice message that does have its
    # audio row — so the flag is a real state and not "the key is missing".
    linked = _create_voice(session, _at(11))
    session.commit()
    available = pages_module._message_view(linked, session)
    assert available["audio_available"] is True
    assert set(available) >= set(missing)
    assert available["audio_file_id"] is not None
    assert available["audio_url"] is not None

    # A text message carries none of the audio keys and is unaffected.
    text = _create(session, "обычная запись", _at(12))
    session.commit()
    text_view = pages_module._message_view(text, session)
    assert "audio_available" not in text_view
    assert "audio_file_id" not in text_view


def test_card_uses_the_newest_audio_row_exactly_as_the_single_message_path(
    api_client: TestClient, session: Session
) -> None:
    """The batch path picks the same ``audio_files`` row as the single path.

    Task 07 stores one row per voice message, but the schema does not forbid an
    older second row. The day page renders through the batch path, so this pins
    that both paths choose the **newest** row — same ``audio_file_id``, same
    URL, same anchor, same transcription state.
    """
    message = MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=_at(9)
    )
    older = AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path="audio/2026/09/synthetic-first.webm",
        size_bytes=1024,
        sha256="0" * 64,
        created_at=_at(10),
    )
    newer = AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/wav",
        storage_path="audio/2026/09/synthetic-second.wav",
        size_bytes=2048,
        sha256="1" * 64,
        created_at=_at(11),
    )
    session.commit()
    assert [row.id for row in AudioFileRepository(session).get_by_message(message.id)] == [
        older.id,
        newer.id,
    ]

    single = pages_module._message_view(message, session)
    batch = pages_module._message_views([message], session)[0]
    assert batch == single
    assert batch["audio_file_id"] == str(newer.id)

    html = api_client.get(_day_url(DAY)).text
    assert f'id="transcription-{newer.id}"' in html
    assert str(older.id) not in html


def test_waveform_is_decorative_and_hidden_from_assistive_technology() -> None:
    """The waveform is a shape, not information: it is ``aria-hidden``, holds
    no numbers and is not derived from the recording (the backend stores none).
    """
    template = (REPO_ROOT / "src/diary/templates/_message.html").read_text(encoding="utf-8")
    waveform = re.search(r'<span class="waveform"[^>]*>', template)
    assert waveform is not None
    assert 'aria-hidden="true"' in waveform.group(0)

    script = (REPO_ROOT / "src/diary/static/js/app.js").read_text(encoding="utf-8")
    # The duration comes from the audio element's own metadata ...
    assert "loadedmetadata" in script
    assert "audio.duration" in script
    # ... and an unusable value hides the element instead of showing something.
    assert "isFinite(seconds) && seconds > 0" in script
    assert "wrapper.hidden = true" in script
    # Appended voice messages get the same decorative waveform.
    assert "buildAudioCard" in script
    assert "waveform.setAttribute" in script and '"aria-hidden", "true"' in script

    css = (REPO_ROOT / "src/diary/static/css/app.css").read_text(encoding="utf-8")
    assert re.search(r"\.audio-duration\s*\{[^}]*font-variant-numeric:\s*tabular-nums", css)


def test_compose_form_declares_whether_the_visible_day_is_today(
    api_client: TestClient,
) -> None:
    """An entry can only be created "now": the page tells the client whether the
    day on screen is today, so a save can never be appended to another day."""
    today_html = api_client.get("/").text
    assert 'data-day-is-today="true"' in today_html
    other_day = _app_today() - timedelta(days=3)
    other_html = api_client.get(_day_url(other_day)).text
    assert 'data-day-is-today="false"' in other_html
    assert "Новая запись будет создана в сегодняшнем дне" in other_html

    script = (REPO_ROOT / "src/diary/static/js/app.js").read_text(encoding="utf-8")
    assert "viewedDayIsToday" in script
    assert "showSavedOnAnotherDay" in script


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


def test_message_delete_control_needs_an_explicit_confirmation() -> None:
    """T15: every message carries a delete action with a clear accessible name,
    and a single click can never delete anything.

    This guards the static contract only. The round trip itself — the existing
    ``DELETE /api/v1/messages/{id}``, the erased text/bytes and the 404 that
    means "already deleted" — is exercised in ``test_message_deletion.py``; that
    the message disappears without a reload is a runtime property of the page.
    """
    template = (REPO_ROOT / "src/diary/templates/_message.html").read_text(encoding="utf-8")
    # The same control is rendered for text and for voice messages.
    assert template.count('{% include "_message_actions.html" %}') == 2

    actions = (REPO_ROOT / "src/diary/templates/_message_actions.html").read_text(encoding="utf-8")
    assert 'data-action="delete-message"' in actions
    assert 'data-message-id="{{ message.id }}"' in actions
    # The repeated "Удалить" is disambiguated by the message timestamp.
    assert 'aria-label="Удалить запись от {{ message.created_at_display }}"' in actions
    # Confirmation is a real second step (its own buttons), not a visual hint.
    assert 'data-action="delete-message-confirm"' in actions
    assert 'data-action="delete-message-cancel"' in actions
    assert "Удалить запись?" in actions
    panel = re.search(r'<div class="message__confirm"[^>]*>', actions)
    assert panel is not None
    assert "hidden" in panel.group(0)

    script = (REPO_ROOT / "src/diary/static/js/app.js").read_text(encoding="utf-8")
    assert '"/api/v1/messages/"' in script
    assert 'method: "DELETE"' in script
    # 404 (unknown or already deleted) is caught with 204 and removes the
    # message — it is never reported as a failure.
    assert "response.status === 204 || response.status === 404" in script
    assert "Escape" in script
    # Messages appended after a save get exactly the same controls.
    assert "buildMessageActions(" in script

    css = (REPO_ROOT / "src/diary/static/css/app.css").read_text(encoding="utf-8")
    assert re.search(r"\.message__confirm\[hidden\]\s*\{[^}]*display:\s*none", css) is not None


def test_app_serves_local_static_assets(api_client: TestClient) -> None:
    for url_path in (
        "/static/css/app.css",
        "/static/js/app.js",
        "/static/vendor/htmx.min.js",
    ):
        response = api_client.get(url_path)
        assert response.status_code == 200
        assert len(response.content) > 0
