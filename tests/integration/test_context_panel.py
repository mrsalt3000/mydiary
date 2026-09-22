"""Integration tests for the day page's processing panel (task T16c).

Covers the third stage of the mockup redesign: the shell's third column on the
day page only, filled with real data from the database — «Записи дня»
(originals, voice messages, transcription) and «Недельные слои» (the weekly
export of the week this day belongs to, with its computed artifacts).

The render assertions run against an isolated, migrated test database with
synthetic data, so counts, states and links are checked against the rows that
were actually inserted. The static contracts at the end check what a rendered
page cannot show: that the panel adds no second transcription trigger, no
script of its own and no invented model name.

Honest scope: whether the panel *looks* right (column width, the icons, the
collapse at <=1080px) is a browser property, verified by driving the page, not
here; that the statuses stay in step after a transcription started in the feed
is deliberately *not* claimed either — the panel is a server-rendered snapshot
and says so on the page (see ``PANEL_SNAPSHOT_NOTE``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session

import diary.api.routes.pages as pages_module
from diary.api.routes.messages import get_session
from diary.api.routes.pages import (
    PAGE_LIMIT,
    PANEL_SNAPSHOT_NOTE,
    PANEL_VOICE_OLDER_NOTE,
    PANEL_WELLBEING_NOTE,
)
from diary.core.models_config import ModelsConfigError, parse_models_config
from diary.core.settings import reload_settings
from diary.db.models import (
    ArtifactLayer,
    JobStatus,
    JobType,
    Message,
    MessageKind,
    TranscriptionStatus,
)
from diary.db.repositories import (
    ArtifactRepository,
    AudioFileRepository,
    JobRepository,
    MessageRepository,
    TranscriptionRepository,
)
from diary.domain.exports.service import export_week
from diary.domain.exports.weeks import sunday_date_for
from diary.domain.pipelines.wellbeing.safety import DISCLAIMER
from diary.main import app

#: Repository root: tests/integration/test_context_panel.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "src/diary/templates"
MOCKUP = REPO_ROOT / "docs/mockups/mydiary-ui-mockup.html"

#: A fixed local day (the default application timezone is Europe/Moscow, so
#: 2026-09-10T09:00Z is 12:00 local on the same day).
DAY = date(2026, 9, 10)
#: That week's Sunday — the week the exports of these tests belong to.
SUNDAY = sunday_date_for(DAY)

PANEL_RE = re.compile(r'<aside class="context-panel".*?</aside>', re.DOTALL)
STEP_RE = re.compile(r'<div class="pipeline-step">.*?(?=<div class="pipeline-step">|\Z)', re.DOTALL)
#: One ``process-card`` of the panel, up to the next card / the closing note.
CARD_RE = re.compile(
    r'<section class="process-card">.*?'
    r'(?=<section class="process-card">|<p class="privacy-note">|</aside>)',
    re.DOTALL,
)


def _panel(html: str) -> str:
    match = PANEL_RE.search(html)
    assert match is not None, "the day page must render the context panel"
    return match.group(0)


def _card(panel: str, title: str) -> str:
    """The rendered fragment of the process card whose title is ``title``."""
    for candidate in CARD_RE.findall(panel):
        if f">{title}<" in candidate:
            return candidate
    raise AssertionError(f"no process card titled {title!r} in the panel")


def _step(panel: str, title: str) -> str:
    """The rendered fragment of the pipeline step whose title is ``title``."""
    for candidate in STEP_RE.findall(panel):
        if f">{title}<" in candidate or f">{title}</a>" in candidate:
            return candidate
    raise AssertionError(f"no pipeline step titled {title!r} in the panel")


def _rendered_message_ids(html: str) -> set[str]:
    """Ids of the entry cards a rendered day page actually holds
    (``id="message-…"``), i.e. the window of the feed."""
    return set(re.findall(r'id="message-([0-9a-f-]+)"', html))


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the app data root at a per-test directory and isolate the env."""
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
    with _client_bound_to(session_factory) as client:
        yield client


def _at(hour: int) -> datetime:
    return datetime(2026, 9, 10, hour, tzinfo=timezone.utc)


def _at_minute(minute: int) -> datetime:
    """A distinct instant of ``DAY`` for the long-day tests (minutes apart, so
    a day can hold more entries than the feed's page window)."""
    return datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)


def _create_text(session: Session, body: str, created_at: datetime) -> Message:
    return MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=created_at
    )


def _create_voice(session: Session, created_at: datetime) -> tuple[Message, object]:
    """A voice message with its audio-file row (no bytes are written)."""
    message = MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=created_at
    )
    audio = AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path="audio/2026/09/synthetic.webm",
        size_bytes=1024,
        sha256="0" * 64,
    )
    return message, audio


def _transcribe_succeeded(session: Session, audio) -> None:
    TranscriptionRepository(session).create(
        audio_file_id=audio.id,
        raw_text="синтетический текст",
        status=TranscriptionStatus.succeeded,
    )


def _week_export(session: Session, *, sunday: date = SUNDAY, re_export: bool = False):
    """One real weekly export (row + file) with a synthetic message in its week."""
    monday = sunday - timedelta(days=6)
    MessageRepository(session).create(
        kind=MessageKind.text,
        body_text="Синтетическая запись недели.",
        created_at=datetime(monday.year, monday.month, monday.day, 9, 0, tzinfo=timezone.utc),
    )
    session.commit()
    return export_week(session, sunday_date=sunday, re_export=re_export)


def _create_artifact(session: Session, export_id, layer: ArtifactLayer, version: int) -> None:
    ArtifactRepository(session).create(
        layer=layer,
        export_id=export_id,
        path=f"exports/{layer.value}/2026-09-10_{layer.value}_0001_r1_v{version}.txt",
        sha256="0" * 64,
        version=version,
    )


# ---------------------------------------------------------------------------
# where the panel appears (DoD 1)
# ---------------------------------------------------------------------------


def test_day_page_renders_the_panel_in_the_third_column(api_client: TestClient) -> None:
    html = api_client.get(f"/?date={DAY.isoformat()}").text
    assert "app-shell--with-context" in html
    panel = _panel(html)
    assert 'aria-label="Обработка дня"' in html
    assert "<h2>Обработка дня</h2>" in panel
    # Three honest blocks, all named.
    assert '<h3 class="process-title">Записи дня</h3>' in panel
    assert '<h3 class="process-title">Недельные слои</h3>' in panel
    assert '<h3 class="process-title">Слои фазы 2</h3>' in panel
    # The panel is a snapshot of the server state and says so.
    assert PANEL_SNAPSHOT_NOTE in panel


def test_only_the_day_page_reserves_the_third_column(
    api_client: TestClient, session: Session
) -> None:
    record = _week_export(session)
    export_id = str(record.id)  # type: ignore[attr-defined]
    paths = (
        "/exports",
        f"/exports/{export_id}",
        f"/exports/{export_id}/normalize",
        f"/exports/{export_id}/summary",
        f"/exports/{export_id}/wellbeing",
        "/processing",
        "/privacy",
    )
    for path in paths:
        response = api_client.get(path)
        assert response.status_code == 200, path
        html = response.text
        assert "context-panel" not in html, path
        assert "app-shell--with-context" not in html, path


def test_panel_describes_the_requested_day_not_today(api_client: TestClient) -> None:
    """The day is the page's unit (T16b): the panel follows ``?date=``."""
    html = api_client.get(f"/?date={DAY.isoformat()}").text
    panel = _panel(html)
    assert "Четверг, 10 сентября" in panel
    assert "Неделя 07.09.2026 — 13.09.2026" in panel


# ---------------------------------------------------------------------------
# block 1 — the day itself, counted from the rows (DoD 2, 3)
# ---------------------------------------------------------------------------


def test_day_block_counts_real_rows_and_voice_states(
    api_client: TestClient, session: Session
) -> None:
    """Two text entries and three voice messages: one transcribed, one never
    requested, one failed. Every number and state is checked against that."""
    _create_text(session, "первая", _at(8))
    _create_text(session, "вторая", _at(9))
    _, done_audio = _create_voice(session, _at(10))
    _, waiting_audio = _create_voice(session, _at(11))
    _, failed_audio = _create_voice(session, _at(12))
    _transcribe_succeeded(session, done_audio)
    TranscriptionRepository(session).create(
        audio_file_id=failed_audio.id,
        raw_text="",
        status=TranscriptionStatus.failed,
    )
    session.commit()

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert "5 записей" in _step(panel, "Оригиналы")
    assert "3 голосовых сообщения" in _step(panel, "Аудио")
    transcription_step = _step(panel, "Транскрипция")
    assert "Расшифровано 1 из 3" in transcription_step
    assert "ждут запуска 1" in transcription_step
    assert "с ошибкой 1" in transcription_step
    # Not everything is transcribed: the step is not marked done.
    assert 'class="step-icon pending"' in transcription_step
    # The per-voice rows carry the same states the cards show.
    assert "Не расшифровано" in panel
    assert "Ошибка расшифровки" in panel
    assert "Расшифровано" in panel


def test_transcription_step_is_done_when_every_voice_message_is_transcribed(
    api_client: TestClient, session: Session
) -> None:
    _, audio = _create_voice(session, _at(10))
    _transcribe_succeeded(session, audio)
    session.commit()

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    step = _step(panel, "Транскрипция")
    assert "Расшифровано 1 из 1" in step
    assert 'class="step-icon done"' in step


def test_day_block_says_so_when_there_is_nothing_to_transcribe(
    api_client: TestClient, session: Session
) -> None:
    _create_text(session, "только текст", _at(9))
    session.commit()

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert "Голосовых записей нет" in _step(panel, "Аудио")
    assert "Расшифровывать нечего" in _step(panel, "Транскрипция")


def test_empty_day_reports_no_entries_instead_of_zeros(api_client: TestClient) -> None:
    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert "За этот день записей нет." in panel
    # The *day* block draws no steps at all for an empty day. (The phase-2 block
    # below it is a separate card: it reports the two layer lineages of the day
    # and of its week as «Не посчитано» — task T37.)
    assert '<div class="pipeline">' not in _card(panel, "Записи дня")


def test_voice_without_an_audio_row_is_reported_separately(
    api_client: TestClient, session: Session
) -> None:
    """An audio message with no ``audio_files`` row is neither "waiting" nor
    "transcribed" — the panel says what is actually missing."""
    MessageRepository(session).create(kind=MessageKind.audio, body_text="", created_at=_at(10))
    session.commit()

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert "Аудиофайл не найден" in panel
    assert "без файла 1" in _step(panel, "Транскрипция")


def test_voice_row_links_to_the_cards_own_action_and_adds_no_second_trigger(
    api_client: TestClient, session: Session
) -> None:
    """No duplicated transcription logic: the panel points at the card's button.

    The STT action and its job poller exist once (``static/js/app.js``), so the
    panel renders a link to the card's anchor instead of a second button, and
    the page still holds exactly one ``data-action="transcribe"`` control.
    """
    _, audio = _create_voice(session, _at(10))
    session.commit()
    audio_file_id = str(audio.id)  # type: ignore[attr-defined]

    html = api_client.get(f"/?date={DAY.isoformat()}").text
    panel = _panel(html)
    assert f'href="#transcription-{audio_file_id}"' in panel
    assert "Расшифровать" in panel
    # The jump target exists (that anchor is the card's transcription block).
    assert f'id="transcription-{audio_file_id}"' in html
    # A short day (every entry rendered): nothing to mark as out of the window.
    assert PANEL_VOICE_OLDER_NOTE not in panel
    # Exactly one trigger, and it is the card's.
    assert html.count('data-action="transcribe"') == 1

    script = (REPO_ROOT / "src/diary/static/js/app.js").read_text(encoding="utf-8")
    # The panel itself has no client behaviour at all.
    assert "panel-voice" not in script
    assert "context-panel" not in script


def test_voice_row_offers_no_action_while_a_job_is_in_flight(
    api_client: TestClient, session: Session
) -> None:
    _, audio = _create_voice(session, _at(10))
    JobRepository(session).create(
        job_type=JobType.transcribe_audio,
        status=JobStatus.pending,
        payload_json={"audio_file_id": str(audio.id)},  # type: ignore[attr-defined]
    )
    session.commit()

    html = api_client.get(f"/?date={DAY.isoformat()}").text
    panel = _panel(html)
    assert "Ожидает расшифровки" in panel
    # The card renders the busy state, the panel offers no jump to a disabled
    # button (and no second way to start STT).
    assert f'href="#transcription-{audio.id}"' not in panel
    assert 'data-active-job-id="' in html
    assert html.count('data-action="transcribe"') == 1


def test_voice_row_links_only_into_the_rendered_window(
    api_client: TestClient, session: Session
) -> None:
    """A day longer than the feed's window (T16c rework).

    The day block counts the *whole* day, the feed renders only its newest
    :data:`PAGE_LIMIT` entries. A voice row whose card is outside that window
    must not offer a jump: the anchor ``#transcription-…`` would have no target
    in the DOM (the card only arrives with "Показать более старые записи").
    The row itself stays — with its real state and an honest mark — because the
    counters count it.
    """
    # The voice message is the oldest entry of a day longer than the window.
    voice, audio = _create_voice(session, _at_minute(0))
    for minute in range(1, PAGE_LIMIT + 6):
        _create_text(session, f"запись {minute}", _at_minute(minute))
    session.commit()
    audio_file_id = str(audio.id)  # type: ignore[attr-defined]
    html = api_client.get(f"/?date={DAY.isoformat()}").text
    # Sanity of the case itself: the voice card is *not* rendered.
    assert str(voice.id) not in _rendered_message_ids(html)  # type: ignore[attr-defined]

    panel = _panel(html)
    # No jump into nowhere, and the reason is stated in words.
    assert 'href="#transcription-' not in panel
    assert f'href="#transcription-{audio_file_id}"' not in panel
    assert f'id="transcription-{audio_file_id}"' not in html
    assert PANEL_VOICE_OLDER_NOTE in panel
    # The row is still there with its real state (the day block counts the day).
    assert "Не расшифровано" in panel
    assert "Расшифровано 0 из 1" in _step(panel, "Транскрипция")
    assert "ждут запуска 1" in _step(panel, "Транскрипция")


def test_voice_row_of_a_long_day_inside_the_window_keeps_its_working_link(
    api_client: TestClient, session: Session
) -> None:
    """The other half of the same rule, on a mixed long day.

    Two voice messages: one outside the rendered window (as above) and one
    inside it. The windowed one keeps its jump and that jump has a target in
    the very same HTML; every ``#transcription-…`` link the panel renders
    resolves, so the panel can never hold a dead anchor.
    """
    _, outside_audio = _create_voice(session, _at_minute(0))
    for minute in range(1, PAGE_LIMIT):
        _create_text(session, f"запись {minute}", _at_minute(minute))
    voice, audio = _create_voice(session, _at_minute(PAGE_LIMIT))
    session.commit()
    audio_file_id = str(audio.id)  # type: ignore[attr-defined]
    html = api_client.get(f"/?date={DAY.isoformat()}").text
    # Sanity of the case itself: one card is rendered, the other is not.
    rendered = _rendered_message_ids(html)
    assert str(voice.id) in rendered  # type: ignore[attr-defined]
    assert str(outside_audio.id) not in rendered  # type: ignore[attr-defined]

    panel = _panel(html)
    assert f'href="#transcription-{audio_file_id}"' in panel
    assert f'id="transcription-{audio_file_id}"' in html
    # Both voice messages are reported; only the rendered card is linked, and
    # every link of the panel has its target on the page.
    assert "Расшифровано 0 из 2" in _step(panel, "Транскрипция")
    assert f'href="#transcription-{outside_audio.id}"' not in panel  # type: ignore[attr-defined]
    # Exactly the out-of-window row carries the mark; the rendered one does not.
    assert panel.count(PANEL_VOICE_OLDER_NOTE) == 1
    links = re.findall(r'href="#(transcription-[0-9a-f-]+)"', panel)
    assert links, "the rendered voice message must keep its jump"
    for anchor in links:
        assert f'id="{anchor}"' in html, anchor


def test_day_block_chip_names_the_configured_stt_profile(
    api_client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chip is the real configured STT provider and model — never a name
    borrowed from the mockup (which says "Whisper · local" / "llama3.2")."""
    config = parse_models_config(
        {
            "profiles": {
                "transcription": {"provider": "faster_whisper", "model": "small"},
                "normalization": _openai_profile(),
                "summarization": _openai_profile(),
                "wellbeing_analysis": _openai_profile(),
            }
        }
    )
    monkeypatch.setattr(pages_module, "_load_models_config", lambda: config)
    _, audio = _create_voice(session, _at(10))
    _transcribe_succeeded(session, audio)
    session.commit()

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert "faster-whisper · small" in _step(panel, "Транскрипция")
    assert "llama3.2" not in panel


def test_day_block_omits_the_chip_when_the_configuration_is_unavailable(
    api_client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No configuration → no chip: the panel does not guess a model."""

    def _raise() -> None:
        raise ModelsConfigError("synthetic: no configuration")

    monkeypatch.setattr(pages_module, "_load_models_config", _raise)
    _, audio = _create_voice(session, _at(10))
    _transcribe_succeeded(session, audio)
    session.commit()

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert 'class="chip"' not in _step(panel, "Транскрипция")


def _openai_profile(model: str = "diary-sum") -> dict[str, object]:
    return {
        "provider": "openai_compatible",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": model,
        "api_key_env": "SYNTHETIC_KEY",
        "max_tokens": 1024,
    }


# ---------------------------------------------------------------------------
# block 2 — the week's layers, anchored to the weekly export (DoD 2, 3, 4)
# ---------------------------------------------------------------------------


def test_week_block_shows_the_computed_layers_and_links_to_them(
    api_client: TestClient, session: Session
) -> None:
    record = _week_export(session)
    export_id = str(record.id)  # type: ignore[attr-defined]
    _create_artifact(session, record.id, ArtifactLayer.normalized, 2)  # type: ignore[attr-defined]
    _create_artifact(session, record.id, ArtifactLayer.summary, 1)  # type: ignore[attr-defined]
    session.commit()

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert f"/exports/{export_id}" in panel
    assert "ревизия r1" in panel
    # Computed layers report their artifact version; the untouched one does not
    # pretend to be computed.
    assert "v2 · создано" in _step(panel, "Нормализованный (слой 2)")
    assert "v1 · создано" in _step(panel, "Саммари (слой 3)")
    assert "Не посчитано" in _step(panel, "Wellbeing-рефлексия (слой 4)")
    # Every layer links to its existing page of this export.
    for suffix in ("normalize", "summary", "wellbeing"):
        assert f'href="/exports/{export_id}/{suffix}"' in panel
    # The wellbeing layer keeps its delicacy wording.
    assert "Деликатная рефлексия" in panel


def test_week_block_does_not_borrow_another_weeks_export(
    api_client: TestClient, session: Session
) -> None:
    """A week without an export says so and links to where one is created."""
    other = _week_export(session, sunday=SUNDAY - timedelta(days=7))
    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert "Выгрузки за эту неделю нет." in panel
    assert 'href="/exports"' in panel
    assert str(other.id) not in panel  # type: ignore[attr-defined]


def test_week_block_prefers_the_newest_revision(api_client: TestClient, session: Session) -> None:
    _week_export(session)
    newest = _week_export(session, re_export=True)

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert "ревизия r2 из 2" in panel
    assert f"/exports/{newest.id}" in panel  # type: ignore[attr-defined]


def test_week_block_reports_a_running_run_instead_of_not_computed(
    api_client: TestClient, session: Session
) -> None:
    record = _week_export(session)
    JobRepository(session).create(
        job_type=JobType.normalize,
        status=JobStatus.running,
        payload_json={"export_id": str(record.id)},  # type: ignore[attr-defined]
    )
    session.commit()

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    step = _step(panel, "Нормализованный (слой 2)")
    assert "Прогон идёт" in step
    assert 'class="step-icon pending"' in step


def test_week_block_uses_the_configured_model_names(
    api_client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weekly-layer chips name the configured models; an external profile is
    labelled as such instead of claiming locality."""
    config = parse_models_config(
        {
            "profiles": {
                "transcription": {"provider": "faster_whisper", "model": "large-v3"},
                "normalization": _openai_profile("diary-stt"),
                "summarization": _openai_profile("diary-sum"),
                "wellbeing_analysis": {
                    **_openai_profile("diary-coach"),
                    "external": True,
                },
            }
        }
    )
    monkeypatch.setattr(pages_module, "_load_models_config", lambda: config)
    _week_export(session)

    panel = _panel(api_client.get(f"/?date={DAY.isoformat()}").text)
    assert "diary-stt · локально" in panel
    assert "diary-sum · локально" in panel
    assert "diary-coach · внешний API" in panel


# ---------------------------------------------------------------------------
# no dead elements, nothing invented (DoD 4, 5, 8)
# ---------------------------------------------------------------------------


def test_panel_carries_no_dead_mockup_actions(api_client: TestClient) -> None:
    """The mockup's "Показать происхождение" (no provenance page exists) and
    "Экспортировать запись" (the product exports weeks, not entries) are not
    reproduced; the weekly export link takes their place."""
    html = api_client.get("/").text
    assert "Показать происхождение" not in html
    assert "Экспортировать запись" not in html
    assert "Экспорт недели" not in html
    panel = _panel(html)
    # No script and no request-issuing element of its own.
    assert "<script" not in panel
    assert "<button" not in panel
    assert "hx-get" not in panel and "hx-post" not in panel
    assert "data-action=" not in panel


def test_panel_wellbeing_note_matches_the_layer_disclaimer(api_client: TestClient) -> None:
    """The short note says the same thing as the wellbeing layer's disclaimer
    and promises nothing new."""
    panel = _panel(api_client.get("/").text)
    assert PANEL_WELLBEING_NOTE in panel
    assert "диагноз" in DISCLAIMER and "профессиональной помощи" in DISCLAIMER
    assert "не ставит диагнозов" in PANEL_WELLBEING_NOTE
    assert "не заменяет профессиональную помощь" in PANEL_WELLBEING_NOTE


def test_panel_markup_follows_the_mockup_and_adds_no_scripts() -> None:
    template = (TEMPLATES_DIR / "_context_panel.html").read_text(encoding="utf-8")
    mockup = MOCKUP.read_text(encoding="utf-8")
    # The mockup's own class names are reused, so the panel cannot drift from
    # the reference design. (``context-panel__head`` is this project's BEM
    # spelling of the mockup's ``context-head``, which base.html already owns.)
    for name in (
        "process-card",
        "process-title",
        "process-subtitle",
        "pipeline",
        "pipeline-step",
        "step-icon",
        "step-content",
        "step-title",
        "step-detail",
        "chip",
        "context-actions",
        "privacy-note",
    ):
        assert name in template, name
        assert name in mockup, name
    assert "context-head" in mockup and "context-panel__head" in template
    # Server-rendered only: no script, no external resource.
    assert "<script" not in template
    assert re.search(r"https?://", template) is None
    # Diary text is not rendered in the panel (no body/raw-text variables).
    assert "message.body" not in template
    assert "transcription_raw" not in template


def test_panel_styles_use_the_token_system_only() -> None:
    css = (REPO_ROOT / "src/diary/static/css/app.css").read_text(encoding="utf-8")
    start = css.index("/* --------------------- processing panel (T16c)")
    end = css.index("@media (max-width: 1080px)", start)
    block = css[start:end]
    for selector in (
        ".process-card",
        ".pipeline-step",
        ".step-icon.done",
        ".chip",
        ".privacy-note",
    ):
        assert selector in block, selector
    # Colours come from the token system, never from literals.
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
    assert "var(--" in block


def test_other_templates_do_not_fill_the_panel() -> None:
    """One page owns the third column (task T16c): the shell renders the panel
    for the day page and reserves nothing on the other pages."""
    chat = (TEMPLATES_DIR / "chat.html").read_text(encoding="utf-8")
    assert "{% block context %}" in chat
    assert "_context_panel.html" in chat
    for name in (
        "exports.html",
        "processing.html",
        "privacy.html",
        "export_detail.html",
        "export_compare.html",
        "export_summary.html",
        "export_wellbeing.html",
    ):
        source = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
        assert "context_panel" not in source, name
        assert "{% block context %}" not in source, name


# ---------------------------------------------------------------------------
# T31 — the day page reads a constant number of rows
# ---------------------------------------------------------------------------

#: SQL statements one render of the day page issues, for a day that holds at
#: least one voice message. The acceptance measured 17 for 1, 2 and 3 voice
#: messages and for a long mixed day, and 11 for a day with no voice message at
#: all; this test pins the voice-message cases (the audio reads are skipped for
#: a day without voice messages, not run with an empty result). Task T37 added
#: two constant reads — the newest ``daily_summary`` of the day and the newest
#: ``weekly_summary`` of its ISO week — so the same three cases now measure 19,
#: and a day with no voice message measures 13.
#:
#: Before T31 the same page issued ``11 + 6 * N`` — 17 / 23 / 29 for
#: 1 / 2 / 3 voice messages: every card resolved its audio state with three
#: single-message reads, and the panel repeated those three per voice row.
#: The numbers are pinned on purpose: a future change that reintroduces a
#: per-message read must fail here and be justified, not pass silently.
DAY_PAGE_QUERY_COUNT = 19


@contextmanager
def _count_queries(engine) -> Iterator[dict[str, int]]:
    """Count the SQL statements sent to the database inside the context.

    ``before_cursor_execute`` fires once per statement, so the counter sees
    every read the render performs. The listener is removed on exit, so a
    failing test cannot leak it into the next one.
    """
    counter = {"statements": 0}

    def _before(conn, cursor, statement, parameters, context, executemany) -> None:
        counter["statements"] += 1

    event.listen(engine, "before_cursor_execute", _before)
    try:
        yield counter
    finally:
        event.remove(engine, "before_cursor_execute", _before)


def test_day_page_reads_a_constant_number_of_rows_whatever_the_voice_count(
    engine, api_client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T31 DoD 2: the render cost does not grow with the number of recordings.

    The same day rendered with 1, 2 and 3 voice messages sends the same number
    of SQL statements (:data:`DAY_PAGE_QUERY_COUNT`); before T31 it grew by six
    per voice message (three in the card, three in the panel). What the panel
    *says* about those messages is checked by the tests above — this one only
    counts reads, and it counts the whole request (panel and feed alike).

    The long-day part keeps the same number with paging switched on: the
    "older entries" fragment is a render of its own (its cost is constant too),
    and the day page never re-reads a window it does not render.
    """
    counts: list[int] = []
    for index in range(1, 4):
        _create_voice(session, _at(9 + index))
        session.commit()
        with _count_queries(engine) as counter:
            response = api_client.get(f"/?date={DAY.isoformat()}")
        assert response.status_code == 200
        counts.append(counter["statements"])
    assert counts == [DAY_PAGE_QUERY_COUNT] * 3, counts

    # A day longer than the feed window (paging on), half of it voice messages.
    monkeypatch.setattr(pages_module, "PAGE_LIMIT", 3)
    for minute in range(12):
        if minute % 2:
            _create_voice(session, _at_minute(minute))
        else:
            _create_text(session, f"запись-{minute}", _at_minute(minute))
    session.commit()
    with _count_queries(engine) as counter:
        response = api_client.get(f"/?date={DAY.isoformat()}")
    assert response.status_code == 200
    assert "Показать более старые записи" in response.text
    assert counter["statements"] == DAY_PAGE_QUERY_COUNT, counter["statements"]
