"""Thin page routes for the server-rendered UI (Tasks 06/10/13/14/15/16/16b).

Mounted at the application root (no prefix; ``/api/v1`` stays the JSON API).

- ``GET /``                              — one day of the diary (task T16b):
  the day's entries (date-as-unit, counters, day navigation) with the compose
  form, the "load older" control when that day holds more than one window and
  the processing panel in the shell's third column (task T16c);
- ``GET /messages/fragments/older``      — HTML fragment of the next older
  window *of the day the cursor belongs to* (htmx ``afterbegin`` into
  ``#message-list``) plus an out-of-band ``#older-control`` that carries the
  next cursor (or stops paging);
- Task 16 exports/processing management UI:
  ``GET /exports`` (weekly exports management), ``GET /exports/{export_id}``
  (per-export lineage detail), ``GET /processing`` (recent jobs) and
  ``GET /processing/fragments/jobs`` (htmx polling fragment).

Rendering decisions
===================
* Timestamps are converted to the application timezone
  (``AppSettings.tzinfo``) and formatted as ``%d.%m.%Y %H:%M``; an entry card
  additionally carries the wall-clock-only ``%H:%M`` (the day is the page's
  unit, so the date lives in the day header). The ``<time datetime>``
  attribute keeps the full ISO-8601 value for the page's own use.
* The day itself is a *local calendar day* in the application timezone, using
  the very same definition as the daily-summary layer
  (:func:`diary.domain.pipelines.sum_daily.day_input.day_bounds`, ADR-007 §1),
  so the page and the pipeline can never disagree about which day a message
  belongs to. No timezone arithmetic is re-implemented here.
* All user text is escaped by Jinja2 autoescape (default for ``.html``).
* The keyset cursor format is owned by the messages API
  (``diary.api.routes.messages``); this module imports the codec instead of
  duplicating it so page and API paging can never drift apart.
* Page actions never run work inside a page request: they POST to the JSON
  API and the client polls the returned job (single Task-16 pattern shared
  with the Task-09 transcription flow).
"""

from __future__ import annotations

import uuid
from collections.abc import Collection
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from diary.api.routes.messages import _decode_cursor, _encode_cursor, get_session
from diary.core.models_config import (
    ModelProfile,
    ModelsConfigError,
    OpenAICompatibleProfile,
    ProviderType,
    TaskType,
    load_models_config,
)
from diary.core.profile_status import ProfileStatus, profile_statuses
from diary.core.settings import get_settings
from diary.db.base import utc_now
from diary.db.models import (
    Artifact,
    ArtifactLayer,
    AudioFile,
    Job,
    JobStatus,
    JobType,
    Message,
    MessageKind,
    PipelineType,
    ProcessingRun,
    RunStatus,
    Transcription,
    TranscriptionStatus,
    WeeklyExport,
)
from diary.db.repositories import (
    ArtifactRepository,
    AudioFileRepository,
    JobRepository,
    MessageRepository,
    ProcessingRunRepository,
    TranscriptionRepository,
    WeeklyExportRepository,
)
from diary.domain.chat.service import MessageCursor
from diary.domain.exports.service import export_filename, list_exports
from diary.domain.exports.weeks import sunday_date_for
from diary.domain.pipelines.naming import (
    JOB_PIPELINE_TYPE,
    PIPELINE_JOB_TYPE,
)
from diary.domain.pipelines.sum_daily.day_input import day_bounds, local_date_of
from diary.domain.pipelines.wellbeing.safety import (
    ESCALATION_LINES,
    ESCALATION_TITLE,
    SafetyLevel,
    classify,
)
from diary.domain.transcription.service import (
    find_active_transcription_job,
    find_active_transcription_jobs,
)

#: This module lives at ``src/diary/api/routes/pages.py``.
_PACKAGE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = _PACKAGE_DIR / "templates"
STATIC_DIR = _PACKAGE_DIR / "static"

#: Entries per day window (the initial page and each "older" fragment).
PAGE_LIMIT = 50

#: Wall-clock display format for message timestamps (application timezone).
#: Numeric-only codes: the rendering does not depend on the process locale.
TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M"

#: Wall-clock-only format for entry cards (task T16b): the day is the page's
#: unit, so the card shows the time and the day header carries the date.
TIME_FORMAT = "%H:%M"

#: Weekday titles for the day header, indexed by ``date.weekday()``. Explicit
#: tables instead of locale data: like the formats above, the rendering must
#: not depend on the process locale (a Russian page on an English machine).
WEEKDAY_TITLES: tuple[str, ...] = (
    "Понедельник",
    "Вторник",
    "Среда",
    "Четверг",
    "Пятница",
    "Суббота",
    "Воскресенье",
)

#: Month names in the genitive case ("20 сентября").
MONTH_TITLES: tuple[str, ...] = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)

#: Russian plural forms of "запись" / "голосовое сообщение" (one, few, many).
ENTRY_FORMS: tuple[str, str, str] = ("запись", "записи", "записей")
VOICE_FORMS: tuple[str, str, str] = (
    "голосовое сообщение",
    "голосовых сообщения",
    "голосовых сообщений",
)

#: Shown when ``?date=`` cannot be read as an unambiguous local calendar day.
#: The page then falls back to today: a malformed link must never be an error
#: page (and never a 500).
INVALID_DAY_NOTICE = "Дата в адресе не распознана — показан сегодняшний день."

#: Human-readable transcription-state captions for audio messages.
TRANSCRIPTION_LABELS: dict[TranscriptionStatus, str] = {
    TranscriptionStatus.not_requested: "Не расшифровано",
    TranscriptionStatus.queued: "Ожидает расшифровки",
    TranscriptionStatus.running: "Расшифровывается…",
    TranscriptionStatus.succeeded: "Расшифровано",
    TranscriptionStatus.failed: "Ошибка расшифровки",
}

#: UI-only state of a voice card whose ``audio_files`` row is missing (T31).
#: Not a :class:`TranscriptionStatus` value: the row the status would describe
#: does not exist. It is carried in the template context so the state is an
#: explicit value rather than an undefined key, and it is never rendered as a
#: ``data-transcription-state`` (no transcription block is rendered at all —
#: the contract with ``static/js/app.js`` stays untouched).
AUDIO_LINK_MISSING_STATE = "missing"

# ---------------------------------------------------------------------------
# Task 16 — labels shared by the exports/processing management templates
# ---------------------------------------------------------------------------

#: Captions of the three LLM pipelines (used for run/job rows and actions).
PIPELINE_LABELS: dict[PipelineType, str] = {
    PipelineType.normalize: "Нормализация",
    PipelineType.summarize: "Саммаризация",
    PipelineType.wellbeing: "Wellbeing-рефлексия",
    # Phase-2 layer (task T04, second anchor family): per-message STT fix.
    PipelineType.stt_corrected: "Правка STT",
    # Phase-2 layer (task T05, second anchor family): per-day summary.
    PipelineType.daily_summary: "Саммари дня",
    # Phase-2 layer (task T06, second anchor family): per-week merge.
    PipelineType.weekly_summary: "Саммари недели",
    # Phase-2 layer (task T07, second anchor family, layer 5): cautious coaching
    # note over one weekly summary — separate from the wellbeing layer 4.
    PipelineType.coach_note: "Заметка коуча",
}

#: Captions of durable job types (processing page).
JOB_TYPE_LABELS: dict[JobType, str] = {
    JobType.transcribe_audio: "Расшифровка аудио",
    JobType.normalize: "Нормализация",
    JobType.summarize: "Саммаризация",
    JobType.wellbeing: "Wellbeing-рефлексия",
    # Phase-2 layer (task T04, second anchor family).
    JobType.stt_corrected: "Правка STT",
    # Phase-2 layer (task T05, second anchor family).
    JobType.daily_summary: "Саммари дня",
    # Phase-2 layer (task T06, second anchor family).
    JobType.weekly_summary: "Саммари недели",
    # Phase-2 layer (task T07, second anchor family, layer 5).
    JobType.coach_note: "Заметка коуча",
}

#: Captions of job statuses (processing page).
JOB_STATUS_LABELS: dict[JobStatus, str] = {
    JobStatus.pending: "В очереди",
    JobStatus.running: "Выполняется",
    JobStatus.retrying: "Ожидает повтора",
    JobStatus.succeeded: "Успешно",
    JobStatus.failed: "Ошибка",
    JobStatus.cancelled: "Отменено",
}

#: Captions of processing-run statuses (export detail page).
RUN_STATUS_LABELS: dict[RunStatus, str] = {
    RunStatus.queued: "В очереди",
    RunStatus.running: "Выполняется",
    RunStatus.succeeded: "Успешно",
    RunStatus.failed: "Ошибка",
}

#: The three LLM pipelines in display order (normalize → summarize → wellbeing).
_PIPELINE_ORDER: tuple[PipelineType, ...] = (
    PipelineType.normalize,
    PipelineType.summarize,
    PipelineType.wellbeing,
)

#: LLM pipeline type → the artifact layer it produces.
_PIPELINE_LAYER: dict[PipelineType, ArtifactLayer] = {
    PipelineType.normalize: ArtifactLayer.normalized,
    PipelineType.summarize: ArtifactLayer.summary,
    PipelineType.wellbeing: ArtifactLayer.wellbeing,
    # Phase-2 layer (task T04, second anchor family); it is anchored to a
    # message, so it never appears in the export-scoped run lists above.
    PipelineType.stt_corrected: ArtifactLayer.stt_corrected,
    # Phase-2 layer (task T05, second anchor family); it is anchored to a local
    # day, so it never appears in the export-scoped run lists above.
    PipelineType.daily_summary: ArtifactLayer.daily_summary,
    # Phase-2 layer (task T06, second anchor family); it is anchored to an ISO
    # week, so it never appears in the export-scoped run lists above.
    PipelineType.weekly_summary: ArtifactLayer.weekly_summary,
    # Phase-2 layer (task T07, second anchor family); it is anchored to an
    # upstream weekly-summary artifact, so it never appears in the
    # export-scoped run lists above.
    PipelineType.coach_note: ArtifactLayer.coach_note,
}

#: Titles of artifact-layer sections on the exports/detail pages.
LAYER_TITLES: dict[ArtifactLayer, str] = {
    ArtifactLayer.original: "Оригинал (слой 1)",
    ArtifactLayer.normalized: "Нормализованный (слой 2)",
    ArtifactLayer.summary: "Саммари (слой 3)",
    ArtifactLayer.wellbeing: "Wellbeing-рефлексия (слой 4)",
    # Phase-2 layer (task T04): the phase-2 layers are not numbered 1-4.
    ArtifactLayer.stt_corrected: "Правка STT (фаза 2)",
    # Phase-2 layer (task T05).
    ArtifactLayer.daily_summary: "Саммари дня (фаза 2)",
    # Phase-2 layer (task T06).
    ArtifactLayer.weekly_summary: "Саммари недели (фаза 2)",
    # Phase-2 layer (task T07): layer 5, deliberately not merged with layer 4.
    ArtifactLayer.coach_note: "Заметка коуча (фаза 2, слой 5)",
}

#: Short per-layer captions for compact chips/links.
LAYER_SHORT_TITLES: dict[ArtifactLayer, str] = {
    ArtifactLayer.original: "Оригинал",
    ArtifactLayer.normalized: "Нормализация",
    ArtifactLayer.summary: "Саммари",
    ArtifactLayer.wellbeing: "Wellbeing",
    ArtifactLayer.stt_corrected: "Правка STT",
    # Phase-2 layer (task T05).
    ArtifactLayer.daily_summary: "Саммари дня",
    # Phase-2 layer (task T06).
    ArtifactLayer.weekly_summary: "Саммари недели",
    # Phase-2 layer (task T07).
    ArtifactLayer.coach_note: "Заметка коуча",
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

    This is the *single-message* path (one message, at most three reads); the
    day page renders a whole window and uses the batch path
    (:func:`_audio_views_for_messages`) instead, which produces the very same
    fields. Returns ``None`` when the linkage is missing — text messages never
    reach here — and the caller then uses
    :func:`_missing_audio_fields` (task T31).
    """
    audio_files = AudioFileRepository(session).get_by_message(message_id)
    if not audio_files:
        return None
    audio = audio_files[-1]
    transcriptions = TranscriptionRepository(session).list_by_audio_file(audio.id)
    active_job = find_active_transcription_job(session, audio.id)
    return _audio_fields(audio, transcriptions, active_job)


def _audio_fields(
    audio: AudioFile,
    transcriptions: list[Transcription],
    active_job: Job | None,
) -> dict[str, Any]:
    """Map one audio file + its transcription rows to template fields.

    Pure mapping shared by :func:`_audio_view` (one card) and
    :func:`_audio_views_for_messages` (a whole page window), so the two paths
    cannot disagree about a status or a button label. ``transcriptions`` is the
    file's history oldest first; the effective status is the in-flight job's
    (it wins over the stored row — the page reload during processing must not
    look idle), otherwise the newest row's.
    """
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
        "audio_available": True,
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


def _missing_audio_fields() -> dict[str, Any]:
    """Template fields of a voice card whose ``audio_files`` row is missing.

    Task T31: an audio message without its audio-file row used to leave every
    audio key undefined, so the card rendered ``id="transcription-"`` (an
    empty anchor), ``<audio src="">`` and a «Расшифровать» button with an empty
    caption — dead controls and a broken jump target. The state is now stated
    explicitly and the template renders the honest caption of the panel
    (:data:`PANEL_AUDIO_FILE_MISSING`) instead of a player and a button.

    Every key the available state carries is present here with an explicit
    ``None``/``False``, so the template always reads a defined value
    (``audio_available`` is the one flag it branches on).
    """
    return {
        "audio_available": False,
        "audio_file_id": None,
        "audio_url": None,
        "audio_mime": None,
        # UI-only state of a missing linkage; it is never rendered as a
        # ``data-transcription-state`` value (no transcription block exists).
        "transcription_status": AUDIO_LINK_MISSING_STATE,
        "transcription_label": PANEL_AUDIO_FILE_MISSING,
        "transcription_raw": None,
        "transcription_button_label": None,
        "active_job_id": None,
    }


def _audio_views_for_messages(
    session: Session, message_ids: Collection[uuid.UUID]
) -> dict[uuid.UUID, dict[str, Any]]:
    """Card audio state of many messages in a constant number of reads (T31).

    The batch path of :func:`_audio_view`, used by the day page and by the
    "older" fragment: one read of the ``audio_files`` rows of every message,
    one of their ``transcriptions`` and one of the in-flight
    ``transcribe_audio`` jobs — three reads for the whole window, whatever its
    number of voice messages (before T31 it was three reads *per card*, and
    the panel added three more per voice row).

    The returned mapping holds an entry only for messages that have an
    ``audio_files`` row; a voice message without one is absent, and its caller
    uses :func:`_missing_audio_fields` (never ``None`` and never an undefined
    key in the template context).
    """
    if not message_ids:
        return {}
    audio_by_message: dict[uuid.UUID, AudioFile] = {}
    for audio in AudioFileRepository(session).list_by_messages(message_ids):
        # The repository groups a message's rows oldest first, so the last one
        # is the row the single-message path picks.
        audio_by_message[audio.message_id] = audio
    if not audio_by_message:
        return {}
    transcriptions_by_audio: dict[uuid.UUID, list[Transcription]] = {}
    for transcription in TranscriptionRepository(session).list_by_audio_files(
        [audio.id for audio in audio_by_message.values()]
    ):
        transcriptions_by_audio.setdefault(transcription.audio_file_id, []).append(transcription)
    active_jobs = find_active_transcription_jobs(
        session, [audio.id for audio in audio_by_message.values()]
    )
    return {
        message_id: _audio_fields(
            audio,
            transcriptions_by_audio.get(audio.id, []),
            active_jobs.get(audio.id),
        )
        for message_id, audio in audio_by_message.items()
    }


def _message_views(records: Collection[Message], session: Session) -> list[dict[str, Any]]:
    """Template fields of many messages — the batch render path (task T31).

    Reads the audio linkage of the whole window with the constant number of
    reads of :func:`_audio_views_for_messages` instead of once per card. The
    produced fields are identical to :func:`_message_view` for every message,
    voice messages without an ``audio_files`` row included (they get
    :func:`_missing_audio_fields`).
    """
    records = list(records)
    audio_views = _audio_views_for_messages(
        session, [record.id for record in records if record.kind is MessageKind.audio]
    )
    return [
        _message_fields(
            record,
            audio_views.get(record.id) if record.kind is MessageKind.audio else None,
        )
        for record in records
    ]


def _message_fields(record: Message, audio: dict[str, Any] | None) -> dict[str, Any]:
    """Template-ready fields of one message in the application timezone.

    ``audio`` is the resolved audio state for a voice message — a
    :func:`_audio_fields` mapping when the ``audio_files`` row exists, ``None``
    when it is missing (then :func:`_missing_audio_fields` states it honestly).
    For text messages ``audio`` is ``None`` as well and is simply not used.
    All values originate from the database, never from user input, and are
    escaped by Jinja2 at render time.
    """
    settings = get_settings()
    created_local = record.created_at.astimezone(settings.tzinfo)
    view: dict[str, Any] = {
        "id": str(record.id),
        "kind": record.kind.value,
        "created_at_iso": created_local.isoformat(),
        "created_at_display": created_local.strftime(TIMESTAMP_FORMAT),
        # T16b: the card shows only the wall-clock time — the date is in the
        # day header, which is what the page is about.
        "created_time_display": created_local.strftime(TIME_FORMAT),
        "body": record.body_text,
    }
    if record.kind is MessageKind.audio:
        view.update(_missing_audio_fields() if audio is None else audio)
    return view


def _message_view(record: Message, session: Session) -> dict[str, Any]:
    """Template fields of a single message — the one-message path (task T31).

    The counterpart of the batch path :func:`_message_views` for a caller that
    renders exactly one card: it resolves the audio state with the
    single-message :func:`_audio_view` (at most three reads) and shares
    :func:`_message_fields` with the batch path, so the two produce identical
    fields. A voice message without an ``audio_files`` row gets the explicit
    :func:`_missing_audio_fields` state here as well.

    **No page calls this any more:** the day page and the older fragment render
    through :func:`_message_views`. It stays as the oracle the parity test pins
    the batch path against (which ``audio_files`` row and which in-flight job a
    card must pick), so the two paths cannot silently drift apart.
    """
    audio = _audio_view(session, record.id) if record.kind is MessageKind.audio else None
    return _message_fields(record, audio)


def _nav_context(session: Session) -> dict[str, Any]:
    """Sidebar counters of the shared shell (task T16a).

    Both numbers come from the database and are never invented:
    ``exports_count`` is the number of weekly export revisions stored,
    ``active_jobs`` the number of jobs still in flight (pending / running /
    retrying) across every job type. Only row counts are read — no diary
    content and no job payload.

    Which item is highlighted travels separately as the ``nav_current``
    template variable, so a page that deliberately stays database-free (the
    privacy page) still highlights its own item; pages that pass no ``nav``
    simply render no counters (``base.html`` guards every use).
    """
    job_repository = JobRepository(session)
    active_jobs = sum(len(job_repository.list_active_of_type(job_type)) for job_type in JobType)
    return {
        "exports_count": len(WeeklyExportRepository(session).list_all_by_sequence()),
        "active_jobs": active_jobs,
    }


@pages_router.get("/", response_class=HTMLResponse)
def chat_page(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    date_raw: Annotated[str | None, Query(alias="date", max_length=32)] = None,
) -> HTMLResponse:
    """Render one local calendar day of the diary (task T16b).

    The day is the page's unit: the header names it, the counters count it and
    the feed holds its entries only. ``?date=`` selects the day
    (``YYYY-MM-DD``, a local calendar date in the application timezone); with
    no parameter — or with an unusable one — the page shows today plus a
    notice, never a 500 (``INVALID_DAY_NOTICE``).

    The day's entries come from :meth:`MessageRepository.list_active_between`
    — the very same day window the weekly export and the daily-summary layer
    use, computed by the *same* ``day_bounds`` helper (ADR-007 §1) — so page
    and pipelines cannot disagree about what "one day" contains. Soft-deleted
    messages are excluded by that repository method (task T15 policy).

    Only the newest :data:`PAGE_LIMIT` entries of a very long day are rendered
    initially; the "older" control then pages *inside that day*
    (:func:`older_messages_fragment`), so nothing is silently dropped and the
    feed never spills over into the neighbouring days.

    The processing panel (task T16c) describes the *whole* requested day, not
    the rendered window: its counters and rows are built from ``day_messages``
    by :func:`_day_processing_panel`. Its jump actions, however, are limited to
    the messages of the rendered ``window`` — the panel is rendered with the
    same window the feed holds, so a row can never point at a card that is not
    on the page. This is the only page that sets ``context_panel``, so the
    shell reserves the third column here and nowhere else.
    """
    settings = get_settings()
    today = local_date_of(utc_now(), settings.tzinfo)
    day, day_notice = _requested_day(date_raw, today=today, tz=settings.tzinfo)
    day_messages = _day_messages(session, day, settings.tzinfo)
    window, next_cursor = _day_history_window(day_messages, limit=PAGE_LIMIT)
    older_cursor = _encode_cursor(next_cursor) if next_cursor is not None else None
    day_fields = _day_view(day, today=today, messages=day_messages, notice=day_notice)
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "messages": _message_views(window, session),
            "has_more": next_cursor is not None,
            # Pre-quoted for the hx-get URL; Jinja never sees raw "+" or "|".
            "older_cursor": quote(older_cursor, safe="") if older_cursor else None,
            "max_message_length": settings.max_message_length,
            "max_audio_size_mb": settings.max_audio_size_mb,
            # Client-side ``accept`` for the file-upload fallback.
            "audio_accept": ",".join(settings.allowed_audio_mime_types),
            "nav_current": "chat",
            "nav": _nav_context(session),
            # The day page is the only page that fills the third column (T16c).
            "context_panel": True,
            "panel": _day_processing_panel(
                session,
                day=day,
                day_title=day_fields["day_title"],
                messages=day_messages,
                # The very window the feed below renders: a panel action may
                # only point at a card that is really on the page.
                rendered_ids={message.id for message in window},
            ),
            **day_fields,
        },
    )


def _russian_plural(count: int, forms: tuple[str, str, str]) -> str:
    """The Russian plural form of ``count``: one / few / many.

    Russian needs three forms and the 11-14 teens behave like "many". The UI
    counts real rows, so the sentence has to agree with the number.
    """
    if count % 10 == 1 and count % 100 != 11:
        return forms[0]
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return forms[1]
    return forms[2]


def _day_title(day: date, today: date) -> str:
    """Human-readable day title: "Воскресенье, 20 сентября" (plus the year when
    the day is not in the current year, so the header is never ambiguous)."""
    title = f"{WEEKDAY_TITLES[day.weekday()]}, {day.day} {MONTH_TITLES[day.month - 1]}"
    if day.year != today.year:
        title = f"{title} {day.year}"
    return title


def _day_counters(messages: list[Message]) -> str:
    """The day's counters, counted, never invented.

    "3 записи · из них 1 голосовое сообщение" — the totals of the whole day
    (not of the rendered window), exactly as stored. An empty day says so
    instead of showing zeros.
    """
    total = len(messages)
    if total == 0:
        return "Записей нет"
    voice = sum(1 for message in messages if message.kind is MessageKind.audio)
    counters = f"{total} {_russian_plural(total, ENTRY_FORMS)}"
    if voice:
        counters = f"{counters} · из них {voice} {_russian_plural(voice, VOICE_FORMS)}"
    return counters


def _day_messages(session: Session, day: date, tz: tzinfo) -> list[Message]:
    """Every active message of the local calendar day ``day``, oldest first.

    The window (``day_bounds``) is the shared definition of "that local day";
    the query itself is the existing
    :meth:`MessageRepository.list_active_between` — no day-shaped SQL is
    written here. Soft-deleted messages are excluded by that method.
    """
    start, end = day_bounds(day, tz)
    return MessageRepository(session).list_active_between(start=start, end=end)


def _local_day_bounds(day: date, tz: tzinfo) -> tuple[datetime, datetime] | None:
    """The UTC instants of local day ``day``, or ``None`` if it has none.

    ``date.fromisoformat`` happily returns dates whose local day window cannot
    be represented (``0001-01-01`` with a positive offset, ``9999-12-31`` —
    the next local midnight overflows). Those are treated as "not a usable
    date" so that a hand-written URL can never reach an unhandled overflow.
    """
    try:
        return day_bounds(day, tz)
    except (OverflowError, ValueError):
        return None


def _requested_day(raw: str | None, *, today: date, tz: tzinfo) -> tuple[date, str | None]:
    """The day to render plus an optional notice about the request.

    Returns ``(today, None)`` when no date was asked for, and
    ``(today, INVALID_DAY_NOTICE)`` when ``?date=`` is unreadable or names a
    day whose window cannot be computed. A named but merely *empty* day (a
    future date, a date before the diary existed) is a perfectly normal day
    and renders as such.
    """
    if raw is None or not raw.strip():
        return today, None
    try:
        day = date.fromisoformat(raw.strip())
    except ValueError:
        return today, INVALID_DAY_NOTICE
    if _local_day_bounds(day, tz) is None:
        return today, INVALID_DAY_NOTICE
    return day, None


def _day_history_window(
    day_messages: list[Message],
    *,
    limit: int,
    older_than: MessageCursor | None = None,
) -> tuple[list[Message], MessageCursor | None]:
    """The ``limit`` newest messages of a day, plus the next-page cursor.

    ``day_messages`` is one day in chronological order (as
    :meth:`MessageRepository.list_active_between` returns it). With
    ``older_than`` set only the messages strictly older than that keyset
    position are considered, which is exactly the contract of the API paging
    (:func:`diary.domain.chat.service.list_messages`) — except that the day
    bounds the window here, so paging can never walk into the day before and
    "load older" always terminates.

    The returned cursor is the position of the oldest returned message when
    the day still holds older ones, else ``None``.
    """
    older = day_messages
    if older_than is not None:
        older = [
            message for message in day_messages if (message.created_at, message.id) < older_than
        ]
    window = older[-limit:]
    next_cursor: MessageCursor | None = None
    if len(older) > len(window):
        oldest = window[0]
        next_cursor = (oldest.created_at, oldest.id)
    return window, next_cursor


def _day_empty_text(*, is_today: bool, today_title: str) -> str:
    """The empty state of a day without entries (task T16b).

    Two sentences and no invented promise: the first says what is true about
    this day, and on another day the second one states where a freshly written
    entry will actually go (an entry can only be created "now").
    """
    text = "Пока нет записей за этот день."
    if not is_today:
        text = (
            f"{text} Запись, созданная сейчас, попадёт в сегодняшний день ({today_title.lower()})."
        )
    return f"{text} Напишите первую ниже или выберите другой день в шапке дня."


def _compose_notice(day_title: str, today_title: str) -> str:
    """What the composer says while a day that is not today is on screen.

    An entry can only be created "now", so the composer states where a new
    entry will actually go instead of silently writing into another day than
    the one being displayed.
    """
    return (
        f"Показан день «{day_title.lower()}». "
        f"Новая запись будет создана в сегодняшнем дне ({today_title.lower()})."
    )


def _day_view(
    day: date,
    *,
    today: date,
    messages: list[Message],
    notice: str | None,
) -> dict[str, Any]:
    """Template fields of the day header (title, counters, day navigation).

    The previous/next links are plain links to the neighbouring local days and
    are *not* capped at today: claiming "there is nothing after today" would
    be an unverified statement about the data (nothing stops an imported row
    from carrying a later timestamp). A future day is simply an empty day, and
    the empty state says so. ``day_is_today`` drives the composer hint and the
    "back to today" link — both factual, both from the clock.
    """
    previous_day = day - timedelta(days=1)
    next_day = day + timedelta(days=1)
    is_today = day == today
    relative = "Сегодня" if is_today else ("Вчера" if day == today - timedelta(days=1) else None)
    today_title = _day_title(today, today)
    return {
        "day_iso": day.isoformat(),
        "day_title": _day_title(day, today),
        "day_relative": relative,
        "day_is_today": is_today,
        "day_error": notice,
        "day_counters": _day_counters(messages),
        "day_entries_total": len(messages),
        "day_voice_total": sum(1 for message in messages if message.kind is MessageKind.audio),
        "day_empty_text": _day_empty_text(is_today=is_today, today_title=today_title),
        "day_compose_notice": _compose_notice(_day_title(day, today), today_title),
        "prev_day_url": f"/?date={previous_day.isoformat()}",
        "prev_day_title": _day_title(previous_day, today),
        "next_day_url": f"/?date={next_day.isoformat()}",
        "next_day_title": _day_title(next_day, today),
    }


# ---------------------------------------------------------------------------
# Task T16c — the day page's processing panel
# ---------------------------------------------------------------------------

#: Day-block step captions; the step rows follow the mockup's pipeline.
PANEL_ORIGINALS_TITLE = "Оригиналы"
PANEL_AUDIO_TITLE = "Аудио"
PANEL_TRANSCRIPTION_TITLE = "Транскрипция"

#: Reference note shown with the wellbeing layer (T16c). The mockup's short
#: sentence, which says the same as ``wellbeing.safety.DISCLAIMER`` (layer 5's
#: disclaimer spells it out) — nothing new is promised here.
PANEL_WELLBEING_NOTE_TITLE = "Деликатная рефлексия"
PANEL_WELLBEING_NOTE = "Wellbeing-слой не ставит диагнозов и не заменяет профессиональную помощь."

#: The panel renders server state once, at page render time. Saying so keeps a
#: number honest after the user starts a transcription from the card below
#: (that flow updates the card, not the panel — there is exactly one job
#: poller in the product, see ``static/js/app.js``).
PANEL_SNAPSHOT_NOTE = "Статусы — по данным на момент загрузки страницы."

#: Caption of the panel's link to the weekly export detail page.
PANEL_EXPORT_LINK = "Открыть выгрузку недели"

#: Caption of the fallback link when this week has no export at all.
PANEL_NO_EXPORT_TEXT = "Выгрузки за эту неделю нет."
PANEL_CREATE_EXPORT_LINK = "Создать выгрузку недели"

#: Caption of a voice row whose audio-file row is missing entirely.
PANEL_AUDIO_FILE_MISSING = "Аудиофайл не найден"
PANEL_LAYER_NOT_RUN = "Не посчитано"

#: Mark of a voice row whose message lies *outside* the rendered feed window
#: (a day longer than :data:`PAGE_LIMIT`). The row keeps its real status — the
#: day block counts the whole day — but offers no jump, because the card it
#: would point at is not on the page yet; the words name the control that
#: brings it in ("Показать более старые записи").
PANEL_VOICE_OLDER_NOTE = "в более старых записях"


def _pipeline_model_chips() -> dict[str, str]:
    """Per-layer model chips for the weekly block, from the real configuration.

    Keyed by pipeline value (``normalize`` / ``summarize`` / ``wellbeing``) and
    built from the *actual* profiles: the model name comes from
    ``configs/models.yaml`` (e.g. ``diary-stt``) and the locality word is
    derived from the profile's ``external`` flag — never asserted. A missing or
    invalid configuration yields no chips at all, so the panel cannot claim a
    model it does not know. Model names in the mockup (``llama3.2``) are
    deliberately not reproduced.
    """
    try:
        config = _load_models_config()
    except ModelsConfigError:
        return {}
    profiles = {
        "normalize": config.normalization,
        "summarize": config.summarization,
        "wellbeing": config.wellbeing_analysis,
    }
    return {
        key: f"{profile.model} · {'внешний API' if profile.external else 'локально'}"
        for key, profile in profiles.items()
    }


def _transcription_chip() -> str | None:
    """Chip of the STT step (day block) from the configured transcription profile.

    ``None`` when no usable configuration is available: the panel then shows no
    chip instead of guessing a model or a provider.
    """
    try:
        config = _load_models_config()
    except ModelsConfigError:
        return None
    profile = config.transcription
    return f"{profile.provider.replace('_', '-')} · {profile.model}"


def _panel_step(
    *,
    key: str,
    title: str,
    detail: str,
    state: str | None,
    chip: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """One pipeline step of the panel (mockup's ``.pipeline-step``).

    ``state`` is ``"done"``, ``"pending"`` or ``None`` (nothing to report) and
    only drives the icon; the fact itself is always in ``detail``, so the state
    is readable as text and not only as a colour. ``url`` — when given — is an
    existing page of the product (never a placeholder action).
    """
    return {
        "key": key,
        "title": title,
        "detail": detail,
        "state": state,
        "chip": chip,
        "url": url,
    }


def _day_voice_rows(
    session: Session, messages: list[Message], *, rendered_ids: Collection[uuid.UUID]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Per-voice rows of the day block plus the transcription counters.

    Every audio message of the day gets one row: its wall-clock time, the
    transcription state as the card below words it (the caption and the button
    label come from the very same mapping the card is rendered with — the
    state is never computed twice) and, when the transcription can actually be
    started, a *link to that card's own button*
    (``#transcription-<audio file id>``).

    The day's audio state is loaded through :func:`_audio_views_for_messages`
    (task T31): three reads for the whole day instead of three per voice
    message, and the same three fields the card path produces, so the panel
    and the feed cannot drift apart. The effective status is unchanged: an
    in-flight ``transcribe_audio`` job wins over the status of the newest
    stored row.

    ``messages`` is the whole day while the feed renders only its newest
    window, so ``rendered_ids`` — the ids of the messages actually on the page
    (built by :func:`chat_page` from the very window it renders) — gates the
    jump: a row of a message outside that window carries ``note``
    (:data:`PANEL_VOICE_OLDER_NOTE`) instead of an action, and never an
    ``href`` to a missing anchor. The row itself is *not* dropped: the block
    counts the whole day, and hiding a record the counters do count would make
    the block disagree with itself.

    The link deliberately replaces a second button: the STT action and its job
    poller exist exactly once in the product (``static/js/app.js``), so a
    second trigger would either duplicate the polling loop or need new client
    logic. While a job is in flight the card's button is disabled, so no action
    is offered here either.

    Counters: ``transcribed`` (succeeded), ``awaiting`` (not requested or
    queued — waiting for a manual run), ``running``, ``failed`` and ``missing``
    (an audio message without an ``audio_files`` row: reported as such, not
    counted as awaiting).
    """
    settings = get_settings()
    rows: list[dict[str, Any]] = []
    counters = {
        "total": len(messages),
        "voice": 0,
        "transcribed": 0,
        "awaiting": 0,
        "running": 0,
        "failed": 0,
        "missing": 0,
    }
    # Task T31: the whole day's audio state is read in a constant number of
    # queries (they were three per voice message before), so the panel's cost
    # no longer grows with the number of recordings. Every field below comes
    # from the same mapping the cards are rendered from, so a row and its card
    # can never disagree about a state or a button label.
    audio_views = _audio_views_for_messages(
        session, [message.id for message in messages if message.kind is MessageKind.audio]
    )
    for message in messages:
        if message.kind is not MessageKind.audio:
            continue
        counters["voice"] += 1
        time_display = message.created_at.astimezone(settings.tzinfo).strftime(TIME_FORMAT)
        audio = audio_views.get(message.id)
        if audio is None:
            counters["missing"] += 1
            rows.append(
                {
                    "time_display": time_display,
                    "status_label": PANEL_AUDIO_FILE_MISSING,
                    "state": None,
                    "action": None,
                    "note": None,
                }
            )
            continue
        status = TranscriptionStatus(audio["transcription_status"])
        if status is TranscriptionStatus.succeeded:
            counters["transcribed"] += 1
        elif status in (TranscriptionStatus.not_requested, TranscriptionStatus.queued):
            counters["awaiting"] += 1
        elif status is TranscriptionStatus.running:
            counters["running"] += 1
        else:
            counters["failed"] += 1
        state = "done" if status is TranscriptionStatus.succeeded else "pending"
        action: dict[str, str] | None = None
        note: str | None = None
        in_flight = audio["active_job_id"] is not None or status in (
            TranscriptionStatus.queued,
            TranscriptionStatus.running,
        )
        if not in_flight and message.id in rendered_ids:
            label = str(audio["transcription_button_label"])
            action = {
                "url": f"#transcription-{audio['audio_file_id']}",
                "label": label,
                "aria_label": f"{label}: перейти к голосовому сообщению за {time_display}",
            }
        elif not in_flight:
            # The card exists in the day, but not in this page: state the real
            # reason instead of offering a jump into nowhere.
            note = PANEL_VOICE_OLDER_NOTE
        rows.append(
            {
                "time_display": time_display,
                "status_label": audio["transcription_label"],
                "state": state,
                "action": action,
                "note": note,
            }
        )
    return rows, counters


def _day_block_steps(counters: dict[str, int], *, stt_chip: str | None) -> list[dict[str, Any]]:
    """The «Записи дня» steps: originals, audio, transcription.

    Counted from the day's rows only; a step that has nothing to report says so
    instead of showing a zero. The transcription step's state is ``done`` only
    when every voice message of the day is transcribed.
    """
    total = counters["total"]
    voice_total = counters["voice"]
    steps = [
        _panel_step(
            key="originals",
            title=PANEL_ORIGINALS_TITLE,
            detail=f"{total} {_russian_plural(total, ENTRY_FORMS)}",
            state="done",
            chip="локальное хранилище",
        )
    ]
    if voice_total == 0:
        steps.append(
            _panel_step(
                key="audio",
                title=PANEL_AUDIO_TITLE,
                detail="Голосовых записей нет",
                state=None,
            )
        )
        steps.append(
            _panel_step(
                key="transcription",
                title=PANEL_TRANSCRIPTION_TITLE,
                detail="Расшифровывать нечего",
                state=None,
                chip=stt_chip,
            )
        )
        return steps

    steps.append(
        _panel_step(
            key="audio",
            title=PANEL_AUDIO_TITLE,
            detail=f"{voice_total} {_russian_plural(voice_total, VOICE_FORMS)}",
            state="done",
        )
    )
    parts = [f"Расшифровано {counters['transcribed']} из {voice_total}"]
    if counters["awaiting"]:
        parts.append(f"ждут запуска {counters['awaiting']}")
    if counters["running"]:
        parts.append(f"расшифровывается {counters['running']}")
    if counters["failed"]:
        parts.append(f"с ошибкой {counters['failed']}")
    if counters["missing"]:
        parts.append(f"без файла {counters['missing']}")
    steps.append(
        _panel_step(
            key="transcription",
            title=PANEL_TRANSCRIPTION_TITLE,
            detail=" · ".join(parts),
            state="done" if counters["transcribed"] == voice_total else "pending",
            chip=stt_chip,
        )
    )
    return steps


def _week_layers_view(session: Session, day: date) -> dict[str, Any]:
    """The «Недельные слои» block: the week's export and its computed layers.

    Normalization, summary and wellbeing are anchored to a *weekly export*
    (ADR-007 §1), not to a message, so this block is the only honest place for
    them. The week is the one containing the day on screen
    (:func:`sunday_date_for`), and the block shows the newest revision of that
    week's export — the one whose layer pages the links open.

    Reuses :func:`_export_management_view`, i.e. the same per-layer view model
    the exports pages render, so the panel and those pages cannot disagree
    about what is computed. When the week has no export the block says so and
    links to ``/exports`` (where an export is created) instead of offering an
    action that cannot work.
    """
    sunday = sunday_date_for(day)
    monday = sunday - timedelta(days=6)
    exports = WeeklyExportRepository(session).list_by_week(sunday)
    view: dict[str, Any] = {
        "week_title": f"{monday.strftime('%d.%m.%Y')} — {sunday.strftime('%d.%m.%Y')}",
        "has_export": bool(exports),
        "exports_count": len(exports),
        "create_url": "/exports",
        "layers": [],
    }
    if not exports:
        return view

    record = exports[-1]  # list_by_week orders by revision: the newest one.
    base = _export_view(record)
    management = _export_management_view(session, record, _active_pipeline_jobs(session))
    layer_urls = {
        "normalize": str(base["compare_url"]),
        "summarize": str(base["summary_url"]),
        "wellbeing": str(base["wellbeing_url"]),
    }
    chips = _pipeline_model_chips()
    active_jobs = management["active_jobs"]
    layers: list[dict[str, Any]] = []
    for item in management["derived"]:
        key = str(item["key"])
        artifact = item.get("artifact")
        artifact_info = artifact if isinstance(artifact, dict) else None
        if artifact_info is not None:
            version_label = artifact_info["version_label"] or "версия"
            detail = f"{version_label} · создано {artifact_info['created_display']}"
            state: str | None = "done"
        elif active_jobs.get(key) is not None:
            # A run of this layer is in flight right now (the exports pages read
            # the same mapping), so "not computed" would be a half-truth.
            detail = "Прогон идёт"
            state = "pending"
        elif item.get("last_run_label"):
            detail = f"Прогон: {item['last_run_label']}"
            in_flight_run = item.get("last_run_status") in (
                RunStatus.queued.value,
                RunStatus.running.value,
            )
            state = "pending" if in_flight_run else None
        else:
            detail = PANEL_LAYER_NOT_RUN
            state = None
        layers.append(
            _panel_step(
                key=key,
                title=LAYER_TITLES[ArtifactLayer(item["layer"])],
                detail=detail,
                state=state,
                chip=chips.get(key),
                url=layer_urls.get(key),
            )
        )
    view.update(
        {
            "export_url": f"/exports/{record.id}",
            "filename": base["filename"],
            "revision": record.revision,
            "layers": layers,
        }
    )
    return view


def _day_processing_panel(
    session: Session,
    *,
    day: date,
    day_title: str,
    messages: list[Message],
    rendered_ids: Collection[uuid.UUID],
) -> dict[str, Any]:
    """Template fields of the day page's processing panel (task T16c).

    Two honest blocks, because the derivational layers do not share one anchor
    (ADR-007 §1, ADR-008): «Записи дня» describes the day itself (originals,
    voice messages, transcription — transcription is stored per audio file),
    and «Недельные слои» describes the weekly export of the week this day
    belongs to (normalization, summary, wellbeing are stored per export).

    ``messages`` is the whole requested day (every number is counted from real
    rows) while ``rendered_ids`` is the message window the feed below renders:
    the day block reports the whole day but only offers jump actions into that
    window, so the panel can never point at a card that is not on the page
    (:func:`_day_voice_rows`).

    Every number is counted from the database and every caption comes either
    from a database row or from the loaded model configuration; nothing is
    invented, and a step with nothing to report says so in words.
    """
    rows, counters = _day_voice_rows(session, messages, rendered_ids=rendered_ids)
    return {
        "day_title": day_title,
        "has_entries": counters["total"] > 0,
        "empty_text": "За этот день записей нет.",
        "steps": _day_block_steps(counters, stt_chip=_transcription_chip()),
        "voice_rows": rows,
        "week": _week_layers_view(session, day),
        "snapshot_note": PANEL_SNAPSHOT_NOTE,
        "wellbeing_note_title": PANEL_WELLBEING_NOTE_TITLE,
        "wellbeing_note": PANEL_WELLBEING_NOTE,
        "export_link_label": PANEL_EXPORT_LINK,
        "no_export_text": PANEL_NO_EXPORT_TEXT,
        "create_export_link_label": PANEL_CREATE_EXPORT_LINK,
    }


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
        "summary_url": f"/exports/{record.id}/summary",
        "wellbeing_url": f"/exports/{record.id}/wellbeing",
    }


# ---------------------------------------------------------------------------
# Task 16 — view helpers for the exports management and processing pages
# ---------------------------------------------------------------------------


def _format_local(value: datetime) -> str:
    """Format an aware timestamp as wall-clock text in the application timezone."""
    return value.astimezone(get_settings().tzinfo).strftime(TIMESTAMP_FORMAT)


def _artifact_download_url(export_id: uuid.UUID, artifact_id: uuid.UUID) -> str:
    """Generic derived-artifact download URL (the Task-12 API route)."""
    return f"/api/v1/exports/{export_id}/artifacts/{artifact_id}/file"


def _artifact_file_path(artifact: Artifact) -> Path:
    """Absolute path of a first-family artifact file.

    A result of the phase-2 second anchor family may be stored inline without a
    rendered file (task T08); every artifact rendered on these pages belongs to
    a weekly export and always has one.
    """
    assert artifact.path is not None
    return get_settings().data_root / artifact.path


def _artifact_file_name(artifact: Artifact) -> str:
    """File name of a first-family artifact file (see :func:`_artifact_file_path`)."""
    assert artifact.path is not None
    return artifact.path.rsplit("/", 1)[-1]


def _load_models_config() -> Any:
    """Return the current model configuration (module-level test seam).

    Page rendering must never fail on a missing/invalid configuration file:
    :func:`_external_profile_info` treats any loading failure as "no external
    profiles" (the local default), so the UI stays usable without a
    ``configs/models.yaml`` present.
    """
    return load_models_config()


def _external_profile_info() -> dict[str, dict[str, object]]:
    """External-provider disclosure info for the LLM pipeline actions.

    Returns a mapping keyed by pipeline value (``normalize`` / ``summarize`` /
    ``wellbeing``); every entry is ``{"external": bool, "host": str | None,
    "model": str | None}``. A profile is external only when the matching
    Task-03 profile sets ``external: true``; the disclosure names the endpoint
    host (never credentials — ``base_url`` forbids them) and the model.
    """
    info: dict[str, dict[str, object]] = {
        key: {"external": False, "host": None, "model": None}
        for key in ("normalize", "summarize", "wellbeing")
    }
    try:
        config = _load_models_config()
    except ModelsConfigError:
        return info
    profiles = {
        "normalize": config.normalization,
        "summarize": config.summarization,
        "wellbeing": config.wellbeing_analysis,
    }
    for key, profile in profiles.items():
        if not profile.external:
            continue
        host = urlsplit(profile.base_url).netloc or None
        info[key] = {"external": True, "host": host, "model": profile.model}
    return info


# ---------------------------------------------------------------------------
# Task 18 — privacy disclosure view (/privacy)
# ---------------------------------------------------------------------------

#: Human-readable task captions for the privacy page, in enum order.
PRIVACY_TASK_LABELS: dict[str, str] = {
    TaskType.transcription.value: "Транскрипция (голос → текст)",
    TaskType.normalization.value: "Нормализация (слой 2)",
    TaskType.summarization.value: "Саммаризация (слой 3)",
    TaskType.wellbeing_analysis.value: "Wellbeing-рефлексия (слой 4)",
}

#: Human-readable provider captions for the privacy page.
PRIVACY_PROVIDER_LABELS: dict[str, str] = {
    ProviderType.faster_whisper.value: "Локальный STT (faster-whisper)",
    ProviderType.openai_compatible.value: "OpenAI-совместимый API",
}


def _privacy_rows() -> tuple[list[dict[str, object]], bool]:
    """Per-task rows for the privacy page plus the "no config" flag.

    Uses :func:`profile_statuses` (Task 03), so only the *booleans*
    ``api_key_configured``/``valid`` and the referenced environment-variable
    name ever reach the page — never a key value. ``external`` mirrors the
    profile's ``external`` flag (faster-whisper is always local). A missing
    or invalid models configuration renders an explicit "not configured"
    state instead of failing the page.
    """
    try:
        config = _load_models_config()
    except ModelsConfigError:
        config = None
    statuses: dict[TaskType, ProfileStatus] = {}
    profiles: dict[TaskType, ModelProfile] = {}
    if config is not None:
        statuses = {status.task: status for status in profile_statuses(config)}
        profiles = dict(config.profiles_by_task())

    rows: list[dict[str, object]] = []
    for task in TaskType:
        status = statuses.get(task)
        profile = profiles.get(task)
        row: dict[str, object] = {
            "key": task.value,
            "label": PRIVACY_TASK_LABELS.get(task.value, task.value),
            "endpoint": None,
            "external": None,
            "key_env": None,
            "key_configured": None,
            "valid": None,
        }
        if config is None or status is None:
            row["provider_label"] = None
            row["model"] = None
            rows.append(row)
            continue
        external = False
        key_env: str | None = None
        if isinstance(profile, OpenAICompatibleProfile):
            external = profile.external
            key_env = profile.api_key_env
        base_url = status.base_url
        row.update(
            {
                "provider_label": PRIVACY_PROVIDER_LABELS.get(status.provider.value),
                "model": status.model,
                "endpoint": urlsplit(base_url).netloc if base_url else None,
                "external": external,
                "key_env": key_env,
                "key_configured": status.api_key_configured,
                "valid": status.valid,
            }
        )
        rows.append(row)
    return rows, config is None


@pages_router.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request) -> HTMLResponse:
    """Privacy disclosure page (Task 18).

    For every task (transcription/normalization/summarization/wellbeing)
    shows the provider, model, endpoint host (never credentials) and whether
    the processing is local or external, plus the key/validity state from
    :func:`profile_statuses`. No secret value is ever rendered; the page only
    states which environment variable a profile references.

    The page renders without a database on purpose (Task 18): it does not
    compute the sidebar counters, only the highlighted navigation item.
    """
    rows, config_missing = _privacy_rows()
    return templates.TemplateResponse(
        request,
        "privacy.html",
        {"rows": rows, "config_missing": config_missing, "nav_current": "privacy"},
    )


def _active_pipeline_jobs(
    session: Session,
) -> dict[tuple[JobType, str], uuid.UUID]:
    """In-flight pipeline jobs by ``(job type, export id)``.

    "In flight" means pending/running/retrying. The mapping lets the templates
    resume job polling after a reload and label running actions; only
    identifiers are used, never payload content.
    """
    repository = JobRepository(session)
    result: dict[tuple[JobType, str], uuid.UUID] = {}
    for pipeline in _PIPELINE_ORDER:
        job_type = PIPELINE_JOB_TYPE[pipeline]
        for job in repository.list_active_of_type(job_type):
            payload = job.payload_json if isinstance(job.payload_json, dict) else None
            export_id = payload.get("export_id") if payload is not None else None
            if export_id is not None:
                result[(job_type, str(export_id))] = job.id
    return result


def _export_management_view(
    session: Session,
    record: WeeklyExport,
    active_jobs: dict[tuple[JobType, str], uuid.UUID],
) -> dict[str, Any]:
    """Template fields of one export row for the management pages (Task 16).

    Extends the base :func:`_export_view` with the detail-page link, the
    original-file presence, an in-flight job handle per pipeline and the
    per-layer overview (newest artifact version + status of the newest run).
    """
    settings = get_settings()
    view = _export_view(record)
    view["detail_url"] = f"/exports/{record.id}"
    original_path = settings.data_root / record.path
    view["original_present"] = original_path.is_file()

    runs = ProcessingRunRepository(session).list_by_input_export(record.id)
    runs_newest_first = list(reversed(runs))
    derived: list[dict[str, Any]] = []
    for pipeline in _PIPELINE_ORDER:
        layer = _PIPELINE_LAYER[pipeline]
        artifacts = ArtifactRepository(session).list_by_export_and_layer(record.id, layer)
        latest = artifacts[-1] if artifacts else None
        file_present = latest is not None and _artifact_file_path(latest).is_file()
        artifact_info: dict[str, Any] | None = None
        if latest is not None:
            artifact_info = {
                "version_label": f"v{latest.version}" if latest.version is not None else "",
                "filename": _artifact_file_name(latest),
                "download_url": _artifact_download_url(record.id, latest.id),
                "file_present": file_present,
                "created_display": _format_local(latest.created_at),
            }
        last_run = next((run for run in runs_newest_first if run.pipeline_type is pipeline), None)
        derived.append(
            {
                "key": pipeline.value,
                "short_label": LAYER_SHORT_TITLES[layer],
                "layer": layer.value,
                "has_artifact": latest is not None,
                "artifact": artifact_info,
                "last_run_status": last_run.status.value if last_run is not None else None,
                "last_run_label": (
                    RUN_STATUS_LABELS[last_run.status] if last_run is not None else None
                ),
                "run_error_summary": last_run.error_summary if last_run is not None else None,
            }
        )
    view["derived"] = derived
    # "Normalized input available" gates the summarize/wellbeing actions: the
    # newest normalized artifact must exist and its file must be on disk.
    normalized_info = derived[0].get("artifact") if derived else None
    view["has_normalized"] = bool(
        isinstance(normalized_info, dict) and normalized_info.get("file_present")
    )
    active: dict[str, str | None] = {}
    for pipeline in _PIPELINE_ORDER:
        job_type = PIPELINE_JOB_TYPE[pipeline]
        job_id = active_jobs.get((job_type, str(record.id)))
        active[pipeline.value] = str(job_id) if job_id is not None else None
    view["active_jobs"] = active
    return view


def _lineage_artifact_view(
    session: Session, artifact: Artifact, export: WeeklyExport
) -> dict[str, Any]:
    """Template fields of one derived artifact on the export detail page.

    Includes a parent caption that makes the lineage explicit: layer-2
    artifacts are derived from the original export (``NNNN_rN``); layer-3/4
    artifacts name the exact normalized artifact they were built from.
    """
    producing_run = (
        ProcessingRunRepository(session).get(artifact.processing_run_id)
        if artifact.processing_run_id is not None
        else None
    )
    parent_caption = f"производный от оригинала {export.seq_number:04d}_r{export.revision}"
    if artifact.layer in (ArtifactLayer.summary, ArtifactLayer.wellbeing):
        if producing_run is not None and producing_run.input_artifact_id is not None:
            input_artifact = ArtifactRepository(session).get(producing_run.input_artifact_id)
            if input_artifact is not None:
                parent_caption = "создан из нормализованного " + _artifact_file_name(input_artifact)
    file_present = _artifact_file_path(artifact).is_file()
    assert artifact.sha256 is not None  # a first-family artifact always has a file hash
    return {
        "id": str(artifact.id),
        "layer": artifact.layer.value,
        "filename": _artifact_file_name(artifact),
        "version_label": f"v{artifact.version}" if artifact.version is not None else "",
        "sha_prefix": artifact.sha256[:16],
        "created_display": _format_local(artifact.created_at),
        "download_url": _artifact_download_url(export.id, artifact.id),
        "file_present": file_present,
        "parent_caption": parent_caption,
        "run_status_label": (
            RUN_STATUS_LABELS[producing_run.status] if producing_run is not None else None
        ),
    }


def _lineage_sections(session: Session, export: WeeklyExport) -> list[dict[str, Any]]:
    """Lineage sections (original → normalized → summary/wellbeing) of an export."""
    settings = get_settings()
    original_path = settings.data_root / export.path
    sections: list[dict[str, Any]] = [
        {
            "key": ArtifactLayer.original.value,
            "title": LAYER_TITLES[ArtifactLayer.original],
            "is_original": True,
            "is_derived": False,
            "input_note": None,
            "empty_message": None,
            "rows": [
                {
                    "filename": export_filename(
                        export.week_end_date, export.seq_number, export.revision
                    ),
                    "revision_label": f"r{export.revision}",
                    "created_display": _format_local(export.created_at),
                    "download_url": f"/api/v1/exports/{export.id}/file",
                    "file_present": original_path.is_file(),
                }
            ],
        }
    ]
    for pipeline in _PIPELINE_ORDER:
        layer = _PIPELINE_LAYER[pipeline]
        artifacts = ArtifactRepository(session).list_by_export_and_layer(export.id, layer)
        if pipeline is PipelineType.normalize:
            input_note: str | None = None
            empty_message = "Для этой выгрузки ещё нет нормализованной версии."
        else:
            input_note = "строится только из нормализованного текста (слой 2)"
            empty_message = "Для этой выгрузки ещё нет результата этого слоя."
        sections.append(
            {
                "key": layer.value,
                "title": LAYER_TITLES[layer],
                "is_original": False,
                "is_derived": True,
                "input_note": input_note,
                "empty_message": empty_message,
                "rows": [
                    _lineage_artifact_view(session, artifact, export) for artifact in artifacts
                ],
            }
        )
    return sections


def _processing_run_view(
    session: Session, run: ProcessingRun, export: WeeklyExport
) -> dict[str, Any]:
    """Template fields of one processing run on the export detail page.

    Status surface and provenance only (pipeline/status/prompt version/safe
    error summary/timestamps) plus the artifact the run produced when it
    succeeded — never diary content or raw provider payloads.
    """
    output: dict[str, str] | None = None
    produced = ArtifactRepository(session).list_by_run(run.id)
    if produced:
        artifact = produced[-1]
        output = {
            "filename": _artifact_file_name(artifact),
            "version_label": f"v{artifact.version}" if artifact.version is not None else "",
            "download_url": _artifact_download_url(export.id, artifact.id),
        }
    return {
        "id": str(run.id),
        "pipeline_type": run.pipeline_type.value,
        "label": PIPELINE_LABELS[run.pipeline_type],
        "status": run.status.value,
        "status_label": RUN_STATUS_LABELS[run.status],
        "prompt_version": run.prompt_version,
        "error_summary": run.error_summary,
        "created_display": _format_local(run.created_at),
        "started_display": _format_local(run.started_at) if run.started_at else None,
        "finished_display": _format_local(run.finished_at) if run.finished_at else None,
        "output": output,
    }


def _export_runs_view(session: Session, export: WeeklyExport) -> list[dict[str, Any]]:
    """Processing runs of one export, newest first (export detail page)."""
    runs = ProcessingRunRepository(session).list_by_input_export(export.id)
    return [_processing_run_view(session, run, export) for run in reversed(runs)]


def _job_export_ref(session: Session, job: Job) -> dict[str, Any] | None:
    """The export a pipeline job belongs to (identifier only), for linking.

    Returns ``None`` for non-pipeline jobs, jobs whose payload carries no
    ``export_id``, unparsable ids and ids of exports that no longer exist.
    """
    pipeline = JOB_PIPELINE_TYPE.get(job.job_type)
    if pipeline is None:
        return None
    payload = job.payload_json if isinstance(job.payload_json, dict) else None
    raw = payload.get("export_id") if payload is not None else None
    if raw is None:
        return None
    try:
        export_id = uuid.UUID(str(raw))
    except ValueError:
        return None
    if WeeklyExportRepository(session).get(export_id) is None:
        return None
    return {"id": str(export_id), "detail_url": f"/exports/{export_id}"}


def _job_view(session: Session, job: Job) -> dict[str, Any]:
    """Template fields of one job row on the processing page (Task 16).

    Status surface only — the payload is never rendered (privacy rule). The
    optional export reference links pipeline jobs to their export detail page.
    """
    return {
        "id": str(job.id),
        "job_type": job.job_type.value,
        "job_label": JOB_TYPE_LABELS.get(job.job_type, job.job_type.value),
        "status": job.status.value,
        "status_label": JOB_STATUS_LABELS.get(job.status, job.status.value),
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "error_summary": job.error_summary,
        "retryable": job.status in (JobStatus.failed, JobStatus.cancelled),
        "export": _job_export_ref(session, job),
        "created_display": _format_local(job.created_at),
        "updated_display": _format_local(job.updated_at),
    }


def _processing_jobs_context(session: Session) -> dict[str, Any]:
    """Template context shared by the processing page and its polling fragment."""
    jobs = JobRepository(session).list_recent(limit=PAGE_LIMIT)
    return {"jobs": [_job_view(session, job) for job in jobs]}


@pages_router.get("/exports", response_class=HTMLResponse)
def exports_page(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Weekly exports management page (Task 16).

    Groups every original export by week (most recent week first) and shows,
    per revision, the derived-layer statuses and the management actions
    (export week / new revision / normalize / summarize / wellbeing /
    downloads / detail links). All values are server-derived; no diary
    content is rendered on this page.
    """
    settings = get_settings()
    active_jobs = _active_pipeline_jobs(session)
    records = list_exports(session)
    rows = [_export_management_view(session, record, active_jobs) for record in records]
    weeks: list[dict[str, Any]] = []
    for row in rows:
        sunday_iso = row["sunday"].isoformat()
        if weeks and weeks[-1]["sunday_iso"] == sunday_iso:
            weeks[-1]["exports"].append(row)
        else:
            weeks.append(
                {
                    "sunday_iso": sunday_iso,
                    "monday": row["monday"],
                    "sunday": row["sunday"],
                    "exports": [row],
                }
            )
    today_local = datetime.now(settings.tzinfo).date()
    return templates.TemplateResponse(
        request,
        "exports.html",
        {
            "weeks": weeks,
            "default_sunday": sunday_date_for(today_local).isoformat(),
            "wellbeing_enabled": settings.wellbeing_enabled,
            "external": _external_profile_info(),
            "nav_current": "exports",
            "nav": _nav_context(session),
        },
    )


@pages_router.get("/exports/{export_id}", response_class=HTMLResponse)
def export_detail_page(
    export_id: uuid.UUID,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Per-export lineage page (Task 16).

    Shows one original export revision with all its derived-layer versions,
    the processing runs and the sibling revisions of the same week. Actions
    (new revision / normalize / summarize / wellbeing) launch jobs through the
    JSON API; diary content and job payloads are never rendered here.
    """
    record = WeeklyExportRepository(session).get(export_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Export not found.")
    settings = get_settings()
    siblings = [
        {
            "id": str(sibling.id),
            "revision": sibling.revision,
            "detail_url": f"/exports/{sibling.id}",
            "current": sibling.id == record.id,
        }
        for sibling in WeeklyExportRepository(session).list_by_week(record.week_end_date)
    ]
    return templates.TemplateResponse(
        request,
        "export_detail.html",
        {
            "export": _export_management_view(session, record, _active_pipeline_jobs(session)),
            "sections": _lineage_sections(session, record),
            "runs": _export_runs_view(session, record),
            "siblings": siblings,
            "wellbeing_enabled": settings.wellbeing_enabled,
            "external": _external_profile_info(),
            "nav_current": "exports",
            "nav": _nav_context(session),
        },
    )


@pages_router.get("/processing", response_class=HTMLResponse)
def processing_page(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Processing page (Task 16): recent jobs with safe status summaries.

    The job table is refreshed locally by htmx polling of
    ``GET /processing/fragments/jobs``; failed/cancelled rows carry an
    explicit "retry" action (``POST /api/v1/jobs/{id}/retry``). Payloads and
    diary content are never shown.
    """
    return templates.TemplateResponse(
        request,
        "processing.html",
        {
            "nav_current": "processing",
            **_processing_jobs_context(session),
            "nav": _nav_context(session),
        },
    )


@pages_router.get("/processing/fragments/jobs", response_class=HTMLResponse)
def processing_jobs_fragment(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Job-table HTML fragment polled by the processing page (htmx every N s)."""
    return templates.TemplateResponse(
        request, "_jobs_panel.html", _processing_jobs_context(session)
    )


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
        normalized_path = _artifact_file_path(latest)
        if normalized_path.is_file():
            normalized_text = normalized_path.read_text(encoding="utf-8")
            normalized_filename = _artifact_file_name(latest)
    return templates.TemplateResponse(
        request,
        "export_compare.html",
        {
            "export": _export_view(record),
            "original_text": original_path.read_text(encoding="utf-8"),
            "normalized_text": normalized_text,
            "normalized_filename": normalized_filename,
            "nav_current": "exports",
            "nav": _nav_context(session),
        },
    )


@pages_router.get("/exports/{export_id}/summary", response_class=HTMLResponse)
def export_summary_page(
    export_id: uuid.UUID,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Read-only summary page for one export (Task 14).

    Shows the newest succeeded summary (layer 3) as escaped plain text inside
    ``<pre>`` — the stored Markdown is never rendered as unsafe HTML — plus a
    download link through the generic derived-artifact route. An export
    without a summary (yet) renders an empty state with the relevant next
    step; an unknown export id is a 404.
    """
    record = WeeklyExportRepository(session).get(export_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Export not found.")
    artifacts = ArtifactRepository(session).list_by_export_and_layer(
        export_id, ArtifactLayer.summary
    )
    latest = artifacts[-1] if artifacts else None
    normalized = ArtifactRepository(session).list_by_export_and_layer(
        export_id, ArtifactLayer.normalized
    )
    summary_text: str | None = None
    summary_filename: str | None = None
    summary_download_url: str | None = None
    if latest is not None:
        summary_path = _artifact_file_path(latest)
        if summary_path.is_file():
            summary_text = summary_path.read_text(encoding="utf-8")
            summary_filename = _artifact_file_name(latest)
            summary_download_url = f"/api/v1/exports/{export_id}/artifacts/{latest.id}/file"
    return templates.TemplateResponse(
        request,
        "export_summary.html",
        {
            "export": _export_view(record),
            "has_normalized": bool(normalized),
            "summary_text": summary_text,
            "summary_filename": summary_filename,
            "summary_download_url": summary_download_url,
            "nav_current": "exports",
            "nav": _nav_context(session),
        },
    )


@pages_router.get("/exports/{export_id}/wellbeing", response_class=HTMLResponse)
def export_wellbeing_page(
    export_id: uuid.UUID,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Read-only wellbeing page for one export (Task 15).

    Shows the newest succeeded wellbeing report (layer 4) as escaped plain
    text inside ``<pre>`` plus a download link, or an empty state with the
    explicit run action. Safety behavior is deterministic and does NOT depend
    on any LLM run having succeeded (ADR-006):

    * the page itself classifies the latest normalized (layer 2) text with
      ``diary.domain.pipelines.wellbeing.safety.classify``;
    * when the level is ``urgent`` a fixed escalation block (constants from
      ``safety.py``, never LLM text) and safe minimal metadata (level, export
      id, detection time — no diary text) are always rendered;
    * the run action is only rendered when the feature is enabled
      (``AppSettings.wellbeing_enabled``) and requires an explicit client-side
      confirmation in addition to the server-side ``confirm: true`` flag.
    """
    record = WeeklyExportRepository(session).get(export_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Export not found.")
    settings = get_settings()
    normalized_rows = ArtifactRepository(session).list_by_export_and_layer(
        export_id, ArtifactLayer.normalized
    )
    normalized = normalized_rows[-1] if normalized_rows else None
    report_rows = ArtifactRepository(session).list_by_export_and_layer(
        export_id, ArtifactLayer.wellbeing
    )
    latest_report = report_rows[-1] if report_rows else None

    has_normalized = False
    normalized_text: str | None = None
    if normalized is not None:
        normalized_path = _artifact_file_path(normalized)
        if normalized_path.is_file():
            has_normalized = True
            normalized_text = normalized_path.read_text(encoding="utf-8")

    # Deterministic page-level safety classification (never from an LLM).
    level = classify(normalized_text) if normalized_text is not None else None
    report_text: str | None = None
    report_filename: str | None = None
    report_download_url: str | None = None
    report_created_display: str | None = None
    if latest_report is not None:
        report_path = _artifact_file_path(latest_report)
        if report_path.is_file():
            report_text = report_path.read_text(encoding="utf-8")
            report_filename = _artifact_file_name(latest_report)
            report_download_url = f"/api/v1/exports/{export_id}/artifacts/{latest_report.id}/file"
            report_created_display = latest_report.created_at.astimezone(settings.tzinfo).strftime(
                TIMESTAMP_FORMAT
            )
    detected_at = (
        normalized.created_at.astimezone(settings.tzinfo).strftime(TIMESTAMP_FORMAT)
        if normalized is not None
        else None
    )
    return templates.TemplateResponse(
        request,
        "export_wellbeing.html",
        {
            "export": _export_view(record),
            "wellbeing_enabled": settings.wellbeing_enabled,
            "has_normalized": has_normalized,
            "safety_level_value": level.value if level is not None else None,
            "safety_level_label": level.display_label if level is not None else None,
            "urgent": level is SafetyLevel.urgent,
            "escalation_lines": list(ESCALATION_LINES) if level is SafetyLevel.urgent else None,
            "escalation_title": ESCALATION_TITLE,
            "export_id_display": str(export_id),
            "detected_at_display": detected_at,
            "report_text": report_text,
            "report_filename": report_filename,
            "report_download_url": report_download_url,
            "report_created_display": report_created_display,
            "nav_current": "exports",
            "nav": _nav_context(session),
        },
    )


@pages_router.get("/messages/fragments/older", response_class=HTMLResponse)
def older_messages_fragment(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    cursor: Annotated[str, Query(max_length=256)],
    limit: Annotated[int, Query(ge=1, le=200)] = PAGE_LIMIT,
) -> HTMLResponse:
    """Return the next older entries window as an HTML fragment.

    The cursor points at the oldest message already shown; the response holds
    the messages strictly older than it in chronological order (they are
    prepended to ``#message-list``), followed by an out-of-band replacement of
    the ``#older-control`` element with the next cursor.

    The window is bounded by the local calendar day the cursor belongs to
    (task T16b): the feed shows one day, so paging stays inside that day and
    stops at its beginning instead of walking into the day before. The cursor
    itself carries the instant, so no extra request parameter is needed and
    the endpoint keeps working for the JSON-API paging contract
    (``GET /api/v1/messages``), which remains unbounded.
    """
    try:
        position = _decode_cursor(cursor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    settings = get_settings()
    day = local_date_of(position[0], settings.tzinfo)
    day_messages = _day_messages(session, day, settings.tzinfo)
    window, next_cursor = _day_history_window(day_messages, limit=limit, older_than=position)
    older_cursor = _encode_cursor(next_cursor) if next_cursor is not None else None
    return templates.TemplateResponse(
        request,
        "_older_messages.html",
        {
            "messages": _message_views(window, session),
            "has_more": next_cursor is not None,
            "older_cursor": quote(older_cursor, safe="") if older_cursor else None,
        },
    )
