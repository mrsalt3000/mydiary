"""Provenance of one artifact — «артефакт → источник → прогон обработки» (task T37).

The page ``/artifacts/{id}/provenance`` renders what the T08 store resolved
(``diary.db.results.trace_layer_result``, ADR-007 §7): which source a stored
result came from, and which processing run produced it. It is reached from the
day page's «Слои фазы 2» block (a second-family result anchored to that day or
to its ISO week) and from the export detail page (a first-family artifact).

Covers, against an isolated migrated database with synthetic data:

* the chain of a second-family result for all four anchor kinds — message,
  local day, ISO week and an upstream artifact — including the link into the
  day page and the link into the source artifact's own chain;
* the honest states: a result whose run was not recorded, a source row that
  cannot be resolved, a deleted source message, a row with no anchor at all,
  and a stale link to an artifact that does not exist (404 in the shell);
* the first-family answer (its lineage runs through the weekly export — the
  page links there instead of inventing a chain) and the control on the export
  detail page that reaches it;
* the day page's phase-2 block: the control exists only for a result that
  exists, the anchors are looked up exactly as the layers write them, and the
  control leads to a page that answers 200;
* privacy: neither the text of the source message nor the stored LLM payload
  (``result_json``) is rendered, and no content of the chain reaches the logs.

What is *not* claimed here: that the page looks right (that is a browser
property, checked by driving the page), and nothing about the resolvers
themselves — they have their own tests (T08) and are used as-is.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import diary.api.routes.pages as pages_module
from diary.api.routes.messages import get_session
from diary.api.routes.pages import (
    ARTIFACT_MISSING_NOTICE,
    FIRST_FAMILY_NOTICE,
    PROVENANCE_LINK_LABEL,
    UNRESOLVED_NOTICE,
)
from diary.core.settings import get_settings, reload_settings
from diary.db.models import (
    Artifact,
    ArtifactLayer,
    MessageKind,
    PipelineType,
    RunStatus,
    WeeklyExport,
)
from diary.db.repositories import (
    ArtifactRepository,
    MessageRepository,
    ProcessingRunRepository,
)
from diary.db.results import LayerResultError, LayerResultTrace, save_layer_result
from diary.domain.exports.service import export_week
from diary.domain.exports.weeks import sunday_date_for
from diary.domain.pipelines.naming import artifact_data_root_relative_path, derived_filename
from diary.domain.pipelines.sum_weekly.week_input import week_id_of
from diary.main import app

#: A fixed local day (the default application timezone is Europe/Moscow, so
#: 2026-09-10T09:00Z is 12:00 local on the same day) and its ISO week.
DAY = date(2026, 9, 10)
DAY_ISO = DAY.isoformat()
WEEK_ID = week_id_of(DAY)
SUNDAY = sunday_date_for(DAY)

#: Markers that must never appear on the provenance page: the body of the
#: source message and a value of the stored LLM payload.
MESSAGE_MARKER = "синтетический текст записи-маркер-9f3a"
PAYLOAD_MARKER = "маркер-в-полезной-нагрузке-4b71"


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


# ---------------------------------------------------------------------------
# synthetic data (never real diary content)
# ---------------------------------------------------------------------------


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(DAY.year, DAY.month, DAY.day, hour, minute, tzinfo=timezone.utc)


def _text_message(session: Session, *, body: str = MESSAGE_MARKER, at: datetime | None = None):
    message = MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=at or _at(9)
    )
    session.commit()
    return message


def _run(
    session: Session,
    *,
    pipeline_type: PipelineType = PipelineType.daily_summary,
    status: RunStatus = RunStatus.succeeded,
    profile_name: str | None = "diary_sum",
    model_name: str | None = "diary-sum",
    prompt_version: str | None = "v1",
    error_summary: str | None = None,
    input_artifact_id=None,
    input_export_id=None,
):
    """One processing run with the provenance fields a real run carries."""
    run = ProcessingRunRepository(session).create(
        pipeline_type=pipeline_type,
        status=status,
        profile_name=profile_name,
        model_name=model_name,
        prompt_version=prompt_version,
        error_summary=error_summary,
        input_artifact_id=input_artifact_id,
        input_export_id=input_export_id,
        created_at=_at(20),
    )
    if status is RunStatus.succeeded:
        run.started_at = _at(20)
        run.finished_at = _at(20, 1)
    session.commit()
    return run


def _day_summary(session: Session, *, run=None, day: date = DAY, result: bool = True) -> Artifact:
    """A ``daily_summary`` result anchored to ``day`` (with a payload by default)."""
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.daily_summary,
        result={"date": day.isoformat(), "facts": [PAYLOAD_MARKER]} if result else None,
        path=None if result else f"exports/03_summaries/{day.isoformat()}_summary.txt",
        sha256=None if result else hashlib.sha256(b"x").hexdigest(),
        processing_run_id=run.id if run is not None else None,
        local_date=day,
        local_tz=get_settings().timezone,
        created_at=_at(21),
    )
    session.commit()
    return artifact


def _message_correction(session: Session, *, raw_id, run=None) -> Artifact:
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.stt_corrected,
        result={"raw_id": str(raw_id), "text": MESSAGE_MARKER, "uncertain": []},
        processing_run_id=run.id if run is not None else None,
        raw_id=raw_id,
        created_at=_at(19),
    )
    session.commit()
    return artifact


def _week_summary(session: Session, *, week_id: str = WEEK_ID, run=None) -> Artifact:
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.weekly_summary,
        result={"week_id": week_id, "daily_ids": [DAY_ISO], "rollup": PAYLOAD_MARKER},
        processing_run_id=run.id if run is not None else None,
        week_id=week_id,
        created_at=_at(22),
    )
    session.commit()
    return artifact


def _coach_note(session: Session, *, summary_id, run=None) -> Artifact:
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.coach_note,
        result={
            "summary_id": WEEK_ID,
            "observations": ["a", "b", "c"],
            "hypotheses": ["a", "b"],
            "experiment_7d": "7 дней вести заметки",
            "disclaimer": "не диагноз",
        },
        processing_run_id=run.id if run is not None else None,
        summary_id=summary_id,
        created_at=_at(23),
    )
    session.commit()
    return artifact


def _export(session: Session) -> WeeklyExport:
    """A real weekly export (row + file) with one synthetic message in its week."""
    MessageRepository(session).create(
        kind=MessageKind.text,
        body_text="синтетическая запись недели",
        created_at=datetime(SUNDAY.year, SUNDAY.month, SUNDAY.day - 3, 9, 0, tzinfo=timezone.utc),
    )
    session.commit()
    return export_week(session, sunday_date=SUNDAY)


def _first_family_artifact(
    session: Session, record: WeeklyExport, *, layer: ArtifactLayer = ArtifactLayer.normalized
) -> Artifact:
    """A layer-2 artifact of ``record`` with a real file on disk."""
    filename = derived_filename(
        sunday=record.week_end_date,
        layer=layer,
        seq_number=record.seq_number,
        revision=record.revision,
        version=1,
    )
    relative = artifact_data_root_relative_path(layer=layer, filename=filename)
    target = get_settings().data_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("синтетический нормализованный текст", encoding="utf-8")
    artifact = ArtifactRepository(session).create(
        layer=layer,
        export_id=record.id,
        path=relative,
        sha256=hashlib.sha256(b"synthetic").hexdigest(),
        version=1,
    )
    session.commit()
    return artifact


def _provenance_url(artifact: Artifact) -> str:
    return f"/artifacts/{artifact.id}/provenance"


def _local(value: datetime) -> str:
    """A moment as the page words it: wall clock in the application timezone."""
    return value.astimezone(get_settings().tzinfo).strftime("%d.%m.%Y %H:%M")


# ---------------------------------------------------------------------------
# the chain of a second-family result (DoD 1)
# ---------------------------------------------------------------------------


def test_day_result_shows_artifact_anchor_and_run(api_client: TestClient, session: Session) -> None:
    """A ``daily_summary`` result shows its identifiers, its day anchor, the
    resolved source (the day) and the whole run record — profile, model, prompt
    version and the three timestamps."""
    run = _run(session)
    artifact = _day_summary(session, run=run)

    response = api_client.get(_provenance_url(artifact))

    assert response.status_code == 200
    html = response.text
    # Artifact identity.
    assert str(artifact.id) in html
    assert "Саммари дня (фаза 2)" in html
    assert "v1" in html
    assert _local(artifact.created_at) in html
    # Anchor: the field and its value, the timezone it was computed in.
    assert "local_date = 2026-09-10" in html
    assert "Europe/Moscow" in html
    assert "Локальный календарный день" in html
    assert "(local_day)" in html
    # The resolved source is the day itself, reachable through the day page.
    assert f'href="/?date={DAY_ISO}"' in html
    # The run: identifier, layer, status, profile/model/prompt version, moments.
    assert str(run.id) in html
    assert "Саммари дня" in html
    assert "Успешно" in html
    assert "(succeeded)" in html
    assert "diary_sum" in html
    assert "diary-sum" in html
    assert "v1" in html
    assert _local(run.created_at) in html
    assert _local(run.started_at) in html
    assert _local(run.finished_at) in html


def test_message_anchor_shows_metadata_without_the_text(
    api_client: TestClient, session: Session
) -> None:
    """A ``stt_corrected`` result of a message shows the message's identity,
    kind and moment and links to its day — and never its text."""
    message = _text_message(session)
    run = _run(session, pipeline_type=PipelineType.stt_corrected)
    artifact = _message_correction(session, raw_id=message.id, run=run)

    response = api_client.get(_provenance_url(artifact))

    assert response.status_code == 200
    html = response.text
    assert f"raw_id = {message.id}" in html
    assert "Запись дневника (сообщение)" in html
    assert "(message)" in html
    assert "Текстовая запись" in html
    assert str(message.id) in html
    assert f'href="/?date={DAY_ISO}"' in html
    # Privacy: neither the message body nor the stored payload is rendered.
    assert MESSAGE_MARKER not in html
    assert PAYLOAD_MARKER not in html


def test_week_anchor_shows_the_iso_week(api_client: TestClient, session: Session) -> None:
    """A ``weekly_summary`` result shows the ISO week its anchor names, and the
    seven days that week covers — derived, not guessed."""
    run = _run(session, pipeline_type=PipelineType.weekly_summary)
    artifact = _week_summary(session, run=run)

    html = api_client.get(_provenance_url(artifact)).text

    assert f"week_id = {WEEK_ID}" in html
    assert "ISO-неделя" in html
    assert "(iso_week)" in html
    assert "07.09.2026 — 13.09.2026" in html
    assert PAYLOAD_MARKER not in html


def test_summary_anchor_links_to_the_source_chain(api_client: TestClient, session: Session) -> None:
    """A ``coach_note`` result resolves to the upstream ``weekly_summary``
    artifact, and the page links into that artifact's own chain — the link
    answers, so it is not a dead control."""
    summary = _week_summary(session)
    run = _run(session, pipeline_type=PipelineType.coach_note)
    note = _coach_note(session, summary_id=summary.id, run=run)

    response = api_client.get(_provenance_url(note))

    assert response.status_code == 200
    html = response.text
    assert f"summary_id = {summary.id}" in html
    assert "Артефакт предыдущего слоя" in html
    assert "(artifact)" in html
    assert "Саммари недели (фаза 2)" in html
    assert str(summary.id) in html
    assert f'href="{_provenance_url(summary)}"' in html
    assert api_client.get(_provenance_url(summary)).status_code == 200


# ---------------------------------------------------------------------------
# honest states (DoD 2-4)
# ---------------------------------------------------------------------------


def test_result_without_a_recorded_run_says_so(api_client: TestClient, session: Session) -> None:
    """A result whose ``processing_run_id`` is empty keeps its chain to the
    source and states that no run was recorded — no invented run, no control."""
    artifact = _day_summary(session, run=None)

    html = api_client.get(_provenance_url(artifact)).text

    assert "Прогон обработки для этого артефакта не записан" in html
    assert "Прогон обработки для этого артефакта не записан" in html
    # The source is still there: the anchor does not depend on the run.
    assert "local_date = 2026-09-10" in html


def test_run_with_empty_provenance_columns_says_not_recorded(
    api_client: TestClient, session: Session
) -> None:
    """A stored run with empty profile/model/prompt columns is shown as such:
    every one of the three says «не записано» instead of printing an empty
    value or inventing the configured profile of the day."""
    run = _run(
        session,
        profile_name=None,
        model_name=None,
        prompt_version=None,
        status=RunStatus.failed,
        error_summary="безопасная сводка ошибки",
    )
    artifact = _day_summary(session, run=run)

    html = api_client.get(_provenance_url(artifact)).text

    assert html.count("не записано") == 3
    assert "безопасная сводка ошибки" in html
    assert str(run.id) in html
    # A failed run has no finish moment yet: the page says so with a dash.
    assert "завершён: —" in html


def test_deleted_source_message_is_reported_as_deleted(
    api_client: TestClient, session: Session
) -> None:
    """The source of a correction may be a deleted message (ADR-011 keeps the
    row and erases its text): the page says so and shows no text at all."""
    message = _text_message(session)
    artifact = _message_correction(session, raw_id=message.id)
    message.deleted_at = _at(10)
    message.body_text = ""
    session.commit()

    html = api_client.get(_provenance_url(artifact)).text

    assert str(message.id) in html
    assert "запись удалена" in html
    assert MESSAGE_MARKER not in html


def test_missing_source_row_says_so_instead_of_an_empty_chain(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source row that cannot be resolved is stated in words.

    The database foreign keys make this state unreachable through normal writes,
    so the resolver is *replaced* for this one test: what is checked here is the
    rendering branch of the page (an honest sentence instead of an empty block),
    not the resolver, which has its own tests.
    """
    message = _text_message(session)
    artifact = _message_correction(session, raw_id=message.id)
    real_trace = pages_module.trace_layer_result

    def _trace_without_source(session_, artifact_id):
        trace = real_trace(session_, artifact_id)
        return LayerResultTrace(
            artifact=trace.artifact,
            layer=trace.layer,
            anchor_field=trace.anchor_field,
            anchor_value=trace.anchor_value,
            source_kind=trace.source_kind,
            source_message=None,
            source_artifact=trace.source_artifact,
            run=trace.run,
        )

    monkeypatch.setattr(pages_module, "trace_layer_result", _trace_without_source)

    html = api_client.get(_provenance_url(artifact)).text

    assert "в хранилище не найдена" in html
    assert str(message.id) in html  # the anchor value itself is still shown


def test_row_without_an_anchor_gets_an_honest_answer(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``LayerResultError`` from the resolver (a row that carries no anchor)
    is answered in words — never a 500 and never an empty chain.

    The table's own CHECK constraint makes such a row unwritable, so the
    resolver is replaced for this test: the branch under test is the page's.
    """
    artifact = _day_summary(session)

    def _raise(session_, artifact_id):
        raise LayerResultError("The artifact carries no second-family anchor.")

    monkeypatch.setattr(pages_module, "trace_layer_result", _raise)

    response = api_client.get(_provenance_url(artifact))

    assert response.status_code == 200
    assert UNRESOLVED_NOTICE in response.text


def test_first_family_artifact_answers_honestly_and_links_to_its_export(
    api_client: TestClient, session: Session
) -> None:
    """A layer 1–4 artifact has no second-family chain: the page says where its
    lineage runs and links to that export (both the export page and the notice
    are real)."""
    record = _export(session)
    artifact = _first_family_artifact(session, record)

    response = api_client.get(_provenance_url(artifact))

    assert response.status_code == 200
    html = response.text
    assert FIRST_FAMILY_NOTICE in html
    assert str(artifact.id) in html
    assert "Нормализованный (слой 2)" in html
    assert f'href="/exports/{record.id}"' in html
    # No second-family chain is drawn for it — and no run block either.
    assert "Якорь:" not in html
    assert "Прогон обработки" not in html
    assert api_client.get(f"/exports/{record.id}").status_code == 200


def test_unknown_artifact_id_is_a_404_in_the_shell(api_client: TestClient) -> None:
    """A stale link is answered with 404 *and* the honest sentence, inside the
    normal shell — not with a raw error page and not with a 200 pretence."""
    import uuid as uuid_module

    response = api_client.get(f"/artifacts/{uuid_module.uuid4()}/provenance")

    assert response.status_code == 404
    html = response.text
    assert ARTIFACT_MISSING_NOTICE in html
    assert '<aside class="sidebar"' in html


# ---------------------------------------------------------------------------
# where the control lives (DoD 1-2)
# ---------------------------------------------------------------------------


def test_export_detail_page_offers_the_control_for_every_artifact(
    api_client: TestClient, session: Session
) -> None:
    """Every derived artifact row of an export carries the control, and each
    link answers — the first family included."""
    record = _export(session)
    normalized = _first_family_artifact(session, record)
    summary = _first_family_artifact(session, record, layer=ArtifactLayer.summary)

    html = api_client.get(f"/exports/{record.id}").text

    for artifact in (normalized, summary):
        assert f'href="{_provenance_url(artifact)}"' in html
        assert PROVENANCE_LINK_LABEL in html
        assert api_client.get(_provenance_url(artifact)).status_code == 200


def test_day_panel_offers_the_control_only_for_stored_results(
    api_client: TestClient, session: Session
) -> None:
    """The phase-2 block names both lineages of the day; the control appears for
    the layer that has a stored result and only for it."""
    run = _run(session)
    artifact = _day_summary(session, run=run)

    html = api_client.get(f"/?date={DAY_ISO}").text

    assert "Слои фазы 2" in html
    assert PROVENANCE_LINK_LABEL in html
    assert f'href="{_provenance_url(artifact)}"' in html
    assert api_client.get(_provenance_url(artifact)).status_code == 200
    # The week lineage has no stored result for this week: «Не посчитано» and no
    # second control.
    assert html.count('href="/artifacts/') == 1
    assert html.count(PROVENANCE_LINK_LABEL) == 1


def test_day_panel_looks_the_layers_up_by_their_own_anchors(
    api_client: TestClient, session: Session
) -> None:
    """The block resolves ``daily_summary`` by this day and ``weekly_summary`` by
    this day's ISO week: results anchored elsewhere are not shown as this day's."""
    other_day = DAY - timedelta(days=7)  # the same weekday one week earlier
    other_week = week_id_of(DAY + timedelta(days=7))  # the following ISO week
    assert other_week != WEEK_ID
    foreign_day = _day_summary(session, day=other_day)
    foreign_week = _week_summary(session, week_id=other_week)

    html = api_client.get(f"/?date={DAY_ISO}").text

    assert _provenance_url(foreign_day) not in html
    assert _provenance_url(foreign_week) not in html
    assert PROVENANCE_LINK_LABEL not in html
    # …and the other day shows its own day result.
    other_html = api_client.get(f"/?date={other_day.isoformat()}").text
    assert _provenance_url(foreign_day) in other_html
    assert _provenance_url(foreign_week) not in other_html


# ---------------------------------------------------------------------------
# privacy (DoD 3)
# ---------------------------------------------------------------------------


def test_nothing_of_the_chain_reaches_the_logs(
    api_client: TestClient, session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    """Rendering a chain logs no content: not the source text, not the stored
    payload. (The request *path* carries the artifact id, exactly as the
    existing ``/exports/{id}`` pages do; no content is logged anywhere.)"""
    message = _text_message(session)
    run = _run(session, pipeline_type=PipelineType.stt_corrected)
    artifact = _message_correction(session, raw_id=message.id, run=run)

    with caplog.at_level(logging.DEBUG):
        assert api_client.get(_provenance_url(artifact)).status_code == 200

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert MESSAGE_MARKER not in logged
    assert PAYLOAD_MARKER not in logged
    # No field name of the stored payload either (the request *path* legitimately
    # carries the artifact id, as every other id-bearing page does).
    assert "uncertain" not in logged


def test_provenance_page_renders_no_diary_text_of_the_result(
    api_client: TestClient, session: Session
) -> None:
    """The stored LLM payload of a day summary is never rendered, whatever it
    holds; the page only states that a payload exists."""
    artifact = _day_summary(session)

    html = api_client.get(_provenance_url(artifact)).text

    assert PAYLOAD_MARKER not in html
    assert "Структурированный результат хранится в базе" in html
