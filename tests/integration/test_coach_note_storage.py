"""Integration tests for the T07 coaching-note layer writing through T08.

Runs against an isolated, migration-created test database (never the production
one) with a fake LLM provider, and covers the parts of the T07 definition of
done that need real persistence:

* the input is the **newest stored weekly summary** of the ISO week — the raw
  tables (``messages``, ``audio_files``, ``transcriptions``, ``weekly_exports``)
  are not read and are unchanged by the run, and no weekly export is involved;
* a valid completion is appended to ``artifacts`` as layer 5 with **3
  observations, 2 hypotheses, one 7-day experiment and the mandatory
  disclaimer**, and with no rendered file;
* the twin ``summary_id`` seam: the stored row's **storage anchor**
  (``artifacts.summary_id``) is the **artifact id** of the input weekly
  summary, while the stored payload's **contract field**
  ``result_json["summary_id"]`` is that summary's **week_id** (ADR-008 §5);
* a repeated run appends ``v+1`` and leaves the previous version untouched
  (ADR-005);
* a failed run stores nothing and leaves the raw rows and the input summary
  alone;
* an enqueued ``coach_note`` job does **not** run the layer: with no registered
  handler the job fails permanently and no note appears (T07 decision 3).

All content is synthetic (AGENTS.md §13); no endpoint, model or real
configuration file is used.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from diary.db.models import Artifact, ArtifactLayer, JobStatus, JobType, MessageKind, PipelineType
from diary.db.repositories import JobRepository, MessageRepository
from diary.db.results import list_layer_results, save_layer_result, trace_layer_result
from diary.domain.contracts import COACH_DISCLAIMER, WeeklySummary
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines.coach.coach_input import CoachInput, CoachInputError
from diary.domain.pipelines.coach.pipeline import SUMMARY_HEADER
from diary.domain.pipelines.coach.service import coach_note
from diary.domain.pipelines.wellbeing import safety as wellbeing_safety
from diary.workers.runner import process_due_jobs

WEEK_ID = "2026-W38"
DAILY_IDS = ["2026-09-14", "2026-09-15", "2026-09-16"]
ROLLUP = "Неделя прошла вокруг отчёта; в среду сон сбился, причина осталась открытой."
OBSERVATIONS = [
    "Работа над отчётом упоминается в начале и середине недели.",
    "В сводке есть один день спорта.",
    "Открытым остаётся вопрос о сне в среду.",
]
HYPOTHESES = [
    "Возможно, отчёт был основной нагрузкой первой половины недели.",
    "Похоже, что темп недели менялся к её середине.",
]
EXPERIMENT = "В течение следующих 7 дней можно попробовать отмечать, что занимало основное время."

#: Synthetic raw text that must never reach the coaching-note layer's input.
RAW_TEXT = "Сырое сообщение дневника, которого не должно быть в заметке коуча."


class _FakeProvider:
    """Deterministic provider: a fixed response, recorded requests."""

    def __init__(self, payload: object | str) -> None:
        text_value = (
            payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        )
        self.response = LLMResponse(text=text_value)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response

    @property
    def request_text(self) -> str:
        """The content of the single request this provider received."""
        assert len(self.requests) == 1
        return self.requests[0].messages[0].content


def _summary(week_id: str = WEEK_ID, *, rollup: str = ROLLUP) -> WeeklySummary:
    return WeeklySummary(week_id=week_id, daily_ids=DAILY_IDS, rollup=rollup)


def _store_week_summary(
    session: Session,
    *,
    week_id: str = WEEK_ID,
    rollup: str = ROLLUP,
) -> Artifact:
    """Store one ``weekly_summary`` result (the coaching note's only input)."""
    return save_layer_result(
        session,
        layer=ArtifactLayer.weekly_summary,
        week_id=week_id,
        result=_summary(week_id, rollup=rollup),
    )


def _note_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "observations": list(OBSERVATIONS),
        "hypotheses": list(HYPOTHESES),
        "experiment_7d": EXPERIMENT,
    }
    payload.update(overrides)
    return payload


def _raw_state(session: Session) -> list[list[tuple]]:
    """Every row of every raw source table, as raw values (value-by-value)."""
    return [
        session.execute(text(f"SELECT * FROM {table} ORDER BY id")).all()  # noqa: S608
        for table in ("messages", "audio_files", "transcriptions", "weekly_exports")
    ]


def _week_summary_state(session: Session) -> list[tuple]:
    """The stored weekly-summary results (the layer's input), newest last."""
    rows = list_layer_results(session, layer=ArtifactLayer.weekly_summary)
    return [(row.id, row.week_id, row.version, row.result_json) for row in rows]


def _notes(session: Session) -> list[Artifact]:
    return list_layer_results(session, layer=ArtifactLayer.coach_note)


# ---------------------------------------------------------------------------
# input: the week's newest weekly summary, and nothing else
# ---------------------------------------------------------------------------


def test_the_input_is_the_stored_weekly_summary_of_the_week(session: Session) -> None:
    _store_week_summary(session)
    # Another week is not this layer's input.
    _store_week_summary(session, week_id="2026-W39", rollup="Другая неделя.")
    session.commit()
    provider = _FakeProvider(_note_payload())

    outcome = coach_note(session, week_id=WEEK_ID, provider=provider)
    session.commit()

    assert outcome.note.summary_id == WEEK_ID
    assert ROLLUP in provider.request_text
    assert SUMMARY_HEADER in provider.request_text
    assert "2026-W39" not in provider.request_text


def test_raw_messages_are_never_read_or_placed_in_the_request(session: Session) -> None:
    """The layer consumes the week rollup, not raw entries or the day layer."""
    _store_week_summary(session)
    MessageRepository(session).create(
        kind=MessageKind.text,
        body_text=RAW_TEXT,
        created_at=datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc),
    )
    session.commit()
    provider = _FakeProvider(_note_payload())

    coach_note(session, week_id=WEEK_ID, provider=provider)
    session.commit()

    assert RAW_TEXT not in provider.request_text
    assert "raw_entry" not in provider.request_text
    assert "daily_summary" not in provider.request_text


def test_only_the_newest_version_of_the_week_is_used(session: Session) -> None:
    _store_week_summary(session, rollup="Старая версия недели.")
    newest = _store_week_summary(session, rollup=ROLLUP)
    session.commit()
    provider = _FakeProvider(_note_payload())

    coach_note(session, week_id=WEEK_ID, provider=provider)
    session.commit()

    assert ROLLUP in provider.request_text
    assert "Старая версия недели." not in provider.request_text
    stored = _notes(session)[0]
    assert stored.summary_id == newest.id


def test_a_week_without_a_weekly_summary_fails_loudly(session: Session) -> None:
    provider = _FakeProvider(_note_payload())

    with pytest.raises(CoachInputError):
        coach_note(session, week_id=WEEK_ID, provider=provider)
    session.rollback()

    assert provider.requests == []
    assert _notes(session) == []


def test_a_weekly_summary_of_another_week_in_the_seam_fails_loudly(session: Session) -> None:
    artifact = _store_week_summary(session, week_id="2026-W39", rollup="Другая неделя.")
    session.commit()
    foreign = CoachInput(summary=_summary("2026-W39", rollup="Другая неделя."), artifact=artifact)
    provider = _FakeProvider(_note_payload())

    with pytest.raises(CoachInputError):
        coach_note(session, week_id=WEEK_ID, provider=provider, coach_input=foreign)

    assert provider.requests == []
    assert _notes(session) == []


def test_a_weekly_summary_stored_only_as_a_file_fails_loudly(session: Session) -> None:
    """A result stored without its inline payload cannot be an input."""
    save_layer_result(
        session,
        layer=ArtifactLayer.weekly_summary,
        week_id=WEEK_ID,
        path="exports/05_weekly_summary/2026-W38_weekly_summary_v1.md",
        sha256="a" * 64,
    )
    session.commit()
    provider = _FakeProvider(_note_payload())

    with pytest.raises(CoachInputError):
        coach_note(session, week_id=WEEK_ID, provider=provider)

    assert provider.requests == []


def test_a_weekly_summary_with_a_disagreeing_week_id_fails_loudly(session: Session) -> None:
    """A payload whose week id disagrees with its anchor is a loud failure."""
    save_layer_result(
        session,
        layer=ArtifactLayer.weekly_summary,
        week_id=WEEK_ID,
        result={"week_id": "2026-W39", "daily_ids": DAILY_IDS, "rollup": ROLLUP},
    )
    session.commit()
    provider = _FakeProvider(_note_payload())

    with pytest.raises(CoachInputError):
        coach_note(session, week_id=WEEK_ID, provider=provider)

    assert provider.requests == []


# ---------------------------------------------------------------------------
# storing a validated note
# ---------------------------------------------------------------------------


def test_valid_note_is_appended_as_layer_5_with_3_2_1_and_the_disclaimer(
    session: Session,
) -> None:
    upstream = _store_week_summary(session)
    session.commit()
    raw_before = _raw_state(session)
    summaries_before = _week_summary_state(session)

    outcome = coach_note(session, week_id=WEEK_ID, provider=_FakeProvider(_note_payload()))
    session.commit()

    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None
    assert stored.layer is ArtifactLayer.coach_note
    assert stored.export_id is None  # second anchor family: no weekly export
    assert stored.version == 1
    assert stored.path is None and stored.sha256 is None  # stored inline (ADR-008 §1)
    # The storage anchor is the upstream artifact; no other anchor is set.
    assert stored.summary_id == upstream.id
    assert stored.raw_id is None and stored.local_date is None and stored.week_id is None
    assert stored.local_tz is None

    payload = stored.result_json
    assert payload is not None
    assert len(payload["observations"]) == 3
    assert len(payload["hypotheses"]) == 2
    assert "7" in payload["experiment_7d"]
    assert payload["disclaimer"] == COACH_DISCLAIMER
    assert payload["disclaimer"] != wellbeing_safety.DISCLAIMER  # layer 5, not layer 4
    # Exactly the contract's fields: no provenance or advice leaked into it.
    assert set(payload) == {
        "summary_id",
        "observations",
        "hypotheses",
        "experiment_7d",
        "disclaimer",
    }

    # The raw source tables and the input weekly summary are untouched.
    assert _raw_state(session) == raw_before
    assert _week_summary_state(session) == summaries_before
    # The layer wrote exactly one new artifact row: the note itself.
    assert [row.id for row in _notes(session)] == [stored.id]


def test_the_twin_summary_id_anchor_and_contract_field_agree(session: Session) -> None:
    """The two values both named ``summary_id``: artifact id vs week id (ADR-008 §5)."""
    upstream = _store_week_summary(session)
    session.commit()

    outcome = coach_note(session, week_id=WEEK_ID, provider=_FakeProvider(_note_payload()))
    session.commit()

    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None
    # Storage anchor (UUID column): the artifact id of the input weekly summary.
    assert stored.summary_id == upstream.id
    assert isinstance(stored.summary_id, type(upstream.id))  # a UUID
    # Contract field (string in the payload): the week id of that same summary.
    assert stored.result_json is not None
    contract_value = stored.result_json["summary_id"]
    assert contract_value == upstream.week_id == WEEK_ID
    assert isinstance(contract_value, str)
    # They are different values in different namespaces — by design, not by accident.
    assert contract_value != str(stored.summary_id)

    # Traceability resolves the stored note back to the exact upstream artifact,
    # and that artifact's week id is the value the payload reports.
    trace = trace_layer_result(session, stored.id)
    assert trace.anchor_field == "summary_id"
    assert trace.anchor_value == upstream.id
    assert trace.source_kind == "artifact"
    assert trace.source_artifact is not None
    assert trace.source_artifact.id == upstream.id
    assert trace.source_artifact.layer is ArtifactLayer.weekly_summary
    assert trace.source_artifact.week_id == contract_value


def test_a_rerun_appends_a_new_version_and_keeps_the_previous_one(session: Session) -> None:
    _store_week_summary(session)
    # Versions are told apart by words, not digits: a figure the weekly summary
    # does not carry is refused by the runtime gate (T17b).
    first_provider = _FakeProvider(
        _note_payload(observations=["Первое наблюдение версии первой.", *OBSERVATIONS[1:]])
    )
    second_provider = _FakeProvider(
        _note_payload(observations=["Первое наблюдение версии второй.", *OBSERVATIONS[1:]])
    )

    first = coach_note(session, week_id=WEEK_ID, provider=first_provider)
    session.commit()
    second = coach_note(session, week_id=WEEK_ID, provider=second_provider)
    session.commit()

    assert first.artifact.version == 1
    assert second.artifact.version == 2
    assert first.artifact.summary_id == second.artifact.summary_id  # one anchor
    lineage = _notes(session)
    assert [row.version for row in lineage] == [1, 2]
    # Append-only: the first version still carries its own note (ADR-005).
    assert lineage[0].result_json is not None
    assert lineage[0].result_json["observations"][0] == "Первое наблюдение версии первой."
    assert lineage[1].result_json is not None
    assert lineage[1].result_json["observations"][0] == "Первое наблюдение версии второй."


def test_a_new_version_of_the_week_starts_a_new_note_lineage(session: Session) -> None:
    """The note is about one specific weekly-summary artifact, not about the week."""
    first_week_version = _store_week_summary(session, rollup="Первая версия недели.")
    session.commit()
    coach_note(session, week_id=WEEK_ID, provider=_FakeProvider(_note_payload()))
    session.commit()

    second_week_version = _store_week_summary(session, rollup="Вторая версия недели.")
    session.commit()
    second = coach_note(session, week_id=WEEK_ID, provider=_FakeProvider(_note_payload()))
    session.commit()

    assert first_week_version.id != second_week_version.id
    assert second.artifact.summary_id == second_week_version.id
    assert second.artifact.version == 1  # a new upstream artifact is a new pair


def test_the_stored_disclaimer_is_not_taken_from_the_model(session: Session) -> None:
    _store_week_summary(session)

    outcome = coach_note(
        session,
        week_id=WEEK_ID,
        provider=_FakeProvider(
            _note_payload(summary_id="1999-W01", disclaimer="Другой дисклеймер от модели.")
        ),
    )
    session.commit()

    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None and stored.result_json is not None
    assert stored.result_json["disclaimer"] == COACH_DISCLAIMER
    assert stored.result_json["summary_id"] == WEEK_ID


# ---------------------------------------------------------------------------
# a failed run stores nothing
# ---------------------------------------------------------------------------


def test_invalid_completion_stores_nothing_and_leaves_the_input_alone(session: Session) -> None:
    upstream = _store_week_summary(session)
    session.commit()
    raw_before = _raw_state(session)
    summaries_before = _week_summary_state(session)

    with pytest.raises(LLMStructuredOutputError):
        coach_note(session, week_id=WEEK_ID, provider=_FakeProvider("Не JSON вовсе."))
    session.rollback()

    assert _notes(session) == []
    assert _raw_state(session) == raw_before
    assert _week_summary_state(session) == summaries_before
    stored_upstream = session.get(Artifact, upstream.id)
    assert stored_upstream is not None and stored_upstream.result_json is not None


@pytest.mark.parametrize(
    "payload",
    [
        {"observations": OBSERVATIONS[:2], "hypotheses": HYPOTHESES, "experiment_7d": EXPERIMENT},
        {"observations": OBSERVATIONS, "hypotheses": HYPOTHESES[:1], "experiment_7d": EXPERIMENT},
        {"observations": OBSERVATIONS, "hypotheses": HYPOTHESES, "experiment_7d": "На неделе."},
        {**_note_payload(), "advice": "Отдыхай больше."},
    ],
)
def test_an_invalid_shape_stores_nothing(session: Session, payload: dict[str, object]) -> None:
    _store_week_summary(session)
    session.commit()

    with pytest.raises(LLMStructuredOutputError):
        coach_note(session, week_id=WEEK_ID, provider=_FakeProvider(payload))
    session.rollback()

    assert _notes(session) == []


# ---------------------------------------------------------------------------
# launch is explicit only: the queue does not run the layer
# ---------------------------------------------------------------------------


def test_an_enqueued_coach_job_fails_instead_of_running_the_layer(
    session: Session, session_factory: sessionmaker[Session]
) -> None:
    """No handler is registered, so a ``coach_note`` job is a permanent failure."""
    _store_week_summary(session)
    job = JobRepository(session).enqueue(
        job_type=JobType.coach_note,
        payload_json={"pipeline_type": PipelineType.coach_note.value},
    )
    session.commit()
    job_id = job.id

    assert process_due_jobs(session_factory, worker_id="coach-test") == 1

    with session_factory() as check:
        stored_job = JobRepository(check).get(job_id)
        assert stored_job is not None and stored_job.status is JobStatus.failed
        assert list_layer_results(check, layer=ArtifactLayer.coach_note) == []


def test_the_coach_layer_never_produces_transcript_or_day_layers(session: Session) -> None:
    """A run adds a layer-5 row only; layers 4 and below are untouched."""
    _store_week_summary(session)
    session.commit()
    before = session.execute(text("SELECT COUNT(*) FROM artifacts")).scalar_one()

    coach_note(session, week_id=WEEK_ID, provider=_FakeProvider(_note_payload()))
    session.commit()

    assert session.execute(text("SELECT COUNT(*) FROM artifacts")).scalar_one() == before + 1
    layers = session.execute(text("SELECT DISTINCT layer FROM artifacts")).scalars().all()
    assert set(layers) == {"weekly_summary", "coach_note"}
