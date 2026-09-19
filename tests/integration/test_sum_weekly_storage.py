"""Integration tests for the T06 weekly-summary layer writing through T08.

Runs against an isolated, migration-created test database (never the production
one) with a fake LLM provider, and covers the parts of the T06 definition of
done that need real persistence:

* the input is the stored **daily summaries** of the ISO week — the raw tables
  (``messages``, ``audio_files``, ``transcriptions``) are not read and are
  unchanged by the run, and no weekly export is involved;
* the newest version of each day's lineage is the one merged, and days without
  a summary are simply absent while every stored day is included;
* the ISO-week anchor works across the New Year (``2026-W53`` runs from
  2026-12-28 to 2027-01-03);
* a valid completion is appended to ``artifacts`` under the ``week_id`` anchor,
  with no weekly export and no rendered file;
* a repeated run appends ``v+1`` and leaves the previous version untouched
  (ADR-005);
* a failed run stores nothing and leaves the raw rows and the day summaries
  alone.

All content is synthetic (AGENTS.md §13); no endpoint, model or real
configuration file is used.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from diary.core.models_config import OpenAICompatibleProfile
from diary.db.models import Artifact, ArtifactLayer, Message, MessageKind
from diary.db.repositories import MessageRepository
from diary.db.results import list_layer_results, save_layer_result, trace_layer_result
from diary.domain.contracts import DailySummary
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines.sum_weekly.pipeline import SUMMARIES_HEADER
from diary.domain.pipelines.sum_weekly.service import sum_weekly
from diary.domain.pipelines.sum_weekly.week_input import WeeklyInputError, week_days

WEEK_ID = "2026-W38"
LOCAL_TZ = "Europe/Moscow"
DAYS = [day.isoformat() for day in week_days(WEEK_ID)]
FACTS = [f"Синтетический факт дня {day}." for day in DAYS]
ROLLUP = "Неделя прошла вокруг отчёта и бассейна; сон сбился к середине недели."

#: Synthetic raw text that must never reach the weekly layer's input.
RAW_TEXT = "Сырое сообщение дневника, которого не должно быть в недельной сводке."


class _FakeProvider:
    """Deterministic provider: a fixed response, recorded requests."""

    def __init__(self, payload: object | str) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.response = LLMResponse(text=text)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response

    @property
    def request_text(self) -> str:
        """The content of the single request this provider received."""
        assert len(self.requests) == 1
        return self.requests[0].messages[0].content

    @property
    def request_days(self) -> list[str]:
        """The ``date`` of every daily summary in the request, in order."""
        _, separator, payload = self.request_text.partition(f"{SUMMARIES_HEADER}\n")
        assert separator, "the request must carry the week's daily summaries"
        parsed = json.loads(payload)
        assert isinstance(parsed, list)
        return [entry["date"] for entry in parsed]


def _profile(model: str = "diary-sum") -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:1/v1",  # never contacted by the fake
        model=model,
        api_key_env="SYNTHETIC_PHASE2_KEY",
        temperature=0.25,
        max_tokens=4096,
    )


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {"rollup": ROLLUP}
    payload.update(overrides)
    return payload


def _store_daily_summary(
    session: Session,
    *,
    day: date,
    facts: list[str],
    local_tz: str = LOCAL_TZ,
) -> Artifact:
    """Store one ``daily_summary`` result (the weekly layer's only input)."""
    return save_layer_result(
        session,
        layer=ArtifactLayer.daily_summary,
        local_date=day,
        local_tz=local_tz,
        result=DailySummary(date=day, facts=facts),
    )


def _store_week(session: Session, *, week_id: str = WEEK_ID) -> list[Artifact]:
    return [
        _store_daily_summary(session, day=day, facts=[fact])
        for day, fact in zip(week_days(week_id), FACTS, strict=True)
    ]


def _raw_state(session: Session) -> list[list[tuple]]:
    """Every row of every raw source table, as raw values (value-by-value)."""
    return [
        session.execute(text(f"SELECT * FROM {table} ORDER BY id")).all()  # noqa: S608
        for table in ("messages", "audio_files", "transcriptions")
    ]


def _day_summary_state(session: Session) -> list[tuple[object, object]]:
    """The stored day-summary results, newest last (the layer's input)."""
    rows = list_layer_results(session, layer=ArtifactLayer.daily_summary)
    return [(row.local_date, row.result_json) for row in rows]


def _text_message(session: Session, *, body: str, created_at: datetime) -> Message:
    return MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=created_at
    )


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# input: the week's daily summaries, and nothing else
# ---------------------------------------------------------------------------


def test_the_input_is_the_weeks_stored_day_summaries(session: Session) -> None:
    _store_week(session)
    # A day of the next week and a weekly export are not this layer's input.
    _store_daily_summary(session, day=date(2026, 9, 21), facts=["Следующая неделя."])
    session.commit()
    provider = _FakeProvider(_payload())

    sum_weekly(session, week_id=WEEK_ID, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)

    assert provider.request_days == DAYS


def test_raw_messages_are_never_read_or_placed_in_the_request(session: Session) -> None:
    """The weekly layer merges day summaries, not raw entries (T06 DoD)."""
    _store_daily_summary(session, day=date(2026, 9, 14), facts=[FACTS[0]])
    for index, day in enumerate(DAYS):
        _text_message(
            session,
            body=f"{RAW_TEXT} {index}",
            created_at=_utc(f"{day}T10:00:00"),
        )
    session.commit()
    provider = _FakeProvider(_payload())

    sum_weekly(session, week_id=WEEK_ID, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)

    # Only the one stored day summary was fed; no raw message became an input.
    assert provider.request_days == [DAYS[0]]
    assert RAW_TEXT not in provider.request_text
    assert "raw_entry" not in provider.request_text


def test_only_the_newest_version_of_a_day_is_merged(session: Session) -> None:
    day = date(2026, 9, 14)
    _store_daily_summary(session, day=day, facts=["Старая версия факта."])
    _store_daily_summary(session, day=day, facts=[FACTS[0]])
    session.commit()
    provider = _FakeProvider(_payload())

    sum_weekly(session, week_id=WEEK_ID, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)

    assert FACTS[0] in provider.request_text
    assert "Старая версия факта." not in provider.request_text


def test_missing_days_are_skipped_but_every_stored_day_is_included(session: Session) -> None:
    present = [0, 2, 4, 6]
    for index in present:
        _store_daily_summary(session, day=date.fromisoformat(DAYS[index]), facts=[FACTS[index]])
    session.commit()
    provider = _FakeProvider(_payload())

    outcome = sum_weekly(
        session, week_id=WEEK_ID, profile=_profile(), provider=provider, local_tz=LOCAL_TZ
    )
    session.commit()

    assert provider.request_days == [DAYS[index] for index in present]
    assert outcome.summary.daily_ids == [DAYS[index] for index in present]


def test_a_day_summary_of_another_week_is_not_used(session: Session) -> None:
    _store_daily_summary(session, day=date(2026, 9, 21), facts=["Следующая неделя."])
    session.commit()
    provider = _FakeProvider(_payload())

    with pytest.raises(WeeklyInputError):
        sum_weekly(
            session, week_id=WEEK_ID, profile=_profile(), provider=provider, local_tz=LOCAL_TZ
        )
    session.rollback()

    assert provider.requests == []
    assert list_layer_results(session, layer=ArtifactLayer.weekly_summary, week_id=WEEK_ID) == []


def test_a_day_summary_computed_in_another_timezone_is_not_used(session: Session) -> None:
    _store_daily_summary(session, day=date(2026, 9, 14), facts=[FACTS[0]], local_tz="UTC")
    session.commit()
    provider = _FakeProvider(_payload())

    with pytest.raises(WeeklyInputError):
        sum_weekly(
            session, week_id=WEEK_ID, profile=_profile(), provider=provider, local_tz=LOCAL_TZ
        )

    assert provider.requests == []


def test_a_day_summary_without_an_inline_payload_fails_loudly(session: Session) -> None:
    """A day result stored only as a file cannot be merged and is not skipped."""
    save_layer_result(
        session,
        layer=ArtifactLayer.daily_summary,
        local_date=date(2026, 9, 14),
        local_tz=LOCAL_TZ,
        path="exports/03_summaries/2026-09-14_daily_summary_v1.md",
        sha256="a" * 64,
    )
    session.commit()
    provider = _FakeProvider(_payload())

    with pytest.raises(WeeklyInputError):
        sum_weekly(
            session, week_id=WEEK_ID, profile=_profile(), provider=provider, local_tz=LOCAL_TZ
        )

    assert provider.requests == []


def test_a_day_summary_whose_date_disagrees_with_its_anchor_fails_loudly(
    session: Session,
) -> None:
    """A mis-anchored day summary is a loud failure, never a silently wrong day."""
    save_layer_result(
        session,
        layer=ArtifactLayer.daily_summary,
        local_date=date(2026, 9, 14),
        local_tz=LOCAL_TZ,
        result=DailySummary(date=date(2026, 9, 15), facts=[FACTS[0]]),
    )
    session.commit()
    provider = _FakeProvider(_payload())

    with pytest.raises(WeeklyInputError):
        sum_weekly(
            session, week_id=WEEK_ID, profile=_profile(), provider=provider, local_tz=LOCAL_TZ
        )

    assert provider.requests == []


# ---------------------------------------------------------------------------
# the ISO week anchor, including the New Year boundary
# ---------------------------------------------------------------------------


def test_a_week_that_spans_the_new_year_is_one_anchor(session: Session) -> None:
    """2026-W53 runs from 2026-12-28 to 2027-01-03 (ISO year != calendar year)."""
    new_year_week = "2026-W53"
    days = [day.isoformat() for day in week_days(new_year_week)]
    for index, day in enumerate(days):
        _store_daily_summary(
            session, day=date.fromisoformat(day), facts=[f"Синтетический факт {index}."]
        )
    session.commit()
    provider = _FakeProvider(_payload())

    outcome = sum_weekly(
        session,
        week_id=new_year_week,
        profile=_profile(),
        provider=provider,
        local_tz=LOCAL_TZ,
    )
    session.commit()

    assert days[0] == "2026-12-28" and days[-1] == "2027-01-03"
    assert outcome.summary.week_id == "2026-W53"
    assert outcome.summary.daily_ids == days
    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None and stored.week_id == "2026-W53"


# ---------------------------------------------------------------------------
# storing a validated summary
# ---------------------------------------------------------------------------


def test_valid_summary_is_appended_under_the_week_anchor(session: Session) -> None:
    _store_week(session)
    session.commit()
    raw_before = _raw_state(session)
    day_summaries_before = _day_summary_state(session)

    outcome = sum_weekly(
        session,
        week_id=WEEK_ID,
        profile=_profile(),
        provider=_FakeProvider(_payload()),
        local_tz=LOCAL_TZ,
    )
    session.commit()

    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None
    assert stored.layer is ArtifactLayer.weekly_summary
    assert stored.export_id is None  # second anchor family: no weekly export
    assert stored.week_id == WEEK_ID
    assert stored.raw_id is None and stored.local_date is None and stored.summary_id is None
    assert stored.local_tz is None  # the timezone belongs to the day anchor only
    assert stored.version == 1
    assert stored.path is None and stored.sha256 is None  # stored inline (ADR-008 §1)
    assert stored.result_json == {"week_id": WEEK_ID, "daily_ids": DAYS, "rollup": ROLLUP}
    # Exactly the contract's fields: no provenance or advice leaked into it.
    assert set(stored.result_json) == {"week_id", "daily_ids", "rollup"}

    # The raw source tables and the input day summaries are untouched.
    assert _raw_state(session) == raw_before
    assert _day_summary_state(session) == day_summaries_before

    # The result is traceable back to its ISO-week anchor.
    trace = trace_layer_result(session, stored.id)
    assert trace.anchor_field == "week_id"
    assert trace.anchor_value == WEEK_ID
    assert trace.source_kind == "iso_week"


def test_a_rerun_appends_a_new_version_and_keeps_the_previous_one(
    session: Session,
) -> None:
    _store_week(session)
    first_provider = _FakeProvider(_payload(rollup="Первый вариант сводки недели."))
    second_provider = _FakeProvider(_payload(rollup="Второй вариант сводки недели."))

    first = sum_weekly(
        session, week_id=WEEK_ID, profile=_profile(), provider=first_provider, local_tz=LOCAL_TZ
    )
    session.commit()
    second = sum_weekly(
        session, week_id=WEEK_ID, profile=_profile(), provider=second_provider, local_tz=LOCAL_TZ
    )
    session.commit()

    assert first.artifact.version == 1
    assert second.artifact.version == 2
    lineage = list_layer_results(session, layer=ArtifactLayer.weekly_summary, week_id=WEEK_ID)
    assert [row.version for row in lineage] == [1, 2]
    # Append-only: the first version still carries its own rollup (ADR-005).
    assert lineage[0].result_json is not None
    assert lineage[0].result_json["rollup"] == "Первый вариант сводки недели."
    assert lineage[1].result_json is not None
    assert lineage[1].result_json["rollup"] == "Второй вариант сводки недели."


def test_two_different_weeks_are_two_separate_lineages(session: Session) -> None:
    _store_week(session)
    _store_week(session, week_id="2026-W39")
    session.commit()

    first = sum_weekly(
        session,
        week_id=WEEK_ID,
        profile=_profile(),
        provider=_FakeProvider(_payload()),
        local_tz=LOCAL_TZ,
    )
    second = sum_weekly(
        session,
        week_id="2026-W39",
        profile=_profile(),
        provider=_FakeProvider(_payload()),
        local_tz=LOCAL_TZ,
    )
    session.commit()

    assert first.artifact.version == 1
    assert second.artifact.version == 1  # a new anchor, a new lineage
    rows = list_layer_results(session, layer=ArtifactLayer.weekly_summary)
    assert sorted(row.week_id for row in rows if row.week_id) == [WEEK_ID, "2026-W39"]


# ---------------------------------------------------------------------------
# a failed run stores nothing
# ---------------------------------------------------------------------------


def test_invalid_completion_stores_nothing_and_leaves_the_input_alone(
    session: Session,
) -> None:
    _store_week(session)
    session.commit()
    raw_before = _raw_state(session)
    day_summaries_before = _day_summary_state(session)

    with pytest.raises(LLMStructuredOutputError):
        sum_weekly(
            session,
            week_id=WEEK_ID,
            profile=_profile(),
            provider=_FakeProvider("Не JSON вовсе."),
            local_tz=LOCAL_TZ,
        )
    session.rollback()

    assert list_layer_results(session, layer=ArtifactLayer.weekly_summary, week_id=WEEK_ID) == []
    assert _raw_state(session) == raw_before
    assert _day_summary_state(session) == day_summaries_before


def test_an_extra_field_in_the_completion_stores_nothing(session: Session) -> None:
    _store_week(session)
    session.commit()

    with pytest.raises(LLMStructuredOutputError):
        sum_weekly(
            session,
            week_id=WEEK_ID,
            profile=_profile(),
            provider=_FakeProvider(_payload(advice="Отдыхай больше.")),
            local_tz=LOCAL_TZ,
        )
    session.rollback()

    assert list_layer_results(session, layer=ArtifactLayer.weekly_summary, week_id=WEEK_ID) == []


def test_the_result_carries_no_weekly_export_reference(session: Session) -> None:
    """The week layer works before (and without) any weekly export (ADR-007)."""
    _store_week(session)

    outcome = sum_weekly(
        session,
        week_id=WEEK_ID,
        profile=_profile(),
        provider=_FakeProvider(_payload()),
        local_tz=LOCAL_TZ,
    )
    session.commit()

    assert outcome.artifact.export_id is None
    assert outcome.artifact.week_id == WEEK_ID
