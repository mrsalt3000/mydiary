"""Integration tests for the T05 daily-summary layer writing through T08.

Runs against an isolated, migration-created test database (never the production
one) with a fake LLM provider, and covers the parts of the T05 definition of
done that need real persistence:

* the day's messages are selected by the **local** calendar day (``APP_TIMEZONE``)
  in chronological order — a UTC-only boundary would put them on the wrong day;
* voice messages map to ``raw_entry`` with their ``audio_path`` and the raw text
  of their newest *succeeded* transcription (empty while untranscribed);
* soft-deleted messages are excluded;
* a valid completion is appended to ``artifacts`` under the ``local_date`` +
  ``local_tz`` anchor, with no weekly export and no rendered file;
* a repeated run appends ``v+1`` and leaves the previous version untouched
  (ADR-005);
* the raw source tables (``messages``, ``audio_files``, ``transcriptions``) are
  unchanged by the run — value by value, row by row;
* a failed run stores nothing and leaves the raw rows alone.

All content is synthetic (AGENTS.md §13); no endpoint, model or real
configuration file is used.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from diary.core.models_config import OpenAICompatibleProfile
from diary.db.models import (
    Artifact,
    ArtifactLayer,
    Message,
    MessageKind,
    PipelineType,
    RunStatus,
    TranscriptionStatus,
)
from diary.db.repositories import (
    AudioFileRepository,
    MessageRepository,
    ProcessingRunRepository,
    TranscriptionRepository,
)
from diary.db.results import list_layer_results, trace_layer_result
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines.base import PipelineInvariantError, PipelinePermanentError
from diary.domain.pipelines.quality import GENERATION_ATTEMPTS
from diary.domain.pipelines.sum_daily.day_input import DayInputError
from diary.domain.pipelines.sum_daily.pipeline import ENTRIES_HEADER
from diary.domain.pipelines.sum_daily.service import sum_daily

DAY = date(2026, 9, 14)
LOCAL_TZ = "Europe/Moscow"

MORNING_TEXT = "Утром сходил в поликлинику за справкой для бассейна."
EVENING_TEXT = "вечером починил велосипед и посмотрел фильм"
NEXT_DAY_TEXT = "На следующий день поехал на дачу к родителям."

#: Synthetic ``raw_entry`` values used by the fake completion.
FACTS = [MORNING_TEXT, "Вечером починил велосипед и посмотрел фильм."]


class _FakeProvider:
    """Deterministic provider: one fixed response, recorded requests."""

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
    def request_entries(self) -> list[dict[str, object]]:
        """The ``raw_entry`` array the request hands to the model.

        The layer appends the day's entries to the prompt after a fixed header;
        parsing them back is how these tests observe the selected day without
        matching substrings that the prompt's own synthetic examples could also
        contain.
        """
        _, separator, payload = self.request_text.partition(f"{ENTRIES_HEADER}\n")
        assert separator, "the request must carry the day's entries"
        parsed = json.loads(payload)
        assert isinstance(parsed, list)
        return parsed


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
    payload: dict[str, object] = {
        "facts": list(FACTS),
        "themes": ["рабочие письма", "бассейн"],
        "energy": None,
        "open_questions": [],
    }
    payload.update(overrides)
    return payload


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _text_message(session: Session, *, body: str, created_at: datetime) -> Message:
    return MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=created_at
    )


def _voice_message(session: Session, *, created_at: datetime) -> Message:
    """A voice message with one audio row and one succeeded transcription."""
    message = MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=created_at
    )
    audio = AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path=f"audio/2026/09/{uuid.uuid4()}.webm",
        size_bytes=123,
        sha256="d" * 64,
        created_at=created_at,
    )
    TranscriptionRepository(session).create(
        audio_file_id=audio.id,
        raw_text=EVENING_TEXT,
        status=TranscriptionStatus.succeeded,
        created_at=created_at,
    )
    return message


def _raw_state(session: Session) -> list[list[tuple]]:
    """Every row of every raw source table, as raw values (value-by-value)."""
    return [
        session.execute(text(f"SELECT * FROM {table} ORDER BY id")).all()  # noqa: S608
        for table in ("messages", "audio_files", "transcriptions")
    ]


def _run(session: Session):
    return ProcessingRunRepository(session).create(
        pipeline_type=PipelineType.daily_summary, status=RunStatus.succeeded
    )


# ---------------------------------------------------------------------------
# day selection
# ---------------------------------------------------------------------------


def test_the_day_input_is_the_local_day_only(session: Session) -> None:
    """Messages of the requested local day, in order, and nothing else."""
    # 2026-09-14 01:00 Moscow == 2026-09-13 22:00 UTC — a UTC boundary would
    # exclude it; the local-day rule includes it (and 09-13 must not).
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    _voice_message(session, created_at=_utc("2026-09-14T16:45:00"))
    # 2026-09-15 00:30 Moscow == 2026-09-14 21:30 UTC — the next local day.
    _text_message(session, body=NEXT_DAY_TEXT, created_at=_utc("2026-09-14T21:30:00"))
    provider = _FakeProvider(_payload())

    outcome = sum_daily(
        session,
        local_date=DAY,
        profile=_profile(),
        provider=provider,
        local_tz=LOCAL_TZ,
    )

    # Exactly the two entries of 2026-09-14, in chronological order.
    assert [entry["text"] for entry in provider.request_entries] == [MORNING_TEXT, EVENING_TEXT]
    # The stored summary is the validated contract of that same day.
    assert outcome.summary.date == DAY
    assert outcome.summary.facts == FACTS


def test_the_voice_entry_carries_its_audio_path_and_raw_stt_text(session: Session) -> None:
    voice = _voice_message(session, created_at=_utc("2026-09-14T16:45:00"))
    audio_path = AudioFileRepository(session).get_by_message(voice.id)[-1].storage_path
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    provider = _FakeProvider(_payload())

    sum_daily(session, local_date=DAY, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)

    text_entry, voice_entry = provider.request_entries
    assert text_entry["source"] == "text"
    assert text_entry["audio_path"] is None
    assert text_entry["text"] == MORNING_TEXT
    assert voice_entry["source"] == "voice"
    assert voice_entry["audio_path"] == audio_path
    assert voice_entry["text"] == EVENING_TEXT  # the raw STT text, unfixed
    assert voice_entry["id"] == str(voice.id)


def test_soft_deleted_messages_are_not_summarized(session: Session) -> None:
    deleted = _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    _text_message(session, body=EVENING_TEXT, created_at=_utc("2026-09-14T16:45:00"))
    deleted.deleted_at = _utc("2026-09-14T17:00:00")
    session.flush()
    provider = _FakeProvider(_payload())

    sum_daily(session, local_date=DAY, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)

    assert [entry["text"] for entry in provider.request_entries] == [EVENING_TEXT]


def test_an_untranscribed_voice_message_enters_with_empty_text(session: Session) -> None:
    """The contract allows ``text=""`` while a voice message has no STT yet."""
    message = MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=_utc("2026-09-13T22:00:00")
    )
    audio = AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path=f"audio/2026/09/{uuid.uuid4()}.webm",
        size_bytes=11,
        sha256="e" * 64,
        created_at=_utc("2026-09-13T22:00:00"),
    )
    provider = _FakeProvider(_payload())

    sum_daily(session, local_date=DAY, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)

    (entry,) = provider.request_entries
    assert entry["text"] == ""
    assert entry["source"] == "voice"
    assert entry["audio_path"] == audio.storage_path


def test_a_failed_newer_transcription_does_not_hide_the_succeeded_one(
    session: Session,
) -> None:
    message = MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=_utc("2026-09-13T22:00:00")
    )
    audio = AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path=f"audio/2026/09/{uuid.uuid4()}.webm",
        size_bytes=11,
        sha256="f" * 64,
        created_at=_utc("2026-09-13T22:00:00"),
    )
    transcripts = TranscriptionRepository(session)
    transcripts.create(
        audio_file_id=audio.id,
        raw_text=EVENING_TEXT,
        status=TranscriptionStatus.succeeded,
        created_at=_utc("2026-09-13T22:00:01"),
    )
    transcripts.create(
        audio_file_id=audio.id,
        raw_text="",
        status=TranscriptionStatus.failed,
        created_at=_utc("2026-09-13T22:00:02"),
    )
    provider = _FakeProvider(_payload())

    sum_daily(session, local_date=DAY, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)

    (entry,) = provider.request_entries
    assert entry["text"] == EVENING_TEXT


def test_a_voice_message_without_audio_metadata_fails_loudly(session: Session) -> None:
    MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=_utc("2026-09-13T22:00:00")
    )
    provider = _FakeProvider(_payload())

    with pytest.raises(DayInputError):
        sum_daily(session, local_date=DAY, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)

    assert provider.requests == []


def test_an_empty_day_fails_loudly_and_stores_nothing(session: Session) -> None:
    provider = _FakeProvider(_payload())

    with pytest.raises(DayInputError):
        sum_daily(session, local_date=DAY, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)
    session.rollback()

    assert provider.requests == []
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY) == []


# ---------------------------------------------------------------------------
# storing a validated summary
# ---------------------------------------------------------------------------


def test_valid_summary_is_appended_under_the_local_day_anchor(session: Session) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    _voice_message(session, created_at=_utc("2026-09-14T16:45:00"))
    session.commit()
    raw_before = _raw_state(session)

    outcome = sum_daily(
        session,
        local_date=DAY,
        profile=_profile(),
        provider=_FakeProvider(_payload()),
        local_tz=LOCAL_TZ,
        processing_run_id=_run(session).id,
    )
    session.commit()

    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None
    assert stored.layer is ArtifactLayer.daily_summary
    assert stored.export_id is None  # second anchor family: no weekly export
    assert stored.local_date == DAY
    assert stored.local_tz == LOCAL_TZ
    assert stored.raw_id is None and stored.week_id is None and stored.summary_id is None
    assert stored.version == 1
    assert stored.path is None and stored.sha256 is None  # stored inline (ADR-008 §1)
    assert stored.result_json == {
        "date": DAY.isoformat(),
        "facts": FACTS,
        "themes": ["рабочие письма", "бассейн"],
        "energy": None,
        "open_questions": [],
    }
    # Exactly the contract's fields: no provenance or advice leaked into it.
    assert set(stored.result_json) == {"date", "facts", "themes", "energy", "open_questions"}

    # The raw source tables are untouched, value by value, row by row.
    assert _raw_state(session) == raw_before

    # The result is traceable back to its local-day anchor.
    trace = trace_layer_result(session, stored.id)
    assert trace.anchor_field == "local_date"
    assert trace.anchor_value == DAY
    assert trace.source_kind == "local_day"
    assert trace.run is not None


def test_a_silent_source_stores_energy_as_null(session: Session) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))

    outcome = sum_daily(
        session,
        local_date=DAY,
        profile=_profile(),
        provider=_FakeProvider(_payload(energy=None)),
        local_tz=LOCAL_TZ,
    )
    session.commit()

    assert outcome.summary.energy is None
    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None and stored.result_json is not None
    assert stored.result_json["energy"] is None


def test_a_stated_energy_is_stored_as_stated(session: Session) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))

    outcome = sum_daily(
        session,
        local_date=DAY,
        profile=_profile(),
        provider=_FakeProvider(_payload(energy="high")),
        local_tz=LOCAL_TZ,
    )
    session.commit()

    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None and stored.result_json is not None
    assert stored.result_json["energy"] == "high"


def test_a_rerun_appends_a_new_version_and_keeps_the_previous_one(
    session: Session,
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    first_provider = _FakeProvider(_payload())
    second_provider = _FakeProvider(_payload(facts=["Другой вариант факта."]))

    first = sum_daily(
        session, local_date=DAY, profile=_profile(), provider=first_provider, local_tz=LOCAL_TZ
    )
    session.commit()
    second = sum_daily(
        session, local_date=DAY, profile=_profile(), provider=second_provider, local_tz=LOCAL_TZ
    )
    session.commit()

    assert first.artifact.version == 1
    assert second.artifact.version == 2
    lineage = list_layer_results(
        session, layer=ArtifactLayer.daily_summary, local_date=DAY, local_tz=LOCAL_TZ
    )
    assert [row.version for row in lineage] == [1, 2]
    # Append-only: the first version still carries its own facts (ADR-005).
    assert lineage[0].result_json is not None and lineage[0].result_json["facts"] == FACTS
    assert lineage[1].result_json is not None
    assert lineage[1].result_json["facts"] == ["Другой вариант факта."]


def test_two_different_local_days_are_two_separate_lineages(session: Session) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    _text_message(session, body=NEXT_DAY_TEXT, created_at=_utc("2026-09-14T21:30:00"))

    first = sum_daily(
        session,
        local_date=DAY,
        profile=_profile(),
        provider=_FakeProvider(_payload()),
        local_tz=LOCAL_TZ,
    )
    next_day = sum_daily(
        session,
        local_date=date(2026, 9, 15),
        profile=_profile(),
        provider=_FakeProvider(_payload(facts=[NEXT_DAY_TEXT])),
        local_tz=LOCAL_TZ,
    )
    session.commit()

    assert first.artifact.version == 1
    assert next_day.artifact.version == 1  # a new anchor, a new lineage
    assert first.artifact.local_date != next_day.artifact.local_date
    rows = list_layer_results(session, layer=ArtifactLayer.daily_summary)
    assert sorted(row.local_date for row in rows if row.local_date) == [
        DAY,
        date(2026, 9, 15),
    ]


def test_the_stored_anchor_timezone_is_the_requested_one(session: Session) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))

    outcome = sum_daily(
        session,
        local_date=DAY,
        profile=_profile(),
        provider=_FakeProvider(_payload()),
        local_tz="Europe/Moscow",
    )
    session.commit()

    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None and stored.local_tz == "Europe/Moscow"


# ---------------------------------------------------------------------------
# a failed run stores nothing
# ---------------------------------------------------------------------------


def test_invalid_completion_stores_nothing_and_leaves_the_raw_rows_alone(
    session: Session,
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    session.commit()  # the raw source rows are persisted before the run
    raw_before = _raw_state(session)
    provider = _FakeProvider("Не JSON вовсе.")

    with pytest.raises(LLMStructuredOutputError):
        sum_daily(session, local_date=DAY, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)
    session.rollback()

    assert list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY) == []
    assert _raw_state(session) == raw_before


def test_an_extra_field_in_the_completion_stores_nothing(session: Session) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    session.commit()
    raw_before = _raw_state(session)

    with pytest.raises(LLMStructuredOutputError):
        sum_daily(
            session,
            local_date=DAY,
            profile=_profile(),
            provider=_FakeProvider(_payload(advice="Ложись раньше")),
            local_tz=LOCAL_TZ,
        )
    session.rollback()

    assert list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY) == []
    assert _raw_state(session) == raw_before
    assert session.scalars(select(Message)).all()  # the source stays readable


def test_a_completion_that_breaks_an_invariant_stores_nothing(
    session: Session,
) -> None:
    """T17b: the layer refuses its own output before it can be stored.

    Before the gate the advice was stored and only the eval harness flagged it,
    which is how the phase acceptance could measure advice in real artifacts.
    Now the run fails permanently, names the check it broke and leaves no
    artifact, no marker and no text of the diary behind in the error.
    """
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    session.commit()
    raw_before = _raw_state(session)
    payload = _payload()
    payload["facts"] = ["Ты молодец, успела многое."]
    provider = _FakeProvider(payload)

    with pytest.raises(PipelineInvariantError) as error:
        sum_daily(session, local_date=DAY, profile=_profile(), provider=provider, local_tz=LOCAL_TZ)
    # Checked before the rollback: "nothing was written" is a stronger claim than
    # "nothing survived a rollback", and only the first one is the guarantee.
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY) == []
    session.rollback()

    assert error.value.checks == ("advice_and_judgement_absent",)
    assert "молодец" not in str(error.value)
    assert len(provider.requests) == GENERATION_ATTEMPTS  # a bounded regeneration
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY) == []
    assert _raw_state(session) == raw_before


class _ContradictionOnce:
    """A judge whose first answer set contradicts and whose later ones support.

    The runtime NLI gate is stochastic-defect tolerant: a contradiction means
    «generate again», not «fail», so the first attempt is expected to be re-asked.
    """

    def load(self) -> None:
        """No weights here; the gate calls this to prove the instrument."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, pairs):
        self.calls += 1
        if self.calls == 1:
            return [{"entailment": 0.02, "contradiction": 0.97, "neutral": 0.01} for _ in pairs]
        return [{"entailment": 0.99, "contradiction": 0.005, "neutral": 0.005} for _ in pairs]


def test_a_contradicting_summary_is_generated_again_and_the_clean_one_is_stored(
    session: Session,
) -> None:
    """T24b: the gate shares the policy of T17b — regenerate, then refuse.

    The premise is the day's own text, so a claim the source contradicts can be
    re-asked like any other stochastic slip; the clean second attempt is stored.
    """
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    session.commit()
    provider = _FakeProvider(_payload())
    judge = _ContradictionOnce()

    outcome = sum_daily(
        session,
        local_date=DAY,
        profile=_profile(),
        provider=provider,
        local_tz=LOCAL_TZ,
        judge=judge,
    )

    assert judge.calls == 2  # one contradiction, then a clean generation
    assert len(provider.requests) == 2
    assert len(list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY)) == 1
    assert outcome.summary.facts == list(_payload()["facts"])


def test_a_persistently_contradicting_summary_stores_nothing(session: Session) -> None:
    """The refusal path of the runtime gate, with the check it names and no text."""
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    session.commit()
    raw_before = _raw_state(session)
    provider = _FakeProvider(_payload())

    class _AlwaysContradicting:
        def load(self) -> None:
            """No weights here; the gate calls this to prove the instrument."""

        def __call__(self, pairs):
            return [{"entailment": 0.01, "contradiction": 0.98, "neutral": 0.01} for _ in pairs]

    with pytest.raises(PipelineInvariantError) as error:
        sum_daily(
            session,
            local_date=DAY,
            profile=_profile(),
            provider=provider,
            local_tz=LOCAL_TZ,
            judge=_AlwaysContradicting(),
        )
    # Checked before the rollback: nothing was written at all.
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY) == []
    session.rollback()

    assert error.value.checks == ("no_contradiction_with_source",)
    assert len(provider.requests) == GENERATION_ATTEMPTS
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY) == []
    assert _raw_state(session) == raw_before
    # The failure message stays content-free (AGENTS.md §7-8).
    assert "Плавал" not in str(error.value)


def test_an_unavailable_instrument_fails_the_day_permanently(session: Session) -> None:
    """The owner's decision for T24: an unavailable instrument never means «passed»."""
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))
    session.commit()

    class _Unavailable:
        def load(self) -> None:
            from diary.domain.nli.onnx_judge import NliUnavailableError

            raise NliUnavailableError("/nowhere/model.onnx is missing")

    with pytest.raises(PipelinePermanentError):
        sum_daily(
            session,
            local_date=DAY,
            profile=_profile(),
            provider=_FakeProvider(_payload()),
            local_tz=LOCAL_TZ,
            judge=_Unavailable(),
        )
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY) == []
    session.rollback()


def test_the_result_carries_no_weekly_export_reference(session: Session) -> None:
    """The day layer works before (and without) any weekly export (ADR-007)."""
    _text_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-13T22:00:00"))

    outcome = sum_daily(
        session,
        local_date=DAY,
        profile=_profile(),
        provider=_FakeProvider(_payload()),
        local_tz=LOCAL_TZ,
    )
    session.commit()

    assert outcome.artifact.export_id is None
    assert outcome.artifact.local_date == DAY
    assert outcome.artifact.local_tz == LOCAL_TZ
