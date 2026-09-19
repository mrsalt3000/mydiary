"""Unit tests for the T05 daily-summary layer (no provider, no database).

Covers the definition of done of task T05:

* a valid completion becomes a validated :class:`DailySummary` and is handed to
  the append-only store exactly once, under the ``local_date`` + ``local_tz``
  anchor;
* the result is strictly the contract's fields: ``facts`` / ``themes`` /
  ``energy`` / ``open_questions`` plus the backend-owned ``date`` — an extra
  field (advice, morality, evaluation) fails the run and stores nothing;
* ``energy`` is never invented: a source that stays silent yields ``None``, an
  explicitly stated level keeps its value, and an out-of-contract level is a
  loud failure;
* an invalid completion (non-JSON, missing/empty required field, extra field)
  raises a loud, non-retryable LLM error and stores **nothing** — no partial
  artifact;
* the anchor field ``date`` comes from the backend, never from the model's
  answer;
* the day's entries are not modified, and the layer has no other write path;
* a day without messages fails loudly before any request is sent.

The provider is a fake and the store is a recording double, so no endpoint,
configuration file, model or database is touched. All content is synthetic
(AGENTS.md §13).
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Any

import pytest

from diary.core.models_config import OpenAICompatibleProfile
from diary.core.settings import get_settings
from diary.db.models import ArtifactLayer, PipelineType
from diary.domain.contracts import DailySummary, RawEntry
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines import naming
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.sum_daily import service as sum_daily_service
from diary.domain.pipelines.sum_daily.day_input import DayInputError
from diary.domain.pipelines.sum_daily.pipeline import (
    ENTRIES_HEADER,
    PROMPT_PATH,
    PROMPT_VERSION,
    SumDailyPipeline,
)
from diary.domain.pipelines.sum_daily.registration import ensure_sum_daily_pipeline_registered

DAY = date(2026, 9, 14)
LOCAL_TZ = "Europe/Moscow"
ENTRY_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
ENTRY_TEXT = "Разобрал входящие письма и подготовил черновик отчёта по складу."
SECOND_TEXT = "Думаю, стоит ли менять поставщика; пока не решил."


def _entry(*, text: str = ENTRY_TEXT, entry_id: uuid.UUID = ENTRY_ID) -> RawEntry:
    return RawEntry(
        id=entry_id,
        ts=datetime(2026, 9, 14, 13, 20, tzinfo=timezone.utc),
        source="text",
        text=text,
    )


def _entries() -> list[RawEntry]:
    return [
        _entry(),
        _entry(
            text=SECOND_TEXT,
            entry_id=uuid.UUID("44444444-4444-4444-8444-444444444444"),
        ),
    ]


def _profile(model: str = "diary-sum") -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:11434/v1",
        model=model,
        api_key_env="SYNTHETIC_PHASE2_KEY",
        temperature=0.25,
        max_tokens=4096,
    )


class _FakeProvider:
    """Deterministic provider returning one fixed response (records requests)."""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response


def _json_response(payload: object) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload, ensure_ascii=False), model="чужой-модель")


#: A contract-valid completion: two facts, themes, no energy, one open question.
_VALID_PAYLOAD: dict[str, object] = {
    "facts": [ENTRY_TEXT, SECOND_TEXT],
    "themes": ["рабочие письма"],
    "energy": None,
    "open_questions": ["Менять ли поставщика?"],
}


class _RecordingStore:
    """Records the arguments of the append-only store call (no database)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, session: object, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return type("StoredArtifact", (), {"id": uuid.uuid4(), "version": 1})()


@pytest.fixture(autouse=True)
def _restore_registries() -> None:
    snapshot = dict(pipeline_registry.SECOND_FAMILY_REGISTRY)
    yield
    pipeline_registry.SECOND_FAMILY_REGISTRY.clear()
    pipeline_registry.SECOND_FAMILY_REGISTRY.update(snapshot)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _RecordingStore:
    recording = _RecordingStore()
    monkeypatch.setattr(sum_daily_service, "save_layer_result", recording)
    return recording


def _run(store: _RecordingStore, provider: _FakeProvider) -> sum_daily_service.SumDailyOutcome:
    return sum_daily_service.sum_daily(
        None,  # type: ignore[arg-type]  # the store double ignores the session
        local_date=DAY,
        profile=_profile(),
        provider=provider,
        local_tz=LOCAL_TZ,
        entries=_entries(),
    )


# ---------------------------------------------------------------------------
# prompt template
# ---------------------------------------------------------------------------


def test_prompt_template_is_v1_and_states_the_energy_rule() -> None:
    assert PROMPT_VERSION == "v1"
    assert PROMPT_PATH.is_file()
    text = PROMPT_PATH.read_text(encoding="utf-8")
    # The four content fields and the "never invent energy" rule are explicit.
    assert "energy" in text
    assert "null" in text
    assert "JSON" in text
    # Advice/morality is a veto of this layer, so the prompt must forbid it.
    assert "совет" in text
    assert "морал" in text


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------


def test_layer_is_registered_in_the_second_family_registry() -> None:
    assert ensure_sum_daily_pipeline_registered() is True
    assert ensure_sum_daily_pipeline_registered() is False  # idempotent
    registered = pipeline_registry.get_second_family_pipeline(PipelineType.daily_summary)
    assert isinstance(registered, SumDailyPipeline)
    assert registered.layer is ArtifactLayer.daily_summary
    assert registered.pipeline_type is PipelineType.daily_summary
    assert registered.profile_name == "diary_sum"
    assert "daily_summary" in pipeline_registry.known_pipeline_types()
    # The export-driven lookup must not return a per-day layer.
    assert pipeline_registry.get_pipeline(PipelineType.daily_summary) is None


def test_registry_rejects_a_misdeclared_layer() -> None:
    class _WrongLayer(SumDailyPipeline):
        layer = ArtifactLayer.normalized

    with pytest.raises(ValueError, match="must declare layer"):
        pipeline_registry.register(_WrongLayer())


def test_registry_rejects_a_misdeclared_profile_name() -> None:
    class _WrongProfile(SumDailyPipeline):
        profile_name = "diary_stt"

    with pytest.raises(ValueError, match="must declare profile name"):
        pipeline_registry.register(_WrongProfile())


def test_naming_maps_know_the_daily_summary_layer() -> None:
    """The layer is mapped by profile name, not by a TaskType (T02/T04 rule)."""
    assert naming.PIPELINE_PROFILE_NAME[PipelineType.daily_summary] == "diary_sum"
    assert PipelineType.daily_summary not in naming.PIPELINE_TASK_TYPE
    assert naming.layer_output_extension(ArtifactLayer.daily_summary) == ".md"
    # The layer renders no file, so it has no per-layer artifact directory.
    with pytest.raises(ValueError):
        naming.layer_dirname(ArtifactLayer.daily_summary)


# ---------------------------------------------------------------------------
# request building
# ---------------------------------------------------------------------------


def test_request_carries_the_day_entries_in_chronological_order() -> None:
    entries = _entries()
    request = SumDailyPipeline().build_request(entries=entries)

    assert request.prompt_version == PROMPT_VERSION
    content = request.messages[0].content
    assert ENTRIES_HEADER in content
    # The day's entries are handed over as the raw_entry JSON array, unchanged.
    for entry in entries:
        assert json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) in content
    assert content.index(ENTRY_TEXT) < content.index(SECOND_TEXT)


def test_a_day_without_messages_fails_loudly_before_the_provider(
    store: _RecordingStore,
) -> None:
    provider = _FakeProvider(_json_response(_VALID_PAYLOAD))

    with pytest.raises(DayInputError):
        sum_daily_service.sum_daily(
            None,  # type: ignore[arg-type]
            local_date=DAY,
            profile=_profile(),
            provider=provider,
            local_tz=LOCAL_TZ,
            entries=[],
        )

    assert provider.requests == []
    assert store.calls == []


# ---------------------------------------------------------------------------
# successful run
# ---------------------------------------------------------------------------


def test_valid_response_yields_a_summary_and_is_stored_once(
    store: _RecordingStore,
) -> None:
    outcome = _run(store, _FakeProvider(_json_response(_VALID_PAYLOAD)))

    assert outcome.summary == DailySummary(
        date=DAY,
        facts=[ENTRY_TEXT, SECOND_TEXT],
        themes=["рабочие письма"],
        energy=None,
        open_questions=["Менять ли поставщика?"],
    )
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["layer"] is ArtifactLayer.daily_summary
    assert call["local_date"] == DAY  # the day anchor, not a weekly export
    assert call["local_tz"] == LOCAL_TZ
    assert call["result"] is outcome.summary
    # No rendered file: the layer's result is stored inline (ADR-008 §1).
    assert "path" not in call or call.get("path") is None
    # The day anchor is the only anchor: the two families never mix.
    assert "raw_id" not in call and "week_id" not in call and "summary_id" not in call


def test_the_anchor_timezone_defaults_to_the_application_timezone(
    store: _RecordingStore,
) -> None:
    sum_daily_service.sum_daily(
        None,  # type: ignore[arg-type]
        local_date=DAY,
        profile=_profile(),
        provider=_FakeProvider(_json_response(_VALID_PAYLOAD)),
        entries=_entries(),
    )

    assert store.calls[0]["local_tz"] == get_settings().timezone


def test_the_date_anchor_comes_from_the_backend_not_the_answer(
    store: _RecordingStore,
) -> None:
    """A completion claiming another day cannot change the stored day."""
    payload = dict(_VALID_PAYLOAD, date="1999-01-01")
    outcome = _run(store, _FakeProvider(_json_response(payload)))

    assert outcome.summary.date == DAY
    stored = store.calls[0]["result"]
    assert stored.date == DAY


# ---------------------------------------------------------------------------
# energy is never invented
# ---------------------------------------------------------------------------


def test_a_silent_source_leaves_energy_none(store: _RecordingStore) -> None:
    outcome = _run(store, _FakeProvider(_json_response(_VALID_PAYLOAD)))

    assert outcome.summary.energy is None
    assert store.calls[0]["result"].energy is None


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_an_explicitly_stated_energy_keeps_its_value(store: _RecordingStore, level: str) -> None:
    payload = dict(_VALID_PAYLOAD, energy=level)
    outcome = _run(store, _FakeProvider(_json_response(payload)))

    assert outcome.summary.energy == level
    assert store.calls[0]["result"].energy == level


def test_an_out_of_contract_energy_fails_loudly(store: _RecordingStore) -> None:
    payload = dict(_VALID_PAYLOAD, energy="very high")

    with pytest.raises(LLMStructuredOutputError):
        _run(store, _FakeProvider(_json_response(payload)))

    assert store.calls == []


# ---------------------------------------------------------------------------
# failed runs: loud, and without any stored result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_text",
    [
        "Не JSON вовсе.",
        '```json\n{"facts": ["Синтетика."]}\n```',
        '[{"facts": ["Синтетика."]}]',
    ],
)
def test_invalid_json_fails_loudly_and_stores_nothing(
    store: _RecordingStore, raw_text: str
) -> None:
    with pytest.raises(LLMStructuredOutputError):
        _run(store, _FakeProvider(LLMResponse(text=raw_text)))

    assert store.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"themes": ["Тема"]},  # missing facts
        {"facts": []},  # empty facts is a fabrication-free summary that says nothing
        {"facts": [""]},  # empty fact string
        # An extra field is advice/morality the contract does not allow.
        dict(_VALID_PAYLOAD, advice="Ложись раньше"),
        dict(_VALID_PAYLOAD, mood="хорошее"),
    ],
)
def test_invalid_contract_fails_loudly_and_stores_nothing(
    store: _RecordingStore, payload: dict[str, object]
) -> None:
    with pytest.raises(LLMStructuredOutputError):
        _run(store, _FakeProvider(_json_response(payload)))

    assert store.calls == []


def test_themes_and_open_questions_may_be_omitted(store: _RecordingStore) -> None:
    """Both have contract defaults; a day with only facts is still valid."""
    outcome = _run(store, _FakeProvider(_json_response({"facts": [ENTRY_TEXT]})))

    assert outcome.summary.facts == [ENTRY_TEXT]
    assert outcome.summary.themes == []
    assert outcome.summary.open_questions == []
    assert outcome.summary.energy is None
    assert len(store.calls) == 1


# ---------------------------------------------------------------------------
# the input is not modified
# ---------------------------------------------------------------------------


def test_the_run_does_not_change_the_day_entries(store: _RecordingStore) -> None:
    entries = _entries()
    before = [entry.model_dump() for entry in entries]

    _run(store, _FakeProvider(_json_response(_VALID_PAYLOAD)))

    assert [entry.model_dump() for entry in entries] == before
    assert entries[0].text == ENTRY_TEXT  # the source text stays untouched
    # The only write performed by the layer is the append-only result store.
    assert [call["layer"] for call in store.calls] == [ArtifactLayer.daily_summary]


def test_the_layer_declares_no_rendered_file_location() -> None:
    """The layer's result is inline; it has no per-layer artifact directory."""
    assert PROMPT_PATH.parent.name == "sum_daily"
    assert PROMPT_PATH.name == "v1.md"
    assert SumDailyPipeline.output_ext == ".md"
