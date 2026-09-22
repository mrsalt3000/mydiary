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
from diary.domain.contracts.golden import estimate_request_tokens, estimate_tokens
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines import naming
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.base import PipelinePermanentError
from diary.domain.pipelines.sum_daily import service as sum_daily_service
from diary.domain.pipelines.sum_daily.day_input import DayInputError
from diary.domain.pipelines.sum_daily.pipeline import (
    ENTRIES_HEADER,
    MERGE_PROMPT_PATH,
    MERGE_PROMPT_VERSION,
    NUM_CTX,
    PARTS_HEADER,
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


def test_run_sum_daily_arms_the_nli_gate(monkeypatch) -> None:
    """The convenience entry is a production entry, so it builds the instrument itself.

    ``sum_daily`` keeps the gate optional (the documented seam for tests and for the
    harness in CI), which means the arming has to happen *inside* this entry — and a
    review of T24b proved that no test caught its removal. This one does: the builder
    is recorded and the instrument is checked to reach the service.
    """
    from datetime import date as _date

    from diary.domain.pipelines.sum_daily import service as sum_daily_service

    class _Judge:
        """A stand-in: this test checks the wiring, and the service is faked anyway."""

        def load(self) -> None:
            """Nothing to load."""

        def __call__(self, pairs):  # pragma: no cover - never called here
            return []

    built: list[str] = []
    judge = _Judge()

    def _recorder() -> _Judge:
        built.append("armed")
        return judge

    captured: dict[str, object] = {}

    def _fake_sum_daily(session: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "stored"

    monkeypatch.setattr(sum_daily_service, "default_nli_judge", _recorder)
    monkeypatch.setattr(sum_daily_service, "sum_daily", _fake_sum_daily)
    monkeypatch.setattr(sum_daily_service, "resolve_phase2_profile", lambda config, name: object())
    monkeypatch.setattr(sum_daily_service, "create_llm_provider", lambda profile: object())

    sum_daily_service.run_sum_daily(None, local_date=_date(2026, 9, 14), models_config=object())

    assert built == ["armed"]
    assert captured["judge"] is judge


# ---------------------------------------------------------------------------
# the context budget (T23): a day that fits is untouched, a longer day is split
# ---------------------------------------------------------------------------

#: A message long enough that a day of them cannot fit one window: with the
#: prompt at ~2124 estimated tokens and the budget at 4096, twelve of these
#: (400 + the JSON envelope) overflow, while each one alone fits with room.
_LONG_TEXT = "Записала, как прошёл день, и что было сделано за это время. " * 7


def _long_entries(count: int) -> list[RawEntry]:
    return [
        _entry(
            text=f"{index:02d} {_LONG_TEXT}",
            entry_id=uuid.UUID(f"{index:08d}-1111-4111-8111-111111111111"),
        )
        for index in range(1, count + 1)
    ]


def test_a_day_that_fits_one_window_is_one_part_and_one_request() -> None:
    """The ordinary case must be exactly the single call it always was (T23 DoD 5)."""
    pipeline = SumDailyPipeline()
    entries = _entries()

    parts = pipeline.chunk_entries(entries, profile=_profile())

    assert parts == [entries]  # one part, the same objects in the same order
    # ...and that part's request is byte-for-byte the request of the old path.
    assert pipeline.build_request(entries=entries).messages[0].content == (
        pipeline.build_request(entries=parts[0]).messages[0].content
    )


def test_a_long_day_is_split_without_losing_or_duplicating_a_message() -> None:
    """The parts partition the day: same messages, same order, each part fits."""
    pipeline = SumDailyPipeline()
    entries = _long_entries(12)
    profile = _profile()

    parts = pipeline.chunk_entries(entries, profile=profile)

    assert len(parts) > 1, "the fixture must exceed one window"
    assert [entry.id for part in parts for entry in part] == [entry.id for entry in entries]
    for part in parts:
        request = pipeline.build_request(entries=part)
        assert pipeline.request_fits(request, profile=profile), len(part)


def test_the_budget_is_the_same_rule_as_the_weekly_layer() -> None:
    """One rule, two layers (T23 DoD 3): no drift between them or with the harness."""
    from diary.domain.pipelines.sum_weekly.pipeline import SumWeeklyPipeline

    profile = _profile()
    assert SumDailyPipeline.input_budget_tokens(profile) == NUM_CTX - profile.max_tokens
    assert SumDailyPipeline.input_budget_tokens(profile) == SumWeeklyPipeline.input_budget_tokens(
        profile
    )


def test_a_message_that_cannot_fit_alone_fails_loudly() -> None:
    """A day cannot be split through the middle of a message."""
    pipeline = SumDailyPipeline()
    huge = _entry(text="x" * 20_000)

    with pytest.raises(PipelinePermanentError, match="does not fit the context window"):
        pipeline.chunk_entries([huge], profile=_profile())


#: A day of voice messages that are not transcribed yet — T39's worst realistic
#: day. Every entry carries an empty ``text`` and the server-generated
#: ``audio_path`` (a second UUID-shaped string), so its ``raw_entry`` is the most
#: structural payload the layer can build.
_UNTRANSCRIBED_VOICE_DAY = 20


def _untranscribed_voice_entries(count: int = _UNTRANSCRIBED_VOICE_DAY) -> list[RawEntry]:
    return [
        RawEntry(
            id=uuid.UUID(f"{index:08d}-2222-4222-8222-222222222222"),
            ts=datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc),
            source="voice",
            text="",
            audio_path=f"audio/2026/09/{index:08d}-2222-4222-8222-222222222222.webm",
        )
        for index in range(1, count + 1)
    ]


def test_the_gate_refuses_the_call_the_prose_estimate_let_through() -> None:
    """T39: the prose coefficient accepted a call the endpoint measured over budget.

    Measured on 2026-09-22 with the ``diary-sum`` tag of this machine (the
    endpoint's own ``prompt_eval_count``; the payload and the whole call are the
    very strings asserted below): this day's **payload** is 3640 characters and
    measured **2602** tokens, its **call** is 9010 characters and measured
    **4543**, and the prose estimate priced that call at **3604** against the 4096
    budget — so the gate sent a request that could not fit the window
    (4543 + 4096 > 8192), exactly the silent-truncation class T23 was built
    against. The structural estimate prices the same call at **4981**, and the day
    is split so that every call it makes fits.
    """
    pipeline = SumDailyPipeline()
    profile = _profile()
    entries = _untranscribed_voice_entries()
    request = pipeline.build_request(entries=entries)
    content = request.messages[0].content
    payload = content.split(f"\n\n{ENTRIES_HEADER}\n", 1)[1]
    budget = SumDailyPipeline.input_budget_tokens(profile)

    assert len(payload) == 3640, "the measured payload"
    assert len(content) == 9010, "the measured call"
    assert estimate_request_tokens(payload) >= 2602  # above the measured payload cost
    assert 4543 + profile.max_tokens > NUM_CTX  # the measured call did not fit the window
    assert estimate_tokens(len(content)) == 3604 <= budget  # the old gate accepted it anyway
    assert not pipeline.request_fits(request, profile=profile)  # the gate does not

    parts = pipeline.chunk_entries(entries, profile=profile)
    assert len(parts) > 1, "the oversized call must be split, not sent"
    assert [entry.id for part in parts for entry in part] == [entry.id for entry in entries]
    for part in parts:
        assert pipeline.request_fits(pipeline.build_request(entries=part), profile=profile), len(
            part
        )


def test_the_payload_markers_stay_out_of_the_prompt_templates() -> None:
    """The structural split (T39) is exact only while the prompts stay out of the way.

    The gate finds a request's JSON payload by the header line the layer itself
    wrote, so a prompt carrying that line would be charged as JSON — and, worse,
    split at the wrong place. Both versioned prompts are pinned here: the
    assumption is a property of the files, not of a comment.
    """
    for prompt in (SumDailyPipeline().load_prompt(), SumDailyPipeline().load_merge_prompt()):
        for header in (ENTRIES_HEADER, PARTS_HEADER):
            assert header not in prompt


def test_the_merge_request_carries_the_parts_content_fields_in_order() -> None:
    pipeline = SumDailyPipeline()
    parts = [
        DailySummary(
            date=DAY, facts=["Первое."], themes=["работа"], energy=None, open_questions=[]
        ),
        DailySummary(
            date=DAY, facts=["Второе."], themes=[], energy="low", open_questions=["Вопрос?"]
        ),
    ]

    request = pipeline.build_merge_request(parts, local_date=DAY)

    assert request.prompt_version == MERGE_PROMPT_VERSION
    content = request.messages[0].content
    assert PARTS_HEADER in content
    assert content.index("Первое.") < content.index("Второе.")
    payload = json.loads(content.split(PARTS_HEADER, 1)[1].strip())
    assert payload == [
        {"facts": ["Первое."], "themes": ["работа"], "energy": None, "open_questions": []},
        {"facts": ["Второе."], "themes": [], "energy": "low", "open_questions": ["Вопрос?"]},
    ]
    # The anchor is the backend's: the parts do not carry a date into the prompt.
    assert "2026-09-14" not in content


def test_a_part_of_another_day_cannot_be_folded_back() -> None:
    pipeline = SumDailyPipeline()
    foreign = DailySummary(date=date(2026, 9, 13), facts=["Чужой день."])

    with pytest.raises(PipelinePermanentError, match="another day"):
        pipeline.build_merge_request([foreign], local_date=DAY)


def test_an_empty_part_list_fails_loudly() -> None:
    with pytest.raises(PipelinePermanentError, match="no partial summaries"):
        SumDailyPipeline().build_merge_request([], local_date=DAY)


def test_a_merge_that_does_not_fit_fails_loudly() -> None:
    """The merge input is bounded by the parts; an unfittable one is not sent."""
    pipeline = SumDailyPipeline()
    profile = _profile()
    parts = [
        DailySummary(date=DAY, facts=[_LONG_TEXT * 2], themes=[], energy=None, open_questions=[])
        for _ in range(20)
    ]

    request = pipeline.build_merge_request(parts, local_date=DAY)

    assert not pipeline.request_fits(request, profile=profile)
    with pytest.raises(PipelinePermanentError, match="do not fit one context window"):
        pipeline.require_merge_fits(request, profile=profile)


def test_the_merge_prompt_is_v2_and_states_the_four_fields_and_energy_rule() -> None:
    assert MERGE_PROMPT_VERSION == "v2"
    assert MERGE_PROMPT_PATH.is_file()
    text = MERGE_PROMPT_PATH.read_text(encoding="utf-8")
    for field in ("facts", "themes", "energy", "open_questions"):
        assert field in text
    # The energy rule is the layer's veto, so the merge must restate it: null
    # unless the parts name the level, and no indirect inference.
    assert "null" in text
    assert "энергии" in text
    # The merge must not become a place where advice/сausality sneak in.
    assert "совет" in text
    assert "причин" in text


class _ScriptedProvider:
    """A provider answering call by call, so the chunked path is observable."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("the layer made more calls than the script provides")
        return self._responses.pop(0)


def test_a_long_day_is_summarized_in_parts_and_then_folded_back_once(
    store: _RecordingStore,
) -> None:
    """End to end: parts get the day prompt, the merge gets the merge prompt.

    What is stored is still one validated :class:`DailySummary` (T23 DoD 4): the
    split changes how the model got there, not the shape or the anchor.
    """
    entries = _long_entries(12)
    part_payload = {"facts": ["Часть дня."], "themes": [], "energy": None}
    merged_payload = {
        "facts": ["Первая часть дня.", "Вторая часть дня."],
        "themes": ["день"],
        "energy": None,
    }
    provider = _ScriptedProvider(
        [
            _json_response(part_payload),
            _json_response(part_payload),
            _json_response(merged_payload),
        ]
    )

    outcome = sum_daily_service.sum_daily(
        None,  # type: ignore[arg-type]
        local_date=DAY,
        profile=_profile(),
        provider=provider,
        local_tz=LOCAL_TZ,
        entries=entries,
    )

    assert len(provider.requests) == 3, "two parts and one merge"
    assert [request.prompt_version for request in provider.requests] == [
        PROMPT_VERSION,
        PROMPT_VERSION,
        MERGE_PROMPT_VERSION,
    ]
    assert outcome.summary.facts == ["Первая часть дня.", "Вторая часть дня."]
    assert outcome.summary.date == DAY
    # One stored artifact, under the day anchor: chunking does not multiply rows.
    assert len(store.calls) == 1
    assert store.calls[0]["result"] is outcome.summary


def test_an_unfittable_merge_is_refused_before_it_is_sent(store: _RecordingStore) -> None:
    """The merge has its own budget check, and it must be the last call (T23 DoD 4).

    The parts partition the day, so they fit by construction; what does not fit by
    construction is their *folded* text, which is bounded only by the generation cap.
    A review of T23 found this check pinned by no test: deleting it left the suite
    green, so a day whose parts cannot be folded back would have been sent anyway and
    silently truncated. Here the parts are large enough that the merge cannot fit —
    the layer must fail loudly and make no merge call at all.
    """
    entries = _long_entries(12)
    huge_part = {
        "facts": ["Запись дня, перечисленная подробно." * 12 for _ in range(30)],
        "themes": [],
        "energy": None,
    }
    provider = _ScriptedProvider([_json_response(huge_part) for _ in range(8)])

    with pytest.raises(PipelinePermanentError, match="do not fit one context window"):
        sum_daily_service.sum_daily(
            None,  # type: ignore[arg-type]
            local_date=DAY,
            profile=_profile(),
            provider=provider,
            local_tz=LOCAL_TZ,
            entries=entries,
        )

    # Every part was summarized; the merge was refused *before* being sent, so the
    # provider saw one call per part and nothing else.
    assert len(provider.requests) > 1
    assert all(request.prompt_version == PROMPT_VERSION for request in provider.requests)
    assert store.calls == []


def test_a_day_that_fits_still_makes_exactly_one_call(store: _RecordingStore) -> None:
    provider = _ScriptedProvider([_json_response(_VALID_PAYLOAD)])

    sum_daily_service.sum_daily(
        None,  # type: ignore[arg-type]
        local_date=DAY,
        profile=_profile(),
        provider=provider,
        local_tz=LOCAL_TZ,
        entries=_entries(),
    )

    assert len(provider.requests) == 1
    assert provider.requests[0].prompt_version == PROMPT_VERSION
