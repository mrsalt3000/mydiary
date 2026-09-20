"""Unit tests for the T06 weekly-summary layer (no provider, no database).

Covers the definition of done of task T06:

* a valid completion becomes a validated :class:`WeeklySummary` and is handed to
  the append-only store exactly once, under the ``week_id`` anchor;
* the input is the week's **daily summaries** — the model receives the
  ``daily_summary[]`` array in day order and nothing else;
* the envelope fields ``week_id`` and ``daily_ids`` come from the backend: a
  completion that reports its own week or day list cannot change the stored
  anchor or the recorded days;
* the result is strictly the contract's fields — ``rollup`` plus the backend
  envelope; an extra field (advice, evaluation) fails the run and stores
  nothing;
* an invalid completion (non-JSON, missing/empty ``rollup``, extra field) raises
  a loud, non-retryable LLM error and stores **nothing** — no partial artifact;
* the context-budget chunker partitions the days in order with no day lost and
  none duplicated, and a chunked run reconciles the partial rollups through one
  merge call into a single ``rollup`` (its stored ``daily_ids`` are still every
  day that was fed);
* a week without daily summaries, or with a day of another week, fails loudly
  before any request is sent;
* the week's day summaries are not modified, and the layer has no other write
  path.

The provider is a fake and the store is a recording double, so no endpoint,
configuration file, model or database is touched. All content is synthetic
(AGENTS.md §13).
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Any, Protocol

import pytest

from diary.core.models_config import ModelsConfig, OpenAICompatibleProfile, load_models_config
from diary.core.settings import PROJECT_ROOT
from diary.db.models import ArtifactLayer, PipelineType
from diary.domain.contracts import DailySummary, WeeklySummary
from diary.domain.contracts.golden import estimate_tokens
from diary.domain.llm.errors import LLMProviderUnavailableError, LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines import naming
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.base import PipelinePermanentError
from diary.domain.pipelines.sum_weekly import service as sum_weekly_service
from diary.domain.pipelines.sum_weekly.pipeline import (
    MERGE_HEADER,
    NUM_CTX,
    PROMPT_PATH,
    PROMPT_VERSION,
    SUMMARIES_HEADER,
    SumWeeklyPipeline,
)
from diary.domain.pipelines.sum_weekly.registration import ensure_sum_weekly_pipeline_registered
from diary.domain.pipelines.sum_weekly.week_input import (
    WeeklyInputError,
    week_days,
)

WEEK_ID = "2026-W38"
DAYS = [day.isoformat() for day in week_days(WEEK_ID)]
FACTS = [f"Синтетический факт дня {day}." for day in DAYS]
ROLLUP = "Неделя прошла вокруг отчёта и бассейна; сон сбился к середине недели."

#: Distinct markers for the partial rollups, spelled as words: a partial naming
#: a quantity its chunk does not contain is refused by the runtime gate (T17b),
#: and a week never needs more chunks than it has days.
PARTIAL_MARKERS = (
    "первая",
    "вторая",
    "третья",
    "четвёртая",
    "пятая",
    "шестая",
    "седьмая",
)


def _summaries() -> list[DailySummary]:
    return [
        DailySummary(date=date.fromisoformat(day), facts=[fact])
        for day, fact in zip(DAYS, FACTS, strict=True)
    ]


def _profile(max_tokens: int = 4096, model: str = "diary-sum") -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:1/v1",  # never contacted by the fake
        model=model,
        api_key_env="SYNTHETIC_PHASE2_KEY",
        temperature=0.25,
        max_tokens=max_tokens,
    )


class _Provider(Protocol):
    """Minimal provider shape the layer uses (a fake records the requests)."""

    requests: list[LLMRequest]

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class _FakeProvider:
    """Deterministic provider: a queue of fixed responses (records requests)."""

    def __init__(self, *responses: LLMResponse) -> None:
        self.queue = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.queue:
            raise AssertionError("the run made more provider calls than the test provided")
        return self.queue.pop(0)


class _ChunkAwareProvider:
    """Answers chunk calls with fresh partial rollups and the merge with the final one.

    The chunk count is decided by the layer (not by the test), so the fake has
    to react to the request kind instead of holding a fixed response queue.
    """

    def __init__(self, *, final_rollup: str, partial_prefix: str) -> None:
        self.final_rollup = final_rollup
        self.partial_prefix = partial_prefix
        self.requests: list[LLMRequest] = []
        self.rollup_calls = 0
        self.merge_calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if MERGE_HEADER in request.messages[0].content:
            self.merge_calls += 1
            return _rollup_response(self.final_rollup)
        index = self.rollup_calls
        self.rollup_calls += 1
        return _rollup_response(f"{self.partial_prefix} {PARTIAL_MARKERS[index]}.")


def _json_response(payload: object) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload, ensure_ascii=False), model="чужой-модель")


def _rollup_response(text: str = ROLLUP) -> LLMResponse:
    return _json_response({"rollup": text})


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
    monkeypatch.setattr(sum_weekly_service, "save_layer_result", recording)
    return recording


def _run(
    store: _RecordingStore,
    provider: _Provider,
    *,
    week_id: str = WEEK_ID,
    summaries: list[DailySummary] | None = None,
    profile: OpenAICompatibleProfile | None = None,
) -> sum_weekly_service.SumWeeklyOutcome:
    return sum_weekly_service.sum_weekly(
        None,  # type: ignore[arg-type]  # the store double ignores the session
        week_id=week_id,
        profile=profile or _profile(),
        provider=provider,
        local_tz="Europe/Moscow",
        summaries=summaries if summaries is not None else _summaries(),
    )


def _request_content(request: LLMRequest) -> str:
    return request.messages[0].content


def _payload_days(content: str) -> list[str]:
    """The ``date`` of every daily summary in a rollup request, in order."""
    _, separator, payload = content.partition(f"{SUMMARIES_HEADER}\n")
    assert separator, "the request must carry the week's daily summaries"
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    return [entry["date"] for entry in parsed]


def _merge_partials(content: str) -> list[dict[str, object]]:
    """The partial rollups of a merge request, in order."""
    _, separator, payload = content.partition(f"{MERGE_HEADER}\n")
    assert separator, "the merge request must carry the partial rollups"
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    return parsed


def _content_tokens(summaries: list[DailySummary]) -> int:
    """Token estimate of the rollup request content for exactly ``summaries``."""
    content = SumWeeklyPipeline().build_request(summaries=summaries).messages[0].content
    return estimate_tokens(len(content))


def _profile_for_budget(tokens: int) -> OpenAICompatibleProfile:
    """A profile whose input budget is exactly ``tokens`` tokens."""
    return _profile(max_tokens=NUM_CTX - tokens)


# ---------------------------------------------------------------------------
# prompt template
# ---------------------------------------------------------------------------


def test_prompt_template_is_v2_and_forbids_advice_and_diagnoses() -> None:
    assert PROMPT_VERSION == "v2"
    assert PROMPT_PATH.is_file()
    text = PROMPT_PATH.read_text(encoding="utf-8")
    # The single output field and the merge mode are explicit.
    assert "rollup" in text
    assert "week_id" in text and "daily_ids" in text
    assert "JSON" in text
    # Advice/morality and diagnoses are vetoes of this layer.
    assert "совет" in text
    assert "морал" in text
    assert "диагноз" in text
    assert "лечение" in text
    # The one-field answer shape must be stated.
    assert '{"rollup"' in text.replace(" ", "").replace("\n", "")


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------


def test_layer_is_registered_in_the_second_family_registry() -> None:
    assert ensure_sum_weekly_pipeline_registered() is True
    assert ensure_sum_weekly_pipeline_registered() is False  # idempotent
    registered = pipeline_registry.get_second_family_pipeline(PipelineType.weekly_summary)
    assert isinstance(registered, SumWeeklyPipeline)
    assert registered.layer is ArtifactLayer.weekly_summary
    assert registered.pipeline_type is PipelineType.weekly_summary
    assert registered.profile_name == "diary_week"
    assert "weekly_summary" in pipeline_registry.known_pipeline_types()
    # The export-driven lookup must not return a per-week layer.
    assert pipeline_registry.get_pipeline(PipelineType.weekly_summary) is None


def test_registry_rejects_a_misdeclared_layer() -> None:
    class _WrongLayer(SumWeeklyPipeline):
        layer = ArtifactLayer.daily_summary

    with pytest.raises(ValueError, match="must declare layer"):
        pipeline_registry.register(_WrongLayer())


def test_registry_rejects_a_misdeclared_profile_name() -> None:
    class _WrongProfile(SumWeeklyPipeline):
        profile_name = "diary_stt"

    with pytest.raises(ValueError, match="must declare profile name"):
        pipeline_registry.register(_WrongProfile())


def test_naming_maps_know_the_weekly_summary_layer() -> None:
    """The layer is mapped by profile name, not by a TaskType (T02/T04 rule).

    Since T13b the weekly layer has its **own** profile (``diary_week``) instead
    of sharing the day's ``diary_sum``: that is what makes "day on the GPU, week
    on the CPU" expressible in the configuration (``docs/device-choice.md`` §11).
    """
    assert naming.PIPELINE_PROFILE_NAME[PipelineType.weekly_summary] == "diary_week"
    # The two sum layers no longer share a profile; the day layer keeps its own.
    assert naming.PIPELINE_PROFILE_NAME[PipelineType.daily_summary] == "diary_sum"
    assert PipelineType.weekly_summary not in naming.PIPELINE_TASK_TYPE
    assert naming.layer_output_extension(ArtifactLayer.weekly_summary) == ".md"
    # The layer renders no file, so it has no per-layer artifact directory.
    with pytest.raises(ValueError):
        naming.layer_dirname(ArtifactLayer.weekly_summary)


# ---------------------------------------------------------------------------
# profile resolution (T13b: day on the GPU, week on the CPU)
# ---------------------------------------------------------------------------


def _example_config() -> ModelsConfig:
    """The tracked hybrid example configuration (no endpoint is contacted)."""
    return load_models_config(PROJECT_ROOT / "configs" / "models.example.yaml")


def test_run_sum_weekly_resolves_the_week_profile_not_the_day_one(
    store: _RecordingStore,
) -> None:
    """The layer takes ``diary_week`` (the CPU twin), never ``diary_sum``."""
    config = _example_config()
    provider = _FakeProvider(_rollup_response())

    outcome = sum_weekly_service.run_sum_weekly(
        None,  # type: ignore[arg-type]  # the store double ignores the session
        week_id=WEEK_ID,
        models_config=config,
        provider=provider,
        local_tz="Europe/Moscow",
        summaries=_summaries(),
    )

    assert outcome.summary.rollup == ROLLUP
    assert config.diary_week is not None and config.diary_week.model == "diary-sum-cpu"
    assert config.diary_sum is not None and config.diary_sum.model == "diary-sum"


def test_run_sum_weekly_is_switched_off_without_the_week_profile(
    store: _RecordingStore,
) -> None:
    """(b) Backward compatibility: a pre-T13b config leaves the layer off.

    An absent ``diary_week`` must be "the layer is switched off" — a
    non-retryable configuration error raised before any request, not a crash
    and not a silent fallback to ``diary_sum`` (which would quietly return the
    weekly rollup to the GPU).
    """
    config = _example_config().model_copy(update={"diary_week": None})
    provider = _FakeProvider(_rollup_response())

    with pytest.raises(LLMProviderUnavailableError) as exc:
        sum_weekly_service.run_sum_weekly(
            None,  # type: ignore[arg-type]
            week_id=WEEK_ID,
            models_config=config,
            provider=provider,
            local_tz="Europe/Moscow",
            summaries=_summaries(),
        )

    assert "diary_week" in str(exc.value)
    assert "switched off" in str(exc.value)
    assert provider.requests == []
    assert store.calls == []


# ---------------------------------------------------------------------------
# request building
# ---------------------------------------------------------------------------


def test_request_carries_the_week_summaries_in_day_order() -> None:
    request = SumWeeklyPipeline().build_request(summaries=_summaries())

    assert request.prompt_version == PROMPT_VERSION
    content = _request_content(request)
    assert SUMMARIES_HEADER in content
    assert _payload_days(content) == DAYS
    # The day summaries are handed over as the daily_summary JSON array.
    for summary in _summaries():
        assert json.dumps(summary.model_dump(mode="json"), ensure_ascii=False) in content


def test_the_request_never_mentions_a_raw_source_field() -> None:
    """The weekly layer's input is the day summaries, not raw entries."""
    content = _request_content(SumWeeklyPipeline().build_request(summaries=_summaries()))

    assert "raw_entry" not in content
    assert "transcription" not in content
    assert "audio_path" not in content


def test_a_week_without_summaries_fails_loudly_before_the_provider(
    store: _RecordingStore,
) -> None:
    provider = _FakeProvider(_rollup_response())

    with pytest.raises(WeeklyInputError):
        _run(store, provider, summaries=[])

    assert provider.requests == []
    assert store.calls == []


def test_a_summary_of_another_week_fails_loudly_before_the_provider(
    store: _RecordingStore,
) -> None:
    provider = _FakeProvider(_rollup_response())
    foreign = [*_summaries(), DailySummary(date=date(2026, 9, 21), facts=[FACTS[0]])]

    with pytest.raises(WeeklyInputError):
        _run(store, provider, summaries=foreign)

    assert provider.requests == []
    assert store.calls == []


@pytest.mark.parametrize("week_id", ["2026-W54", "not-a-week"])
def test_a_malformed_week_id_fails_loudly_before_the_provider(
    store: _RecordingStore, week_id: str
) -> None:
    provider = _FakeProvider(_rollup_response())

    with pytest.raises(WeeklyInputError):
        _run(store, provider, week_id=week_id)

    assert provider.requests == []
    assert store.calls == []


# ---------------------------------------------------------------------------
# successful run
# ---------------------------------------------------------------------------


def test_valid_response_yields_a_summary_and_is_stored_once(
    store: _RecordingStore,
) -> None:
    provider = _FakeProvider(_rollup_response())

    outcome = _run(store, provider)

    assert outcome.summary == WeeklySummary(week_id=WEEK_ID, daily_ids=DAYS, rollup=ROLLUP)
    assert len(provider.requests) == 1
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["layer"] is ArtifactLayer.weekly_summary
    assert call["week_id"] == WEEK_ID  # the ISO-week anchor, not a weekly export
    assert call["result"] is outcome.summary
    # No rendered file: the layer's result is stored inline (ADR-008 §1).
    assert "path" not in call or call.get("path") is None
    # The week anchor is the only anchor: the two families never mix.
    assert "raw_id" not in call and "local_date" not in call and "summary_id" not in call


def test_the_daily_ids_are_the_days_that_were_fed(store: _RecordingStore) -> None:
    summaries = _summaries()[:3]  # a partially summarized week
    outcome = _run(store, _FakeProvider(_rollup_response()), summaries=summaries)

    assert outcome.summary.daily_ids == [summary.date.isoformat() for summary in summaries]
    assert store.calls[0]["result"].daily_ids == [summary.date.isoformat() for summary in summaries]


def test_the_envelope_fields_come_from_the_backend_not_the_answer(
    store: _RecordingStore,
) -> None:
    """A completion claiming another week or day list cannot change the record."""
    payload = {
        "rollup": ROLLUP,
        "week_id": "1999-W01",
        "daily_ids": ["1999-01-01"],
    }
    outcome = _run(store, _FakeProvider(_json_response(payload)))

    assert outcome.summary.week_id == WEEK_ID
    assert outcome.summary.daily_ids == DAYS
    stored = store.calls[0]["result"]
    assert stored.week_id == WEEK_ID
    assert stored.daily_ids == DAYS


# ---------------------------------------------------------------------------
# failed runs: loud, and without any stored result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_text",
    [
        "Не JSON вовсе.",
        '```json\n{"rollup": "Синтетика."}\n```',
        '[{"rollup": "Синтетика."}]',
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
        {},  # missing rollup
        {"rollup": ""},  # empty rollup
        {"rollup": "Текст.", "advice": "Отдыхай больше."},  # advice the contract forbids
        {"rollup": "Текст.", "facts": ["Чужой факт."]},  # the day layer's fields
    ],
)
def test_invalid_contract_fails_loudly_and_stores_nothing(
    store: _RecordingStore, payload: dict[str, object]
) -> None:
    with pytest.raises(LLMStructuredOutputError):
        _run(store, _FakeProvider(_json_response(payload)))

    assert store.calls == []


# ---------------------------------------------------------------------------
# chunking: no day lost, none duplicated
# ---------------------------------------------------------------------------


def test_one_chunk_when_the_whole_week_fits_the_budget() -> None:
    summaries = _summaries()

    chunks = SumWeeklyPipeline().chunk_summaries(summaries, profile=_profile())

    assert chunks == [summaries]


def test_the_chunker_partitions_in_order_without_loss_or_duplication() -> None:
    summaries = _summaries()
    full_budget = _content_tokens(summaries)
    pipeline = SumWeeklyPipeline()

    for budget in (full_budget - 1, _content_tokens(summaries[:2]), _content_tokens(summaries[:1])):
        chunks = pipeline.chunk_summaries(summaries, profile=_profile_for_budget(budget))
        flat = [summary for chunk in chunks for summary in chunk]
        assert flat == summaries  # same order, no day lost, none duplicated
        assert len(chunks) > 1
        # Every chunk is itself non-empty and fits the budget.
        for chunk in chunks:
            assert chunk
            request = pipeline.build_request(summaries=chunk)
            assert pipeline.request_fits(request, profile=_profile_for_budget(budget))


def test_a_chunk_boundary_does_not_drop_the_day_that_exactly_fits() -> None:
    summaries = _summaries()
    # A budget that fits exactly the first two days: the second day must stay in
    # the first chunk and the third must start the next one.
    budget = _content_tokens(summaries[:2])

    chunks = SumWeeklyPipeline().chunk_summaries(summaries, profile=_profile_for_budget(budget))

    assert len(chunks[0]) >= 2
    assert [summary.date for summary in chunks[0][:2]] == [
        summary.date for summary in summaries[:2]
    ]
    assert chunks[0][-1].date < chunks[1][0].date  # ordered, non-overlapping


def test_a_day_that_cannot_fit_alone_fails_loudly(store: _RecordingStore) -> None:
    provider = _FakeProvider(_rollup_response())

    with pytest.raises(PipelinePermanentError):
        _run(store, provider, profile=_profile_for_budget(0))

    assert provider.requests == []
    assert store.calls == []


def test_a_chunked_run_reconciles_the_partials_with_one_merge_call(
    store: _RecordingStore,
) -> None:
    summaries = _summaries()
    # One token short of the whole week: the week splits into ordered chunks.
    budget = _content_tokens(summaries) - 1
    provider = _ChunkAwareProvider(final_rollup=ROLLUP, partial_prefix="Частичная сводка")

    outcome = _run(store, provider, profile=_profile_for_budget(budget))

    # The week needed more than one piece: chunk calls plus exactly one merge.
    assert provider.rollup_calls > 1
    assert provider.merge_calls == 1
    assert len(provider.requests) == provider.rollup_calls + 1
    rollup_requests = provider.requests[:-1]
    merge_request = provider.requests[-1]
    assert all(SUMMARIES_HEADER in _request_content(request) for request in rollup_requests)
    assert MERGE_HEADER in _request_content(merge_request)
    # Every day was fed exactly once across the chunk calls, in order.
    fed = [day for request in rollup_requests for day in _payload_days(_request_content(request))]
    assert fed == DAYS
    # The merge input is the ordered partial rollups — not the raw day summaries.
    partials = _merge_partials(_request_content(merge_request))
    assert [partial["rollup"] for partial in partials] == [
        f"Частичная сводка {PARTIAL_MARKERS[index]}." for index in range(provider.rollup_calls)
    ]
    assert [item for partial in partials for item in partial["daily_ids"]] == DAYS
    # The stored result covers every day that was fed and is the merged rollup.
    assert outcome.summary.rollup == ROLLUP
    assert outcome.summary.daily_ids == DAYS
    assert store.calls[0]["result"].daily_ids == DAYS


def test_a_merge_that_cannot_fit_fails_loudly_before_storing(
    store: _RecordingStore,
) -> None:
    summaries = _summaries()
    # A budget that fits one day per chunk: seven chunk calls, then a merge whose
    # input is the seven partial rollups.
    budget = _content_tokens(summaries[:1])
    long_partial = "Синтетическая частичная сводка недели. " * 40
    provider = _FakeProvider(*[_rollup_response(long_partial) for _ in range(7)])

    with pytest.raises(PipelinePermanentError):
        _run(store, provider, profile=_profile_for_budget(budget))

    # The failure happens before the merge call and before any write.
    assert len(provider.requests) == 7
    assert store.calls == []


# ---------------------------------------------------------------------------
# the input is not modified
# ---------------------------------------------------------------------------


def test_the_run_does_not_change_the_week_summaries(store: _RecordingStore) -> None:
    summaries = _summaries()
    before = [summary.model_dump() for summary in summaries]

    _run(store, _FakeProvider(_rollup_response()), summaries=summaries)

    assert [summary.model_dump() for summary in summaries] == before
    # The only write performed by the layer is the append-only result store.
    assert [call["layer"] for call in store.calls] == [ArtifactLayer.weekly_summary]


def test_the_layer_declares_no_rendered_file_location() -> None:
    """The layer's result is inline; it has no per-layer artifact directory."""
    assert PROMPT_PATH.parent.name == "sum_weekly"
    assert PROMPT_PATH.name == "v2.md"
    assert SumWeeklyPipeline.output_ext == ".md"


def test_the_budget_is_the_context_window_minus_the_generation_cap() -> None:
    assert SumWeeklyPipeline.input_budget_tokens(_profile(max_tokens=4096)) == NUM_CTX - 4096
    # A cap that leaves no room makes every request overflow (loudly).
    assert SumWeeklyPipeline.input_budget_tokens(_profile(max_tokens=NUM_CTX)) == 0


def test_created_at_and_run_are_forwarded_to_the_store(store: _RecordingStore) -> None:
    created_at = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    run_id = uuid.uuid4()

    sum_weekly_service.sum_weekly(
        None,  # type: ignore[arg-type]
        week_id=WEEK_ID,
        profile=_profile(),
        provider=_FakeProvider(_rollup_response()),
        local_tz="Europe/Moscow",
        summaries=_summaries(),
        processing_run_id=run_id,
        created_at=created_at,
    )

    assert store.calls[0]["processing_run_id"] == run_id
    assert store.calls[0]["created_at"] == created_at
