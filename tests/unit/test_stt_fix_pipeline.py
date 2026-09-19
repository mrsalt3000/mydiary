"""Unit tests for the T04 STT-correction layer (no provider, no database).

Covers the definition of done of task T04:

* a valid completion becomes a validated :class:`SttCorrection` and is handed to
  the append-only store exactly once;
* an invalid completion (non-JSON, missing/extra field, non-verbatim
  ``uncertain`` fragment) raises a loud, non-retryable LLM error and stores
  **nothing** — no partial artifact;
* the envelope fields (``raw_id``/``model``/``prompt_ver``) come from the
  backend, never from the model's answer;
* doubt is expressed only through ``uncertain[]``: the corrected ``text`` is
  returned exactly as the model produced it, with no marker added by the layer;
* the input entry is not modified, and the layer has no other write path.

The provider is a fake and the store is a recording double, so no endpoint,
configuration file, model or database is touched. All content is synthetic
(AGENTS.md §13).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from diary.core.models_config import OpenAICompatibleProfile
from diary.db.models import ArtifactLayer, PipelineType
from diary.domain.contracts import RawEntry, SttCorrection
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines import naming
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.stt_fix import service as stt_fix_service
from diary.domain.pipelines.stt_fix.pipeline import (
    PROMPT_PATH,
    PROMPT_VERSION,
    SttFixPipeline,
)
from diary.domain.pipelines.stt_fix.registration import ensure_stt_fix_pipeline_registered

ENTRY_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
RAW_TEXT = "сегодня утром я превет какдила ходил в парк"
CORRECTED_TEXT = "Сегодня утром я привет как дела, ходил в парк."


def _entry(*, text: str = RAW_TEXT) -> RawEntry:
    return RawEntry(
        id=ENTRY_ID,
        ts=datetime(2026, 9, 19, 9, 30, tzinfo=timezone.utc),
        source="text",
        text=text,
    )


def _profile(model: str = "diary-stt") -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:11434/v1",
        model=model,
        api_key_env="SYNTHETIC_PHASE2_KEY",
        temperature=0.15,
        max_tokens=3072,
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
    monkeypatch.setattr(stt_fix_service, "save_layer_result", recording)
    return recording


# ---------------------------------------------------------------------------
# prompt template
# ---------------------------------------------------------------------------


def test_prompt_template_is_v1_and_contains_no_doubt_marker() -> None:
    assert PROMPT_VERSION == "v1"
    assert PROMPT_PATH.is_file()
    text = PROMPT_PATH.read_text(encoding="utf-8")
    # Doubt is expressed only through ``uncertain``: the prompt must not
    # introduce a marker literal at all (ADR-007 §3).
    assert "[?]" not in text
    assert "uncertain" in text
    assert "JSON" in text


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------


def test_layer_is_registered_in_the_second_family_registry() -> None:
    assert ensure_stt_fix_pipeline_registered() is True
    assert ensure_stt_fix_pipeline_registered() is False  # idempotent
    registered = pipeline_registry.get_second_family_pipeline(PipelineType.stt_corrected)
    assert isinstance(registered, SttFixPipeline)
    assert registered.layer is ArtifactLayer.stt_corrected
    assert registered.pipeline_type is PipelineType.stt_corrected
    assert registered.profile_name == "diary_stt"
    assert "stt_corrected" in pipeline_registry.known_pipeline_types()
    # The export-driven lookup must not return a per-message layer.
    assert pipeline_registry.get_pipeline(PipelineType.stt_corrected) is None


def test_registry_rejects_a_misdeclared_layer() -> None:
    class _WrongLayer(SttFixPipeline):
        layer = ArtifactLayer.normalized

    with pytest.raises(ValueError, match="must declare layer"):
        pipeline_registry.register(_WrongLayer())


def test_registry_rejects_a_misdeclared_profile_name() -> None:
    class _WrongProfile(SttFixPipeline):
        profile_name = "diary_coach"

    with pytest.raises(ValueError, match="must declare profile name"):
        pipeline_registry.register(_WrongProfile())


# ---------------------------------------------------------------------------
# request building
# ---------------------------------------------------------------------------


def test_request_carries_the_entry_text_and_the_prompt_version() -> None:
    request = SttFixPipeline().build_request(entry=_entry())
    assert request.prompt_version == PROMPT_VERSION
    content = request.messages[0].content
    assert json.dumps({"text": RAW_TEXT}, ensure_ascii=False) in content
    # The layer adds no marker of its own to the request or the entry.
    assert "[?]" not in content


# ---------------------------------------------------------------------------
# successful run
# ---------------------------------------------------------------------------


def test_valid_response_yields_a_correction_and_is_stored_once(
    store: _RecordingStore,
) -> None:
    entry = _entry()
    provider = _FakeProvider(
        _json_response({"text": CORRECTED_TEXT, "uncertain": []}),
    )

    outcome = stt_fix_service.correct_stt(
        None,  # type: ignore[arg-type]  # the store double ignores the session
        entry=entry,
        profile=_profile(),
        provider=provider,
    )

    assert outcome.correction == SttCorrection(
        raw_id=ENTRY_ID,
        text=CORRECTED_TEXT,
        uncertain=[],
        model="diary-stt",
        prompt_ver="v1",
    )
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["layer"] is ArtifactLayer.stt_corrected
    assert call["raw_id"] == ENTRY_ID
    assert call["result"] is outcome.correction
    # No rendered file: the layer's result is stored inline (ADR-008 §1).
    assert "path" not in call or call.get("path") is None
    assert len(provider.requests) == 1


def test_envelope_fields_come_from_the_backend_not_the_answer(
    store: _RecordingStore,
) -> None:
    """A completion claiming its own model/prompt version cannot change them."""
    provider = _FakeProvider(
        _json_response(
            {
                "text": CORRECTED_TEXT,
                "uncertain": [],
                "model": "чужой-модель",
                "prompt_ver": "v99",
            }
        )
    )

    outcome = stt_fix_service.correct_stt(
        None,  # type: ignore[arg-type]
        entry=_entry(),
        profile=_profile(model="diary-stt"),
        provider=provider,
    )

    assert outcome.correction.model == "diary-stt"
    assert outcome.correction.prompt_ver == "v1"
    stored = store.calls[0]["result"]
    assert stored.model == "diary-stt"
    assert stored.prompt_ver == "v1"


def test_doubt_is_expressed_only_through_uncertain(store: _RecordingStore) -> None:
    uncertain_fragment = "три тыщи восемьсот"
    text = f"Вечером позвоню в банк, {uncertain_fragment}."
    provider = _FakeProvider(_json_response({"text": text, "uncertain": [uncertain_fragment]}))

    outcome = stt_fix_service.correct_stt(
        None,  # type: ignore[arg-type]
        entry=_entry(),
        profile=_profile(),
        provider=provider,
    )

    # The text is exactly what the model produced: the layer adds no marker.
    assert outcome.correction.text == text
    assert outcome.correction.uncertain == [uncertain_fragment]
    assert all(fragment in outcome.correction.text for fragment in outcome.correction.uncertain)
    assert "[?]" not in outcome.correction.text


def test_a_text_without_doubt_has_an_empty_uncertain_list(store: _RecordingStore) -> None:
    provider = _FakeProvider(_json_response({"text": CORRECTED_TEXT, "uncertain": []}))
    outcome = stt_fix_service.correct_stt(
        None,  # type: ignore[arg-type]
        entry=_entry(),
        profile=_profile(),
        provider=provider,
    )
    assert outcome.correction.uncertain == []


# ---------------------------------------------------------------------------
# failed runs: loud, and without any stored result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_text",
    [
        "Не JSON вовсе.",
        '```json\n{"text": "Синтетика.", "uncertain": []}\n```',
        '[{"text": "Синтетика."}]',
    ],
)
def test_invalid_json_fails_loudly_and_stores_nothing(
    store: _RecordingStore, raw_text: str
) -> None:
    provider = _FakeProvider(LLMResponse(text=raw_text))

    with pytest.raises(LLMStructuredOutputError):
        stt_fix_service.correct_stt(
            None,  # type: ignore[arg-type]
            entry=_entry(),
            profile=_profile(),
            provider=provider,
        )

    assert store.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"uncertain": []},  # missing text
        {"text": "", "uncertain": []},  # empty text
        {"text": CORRECTED_TEXT, "uncertain": ["которого нет в тексте"]},  # not verbatim
        {"text": CORRECTED_TEXT, "uncertain": [], "advice": "Ложись раньше"},  # extra field
    ],
)
def test_invalid_contract_fails_loudly_and_stores_nothing(
    store: _RecordingStore, payload: dict[str, object]
) -> None:
    provider = _FakeProvider(_json_response(payload))

    with pytest.raises(LLMStructuredOutputError):
        stt_fix_service.correct_stt(
            None,  # type: ignore[arg-type]
            entry=_entry(),
            profile=_profile(),
            provider=provider,
        )

    assert store.calls == []


def test_a_response_without_uncertain_is_stored_with_an_empty_list(
    store: _RecordingStore,
) -> None:
    """``uncertain`` has a contract default, so an omitted field is not an error."""
    provider = _FakeProvider(_json_response({"text": CORRECTED_TEXT}))

    outcome = stt_fix_service.correct_stt(
        None,  # type: ignore[arg-type]
        entry=_entry(),
        profile=_profile(),
        provider=provider,
    )

    assert outcome.correction.uncertain == []
    assert len(store.calls) == 1


# ---------------------------------------------------------------------------
# the input is not modified
# ---------------------------------------------------------------------------


def test_the_run_does_not_change_the_input_entry(store: _RecordingStore) -> None:
    entry = _entry(text=RAW_TEXT)
    before = entry.model_dump()
    provider = _FakeProvider(
        _json_response({"text": CORRECTED_TEXT, "uncertain": ["ходил в парк"]})
    )

    stt_fix_service.correct_stt(
        None,  # type: ignore[arg-type]
        entry=entry,
        profile=_profile(),
        provider=provider,
    )

    assert entry.model_dump() == before
    assert entry.text == RAW_TEXT  # the raw text stays untouched
    # The only write performed by the layer is the append-only result store.
    assert [call["layer"] for call in store.calls] == [ArtifactLayer.stt_corrected]


def test_the_layer_declares_no_rendered_file_location() -> None:
    """The layer's result is inline; it has no per-layer artifact directory."""
    assert PROMPT_PATH.parent.name == "stt_fix"
    assert PROMPT_PATH.name == "v1.md"
    with pytest.raises(ValueError):
        naming.layer_dirname(ArtifactLayer.stt_corrected)
