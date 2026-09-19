"""Unit tests for the strict structured-output bridge (task T04).

The bridge (:mod:`diary.domain.llm.structured`) is what turns a completion into
a validated phase-2 contract. It is pure — no provider, no database, no prompt
file — so every branch of it is unit-testable with hand-written responses.
All content is synthetic (AGENTS.md §13).
"""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import BaseModel, ConfigDict

from diary.core.models_config import (
    FasterWhisperProfile,
    ModelsConfig,
    OpenAICompatibleProfile,
)
from diary.domain.contracts import SttCorrection
from diary.domain.llm.errors import LLMError, LLMProviderUnavailableError, LLMStructuredOutputError
from diary.domain.llm.models import LLMResponse
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.llm.structured import parse_contract_json, parse_contract_response

RAW_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")

#: The envelope the backend supplies for this layer (never read from a model).
ENVELOPE: dict[str, object] = {
    "raw_id": str(RAW_ID),
    "model": "diary-stt",
    "prompt_ver": "v1",
}


def _response(payload: object) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload, ensure_ascii=False))


def _profile(model: str = "diary-stt") -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:11434/v1",
        model=model,
        api_key_env="SYNTHETIC_PHASE2_KEY",
        temperature=0.15,
        max_tokens=3072,
    )


# ---------------------------------------------------------------------------
# strict JSON parsing
# ---------------------------------------------------------------------------


def test_valid_json_object_is_parsed() -> None:
    assert parse_contract_json('{"text": "Синтетика.", "uncertain": []}') == {
        "text": "Синтетика.",
        "uncertain": [],
    }


def test_surrounding_whitespace_is_tolerated() -> None:
    assert parse_contract_json('\n  {"text": "Синтетика."}  \n') == {"text": "Синтетика."}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "Не JSON вовсе.",
        '[{"text": "Синтетика."}]',  # an array, not an object
        '"Синтетика."',  # a scalar
        '{"text": "Синтетика."',  # truncated
        '```json\n{"text": "Синтетика."}\n```',  # fenced block: no soft parsing
        'Вот ответ: {"text": "Синтетика."}',  # prose around the object
        '{"text": "Синтетика."} Спасибо!',
    ],
)
def test_non_object_or_wrapped_responses_are_rejected(raw: str) -> None:
    """No soft parsing: fences and prose are invalid output, not repaired JSON."""
    with pytest.raises(LLMStructuredOutputError):
        parse_contract_json(raw)


# ---------------------------------------------------------------------------
# contract validation and envelope fields
# ---------------------------------------------------------------------------


def test_valid_response_validates_into_the_contract() -> None:
    correction = parse_contract_response(
        _response({"text": "Синтетический текст.", "uncertain": ["Синтетический"]}),
        SttCorrection,
        envelope=ENVELOPE,
    )
    assert correction == SttCorrection(
        raw_id=RAW_ID,
        text="Синтетический текст.",
        uncertain=["Синтетический"],
        model="diary-stt",
        prompt_ver="v1",
    )


def test_envelope_wins_over_foreign_model_and_prompt_version() -> None:
    """A completion cannot claim a foreign model or prompt version."""
    correction = parse_contract_response(
        _response(
            {
                "text": "Синтетический текст.",
                "uncertain": [],
                "model": "чужой-модель",
                "prompt_ver": "v99",
            }
        ),
        SttCorrection,
        envelope=ENVELOPE,
    )
    assert correction.model == "diary-stt"
    assert correction.prompt_ver == "v1"
    assert correction.raw_id == RAW_ID


def test_envelope_wins_over_a_foreign_raw_id() -> None:
    """The stored correction is anchored to the entry the backend corrected."""
    correction = parse_contract_response(
        _response(
            {
                "raw_id": str(uuid.uuid4()),
                "text": "Синтетический текст.",
                "uncertain": [],
            }
        ),
        SttCorrection,
        envelope=ENVELOPE,
    )
    assert correction.raw_id == RAW_ID


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(LLMStructuredOutputError):
        parse_contract_response(
            _response({"text": "Синтетика.", "uncertain": [], "advice": "Ложись раньше"}),
            SttCorrection,
            envelope=ENVELOPE,
        )


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(LLMStructuredOutputError):
        parse_contract_response(_response({"uncertain": []}), SttCorrection, envelope=ENVELOPE)


def test_non_verbatim_uncertain_fragment_is_rejected() -> None:
    with pytest.raises(LLMStructuredOutputError):
        parse_contract_response(
            _response({"text": "Синтетический текст.", "uncertain": ["которого нет"]}),
            SttCorrection,
            envelope=ENVELOPE,
        )


def test_contract_without_envelope_values_is_rejected() -> None:
    """The envelope is required: a completion alone cannot fill the contract."""
    with pytest.raises(LLMStructuredOutputError):
        parse_contract_response(
            _response({"text": "Синтетика.", "uncertain": []}),
            SttCorrection,
            envelope={},
        )


def test_failure_is_a_non_retryable_llm_error_with_a_static_message() -> None:
    with pytest.raises(LLMStructuredOutputError) as excinfo:
        parse_contract_response(_response("не JSON"), SttCorrection, envelope=ENVELOPE)
    assert isinstance(excinfo.value, LLMError)
    assert excinfo.value.retryable is False
    # The message is static: no response text, no diary content.
    assert "не JSON" not in str(excinfo.value)


def test_bridge_works_for_any_contract() -> None:
    class _SyntheticContract(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: str
        model: str

    parsed = parse_contract_response(
        _response({"value": "Синтетика", "model": "чужое"}),
        _SyntheticContract,
        envelope={"model": "diary-sum"},
    )
    assert parsed.model == "diary-sum"


# ---------------------------------------------------------------------------
# phase-2 profile resolution
# ---------------------------------------------------------------------------


def _models_config(*, with_stt: bool) -> ModelsConfig:
    return ModelsConfig(
        transcription=FasterWhisperProfile(provider="faster_whisper", model="tiny"),
        normalization=_profile("norm"),
        summarization=_profile("sum"),
        wellbeing_analysis=_profile("well"),
        diary_stt=_profile("diary-stt") if with_stt else None,
    )


def test_phase2_profile_is_resolved_by_configuration_field_name() -> None:
    assert resolve_phase2_profile(_models_config(with_stt=True), "diary_stt").model == "diary-stt"


def test_absent_phase2_profile_means_the_layer_is_switched_off() -> None:
    with pytest.raises(LLMProviderUnavailableError) as excinfo:
        resolve_phase2_profile(_models_config(with_stt=False), "diary_stt")
    assert excinfo.value.retryable is False
    assert "diary_stt" in str(excinfo.value)
