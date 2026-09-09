"""Unit tests for LLM profile resolution and provenance recording (Task 11).

Covers :func:`resolve_profile` (per-task profile mapping, transcription is not
an LLM task) and the success/failure provenance builders — including the
guarantee that provenance dumps never carry request content or keys.
"""

from __future__ import annotations

import pytest

from diary.core.models_config import (
    FasterWhisperProfile,
    ModelsConfig,
    OpenAICompatibleProfile,
    TaskType,
)
from diary.domain.llm import errors as llm_errors
from diary.domain.llm.models import (
    LLMCallOutcome,
    LLMCallProvenance,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    request_input_sha256,
    utc_now,
)
from diary.domain.llm.service import (
    LLM_TASK_TYPES,
    failure_provenance,
    resolve_profile,
    success_provenance,
)


def _llm_profile(
    base_url: str,
    model: str,
    api_key_env: str,
    *,
    external: bool = False,
) -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        timeout_s=10.0,
        temperature=0.1,
        max_tokens=256,
        external=external,
    )


def _config() -> ModelsConfig:
    return ModelsConfig(
        transcription=FasterWhisperProfile(provider="faster_whisper", model="small"),
        normalization=_llm_profile(
            "http://127.0.0.1:9000/v1", "norm-model", "NORMALIZATION_TEST_KEY"
        ),
        summarization=_llm_profile(
            "https://api.remote.example/v1", "summ-model", "SUMMARIZATION_TEST_KEY"
        ),
        wellbeing_analysis=_llm_profile(
            "http://127.0.0.1:9000/v1", "wb-model", "WELLBEING_TEST_KEY"
        ),
    )


# ---------------------------------------------------------------------------
# resolve_profile
# ---------------------------------------------------------------------------


def test_llm_task_types_covers_three_llm_tasks() -> None:
    assert LLM_TASK_TYPES == frozenset(
        {TaskType.normalization, TaskType.summarization, TaskType.wellbeing_analysis}
    )
    assert TaskType.transcription not in LLM_TASK_TYPES


def test_resolve_profile_returns_per_task_profiles() -> None:
    config = _config()

    normalization = resolve_profile(config, TaskType.normalization)
    summarization = resolve_profile(config, TaskType.summarization)
    wellbeing = resolve_profile(config, TaskType.wellbeing_analysis)

    assert normalization.base_url == "http://127.0.0.1:9000/v1"
    assert normalization.model == "norm-model"
    assert normalization.api_key_env == "NORMALIZATION_TEST_KEY"
    assert summarization.base_url == "https://api.remote.example/v1"
    assert summarization.model == "summ-model"
    assert wellbeing.model == "wb-model"
    # Different tasks may point to different endpoints/models by configuration.
    assert normalization.base_url != summarization.base_url
    assert normalization.model != summarization.model


def test_resolve_profile_rejects_transcription() -> None:
    with pytest.raises(llm_errors.LLMProviderUnavailableError) as excinfo:
        resolve_profile(_config(), TaskType.transcription)
    assert "does not use an LLM provider" in str(excinfo.value)
    assert excinfo.value.retryable is False


def test_resolve_profile_rejects_non_openai_compatible_profile() -> None:
    config = _config()
    config.normalization = FasterWhisperProfile(  # type: ignore[assignment]
        provider="faster_whisper", model="small"
    )
    with pytest.raises(llm_errors.LLMProviderUnavailableError) as excinfo:
        resolve_profile(config, TaskType.normalization)
    assert "openai_compatible" in str(excinfo.value)
    assert "normalization" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Provenance recording
# ---------------------------------------------------------------------------


def _request() -> LLMRequest:
    return LLMRequest.from_prompt("Содержимое дневника для обработки", prompt_version="v1")


def test_success_provenance_records_metadata_only() -> None:
    profile = _llm_profile("http://127.0.0.1:9000/v1", "norm-model", "NORMALIZATION_TEST_KEY")
    request = _request()
    started = utc_now()
    response = LLMResponse(
        text="Обработанный текст",
        model="norm-model",
        usage=LLMUsage(input_tokens=30, output_tokens=5),
        created_at=utc_now(),
    )
    provenance = success_provenance(
        profile, request, response, started_at=started, finished_at=utc_now()
    )

    assert isinstance(provenance, LLMCallProvenance)
    assert provenance.provider_type == "openai_compatible"
    assert provenance.model == "norm-model"
    assert provenance.prompt_version == "v1"
    assert provenance.status is LLMCallOutcome.success
    assert provenance.error_class is None
    assert provenance.status_code is None
    assert provenance.usage == LLMUsage(input_tokens=30, output_tokens=5)
    assert provenance.input_sha256 == request_input_sha256(request)
    assert provenance.started_at == started
    assert provenance.finished_at >= provenance.started_at


def test_success_provenance_request_model_override_wins() -> None:
    profile = _llm_profile("http://127.0.0.1:9000/v1", "norm-model", "NORMALIZATION_TEST_KEY")
    request = LLMRequest.from_prompt("текст", model="override-model")
    response = LLMResponse(text="Ответ")
    provenance = success_provenance(profile, request, response)
    assert provenance.model == "override-model"


def test_failure_provenance_records_error_class_and_status() -> None:
    profile = _llm_profile("http://127.0.0.1:9000/v1", "norm-model", "NORMALIZATION_TEST_KEY")
    request = _request()
    error = llm_errors.LLMAuthError("rejected", status_code=401)
    provenance = failure_provenance(profile, request, error, prompt_version="v2")

    assert provenance.status is LLMCallOutcome.failure
    assert provenance.error_class == "LLMAuthError"
    assert provenance.status_code == 401
    assert provenance.usage is None
    assert provenance.prompt_version == "v2"
    assert provenance.input_sha256 == request_input_sha256(request)
    assert provenance.model == "norm-model"


def test_provenance_dumps_never_contain_request_content_or_keys() -> None:
    profile = _llm_profile("http://127.0.0.1:9000/v1", "norm-model", "NORMALIZATION_TEST_KEY")
    secret_marker = "sk-test-value-must-not-leak"
    request_text = "Личный текст дневника, который никогда не должен попасть в метаданные"
    request = LLMRequest.from_prompt(request_text)

    success = success_provenance(profile, request, LLMResponse(text="Ответ модели"))
    failure = failure_provenance(
        profile,
        request,
        llm_errors.LLMServerError(f"failed (HTTP 503) but {secret_marker}"),  # defensive
    )

    for provenance in (success, failure):
        dumped = provenance.model_dump_json()
        assert secret_marker not in dumped
        assert request_text not in dumped
        assert "Ответ модели" not in dumped
