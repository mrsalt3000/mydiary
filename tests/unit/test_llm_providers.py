"""Unit tests for the OpenAI-compatible LLM provider (Task 11).

No network is ever used: every call goes through ``httpx.MockTransport``. These
tests pin the request/URL construction, per-request API-key resolution (never
stored/echoed), the error taxonomy mapped from HTTP statuses and transport
failures, the ``retryable`` flags, and the registry-based factory seam.
"""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest
from pydantic import ValidationError

from diary.core.models_config import OpenAICompatibleProfile
from diary.domain.llm import errors as llm_errors
from diary.domain.llm import providers as llm_providers
from diary.domain.llm.models import LLMRequest, LLMResponse

KEY_ENV = "LLM_TEST_API_KEY"
KEY_VALUE = "sk-llm-test-fake-0001"
ENDPOINT = "http://127.0.0.1:11434/v1"

_OK_BODY: dict[str, object] = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "created": 1780000000,
    "model": "local-test",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Привет, дневник!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
}


def _profile(**overrides: object) -> OpenAICompatibleProfile:
    fields: dict[str, object] = {
        "provider": "openai_compatible",
        "base_url": ENDPOINT,
        "model": "local-test",
        "api_key_env": KEY_ENV,
        "timeout_s": 5.0,
        "temperature": 0.2,
        "max_tokens": 512,
        "external": False,
    }
    fields.update(overrides)
    return OpenAICompatibleProfile(**fields)  # type: ignore[arg-type]


def _request(text: str = "Привет, это тестовый запрос") -> LLMRequest:
    return LLMRequest.from_prompt(text, prompt_version="v1")


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    profile: OpenAICompatibleProfile | None = None,
) -> llm_providers.OpenAICompatibleProvider:
    return llm_providers.OpenAICompatibleProvider(
        profile or _profile(), transport=httpx.MockTransport(handler)
    )


@pytest.fixture(autouse=True)
def _clean_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    snapshot = dict(llm_providers.PROVIDER_REGISTRY)
    yield
    llm_providers.PROVIDER_REGISTRY.clear()
    llm_providers.PROVIDER_REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


def test_chat_completions_url_building() -> None:
    assert (
        llm_providers.chat_completions_url("http://127.0.0.1:11434/v1")
        == "http://127.0.0.1:11434/v1/chat/completions"
    )
    assert (
        llm_providers.chat_completions_url("http://127.0.0.1:11434")
        == "http://127.0.0.1:11434/chat/completions"
    )
    assert (
        llm_providers.chat_completions_url("http://host:8080/v1/")
        == "http://host:8080/v1/chat/completions"
    )


def test_request_hits_configured_v1_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_OK_BODY)

    response = _provider(handler).complete(_request())
    assert response.text == "Привет, дневник!"
    assert seen["url"] == f"{ENDPOINT}/chat/completions"


# ---------------------------------------------------------------------------
# Successful completion and request payload
# ---------------------------------------------------------------------------


def test_success_parses_text_model_usage_and_created_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    provider = _provider(lambda request: httpx.Response(200, json=_OK_BODY))

    response = provider.complete(_request())

    assert response.text == "Привет, дневник!"
    assert response.model == "local-test"
    assert response.usage is not None
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 4
    assert response.created_at is not None
    assert response.created_at.tzinfo is not None


def test_success_without_usage_or_created_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    body = {"choices": [{"message": {"role": "assistant", "content": "текст"}}]}
    response = _provider(lambda request: httpx.Response(200, json=body)).complete(_request())
    assert response.text == "текст"
    assert response.model is None
    assert response.usage is None
    assert response.created_at is None


def test_request_payload_is_openai_chat_completions_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_OK_BODY)

    _provider(handler).complete(_request("Содержимое для обработки"))
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "local-test"
    assert payload["messages"] == [{"role": "user", "content": "Содержимое для обработки"}]
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 512
    assert "prompt_version" not in payload  # prompt version is provenance-only
    assert seen["auth"] == f"Bearer {KEY_VALUE}"


def test_request_overrides_beat_profile_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_OK_BODY)

    request = LLMRequest.from_prompt("x", model="override-model", temperature=0.9, max_tokens=7)
    _provider(handler).complete(request)
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "override-model"
    assert payload["temperature"] == 0.9
    assert payload["max_tokens"] == 7


# ---------------------------------------------------------------------------
# reasoning_effort (T02b, ADR-009)
# ---------------------------------------------------------------------------


def _sent_payload(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: OpenAICompatibleProfile | None = None,
    request: LLMRequest | None = None,
) -> dict[str, object]:
    """Run one mocked completion and return the JSON payload sent."""
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    seen: dict[str, object] = {}

    def handler(incoming: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(incoming.content)
        return httpx.Response(200, json=_OK_BODY)

    provider = _provider(handler, profile) if profile is not None else _provider(handler)
    provider.complete(request if request is not None else _request())
    payload = seen["payload"]
    assert isinstance(payload, dict)
    return payload


def test_reasoning_effort_absent_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """(a) Backward compatibility: unset field is not sent at all (not ``null``)."""
    payload = _sent_payload(monkeypatch)
    assert "reasoning_effort" not in payload


def test_resolved_params_report_none_when_unset() -> None:
    params = llm_providers.resolve_completion_params(_request(), _profile())
    assert params["reasoning_effort"] is None


def test_profile_reasoning_effort_reaches_the_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """(b) A profile-level value is sent verbatim."""
    payload = _sent_payload(monkeypatch, profile=_profile(reasoning_effort="none"))
    assert payload["reasoning_effort"] == "none"


def test_request_reasoning_effort_beats_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """(c) Request-level override wins over the profile default."""
    payload = _sent_payload(
        monkeypatch,
        profile=_profile(reasoning_effort="none"),
        request=LLMRequest.from_prompt("x", reasoning_effort="high"),
    )
    assert payload["reasoning_effort"] == "high"


def test_request_reasoning_effort_applies_without_profile_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) The override also works when the profile does not set the field."""
    payload = _sent_payload(
        monkeypatch,
        profile=_profile(),
        request=LLMRequest.from_prompt("x", reasoning_effort="low"),
    )
    assert payload["reasoning_effort"] == "low"


@pytest.mark.parametrize("bad", ["minimal", "NONE", "yes", 1, ""])
def test_invalid_reasoning_effort_rejected(bad: object) -> None:
    """(d) Only the four documented levels are accepted, in profile and request."""
    with pytest.raises(ValidationError):
        _profile(reasoning_effort=bad)
    with pytest.raises(ValidationError):
        LLMRequest.from_prompt("x", reasoning_effort=bad)  # type: ignore[arg-type]


def test_resolved_params_prefer_request_over_profile() -> None:
    params = llm_providers.resolve_completion_params(
        LLMRequest.from_prompt("x", reasoning_effort="medium"),
        _profile(reasoning_effort="none"),
    )
    assert params["reasoning_effort"] == "medium"


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------


def test_key_is_read_from_env_on_every_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.headers.get("authorization")))
        return httpx.Response(200, json=_OK_BODY)

    provider = _provider(handler)
    monkeypatch.setenv(KEY_ENV, "first-key")
    provider.complete(_request())
    monkeypatch.setenv(KEY_ENV, "second-key")
    provider.complete(_request())
    assert captured == ["Bearer first-key", "Bearer second-key"]


def test_missing_key_raises_provider_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    provider = _provider(lambda request: httpx.Response(200, json=_OK_BODY))
    with pytest.raises(llm_errors.LLMProviderUnavailableError) as excinfo:
        provider.complete(_request())
    message = str(excinfo.value)
    assert KEY_ENV in message
    assert KEY_VALUE not in message
    assert excinfo.value.retryable is False


def test_blank_key_raises_provider_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "   ")
    provider = _provider(lambda request: httpx.Response(200, json=_OK_BODY))
    with pytest.raises(llm_errors.LLMProviderUnavailableError):
        provider.complete(_request())


def test_key_never_appears_in_repr_or_instance_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    provider = _provider(lambda request: httpx.Response(200, json=_OK_BODY))
    assert KEY_VALUE not in repr(provider)
    assert KEY_VALUE not in str(provider)
    for attribute in vars(provider).values():
        if isinstance(attribute, str):
            assert KEY_VALUE not in attribute


# ---------------------------------------------------------------------------
# HTTP-status → error taxonomy mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, llm_errors.LLMAuthError),
        (403, llm_errors.LLMAuthError),
        (429, llm_errors.LLMRateLimitError),
        (500, llm_errors.LLMServerError),
        (503, llm_errors.LLMServerError),
    ],
)
def test_http_status_mapping(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected: type[llm_errors.LLMError],
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)

    def handler(request: httpx.Request) -> httpx.Response:
        # The body may contain anything (even the key); it must never surface.
        return httpx.Response(status, json={"error": {"message": KEY_VALUE}})

    with pytest.raises(expected) as excinfo:
        _provider(handler).complete(_request())
    assert excinfo.value.retryable is expected.retryable
    assert excinfo.value.status_code == status
    assert KEY_VALUE not in str(excinfo.value)
    assert KEY_VALUE not in repr(excinfo.value)


def test_unlisted_status_raises_base_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    provider = _provider(lambda request: httpx.Response(400, json={"error": "bad"}))

    with pytest.raises(llm_errors.LLMError) as excinfo:
        provider.complete(_request())
    assert type(excinfo.value) is llm_errors.LLMError
    assert excinfo.value.retryable is False
    assert excinfo.value.status_code == 400
    assert "bad" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Transport failures
# ---------------------------------------------------------------------------


def test_timeout_is_mapped_to_llm_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)

    def handler(request: httpx.Request) -> httpx.Response:
        # MockTransport bypasses httpcore's clock, so a real slow endpoint is
        # simulated by the exact exception httpcore raises on a read timeout.
        raise httpx.ReadTimeout("simulated slow endpoint", request=request)

    with pytest.raises(llm_errors.LLMTimeoutError) as excinfo:
        _provider(handler).complete(_request())
    assert excinfo.value.retryable is True
    assert excinfo.value.status_code is None


def test_connect_error_is_mapped_to_llm_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(llm_errors.LLMNetworkError) as excinfo:
        _provider(handler).complete(_request())
    assert excinfo.value.retryable is True


# ---------------------------------------------------------------------------
# Malformed 2xx responses
# ---------------------------------------------------------------------------


def _malformed(
    monkeypatch: pytest.MonkeyPatch,
    body: object | None,
    *,
    text: str | None = None,
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)

    def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(200, text=text)
        return httpx.Response(200, json=body)

    with pytest.raises(llm_errors.LLMMalformedResponseError) as excinfo:
        _provider(handler).complete(_request())
    assert excinfo.value.retryable is False
    assert excinfo.value.status_code is None


def test_non_json_body_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    _malformed(monkeypatch, None, text="<html>proxy page</html>")


@pytest.mark.parametrize(
    "body",
    [
        ["not", "a", "dict"],
        {},
        {"choices": []},
        {"choices": ["assistant"]},
        {"choices": [{"index": 0}]},
        {"choices": [{"index": 0, "message": {"role": "assistant", "content": None}}]},
        {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}]},
        {"choices": [{"index": 0, "message": {"role": "assistant", "content": "   "}}]},
    ],
)
def test_structural_garbage_is_malformed(monkeypatch: pytest.MonkeyPatch, body: object) -> None:
    _malformed(monkeypatch, body)


# ---------------------------------------------------------------------------
# Factory / registry seam
# ---------------------------------------------------------------------------


def test_create_llm_provider_returns_openai_compatible_provider() -> None:
    provider = llm_providers.create_llm_provider(_profile())
    assert isinstance(provider, llm_providers.OpenAICompatibleProvider)


def test_registry_is_the_factory_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing the registry entry swaps the real adapter for a fake one —
    the mechanism Task-12+ pipeline tests use instead of a real endpoint."""

    class FakeLLMProvider:
        def __init__(self, profile: OpenAICompatibleProfile) -> None:
            self.profile = profile
            self.calls: list[LLMRequest] = []

        def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls.append(request)
            return LLMResponse(text="fake completion", model=self.profile.model)

    monkeypatch.setitem(llm_providers.PROVIDER_REGISTRY, "openai_compatible", FakeLLMProvider)
    provider = llm_providers.create_llm_provider(_profile())
    assert isinstance(provider, FakeLLMProvider)
    response = provider.complete(_request())
    assert response.text == "fake completion"
    assert len(provider.calls) == 1


def test_create_llm_provider_rejects_non_llm_profile() -> None:
    """A non-LLM profile (e.g. faster-whisper STT) has no registered provider."""
    from diary.core.models_config import FasterWhisperProfile

    profile = FasterWhisperProfile(provider="faster_whisper", model="small")
    with pytest.raises(llm_errors.LLMProviderUnavailableError, match="registered"):
        llm_providers.create_llm_provider(profile)  # type: ignore[arg-type]


def test_create_llm_provider_rejects_empty_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_providers, "PROVIDER_REGISTRY", {})
    with pytest.raises(llm_errors.LLMProviderUnavailableError, match="registered"):
        llm_providers.create_llm_provider(_profile())
