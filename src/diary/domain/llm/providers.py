"""LLM provider abstraction and the OpenAI-compatible HTTPX adapter (Task 11).

Design (``docs/architecture-brief.md`` §9–10, ``skills/llm-pipelines``,
ADR-004):

* :class:`LLMProvider` — the ``complete(request) -> LLMResponse`` protocol
  every adapter implements;
* :class:`OpenAICompatibleProvider` — one HTTPX-based synchronous adapter that
  covers both local OpenAI-compatible endpoints (Ollama, llama.cpp, ...) and
  commercial ones. It reads the API key from ``os.environ`` *at request time*
  through the profile's ``api_key_env`` name and never stores, logs or echoes
  the value;
* :func:`create_llm_provider` — a registry-based factory. Tests replace the
  registry entry (or the function) with a fake provider, so no real endpoint
  is ever called in CI.

Endpoint URL: ``<base_url>/chat/completions``. ``base_url`` may or may not end
with ``/v1`` (e.g. ``http://127.0.0.1:11434/v1``); the adapter never appends a
``/v1`` itself, so a configured ``/v1`` is not duplicated.

Privacy contract (AGENTS.md §7–8): this module never logs request content,
prompts, completions or keys. Errors carry only the static safe messages from
:mod:`diary.domain.llm.errors`; response bodies are never part of messages,
``repr`` or logs.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol, TypedDict

import httpx

from diary.core.models_config import OpenAICompatibleProfile
from diary.domain.llm.errors import (
    LLMAuthError,
    LLMError,
    LLMMalformedResponseError,
    LLMNetworkError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from diary.domain.llm.models import LLMRequest, LLMResponse, LLMUsage


def chat_completions_url(base_url: str) -> str:
    """The chat-completions endpoint for an OpenAI-compatible ``base_url``.

    ``base_url`` keeps whatever path the configuration declares (it may or may
    not end in ``/v1``); the adapter only appends ``/chat/completions`` and
    never inserts a ``/v1``, so a configured ``/v1`` is not duplicated.
    """
    return f"{base_url.rstrip('/')}/chat/completions"


class CompletionParams(TypedDict):
    """Resolved completion parameters actually sent to the endpoint."""

    model: str
    temperature: float
    max_tokens: int


def resolve_completion_params(
    request: LLMRequest, profile: OpenAICompatibleProfile
) -> CompletionParams:
    """Merge request-level overrides over the task-profile defaults.

    The profile stays authoritative for the endpoint and supplies defaults for
    ``model``, ``temperature`` and ``max_tokens``; an explicitly set request
    field wins over the profile default.
    """
    return {
        "model": request.model if request.model is not None else profile.model,
        "temperature": (
            request.temperature if request.temperature is not None else profile.temperature
        ),
        "max_tokens": request.max_tokens if request.max_tokens is not None else profile.max_tokens,
    }


class LLMProvider(Protocol):
    """Completes a chat request into normalised text (never side effects;
    the caller persists the result)."""

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class OpenAICompatibleProvider:
    """OpenAI-compatible chat-completions adapter (synchronous HTTPX).

    All tunables come from the :class:`OpenAICompatibleProfile`; the API key
    is resolved from the environment on every request via ``api_key_env`` and
    is never stored on the instance, so ``repr``/attribute dumps cannot leak
    it.
    """

    def __init__(
        self,
        profile: OpenAICompatibleProfile,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._profile = profile
        #: Test seam: a custom HTTPX transport (e.g. ``httpx.MockTransport``).
        #: Production callers leave it ``None`` and HTTPX uses its real
        #: network transport.
        self._transport = transport

    def __repr__(self) -> str:
        # base_url is credential-free by Task-03 validation; the key and even
        # its environment-variable name never appear here.
        return (
            f"OpenAICompatibleProvider(model={self._profile.model!r}, "
            f"base_url={self._profile.base_url!r})"
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run ``request`` against the configured endpoint.

        Raises the :mod:`~diary.domain.llm.errors` taxonomy mapped from HTTP
        status codes and transport failures. The key is read from the
        environment here, per request, and is used only for the Authorization
        header of this call.
        """
        key = _resolve_api_key(self._profile)
        headers = {"Authorization": f"Bearer {key}"}
        payload: dict[str, object] = dict(resolve_completion_params(request, self._profile))
        payload["messages"] = [message.model_dump() for message in request.messages]
        timeout = httpx.Timeout(self._profile.timeout_s)
        try:
            with httpx.Client(transport=self._transport, timeout=timeout) as client:
                response = client.post(
                    chat_completions_url(self._profile.base_url),
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            # Covers connect/read/write/pool timeouts (ReadTimeout, ...).
            raise LLMTimeoutError("The LLM endpoint timed out.") from exc
        except httpx.TransportError as exc:
            # DNS, connection refused/reset and other transport failures.
            raise LLMNetworkError("The LLM endpoint is unreachable (network error).") from exc
        return _parse_completion_response(response)


def _resolve_api_key(profile: OpenAICompatibleProfile) -> str:
    """The API key for ``profile`` from the environment (never cached/stored).

    Raises :class:`LLMProviderUnavailableError` naming only the environment
    variable — never a key value.
    """
    env_name = profile.api_key_env
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise LLMProviderUnavailableError(
            f"The API-key environment variable {env_name!r} (api_key_env) is not set or is empty."
        )
    return value


def _parse_completion_response(response: httpx.Response) -> LLMResponse:
    """Map an HTTP response to an :class:`LLMResponse` or a safe error.

    Only the status code and a structurally parsed body are used; the body is
    never echoed into messages, so a malicious/broken endpoint cannot leak
    content through exceptions.
    """
    status = response.status_code
    if status in (401, 403):
        raise LLMAuthError(
            f"The LLM endpoint rejected the API key (HTTP {status}).", status_code=status
        )
    if status == 429:
        raise LLMRateLimitError(
            "The LLM endpoint rate-limited the request (HTTP 429).", status_code=status
        )
    if status >= 500:
        raise LLMServerError(f"The LLM endpoint failed (HTTP {status}).", status_code=status)
    if not 200 <= status < 300:
        raise LLMError(
            f"The LLM endpoint returned an unexpected response (HTTP {status}).",
            status_code=status,
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise LLMMalformedResponseError(
            f"The LLM endpoint returned a non-JSON response (HTTP {status})."
        ) from exc
    if not isinstance(data, dict):
        raise LLMMalformedResponseError(
            f"The LLM endpoint response is not a JSON object (HTTP {status})."
        )

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMMalformedResponseError(
            f"The LLM endpoint response has no usable 'choices' (HTTP {status})."
        )
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LLMMalformedResponseError(
            f"The LLM endpoint response has a malformed choice (HTTP {status})."
        )
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LLMMalformedResponseError(
            f"The LLM endpoint response has no usable message (HTTP {status})."
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMMalformedResponseError(
            f"The LLM endpoint response contains no text content (HTTP {status})."
        )

    reported_model = data.get("model")
    return LLMResponse(
        text=content,
        model=reported_model if isinstance(reported_model, str) else None,
        usage=_parse_usage(data.get("usage")),
        created_at=_parse_created_at(data.get("created")),
    )


def _parse_usage(raw: object) -> LLMUsage | None:
    """Usage counters from OpenAI-style ``prompt/completion_tokens``.

    Unknown or missing usage shapes are ignored — the completion text is still
    valid and provenance simply records ``usage=None``.
    """
    if not isinstance(raw, dict):
        return None
    input_tokens = raw.get("prompt_tokens")
    output_tokens = raw.get("completion_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _parse_created_at(raw: object) -> datetime | None:
    """Timezone-aware UTC datetime from an OpenAI-style ``created`` epoch."""
    if not isinstance(raw, int):
        return None
    try:
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


#: Provider factories, keyed by ``OpenAICompatibleProfile.provider``. Every
#: registered class takes just the profile in its constructor.
PROVIDER_REGISTRY: dict[str, Callable[[OpenAICompatibleProfile], LLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
}


def create_llm_provider(profile: OpenAICompatibleProfile) -> LLMProvider:
    """Return a provider instance for ``profile`` (registry-based factory).

    The registry is a module-level hook: tests monkeypatch it (or this
    function) with a fake provider, so LLM tests never touch a real endpoint.
    """
    provider_cls = PROVIDER_REGISTRY.get(profile.provider)
    if provider_cls is None:
        raise LLMProviderUnavailableError(
            f"No LLM provider is registered for {profile.provider!r}."
        )
    return provider_cls(profile)


__all__ = [
    "CompletionParams",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "PROVIDER_REGISTRY",
    "chat_completions_url",
    "create_llm_provider",
    "resolve_completion_params",
]
