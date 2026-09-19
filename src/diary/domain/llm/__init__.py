"""LLM domain package (Task 11, ADR-004).

A safe, configurable OpenAI-compatible LLM client reused by the normalization,
summarization and wellbeing pipelines (used from Task 12 onward; never wired
into workers/API here):

* :mod:`diary.domain.llm.models` — normalised request/response models, usage
  counters and processing provenance (``LLMCallProvenance``);
* :mod:`diary.domain.llm.errors` — the error taxonomy with ``retryable`` flags
  and safe, content-free messages;
* :mod:`diary.domain.llm.providers` — the ``LLMProvider`` protocol, the
  HTTPX-based ``OpenAICompatibleProvider`` and the registry-backed factory;
* :mod:`diary.domain.llm.service` — per-task profile resolution and
  provenance recording helpers;
* :mod:`diary.domain.llm.structured` — the strict bridge from a completion to a
  validated phase-2 contract, with backend-filled envelope fields (task T04).

Secrets are resolved from the environment by variable *name* on every request
and are never stored, logged or echoed. Diary content/prompts/completions are
never logged by this package (AGENTS.md §7–8).
"""

from diary.domain.llm.errors import (
    LLMAuthError,
    LLMError,
    LLMMalformedResponseError,
    LLMNetworkError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMServerError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)
from diary.domain.llm.models import (
    LLMCallOutcome,
    LLMCallProvenance,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    request_input_sha256,
    utc_now,
)
from diary.domain.llm.providers import (
    PROVIDER_REGISTRY,
    LLMProvider,
    OpenAICompatibleProvider,
    chat_completions_url,
    create_llm_provider,
    resolve_completion_params,
)
from diary.domain.llm.service import (
    LLM_TASK_TYPES,
    failure_provenance,
    resolve_phase2_profile,
    resolve_profile,
    success_provenance,
)
from diary.domain.llm.structured import parse_contract_json, parse_contract_response

__all__ = [
    "LLMAuthError",
    "LLMCallOutcome",
    "LLMCallProvenance",
    "LLMError",
    "LLMMalformedResponseError",
    "LLMMessage",
    "LLMNetworkError",
    "LLMProvider",
    "LLMProviderUnavailableError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMServerError",
    "LLMStructuredOutputError",
    "LLMTimeoutError",
    "LLM_TASK_TYPES",
    "LLMUsage",
    "OpenAICompatibleProvider",
    "PROVIDER_REGISTRY",
    "chat_completions_url",
    "create_llm_provider",
    "failure_provenance",
    "parse_contract_json",
    "parse_contract_response",
    "request_input_sha256",
    "resolve_completion_params",
    "resolve_phase2_profile",
    "resolve_profile",
    "success_provenance",
    "utc_now",
]
