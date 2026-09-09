"""Error taxonomy for the LLM provider layer (Task 11, ADR-004).

Privacy contract: every message is a static, non-sensitive summary — the
error class plus a short safe reason such as an HTTP status code. Response
bodies, request bodies, diary content, prompts and API keys never appear in
``str()``, ``repr()``, ``args`` or the attributes of these exceptions
(``status_code`` is an integer and is never sensitive).

Retry policy: each error exposes a ``retryable`` class attribute that the
Task-08 worker policy and the Task-12 pipeline handlers consult when they map
a failure to a transient retry or to a permanent job failure:

- **not retryable** (deterministic — repeating the identical request cannot
  help): provider unavailable (unset/empty API-key env var, unsupported
  provider), HTTP 401/403 authentication errors, and a 2xx response that
  cannot be parsed into a usable completion (a wrong endpoint or a
  non-compatible proxy returns the same broken shape on every attempt);
- **retryable** (transient — a later attempt may succeed): HTTP 429 rate
  limits, HTTP 5xx server errors, transport/network failures and timeouts.
"""

from __future__ import annotations

from typing import ClassVar


class LLMError(Exception):
    """Base class for all LLM-provider errors (never diary content/keys).

    ``retryable`` is a class attribute — ``False`` for deterministic failures
    that must not burn the job retry budget, ``True`` for transient failures
    that may succeed on a later attempt. ``status_code`` carries the optional
    HTTP status that produced the error when one exists.
    """

    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMProviderUnavailableError(LLMError):
    """The configured profile is unusable and retrying cannot fix it.

    Raised when the ``api_key_env`` variable is unset or empty, when the task
    has no LLM profile, or when no provider is registered for the profile
    type. The message names only the environment-variable *name*, never a key
    value.
    """

    retryable: ClassVar[bool] = False


class LLMAuthError(LLMError):
    """The endpoint rejected the credentials (HTTP 401 or 403)."""

    retryable: ClassVar[bool] = False


class LLMRateLimitError(LLMError):
    """The endpoint rate-limited the request (HTTP 429)."""

    retryable: ClassVar[bool] = True


class LLMServerError(LLMError):
    """The endpoint failed internally (HTTP 5xx)."""

    retryable: ClassVar[bool] = True


class LLMNetworkError(LLMError):
    """Transport-level failure: DNS, connection refused/reset, etc."""

    retryable: ClassVar[bool] = True


class LLMTimeoutError(LLMError):
    """The request exceeded the configured timeout (connect/read/write/pool)."""

    retryable: ClassVar[bool] = True


class LLMMalformedResponseError(LLMError):
    """A 2xx response that cannot be parsed into a usable completion.

    Non-JSON bodies, missing/empty ``choices``, or a choice without a text
    ``content`` mean the endpoint does not actually implement the
    OpenAI-compatible chat-completions contract for this request path. The
    identical request would return the same shape again, so this failure is
    **not retryable** at the client level. Structured-output (schema)
    validation of a *valid* completion is a pipeline concern (Task 13+) and is
    deliberately out of scope for this error.
    """

    retryable: ClassVar[bool] = False


__all__ = [
    "LLMAuthError",
    "LLMError",
    "LLMMalformedResponseError",
    "LLMNetworkError",
    "LLMProviderUnavailableError",
    "LLMRateLimitError",
    "LLMServerError",
    "LLMTimeoutError",
]
