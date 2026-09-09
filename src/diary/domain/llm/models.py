"""Normalised, provider-independent LLM models (Task 11, ADR-004).

Design notes
============

* :class:`LLMMessage`/:class:`LLMRequest` — the logical chat completion a
  pipeline wants to run. ``content`` may carry text derived from diary
  entries: it is an in-memory object only and must never be logged
  (AGENTS.md §8). Optional request fields (``model``, ``temperature``,
  ``max_tokens``) override the task profile's defaults when set.
* :class:`LLMResponse` — the normalised result: generated text plus the
  server-reported model, token-usage counters and timestamp when present.
  Metadata never embeds diary content — only integers and identifiers.
* :class:`LLMCallProvenance` — processing metadata for one completion
  attempt: provider type, model, prompt version, outcome, optional error
  class and HTTP status, usage counters, an SHA-256 *digest* of the
  normalised request content (never the content itself) and start/finish
  timestamps. It never contains secrets or diary text.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Roles supported by the normalised chat format (OpenAI-compatible subset).
LLM_ROLE = Literal["system", "user", "assistant"]


def utc_now() -> datetime:
    """Current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class LLMMessage(BaseModel):
    """One chat message (role + content). Content is never logged."""

    model_config = ConfigDict(extra="forbid")

    role: LLM_ROLE
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class LLMRequest(BaseModel):
    """A chat-completion request, independent of the provider implementation.

    ``messages`` is the payload content (never logged). Optional
    ``model``/``temperature``/``max_tokens`` override the task profile's
    values; the profile remains authoritative for the endpoint (``base_url``)
    and for defaults (Task 03, ADR-004).
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[LLMMessage] = Field(min_length=1)
    model: str | None = Field(default=None, min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    #: Version of the prompt template this request was built from. Recorded in
    #: provenance only; never sent to the endpoint.
    prompt_version: str | None = Field(default=None, min_length=1)

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        prompt_version: str | None = None,
    ) -> LLMRequest:
        """A single-turn request wrapping ``prompt`` as the only user message."""
        return cls(
            messages=[LLMMessage(role="user", content=prompt)],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
        )


class LLMUsage(BaseModel):
    """Token-usage counters as reported by the endpoint (never content)."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class LLMResponse(BaseModel):
    """The normalised completion result of one provider call."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    #: Model name as echoed by the endpoint, when provided.
    model: str | None = None
    usage: LLMUsage | None = None
    #: Endpoint-reported creation time (timezone-aware, UTC), when provided.
    created_at: datetime | None = None

    @field_validator("created_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (UTC)")
        return value


class LLMCallOutcome(str, Enum):
    """Outcome of one LLM completion attempt."""

    success = "success"
    failure = "failure"


class LLMCallProvenance(BaseModel):
    """Processing metadata for one LLM call. Never contains content or keys."""

    model_config = ConfigDict(extra="forbid")

    #: Provider kind, e.g. ``openai_compatible`` (from the task profile).
    provider_type: str
    #: Model actually requested for this call.
    model: str = Field(min_length=1)
    #: Prompt-template version, passed by the caller.
    prompt_version: str | None = None
    status: LLMCallOutcome
    #: Exception class name of a failed call (e.g. ``LLMAuthError``).
    error_class: str | None = None
    #: HTTP status code of a failed call, when known.
    status_code: int | None = None
    usage: LLMUsage | None = None
    #: SHA-256 digest of the normalised request content (never the content).
    input_sha256: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime

    @field_validator("started_at", "finished_at")
    @classmethod
    def _require_tz_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware (UTC)")
        return value


def request_input_sha256(request: LLMRequest) -> str:
    """SHA-256 digest of the normalised request content.

    Computed over deterministic canonical JSON of the message list (role +
    content only), so identical requests always produce the same digest while
    the content itself is never stored in provenance.
    """
    canonical = json.dumps(
        [{"role": message.role, "content": message.content} for message in request.messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "LLMCallOutcome",
    "LLMCallProvenance",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "request_input_sha256",
    "utc_now",
]
