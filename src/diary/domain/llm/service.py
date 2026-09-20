"""LLM task-profile resolution and provenance recording (Task 11).

* :func:`resolve_profile` maps a diary task to its Task-03
  ``openai_compatible`` profile. Only normalization, summarization and
  wellbeing analysis run through an LLM provider; transcription uses the local
  STT provider (Task 09) and is rejected here with a clear error.
* :func:`resolve_phase2_profile` maps a phase-2 profile *name*
  (``diary_stt``/``diary_sum``/``diary_week``/``diary_coach``) to its optional
  profile; an absent profile means the layer is switched off (task T02/T04).
* :func:`success_provenance` / :func:`failure_provenance` build a
  :class:`~diary.domain.llm.models.LLMCallProvenance` for one completion
  attempt. Provenance carries metadata only — never diary content, prompts or
  API keys (AGENTS.md §8); the request content appears solely as an SHA-256
  digest.

The helpers are the recording seam the Task-12 pipeline framework uses when it
persists per-run provenance; nothing here is wired to the worker or the API
yet (use starts in Task 12).
"""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from diary.core.models_config import ModelsConfig, OpenAICompatibleProfile, TaskType
from diary.domain.llm.errors import LLMError, LLMProviderUnavailableError
from diary.domain.llm.models import (
    LLMCallOutcome,
    LLMCallProvenance,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    request_input_sha256,
    utc_now,
)
from diary.domain.llm.providers import resolve_completion_params

#: Diary tasks that run through an LLM provider (Task 03/11).
LLM_TASK_TYPES: frozenset[TaskType] = frozenset(
    {TaskType.normalization, TaskType.summarization, TaskType.wellbeing_analysis}
)

_TASK_NAMES = ", ".join(sorted(task.value for task in LLM_TASK_TYPES))


class _ProvenanceFields(TypedDict):
    """Every field of :class:`LLMCallProvenance`, fully typed for expansion."""

    provider_type: str
    model: str
    prompt_version: str | None
    status: LLMCallOutcome
    error_class: str | None
    status_code: int | None
    usage: LLMUsage | None
    input_sha256: str
    started_at: datetime
    finished_at: datetime


def resolve_profile(models_config: ModelsConfig, task_type: TaskType) -> OpenAICompatibleProfile:
    """Return the ``openai_compatible`` LLM profile configured for ``task_type``.

    Raises :class:`LLMProviderUnavailableError` (never retryable) when the task
    does not use an LLM provider (e.g. transcription) or its profile is not an
    ``openai_compatible`` profile.
    """
    if task_type not in LLM_TASK_TYPES:
        raise LLMProviderUnavailableError(
            f"Task {task_type.value!r} does not use an LLM provider (only {_TASK_NAMES} do)."
        )
    profile = getattr(models_config, task_type.value, None)
    if not isinstance(profile, OpenAICompatibleProfile):
        raise LLMProviderUnavailableError(
            f"Task {task_type.value!r} is not configured with an openai_compatible LLM profile."
        )
    return profile


def resolve_phase2_profile(
    models_config: ModelsConfig, profile_name: str
) -> OpenAICompatibleProfile:
    """Return the phase-2 layer profile stored under ``profile_name``.

    The phase-2 layers (``stt_corrected``, ``daily_summary``, ``weekly_summary``,
    ``coach_note``) are configured by *profile name* — ``diary_stt``,
    ``diary_sum``, ``diary_week``, ``diary_coach`` — rather than by a
    :class:`TaskType`: their
    profiles are optional additions to ``configs/models.yaml`` (T02), and the
    four MVP task types are not extended by the phase-2 backlog.

    An absent profile means "this layer is switched off" and raises
    :class:`LLMProviderUnavailableError` (never retryable), naming only the
    configuration field (AGENTS.md §8).
    """
    profile = getattr(models_config, profile_name, None)
    if not isinstance(profile, OpenAICompatibleProfile):
        raise LLMProviderUnavailableError(
            f"The phase-2 profile {profile_name!r} is not configured; the layer is switched off."
        )
    return profile


def _provenance_fields(
    *,
    profile: OpenAICompatibleProfile,
    request: LLMRequest,
    outcome: LLMCallOutcome,
    prompt_version: str | None,
    started_at: datetime | None,
    finished_at: datetime | None,
) -> _ProvenanceFields:
    """Shared provenance fields (model/digest/timestamps) for one attempt."""
    effective_model = resolve_completion_params(request, profile)["model"]
    now = utc_now()
    return {
        "provider_type": profile.provider,
        "model": effective_model,
        "prompt_version": (
            prompt_version if prompt_version is not None else request.prompt_version
        ),
        "status": outcome,
        "error_class": None,
        "status_code": None,
        "usage": None,
        "input_sha256": request_input_sha256(request),
        "started_at": started_at if started_at is not None else now,
        "finished_at": finished_at if finished_at is not None else now,
    }


def success_provenance(
    profile: OpenAICompatibleProfile,
    request: LLMRequest,
    response: LLMResponse,
    *,
    prompt_version: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> LLMCallProvenance:
    """Provenance of a successful completion (usage recorded when reported)."""
    fields = _provenance_fields(
        profile=profile,
        request=request,
        outcome=LLMCallOutcome.success,
        prompt_version=prompt_version,
        started_at=started_at,
        finished_at=finished_at,
    )
    fields["usage"] = response.usage
    return LLMCallProvenance(**fields)


def failure_provenance(
    profile: OpenAICompatibleProfile,
    request: LLMRequest,
    error: LLMError,
    *,
    prompt_version: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> LLMCallProvenance:
    """Provenance of a failed completion (error class + HTTP status only)."""
    fields = _provenance_fields(
        profile=profile,
        request=request,
        outcome=LLMCallOutcome.failure,
        prompt_version=prompt_version,
        started_at=started_at,
        finished_at=finished_at,
    )
    fields["error_class"] = type(error).__name__
    status_code = error.status_code
    fields["status_code"] = status_code if isinstance(status_code, int) else None
    return LLMCallProvenance(**fields)


__all__ = [
    "LLM_TASK_TYPES",
    "failure_provenance",
    "resolve_phase2_profile",
    "resolve_profile",
    "success_provenance",
]
