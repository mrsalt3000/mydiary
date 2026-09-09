"""Safe, non-sensitive diagnostics for configured model profiles (Task 03).

Internal service only — not an HTTP endpoint (API diagnostics arrive in a
later task). For each of the four task profiles it reports whether the
profile is present, which provider/model it uses, a credentials-free host
view of the endpoint, and whether the referenced API-key environment
variable is configured.

Privacy contract: a resolved key value is immediately reduced to a boolean
and is never stored on, serialized from, or echoed by any object in this
module. Only environment-variable *names* may appear in error messages.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from pydantic import BaseModel

from diary.core.models_config import (
    FasterWhisperProfile,
    ModelProfile,
    ModelsConfig,
    ModelsConfigError,
    ProviderType,
    TaskType,
)


class ProfileStatus(BaseModel):
    """Diagnostic view of one task profile. Never contains secret values."""

    task: TaskType
    provider: ProviderType
    model: str
    #: ``scheme://host[:port]`` only — no path, query, or credentials.
    base_url: str | None = None
    #: Whether the referenced ``api_key_env`` variable is set and non-empty.
    api_key_configured: bool
    #: Whether the profile is valid and ready to use.
    valid: bool


def _env_is_configured(env_name: str) -> bool:
    value = os.environ.get(env_name, "")
    return bool(value.strip())


def _host_only_url(base_url: str) -> str | None:
    """Return ``scheme://host[:port]`` without path/query/credentials."""
    parts = urlsplit(base_url)
    host = parts.hostname
    if host is None:
        return None
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is not None:
        host = f"{host}:{port}"
    return f"{parts.scheme}://{host}"


def status_for(task: TaskType, profile: ModelProfile) -> ProfileStatus:
    """Build a diagnostics view for a single task profile."""
    if isinstance(profile, FasterWhisperProfile):
        # Local STT requires no API key, so it is trivially configured.
        return ProfileStatus(
            task=task,
            provider=ProviderType.faster_whisper,
            model=profile.model,
            api_key_configured=True,
            valid=True,
        )
    key_configured = _env_is_configured(profile.api_key_env)
    return ProfileStatus(
        task=task,
        provider=ProviderType.openai_compatible,
        model=profile.model,
        base_url=_host_only_url(profile.base_url),
        api_key_configured=key_configured,
        valid=key_configured,
    )


def profile_statuses(config: ModelsConfig) -> list[ProfileStatus]:
    """Return diagnostic status for all four task profiles."""
    return [status_for(task, profile) for task, profile in config.profiles_by_task()]


def validate_all(config: ModelsConfig) -> None:
    """Raise :class:`ModelsConfigError` when a profile is not ready to use.

    Currently the only runtime readiness check is secret presence: every
    ``openai_compatible`` profile must reference an environment variable
    that is set and non-empty. Local ``faster_whisper`` STT needs no secret.
    The error names only the task and the environment-variable name.
    """
    problems: list[str] = []
    for task, profile in config.profiles_by_task():
        if isinstance(profile, FasterWhisperProfile):
            continue
        if not _env_is_configured(profile.api_key_env):
            problems.append(
                f"task {task.value!r} (provider "
                f"{ProviderType.openai_compatible.value!r}): required secret "
                f"environment variable {profile.api_key_env!r} (api_key_env) "
                "is not set or is empty"
            )
    if problems:
        raise ModelsConfigError("model configuration is not ready:\n- " + "\n- ".join(problems))
