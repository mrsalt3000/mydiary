"""Per-task model/provider configuration (Task 03).

Structure follows ``docs/architecture-brief.md`` §9. Profiles are loaded
from a YAML file whose path comes from the ``MODELS_CONFIG`` environment
variable and defaults to ``<repository>/configs/models.yaml`` — the real,
per-user working file, which is never committed; only the synthetic
``configs/models.example.yaml`` is tracked.

Privacy contract: profiles hold no secret values. An ``openai_compatible``
profile stores the *name* of the environment variable that holds the key
(``api_key_env``). Values are resolved from ``os.environ`` at runtime and
never enter model objects, logs, serialized config, or error messages.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from diary.core.settings import PROJECT_ROOT

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_COMPUTE_TYPES = frozenset({"int8", "int8_float16", "int8_float32", "float16", "float32"})

#: Environment variable that overrides the model-configuration file path.
MODELS_CONFIG_ENV_VAR = "MODELS_CONFIG"

#: Default path of the real (user-local, untracked) configuration file.
DEFAULT_MODELS_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"

_CONFIG_HINT = (
    f"Create it from configs/models.example.yaml or point "
    f"{MODELS_CONFIG_ENV_VAR} at an existing file."
)


class ModelsConfigError(Exception):
    """Raised when the model configuration is missing, malformed, or invalid."""


class TaskType(str, Enum):
    """Diary task types that carry an independent model profile."""

    transcription = "transcription"
    normalization = "normalization"
    summarization = "summarization"
    wellbeing_analysis = "wellbeing_analysis"


class ProviderType(str, Enum):
    """Supported provider kinds."""

    faster_whisper = "faster_whisper"
    openai_compatible = "openai_compatible"


class FasterWhisperProfile(BaseModel):
    """Local speech-to-text profile (faster-whisper)."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["faster_whisper"]
    #: Model name/size, e.g. ``small`` or ``large-v3``.
    model: str = Field(min_length=1)
    #: STT language code; ``ru`` by default.
    language: str = Field(default="ru", min_length=1)
    #: Inference device.
    device: Literal["cpu", "cuda"] = "cpu"
    #: faster-whisper compute type.
    compute_type: str = Field(default="int8")
    #: Beam-search width used during decoding.
    beam_size: int = Field(default=5, ge=1)

    @field_validator("compute_type")
    @classmethod
    def _validate_compute_type(cls, value: str) -> str:
        if value not in _COMPUTE_TYPES:
            allowed = ", ".join(sorted(_COMPUTE_TYPES))
            raise ValueError(f"unsupported compute_type {value!r}; expected one of: {allowed}")
        return value


class OpenAICompatibleProfile(BaseModel):
    """OpenAI-compatible LLM profile (local or commercial endpoint)."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai_compatible"]
    #: Endpoint base URL, e.g. ``http://127.0.0.1:11434/v1``.
    base_url: str
    #: Model identifier served by the endpoint.
    model: str = Field(min_length=1)
    #: Name of the environment variable holding the API key (never the value).
    api_key_env: str = Field(min_length=1)
    #: Request timeout in seconds.
    timeout_s: float = Field(default=120.0, gt=0)
    #: Sampling temperature.
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    #: Maximum number of output tokens for a single request.
    max_tokens: int = Field(gt=0)
    #: Whether the endpoint is external (diary content may leave the machine).
    external: bool = False

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"}:
            raise ValueError("base_url must start with http:// or https://")
        if not parts.hostname:
            raise ValueError("base_url must include a host")
        if parts.username or parts.password:
            raise ValueError(
                "base_url must not embed credentials; put the API key into the "
                "environment variable named by api_key_env instead"
            )
        try:
            parts.port
        except ValueError as exc:
            raise ValueError("base_url contains an invalid port") from exc
        return value

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str) -> str:
        name = value.strip()
        if not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(
                "api_key_env must be a plain environment-variable name "
                "(letters, digits, underscore); it must reference a variable "
                "and must not contain a key value"
            )
        return name


#: Any concrete task profile.
ModelProfile = FasterWhisperProfile | OpenAICompatibleProfile


class ModelsConfig(BaseModel):
    """Validated per-task model profiles; all four tasks are required."""

    model_config = ConfigDict(extra="forbid")

    transcription: FasterWhisperProfile
    normalization: OpenAICompatibleProfile
    summarization: OpenAICompatibleProfile
    wellbeing_analysis: OpenAICompatibleProfile

    def profiles_by_task(self) -> list[tuple[TaskType, ModelProfile]]:
        """Return the four fixed task profiles in deterministic order."""
        return [
            (TaskType.transcription, self.transcription),
            (TaskType.normalization, self.normalization),
            (TaskType.summarization, self.summarization),
            (TaskType.wellbeing_analysis, self.wellbeing_analysis),
        ]


def _resolve_config_path(path: Path | None) -> Path:
    if path is not None:
        return path
    env_value = os.environ.get(MODELS_CONFIG_ENV_VAR, "").strip()
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_MODELS_CONFIG_PATH


def load_models_config(path: Path | str | None = None) -> ModelsConfig:
    """Load and validate model profiles from a YAML file.

    ``path`` falls back to the ``MODELS_CONFIG`` environment variable and
    then to ``configs/models.yaml`` (the real user file; not committed).
    """
    config_path = _resolve_config_path(Path(path) if path is not None else None)
    if not config_path.is_file():
        raise ModelsConfigError(
            f"model configuration file not found: {config_path}. {_CONFIG_HINT}"
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelsConfigError(f"cannot read model configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ModelsConfigError(
            f"model configuration {config_path} is not valid YAML: {exc}"
        ) from exc
    return parse_models_config(raw, source=str(config_path))


def parse_models_config(raw: object, *, source: str = "<config>") -> ModelsConfig:
    """Validate an already-parsed YAML mapping into a :class:`ModelsConfig`."""
    if not isinstance(raw, dict):
        raise ModelsConfigError(f"{source}: expected a YAML mapping at the top level")
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict):
        raise ModelsConfigError(
            f"{source}: expected a mapping under the 'profiles' key with one "
            "profile per task (transcription, normalization, summarization, "
            "wellbeing_analysis)"
        )
    try:
        return ModelsConfig.model_validate(profiles)
    except ValidationError as exc:
        raise ModelsConfigError(_format_validation_error(exc, source)) from exc


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines = [f"{source}: model configuration is invalid:"]
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)
