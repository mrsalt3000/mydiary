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
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from diary.core.settings import PROJECT_ROOT

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_COMPUTE_TYPES = frozenset({"int8", "int8_float16", "int8_float32", "float16", "float32"})

#: Reasoning-effort levels accepted on the wire (standard OpenAI API field).
#: ``None`` means "do not send the field at all" — that is the default, and it
#: keeps the pre-T02b behaviour identical for every existing profile (ADR-009).
ReasoningEffort = Literal["none", "low", "medium", "high"]

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
    gigaam2_ctc_lm = "gigaam2_ctc_lm"
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


class GigaamCtcLmProfile(BaseModel):
    """Local speech-to-text profile for the GigaAM-v2 CTC+LM backend (T43).

    Second STT backend behind the same provider factory as
    :class:`FasterWhisperProfile`; the choice between the two is made by the
    ``provider`` discriminator of ``profiles.transcription`` alone, never by an
    environment variable and never by a silent fallback (``ADR-017``). The
    runtime itself (weights, a separate venv) is provisioned in T45 — this
    profile only describes what to run once it is installed.

    ``model_id`` is the Hugging Face repository and ``revision`` is **required**
    on purpose: the wrapper is executed locally through ``trust_remote_code``,
    so an unpinned revision would run whatever ``main`` happens to be
    (``docs/stt-backends-analysis.md`` §11, risk 1). The revision measured by
    the spike is ``07d97cf8a4f178b68679cafae0600c8487877301``.

    ``alpha``/``beta`` default to ``None``, which means "take the checkpoint's
    own values from ``language_model/attrs.json``" (measured:
    ``alpha = 0.4612549346468768``, ``beta = 0.3271780420615455``); only
    explicitly set values are sent to the decoder, and no upper bound is
    invented for them.

    ``max_chunk_s``/``chunk_overlap_s`` split a long recording into chunks the
    model can swallow. The 20 s default is the measured cheap point: on the
    RTX 3060 6144 MiB the GigaAM contour added **+962…981 MiB** of VRAM at 20 s
    chunks against **+1904 MiB** at 120 s (``docs/stt-backends-analysis.md``
    §6.5, §7.2), while the language model costs time but no VRAM.

    ``unload_after`` is the policy switch: the sidecar chosen by the spike
    (``docs/stt-backends-analysis.md`` §10) makes "unload" a process exit, and
    T44/T45 read this flag; nothing implements it yet in T43.

    Deliberately **not** fields (so a later session does not "helpfully" add
    them back):

    * ``sample_rate`` — GigaAM is hard-wired to mono 16 kHz; that is a property
      of the provider, not a knob to configure;
    * ``dtype`` — the working dtype is the checkpoint's native fp16; fp32 would
      only cost more VRAM for the same result.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["gigaam2_ctc_lm"]
    #: Hugging Face repository of the CTC+LM checkpoint, e.g.
    #: ``waveletdeboshir/gigaam-ctc-with-lm``.
    model_id: str = Field(min_length=1)
    #: Immutable revision (commit sha or tag) of that repository. Required: the
    #: wrapper runs locally via ``trust_remote_code``.
    revision: str = Field(min_length=1)
    #: Whether the Hugging Face wrapper may execute its own code locally.
    #: ``True`` is what makes the wrapper work; the trust boundary is
    #: documented in T45 (execution of third-party code in a separate venv).
    trust_remote_code: bool = True
    #: Inference device; ``cpu`` keeps the same default as the Whisper profile.
    device: Literal["cpu", "cuda"] = "cpu"
    #: Beam-search width of the CTC+LM decoder.
    beam_width: int = Field(default=64, ge=1)
    #: Language-model weight; ``None`` means "use the checkpoint's own value".
    alpha: float | None = Field(default=None, ge=0.0)
    #: Word-insertion bonus; ``None`` means "use the checkpoint's own value".
    beta: float | None = Field(default=None, ge=0.0)
    #: Longest chunk handed to the model, in seconds (see the class docstring).
    max_chunk_s: float = Field(default=20.0, gt=0)
    #: Overlap between neighbouring chunks, in seconds.
    chunk_overlap_s: float = Field(default=0.5, ge=0)
    #: Whether the backend is released after a run (sidecar policy, T44/T45).
    unload_after: bool = True

    @model_validator(mode="after")
    def _validate_chunking(self) -> GigaamCtcLmProfile:
        if self.chunk_overlap_s >= self.max_chunk_s:
            raise ValueError(
                f"chunk_overlap_s ({self.chunk_overlap_s}) must be smaller than "
                f"max_chunk_s ({self.max_chunk_s}); with an overlap that large the "
                "chunk is never shortened and the overlap is meaningless"
            )
        return self


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
    #: Optional reasoning-effort hint sent as the standard OpenAI field
    #: ``reasoning_effort``; ``None`` (the default) means the field is **not**
    #: sent at all, so the endpoint/provider default applies and no behaviour
    #: changes for profiles that do not opt in (ADR-009). ``"none"`` switches
    #: the reasoning channel off where the endpoint supports it, which keeps
    #: the whole ``max_tokens`` budget for the answer (phase-2 layers).
    reasoning_effort: ReasoningEffort | None = None
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


#: A transcription (speech-to-text) profile, **discriminated by ``provider``**.
#: Which backend runs is a configuration decision, not an environment variable:
#: ``faster_whisper`` (the MVP backend) or ``gigaam2_ctc_lm`` (the GigaAM-v2
#: CTC+LM contour chosen by the T42 spike). An unknown ``provider`` is a load
#: error listing the accepted tags — never a silent fallback to Whisper
#: (``ADR-017``).
TranscriptionProfile = Annotated[
    FasterWhisperProfile | GigaamCtcLmProfile, Field(discriminator="provider")
]

#: Any concrete task profile.
ModelProfile = FasterWhisperProfile | GigaamCtcLmProfile | OpenAICompatibleProfile


def transcription_model_name(profile: TranscriptionProfile) -> str:
    """The model identifier of a transcription profile.

    The two STT backends name their model differently — Whisper's field is
    ``model``, the GigaAM-v2 CTC+LM profile names the Hugging Face repository
    ``model_id`` — so provenance (``transcriptions.model_name``) and the STT
    stage logs read it through this one accessor instead of branching on the
    provider in every caller.
    """
    if isinstance(profile, GigaamCtcLmProfile):
        return profile.model_id
    return profile.model


class ModelsConfig(BaseModel):
    """Validated per-task model profiles.

    The four MVP task profiles are required. The phase-2 layer profiles
    (:attr:`diary_stt`, :attr:`diary_sum`, :attr:`diary_week`,
    :attr:`diary_coach`; tasks T02 and T13b) are **optional**: ``None`` means
    the layer is switched off, which is a valid state and not an error
    (``docs/tasklist-phase2.md`` §9.8). The real working ``configs/models.yaml``
    is user-owned and uncommitted, so a required new field would break the start
    of the already-accepted application.
    """

    model_config = ConfigDict(extra="forbid")

    transcription: TranscriptionProfile
    normalization: OpenAICompatibleProfile
    summarization: OpenAICompatibleProfile
    wellbeing_analysis: OpenAICompatibleProfile

    # --- профили слоёв фазы 2 (T02), все опциональные -----------------------
    #: STT-правка (слой 2 фазы 2); тег Ollama ``diary-stt``.
    diary_stt: OpenAICompatibleProfile | None = None
    #: Сводка дня (слой 3 фазы 2); тег Ollama ``diary-sum``.
    diary_sum: OpenAICompatibleProfile | None = None
    #: Сводка **недели** (слой 3 фазы 2, задача T13b). Отдельный профиль нужен,
    #: потому что день и неделя — один и тот же слой с одним профилем
    #: ``diary_sum``, и решение «день интерактивен и живёт на GPU, неделя
    #: фоновая и живёт на CPU» иначе не выражается конфигом. Отсутствие профиля
    #: = недельный слой выключен (как и у остальных профилей фазы 2).
    diary_week: OpenAICompatibleProfile | None = None
    #: Заметка слоя 5 (wellbeing); тег Ollama ``diary-coach``.
    diary_coach: OpenAICompatibleProfile | None = None

    def profiles_by_task(self) -> list[tuple[TaskType, ModelProfile]]:
        """Return the four fixed task profiles in deterministic order."""
        return [
            (TaskType.transcription, self.transcription),
            (TaskType.normalization, self.normalization),
            (TaskType.summarization, self.summarization),
            (TaskType.wellbeing_analysis, self.wellbeing_analysis),
        ]

    def phase2_profiles(self) -> list[tuple[str, OpenAICompatibleProfile]]:
        """Return the configured phase-2 profiles in deterministic order.

        Profiles that are absent are simply not returned: absence is a valid
        state ("layer switched off"), not an error (``docs/tasklist-phase2.md``
        §9.8). The accessor is keyed by the configuration field name on purpose
        and does **not** depend on :class:`TaskType`: extending that enum is a
        change to an existing MVP enum and needs its own ADR (T02 decision).
        """
        candidates: tuple[tuple[str, OpenAICompatibleProfile | None], ...] = (
            ("diary_stt", self.diary_stt),
            ("diary_sum", self.diary_sum),
            ("diary_week", self.diary_week),
            ("diary_coach", self.diary_coach),
        )
        return [(name, profile) for name, profile in candidates if profile is not None]


def _resolve_config_path(path: Path | None) -> Path:
    if path is not None:
        return path
    env_value = os.environ.get(MODELS_CONFIG_ENV_VAR, "").strip()
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_MODELS_CONFIG_PATH


class _DuplicateKeyError(Exception):
    """A YAML mapping repeated a key. Carries the path and the line, no values."""

    def __init__(self, path: str, key: object, line: int) -> None:
        super().__init__(f"{path}: {key!r} (line {line})")
        self.path = path
        self.key = key
        self.line = line


class _StrictLoader(yaml.SafeLoader):
    """``SafeLoader`` that refuses a repeated key inside one mapping (T43).

    PyYAML's default is to keep the **last** value and drop the earlier one
    silently. For ``profiles.transcription`` that would mean a user who
    uncomments the second backend and forgets to delete the first gets a silent
    backend substitution — exactly the class of defect ADR-017 forbids, and the
    same one that cost acceptance rounds in T38. Refusing is the only honest
    option: with one key, which backend runs is never ambiguous.

    The load path keeps a key stack so the message can name the full path
    (``profiles.transcription.provider``). Only keys, a path and a line number
    are ever reported — never a value, because this file may hold endpoint URLs
    and environment-variable names (AGENTS.md §7).

    YAML merge keys (``<<``) are not used by this project's configuration files.
    A plain alias and a merge that overrides nothing keep working; a field given
    both through a merge and explicitly is reported as a repeated key and the
    file is refused — standard YAML would silently let the explicit key win.
    That is the deliberate price of one unambiguous reading: these configs are
    written out in full, without inheritance.
    """

    def __init__(self, stream: object) -> None:
        super().__init__(stream)
        self._key_path: list[str] = []

    def key_path(self) -> str:
        """Dotted path of the mapping currently being constructed."""
        return ".".join(self._key_path) if self._key_path else "<root>"


def _construct_mapping_strict(
    loader: _StrictLoader, node: yaml.Node, deep: bool = False
) -> dict[object, object]:
    """Build a mapping, raising on a repeated key instead of overwriting."""
    if isinstance(node, yaml.MappingNode):
        loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            already_present = key in mapping
        except TypeError as exc:  # unhashable key: standard PyYAML diagnosis
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key ({exc})",
                key_node.start_mark,
            ) from exc
        if already_present:
            raise _DuplicateKeyError(loader.key_path(), key, key_node.start_mark.line + 1)
        loader._key_path.append(str(key))
        try:
            mapping[key] = loader.construct_object(value_node, deep=deep)
        finally:
            loader._key_path.pop()
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_strict,
)


def load_models_config(path: Path | str | None = None) -> ModelsConfig:
    """Load and validate model profiles from a YAML file.

    ``path`` falls back to the ``MODELS_CONFIG`` environment variable and
    then to ``configs/models.yaml`` (the real user file; not committed).

    Parsing is strict about repeated keys (:class:`_StrictLoader`): two
    ``transcription:`` blocks in one ``profiles:`` mapping are a load error
    naming the key, not "the last one wins".
    """
    config_path = _resolve_config_path(Path(path) if path is not None else None)
    if not config_path.is_file():
        raise ModelsConfigError(
            f"model configuration file not found: {config_path}. {_CONFIG_HINT}"
        )
    try:
        raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except OSError as exc:
        raise ModelsConfigError(f"cannot read model configuration {config_path}: {exc}") from exc
    except _DuplicateKeyError as exc:
        # Name the key, its path and the line — never a value: the file may hold
        # endpoint URLs and environment-variable names.
        consequence = (
            " — for `transcription` that would silently change the STT backend"
            if exc.key == "transcription"
            else ""
        )
        raise ModelsConfigError(
            f"model configuration {config_path} repeats the key {exc.key!r} at "
            f"line {exc.line} (under {exc.path}). A repeated key silently drops "
            f"the earlier value{consequence}, so the file is refused: delete the "
            "extra key instead of keeping both."
        ) from exc
    except yaml.YAMLError as exc:
        raise ModelsConfigError(
            f"model configuration {config_path} is not valid YAML: {exc}"
        ) from exc
    return parse_models_config(raw, source=str(config_path))


def parse_models_config(raw: object, *, source: str = "<config>") -> ModelsConfig:
    """Validate an already-parsed YAML mapping into a :class:`ModelsConfig`.

    Repeated keys cannot reach this function: it takes a plain mapping, and a
    repeated YAML key is rejected earlier, while the file is parsed
    (:func:`load_models_config`).
    """
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
