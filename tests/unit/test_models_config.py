"""Unit tests for model configuration loading/validation and diagnostics (Task 03)
plus the phase-2 optional profiles (task T02).

Synthetic only: no network calls (the one payload test uses
``httpx.MockTransport``) and no real API keys. Secrets are simulated with
obviously fake values to prove they never leak into dumps or errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from diary.core.models_config import (
    MODELS_CONFIG_ENV_VAR,
    FasterWhisperProfile,
    GigaamCtcLmProfile,
    ModelsConfig,
    ModelsConfigError,
    OpenAICompatibleProfile,
    ProviderType,
    TaskType,
    load_models_config,
    parse_models_config,
    transcription_model_name,
)
from diary.core.profile_status import (
    profile_statuses,
    status_for,
    validate_all,
)
from diary.core.settings import PROJECT_ROOT
from diary.domain.llm.models import LLMRequest
from diary.domain.llm.providers import OpenAICompatibleProvider

EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "models.example.yaml"

API_KEY_NAMES = (
    "NORMALIZATION_API_KEY",
    "SUMMARIZATION_API_KEY",
    "WELLBEING_API_KEY",
    "OLLAMA_API_KEY",
)

SECRET_VALUES = {
    "NORMALIZATION_API_KEY": "sk-normalization-fake-1111",
    "SUMMARIZATION_API_KEY": "sk-summarization-fake-2222",
    "WELLBEING_API_KEY": "sk-wellbeing-fake-3333",
}

#: Phase-2 optional profile fields (T02, extended by T13b): field → aligned
#: artifact values. ``temperature`` must equal ``PARAMETER temperature`` of the
#: paired Modelfile and ``model`` must equal the Ollama tag:
#: ``resolve_completion_params`` always sends the profile temperature, so a
#: divergence would be a silent substitution. ``reasoning_effort`` is the T02b
#: (ADR-009) switch that turns the reasoning channel off for the phase-2 layers.
#: ``diary_week`` (T13b) is the weekly layer's own profile; its tag is the CPU
#: twin of the day tag, which is what the hybrid rule (day on GPU, week on CPU)
#: is made of.
PHASE2_PROFILES: dict[str, dict[str, object]] = {
    "diary_stt": {
        "model": "diary-stt",
        "modelfile": "Modelfile.stt",
        "temperature": 0.15,
        "max_tokens": 3072,
        "reasoning_effort": "none",
    },
    "diary_sum": {
        "model": "diary-sum",
        "modelfile": "Modelfile.sum",
        "temperature": 0.25,
        "max_tokens": 4096,
        "reasoning_effort": "none",
    },
    "diary_week": {
        "model": "diary-sum-cpu",
        "modelfile": "Modelfile.sum-cpu",
        "temperature": 0.25,
        "max_tokens": 4096,
        "reasoning_effort": "none",
    },
    "diary_coach": {
        "model": "diary-coach",
        "modelfile": "Modelfile.coach",
        "temperature": 0.4,
        # T20: the blanket 4096 did not fit the window together with the coach
        # prompt (4113 estimated tokens of 8192), so the cap went to 2048 — ten
        # times the ~200 tokens a live coaching note used.
        "max_tokens": 2048,
        "reasoning_effort": "none",
    },
}

PHASE2_FIELDS = tuple(PHASE2_PROFILES)

NUM_CTX = 8192
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

_OK_CHAT_BODY: dict[str, object] = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "created": 1780000000,
    "model": "diary-stub",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "готово"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
}


@pytest.fixture(autouse=True)
def _clean_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in API_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def example_config() -> ModelsConfig:
    return load_models_config(EXAMPLE_CONFIG)


# ---------------------------------------------------------------------------
# YAML builders (pure Python dicts, dumped with safe_dump)
# ---------------------------------------------------------------------------


def _default_profiles() -> dict[str, dict[str, object]]:
    return {
        "transcription": {
            "provider": "faster_whisper",
            "model": "small",
            "language": "ru",
            "device": "cpu",
            "compute_type": "int8",
            "beam_size": 5,
        },
        "normalization": {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "local-normalizer",
            "api_key_env": "NORMALIZATION_API_KEY",
            "timeout_s": 120,
            "temperature": 0.0,
            "max_tokens": 8000,
            "external": False,
        },
        "summarization": {
            "provider": "openai_compatible",
            "base_url": "https://api.example.com/v1",
            "model": "summary-model",
            "api_key_env": "SUMMARIZATION_API_KEY",
            "timeout_s": 120,
            "temperature": 0.2,
            "max_tokens": 2000,
            "external": True,
        },
        "wellbeing_analysis": {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "local-wellbeing",
            "api_key_env": "WELLBEING_API_KEY",
            "timeout_s": 120,
            "temperature": 0.2,
            "max_tokens": 1500,
            "external": False,
        },
    }


def _profiles_dict() -> dict[str, dict[str, object]]:
    # Deep-enough copy so tests can mutate their own fixture independently.
    return {task: dict(fields) for task, fields in _default_profiles().items()}


def _dump(profiles: dict[str, dict[str, object]]) -> str:
    return yaml.safe_dump({"profiles": profiles}, sort_keys=False)


def _yaml_for(
    tweaks: dict[str, dict[str, object]] | None = None,
    drop: set[str] | None = None,
) -> str:
    profiles = _profiles_dict()
    for task in drop or set():
        profiles.pop(task, None)
    for task, fields in (tweaks or {}).items():
        profiles.setdefault(task, {}).update(fields)
    return _dump(profiles)


def _config_from_text(tmp_path: Path, text: str) -> ModelsConfig:
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return load_models_config(path)


# ---------------------------------------------------------------------------
# Loading the repository example
# ---------------------------------------------------------------------------


def test_example_file_loads(example_config: ModelsConfig) -> None:
    transcription = example_config.transcription
    assert isinstance(transcription, FasterWhisperProfile)
    assert transcription.provider == "faster_whisper"
    assert transcription.model == "small"
    assert transcription.language == "ru"
    assert transcription.device == "cpu"
    assert transcription.compute_type == "int8"

    normalization = example_config.normalization
    assert isinstance(normalization, OpenAICompatibleProfile)
    assert normalization.provider == "openai_compatible"
    assert normalization.base_url == "http://127.0.0.1:11434/v1"
    assert normalization.api_key_env == "NORMALIZATION_API_KEY"
    assert normalization.max_tokens == 8000
    assert normalization.external is False

    assert isinstance(example_config.summarization, OpenAICompatibleProfile)
    assert isinstance(example_config.wellbeing_analysis, OpenAICompatibleProfile)


def test_example_provides_four_independent_profiles(example_config: ModelsConfig) -> None:
    models = {
        example_config.transcription.model,
        example_config.normalization.model,
        example_config.summarization.model,
        example_config.wellbeing_analysis.model,
    }
    assert models == {"small", "local-normalizer", "summary-model", "local-wellbeing"}
    assert example_config.normalization.base_url != example_config.summarization.base_url
    assert example_config.summarization.external is True


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def test_all_four_tasks_required(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, _yaml_for(drop={"wellbeing_analysis"}))
    assert "wellbeing_analysis" in str(exc.value)


def test_unknown_task_profile_name_rejected(tmp_path: Path) -> None:
    tweaks = {
        "sentiment": {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "x",
            "api_key_env": "SENTIMENT_API_KEY",
            "max_tokens": 100,
        }
    }
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, _yaml_for(tweaks=tweaks))
    assert "sentiment" in str(exc.value)


def test_unknown_inner_field_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, _yaml_for(tweaks={"transcription": {"modle": "tiny"}}))
    # Since T43 the transcription profile is a discriminated union, so the
    # location names the resolved backend variant as well (still the offending
    # field, one level deeper).
    assert "transcription.faster_whisper.modle" in str(exc.value)


def test_provider_type_enforced_per_task(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(
            tmp_path, _yaml_for(tweaks={"normalization": {"provider": "faster_whisper"}})
        )
    assert "normalization.provider" in str(exc.value)

    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(
            tmp_path,
            _yaml_for(tweaks={"transcription": {"provider": "openai_compatible"}}),
        )
    # The transcription profile accepts only the STT tags (T43): an LLM provider
    # is an unknown discriminator, and the message lists what *is* accepted.
    transcription_message = str(exc.value)
    assert "transcription" in transcription_message
    assert "openai_compatible" in transcription_message
    assert "faster_whisper" in transcription_message
    assert "gigaam2_ctc_lm" in transcription_message


def test_invalid_values_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, _yaml_for(tweaks={"transcription": {"device": "tpu"}}))
    # Tag-qualified location since T43 (transcription is a discriminated union).
    assert "transcription.faster_whisper.device" in str(exc.value)

    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(
            tmp_path,
            _yaml_for(tweaks={"transcription": {"compute_type": "float64"}}),
        )
    assert "transcription.faster_whisper.compute_type" in str(exc.value)

    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(
            tmp_path,
            _yaml_for(tweaks={"normalization": {"temperature": 3.0}}),
        )
    assert "normalization.temperature" in str(exc.value)

    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, _yaml_for(tweaks={"normalization": {"max_tokens": 0}}))
    assert "normalization.max_tokens" in str(exc.value)


def test_missing_required_field_reported(tmp_path: Path) -> None:
    profiles = _profiles_dict()
    del profiles["summarization"]["api_key_env"]
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, _dump(profiles))
    assert "summarization.api_key_env" in str(exc.value)


def test_invalid_api_key_env_names_rejected(tmp_path: Path) -> None:
    for bad in ("sk-ant-fake-key-with-dashes", "not a name", "VAR/NAME"):
        with pytest.raises(ModelsConfigError) as exc:
            _config_from_text(
                tmp_path,
                _yaml_for(tweaks={"normalization": {"api_key_env": bad}}),
            )
        assert "api_key_env" in str(exc.value)


def test_base_url_with_credentials_rejected_without_leaking(tmp_path: Path) -> None:
    secret = "supersecret-password"
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(
            tmp_path,
            _yaml_for(
                tweaks={"normalization": {"base_url": f"https://user:{secret}@api.example.com/v1"}}
            ),
        )
    message = str(exc.value)
    assert "must not embed credentials" in message
    assert secret not in message


def test_invalid_base_urls_rejected(tmp_path: Path) -> None:
    for bad in ("ftp://127.0.0.1/v1", "127.0.0.1:11434/v1", "http://127.0.0.1:99999/v1"):
        with pytest.raises(ModelsConfigError) as exc:
            _config_from_text(tmp_path, _yaml_for(tweaks={"normalization": {"base_url": bad}}))
        assert "base_url" in str(exc.value)


def test_invalid_yaml_syntax_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, "profiles:\n  transcription: [unclosed")
    assert "not valid YAML" in str(exc.value)


def test_missing_profiles_key_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, yaml.safe_dump({"foo": "bar"}))
    assert "profiles" in str(exc.value)


def test_empty_and_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigError):
        _config_from_text(tmp_path, "")
    with pytest.raises(ModelsConfigError):
        _config_from_text(tmp_path, "- just\n- a\n- list")


def test_missing_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigError) as exc:
        load_models_config(tmp_path / "absent.yaml")
    assert "not found" in str(exc.value)


def test_config_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    valid_path = tmp_path / "custom.yaml"
    valid_path.write_text(_yaml_for(), encoding="utf-8")
    monkeypatch.setenv(MODELS_CONFIG_ENV_VAR, str(valid_path))
    config = load_models_config()
    assert isinstance(config.normalization, OpenAICompatibleProfile)

    monkeypatch.setenv(MODELS_CONFIG_ENV_VAR, str(tmp_path / "missing.yaml"))
    with pytest.raises(ModelsConfigError) as exc:
        load_models_config()
    assert str(tmp_path / "missing.yaml") in str(exc.value)


def test_parse_models_config_accepts_plain_dict() -> None:
    raw = yaml.safe_load(_yaml_for())
    config = parse_models_config(raw, source="<test>")
    assert isinstance(config.summarization, OpenAICompatibleProfile)


# ---------------------------------------------------------------------------
# Secret presence and leak prevention
# ---------------------------------------------------------------------------


def _set_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in SECRET_VALUES.items():
        monkeypatch.setenv(name, value)


def test_validate_all_requires_every_declared_secret(
    example_config: ModelsConfig,
) -> None:
    with pytest.raises(ModelsConfigError) as exc:
        validate_all(example_config)
    message = str(exc.value)
    assert "model configuration is not ready" in message
    assert "NORMALIZATION_API_KEY" in message
    assert "SUMMARIZATION_API_KEY" in message
    assert "WELLBEING_API_KEY" in message
    assert "normalization" in message


def test_validate_all_passes_when_secrets_present(
    monkeypatch: pytest.MonkeyPatch, example_config: ModelsConfig
) -> None:
    _set_all_keys(monkeypatch)
    validate_all(example_config)


def test_empty_env_value_counts_as_not_configured(
    monkeypatch: pytest.MonkeyPatch, example_config: ModelsConfig
) -> None:
    for name in API_KEY_NAMES:
        monkeypatch.setenv(name, "   ")
    with pytest.raises(ModelsConfigError) as exc:
        validate_all(example_config)
    assert "NORMALIZATION_API_KEY" in str(exc.value)


def test_profile_status_flags_follow_secret_presence(
    monkeypatch: pytest.MonkeyPatch, example_config: ModelsConfig
) -> None:
    statuses = {status.task: status for status in profile_statuses(example_config)}

    transcription = statuses[TaskType.transcription]
    assert transcription.base_url is None
    assert transcription.api_key_configured is True
    assert transcription.valid is True

    normalization = statuses[TaskType.normalization]
    assert normalization.base_url == "http://127.0.0.1:11434"
    assert normalization.api_key_configured is False
    assert normalization.valid is False

    _set_all_keys(monkeypatch)
    refreshed = {status.task: status for status in profile_statuses(example_config)}
    assert refreshed[TaskType.normalization].api_key_configured is True
    assert refreshed[TaskType.normalization].valid is True
    assert refreshed[TaskType.summarization].valid is True


def test_status_base_url_never_contains_path_or_credentials(
    monkeypatch: pytest.MonkeyPatch, example_config: ModelsConfig
) -> None:
    _set_all_keys(monkeypatch)
    for status in profile_statuses(example_config):
        if status.base_url is not None:
            assert "/v1" not in status.base_url
            assert "@" not in status.base_url
            assert status.provider.value == "openai_compatible"


def test_no_secret_values_in_dumps_or_str(
    monkeypatch: pytest.MonkeyPatch, example_config: ModelsConfig
) -> None:
    _set_all_keys(monkeypatch)
    rendered: list[str] = []
    for status in profile_statuses(example_config):
        rendered.append(str(status))
        rendered.append(repr(status))
        rendered.append(str(status.model_dump()))
        rendered.append(str(status.model_dump(mode="json")))
        rendered.append(status.model_dump_json())
    profiles = [
        example_config.transcription,
        example_config.normalization,
        example_config.summarization,
        example_config.wellbeing_analysis,
    ]
    for profile in profiles:
        rendered.append(str(profile))
        rendered.append(str(profile.model_dump(mode="json")))
        rendered.append(profile.model_dump_json())
    rendered.append(str(example_config))
    rendered.append(example_config.model_dump_json())

    joined = "\n".join(rendered)
    assert "api_key_env" in joined  # names are fine and expected
    for value in SECRET_VALUES.values():
        assert value not in joined


def test_validation_error_never_contains_secret_values(
    monkeypatch: pytest.MonkeyPatch, example_config: ModelsConfig
) -> None:
    # Normalization and wellbeing keys are present; summarization is missing.
    monkeypatch.setenv("NORMALIZATION_API_KEY", SECRET_VALUES["NORMALIZATION_API_KEY"])
    monkeypatch.setenv("WELLBEING_API_KEY", SECRET_VALUES["WELLBEING_API_KEY"])
    with pytest.raises(ModelsConfigError) as exc:
        validate_all(example_config)
    message = str(exc.value)
    assert "SUMMARIZATION_API_KEY" in message
    for value in SECRET_VALUES.values():
        assert value not in message


# ---------------------------------------------------------------------------
# Phase-2 optional profiles (T02)
# ---------------------------------------------------------------------------


def _example_profiles() -> dict[str, dict[str, object]]:
    """The ``profiles`` mapping of the tracked example file, freshly parsed."""
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return raw["profiles"]


def _config_from_profiles(tmp_path: Path, profiles: dict[str, dict[str, object]]) -> ModelsConfig:
    return _config_from_text(tmp_path, yaml.safe_dump({"profiles": profiles}, sort_keys=False))


def _modelfile_parameters(name: str) -> dict[str, str]:
    """``PARAMETER`` name → value for ``ollama/<name>``."""
    path = PROJECT_ROOT / "ollama" / name
    assert path.is_file(), f"missing {path}"
    params: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, value = line.partition(" ")
        if keyword.upper() != "PARAMETER":
            continue
        param_name, _, param_value = value.strip().partition(" ")
        params[param_name] = param_value.strip()
    return params


def test_example_provides_the_four_phase2_profiles(example_config: ModelsConfig) -> None:
    pairs = example_config.phase2_profiles()
    assert [name for name, _ in pairs] == list(PHASE2_FIELDS)
    for name, profile in pairs:
        expected = PHASE2_PROFILES[name]
        assert isinstance(profile, OpenAICompatibleProfile)
        assert profile.provider == "openai_compatible"
        assert profile.base_url == OLLAMA_BASE_URL
        assert profile.model == expected["model"]
        assert profile.api_key_env == "OLLAMA_API_KEY"
        assert profile.temperature == pytest.approx(expected["temperature"])
        assert profile.max_tokens == expected["max_tokens"]
        assert profile.reasoning_effort == expected["reasoning_effort"]
        assert profile.external is False


def test_config_without_phase2_profiles_is_valid(tmp_path: Path) -> None:
    """(a) Backward compatibility: the pre-T02 config still loads.

    Since T13b a config without ``diary_week`` also leaves the weekly layer
    switched off — absent profile, not an error (checked here and in
    ``tests/unit/test_sum_weekly_pipeline.py``).
    """
    profiles = _example_profiles()
    for field in PHASE2_FIELDS:
        profiles.pop(field, None)
    config = _config_from_profiles(tmp_path, profiles)
    assert config.phase2_profiles() == []
    assert config.diary_stt is None
    assert config.diary_sum is None
    assert config.diary_week is None
    assert config.diary_coach is None
    # The four MVP profiles are untouched by the new optional fields.
    assert [task for task, _ in config.profiles_by_task()] == list(TaskType)


def test_week_profile_alone_does_not_bring_the_other_layers_back(tmp_path: Path) -> None:
    """T13b: the example's MVP profiles plus ``diary_week`` only — a valid state."""
    profiles = _example_profiles()
    for other in PHASE2_FIELDS:
        if other != "diary_week":
            profiles.pop(other, None)
    config = _config_from_profiles(tmp_path, profiles)
    assert [name for name, _ in config.phase2_profiles()] == ["diary_week"]
    assert config.diary_stt is None
    assert config.diary_sum is None
    assert config.diary_coach is None


@pytest.mark.parametrize("field", PHASE2_FIELDS)
def test_each_phase2_profile_is_optional_on_its_own(tmp_path: Path, field: str) -> None:
    """(b) Any single phase-2 profile (and no other) is enough to load."""
    profiles = _example_profiles()
    for other in PHASE2_FIELDS:
        if other != field:
            profiles.pop(other, None)
    config = _config_from_profiles(tmp_path, profiles)
    assert [name for name, _ in config.phase2_profiles()] == [field]
    single = getattr(config, field)
    assert isinstance(single, OpenAICompatibleProfile)
    assert single.model == PHASE2_PROFILES[field]["model"]
    for other in PHASE2_FIELDS:
        if other != field:
            assert getattr(config, other) is None


def test_phase2_profile_is_still_validated_when_present(tmp_path: Path) -> None:
    """Optional means "may be absent", not "unchecked"."""
    profiles = _example_profiles()
    profiles["diary_stt"]["max_tokens"] = 0
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_profiles(tmp_path, profiles)
    assert "diary_stt.max_tokens" in str(exc.value)


def test_unknown_key_next_to_phase2_profiles_still_rejected(tmp_path: Path) -> None:
    """(c) ``extra="forbid"`` keeps its grip beside the new optional fields."""
    wrong_name = {"diary_stt_fix": dict(_example_profiles()["diary_stt"])}
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, _yaml_for(tweaks=wrong_name))
    assert "diary_stt_fix" in str(exc.value)


def test_phase2_yaml_temperature_matches_modelfile() -> None:
    """(d) Modelfile ⇄ ``configs/models.example.yaml`` agreement (T02/T13b).

    The profile temperature is what actually goes on the wire
    (``resolve_completion_params``), so a divergence between the YAML and the
    Modelfile ``PARAMETER temperature`` is a silent substitution — this test
    fails on any mismatch, in either direction, and on drift of the decided
    numbers themselves. Since T13b the weekly profile is checked against its own
    Modelfile: ``diary_week`` ⇄ ``ollama/Modelfile.sum-cpu``.
    """
    profiles = _example_profiles()
    for field, expected in PHASE2_PROFILES.items():
        profile = profiles[field]
        params = _modelfile_parameters(str(expected["modelfile"]))
        assert params["num_ctx"] == str(NUM_CTX), f"{field}: num_ctx must be {NUM_CTX}"
        assert float(params["temperature"]) == pytest.approx(profile["temperature"]), (
            f"{field}: Modelfile temperature {params['temperature']} != "
            f"profile temperature {profile['temperature']}"
        )
        assert float(params["temperature"]) == pytest.approx(expected["temperature"])
        assert float(profile["temperature"]) == pytest.approx(expected["temperature"])
        assert profile["model"] == expected["model"]


def test_week_profile_is_the_cpu_twin_of_the_day_profile() -> None:
    """T13b hybrid: day stays on the GPU tag, week runs on the CPU twin.

    The whole point of the separate profile is the device split, so the tags
    themselves are asserted here: ``diary-sum`` (GPU) for the interactive day
    layer, ``diary-sum-cpu`` (``num_gpu 0``) for the background weekly rollup.
    """
    profiles = _example_profiles()
    day = profiles["diary_sum"]
    week = profiles["diary_week"]

    assert day["model"] == "diary-sum"
    assert week["model"] == "diary-sum-cpu"
    assert not str(day["model"]).endswith("-cpu")

    cpu_params = _modelfile_parameters("Modelfile.sum-cpu")
    gpu_params = _modelfile_parameters("Modelfile.sum")
    assert cpu_params["num_gpu"] == "0"
    assert "num_gpu" not in gpu_params

    # Same prompt/temperature family and the same generation cap: the device is
    # the only difference, plus the transport headroom a CPU call needs.
    assert week["temperature"] == day["temperature"]
    assert week["max_tokens"] == day["max_tokens"]
    assert float(week["timeout_s"]) > float(day["timeout_s"])
    assert float(week["timeout_s"]) >= 180.0


def test_phase2_profile_temperature_reaches_the_payload(
    monkeypatch: pytest.MonkeyPatch, example_config: ModelsConfig
) -> None:
    """T02 DoD: the phase-2 profile really reaches the HTTP payload.

    ``httpx.MockTransport`` only — no endpoint is contacted. Payload keys are
    exactly what :meth:`OpenAICompatibleProvider.complete` builds: model,
    temperature, max_tokens and messages.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-local-fake")
    seen: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[str(request.url)] = json.loads(request.content)
        return httpx.Response(200, json=_OK_CHAT_BODY)

    pairs = example_config.phase2_profiles()
    assert [name for name, _ in pairs] == list(PHASE2_FIELDS)
    url = f"{OLLAMA_BASE_URL}/chat/completions"
    for field, profile in pairs:
        seen.clear()
        provider = OpenAICompatibleProvider(profile, transport=httpx.MockTransport(handler))
        provider.complete(LLMRequest.from_prompt("синтетический запрос", prompt_version="v1"))
        assert list(seen) == [url]
        payload = seen[url]
        expected = PHASE2_PROFILES[field]
        assert payload["temperature"] == pytest.approx(expected["temperature"])
        assert payload["model"] == expected["model"]
        assert payload["max_tokens"] == expected["max_tokens"]
        assert payload["reasoning_effort"] == expected["reasoning_effort"]


# ---------------------------------------------------------------------------
# reasoning_effort in the config file (T02b, ADR-009)
# ---------------------------------------------------------------------------


def test_config_without_reasoning_effort_is_valid(tmp_path: Path) -> None:
    """(д) Backward compatibility: a pre-T02b config (no such field anywhere)
    keeps loading, and the resolved default is ``None`` — "not sent"."""
    profiles = _example_profiles()
    for fields in profiles.values():
        fields.pop("reasoning_effort", None)
    config = _config_from_profiles(tmp_path, profiles)

    llm_profiles = [
        profile
        for _, profile in config.profiles_by_task()
        if isinstance(profile, OpenAICompatibleProfile)
    ]
    assert llm_profiles  # the four MVP profiles are still there
    for profile in llm_profiles:
        assert profile.reasoning_effort is None
    for _, profile in config.phase2_profiles():
        assert profile.reasoning_effort is None


def test_mvp_profiles_keep_the_field_unset(example_config: ModelsConfig) -> None:
    """T02b sets the switch for the phase-2 layers only; MVP profiles are
    untouched, so their wire behaviour is byte-for-byte the old one."""
    for task, profile in example_config.profiles_by_task():
        if isinstance(profile, OpenAICompatibleProfile):
            assert profile.reasoning_effort is None, task
        else:
            assert not hasattr(profile, "reasoning_effort"), task


def test_invalid_reasoning_effort_rejected(tmp_path: Path) -> None:
    """(г) An unknown level is a configuration error, reported with its path."""
    for task in ("diary_stt", "normalization"):
        with pytest.raises(ModelsConfigError) as exc:
            _config_from_text(
                tmp_path,
                _yaml_for(tweaks={task: {"reasoning_effort": "minimal"}}),
            )
        assert f"{task}.reasoning_effort" in str(exc.value)


def test_reasoning_effort_reaches_the_payload(
    monkeypatch: pytest.MonkeyPatch, example_config: ModelsConfig
) -> None:
    """(б) The example config's ``reasoning_effort: none`` really goes on the
    wire; a profile without it sends no such key (not ``null``)."""
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-local-fake")
    seen: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[str(request.url)] = json.loads(request.content)
        return httpx.Response(200, json=_OK_CHAT_BODY)

    url = f"{OLLAMA_BASE_URL}/chat/completions"

    def payload_for(profile: OpenAICompatibleProfile) -> dict[str, object]:
        seen.clear()
        provider = OpenAICompatibleProvider(profile, transport=httpx.MockTransport(handler))
        provider.complete(LLMRequest.from_prompt("синтетический запрос", prompt_version="v1"))
        assert list(seen) == [url]
        return seen[url]

    profile = example_config.diary_stt
    assert isinstance(profile, OpenAICompatibleProfile)
    assert payload_for(profile)["reasoning_effort"] == "none"
    assert "reasoning_effort" not in payload_for(
        profile.model_copy(update={"reasoning_effort": None})
    )


# ---------------------------------------------------------------------------
# Second STT backend behind the ``provider`` discriminator (T43, ADR-017)
# ---------------------------------------------------------------------------

#: The revision the T42 spike measured the GigaAM-v2 CTC+LM wrapper at
#: (``docs/stt-backends-analysis.md`` §11). The example config must pin exactly
#: it: the wrapper runs locally through ``trust_remote_code``.
GIGAAM_PINNED_REVISION = "07d97cf8a4f178b68679cafae0600c8487877301"
GIGAAM_MODEL_ID = "waveletdeboshir/gigaam-ctc-with-lm"

GIGAAM_FIELDS: dict[str, object] = {
    "provider": "gigaam2_ctc_lm",
    "model_id": GIGAAM_MODEL_ID,
    "revision": GIGAAM_PINNED_REVISION,
}


def _giga_profiles(overrides: dict[str, object] | None = None) -> dict[str, dict[str, object]]:
    """The default profiles with ``transcription`` switched to GigaAM."""
    profiles = _profiles_dict()
    profiles["transcription"] = dict(GIGAAM_FIELDS, **(overrides or {}))
    return profiles


def test_gigaam_transcription_profile_loads_with_defaults(tmp_path: Path) -> None:
    """(а) DoD 1: the second profile shape loads and its defaults are the decided ones."""
    config = _config_from_profiles(tmp_path, _giga_profiles())
    profile = config.transcription
    assert isinstance(profile, GigaamCtcLmProfile)
    assert profile.provider == "gigaam2_ctc_lm"
    assert profile.model_id == GIGAAM_MODEL_ID
    assert profile.revision == GIGAAM_PINNED_REVISION
    assert profile.trust_remote_code is True
    assert profile.device == "cpu"
    assert profile.beam_width == 64
    # None means "use the checkpoint's own alpha/beta", not a silent 0.0.
    assert profile.alpha is None
    assert profile.beta is None
    assert profile.max_chunk_s == pytest.approx(20.0)
    assert profile.chunk_overlap_s == pytest.approx(0.5)
    assert profile.unload_after is True
    # The provenance accessor names the Hugging Face repository, not ``model``.
    assert transcription_model_name(profile) == GIGAAM_MODEL_ID


def test_gigaam_transcription_profile_reads_explicit_values(tmp_path: Path) -> None:
    config = _config_from_profiles(
        tmp_path,
        _giga_profiles(
            {
                "device": "cuda",
                "beam_width": 8,
                "alpha": 0.5,
                "beta": 0.25,
                "max_chunk_s": 120.0,
                "chunk_overlap_s": 0.0,
                "trust_remote_code": False,
                "unload_after": False,
            }
        ),
    )
    profile = config.transcription
    assert isinstance(profile, GigaamCtcLmProfile)
    assert profile.device == "cuda"
    assert profile.beam_width == 8
    assert profile.alpha == pytest.approx(0.5)
    assert profile.beta == pytest.approx(0.25)
    assert profile.max_chunk_s == pytest.approx(120.0)
    assert profile.chunk_overlap_s == pytest.approx(0.0)
    assert profile.trust_remote_code is False
    assert profile.unload_after is False


def test_unknown_transcription_provider_is_a_load_error(tmp_path: Path) -> None:
    """(б) DoD 1: a third ``provider`` value is refused, not silently downgraded.

    The message must name the offending value **and** the accepted tags, so the
    user can fix the config without reading the source.
    """
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_profiles(tmp_path, _giga_profiles({"provider": "whisper_cpp"}))
    message = str(exc.value)
    assert "transcription" in message
    assert "whisper_cpp" in message
    assert "faster_whisper" in message
    assert "gigaam2_ctc_lm" in message


def test_whisper_only_config_still_loads_unchanged(tmp_path: Path) -> None:
    """(в) DoD 5: a user config with only ``faster_whisper`` needs no edits."""
    config = _config_from_text(tmp_path, _yaml_for())
    profile = config.transcription
    assert isinstance(profile, FasterWhisperProfile)
    assert profile.provider == "faster_whisper"
    assert profile.model == "small"
    assert profile.language == "ru"
    assert profile.compute_type == "int8"
    assert profile.beam_size == 5
    assert transcription_model_name(profile) == "small"


@pytest.mark.parametrize(
    ("max_chunk_s", "chunk_overlap_s"),
    [(10.0, 10.0), (10.0, 11.0)],
)
def test_chunk_overlap_must_be_smaller_than_max_chunk(
    tmp_path: Path, max_chunk_s: float, chunk_overlap_s: float
) -> None:
    """(г) An overlap that removes no audio is a config error naming both values."""
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_profiles(
            tmp_path,
            _giga_profiles({"max_chunk_s": max_chunk_s, "chunk_overlap_s": chunk_overlap_s}),
        )
    message = str(exc.value)
    assert "chunk_overlap_s" in message
    assert "max_chunk_s" in message
    assert str(chunk_overlap_s) in message
    assert str(max_chunk_s) in message


def test_overlap_equal_to_max_chunk_is_rejected(tmp_path: Path) -> None:
    """The boundary is strict: equal values are refused as well."""
    with pytest.raises(ModelsConfigError):
        _config_from_profiles(
            tmp_path, _giga_profiles({"max_chunk_s": 20.0, "chunk_overlap_s": 20.0})
        )


def test_duplicate_transcription_key_is_a_load_error(tmp_path: Path) -> None:
    """D1: two ``transcription:`` blocks are refused — never "the last one wins".

    PyYAML's default keeps the last value and drops the earlier one, so a user
    who uncomments the GigaAM twin and forgets to delete the Whisper block would
    get a silent backend substitution. The file must be rejected instead, and
    the message must name the key without echoing any value.
    """
    text = (
        "profiles:\n"
        "  transcription:\n"
        "    provider: faster_whisper\n"
        "    model: small\n"
        "  transcription:\n"
        "    provider: gigaam2_ctc_lm\n"
        f"    model_id: {GIGAAM_MODEL_ID}\n"
        f"    revision: {GIGAAM_PINNED_REVISION}\n"
        "  normalization:\n"
        "    provider: openai_compatible\n"
        "    base_url: http://127.0.0.1:11434/v1\n"
        "    model: local-normalizer\n"
        "    api_key_env: NORMALIZATION_API_KEY\n"
        "    max_tokens: 100\n"
    )
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, text)
    message = str(exc.value)
    assert "repeats the key 'transcription'" in message
    assert "profiles" in message
    # Privacy: no value from the file may be echoed (model ids, base_url, ...).
    assert GIGAAM_MODEL_ID not in message
    assert GIGAAM_PINNED_REVISION not in message
    assert "waveletdeboshir" not in message
    assert "small" not in message
    assert "127.0.0.1" not in message


def test_duplicate_key_inside_a_profile_is_a_load_error(tmp_path: Path) -> None:
    """The guard is not limited to the backend switch: any repeated key fails.

    The message names the path, so a duplicate deep in a profile is as
    locatable as a second ``transcription:`` block.
    """
    text = _yaml_for().replace("    beam_size: 5", "    beam_size: 5\n    beam_size: 9", 1)
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, text)
    message = str(exc.value)
    assert "repeats the key 'beam_size'" in message
    assert "profiles.transcription" in message


def test_strict_parsing_leaves_well_formed_configs_alone(tmp_path: Path) -> None:
    """D1: the duplicate guard changes nothing for a file with unique keys."""
    config = _config_from_text(tmp_path, _yaml_for())
    assert isinstance(config.transcription, FasterWhisperProfile)
    assert config.normalization.model == "local-normalizer"
    for name in ("models.example.yaml", "models.cpu.example.yaml"):
        loaded = load_models_config(PROJECT_ROOT / "configs" / name)
        assert isinstance(loaded.transcription, FasterWhisperProfile), name
        assert loaded.phase2_profiles(), name


def test_aliases_work_and_an_explicit_key_over_a_merge_is_reported(tmp_path: Path) -> None:
    """The documented nuance of the strict loader (README, ADR-017).

    A plain alias and a merge that overrides nothing keep working; a field given
    both through ``<<`` and explicitly is reported as a repeated key, where
    standard YAML would silently let the explicit key win. Pinned here so the
    wording in the docs and the loader cannot drift apart.
    """
    llm = (
        "  normalization: &norm\n"
        "    provider: openai_compatible\n"
        "    base_url: http://127.0.0.1:11434/v1\n"
        "    model: local-normalizer\n"
        "    api_key_env: NORMALIZATION_API_KEY\n"
        "    max_tokens: 100\n"
    )
    text = (
        "profiles:\n"
        "  transcription:\n"
        "    provider: faster_whisper\n"
        "    model: small\n"
        + llm
        + "  summarization:\n    <<: *norm\n"
        + "  wellbeing_analysis:\n    provider: openai_compatible\n"
        "    base_url: http://127.0.0.1:11434/v1\n"
        "    model: local-wellbeing\n"
        "    api_key_env: WELLBEING_API_KEY\n"
        "    max_tokens: 100\n"
    )
    config = _config_from_text(tmp_path, text)
    assert config.normalization.model == "local-normalizer"
    assert config.summarization.model == "local-normalizer"  # merged, no override

    overridden = text.replace("    <<: *norm\n", "    <<: *norm\n    model: overridden\n", 1)
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, overridden)
    assert "repeats the key 'model'" in str(exc.value)


def test_gigaam_revision_is_required(tmp_path: Path) -> None:
    """(д) The revision pin is mandatory: the wrapper runs ``trust_remote_code``."""
    for revision in (None, ""):
        fields = dict(GIGAAM_FIELDS)
        if revision is None:
            fields.pop("revision")
        else:
            fields["revision"] = revision
        profiles = _profiles_dict()
        profiles["transcription"] = fields
        with pytest.raises(ModelsConfigError) as exc:
            _config_from_profiles(tmp_path, profiles)
        assert "revision" in str(exc.value)


def test_gigaam_rejects_unknown_fields(tmp_path: Path) -> None:
    """``extra="forbid"`` holds for the new profile too (no silent knobs)."""
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_profiles(tmp_path, _giga_profiles({"sample_rate": 16000}))
    assert "sample_rate" in str(exc.value)


def test_example_config_still_uses_whisper_and_documents_gigaam() -> None:
    """(е) DoD 3: both examples load, keep Whisper active and show the twin block."""
    loaded = load_models_config(EXAMPLE_CONFIG)
    assert isinstance(loaded.transcription, FasterWhisperProfile)
    assert loaded.transcription.provider == "faster_whisper"

    blocks: list[list[str]] = []
    for name in ("models.example.yaml", "models.cpu.example.yaml"):
        path = PROJECT_ROOT / "configs" / name
        raw = path.read_text(encoding="utf-8")
        # Both backends are described, and the GigaAM twin is fully specified.
        for needle in (
            "gigaam2_ctc_lm",
            GIGAAM_MODEL_ID,
            GIGAAM_PINNED_REVISION,
            "beam_width",
            "max_chunk_s",
            "chunk_overlap_s",
            "unload_after",
        ):
            assert needle in raw, f"{name}: missing {needle}"
        # The active profile is still Whisper — the file parses as before.
        active = yaml.safe_load(raw)["profiles"]["transcription"]
        assert active["provider"] == "faster_whisper", name
        blocks.append(_commented_gigaam_block(path))
    # The CPU example claims exactly two differences from the main example; the
    # commented GigaAM block must not become a third one.
    assert blocks[0] == blocks[1]
    assert any("provider: gigaam2_ctc_lm" in line for line in blocks[0])


def _commented_gigaam_block(path: Path) -> list[str]:
    """The contiguous commented GigaAM block of an example config file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if "Второй STT-бэкенд" in line)
    block: list[str] = []
    for line in lines[start:]:
        if line and not line.lstrip().startswith("#"):
            break
        block.append(line)
    return block


def _uncommented_gigaam_profile(path: Path) -> dict[str, object]:
    """The commented GigaAM twin of an example config, as real YAML fields.

    Only the profile's own lines are uncommented (prose comments are dropped,
    an inner ``#`` comment stays a comment): the point is to validate the block
    a user is told to paste, not to re-parse the explanation around it.
    """
    lines = _commented_gigaam_block(path)
    start = next(i for i, line in enumerate(lines) if line.strip() == "# transcription:")
    body_lines: list[str] = []
    for line in lines[start + 1 :]:
        text = line.strip()
        if not text.startswith("#"):
            break
        body = text[1:].strip()
        if not body:
            break
        body_lines.append("  " + body)
        if body.startswith("unload_after:"):
            break
    document = yaml.safe_load("transcription:\n" + "\n".join(body_lines))
    assert isinstance(document, dict)
    return document["transcription"]


def test_the_commented_gigaam_twin_is_a_loadable_profile() -> None:
    """The documented second block must be a real, loadable profile.

    A commented example is documentation, and documentation drifts: rename a
    field in ``models_config`` and the block a user is told to paste would fail
    to load. Uncommenting it here and validating it against the live schema is
    what keeps DoD 3 ("switching is the profile's ``provider`` field") true.
    """
    for name in ("models.example.yaml", "models.cpu.example.yaml"):
        path = PROJECT_ROOT / "configs" / name
        profiles = _profiles_dict()
        profiles["transcription"] = _uncommented_gigaam_profile(path)
        config = parse_models_config({"profiles": profiles}, source=name)
        profile = config.transcription
        assert isinstance(profile, GigaamCtcLmProfile)
        assert profile.model_id == GIGAAM_MODEL_ID
        assert profile.revision == GIGAAM_PINNED_REVISION
        assert profile.alpha == pytest.approx(0.4612549346468768)
        assert profile.beta == pytest.approx(0.3271780420615455)
        assert profile.max_chunk_s == pytest.approx(20.0)
        assert profile.chunk_overlap_s < profile.max_chunk_s


def test_gigaam_profile_status_is_valid_without_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(ж) DoD 2: the local GigaAM backend needs no secret and reports valid."""
    for name in API_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    config = _config_from_profiles(tmp_path, _giga_profiles())

    status = status_for(TaskType.transcription, config.transcription)
    assert status.provider is ProviderType.gigaam2_ctc_lm
    assert status.model == GIGAAM_MODEL_ID
    assert status.base_url is None
    assert status.api_key_configured is True
    assert status.valid is True

    statuses = {row.task: row for row in profile_statuses(config)}
    assert statuses[TaskType.transcription].provider is ProviderType.gigaam2_ctc_lm
    # The LLM profiles keep their own (secret-driven) verdicts: switching the
    # STT backend changes nothing for them.
    assert statuses[TaskType.normalization].provider is ProviderType.openai_compatible
    assert statuses[TaskType.normalization].valid is False


def test_validate_all_does_not_trip_over_the_gigaam_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(з) ``validate_all`` must not touch a local STT profile's missing secret.

    Before T43 it skipped only ``FasterWhisperProfile`` and read
    ``api_key_env`` off every other profile — an ``AttributeError`` on GigaAM.
    """
    config = _config_from_profiles(tmp_path, _giga_profiles())

    _set_all_keys(monkeypatch)
    validate_all(config)

    for name in API_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ModelsConfigError) as exc:
        validate_all(config)
    message = str(exc.value)
    assert "NORMALIZATION_API_KEY" in message
    # The transcription task is never named: local STT has no secret at all.
    assert "transcription" not in message
