"""Unit tests for model configuration loading/validation and diagnostics (Task 03).

Synthetic only: no network calls and no real API keys. Secrets are simulated
with obviously fake values to prove they never leak into dumps or errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from diary.core.models_config import (
    MODELS_CONFIG_ENV_VAR,
    FasterWhisperProfile,
    ModelsConfig,
    ModelsConfigError,
    OpenAICompatibleProfile,
    TaskType,
    load_models_config,
    parse_models_config,
)
from diary.core.profile_status import (
    profile_statuses,
    validate_all,
)
from diary.core.settings import PROJECT_ROOT

EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "models.example.yaml"

API_KEY_NAMES = (
    "NORMALIZATION_API_KEY",
    "SUMMARIZATION_API_KEY",
    "WELLBEING_API_KEY",
)

SECRET_VALUES = {
    "NORMALIZATION_API_KEY": "sk-normalization-fake-1111",
    "SUMMARIZATION_API_KEY": "sk-summarization-fake-2222",
    "WELLBEING_API_KEY": "sk-wellbeing-fake-3333",
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
    assert "transcription.modle" in str(exc.value)


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
    assert "transcription.provider" in str(exc.value)


def test_invalid_values_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(tmp_path, _yaml_for(tweaks={"transcription": {"device": "tpu"}}))
    assert "transcription.device" in str(exc.value)

    with pytest.raises(ModelsConfigError) as exc:
        _config_from_text(
            tmp_path,
            _yaml_for(tweaks={"transcription": {"compute_type": "float64"}}),
        )
    assert "transcription.compute_type" in str(exc.value)

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
