"""Unit tests for the audio-related settings fields (Task 07)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from diary.core.settings import AppSettings

_AUDIO_ENV_VARS = (
    "APP_MAX_AUDIO_MB",
    "APP_ALLOWED_AUDIO_MIME_TYPES",
)


@pytest.fixture(autouse=True)
def _isolate_audio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AUDIO_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: object) -> AppSettings:
    return AppSettings(_env_file=None, **overrides)


def test_default_audio_settings() -> None:
    settings = _settings()
    assert settings.max_audio_size_mb == 25
    assert settings.max_audio_size_bytes == 25 * 1024 * 1024
    assert settings.allowed_audio_mime_types == [
        "audio/webm",
        "audio/ogg",
        "audio/wav",
        "audio/mpeg",
        "audio/mp4",
    ]


def test_env_override_reads_app_max_audio_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MAX_AUDIO_MB", "3")
    settings = AppSettings(_env_file=None)
    assert settings.max_audio_size_mb == 3
    assert settings.max_audio_size_bytes == 3 * 1024 * 1024


def test_env_override_reads_allowed_mime_types_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ALLOWED_AUDIO_MIME_TYPES", '["audio/webm", "audio/wav"]')
    settings = AppSettings(_env_file=None)
    assert settings.allowed_audio_mime_types == ["audio/webm", "audio/wav"]


def test_allowlist_is_normalized_and_deduplicated() -> None:
    settings = _settings(allowed_audio_mime_types=["  audio/WAV ", "audio/wav", "AUDIO/OGG"])
    assert settings.allowed_audio_mime_types == ["audio/wav", "audio/ogg"]


def test_invalid_max_audio_mb_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(max_audio_size_mb=0)


def test_allowlist_rejects_mime_with_parameters_and_empties() -> None:
    with pytest.raises(ValidationError, match="without parameters"):
        _settings(allowed_audio_mime_types=["audio/webm;codecs=opus"])
    with pytest.raises(ValidationError, match="empty entries"):
        _settings(allowed_audio_mime_types=["audio/webm", "   "])
    with pytest.raises(ValidationError, match="At least one"):
        _settings(allowed_audio_mime_types=[])
