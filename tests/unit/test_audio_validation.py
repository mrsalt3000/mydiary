"""Unit tests for the pure audio-upload validation helpers (Task 07).

Only in-memory helpers are exercised here (no DB, no filesystem): MIME
normalization, MIME→extension mapping and the container-signature checks.
The storage/cleanup behaviour is covered by the integration tests.

Payload bytes are synthetic (magic-byte-valid containers only, never real
diary recordings).
"""

from __future__ import annotations

import pytest

from diary.core.settings import AppSettings
from diary.domain.audio.service import (
    SUPPORTED_AUDIO_MIME_TYPES,
    AudioTooLargeError,
    EmptyAudioError,
    InvalidAudioContentError,
    UnsupportedAudioTypeError,
    extension_for_mime,
    normalize_mime,
    validate_upload,
)

WAV_MIME = "audio/wav"
WEBM_MIME = "audio/webm"


def _wav_bytes() -> bytes:
    return b"RIFF" + b"\x24\x00\x00\x00" + b"WAVE" + b"\x00" * 24


def _webm_bytes() -> bytes:
    return b"\x1a\x45\xdf\xa3" + bytes(64)


def _ogg_bytes() -> bytes:
    return b"OggS" + bytes(64)


def _mp3_bytes() -> bytes:
    return b"ID3" + bytes(64)


def _mp4_bytes() -> bytes:
    return b"\x00\x00\x00\x18ftypM4A " + bytes(64)


def _not_audio_bytes() -> bytes:
    return b"plain text that is definitely not audio" * 4


@pytest.fixture(autouse=True)
def _isolate_audio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "APP_MAX_AUDIO_MB",
        "APP_ALLOWED_AUDIO_MIME_TYPES",
        "APP_TIMEZONE",
        "APP_DATA_ROOT",
        "APP_AUDIO_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: object) -> AppSettings:
    return AppSettings(_env_file=None, **overrides)


# ---------------------------------------------------------------------------
# MIME normalization and extension mapping
# ---------------------------------------------------------------------------


def test_normalize_mime_strips_parameters_and_case() -> None:
    assert normalize_mime("audio/webm;codecs=opus") == "audio/webm"
    assert normalize_mime("  AUDIO/WAV ") == "audio/wav"
    assert normalize_mime("audio/ogg; codecs=opus") == "audio/ogg"
    assert normalize_mime(None) is None
    assert normalize_mime("") is None
    assert normalize_mime("   ") is None


def test_extension_mapping_is_safe_and_fixed() -> None:
    assert extension_for_mime("audio/webm;codecs=opus") == "webm"
    assert extension_for_mime("audio/ogg") == "ogg"
    assert extension_for_mime("audio/wav") == "wav"
    assert extension_for_mime("audio/x-wav") == "wav"
    assert extension_for_mime("audio/mpeg") == "mp3"
    assert extension_for_mime("audio/mp4") == "m4a"
    assert extension_for_mime("audio/unknown") is None
    assert extension_for_mime(None) is None


def test_supported_mime_types_cover_the_default_allowlist() -> None:
    defaults = AppSettings(_env_file=None).allowed_audio_mime_types
    assert set(defaults) <= SUPPORTED_AUDIO_MIME_TYPES


# ---------------------------------------------------------------------------
# validate_upload
# ---------------------------------------------------------------------------


def test_valid_wav_passes_validation() -> None:
    mime = validate_upload(WAV_MIME, _wav_bytes())
    assert mime == WAV_MIME


@pytest.mark.parametrize(
    ("content_type", "data_builder"),
    [
        (WEBM_MIME, _webm_bytes),
        ("audio/ogg", _ogg_bytes),
        ("audio/mpeg", _mp3_bytes),
        ("audio/mp4", _mp4_bytes),
    ],
)
def test_container_signatures_pass(content_type: str, data_builder) -> None:
    assert validate_upload(content_type, data_builder()) == content_type


def test_rejects_mime_outside_allowlist() -> None:
    with pytest.raises(UnsupportedAudioTypeError):
        validate_upload("text/plain", _wav_bytes())
    with pytest.raises(UnsupportedAudioTypeError):
        validate_upload("audio/flac", _wav_bytes())  # not in the default allowlist


def test_rejects_allowlisted_mime_without_storage_profile() -> None:
    settings = _settings(allowed_audio_mime_types=["audio/flac"])
    with pytest.raises(UnsupportedAudioTypeError):
        validate_upload("audio/flac", _webm_bytes(), settings=settings)


def test_rejects_empty_payload() -> None:
    with pytest.raises(EmptyAudioError):
        validate_upload(WAV_MIME, b"")


def test_rejects_payload_above_limit() -> None:
    settings = _settings(max_audio_size_mb=1)
    with pytest.raises(AudioTooLargeError):
        validate_upload(WAV_MIME, b"\x00" * (1024 * 1024 + 1), settings=settings)


def test_rejects_content_not_matching_declared_type() -> None:
    with pytest.raises(InvalidAudioContentError):
        validate_upload(WAV_MIME, _webm_bytes())  # WebM bytes, WAV declared
    with pytest.raises(InvalidAudioContentError):
        validate_upload(WAV_MIME, _not_audio_bytes())
