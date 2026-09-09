"""Unit tests for the transcription provider abstraction (Task 09).

No database and no model are used: faster-whisper is an optional dependency,
and its import is stubbed out to prove that a missing package produces a
clean :class:`ProviderUnavailableError` without attempting any download or
model load, and that runtime failures are sanitised at the adapter boundary.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from diary.core.models_config import FasterWhisperProfile
from diary.domain.transcription import providers


def _profile() -> FasterWhisperProfile:
    return FasterWhisperProfile(
        provider="faster_whisper",
        model="tiny",
        language="ru",
        device="cpu",
        compute_type="int8",
        beam_size=1,
    )


def _audio_path(tmp_path: Path) -> Path:
    return tmp_path / "synthetic-voice.wav"


def test_create_provider_returns_faster_whisper_for_profile() -> None:
    provider = providers.create_provider(_profile())
    assert isinstance(provider, providers.FasterWhisperProvider)


def test_registry_is_the_factory_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing the registry entry swaps the real adapter for a fake one —
    the mechanism integration tests use instead of a GPU/model in CI."""

    class FakeProvider:
        version = "0.0.0"

        def transcribe(self, audio_path: Path, profile: FasterWhisperProfile) -> str:
            return "fake raw text"

    monkeypatch.setitem(providers.PROVIDER_REGISTRY, "faster_whisper", FakeProvider)
    provider = providers.create_provider(_profile())
    assert isinstance(provider, FakeProvider)


def test_create_provider_rejects_empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "PROVIDER_REGISTRY", {})
    with pytest.raises(providers.TranscriptionProviderError, match="registered"):
        providers.create_provider(_profile())


def test_missing_faster_whisper_raises_unavailable_without_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing faster-whisper package must fail cleanly *before* any model is
    constructed or downloaded (the import is stubbed to raise ImportError)."""
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # forces ImportError
    provider = providers.FasterWhisperProvider()
    with pytest.raises(providers.ProviderUnavailableError, match=r"\[stt\]"):
        provider.transcribe(_audio_path(tmp_path), _profile())


def test_execution_error_is_sanitised_at_the_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A decoder failure must not leak its message (it could carry audio/
    transcription internals); only the exception class name is echoed."""

    class _ExplodingModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("sensitive decoder detail about the audio content")

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=_ExplodingModel),
    )
    provider = providers.FasterWhisperProvider()
    with pytest.raises(providers.TranscriptionExecutionError) as excinfo:
        provider.transcribe(_audio_path(tmp_path), _profile())
    assert "sensitive decoder detail" not in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)
