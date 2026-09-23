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

from diary.core.models_config import FasterWhisperProfile, GigaamCtcLmProfile
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


def _giga_profile() -> GigaamCtcLmProfile:
    return GigaamCtcLmProfile(
        provider="gigaam2_ctc_lm",
        model_id="waveletdeboshir/gigaam-ctc-with-lm",
        revision="07d97cf8a4f178b68679cafae0600c8487877301",
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


def test_unregistered_gigaam_backend_fails_loudly_without_falling_back() -> None:
    """T43/ADR-017: an unimplemented backend is an error, never a Whisper run.

    The GigaAM-v2 CTC+LM profile parses (T43) but has no adapter until T44, so
    the factory must refuse — the alternative, silently transcribing with
    Whisper, would both mislabel the provenance and ignore the configured
    backend.
    """
    with pytest.raises(providers.TranscriptionProviderError, match="gigaam2_ctc_lm"):
        providers.create_provider(_giga_profile())


def test_faster_whisper_provider_refuses_a_foreign_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The adapter serves its own provider only; no package is even imported."""
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # any import would fail
    provider = providers.FasterWhisperProvider()
    with pytest.raises(providers.TranscriptionProviderError, match="gigaam2_ctc_lm"):
        provider.transcribe(_audio_path(tmp_path), _giga_profile())


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


# ---------------------------------------------------------------------------
# Explicit model lifecycle: caching + unload (phase 2, T03)
# ---------------------------------------------------------------------------
# Before T03 the adapter built a WhisperModel per call and had no release
# operation at all, so ``asr.run()`` ("Whisper unloaded before the LLM is
# called") had nothing to call. These tests pin the replacement contract with a
# stubbed faster-whisper: the model is loaded once, reused while the profile
# stays the same, and ``unload()`` releases it (library hook + dropped
# reference) — still without the real package, a model file or a GPU.


class _StubCTranslate2Model:
    """The ``faster_whisper.WhisperModel.model`` surface we release."""

    def __init__(self) -> None:
        self.unload_calls = 0
        self.fail_unload = False

    def unload_model(self) -> None:
        self.unload_calls += 1
        if self.fail_unload:
            raise RuntimeError("ct2 release hook failed")


class _StubSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubWhisperModel:
    """Minimal stand-in for ``faster_whisper.WhisperModel``."""

    instances: list[_StubWhisperModel] = []

    def __init__(self, model_size: str, *, device: str, compute_type: str) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = _StubCTranslate2Model()
        self.transcribe_calls = 0
        _StubWhisperModel.instances.append(self)

    def transcribe(self, audio: str, *, language: str, beam_size: int):
        self.transcribe_calls += 1
        return iter([_StubSegment(" Привет"), _StubSegment(", мир")]), object()


@pytest.fixture
def stub_whisper(monkeypatch: pytest.MonkeyPatch) -> list[_StubWhisperModel]:
    """Install the stub package and return the list of constructed models."""
    _StubWhisperModel.instances = []
    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=_StubWhisperModel)
    )
    return _StubWhisperModel.instances


def test_transcribe_loads_once_and_reuses_the_model_for_the_same_profile(
    stub_whisper: list[_StubWhisperModel], tmp_path: Path
) -> None:
    provider = providers.FasterWhisperProvider()

    assert provider.transcribe(_audio_path(tmp_path), _profile()) == "Привет, мир"
    assert provider.transcribe(_audio_path(tmp_path), _profile()) == "Привет, мир"

    assert len(stub_whisper) == 1
    assert stub_whisper[0].transcribe_calls == 2
    assert stub_whisper[0].model.unload_calls == 0


def test_profile_change_releases_the_previous_model(
    stub_whisper: list[_StubWhisperModel], tmp_path: Path
) -> None:
    provider = providers.FasterWhisperProvider()
    provider.transcribe(_audio_path(tmp_path), _profile())
    other = FasterWhisperProfile(
        provider="faster_whisper",
        model="base",
        language="ru",
        device="cuda",
        compute_type="float16",
        beam_size=1,
    )

    provider.transcribe(_audio_path(tmp_path), other)

    assert len(stub_whisper) == 2
    assert stub_whisper[0].model.unload_calls == 1
    assert provider._model is stub_whisper[1]


def test_unload_releases_the_model_and_is_idempotent(
    stub_whisper: list[_StubWhisperModel], tmp_path: Path
) -> None:
    provider = providers.FasterWhisperProvider()
    provider.transcribe(_audio_path(tmp_path), _profile())

    provider.unload()

    assert provider._model is None
    assert provider._model_key is None
    assert stub_whisper[0].model.unload_calls == 1

    provider.unload()  # idempotent: nothing loaded, no error, no extra release

    assert stub_whisper[0].model.unload_calls == 1


def test_unload_drops_the_reference_even_when_the_library_hook_fails(
    stub_whisper: list[_StubWhisperModel], tmp_path: Path
) -> None:
    """A failing library hook must not raise (it would turn a successful
    transcription into a failed job) — and must not keep the model either."""
    provider = providers.FasterWhisperProvider()
    provider.transcribe(_audio_path(tmp_path), _profile())
    stub_whisper[0].model.fail_unload = True

    provider.unload()

    assert provider._model is None
    assert provider._model_key is None


def test_unload_without_a_loaded_model_never_imports_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``unload()`` must work on a machine without the optional ``stt`` extra."""
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # forces ImportError
    provider = providers.FasterWhisperProvider()

    provider.unload()

    assert provider._model is None
