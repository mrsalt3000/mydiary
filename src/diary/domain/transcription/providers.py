"""Transcription provider abstraction and the local faster-whisper adapter (Task 09).

Design (``docs/architecture-brief.md`` §8.2–8.3, ``skills/local-stt``):

- :class:`TranscriptionProvider` — the ``transcribe(audio_path, profile) -> raw_text``
  protocol implemented by every adapter;
- :class:`FasterWhisperProvider` — a thin adapter around the *optional*
  ``faster-whisper`` package. The import happens lazily inside
  :meth:`FasterWhisperProvider.transcribe`, so installing the ``stt`` extra is
  never required to run the application; without the package the adapter raises
  :class:`ProviderUnavailableError` before any model is constructed or
  downloaded;
- :func:`create_provider` — a small registry-based factory over the profile.
  Tests replace this function (or the registry entry) with a fake provider, so
  no model or GPU is ever needed in CI.

Privacy contract (AGENTS.md §7–8): raw audio and transcription text are never
logged by this module. Every exception it raises carries only a static,
non-sensitive message — never diary content, file bytes or provider payloads —
so ``jobs.error_summary`` stays clean at the worker boundary.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Protocol

from diary.core.models_config import FasterWhisperProfile

#: Faster-whisper distribution name used by ``importlib.metadata``.
_FASTER_WHISPER_DIST = "faster-whisper"

#: Static installation hint shown in the provider-unavailable error.
_INSTALL_HINT = "Install the optional dependency with 'pip install -e \".[stt]\"'."


class TranscriptionProviderError(Exception):
    """Base class for transcription-provider errors.

    Messages are always static and never contain diary content or secrets.
    """


class ProviderUnavailableError(TranscriptionProviderError):
    """The configured provider runtime (faster-whisper) is not installed.

    Raised before any model is loaded or downloaded; the job fails permanently
    at the worker boundary (retrying would never help without an explicit
    install).
    """


class TranscriptionExecutionError(TranscriptionProviderError):
    """The provider failed at runtime; the message is a sanitised summary.

    Only the exception *class name* is echoed — never the original message,
    which could carry audio/decoder internals.
    """


class TranscriptionProvider(Protocol):
    """Transcribes one local audio file into raw text (never diary-adjacent
    side effects; the caller persists the result)."""

    def transcribe(self, audio_path: Path, profile: FasterWhisperProfile) -> str: ...


class FasterWhisperProvider:
    """Local speech-to-text adapter around the optional ``faster-whisper``.

    Settings come from the Task-03 ``transcription`` profile (``FasterWhisperProfile``):
    model size/name, language (``ru`` by default), ``cpu``/``cuda`` device,
    compute type and beam size. ``cpu``/``int8`` are the defaults, so a plain
    CPU machine works out of the box once the package and the model are present.
    """

    def __init__(self) -> None:
        #: Installed faster-whisper distribution version, when resolvable
        #: (metadata lookup only — it never imports or loads the package).
        self.version: str | None = None
        try:
            self.version = _distribution_version(_FASTER_WHISPER_DIST)
        except PackageNotFoundError:
            self.version = None

    def transcribe(self, audio_path: Path, profile: FasterWhisperProfile) -> str:
        """Transcribe ``audio_path`` and return the raw text.

        Raises :class:`ProviderUnavailableError` when faster-whisper is not
        installed (nothing is downloaded or loaded in that case) and
        :class:`TranscriptionExecutionError` for any runtime failure (message
        is static and carries only the exception class name).
        """
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via a stubbed import
            raise ProviderUnavailableError(
                "faster-whisper is not installed; local transcription is unavailable. "
                f"{_INSTALL_HINT}"
            ) from exc
        try:
            model = WhisperModel(
                profile.model,
                device=profile.device,
                compute_type=profile.compute_type,
            )
            segments, _info = model.transcribe(
                str(audio_path),
                language=profile.language,
                beam_size=profile.beam_size,
            )
            return "".join(segment.text for segment in segments).strip()
        except Exception as exc:  # decoder/model errors are sanitised here
            raise TranscriptionExecutionError(
                f"Speech-to-text failed while transcribing ({type(exc).__name__})."
            ) from exc


#: Concrete provider classes, keyed by ``FasterWhisperProfile.provider``.
PROVIDER_REGISTRY: dict[str, type[TranscriptionProvider]] = {
    "faster_whisper": FasterWhisperProvider,
}


def create_provider(profile: FasterWhisperProfile) -> TranscriptionProvider:
    """Return a provider instance for ``profile`` (registry-based factory).

    The registry is a module-level hook: tests monkeypatch it (or this
    function) with a fake provider, so transcription tests never download a
    model or touch a GPU.
    """
    provider_cls = PROVIDER_REGISTRY.get(profile.provider)
    if provider_cls is None:
        raise TranscriptionProviderError(
            f"No transcription provider is registered for {profile.provider!r}."
        )
    return provider_cls()


__all__ = [
    "FasterWhisperProvider",
    "PROVIDER_REGISTRY",
    "ProviderUnavailableError",
    "TranscriptionExecutionError",
    "TranscriptionProvider",
    "TranscriptionProviderError",
    "create_provider",
]
