"""Transcription provider abstraction and the local faster-whisper adapter (Task 09).

Design (``docs/architecture-brief.md`` §8.2–8.3, ``skills/local-stt``):

- :class:`TranscriptionProvider` — the ``transcribe(audio_path, profile) -> raw_text``
  protocol implemented by every adapter, plus the explicit
  ``unload() -> None`` resource-release operation (phase 2, T03) that
  :func:`diary.domain.transcription.asr.run` calls after every STT run so
  Whisper is out of VRAM before the LLM is used again;
- :class:`FasterWhisperProvider` — a thin adapter around the *optional*
  ``faster-whisper`` package. The import happens lazily inside
  :meth:`FasterWhisperProvider.transcribe`, so installing the ``stt`` extra is
  never required to run the application; without the package the adapter raises
  :class:`ProviderUnavailableError` before any model is constructed or
  downloaded;
- :class:`~diary.domain.transcription.gigaam.GigaamSidecarProvider` — the second
  backend (T44), which drives an isolated sidecar process instead of a library in
  this interpreter (``docs/stt-backends-analysis.md`` §10). It is imported here
  only to register it: the adapter itself imports no application dependency
  beyond settings and the profile schema;
- :func:`create_provider` — a small registry-based factory over the profile.
  Tests replace this function (or the registry entry) with a fake provider, so
  no model or GPU is ever needed in CI. Since T43 the profile is a discriminated
  union: the factory dispatches on ``provider``, and an unregistered backend
  raises instead of falling back to Whisper.

The error taxonomy (including which failures are retryable) lives in
:mod:`diary.domain.transcription.errors`, and is re-exported here because this
module has always been its public face.

Privacy contract (AGENTS.md §7–8): raw audio and transcription text are never
logged by this module. Every exception it raises carries only a static,
non-sensitive message — never diary content, file bytes or provider payloads —
so ``jobs.error_summary`` stays clean at the worker boundary.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Any, Protocol

from diary.core.models_config import FasterWhisperProfile, TranscriptionProfile
from diary.domain.transcription.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
    TranscriptionExecutionError,
    TranscriptionProviderError,
    TranscriptionResultError,
)
from diary.domain.transcription.gigaam import GigaamSidecarProvider

logger = logging.getLogger("diary.domain.transcription.providers")

#: Faster-whisper distribution name used by ``importlib.metadata``.
_FASTER_WHISPER_DIST = "faster-whisper"

#: Static installation hint shown in the provider-unavailable error.
_INSTALL_HINT = "Install the optional dependency with 'pip install -e \".[stt]\"'."


class TranscriptionProvider(Protocol):
    """Transcribes one local audio file into raw text (never diary-adjacent
    side effects; the caller persists the result)."""

    def transcribe(self, audio_path: Path, profile: TranscriptionProfile) -> str: ...

    def unload(self) -> None:
        """Release every model/resource the provider holds (idempotent).

        Called by :func:`diary.domain.transcription.asr.run` after a run so the
        GPU veto "Whisper and the LLM are never resident at the same time"
        holds (``docs/tasklist-phase2.md`` §2). A provider that never loaded
        anything must make this a no-op and must not raise.
        """
        ...


def _release_library_model(model: Any) -> None:
    """Best-effort ``ctranslate2`` release hook of a loaded Whisper model.

    ``faster_whisper.WhisperModel`` keeps the actual CTranslate2 model in
    ``.model``, which exposes ``unload_model()``. The hook is optional: any
    other model shape (or a newer API) simply skips it — the caller still drops
    its own reference, which is what guarantees the object becomes collectable.
    A hook failure never propagates: it must not turn a successful
    transcription into a failed job.
    """
    unload_model = getattr(getattr(model, "model", None), "unload_model", None)
    if not callable(unload_model):
        return
    try:
        unload_model()
    except Exception as exc:  # pragma: no cover - third-party hook failure
        logger.warning("whisper model release hook failed (%s).", type(exc).__name__)


class FasterWhisperProvider:
    """Local speech-to-text adapter around the optional ``faster-whisper``.

    Settings come from the Task-03 ``transcription`` profile (``FasterWhisperProfile``):
    model size/name, language (``ru`` by default), ``cpu``/``cuda`` device,
    compute type and beam size. ``cpu``/``int8`` are the defaults, so a plain
    CPU machine works out of the box once the package and the model are present.

    Lifecycle: the model is loaded lazily on the first :meth:`transcribe` call
    and cached on the instance, so :meth:`unload` has something concrete to
    release. Nothing is loaded at construction time and nothing is downloaded
    before a real transcription is requested.
    """

    def __init__(self) -> None:
        #: Installed faster-whisper distribution version, when resolvable
        #: (metadata lookup only — it never imports or loads the package).
        self.version: str | None = None
        try:
            self.version = _distribution_version(_FASTER_WHISPER_DIST)
        except PackageNotFoundError:
            self.version = None
        #: Cached faster-whisper model (an untyped third-party object);
        #: ``None`` means nothing is loaded.
        self._model: Any = None
        #: Cache key ``(model, device, compute_type)`` of :attr:`_model`.
        self._model_key: tuple[str, str, str] | None = None

    def transcribe(self, audio_path: Path, profile: TranscriptionProfile) -> str:
        """Transcribe ``audio_path`` and return the raw text.

        The model is loaded on first use and kept until :meth:`unload` (or a
        profile change) — this is what makes an explicit unload a real
        operation instead of a no-op. The returned text and the error surface
        are unchanged from the pre-T03 adapter.

        The parameter is the whole transcription union (a registry adapter must
        accept what the factory may hand it); this adapter serves the
        ``faster_whisper`` provider only and refuses any other profile loudly —
        the factory never routes another backend here, so that branch is a
        programming-error guard, not a fallback.

        Raises :class:`ProviderUnavailableError` when faster-whisper is not
        installed (nothing is downloaded or loaded in that case) and
        :class:`TranscriptionExecutionError` for any runtime failure (message
        is static and carries only the exception class name).
        """
        if not isinstance(profile, FasterWhisperProfile):
            raise TranscriptionProviderError(
                "FasterWhisperProvider serves the 'faster_whisper' provider only; "
                f"got {profile.provider!r}."
            )
        model = self._model_for(profile)
        try:
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

    def unload(self) -> None:
        """Release the loaded Whisper model and every reference to it.

        Idempotent and never raises. Called by
        :func:`diary.domain.transcription.asr.run` after each run so the
        phase-2 GPU veto (Whisper and the LLM must not be resident at the same
        time) is honoured before the next LLM call; the CTranslate2 release
        hook is best effort, while dropping this adapter's reference is the
        hard guarantee that the model object becomes collectable (the historical
        refcount-only path had no explicit release at all — see
        ``docs/tasklist-phase2.md`` §10 "Известный пробел T03").

        Imports nothing: ``unload()`` is safe on a machine without the optional
        ``stt`` extra.
        """
        model, self._model = self._model, None
        self._model_key = None
        if model is None:
            return
        _release_library_model(model)
        del model

    def _model_for(self, profile: FasterWhisperProfile) -> Any:
        """The loaded Whisper model for ``profile``, loading it when needed.

        The ``faster_whisper`` import is lazy: without the package this raises
        :class:`ProviderUnavailableError` *before* any download, model
        construction or GPU work. A profile change (model/device/compute type)
        releases the previously loaded model first, so at most one Whisper model
        is ever resident in this adapter.
        """
        key = (profile.model, profile.device, profile.compute_type)
        if self._model is not None and self._model_key == key:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised via a stubbed import
            raise ProviderUnavailableError(
                "faster-whisper is not installed; local transcription is unavailable. "
                f"{_INSTALL_HINT}"
            ) from exc
        self.unload()
        try:
            model = WhisperModel(
                profile.model,
                device=profile.device,
                compute_type=profile.compute_type,
            )
        except Exception as exc:
            raise TranscriptionExecutionError(
                f"Speech-to-text failed while loading the model ({type(exc).__name__})."
            ) from exc
        self._model = model
        self._model_key = key
        return model


#: Concrete provider classes, keyed by the ``provider`` discriminator of a
#: :data:`~diary.core.models_config.TranscriptionProfile`.
PROVIDER_REGISTRY: dict[str, type[TranscriptionProvider]] = {
    "faster_whisper": FasterWhisperProvider,
    "gigaam2_ctc_lm": GigaamSidecarProvider,
}


def create_provider(profile: TranscriptionProfile) -> TranscriptionProvider:
    """Return a provider instance for ``profile`` (registry-based factory).

    The registry is a module-level hook: tests monkeypatch it (or this
    function) with a fake provider, so transcription tests never download a
    model or touch a GPU.

    A provider that is not registered is a **loud error**, not a fallback: a
    configuration naming a backend with no adapter fails here, before any model
    is loaded, instead of silently running Whisper (ADR-017 §4). The refusal is
    a :class:`~diary.domain.transcription.errors.ProviderConfigurationError` —
    deliberately a *non-retryable* error: the configuration cannot change
    between two attempts of the same job, so retrying would only burn the
    attempt budget (T43 acceptance, T44).

    Where that lands in the call order is the caller's business, and the two
    callers differ: :func:`~diary.domain.transcription.service.run_transcription`
    resolves the provider **before** it enters
    :func:`~diary.domain.transcription.asr.run`, so its refusal is visible before
    the GPU slot; ``asr.run`` itself calls the factory **inside** the slot and
    releases the slot in its ``finally``, so a direct call takes and frees the
    slot before failing.
    """
    provider_cls = PROVIDER_REGISTRY.get(profile.provider)
    if provider_cls is None:
        raise ProviderConfigurationError(
            f"No transcription provider is registered for {profile.provider!r}."
        )
    return provider_cls()


__all__ = [
    "FasterWhisperProvider",
    "GigaamSidecarProvider",
    "PROVIDER_REGISTRY",
    "ProviderConfigurationError",
    "ProviderUnavailableError",
    "TranscriptionExecutionError",
    "TranscriptionProvider",
    "TranscriptionProviderError",
    "TranscriptionResultError",
    "create_provider",
]
