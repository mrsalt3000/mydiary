"""Sequential speech-to-text access point for the phase-2 layers (T03, role A2).

One invariant lives here and nowhere else:

    **Whisper is never resident in VRAM at the same time as the LLM.**

The 6 GiB laptop GPU cannot host both, so ``docs/tasklist-phase2.md`` §2 makes
"STT and LLM in VRAM at once" a hard veto. :func:`run` enforces the *order* of
a single STT turn:

1. take the exclusive process-wide GPU slot (:data:`_GPU_SLOT`);
2. **unload the LLM first** by calling the caller-supplied ``unload_llm``
   callable — the LLM tool is deliberately *not* named or imported here.
   Dependency injection keeps this module free of any Ollama/HTTP assumption
   and lets a test assert the order with a fake callable;
3. run ``provider.transcribe(...)``;
4. release Whisper (:meth:`~diary.domain.transcription.providers.TranscriptionProvider.unload`);
5. release the slot.

Steps 2–4 run inside a ``try/finally``, so a failed transcription still
releases Whisper, and a failed unload still releases everything that was
acquired. The order is what the task's DoD asks for: ``run`` returns only after
Whisper is unloaded, so the caller's next LLM call can never overlap it.

Failure policy (deliberate): if ``unload_llm`` raises, STT is **not** started
and :class:`LlmUnloadError` propagates. Proceeding "without the unload" would
mean transcribing while the LLM is still on the GPU — exactly the veto case —
so a loud failure is the only safe behaviour. The message is static and carries
only the exception class name, never the original message (AGENTS.md §7–8).

Scope and deliberate simplifications (documented, not forgotten)
==============================================================

* **The slot is a process-level** :class:`threading.Lock`, not a file lock, an
  IPC primitive or a GPU semaphore. The MVP is single-user and runs exactly one
  worker process, and that worker executes jobs one at a time
  (``diary.workers.runner.process_due_jobs`` is sequential), so cross-process
  arbitration is unnecessary today. The lock still makes the invariant exact for
  any in-process caller (future threaded callers, tests). A plain ``Lock`` is
  used rather than ``RLock``: :func:`run` is not reentrant by contract, so a
  nested call is a programming error instead of a silently nested STT.
* **The slot serialises STT against STT.** It cannot stop a *different* thread
  from calling the LLM while :func:`run` is in progress; keeping LLM traffic away
  from the STT window is the caller's (orchestrator T09) responsibility, backed
  by the runtime configuration ``OLLAMA_MAX_LOADED_MODELS=1`` /
  ``OLLAMA_NUM_PARALLEL=1`` (``docs/tasklist-phase2.md`` §9.10).
* **Wiring**: :func:`run` is the entry point the phase-2 orchestrator (T09) and
  the manual STT path are meant to share; the phase-1 ``run_transcription``
  service is intentionally left untouched by T03 (it constructs a provider per
  call and never holds a model across calls).

Privacy (AGENTS.md §7–8): stage logs carry only stage names, the profile's
model/device and durations — never audio, its path or the recognised text.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from diary.core.models_config import FasterWhisperProfile
from diary.domain.transcription.providers import create_provider

logger = logging.getLogger("diary.domain.transcription.asr")

#: The single process-wide STT slot (see the module docstring for the rationale).
_GPU_SLOT = threading.Lock()


class AsrError(Exception):
    """Base class for sequential-STT errors (messages are always static)."""


class LlmUnloadError(AsrError):
    """The injected LLM-unload callable failed, so STT was not started.

    Loud by design: continuing without a confirmed unload would put Whisper and
    the LLM into VRAM together (phase-2 veto). The message names only the
    original exception *class*.
    """


def run(
    audio_path: Path,
    profile: FasterWhisperProfile,
    *,
    unload_llm: Callable[[], None],
) -> str:
    """Transcribe one audio file with the LLM unloaded before and after.

    ``unload_llm`` is supplied by the caller (dependency injection) and must
    release the LLM from the GPU synchronously; it is called exactly once per
    :func:`run`, before any speech-to-text work. The provider comes from the
    profile through the regular registry factory
    (:func:`~diary.domain.transcription.providers.create_provider`), so tests
    substitute a fake provider and never need faster-whisper, a model or a GPU.

    Returns the raw recognised text (the caller persists it — this function has
    no database side effects). Raises :class:`LlmUnloadError` when the injected
    unload fails (no STT started) and lets provider errors
    (``ProviderUnavailableError`` / ``TranscriptionExecutionError``) propagate
    unchanged so the worker can keep mapping them to permanent/transient
    failures. A provider whose ``unload()`` raises (a protocol violation) also
    propagates: the caller must then treat the turn as *not* finished rather
    than move on to the LLM, because Whisper may still be resident.
    """
    wait_started = time.monotonic()
    logger.info("asr stage=slot-wait model=%s device=%s", profile.model, profile.device)
    with _GPU_SLOT:
        logger.info(
            "asr stage=slot-acquired model=%s wait_ms=%d",
            profile.model,
            _elapsed_ms(wait_started),
        )
        provider = create_provider(profile)
        try:
            unload_started = time.monotonic()
            logger.info("asr stage=llm-unload-start")
            try:
                unload_llm()
            except Exception as exc:
                raise LlmUnloadError(
                    "The LLM unload step failed "
                    f"({type(exc).__name__}); speech-to-text was not started, so "
                    "Whisper and the LLM can never be resident at the same time."
                ) from exc
            logger.info("asr stage=llm-unload-done duration_ms=%d", _elapsed_ms(unload_started))
            stt_started = time.monotonic()
            logger.info("asr stage=stt-start model=%s device=%s", profile.model, profile.device)
            text = provider.transcribe(audio_path, profile)
            logger.info("asr stage=stt-done duration_ms=%d", _elapsed_ms(stt_started))
        finally:
            # Runs on success, on a provider failure and on a failed LLM unload:
            # whatever was loaded must leave the GPU before this slot is free.
            provider.unload()
            logger.info("asr stage=whisper-unloaded")
    logger.info("asr stage=slot-released total_ms=%d", _elapsed_ms(wait_started))
    return text


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since the ``time.monotonic()`` instant ``started``."""
    return int((time.monotonic() - started) * 1000.0)


__all__ = ["AsrError", "LlmUnloadError", "run"]
