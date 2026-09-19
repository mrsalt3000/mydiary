"""Unit tests for the sequential STT access point (phase 2, T03, role A2).

Hermetic by construction: the provider is a fake installed through the
providers registry seam, the LLM unload is a fake callable, and no
faster-whisper, model, GPU or network is ever touched (the CI convention of
Task 09 — the ``stt`` extra is optional).

What is pinned here:

* the DoD order of a single STT turn — ``llm_unload`` **before** speech-to-text
  and Whisper released **after** it (the shared journal makes the order
  assertable, and the same order is visible in the stage logs);
* the failure policy — a failed LLM unload aborts *before* STT and raises
  :class:`~diary.domain.transcription.asr.LlmUnloadError` (never "continue
  without the unload", which would put Whisper and the LLM into VRAM together);
* resource release — Whisper is unloaded on the success path, on a provider
  failure and on a failed LLM unload; the GPU slot is always released;
* serialisation — two concurrent :func:`~diary.domain.transcription.asr.run`
  calls never transcribe at the same time;
* privacy — the recognised text never reaches the logs of this module.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest

from diary.core.models_config import FasterWhisperProfile
from diary.domain.transcription import asr
from diary.domain.transcription import providers as transcription_providers

#: Stage names of the shared journal (fake unload + fake provider events).
LLM_UNLOAD = "llm_unload"
STT = "stt"
WHISPER_UNLOAD = "whisper_unload"


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
    path = tmp_path / "synthetic-asr.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")
    return path


class FakeProvider:
    """Fake STT provider: journals its calls and returns fixed text.

    Mirrors the phase-09 fake-provider convention (no faster-whisper, no model,
    no GPU) extended with the T03 ``unload`` operation.
    """

    def __init__(
        self,
        journal: list[str],
        *,
        text: str = "сырой текст",
        error: Exception | None = None,
    ) -> None:
        self.journal = journal
        self.text = text
        self.error = error
        self.transcribe_calls: list[tuple[Path, FasterWhisperProfile]] = []
        self.unload_calls = 0

    def transcribe(self, audio_path: Path, profile: FasterWhisperProfile) -> str:
        self.journal.append(STT)
        self.transcribe_calls.append((audio_path, profile))
        if self.error is not None:
            raise self.error
        return self.text

    def unload(self) -> None:
        self.journal.append(WHISPER_UNLOAD)
        self.unload_calls += 1


def _use_provider(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
    """Install ``provider`` through the registry seam (the factory is called
    with no arguments, exactly like a real provider class)."""
    monkeypatch.setitem(
        transcription_providers.PROVIDER_REGISTRY, "faster_whisper", lambda: provider
    )


def _unload_llm(journal: list[str], *, error: Exception | None = None):
    def unload() -> None:
        journal.append(LLM_UNLOAD)
        if error is not None:
            raise error

    return unload


# ---------------------------------------------------------------------------
# The DoD: order of one STT turn
# ---------------------------------------------------------------------------


def test_llm_unload_runs_before_stt_and_whisper_unload_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The DoD of T03: LLM unloaded → STT → Whisper unloaded."""
    journal: list[str] = []
    provider = FakeProvider(journal)
    _use_provider(monkeypatch, provider)

    text = asr.run(_audio_path(tmp_path), _profile(), unload_llm=_unload_llm(journal))

    assert text == "сырой текст"
    assert journal == [LLM_UNLOAD, STT, WHISPER_UNLOAD]


def test_run_returns_the_provider_text_and_passes_audio_and_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal: list[str] = []
    provider = FakeProvider(journal, text="  приведённый текст  ")
    _use_provider(monkeypatch, provider)
    audio_path = _audio_path(tmp_path)
    profile = _profile()

    text = asr.run(audio_path, profile, unload_llm=_unload_llm(journal))

    assert text == "  приведённый текст  "
    assert provider.transcribe_calls == [(audio_path, profile)]


def test_llm_unload_is_called_exactly_once_and_whisper_released_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal: list[str] = []
    provider = FakeProvider(journal)
    _use_provider(monkeypatch, provider)

    asr.run(_audio_path(tmp_path), _profile(), unload_llm=_unload_llm(journal))

    assert journal.count(LLM_UNLOAD) == 1
    assert provider.unload_calls == 1


def test_stage_logs_show_llm_unload_completed_before_stt_started(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The same order is observable in the stage logs (the runtime evidence the
    real-GPU check greps for)."""
    journal: list[str] = []
    _use_provider(monkeypatch, FakeProvider(journal))
    caplog.set_level(logging.INFO, logger="diary.domain.transcription.asr")

    asr.run(_audio_path(tmp_path), _profile(), unload_llm=_unload_llm(journal))

    stages = [
        record.getMessage().split()[1]
        for record in caplog.records
        if "stage=" in record.getMessage()
    ]
    assert stages.index("stage=llm-unload-done") < stages.index("stage=stt-start")
    assert stages.index("stage=stt-done") < stages.index("stage=whisper-unloaded")
    assert "stage=slot-acquired" in stages
    assert "stage=slot-released" in stages


# ---------------------------------------------------------------------------
# Failure policy: a failed LLM unload aborts loudly, before STT
# ---------------------------------------------------------------------------


def test_stt_is_not_started_when_llm_unload_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal: list[str] = []
    provider = FakeProvider(journal)
    _use_provider(monkeypatch, provider)
    failure = RuntimeError("sensitive provider detail")

    with pytest.raises(asr.LlmUnloadError) as excinfo:
        asr.run(_audio_path(tmp_path), _profile(), unload_llm=_unload_llm(journal, error=failure))

    assert STT not in journal
    assert provider.transcribe_calls == []
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    # Loud but safe: the class name is reported, the original message is not.
    assert "RuntimeError" in str(excinfo.value)
    assert "sensitive provider detail" not in str(excinfo.value)


def test_whisper_is_released_even_when_the_llm_unload_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal: list[str] = []
    provider = FakeProvider(journal)
    _use_provider(monkeypatch, provider)

    with pytest.raises(asr.LlmUnloadError):
        asr.run(
            _audio_path(tmp_path),
            _profile(),
            unload_llm=_unload_llm(journal, error=RuntimeError("boom")),
        )

    assert journal == [LLM_UNLOAD, WHISPER_UNLOAD]


def test_whisper_is_released_when_transcription_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal: list[str] = []
    failure = transcription_providers.TranscriptionExecutionError(
        "Speech-to-text failed while transcribing (RuntimeError)."
    )
    provider = FakeProvider(journal, error=failure)
    _use_provider(monkeypatch, provider)

    with pytest.raises(transcription_providers.TranscriptionExecutionError) as excinfo:
        asr.run(_audio_path(tmp_path), _profile(), unload_llm=_unload_llm(journal))

    assert excinfo.value is failure  # provider errors propagate unchanged
    assert journal == [LLM_UNLOAD, STT, WHISPER_UNLOAD]
    assert provider.unload_calls == 1


# ---------------------------------------------------------------------------
# GPU slot: release and serialisation
# ---------------------------------------------------------------------------


def test_the_slot_is_released_after_a_failed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed run must not leave the exclusive slot held (no deadlock later)."""
    journal: list[str] = []
    provider = FakeProvider(journal, error=RuntimeError("decoder exploded"))
    _use_provider(monkeypatch, provider)

    with pytest.raises(RuntimeError):
        asr.run(_audio_path(tmp_path), _profile(), unload_llm=_unload_llm(journal))

    result: list[str] = []

    def _second_run() -> None:
        provider.error = None
        result.append(asr.run(_audio_path(tmp_path), _profile(), unload_llm=_unload_llm(journal)))

    thread = threading.Thread(target=_second_run)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "the ASR slot was not released after a failed run"
    assert result == ["сырой текст"]


def test_concurrent_runs_never_transcribe_at_the_same_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The process-wide slot serialises STT: max one in-flight transcription."""
    start_barrier = threading.Barrier(2)
    guard = threading.Lock()
    state = {"active": 0, "max_active": 0}

    class ConcurrencyProbe(FakeProvider):
        def transcribe(self, audio_path: Path, profile: FasterWhisperProfile) -> str:
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.15)  # a real transcription takes far longer than this
            with guard:
                state["active"] -= 1
            return super().transcribe(audio_path, profile)

    provider = ConcurrencyProbe([])
    _use_provider(monkeypatch, provider)
    audio_path = _audio_path(tmp_path)

    def _call() -> None:
        start_barrier.wait(timeout=5.0)  # both threads call ``run`` together
        asr.run(audio_path, _profile(), unload_llm=lambda: None)

    threads = [threading.Thread(target=_call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert state["max_active"] == 1
    assert len(provider.transcribe_calls) == 2  # both runs completed, serialised


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def test_recognised_text_never_reaches_the_module_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    journal: list[str] = []
    secret_text = "содержимое дневника не должно попасть в лог"
    _use_provider(monkeypatch, FakeProvider(journal, text=secret_text))
    caplog.set_level(logging.DEBUG, logger="diary.domain.transcription.asr")

    asr.run(_audio_path(tmp_path), _profile(), unload_llm=_unload_llm(journal))

    assert secret_text not in caplog.text
    assert str(tmp_path) not in caplog.text
