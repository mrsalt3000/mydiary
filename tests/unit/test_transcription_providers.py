"""Unit tests for the transcription provider abstraction (Task 09).

No database and no model are used: faster-whisper is an optional dependency,
and its import is stubbed out to prove that a missing package produces a
clean :class:`ProviderUnavailableError` without attempting any download or
model load, and that runtime failures are sanitised at the adapter boundary.

The GigaAM sidecar adapter (T44) is driven through an injected **runner** and a
fake **process**: no interpreter, no sidecar virtualenv and no GPU are involved,
and the two tests that start a real child use ``sys.executable`` with a trivial
script (no network, no model).
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from diary.core.models_config import FasterWhisperProfile, GigaamCtcLmProfile
from diary.core.settings import AppSettings
from diary.domain.transcription import gigaam, providers


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


def test_unregistered_backend_is_a_permanent_refusal() -> None:
    """T43/ADR-017 + T44 DoD 6: an unknown tag is refused, never retried.

    The profile union cannot express a third ``provider`` value, so the stand-in
    carries only what the factory reads (``provider``). The refusal is a
    :class:`ProviderConfigurationError` whose ``retryable`` flag is ``False`` —
    the property the worker boundary reads, instead of a list of classes that a
    new error could silently fall out of (the T43 acceptance found the factory's
    refusal being retried although no retry can help).
    """
    unknown = types.SimpleNamespace(provider="whisper_cpp")
    with pytest.raises(providers.ProviderConfigurationError, match="whisper_cpp") as excinfo:
        providers.create_provider(unknown)  # type: ignore[arg-type]
    assert excinfo.value.retryable is False


def test_gigaam_backend_is_registered_behind_the_same_factory() -> None:
    """T44: ``gigaam2_ctc_lm`` has an adapter, reached through the same registry."""
    provider = providers.create_provider(_giga_profile())
    assert isinstance(provider, providers.GigaamSidecarProvider)
    assert providers.PROVIDER_REGISTRY["gigaam2_ctc_lm"] is providers.GigaamSidecarProvider


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


# ---------------------------------------------------------------------------
# GigaAM sidecar adapter (T44)
# ---------------------------------------------------------------------------
# Everything here is hermetic: the runner is injected, the process is fake, and
# the settings point at files this test creates. Nothing imports torch, spawns a
# sidecar virtualenv or touches the network.

_GIGAAM_REVISION = "07d97cf8a4f178b68679cafae0600c8487877301"


@pytest.fixture(autouse=True)
def _isolate_gigaam_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests independent of a developer's own APP_GIGAAM_* variables."""
    for name in (
        "APP_GIGAAM_PYTHON",
        "APP_GIGAAM_WORKER_SCRIPT",
        "APP_GIGAAM_WORKER_TIMEOUT_S",
        "APP_GIGAAM_MODEL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


class _FakeProcess:
    """The process surface the adapter uses (``SidecarProcess``)."""

    def __init__(self, returncode: int | None = 0) -> None:
        self.returncode = returncode
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        assert self.returncode is not None, "wait() on a process that never exited"
        return self.returncode


class _RefusingProcess(_FakeProcess):
    """A process whose ``terminate()`` raises: the pid was not there (T45).

    Modelled on ``ProcessLookupError``, which is exactly what the real
    ``Popen.terminate()`` raises when the child is already gone or has been
    reaped — the case in which "the signal was refused" must not be read as
    "the process is dead".
    """

    def terminate(self) -> None:
        self.terminate_calls += 1
        raise ProcessLookupError("synthetic: the signal did not land")


class _StubbornProcess(_FakeProcess):
    """A process that survives ``terminate()`` (T45): only ``kill()`` ends it.

    ``wait`` is inherited, so while the process is alive it raises instead of
    returning — standing in for ``subprocess.TimeoutExpired`` after the grace
    period.
    """

    def terminate(self) -> None:
        self.terminate_calls += 1  # sent, and ignored by the child

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


class _FakeRunner:
    """A runner that returns a scripted outcome and records its arguments."""

    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        process: _FakeProcess | None = None,
        error: Exception | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.process = process if process is not None else _FakeProcess(returncode)
        self.error = error
        self.calls: list[tuple[list[str], dict[str, str], float]] = []

    def run(self, argv, *, env, timeout_s):  # type: ignore[no-untyped-def]
        self.calls.append((list(argv), dict(env), timeout_s))
        if self.error is not None:
            raise self.error
        self.process.returncode = self.returncode
        return gigaam.SidecarRun(process=self.process, stdout=self.stdout, stderr=self.stderr)


def _result_json(text: str = "сырой текст", *, gap: float = 0.0, double: float = 0.0) -> str:
    return json.dumps({"text": text, "coverage": {"gap_s": gap, "double_s": double}})


def _gigaam_settings(tmp_path: Path, **overrides: object) -> AppSettings:
    """Settings whose interpreter and worker script exist (so only the runner is faked)."""
    interpreter = tmp_path / "sidecar-venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n")
    script = tmp_path / "gigaam-stt-worker.py"
    script.write_text("#!/usr/bin/env python\n")
    defaults: dict[str, object] = {
        "gigaam_python": interpreter,
        "gigaam_worker_script": script,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)


def _sidecar(settings: AppSettings, runner: _FakeRunner) -> gigaam.GigaamSidecarProvider:
    return gigaam.GigaamSidecarProvider(settings=settings, runner=runner)


def test_gigaam_transcribe_returns_the_text_and_passes_the_profile(
    tmp_path: Path,
) -> None:
    """The happy path: one JSON result in, the recognised text out."""
    runner = _FakeRunner(stdout=_result_json("сырой текст") + "\n")
    settings = _gigaam_settings(tmp_path)
    provider = _sidecar(settings, runner)

    text = provider.transcribe(_audio_path(tmp_path), _giga_profile())

    assert text == "сырой текст"
    assert len(runner.calls) == 1
    argv, env, timeout_s = runner.calls[0]
    assert argv[0] == str(settings.gigaam_python)
    assert argv[1] == str(settings.gigaam_worker_script)
    assert argv[2] == str(_audio_path(tmp_path))
    for flag, value in (
        ("--model-id", "waveletdeboshir/gigaam-ctc-with-lm"),
        ("--revision", _GIGAAM_REVISION),
        ("--device", "cpu"),
        ("--beam-width", "64"),
        ("--max-chunk-s", "20.0"),
        ("--chunk-overlap-s", "0.5"),
    ):
        assert argv[argv.index(flag) + 1] == value
    # alpha/beta are absent when the profile does not set them: the worker then
    # keeps the checkpoint's own values instead of receiving a substituted one.
    assert "--alpha" not in argv
    assert "--beta" not in argv
    # The child can never start a download inside a transcription request.
    assert env["HF_HUB_OFFLINE"] == "1"
    assert timeout_s == settings.gigaam_worker_timeout_s


def test_gigaam_alpha_and_beta_are_sent_only_when_the_profile_sets_them(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner(stdout=_result_json())
    provider = _sidecar(_gigaam_settings(tmp_path), runner)
    profile = GigaamCtcLmProfile(
        provider="gigaam2_ctc_lm",
        model_id="waveletdeboshir/gigaam-ctc-with-lm",
        revision=_GIGAAM_REVISION,
        alpha=0.4612549346468768,
        beta=0.3271780420615455,
    )

    provider.transcribe(_audio_path(tmp_path), profile)

    argv = runner.calls[0][0]
    assert argv[argv.index("--alpha") + 1] == repr(0.4612549346468768)
    assert argv[argv.index("--beta") + 1] == repr(0.3271780420615455)


def test_gigaam_missing_interpreter_is_unavailable_before_any_process(tmp_path: Path) -> None:
    """No interpreter configured: refuse without starting anything (DoD 1)."""
    runner = _FakeRunner(stdout=_result_json())
    settings = _gigaam_settings(tmp_path, gigaam_python=None)
    provider = _sidecar(settings, runner)

    with pytest.raises(providers.ProviderUnavailableError, match="APP_GIGAAM_PYTHON"):
        provider.transcribe(_audio_path(tmp_path), _giga_profile())

    assert runner.calls == []
    assert providers.ProviderUnavailableError.retryable is False


def test_gigaam_missing_worker_script_is_unavailable(tmp_path: Path) -> None:
    runner = _FakeRunner(stdout=_result_json())
    settings = _gigaam_settings(tmp_path, gigaam_worker_script=tmp_path / "absent-worker.py")
    provider = _sidecar(settings, runner)

    with pytest.raises(providers.ProviderUnavailableError, match="APP_GIGAAM_WORKER_SCRIPT"):
        provider.transcribe(_audio_path(tmp_path), _giga_profile())

    assert runner.calls == []


def test_gigaam_unload_after_false_is_an_explicit_configuration_error(tmp_path: Path) -> None:
    """The switch the sidecar cannot honour is refused, not silently ignored."""
    runner = _FakeRunner(stdout=_result_json())
    provider = _sidecar(_gigaam_settings(tmp_path), runner)
    profile = GigaamCtcLmProfile(
        provider="gigaam2_ctc_lm",
        model_id="waveletdeboshir/gigaam-ctc-with-lm",
        revision=_GIGAAM_REVISION,
        unload_after=False,
    )

    with pytest.raises(providers.ProviderConfigurationError, match="unload_after") as excinfo:
        provider.transcribe(_audio_path(tmp_path), profile)

    assert runner.calls == []
    assert excinfo.value.retryable is False


def test_gigaam_trust_remote_code_false_is_refused(tmp_path: Path) -> None:
    """The wrapper cannot run without executing its own code; say so, don't pretend."""
    runner = _FakeRunner(stdout=_result_json())
    provider = _sidecar(_gigaam_settings(tmp_path), runner)
    profile = GigaamCtcLmProfile(
        provider="gigaam2_ctc_lm",
        model_id="waveletdeboshir/gigaam-ctc-with-lm",
        revision=_GIGAAM_REVISION,
        trust_remote_code=False,
    )

    with pytest.raises(providers.ProviderConfigurationError, match="trust_remote_code"):
        provider.transcribe(_audio_path(tmp_path), profile)
    assert runner.calls == []


def test_gigaam_provider_refuses_a_foreign_profile(tmp_path: Path) -> None:
    """The adapter serves its own provider only; the runner is never asked."""
    runner = _FakeRunner(stdout=_result_json())
    provider = _sidecar(_gigaam_settings(tmp_path), runner)

    with pytest.raises(providers.TranscriptionProviderError, match="faster_whisper"):
        provider.transcribe(_audio_path(tmp_path), _profile())

    assert runner.calls == []


def test_gigaam_runtime_failure_is_transient_and_sanitised(tmp_path: Path) -> None:
    """A non-zero exit stays transient and echoes only the worker's class name."""
    poisoned = "секретный текст дневника"
    runner = _FakeRunner(returncode=1, stderr=f"{poisoned}\nRuntimeError\n")
    provider = _sidecar(_gigaam_settings(tmp_path), runner)

    with pytest.raises(providers.TranscriptionExecutionError) as excinfo:
        provider.transcribe(_audio_path(tmp_path), _giga_profile())

    assert "RuntimeError" in str(excinfo.value)
    assert poisoned not in str(excinfo.value)
    assert excinfo.value.retryable is True


def test_gigaam_unavailable_exit_is_a_permanent_refusal(tmp_path: Path) -> None:
    """Missing weights/packages: exit 3 is permanent, and named in the message."""
    runner = _FakeRunner(returncode=3, stderr="LocalEntryNotFoundError\n")
    provider = _sidecar(_gigaam_settings(tmp_path), runner)

    with pytest.raises(providers.ProviderUnavailableError, match="LocalEntryNotFoundError"):
        provider.transcribe(_audio_path(tmp_path), _giga_profile())


def test_gigaam_malformed_stdout_is_a_permanent_failure(tmp_path: Path) -> None:
    """A successful exit we cannot read is deterministic, so it is not retried.

    T45 DoD 6c: the worker exited 0, therefore what arrived is a function of the
    recording and the profile — repeating the request repeats the same payload.
    Retrying would only burn the job's attempt budget.
    """
    runner = _FakeRunner(stdout="это не JSON\n")
    provider = _sidecar(_gigaam_settings(tmp_path), runner)

    with pytest.raises(providers.TranscriptionResultError) as excinfo:
        provider.transcribe(_audio_path(tmp_path), _giga_profile())

    assert "не JSON" not in str(excinfo.value)
    assert excinfo.value.retryable is False


def test_gigaam_nonzero_coverage_is_a_permanent_refusal(tmp_path: Path) -> None:
    """A plan that does not cover the recording is a failed run, never a transcript.

    The plan depends on the file and the profile only, so the same request repeats
    it: the classification is *permanent* (T45 DoD 6c), which is what stops the
    worker from spending all three attempts on a deterministic defect.
    """
    runner = _FakeRunner(stdout=_result_json(gap=1.0))
    provider = _sidecar(_gigaam_settings(tmp_path), runner)

    with pytest.raises(providers.TranscriptionResultError, match="does not cover") as excinfo:
        provider.transcribe(_audio_path(tmp_path), _giga_profile())

    assert excinfo.value.retryable is False


def test_gigaam_unavailable_message_names_the_provisioning_scripts(tmp_path: Path) -> None:
    """T45 DoD 2: a missing runtime says what to *run*, not just that it is missing."""
    runner = _FakeRunner(stdout=_result_json())
    settings = _gigaam_settings(tmp_path, gigaam_python=None)
    provider = _sidecar(settings, runner)

    with pytest.raises(providers.ProviderUnavailableError) as excinfo:
        provider.transcribe(_audio_path(tmp_path), _giga_profile())

    message = str(excinfo.value)
    assert "scripts/setup-gigaam-venv.sh" in message
    assert "scripts/fetch-gigaam-model.sh" in message
    assert "APP_GIGAAM_MODEL_DIR" in message


def test_gigaam_child_env_is_an_allow_list_with_the_provisioned_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T45 DoD 8: the sidecar gets a minimal environment, not the application's own.

    The child is a different program: a locale (it prints UTF-8 JSON), a home, a
    ``PATH``, a scratch directory, the GPU selection and a library path — and
    nothing else. An exported LLM API key, the case the T44 acceptance named, must
    not reach it at all; and ``HF_HOME`` must be the provisioned cache rather than
    whatever the user has exported.
    """
    monkeypatch.setenv("NORMALIZATION_API_KEY", "синтетический-ключ")
    monkeypatch.setenv("HF_HOME", "/home/someone-else/.cache/huggingface")
    monkeypatch.setenv("APP_GIGAAM_MODEL_DIR", str(tmp_path / "weights"))
    runner = _FakeRunner(stdout=_result_json())
    settings = _gigaam_settings(tmp_path)
    provider = _sidecar(settings, runner)

    provider.transcribe(_audio_path(tmp_path), _giga_profile())

    env = runner.calls[0][1]
    allowed = set(gigaam._ENV_ALLOW_LIST) | {
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    }
    assert set(env) <= allowed
    assert "NORMALIZATION_API_KEY" not in env
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert env["HF_HOME"] == str(settings.gigaam_model_dir)
    assert env["HF_HOME"] != os.environ["HF_HOME"]


def test_gigaam_timeout_is_transient(tmp_path: Path) -> None:
    runner = _FakeRunner(error=gigaam.SidecarTimeoutError("synthetic timeout"))
    provider = _sidecar(_gigaam_settings(tmp_path), runner)

    with pytest.raises(gigaam.SidecarTimeoutError) as excinfo:
        provider.transcribe(_audio_path(tmp_path), _giga_profile())

    assert excinfo.value.retryable is True


def test_unload_terminates_a_sidecar_that_outlived_its_result(tmp_path: Path) -> None:
    """The lifecycle guarantee: a live sidecar is terminated, and only once."""
    process = _FakeProcess(returncode=None)  # still running
    provider = _sidecar(_gigaam_settings(tmp_path), _FakeRunner())
    provider._process = process

    provider.unload()

    assert process.terminate_calls == 1
    assert process.returncode is not None

    provider.unload()  # idempotent: the handle was dropped, no second signal

    assert process.terminate_calls == 1


def test_unload_escalates_to_kill_when_terminate_raises(tmp_path: Path) -> None:
    """T45 DoD 6a: a refused signal must not leave the sidecar alive.

    ``terminate()`` raising says only that the pid was not there at that instant,
    so it is not evidence the process died. The release guarantee ``asr.run``
    leans on (ADR-010) therefore continues to ``kill()``.
    """
    process = _RefusingProcess(returncode=None)
    provider = _sidecar(_gigaam_settings(tmp_path), _FakeRunner())
    provider._process = process

    provider.unload()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.returncode == -9
    assert provider._process is None


def test_unload_kills_a_sidecar_that_outlives_the_grace_period(tmp_path: Path) -> None:
    """T45 DoD 6a: a sidecar that ignores the request is killed, and reaped."""
    process = _StubbornProcess(returncode=None)
    provider = _sidecar(_gigaam_settings(tmp_path), _FakeRunner())
    provider._process = process

    provider.unload()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.returncode == -9
    assert process.wait_calls  # the child was reaped, not just signalled
    assert provider._process is None


def test_unload_is_a_noop_for_a_finished_sidecar(tmp_path: Path) -> None:
    """The normal path: the worker printed its JSON and exited on its own."""
    runner = _FakeRunner(stdout=_result_json())
    provider = _sidecar(_gigaam_settings(tmp_path), runner)

    provider.transcribe(_audio_path(tmp_path), _giga_profile())
    assert provider._process is runner.process

    provider.unload()

    assert runner.process.terminate_calls == 0
    assert provider._process is None


def test_unload_without_a_process_is_a_noop(tmp_path: Path) -> None:
    provider = _sidecar(_gigaam_settings(tmp_path), _FakeRunner())

    provider.unload()

    assert provider._process is None


def test_subprocess_runner_returns_the_child_output() -> None:
    """The real runner, on a trivial child: no model, no GPU, no network."""
    runner = gigaam.SubprocessRunner()

    run = runner.run(
        [sys.executable, "-c", "import sys; print('{}'); sys.stderr.write('note\\n')"],
        env=dict(os.environ),
        timeout_s=60.0,
    )

    assert run.process.returncode == 0
    assert run.stdout.strip() == "{}"
    assert "note" in run.stderr


def test_subprocess_runner_kills_the_child_on_timeout() -> None:
    """A hung child is killed and the attempt fails as a transient error."""
    runner = gigaam.SubprocessRunner()

    with pytest.raises(gigaam.SidecarTimeoutError, match="within"):
        runner.run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=dict(os.environ),
            timeout_s=0.5,
        )
