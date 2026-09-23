"""GigaAM-v2 CTC+LM adapter driven through an isolated sidecar process (T44).

The backend lives in its **own virtualenv** and its **own process**
(``docs/stt-backends-analysis.md`` §10, ADR-017 §7): measured, ``unload()``
returns VRAM but not RAM (``VmHWM`` 4553.2 MiB), the wrapper runs through
``trust_remote_code``, and the application venv has neither torch nor numpy. So
"transcribe a file" here means "start ``scripts/gigaam-stt-worker.py`` with that
venv's interpreter, hand it the audio path and the decoding parameters, read one
JSON object back, and let the process end".

Lifecycle
=========

* the runtime is resolved from **settings**, not from the model profile
  (``APP_GIGAAM_PYTHON``, ``APP_GIGAAM_WORKER_SCRIPT``, ``APP_GIGAAM_MODEL_DIR``):
  a profile describes what to decode, never where a machine keeps its
  interpreters and its weights (T38: a path a source set keeps its value). The
  two provisioning scripts are the way those paths get filled:
  ``scripts/setup-gigaam-venv.sh`` builds the interpreter,
  ``scripts/fetch-gigaam-model.sh`` downloads the pinned weights into
  ``APP_GIGAAM_MODEL_DIR`` and verifies their ``sha256`` (T45);
* missing interpreter, missing worker script or a worker that reports an
  unusable runtime (no weights in the provisioned cache) raise
  :class:`~diary.domain.transcription.errors.ProviderUnavailableError`
  **before** anything is loaded or downloaded — the adapter runs the child with
  ``HF_HUB_OFFLINE=1`` and points its ``HF_HOME`` at the provisioned cache, so a
  transcription request can neither start a multi-gigabyte download nor read a
  copy of the model the user happens to keep elsewhere;
* ``unload_after: false`` is refused loudly rather than ignored. The phase-2 veto
  forbids keeping STT resident in VRAM next to the LLM, and the sidecar's way of
  releasing a model is to exit; there is no resident mode to switch on, so the
  flag would otherwise be a control that does nothing;
* :meth:`unload` terminates a sidecar that is still alive and is idempotent and
  total (it never raises). The worker prints its JSON as its last line and exits,
  so in the normal path the process is already gone when ``unload`` runs and the
  release is structural; the termination path is the guarantee behind
  ``asr.run``'s "release after every run" (ADR-010) for a worker that outlives
  its result (a hung interpreter teardown, a timeout that killed nothing). A
  ``terminate()`` that raises does not end the attempt: the escalation goes on to
  ``kill()``, because a signal that never landed is not evidence the process died.

Privacy (AGENTS.md §7–8): no exception here carries diary content, an audio
path, or the worker's stdout/stderr text — only a static message plus, at most,
the *class name* the worker put last on stderr. The worker's stdout is never
logged; the recognised text travels from it into the ``transcriptions`` table
and nowhere else. This module imports nothing heavy: the sidecar's own venv
carries the runtime and the child process isolates ``trust_remote_code``.

The child's environment is an **allow-list**, not this process' environment minus
secrets (T45, ``docs/privacy-and-safety.md`` §8): the sidecar is a different
program that needs a locale (it prints UTF-8 JSON), a home, a ``PATH``, a scratch
directory and the GPU selection — and nothing else. The names of the LLM API keys
the user may have exported are therefore not inherited at all, which is a
narrower rule than a deny-list of "variables that look secret": the child cannot
read a key that was never passed, and a diary secret named something we did not
think of cannot slip through.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from diary.core.models_config import GigaamCtcLmProfile, TranscriptionProfile
from diary.core.settings import AppSettings, get_settings
from diary.domain.transcription import sidecar_protocol
from diary.domain.transcription.chunking import COVERAGE_TOLERANCE_S
from diary.domain.transcription.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
    TranscriptionExecutionError,
    TranscriptionProviderError,
    TranscriptionResultError,
)

#: How long a terminated sidecar is given to exit before it is killed.
_TERMINATE_GRACE_S = 5.0

#: The child's environment always carries this: a transcription request must not
#: start a download (the weights are provisioned outside the application, T45),
#: and an offline cache miss fails fast instead of fetching gigabytes.
_OFFLINE_ENV = "HF_HUB_OFFLINE"

#: Set together with the offline pin: an offline child should not even try to
#: report anything to a remote service (defence in depth — with
#: ``HF_HUB_OFFLINE=1`` the library does not open a connection at all).
_TELEMETRY_ENV = "HF_HUB_DISABLE_TELEMETRY"

#: Environment variables the sidecar child inherits — the whole of it (T45). The
#: sidecar is a different program: it needs a locale (it prints a UTF-8 JSON with
#: the recognised text), a home (library defaults land under it), a ``PATH``, a
#: scratch directory, the GPU selection and a library path. Everything else is
#: deliberately **not** passed, so the LLM API keys the user exported cannot
#: reach it (``docs/privacy-and-safety.md`` §8). ``HF_HOME``/``HF_HUB_OFFLINE``
#: are not listed because the adapter sets them itself, to the provisioned cache.
#: ``LD_LIBRARY_PATH`` *is* listed: on this machine it carries the cuBLAS shim
#: that ctranslate2 needs (a property of the application's environment, not a
#: secret), and dropping it would make the child depend on where its CUDA
#: libraries happen to live.
_ENV_ALLOW_LIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
)

#: Last stderr line of a failed worker is the error class name — a plain
#: identifier (optionally dotted), never a message.
_CLASS_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_MAX_CLASS_NAME = 128

#: Static installation hint shown when the sidecar runtime cannot be used. It
#: names the two provisioning scripts (T45 DoD 2): a missing runtime is a
#: user-fixable condition, and the message should say what to run rather than
#: leave the user inside somebody else's library traceback.
_PROVISION_HINT = (
    "Provision the GigaAM sidecar runtime before using the gigaam2_ctc_lm backend: "
    "scripts/setup-gigaam-venv.sh builds its interpreter and "
    "scripts/fetch-gigaam-model.sh downloads the pinned weights; then point "
    "APP_GIGAAM_PYTHON (and, if the weights went elsewhere, APP_GIGAAM_MODEL_DIR) at them."
)


class SidecarTimeoutError(TranscriptionExecutionError):
    """The sidecar did not finish inside the configured timeout.

    Transient by contract (a busy machine may be faster next time); the child
    process is killed before this is raised, so no half-dead worker survives the
    failed attempt.
    """


class SidecarProcess(Protocol):
    """The slice of ``subprocess.Popen`` this adapter uses.

    A protocol rather than the concrete type so the tests can drive the adapter
    with a fake process (no interpreter, no sidecar venv, no GPU): the adapter
    must be able to terminate a live process it started.
    """

    returncode: int | None

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


@dataclass(frozen=True)
class SidecarRun:
    """One finished (or timed-out) sidecar invocation."""

    process: SidecarProcess
    stdout: str
    stderr: str


class SidecarRunner(Protocol):
    """Starts one sidecar process and returns what it wrote."""

    def run(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout_s: float
    ) -> SidecarRun: ...


class SubprocessRunner:
    """The production runner: ``subprocess``, no shell, separate pipes, timeout."""

    def run(self, argv: Sequence[str], *, env: Mapping[str, str], timeout_s: float) -> SidecarRun:
        """Run ``argv`` to completion; kill it on timeout.

        ``shell=False`` is not a detail: the audio path comes from the database
        and must never reach a shell. Both pipes are separate, so the recognised
        text (stdout) and the diagnostics (stderr) stay apart.
        """
        try:
            process: subprocess.Popen[str] = subprocess.Popen(
                list(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(env),
                shell=False,
            )
        except OSError as exc:
            raise TranscriptionExecutionError(
                f"The GigaAM sidecar could not be started ({type(exc).__name__})."
            ) from exc
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise SidecarTimeoutError(
                f"The GigaAM sidecar did not finish within {timeout_s:g} s."
            ) from exc
        return SidecarRun(process=process, stdout=stdout or "", stderr=stderr or "")


def _send_signal(send: Callable[[], None]) -> bool:
    """Send one signal to the sidecar; ``False`` when the call itself failed.

    Signal delivery is a request, not a fact: ``terminate()`` raises when the pid
    is not there (or is not ours), and that says nothing about whether the process
    is still running. So a caller that must guarantee the release cannot treat a
    failed signal as a finished job — it has to escalate (see ``unload``), and
    this helper is what makes that readable.
    """
    try:
        send()
    except Exception:
        return False
    return True


def _reported_class(stderr: str) -> str | None:
    """The worker's error *class name*, from the last non-empty stderr line.

    Only a bare identifier is accepted: the adapter never echoes free-form stderr
    text (it could carry a path or a decoder payload), so anything else is
    reported as an unknown failure.
    """
    for line in reversed(stderr.splitlines()):
        token = line.strip()
        if not token:
            continue
        if _CLASS_NAME_RE.match(token) and len(token) <= _MAX_CLASS_NAME:
            return token
        return None
    return None


def _parse_result(stdout: str) -> dict[str, object]:
    """Parse the worker's last stdout line; validate the fields the row needs.

    A malformed result is a failed run, never a partially trusted one: raising
    here means no transcript is stored (AGENTS.md §15 applies the same rule to
    structured LLM output). Every failure in this function is
    :class:`~diary.domain.transcription.errors.TranscriptionResultError`, that
    is *permanent*: the worker has already exited successfully, so what we are
    looking at is a function of the recording and the profile — the identical
    request produces the identical unusable payload, and retrying only burns the
    job's attempts. A runtime failure (a killed or crashed worker) never reaches
    this function: it arrives as a non-zero exit code earlier.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise TranscriptionResultError("The GigaAM sidecar produced no result.")
    try:
        payload = json.loads(lines[-1])
    except ValueError as exc:
        raise TranscriptionResultError("The GigaAM sidecar result is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise TranscriptionResultError("The GigaAM sidecar result is not a JSON object.")
    if not isinstance(payload.get(sidecar_protocol.RESULT_KEY_TEXT), str):
        raise TranscriptionResultError("The GigaAM sidecar result carries no recognised text.")
    coverage = payload.get(sidecar_protocol.RESULT_KEY_COVERAGE)
    if not isinstance(coverage, dict):
        raise TranscriptionResultError("The GigaAM sidecar result carries no chunk coverage.")
    gap = coverage.get(sidecar_protocol.COVERAGE_KEY_GAP_S)
    double = coverage.get(sidecar_protocol.COVERAGE_KEY_DOUBLE_S)
    if not isinstance(gap, (int, float)) or not isinstance(double, (int, float)):
        raise TranscriptionResultError("The GigaAM sidecar coverage report is malformed.")
    if abs(float(gap)) > COVERAGE_TOLERANCE_S or abs(float(double)) > COVERAGE_TOLERANCE_S:
        raise TranscriptionResultError(
            "The GigaAM sidecar reported a chunk plan that does not cover the "
            f"recording ({float(gap):.6f}s uncovered, {float(double):.6f}s duplicated)."
        )
    return payload


class GigaamSidecarProvider:
    """``transcribe(audio_path, profile) -> str`` over the GigaAM sidecar worker.

    Satisfies :class:`~diary.domain.transcription.providers.TranscriptionProvider`
    structurally and is registered under ``gigaam2_ctc_lm``.
    """

    #: No provenance version is resolvable here: the runtime lives in another
    #: virtualenv, so the transcription row records no ``provider_version`` for
    #: this backend (the revision and the decoding knobs go into
    #: ``metadata_json`` instead).
    version: str | None = None

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        runner: SidecarRunner | None = None,
    ) -> None:
        #: Settings are resolved lazily (the factory constructs adapters with no
        #: arguments, and the worker is long-lived): ``None`` means "the current
        #: application settings at call time".
        self._settings = settings
        self._runner: SidecarRunner = runner if runner is not None else SubprocessRunner()
        #: The last sidecar process this adapter started, kept so :meth:`unload`
        #: can terminate it if it outlived its result. ``None`` before the first
        #: run and after the first :meth:`unload`.
        self._process: SidecarProcess | None = None

    def transcribe(self, audio_path: Path, profile: TranscriptionProfile) -> str:
        """Transcribe ``audio_path`` and return the recognised text.

        ``profile`` is the whole transcription union (a registry adapter must
        accept what the factory may hand it); this adapter serves
        ``gigaam2_ctc_lm`` only and refuses anything else loudly — the factory
        never routes another backend here, so that branch is a programming-error
        guard, not a fallback.

        Raises :class:`ProviderUnavailableError` when the sidecar runtime or its
        weights are not usable here (nothing is started in that case),
        :class:`ProviderConfigurationError` for an unsupported profile switch,
        :class:`TranscriptionExecutionError` for a runtime failure or timeout, and
        :class:`~diary.domain.transcription.errors.TranscriptionResultError` when
        the worker exits successfully but its payload fails validation (a
        deterministic refusal, see :func:`_parse_result`). No message carries
        diary content or a path.
        """
        if not isinstance(profile, GigaamCtcLmProfile):
            raise TranscriptionProviderError(
                "GigaamSidecarProvider serves the 'gigaam2_ctc_lm' provider only; "
                f"got {profile.provider!r}."
            )
        if not profile.unload_after:
            raise ProviderConfigurationError(
                "unload_after: false is not supported by the gigaam2_ctc_lm backend: the "
                "sidecar releases the model and its CUDA cache by exiting, so the model cannot "
                "stay resident between runs (the phase-2 veto forbids keeping STT in VRAM next "
                "to the LLM). Set unload_after: true."
            )
        if not profile.trust_remote_code:
            raise ProviderConfigurationError(
                "trust_remote_code: false is not supported by the gigaam2_ctc_lm backend: the "
                "checkpoint's own code implements its forward pass, so the wrapper cannot be "
                "loaded without executing it. It runs inside the isolated sidecar process. "
                "Set trust_remote_code: true."
            )

        argv = self._build_argv(audio_path, profile)
        run = self._runner.run(
            argv,
            env=self._child_env(),
            timeout_s=self._effective_settings().gigaam_worker_timeout_s,
        )
        # Kept before any branch: a worker that failed is still terminated by
        # ``unload()`` if it did not exit on its own.
        self._process = run.process

        if run.process.returncode == sidecar_protocol.SIDECAR_EXIT_UNAVAILABLE:
            raise ProviderUnavailableError(
                "The GigaAM sidecar runtime is not usable on this machine "
                f"({_reported_class(run.stderr) or 'unknown error'}). {_PROVISION_HINT}"
            )
        if run.process.returncode != sidecar_protocol.SIDECAR_EXIT_OK:
            raise TranscriptionExecutionError(
                "The GigaAM sidecar failed during transcription "
                f"({_reported_class(run.stderr) or 'unknown error'})."
            )
        payload = _parse_result(run.stdout)
        text = payload[sidecar_protocol.RESULT_KEY_TEXT]
        if not isinstance(text, str):  # pragma: no cover - guarded by _parse_result
            raise TranscriptionResultError("The GigaAM sidecar result carries no recognised text.")
        return text

    def unload(self) -> None:
        """Terminate a sidecar that is still running; never raises, idempotent.

        The worker prints its JSON result as its last line and exits, so on the
        normal path the process is already gone and this is a no-op — the
        release happened structurally, which is the whole reason the sidecar was
        chosen (``docs/stt-backends-analysis.md`` §10: the RAM of the contour
        comes back only when the process ends). The termination path exists so
        ``asr.run``'s contract ("whatever was loaded is released before the slot
        is free", ADR-010) holds even when the worker outlives its result.

        The escalation is written out rather than nested: a ``terminate()`` that
        raises is **not** evidence the process is gone (``ProcessLookupError``
        only says the pid was not there at that instant), so it goes on to
        ``kill()`` — a single refused signal must not silently leave a sidecar
        holding the RAM of the contour (measured peak ``VmHWM`` **4553.2 MiB**,
        ``docs/stt-backends-analysis.md`` §9). Every step is wrapped, so the
        method stays total.
        """
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        if not _send_signal(process.terminate):
            _send_signal(process.kill)
        else:
            try:
                process.wait(timeout=_TERMINATE_GRACE_S)
                return
            except Exception:
                # Still alive after the grace period: kill and reap, both
                # best-effort — ``unload`` never raises.
                _send_signal(process.kill)
        with contextlib.suppress(Exception):
            process.wait(timeout=_TERMINATE_GRACE_S)

    def _effective_settings(self) -> AppSettings:
        """The settings to resolve the runtime from (injected or current)."""
        return self._settings if self._settings is not None else get_settings()

    def _build_argv(self, audio_path: Path, profile: GigaamCtcLmProfile) -> list[str]:
        """The worker command line: one audio path plus the profile's parameters.

        ``alpha``/``beta`` are sent only when the profile sets them; omitted, the
        worker lets the checkpoint's own values stand (they are not substituted
        with ``0.0`` or with the values the probe measured).
        """
        python, script = self._resolve_runtime()
        argv = [
            str(python),
            str(script),
            str(audio_path),
            "--model-id",
            profile.model_id,
            "--revision",
            profile.revision,
            "--device",
            profile.device,
            "--beam-width",
            str(profile.beam_width),
            "--max-chunk-s",
            repr(profile.max_chunk_s),
            "--chunk-overlap-s",
            repr(profile.chunk_overlap_s),
        ]
        if profile.alpha is not None:
            argv += ["--alpha", repr(profile.alpha)]
        if profile.beta is not None:
            argv += ["--beta", repr(profile.beta)]
        return argv

    def _child_env(self) -> dict[str, str]:
        """The child's environment: an allow-list plus the pins the adapter owns.

        Three groups, and nothing else (``_ENV_ALLOW_LIST`` carries the reasons):

        * the allow-listed variables this process actually has — a locale, a home,
          a ``PATH``, a scratch directory, the GPU selection, a library path;
        * ``HF_HOME``, set to :attr:`~diary.core.settings.AppSettings.gigaam_model_dir`
          — the provisioned cache. It is set **here** rather than inherited, so the
          weights a transcription reads are the ones ``scripts/fetch-gigaam-model.sh``
          placed there; the worker derives its ``--cache-dir`` (``$HF_HOME/hub``)
          from the same value, so there is one source of truth on both sides;
        * ``HF_HUB_OFFLINE=1``, unconditionally: the application never downloads
          model weights inside a transcription request. A cache miss then fails
          immediately and is reported as an unavailable runtime, which is the
          honest answer — provisioning is an explicit step
          (``scripts/fetch-gigaam-model.sh``), not a side effect of pressing
          «Расшифровать».
        """
        settings = self._effective_settings()
        env = {name: os.environ[name] for name in _ENV_ALLOW_LIST if name in os.environ}
        env[_OFFLINE_ENV] = "1"
        env[_TELEMETRY_ENV] = "1"
        env["HF_HOME"] = str(settings.gigaam_model_dir)
        return env

    def _resolve_runtime(self) -> tuple[Path, Path]:
        """The configured interpreter and worker script, or a loud refusal.

        Nothing is started and nothing is resolved from the network here: a
        missing path is a permanent, user-fixable condition (AGENTS.md §10 — a
        heavy optional runtime is provisioned explicitly).
        """
        settings = self._effective_settings()
        interpreter = settings.gigaam_python
        if interpreter is None:
            raise ProviderUnavailableError(
                "The GigaAM sidecar interpreter is not configured (APP_GIGAAM_PYTHON). "
                + _PROVISION_HINT
            )
        if not Path(interpreter).is_file():
            raise ProviderUnavailableError(
                "The GigaAM sidecar interpreter configured by APP_GIGAAM_PYTHON does not exist. "
                + _PROVISION_HINT
            )
        script = Path(settings.gigaam_worker_script)
        if not script.is_file():
            raise ProviderUnavailableError(
                "The GigaAM sidecar worker script configured by APP_GIGAAM_WORKER_SCRIPT "
                "does not exist. " + _PROVISION_HINT
            )
        return Path(interpreter), script


__all__ = [
    "GigaamSidecarProvider",
    "SidecarProcess",
    "SidecarRun",
    "SidecarRunner",
    "SidecarTimeoutError",
    "SubprocessRunner",
]
