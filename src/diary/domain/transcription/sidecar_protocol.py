"""Wire contract between the application and the GigaAM sidecar worker (T44).

The sidecar is a separate process with its own virtualenv
(``docs/stt-backends-analysis.md`` §10): the application starts it, hands it one
audio path and the decoding parameters, and reads back **one JSON object** from
the process' stdout. This module is the single source of truth for that
contract's constants, so the adapter (:mod:`diary.domain.transcription.gigaam`)
and the worker (``scripts/gigaam-stt-worker.py``) cannot drift apart. It imports
only the standard library: the worker's venv has numpy/torch/transformers but
neither pydantic nor sqlalchemy, so it must never import the application's
dependency graph.

The contract
============

**Invocation** (arguments, never a shell string)::

    <python> <worker-script> <audio-path> \
        --model-id <hf-repo> --revision <sha> --device cpu|cuda \
        --beam-width <n> --max-chunk-s <s> --chunk-overlap-s <s> \
        [--alpha <f>] [--beta <f>] [--greedy]

``--alpha``/``--beta`` are passed only when the profile sets them; omitted, the
worker lets the checkpoint's own ``language_model/attrs.json`` values stand.

**Success** — exit code :data:`SIDECAR_EXIT_OK`, and the **last stdout line** is
a single JSON object with at least ``text`` (the recognised string) and
``coverage`` (``{"gap_s": …, "double_s": …}``, both zero on a valid chunk plan).
Diagnostics (timings, chunk kinds, coverage, removed words) go to stderr and are
never read back into the application.

**Failure** — a non-zero exit code, and the **last stderr line** is the static
name of the error class (a plain identifier, never a message):

* :data:`SIDECAR_EXIT_UNAVAILABLE` — the runtime or the weights are not usable
  on this machine, or the input cannot be planned (a repeat cannot help) →
  the adapter raises :class:`~diary.domain.transcription.errors.ProviderUnavailableError`;
* any other non-zero code — inference/load failure → the adapter raises
  :class:`~diary.domain.transcription.errors.TranscriptionExecutionError`.

**Privacy** (AGENTS.md §7–8): the worker never writes recognised text or audio
data to stderr, and the application never writes the worker's stdout to a log —
the recognised text travels from stdout into the ``transcriptions`` table and
nowhere else.
"""

from __future__ import annotations

#: Exit code of a successful run.
SIDECAR_EXIT_OK = 0
#: Exit code of a run that failed at runtime (transient).
SIDECAR_EXIT_FAILED = 1
#: Exit code of a run whose runtime/weights/input cannot work here (permanent).
SIDECAR_EXIT_UNAVAILABLE = 3

#: JSON keys the adapter reads from the worker's last stdout line.
RESULT_KEY_TEXT = "text"
RESULT_KEY_COVERAGE = "coverage"
COVERAGE_KEY_GAP_S = "gap_s"
COVERAGE_KEY_DOUBLE_S = "double_s"

__all__ = [
    "COVERAGE_KEY_DOUBLE_S",
    "COVERAGE_KEY_GAP_S",
    "RESULT_KEY_COVERAGE",
    "RESULT_KEY_TEXT",
    "SIDECAR_EXIT_FAILED",
    "SIDECAR_EXIT_OK",
    "SIDECAR_EXIT_UNAVAILABLE",
]
