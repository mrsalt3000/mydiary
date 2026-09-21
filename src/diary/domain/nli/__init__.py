"""Local NLI validation runtime (T24a).

The rules of the check live in :mod:`diary.domain.contracts.nli` and are
model-agnostic; this package holds the production adapter that runs a pinned
Russian NLI cross-encoder through ONNX Runtime on the CPU
(:mod:`diary.domain.nli.onnx_judge`). Analysis, measurements and the decision
behind the whole thing: ``docs/nli-validation-analysis.md``; the task and its
definition of done: ``docs/tasklist-phase2.md`` §15.
"""

from __future__ import annotations

from diary.domain.nli.onnx_judge import NliUnavailableError, OnnxNliJudge

__all__ = ["NliUnavailableError", "OnnxNliJudge"]
