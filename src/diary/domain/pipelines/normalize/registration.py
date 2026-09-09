"""Explicit registration wiring of the normalization pipeline (Task 13).

The pipeline registry (Task 12) is filled by *callers*, never as an import
side effect: executing code that merely imports a module must not mutate the
global registry, otherwise the framework tests that install their own fake
``normalize`` pipeline (and the unit-test baseline that expects the registry
to stay empty) would break. The single entry point is
:func:`ensure_normalize_pipeline_registered`, which is:

* guarded — registering the real pipeline while a fake/other normalize
  implementation is already present is a no-op, so test substitutions of the
  registry keep working;
* wired into the **worker** startup (``diary.workers.runner.main``), where
  real ``normalize`` jobs are executed — without it every job would fail
  permanently with "pipeline not registered yet".

The web process never executes pipeline jobs (it only enqueues them), so it
deliberately does not register the pipeline; doing so inside a FastAPI startup
event would mutate the shared registry in every test that opens a
``TestClient`` and break the fake-pipeline tests of Task 12.
"""

from __future__ import annotations

from diary.db.models import PipelineType
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.normalize.pipeline import NormalizePipeline


def ensure_normalize_pipeline_registered() -> bool:
    """Register the real normalization pipeline unless one is already present.

    Returns ``True`` when this call performed the registration and ``False``
    when a normalize pipeline was already registered (e.g. a test fake) or
    when this instance already registered it. Calling the function repeatedly
    is safe (idempotent) — the worker startup may call it on every boot.
    """
    if pipeline_registry.get_pipeline(PipelineType.normalize) is not None:
        return False
    pipeline_registry.register(NormalizePipeline())
    return True


__all__ = ["ensure_normalize_pipeline_registered"]
