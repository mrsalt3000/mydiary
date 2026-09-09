"""Explicit registration wiring of the wellbeing pipeline (Task 15).

Mirrors the normalization/summarization registrations (Tasks 13-14): the
pipeline registry is filled by *callers*, never as an import side effect, and
the single entry point :func:`ensure_wellbeing_pipeline_registered` is guarded
and wired into the **worker** startup (``diary.workers.runner.main``), where
real ``wellbeing`` jobs are executed. The web process never executes pipeline
jobs and therefore does not register the pipeline.
"""

from __future__ import annotations

from diary.db.models import PipelineType
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.wellbeing.pipeline import WellbeingPipeline


def ensure_wellbeing_pipeline_registered() -> bool:
    """Register the real wellbeing pipeline unless one is already present.

    Returns ``True`` when this call performed the registration and ``False``
    when a wellbeing pipeline was already registered (e.g. a test fake) or
    when this instance already registered it. Calling the function repeatedly
    is safe (idempotent) — the worker startup may call it on every boot.
    """
    if pipeline_registry.get_pipeline(PipelineType.wellbeing) is not None:
        return False
    pipeline_registry.register(WellbeingPipeline())
    return True


__all__ = ["ensure_wellbeing_pipeline_registered"]
