"""Explicit registration wiring of the weekly-summary layer (task T06).

Mirrors the daily-summary registration (task T05) and the STT-correction
registration (task T04): the pipeline registry is filled by *callers*, never as
an import side effect, so importing this package cannot mutate a registry that a
test (or the web process) expects to find in its own state. The single entry
point :func:`ensure_sum_weekly_pipeline_registered` is:

* guarded — registering the real layer while a fake/other ``weekly_summary``
  implementation is already present is a no-op, so test substitutions keep
  working;
* idempotent — the worker startup may call it on every boot.

The layer is registered in the **second-family** half of the registry
(``SECOND_FAMILY_REGISTRY``): it is per ISO week and is not driven by a weekly
export, so it is looked up with
:func:`~diary.domain.pipelines.registry.get_second_family_pipeline`, not with
``get_pipeline``.
"""

from __future__ import annotations

from diary.db.models import PipelineType
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.sum_weekly.pipeline import SumWeeklyPipeline


def ensure_sum_weekly_pipeline_registered() -> bool:
    """Register the weekly-summary layer unless one is already present.

    Returns ``True`` when this call performed the registration and ``False``
    when a ``weekly_summary`` layer was already registered (e.g. a test fake).
    """
    if pipeline_registry.get_second_family_pipeline(PipelineType.weekly_summary) is not None:
        return False
    pipeline_registry.register(SumWeeklyPipeline())
    return True


__all__ = ["ensure_sum_weekly_pipeline_registered"]
