"""Explicit registration wiring of the daily-summary layer (task T05).

Mirrors the STT-correction registration (task T04) and the MVP
normalization/summarization/wellbeing registrations: the pipeline registry is
filled by *callers*, never as an import side effect, so importing this package
cannot mutate a registry that a test (or the web process) expects to find in its
own state. The single entry point :func:`ensure_sum_daily_pipeline_registered`
is:

* guarded — registering the real layer while a fake/other ``daily_summary``
  implementation is already present is a no-op, so test substitutions keep
  working;
* idempotent — the worker startup may call it on every boot.

The layer is registered in the **second-family** half of the registry
(``SECOND_FAMILY_REGISTRY``): it is per local day and is not driven by a weekly
export, so it is looked up with
:func:`~diary.domain.pipelines.registry.get_second_family_pipeline`, not with
``get_pipeline``.
"""

from __future__ import annotations

from diary.db.models import PipelineType
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.sum_daily.pipeline import SumDailyPipeline


def ensure_sum_daily_pipeline_registered() -> bool:
    """Register the daily-summary layer unless one is already present.

    Returns ``True`` when this call performed the registration and ``False``
    when a ``daily_summary`` layer was already registered (e.g. a test fake).
    """
    if pipeline_registry.get_second_family_pipeline(PipelineType.daily_summary) is not None:
        return False
    pipeline_registry.register(SumDailyPipeline())
    return True


__all__ = ["ensure_sum_daily_pipeline_registered"]
