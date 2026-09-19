"""Explicit registration wiring of the STT-correction layer (task T04).

Mirrors the normalization/summarization/wellbeing registrations (Tasks 13-15):
the pipeline registry is filled by *callers*, never as an import side effect, so
importing this package cannot mutate a registry that a test (or the web process)
expects to find in its own state. The single entry point
:func:`ensure_stt_fix_pipeline_registered` is:

* guarded — registering the real layer while a fake/other ``stt_corrected``
  implementation is already present is a no-op, so test substitutions keep
  working;
* idempotent — the worker startup may call it on every boot.

The layer is registered in the **second-family** half of the registry
(``SECOND_FAMILY_REGISTRY``): it is per message and is not driven by a weekly
export, so it is looked up with
:func:`~diary.domain.pipelines.registry.get_second_family_pipeline`, not with
``get_pipeline``.
"""

from __future__ import annotations

from diary.db.models import PipelineType
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.stt_fix.pipeline import SttFixPipeline


def ensure_stt_fix_pipeline_registered() -> bool:
    """Register the STT-correction layer unless one is already present.

    Returns ``True`` when this call performed the registration and ``False``
    when an ``stt_corrected`` layer was already registered (e.g. a test fake).
    """
    if pipeline_registry.get_second_family_pipeline(PipelineType.stt_corrected) is not None:
        return False
    pipeline_registry.register(SttFixPipeline())
    return True


__all__ = ["ensure_stt_fix_pipeline_registered"]
