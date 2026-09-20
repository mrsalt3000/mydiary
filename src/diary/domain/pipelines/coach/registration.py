"""Explicit registration wiring of the coaching-note layer (task T07).

Mirrors the STT-correction (task T04), daily-summary (task T05) and
weekly-summary (task T06) registrations: the pipeline registry is filled by
*callers*, never as an import side effect, so importing this package cannot
mutate a registry that a test (or the web process) expects to find in its own
state. The single entry point :func:`ensure_coach_pipeline_registered` is:

* guarded — registering the real layer while a fake/other ``coach_note``
  implementation is already present is a no-op, so test substitutions keep
  working;
* idempotent — a caller may invoke it on every boot.

The layer is registered in the **second-family** half of the registry
(``SECOND_FAMILY_REGISTRY``): it is built from an upstream weekly-summary
artifact and is not driven by a weekly export, so it is looked up with
:func:`~diary.domain.pipelines.registry.get_second_family_pipeline`, not with
``get_pipeline``.

**Nothing calls this function automatically** (T07 decision 3: «запуск только
явный»). The worker startup (``diary.workers.runner.main``) **registers** the
layer since T09 — the orchestrator resolves layers through
:func:`~diary.domain.pipelines.registry.get_second_family_pipeline`, so an
unregistered layer could not be run explicitly in the worker process either —
but registration is not a launch: the worker registers **no job handler** for
``JobType.coach_note`` (an enqueued ``coach_note`` job therefore still fails
permanently instead of running the layer) and no route or worker code calls the
layer service. Launching the layer stays a deliberate action of a caller
(T09's ``run_coach_step`` and T11's CLI/UI own that wiring).
"""

from __future__ import annotations

from diary.db.models import PipelineType
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.coach.pipeline import CoachNotePipeline


def ensure_coach_pipeline_registered() -> bool:
    """Register the coaching-note layer unless one is already present.

    Returns ``True`` when this call performed the registration and ``False``
    when a ``coach_note`` layer was already registered (e.g. a test fake).
    """
    if pipeline_registry.get_second_family_pipeline(PipelineType.coach_note) is not None:
        return False
    pipeline_registry.register(CoachNotePipeline())
    return True


__all__ = ["ensure_coach_pipeline_registered"]
