"""Wellbeing enqueueing use case (Task 15).

Layer 4 is a deliberate, single action on normalized text (AGENTS.md §16), so
its enqueue path is stricter than the generic pipeline endpoint:

* **feature switch** — when ``AppSettings.wellbeing_enabled`` is ``False`` the
  request is rejected up front (:class:`WellbeingDisabledError` → HTTP 403);
* **explicit confirmation** — the caller must pass ``confirmed=True``
  (:class:`WellbeingConfirmationRequiredError` → HTTP 409). The UI enforces a
  second, client-side confirmation, so skipping the flag is impossible from
  the page and hard from the API;
* **input eligibility** — wellbeing consumes only a normalized (layer 2)
  artifact, so an export without a normalized artifact (row + file on disk) is
  rejected before any job is created (:class:`WellbeingInputUnavailableError`
  → HTTP 404). Running normalization upstream first is the documented fix.

The actual durable job is created through the shared
:func:`diary.domain.pipelines.service.enqueue_pipeline_run`, so deduplication
of in-flight jobs and rerun versioning behave exactly like the other pipelines.

Privacy: the job payload carries identifiers only (the shared service owns
that); nothing here logs or persists diary content.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from diary.core.settings import get_settings
from diary.db.models import ArtifactLayer, Job, PipelineType
from diary.db.repositories import ArtifactRepository, WeeklyExportRepository
from diary.domain.pipelines import service as pipeline_service
from diary.domain.pipelines.base import PipelineExportNotFoundError


class WellbeingEnqueueError(Exception):
    """Base class for wellbeing enqueueing rejections (safe, static)."""


class WellbeingDisabledError(WellbeingEnqueueError):
    """Wellbeing analysis is disabled by configuration (HTTP 403)."""


class WellbeingConfirmationRequiredError(WellbeingEnqueueError):
    """The run was not explicitly confirmed by the user (HTTP 409)."""


class WellbeingInputUnavailableError(WellbeingEnqueueError):
    """The export has no normalized (layer 2) input available yet (HTTP 404)."""


def enqueue_wellbeing_run(session: Session, *, export_id: uuid.UUID, confirmed: bool) -> Job:
    """Enqueue one wellbeing run for ``export_id`` after wellbeing-specific
    guards; return the durable job.

    Raises :class:`WellbeingDisabledError` when the feature is disabled,
    :class:`WellbeingConfirmationRequiredError` when ``confirmed`` is false,
    :class:`PipelineExportNotFoundError` (→ HTTP 404) for an unknown export
    and :class:`WellbeingInputUnavailableError` (→ HTTP 404) when the export
    has no normalized artifact whose file is present. Commits the new job row
    through the shared pipeline service.
    """
    if not get_settings().wellbeing_enabled:
        raise WellbeingDisabledError("Wellbeing analysis is disabled by configuration.")
    if not confirmed:
        raise WellbeingConfirmationRequiredError(
            "Explicit confirmation is required before a wellbeing analysis run."
        )
    export = WeeklyExportRepository(session).get(export_id)
    if export is None:
        raise PipelineExportNotFoundError("Export not found.")
    artifacts = ArtifactRepository(session).list_by_export_and_layer(
        export_id, ArtifactLayer.normalized
    )
    if not artifacts:
        raise WellbeingInputUnavailableError(
            "The export has no normalized text yet; run normalization first."
        )
    assert artifacts[-1].path is not None  # a normalized artifact is file-backed
    normalized_path = get_settings().data_root / artifacts[-1].path
    if not normalized_path.is_file():
        raise WellbeingInputUnavailableError(
            "The export has no normalized text yet; run normalization first."
        )
    return pipeline_service.enqueue_pipeline_run(
        session, pipeline_type=PipelineType.wellbeing, export_id=export_id
    )


__all__ = [
    "WellbeingConfirmationRequiredError",
    "WellbeingDisabledError",
    "WellbeingEnqueueError",
    "WellbeingInputUnavailableError",
    "enqueue_wellbeing_run",
]
