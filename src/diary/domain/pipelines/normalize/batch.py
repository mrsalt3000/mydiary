"""Batch normalization enqueueing for eligible original exports (Task 13).

``POST /api/v1/pipelines/normalize/batch`` enqueues one ``normalize`` job for
every **eligible** original export. An export is eligible when:

1. its original artifact file is present on disk (resolved by database
   identity, never by globs), **and**
2. it has **no succeeded normalized artifact** yet.

The second rule is read from the ``artifacts`` table: the framework creates
an artifact row of layer ``normalized`` only *after* a validated layer-2 file
was written atomically by a succeeded run, so the presence of such a row
means the export already has a normalized result. Re-normalizing an already
normalized export is an explicit single-run action
(``POST .../normalize/runs``), never part of a batch. A previous *failed* run
leaves no artifact row, so the export stays eligible and a later batch retries
it.

Deduplication: batch enqueueing reuses :func:`~diary.domain.pipelines.service
.enqueue_pipeline_run`, which never creates a second job while one is already
in flight for the same (export, pipeline). Running the batch twice while jobs
are pending/running returns the same job ids instead of duplicating rows.
"""

from __future__ import annotations

from diary.core.settings import get_settings
from diary.db.models import ArtifactLayer, Job, PipelineType, WeeklyExport
from diary.db.repositories import ArtifactRepository, WeeklyExportRepository
from diary.domain.pipelines import service as pipeline_service


def list_eligible_exports(session) -> list[WeeklyExport]:
    """Original exports that may receive a fresh normalization run, in
    deterministic sequence order (see the module docstring for the rule)."""
    exports = WeeklyExportRepository(session).list_all_by_sequence()
    eligible: list[WeeklyExport] = []
    for export in exports:
        original_path = get_settings().data_root / export.path
        if not original_path.is_file():
            continue
        has_result = ArtifactRepository(session).list_by_export_and_layer(
            export.id, ArtifactLayer.normalized
        )
        if has_result:
            continue
        eligible.append(export)
    return eligible


def enqueue_normalize_batch(session) -> list[Job]:
    """Enqueue one normalization job per eligible export; return the jobs.

    Jobs are returned in the same deterministic order as
    :func:`list_eligible_exports`. Each job row is committed by
    ``enqueue_pipeline_run``; an already active job for an export is returned
    unchanged (no duplicates).
    """
    jobs: list[Job] = []
    for export in list_eligible_exports(session):
        jobs.append(
            pipeline_service.enqueue_pipeline_run(
                session, pipeline_type=PipelineType.normalize, export_id=export.id
            )
        )
    return jobs


__all__ = ["enqueue_normalize_batch", "list_eligible_exports"]
