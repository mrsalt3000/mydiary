"""Batch summarization enqueueing for eligible exports (Task 14).

``POST /api/v1/pipelines/summarize/batch`` enqueues one ``summarize`` job for
every **eligible** export. Layer 3 is built only from layer 2, so an export is
eligible when:

1. its original artifact file is present on disk (the shared enqueue path
   verifies it; existence only — the layer-3 input is never the original),
2. it has at least one **normalized** artifact (a derived-layer row exists only
   after a validated layer-2 file was written by a succeeded run) and the
   newest normalized file is present on disk,
3. it has **no summary artifact yet**.

The summary version is created by the newest succeeded normalized artifact the
framework resolves at run time (see
:func:`~diary.domain.pipelines.service.run_pipeline`), so a rerun of
normalization followed by a rerun of summarization always reads the newest
layer-2 version. A previous *failed* summary run leaves no artifact row, so the
export stays eligible and a later batch retries it.

Deduplication: batch enqueueing reuses
:func:`~diary.domain.pipelines.service.enqueue_pipeline_run`, which never
creates a second job while one is already in flight for the same (export,
pipeline).
"""

from __future__ import annotations

from diary.core.settings import get_settings
from diary.db.models import ArtifactLayer, Job, PipelineType, WeeklyExport
from diary.db.repositories import ArtifactRepository, WeeklyExportRepository
from diary.domain.pipelines import service as pipeline_service


def list_eligible_exports(session) -> list[WeeklyExport]:
    """Exports that may receive a fresh summarization run, in deterministic
    sequence order (see the module docstring for the rule)."""
    exports = WeeklyExportRepository(session).list_all_by_sequence()
    eligible: list[WeeklyExport] = []
    for export in exports:
        original_path = get_settings().data_root / export.path
        if not original_path.is_file():
            continue
        normalized = ArtifactRepository(session).list_by_export_and_layer(
            export.id, ArtifactLayer.normalized
        )
        if not normalized:
            continue
        normalized_path = get_settings().data_root / normalized[-1].path
        if not normalized_path.is_file():
            continue
        has_summary = ArtifactRepository(session).list_by_export_and_layer(
            export.id, ArtifactLayer.summary
        )
        if has_summary:
            continue
        eligible.append(export)
    return eligible


def enqueue_summarize_batch(session) -> list[Job]:
    """Enqueue one summarization job per eligible export; return the jobs.

    Jobs are returned in the same deterministic order as
    :func:`list_eligible_exports`. Each job row is committed by
    ``enqueue_pipeline_run``; an already active job for an export is returned
    unchanged (no duplicates).
    """
    jobs: list[Job] = []
    for export in list_eligible_exports(session):
        jobs.append(
            pipeline_service.enqueue_pipeline_run(
                session, pipeline_type=PipelineType.summarize, export_id=export.id
            )
        )
    return jobs


__all__ = ["enqueue_summarize_batch", "list_eligible_exports"]
