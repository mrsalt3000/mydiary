"""Thin HTTP routes for the pipeline framework (Task 12).

- ``POST /api/v1/pipelines/{pipeline_type}/runs``  — enqueue a pipeline run
  for one original export (202 + job); body ``{export_id}``. The pipeline
  implementation itself may not be registered yet (Tasks 13-15) — enqueueing
  still succeeds; executing the job then fails permanently with a clear
  summary;
- ``GET /api/v1/exports/{export_id}/runs``       — the export's processing
  runs, newest first (status surface + provenance columns, never content);
- ``GET /api/v1/exports/{export_id}/artifacts``  — the export's artifacts by
  layer (metadata only: layer/version/filename/sha256; no file bytes).

Handlers stay thin: enqueueing and the run lifecycle live in
:mod:`diary.domain.pipelines.service`. Error details are static and never echo
diary content (AGENTS.md §8). Status codes: 404 for unknown exports, 409 when
the export's original file is gone, 422 for unknown pipeline types, malformed
UUIDs or a missing body.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.api.schemas import (
    ExportArtifactRead,
    JobRead,
    PipelineRunCreate,
    ProcessingRunRead,
    artifact_read_from,
    job_read_from,
    processing_run_read_from,
)
from diary.core.settings import get_settings
from diary.db.models import PipelineType
from diary.db.repositories import (
    ArtifactRepository,
    ProcessingRunRepository,
    WeeklyExportRepository,
)
from diary.domain.pipelines import service as pipeline_service
from diary.domain.pipelines.base import (
    PipelineExportNotFoundError,
    PipelineOriginalFileMissingError,
)
from diary.domain.pipelines.naming import (
    EXPORTS_PREFIX,
    layer_dirname,
    layer_output_extension,
)
from diary.domain.pipelines.normalize import batch as normalize_batch
from diary.domain.pipelines.summarize import batch as summarize_batch

router = APIRouter(prefix="/pipelines", tags=["pipelines"])

#: Media types of rendered artifact files served by the generic download route.
_ARTIFACT_MEDIA_TYPES = {
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


@router.post("/normalize/batch", response_model=list[JobRead], status_code=202)
def enqueue_normalize_batch(
    session: Annotated[Session, Depends(get_session)],
) -> list[JobRead]:
    """Enqueue one normalization job per eligible original export.

    An export is eligible when its original file is present and it has no
    normalized artifact yet (see
    :func:`diary.domain.pipelines.normalize.batch.enqueue_normalize_batch`).
    Running the batch again while jobs are in flight reuses the active jobs —
    no duplicates are created. Returns the jobs of the eligible exports in
    deterministic export order.
    """
    jobs = normalize_batch.enqueue_normalize_batch(session)
    return [job_read_from(job) for job in jobs]


@router.post("/summarize/batch", response_model=list[JobRead], status_code=202)
def enqueue_summarize_batch(
    session: Annotated[Session, Depends(get_session)],
) -> list[JobRead]:
    """Enqueue one summarization job per eligible export.

    An export is eligible when it has a normalized artifact whose file is
    present and no summary artifact yet (see
    :func:`diary.domain.pipelines.summarize.batch.enqueue_summarize_batch`).
    Running the batch again while jobs are in flight reuses the active jobs —
    no duplicates are created. Returns the jobs of the eligible exports in
    deterministic export order.
    """
    jobs = summarize_batch.enqueue_summarize_batch(session)
    return [job_read_from(job) for job in jobs]


@router.post("/{pipeline_type}/runs", response_model=JobRead, status_code=202)
def enqueue_pipeline_run(
    pipeline_type: PipelineType,
    payload: PipelineRunCreate,
    session: Annotated[Session, Depends(get_session)],
) -> JobRead:
    """Queue one pipeline run for the export given in ``payload.export_id``."""
    try:
        job = pipeline_service.enqueue_pipeline_run(
            session, pipeline_type=pipeline_type, export_id=payload.export_id
        )
    except PipelineExportNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Export not found.") from exc
    except PipelineOriginalFileMissingError as exc:
        raise HTTPException(
            status_code=409,
            detail="The original export file is missing; the pipeline cannot run.",
        ) from exc
    return job_read_from(job)


# ---------------------------------------------------------------------------
# Export-scoped listing of runs and artifacts
# ---------------------------------------------------------------------------

runs_router = APIRouter(prefix="/exports", tags=["exports"])


def _require_export(session: Session, export_id: uuid.UUID) -> None:
    """Raise HTTP 404 when no export row exists for ``export_id``."""
    if WeeklyExportRepository(session).get(export_id) is None:
        raise HTTPException(status_code=404, detail="Export not found.")


@runs_router.get("/{export_id}/runs", response_model=list[ProcessingRunRead])
def list_export_runs(
    export_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> list[ProcessingRunRead]:
    """Return the processing runs of one export, newest first, metadata only."""
    _require_export(session, export_id)
    runs = ProcessingRunRepository(session).list_by_input_export(export_id)
    return [processing_run_read_from(run) for run in reversed(runs)]


@runs_router.get("/{export_id}/artifacts", response_model=list[ExportArtifactRead])
def list_export_artifacts(
    export_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> list[ExportArtifactRead]:
    """Return the artifacts of one export grouped by layer and version."""
    _require_export(session, export_id)
    artifacts = ArtifactRepository(session).list_by_export(export_id)
    return [artifact_read_from(artifact) for artifact in artifacts]


@runs_router.get("/{export_id}/artifacts/{artifact_id}/file")
def download_artifact_file(
    export_id: uuid.UUID,
    artifact_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> FileResponse:
    """Download one derived artifact file (normalized/summary/wellbeing).

    Generic download for layers 2-4 (the layer-1 download route lives on the
    exports router). The artifact must belong to ``export_id`` and its stored
    data-root-relative path must resolve inside the artifact's own layer
    directory (server-generated paths; the containment check is defensive).
    The media type follows the layer extension (``.txt`` plain, ``.md``
    markdown), never diary content or user-supplied names.
    """
    artifact = ArtifactRepository(session).get(artifact_id)
    if artifact is None or artifact.export_id != export_id:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    settings = get_settings()
    # Derived-artifact paths are data-root-relative ``exports/<layer>/<file>``
    # (the naming module is the single source of truth), so the containment
    # check resolves against that fixed location.
    layer_dir = (settings.data_root / EXPORTS_PREFIX / layer_dirname(artifact.layer)).resolve()
    candidate = (settings.data_root / artifact.path).resolve()
    if not candidate.is_relative_to(layer_dir):
        # Stored paths are server-generated; this branch is purely defensive.
        raise HTTPException(status_code=404, detail="Artifact not found.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")
    extension = layer_output_extension(artifact.layer)
    filename = artifact.path.rsplit("/", 1)[-1]
    return FileResponse(
        str(candidate),
        media_type=_ARTIFACT_MEDIA_TYPES.get(extension, "application/octet-stream"),
        filename=filename,
    )
