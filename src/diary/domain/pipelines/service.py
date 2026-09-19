"""Pipeline orchestration use cases: enqueueing and run lifecycle (Task 12).

Responsibilities
================

* :func:`enqueue_pipeline_run` — the explicit user action behind
  ``POST /api/v1/pipelines/{type}/runs``. It validates that the target export
  exists and its original artifact file is present, then enqueues a durable
  job whose payload holds only identifiers (``export_id`` and the pipeline
  type — never diary content). While a job for the same (export, pipeline) is
  in flight (pending/running/retrying), the existing job is returned instead
  of creating a duplicate; after a terminal outcome an explicit repeat request
  starts a fresh run (reruns produce ``v2``, never an overwrite).
* :func:`run_pipeline` — the framework core, executed by the generic worker
  handler for the ``normalize``/``summarize``/``wellbeing`` job types:

  - the input is resolved **by database identity**: the job payload's
    ``export_id`` (and, for layers 3-4 later, an optional ``input_artifact_id``)
    — no filesystem globs;
  - the processing-run lifecycle is ``queued → running → succeeded | failed``
    with ``started_at``/``finished_at`` and a safe ``error_summary``; each job
    records **one** terminal run, and replaying a job whose run already
    succeeded is a no-op (worker crash between the handler commit and
    ``mark_succeeded`` + stale-``running`` reset, mirroring Task 09);
  - the pipeline implementation is looked up in the registry: an unregistered
    pipeline (e.g. wellbeing before Task 15) fails permanently with a clear
    summary and is never retried;
  - the LLM profile is resolved by the pipeline's ``task_type``, the request is
    built, the provider is called, the response is parsed into the pipeline's
    Pydantic result schema **before any file is written**; invalid output is a
    failed run with no artifact;
  - the final file is rendered by the backend from validated structured data
    and written atomically (temp → fsync → rename) into the layer directory;
  - provenance (profile/model/prompt version/usage/input hash) is recorded in
    the run — never diary content or raw LLM text.

Error policy: deterministic failures (unregistered pipeline, invalid input,
invalid structured output, config/storage errors, non-retryable LLM errors)
fail the run permanently; transient LLM errors (rate limit, HTTP 5xx,
network, timeout) propagate unchanged so the Task-08 retry policy applies.

Privacy contract (AGENTS.md §7-8): job payloads and run metadata carry only
identifiers and provenance. Diary content, prompts and LLM responses are
never logged, never stored in metadata and never placed in error summaries.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from diary.core.models_config import (
    ModelsConfig,
    ModelsConfigError,
    OpenAICompatibleProfile,
    load_models_config,
)
from diary.core.settings import get_settings
from diary.db.base import utc_now
from diary.db.models import (
    Artifact,
    ArtifactLayer,
    Job,
    PipelineType,
    ProcessingRun,
    RunStatus,
)
from diary.db.repositories import (
    MAX_ERROR_SUMMARY_LENGTH,
    ArtifactRepository,
    JobRepository,
    ProcessingRunRepository,
    WeeklyExportRepository,
)
from diary.domain.llm.errors import LLMError
from diary.domain.llm.models import LLMRequest, LLMResponse, request_input_sha256
from diary.domain.llm.providers import create_llm_provider, resolve_completion_params
from diary.domain.llm.service import resolve_profile
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.base import (
    Pipeline,
    PipelineError,
    PipelineExportNotFoundError,
    PipelineInputError,
    PipelineNotRegisteredError,
    PipelineOriginalFileMissingError,
    PipelineOutputError,
    PipelinePermanentError,
    PipelineStorageError,
)
from diary.domain.pipelines.naming import (
    JOB_PIPELINE_TYPE,
    PIPELINE_JOB_TYPE,
    artifact_data_root_relative_path,
    derived_filename,
)

#: Job-payload keys (identifiers only, never diary content).
PAYLOAD_EXPORT_ID = "export_id"
PAYLOAD_INPUT_ARTIFACT_ID = "input_artifact_id"
PAYLOAD_PIPELINE_TYPE = "pipeline_type"

#: Static, non-sensitive summary for configuration failures.
_CONFIG_ERROR_MESSAGE = "The LLM model configuration is missing or invalid."
#: Static, non-sensitive summary for structured-output validation failures.
_OUTPUT_INVALID_MESSAGE = "The pipeline output did not pass structured validation."


class PipelineServiceError(PipelineError):
    """Base class for pipeline-service errors that carry no diary content."""


def _load_models_config() -> ModelsConfig:
    """Return the current models configuration.

    Module-level hook so tests can substitute an in-memory configuration
    without a YAML file (mirrors the Task-09 transcription service hook).
    """
    return load_models_config()


# ---------------------------------------------------------------------------
# Enqueueing (explicit user action)
# ---------------------------------------------------------------------------


def find_active_pipeline_job(
    session: Session, *, pipeline_type: PipelineType, export_id: uuid.UUID
) -> Job | None:
    """The in-flight job for one (``pipeline_type``, ``export_id``), if any.

    "In flight" means ``pending``, ``running`` or ``retrying``. Terminal jobs
    do not count, so an explicit repeat request after a success (rerun → v2)
    or a failure starts a fresh job.
    """
    target = str(export_id)
    job_type = PIPELINE_JOB_TYPE[pipeline_type]
    for job in JobRepository(session).list_active_of_type(job_type):
        payload = job.payload_json if isinstance(job.payload_json, dict) else {}
        if str(payload.get(PAYLOAD_EXPORT_ID, "")) == target:
            return job
    return None


def enqueue_pipeline_run(
    session: Session, *, pipeline_type: PipelineType, export_id: uuid.UUID
) -> Job:
    """Queue one pipeline run for ``export_id``; return the job.

    The export must exist and its original artifact file must be present (both
    checks resolve by database identity, never by file globs). A job for the
    same (export, pipeline) that is already in flight is returned unchanged —
    no duplicate row. Raises :class:`PipelineExportNotFoundError` (→ HTTP 404)
    for an unknown export and :class:`PipelineOriginalFileMissingError` (→ HTTP
    409) when the original file is gone. Commits the new job row.
    """
    export = WeeklyExportRepository(session).get(export_id)
    if export is None:
        raise PipelineExportNotFoundError("Export not found.")
    original_path = get_settings().data_root / export.path
    if not original_path.is_file():
        raise PipelineOriginalFileMissingError(
            "The original export file is missing; the pipeline cannot run."
        )
    active = find_active_pipeline_job(session, pipeline_type=pipeline_type, export_id=export_id)
    if active is not None:
        return active
    job = JobRepository(session).enqueue(
        job_type=PIPELINE_JOB_TYPE[pipeline_type],
        payload_json={
            PAYLOAD_PIPELINE_TYPE: pipeline_type.value,
            PAYLOAD_EXPORT_ID: str(export_id),
        },
    )
    session.commit()
    return job


# ---------------------------------------------------------------------------
# Run lifecycle (worker handler core)
# ---------------------------------------------------------------------------


def run_pipeline(session: Session, job: Job) -> None:
    """Execute one claimed pipeline job (called by the generic worker handler).

    The caller (``diary.workers.runner``) owns the transaction: on a normal
    return it commits the session and marks the job ``succeeded``; on an
    exception it applies the retry policy. This function commits itself only
    when it records a terminal *failed* run (permanent failure, or the final
    attempt of a transient failure), mirroring the Task-09 transcription flow.
    """
    pipeline_type = _pipeline_type_for_job(job)
    export_id = _payload_export_id(job)

    terminal = _terminal_run_for_job(session, job.id)
    if terminal is not None:
        if terminal.status is RunStatus.succeeded:
            # A previous execution of this very job already committed a
            # succeeded run and its artifact (worker crash between the handler
            # commit and mark_succeeded; stale-running reset replayed the job).
            # Re-running would duplicate the artifact, so the replay is a no-op.
            return
        # The job already recorded its terminal failed run; do not burn more
        # attempts (mirrors the Task-09 failure-replay guard).
        raise PipelinePermanentError("This pipeline job already recorded a terminal failure.")

    run = ProcessingRunRepository(session).create(
        pipeline_type=pipeline_type,
        status=RunStatus.queued,
        metadata_json={"job_id": str(job.id)},
    )
    run.status = RunStatus.running
    run.started_at = run.started_at or utc_now()
    try:
        _execute_run(session, run, job, pipeline_type=pipeline_type, export_id=export_id)
    except PipelinePermanentError as exc:
        raise _record_failed_run(session, run, job, exc) from exc
    except ValidationError as exc:
        # Defense in depth: a pipeline whose parse_result let a Pydantic
        # ValidationError escape must still fail permanently, never retry.
        raise _record_failed_run(
            session, run, job, PipelineOutputError(_OUTPUT_INVALID_MESSAGE)
        ) from exc
    except ModelsConfigError as exc:
        raise _record_failed_run(session, run, job, PipelineError(_CONFIG_ERROR_MESSAGE)) from exc
    except LLMError as exc:
        if exc.retryable:
            # Transient: the worker retries with backoff. Only the final
            # attempt records a failed run (never multiply failure rows).
            if _is_terminal_attempt(job):
                raise _record_failed_run(session, run, job, exc) from exc
            raise
        # Deterministic provider failure (auth, malformed response, provider
        # unavailable): permanent, like the Task-09 provider mapping.
        raise _record_failed_run(session, run, job, exc) from exc
    except Exception as exc:
        # Unexpected failure: treat as transient (Task-08 policy); the final
        # attempt records the failed run before the job exhausts its budget.
        if _is_terminal_attempt(job):
            raise _record_failed_run(session, run, job, exc) from exc
        raise


def _execute_run(
    session: Session,
    run: ProcessingRun,
    job: Job,
    *,
    pipeline_type: PipelineType,
    export_id: uuid.UUID,
) -> None:
    """Run one pipeline attempt after the run row is ``running``.

    Raises permanent pipeline errors for deterministic failures and lets
    retryable LLM errors propagate. On success finalizes ``run`` as succeeded
    (flush only — the worker commits) and creates the artifact row.
    """
    export = WeeklyExportRepository(session).get(export_id)
    if export is None:
        raise PipelineExportNotFoundError("Export not found.")
    run.input_export_id = export.id

    pipeline = pipeline_registry.get_pipeline(pipeline_type)
    if pipeline is None:
        raise PipelineNotRegisteredError(
            f"Pipeline {pipeline_type.value!r} is not registered yet (implemented in Tasks 13-15)."
        )

    input_artifact = _resolve_input_artifact(session, job, export_id=export.id)
    if input_artifact is None:
        # The job payload carried no explicit input artifact. Pipelines that
        # consume a lower derived layer (layers 3-4 consume ``normalized``)
        # resolve the export's latest succeeded artifact of that layer here,
        # so their input is always layer 2 by database identity and the
        # original export is never read (AGENTS.md §6).
        input_artifact = _resolve_default_input_artifact(
            session, export_id=export.id, pipeline=pipeline
        )
    run.input_artifact_id = input_artifact.id if input_artifact is not None else None

    _ensure_input_file(export=export, input_artifact=input_artifact)

    models_config = _load_models_config()
    profile = resolve_profile(models_config, pipeline.task_type)
    provider = create_llm_provider(profile)

    requests = pipeline.build_requests(export=export, input_artifact=input_artifact)
    if not requests:
        raise PipelineInputError("The pipeline produced no LLM request for this run.")
    call_started = utc_now()
    responses = [provider.complete(request) for request in requests]
    call_finished = utc_now()
    structured = pipeline.parse_responses(responses)

    # The derived version must be known before the artifact is rendered: the
    # layer-2+ headers embed ``vM`` (the Task-13 render-with-context hook).
    version = _next_artifact_version(session, export.id, pipeline.layer)
    if pipeline.render_requires_context:
        rendered = pipeline.render_artifact(structured, export=export, version=version)
    else:
        rendered = pipeline.render(structured)

    filename = derived_filename(
        sunday=export.week_end_date,
        layer=pipeline.layer,
        seq_number=export.seq_number,
        revision=export.revision,
        version=version,
    )
    relative_path = artifact_data_root_relative_path(layer=pipeline.layer, filename=filename)
    target_path = get_settings().data_root / relative_path
    data = rendered.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    _write_artifact_atomically(data, target_path, get_settings().temp_dir)

    ArtifactRepository(session).create(
        layer=pipeline.layer,
        export_id=export.id,
        path=relative_path,
        sha256=digest,
        processing_run_id=run.id,
        version=version,
    )

    _finalize_succeeded_run(
        session,
        run,
        job,
        pipeline=pipeline,
        profile=profile,
        requests=requests,
        responses=responses,
        call_started=call_started,
        call_finished=call_finished,
    )


def _finalize_succeeded_run(
    session: Session,
    run: ProcessingRun,
    job: Job,
    *,
    pipeline: Pipeline[Any],
    profile: OpenAICompatibleProfile,
    requests: list[LLMRequest],
    responses: list[LLMResponse],
    call_started: datetime,
    call_finished: datetime,
) -> None:
    """Record a succeeded run: status, timestamps and merged provenance.

    Provenance never carries diary content or prompts — only the profile,
    the effective model, the prompt version and content digests. Single-request
    runs keep the exact Task-12 metadata shape (``input_sha256`` is a string,
    ``usage`` is one object). Chunked runs (Task 13) record ``input_sha256``
    as the ordered list of per-chunk digests, the summed ``usage`` and a
    ``call_count``; ``metadata_json`` is the deliberately flexible JSON
    metadata column, so the shape difference is documented here.
    """
    model = resolve_completion_params(requests[0], profile)["model"]
    prompt_version = requests[0].prompt_version
    run.status = RunStatus.succeeded
    run.finished_at = call_finished
    run.profile_name = pipeline.task_type.value
    run.model_name = model
    run.prompt_version = prompt_version
    if len(requests) == 1:
        usage = responses[0].usage.model_dump() if responses[0].usage is not None else None
        run.metadata_json = {
            "job_id": str(job.id),
            "profile": pipeline.task_type.value,
            "provider_type": profile.provider,
            "model": model,
            "prompt_version": prompt_version,
            "input_sha256": request_input_sha256(requests[0]),
            "usage": usage,
        }
        return
    run.metadata_json = {
        "job_id": str(job.id),
        "profile": pipeline.task_type.value,
        "provider_type": profile.provider,
        "model": model,
        "prompt_version": prompt_version,
        "input_sha256": [request_input_sha256(request) for request in requests],
        "usage": _merged_usage(responses),
        "call_count": len(requests),
    }


def _merged_usage(responses: list[LLMResponse]) -> dict[str, int] | None:
    """Sum token usage across chunk responses (``None`` when any is missing)."""
    if not responses or any(response.usage is None for response in responses):
        return None
    totals = {"input_tokens": 0, "output_tokens": 0}
    for response in responses:
        usage = response.usage
        assert usage is not None  # guaranteed by the guard above
        totals["input_tokens"] += usage.input_tokens
        totals["output_tokens"] += usage.output_tokens
    return totals


def _resolve_input_artifact(session: Session, job: Job, *, export_id: uuid.UUID) -> Artifact | None:
    """The optional derived input artifact referenced by the job payload.

    Tasks 13-15 use this seam: summarization and wellbeing consume a specific
    normalized artifact (layer 2), and the AGENTS.md invariant "layers 3-4
    build only from layer 2" is enforced by those pipelines' own input checks.
    When the payload has no ``input_artifact_id`` the input is the original
    export itself (``None`` here).
    """
    payload = job.payload_json if isinstance(job.payload_json, dict) else {}
    value = payload.get(PAYLOAD_INPUT_ARTIFACT_ID)
    if value is None:
        return None
    artifact_id = _coerce_uuid(value, field=PAYLOAD_INPUT_ARTIFACT_ID)
    artifact = ArtifactRepository(session).get(artifact_id)
    if artifact is None:
        raise PipelineInputError("The input artifact for this job no longer exists.")
    if artifact.export_id != export_id:
        raise PipelineInputError("The input artifact does not belong to the job's export lineage.")
    return artifact


def _resolve_default_input_artifact(
    session: Session, *, export_id: uuid.UUID, pipeline: Pipeline[Any]
) -> Artifact | None:
    """Resolve the export's input artifact for a layer-consuming pipeline.

    Pipelines that declare ``input_layer`` (summarization/wellbeing consume
    ``normalized``) run against the export's latest succeeded artifact of that
    layer when the job payload carries no explicit ``input_artifact_id``: an
    artifact row of a derived layer exists only after a validated file was
    written by a succeeded run, so the newest row of the layer is the newest
    succeeded version. Returns ``None`` for pipelines without ``input_layer``
    (normalization consumes the original export itself) and raises
    :class:`PipelineInputError` when no artifact of the layer exists yet —
    a permanent, content-free failure instructing the user to run the upstream
    pipeline first.
    """
    if pipeline.input_layer is None:
        return None
    artifacts = ArtifactRepository(session).list_by_export_and_layer(
        export_id, pipeline.input_layer
    )
    if not artifacts:
        raise PipelineInputError(
            f"The export has no {pipeline.input_layer.value} artifact yet; "
            "run the upstream pipeline first."
        )
    return artifacts[-1]


def _ensure_input_file(*, export, input_artifact: Artifact | None) -> None:
    """Verify the input file exists on disk (path resolved by DB identity)."""
    if input_artifact is not None:
        assert input_artifact.path is not None  # a derived input artifact is file-backed
        artifact_path = get_settings().data_root / input_artifact.path
        if not artifact_path.is_file():
            raise PipelineInputError("The input artifact file is missing on disk.")
        return
    original_path = get_settings().data_root / export.path
    if not original_path.is_file():
        raise PipelineOriginalFileMissingError(
            "The original export file is missing; the pipeline cannot run."
        )


# ---------------------------------------------------------------------------
# Run bookkeeping helpers
# ---------------------------------------------------------------------------


def _terminal_run_for_job(session: Session, job_id: uuid.UUID) -> ProcessingRun | None:
    """The last recorded run of ``job_id`` that reached a terminal state."""
    runs = ProcessingRunRepository(session).list_by_job_id(job_id)
    return runs[-1] if runs else None


def _record_failed_run(
    session: Session, run: ProcessingRun, job: Job, error: BaseException
) -> PipelinePermanentError:
    """Mark ``run`` failed with a safe summary, commit, and return a permanent
    error carrying the same summary for the worker to store on the job.

    ``error`` is an exception whose message is already static and
    non-sensitive (domain errors) or is safely summarized here.
    """
    summary = _safe_summary(error)
    run.status = RunStatus.failed
    run.finished_at = utc_now()
    run.error_summary = summary
    run.metadata_json = {
        "job_id": str(job.id),
        "error_class": type(error).__name__,
        "attempts": job.attempts,
        "terminal": True,
    }
    session.commit()
    return PipelinePermanentError(summary)


def _safe_summary(error: BaseException) -> str:
    """A short, non-sensitive summary (class + truncated message, no trace)."""
    message = " ".join(str(error).split())
    if not message:
        return type(error).__name__
    return f"{type(error).__name__}: {message[:MAX_ERROR_SUMMARY_LENGTH]}"


def _is_terminal_attempt(job: Job) -> bool:
    """Whether this claim is the job's last allowed attempt."""
    effective_max = (
        job.max_attempts if job.max_attempts is not None else get_settings().job_max_attempts
    )
    return job.attempts >= effective_max


def _next_artifact_version(session: Session, export_id: uuid.UUID, layer: ArtifactLayer) -> int:
    """Next ``vM`` for the pair (export row, layer): max existing + 1, v1 first."""
    rows = ArtifactRepository(session).list_by_export_and_layer(export_id, layer)
    if not rows:
        return 1
    return (rows[-1].version or 0) + 1


# ---------------------------------------------------------------------------
# Payload helpers (identifiers only)
# ---------------------------------------------------------------------------


def _pipeline_type_for_job(job: Job) -> PipelineType:
    """Map the job type to a pipeline type; non-pipeline jobs are permanent."""
    pipeline_type = JOB_PIPELINE_TYPE.get(job.job_type)
    if pipeline_type is None:
        raise PipelinePermanentError(f"Job type {job.job_type.value!r} is not a pipeline job type.")
    return pipeline_type


def _payload_export_id(job: Job) -> uuid.UUID:
    """The ``export_id`` from a pipeline job payload as a UUID."""
    payload = job.payload_json if isinstance(job.payload_json, dict) else None
    value = payload.get(PAYLOAD_EXPORT_ID) if payload is not None else None
    return _coerce_uuid(value, field=PAYLOAD_EXPORT_ID)


def _coerce_uuid(value: object, *, field: str) -> uuid.UUID:
    """Coerce a payload value (str or UUID) into a UUID."""
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        try:
            return uuid.UUID(value)
        except ValueError:
            pass
    raise PipelineInputError(f"The pipeline job payload has no valid {field}.")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _unlink_safe(path: Path) -> None:
    """Remove a file, ignoring "already gone" errors."""
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort cleanup
        pass


def _write_artifact_atomically(data: bytes, target_path: Path, temp_dir: Path) -> None:
    """Write a derived artifact atomically (AGENTS.md §12, data-lineage §5).

    A unique temporary file inside ``temp_dir`` is written, flushed and
    fsynced, then renamed over the final path. When the temporary directory
    and the target directory live on different filesystems (``rename`` raises
    ``EXDEV``) a copy+delete fallback is used — the payload is already fsynced,
    so durability does not depend on the rename being atomic. No partial file
    ever appears at the target path. Every filesystem failure — including an
    unavailable target directory — is mapped to a permanent
    :class:`PipelineStorageError` with a safe, static summary.
    """
    try:
        temp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_path = temp_dir / f"artifact-{uuid.uuid4().hex}.tmp"
        try:
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            raise PipelineStorageError("Failed to write the pipeline artifact file.") from exc
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            _unlink_safe(temp_path)
            raise
        try:
            os.replace(temp_path, target_path)
        except OSError as exc:
            _unlink_safe(temp_path)
            if exc.errno != errno.EXDEV:
                raise PipelineStorageError("Failed to write the pipeline artifact file.") from exc
            try:
                shutil.move(str(temp_path), str(target_path))
            except BaseException:
                _unlink_safe(target_path)
                _unlink_safe(temp_path)
                raise
    except OSError as exc:
        raise PipelineStorageError("Failed to write the pipeline artifact file.") from exc


__all__ = [
    "PAYLOAD_EXPORT_ID",
    "PAYLOAD_INPUT_ARTIFACT_ID",
    "PAYLOAD_PIPELINE_TYPE",
    "PipelineServiceError",
    "enqueue_pipeline_run",
    "find_active_pipeline_job",
    "run_pipeline",
]
