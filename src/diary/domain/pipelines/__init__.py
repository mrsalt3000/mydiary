"""Derived-artifact pipeline framework (Task 12, architecture brief §10).

The package provides the shared orchestration used by the normalization,
summarization and wellbeing pipelines (implemented in Tasks 13-15):

* :mod:`diary.domain.pipelines.base` — the abstract :class:`Pipeline`
  interface (request building, structured-output parsing, backend rendering)
  and the permanent-failure error taxonomy;
* :mod:`diary.domain.pipelines.naming` — derived-artifact naming and layer
  conventions (``YYYY-MM-DD_<layer>_NNNN_rN_vM.<ext>``, per-layer
  directories/extensions, pipeline/job/task type mappings);
* :mod:`diary.domain.pipelines.registry` — the registry of pipeline
  implementations (empty by default; Tasks 13-15 register the real ones);
* :mod:`diary.domain.pipelines.service` — the use cases: job enqueueing and
  the shared ``queued → running → succeeded | failed`` processing-run
  lifecycle with replay idempotency.

No diary content, prompts or LLM responses are ever logged or persisted
anywhere in this package (AGENTS.md §7-8). Concrete pipelines (prompts,
result schemas, renderers) are intentionally out of scope for Task 12.
"""

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
    LAYER_DIRNAMES,
    LAYER_OUTPUT_EXTENSIONS,
    PIPELINE_JOB_TYPE,
    PIPELINE_LAYER,
    PIPELINE_TASK_TYPE,
    artifact_data_root_relative_path,
    derived_filename,
    layer_dirname,
    layer_output_extension,
)
from diary.domain.pipelines.registry import (
    REGISTRY,
    get_pipeline,
    known_pipeline_types,
    register,
)
from diary.domain.pipelines.service import (
    enqueue_pipeline_run,
    find_active_pipeline_job,
    run_pipeline,
)

__all__ = [
    "JOB_PIPELINE_TYPE",
    "LAYER_DIRNAMES",
    "LAYER_OUTPUT_EXTENSIONS",
    "PIPELINE_JOB_TYPE",
    "PIPELINE_LAYER",
    "PIPELINE_TASK_TYPE",
    "REGISTRY",
    "Pipeline",
    "PipelineError",
    "PipelineExportNotFoundError",
    "PipelineInputError",
    "PipelineNotRegisteredError",
    "PipelineOriginalFileMissingError",
    "PipelineOutputError",
    "PipelinePermanentError",
    "PipelineStorageError",
    "artifact_data_root_relative_path",
    "derived_filename",
    "enqueue_pipeline_run",
    "find_active_pipeline_job",
    "get_pipeline",
    "known_pipeline_types",
    "layer_dirname",
    "layer_output_extension",
    "register",
    "run_pipeline",
]
