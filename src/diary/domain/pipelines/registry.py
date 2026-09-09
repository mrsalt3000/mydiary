"""Registry of pipeline implementations (Task 12).

Concrete pipelines (normalization in Task 13, summarization in Task 14,
wellbeing in Task 15) register a stateless :class:`Pipeline` instance under
its ``pipeline_type`` here. The framework core (``diary.domain.pipelines
.service``) only *looks up* the registry, so a job may be enqueued before its
pipeline implementation exists; executing such a job fails permanently with a
clear "not registered yet" summary instead of retrying.

Registering is validated against the naming-convention maps so a pipeline
cannot accidentally advertise a layer/extension/task that contradicts its
declared pipeline type (e.g. a "summarize" pipeline producing ``.txt``).
"""

from __future__ import annotations

from typing import Any

from diary.db.models import PipelineType
from diary.domain.pipelines.base import Pipeline
from diary.domain.pipelines.naming import (
    LAYER_OUTPUT_EXTENSIONS,
    PIPELINE_LAYER,
    PIPELINE_TASK_TYPE,
)

#: Registered pipeline instances, keyed by ``PipelineType.value``.
REGISTRY: dict[str, Pipeline[Any]] = {}


def register(pipeline: Pipeline[Any]) -> None:
    """Register ``pipeline`` for its ``pipeline_type`` (raises on duplicates).

    The pipeline's class attributes are checked against the naming-convention
    maps before registration, so a mis-declared layer/extension/task cannot
    silently produce artifacts in the wrong directory or with the wrong name.
    """
    pipeline_type = pipeline.pipeline_type
    if not isinstance(pipeline_type, PipelineType):
        raise TypeError(f"A pipeline must declare a PipelineType, got {pipeline_type!r}.")
    expected_layer = PIPELINE_LAYER[pipeline_type]
    if pipeline.layer is not expected_layer:
        raise ValueError(
            f"Pipeline {pipeline_type.value!r} must declare layer "
            f"{expected_layer.value!r}, got {pipeline.layer.value!r}."
        )
    expected_extension = LAYER_OUTPUT_EXTENSIONS[expected_layer]
    if pipeline.output_ext != expected_extension:
        raise ValueError(
            f"Pipeline {pipeline_type.value!r} must declare output extension "
            f"{expected_extension!r}, got {pipeline.output_ext!r}."
        )
    expected_task = PIPELINE_TASK_TYPE[pipeline_type]
    if pipeline.task_type is not expected_task:
        raise ValueError(
            f"Pipeline {pipeline_type.value!r} must declare task type "
            f"{expected_task.value!r}, got {pipeline.task_type.value!r}."
        )
    if pipeline_type.value in REGISTRY:
        raise ValueError(f"A pipeline for type {pipeline_type.value!r} is already registered.")
    REGISTRY[pipeline_type.value] = pipeline


def get_pipeline(pipeline_type: PipelineType | str) -> Pipeline[Any] | None:
    """Return the registered pipeline for ``pipeline_type``, or ``None``."""
    key = pipeline_type.value if isinstance(pipeline_type, PipelineType) else pipeline_type
    return REGISTRY.get(key)


def known_pipeline_types() -> frozenset[str]:
    """The pipeline types that currently have a registered implementation."""
    return frozenset(REGISTRY)


__all__ = [
    "REGISTRY",
    "get_pipeline",
    "known_pipeline_types",
    "register",
]
