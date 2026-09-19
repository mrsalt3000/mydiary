"""Registry of pipeline implementations (Task 12, extended by task T04).

Concrete pipelines register a stateless instance under its ``pipeline_type``
here, keyed by that type. The framework core (``diary.domain.pipelines
.service``) only *looks up* the registry, so a job may be enqueued before its
pipeline implementation exists; executing such a job fails permanently with a
clear "not registered yet" summary instead of retrying.

Two kinds of layer are registered, in two dicts:

* :data:`REGISTRY` — the export-driven layers of the MVP (normalization in
  Task 13, summarization in Task 14, wellbeing in Task 15), which subclass
  :class:`~diary.domain.pipelines.base.Pipeline` and are driven through a
  weekly export;
* :data:`SECOND_FAMILY_REGISTRY` — the phase-2 layers of the second anchor
  family (ADR-007; ``stt_corrected`` first, task T04), which subclass
  :class:`~diary.domain.pipelines.base.SecondFamilyLayer` and are driven
  directly from their own source object (message/day/week/upstream summary).
  They are read through :func:`get_second_family_pipeline`, never through
  :func:`get_pipeline`: the export-shaped framework cannot execute them.

:func:`register` is the single registration entry point for both kinds and
dispatches on the class kind; each kind is validated against the naming-
convention maps, so a pipeline cannot accidentally advertise a layer/extension
(or task type/profile name) that contradicts its declared pipeline type.
"""

from __future__ import annotations

from typing import Any

from diary.db.models import PipelineType
from diary.domain.pipelines.base import Pipeline, SecondFamilyLayer
from diary.domain.pipelines.naming import (
    LAYER_OUTPUT_EXTENSIONS,
    PIPELINE_LAYER,
    PIPELINE_PROFILE_NAME,
    PIPELINE_TASK_TYPE,
)

#: Registered export-driven pipeline instances, keyed by ``PipelineType.value``.
REGISTRY: dict[str, Pipeline[Any]] = {}

#: Registered phase-2 (second anchor family) layers, keyed by ``PipelineType.value``.
SECOND_FAMILY_REGISTRY: dict[str, SecondFamilyLayer[Any]] = {}


def register(pipeline: Pipeline[Any] | SecondFamilyLayer[Any]) -> None:
    """Register ``pipeline`` for its ``pipeline_type`` (raises on duplicates).

    Dispatches on the kind of the implementation: an export-driven
    :class:`~diary.domain.pipelines.base.Pipeline` goes to :data:`REGISTRY`, a
    :class:`~diary.domain.pipelines.base.SecondFamilyLayer` to
    :data:`SECOND_FAMILY_REGISTRY`. In both cases the class attributes are
    checked against the naming-convention maps before registration, so a
    mis-declared layer, extension, task type or profile name cannot silently
    produce artifacts in the wrong place or under a wrong profile.
    """
    if isinstance(pipeline, Pipeline):
        _register_export_pipeline(pipeline)
    elif isinstance(pipeline, SecondFamilyLayer):
        _register_second_family_layer(pipeline)
    else:
        raise TypeError(
            "A pipeline must be a Pipeline or a SecondFamilyLayer, "
            f"got {type(pipeline).__name__!r}."
        )


def _register_export_pipeline(pipeline: Pipeline[Any]) -> None:
    """Validate and register an export-driven pipeline (MVP layers 2-4)."""
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


def _register_second_family_layer(layer: SecondFamilyLayer[Any]) -> None:
    """Validate and register a phase-2 layer of the second anchor family.

    Such a layer has no ``TaskType``: it selects its LLM profile by the
    configuration field name declared in ``PIPELINE_PROFILE_NAME``.
    """
    pipeline_type = layer.pipeline_type
    if not isinstance(pipeline_type, PipelineType):
        raise TypeError(f"A layer must declare a PipelineType, got {pipeline_type!r}.")
    if pipeline_type in PIPELINE_TASK_TYPE:
        raise ValueError(
            f"Pipeline {pipeline_type.value!r} belongs to the export anchor family and must "
            "be registered as an export-driven Pipeline, not as a second-family layer."
        )
    expected_layer = PIPELINE_LAYER.get(pipeline_type)
    if expected_layer is None:
        raise ValueError(
            f"No artifact layer is declared for pipeline type {pipeline_type.value!r}."
        )
    if layer.layer is not expected_layer:
        raise ValueError(
            f"Pipeline {pipeline_type.value!r} must declare layer "
            f"{expected_layer.value!r}, got {layer.layer.value!r}."
        )
    expected_extension = LAYER_OUTPUT_EXTENSIONS.get(expected_layer)
    if layer.output_ext != expected_extension:
        raise ValueError(
            f"Pipeline {pipeline_type.value!r} must declare output extension "
            f"{expected_extension!r}, got {layer.output_ext!r}."
        )
    expected_profile = PIPELINE_PROFILE_NAME.get(pipeline_type)
    if layer.profile_name != expected_profile:
        raise ValueError(
            f"Pipeline {pipeline_type.value!r} must declare profile name "
            f"{expected_profile!r}, got {layer.profile_name!r}."
        )
    if pipeline_type.value in SECOND_FAMILY_REGISTRY:
        raise ValueError(f"A pipeline for type {pipeline_type.value!r} is already registered.")
    SECOND_FAMILY_REGISTRY[pipeline_type.value] = layer


def get_pipeline(pipeline_type: PipelineType | str) -> Pipeline[Any] | None:
    """Return the registered export-driven pipeline for ``pipeline_type``, or ``None``.

    Phase-2 layers are deliberately not returned here: they are not driven by a
    weekly export and must be read through
    :func:`get_second_family_pipeline`.
    """
    key = pipeline_type.value if isinstance(pipeline_type, PipelineType) else pipeline_type
    return REGISTRY.get(key)


def get_second_family_pipeline(
    pipeline_type: PipelineType | str,
) -> SecondFamilyLayer[Any] | None:
    """Return the registered phase-2 layer for ``pipeline_type``, or ``None``."""
    key = pipeline_type.value if isinstance(pipeline_type, PipelineType) else pipeline_type
    return SECOND_FAMILY_REGISTRY.get(key)


def known_pipeline_types() -> frozenset[str]:
    """The pipeline types that currently have a registered implementation."""
    return frozenset(REGISTRY) | frozenset(SECOND_FAMILY_REGISTRY)


__all__ = [
    "REGISTRY",
    "SECOND_FAMILY_REGISTRY",
    "get_pipeline",
    "get_second_family_pipeline",
    "known_pipeline_types",
    "register",
]
