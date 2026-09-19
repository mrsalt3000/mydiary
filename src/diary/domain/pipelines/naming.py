"""Derived-artifact naming and layer conventions (Task 12).

Implements the derived-artifact naming rules fixed in tasklist Task 12 and
``docs/data-lineage.md`` §3-4:

``YYYY-MM-DD_<layer>_NNNN_rN_vM.<ext>``

- ``YYYY-MM-DD`` — the Sunday date of the source week (from the export row);
- ``<layer>`` — the layer token (``normalized``/``summary``/``wellbeing``);
- ``NNNN`` — the global no-gap sequence of the **source original export**;
- ``rN`` — the revision of the source original export;
- ``vM`` — the version of the derived result for the pair (export row, layer);
- ``<ext>`` — ``.txt`` for normalized text, ``.md`` for summary/wellbeing.

The module is the single source of truth for the per-layer directories below
``<exports_dir>``, the per-layer file extensions, and the mappings between
pipeline/job/task enum values and artifact layers (so a registered pipeline
can be validated for internal consistency). Stored artifact paths are relative
to the data root with forward slashes, mirroring
``diary.domain.exports.service`` (e.g.
``exports/02_normalized/2026-09-13_normalized_0001_r1_v1.txt``).

Task T04 adds the first **phase-2** entry: the ``stt_corrected`` layer of the
second anchor family (ADR-007); task T05 adds the ``daily_summary`` layer, task
T06 the ``weekly_summary`` layer and task T07 the ``coach_note`` layer (layer 5,
separate from the layer-4 ``wellbeing`` pipeline) of the same family. Such a layer has no
weekly export, so it has no ``NNNN_rN`` file name and no layer directory — its
validated result is stored inline by ``diary.db.results.save_layer_result``
(ADR-008). Only the three maps the registry validates against are extended for
each of them: its artifact layer, its rendered-file extension and its job type;
its LLM profile is selected by the configuration field name
(``PIPELINE_PROFILE_NAME``), not by a ``TaskType``.
"""

from __future__ import annotations

from datetime import date

from diary.core.models_config import TaskType
from diary.db.models import ArtifactLayer, JobType, PipelineType

#: Data-root-relative prefix of every export artifact directory.
EXPORTS_PREFIX = "exports"

#: Per-layer directory below ``<exports_dir>`` (data-lineage.md §4).
LAYER_DIRNAMES: dict[ArtifactLayer, str] = {
    ArtifactLayer.original: "01_original",
    ArtifactLayer.normalized: "02_normalized",
    ArtifactLayer.summary: "03_summaries",
    ArtifactLayer.wellbeing: "04_wellbeing",
}

#: Per-layer file extension of rendered artifacts.
LAYER_OUTPUT_EXTENSIONS: dict[ArtifactLayer, str] = {
    ArtifactLayer.original: ".txt",
    ArtifactLayer.normalized: ".txt",
    ArtifactLayer.summary: ".md",
    ArtifactLayer.wellbeing: ".md",
    # Phase-2 layer (task T04): the corrected message text is plain text.
    ArtifactLayer.stt_corrected: ".txt",
    # Phase-2 layer (task T05): a day summary is rendered as Markdown.
    ArtifactLayer.daily_summary: ".md",
    # Phase-2 layer (task T06): a week summary is rendered as Markdown.
    ArtifactLayer.weekly_summary: ".md",
    # Phase-2 layer (task T07, layer 5): a coaching note is rendered as Markdown.
    ArtifactLayer.coach_note: ".md",
}

#: Derived pipeline kind → the artifact layer it produces.
PIPELINE_LAYER: dict[PipelineType, ArtifactLayer] = {
    PipelineType.normalize: ArtifactLayer.normalized,
    PipelineType.summarize: ArtifactLayer.summary,
    PipelineType.wellbeing: ArtifactLayer.wellbeing,
    # Phase-2 layer (task T04, second anchor family): one message per run.
    PipelineType.stt_corrected: ArtifactLayer.stt_corrected,
    # Phase-2 layer (task T05, second anchor family): one local day per run.
    PipelineType.daily_summary: ArtifactLayer.daily_summary,
    # Phase-2 layer (task T06, second anchor family): one ISO week per run.
    PipelineType.weekly_summary: ArtifactLayer.weekly_summary,
    # Phase-2 layer (task T07, layer 5, second anchor family): one upstream
    # weekly-summary artifact per run. Separate from ``PipelineType.wellbeing``.
    PipelineType.coach_note: ArtifactLayer.coach_note,
}

#: Derived pipeline kind → the Task-03 LLM profile task type it uses.
#:
#: Only the export-driven MVP pipelines appear here: their LLM profile is
#: selected by :class:`~diary.core.models_config.TaskType`. The phase-2 layers
#: are configured by *profile name* instead (``PIPELINE_PROFILE_NAME``), because
#: their profiles (``diary_stt``/``diary_sum``/``diary_coach``) are optional
#: additions to ``configs/models.yaml`` and the four MVP task types are not
#: extended by the phase-2 backlog (task T02/T04).
PIPELINE_TASK_TYPE: dict[PipelineType, TaskType] = {
    PipelineType.normalize: TaskType.normalization,
    PipelineType.summarize: TaskType.summarization,
    PipelineType.wellbeing: TaskType.wellbeing_analysis,
}

#: Phase-2 pipeline kind → the ``configs/models.yaml`` profile field it uses.
#:
#: The phase-2 counterpart of :data:`PIPELINE_TASK_TYPE`: a layer of the second
#: anchor family selects its LLM profile by the configuration field name, and an
#: absent profile means "the layer is switched off" (``docs/tasklist-phase2.md``
#: §9.8). Populated per layer as T04-T07 are implemented.
PIPELINE_PROFILE_NAME: dict[PipelineType, str] = {
    PipelineType.stt_corrected: "diary_stt",
    PipelineType.daily_summary: "diary_sum",
    # Phase-2 layer (task T06): the week merge runs under the same summarization
    # profile as the day layer (``diary-sum``, temperature 0.25).
    PipelineType.weekly_summary: "diary_sum",
    # Phase-2 layer (task T07, layer 5): its own coaching profile (temperature
    # 0.4). It is not the layer-4 wellbeing task type.
    PipelineType.coach_note: "diary_coach",
}

#: Derived pipeline kind → the durable job type that drives its execution.
PIPELINE_JOB_TYPE: dict[PipelineType, JobType] = {
    PipelineType.normalize: JobType.normalize,
    PipelineType.summarize: JobType.summarize,
    PipelineType.wellbeing: JobType.wellbeing,
    # Phase-2 layer (task T04): the job type of the same name (ADR-008 §3).
    PipelineType.stt_corrected: JobType.stt_corrected,
    # Phase-2 layer (task T05): the job type of the same name (ADR-008 §3).
    PipelineType.daily_summary: JobType.daily_summary,
    # Phase-2 layer (task T06): the job type of the same name (ADR-008 §3).
    PipelineType.weekly_summary: JobType.weekly_summary,
    # Phase-2 layer (task T07, layer 5): the job type of the same name
    # (ADR-008 §3). The layer registers no handler for it, so an enqueued
    # ``coach_note`` job fails permanently instead of running by itself.
    PipelineType.coach_note: JobType.coach_note,
}

#: Reverse mapping: job type → pipeline kind.
JOB_PIPELINE_TYPE: dict[JobType, PipelineType] = {
    job_type: pipeline_type for pipeline_type, job_type in PIPELINE_JOB_TYPE.items()
}


def layer_dirname(layer: ArtifactLayer) -> str:
    """Subdirectory of an artifact layer below ``<exports_dir>``.

    For example ``ArtifactLayer.normalized`` → ``"02_normalized"``.
    """
    try:
        return LAYER_DIRNAMES[layer]
    except KeyError as exc:
        raise ValueError(f"Unknown artifact layer {layer!r}.") from exc


def layer_output_extension(layer: ArtifactLayer) -> str:
    """File extension of rendered artifacts of ``layer`` (``.txt``/``.md``)."""
    try:
        return LAYER_OUTPUT_EXTENSIONS[layer]
    except KeyError as exc:
        raise ValueError(f"Unknown artifact layer {layer!r}.") from exc


def derived_filename(
    *,
    sunday: date,
    layer: ArtifactLayer,
    seq_number: int,
    revision: int,
    version: int,
) -> str:
    """Derived artifact file name per the Task-12 naming rules.

    ``sunday``/``seq_number``/``revision`` come from the source original
    export row (the layer-1 lineage anchor); ``version`` is the derived
    version ``vM`` for the pair (export row, layer). The layer token and the
    extension are derived from ``layer`` so the stored name can never diverge
    from the layer mapping. Only derived layers (2-4) carry a ``vM`` suffix.
    """
    extension = layer_output_extension(layer)
    return f"{sunday.isoformat()}_{layer.value}_{seq_number:04d}_r{revision}_v{version}{extension}"


def artifact_data_root_relative_path(*, layer: ArtifactLayer, filename: str) -> str:
    """Data-root-relative artifact path for ``filename`` of ``layer``.

    Returns e.g. ``"exports/02_normalized/<filename>"``; joining it with the
    data root yields the absolute file location stored in ``artifacts.path``.
    """
    return f"{EXPORTS_PREFIX}/{layer_dirname(layer)}/{filename}"


__all__ = [
    "EXPORTS_PREFIX",
    "JOB_PIPELINE_TYPE",
    "LAYER_DIRNAMES",
    "LAYER_OUTPUT_EXTENSIONS",
    "PIPELINE_JOB_TYPE",
    "PIPELINE_LAYER",
    "PIPELINE_PROFILE_NAME",
    "PIPELINE_TASK_TYPE",
    "artifact_data_root_relative_path",
    "derived_filename",
    "layer_dirname",
    "layer_output_extension",
]
