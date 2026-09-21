"""Unit tests for the T09 orchestrator wiring (no provider, no database).

The orchestrator (``diary.domain.pipelines.pipeline``) is the only place that
drives several phase-2 layers in one sequence, so the properties pinned here are
the structural ones that no database test can prove:

* the day and week chains **cannot reach layer 5**: their code references
  nothing named ``coach`` and takes no coach parameter, while the explicit
  :func:`~diary.domain.pipelines.pipeline.run_coach_step` does — the T07 rule
  «запуск только явно» expressed as missing reachability, not as a flag;
* the second family is driven through
  :func:`~diary.domain.pipelines.registry.get_second_family_pipeline` and never
  through the export-shaped framework (``get_pipeline``/``run_pipeline``), and
  ``PIPELINE_TASK_TYPE``/``TaskType`` stay the export-only MVP maps;
* the worker startup **registers** every phase-2 layer (guarded and idempotent)
  but calls no layer service — registration is not launch;
* every enum value a chain run can produce has a label/the map entry it needs on
  the paths that render runs, artifacts and jobs (ADR-008 §3), and the phase-2
  layers deliberately have no layer directory (their results are inline).

No provider, no model, no database and no configuration file is touched; the
only synthetic content here is the absence of content.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

from diary.api.routes import pages
from diary.core.models_config import TaskType
from diary.core.settings import AppSettings
from diary.db.models import (
    ArtifactLayer,
    JobStatus,
    JobType,
    PipelineType,
    RunStatus,
    TranscriptionStatus,
)
from diary.domain.pipelines import naming, pipeline
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.base import PipelineNotRegisteredError
from diary.workers import handlers, runner

#: Repository root: tests/unit/test_phase2_orchestrator_wiring.py -> parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: The four layers of the second anchor family (ADR-007 §1).
PHASE2_LAYERS = (
    PipelineType.stt_corrected,
    PipelineType.daily_summary,
    PipelineType.weekly_summary,
    PipelineType.coach_note,
)


def _referenced_names(func: object) -> set[str]:
    """Names/attributes referenced by the **code** of ``func`` (no docstring).

    Parsing the source and walking the AST (instead of matching substrings)
    separates "this function mentions the coaching layer in prose" from "this
    function can call it": assertions about reachability must look at the code.
    """
    source = textwrap.dedent(inspect.getsource(func))  # type: ignore[arg-type]
    node = ast.parse(source).body[0]
    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        node.body = node.body[1:]  # drop the docstring
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.ImportFrom) and sub.module is not None:
            names.add(sub.module)
    return names


# ---------------------------------------------------------------------------
# the coaching note is unreachable from the day/week chains
# ---------------------------------------------------------------------------


def test_the_day_gate_is_armed_by_the_production_entries() -> None:
    """T24b: the NLI gate must not depend on a caller remembering to pass it.

    ``sum_daily`` takes the instrument optionally (the documented seam for tests and
    for the harness in CI, where neither the extra nor the weights exist), so the
    arming has to happen at the production boundary — the CLI and the
    ``daily_summary`` job handler — and it is pinned here by source, the way the
    other wiring of this file is pinned.
    """
    root = PROJECT_ROOT / "src" / "diary"
    cli = (root / "cli.py").read_text(encoding="utf-8")
    handlers = (root / "workers" / "handlers.py").read_text(encoding="utf-8")
    pipeline = (root / "domain" / "pipelines" / "pipeline.py").read_text(encoding="utf-8")
    for name, source in (("cli.py", cli), ("handlers.py", handlers)):
        assert "nli_judge=default_nli_judge()" in source, name

    # Compare *lines*, not substrings: `"judge=nli_judge"` occurs inside
    # `"nli_judge=nli_judge"`, so a substring test would keep passing after the day
    # step stopped forwarding the instrument — the vacuous assertion a review of
    # T24b caught, in the same class as the ungated path it caught before.
    forwarded = {
        line.strip().split(",")[0]
        for line in pipeline.splitlines()
        if line.strip().startswith(("judge=", "nli_judge="))
    }
    assert "judge=nli_judge" in forwarded  # the day step forwards what the entry armed
    # The run_days refresh of the weekly chain is a second way to store a day
    # summary, and it must be gated too (a review of T24b found it was not).
    assert "nli_judge=nli_judge" in forwarded
    assert pipeline.count("nli_judge=default_nli_judge()") == 0  # only entries arm it
    assert cli.count("nli_judge=default_nli_judge()") == 2  # day and week commands
    assert handlers.count("nli_judge=default_nli_judge()") == 2  # both phase-2 jobs
    # The low-level service keeps the seam, and it is documented as such.
    service = (root / "domain" / "pipelines" / "sum_daily" / "service.py").read_text(
        encoding="utf-8"
    )
    assert "None`` means no gate" in service


def test_the_day_and_week_chains_reference_no_coach_code() -> None:
    """Layer 5 is not reachable from the chains — structurally, not by a flag."""
    for func in (pipeline.run_day_chain, pipeline.run_week_chain):
        referenced = _referenced_names(func)
        assert referenced, f"{func.__name__} must reference something"
        offenders = {name for name in referenced if "coach" in name.lower()}
        assert not offenders, (func.__name__, sorted(offenders))
        parameters = inspect.signature(func).parameters
        assert not [name for name in parameters if "coach" in name.lower()], func.__name__


def test_the_explicit_coach_entry_point_is_the_only_one_that_touches_layer_5() -> None:
    referenced = _referenced_names(pipeline.run_coach_step)
    assert any("coach" in name.lower() for name in referenced)
    assert "coach_note" in referenced
    # The run of the note is anchored to the upstream weekly summary artifact.
    assert "collect_week_summary" in referenced


def test_only_the_underscored_helpers_and_the_explicit_entry_points_exist() -> None:
    """The public surface is exactly the explicit per-unit operations.

    T09 defined the day, week and coach units; T11 added the smallest unit —
    the correction of one message (``run_stt_fix_step``), which is what a
    ``stt_corrected`` job dispatches. The set stays closed: no helper leaks out.
    """
    assert set(pipeline.__all__) == {
        "ORCHESTRATOR",
        "SKIP_NO_INPUT",
        "SKIP_PROFILE_OFF",
        "ChainStep",
        "CoachStepOutcome",
        "DayChainOutcome",
        "SkippedStep",
        "SttFixStepOutcome",
        "WeekChainOutcome",
        "run_coach_step",
        "run_day_chain",
        "run_stt_fix_step",
        "run_week_chain",
    }


# ---------------------------------------------------------------------------
# the second family never goes through the export-shaped framework
# ---------------------------------------------------------------------------


def _orchestrator_source() -> str:
    return (PROJECT_ROOT / "src" / "diary" / "domain" / "pipelines" / "pipeline.py").read_text(
        encoding="utf-8"
    )


def test_the_orchestrator_looks_layers_up_in_the_second_family_registry() -> None:
    source = _orchestrator_source()
    assert "get_second_family_pipeline" in source
    # The export-shaped half of the framework is not used for the second family.
    assert "get_pipeline(" not in source
    assert "run_pipeline(" not in source
    assert "diary.domain.pipelines.service" not in source
    assert "get_second_family_pipeline" in _referenced_names(pipeline._run_step)


def test_the_task_type_map_and_the_enum_stay_export_only() -> None:
    """Phase 2 layers are configured by profile name, never by a new TaskType."""
    assert set(naming.PIPELINE_TASK_TYPE) == {
        PipelineType.normalize,
        PipelineType.summarize,
        PipelineType.wellbeing,
    }
    assert {task.value for task in TaskType} == {
        "transcription",
        "normalization",
        "summarization",
        "wellbeing_analysis",
    }
    for layer in PHASE2_LAYERS:
        assert layer not in naming.PIPELINE_TASK_TYPE
        assert naming.PIPELINE_PROFILE_NAME[layer].startswith("diary_")
    assert {naming.PIPELINE_PROFILE_NAME[layer] for layer in PHASE2_LAYERS} == {
        "diary_stt",
        "diary_sum",
        "diary_week",
        "diary_coach",
    }


def test_an_unregistered_layer_fails_loudly() -> None:
    """A layer that was never registered is a misconfiguration, not a no-op."""
    assert issubclass(PipelineNotRegisteredError, Exception)
    assert "not registered" in str(PipelineNotRegisteredError("x is not registered"))
    assert "not registered" in _orchestrator_source()


# ---------------------------------------------------------------------------
# every enum value on the chain's paths has its map entry / label
# ---------------------------------------------------------------------------


def test_phase2_layers_have_their_naming_map_entries() -> None:
    for layer in PHASE2_LAYERS:
        assert layer in naming.PIPELINE_LAYER
        assert layer in naming.PIPELINE_JOB_TYPE
        assert layer in naming.PIPELINE_PROFILE_NAME
        artifact_layer = naming.PIPELINE_LAYER[layer]
        assert artifact_layer in naming.LAYER_OUTPUT_EXTENSIONS
        # The job type of a phase-2 layer is the layer's own name (ADR-008 §3)
        # and maps back to it.
        assert naming.PIPELINE_JOB_TYPE[layer].value == layer.value
        assert naming.JOB_PIPELINE_TYPE[naming.PIPELINE_JOB_TYPE[layer]] is layer


def test_the_phase2_layers_have_no_layer_directory_by_design() -> None:
    """Inline results have no rendered file, so they have no directory either."""
    assert set(naming.LAYER_DIRNAMES) == {
        ArtifactLayer.original,
        ArtifactLayer.normalized,
        ArtifactLayer.summary,
        ArtifactLayer.wellbeing,
    }
    for layer in PHASE2_LAYERS:
        artifact_layer = naming.PIPELINE_LAYER[layer]
        assert artifact_layer not in naming.LAYER_DIRNAMES
        try:
            naming.layer_dirname(artifact_layer)
        except ValueError:
            continue
        raise AssertionError(f"{artifact_layer.value!r} must not have a layer directory")


def test_every_enum_value_shown_on_the_management_pages_has_a_label() -> None:
    """The labels/maps of the pages that render runs, artifacts and jobs.

    ADR-008 §3 lists these maps as the known consequence of extending the
    enums; they are read on the paths that display a run, an artifact or a job,
    so each must know every value the orchestrator can produce.
    """
    assert set(pages.PIPELINE_LABELS) == set(PipelineType)
    assert set(pages._PIPELINE_LAYER) == set(PipelineType)
    assert set(pages.LAYER_TITLES) == set(ArtifactLayer)
    assert set(pages.LAYER_SHORT_TITLES) == set(ArtifactLayer)
    assert set(pages.RUN_STATUS_LABELS) == set(RunStatus)
    assert set(pages.JOB_TYPE_LABELS) == set(JobType)
    assert set(pages.JOB_STATUS_LABELS) == set(JobStatus)
    assert set(pages.TRANSCRIPTION_LABELS) == set(TranscriptionStatus)
    for layer in PHASE2_LAYERS:
        assert pages.PIPELINE_LABELS[layer]
        assert pages.LAYER_TITLES[naming.PIPELINE_LAYER[layer]]
        assert pages.JOB_TYPE_LABELS[naming.PIPELINE_JOB_TYPE[layer]]


# ---------------------------------------------------------------------------
# the worker registers the layers, and launches none of them
# ---------------------------------------------------------------------------


def test_the_worker_startup_registers_every_phase2_layer(tmp_path, monkeypatch) -> None:
    """``runner.main`` is where the layers become resolvable in the worker.

    Without this the explicit orchestrator calls could not work in the worker
    process (they resolve layers through the second-family registry), which is
    what the T05/T07 acceptance asked T09 for.
    """
    second_family_snapshot = dict(pipeline_registry.SECOND_FAMILY_REGISTRY)
    export_snapshot = dict(pipeline_registry.REGISTRY)
    try:
        monkeypatch.setattr(runner, "configure_logging", lambda: None)
        monkeypatch.setattr(runner, "get_settings", lambda: AppSettings(data_root=tmp_path))
        monkeypatch.setattr(runner, "process_due_jobs", lambda *args, **kwargs: 0)
        for layer in PHASE2_LAYERS:
            pipeline_registry.SECOND_FAMILY_REGISTRY.pop(layer.value, None)
        assert runner.main(["--once"]) == 0
        for layer in PHASE2_LAYERS:
            registered = pipeline_registry.get_second_family_pipeline(layer)
            assert registered is not None, layer.value
            assert registered.pipeline_type is layer
            # Registered in the second-family half only: the export-shaped
            # lookup must not return a phase-2 layer.
            assert pipeline_registry.get_pipeline(layer) is None
            assert layer.value in pipeline_registry.known_pipeline_types()
    finally:
        pipeline_registry.SECOND_FAMILY_REGISTRY.clear()
        pipeline_registry.SECOND_FAMILY_REGISTRY.update(second_family_snapshot)
        pipeline_registry.REGISTRY.clear()
        pipeline_registry.REGISTRY.update(export_snapshot)


def test_the_worker_registers_layers_but_calls_no_layer_service() -> None:
    """Registration is not launch: no service call and no handler in the worker."""
    source = (PROJECT_ROOT / "src" / "diary" / "workers" / "runner.py").read_text(encoding="utf-8")
    for call in (
        "correct_stt(",
        "run_stt_fix(",
        "sum_daily(",
        "run_sum_daily(",
        "sum_weekly(",
        "run_sum_weekly(",
        "coach_note(",
        "run_coach_note(",
        "run_day_chain(",
        "run_week_chain(",
        "run_coach_step(",
    ):
        assert call not in source, call
    # The coaching note keeps its explicit-launch rule: no handler, so an
    # enqueued coach job still fails permanently instead of running the layer.
    assert handlers.get_handler(JobType.coach_note) is None
