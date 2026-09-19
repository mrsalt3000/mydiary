"""Unit tests for pipeline naming, registry validation and atomic writes.

The naming functions and the atomic artifact writer are pure (explicit paths,
no database), and the pipeline registry is validated against the naming maps —
all unit-testable without a database or a models configuration file.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from pydantic import BaseModel

from diary.core.models_config import TaskType
from diary.db.models import Artifact, ArtifactLayer, PipelineType, WeeklyExport
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.base import Pipeline, PipelineStorageError
from diary.domain.pipelines.naming import (
    JOB_PIPELINE_TYPE,
    LAYER_DIRNAMES,
    PIPELINE_JOB_TYPE,
    PIPELINE_LAYER,
    PIPELINE_PROFILE_NAME,
    PIPELINE_TASK_TYPE,
    artifact_data_root_relative_path,
    derived_filename,
    layer_dirname,
    layer_output_extension,
)
from diary.domain.pipelines.service import _write_artifact_atomically

SUNDAY = date(2026, 9, 13)


class _DummyResult(BaseModel):
    value: str


# ---------------------------------------------------------------------------
# naming conventions
# ---------------------------------------------------------------------------


def test_derived_filename_formats_per_layer() -> None:
    assert (
        derived_filename(
            sunday=SUNDAY, layer=ArtifactLayer.normalized, seq_number=1, revision=1, version=1
        )  # noqa: E501
        == "2026-09-13_normalized_0001_r1_v1.txt"
    )
    assert (
        derived_filename(
            sunday=SUNDAY, layer=ArtifactLayer.summary, seq_number=1, revision=1, version=2
        )  # noqa: E501
        == "2026-09-13_summary_0001_r1_v2.md"
    )
    assert (
        derived_filename(
            sunday=SUNDAY, layer=ArtifactLayer.wellbeing, seq_number=12, revision=2, version=3
        )  # noqa: E501
        == "2026-09-13_wellbeing_0012_r2_v3.md"
    )


def test_layer_directory_and_extension_maps() -> None:
    assert LAYER_DIRNAMES == {
        ArtifactLayer.original: "01_original",
        ArtifactLayer.normalized: "02_normalized",
        ArtifactLayer.summary: "03_summaries",
        ArtifactLayer.wellbeing: "04_wellbeing",
    }
    assert layer_dirname(ArtifactLayer.normalized) == "02_normalized"
    assert layer_output_extension(ArtifactLayer.normalized) == ".txt"
    assert layer_output_extension(ArtifactLayer.summary) == ".md"
    assert layer_output_extension(ArtifactLayer.wellbeing) == ".md"


def test_artifact_path_is_data_root_relative() -> None:
    assert (
        artifact_data_root_relative_path(
            layer=ArtifactLayer.normalized, filename="2026-09-13_normalized_0001_r1_v1.txt"
        )
        == "exports/02_normalized/2026-09-13_normalized_0001_r1_v1.txt"
    )
    assert (
        artifact_data_root_relative_path(
            layer=ArtifactLayer.wellbeing, filename="2026-09-13_wellbeing_0001_r1_v1.md"
        )
        == "exports/04_wellbeing/2026-09-13_wellbeing_0001_r1_v1.md"
    )


def test_type_mappings_are_consistent() -> None:
    assert PIPELINE_LAYER == {
        PipelineType.normalize: ArtifactLayer.normalized,
        PipelineType.summarize: ArtifactLayer.summary,
        PipelineType.wellbeing: ArtifactLayer.wellbeing,
        # Phase-2 layer (task T04, second anchor family).
        PipelineType.stt_corrected: ArtifactLayer.stt_corrected,
        # Phase-2 layer (task T05, second anchor family).
        PipelineType.daily_summary: ArtifactLayer.daily_summary,
        # Phase-2 layer (task T06, second anchor family).
        PipelineType.weekly_summary: ArtifactLayer.weekly_summary,
        # Phase-2 layer (task T07, layer 5, second anchor family).
        PipelineType.coach_note: ArtifactLayer.coach_note,
    }
    assert PIPELINE_TASK_TYPE[PipelineType.normalize] is TaskType.normalization
    assert PIPELINE_TASK_TYPE[PipelineType.summarize] is TaskType.summarization
    assert PIPELINE_TASK_TYPE[PipelineType.wellbeing] is TaskType.wellbeing_analysis
    # The job-type map is the exact inverse of the pipeline-type map.
    assert set(JOB_PIPELINE_TYPE) == set(PIPELINE_JOB_TYPE.values())
    assert {job.value for job in JOB_PIPELINE_TYPE} == {
        "normalize",
        "summarize",
        "wellbeing",
        "stt_corrected",
        "daily_summary",
        "weekly_summary",
        "coach_note",
    }
    assert JOB_PIPELINE_TYPE[PIPELINE_JOB_TYPE[PipelineType.normalize]] is PipelineType.normalize
    assert JOB_PIPELINE_TYPE[PIPELINE_JOB_TYPE[PipelineType.stt_corrected]] is (
        PipelineType.stt_corrected
    )
    assert JOB_PIPELINE_TYPE[PIPELINE_JOB_TYPE[PipelineType.daily_summary]] is (
        PipelineType.daily_summary
    )
    assert JOB_PIPELINE_TYPE[PIPELINE_JOB_TYPE[PipelineType.weekly_summary]] is (
        PipelineType.weekly_summary
    )
    assert JOB_PIPELINE_TYPE[PIPELINE_JOB_TYPE[PipelineType.coach_note]] is (
        PipelineType.coach_note
    )


def test_phase2_layer_maps_keep_the_task_type_map_export_only() -> None:
    """The phase-2 layer is mapped by profile name, not by a TaskType.

    ``TaskType`` has no phase-2 member (extending an MVP enum needs its own
    ADR, task T02), so the phase-2 profile is selected by the configuration
    field name and the layer must not appear in the task-type map.
    """
    assert PipelineType.stt_corrected not in PIPELINE_TASK_TYPE
    assert PIPELINE_PROFILE_NAME[PipelineType.stt_corrected] == "diary_stt"
    assert layer_output_extension(ArtifactLayer.stt_corrected) == ".txt"
    # Layer 5 (task T07) is a phase-2 layer too: profile by name, no TaskType,
    # and its own profile — never the layer-4 wellbeing task type.
    assert PipelineType.coach_note not in PIPELINE_TASK_TYPE
    assert PIPELINE_PROFILE_NAME[PipelineType.coach_note] == "diary_coach"
    assert layer_output_extension(ArtifactLayer.coach_note) == ".md"


# ---------------------------------------------------------------------------
# pipeline registry validation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    snapshot = dict(pipeline_registry.REGISTRY)
    yield
    pipeline_registry.REGISTRY.clear()
    pipeline_registry.REGISTRY.update(snapshot)


class _WellDeclaredPipeline(Pipeline[_DummyResult]):
    pipeline_type = PipelineType.normalize
    layer = ArtifactLayer.normalized
    output_ext = ".txt"
    task_type = TaskType.normalization
    result_schema = _DummyResult

    def build_request(self, *, export: WeeklyExport, input_artifact: Artifact | None) -> LLMRequest:
        raise AssertionError("not exercised in unit tests")

    def parse_result(self, response: LLMResponse) -> _DummyResult:
        raise AssertionError("not exercised in unit tests")

    def render(self, structured: _DummyResult) -> str:
        raise AssertionError("not exercised in unit tests")


def test_register_and_lookup_roundtrip() -> None:
    pipeline = _WellDeclaredPipeline()
    pipeline_registry.register(pipeline)
    assert pipeline_registry.get_pipeline(PipelineType.normalize) is pipeline
    assert pipeline_registry.get_pipeline("normalize") is pipeline
    assert pipeline_registry.get_pipeline(PipelineType.summarize) is None
    assert "normalize" in pipeline_registry.known_pipeline_types()


def test_register_rejects_duplicate_type() -> None:
    pipeline_registry.register(_WellDeclaredPipeline())
    with pytest.raises(ValueError, match="already registered"):
        pipeline_registry.register(_WellDeclaredPipeline())


def test_register_rejects_misdeclared_layer() -> None:
    class _WrongLayerPipeline(_WellDeclaredPipeline):
        layer = ArtifactLayer.summary

    with pytest.raises(ValueError, match="must declare layer"):
        pipeline_registry.register(_WrongLayerPipeline())


def test_register_rejects_misdeclared_task() -> None:
    class _WrongTaskPipeline(_WellDeclaredPipeline):
        task_type = TaskType.summarization

    with pytest.raises(ValueError, match="must declare task type"):
        pipeline_registry.register(_WrongTaskPipeline())


def test_register_rejects_misdeclared_extension() -> None:
    class _WrongExtPipeline(_WellDeclaredPipeline):
        output_ext = ".md"

    with pytest.raises(ValueError, match="output extension"):
        pipeline_registry.register(_WrongExtPipeline())


# ---------------------------------------------------------------------------
# atomic artifact writer
# ---------------------------------------------------------------------------


def test_atomic_writer_writes_and_leaves_no_scratch(tmp_path) -> None:
    target_dir = tmp_path / "exports" / "02_normalized"
    target = target_dir / "2026-09-13_normalized_0001_r1_v1.txt"
    temp_dir = tmp_path / "temp"

    _write_artifact_atomically(b"content\n", target, temp_dir)

    assert target.read_bytes() == b"content\n"
    scratch = [p for p in temp_dir.rglob("*") if p.is_file()]
    assert scratch == []


def test_atomic_writer_failure_raises_storage_error_and_leaves_no_partial(
    tmp_path,
) -> None:
    # Occupying the target directory node makes the writer's mkdir fail.
    target_parent = tmp_path / "02_normalized"
    target_parent.write_text("occupied", encoding="utf-8")
    target = target_parent / "file.txt"
    temp_dir = tmp_path / "temp"

    with pytest.raises(PipelineStorageError):
        _write_artifact_atomically(b"content\n", target, temp_dir)

    assert target_parent.read_text(encoding="utf-8") == "occupied"
    assert not target.exists()
    scratch = [p for p in temp_dir.rglob("*") if p.is_file()] if temp_dir.exists() else []
    assert scratch == []


def test_atomic_writer_overwrites_previous_target(tmp_path) -> None:
    target_dir = tmp_path / "out"
    target_dir.mkdir()
    target = target_dir / f"artifact-{uuid.uuid4().hex}.txt"
    target.write_bytes(b"old")

    _write_artifact_atomically(b"new", target, tmp_path / "temp")

    assert target.read_bytes() == b"new"
