"""Unit tests for the phase-2 result store (task T08).

Covers the parts of ``diary.db.results`` that need no database:

* the layer ↔ anchor mapping and its schema expression;
* the 1:1 mapping of the four phase-2 layer values onto the pipeline/job values
  added by migration ``0003``;
* anchor validation (wrong anchor, ambiguous anchor, timezone pairing, ISO week
  shape);
* result payload and rendered-file validation.

The database-backed behaviour lives in
``tests/integration/test_layer_results.py``.
"""

from __future__ import annotations

import importlib.util
import uuid
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel

from diary.core.settings import get_settings
from diary.db.models import (
    ANCHOR_FAMILY_CHECK,
    ANCHOR_LOCAL_TZ_CHECK,
    ANCHOR_PAYLOAD_CHECK,
    EXPORT_FAMILY_LAYERS,
    SECOND_FAMILY_ANCHORS,
    SECOND_FAMILY_LAYER_ANCHORS,
    ArtifactLayer,
    JobType,
    PipelineType,
)
from diary.db.results import (
    AnchorError,
    ResultPayloadError,
    UnsupportedLayerError,
    _normalize_file,
    _resolve_anchor,
    anchor_field_for,
    normalize_result_payload,
)

#: Repository root: tests/unit/test_layer_results.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_0003 = REPO_ROOT / "alembic" / "versions" / "0003_phase2_anchor_family.py"

#: The four layers of the second anchor family (ADR-007 §1).
PHASE2_LAYERS = (
    ArtifactLayer.stt_corrected,
    ArtifactLayer.daily_summary,
    ArtifactLayer.weekly_summary,
    ArtifactLayer.coach_note,
)


def _load_migration():
    """Import the frozen migration module (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("migration_0003", MIGRATION_0003)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalized(sql: str) -> str:
    return " ".join(sql.split())


# ---------------------------------------------------------------------------
# Layer/anchor/pipeline mapping
# ---------------------------------------------------------------------------


def test_phase2_layers_are_anchored_to_their_own_source() -> None:
    assert SECOND_FAMILY_LAYER_ANCHORS == {
        ArtifactLayer.stt_corrected: "raw_id",
        ArtifactLayer.daily_summary: "local_date",
        ArtifactLayer.weekly_summary: "week_id",
        ArtifactLayer.coach_note: "summary_id",
    }
    for layer in PHASE2_LAYERS:
        assert anchor_field_for(layer) == SECOND_FAMILY_LAYER_ANCHORS[layer]


def test_first_family_layers_are_rejected_by_the_result_store() -> None:
    for layer in EXPORT_FAMILY_LAYERS:
        with pytest.raises(UnsupportedLayerError):
            anchor_field_for(layer)


def test_layer_anchor_map_and_check_cover_every_layer() -> None:
    """The schema CHECK and the Python mapping must not drift apart."""
    assert set(ArtifactLayer) == set(EXPORT_FAMILY_LAYERS) | set(SECOND_FAMILY_LAYER_ANCHORS)
    for layer in EXPORT_FAMILY_LAYERS:
        assert f"'{layer.value}'" in ANCHOR_FAMILY_CHECK
    for layer, anchor in SECOND_FAMILY_LAYER_ANCHORS.items():
        assert f"'{layer.value}'" in ANCHOR_FAMILY_CHECK
        assert anchor in ANCHOR_FAMILY_CHECK
    for anchor in SECOND_FAMILY_ANCHORS:
        assert anchor in ANCHOR_FAMILY_CHECK
    assert "version IS NOT NULL" in ANCHOR_FAMILY_CHECK


def test_phase2_layer_pipeline_and_job_values_are_identical() -> None:
    """A phase-2 result names its layer, its pipeline and its job the same way."""
    for layer in PHASE2_LAYERS:
        assert layer.value in {member.value for member in PipelineType}
        assert layer.value in {member.value for member in JobType}
    assert len(list(PipelineType)) == len({member.value for member in PipelineType})
    assert len(list(JobType)) == len({member.value for member in JobType})


def test_first_family_enum_values_are_unchanged() -> None:
    """Migration 0003 only widens the enums; MVP values keep their meaning."""
    assert [member.value for member in ArtifactLayer][:4] == [
        "original",
        "normalized",
        "summary",
        "wellbeing",
    ]
    assert [member.value for member in PipelineType][:3] == ["normalize", "summarize", "wellbeing"]
    assert [member.value for member in JobType][:4] == [
        "transcribe_audio",
        "normalize",
        "summarize",
        "wellbeing",
    ]


def test_migration_0003_freezes_the_same_check_sql_as_the_model() -> None:
    """The migration and the model must describe the same anchor invariant."""
    migration = _load_migration()
    assert _normalized(migration._ANCHOR_FAMILY_CHECK) == _normalized(ANCHOR_FAMILY_CHECK)
    assert _normalized(migration._ANCHOR_LOCAL_TZ_CHECK) == _normalized(ANCHOR_LOCAL_TZ_CHECK)
    assert _normalized(migration._ANCHOR_PAYLOAD_CHECK) == _normalized(ANCHOR_PAYLOAD_CHECK)
    for layer in PHASE2_LAYERS:
        assert layer.value in migration._ARTIFACT_LAYER_CHECK
        assert layer.value in migration._PIPELINE_TYPE_CHECK
        assert layer.value in migration._JOB_TYPE_CHECK


# ---------------------------------------------------------------------------
# Anchor validation
# ---------------------------------------------------------------------------


def test_anchor_requires_the_anchor_of_its_layer() -> None:
    with pytest.raises(AnchorError, match="requires the 'raw_id' anchor"):
        _resolve_anchor(
            ArtifactLayer.stt_corrected,
            raw_id=None,
            local_date=None,
            local_tz=None,
            week_id=None,
            summary_id=None,
        )


def test_anchor_rejects_a_foreign_anchor() -> None:
    with pytest.raises(AnchorError, match="must not be set"):
        _resolve_anchor(
            ArtifactLayer.daily_summary,
            raw_id=uuid.uuid4(),
            local_date=None,
            local_tz=None,
            week_id=None,
            summary_id=None,
        )


def test_daily_anchor_keeps_date_and_timezone_together() -> None:
    resolved = _resolve_anchor(
        ArtifactLayer.daily_summary,
        raw_id=None,
        local_date=date(2026, 9, 6),
        local_tz=None,
        week_id=None,
        summary_id=None,
    )
    assert resolved == {
        "local_date": date(2026, 9, 6),
        "local_tz": get_settings().timezone,
    }
    explicit = _resolve_anchor(
        ArtifactLayer.daily_summary,
        raw_id=None,
        local_date=date(2026, 9, 6),
        local_tz="UTC",
        week_id=None,
        summary_id=None,
    )
    assert explicit["local_tz"] == "UTC"


def test_week_id_must_be_an_iso_week() -> None:
    for bad in ("2026-54", "2026W38", "2026-w38", "2026-W3", ""):
        with pytest.raises(AnchorError, match="YYYY-Www"):
            _resolve_anchor(
                ArtifactLayer.weekly_summary,
                raw_id=None,
                local_date=None,
                local_tz=None,
                week_id=bad,
                summary_id=None,
            )
    assert _resolve_anchor(
        ArtifactLayer.weekly_summary,
        raw_id=None,
        local_date=None,
        local_tz=None,
        week_id="2026-W38",
        summary_id=None,
    ) == {"week_id": "2026-W38"}


def test_timezone_only_belongs_to_the_day_anchor() -> None:
    with pytest.raises(AnchorError, match="daily_summary"):
        _resolve_anchor(
            ArtifactLayer.weekly_summary,
            raw_id=None,
            local_date=None,
            local_tz="Europe/Moscow",
            week_id="2026-W38",
            summary_id=None,
        )


# ---------------------------------------------------------------------------
# Result payload / rendered file
# ---------------------------------------------------------------------------


class _SyntheticResult(BaseModel):
    """Stand-in for a validated phase-2 contract in the payload tests."""

    text: str
    note: str | None = None


def test_payload_of_a_contract_model_is_serialized_deterministically() -> None:
    payload = normalize_result_payload(_SyntheticResult(text="Синтетическая правка"))
    assert payload == {"text": "Синтетическая правка", "note": None}


def test_payload_of_a_mapping_is_copied() -> None:
    payload = normalize_result_payload({"week_id": "2026-W38", "rollup": "Синтетика"})
    assert payload == {"week_id": "2026-W38", "rollup": "Синтетика"}


def test_payload_none_stays_none() -> None:
    assert normalize_result_payload(None) is None


def test_payload_rejects_a_non_serializable_value() -> None:
    with pytest.raises(ResultPayloadError, match="not JSON-serializable"):
        normalize_result_payload({"raw_id": uuid.uuid4()})


def test_payload_rejects_an_unsupported_type() -> None:
    with pytest.raises(ResultPayloadError, match="Pydantic contract or a mapping"):
        normalize_result_payload(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_rendered_file_requires_path_and_digest_together() -> None:
    with pytest.raises(ResultPayloadError, match="without a rendered file path"):
        _normalize_file(None, "a" * 64)
    with pytest.raises(ResultPayloadError, match="SHA-256"):
        _normalize_file("exports/05_coach_note/x.md", None)
    with pytest.raises(ResultPayloadError, match="SHA-256"):
        _normalize_file("exports/05_coach_note/x.md", "not-a-digest")
    assert _normalize_file("exports/05_coach_note/x.md", "a" * 64) == (
        "exports/05_coach_note/x.md",
        "a" * 64,
    )


def test_rendered_file_path_must_stay_inside_the_data_root() -> None:
    for bad in ("/etc/passwd", "..\\x", "exports/../secret.txt", " exports/x.md", ""):
        with pytest.raises(ResultPayloadError):
            _normalize_file(bad, "a" * 64)
