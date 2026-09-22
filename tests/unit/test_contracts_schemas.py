"""Unit tests for the phase-2 contracts and their generated JSON Schemas (T00).

Two properties are checked here:

1. every contract model carries the expected metadata and rejects unknown
   fields (``extra="forbid"``), so an LLM answer with a stray key fails the run
   instead of being silently half-accepted;
2. ``schemas/*.json`` is byte-for-byte what ``scripts/generate_contract_schemas.py``
   renders from the models today — the files are derived artifacts, never hand
   edited (``make contracts`` regenerates them).

Contract validators are exercised directly as well: the coaching-note counts
and disclaimer, the ISO week id, the energy rule and the ``uncertain``
substring rule (the marker of doubt is ``uncertain[]``, not a ``[?]`` literal
inside the text — ADR-007).
"""

from __future__ import annotations

import importlib.util
import uuid
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from diary.core.settings import PROJECT_ROOT
from diary.domain.contracts import (
    COACH_DISCLAIMER,
    CoachNote,
    DailySummary,
    RawEntry,
    SttCorrection,
    WeeklySummary,
)
from diary.domain.contracts._base import ContractModel
from diary.domain.pipelines.wellbeing.safety import DISCLAIMER as WELLBEING_DISCLAIMER

SCHEMAS_DIR = PROJECT_ROOT / "schemas"
GENERATOR_PATH = PROJECT_ROOT / "scripts" / "generate_contract_schemas.py"

CONTRACT_MODELS: tuple[type[ContractModel], ...] = (
    RawEntry,
    SttCorrection,
    DailySummary,
    WeeklySummary,
    CoachNote,
)

_RAW_ID = str(uuid.uuid4())
_ENTRY_ID = str(uuid.uuid4())

VALID_PAYLOADS: dict[str, dict[str, Any]] = {
    "raw_entry": {
        "id": _ENTRY_ID,
        "ts": "2026-09-14T09:12:00+03:00",
        "source": "text",
        "text": "Синтетическое сообщение для теста.",
    },
    "stt_corrected": {
        "raw_id": _RAW_ID,
        "text": "Синтетический текст без пометок.",
        "uncertain": [],
        "model": "local-normalizer",
        "prompt_ver": "v1",
    },
    "daily_summary": {"date": "2026-09-14", "facts": ["Синтетический факт."]},
    "weekly_summary": {
        "week_id": "2026-W38",
        "daily_ids": ["2026-09-14"],
        "rollup": "Синтетический итог недели.",
    },
    "coach_note": {
        "summary_id": "2026-W38",
        "observations": ["Наблюдение 1.", "Наблюдение 2.", "Наблюдение 3."],
        "hypotheses": ["Гипотеза 1.", "Гипотеза 2."],
        # Contract-valid, and deliberately kept so: this fixture proves the
        # *schema*, not the invariants. The hour is not grounded in a weekly
        # rollup that never names it, so the runtime gate refuses this form —
        # measured in T32 on both coaching chunks of the golden set (ADR-012,
        # docs/tasklist-phase2.md §16); see the test that pins it.
        "experiment_7d": "Попробовать 7 дней ложиться до 23:00.",
        "disclaimer": COACH_DISCLAIMER,
    },
}


def _generator() -> Any:
    """Load ``scripts/generate_contract_schemas.py`` as a module."""
    spec = importlib.util.spec_from_file_location("_generate_contract_schemas", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _generator()


def _payload(contract_name: str) -> dict[str, Any]:
    return dict(VALID_PAYLOADS[contract_name])


# ---------------------------------------------------------------------------
# contract metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", CONTRACT_MODELS)
def test_contract_metadata(model: type[ContractModel]) -> None:
    assert issubclass(model, ContractModel)
    assert model.contract_name in VALID_PAYLOADS
    assert model.schema_version == 1
    assert model.model_config.get("extra") == "forbid"


@pytest.mark.parametrize("model", CONTRACT_MODELS)
def test_valid_payload_is_accepted(model: type[ContractModel]) -> None:
    instance = model.model_validate(_payload(model.contract_name))
    assert isinstance(instance, model)


@pytest.mark.parametrize("model", CONTRACT_MODELS)
def test_unknown_field_is_rejected(model: type[ContractModel]) -> None:
    payload = _payload(model.contract_name)
    payload["unexpected_field"] = "не должно приниматься"
    with pytest.raises(ValidationError):
        model.model_validate(payload)


# ---------------------------------------------------------------------------
# generated schemas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", CONTRACT_MODELS)
def test_schema_file_matches_regeneration(model: type[ContractModel], generator: Any) -> None:
    path = SCHEMAS_DIR / f"{model.contract_name}.json"
    assert path.is_file(), f"missing generated schema: {path}"
    expected = generator.render_schema(model).encode("utf-8")
    assert path.read_bytes() == expected


@pytest.mark.parametrize("model", CONTRACT_MODELS)
def test_generated_schema_forbids_additional_properties(model: type[ContractModel]) -> None:
    schema = model.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["title"] == model.__name__


def test_generator_and_contract_set_agree(generator: Any) -> None:
    assert set(generator.CONTRACT_MODELS) == {model.contract_name for model in CONTRACT_MODELS}


# ---------------------------------------------------------------------------
# stt_corrected validators
# ---------------------------------------------------------------------------


def test_uncertain_entries_must_be_verbatim_substrings_of_text() -> None:
    correction = SttCorrection.model_validate(
        {
            "raw_id": _RAW_ID,
            "text": "Купить билеты на 14:00 в музей, как его, на субботу.",
            "uncertain": ["как его"],
            "model": "local-normalizer",
            "prompt_ver": "v2",
        }
    )
    assert correction.uncertain == ["как его"]

    with pytest.raises(ValidationError, match="not a verbatim substring"):
        SttCorrection.model_validate(
            {
                "raw_id": _RAW_ID,
                "text": "Купить билеты на 14:00 в музей.",
                "uncertain": ["как его"],
                "model": "local-normalizer",
                "prompt_ver": "v1",
            }
        )


def test_uncertain_entries_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        SttCorrection.model_validate(
            {
                "raw_id": _RAW_ID,
                "text": "Текст без пометок.",
                "uncertain": [""],
                "model": "local-normalizer",
                "prompt_ver": "v1",
            }
        )


@pytest.mark.parametrize("prompt_ver", ["v1", "v12"])
def test_prompt_version_follows_the_convention(prompt_ver: str) -> None:
    payload = _payload("stt_corrected")
    payload["prompt_ver"] = prompt_ver
    assert SttCorrection.model_validate(payload).prompt_ver == prompt_ver


@pytest.mark.parametrize("prompt_ver", ["1", "v", "v1.0", "V2", ""])
def test_prompt_version_rejects_other_forms(prompt_ver: str) -> None:
    payload = _payload("stt_corrected")
    payload["prompt_ver"] = prompt_ver
    with pytest.raises(ValidationError):
        SttCorrection.model_validate(payload)


def test_no_contract_introduces_a_text_doubt_marker() -> None:
    # ADR-007: doubt lives in uncertain[]; no contract schema defines a `[?]`
    # marker and the corrected text carries the phrase without annotation.
    for model in CONTRACT_MODELS:
        assert "[?]" not in str(model.model_json_schema())
    correction = SttCorrection.model_validate(_payload("stt_corrected"))
    assert "[?]" not in correction.text


# ---------------------------------------------------------------------------
# raw_entry validators
# ---------------------------------------------------------------------------


def test_raw_entry_requires_a_timezone_aware_timestamp() -> None:
    payload = _payload("raw_entry")
    payload["ts"] = "2026-09-14T09:12:00"
    with pytest.raises(ValidationError, match="timezone-aware"):
        RawEntry.model_validate(payload)


def test_raw_entry_voice_entry_requires_a_safe_audio_path() -> None:
    payload = _payload("raw_entry")
    payload.update({"source": "voice", "audio_path": "audio/2026/09/entry.webm"})
    assert RawEntry.model_validate(payload).audio_path == "audio/2026/09/entry.webm"

    for unsafe in ("/etc/passwd", "audio/../../secret.webm", "audio\\2026\\entry.webm", ""):
        with pytest.raises(ValidationError):
            RawEntry.model_validate({**payload, "audio_path": unsafe})

    with pytest.raises(ValidationError, match="must carry its audio_path"):
        RawEntry.model_validate({**payload, "audio_path": None})

    with pytest.raises(ValidationError, match="must not carry an audio_path"):
        RawEntry.model_validate({**_payload("raw_entry"), "audio_path": "audio/2026/09/entry.webm"})


# ---------------------------------------------------------------------------
# daily_summary validators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("energy", ["low", "medium", "high", None])
def test_daily_summary_energy_accepts_levels_and_none(energy: str | None) -> None:
    payload = _payload("daily_summary")
    payload["energy"] = energy
    assert DailySummary.model_validate(payload).energy == energy


@pytest.mark.parametrize("energy", ["HIGH", "very_low", "", 3, "normal"])
def test_daily_summary_rejects_invented_energy_levels(energy: object) -> None:
    payload = _payload("daily_summary")
    payload["energy"] = energy
    with pytest.raises(ValidationError):
        DailySummary.model_validate(payload)


def test_daily_summary_requires_at_least_one_fact() -> None:
    with pytest.raises(ValidationError):
        DailySummary.model_validate({"date": "2026-09-14", "facts": []})


# ---------------------------------------------------------------------------
# weekly_summary validators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("week_id", ["2026-W38", "2020-W53", "2026-W01"])
def test_week_id_accepts_existing_iso_weeks(week_id: str) -> None:
    payload = _payload("weekly_summary")
    payload["week_id"] = week_id
    assert WeeklySummary.model_validate(payload).week_id == week_id


@pytest.mark.parametrize(
    "week_id", ["2026-W54", "2026-W00", "2026-W3", "2026-38", "2026W38", "abc"]
)
def test_week_id_rejects_malformed_or_missing_weeks(week_id: str) -> None:
    payload = _payload("weekly_summary")
    payload["week_id"] = week_id
    with pytest.raises(ValidationError):
        WeeklySummary.model_validate(payload)


def test_daily_ids_must_be_iso_dates_and_unique() -> None:
    payload = _payload("weekly_summary")
    payload["daily_ids"] = ["2026-09-14", "2026-09-15"]
    assert WeeklySummary.model_validate(payload).daily_ids == ["2026-09-14", "2026-09-15"]

    for bad in (["2026-9-14"], ["14.09.2026"], ["2026-09-31"], ["2026-09-14", "2026-09-14"], []):
        with pytest.raises(ValidationError):
            WeeklySummary.model_validate({**payload, "daily_ids": bad})


def test_rollup_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        WeeklySummary.model_validate(
            {"week_id": "2026-W38", "daily_ids": ["2026-09-14"], "rollup": ""}
        )


# ---------------------------------------------------------------------------
# coach_note validators
# ---------------------------------------------------------------------------


def test_coach_note_requires_exactly_three_observations_and_two_hypotheses() -> None:
    payload = _payload("coach_note")
    payload["observations"] = payload["observations"][:2]
    with pytest.raises(ValidationError):
        CoachNote.model_validate(payload)

    payload = _payload("coach_note")
    payload["hypotheses"] = [*payload["hypotheses"], "Лишняя гипотеза."]
    with pytest.raises(ValidationError):
        CoachNote.model_validate(payload)

    payload = _payload("coach_note")
    payload["observations"] = ["", "Наблюдение 2.", "Наблюдение 3."]
    with pytest.raises(ValidationError):
        CoachNote.model_validate(payload)


def test_coach_note_disclaimer_must_equal_the_constant() -> None:
    payload = _payload("coach_note")
    payload["disclaimer"] = "Краткая версия оговорки."
    with pytest.raises(ValidationError, match="COACH_DISCLAIMER"):
        CoachNote.model_validate(payload)


def test_coach_disclaimer_is_a_separate_constant_from_layer_four() -> None:
    # ADR-007: layer 4 keeps its own disclaimer; layer 5 must not reuse it.
    assert COACH_DISCLAIMER != WELLBEING_DISCLAIMER
    assert "диагноз" in COACH_DISCLAIMER


def test_coach_note_experiment_must_state_the_seven_day_horizon() -> None:
    payload = _payload("coach_note")
    payload["experiment_7d"] = "Попробовать ложиться раньше."
    with pytest.raises(ValidationError, match="horizon"):
        CoachNote.model_validate(payload)


def test_coach_note_requires_a_summary_id() -> None:
    with pytest.raises(ValidationError):
        CoachNote.model_validate({**_payload("coach_note"), "summary_id": ""})


def test_contract_model_is_a_pydantic_base_model() -> None:
    assert issubclass(ContractModel, BaseModel)
