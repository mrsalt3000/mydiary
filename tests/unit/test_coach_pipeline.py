"""Unit tests for the T07 coaching-note layer (no provider, no database).

Covers the definition of done of task T07 (``docs/tasklist-phase2.md`` §6):

* the prompt template demands exactly 3 observations, exactly 2 hypotheses and
  one experiment that states the 7-day horizon; it forbids asserted causes,
  diagnoses, treatment, advice, morality and new facts, and it carries only
  synthetic examples;
* a valid completion becomes a validated :class:`CoachNote` and is handed to
  the append-only store exactly once, under the storage anchor
  ``summary_id`` = the **artifact id** of the upstream weekly summary, while
  the contract field ``summary_id`` carries that summary's ``week_id`` (the
  twin-identity seam of T00 / ADR-008 §5);
* both ``summary_id`` and ``disclaimer`` are envelope fields of the backend: a
  completion that reports a foreign week id or its own disclaimer wording
  cannot change the stored note, and the note always carries the layer-5
  ``COACH_DISCLAIMER`` — never the layer-4 ``wellbeing`` one;
* an invalid completion (non-JSON, extra field, missing field, 2 or 4
  observations, 1 or 3 hypotheses, an experiment without the digit 7) raises a
  loud, non-retryable LLM error and stores **nothing** — no partial artifact;
* a week without a weekly summary, or a supplied input of another week, fails
  loudly before any request is sent;
* the input weekly summary is not modified, and the layer has no other write
  path;
* **launch is explicit only**: importing the package registers nothing, no job
  handler exists for the layer, the worker startup registers the layer (since
  T09, so the orchestrator can resolve it) but never calls it, and no API route
  calls it;
* **layer 4 is untouched**: the coach modules do not import the wellbeing
  package, the two disclaimer constants stay separate and the layer-5 layer /
  pipeline values are distinct from the layer-4 ones.

The provider is a fake and the store is a recording double, so no endpoint,
configuration file, model or database is touched. All content is synthetic
(AGENTS.md §13).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

import pytest

from diary.core.settings import PROJECT_ROOT
from diary.db.models import Artifact, ArtifactLayer, JobType, PipelineType
from diary.domain.contracts import COACH_DISCLAIMER, CoachNote, WeeklySummary
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines import naming
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.coach import service as coach_service
from diary.domain.pipelines.coach.coach_input import (
    CoachInput,
    CoachInputError,
    ensure_input_matches_week,
)
from diary.domain.pipelines.coach.pipeline import (
    PROMPT_PATH,
    PROMPT_VERSION,
    SUMMARY_HEADER,
    CoachNotePipeline,
)
from diary.domain.pipelines.coach.registration import ensure_coach_pipeline_registered
from diary.domain.pipelines.wellbeing import safety as wellbeing_safety
from diary.workers import handlers

WEEK_ID = "2026-W38"
DAILY_IDS = ["2026-09-14", "2026-09-15", "2026-09-16"]
ROLLUP = "Неделя прошла вокруг отчёта; в среду сон сбился, причина осталась открытой."
OBSERVATIONS = [
    "Работа над отчётом упоминается в начале и середине недели.",
    "В сводке есть один день спорта.",
    "Открытым остаётся вопрос о сне в среду.",
]
HYPOTHESES = [
    "Возможно, отчёт был основной нагрузкой первой половины недели.",
    "Похоже, что темп недели менялся к её середине.",
]
EXPERIMENT = "В течение следующих 7 дней можно попробовать отмечать, что занимало основное время."


def _summary(week_id: str = WEEK_ID) -> WeeklySummary:
    return WeeklySummary(week_id=week_id, daily_ids=DAILY_IDS, rollup=ROLLUP)


def _artifact(
    *,
    summary: WeeklySummary | None = None,
    artifact_id: uuid.UUID | None = None,
) -> Artifact:
    """A detached ``weekly_summary`` artifact row (the layer only reads it)."""
    value = summary if summary is not None else _summary()
    return Artifact(
        id=artifact_id if artifact_id is not None else uuid.uuid4(),
        layer=ArtifactLayer.weekly_summary,
        week_id=value.week_id,
        version=1,
        result_json=value.model_dump(mode="json"),
    )


def _coach_input(*, week_id: str = WEEK_ID) -> CoachInput:
    summary = _summary(week_id)
    return CoachInput(summary=summary, artifact=_artifact(summary=summary))


def _note_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "observations": list(OBSERVATIONS),
        "hypotheses": list(HYPOTHESES),
        "experiment_7d": EXPERIMENT,
    }
    payload.update(overrides)
    return payload


def _json_response(payload: object) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload, ensure_ascii=False), model="чужой-модель")


class _Provider(Protocol):
    """Minimal provider shape the layer uses (a fake records the requests)."""

    requests: list[LLMRequest]

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class _FakeProvider:
    """Deterministic provider: a fixed response, recorded requests."""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response

    @property
    def request_content(self) -> str:
        assert len(self.requests) == 1
        return self.requests[0].messages[0].content


class _RecordingStore:
    """Records the arguments of the append-only store call (no database)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, session: object, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return type("StoredArtifact", (), {"id": uuid.uuid4(), "version": 1})()


@pytest.fixture(autouse=True)
def _restore_registries() -> None:
    snapshot = dict(pipeline_registry.SECOND_FAMILY_REGISTRY)
    handler_snapshot = dict(handlers.REGISTRY)
    yield
    pipeline_registry.SECOND_FAMILY_REGISTRY.clear()
    pipeline_registry.SECOND_FAMILY_REGISTRY.update(snapshot)
    handlers.REGISTRY.clear()
    handlers.REGISTRY.update(handler_snapshot)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _RecordingStore:
    recording = _RecordingStore()
    monkeypatch.setattr(coach_service, "save_layer_result", recording)
    return recording


def _run(
    store: _RecordingStore,
    provider: _Provider,
    *,
    week_id: str = WEEK_ID,
    coach_input: CoachInput | None = None,
) -> coach_service.CoachNoteOutcome:
    return coach_service.coach_note(
        None,  # type: ignore[arg-type]  # the store double ignores the session
        week_id=week_id,
        provider=provider,
        coach_input=coach_input if coach_input is not None else _coach_input(week_id=week_id),
    )


# ---------------------------------------------------------------------------
# prompt template
# ---------------------------------------------------------------------------


def test_prompt_template_is_v2_and_demands_3_2_1() -> None:
    assert PROMPT_VERSION == "v2"
    assert PROMPT_PATH.is_file()
    text = PROMPT_PATH.read_text(encoding="utf-8")
    # The three answer fields and their counts are explicit.
    assert "observations" in text and "hypotheses" in text and "experiment_7d" in text
    assert "ровно из 3" in text
    assert "ровно из 2" in text
    assert "7 дней" in text
    assert "JSON" in text
    # The one-object answer shape must be stated.
    assert '{"observations"' in text.replace(" ", "").replace("\n", "")
    # The fields the backend owns are named as system-owned, not asked of the model.
    assert "summary_id" in text and "disclaimer" in text


def test_prompt_template_forbids_diagnoses_causes_advice_and_morality() -> None:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    haystack = text.lower()
    for lexeme in ("диагноз", "лечение", "совет", "морал", "симптомы"):
        assert lexeme in haystack
    # Hypotheses must be phrased as assumptions, and asserted causes are vetoed.
    assert "Возможно" in text and "Похоже" in text
    assert "потому что" in text  # named in the prohibition itself
    assert "не установленная причина" in " ".join(text.split())


def test_prompt_template_carries_only_synthetic_examples() -> None:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "Синтетический пример" in text
    assert "2026-W38" in text  # the authored example week id


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------


def test_layer_is_registered_in_the_second_family_registry_on_request() -> None:
    assert ensure_coach_pipeline_registered() is True
    assert ensure_coach_pipeline_registered() is False  # idempotent
    registered = pipeline_registry.get_second_family_pipeline(PipelineType.coach_note)
    assert isinstance(registered, CoachNotePipeline)
    assert registered.layer is ArtifactLayer.coach_note
    assert registered.pipeline_type is PipelineType.coach_note
    assert registered.profile_name == "diary_coach"
    assert "coach_note" in pipeline_registry.known_pipeline_types()
    # The export-driven lookup must not return this layer.
    assert pipeline_registry.get_pipeline(PipelineType.coach_note) is None


def test_registry_rejects_a_misdeclared_layer() -> None:
    class _WrongLayer(CoachNotePipeline):
        layer = ArtifactLayer.wellbeing

    with pytest.raises(ValueError, match="must declare layer"):
        pipeline_registry.register(_WrongLayer())


def test_registry_rejects_a_misdeclared_profile_name() -> None:
    class _WrongProfile(CoachNotePipeline):
        profile_name = "diary_sum"

    with pytest.raises(ValueError, match="must declare profile name"):
        pipeline_registry.register(_WrongProfile())


def test_the_layer_is_mapped_by_profile_name_and_has_no_task_type() -> None:
    """``coach_note`` is a second-family layer, not a weekly-export pipeline."""
    assert PipelineType.coach_note not in naming.PIPELINE_TASK_TYPE
    assert naming.PIPELINE_LAYER[PipelineType.coach_note] is ArtifactLayer.coach_note
    assert naming.PIPELINE_PROFILE_NAME[PipelineType.coach_note] == "diary_coach"
    assert naming.PIPELINE_JOB_TYPE[PipelineType.coach_note] is JobType.coach_note
    assert naming.layer_output_extension(ArtifactLayer.coach_note) == ".md"
    # The layer renders no file, so it has no per-layer artifact directory.
    with pytest.raises(ValueError):
        naming.layer_dirname(ArtifactLayer.coach_note)


# ---------------------------------------------------------------------------
# request building
# ---------------------------------------------------------------------------


def test_request_carries_only_the_weekly_summary() -> None:
    summary = _summary()
    request = CoachNotePipeline().build_request(summary=summary)

    assert request.prompt_version == PROMPT_VERSION
    content = request.messages[0].content
    assert SUMMARY_HEADER in content
    assert WEEK_ID in content
    assert ROLLUP in content
    assert json.dumps(summary.model_dump(mode="json"), ensure_ascii=False) in content
    # The input is the week rollup, not raw entries, the day layer or a file.
    assert "raw_entry" not in content
    assert "transcription" not in content
    assert "daily_summary" not in content
    assert "audio_path" not in content


# ---------------------------------------------------------------------------
# successful run
# ---------------------------------------------------------------------------


def test_valid_response_yields_a_note_and_is_stored_once(store: _RecordingStore) -> None:
    provider = _FakeProvider(_json_response(_note_payload()))

    outcome = _run(store, provider)

    assert outcome.note == CoachNote(
        summary_id=WEEK_ID,
        observations=OBSERVATIONS,
        hypotheses=HYPOTHESES,
        experiment_7d=EXPERIMENT,
        disclaimer=COACH_DISCLAIMER,
    )
    assert len(provider.requests) == 1
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["layer"] is ArtifactLayer.coach_note
    assert call["result"] is outcome.note
    # No rendered file: the layer's result is stored inline (ADR-008 §1).
    assert call.get("path") is None
    # The upstream-artifact anchor is the only anchor the layer passes.
    assert isinstance(call["summary_id"], uuid.UUID)
    assert "raw_id" not in call and "local_date" not in call and "week_id" not in call


def test_the_note_has_exactly_3_observations_2_hypotheses_and_a_7_day_experiment(
    store: _RecordingStore,
) -> None:
    outcome = _run(store, _FakeProvider(_json_response(_note_payload())))

    assert len(outcome.note.observations) == 3
    assert len(outcome.note.hypotheses) == 2
    assert "7" in outcome.note.experiment_7d
    assert outcome.note.disclaimer == COACH_DISCLAIMER


def test_the_contract_summary_id_and_the_storage_anchor_are_two_different_values(
    store: _RecordingStore,
) -> None:
    """Twin ``summary_id``: contract field = week id, storage anchor = artifact id."""
    coach_input = _coach_input()
    artifact_id = coach_input.artifact.id

    outcome = _run(store, _FakeProvider(_json_response(_note_payload())), coach_input=coach_input)

    # Storage anchor: the artifact id of the input weekly summary (a UUID).
    assert store.calls[0]["summary_id"] == artifact_id
    assert isinstance(store.calls[0]["summary_id"], uuid.UUID)
    # Contract field: the week id of that same summary (a string).
    assert outcome.note.summary_id == WEEK_ID
    assert isinstance(outcome.note.summary_id, str)
    # The two are different values in different namespaces — by design.
    assert outcome.note.summary_id != str(artifact_id)


def test_the_envelope_fields_come_from_the_backend_not_the_answer(
    store: _RecordingStore,
) -> None:
    """A completion claiming another week or its own disclaimer cannot change the record."""
    payload = _note_payload(summary_id="1999-W01", disclaimer="Это мой собственный текст.")

    outcome = _run(store, _FakeProvider(_json_response(payload)))

    assert outcome.note.summary_id == WEEK_ID
    assert outcome.note.disclaimer == COACH_DISCLAIMER
    stored = store.calls[0]["result"]
    assert stored.summary_id == WEEK_ID
    assert stored.disclaimer == COACH_DISCLAIMER


def test_the_backend_supplies_the_disclaimer_when_the_model_omits_it(
    store: _RecordingStore,
) -> None:
    outcome = _run(store, _FakeProvider(_json_response(_note_payload())))

    assert outcome.note.disclaimer == COACH_DISCLAIMER
    assert store.calls[0]["result"].disclaimer == COACH_DISCLAIMER


def test_the_storage_anchor_is_not_influenced_by_the_model(store: _RecordingStore) -> None:
    """The model cannot move the note to another upstream summary."""
    coach_input = _coach_input()
    artifact_id = coach_input.artifact.id
    payload = _note_payload(summary_id=str(uuid.uuid4()))

    _run(store, _FakeProvider(_json_response(payload)), coach_input=coach_input)

    assert store.calls[0]["summary_id"] == artifact_id
    assert store.calls[0]["summary_id"] != payload["summary_id"]


# ---------------------------------------------------------------------------
# failed runs: loud, and without any stored result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_text",
    [
        "Не JSON вовсе.",
        '```json\n{"observations": ["a"], "hypotheses": ["b"], "experiment_7d": "7 дней"}\n```',
        '[{"observations": ["a"]}]',
    ],
)
def test_invalid_json_fails_loudly_and_stores_nothing(
    store: _RecordingStore, raw_text: str
) -> None:
    with pytest.raises(LLMStructuredOutputError):
        _run(store, _FakeProvider(LLMResponse(text=raw_text)))

    assert store.calls == []


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("extra advice field", _note_payload(advice="Отдыхай больше.")),
        ("missing observations", {"hypotheses": HYPOTHESES, "experiment_7d": EXPERIMENT}),
        ("missing hypotheses", {"observations": OBSERVATIONS, "experiment_7d": EXPERIMENT}),
        ("missing experiment_7d", {"observations": OBSERVATIONS, "hypotheses": HYPOTHESES}),
        ("two observations", _note_payload(observations=OBSERVATIONS[:2])),
        ("four observations", _note_payload(observations=[*OBSERVATIONS, "Четвёртое."])),
        ("one hypothesis", _note_payload(hypotheses=HYPOTHESES[:1])),
        ("three hypotheses", _note_payload(hypotheses=[*HYPOTHESES, "Третья."])),
        ("experiment without the digit 7", _note_payload(experiment_7d="Попробовать на неделе.")),
        ("empty observation", _note_payload(observations=["", OBSERVATIONS[1], OBSERVATIONS[2]])),
    ],
)
def test_invalid_contract_fails_loudly_and_stores_nothing(
    store: _RecordingStore, label: str, payload: dict[str, object]
) -> None:
    with pytest.raises(LLMStructuredOutputError):
        _run(store, _FakeProvider(_json_response(payload)))

    assert store.calls == []


def test_a_week_without_a_weekly_summary_fails_loudly_before_the_provider(
    store: _RecordingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_summary(session: object, *, week_id: str) -> CoachInput:
        raise CoachInputError("The ISO week has no weekly summary.")

    monkeypatch.setattr(coach_service, "collect_week_summary", _no_summary)
    provider = _FakeProvider(_json_response(_note_payload()))

    with pytest.raises(CoachInputError):
        coach_service.coach_note(
            None,  # type: ignore[arg-type]
            week_id=WEEK_ID,
            provider=provider,
        )

    assert provider.requests == []
    assert store.calls == []


def test_a_supplied_input_of_another_week_fails_loudly_before_the_provider(
    store: _RecordingStore,
) -> None:
    provider = _FakeProvider(_json_response(_note_payload()))

    with pytest.raises(CoachInputError):
        _run(store, provider, week_id=WEEK_ID, coach_input=_coach_input(week_id="2026-W39"))

    assert provider.requests == []
    assert store.calls == []


def test_the_membership_guard_rejects_a_malformed_week_id() -> None:
    with pytest.raises(CoachInputError):
        ensure_input_matches_week(_coach_input(), "2026-W54")


# ---------------------------------------------------------------------------
# the input is not modified
# ---------------------------------------------------------------------------


def test_the_run_does_not_change_the_weekly_summary(store: _RecordingStore) -> None:
    coach_input = _coach_input()
    before = coach_input.summary.model_dump()

    _run(store, _FakeProvider(_json_response(_note_payload())), coach_input=coach_input)

    assert coach_input.summary.model_dump() == before
    # The only write performed by the layer is the append-only result store.
    assert [call["layer"] for call in store.calls] == [ArtifactLayer.coach_note]


def test_created_at_and_run_are_forwarded_to_the_store(store: _RecordingStore) -> None:
    created_at = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    run_id = uuid.uuid4()

    coach_service.coach_note(
        None,  # type: ignore[arg-type]
        week_id=WEEK_ID,
        provider=_FakeProvider(_json_response(_note_payload())),
        coach_input=_coach_input(),
        processing_run_id=run_id,
        created_at=created_at,
    )

    assert store.calls[0]["processing_run_id"] == run_id
    assert store.calls[0]["created_at"] == created_at


# ---------------------------------------------------------------------------
# launch is explicit only
# ---------------------------------------------------------------------------


def test_no_job_handler_runs_the_layer() -> None:
    """Without a handler an enqueued ``coach_note`` job fails instead of running."""
    assert handlers.get_handler(JobType.coach_note) is None
    assert JobType.coach_note.value not in handlers.known_job_types()


def test_importing_the_package_registers_nothing_and_pulls_in_no_wellbeing() -> None:
    """A clean interpreter: import the layer, touch no registry, load no layer 4."""
    code = """
import json, sys
import diary.domain.pipelines.coach  # noqa: F401
from diary.db.models import JobType, PipelineType
from diary.domain.pipelines import registry
from diary.workers import handlers
print(json.dumps({
    "coach_registered": registry.get_second_family_pipeline(PipelineType.coach_note) is not None,
    "coach_exports": registry.get_pipeline(PipelineType.coach_note) is not None,
    "coach_handler": handlers.get_handler(JobType.coach_note) is not None,
    "wellbeing_loaded": any(
        name.startswith("diary.domain.pipelines.wellbeing") for name in sys.modules
    ),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    assert facts == {
        "coach_registered": False,
        "coach_exports": False,
        "coach_handler": False,
        "wellbeing_loaded": False,
    }


def test_the_worker_startup_registers_but_never_launches_the_layer() -> None:
    """The queue must never start the layer; T09/T11 own the explicit wiring.

    Since T09 the worker **registers** the layer (the orchestrator resolves
    layers through the second-family registry, so an unregistered layer could
    not be launched explicitly in the worker process either) — but registering
    is not launching: with no handler for ``JobType.coach_note`` an enqueued job
    still fails permanently, and no worker code calls the layer service. This is
    the narrowed, still-structural form of the T07 rule «запуск только явный».
    """
    runner_source = (PROJECT_ROOT / "src" / "diary" / "workers" / "runner.py").read_text(
        encoding="utf-8"
    )
    # Registered at startup, so the explicit launch can resolve the layer here.
    assert "ensure_coach_pipeline_registered" in runner_source
    # Never launched: no call of the layer or of its service in the worker.
    assert "run_coach_note(" not in runner_source
    assert "coach_note(" not in runner_source
    assert handlers.get_handler(JobType.coach_note) is None


def test_no_api_route_runs_the_layer() -> None:
    """No route may launch the layer: the explicit call is the only entry point."""
    api_root = PROJECT_ROOT / "src" / "diary" / "api"
    sources = {path: path.read_text(encoding="utf-8") for path in api_root.rglob("*.py")}
    assert sources  # the scan is not vacuous
    for path, source in sources.items():
        assert "run_coach_note" not in source, path
        assert "coach_note(" not in source, path
        assert "ensure_coach_pipeline_registered" not in source, path


# ---------------------------------------------------------------------------
# layer 4 is untouched
# ---------------------------------------------------------------------------


def test_the_two_disclaimers_are_separate_constants() -> None:
    assert COACH_DISCLAIMER
    assert wellbeing_safety.DISCLAIMER
    assert COACH_DISCLAIMER != wellbeing_safety.DISCLAIMER
    # The layer-4 limitation still names the diary-only, non-medical position.
    assert wellbeing_safety.DISCLAIMER.startswith("Рефлексия основана только на записях дневника")


def test_the_coach_layer_has_no_layer_4_import_or_value() -> None:
    """Import graph and enum values: layer 5 never borrows layer 4."""
    package_dir = PROJECT_ROOT / "src" / "diary" / "domain" / "pipelines" / "coach"
    modules = sorted(package_dir.glob("*.py"))
    assert modules
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("diary.domain.pipelines.wellbeing"), module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("diary.domain.pipelines.wellbeing"), module
    # Distinct enum members and unchanged layer-4 safety values.
    assert PipelineType.coach_note is not PipelineType.wellbeing
    assert ArtifactLayer.coach_note is not ArtifactLayer.wellbeing
    assert naming.PIPELINE_LAYER[PipelineType.wellbeing] is ArtifactLayer.wellbeing
    assert wellbeing_safety.SafetyLevel.urgent.value == "urgent"
    # The layer-4 crisis classifier is still the deterministic code rule.
    assert wellbeing_safety.classify("") is wellbeing_safety.SafetyLevel.normal


def test_the_layer_4_pipeline_is_not_replaced_by_the_coach_layer() -> None:
    """The two layers keep their own numbers on the UI label maps."""
    from diary.api.routes import pages

    assert pages.LAYER_TITLES[ArtifactLayer.wellbeing] == "Wellbeing-рефлексия (слой 4)"
    assert pages.LAYER_TITLES[ArtifactLayer.coach_note] == "Заметка коуча (фаза 2, слой 5)"
    assert pages.PIPELINE_LABELS[PipelineType.wellbeing] == "Wellbeing-рефлексия"
    assert pages.PIPELINE_LABELS[PipelineType.coach_note] == "Заметка коуча"
    assert pages.JOB_TYPE_LABELS[JobType.coach_note] == "Заметка коуча"


def test_prompt_and_layer_files_live_under_the_t07_locations() -> None:
    assert PROMPT_PATH.parent.name == "coach_note"
    assert PROMPT_PATH.name == "v2.md"
    assert CoachNotePipeline.output_ext == ".md"
