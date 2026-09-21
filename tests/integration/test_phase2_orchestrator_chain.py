"""Integration tests for the T09 orchestrator (isolated migration-created DB).

The orchestrator (``diary.domain.pipelines.pipeline``) drives the second anchor
family chain over real persistence — an isolated database created solely by the
Alembic migrations, never the production one — with fake per-layer providers
(one fake per phase-2 profile) and an in-memory models configuration. It covers
the parts of the T09 definition of done that only a database can show:

* the day chain is ``stt_fix`` (one run per message) then ``sum_daily``, and the
  week chain merges the **stored** day summaries (never raw text);
* every step has its own ``ProcessingRun``, the layers receive
  ``processing_run_id`` and the stored artifact points back at exactly that run;
* the runs carry the provenance the T05 acceptance asked for — ``profile_name``,
  ``model_name``, ``prompt_version`` — plus content-free usage metadata;
* the raw source rows (``messages``/``audio_files``/``transcriptions``) are
  unchanged, value by value, by any run of the chain;
* a rerun appends ``v+1`` for every step and leaves the previous versions and
  their runs intact (append-only, ADR-005);
* a failing step records a failed run, stores no partial artifact and stops the
  chain, while the steps that already succeeded stay committed;
* the coaching note (layer 5) is produced only by the explicit call and its
  anchor is the upstream weekly-summary artifact;
* runs of the second family never appear in the export-scoped views
  (``list_by_input_export`` / ``_export_runs_view``), which require a rendered
  file (ADR-008 §7);
* a layer without a profile is switched off (skip), and an empty input is
  skipped rather than invented.

All content is synthetic (AGENTS.md §13); no endpoint, model or real
configuration file is used.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from diary.api.routes import pages
from diary.core.models_config import (
    FasterWhisperProfile,
    ModelsConfig,
    OpenAICompatibleProfile,
)
from diary.db.models import (
    Artifact,
    ArtifactLayer,
    ExportStatus,
    Message,
    MessageKind,
    PipelineType,
    ProcessingRun,
    RunStatus,
)
from diary.db.repositories import (
    ArtifactRepository,
    AudioFileRepository,
    MessageRepository,
    ProcessingRunRepository,
    WeeklyExportRepository,
)
from diary.db.results import list_layer_results
from diary.domain.contracts import COACH_DISCLAIMER
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse, LLMUsage
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.base import PipelineInputError, PipelineInvariantError
from diary.domain.pipelines.coach.coach_input import CoachInputError
from diary.domain.pipelines.coach.pipeline import PROMPT_VERSION as COACH_PROMPT_VERSION
from diary.domain.pipelines.coach.pipeline import SUMMARY_HEADER
from diary.domain.pipelines.coach.registration import ensure_coach_pipeline_registered
from diary.domain.pipelines.pipeline import (
    ORCHESTRATOR,
    SKIP_NO_INPUT,
    SKIP_PROFILE_OFF,
    SkippedStep,
    run_coach_step,
    run_day_chain,
    run_stt_fix_step,
    run_week_chain,
)
from diary.domain.pipelines.stt_fix.pipeline import ENTRY_HEADER
from diary.domain.pipelines.stt_fix.registration import ensure_stt_fix_pipeline_registered
from diary.domain.pipelines.sum_daily.pipeline import ENTRIES_HEADER
from diary.domain.pipelines.sum_daily.registration import ensure_sum_daily_pipeline_registered
from diary.domain.pipelines.sum_weekly.pipeline import SUMMARIES_HEADER
from diary.domain.pipelines.sum_weekly.registration import ensure_sum_weekly_pipeline_registered
from diary.domain.pipelines.sum_weekly.week_input import week_id_of

#: Local day of the chain (a Monday; ``2026-W38`` is its ISO week).
DAY = date(2026, 9, 14)
#: The Sunday of that week (used by the export row of the isolation test).
SUNDAY = date(2026, 9, 20)
WEEK_ID = "2026-W38"
LOCAL_TZ = "Europe/Moscow"

#: UTC instants of the synthetic messages (07:10 and 20:00 local time).
DAY_MORNING_UTC = datetime(2026, 9, 14, 4, 10, tzinfo=timezone.utc)
DAY_EVENING_UTC = datetime(2026, 9, 14, 17, 0, tzinfo=timezone.utc)
NEXT_DAY_UTC = datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)

MORNING_TEXT = "утрам схадил в паликлинику за спрафкой"
EVENING_TEXT = "вечиром пачинил веласипет"
OTHER_DAY_TEXT = "На следующий день поехал на дачу."

CORRECTED_MORNING = "Утром сходил в поликлинику за справкой."
CORRECTED_EVENING = "Вечером починил велосипед."

#: Synthetic token usage the fake providers report (recorded as provenance).
USAGE = {"input_tokens": 11, "output_tokens": 7}


class FakeLayerProvider:
    """The fake provider of one phase-2 layer.

    Every call is answered from ``payloads`` cyclically (call *n* gets
    ``payloads[n % len(payloads)]``), so a rerun gets a different, deterministic
    payload without extra bookkeeping. The request must carry the header of the
    layer this fake belongs to, which is how a test proves that the right layer
    was called with the right input; ``fail_with`` makes the next calls fail
    instead of answering.
    """

    def __init__(self, *, header: str, payloads: list[object], model: str) -> None:
        self.header = header
        self.payloads = payloads
        self.model = model
        self.requests: list[LLMRequest] = []
        self.fail_with: Exception | None = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        content = request.messages[0].content
        assert self.header in content, f"the request must be shaped like {self.header!r}"
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        payload = self.payloads[(len(self.requests) - 1) % len(self.payloads)]
        text_value = (
            payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        )
        return LLMResponse(
            text=text_value,
            model=self.model,
            usage=LLMUsage(**USAGE),
        )


def _stt_payloads() -> list[object]:
    return [
        {"text": CORRECTED_MORNING, "uncertain": []},
        {"text": CORRECTED_EVENING, "uncertain": []},
    ]


def _day_payloads() -> list[object]:
    return [
        {
            "facts": ["Сходил в поликлинику за справкой.", "Починил велосипед."],
            "themes": ["поликлиника", "велосипед"],
            "energy": None,
            "open_questions": [],
        },
        {
            "facts": ["День прошёл вокруг бытовых дел."],
            "themes": ["быт"],
            "energy": "medium",
            "open_questions": ["Осталось неясным, что заняло вечер."],
        },
    ]


def _week_payloads() -> list[object]:
    return [{"rollup": "Неделя прошла вокруг бытовых дел."}]


def _coach_payloads() -> list[object]:
    return [
        {
            "observations": [
                "В сводке упомянуты бытовые дела.",
                "Есть один день с походом в поликлинику.",
                "Открытым остался вопрос о вечере.",
            ],
            "hypotheses": [
                "Возможно, неделя была занята бытовыми делами.",
                "Похоже, что темп недели был ровным.",
            ],
            "experiment_7d": "В течение следующих 7 дней можно отмечать бытовые дела.",
        }
    ]


@pytest.fixture(autouse=True)
def _registered_layers() -> None:
    """Register the four phase-2 layers and restore the registry afterwards."""
    snapshot = dict(pipeline_registry.SECOND_FAMILY_REGISTRY)
    ensure_stt_fix_pipeline_registered()
    ensure_sum_daily_pipeline_registered()
    ensure_sum_weekly_pipeline_registered()
    ensure_coach_pipeline_registered()
    yield
    pipeline_registry.SECOND_FAMILY_REGISTRY.clear()
    pipeline_registry.SECOND_FAMILY_REGISTRY.update(snapshot)


@pytest.fixture
def providers() -> dict[str, FakeLayerProvider]:
    """One fake provider per phase-2 profile (the injection seam)."""
    return {
        "diary_stt": FakeLayerProvider(
            header=ENTRY_HEADER, payloads=_stt_payloads(), model="fake-stt"
        ),
        "diary_sum": FakeLayerProvider(
            header=ENTRIES_HEADER, payloads=_day_payloads(), model="fake-day"
        ),
        "diary_week": FakeLayerProvider(
            header=SUMMARIES_HEADER, payloads=_week_payloads(), model="fake-week"
        ),
        "diary_coach": FakeLayerProvider(
            header=SUMMARY_HEADER, payloads=_coach_payloads(), model="fake-coach"
        ),
    }


def _llm_profile(model: str) -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:1/v1",  # never contacted by the fakes
        model=model,
        api_key_env="SYNTHETIC_PHASE2_KEY",
        temperature=0.25,
        max_tokens=512,
    )


def _models_config(
    *, stt: bool = True, day: bool = True, week: bool = True, coach: bool = True
) -> ModelsConfig:
    """An in-memory configuration; a ``False`` switch means "layer off"."""
    return ModelsConfig(
        transcription=FasterWhisperProfile(provider="faster_whisper", model="tiny"),
        normalization=_llm_profile("fake-normalizer"),
        summarization=_llm_profile("fake-summarizer"),
        wellbeing_analysis=_llm_profile("fake-wellbeing"),
        diary_stt=_llm_profile("fake-stt") if stt else None,
        diary_sum=_llm_profile("fake-day") if day else None,
        diary_week=_llm_profile("fake-week") if week else None,
        diary_coach=_llm_profile("fake-coach") if coach else None,
    )


def _text_message(session: Session, *, body: str, created_at: datetime) -> Message:
    return MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=created_at
    )


def _voice_message(session: Session, *, created_at: datetime) -> Message:
    """A voice message with an audio row but **no** transcription yet."""
    message = MessageRepository(session).create(
        kind=MessageKind.audio, body_text="", created_at=created_at
    )
    AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path=f"audio/2026/09/{uuid.uuid4()}.webm",
        size_bytes=123,
        sha256="d" * 64,
        created_at=created_at,
    )
    return message


def _raw_state(session: Session) -> list[list[tuple]]:
    """Every row of every raw source table, as raw values (value-by-value)."""
    return [
        session.execute(text(f"SELECT * FROM {table} ORDER BY id")).all()  # noqa: S608
        for table in ("messages", "audio_files", "transcriptions")
    ]


def _runs(session: Session) -> list[ProcessingRun]:
    """Every processing run of the test database, in creation order."""
    return list(
        session.scalars(
            select(ProcessingRun).order_by(ProcessingRun.created_at.asc(), ProcessingRun.id.asc())
        )
    )


def _run_of(session: Session, run_id: uuid.UUID) -> ProcessingRun:
    run = session.get(ProcessingRun, run_id)
    assert run is not None
    return run


def _artifact_of(session: Session, artifact_id: uuid.UUID | None) -> Artifact:
    assert artifact_id is not None
    artifact = session.get(Artifact, artifact_id)
    assert artifact is not None
    return artifact


def test_week_id_of_the_synthetic_day_is_the_iso_week_of_the_chain() -> None:
    """Guard for the fixtures below: the day and the week id belong together."""
    assert week_id_of(DAY) == WEEK_ID


# ---------------------------------------------------------------------------
# the day chain: raw → stt_fix (per message) → sum_daily
# ---------------------------------------------------------------------------


def test_the_day_chain_corrects_every_message_then_summarizes_the_day(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    _text_message(session, body=EVENING_TEXT, created_at=DAY_EVENING_UTC)
    _text_message(session, body=OTHER_DAY_TEXT, created_at=NEXT_DAY_UTC)
    session.commit()
    raw_before = _raw_state(session)

    outcome = run_day_chain(
        session,
        local_date=DAY,
        models_config=_models_config(),
        providers=providers,
        local_tz=LOCAL_TZ,
    )

    assert outcome.layers == (
        PipelineType.stt_corrected,
        PipelineType.stt_corrected,
        PipelineType.daily_summary,
    )
    assert outcome.skipped == ()
    assert outcome.summary is not None and outcome.summary.date == DAY

    # One call per message of the day (the other day's message is out of scope)
    # and the raw text is what the correction layer received.
    assert len(providers["diary_stt"].requests) == 2
    assert MORNING_TEXT in providers["diary_stt"].requests[0].messages[0].content
    assert OTHER_DAY_TEXT not in providers["diary_sum"].requests[0].messages[0].content

    # Every step: its own run, the artifact pointing back at that run, and the
    # provenance the T05 acceptance asked for on the run.
    for step in outcome.steps:
        artifact = _artifact_of(session, step.artifact_id)
        assert artifact.processing_run_id == step.run_id
        assert artifact.export_id is None  # second anchor family
        assert artifact.path is None and artifact.sha256 is None  # stored inline
        assert artifact.version == 1
        run = _run_of(session, step.run_id)
        assert run.status is RunStatus.succeeded
        assert run.input_export_id is None
        assert run.profile_name == step.profile_name
        assert run.model_name == step.model_name
        assert run.prompt_version == step.prompt_version
        assert run.started_at is not None and run.finished_at is not None
        metadata = run.metadata_json
        assert isinstance(metadata, dict)
        assert metadata["orchestrator"] == ORCHESTRATOR
        assert metadata["layer"] == step.layer.value
        assert metadata["prompt_version"] == step.prompt_version
        assert metadata["usage"] == USAGE
        assert isinstance(metadata["input_sha256"], str) and len(metadata["input_sha256"]) == 64

    assert [step.model_name for step in outcome.steps] == [
        "fake-stt",
        "fake-stt",
        "fake-day",
    ]
    assert [step.profile_name for step in outcome.steps] == [
        "diary_stt",
        "diary_stt",
        "diary_sum",
    ]

    day_artifact = _artifact_of(session, outcome.summary_artifact_id)
    assert day_artifact.layer is ArtifactLayer.daily_summary
    assert day_artifact.local_date == DAY and day_artifact.local_tz == LOCAL_TZ
    assert len(list_layer_results(session, layer=ArtifactLayer.daily_summary)) == 1

    # The input is read-only: every raw row is unchanged, value by value.
    assert _raw_state(session) == raw_before


# ---------------------------------------------------------------------------
# the week chain: sum_weekly over the stored day summaries
# ---------------------------------------------------------------------------


def test_the_week_chain_merges_the_stored_day_summaries_not_the_raw_text(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()
    config = _models_config()
    run_day_chain(
        session, local_date=DAY, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )

    outcome = run_week_chain(
        session, week_id=WEEK_ID, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )

    assert outcome.layers == (PipelineType.weekly_summary,)
    assert outcome.skipped == ()
    assert outcome.summary is not None
    assert outcome.summary.week_id == WEEK_ID
    assert outcome.summary.daily_ids == [DAY.isoformat()]
    # The week layer is fed the day summary, never the raw message.
    week_request = providers["diary_week"].requests[0].messages[0].content
    assert DAY.isoformat() in week_request
    assert MORNING_TEXT not in week_request

    step = outcome.steps[0]
    artifact = _artifact_of(session, step.artifact_id)
    assert artifact.week_id == WEEK_ID and artifact.processing_run_id == step.run_id
    run = _run_of(session, step.run_id)
    assert run.profile_name == "diary_week" and run.model_name == "fake-week"
    assert run.prompt_version == step.prompt_version == "v2"
    assert run.input_export_id is None
    assert run.input_artifact_id is None  # the input is a set of day summaries


def test_the_full_chain_from_raw_refreshes_the_days_and_then_the_week(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    _text_message(session, body=EVENING_TEXT, created_at=DAY_EVENING_UTC)
    session.commit()

    outcome = run_week_chain(
        session,
        week_id=WEEK_ID,
        run_days=True,
        models_config=_models_config(),
        providers=providers,
        local_tz=LOCAL_TZ,
    )

    assert outcome.layers == (
        PipelineType.stt_corrected,
        PipelineType.stt_corrected,
        PipelineType.daily_summary,
        PipelineType.weekly_summary,
    )
    assert outcome.summary is not None and outcome.summary.daily_ids == [DAY.isoformat()]
    # The days of the week without messages are skipped, never invented.
    assert outcome.skipped
    assert {step.layer for step in outcome.skipped} == {PipelineType.daily_summary}
    assert {step.reason for step in outcome.skipped} == {SKIP_NO_INPUT}


# ---------------------------------------------------------------------------
# append-only: a rerun appends v+1 and keeps the previous versions
# ---------------------------------------------------------------------------


def test_a_rerun_appends_new_versions_and_keeps_the_previous_ones(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()
    raw_before = _raw_state(session)
    config = _models_config()

    first = run_day_chain(
        session, local_date=DAY, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )
    second = run_day_chain(
        session, local_date=DAY, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )

    assert first.layers == second.layers
    for first_step, second_step in zip(first.steps, second.steps, strict=True):
        assert first_step.run_id != second_step.run_id
        old_artifact = _artifact_of(session, first_step.artifact_id)
        new_artifact = _artifact_of(session, second_step.artifact_id)
        assert old_artifact.version == 1
        assert new_artifact.version == 2
        # The previous version keeps its own content and its own run.
        assert old_artifact.processing_run_id == first_step.run_id
        assert new_artifact.processing_run_id == second_step.run_id

    rows = list_layer_results(
        session, layer=ArtifactLayer.daily_summary, local_date=DAY, local_tz=LOCAL_TZ
    )
    assert [row.version for row in rows] == [1, 2]
    assert rows[0].result_json != rows[1].result_json  # the payloads differ
    assert rows[0].result_json is not None
    assert rows[1].result_json is not None
    # One run per step and rerun, all succeeded, two per message and per day.
    assert len(_runs(session)) == 2 * len(first.steps)
    assert all(run.status is RunStatus.succeeded for run in _runs(session))
    assert _raw_state(session) == raw_before


# ---------------------------------------------------------------------------
# the coaching note: explicit launch only
# ---------------------------------------------------------------------------


def test_the_chains_never_run_the_coaching_note(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()
    config = _models_config()

    run_day_chain(
        session, local_date=DAY, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )
    run_week_chain(
        session, week_id=WEEK_ID, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )
    run_week_chain(
        session,
        week_id=WEEK_ID,
        run_days=True,
        models_config=config,
        providers=providers,
        local_tz=LOCAL_TZ,
    )

    assert providers["diary_coach"].requests == []  # the layer was never called
    assert list_layer_results(session, layer=ArtifactLayer.coach_note) == []
    assert not [run for run in _runs(session) if run.pipeline_type is PipelineType.coach_note]


def test_the_explicit_coach_step_is_anchored_to_the_weekly_summary_artifact(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()
    config = _models_config()
    run_day_chain(
        session, local_date=DAY, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )
    week = run_week_chain(
        session, week_id=WEEK_ID, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )

    outcome = run_coach_step(session, week_id=WEEK_ID, models_config=config, providers=providers)

    assert outcome.skipped is None
    step = outcome.step
    assert step is not None and step.layer is PipelineType.coach_note
    assert outcome.note is not None
    assert outcome.note.summary_id == WEEK_ID  # the contract field
    assert outcome.note.disclaimer == COACH_DISCLAIMER
    artifact = _artifact_of(session, step.artifact_id)
    assert artifact.layer is ArtifactLayer.coach_note
    assert artifact.summary_id == week.summary_artifact_id  # the storage anchor
    assert artifact.processing_run_id == step.run_id
    run = _run_of(session, step.run_id)
    assert run.input_export_id is None
    assert run.input_artifact_id == week.summary_artifact_id
    assert run.profile_name == "diary_coach" and run.model_name == "fake-coach"
    assert run.prompt_version == step.prompt_version == COACH_PROMPT_VERSION
    assert len(providers["diary_coach"].requests) == 1


# ---------------------------------------------------------------------------
# a failure is durable, leaves no partial artifact and stops the chain
# ---------------------------------------------------------------------------


def test_a_failed_stt_step_records_a_failed_run_and_stops_the_chain(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()
    raw_before = _raw_state(session)
    providers["diary_stt"].fail_with = LLMStructuredOutputError("Synthetic invalid completion.")

    with pytest.raises(LLMStructuredOutputError):
        run_day_chain(
            session,
            local_date=DAY,
            models_config=_models_config(),
            providers=providers,
            local_tz=LOCAL_TZ,
        )

    runs = _runs(session)
    assert [run.pipeline_type for run in runs] == [PipelineType.stt_corrected]
    assert runs[0].status is RunStatus.failed
    assert runs[0].finished_at is not None
    assert runs[0].error_summary is not None
    assert runs[0].error_summary.startswith("LLMStructuredOutputError:")
    assert runs[0].metadata_json is not None
    assert runs[0].metadata_json["terminal"] is True
    # No partial artifact anywhere, and the raw rows are untouched.
    assert list_layer_results(session, layer=ArtifactLayer.stt_corrected) == []
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary) == []
    assert _raw_state(session) == raw_before


def test_a_failed_day_step_keeps_the_committed_stt_steps(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()
    providers["diary_sum"].fail_with = LLMStructuredOutputError("Synthetic invalid completion.")

    with pytest.raises(LLMStructuredOutputError):
        run_day_chain(
            session,
            local_date=DAY,
            models_config=_models_config(),
            providers=providers,
            local_tz=LOCAL_TZ,
        )

    runs = _runs(session)
    assert [run.pipeline_type for run in runs] == [
        PipelineType.stt_corrected,
        PipelineType.daily_summary,
    ]
    assert [run.status for run in runs] == [RunStatus.succeeded, RunStatus.failed]
    # The earlier step is committed with its artifact and its run; the failed
    # step stored nothing.
    stored = list_layer_results(session, layer=ArtifactLayer.stt_corrected)
    assert len(stored) == 1 and stored[0].processing_run_id == runs[0].id
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary) == []


# ---------------------------------------------------------------------------
# isolation from the export-scoped views (ADR-008 §7)
# ---------------------------------------------------------------------------


def test_second_family_runs_never_enter_the_export_scoped_views(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    """The variant chosen for ADR-008 §7: the chain's runs carry no export id.

    A first-family run of a weekly export is created alongside the whole chain,
    so the export-scoped selection and the page helper that renders it are
    exercised for real: they must return exactly the first-family run and must
    never meet an artifact without a rendered file.
    """
    export = WeeklyExportRepository(session).create(
        week_end_date=SUNDAY,
        seq_number=1,
        revision=1,
        path="exports/01_original/2026-09-20_original_0001_r1.txt",
        sha256="a" * 64,
        status=ExportStatus.created,
    )
    first_family_run = ProcessingRunRepository(session).create(
        pipeline_type=PipelineType.normalize,
        status=RunStatus.succeeded,
        input_export_id=export.id,
        profile_name="normalization",
        model_name="fake-normalizer",
        prompt_version="v1",
    )
    session.commit()

    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()
    config = _models_config()
    run_week_chain(
        session,
        week_id=WEEK_ID,
        run_days=True,
        models_config=config,
        providers=providers,
        local_tz=LOCAL_TZ,
    )
    run_coach_step(session, week_id=WEEK_ID, models_config=config, providers=providers)

    chain_runs = [
        run
        for run in _runs(session)
        if run.pipeline_type.value
        in {
            "stt_corrected",
            "daily_summary",
            "weekly_summary",
            "coach_note",
        }
    ]
    assert chain_runs
    assert all(run.input_export_id is None for run in chain_runs)

    # The export-scoped selection cannot see them ...
    assert ProcessingRunRepository(session).list_by_input_export(export.id) == [first_family_run]
    # ... although every one of them exists in the same table (the filter, not
    # an empty database, is what keeps the view away from them).
    assert len(_runs(session)) == len(chain_runs) + 1
    # ... and the page helper that requires a rendered file renders exactly the
    # first-family run (the second family's file-less results are unreachable).
    views = pages._export_runs_view(session, export)
    assert [view["pipeline_type"] for view in views] == [PipelineType.normalize.value]
    assert views[0]["output"] is None
    # Export-scoped artifact listings filter on export_id for the same reason.
    assert ArtifactRepository(session).list_by_export(export.id) == []
    # The inline results of the chain exist, but none of them carries a file.
    assert all(
        artifact.path is None
        for layer in (
            ArtifactLayer.stt_corrected,
            ArtifactLayer.daily_summary,
            ArtifactLayer.weekly_summary,
            ArtifactLayer.coach_note,
        )
        for artifact in list_layer_results(session, layer=layer)
    )


# ---------------------------------------------------------------------------
# switched-off layers and empty inputs
# ---------------------------------------------------------------------------


def test_an_injected_providers_mapping_missing_a_layer_fails_loudly(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    """The injection seam must never fall back to a real endpoint silently."""
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()
    providers.pop("diary_sum")

    with pytest.raises(ValueError, match="no entry for profile 'diary_sum'"):
        run_day_chain(
            session,
            local_date=DAY,
            models_config=_models_config(),
            providers=providers,
            local_tz=LOCAL_TZ,
        )

    # The step that could not get a provider recorded nothing at all; the steps
    # before it are already committed.
    assert [run.pipeline_type for run in _runs(session)] == [PipelineType.stt_corrected]
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary) == []


def test_the_day_chain_gate_refuses_a_contradicting_summary(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    """T24b, behavioural: the instrument the entry armed really acts inside the chain.

    Reviews of T24b found two ways for this wiring to rot quietly — an ungated path
    (the ``run_days`` refresh) and a vacuous source assertion — so the forwarding is
    pinned by behaviour as well: a contradicting judge must make the day step fail
    and store nothing, while a supporting one lets the same chain pass.
    """
    from _pipeline_fakes import contradicting_nli_judge

    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()

    with pytest.raises(PipelineInvariantError) as error:
        run_day_chain(
            session,
            local_date=DAY,
            models_config=_models_config(),
            providers=providers,
            local_tz=LOCAL_TZ,
            nli_judge=contradicting_nli_judge(),
        )

    # The refusal is loud and permanent (ADR-012): the chain does not swallow it,
    # the worker maps the permanent error to a failed job, and nothing is stored.
    assert error.value.checks == ("no_contradiction_with_source",)
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY) == []


def test_the_day_chain_gate_lets_a_supported_summary_through(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    """The control for the test above: with a supporting judge the same chain stores."""
    from _pipeline_fakes import supported_nli_judge

    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()

    outcome = run_day_chain(
        session,
        local_date=DAY,
        models_config=_models_config(),
        providers=providers,
        local_tz=LOCAL_TZ,
        nli_judge=supported_nli_judge(),
    )

    assert outcome.summary is not None
    assert len(list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=DAY)) == 1


def test_the_week_refresh_forwards_the_nli_gate_to_every_day(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    """T24b review: the ``run_days`` path is a second way to store a day summary.

    The arming of the weekly entries used to rest on a source-text test alone, which
    the review called out: the gate could be dropped from the refresh while the test
    stayed green. This one proves the instrument actually reaches every refreshed day
    — the days are faked, the forwarding is real, and the week part is switched off
    by the configuration so nothing else runs.
    """
    import types

    from diary.domain.pipelines import pipeline as phase2_pipeline

    sentinel = object()
    seen: list[object] = []

    def _fake_run_day_chain(session: object, *, local_date: object, **kwargs: object):
        seen.append(kwargs.get("nli_judge"))
        return types.SimpleNamespace(steps=[], skipped=[])

    original = phase2_pipeline.run_day_chain
    phase2_pipeline.run_day_chain = _fake_run_day_chain
    try:
        outcome = run_week_chain(
            session,
            week_id=WEEK_ID,
            run_days=True,
            models_config=_models_config(stt=False, day=False, week=False, coach=False),
            providers=providers,
            local_tz=LOCAL_TZ,
            nli_judge=sentinel,
        )
    finally:
        phase2_pipeline.run_day_chain = original

    assert len(seen) == 7  # Monday through Sunday of the ISO week
    assert all(item is sentinel for item in seen)
    assert outcome.steps == ()  # the week layer is off in this configuration


def test_a_layer_without_a_profile_is_switched_off_not_an_error(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()
    config = _models_config(stt=False, day=False, week=False, coach=False)

    day = run_day_chain(
        session, local_date=DAY, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )
    assert day.steps == ()
    assert {(step.layer, step.reason) for step in day.skipped} == {
        (PipelineType.stt_corrected, SKIP_PROFILE_OFF),
        (PipelineType.daily_summary, SKIP_PROFILE_OFF),
    }
    assert day.summary is None and day.summary_artifact_id is None

    week = run_week_chain(
        session, week_id=WEEK_ID, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )
    assert week.steps == () and week.summary is None
    assert [(step.layer, step.reason) for step in week.skipped] == [
        (PipelineType.weekly_summary, SKIP_PROFILE_OFF)
    ]

    coach = run_coach_step(session, week_id=WEEK_ID, models_config=config, providers=providers)
    assert coach.step is None and coach.note is None
    assert coach.skipped == SkippedStep(
        PipelineType.coach_note, SKIP_PROFILE_OFF, f"week:{WEEK_ID}"
    )

    # A switched-off layer records no run and calls no provider.
    assert _runs(session) == []
    assert all(provider.requests == [] for provider in providers.values())


def test_a_partially_configured_chain_skips_only_the_missing_layer(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()

    outcome = run_day_chain(
        session,
        local_date=DAY,
        models_config=_models_config(day=False),
        providers=providers,
        local_tz=LOCAL_TZ,
    )

    assert outcome.layers == (PipelineType.stt_corrected,)
    assert [(step.layer, step.reason) for step in outcome.skipped] == [
        (PipelineType.daily_summary, SKIP_PROFILE_OFF)
    ]
    assert providers["diary_sum"].requests == []


def test_a_day_without_messages_and_a_week_without_summaries_are_skipped(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    config = _models_config()

    day = run_day_chain(
        session, local_date=DAY, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )
    assert day.steps == ()
    assert day.skipped == (
        SkippedStep(PipelineType.daily_summary, SKIP_NO_INPUT, f"day:{DAY.isoformat()}"),
    )

    week = run_week_chain(
        session, week_id=WEEK_ID, models_config=config, providers=providers, local_tz=LOCAL_TZ
    )
    assert week.steps == ()
    assert week.skipped == (
        SkippedStep(PipelineType.weekly_summary, SKIP_NO_INPUT, f"week:{WEEK_ID}"),
    )

    # The explicit coach call has nothing to build on: it fails loudly and
    # records the failure instead of inventing a week.
    with pytest.raises(CoachInputError):
        run_coach_step(session, week_id=WEEK_ID, models_config=config, providers=providers)
    runs = _runs(session)
    assert [run.pipeline_type for run in runs] == [PipelineType.coach_note]
    assert runs[0].status is RunStatus.failed
    assert list_layer_results(session, layer=ArtifactLayer.coach_note) == []
    assert providers["diary_sum"].requests == [] and providers["diary_week"].requests == []


def test_a_voice_message_without_a_transcription_is_skipped_per_message(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    voice = _voice_message(session, created_at=DAY_MORNING_UTC)
    _text_message(session, body=EVENING_TEXT, created_at=DAY_EVENING_UTC)
    session.commit()

    outcome = run_day_chain(
        session,
        local_date=DAY,
        models_config=_models_config(),
        providers=providers,
        local_tz=LOCAL_TZ,
    )

    assert outcome.layers == (PipelineType.stt_corrected, PipelineType.daily_summary)
    assert outcome.skipped == (
        SkippedStep(PipelineType.stt_corrected, SKIP_NO_INPUT, f"message:{voice.id}"),
    )
    assert len(providers["diary_stt"].requests) == 1
    # The untranscribed voice message still belongs to the day's input (its raw
    # text is empty by contract) — it is skipped, not dropped silently.
    assert '"source": "voice"' in providers["diary_sum"].requests[0].messages[0].content


# ---------------------------------------------------------------------------
# one message: the stt_fix unit added by T11 (the job dispatch's smallest unit)
# ---------------------------------------------------------------------------


def test_the_message_step_corrects_exactly_one_message_of_the_day(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    """``run_stt_fix_step`` must not reach beyond the message it was given."""
    first = _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    _text_message(session, body=EVENING_TEXT, created_at=DAY_EVENING_UTC)
    session.commit()

    outcome = run_stt_fix_step(
        session,
        raw_id=first.id,
        models_config=_models_config(),
        providers=providers,
        local_tz=LOCAL_TZ,
    )

    assert outcome.skipped is None
    step = outcome.step
    assert step is not None and step.layer is PipelineType.stt_corrected
    assert step.input_ref == f"message:{first.id}"
    assert outcome.correction is not None and outcome.correction.raw_id == first.id
    # Exactly one LLM call: the other message of the day was not touched, and no
    # day summary was produced (that is the day chain's job, not this unit's).
    assert len(providers["diary_stt"].requests) == 1
    assert MORNING_TEXT in providers["diary_stt"].requests[0].messages[0].content
    assert providers["diary_sum"].requests == []
    stored = list_layer_results(session, layer=ArtifactLayer.stt_corrected)
    assert len(stored) == 1 and stored[0].raw_id == first.id


def test_the_message_step_is_skipped_when_the_layer_is_switched_off(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    message = _text_message(session, body=MORNING_TEXT, created_at=DAY_MORNING_UTC)
    session.commit()

    outcome = run_stt_fix_step(
        session,
        raw_id=message.id,
        models_config=_models_config(stt=False),
        providers=providers,
        local_tz=LOCAL_TZ,
    )

    assert outcome.step is None
    assert outcome.skipped == SkippedStep(
        PipelineType.stt_corrected, SKIP_PROFILE_OFF, f"message:{message.id}"
    )
    assert providers["diary_stt"].requests == []
    assert list_layer_results(session, layer=ArtifactLayer.stt_corrected) == []


def test_the_message_step_skips_a_voice_message_without_a_transcription(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    voice = _voice_message(session, created_at=DAY_MORNING_UTC)
    session.commit()

    outcome = run_stt_fix_step(
        session,
        raw_id=voice.id,
        models_config=_models_config(),
        providers=providers,
        local_tz=LOCAL_TZ,
    )

    assert outcome.step is None
    assert outcome.skipped == SkippedStep(
        PipelineType.stt_corrected, SKIP_NO_INPUT, f"message:{voice.id}"
    )
    assert providers["diary_stt"].requests == []


def test_the_message_step_fails_loudly_for_an_unknown_message(
    session: Session, providers: dict[str, FakeLayerProvider]
) -> None:
    with pytest.raises(PipelineInputError):
        run_stt_fix_step(
            session,
            raw_id=uuid.uuid4(),
            models_config=_models_config(),
            providers=providers,
            local_tz=LOCAL_TZ,
        )
    assert providers["diary_stt"].requests == []
    assert list_layer_results(session, layer=ArtifactLayer.stt_corrected) == []
