"""Integration tests for the phase-2 job dispatch (task T11, migrated DB).

Before T11 the four phase-2 job types had no handler, so an enqueued
``stt_corrected``/``daily_summary``/``weekly_summary``/``coach_note`` job failed
permanently with "no job handler is registered". This module drives the real
queue (``process_due_jobs`` → handler → orchestrator → append-only result store)
over an isolated database created solely by the migrations, with fake per-layer
providers injected through the orchestrator's provider factory.

Pinned here:

* the three dispatched job types run their layer and store the result, with the
  job ending ``succeeded`` and the artifact carrying the run of its step;
* the payload carries the anchor only (``raw_id``/``local_date``/``week_id``),
  which is what makes the job content-free;
* ``coach_note`` still has **no** handler: the job fails permanently and no
  coaching note is produced behind the user's back (T07 decision 3);
* a layer that is switched off (no profile) fails permanently instead of
  silently succeeding, while a legitimately empty input (a day without
  messages, a week without day summaries) succeeds as a no-op.

All content is synthetic (AGENTS.md §13); no endpoint, model or real
configuration file is used.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from diary.core.models_config import (
    FasterWhisperProfile,
    ModelsConfig,
    OpenAICompatibleProfile,
)
from diary.db.models import (
    ArtifactLayer,
    JobStatus,
    JobType,
    MessageKind,
    PipelineType,
    ProcessingRun,
)
from diary.db.repositories import JobRepository, MessageRepository, ProcessingRunRepository
from diary.db.results import list_layer_results
from diary.domain.llm.models import LLMRequest, LLMResponse, LLMUsage
from diary.domain.pipelines import pipeline as phase2_pipeline
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.coach.registration import ensure_coach_pipeline_registered
from diary.domain.pipelines.stt_fix.pipeline import ENTRY_HEADER
from diary.domain.pipelines.stt_fix.registration import ensure_stt_fix_pipeline_registered
from diary.domain.pipelines.sum_daily.pipeline import ENTRIES_HEADER
from diary.domain.pipelines.sum_daily.registration import ensure_sum_daily_pipeline_registered
from diary.domain.pipelines.sum_weekly.pipeline import SUMMARIES_HEADER
from diary.domain.pipelines.sum_weekly.registration import ensure_sum_weekly_pipeline_registered
from diary.workers.runner import process_due_jobs

#: The synthetic local day (a Monday) and its ISO week.
DAY = date(2026, 9, 14)
WEEK_ID = "2026-W38"
LOCAL_TZ = "Europe/Moscow"

#: A UTC instant inside :data:`DAY` in ``Europe/Moscow`` (07:10 local).
MORNING_UTC = datetime(2026, 9, 14, 4, 10, tzinfo=timezone.utc)

RAW_TEXT = "утрам схадил в паликлинику за спрафкой"
CORRECTED_TEXT = "Утром сходил в поликлинику за справкой."

USAGE = {"input_tokens": 11, "output_tokens": 7}


class FakeLayerProvider:
    """The fake provider of one phase-2 layer (header-checked, deterministic)."""

    def __init__(self, *, header: str, payload: object, model: str) -> None:
        self.header = header
        self.payload = payload
        self.model = model
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        content = request.messages[0].content
        assert self.header in content, f"the request must be shaped like {self.header!r}"
        self.requests.append(request)
        text = (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload, ensure_ascii=False)
        )
        return LLMResponse(text=text, model=self.model, usage=LLMUsage(**USAGE))


def _llm_profile(model: str) -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:1/v1",  # never contacted
        model=model,
        api_key_env="SYNTHETIC_PHASE2_KEY",
        temperature=0.25,
        max_tokens=512,
    )


def _models_config(*, stt: bool = True, day: bool = True, week: bool = True) -> ModelsConfig:
    return ModelsConfig(
        transcription=FasterWhisperProfile(provider="faster_whisper", model="tiny"),
        normalization=_llm_profile("fake-normalizer"),
        summarization=_llm_profile("fake-summarizer"),
        wellbeing_analysis=_llm_profile("fake-wellbeing"),
        diary_stt=_llm_profile("fake-stt") if stt else None,
        diary_sum=_llm_profile("fake-day") if day else None,
        diary_week=_llm_profile("fake-week") if week else None,
        diary_coach=_llm_profile("fake-coach"),
    )


def _fake_providers() -> dict[str, FakeLayerProvider]:
    """One fake per layer, keyed by the effective model of its profile."""
    return {
        "fake-stt": FakeLayerProvider(
            header=ENTRY_HEADER,
            payload={"text": CORRECTED_TEXT, "uncertain": []},
            model="fake-stt",
        ),
        "fake-day": FakeLayerProvider(
            header=ENTRIES_HEADER,
            payload={
                "facts": ["Сходил в поликлинику за справкой."],
                "themes": ["поликлиника"],
                "energy": None,
                "open_questions": [],
            },
            model="fake-day",
        ),
        "fake-week": FakeLayerProvider(
            header=SUMMARIES_HEADER,
            payload={"rollup": "Неделя прошла вокруг бытовых дел."},
            model="fake-week",
        ),
    }


@pytest.fixture(autouse=True)
def _registered_layers() -> None:
    """Register the phase-2 layers and restore the registry afterwards."""
    second_family_snapshot = dict(pipeline_registry.SECOND_FAMILY_REGISTRY)
    ensure_stt_fix_pipeline_registered()
    ensure_sum_daily_pipeline_registered()
    ensure_sum_weekly_pipeline_registered()
    ensure_coach_pipeline_registered()
    yield
    pipeline_registry.SECOND_FAMILY_REGISTRY.clear()
    pipeline_registry.SECOND_FAMILY_REGISTRY.update(second_family_snapshot)


@pytest.fixture
def layer_providers() -> dict[str, FakeLayerProvider]:
    return _fake_providers()


@pytest.fixture(autouse=True)
def _wire_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    layer_providers: dict[str, FakeLayerProvider],
) -> None:
    """Give the orchestrator the in-memory configuration and the fake providers.

    The handlers call the orchestrator without an injection seam (production
    path), so the seam is patched at the orchestrator module — the same trick
    every other pipeline test uses.
    """

    def _models() -> ModelsConfig:
        return _models_config()

    monkeypatch.setattr(phase2_pipeline, "load_models_config", _models)

    def _create_provider(profile: OpenAICompatibleProfile) -> FakeLayerProvider:
        return layer_providers[profile.model]

    monkeypatch.setattr(phase2_pipeline, "create_llm_provider", _create_provider)


def _enqueue(session: Session, job_type: JobType, payload: dict[str, object]) -> uuid.UUID:
    job = JobRepository(session).enqueue(job_type=job_type, payload_json=payload)
    session.commit()
    return job.id


def _text_message(session: Session, *, body: str, created_at: datetime = MORNING_UTC) -> uuid.UUID:
    message = MessageRepository(session).create(
        kind=MessageKind.text,
        body_text=body,
        created_at=created_at,
    )
    session.commit()
    return message.id


# ---------------------------------------------------------------------------
# the three dispatched layers
# ---------------------------------------------------------------------------


def test_a_stt_corrected_job_corrects_exactly_one_message(
    session: Session, session_factory, layer_providers: dict[str, FakeLayerProvider]
) -> None:
    message_id = _text_message(session, body=RAW_TEXT)
    job_id = _enqueue(session, JobType.stt_corrected, {"raw_id": str(message_id)})

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    job = JobRepository(session).get(job_id)
    assert job is not None and job.status is JobStatus.succeeded
    stored = list_layer_results(session, layer=ArtifactLayer.stt_corrected)
    assert len(stored) == 1
    assert stored[0].raw_id == message_id
    # Exactly one message was corrected — a day chain would also summarize the day.
    assert len(layer_providers["fake-stt"].requests) == 1
    assert not list_layer_results(session, layer=ArtifactLayer.daily_summary)


def test_a_daily_summary_job_runs_the_day_chain(
    session: Session, session_factory, layer_providers: dict[str, FakeLayerProvider]
) -> None:
    message_id = _text_message(session, body=RAW_TEXT)
    job_id = _enqueue(
        session,
        JobType.daily_summary,
        {"local_date": DAY.isoformat(), "local_tz": LOCAL_TZ},
    )

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    job = JobRepository(session).get(job_id)
    assert job is not None and job.status is JobStatus.succeeded
    stored = list_layer_results(session, layer=ArtifactLayer.daily_summary)
    assert len(stored) == 1
    assert stored[0].local_date == DAY and stored[0].local_tz == LOCAL_TZ
    assert stored[0].result_json is not None
    # The day chain corrects the day's messages first (T09 semantics).
    corrections = list_layer_results(session, layer=ArtifactLayer.stt_corrected)
    assert [row.raw_id for row in corrections] == [message_id]


def test_a_weekly_summary_job_merges_the_stored_day_summaries(
    session: Session, session_factory, layer_providers: dict[str, FakeLayerProvider]
) -> None:
    _text_message(session, body=RAW_TEXT)
    _enqueue(session, JobType.daily_summary, {"local_date": DAY.isoformat(), "local_tz": LOCAL_TZ})
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    week_job = _enqueue(session, JobType.weekly_summary, {"week_id": WEEK_ID})

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    job = JobRepository(session).get(week_job)
    assert job is not None and job.status is JobStatus.succeeded
    stored = list_layer_results(session, layer=ArtifactLayer.weekly_summary)
    assert len(stored) == 1
    assert stored[0].week_id == WEEK_ID
    # The week layer consumed a day summary, never the raw text.
    week_request = layer_providers["fake-week"].requests[0].messages[0].content
    assert DAY.isoformat() in week_request
    assert RAW_TEXT not in week_request


def test_a_weekly_summary_job_with_run_days_refreshes_the_week_first(
    session: Session, session_factory
) -> None:
    _text_message(session, body=RAW_TEXT)
    week_job = _enqueue(session, JobType.weekly_summary, {"week_id": WEEK_ID, "run_days": True})

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    job = JobRepository(session).get(week_job)
    assert job is not None and job.status is JobStatus.succeeded
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary)
    assert list_layer_results(session, layer=ArtifactLayer.weekly_summary)


def test_the_coach_note_job_still_fails_permanently(session: Session, session_factory) -> None:
    """Layer 5 keeps «explicit launch only» (T07 decision 3)."""
    _text_message(session, body=RAW_TEXT)
    _enqueue(session, JobType.daily_summary, {"local_date": DAY.isoformat(), "local_tz": LOCAL_TZ})
    process_due_jobs(session_factory, worker_id="test-worker")
    coach_job = _enqueue(session, JobType.coach_note, {"week_id": WEEK_ID})

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    job = JobRepository(session).get(coach_job)
    assert job is not None and job.status is JobStatus.failed
    assert job.error_summary is not None
    assert "No job handler is registered for job type 'coach_note'" in job.error_summary
    assert list_layer_results(session, layer=ArtifactLayer.coach_note) == []
    assert ProcessingRunRepository(session).list_by_job_id(coach_job) == []


# ---------------------------------------------------------------------------
# switched off vs legitimately empty
# ---------------------------------------------------------------------------


def test_a_job_for_a_switched_off_layer_fails_permanently(
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``diary_sum`` profile = the day layer is off: the job says so.

    The orchestrator *skips* the step (a valid state, §9.8), but a job that was
    enqueued for that layer must not report success while doing nothing.
    """
    monkeypatch.setattr(phase2_pipeline, "load_models_config", lambda: _models_config(day=False))
    _text_message(session, body=RAW_TEXT)
    job_id = _enqueue(session, JobType.daily_summary, {"local_date": DAY.isoformat()})

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    job = JobRepository(session).get(job_id)
    assert job is not None and job.status is JobStatus.failed
    assert job.error_summary is not None
    assert "daily_summary" in job.error_summary and "diary_sum" in job.error_summary
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary) == []


def test_a_job_for_an_empty_day_succeeds_as_a_noop(session: Session, session_factory) -> None:
    """A day without messages has nothing to summarize — not a failure."""
    job_id = _enqueue(session, JobType.daily_summary, {"local_date": DAY.isoformat()})

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    job = JobRepository(session).get(job_id)
    assert job is not None and job.status is JobStatus.succeeded
    assert list_layer_results(session, layer=ArtifactLayer.daily_summary) == []


def test_a_job_for_an_unknown_message_fails_permanently(session: Session, session_factory) -> None:
    """A message that no longer exists cannot be corrected (permanent)."""
    job_id = _enqueue(session, JobType.stt_corrected, {"raw_id": str(uuid.uuid4())})

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    job = JobRepository(session).get(job_id)
    assert job is not None and job.status is JobStatus.failed
    assert job.error_summary is not None
    assert "does not exist" in job.error_summary


def test_the_dispatched_runs_are_recognisable_as_the_phase2_chain(
    session: Session, session_factory, layer_providers: dict[str, FakeLayerProvider]
) -> None:
    """A job-dispatched step produces the same run shape as an explicit call."""
    _text_message(session, body=RAW_TEXT)
    _enqueue(session, JobType.daily_summary, {"local_date": DAY.isoformat(), "local_tz": LOCAL_TZ})

    process_due_jobs(session_factory, worker_id="test-worker")

    runs = list(session.scalars(select(ProcessingRun).order_by(ProcessingRun.created_at.asc())))
    assert [run.pipeline_type for run in runs] == [
        PipelineType.stt_corrected,
        PipelineType.daily_summary,
    ]
    for run in runs:
        metadata = run.metadata_json if isinstance(run.metadata_json, dict) else {}
        assert metadata.get("orchestrator") == phase2_pipeline.ORCHESTRATOR
        assert run.input_export_id is None  # second anchor family
        assert run.profile_name in {"diary_stt", "diary_sum"}
