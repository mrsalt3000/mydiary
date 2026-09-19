"""Run one daily summary and store it (task T05).

:func:`sum_daily` is the run itself: select the messages of one local calendar
day (``APP_TIMEZONE``, or an explicit ``local_tz``) into ``raw_entry[]``, build
the request, call the provider, validate the completion into
:class:`~diary.domain.contracts.DailySummary` and append the result to the
phase-2 store under the ``local_date`` + ``local_tz`` anchor (task T08).
:func:`run_sum_daily` is the convenience entry point that resolves the optional
``diary_sum`` profile and the provider from the configuration — the shape task
T09's orchestrator needs for a ``daily_summary`` job.

Properties that matter for the T05 definition of done:

* the **input is never written**: the day's messages, audio metadata and
  transcriptions are read only, and the only write is ``save_layer_result`` into
  ``artifacts`` (the selector in :mod:`.day_input` has no write path at all);
* exactly one day is summarized per run — the entries come from a single local
  calendar day, never from a range, and there is no "aggregate a few days while
  you are at it" option;
* an invalid completion raises
  :class:`~diary.domain.llm.errors.LLMStructuredOutputError` **before** any
  result is stored, so a failed run leaves no partial artifact; a day with no
  messages raises :class:`~diary.domain.pipelines.sum_daily.day_input.DayInputError`
  before the provider is called;
* the stored result is the validated contract — the four content fields the
  model produced plus the backend's ``date`` — and nothing else;
* the version of the (day, layer) pair is computed by the store: a repeated run
  appends ``v+1`` and keeps the previous result intact (ADR-005);
* the caller owns the transaction (``save_layer_result`` flushes but never
  commits), so the result row and the processing run it belongs to are
  committed atomically by the orchestrator.

Privacy: the day's texts exist only inside the in-memory request and the
validated contract; nothing here logs them (AGENTS.md §7-8).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from diary.core.models_config import ModelsConfig, OpenAICompatibleProfile, load_models_config
from diary.core.settings import get_settings
from diary.db.models import Artifact, ArtifactLayer
from diary.db.results import save_layer_result
from diary.domain.contracts import DailySummary, RawEntry
from diary.domain.llm.providers import LLMProvider, create_llm_provider
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.pipelines.sum_daily.day_input import collect_day_entries
from diary.domain.pipelines.sum_daily.pipeline import PROFILE_NAME, SumDailyPipeline


@dataclass(frozen=True)
class SumDailyOutcome:
    """The validated day summary and the appended result row of one run."""

    summary: DailySummary
    artifact: Artifact


def sum_daily(
    session: Session,
    *,
    local_date: date,
    profile: OpenAICompatibleProfile,
    provider: LLMProvider,
    local_tz: str | None = None,
    entries: list[RawEntry] | None = None,
    processing_run_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
) -> SumDailyOutcome:
    """Summarize one local day with ``provider`` and append the result (no commit).

    ``local_tz`` is the timezone the calendar-day boundary is computed in and is
    stored with the anchor (``save_layer_result`` rejects a local day without
    its timezone); it defaults to ``APP_TIMEZONE``. ``entries`` overrides the
    day selection — used by tests and by a caller that already holds the day's
    ``raw_entry[]`` — and defaults to the day's active messages mapped by
    :func:`~diary.domain.pipelines.sum_daily.day_input.collect_day_entries`.

    ``profile`` is the resolved phase-2 profile of the layer; the provider is
    injected so tests (and the GPU-arbitrated orchestrator of T09) control the
    call. The result is anchored to the pair (``local_date``, ``local_tz``), and
    the version is computed by the store: a repeated run appends ``v+1`` and
    keeps the previous result (ADR-005).
    """
    timezone_name = local_tz or get_settings().timezone
    day_entries = (
        entries
        if entries is not None
        else collect_day_entries(session, local_date=local_date, tz=ZoneInfo(timezone_name))
    )
    pipeline = SumDailyPipeline()
    request = pipeline.build_request(entries=day_entries)
    response = provider.complete(request)
    summary = pipeline.parse_result(response, local_date=local_date)
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.daily_summary,
        local_date=local_date,
        local_tz=timezone_name,
        result=summary,
        processing_run_id=processing_run_id,
        created_at=created_at,
    )
    return SumDailyOutcome(summary=summary, artifact=artifact)


def run_sum_daily(
    session: Session,
    *,
    local_date: date,
    models_config: ModelsConfig | None = None,
    provider: LLMProvider | None = None,
    local_tz: str | None = None,
    entries: list[RawEntry] | None = None,
    processing_run_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
) -> SumDailyOutcome:
    """Resolve the ``diary_sum`` profile/provider and summarize one local day.

    An absent ``diary_sum`` profile means the layer is switched off and raises
    :class:`~diary.domain.llm.errors.LLMProviderUnavailableError` (never
    retryable) before any request is sent. ``provider`` overrides the factory —
    tests inject a fake, and T09 may pass its own instance.
    """
    config = models_config if models_config is not None else load_models_config()
    profile = resolve_phase2_profile(config, PROFILE_NAME)
    active_provider = provider if provider is not None else create_llm_provider(profile)
    return sum_daily(
        session,
        local_date=local_date,
        profile=profile,
        provider=active_provider,
        local_tz=local_tz,
        entries=entries,
        processing_run_id=processing_run_id,
        created_at=created_at,
    )


__all__ = ["SumDailyOutcome", "run_sum_daily", "sum_daily"]
