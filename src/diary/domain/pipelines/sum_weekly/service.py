"""Run one weekly summary and store it (task T06).

:func:`sum_weekly` is the run itself: select the daily summaries of one ISO week
(:func:`~diary.domain.pipelines.sum_weekly.week_input.collect_week_summaries`),
build the request(s), call the provider, validate the completion into
:class:`~diary.domain.contracts.WeeklySummary` and append the result to the
phase-2 store under the ``week_id`` anchor (task T08). :func:`run_sum_weekly` is
the convenience entry point that resolves the optional ``diary_week`` profile and
the provider from the configuration — the shape task T09's orchestrator needs
for a ``weekly_summary`` job.

Properties that matter for the T06 definition of done:

* **the input is the daily summaries only**, never raw messages and never a
  weekly export (the reading happens in ``week_input`` through the
  ``daily_summary`` results of the store); the summaries are read only and the
  layer's single write is ``save_layer_result`` into ``artifacts``;
* **the anchor is the ISO week**, computed from the local day of the summaries
  (``APP_TIMEZONE`` weeks are Monday–Sunday and map onto the ISO week of their
  Monday, ADR-007 §5) — not a ``weekly_exports`` row, which this layer never
  reads;
* exactly one week is summarized per run;
* an invalid completion (or an invalid day summary) raises before any result is
  stored, so a failed run leaves no partial artifact;
* a context-budget overflow splits the days into ordered chunks that each fit
  and reconciles the partial rollups into **one** coherent ``rollup`` with a
  final merge call (never by concatenating the partial texts);
* the stored result is the validated contract — ``rollup`` plus the backend's
  ``week_id`` and ``daily_ids`` (the days actually fed) — and nothing else;
* the version of the (week, layer) pair is computed by the store: a repeated
  run appends ``v+1`` and keeps the previous result intact (ADR-005);
* the caller owns the transaction (``save_layer_result`` flushes but never
  commits), so the result row and the processing run it belongs to are
  committed atomically by the orchestrator.

Privacy: the day summaries exist only inside the in-memory requests and the
validated contract; nothing here logs them (AGENTS.md §7-8).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from diary.core.models_config import ModelsConfig, OpenAICompatibleProfile, load_models_config
from diary.core.settings import get_settings
from diary.db.models import Artifact, ArtifactLayer
from diary.db.results import save_layer_result
from diary.domain.contracts import DailySummary, WeeklySummary
from diary.domain.llm.providers import LLMProvider, create_llm_provider
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.pipelines.quality import generate_with_invariants
from diary.domain.pipelines.sum_weekly.pipeline import PROFILE_NAME, SumWeeklyPipeline
from diary.domain.pipelines.sum_weekly.week_input import (
    collect_week_summaries,
    ensure_summaries_match_week,
)


@dataclass(frozen=True)
class SumWeeklyOutcome:
    """The validated week summary and the appended result row of one run."""

    summary: WeeklySummary
    artifact: Artifact


def sum_weekly(
    session: Session,
    *,
    week_id: str,
    profile: OpenAICompatibleProfile,
    provider: LLMProvider,
    local_tz: str | None = None,
    summaries: list[DailySummary] | None = None,
    processing_run_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
) -> SumWeeklyOutcome:
    """Summarize one ISO week with ``provider`` and append the result (no commit).

    ``local_tz`` is the timezone the day anchors were computed in and is used to
    select them; it defaults to ``APP_TIMEZONE``. ``summaries`` overrides the
    week selection — used by tests and by a caller that already holds the week's
    ``DailySummary`` list — and defaults to the week's stored day summaries
    (newest version of each day). Every supplied summary must belong to
    ``week_id``: the stored ``daily_ids`` are the days actually fed, so a
    foreign day would make the anchor a lie.

    ``profile`` is the resolved phase-2 profile of the layer; the provider is
    injected so tests (and the GPU-arbitrated orchestrator of T09) control the
    call. The result is anchored to ``week_id`` and the version is computed by
    the store: a repeated run appends ``v+1`` and keeps the previous result
    (ADR-005).

    When the days do not fit one context window, they are split into ordered
    chunks (no day lost, none duplicated) and each chunk produces a partial
    rollup; a final merge call turns the partials into the one coherent
    ``rollup`` that is stored. The stored ``daily_ids`` always list every day
    that was fed, whatever the number of calls.
    """
    timezone_name = local_tz or get_settings().timezone
    week_summaries = (
        summaries
        if summaries is not None
        else collect_week_summaries(session, week_id=week_id, local_tz=timezone_name)
    )
    # Fails loudly on a malformed week_id or a day of another week, before any
    # provider call.
    ensure_summaries_match_week(week_summaries, week_id)
    daily_ids = [summary.date.isoformat() for summary in week_summaries]

    pipeline = SumWeeklyPipeline()
    chunks = pipeline.chunk_summaries(week_summaries, profile=profile)
    layer = ArtifactLayer.weekly_summary.value

    def generate_from(chunk: list[DailySummary]) -> Callable[[int], WeeklySummary]:
        """One full attempt at the rollup of ``chunk`` (request, call, contract)."""

        def generate(_attempt: int) -> WeeklySummary:
            response = provider.complete(pipeline.build_request(summaries=chunk))
            return pipeline.parse_result(
                response,
                week_id=week_id,
                daily_ids=[summary.date.isoformat() for summary in chunk],
            )

        return generate

    if len(chunks) == 1:
        summary = generate_with_invariants(
            layer=layer, source=chunks[0], generate=generate_from(chunks[0])
        )
    else:
        # A chunk's partial is graded against the chunk it was written from, so a
        # partial that invents a fact cannot be laundered through the merge.
        partials: list[WeeklySummary] = [
            generate_with_invariants(layer=layer, source=chunk, generate=generate_from(chunk))
            for chunk in chunks
        ]

        def generate_merge(_attempt: int) -> WeeklySummary:
            merge_request = pipeline.build_merge_request(partials=partials)
            pipeline.require_merge_fits(merge_request, profile=profile)
            response = provider.complete(merge_request)
            return pipeline.parse_result(response, week_id=week_id, daily_ids=daily_ids)

        # The merge is graded against the week's own days, not against the
        # partials it was handed: the days are the normative source of the week
        # (and the facts the eval harness grades the same output against), while
        # a number that exists only in a partial is not a fact about the week.
        summary = generate_with_invariants(
            layer=layer, source=week_summaries, generate=generate_merge
        )
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.weekly_summary,
        week_id=week_id,
        result=summary,
        processing_run_id=processing_run_id,
        created_at=created_at,
    )
    return SumWeeklyOutcome(summary=summary, artifact=artifact)


def run_sum_weekly(
    session: Session,
    *,
    week_id: str,
    models_config: ModelsConfig | None = None,
    provider: LLMProvider | None = None,
    local_tz: str | None = None,
    summaries: list[DailySummary] | None = None,
    processing_run_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
) -> SumWeeklyOutcome:
    """Resolve the ``diary_week`` profile/provider and summarize one ISO week.

    An absent ``diary_week`` profile means the layer is switched off and raises
    :class:`~diary.domain.llm.errors.LLMProviderUnavailableError` (never
    retryable) before any request is sent. ``provider`` overrides the factory —
    tests inject a fake, and T09 may pass its own instance.
    """
    config = models_config if models_config is not None else load_models_config()
    profile = resolve_phase2_profile(config, PROFILE_NAME)
    active_provider = provider if provider is not None else create_llm_provider(profile)
    return sum_weekly(
        session,
        week_id=week_id,
        profile=profile,
        provider=active_provider,
        local_tz=local_tz,
        summaries=summaries,
        processing_run_id=processing_run_id,
        created_at=created_at,
    )


__all__ = ["SumWeeklyOutcome", "run_sum_weekly", "sum_weekly"]
