"""T10 regression harness for the phase-2 golden set (``eval/golden``).

The harness is the machine check the T05-T07 acceptances asked for: the
checklist items «саммари: нет советов и домыслов» and «coach: нет диагнозов»
used to hold only through prompts, because no code checked the *content* of a
generated field. Here every chunk of the golden set is run through the real
phase-2 layers on a throwaway SQLite database (created by the Alembic
migrations) and every machine-checkable invariant of the chunk is asserted
against the stored result.

Two modes
=========

``ci`` (default)
    No network, no GPU, no API key: the layers run on a deterministic mock
    provider built from the chunk itself, so the harness checks the invariants,
    the context budget and the whole plumbing. The mock is *constructed to
    satisfy* the invariants — it proves the plumbing, not the model. The checks
    themselves are proven by the negative tests in
    ``tests/integration/test_eval_harness.py``, which feed a deliberately
    violating completion through the same code path.

``--live``
    Runs against the configured endpoint (``configs/models.yaml`` or
    ``--models-config``) with the real ``OpenAICompatibleProvider`` and reports
    latency, pass/fail and the failed check names per chunk — never any diary
    content, only chunk ids and metrics. The definition of done of T10 is
    asserted here as **two separate numbers**: at least 8 of 10 chunks clean on
    the invariants, and every measured chunk inside the latency budget of its
    route (T18/T21) — a cold model load fails the run by the latency count and is
    never reported as a content defect.

What is checked
===============

The harness reads the golden set **only** through
:mod:`diary.domain.contracts.golden`: that module owns the chunk format, the
lexeme lists, the numeral words and the budget reserve. The checks are:

* every ``kind`` the set declares has a checker; an unknown ``kind`` is a
  harness error, never a silent skip;
* every enabled invariant field of every chunk maps to an implemented check
  (asserted by a test, so a new field cannot be added and forgotten);
* the generic eval-contract checks added by T10 — no advice/evaluation, no
  causal claims, no new facts (digits, numeral words, proper names) — plus the
  invariant fields the manifest already described;
* the context budget of **every call** the layer makes (T35): ``input_tokens_est +
  prompt_tokens_est + max_tokens <= num_ctx`` per call, where the prompt overhead
  is estimated from the versioned template the layer itself loads (T20, so the
  reserve cannot be understated) and a layer that honestly splits a long input is
  measured per part (T23, ``ADR-016``) instead of as one whole-chunk call; when the
  endpoint reports usage, the *measured* ``input_tokens + profile.max_tokens <=
  num_ctx`` is checked against the largest call, not the last one;
* the raw source rows of an STT/day chunk are unchanged by the run, and the
  in-memory chunk input is not mutated;
* the latency budget declared for the **route of the category** (T21: an STT
  correction under 5 s and a day under 15 s on the GPU tags, a weekly rollup
  under 180 s on the CPU twin ``diary-sum-cpu``; the coaching note declares none
  and is reported as ``n/a``) applied to the **slowest single call** of the chunk
  (T40: the budgets were calibrated on one call, so one call is the unit they are
  compared with; the calls of a layer that honestly splits its input are timed
  one by one, while the chunk's own total stays in the report, so the price of
  splitting is visible instead of being absorbed by the budget). Inside the day
  route the fold-back call of a split day is graded by a budget of its own (T41,
  ``daily_merge`` = 25 s): it is a different call from a part — another template
  and an output the size of the whole day — and the harness tells it from a part
  by the prompt version the layer stamped on the request, never by the call's
  position in the run (a regenerated day makes eight calls, not four).

Two counts, never one (T18): a chunk can fail because its *content* broke an
invariant or because it missed its latency budget, and the report names the two
separately — the manifest threshold applies to the invariant count alone, while a
latency overrun still fails the run. A cold model load is not a content defect,
and reading the two as one number is what once made a red run look like a
checklist violation.

A chunk whose layer is switched off in the configuration is reported as
``SKIP:profile_off`` and excluded from both counts — the product treats an absent
phase-2 profile as «the layer is off» (§9.8), not as an error, and the harness
follows instead of aborting the run (T19). A run that measured nothing is still a
failure: verifying nothing is not the same as verifying that everything is fine.

A chunk whose layer fails **deterministically** — an unusable input, a structured
completion that does not validate, a day that cannot be split into its context
window — is reported as the failure of that error, named by its class
(``pipeline_error:PipelineInputError``), and the run goes on to the remaining
chunks (T32). Before that, the first such chunk ended the whole run with a
traceback and every chunk after it went unmeasured. A refusal by the invariant
gate is not folded into this bucket: it keeps grading the chunk by the checks the
gate named (T17b). A ``HarnessError`` — the harness itself cannot run — stays
fatal on purpose: it says the measurement is broken, not the layer. Transient
provider failures (timeouts, 5xx) also stay fatal: they are properties of the
moment, not of the chunk, so a run that met one is not comparable with another.

Deliberate limits (stated, not hidden)
======================================

* ``no_new_facts`` is exact for digits, numeral words and proper names.
  Free-form content words are **not** machine-checkable without a semantic
  model: a paraphrase and an invention look identical to a lexical check. The
  harness reports ungrounded long content tokens as *diagnostics* (a hint for a
  human), never as a pass/fail criterion — a fabricated "green" check would be
  worse than an honest gap.
* Grounding compares against the **record content** of the input: entry texts,
  ``facts``, ``themes``, ``open_questions``, ``rollup``. Identifiers, audio paths
  and the moment/anchor fields (``ts``, ``date``, ``week_id``, ``daily_ids``,
  ``summary_id``, ``local_date``, ``local_tz``, ``created_at``) are not facts
  about the diary and do not ground a number of the output (T32; before that the
  digits of ``2026-09-23`` grounded a claimed «до 23:00»).
* The context reserve is an estimate (2.5 characters per token) plus the
  profile's generation cap; the live run reports the measured ``input_tokens``
  next to it so the estimate can be re-checked against reality.

Privacy: the harness never writes anything into the repository, never prints
generated text and never logs request content; violations are reported by check
name only. The golden set is synthetic (AGENTS.md §7).

Usage::

    .venv/bin/python eval/run.py                       # CI mode, all chunks
    .venv/bin/python eval/run.py --live                # real provider
    .venv/bin/python eval/run.py --chunk day-01-quiet-monday
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from alembic import command
from alembic.config import Config as AlembicConfig
from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlalchemy.orm import Session

from diary.core.models_config import ModelsConfig, OpenAICompatibleProfile, load_models_config
from diary.core.settings import PROJECT_ROOT, get_settings
from diary.db.models import (
    ArtifactLayer,
    Message,
    MessageKind,
    PipelineType,
    RunStatus,
    TranscriptionStatus,
)
from diary.db.repositories import (
    AudioFileRepository,
    ProcessingRunRepository,
    TranscriptionRepository,
)
from diary.db.results import list_layer_results, save_layer_result
from diary.db.session import create_engine, create_session_factory
from diary.domain.contracts import (
    CoachNote,
    DailySummary,
    RawEntry,
    SttCorrection,
    WeeklySummary,
)
from diary.domain.contracts.golden import (
    ADVICE_EVALUATION_LEXEMES,
    PROFILE_MAX_TOKENS,
    CoachInvariants,
    DailyInvariants,
    GoldenChunk,
    GoldenManifest,
    WeekInvariants,
    canonical_input_json,
    chunk_path,
    coach_disclaimer,
    estimate_tokens,
    generated_text_fields,
    input_payload,
    load_chunk,
    load_manifest,
)
from diary.domain.contracts.invariants import (
    InputFacts,
    InvariantViolation,
    advice_and_judgement_violations,
    causal_claim_violations,
    generated_values,
    input_facts_from_models,
    introduced_lexemes,
    new_fact_violations,
    numbers,
    proper_names,
)
from diary.domain.contracts.nli import (
    CLAIM_FIELDS,
    NliError,
    NliJudge,
    NliMetrics,
    claims_for,
    judge_claims,
    metrics_for,
)
from diary.domain.llm.errors import LLMProviderUnavailableError, LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse, LLMUsage
from diary.domain.llm.providers import LLMProvider, create_llm_provider
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.nli.onnx_judge import OnnxNliJudge
from diary.domain.pipelines.base import PipelineInvariantError, PipelinePermanentError
from diary.domain.pipelines.coach.pipeline import (
    PROFILE_NAME as COACH_PROFILE_NAME,
)
from diary.domain.pipelines.coach.pipeline import (
    PROMPT_PATH as COACH_PROMPT_PATH,
)
from diary.domain.pipelines.coach.service import coach_note
from diary.domain.pipelines.nli_guard import default_nli_judge
from diary.domain.pipelines.stt_fix.pipeline import (
    PROFILE_NAME as STT_PROFILE_NAME,
)
from diary.domain.pipelines.stt_fix.pipeline import (
    PROMPT_PATH as STT_PROMPT_PATH,
)
from diary.domain.pipelines.stt_fix.service import correct_stt
from diary.domain.pipelines.sum_daily.day_input import collect_day_entries
from diary.domain.pipelines.sum_daily.pipeline import (
    ENTRIES_HEADER as SUM_ENTRIES_HEADER,
)
from diary.domain.pipelines.sum_daily.pipeline import (
    MERGE_PROMPT_VERSION as SUM_MERGE_PROMPT_VERSION,
)
from diary.domain.pipelines.sum_daily.pipeline import (
    PARTS_HEADER as SUM_PARTS_HEADER,
)
from diary.domain.pipelines.sum_daily.pipeline import (
    PROFILE_NAME as SUM_PROFILE_NAME,
)
from diary.domain.pipelines.sum_daily.pipeline import (
    PROMPT_PATH as SUM_PROMPT_PATH,
)
from diary.domain.pipelines.sum_daily.pipeline import SumDailyPipeline
from diary.domain.pipelines.sum_daily.service import sum_daily
from diary.domain.pipelines.sum_weekly.pipeline import (
    PROFILE_NAME as WEEK_PROFILE_NAME,
)
from diary.domain.pipelines.sum_weekly.pipeline import (
    PROMPT_PATH as WEEK_PROMPT_PATH,
)
from diary.domain.pipelines.sum_weekly.service import sum_weekly

#: Tracked, secret-free configuration used by the CI mode (no endpoint is ever
#: contacted there and the mock provider is injected, so ``api_key_env`` is
#: never resolved).
CI_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.example.yaml"

#: Raw source tables whose rows must be identical after a run.
RAW_TABLES: tuple[str, ...] = ("messages", "audio_files", "transcriptions")

#: Phase-2 profile each golden category runs under, taken from the layer
#: packages so that a rename cannot drift silently **and** so that the harness
#: measures the profile the layer really uses. Finding F6 was exactly this kind
#: of drift: ``week`` pointed at the day profile, so the hybrid routing of T13b
#: and the CPU tag ``diary-sum-cpu`` had no regression coverage at all (T19).
PROFILE_NAME_BY_CATEGORY: dict[str, str] = {
    "stt_error": STT_PROFILE_NAME,
    "day": SUM_PROFILE_NAME,
    "week": WEEK_PROFILE_NAME,
    "wellbeing": COACH_PROFILE_NAME,
}

#: Versioned prompt template of each category, again taken from the layer
#: packages. The harness estimates the prompt overhead from these files at run
#: time (T20) instead of trusting a hand-maintained constant that a prompt can
#: outgrow unnoticed.
PROMPT_PATH_BY_CATEGORY: dict[str, Path] = {
    "stt_error": STT_PROMPT_PATH,
    "day": SUM_PROMPT_PATH,
    "week": WEEK_PROMPT_PATH,
    "wellbeing": COACH_PROMPT_PATH,
}

#: Check the runner itself adds when a chunk misses its latency budget. It is
#: not a content check: T18 keeps the two counts apart, because a cold model load
#: and a broken invariant are different events that used to share one number.
#: Since T40 the figure compared with the budget is the **slowest single call**
#: of the chunk, not the sum of its calls (see :func:`_apply_latency_budget`);
#: since T41 the day layer's fold-back call is compared with a budget of its own.
LATENCY_CHECK = "latency_budget"

#: Template version each layer stamps on its **fold-back** call (T41).
#:
#: A split day (T23, ``ADR-016``) is summarized part by part with the ordinary
#: template and folded back with another one: the day layer's merge request is
#: built by :meth:`~diary.domain.pipelines.sum_daily.pipeline.SumDailyPipeline.
#: build_merge_request`, which stamps ``prompt_version`` with
#: ``MERGE_PROMPT_VERSION`` ("v2", ``prompts/sum_daily/v2.md``), while its part
#: requests carry ``PROMPT_VERSION`` ("v1"). The harness reads the **same
#: constant the layer writes with**, so a version bump cannot leave the
#: classification behind.
#:
#: Why the request and not its place in the run: a day whose facts the NLI gate
#: rejects is regenerated (``generate_with_invariants``), and a regenerated day
#: makes **eight** calls — parts, fold-back, parts, fold-back (measured live in
#: T40: 8 calls, the slowest 17.20 and 19.74 s). «The last call» would then grade
#: the second part's tail as if it were the fold-back and miss the first one
#: entirely; «every N-th call» needs N *and* the assumption that no other step
#: ever gets retried. The prompt version is set by construction, travels with the
#: request, and is already recorded as provenance — one source, no heuristics.
MERGE_PROMPT_VERSION_BY_CATEGORY: dict[str, str] = {"day": SUM_MERGE_PROMPT_VERSION}

#: Pipeline type recorded on the processing run of each category.
PIPELINE_TYPE_BY_CATEGORY: dict[str, PipelineType] = {
    "stt_error": PipelineType.stt_corrected,
    "day": PipelineType.daily_summary,
    "week": PipelineType.weekly_summary,
    "wellbeing": PipelineType.coach_note,
}

#: Layer whose result the run appends, per category (read back by the harness).
LAYER_BY_CATEGORY: dict[str, ArtifactLayer] = {
    "stt_error": ArtifactLayer.stt_corrected,
    "day": ArtifactLayer.daily_summary,
    "week": ArtifactLayer.weekly_summary,
    "wellbeing": ArtifactLayer.coach_note,
}

#: Contract model of each golden contract (used for ``check_output_contract``;
#: these models are what generates ``schemas/*.json``, so validating through
#: them is validating the contract schema).
CONTRACT_MODELS: dict[str, type[BaseModel]] = {
    "stt_corrected": SttCorrection,
    "daily_summary": DailySummary,
    "weekly_summary": WeeklySummary,
    "coach_note": CoachNote,
}

#: Invariant field of the manifest → the check name that implements it. Every
#: field of every ``GoldenInvariants`` subclass must appear here: a test asserts
#: it, so an invariant can never be declared and then silently skipped.
CHECK_BY_INVARIANT_FIELD: dict[str, str] = {
    "check_output_contract": "output_contract",
    "check_input_unchanged": "input_unchanged",
    "forbidden_lexemes": "forbidden_lexemes",
    "no_advice_or_judgement": "advice_and_judgement_absent",
    "no_causal_claims": "causal_claims_absent",
    "no_new_facts": "no_new_facts",
    "names_and_numbers_grounded_in_input": "names_and_numbers_grounded",
    "uncertain_values_are_verbatim_substrings_of_text": "uncertain_substrings",
    "expected_date": "expected_date",
    "expected_energy": "expected_energy",
    "facts_non_empty": "facts_non_empty",
    "numbers_grounded_in_input": "numbers_grounded",
    "expected_week_id": "expected_week_id",
    "daily_ids_subset_of_input": "daily_ids_subset",
    "rollup_non_empty": "rollup_non_empty",
    "observations_count": "observations_count",
    "hypotheses_count": "hypotheses_count",
    "experiment_horizon_days": "experiment_horizon_days",
    "disclaimer_equals_coach_disclaimer": "disclaimer_equals",
}

#: Checks every chunk runs, whatever its kind.
GENERIC_CHECKS: tuple[str, ...] = (
    "output_contract",
    "input_unchanged",
    "forbidden_lexemes",
    "advice_and_judgement_absent",
    "causal_claims_absent",
    "no_new_facts",
)

#: Additional checks of each contract kind. A kind absent from this mapping is
#: a harness error: an unknown ``kind`` is never skipped silently.
KIND_CHECKS: dict[str, tuple[str, ...]] = {
    "stt_corrected": ("uncertain_substrings", "names_and_numbers_grounded"),
    "daily_summary": ("expected_date", "expected_energy", "facts_non_empty", "numbers_grounded"),
    "weekly_summary": ("expected_week_id", "daily_ids_subset", "rollup_non_empty"),
    "coach_note": (
        "observations_count",
        "hypotheses_count",
        "experiment_horizon_days",
        "disclaimer_equals",
    ),
}

#: Extra check names produced by the runner itself because they need a
#: measurement the pure payload checks do not have.
EXTRA_CHECKS: tuple[str, ...] = (
    "latency_budget",
    "measured_context_budget",
    "raw_rows_unchanged",
    "context_budget",
    "expected_parts",
)

_WORD_RE = re.compile(r"[а-яёa-z]+")

#: Function words ignored by the (informational) content-word diagnostics.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "которая",
        "которого",
        "которой",
        "которые",
        "который",
        "возможно",
        "кажется",
        "например",
        "немного",
        "несколько",
        "оказалось",
        "остается",
        "остаётся",
        "особенно",
        "получилось",
        "похоже",
        "сегодня",
    }
)

#: Minimum length of a content word the diagnostics look at, and the stem length
#: used for its (informational) grounding test.
_DIAGNOSTIC_MIN_LEN = 6
_DIAGNOSTIC_STEM_LEN = 5


class HarnessError(Exception):
    """The harness itself cannot run: unknown kind, unchecked invariant, bad set."""


#: Check-name prefix the harness reports when a layer failed permanently without
#: the invariant gate naming any check (T32). The class name of the error is
#: appended — ``pipeline_error:PipelineInputError`` — so the report says *which*
#: deterministic failure hit the chunk while the error's own message stays out of
#: the report, exactly like every other ``Violation.detail``.
PIPELINE_ERROR_CHECK = "pipeline_error"

#: Check the runner raises when a chunk declares that its layer feeds the input in
#: several calls and the layer does not do it (T35). The declaration lives in the
#: chunk's ``budget.chunks_planned``; grading it is what keeps the manifest
#: expectation from being prose that nothing verifies.
PARTS_CHECK = "expected_parts"


@dataclass(frozen=True)
class Violation:
    """One failed check of one chunk.

    ``check`` is the machine name of the check; ``detail`` explains the failure
    for tests and is never printed by the CLI (it can carry generated text).
    """

    check: str
    detail: str


@dataclass
class ChunkOutcome:
    """Everything the harness knows about one chunk after a run.

    Only metrics, chunk identity and check *names* are meant for reporting; the
    generated text stays in the process (and in the throwaway database).
    """

    chunk_id: str
    category: str
    kind: str
    profile_name: str = ""
    skipped_reason: str = ""
    #: Wall time of the whole layer run of this chunk (T18) — the price the user
    #: pays, including every call a split input needs and the database work around
    #: them. Since T40 this is what the report prints in the chunk's ``latency``
    #: column, while the **budget** is compared with :attr:`slowest_call_s`.
    latency_s: float = 0.0
    latency_budget_s: float | None = None
    #: Budget of the layer's **fold-back** call of this chunk, when its route
    #: declares one (T41; only the day layer does — ``daily_merge``). ``None``
    #: means every call of the chunk is graded by :attr:`latency_budget_s`, which
    #: is the T40 behaviour and stays the rule for the other routes.
    merge_budget_s: float | None = None
    #: Wall time of every ``complete`` call the layer made, in call order (T40).
    #: A day that the layer honestly splits (T23, ``ADR-016``) makes several calls;
    #: timing them one by one is what lets the route budget stay calibrated on one
    #: call. Empty when nothing was timed — a layer that failed before its first
    #: request — and then :attr:`slowest_call_s` falls back to the chunk's total.
    call_seconds: list[float] = field(default_factory=list)
    #: ``prompt_version`` of each of those calls, in the same order (T41): the
    #: field the layer stamps when it builds the request, which is how the harness
    #: tells a part from the fold-back without guessing from the order. Shorter
    #: than ``call_seconds`` only for a hand-built outcome, and a call without a
    #: version is graded by the route budget like every other call.
    call_prompt_versions: list[str | None] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    #: Number of calls the layer made, as the harness counted them (T40 — the count
    #: is ``len(call_seconds)``, not a counter a provider reports, so one number has
    #: one source; a call that raised is counted like any other).
    requests: int = 0
    #: The **largest** usage the endpoint reported for one call of this chunk (T35:
    #: a split day makes several calls, and the fold-back is the smallest of them).
    usage: LLMUsage | None = None
    #: NLI metrics of a summary chunk when ``--nli`` asked for them (T24a); the
    #: metrics are content-free (counts, probabilities, claim coordinates).
    nli: NliMetrics | None = None

    @property
    def skipped(self) -> bool:
        """Whether the layer was switched off, so nothing was measured."""
        return bool(self.skipped_reason)

    @property
    def passed(self) -> bool:
        """Whether the chunk passed every check (latency included)."""
        return not self.violations

    @property
    def content_violations(self) -> list[Violation]:
        """Violations about the content or the plumbing, not the clock.

        Everything but the latency budget: contract shape, blacklists, grounding,
        context budget, raw-row immutability. T18 counts these apart from latency
        so that a cold model load can no longer be read as a broken invariant.
        """
        return [item for item in self.violations if item.check != LATENCY_CHECK]

    @property
    def latency_violations(self) -> list[Violation]:
        """Violations of this chunk's latency budget (none when it has no budget)."""
        return [item for item in self.violations if item.check == LATENCY_CHECK]

    @property
    def slowest_call_s(self) -> float:
        """The slowest single call of the chunk (T40) — the figure the row prints.

        The budgets of T21 (5/15/180 s) were calibrated on **one** call, so one call
        is the unit they are graded in: a layer that splits a long input into parts
        is not punished for the number of parts, only for the slowest of them. When
        no call was timed at all — a layer that refused its input before its first
        request — the chunk's own wall time is the only measurement there is, and
        the fallback is deliberately the *total* rather than zero: an unmeasurable
        chunk must not read as a fast one.

        Which budget that figure is graded with is decided per call, not here
        (:meth:`call_budget_s`, T41): a chunk with a fold-back call prints the
        slowest call of either kind, and the report names them separately.
        """
        return max(self.call_seconds) if self.call_seconds else self.latency_s

    def call_is_merge(self, index: int) -> bool:
        """Whether call ``index`` is the layer's **fold-back** call (T41).

        The answer comes from the request itself: the layer stamps the template
        version it built the request from, and the harness compares it with the
        version the same layer uses for its fold-back template
        (:data:`MERGE_PROMPT_VERSION_BY_CATEGORY`). Position in the run is
        deliberately not consulted — a regenerated day interleaves parts and
        fold-backs, so position is exactly the signal that lies here.
        """
        expected = MERGE_PROMPT_VERSION_BY_CATEGORY.get(self.category)
        if expected is None or index >= len(self.call_prompt_versions):
            return False
        return self.call_prompt_versions[index] == expected

    def call_budget_s(self, index: int) -> float | None:
        """The latency budget call ``index`` is graded with (T41).

        A fold-back call is compared with the fold-back budget of its route, every
        other call with the route budget — the same number for the other routes,
        whose layers declare no separate fold-back budget (``None`` falls back to
        the route budget rather than exempting the call).
        """
        if self.call_is_merge(index) and self.merge_budget_s is not None:
            return self.merge_budget_s
        return self.latency_budget_s

    @property
    def merge_call_seconds(self) -> list[float]:
        """Wall time of the fold-back calls of this chunk, in call order (T41)."""
        return [
            seconds for index, seconds in enumerate(self.call_seconds) if self.call_is_merge(index)
        ]

    @property
    def part_call_seconds(self) -> list[float]:
        """Wall time of the calls that are **not** fold-backs (T41).

        For every chunk that splits nothing this is every call it made, so the
        route budget keeps grading exactly what it graded in T40.
        """
        return [
            seconds
            for index, seconds in enumerate(self.call_seconds)
            if not self.call_is_merge(index)
        ]

    @property
    def slowest_part_call_s(self) -> float | None:
        """The slowest non-fold-back call, or ``None`` when the chunk made none."""
        return max(self.part_call_seconds) if self.part_call_seconds else None

    @property
    def slowest_merge_call_s(self) -> float | None:
        """The slowest fold-back call, or ``None`` when the chunk made none."""
        return max(self.merge_call_seconds) if self.merge_call_seconds else None

    @property
    def latency_within_budget(self) -> bool:
        """Whether the chunk stayed inside its route's latency budget."""
        return not self.latency_violations

    @property
    def failed_checks(self) -> list[str]:
        """Names of the failed checks, in first-failure order, without repeats."""
        seen: list[str] = []
        for violation in self.violations:
            if violation.check not in seen:
                seen.append(violation.check)
        return seen


# ---------------------------------------------------------------------------
# Golden set loading (only through diary.domain.contracts.golden)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def golden_manifest() -> GoldenManifest:
    """The repository golden manifest (loaded once per process)."""
    return load_manifest()


def load_golden_set(
    *,
    chunk_ids: Sequence[str] | None = None,
    base_dir: Path | str | None = None,
) -> tuple[GoldenManifest, list[GoldenChunk]]:
    """Load the manifest and the selected chunks, in manifest order.

    An id the manifest does not declare is a harness error — the run must never
    quietly cover fewer chunks than asked for.
    """
    manifest = (
        golden_manifest() if base_dir is None else load_manifest(Path(base_dir) / "manifest.json")
    )
    requested = set(chunk_ids) if chunk_ids else set(manifest.chunks)
    unknown = sorted(requested - set(manifest.chunks))
    if unknown:
        raise HarnessError(f"unknown chunk id(s): {', '.join(unknown)}")
    # Manifest order, whatever order the caller listed the ids in.
    selected = [chunk_id for chunk_id in manifest.chunks if chunk_id in requested]
    chunks = [load_chunk(chunk_path(chunk_id, base_dir=base_dir)) for chunk_id in selected]
    return manifest, chunks


# ---------------------------------------------------------------------------
# Check context and the checks themselves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckContext:
    """Everything a check may look at: the chunk, its validated output and input.

    ``input_json_before`` is the canonical serialization captured *before* the
    run; ``input_text``/``input_numbers`` are the casefolded content text of the
    input and its digit tokens, precomputed once.
    """

    chunk: GoldenChunk
    model: BaseModel
    payload: Mapping[str, object]
    input_json_before: str
    input_text: str
    input_numbers: frozenset[str]


def _validate_output(
    chunk: GoldenChunk, payload: object
) -> tuple[BaseModel | None, list[Violation]]:
    """Validate an output payload against the contract of the chunk."""
    contract = CONTRACT_MODELS.get(chunk.contract)
    if contract is None:
        raise HarnessError(f"no contract model is registered for {chunk.contract!r}")
    if not isinstance(payload, Mapping):
        return None, [Violation("output_contract", "the stored result is not a JSON object")]
    try:
        return contract.model_validate(dict(payload)), []
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"]) or "<root>"
        return None, [Violation("output_contract", f"{location}: {first['msg']}")]


def _check_output_contract(ctx: CheckContext) -> list[Violation]:
    """The output validates against the contract that generates its schema."""
    _, violations = _validate_output(ctx.chunk, ctx.payload)
    return violations


def _check_input_unchanged(ctx: CheckContext) -> list[Violation]:
    """The chunk input is not mutated by the run (raw rows are checked apart)."""
    if canonical_input_json(ctx.chunk.input) != ctx.input_json_before:
        return [Violation("input_unchanged", "the serialized chunk input changed during the run")]
    return []


def _check_forbidden_lexemes(ctx: CheckContext) -> list[Violation]:
    """The chunk blacklist plus the eval-contract advice/evaluation baseline."""
    lexemes = sorted(set(ctx.chunk.invariants.forbidden_lexemes) | set(ADVICE_EVALUATION_LEXEMES))
    violations: list[Violation] = []
    for value in generated_values(ctx.payload, generated_text_fields(ctx.chunk.contract)):
        for hit in introduced_lexemes(ctx.chunk.contract, value, lexemes, ctx.input_text):
            violations.append(
                Violation("forbidden_lexemes", f"forbidden lexeme {hit!r} is present")
            )
    return violations


def _chunk_facts(chunk: GoldenChunk) -> InputFacts:
    """The reduced source of a chunk, as the shared checks expect it (T17)."""
    return input_facts_from_models(chunk.input)


#: Contracts whose claims the NLI instrument checks (T24a: summarisation only —
#: the STT correction has no claims and the coaching note is out of this scope).
NLI_CONTRACTS: frozenset[str] = frozenset(CLAIM_FIELDS)


def nli_judge() -> NliJudge:
    """The judge ``--nli`` uses: the pinned ONNX cross-encoder, built on demand.

    A module-level hook, so tests install a scripted judge and so the instrument
    is only built when the flag asks for it. An instrument that cannot be built
    raises :class:`~diary.domain.nli.onnx_judge.NliUnavailableError`, which the
    runner reports as a **failed run** rather than as an absent check — the
    owner's decision for T24 («если NLI недоступен — Fail»): a requested check
    that silently does nothing is worse than a loud failure.
    """
    return OnnxNliJudge.from_settings()


def _nli_premise(chunk: GoldenChunk) -> str:
    """The text a claim is compared against: the content of the layer's input.

    A layer asserts about the *meaning* of what it was given, so the premise is
    the entry texts of a day or the day rollups of a week — not row identifiers,
    timestamps or file paths. That also keeps the pair inside the model's
    512-token budget for the chunks the set carried before T35, where a truncation
    would silently hide the very claim the check exists for. An empty result is a
    loud error, not «everything is neutral» (T24a DoD 2).

    Named limit for a split day (T35): the premise is the **whole** day, exactly
    what the layer was asked to summarize, so for ``day-04-long-wednesday`` it is
    longer than one pair and the cross-encoder truncates it — that chunk's
    contradiction verdicts rest on the beginning of the day, and the fact that the
    measurement does not cover every sentence is stated here rather than implied.
    The loopback is deliberate: taking the premise from one part instead would
    judge a day summary against a fraction of its source, which is the mistake
    T24a was built to avoid.
    """
    items = chunk.input if isinstance(chunk.input, list) else [chunk.input]
    parts: list[str] = []
    for item in items:
        if isinstance(item, RawEntry):
            parts.append(item.text)
        elif isinstance(item, DailySummary):
            parts.extend(item.facts)
            parts.extend(item.themes)
            parts.extend(item.open_questions)
        elif isinstance(item, WeeklySummary):
            parts.append(item.rollup)
        else:
            raise HarnessError(f"chunk {chunk.id} carries an input the NLI premise cannot read")
    return "\n".join(part.strip() for part in parts if part.strip())


def _nli_metrics(
    chunk: GoldenChunk, payload: Mapping[str, object], judge: NliJudge
) -> NliMetrics | None:
    """NLI metrics of a summary chunk, or ``None`` for contracts it does not cover."""
    if chunk.contract not in NLI_CONTRACTS:
        return None
    claims = claims_for(chunk.contract, payload)
    return metrics_for(judge_claims(_nli_premise(chunk), claims, judge))


def _ctx_facts(ctx: CheckContext) -> InputFacts:
    """The reduced source carried by a check context."""
    return InputFacts(text=ctx.input_text, numbers=ctx.input_numbers)


def _shared_violations(found: Sequence[InvariantViolation]) -> list[Violation]:
    """Adapt shared invariant violations to the harness ``Violation`` type."""
    return [Violation(item.check, item.detail) for item in found]


def _check_advice_and_judgement_absent(ctx: CheckContext) -> list[Violation]:
    """No advice, praise, evaluation or morality in the generated text fields.

    Delegates to :mod:`diary.domain.contracts.invariants` — the implementation
    the runtime gate enforces (T17) — so the harness and production cannot
    drift apart.
    """
    return _shared_violations(
        advice_and_judgement_violations(ctx.chunk.contract, ctx.payload, _ctx_facts(ctx))
    )


def _check_causal_claims_absent(ctx: CheckContext) -> list[Violation]:
    """No causal connector in the assertive generated text fields."""
    return _shared_violations(
        causal_claim_violations(ctx.chunk.contract, ctx.payload, _ctx_facts(ctx))
    )


def _check_no_new_facts(ctx: CheckContext) -> list[Violation]:
    """Grounding of digits, numeral words and proper names in the input.

    Exact for the three classes it covers; free-form content words are left to
    the diagnostics, because a lexical check cannot tell a paraphrase from an
    invention (see the module docstring).
    """
    return _shared_violations(new_fact_violations(ctx.chunk.contract, ctx.payload, _ctx_facts(ctx)))


def _check_uncertain_substrings(ctx: CheckContext) -> list[Violation]:
    """Every uncertain entry is a verbatim substring of the corrected text."""
    assert isinstance(ctx.model, SttCorrection)  # guaranteed by the kind dispatch
    if any(fragment not in ctx.model.text for fragment in ctx.model.uncertain):
        return [Violation("uncertain_substrings", "an uncertain entry is not a substring of text")]
    return []


def _check_names_and_numbers_grounded(ctx: CheckContext) -> list[Violation]:
    """The strict STT invariant: names and numbers come from the source."""
    assert isinstance(ctx.model, SttCorrection)
    violations: list[Violation] = []
    for digit in sorted(numbers(ctx.model.text)):
        if digit not in ctx.input_numbers:
            violations.append(
                Violation("names_and_numbers_grounded", f"number {digit!r} is not in the input")
            )
    for name in proper_names(ctx.model.text):
        if name.casefold() not in ctx.input_text:
            violations.append(
                Violation("names_and_numbers_grounded", f"proper name {name!r} is not in the input")
            )
    return violations


def _check_expected_date(ctx: CheckContext) -> list[Violation]:
    """The summary carries the local day of its entries (its anchor)."""
    assert isinstance(ctx.model, DailySummary)
    assert isinstance(ctx.chunk.invariants, DailyInvariants)
    expected: date = ctx.chunk.invariants.expected_date
    if ctx.model.date != expected:
        return [
            Violation("expected_date", f"the summary is dated {ctx.model.date}, not {expected}")
        ]
    return []


def _check_expected_energy(ctx: CheckContext) -> list[Violation]:
    """``energy`` equals the expectation, ``None`` included as an expectation."""
    assert isinstance(ctx.model, DailySummary)
    assert isinstance(ctx.chunk.invariants, DailyInvariants)
    expected = ctx.chunk.invariants.expected_energy
    if ctx.model.energy != expected:
        return [
            Violation(
                "expected_energy",
                f"energy is {ctx.model.energy!r} where the source expects {expected!r}",
            )
        ]
    return []


def _check_facts_non_empty(ctx: CheckContext) -> list[Violation]:
    """``facts`` is never empty."""
    assert isinstance(ctx.model, DailySummary)
    if not ctx.model.facts:
        return [Violation("facts_non_empty", "facts is empty")]
    return []


def _check_numbers_grounded(ctx: CheckContext) -> list[Violation]:
    """Every number of the day summary occurs in the input."""
    assert isinstance(ctx.model, DailySummary)
    violations: list[Violation] = []
    for value in [*ctx.model.facts, *ctx.model.themes, *ctx.model.open_questions]:
        for digit in sorted(numbers(value)):
            if digit not in ctx.input_numbers:
                violations.append(
                    Violation("numbers_grounded", f"number {digit!r} does not occur in the input")
                )
    return violations


def _check_expected_week_id(ctx: CheckContext) -> list[Violation]:
    """The rollup carries the ISO week of its days (its anchor)."""
    assert isinstance(ctx.model, WeeklySummary)
    assert isinstance(ctx.chunk.invariants, WeekInvariants)
    expected = ctx.chunk.invariants.expected_week_id
    if ctx.model.week_id != expected:
        return [
            Violation("expected_week_id", f"the rollup carries {ctx.model.week_id}, not {expected}")
        ]
    return []


def _check_daily_ids_subset(ctx: CheckContext) -> list[Violation]:
    """``daily_ids`` is a subset of the days actually fed to the layer."""
    assert isinstance(ctx.model, WeeklySummary)
    declared = {item.date.isoformat() for item in ctx.chunk.input if isinstance(item, DailySummary)}
    if set(ctx.model.daily_ids) - declared:
        return [Violation("daily_ids_subset", "daily_ids contain days that were not fed")]
    return []


def _check_rollup_non_empty(ctx: CheckContext) -> list[Violation]:
    """``rollup`` is never empty."""
    assert isinstance(ctx.model, WeeklySummary)
    if not ctx.model.rollup.strip():
        return [Violation("rollup_non_empty", "rollup is empty")]
    return []


def _check_observations_count(ctx: CheckContext) -> list[Violation]:
    """Exactly the declared number of observations."""
    assert isinstance(ctx.model, CoachNote)
    assert isinstance(ctx.chunk.invariants, CoachInvariants)
    expected = ctx.chunk.invariants.observations_count
    if len(ctx.model.observations) != expected:
        return [
            Violation(
                "observations_count",
                f"{len(ctx.model.observations)} observations, {expected} expected",
            )
        ]
    return []


def _check_hypotheses_count(ctx: CheckContext) -> list[Violation]:
    """Exactly the declared number of hypotheses."""
    assert isinstance(ctx.model, CoachNote)
    assert isinstance(ctx.chunk.invariants, CoachInvariants)
    expected = ctx.chunk.invariants.hypotheses_count
    if len(ctx.model.hypotheses) != expected:
        return [
            Violation(
                "hypotheses_count",
                f"{len(ctx.model.hypotheses)} hypotheses, {expected} expected",
            )
        ]
    return []


def _check_experiment_horizon_days(ctx: CheckContext) -> list[Violation]:
    """The experiment states its declared horizon explicitly."""
    assert isinstance(ctx.model, CoachNote)
    assert isinstance(ctx.chunk.invariants, CoachInvariants)
    expected = ctx.chunk.invariants.experiment_horizon_days
    if str(expected) not in ctx.model.experiment_7d:
        return [
            Violation(
                "experiment_horizon_days",
                f"the experiment does not state a {expected}-day horizon",
            )
        ]
    return []


def _check_disclaimer_equals(ctx: CheckContext) -> list[Violation]:
    """The disclaimer equals the layer-5 constant verbatim."""
    assert isinstance(ctx.model, CoachNote)
    if ctx.model.disclaimer != coach_disclaimer():
        return [Violation("disclaimer_equals", "the disclaimer is not the layer-5 constant")]
    return []


#: Every implemented check, by name. A test asserts that no name of
#: :data:`CHECK_BY_INVARIANT_FIELD` is missing here.
CHECKERS: dict[str, Callable[[CheckContext], list[Violation]]] = {
    "output_contract": _check_output_contract,
    "input_unchanged": _check_input_unchanged,
    "forbidden_lexemes": _check_forbidden_lexemes,
    "advice_and_judgement_absent": _check_advice_and_judgement_absent,
    "causal_claims_absent": _check_causal_claims_absent,
    "no_new_facts": _check_no_new_facts,
    "uncertain_substrings": _check_uncertain_substrings,
    "names_and_numbers_grounded": _check_names_and_numbers_grounded,
    "expected_date": _check_expected_date,
    "expected_energy": _check_expected_energy,
    "facts_non_empty": _check_facts_non_empty,
    "numbers_grounded": _check_numbers_grounded,
    "expected_week_id": _check_expected_week_id,
    "daily_ids_subset": _check_daily_ids_subset,
    "rollup_non_empty": _check_rollup_non_empty,
    "observations_count": _check_observations_count,
    "hypotheses_count": _check_hypotheses_count,
    "experiment_horizon_days": _check_experiment_horizon_days,
    "disclaimer_equals": _check_disclaimer_equals,
}


def content_word_diagnostics(ctx: CheckContext) -> list[str]:
    """Long content tokens of the output that no input text carries.

    Informational only: it is the honest answer to "can invented content be
    checked?" — a lexical stem test cannot separate a legitimate paraphrase from
    an invented fact, so these tokens are reported as a hint for a human and are
    **not** a pass/fail criterion.
    """
    found: set[str] = set()
    for value in generated_values(ctx.payload, generated_text_fields(ctx.chunk.contract)):
        for token in _WORD_RE.findall(value.casefold()):
            if len(token) < _DIAGNOSTIC_MIN_LEN or token in _STOPWORDS:
                continue
            if token[:_DIAGNOSTIC_STEM_LEN] in ctx.input_text:
                continue
            found.add(token)
    return sorted(found)


def check_budget(chunk: GoldenChunk, *, profile: OpenAICompatibleProfile) -> list[Violation]:
    """The context budget of every call the layer will make (T20; T35).

    The reserve is the layer's versioned prompt — estimated from the template the
    layer itself loads, never from a hand-maintained constant (T20) — plus the
    generation cap of the profile the category runs under.

    The unit of this check is **one call**, not one chunk (T35). A layer that
    honestly splits a long input (``chunks_planned > 1``: the day layer of T23,
    ``ADR-016``) sends several smaller requests, so comparing the whole input with
    the window would report a false ``context_budget`` exactly where the layer did
    the right thing. The check therefore asks the *layer's own* chunker for the
    parts (one implementation of the split, the discipline of T17) and bounds each
    part's call; a chunk that declares a split its layer does not produce is
    reported as :data:`PARTS_CHECK`, so the declared expectation is checked rather
    than trusted.
    """
    reserve = prompt_tokens_est_for(chunk.category) + PROFILE_MAX_TOKENS[chunk.category]
    if chunk.budget.chunks_planned == 1:
        if chunk.budget.chunk_tokens_est + reserve > chunk.budget.num_ctx:
            return [
                Violation(
                    "context_budget",
                    f"{chunk.budget.chunk_tokens_est} + {reserve} > {chunk.budget.num_ctx}",
                )
            ]
        return []
    try:
        parts = _planned_call_tokens(chunk, profile=profile)
    except PipelinePermanentError:
        # The chunk declares several calls, but its layer cannot split the input
        # at all (one message larger than the window, a generation cap that leaves
        # no room). The layer refuses loudly at run time; the declaration is what
        # this check grades, and it does not hold.
        return [
            Violation(
                PARTS_CHECK,
                "the chunk declares several calls but the layer cannot split its input",
            )
        ]
    violations: list[Violation] = []
    if len(parts) < chunk.budget.chunks_planned:
        violations.append(
            Violation(
                PARTS_CHECK,
                f"the declared split does not happen: {len(parts)} call(s) "
                f"< declared {chunk.budget.chunks_planned}",
            )
        )
    for index, tokens in enumerate(parts, start=1):
        if tokens + reserve > chunk.budget.num_ctx:
            violations.append(
                Violation(
                    "context_budget",
                    f"part {index}: {tokens} + {reserve} > {chunk.budget.num_ctx}",
                )
            )
    return violations


def _planned_call_tokens(chunk: GoldenChunk, *, profile: OpenAICompatibleProfile) -> list[int]:
    """Estimated input tokens of each call the layer builds for ``chunk``.

    The parts come from the layer's own chunker — ``SumDailyPipeline.chunk_entries``,
    the single implementation the runtime uses (T23, ``ADR-016``) — and are measured
    with the eval contract's canonical serialization and coefficient, exactly like
    ``budget.chunk_tokens_est`` measures the whole input, so the two are
    commensurable.

    Only a day chunk is split here: it is the layer that divides an input inside one
    and the same window. A weekly chunk that declares several pieces instead gets the
    larger window (``num_ctx_for``), which its declared ``num_ctx`` already carries,
    so its whole input is its one call. A ``PipelinePermanentError`` from the chunker
    travels up: the check grades it as a broken declaration.
    """
    if chunk.category != "day":
        return [chunk.budget.chunk_tokens_est]
    entries = [item for item in chunk.input if isinstance(item, RawEntry)]
    if len(entries) != len(chunk.input):
        raise HarnessError(f"chunk {chunk.id} declares a day input that is not a raw_entry list")
    parts = SumDailyPipeline().chunk_entries(entries, profile=profile)
    return [estimate_tokens(len(canonical_input_json(part))) for part in parts]


@lru_cache(maxsize=1)
def _prompt_tokens_est() -> dict[str, int]:
    """Estimated tokens of each category's prompt template, read from its file.

    The estimate uses the eval-contract coefficient
    (:data:`~diary.domain.contracts.golden.CHARS_PER_TOKEN_EST`), the same one
    the chunk input is estimated with, so both halves of the envelope are
    commensurable. Reading the file is what makes an understated reserve
    impossible: a prompt can only grow the reserve.
    """
    return {
        category: estimate_tokens(len(path.read_text(encoding="utf-8")))
        for category, path in PROMPT_PATH_BY_CATEGORY.items()
    }


def prompt_tokens_est_for(category: str) -> int:
    """Estimated overhead of the prompt of ``category``, from its template file."""
    try:
        return _prompt_tokens_est()[category]
    except KeyError as exc:
        raise HarnessError(f"no prompt template is mapped to category {category!r}") from exc


# ---------------------------------------------------------------------------
# Running one chunk through the real layers
# ---------------------------------------------------------------------------


def _upgrade_database(database_url: str) -> None:
    """Create a fresh schema through the Alembic migrations (AGENTS.md §9)."""
    config = AlembicConfig()
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


@contextmanager
def temp_database() -> Iterator[Session]:
    """A migrated throwaway SQLite database in the system temp directory."""
    with tempfile.TemporaryDirectory(prefix="diary-eval-") as directory:
        url = f"sqlite:///{(Path(directory) / 'eval.db').as_posix()}"
        _upgrade_database(url)
        engine = create_engine(url)
        try:
            with create_session_factory(engine)() as session:
                yield session
        finally:
            engine.dispose()


def _seed_entry(session: Session, entry: RawEntry) -> None:
    """Insert the raw rows a chunk declares as its input.

    The golden set pins the identity of every message (``RawEntry.id``), and the
    STT layer anchors its result to exactly that id, while
    ``MessageRepository.create`` generates a fresh id — so the message row is
    built from the model directly. Audio metadata and the transcription go
    through the repositories.
    """
    is_voice = entry.source == "voice"
    message = Message(
        id=entry.id,
        kind=MessageKind.audio if is_voice else MessageKind.text,
        body_text="" if is_voice else entry.text,
        created_at=entry.ts,
    )
    session.add(message)
    session.flush()
    if not is_voice:
        return
    assert entry.audio_path is not None  # the raw_entry contract guarantees it
    audio_file = AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path=entry.audio_path,
        size_bytes=0,
        sha256="0" * 64,
    )
    TranscriptionRepository(session).create(
        audio_file_id=audio_file.id,
        raw_text=entry.text,
        status=TranscriptionStatus.succeeded,
    )


def _seed_chunk_input(session: Session, chunk: GoldenChunk) -> None:
    """Seed the raw rows of an STT/day chunk (weeks and notes need none)."""
    if chunk.category not in {"stt_error", "day"}:
        return
    for item in chunk.input:
        if not isinstance(item, RawEntry):
            raise HarnessError(f"chunk {chunk.id} declares a non-raw_entry input")
        _seed_entry(session, item)
    session.flush()


def _raw_state(session: Session) -> tuple[tuple[tuple[object, ...], ...], ...]:
    """Every row of every raw source table, as raw values."""
    state: list[tuple[tuple[object, ...], ...]] = []
    for table in RAW_TABLES:
        rows = session.execute(text(f"SELECT * FROM {table} ORDER BY id")).all()  # noqa: S608
        state.append(tuple(tuple(value for value in row) for row in rows))
    return tuple(state)


def _collected_day_matches(session: Session, chunk: GoldenChunk) -> bool:
    """Whether the seeded day really reads back as the chunk's declared input."""
    assert isinstance(chunk.invariants, DailyInvariants)
    expected = input_payload(chunk.input)
    actual = input_payload(
        collect_day_entries(
            session,
            local_date=chunk.invariants.expected_date,
            tz=ZoneInfo(get_settings().timezone),
        )
    )
    if not isinstance(expected, list) or not isinstance(actual, list):
        return False
    if len(expected) != len(actual):
        return False
    for left, right in zip(expected, actual, strict=True):
        if left.get("id") != right.get("id") or left.get("source") != right.get("source"):
            return False
        if left.get("text") != right.get("text"):
            return False
        if left.get("audio_path") != right.get("audio_path"):
            return False
    return True


def _single_raw_entry(chunk: GoldenChunk) -> RawEntry:
    """The one ``raw_entry`` an STT chunk corrects (a loud failure otherwise)."""
    items = chunk.input if isinstance(chunk.input, list) else []
    entries = [item for item in items if isinstance(item, RawEntry)]
    if len(entries) != 1:
        raise HarnessError(f"chunk {chunk.id} must declare exactly one raw_entry input")
    return entries[0]


def _run_layer(
    session: Session,
    chunk: GoldenChunk,
    *,
    profile: OpenAICompatibleProfile,
    provider: LLMProvider,
    gate_judge: NliJudge | None = None,
) -> None:
    """Run one chunk through the service of its layer (the run itself)."""
    run = ProcessingRunRepository(session).create(
        pipeline_type=PIPELINE_TYPE_BY_CATEGORY[chunk.category],
        status=RunStatus.succeeded,
        profile_name=PROFILE_NAME_BY_CATEGORY[chunk.category],
    )
    run_id = run.id
    if chunk.category == "stt_error":
        entry = _single_raw_entry(chunk)
        correct_stt(
            session, entry=entry, profile=profile, provider=provider, processing_run_id=run_id
        )
    elif chunk.category == "day":
        assert isinstance(chunk.invariants, DailyInvariants)
        sum_daily(
            session,
            local_date=chunk.invariants.expected_date,
            profile=profile,
            provider=provider,
            processing_run_id=run_id,
            judge=gate_judge,
        )
    elif chunk.category == "week":
        assert isinstance(chunk.invariants, WeekInvariants)
        summaries = list(chunk.input)
        if not all(isinstance(item, DailySummary) for item in summaries):
            raise HarnessError(f"chunk {chunk.id} declares a non-daily_summary input")
        sum_weekly(
            session,
            week_id=chunk.invariants.expected_week_id,
            profile=profile,
            provider=provider,
            summaries=summaries,  # type: ignore[arg-type]
            processing_run_id=run_id,
        )
    elif chunk.category == "wellbeing":
        upstream = chunk.input
        if not isinstance(upstream, WeeklySummary):
            raise HarnessError(f"chunk {chunk.id} declares a non-weekly_summary input")
        save_layer_result(
            session,
            layer=ArtifactLayer.weekly_summary,
            week_id=upstream.week_id,
            result=upstream,
            processing_run_id=run_id,
        )
        coach_note(
            session,
            week_id=upstream.week_id,
            provider=provider,
            profile=profile,
            processing_run_id=run_id,
        )
    else:
        raise HarnessError(f"chunk {chunk.id} has no runner for category {chunk.category!r}")


def _latest_result_payload(session: Session, chunk: GoldenChunk) -> Mapping[str, object]:
    """The result the run just appended, read back from the store (task T08)."""
    rows = list_layer_results(session, layer=LAYER_BY_CATEGORY[chunk.category])
    if not rows:
        raise HarnessError(f"chunk {chunk.id} produced no stored result")
    payload = rows[-1].result_json
    if not isinstance(payload, Mapping):
        raise HarnessError(f"chunk {chunk.id} stored a result without an inline payload")
    return payload


# ---------------------------------------------------------------------------
# Evaluating one chunk
# ---------------------------------------------------------------------------


def _clock() -> float:
    """The monotonic clock every measurement of a run reads.

    ``time.perf_counter`` in a real run; a **test seam** for the latency tests,
    which script a run of known duration (T40) instead of sleeping. It is one
    function so that the chunk's total and the times of its individual calls are
    read off the same clock and cannot drift apart.
    """
    return time.perf_counter()


class TimedProvider:
    """Wraps a provider and records the wall time of **every** call (T40).

    The unit of the latency budget is one call (the budgets of T21 were calibrated
    on one call), so the harness has to know how long each call took — not only how
    long the layer as a whole took. The measurement lives here rather than in the
    live provider wrapper on purpose: it is the harness's own measurement of the
    layer it runs, it applies to the CI mock as well, and it cannot be lost by a
    factory that forgets to wrap a provider.

    A call that raises is timed too: it cost the same wall time, and the layer may
    well retry it (``generate_with_invariants``) or turn it into a named failure.
    Metrics only — the request and the response are passed through untouched.

    Beside the time, the ``prompt_version`` the layer stamped on the request is
    recorded (T41): that is what tells a part from the fold-back without inferring
    it from the order of the calls (see
    :data:`MERGE_PROMPT_VERSION_BY_CATEGORY`). The version is provenance, not
    content — it is the template's number, never anything the request carries.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.call_seconds: list[float] = []
        self.call_prompt_versions: list[str | None] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = _clock()
        try:
            return self._provider.complete(request)
        finally:
            self.call_seconds.append(_clock() - started)
            self.call_prompt_versions.append(request.prompt_version)

    @property
    def usages(self) -> list[LLMUsage] | None:
        """The wrapped provider's reported usages, when it reports any."""
        usages = getattr(self._provider, "usages", None)
        if not isinstance(usages, list):
            return None
        return [usage for usage in usages if isinstance(usage, LLMUsage)]

    @property
    def last_usage(self) -> LLMUsage | None:
        """The wrapped provider's last reported usage, when it reports one."""
        usage = getattr(self._provider, "last_usage", None)
        return usage if isinstance(usage, LLMUsage) else None


def evaluate_chunk(
    chunk: GoldenChunk,
    *,
    provider: LLMProvider,
    models_config: ModelsConfig,
    profile: OpenAICompatibleProfile | None = None,
    judge: NliJudge | None = None,
    gate_judge: NliJudge | None = None,
) -> ChunkOutcome:
    """Run one chunk on a throwaway database and check it.

    The provider is injected (a scripted mock in CI mode, the real
    OpenAI-compatible adapter in a live run); the layers, the store and the
    migrations are the real ones. Latency is measured around the layer call
    itself, without the database setup: the chunk's total lands in
    :attr:`ChunkOutcome.latency_s` and every individual call in
    :attr:`ChunkOutcome.call_seconds`, and the manifest budget of the category is
    compared with the **slowest call** (T40 — the unit the budgets of T21 were
    calibrated in). The provider handed to the layer is a :class:`TimedProvider`
    around the injected one, so the per-call timing does not depend on what the
    caller's provider chooses to report.

    ``judge`` is the NLI instrument of ``--nli`` (T24a): when it is given, the
    claims of a summary chunk are checked against the canonical input the layer
    saw, and the content-free metrics land in :attr:`ChunkOutcome.nli`. The
    metrics never influence the invariant or latency counts — the measurement is a
    measurement.

    ``gate_judge`` arms the **layer's own** NLI gate (T24b). A live run passes the
    instrument the product uses, so what the harness measures is the gated layer
    rather than a layer nobody runs in production; the CI mode passes nothing,
    because it has neither the optional extra nor the weights (documented in
    ``docs/nli-validation-analysis.md`` §9).
    """
    if chunk.invariants.kind not in KIND_CHECKS:
        raise HarnessError(
            f"chunk {chunk.id} declares invariant kind {chunk.invariants.kind!r}, "
            "which no checker implements"
        )
    try:
        resolved_profile = (
            profile
            if profile is not None
            else resolve_phase2_profile(models_config, PROFILE_NAME_BY_CATEGORY[chunk.category])
        )
    except LLMProviderUnavailableError:
        # The layer is switched off in this configuration, which the product
        # reads as a skip and not as an error (§9.8: an absent phase-2 profile
        # means «the layer is off»). The harness says the same instead of
        # aborting the run with a traceback, and the skip is visible in the
        # report and in the summary counts (T19 DoD 4).
        return _skipped_outcome(chunk)
    outcome = ChunkOutcome(
        chunk_id=chunk.id,
        category=chunk.category,
        kind=chunk.invariants.kind,
        profile_name=PROFILE_NAME_BY_CATEGORY[chunk.category],
        latency_budget_s=_latency_budget_for(chunk),
        merge_budget_s=_merge_latency_budget_for(chunk),
    )
    outcome.violations.extend(check_budget(chunk, profile=resolved_profile))

    input_json_before = canonical_input_json(chunk.input)
    with temp_database() as session:
        _seed_chunk_input(session, chunk)
        if chunk.category == "day" and not _collected_day_matches(session, chunk):
            raise HarnessError(
                f"chunk {chunk.id}: the seeded day does not read back as its declared input"
            )
        raw_state_before = _raw_state(session) if chunk.category in {"stt_error", "day"} else None

        timed = TimedProvider(provider)
        started = _clock()
        refused: PipelineInvariantError | None = None
        permanent: PipelinePermanentError | LLMStructuredOutputError | None = None
        try:
            _run_layer(
                session,
                chunk,
                profile=resolved_profile,
                provider=timed,
                gate_judge=gate_judge,
            )
        except PipelineInvariantError as exc:
            # The layer refused its own output (T17b): the gate found a violated
            # invariant and stored nothing. That is the production behaviour, so
            # the chunk is graded as a failure of exactly those checks — grading
            # a stored artifact here instead would measure something production
            # can no longer produce, which is the drift finding F1 was about.
            refused = exc
        except (PipelinePermanentError, LLMStructuredOutputError) as exc:
            # Any *other* deterministic layer failure (T32): an unusable input,
            # a structured completion that does not validate, a day that cannot
            # be split into its context window. Before T32 the first such chunk
            # ended the whole run with a traceback and the remaining chunks were
            # never measured at all — the partially closed T10b finding. The
            # chunk is graded as a failure of that error and the run continues.
            # `HarnessError` is deliberately **not** caught here: it says the
            # measurement itself is broken, which must not read as a defect of
            # the layer.
            permanent = exc
        outcome.latency_s = _clock() - started
        outcome.call_seconds = list(timed.call_seconds)
        # Which step each call was (T41): read from the request the layer built, so
        # the fold-back is identified by what it is and not by where it happened to
        # land in the run (a regenerated day interleaves parts and fold-backs).
        outcome.call_prompt_versions = list(timed.call_prompt_versions)
        # The call count is the harness's own measurement (T40), not what a provider
        # chooses to report: one source for one number, and it counts a call that
        # raised like any other call the layer made.
        outcome.requests = len(outcome.call_seconds)
        session.commit()

        if raw_state_before is not None and _raw_state(session) != raw_state_before:
            outcome.violations.append(
                Violation("raw_rows_unchanged", "a raw source row changed during the run")
            )
        if permanent is not None:
            outcome.violations.append(
                Violation(
                    f"{PIPELINE_ERROR_CHECK}:{type(permanent).__name__}",
                    "the layer failed permanently, so the chunk is graded as that failure",
                )
            )
            _record_measurements(outcome, chunk, resolved_profile, timed)
            _apply_latency_budget(outcome)
            return outcome
        if refused is not None:
            # Fail-closed here too: a refusal that names no check still failed the
            # chunk. `quality.py` always fills `checks` from a non-empty violation
            # list, so this is a guard against a future caller that raises the
            # error bare — grading it as a pass would hide the refusal entirely.
            named = refused.checks or ("pipeline_invariant_refused",)
            outcome.violations.extend(
                Violation(check, "the layer refused to store its output") for check in named
            )
            _record_measurements(outcome, chunk, resolved_profile, timed)
            _apply_latency_budget(outcome)
            return outcome
        payload = _latest_result_payload(session, chunk)

    model, contract_violations = _validate_output(chunk, payload)
    outcome.violations.extend(contract_violations)
    if model is not None:
        ctx = CheckContext(
            chunk=chunk,
            model=model,
            payload=payload,
            input_json_before=input_json_before,
            input_text=_chunk_facts(chunk).text,
            input_numbers=_chunk_facts(chunk).numbers,
        )
        for name in (*GENERIC_CHECKS, *KIND_CHECKS[chunk.invariants.kind]):
            outcome.violations.extend(CHECKERS[name](ctx))
        outcome.diagnostics = content_word_diagnostics(ctx)

    if judge is not None:
        # Judged for every chunk that produced a payload, including one whose
        # contract check failed: an invalid shape and an unfaithful claim are
        # different defects, and the second is worth measuring either way.
        outcome.nli = _nli_metrics(chunk, payload, judge)

    _record_measurements(outcome, chunk, resolved_profile, timed)
    _apply_latency_budget(outcome)
    return outcome


def _skipped_outcome(chunk: GoldenChunk) -> ChunkOutcome:
    """The outcome of a chunk whose layer is switched off in this configuration.

    An absent phase-2 profile is not an error in this project: the product skips
    the layer honestly (``SkippedStep(profile_off)``, §9.8), and the harness must
    not turn the same configuration into a traceback. The chunk is reported as
    skipped, carries no violations and is excluded from both counts — the skip
    itself is named in the report.
    """
    profile = PROFILE_NAME_BY_CATEGORY[chunk.category]
    return ChunkOutcome(
        chunk_id=chunk.id,
        category=chunk.category,
        kind=chunk.invariants.kind,
        profile_name=profile,
        latency_budget_s=_latency_budget_for(chunk),
        merge_budget_s=_merge_latency_budget_for(chunk),
        skipped_reason="profile_off",
    )


@lru_cache(maxsize=1)
def _latency_budgets() -> dict[str, float]:
    """Latency budget per category, as declared by the manifest (T21).

    A budget belongs to the **inference route** of a category rather than to the
    run as a whole: the STT and day layers run on the GPU tags, while the weekly
    rollup runs on the CPU twin ``diary-sum-cpu``, where a GPU-calibrated budget
    is unreachable. A category without a declared budget (the coaching note,
    launched explicitly and rarely) stays at ``None`` and is reported as ``n/a``
    instead of being compared with a foreign number.

    This is the budget of a **part** of the route; the day layer's fold-back call
    has a budget of its own (:func:`_merge_latency_budgets`, T41).
    """
    budget = golden_manifest().latency_budget_s
    return {
        "stt_error": float(budget.stt_fix),
        "day": float(budget.daily),
        "week": float(budget.weekly),
    }


@lru_cache(maxsize=1)
def _merge_latency_budgets() -> dict[str, float]:
    """Budget of the **fold-back** call of each category that declares one (T41).

    Only the day route does. Its fold-back is not a part: it is built from another
    template (``prompts/sum_daily/v2.md``) and returns the summary of the whole
    day (924 output tokens against 65-524 for the parts in the live run of
    2026-09-22), so the measured call is consistently slower than any part —
    18.93 / 18.95 / 18.95 s in three live runs, 18.56 and 18.43 s in acceptance
    runs, 19.74 s in the run that caught an NLI-gate regeneration (8 calls). 25 s
    is the project owner's decision of 2026-09-22: the same 15 s would keep grading
    a different call by the parts' number, and the margin over the worst measured
    call is ~27 %. The routes that declare no entry here keep the behaviour of T40
    (every call graded by the route budget), so no other threshold moves.

    The keys must stay in step with :data:`MERGE_PROMPT_VERSION_BY_CATEGORY` — a
    category the harness can recognise a fold-back in, but has no budget for, would
    silently fall back to the part budget. A test pins that they agree.
    """
    budget = golden_manifest().latency_budget_s
    return {"day": float(budget.daily_merge)}


def _latency_budget_for(chunk: GoldenChunk) -> float | None:
    """Latency budget of the parts of a category, or ``None`` when it declares none."""
    return _latency_budgets().get(chunk.category)


def _merge_latency_budget_for(chunk: GoldenChunk) -> float | None:
    """Budget of the fold-back call of a category, or ``None`` when it declares none."""
    return _merge_latency_budgets().get(chunk.category)


def _record_measurements(
    outcome: ChunkOutcome,
    chunk: GoldenChunk,
    profile: OpenAICompatibleProfile,
    provider: LLMProvider,
) -> None:
    """Record and check what the endpoint reported for **every** call (T35).

    The token figure is the **largest** measured prompt among the calls the layer
    made — not the last one. A layer that splits a long input (T23, ``ADR-016``)
    sends several requests, and reading the last of them (the fold-back) reported a
    number that measured the smallest call of the run while claiming to measure the
    chunk's input. The budget check uses the same maximum, because the window is a
    property of one call: the largest call is what can overflow it.

    The **call count** is not read here: it is the harness's own count of the calls
    it timed (T40, ``ChunkOutcome.requests``), so the report has one source for that
    number instead of two that a provider could disagree with. A provider that
    reports a single usage (an older or simpler adapter) is read through
    ``last_usage``; metrics only — never content.
    """
    usages = _measured_usages(provider)
    if not usages:
        return
    largest = max(usages, key=lambda usage: usage.input_tokens)
    outcome.usage = largest
    if largest.input_tokens + profile.max_tokens > chunk.budget.num_ctx:
        outcome.violations.append(
            Violation(
                "measured_context_budget",
                f"measured {largest.input_tokens} input tokens (largest of "
                f"{len(usages)} call(s)) + {profile.max_tokens} cap "
                f"> num_ctx {chunk.budget.num_ctx}",
            )
        )


def _measured_usages(provider: LLMProvider) -> list[LLMUsage]:
    """Every usage the provider reported for this chunk, in call order."""
    usages = getattr(provider, "usages", None)
    if isinstance(usages, list):
        return [usage for usage in usages if isinstance(usage, LLMUsage)]
    last = getattr(provider, "last_usage", None)
    return [last] if isinstance(last, LLMUsage) else []


def _call_step_label(outcome: ChunkOutcome, index: int) -> str:
    """How the violation and the report name the graded call (T41).

    A fold-back call is named as such — it is the one call whose budget differs —
    while the calls of a chunk that splits nothing keep the T40 wording (they are
    simply "the chunk's call"), so an ordinary day's failure reads as it did before.
    """
    if outcome.call_is_merge(index):
        return "the fold-back call"
    if len(outcome.call_seconds) == 1:
        return "the chunk's only call"
    return "a part call"


def _apply_latency_budget(outcome: ChunkOutcome) -> None:
    """Grade **every** call of the chunk against the budget of its step (T40, T41).

    Two decisions are held here. The first is T40's: the budget is compared with
    one **call**, not with the whole layer run. Until T40 the figure was the chunk,
    which on a day the layer honestly splits (T23, ``ADR-016``) is the sum of its
    parts plus the fold-back (17.76 s + 8.14 s + 18.77 s in the T35 measurement), so
    any correctly split day was red by construction and the number said nothing
    about the layer. Now every call is timed (:class:`TimedProvider`) and each of
    them is graded — equivalently, the slowest one when they share a budget. The
    work outside the calls (the store, the gate) leaves the budget; measured live
    on the three short golden days that is 0.24-3.03 s out of 15 s. The chunk total
    is still printed next to the graded figure, and the number of calls is never
    what fails a chunk: a split day with one slow call is as red as a single-call
    day that is slow.

    The second is T41's: which budget a call is graded with follows from **what the
    call is** (``call_budget_s`` — the layer's fold-back template against the
    fold-back budget of the route, everything else against the route budget), not
    from the call's position. The fold-back of a split day is a different call from
    a part — another template, and an output the size of the whole day — and grading
    it with the parts' number mixed two categories: measured live, the parts of the
    long golden day land in 5.3-13.7 s while the fold-back lands in 18.43-19.74 s.
    Every call over **its own** budget produces a violation, so a single slow part,
    a single slow fold-back and a chunk with both are each named as what they are;
    a route that declares no fold-back budget behaves exactly as in T40.
    """
    if outcome.latency_budget_s is None and outcome.merge_budget_s is None:
        return
    if not outcome.call_seconds:
        # Nothing was timed (the layer failed before its first request), so there is
        # no call to grade: the chunk's own wall time is the only measurement there
        # is, and it is compared with the route budget. Grading it as ``0.00s``
        # would read as «fast».
        budget = outcome.latency_budget_s
        if budget is None or outcome.latency_s <= budget:
            return
        outcome.violations.append(
            Violation(
                LATENCY_CHECK,
                f"{outcome.latency_s:.2f}s > {budget:.0f}s (the chunk — no call was timed; "
                f"chunk total {outcome.latency_s:.2f}s, profile {outcome.profile_name}, "
                f"category {outcome.category})",
            )
        )
        return
    calls = len(outcome.call_seconds)
    for index, seconds in enumerate(outcome.call_seconds):
        budget = outcome.call_budget_s(index)
        if budget is None or seconds <= budget:
            continue
        step = _call_step_label(outcome, index)
        outcome.violations.append(
            Violation(
                LATENCY_CHECK,
                f"{seconds:.2f}s > {budget:.0f}s (call {index + 1}/{calls} — {step}; "
                f"chunk total {outcome.latency_s:.2f}s, profile {outcome.profile_name}, "
                f"category {outcome.category})",
            )
        )


def evaluate_all(
    chunks: Sequence[GoldenChunk],
    *,
    provider_factory: Callable[[GoldenChunk], LLMProvider],
    models_config: ModelsConfig,
    judge: NliJudge | None = None,
    gate_judge: NliJudge | None = None,
) -> list[ChunkOutcome]:
    """Evaluate the selected chunks, one isolated database per chunk.

    A live provider is built from the chunk's profile, so ``profile_off`` can
    surface here as well as inside :func:`evaluate_chunk`; both places turn it
    into a skipped chunk instead of aborting the run (T19 DoD 4, §9.8).
    """
    outcomes: list[ChunkOutcome] = []
    for chunk in chunks:
        try:
            provider = provider_factory(chunk)
        except LLMProviderUnavailableError:
            outcomes.append(_skipped_outcome(chunk))
            continue
        outcomes.append(
            evaluate_chunk(
                chunk,
                provider=provider,
                models_config=models_config,
                judge=judge,
                gate_judge=gate_judge,
            )
        )
    return outcomes


# ---------------------------------------------------------------------------
# Providers: the CI mock and the live wrapper
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Deterministic provider: one fixed completion, or one answer per request.

    ``payload`` is either a fixed mapping (or raw JSON text) served for every
    call, or a callable that builds the answer of one call from its request —
    which is what a chunk whose layer splits its input needs (T35).
    """

    def __init__(
        self,
        payload: Mapping[str, object] | str | Callable[[LLMRequest], Mapping[str, object]],
    ) -> None:
        self._payload = payload
        self.requests: list[LLMRequest] = []

    @property
    def request_count(self) -> int:
        """Number of ``complete`` calls served (a week or a long day may need several)."""
        return len(self.requests)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        payload = self._payload(request) if callable(self._payload) else self._payload
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return LLMResponse(text=text)


def ci_payload(chunk: GoldenChunk) -> dict[str, object]:
    """A contract-valid, invariant-satisfying completion for one chunk.

    The mock is built *from the chunk and its declared expectations*: it exists
    to exercise the harness plumbing (migrations, layers, store, checks) without
    network or GPU, not to imitate a model. Its honesty is in that sentence —
    the checks are proven by the negative tests, not by this provider.
    """
    if chunk.contract == "stt_corrected":
        entry = _single_raw_entry(chunk)
        text_value = entry.text.strip()
        corrected = text_value[:1].upper() + text_value[1:]
        if corrected and corrected[-1] not in ".!?":
            corrected = f"{corrected}."
        return {"text": corrected, "uncertain": []}
    if chunk.contract == "daily_summary":
        facts = [item.text for item in chunk.input if isinstance(item, RawEntry) and item.text]
        assert isinstance(chunk.invariants, DailyInvariants)
        return {
            "facts": facts or ["Запись без текста."],
            "themes": [],
            "energy": chunk.invariants.expected_energy,
            "open_questions": [],
        }
    if chunk.contract == "weekly_summary":
        facts = [
            fact for item in chunk.input if isinstance(item, DailySummary) for fact in item.facts
        ]
        return {"rollup": " ".join(facts)}
    rollup = chunk.input.rollup if isinstance(chunk.input, WeeklySummary) else ""
    sentences = [part.strip() for part in re.split(r"\.\s+|\n+", rollup) if part.strip()]
    observations = (sentences + ["Наблюдение по сводке недели."] * 3)[:3]
    return {
        "observations": observations,
        "hypotheses": [
            "Возможно, неделя шла неравномерно.",
            "Похоже, что нагрузка распределялась по-разному.",
        ],
        "experiment_7d": "В течение следующих 7 дней отмечать, что занимало основное время.",
    }


class MeasuringProvider:
    """Wraps a provider and remembers every usage and the number of calls.

    All reported :class:`~diary.domain.llm.models.LLMUsage` counters are kept, in
    call order, because a layer may make several calls (T35) and the chunk's
    measurement is the largest of them; ``last_usage`` stays for a caller that
    wants the last one. Metrics only — never content.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.request_count = 0
        self.usages: list[LLMUsage] = []
        self.last_usage: LLMUsage | None = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._provider.complete(request)
        self.request_count += 1
        if response.usage is not None:
            self.usages.append(response.usage)
            self.last_usage = response.usage
        return response


def ci_provider_factory(chunk: GoldenChunk) -> LLMProvider:
    """Provider factory of the CI mode (no network, no GPU, no API key).

    A chunk whose layer splits its input — a day that does not fit one window (T35)
    — gets a provider answering each call about the piece that call carries
    (:func:`ci_call_payload`); every other chunk keeps the single fixed completion.
    """
    if chunk.contract == "daily_summary" and chunk.budget.chunks_planned > 1:
        return ScriptedProvider(lambda request: ci_call_payload(chunk, request))
    return ScriptedProvider(ci_payload(chunk))


def ci_call_payload(chunk: GoldenChunk, request: LLMRequest) -> dict[str, object]:
    """The mock completion of one call of a **split** day chunk (T35).

    The fixed whole-chunk completion of :func:`ci_payload` cannot answer a split
    day: every part request carries one part of the day, and the fold-back request
    carries the parts. A fixed whole-day answer would put the whole day into the
    fold-back request once per part, and the *mock* — not the layer — would be what
    fails on the merge budget: with this chunk's answer (2369 characters) the
    fold-back would spend 3849 of its 4096 estimated tokens already at two parts,
    and 4796 from three parts.

    The rule stays the mock's own rule, applied to what the call carries: one fact
    per entry text (part requests), and the parts' facts in part order for the
    fold-back. The level is the chunk's declared expectation, as in the single-call
    mock, so the fold-back receives the level the chunk declares. The result is
    therefore the completion a single call would have produced — the mock's
    honesty is unchanged: it exercises the plumbing, not the model.
    """
    if not isinstance(chunk.invariants, DailyInvariants):
        raise HarnessError(f"chunk {chunk.id} is not a day chunk, so a split call cannot be mocked")
    content = request.messages[0].content
    if SUM_PARTS_HEADER in content:
        parts = _json_items(content.rsplit(SUM_PARTS_HEADER, 1)[-1])
        facts = [
            fact
            for part in parts
            if isinstance(part, Mapping)
            for fact in _strings_of(part.get("facts"))
        ]
    else:
        entries = _json_items(content.rsplit(SUM_ENTRIES_HEADER, 1)[-1])
        facts = [
            text
            for entry in entries
            if isinstance(entry, Mapping)
            for text in _strings_of(entry.get("text"))
        ]
    return {
        "facts": facts or ["Запись без текста."],
        "themes": [],
        "energy": chunk.invariants.expected_energy,
        "open_questions": [],
    }


def _json_items(text: str) -> list[object]:
    """The JSON array a request carries after its content header."""
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise HarnessError("a request of the split mock does not carry a JSON array") from exc
    if not isinstance(payload, list):
        raise HarnessError("a request of the split mock does not carry a JSON array")
    return payload


def _strings_of(value: object) -> list[str]:
    """The non-empty strings of a JSON value (an absent field yields none)."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def live_provider_factory(models_config: ModelsConfig) -> Callable[[GoldenChunk], LLMProvider]:
    """Provider factory of a live run: the real adapter, wrapped for metrics."""

    def factory(chunk: GoldenChunk) -> LLMProvider:
        profile = resolve_phase2_profile(models_config, PROFILE_NAME_BY_CATEGORY[chunk.category])
        return MeasuringProvider(create_llm_provider(profile))

    return factory


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    """Aggregate result of a harness run (metrics only, never content).

    T18 keeps the two natures of a failure apart instead of adding them into one
    number: ``invariants_passed`` is what the manifest threshold talks about,
    ``latency_within_budget`` counts the chunks that stayed inside their route's
    budget (out of ``latency_measured``; each chunk is graded by its **slowest
    single call** of each kind, T40/T41), and ``fully_passed`` is their
    intersection. The report never blends them, because a gate refusal and a cold
    model load are different events.
    """

    total: int
    skipped: int
    skipped_categories: frozenset[str]
    invariants_passed: int
    latency_measured: int
    latency_within_budget: int
    fully_passed: int
    #: The slowest **single part call** of each category (T40) — the figure the
    #: route budget was compared with, and the one the report prints as ``slowest
    #: call`` (or as ``slowest part call`` where a fold-back figure is printed next
    #: to it). A chunk that splits nothing contributes its only call here.
    max_call_by_category: dict[str, float]
    #: The slowest **fold-back** call of each category that made one (T41), graded
    #: with the fold-back budget of its route and printed with it. Separate from
    #: ``max_call_by_category`` because the two figures answer two different
    #: questions: how fast the parts are, and how long the summary of the whole day
    #: takes.
    max_merge_call_by_category: dict[str, float]
    #: The slowest **chunk total** of each category: every call of the layer plus
    #: the work around them. Printed next to the call figures, so the price of
    #: splitting a day is visible in the summary as well as in the chunk row.
    max_latency_by_category: dict[str, float]
    latency_budgets: dict[str, float | None]
    #: Fold-back budget of each category that declares one (T41): the number the
    #: calls of :attr:`max_merge_call_by_category` were graded with. ``None`` for a
    #: route whose layer makes no separate fold-back — there the route budget
    #: grades every call.
    merge_budgets: dict[str, float | None]
    diagnostics_total: int
    #: NLI measurement block (T24a). Zeroed when ``--nli`` was not requested; the
    #: counters are deliberately separate from the invariant and latency counts.
    nli_summaries: int
    nli_claims: int
    nli_contradictions: int
    nli_unsupported: int
    nli_max_contradiction: float
    nli_flagged: tuple[str, ...]

    @property
    def measured(self) -> int:
        """Chunks that were actually run: the selected ones minus the skips."""
        return self.total - self.skipped


def summarize(outcomes: Sequence[ChunkOutcome]) -> Summary:
    """Aggregate outcomes into the numbers the definition of done talks about.

    Skipped chunks (a layer switched off in this configuration) stay out of every
    count: nothing was measured for them, so counting them as clean would inflate
    the score (T19 DoD 4) — and a category with no measured chunk reports nothing
    rather than a flattering ``0.00s``.
    """
    max_latency: dict[str, float] = {}
    max_call: dict[str, float] = {}
    max_merge_call: dict[str, float] = {}
    budgets: dict[str, float | None] = {}
    merge_budgets: dict[str, float | None] = {}
    skipped_categories: set[str] = set()
    for outcome in outcomes:
        if outcome.skipped:
            skipped_categories.add(outcome.category)
            continue
        max_latency[outcome.category] = max(
            max_latency.get(outcome.category, 0.0), outcome.latency_s
        )
        # Two figures, two questions (T41): how fast the parts are, and how long the
        # fold-back of the whole day takes. A chunk that splits nothing has no
        # fold-back and contributes its only call to the part figure, exactly as in
        # T40; the fold-back figure is absent rather than zeroed, so the report says
        # «nothing of this kind was measured» instead of «it took no time».
        part_call = outcome.slowest_part_call_s
        if part_call is not None:
            max_call[outcome.category] = max(max_call.get(outcome.category, 0.0), part_call)
        merge_call = outcome.slowest_merge_call_s
        if merge_call is not None:
            max_merge_call[outcome.category] = max(
                max_merge_call.get(outcome.category, 0.0), merge_call
            )
        budgets.setdefault(outcome.category, outcome.latency_budget_s)
        merge_budgets.setdefault(outcome.category, outcome.merge_budget_s)
    measured = [outcome for outcome in outcomes if not outcome.skipped]
    timed = [outcome for outcome in measured if outcome.latency_budget_s is not None]
    judged = [outcome for outcome in measured if outcome.nli is not None]
    metrics = [outcome.nli for outcome in judged if outcome.nli is not None]
    flagged = tuple(
        f"{outcome.chunk_id} {coordinate}"
        for outcome in judged
        if outcome.nli is not None
        for coordinate in outcome.nli.contradicted_coordinates
    )
    return Summary(
        total=len(outcomes),
        skipped=len(outcomes) - len(measured),
        skipped_categories=frozenset(skipped_categories - set(max_latency)),
        invariants_passed=sum(1 for outcome in measured if not outcome.content_violations),
        latency_measured=len(timed),
        latency_within_budget=sum(1 for outcome in timed if outcome.latency_within_budget),
        fully_passed=sum(1 for outcome in measured if outcome.passed),
        max_call_by_category=max_call,
        max_merge_call_by_category=max_merge_call,
        max_latency_by_category=max_latency,
        latency_budgets=budgets,
        merge_budgets=merge_budgets,
        diagnostics_total=sum(len(outcome.diagnostics) for outcome in outcomes),
        nli_summaries=len(judged),
        nli_claims=sum(item.claims for item in metrics),
        nli_contradictions=sum(item.contradiction for item in metrics),
        nli_unsupported=sum(len(item.unsupported_coordinates) for item in metrics),
        nli_max_contradiction=max((item.max_contradiction for item in metrics), default=0.0),
        nli_flagged=flagged,
    )


def _result_label(outcome: ChunkOutcome) -> str:
    """``PASS``, ``SKIP`` or which nature of failure hit the chunk (T18).

    Naming the nature in the row is what lets a cold start be told from a content
    violation without reading the detail of a check; a skipped chunk says that
    its layer is switched off rather than pretending it passed.
    """
    if outcome.skipped:
        return f"SKIP:{outcome.skipped_reason}"
    natures: list[str] = []
    if outcome.content_violations:
        natures.append("content")
    if outcome.latency_violations:
        natures.append("latency")
    return "PASS" if not natures else "FAIL:" + "+".join(natures)


def _measured_usage_lines(outcomes: Sequence[ChunkOutcome]) -> list[str]:
    """One line naming the largest measured **call** of each category (live only).

    The endpoint's own count is the counter that can judge the character-based
    estimate, so the live report shows the maxima next to the generation cap of
    each profile. The figure of a chunk is the largest of its calls, not its last
    one (T35): a layer that splits a long day sends several requests, and the
    fold-back is the smallest of them, so reading the last call would report a
    prompt that measured nothing. CI mode receives no usage and the line is absent.
    The values are metrics only — never content (AGENTS.md §7-8).
    """
    maxima: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.usage is None:
            continue
        maxima[outcome.category] = max(maxima.get(outcome.category, 0), outcome.usage.input_tokens)
    if not maxima:
        return []
    parts = ", ".join(
        f"{category} {maxima[category]} (cap {PROFILE_MAX_TOKENS[category]})"
        for category in sorted(maxima)
    )
    return [f"measured input tokens (endpoint), max call per category: {parts}"]


def _per_call_latency_lines(outcomes: Sequence[ChunkOutcome]) -> list[str]:
    """Name every call of a chunk that made several, each with **its** budget (T40/T41).

    The budget is graded per call, so the report has to show the calls: a reader who
    sees only the slowest one cannot tell a layer that made three fast calls from
    one that made a single slow call, and the T35 finding («any correctly split day
    is red by construction») was exactly about that difference. Since T41 the budget
    of a call is printed next to the call rather than once per chunk, because the
    calls of a split day are no longer all graded by the same number: a part is
    graded by the route budget and the fold-back by the budget of the fold-back. A
    reader must not have to infer which threshold a figure was compared with.

    Chunks with a single call print nothing extra — their row already is the call.
    Metrics only, never content (AGENTS.md §7-8).
    """
    several = [
        outcome for outcome in outcomes if not outcome.skipped and len(outcome.call_seconds) > 1
    ]
    if not several:
        return []
    lines = [
        "",
        "per-call latency of the chunks that made more than one call (each call is "
        "graded by the budget printed with it, T40/T41):",
    ]
    for outcome in several:
        calls = ", ".join(
            f"{seconds:.2f}s ({_call_budget_label(outcome, index)})"
            for index, seconds in enumerate(outcome.call_seconds)
        )
        lines.append(
            f"  {outcome.chunk_id} ({len(outcome.call_seconds)} call(s)): {calls} — "
            f"slowest {outcome.slowest_call_s:.2f}s, chunk total {outcome.latency_s:.2f}s"
        )
    return lines


def _call_budget_label(outcome: ChunkOutcome, index: int) -> str:
    """``part < 15s`` / ``fold-back < 25s`` — the step and the number it is graded by.

    One label for the report and one for nothing else: the step comes from
    ``call_is_merge`` (the request's template version) and the number from
    ``call_budget_s``, so what is printed is what was graded.
    """
    step = "fold-back" if outcome.call_is_merge(index) else "part"
    budget = outcome.call_budget_s(index)
    verified = "n/a" if budget is None else f"< {budget:.0f}s"
    return f"{step} {verified}"


def format_report(outcomes: Sequence[ChunkOutcome], manifest: GoldenManifest) -> str:
    """The human report: chunk ids, metrics and failed check names only.

    No generated text, no input text and no diary content appears here (AGENTS.md
    §7-8): only ids, categories, the profile each chunk ran under, latency,
    pass/fail, check names and — when the endpoint reports usage — the measured
    token maxima. Invariants and latency are reported as two separate counts
    (T18) — the threshold applies to the invariant count alone. Latency appears as
    two figures (T40): the chunk's **total** (``latency``, the price of splitting a
    long day included) and the **slowest call** it was graded by (``slowest``),
    with the calls of a multi-call chunk listed under the table. The category
    summary names the slowest part call and the slowest fold-back call apart, each
    with **its own** budget (T41), so which threshold graded which call is read off
    the report rather than inferred.
    """
    lines: list[str] = []
    header = (
        f"{'chunk':<34}{'category':<12}{'profile':<14}{'latency':>9}{'calls':>7}"
        f"{'slowest':>9}  {'result':<18} failed checks"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for outcome in outcomes:
        failed = ", ".join(outcome.failed_checks) if outcome.failed_checks else "-"
        # A skipped chunk has no measurement: `0.00s` would read as «fast».
        latency = "n/a" if outcome.skipped else f"{outcome.latency_s:.2f}s"
        # The call count is what makes a split day visible in a live run (T35): a
        # chunk that needed several calls says so, instead of looking like one call
        # whose measured prompt was small.
        calls = "-" if outcome.skipped else str(outcome.requests)
        # `latency` is the chunk's total (what the user waits for), `slowest` is the
        # figure the route budget is compared with since T40 (the largest prompt the
        # endpoint counted is a different figure and keeps its own name in the token
        # line). Printing both is what keeps the price of splitting a day visible
        # instead of hiding it behind the budget.
        slowest = "n/a" if outcome.skipped else f"{outcome.slowest_call_s:.2f}s"
        lines.append(
            f"{outcome.chunk_id:<34}{outcome.category:<12}{outcome.profile_name:<14}"
            f"{latency:>9}{calls:>7}{slowest:>9}  {_result_label(outcome):<18} {failed}"
        )
    lines.extend(_per_call_latency_lines(outcomes))
    summary = summarize(outcomes)
    lines.append("")
    threshold = f"{manifest.threshold.pass_count}/{manifest.threshold.total}"
    if summary.total == manifest.threshold.total and not summary.skipped:
        verdict = "MET" if summary.invariants_passed >= manifest.threshold.pass_count else "MISSED"
        scope = f"threshold {threshold}: {verdict}"
    else:
        # A subset run cannot meet the threshold of the whole set, and a skipped
        # chunk was not measured at all — both are said instead of implied.
        measured = f"{summary.measured}/{manifest.threshold.total}"
        scope = f"threshold {threshold}, {measured} chunks measured"
        if summary.skipped:
            scope = f"{scope}, {summary.skipped} skipped (profile off)"
    lines.append(f"invariants: {summary.invariants_passed}/{summary.measured} passed ({scope})")
    # The budgets named here are the manifest's declaration, not the run's scope:
    # they stay true even when a category was skipped and nothing was measured.
    # The fold-back budget of the day route is listed with its owner (T41), so a
    # reader of the summary line knows the day route is graded by two numbers.
    scopes = ", ".join(
        [f"{category} < {budget:.0f}s" for category, budget in sorted(_latency_budgets().items())]
        + [
            f"{category} fold-back < {budget:.0f}s"
            for category, budget in sorted(_merge_latency_budgets().items())
        ]
    )
    if summary.latency_measured:
        lines.append(
            f"latency: {summary.latency_within_budget}/{summary.latency_measured} within budget "
            f"({scopes})"
        )
    else:
        lines.append(f"latency: not measured ({scopes})")
    if summary.skipped:
        names = ", ".join(outcome.chunk_id for outcome in outcomes if outcome.skipped)
        lines.append(f"skipped (profile off): {summary.skipped} chunk(s) — {names}")
    for category in sorted(set(summary.max_latency_by_category) | summary.skipped_categories):
        if category in summary.skipped_categories:
            # Every chunk of this category was skipped: there is no measurement to
            # report, and `max 0.00s` would read as «measured and fast».
            lines.append(f"latency {category}: n/a (skipped, layer off)")
            continue
        budget = summary.latency_budgets.get(category)
        verified = "n/a" if budget is None else f"< {budget:.0f}s"
        # Two figures on purpose (T40): the budget is compared with one call, and the
        # chunk's total — the price of splitting a long day into parts — is printed
        # next to it so a reader can see both without opening the chunk row.
        fragments: list[str] = []
        part_call = summary.max_call_by_category.get(category)
        merge_call = summary.max_merge_call_by_category.get(category)
        if part_call is not None:
            # «part call» only where a fold-back figure stands next to it (T41):
            # otherwise there is one kind of call and the old wording holds.
            named = "slowest part call" if merge_call is not None else "slowest call"
            fragments.append(f"{named} {part_call:.2f}s (budget {verified})")
        if merge_call is not None:
            merge_budget = summary.merge_budgets.get(category)
            merge_verified = "n/a" if merge_budget is None else f"< {merge_budget:.0f}s"
            fragments.append(f"slowest fold-back call {merge_call:.2f}s (budget {merge_verified})")
        if not fragments:
            # No call of this category was timed at all (the layers refused their
            # input before their first request): the fallback figure is the chunk's
            # own wall time, and that is what is named rather than a call that does
            # not exist (T40).
            fragments.append("no call was timed")
        lines.append(
            f"latency {category}: {', '.join(fragments)}, slowest chunk total "
            f"{summary.max_latency_by_category[category]:.2f}s"
        )
    lines.append(
        f"diagnostics (informational, not a failure): {summary.diagnostics_total} token(s)"
    )
    lines.extend(_measured_usage_lines(outcomes))
    if summary.nli_summaries:
        # A measurement of its own (T24a): it must not be read as a pass
        # criterion, and a contradiction here does not change the exit code until
        # T24b promotes the rule to the runtime gate.
        lines.append(
            f"nli (measurement, not a gate): summaries {summary.nli_summaries}, "
            f"claims {summary.nli_claims}, contradictions {summary.nli_contradictions}, "
            f"unsupported {summary.nli_unsupported}, "
            f"max p_contradiction {summary.nli_max_contradiction:.2f}"
        )
        if summary.nli_flagged:
            lines.append(f"nli flagged contradictions: {', '.join(summary.nli_flagged)}")
    return "\n".join(lines)


def _warn_on_reasoning(models_config: ModelsConfig) -> str | None:
    """Warn when a phase-2 profile may pay for the reasoning channel (ADR-009)."""
    for name, profile in models_config.phase2_profiles():
        if profile.reasoning_effort is None:
            return (
                f"warning: profile {name!r} has no reasoning_effort; the latency budgets of T10 "
                "are only reachable with the reasoning channel switched off (ADR-009)"
            )
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T10 golden regression harness")
    parser.add_argument(
        "--live",
        action="store_true",
        help="run against the configured endpoint instead of the deterministic mock",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=None,
        help="model configuration file (default: the example config in CI mode)",
    )
    parser.add_argument(
        "--chunk",
        action="append",
        default=None,
        help="run only this chunk id (repeatable)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "require every selected chunk to be measured and to pass, not just the "
            "manifest threshold; a skipped chunk (layer switched off) fails a strict run"
        ),
    )
    parser.add_argument(
        "--nli",
        action="store_true",
        help=(
            "also check the claims of every summary against the canonical input of its "
            "layer with the local NLI instrument (T24a): a separate measurement block, "
            "never part of the invariant or latency counts, and an unavailable "
            "instrument fails the run"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the harness; return the process exit code."""
    args = _parse_args(argv)
    manifest, chunks = load_golden_set(chunk_ids=args.chunk)
    config_path = args.models_config or (None if args.live else CI_MODELS_CONFIG)
    models_config = load_models_config(config_path)
    if args.live:
        warning = _warn_on_reasoning(models_config)
        if warning:
            print(warning, file=sys.stderr)
        factory = live_provider_factory(models_config)
        mode = "live"
    else:
        factory = ci_provider_factory
        mode = "ci"
    print(f"mode: {mode}; chunks: {len(chunks)}; config: {config_path or '<default>'}")
    # A live run measures the layer as production runs it: with the NLI gate armed
    # (T24b). The CI mode has neither the extra nor the weights and passes nothing.
    gate_judge: NliJudge | None = None if mode == "ci" else default_nli_judge()
    judge: NliJudge | None = None
    if args.nli:
        try:
            judge = nli_judge()
            # Prove the instrument *before* the run: an unavailable judge must fail
            # the run even when the selection turns out to contain no summary chunk
            # at all, which would otherwise leave the block empty and the run green
            # (found in the first review of T24a). A scripted judge has no ``load``.
            load = getattr(judge, "load", None)
            if callable(load):
                load()
            outcomes = evaluate_all(
                chunks,
                provider_factory=factory,
                models_config=models_config,
                judge=judge,
                gate_judge=gate_judge,
            )
        except NliError as exc:
            # Fail-closed on the owner's decision for T24 («если NLI недоступен —
            # Fail»): a requested check that cannot run is a failed run, never an
            # absent one. The message names the fix (extra or fetch script), not
            # any diary content.
            print(f"FAILED: the NLI check could not run: {exc}")
            return 1
    else:
        outcomes = evaluate_all(
            chunks,
            provider_factory=factory,
            models_config=models_config,
            gate_judge=gate_judge,
        )
    print(format_report(outcomes, manifest))
    summary = summarize(outcomes)
    required = len(chunks) if (args.strict or args.chunk) else manifest.threshold.pass_count
    failures: list[str] = []
    if args.nli and summary.nli_summaries == 0:
        # The instrument was requested and worked, but there was nothing for it to
        # verify: a green run would read as «the claims were checked» (T24a review).
        failures.append(
            "the NLI check verified nothing: the selected chunks declare no summary claims"
        )
    if summary.measured == 0:
        # Fail-closed: a run that measured nothing verified nothing, whatever the
        # reason for the skips.
        skipped = summary.skipped
        failures.append(
            f"nothing was measured: all {skipped} selected chunk(s) are skipped (profile off)"
        )
    elif summary.invariants_passed < required:
        detail = f"{summary.invariants_passed}/{summary.measured} measured"
        if summary.skipped:
            detail = f"{detail}, {summary.skipped} skipped (profile off)"
        failures.append(f"invariants {detail}, {required} required")
    if summary.latency_within_budget < summary.latency_measured:
        over = summary.latency_measured - summary.latency_within_budget
        failures.append(f"latency {over} chunk(s) over budget")
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    skipped = f"; skipped {summary.skipped} (profile off)" if summary.skipped else ""
    print(
        f"OK: invariants {summary.invariants_passed}/{summary.measured} "
        f"(required {required}); "
        f"latency {summary.latency_within_budget}/{summary.latency_measured} within budget{skipped}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
