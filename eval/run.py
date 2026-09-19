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
    asserted here: an STT correction under 5 s, a day summary under 15 s and at
    least 8 of 10 chunks passing.

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
* the context budget ``chunk_tokens_est + reserve_tokens_for(category) <=
  num_ctx`` and, when the endpoint reports usage, the *measured*
  ``input_tokens + profile.max_tokens <= num_ctx``;
* the raw source rows of an STT/day chunk are unchanged by the run, and the
  in-memory chunk input is not mutated;
* the latency budget of the manifest (STT under 5 s, day under 15 s).

Deliberate limits (stated, not hidden)
======================================

* ``no_new_facts`` is exact for digits, numeral words and proper names.
  Free-form content words are **not** machine-checkable without a semantic
  model: a paraphrase and an invention look identical to a lexical check. The
  harness reports ungrounded long content tokens as *diagnostics* (a hint for a
  human), never as a pass/fail criterion — a fabricated "green" check would be
  worse than an honest gap.
* Grounding compares against every content value of the input, timestamps and
  anchors included (they are part of the input). A number that only occurs
  inside a timestamp or an ISO date is therefore considered grounded.
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
    CAUSAL_CONNECTOR_LEXEMES,
    CONTRACT_CONSTANT_DIGITS,
    LEXEME_FIDELITY_CONTRACTS,
    NON_CONTENT_INPUT_KEYS,
    NUMERAL_WORDS,
    CoachInvariants,
    DailyInvariants,
    GoldenChunk,
    GoldenManifest,
    WeekInvariants,
    assertive_text_fields,
    canonical_input_json,
    chunk_path,
    coach_disclaimer,
    generated_text_fields,
    input_payload,
    load_chunk,
    load_manifest,
    reserve_tokens_for,
)
from diary.domain.llm.models import LLMRequest, LLMResponse, LLMUsage
from diary.domain.llm.providers import LLMProvider, create_llm_provider
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.pipelines.coach.pipeline import PROFILE_NAME as COACH_PROFILE_NAME
from diary.domain.pipelines.coach.service import coach_note
from diary.domain.pipelines.stt_fix.pipeline import PROFILE_NAME as STT_PROFILE_NAME
from diary.domain.pipelines.stt_fix.service import correct_stt
from diary.domain.pipelines.sum_daily.day_input import collect_day_entries
from diary.domain.pipelines.sum_daily.pipeline import PROFILE_NAME as SUM_PROFILE_NAME
from diary.domain.pipelines.sum_daily.service import sum_daily
from diary.domain.pipelines.sum_weekly.service import sum_weekly

#: Tracked, secret-free configuration used by the CI mode (no endpoint is ever
#: contacted there and the mock provider is injected, so ``api_key_env`` is
#: never resolved).
CI_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.example.yaml"

#: Raw source tables whose rows must be identical after a run.
RAW_TABLES: tuple[str, ...] = ("messages", "audio_files", "transcriptions")

#: Phase-2 profile each golden category runs under (taken from the layer
#: packages so a rename cannot drift silently).
PROFILE_NAME_BY_CATEGORY: dict[str, str] = {
    "stt_error": STT_PROFILE_NAME,
    "day": SUM_PROFILE_NAME,
    "week": SUM_PROFILE_NAME,
    "wellbeing": COACH_PROFILE_NAME,
}

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
)

_WORD_RE = re.compile(r"[а-яёa-z]+")
_DIGIT_RE = re.compile(r"\d+")
#: Sentence split used to exclude a sentence-initial capital from the names:
#: sentence-ending punctuation followed by whitespace, or a line break.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
#: A capitalized word (Cyrillic or Latin) — candidate proper name.
_CAPITALIZED_RE = re.compile(r"[А-ЯЁ][а-яё]+|[A-Z][a-z]+")

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

#: Stem length used when checking that a proper name occurs in the input (long
#: enough to tolerate Russian inflection, short enough to reject a new name).
_NAME_STEM_LEN = 5


class HarnessError(Exception):
    """The harness itself cannot run: unknown kind, unchecked invariant, bad set."""


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
    latency_s: float = 0.0
    latency_budget_s: float | None = None
    violations: list[Violation] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    requests: int = 0
    usage: LLMUsage | None = None

    @property
    def passed(self) -> bool:
        """Whether the chunk passed every check (latency included)."""
        return not self.violations

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
# Text helpers
# ---------------------------------------------------------------------------


def _content_strings(value: object) -> Iterator[str]:
    """String leaves of a payload, skipping non-content keys (ids, paths)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if key in NON_CONTENT_INPUT_KEYS:
                continue
            yield from _content_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _content_strings(item)


def _generated_values(payload: Mapping[str, object], fields: Sequence[str]) -> Iterator[str]:
    """Values of the generated text ``fields`` of one output payload."""
    for _, value in _generated_field_values(payload, fields):
        yield value


def _generated_field_values(
    payload: Mapping[str, object], fields: Sequence[str]
) -> Iterator[tuple[str, str]]:
    """``(field, value)`` pairs of the generated text fields of a payload."""
    for name in fields:
        value = payload.get(name)
        if isinstance(value, str):
            yield name, value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    yield name, item


def _input_content_text(chunk: GoldenChunk) -> str:
    """Casefolded content text of a chunk input, used for grounding checks."""
    return " ".join(_content_strings(input_payload(chunk.input))).casefold()


def _input_numbers(chunk: GoldenChunk) -> frozenset[str]:
    """Digit tokens of the chunk input (identifiers and audio paths excluded)."""
    return frozenset(_numbers(_input_content_text(chunk)))


def _numbers(text_value: str) -> set[str]:
    """Digit sequences of a text value (``15:30`` contributes ``15`` and ``30``)."""
    return set(_DIGIT_RE.findall(text_value))


def _numeral_word_digits(text_value: str) -> dict[str, str]:
    """Map ``digit`` → the numeral word that produced it, for the output text."""
    found: dict[str, str] = {}
    for token in _WORD_RE.findall(text_value.casefold()):
        value = NUMERAL_WORDS.get(token)
        if value is not None:
            found.setdefault(str(value), token)
    return found


def _proper_names(text_value: str) -> list[str]:
    """Capitalized words of ``text_value`` that are not sentence-initial.

    Sentence starts are excluded because every Russian sentence begins with a
    capital by orthography; the first capitalized word of each sentence is
    therefore not a name candidate.
    """
    names: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text_value):
        stripped = sentence.lstrip()
        skip = _CAPITALIZED_RE.match(stripped)
        skipped = skip.group(0) if skip else ""
        for token in _CAPITALIZED_RE.findall(sentence):
            if token == skipped:
                skipped = ""
                continue
            names.append(token)
    return names


def _lexeme_hits(text_value: str, lexemes: Sequence[str]) -> list[str]:
    """Lexemes of ``lexemes`` occurring in ``text_value`` (case-insensitive)."""
    folded = text_value.casefold()
    return [lexeme for lexeme in lexemes if lexeme.casefold() in folded]


def _name_is_grounded(name: str, input_text: str) -> bool:
    """Whether ``name`` occurs in the (casefolded) input, inflections allowed.

    A stem match tolerates Russian inflection (``Марина`` vs ``мариной``) while
    still rejecting a name that is nowhere in the input.
    """
    folded = name.casefold()
    return folded in input_text or folded[:_NAME_STEM_LEN] in input_text


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


def _introduced_lexemes(ctx: CheckContext, value: str, lexemes: Sequence[str]) -> list[str]:
    """Lexeme hits the layer introduced rather than copied from the source.

    For a correction layer (:data:`LEXEME_FIDELITY_CONTRACTS`) the generated
    text must stay faithful to the input, so a lexeme already present in the
    input is not a violation — only wording the layer added is. Generative
    layers get no exemption at all.
    """
    hits = _lexeme_hits(value, lexemes)
    if ctx.chunk.contract not in LEXEME_FIDELITY_CONTRACTS:
        return hits
    return [hit for hit in hits if hit.casefold() not in ctx.input_text]


def _check_forbidden_lexemes(ctx: CheckContext) -> list[Violation]:
    """The chunk blacklist plus the eval-contract advice/evaluation baseline."""
    lexemes = sorted(set(ctx.chunk.invariants.forbidden_lexemes) | set(ADVICE_EVALUATION_LEXEMES))
    violations: list[Violation] = []
    for value in _generated_values(ctx.payload, generated_text_fields(ctx.chunk.contract)):
        for hit in _introduced_lexemes(ctx, value, lexemes):
            violations.append(
                Violation("forbidden_lexemes", f"forbidden lexeme {hit!r} is present")
            )
    return violations


def _check_advice_and_judgement_absent(ctx: CheckContext) -> list[Violation]:
    """No advice, praise, evaluation or morality in the generated text fields."""
    violations: list[Violation] = []
    for value in _generated_values(ctx.payload, generated_text_fields(ctx.chunk.contract)):
        for hit in _introduced_lexemes(ctx, value, ADVICE_EVALUATION_LEXEMES):
            violations.append(
                Violation("advice_and_judgement_absent", f"advice/evaluation lexeme {hit!r}")
            )
    return violations


def _check_causal_claims_absent(ctx: CheckContext) -> list[Violation]:
    """No causal connector in the assertive generated text fields."""
    violations: list[Violation] = []
    for value in _generated_values(ctx.payload, assertive_text_fields(ctx.chunk.contract)):
        for hit in _introduced_lexemes(ctx, value, CAUSAL_CONNECTOR_LEXEMES):
            violations.append(Violation("causal_claims_absent", f"causal connector {hit!r}"))
    return violations


def _check_no_new_facts(ctx: CheckContext) -> list[Violation]:
    """Grounding of digits, numeral words and proper names in the input.

    Exact for the three classes it covers; free-form content words are left to
    the diagnostics, because a lexical check cannot tell a paraphrase from an
    invention (see the module docstring). Digits a field is contractually
    required to carry (:data:`CONTRACT_CONSTANT_DIGITS`) are exempt.
    """
    violations: list[Violation] = []
    for field_name, value in _generated_field_values(
        ctx.payload, generated_text_fields(ctx.chunk.contract)
    ):
        allowed = CONTRACT_CONSTANT_DIGITS.get((ctx.chunk.contract, field_name), frozenset())
        for digit in sorted(_numbers(value)):
            if digit not in ctx.input_numbers and digit not in allowed:
                violations.append(
                    Violation("no_new_facts", f"number {digit!r} does not occur in the input")
                )
        for digit, word in sorted(_numeral_word_digits(value).items()):
            if digit not in ctx.input_numbers and digit not in allowed:
                violations.append(
                    Violation(
                        "no_new_facts",
                        f"numeral word {word!r} ({digit}) does not occur in the input",
                    )
                )
        for proper in _proper_names(value):
            if not _name_is_grounded(proper, ctx.input_text):
                violations.append(
                    Violation("no_new_facts", f"proper name {proper!r} does not occur in the input")
                )
    return violations


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
    for digit in sorted(_numbers(ctx.model.text)):
        if digit not in ctx.input_numbers:
            violations.append(
                Violation("names_and_numbers_grounded", f"number {digit!r} is not in the input")
            )
    for name in _proper_names(ctx.model.text):
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
        for digit in sorted(_numbers(value)):
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
    for value in _generated_values(ctx.payload, generated_text_fields(ctx.chunk.contract)):
        for token in _WORD_RE.findall(value.casefold()):
            if len(token) < _DIAGNOSTIC_MIN_LEN or token in _STOPWORDS:
                continue
            if token[:_DIAGNOSTIC_STEM_LEN] in ctx.input_text:
                continue
            found.add(token)
    return sorted(found)


def check_budget(chunk: GoldenChunk) -> list[Violation]:
    """The declared context budget: input estimate plus reserve within ``num_ctx``."""
    reserve = reserve_tokens_for(chunk.category)
    if chunk.budget.chunk_tokens_est + reserve > chunk.budget.num_ctx:
        return [
            Violation(
                "context_budget",
                f"{chunk.budget.chunk_tokens_est} + {reserve} > {chunk.budget.num_ctx}",
            )
        ]
    return []


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
        coach_note(session, week_id=upstream.week_id, provider=provider, processing_run_id=run_id)
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


def evaluate_chunk(
    chunk: GoldenChunk,
    *,
    provider: LLMProvider,
    models_config: ModelsConfig,
    profile: OpenAICompatibleProfile | None = None,
) -> ChunkOutcome:
    """Run one chunk on a throwaway database and check it.

    The provider is injected (a scripted mock in CI mode, the real
    OpenAI-compatible adapter in a live run); the layers, the store and the
    migrations are the real ones. Latency is measured around the layer call
    itself, without the database setup, and is compared with the manifest budget
    of the category.
    """
    if chunk.invariants.kind not in KIND_CHECKS:
        raise HarnessError(
            f"chunk {chunk.id} declares invariant kind {chunk.invariants.kind!r}, "
            "which no checker implements"
        )
    resolved_profile = (
        profile
        if profile is not None
        else resolve_phase2_profile(models_config, PROFILE_NAME_BY_CATEGORY[chunk.category])
    )
    outcome = ChunkOutcome(
        chunk_id=chunk.id,
        category=chunk.category,
        kind=chunk.invariants.kind,
        latency_budget_s=_latency_budget_for(chunk),
    )
    outcome.violations.extend(check_budget(chunk))

    input_json_before = canonical_input_json(chunk.input)
    with temp_database() as session:
        _seed_chunk_input(session, chunk)
        if chunk.category == "day" and not _collected_day_matches(session, chunk):
            raise HarnessError(
                f"chunk {chunk.id}: the seeded day does not read back as its declared input"
            )
        raw_state_before = _raw_state(session) if chunk.category in {"stt_error", "day"} else None

        started = time.perf_counter()
        _run_layer(session, chunk, profile=resolved_profile, provider=provider)
        outcome.latency_s = time.perf_counter() - started
        session.commit()

        if raw_state_before is not None and _raw_state(session) != raw_state_before:
            outcome.violations.append(
                Violation("raw_rows_unchanged", "a raw source row changed during the run")
            )
        payload = _latest_result_payload(session, chunk)

    model, contract_violations = _validate_output(chunk, payload)
    outcome.violations.extend(contract_violations)
    if model is not None:
        ctx = CheckContext(
            chunk=chunk,
            model=model,
            payload=payload,
            input_json_before=input_json_before,
            input_text=_input_content_text(chunk),
            input_numbers=_input_numbers(chunk),
        )
        for name in (*GENERIC_CHECKS, *KIND_CHECKS[chunk.invariants.kind]):
            outcome.violations.extend(CHECKERS[name](ctx))
        outcome.diagnostics = content_word_diagnostics(ctx)

    _record_measurements(outcome, chunk, resolved_profile, provider)
    _apply_latency_budget(outcome)
    return outcome


@lru_cache(maxsize=1)
def _latency_budgets() -> dict[str, float]:
    """Latency budget per category, as declared by the manifest."""
    budget = golden_manifest().latency_budget_s
    return {"stt_error": float(budget.stt_fix), "day": float(budget.daily)}


def _latency_budget_for(chunk: GoldenChunk) -> float | None:
    """Latency budget of a category, or ``None`` when the manifest sets none."""
    return _latency_budgets().get(chunk.category)


def _record_measurements(
    outcome: ChunkOutcome,
    chunk: GoldenChunk,
    profile: OpenAICompatibleProfile,
    provider: LLMProvider,
) -> None:
    """Record and check endpoint measurements, when the provider reports them."""
    requests = getattr(provider, "request_count", None)
    if isinstance(requests, int):
        outcome.requests = requests
    usage = getattr(provider, "last_usage", None)
    if not isinstance(usage, LLMUsage):
        return
    outcome.usage = usage
    if usage.input_tokens + profile.max_tokens > chunk.budget.num_ctx:
        outcome.violations.append(
            Violation(
                "measured_context_budget",
                f"measured {usage.input_tokens} input tokens + {profile.max_tokens} cap "
                f"> num_ctx {chunk.budget.num_ctx}",
            )
        )


def _apply_latency_budget(outcome: ChunkOutcome) -> None:
    budget = outcome.latency_budget_s
    if budget is not None and outcome.latency_s > budget:
        outcome.violations.append(
            Violation("latency_budget", f"{outcome.latency_s:.2f}s > {budget:.0f}s")
        )


def evaluate_all(
    chunks: Sequence[GoldenChunk],
    *,
    provider_factory: Callable[[GoldenChunk], LLMProvider],
    models_config: ModelsConfig,
) -> list[ChunkOutcome]:
    """Evaluate the selected chunks, one isolated database per chunk."""
    return [
        evaluate_chunk(
            chunk,
            provider=provider_factory(chunk),
            models_config=models_config,
            profile=resolve_phase2_profile(models_config, PROFILE_NAME_BY_CATEGORY[chunk.category]),
        )
        for chunk in chunks
    ]


# ---------------------------------------------------------------------------
# Providers: the CI mock and the live wrapper
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Deterministic provider: one fixed completion, recorded requests."""

    def __init__(self, payload: Mapping[str, object] | str) -> None:
        self._text = (
            payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        )
        self.requests: list[LLMRequest] = []

    @property
    def request_count(self) -> int:
        """Number of ``complete`` calls served (a week may need several)."""
        return len(self.requests)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self._text)


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
    """Wraps a provider and remembers the usage and the number of calls.

    The last reported :class:`~diary.domain.llm.models.LLMUsage` and the call
    count are metrics only — never content.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.request_count = 0
        self.last_usage: LLMUsage | None = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._provider.complete(request)
        self.request_count += 1
        if response.usage is not None:
            self.last_usage = response.usage
        return response


def ci_provider_factory(chunk: GoldenChunk) -> LLMProvider:
    """Provider factory of the CI mode (no network, no GPU, no API key)."""
    return ScriptedProvider(ci_payload(chunk))


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
    """Aggregate result of a harness run (metrics only, never content)."""

    total: int
    passed: int
    invariants_passed: int
    max_latency_by_category: dict[str, float]
    latency_budgets: dict[str, float | None]
    diagnostics_total: int


def summarize(outcomes: Sequence[ChunkOutcome]) -> Summary:
    """Aggregate outcomes into the numbers the definition of done talks about."""
    max_latency: dict[str, float] = {}
    budgets: dict[str, float | None] = {}
    for outcome in outcomes:
        max_latency[outcome.category] = max(
            max_latency.get(outcome.category, 0.0), outcome.latency_s
        )
        budgets.setdefault(outcome.category, outcome.latency_budget_s)
    invariants_passed = sum(
        1
        for outcome in outcomes
        if not [item for item in outcome.violations if item.check != "latency_budget"]
    )
    return Summary(
        total=len(outcomes),
        passed=sum(1 for outcome in outcomes if outcome.passed),
        invariants_passed=invariants_passed,
        max_latency_by_category=max_latency,
        latency_budgets=budgets,
        diagnostics_total=sum(len(outcome.diagnostics) for outcome in outcomes),
    )


def format_report(outcomes: Sequence[ChunkOutcome], manifest: GoldenManifest) -> str:
    """The human report: chunk ids, metrics and failed check names only.

    No generated text, no input text and no diary content appears here (AGENTS.md
    §7-8): only ids, categories, latency, pass/fail and check names.
    """
    lines: list[str] = []
    header = f"{'chunk':<34}{'category':<12}{'latency':>9}  {'result':<6} failed checks"
    lines.append(header)
    lines.append("-" * len(header))
    for outcome in outcomes:
        failed = ", ".join(outcome.failed_checks) if outcome.failed_checks else "-"
        lines.append(
            f"{outcome.chunk_id:<34}{outcome.category:<12}"
            f"{outcome.latency_s:>8.2f}s  {'PASS' if outcome.passed else 'FAIL':<6} {failed}"
        )
    summary = summarize(outcomes)
    lines.append("")
    lines.append(
        f"chunks: {summary.passed}/{summary.total} passed "
        f"(invariants only: {summary.invariants_passed}/{summary.total}); "
        f"threshold {manifest.threshold.pass_count}/{manifest.threshold.total}"
    )
    for category in sorted(summary.max_latency_by_category):
        budget = summary.latency_budgets.get(category)
        verified = "n/a" if budget is None else f"< {budget:.0f}s"
        lines.append(
            f"latency {category}: max {summary.max_latency_by_category[category]:.2f}s "
            f"(budget {verified})"
        )
    lines.append(
        f"diagnostics (informational, not a failure): {summary.diagnostics_total} token(s)"
    )
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
        help="require every selected chunk to pass, not just the manifest threshold",
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
    outcomes = evaluate_all(chunks, provider_factory=factory, models_config=models_config)
    print(format_report(outcomes, manifest))
    required = len(chunks) if (args.strict or args.chunk) else manifest.threshold.pass_count
    summary = summarize(outcomes)
    if summary.passed < required:
        print(f"FAILED: {summary.passed}/{len(chunks)} passed, {required} required")
        return 1
    print(f"OK: {summary.passed}/{len(chunks)} passed")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
