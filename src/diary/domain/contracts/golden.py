"""Machine-checkable format of the golden eval set (task T00; harness T10).

``eval/golden/`` holds ten synthetic Russian chunks — three STT corrections,
three day summaries, two weekly summaries and two coaching notes — plus a
manifest. This module is the **single definition of that format**: the T00 test
and the future T10 harness both load the set through it, so the on-disk JSON and
the code that consumes it can never drift apart.

Chunk shape (one file per chunk, ``eval/golden/chunks/<id>.json``)::

    id, contract, category, lang, synthetic, origin, budget, input, invariants

- ``contract`` — which phase-2 contract the chunk exercises; ``category`` — the
  grouping used for the 3/3/2/2 budget. The two are linked by
  :data:`CONTRACT_BY_CATEGORY` and validated on load.
- ``lang`` — fixed to ``ru`` (the project's primary language).
- ``synthetic`` — must be ``true``: the golden set never contains diary
  content, only authored synthetic text (AGENTS.md §7).
- ``origin`` — where the chunk came from; must be in :data:`ALLOWED_ORIGINS`.
- ``input`` — the *upstream* contract objects the layer consumes, never the
  expected output (the set carries invariants, not reference strings):
  ``raw_entry[]`` for STT and day chunks, ``daily_summary[]`` for week chunks,
  one ``weekly_summary`` for coaching chunks.
- ``invariants`` — the machine-checkable expectations for the produced output.
  The generic ones (output validates its contract, the input stays unchanged,
  forbidden lexemes absent) hold for every chunk; the category-specific ones
  are listed in the invariant models below.

Context budget: ``budget.num_ctx`` is 8192 by default and 16384 for a weekly
chunk whose input has to be fed in several pieces
(``chunks_planned > 1``) — the rule is enforced by
:func:`num_ctx_for` and validated on load. ``budget.chunk_chars`` is the length
of the canonical serialized ``input`` (:func:`canonical_input_json`) and
``budget.chunk_tokens_est`` its token estimate
(:func:`estimate_tokens`, ~:data:`CHARS_PER_TOKEN_EST` characters per token for
Russian). T10 asserts ``chunk_tokens_est <= num_ctx - reserve``.
``num_ctx`` itself is an eval-contract rule only: it is **not** part of the
model profiles yet (task T02/T04-T07 wire it into the profiles; recorded in
ADR-007).

Blacklist semantics: ``invariants.forbidden_lexemes`` is checked against the
**generated text fields** of the output (``text``, ``facts``, ``themes``,
``open_questions``, ``rollup``, ``observations``, ``hypotheses``,
``experiment_7d``) — not against fixed constant fields (``disclaimer``), whose
wording is mandated by the contract and disclaims rather than asserts.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from diary.core.settings import PROJECT_ROOT
from diary.domain.contracts.coach_note import COACH_DISCLAIMER
from diary.domain.contracts.daily_summary import DailySummary, EnergyLevel
from diary.domain.contracts.raw_entry import RawEntry
from diary.domain.contracts.weekly_summary import WEEK_ID_PATTERN, WeeklySummary

#: Directory of the golden set inside the repository.
GOLDEN_DIR = PROJECT_ROOT / "eval" / "golden"

#: Subdirectory holding one JSON file per chunk.
GOLDEN_CHUNKS_DIRNAME = "chunks"

#: File name of the manifest inside :data:`GOLDEN_DIR`.
GOLDEN_MANIFEST_NAME = "manifest.json"

#: Default context window of an eval chunk (tokens).
DEFAULT_NUM_CTX = 8192

#: Context window of a weekly chunk whose input does not fit in one piece.
WEEKLY_SPLIT_NUM_CTX = 16384

#: Token estimate for Russian text: approximately this many characters per
#: token. Deliberately a documented constant so the estimate is reproducible.
CHARS_PER_TOKEN_EST = 2.5

#: Lexemes that must not appear in any generated text field of a golden
#: output: diagnostic labels, treatment plans and prescriptive advice
#: (AGENTS.md §16; ADR-006). The check excludes fixed constant fields.
DEFAULT_FORBIDDEN_LEXEMES: tuple[str, ...] = ("диагноз", "лечение", "тебе следует")

#: Accepted values of ``GoldenChunk.origin``. Extend together with the eval
#: README when a chunk may come from a source other than authoring.
ALLOWED_ORIGINS: frozenset[str] = frozenset({"authored-for-eval"})

#: Names of the phase-2 contracts a golden chunk may exercise.
ContractName = Literal[
    "raw_entry",
    "stt_corrected",
    "daily_summary",
    "weekly_summary",
    "coach_note",
]

#: Grouping of golden chunks used for the pass threshold and the 3/3/2/2 mix.
ChunkCategory = Literal["stt_error", "day", "week", "wellbeing"]

#: Category → the contract its chunks exercise (ADR-007: the ``wellbeing``
#: category is written against ``coach_note``; no fifth contract is added).
CONTRACT_BY_CATEGORY: dict[str, str] = {
    "stt_error": "stt_corrected",
    "day": "daily_summary",
    "week": "weekly_summary",
    "wellbeing": "coach_note",
}

#: Contract names of the phase-2 family (schema files in ``schemas/``).
CONTRACT_NAMES: frozenset[str] = frozenset(
    {"raw_entry", "stt_corrected", "daily_summary", "weekly_summary", "coach_note"}
)


def num_ctx_for(category: str, chunks_planned: int) -> int:
    """Context budget of a chunk per the eval-contract rule.

    ``8192`` by default and ``16384`` for a weekly chunk whose input must be
    split into more than one piece (``chunks_planned > 1``).
    """
    if category == "week" and chunks_planned > 1:
        return WEEKLY_SPLIT_NUM_CTX
    return DEFAULT_NUM_CTX


# ---------------------------------------------------------------------------
# Manifest models
# ---------------------------------------------------------------------------


class GoldenCounts(BaseModel):
    """Required number of chunks per category (3/3/2/2 for this set)."""

    model_config = ConfigDict(extra="forbid")

    stt_error: int = Field(ge=0)
    day: int = Field(ge=0)
    week: int = Field(ge=0)
    wellbeing: int = Field(ge=0)


class GoldenThreshold(BaseModel):
    """Pass threshold of the golden run (``{"pass": 8, "total": 10}``)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    #: Number of chunks that must pass (JSON key ``pass``, a Python keyword).
    pass_count: int = Field(alias="pass", ge=1)

    #: Total number of chunks in the set.
    total: int = Field(ge=1)

    @model_validator(mode="after")
    def _pass_fits_total(self) -> GoldenThreshold:
        if self.pass_count > self.total:
            raise ValueError("threshold.pass must not exceed threshold.total")
        return self


class GoldenBudgetDefault(BaseModel):
    """Default per-chunk context budget of the set."""

    model_config = ConfigDict(extra="forbid")

    num_ctx: int = Field(ge=1024)


class GoldenLatencyBudget(BaseModel):
    """Latency budget of the T10 harness, in seconds."""

    model_config = ConfigDict(extra="forbid")

    #: One STT correction.
    stt_fix: int = Field(ge=1)

    #: One day summary.
    daily: int = Field(ge=1)


class GoldenManifest(BaseModel):
    """The golden-set manifest: mix, thresholds and the list of chunk ids."""

    model_config = ConfigDict(extra="forbid")

    #: Format version of the manifest and of the chunk files.
    version: int = Field(ge=1)

    #: Language of every chunk (fixed: the project's primary language).
    lang: Literal["ru"]

    #: Marks the whole set as synthetic (never real diary content).
    synthetic: Literal[True]

    #: Pass threshold used by the T10 harness.
    threshold: GoldenThreshold

    #: Required mix of chunk categories.
    counts: GoldenCounts

    #: Default context budget used when a chunk does not override it.
    budget_default: GoldenBudgetDefault

    #: Latency budget of the harness.
    latency_budget_s: GoldenLatencyBudget

    #: Chunk ids, in harness order; the file is
    #: ``<GOLDEN_DIR>/chunks/<id>.json``.
    chunks: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _chunk_list_is_consistent(self) -> GoldenManifest:
        if len(set(self.chunks)) != len(self.chunks):
            raise ValueError("manifest.chunks must not repeat an id")
        declared = sum(self.counts.model_dump().values())
        if declared != len(self.chunks):
            raise ValueError(
                f"manifest.counts sum to {declared} but manifest.chunks lists "
                f"{len(self.chunks)} chunks"
            )
        if self.threshold.total != len(self.chunks):
            raise ValueError(
                f"threshold.total is {self.threshold.total} but manifest.chunks "
                f"lists {len(self.chunks)} chunks"
            )
        return self


# ---------------------------------------------------------------------------
# Per-contract invariants
# ---------------------------------------------------------------------------


class GoldenInvariants(BaseModel):
    """Generic, machine-checkable expectations for a chunk's output.

    ``expected_energy`` in :class:`DailyInvariants` is an expectation, not an
    absence of a check — see the subclass docstrings.
    """

    model_config = ConfigDict(extra="forbid")

    #: The output must validate against the JSON Schema of its contract.
    check_output_contract: Literal[True] = True

    #: The serialized input must be unchanged after the run.
    check_input_unchanged: Literal[True] = True

    #: Lexemes that must not occur in the generated text fields (see the module
    #: docstring for exactly which fields are checked).
    forbidden_lexemes: list[str] = Field(default_factory=lambda: list(DEFAULT_FORBIDDEN_LEXEMES))


class SttInvariants(GoldenInvariants):
    """Expectations for an ``stt_corrected`` chunk."""

    #: Discriminator; always equals the chunk's contract name.
    kind: Literal["stt_corrected"]

    #: Every number in the output occurs in the input, and every proper name in
    #: the output occurs in the input (compared case-insensitively): the
    #: correction may fix slurred *common* words, punctuation and casing, but
    #: must not introduce a name or a number that is not in the source.
    names_and_numbers_grounded_in_input: Literal[True] = True

    #: Every ``uncertain`` entry is a verbatim substring of ``text`` (also
    #: enforced by the contract itself).
    uncertain_values_are_verbatim_substrings_of_text: Literal[True] = True


class DailyInvariants(GoldenInvariants):
    """Expectations for a ``daily_summary`` chunk."""

    kind: Literal["daily_summary"]

    #: The summary must carry exactly this date (the local day of its entries).
    expected_date: date

    #: ``energy`` expected in the output. ``None`` is a positive expectation:
    #: the source of this chunk never states the day's energy, so the only
    #: valid output is ``energy: null`` — inventing a level fails the chunk.
    expected_energy: EnergyLevel | None

    #: ``facts`` is never empty.
    facts_non_empty: Literal[True] = True

    #: Every numeric token of the output occurs in the input (grounding).
    numbers_grounded_in_input: Literal[True] = True


class WeekInvariants(GoldenInvariants):
    """Expectations for a ``weekly_summary`` chunk."""

    kind: Literal["weekly_summary"]

    #: The rollup must carry exactly this ISO week id.
    expected_week_id: str = Field(pattern=WEEK_ID_PATTERN)

    #: ``set(daily_ids)`` must be a subset of the day summaries in the input.
    daily_ids_subset_of_input: Literal[True] = True

    #: ``rollup`` is never empty.
    rollup_non_empty: Literal[True] = True


class CoachInvariants(GoldenInvariants):
    """Expectations for a ``coach_note`` chunk (category ``wellbeing``)."""

    kind: Literal["coach_note"]

    #: Exactly three observations.
    observations_count: Literal[3] = 3

    #: Exactly two hypotheses.
    hypotheses_count: Literal[2] = 2

    #: The experiment states its 7-day horizon.
    experiment_horizon_days: Literal[7] = 7

    #: ``disclaimer`` equals :data:`COACH_DISCLAIMER` verbatim.
    disclaimer_equals_coach_disclaimer: Literal[True] = True


#: Any per-category invariant block, discriminated by ``kind``.
GoldenInvariantSet = Annotated[
    SttInvariants | DailyInvariants | WeekInvariants | CoachInvariants,
    Field(discriminator="kind"),
]

#: Any validated chunk input: upstream contract objects only.
GoldenInput = list[RawEntry] | list[DailySummary] | WeeklySummary


# ---------------------------------------------------------------------------
# Chunk models
# ---------------------------------------------------------------------------


class GoldenBudget(BaseModel):
    """Context budget of one chunk."""

    model_config = ConfigDict(extra="forbid")

    #: Context window in tokens (see :func:`num_ctx_for`).
    num_ctx: int = Field(ge=1024)

    #: Length in characters of the canonical serialized ``input``.
    chunk_chars: int = Field(ge=1)

    #: How many pieces the input has to be fed in.
    chunks_planned: int = Field(ge=1)

    #: Token estimate of the input (see :func:`estimate_tokens`).
    chunk_tokens_est: int = Field(ge=1)


class GoldenChunk(BaseModel):
    """One golden chunk: synthetic upstream input plus output invariants."""

    model_config = ConfigDict(extra="forbid")

    #: Stable chunk id; also the file name (``chunks/<id>.json``).
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")

    #: Contract the chunk exercises (one of :data:`CONTRACT_NAMES`).
    contract: ContractName

    #: Category used for the set mix and the threshold.
    category: ChunkCategory

    #: Language of the chunk (fixed: Russian).
    lang: Literal["ru"]

    #: Must be ``true``: golden chunks are authored synthetic text.
    synthetic: Literal[True]

    #: Provenance of the chunk; must be in :data:`ALLOWED_ORIGINS`.
    origin: str

    #: Context budget of this chunk.
    budget: GoldenBudget

    #: Upstream contract objects the layer consumes.
    input: GoldenInput

    #: Machine-checkable expectations for the produced output.
    invariants: GoldenInvariantSet

    @model_validator(mode="after")
    def _chunk_is_self_consistent(self) -> GoldenChunk:
        if self.origin not in ALLOWED_ORIGINS:
            allowed = ", ".join(sorted(ALLOWED_ORIGINS))
            raise ValueError(f"origin {self.origin!r} is not allowed (allowed: {allowed})")
        expected_contract = CONTRACT_BY_CATEGORY[self.category]
        if self.contract != expected_contract:
            raise ValueError(
                f"category {self.category!r} must use contract "
                f"{expected_contract!r}, not {self.contract!r}"
            )
        if self.invariants.kind != self.contract:
            raise ValueError(
                f"invariants kind {self.invariants.kind!r} does not match "
                f"contract {self.contract!r}"
            )
        if self.category in {"stt_error", "day"}:
            if not isinstance(self.input, list) or not all(
                isinstance(item, RawEntry) for item in self.input
            ):
                raise ValueError(f"input of a {self.category} chunk must be a list of raw_entry")
            if not self.input:
                raise ValueError(f"input of a {self.category} chunk must not be empty")
        elif self.category == "week":
            if not isinstance(self.input, list) or not all(
                isinstance(item, DailySummary) for item in self.input
            ):
                raise ValueError("input of a week chunk must be a list of daily_summary")
            if not self.input:
                raise ValueError("input of a week chunk must not be empty")
        elif not isinstance(self.input, WeeklySummary):
            raise ValueError("input of a wellbeing chunk must be one weekly_summary")
        expected_num_ctx = num_ctx_for(self.category, self.budget.chunks_planned)
        if self.budget.num_ctx != expected_num_ctx:
            raise ValueError(
                f"budget.num_ctx for a {self.category} chunk with "
                f"chunks_planned={self.budget.chunks_planned} must be "
                f"{expected_num_ctx}, not {self.budget.num_ctx}"
            )
        return self


# ---------------------------------------------------------------------------
# Budget helpers (shared with the T00 test and the T10 harness)
# ---------------------------------------------------------------------------


def input_payload(chunk_input: GoldenInput) -> object:
    """JSON-serializable payload of a validated chunk input."""
    if isinstance(chunk_input, list):
        return [item.model_dump(mode="json") for item in chunk_input]
    return chunk_input.model_dump(mode="json")


def canonical_input_json(chunk_input: GoldenInput) -> str:
    """Canonical serialization of a chunk input (compact, sorted, unescaped).

    ``budget.chunk_chars`` is the length of exactly this string.
    """
    payload = input_payload(chunk_input)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def input_char_count(chunk: GoldenChunk) -> int:
    """Length in characters of the chunk's serialized input."""
    return len(canonical_input_json(chunk.input))


def estimate_tokens(char_count: int) -> int:
    """Token estimate of ``char_count`` characters of Russian text."""
    return math.ceil(char_count / CHARS_PER_TOKEN_EST)


def coach_disclaimer() -> str:
    """The fixed layer-5 disclaimer (re-exported for the harness)."""
    return COACH_DISCLAIMER


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def manifest_path(base_dir: Path | str | None = None) -> Path:
    """Path of the manifest inside ``base_dir`` (default: the repository set)."""
    return Path(base_dir or GOLDEN_DIR) / GOLDEN_MANIFEST_NAME


def chunk_path(chunk_id: str, *, base_dir: Path | str | None = None) -> Path:
    """Path of one chunk file by its id (``chunks/<id>.json``)."""
    return Path(base_dir or GOLDEN_DIR) / GOLDEN_CHUNKS_DIRNAME / f"{chunk_id}.json"


def load_manifest(path: Path | str | None = None) -> GoldenManifest:
    """Load and validate the golden manifest (default: ``eval/golden``)."""
    manifest_file = Path(path) if path is not None else manifest_path()
    return GoldenManifest.model_validate(json.loads(manifest_file.read_text(encoding="utf-8")))


def load_chunk(path: Path | str) -> GoldenChunk:
    """Load and validate one golden chunk file."""
    chunk_file = Path(path)
    return GoldenChunk.model_validate(json.loads(chunk_file.read_text(encoding="utf-8")))
