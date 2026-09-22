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
**generated text fields** of the output (:data:`GENERATED_TEXT_FIELDS`) up to
fixed constant fields (``disclaimer``), whose wording is mandated by the
contract and disclaims rather than asserts. Three eval-contract checks are
always on and cannot be switched off by a chunk file:

- ``no_advice_or_judgement`` — advice, praise, evaluation and morality lexemes
  (:data:`ADVICE_EVALUATION_LEXEMES`) are absent from the generated fields;
- ``no_causal_claims`` — causal connectors and asserted cause-effect phrasing
  (:data:`CAUSAL_CONNECTOR_LEXEMES`) are absent from the **assertive** generated
  fields (:data:`ASSERTIVE_TEXT_FIELDS`; ``open_questions`` is excluded on
  purpose — a question marks an open point instead of asserting a cause, and
  the set itself carries ``из-за чего сбился сон`` as a legitimate open
  question);
- ``no_new_facts`` — numbers, numeral words (:data:`NUMERAL_WORDS`) and proper
  names of the output are grounded in the input.

Documented limit of ``no_new_facts``: it is exact for digits, numeral words and
proper names, but free-form content words are **not** machine-checkable without
a semantic model — a paraphrase ("трудовой" for "работа") and an invention
("плохое самочувствие") are indistinguishable to a lexical check. The harness
therefore reports ungrounded long content tokens as *diagnostics* (never as a
failure) instead of pretending to verify them.

Context budget: ``chunk_tokens_est + reserve_tokens_for(category) <= num_ctx``
where the reserve is the fixed overhead of the layer's call — its versioned
prompt plus the generation cap of its profile (:data:`BUDGET_RESERVE_TOKENS`).
The prompt half of that reserve is *recomputed from the template file* by the
harness on every run (T20): :data:`LAYER_PROMPT_TOKENS_EST` is the declared
expectation a test pins against the file, not the number the check trusts.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from diary.core.settings import PROJECT_ROOT
from diary.domain.contracts.coach_note import COACH_DISCLAIMER, EXPERIMENT_HORIZON_DAYS
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
#:
#: This trio is the *base* blacklist embedded in every chunk file. It is kept
#: byte-for-byte because the chunk files carry it explicitly; the harness adds
#: :data:`ADVICE_EVALUATION_LEXEMES` on top of it for every chunk, so the
#: effective check is the union.
DEFAULT_FORBIDDEN_LEXEMES: tuple[str, ...] = ("диагноз", "лечение", "тебе следует")

#: Eval-contract baseline of advice / praise / evaluation / morality lexemes.
#: Added by T10 because ``DEFAULT_FORBIDDEN_LEXEMES`` alone accepted
#: «ты молодец», «правильно сделала», «лучше было бы», «стоило бы» (the gap
#: recorded by the T05 acceptance). Each entry is a **deliberately narrow
#: phrase** rather than a bare root: a summary may legitimately contain
#: «как следует», «правильный ответ» and similar, so the check only fires on
#: wording that is advice, praise or a value judgement. The check is a
#: substring test over the generated text fields, case-insensitive.
ADVICE_EVALUATION_LEXEMES: tuple[str, ...] = (
    # praise / evaluation of the person
    "ты молодец",
    "молодец",
    "умница",
    "правильно сделала",
    "правильно поступила",
    "правильно решила",
    # counterfactual recommendation
    "лучше было бы",
    "лучше бы",
    "стоило бы",
    "стоит попробовать",
    "стоит начать",
    # direct prescription
    "тебе следует",
    "рекомендую",
    "рекомендуется",
    "советую",
    "советует",
    "попробуй",
    "попробуйте",
    # morality / obligation
    "ты должен",
    "ты должна",
    "надо было",
    "нужно было",
)

#: Eval-contract baseline of causal connectors and asserted cause-effect
#: phrasing (T10; the T06 acceptance found «что привело к плохому
#: самочувствию», «из-за ночных пробуждений, что повлияло на энергию» and the
#: T07 acceptance found «привела к», «повлияла на» in live output while the
#: prompts forbid them). Checked against the assertive generated fields only
#: (:data:`ASSERTIVE_TEXT_FIELDS`), case-insensitive substring test. Formats
#: with and without ``ё`` are both listed because models mix them.
CAUSAL_CONNECTOR_LEXEMES: tuple[str, ...] = (
    "из-за",
    "потому что",
    "поэтому",
    "оттого",
    "так как",
    "поскольку",
    "в результате",
    "вследствие",
    "причин",
    "привел",
    "привёл",
    "привело к",
    "привела к",
    "привели к",
    "повлиял",
    "повлияла",
    "повлияло",
    "повлияли",
    "влияние",
)

#: Cardinal numerals as words, mapped to their digit value. Used by the
#: ``no_new_facts`` check: the golden set writes every quantity as digits
#: (authoring rule 2), so a numeral word in the output is checked by
#: normalizing it to digits and requiring that digit to occur in the input —
#: this is exactly how the T07 finding «два созвона при трёх во входе» becomes
#: machine-checkable. ``один/одна/одно`` are deliberately excluded (they are
#: routinely non-counting: «один из», «однажды»), as are ``пара``, ``полтора``,
#: ``несколько`` and ``много`` (vague, not numerals).
NUMERAL_WORDS: dict[str, int] = {
    "два": 2,
    "две": 2,
    "двух": 2,
    "двум": 2,
    "двумя": 2,
    "три": 3,
    "трёх": 3,
    "трех": 3,
    "трём": 3,
    "трем": 3,
    "тремя": 3,
    "четыре": 4,
    "четырёх": 4,
    "четырех": 4,
    "четырём": 4,
    "четырем": 4,
    "четырьмя": 4,
    "пять": 5,
    "пяти": 5,
    "пятью": 5,
    "шесть": 6,
    "шести": 6,
    "шестью": 6,
    "семь": 7,
    "семи": 7,
    "семью": 7,
    "восемь": 8,
    "восьми": 8,
    "восемью": 8,
    "девять": 9,
    "девяти": 9,
    "десять": 10,
    "десяти": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
    "тысяча": 1000,
    "тысячи": 1000,
    "тысяч": 1000,
}

#: Declared prompt overhead of each layer, in estimated tokens — the
#: **expectation**, not the source of truth. T20 stopped treating a
#: hand-maintained number as the authority: the harness recomputes the estimate
#: from the versioned prompt template at run time (the same
#: :data:`CHARS_PER_TOKEN_EST` estimate as the chunk input) and a test in
#: ``tests/integration/test_eval_harness.py`` fails when these numbers and the
#: files disagree in either direction. A prompt that grows must be re-measured
#: and the budget re-decided; it can no longer be absorbed silently.
#:
#: The values are the file-derived estimates of the prompt versions the layers
#: load today (T20 measurement). The earlier numbers were stale by up to two
#: times (``week`` remembered 1629 against 3132, ``wellbeing`` 1968 against
#: 4113), which made the declared budget check optimistic exactly where the
#: reserve was tight.
LAYER_PROMPT_TOKENS_EST: dict[str, int] = {
    "stt_error": 1509,  # prompts/stt_fix/v1.md
    "day": 2124,  # prompts/sum_daily/v1.md
    "week": 3132,  # prompts/sum_weekly/v2.md
    "wellbeing": 4113,  # prompts/coach_note/v3.md
}

#: Generation cap of the profile each category runs under, mirroring
#: ``configs/models.example.yaml`` (``diary_stt`` 3072, ``diary_sum`` and
#: ``diary_week`` 4096, ``diary_coach`` 2048). Kept next to the reserve so a
#: configuration change is a one-line diff; a test cross-checks it against the
#: example configuration.
#:
#: ``wellbeing`` left the blanket 4096 in T20 because with the honest prompt
#: estimate the coach envelope did not close: 4113 + 4096 = 8209 estimated tokens
#: is **more than the 8192 window** before a single character of the week rollup,
#: so the old check passed only because the constant was understated. The coach
#: contract asks for 3 observations, 2 hypotheses and one short experiment — the
#: live run of T07 produced ~200 tokens — so 2048 keeps an order of magnitude of
#: headroom over the observed output and leaves 2031 estimated tokens for the
#: input. The layer enforces the same envelope at run time before the call.
PROFILE_MAX_TOKENS: dict[str, int] = {
    "stt_error": 3072,
    "day": 4096,
    "week": 4096,
    "wellbeing": 2048,
}

#: Tokens the harness reserves on top of ``chunk_tokens_est`` inside
#: ``num_ctx``: the layer's versioned prompt plus the generation cap of its
#: profile. Since T20 the reserve is **derived from the prompt file at run
#: time** — the harness reads the template the layer itself loads and estimates
#: it with :func:`estimate_tokens` — while this mapping stays as the declared
#: expectation that a test cross-checks against the file. A prompt can therefore
#: grow loudly instead of eating the window unnoticed.
#:
#: Source of the declared numbers: the T20 measurement of the prompt versions
#: the layers load today (``estimate_tokens`` over the template file), with the
#: live runs of 2026-09-20 as the sanity check of the estimate's pessimism: an
#: input of 2420 estimated tokens measured 2324 for a day chunk and 3819
#: estimated measured ~2900 for a week chunk, so the character-based estimate
#: runs 4-24 % above what the model really counts. That is the safe direction —
#: an envelope that holds under the estimate holds in the window.
#:
#: The check ``chunk_tokens_est + reserve <= num_ctx`` therefore bounds the
#: whole call, not just the payload, and the real run additionally compares the
#: endpoint-reported ``input_tokens`` against the same budget.
BUDGET_RESERVE_TOKENS: dict[str, int] = {
    category: LAYER_PROMPT_TOKENS_EST[category] + PROFILE_MAX_TOKENS[category]
    for category in LAYER_PROMPT_TOKENS_EST
}

#: Generated text fields of each contract: the fields the blacklist checks are
#: applied to (fixed constant fields such as ``disclaimer`` are excluded).
GENERATED_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "stt_corrected": ("text",),
    "daily_summary": ("facts", "themes", "open_questions"),
    "weekly_summary": ("rollup",),
    "coach_note": ("observations", "hypotheses", "experiment_7d"),
}

#: Assertive generated fields of each contract — the subset that *states*
#: something, used by the causal check. ``open_questions`` is excluded: a
#: question marks an open point instead of asserting a cause, and the golden
#: set itself carries «Непонятно, из-за чего сбился сон.» as a legitimate open
#: question of the week-01 input.
ASSERTIVE_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "stt_corrected": ("text",),
    "daily_summary": ("facts", "themes"),
    "weekly_summary": ("rollup",),
    "coach_note": ("observations", "hypotheses", "experiment_7d"),
}

#: Input fields whose values are bookkeeping rather than content, and therefore
#: neither ground a number of the generated text nor count as source wording:
#: identifiers and server-generated audio paths (a UUID is not a fact) and the
#: **moment/anchor** fields — ``ts``, ``date``, ``week_id``, ``daily_ids``,
#: ``summary_id``, ``local_date``, ``local_tz``, ``created_at``. A moment says
#: *where a record sits in time*, not what the diary says: before T32 the digits
#: of an entry's ``ts`` (or of a week's ``daily_ids``) grounded a number the
#: source never mentions, so a claimed «ложиться до 23:00» passed on a week that
#: merely contained the 23rd. T32 measured the class on the golden set (every
#: chunk accepted such a number) and narrowed the source to record content; the
#: numbers are in ``docs/tasklist-phase2.md`` §16 and ``ADR-012``.
NON_CONTENT_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "audio_path",
        "ts",
        "date",
        "week_id",
        "daily_ids",
        "summary_id",
        "local_date",
        "local_tz",
        "created_at",
    }
)

#: Contracts whose generated text is a *correction of the input* rather than a
#: newly written text (``stt_corrected``). For them the advice/causal lexeme
#: checks only fire on wording the layer **introduced**: a faithful STT
#: correction must keep the speaker's words (T04 DoD: «не сглаживать эмоции»,
#: «не улучшать стиль»), so a lexeme that is already in the source is not a
#: violation — while the same lexeme appearing out of nowhere is. The
#: generative layers (day/week/coach) get no such exemption: their prompts
#: forbid advice and causes outright.
LEXEME_FIDELITY_CONTRACTS: frozenset[str] = frozenset({"stt_corrected"})

#: Digits a generated field is contractually required to carry, exempt from the
#: ``no_new_facts`` grounding check for that field. The only such digit today is
#: the 7-day horizon of the coaching experiment: the contract demands the digit
#: ``7`` in ``experiment_7d`` (checked by ``experiment_horizon_days``), and the
#: upstream week rollup is not required to mention it.
CONTRACT_CONSTANT_DIGITS: dict[tuple[str, str], frozenset[str]] = {
    ("coach_note", "experiment_7d"): frozenset({str(EXPERIMENT_HORIZON_DAYS)}),
}

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
    """Latency budget of the T10 harness, in seconds — **per inference route**.

    T21 stopped treating the budget as one number for the whole run: the STT and
    day layers run on the GPU tags, while the weekly rollup runs on the CPU twin
    ``diary-sum-cpu`` (the hybrid decision of T13b, ``docs/device-choice.md``),
    and a GPU-calibrated budget is unreachable there. A category without a
    declared budget (the coaching note, which is launched explicitly and rarely)
    is measured and reported as ``n/a`` instead of being silently compared with
    a foreign number.
    """

    model_config = ConfigDict(extra="forbid")

    #: One STT correction (GPU, ``diary-stt``).
    stt_fix: int = Field(ge=1)

    #: One day summary (GPU, ``diary-sum``).
    daily: int = Field(ge=1)

    #: One weekly rollup (CPU, ``diary-sum-cpu``). Deliberately much looser than
    #: the GPU budgets: the device-choice measurements put a CPU week at up to
    #: 126.5 s, and the project owner's own bound for a batch operation is «до 3
    #: минут». The value is a decision, not a measurement rounded up.
    weekly: int = Field(ge=1)


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
    #: docstring for exactly which fields are checked). The harness checks the
    #: union of this list and :data:`ADVICE_EVALUATION_LEXEMES`.
    forbidden_lexemes: list[str] = Field(default_factory=lambda: list(DEFAULT_FORBIDDEN_LEXEMES))

    #: No advice, praise, evaluation or morality in the generated text fields;
    #: the check uses :data:`ADVICE_EVALUATION_LEXEMES`. Always ``True``: a
    #: chunk file can add lexemes but cannot switch the check off.
    no_advice_or_judgement: Literal[True] = True

    #: No causal connector or asserted cause-effect phrasing in the assertive
    #: generated text fields; the check uses :data:`CAUSAL_CONNECTOR_LEXEMES`
    #: and :data:`ASSERTIVE_TEXT_FIELDS`. Always ``True``.
    no_causal_claims: Literal[True] = True

    #: No facts beyond the input: digits, numeral words
    #: (:data:`NUMERAL_WORDS`) and proper names of the output are grounded in
    #: the input. Always ``True``; free-form content words are reported as
    #: diagnostics instead (see the module docstring for why).
    no_new_facts: Literal[True] = True


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


def reserve_tokens_for(category: str) -> int:
    """The **declared** reserve of a category: its prompt estimate plus the cap.

    This is the expectation, not the number the harness checks with: since T20 the
    harness derives the prompt half from the template file itself
    (``eval/run.py: prompt_tokens_est_for``) and compares the two here through
    ``tests/integration/test_eval_harness.py``. The declared value is what the
    documents and the chunk budgets talk about, so it stays available — see
    :data:`BUDGET_RESERVE_TOKENS` for the sources. A category without a reserve is
    a harness error rather than a silently unchecked budget.
    """
    try:
        return BUDGET_RESERVE_TOKENS[category]
    except KeyError as exc:
        raise ValueError(f"no context-budget reserve is defined for category {category!r}") from exc


def generated_text_fields(contract: str) -> tuple[str, ...]:
    """Generated text fields of ``contract`` (blacklist check surface)."""
    try:
        return GENERATED_TEXT_FIELDS[contract]
    except KeyError as exc:
        raise ValueError(f"no generated-text field list is defined for {contract!r}") from exc


def assertive_text_fields(contract: str) -> tuple[str, ...]:
    """Assertive generated fields of ``contract`` (causal check surface)."""
    try:
        return ASSERTIVE_TEXT_FIELDS[contract]
    except KeyError as exc:
        raise ValueError(f"no assertive-text field list is defined for {contract!r}") from exc


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
