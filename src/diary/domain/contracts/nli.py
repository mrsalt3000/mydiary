"""NLI-based faithfulness rules for the summary layers (T24a).

**What this module is:** the *rules* of NLI validation — how a stored summary is
split into claims, how a judge's probabilities become a verdict, and which
verdicts matter. **What it is not:** a model. The judge is injected (a callable
returning three probabilities per claim), so the rules are exercised
deterministically and offline with a scripted judge in tests and CI, while
production uses the ONNX cross-encoder of :mod:`diary.domain.nli.onnx_judge` —
one implementation of the rules for both sides, the pattern of T17
(``contracts/invariants.py``).

**Why it exists.** The phase-2 acceptance (T12) named the gap: content words in
free form are not machine-checkable, so an invented retelling («плохое
самочувствие») and an inverted meaning reach the diary as long as no digit, no
proper name and no lexicon entry is touched. NLI asks exactly the missing
question — does the source support the claim.

**Measured basis** (``docs/nli-validation-analysis.md`` §3, 2026-09-21):

* the instrument flags the documented defect classes — an inversion of meaning
  and a swapped number — and on a real generated day summary gives six of six
  claims entailment with **zero false alarms**;
* it does **not** catch a fabrication that contradicts nothing (entailment 0.53
  on the very «плохое самочувствие» example), and a rule «everything must be
  entailed» would reject a legitimate generalisation («плавал в бассейне» →
  «тренировка по плаванию» is neutral to a strict reader);
* therefore only the contradiction rule is destined for a runtime gate (T24b),
  while the support rule stays a diagnostic signal reported by the harness.

**The premise is split into sentences and the verdict is aggregated in a fixed
order.** The model is fine-tuned «to predict the logical relationship between two
short texts» on sentence-pair corpora (SNLI, MNLI, SICK, FEVER); a whole day fed
in as a single premise is outside what it was trained for, and the artefact is
measurable — a source entry compared with the day it belongs to reads as a
contradiction 0.79, while the same entry against itself reads as entailment 0.98.
So each claim is compared with **every sentence of the source**, and the verdict
is taken from the sentence that governs it: a claim that any sentence fully
entails is *supported*, whatever the other sentences would say; a claim no
sentence supports is judged by its strongest contradiction; otherwise the claim
is neutral. The governing and supporting sentence indices travel with the
judgement as provenance — indices, never text (AGENTS.md §7-8).

**Two named rules, never one number** (the discipline of T18):

* ``no_contradiction_with_source`` — a **propositional** claim (see
  :data:`PROPOSITIONAL_FIELDS`) whose ``p_contradiction`` reaches
  :data:`CONTRADICTION_THRESHOLD`. Definite defect: inversion, negation, swapped
  number or date. Nominal labels — themes, open questions — are judged but never
  condemned by it: a topic is not an assertion.
* ``claim_supported_by_source`` — a claim whose ``p_entailment`` is below
  :data:`SUPPORT_THRESHOLD`. This mixes two very different situations, the source
  being silent and the source saying something else, which is why it is *not* a
  gate in T24a/T24b.

**Premise discipline.** The premise is the content of the canonical input the layer
saw — the entry texts of a day, the day rollups of a week — not the fully reduced
form of :func:`diary.domain.contracts.invariants.input_facts` (which additionally
carries timestamps and lower-cases everything): the judge measures entailment of
meaning, and metadata about a row is not an assertion to be entailed. An empty
premise is a **check error** (:class:`NliPremiseError`) and never a silent
«everything is neutral». That rule was paid for by measurement: a probe built its
premise from the wrong field and the resulting report was convincing and
meaningless — and the same class of mistake later hid inside the calibration
script itself (it compared claims against a single source sentence while the
harness used all of them), which is why the calibration now takes the premise from
the pairs the product actually asked about.

**Privacy.** Metrics carry counts, probabilities and *claim coordinates*
(``facts#2``), never claim text (AGENTS.md §7-8): a violation may quote a token
for the harness (as ``invariants`` does), but the aggregate report stays
content-free. The judge receives the premise and the claim texts because that is
the only way to compute an entailment — the same disclosure the layer itself
already makes to its own model, and nothing more.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "CLAIM_FIELDS",
    "CONTRADICTION",
    "CONTRADICTION_CHECK",
    "CONTRADICTION_THRESHOLD",
    "ENTAILMENT",
    "LABELS",
    "NEUTRAL",
    "SENTENCE_SPLIT_RE",
    "SUPPORT_CHECK",
    "SUPPORT_THRESHOLD",
    "Claim",
    "Judgement",
    "NliError",
    "NliJudge",
    "NliJudgeError",
    "NliMetrics",
    "NliPremiseError",
    "PROPOSITIONAL_FIELDS",
    "claims_for",
    "contradiction_judgements",
    "ensure_premise",
    "judge_claims",
    "metrics_for",
    "premise_from_texts",
    "premise_sentences",
    "split_sentences",
    "unsupported_judgements",
]

#: Verdict labels, in the order the ONNX conversion emits its logits
#: (``config.json`` ``id2label``); a test pins the two against each other.
ENTAILMENT = "entailment"
CONTRADICTION = "contradiction"
NEUTRAL = "neutral"
LABELS: tuple[str, str, str] = (ENTAILMENT, CONTRADICTION, NEUTRAL)

#: Probability at which a claim counts as contradicting its source.
#:
#: Calibrated on the T24a corpus (``scripts/calibrate_nli.py``; tables in
#: ``docs/nli-validation-analysis.md`` §8). On 151 live day claims the sweep was
#: flat exactly where it matters: at every threshold in 0.30-0.90 the rule caught
#: all 34 planted number swaps and all 10 planted inversions and rejected the same
#: 2 of 151 correct claims (both above 0.90), so a higher threshold buys nothing —
#: it only loses recall on claims whose source is silent. 0.60 sits in the middle
#: of that flat region, well above the model's noise and below the measured defects
#: (0.96-1.00).
#:
#: The first calibration pass reported 0.85 from a broken measurement (a premise of
#: one sentence, repetitions discarded, the sweep reading an evidence row instead of
#: the verdict) — the second review found it, and the analysis document records the
#: correction rather than hiding it.
CONTRADICTION_THRESHOLD = 0.60

#: Probability below which a claim counts as unsupported by its source.
#:
#: Measured on the corrected T24a corpus: 21 of 151 correct claims of live day
#: summaries are not fully entailed by any single source sentence (an abstractive
#: summary merges entries, generalises and names themes — none of which is an
#: entailment), and among the claims the rule can even judge (the 72 propositions)
#: the share is 21 of 72. That is why this rule is a **diagnostic** and never a
#: gate: it answers «did the source actually say this», not «is the summary wrong».
SUPPORT_THRESHOLD = 0.80

#: Check name of the contradiction rule (intended for the runtime gate, T24b).
CONTRADICTION_CHECK = "no_contradiction_with_source"

#: Check name of the support rule (diagnostic only).
SUPPORT_CHECK = "claim_supported_by_source"

#: Sentence boundary of a premise: sentence-final punctuation and line breaks.
#:
#: The premise must be split because the model was trained on **sentence pairs**
#: (see the module docstring): feeding it a whole day was measured to turn a
#: verbatim source entry into «contradiction 0.79». The splitter mirrors the one
#: the invariants module uses for its own sentence-level checks.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

#: Contract field -> the fields whose values are claims about the source.
#:
#: Only the summary layers of T24a are listed. The coaching note can be added by
#: one line when the gate is extended to it (its measured inversion of meaning
#: is exactly what the contradiction rule catches); it is left out here to keep
#: this task's scope to summarisation.
CLAIM_FIELDS: dict[str, tuple[str, ...]] = {
    "daily_summary": ("facts", "themes", "open_questions"),
    "weekly_summary": ("rollup",),
}


#: Contract -> fields whose values are **propositions** about the source, i.e. the
#: only ones the contradiction rule may judge.
#:
#: Measured reason (T24a, second review pass): the themes of a day summary are
#: nominal labels («чтение», «обед с детьми»), and the judge reads such a label
#: against a full sentence as a contradiction — «чтение» against «почитала книгу
#: минут 40» scored ``p_contradiction`` 0.90 on a faithful summary, while the
#: propositions of the same summaries scored entailment 0.97-0.99 with
#: contradictions at 0.00-0.04. Label fields are still judged (the support
#: diagnostics see everything), but they cannot condemn a summary: a topic is not
#: an assertion, and a rule that treats it as one rejects correct work.
#:
#: The **weekly rollup stays excluded after its own formulation was measured** (T26,
#: ``docs/nli-validation-analysis.md`` §10, ``ADR-015``). Two measurements say the same
#: thing from two directions. T24a judged the whole rollup as one claim and the
#: sentence-level conjunction condemned **all four** correct rollups at every
#: threshold (4/4, artifact ``/tmp/nli-cal-week5.json``, §8.3.3). T26 then split the
#: rollup into its own sentences and judged them individually — the natural next
#: formulation — and restricted them to the sentences that name a number, a quantity
#: or a name. Four independent runs of ten live week summaries each say the same
#: (``/tmp/nli-cal-t26.json`` is the shipped script's artifact that carries the refusal
#: count; §10.1 names all four): **40–67 %** of those
#: sentences and **53–78 %** of the narrative ones were condemned at the 0.60
#: threshold, and at the level of the layer the formulation would have refused **every
#: one of the ten summaries** in each of the four runs — the refusal is derived from the
#: logs of the three earlier ones and read from the artifact field of the last, where it
#: also holds at every threshold of the 0.30–0.90 sweep (§10.1).
#: The mechanism is measured too: the rollup is a *re-rendering* of the week, and the
#: instrument reads its week-level anchors as contradiction — the same sentence scores
#: entailment 0.97 against its own source sentence, and adding a week-level anchor
#: («В субботу …», «В первый день …») drops that support to 0.01 and makes the pair a
#: contradiction instead (0.99 and 0.97 respectively, §10.2) — and 50–74 % of the
#: measurable sentences are not entailed by the days at all, so the rule's contradiction
#: fallback governs them. A rule that refuses every summary is not a gate, so the week is judged
#: by the deterministic invariants and the grounding checks, not by this one.
PROPOSITIONAL_FIELDS: dict[str, tuple[str, ...]] = {
    "daily_summary": ("facts",),
}


class NliError(RuntimeError):
    """Base class of the NLI rule-and-judge errors."""


class NliPremiseError(NliError):
    """The premise is empty or not a usable canonical input.

    Raised instead of judging: with no source to compare against, every verdict
    would be neutral and the report would look like a measurement while being an
    artefact of the plumbing.
    """


class NliJudgeError(NliError):
    """The judge answered in a shape the rules cannot use.

    A judge must answer exactly one three-way probability distribution per claim,
    with the three labels of :data:`LABELS` summing to one. Anything else is a
    broken instrument, not a verdict, and must not be folded into a metric.
    """


@dataclass(frozen=True)
class Claim:
    """One assertion of a summary, addressed by where it came from."""

    contract: str
    field: str
    index: int
    text: str

    @property
    def coordinate(self) -> str:
        """Content-free address of the claim, safe for reports (``facts#2``)."""
        return f"{self.field}#{self.index}"

    @property
    def is_propositional(self) -> bool:
        """Whether this claim is an assertion the contradiction rule may judge.

        Label fields (themes, open questions) are topics, not assertions; the
        measured reason they are exempt is in :data:`PROPOSITIONAL_FIELDS`.
        """
        return self.field in PROPOSITIONAL_FIELDS.get(self.contract, ())


@dataclass(frozen=True)
class Judgement:
    """The judge's verdict on one claim, aggregated over sentences on both sides.

    ``probabilities`` is the *evidence* row the verdict rests on (the worst claim
    sentence against the source sentence that governs it); the verdict itself is
    the two flags, because a claim is a conjunction of sentences and no single
    distribution expresses that (see the module docstring).
    """

    claim: Claim
    probabilities: Mapping[str, float]
    #: True when some sentence of the claim contradicts the source.
    contradicted: bool
    #: True when every sentence of the claim is entailed by some source sentence.
    supported: bool
    #: Best entailment over the claim's sentences: the support evidence, decided by
    #: :data:`SUPPORT_THRESHOLD` alone (so it does not move with the contradiction
    #: threshold) and therefore the quantity a threshold sweep needs.
    support_probability: float = 0.0
    #: Strongest contradiction among the claim's *unsupported* sentences: the
    #: contradiction evidence, and the quantity :data:`CONTRADICTION_THRESHOLD` is
    #: compared with. Kept apart from ``probabilities`` because the evidence row of a
    #: supported claim is its weakest sentence, whose contradiction probability says
    #: nothing about the verdict (found in T24a's second review).
    contradiction_probability: float = 0.0
    #: 1-based index of the source sentence the evidence rests on (provenance, no text).
    governing_sentence: int = 0
    #: 1-based index of the claim sentence the evidence rests on (0 = whole claim).
    governing_piece: int = 0

    @property
    def label(self) -> str:
        """The verdict as a label, for reporting only."""
        if self.contradicted:
            return CONTRADICTION
        return ENTAILMENT if self.supported else NEUTRAL

    @property
    def p_contradiction(self) -> float:
        """Probability of contradiction on the evidence row."""
        return self.probabilities[CONTRADICTION]

    @property
    def p_entailment(self) -> float:
        """Probability of entailment on the evidence row."""
        return self.probabilities[ENTAILMENT]

    @property
    def contradicts(self) -> bool:
        """Whether the claim is contradicted by its source."""
        return self.contradicted

    @property
    def is_supported(self) -> bool:
        """Whether every sentence of the claim is supported by the source."""
        return self.supported


@dataclass(frozen=True)
class NliMetrics:
    """Content-free aggregate of one summary's judgements.

    Counts and probabilities only: the coordinates of the flagged claims are
    addresses, never text, so the metrics can be logged and reported (AGENTS.md
    §7-8).
    """

    claims: int
    entailment: int
    contradiction: int
    neutral: int
    max_contradiction: float
    min_entailment: float
    #: Claims of the contradiction rule, and claims that are neither contradicted
    #: nor fully supported (the diagnostic class). The two are disjoint: a
    #: contradicted claim is a defect of its own, not a second «unsupported» one.
    contradicted_coordinates: tuple[str, ...]
    unsupported_coordinates: tuple[str, ...]


#: Signature of an injected judge: (premise, hypothesis) pairs in, probabilities
#: out, one row per pair. Batching is the judge's business — the production
#: adapter runs the whole batch through the cross-encoder in one call.
NliJudge = Callable[[Sequence[tuple[str, str]]], Sequence[Mapping[str, float]]]


def ensure_premise(premise: str) -> str:
    """Return the premise, refusing an empty one.

    A blank premise cannot support any claim, so judging against it would turn
    every claim into «not entailed» and the report into a plausible artefact.
    """
    stripped = premise.strip()
    if not stripped:
        raise NliPremiseError(
            "the NLI premise is empty: the source of the layer was not reconstructed"
        )
    return stripped


def split_sentences(text: str) -> list[str]:
    """Split a source text into the sentences the judge compares claims against."""
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def premise_from_texts(texts: Sequence[str]) -> str:
    """Join the content texts of a layer's input into the premise of one run.

    One implementation for both sides (the runtime guard and the harness), so the
    premise of a run is built the same way wherever it is checked: the texts the
    layer is about to summarize, joined by newlines. An empty result is a loud
    error rather than an empty premise (see :func:`ensure_premise`).
    """
    return ensure_premise("\n".join(text.strip() for text in texts if text.strip()))


def premise_sentences(premise: str) -> list[str]:
    """Sentences of a non-empty premise (the judge's unit of comparison)."""
    sentences = split_sentences(ensure_premise(premise))
    if not sentences:
        raise NliPremiseError("the NLI premise contains no sentences to compare against")
    return sentences


def claims_for(contract: str, payload: Mapping[str, object]) -> list[Claim]:
    """Split a stored summary into the claims to check, per its contract.

    Claims come from the contract's own fields (facts, themes, open questions,
    the weekly rollup) rather than from sentence splitting: the layer already
    decided what its assertions are, and its contract is the stable source of
    that decision.
    """
    try:
        fields = CLAIM_FIELDS[contract]
    except KeyError as exc:
        raise NliError(f"contract {contract!r} declares no NLI claim fields") from exc
    claims: list[Claim] = []
    for field in fields:
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            texts = [value]
        elif isinstance(value, Sequence):
            texts = [item for item in value if isinstance(item, str)]
        else:
            raise NliJudgeError(
                f"field {field!r} of contract {contract!r} is neither text nor a list of texts"
            )
        for index, text in enumerate(texts, 1):
            stripped = text.strip()
            if stripped:
                claims.append(Claim(contract=contract, field=field, index=index, text=stripped))
    if not claims:
        raise NliJudgeError(f"contract {contract!r} produced no claims to check")
    return claims


def _validate(row: Mapping[str, float], claim: Claim) -> dict[str, float]:
    """Validate one judge answer and normalize it into a label->probability map."""
    missing = [label for label in LABELS if label not in row]
    if missing:
        raise NliJudgeError(f"the judge omitted labels {missing!r} for {claim.coordinate}")
    probabilities: dict[str, float] = {}
    for label in LABELS:
        value = float(row[label])
        if not 0.0 <= value <= 1.0:
            raise NliJudgeError(f"the judge returned {value!r} for {label!r} of {claim.coordinate}")
        probabilities[label] = value
    total = sum(probabilities.values())
    if abs(total - 1.0) > 0.02:
        raise NliJudgeError(
            f"the judge probabilities of {claim.coordinate} sum to {total:.4f}, not one"
        )
    return probabilities


def judge_claims(
    premise: str | Sequence[str], claims: Sequence[Claim], judge: NliJudge
) -> list[Judgement]:
    """Judge every claim against every sentence of the source, then aggregate.

    The unit of comparison is a **sentence pair** on both sides, because that is
    what the model was trained on and because a claim may itself be several
    sentences (a day summary's ``facts`` entry often is). For every claim
    sentence the rule asks: does some source sentence entail it? A sentence that
    does decides in favour of that claim sentence, whatever the other source
    sentences would say (the artefact of comparing unrelated sentences must not
    outvote a supporting one); otherwise the strongest contradiction decides, and
    failing that the claim sentence is merely unsupported.

    The claim as a whole is then composed: it is *contradicted* when any of its
    sentences is contradicted, and *supported* when every one of them is — a
    conjunction of assertions can only be faithful if all of its parts are.
    """
    sentences = (
        premise_sentences(premise)
        if isinstance(premise, str)
        else [sentence.strip() for sentence in premise if sentence and sentence.strip()]
    )
    if not sentences:
        raise NliPremiseError("the NLI premise contains no sentences to compare against")
    if not claims:
        raise NliJudgeError("no claims to judge")
    pieces: list[list[str]] = []
    pairs: list[tuple[str, str]] = []
    for claim in claims:
        own = split_sentences(claim.text) or [claim.text.strip()]
        pieces.append(own)
        pairs.extend((sentence, piece) for piece in own for sentence in sentences)
    rows = list(judge(pairs))
    if len(rows) != len(pairs):
        raise NliJudgeError(f"the judge answered {len(rows)} row(s) for {len(pairs)} pair(s)")

    width = len(sentences)
    cursor = 0
    judgements: list[Judgement] = []
    for claim, own in zip(claims, pieces, strict=True):
        verdicts: list[tuple[Mapping[str, float], int, int | None, float, float]] = []
        for _piece in own:
            block = rows[cursor : cursor + width]
            cursor += width
            validated = [_validate(row, claim) for row in block]
            verdicts.append(_governing_row(validated))
        judgements.append(_compose_judgement(claim, verdicts))
    return judgements


def _governing_row(
    rows: Sequence[Mapping[str, float]],
) -> tuple[Mapping[str, float], int, int | None, float, float]:
    """Pick the source sentence that governs one claim sentence, and why.

    Support first: the artefact of comparing unrelated sentences (a verbatim
    source entry read as a contradiction against another sentence of the same
    day) must not be able to outvote a sentence that entails the claim. Only when
    no sentence supports the claim sentence does the strongest contradiction
    decide — the shape the runtime gate needs (T24b).

    Returns the evidence row, its sentence index, the supporting index (``None``
    when nothing supports the sentence), the best entailment of the piece and its
    strongest contradiction. The last two are what a threshold sweep needs: support
    and contradiction are decided by two different thresholds, and the evidence row
    carries only one of them.
    """
    best_support = max(range(len(rows)), key=lambda index: rows[index][ENTAILMENT])
    best_contradiction = max(range(len(rows)), key=lambda index: rows[index][CONTRADICTION])
    best_entailment = rows[best_support][ENTAILMENT]
    strongest_contradiction = rows[best_contradiction][CONTRADICTION]
    if best_entailment >= SUPPORT_THRESHOLD:
        return rows[best_support], best_support, best_support, best_entailment, 0.0
    if strongest_contradiction >= CONTRADICTION_THRESHOLD:
        return (
            rows[best_contradiction],
            best_contradiction,
            None,
            best_entailment,
            strongest_contradiction,
        )
    return rows[best_support], best_support, None, best_entailment, strongest_contradiction


def _compose_judgement(
    claim: Claim,
    verdicts: Sequence[tuple[Mapping[str, float], int, int | None, float, float]],
) -> Judgement:
    """Compose the per-sentence verdicts of one claim into the claim's verdict.

    The two aggregates are computed here and reported beside the flags: support is
    the best entailment over the claim's sentences (decided by
    :data:`SUPPORT_THRESHOLD` alone, so it does not move with the contradiction
    threshold), and contradiction is the strongest contradiction among the
    sentences nothing supports — the quantity :data:`CONTRADICTION_THRESHOLD` is
    compared with, and not the evidence row of a supported claim (whose weakest
    sentence can carry a high contradiction probability and still be irrelevant to
    the verdict). Evidence for the report is the first contradicted sentence; if
    none is, the weakest sentence of the claim.
    """
    support_probability = max(item[3] for item in verdicts)
    unsupported = [item for item in verdicts if item[2] is None]
    contradiction_probability = max((item[4] for item in unsupported), default=0.0)
    contradicted = [
        (row, index, position)
        for position, (row, index, support, _entailment, _contradiction) in enumerate(verdicts, 1)
        # The rule is applied to propositions only: a theme label judged against a
        # sentence reads as a contradiction on faithful summaries (measured 0.90),
        # so the exemption lives here, where the flag is decided, and nowhere else.
        if claim.is_propositional
        and support is None
        and row[CONTRADICTION] >= CONTRADICTION_THRESHOLD
    ]
    if contradicted:
        row, index, position = contradicted[0]
        return Judgement(
            claim=claim,
            probabilities=row,
            contradicted=True,
            supported=False,
            support_probability=support_probability,
            contradiction_probability=contradiction_probability,
            governing_sentence=index + 1,
            governing_piece=position,
        )
    weakest = min(range(len(verdicts)), key=lambda position: verdicts[position][3])
    row, index, support, _entailment, _contradiction = verdicts[weakest]
    return Judgement(
        claim=claim,
        probabilities=row,
        contradicted=False,
        supported=support is not None,
        support_probability=support_probability,
        contradiction_probability=contradiction_probability,
        governing_sentence=index + 1,
        governing_piece=weakest + 1,
    )


def contradiction_judgements(judgements: Sequence[Judgement]) -> list[Judgement]:
    """Claims that reach the contradiction threshold, in document order."""
    return [item for item in judgements if item.contradicts]


def unsupported_judgements(judgements: Sequence[Judgement]) -> list[Judgement]:
    """Claims not fully supported by the source, in document order."""
    return [item for item in judgements if not item.is_supported]


def metrics_for(judgements: Sequence[Judgement]) -> NliMetrics:
    """Aggregate judgements into the content-free metrics of one summary."""
    if not judgements:
        raise NliJudgeError("no judgements to aggregate")
    counts = {label: 0 for label in LABELS}
    for item in judgements:
        counts[item.label] += 1
    return NliMetrics(
        claims=len(judgements),
        entailment=counts[ENTAILMENT],
        contradiction=counts[CONTRADICTION],
        neutral=counts[NEUTRAL],
        # Claim-level aggregates, not evidence rows: a supported claim's evidence row
        # is its weakest sentence, whose numbers say nothing about the verdict.
        max_contradiction=max(item.contradiction_probability for item in judgements),
        min_entailment=min(item.support_probability for item in judgements),
        contradicted_coordinates=tuple(
            item.claim.coordinate for item in contradiction_judgements(judgements)
        ),
        unsupported_coordinates=tuple(
            item.claim.coordinate
            for item in judgements
            if not item.is_supported and not item.contradicts
        ),
    )
