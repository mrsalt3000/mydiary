#!/usr/bin/env python
"""Calibrate the NLI thresholds on the golden corpus (T24a).

The rules of the check live in :mod:`diary.domain.contracts.nli`; this script is
the *measurement* that decides their thresholds, and it is kept in the repository
because the thresholds may only change together with a new measurement
(``docs/tasklist-phase2.md`` §15, T24a DoD 5).

What it does:

* runs the summary chunks of the golden set through the real layer path with a
  **recording judge** wrapped around the pinned ONNX instrument, so the corpus is
  exactly what the product would see (premise = the canonical input of the layer,
  claims = the contract's own fields);
* labels those claims ``clean`` while the summary passes every deterministic
  check (that is the honest definition available: «not caught by anything else»
  is not the same as «faithful», which is why the instrument is being measured);
* plants negative claims out of the clean ones — ``number_swap`` (a digit is
  replaced), ``antonym`` (a curated inversion such as «раньше» → «позже») and
  ``foreign_claim`` (a claim taken from another day, so the source is silent);
* judges, as a **counterfactual**, the field the contradiction rule does not judge
  (the weekly rollup, T26) — split into its own sentences and reported by
  measurability — so the sweep carries what such a rule would have caught *and* what
  it would have condemned among correct sentences. T26 decided on those numbers, and
  this is how the decision stays reproducible (§10 of the analysis);
* skips and counts an evaluation the harness did not judge — the layer refused its own
  output (T17b/T24b) and stored nothing, so there is no summary to measure — instead of
  judging the previous run's summary against an empty premise, which is both an invented
  measurement and a crash (met by a live re-run of T26's own measurement);
* sweeps the contradiction threshold and reports, per value, how much planted
  damage is caught and how many clean claims are flagged.

Usage::

    python scripts/calibrate_nli.py --source ci  --show-text     # fast, offline
    python scripts/calibrate_nli.py --source live --runs 6       # real layers
    python scripts/calibrate_nli.py --source live --runs 2 --json /tmp/nli-cal.json

``--source live`` runs the real layers (a day on the GPU, a week on the CPU, so
minutes on the weekly chunks) and needs ``OLLAMA_API_KEY``; ``--source ci`` uses
the scripted payloads of the harness and needs no endpoint. Weights come from
``APP_NLI_MODEL_DIR`` (``scripts/fetch-nli-model.sh``).

Privacy: the golden corpus is synthetic. Claim text is printed only with
``--show-text``, and the JSON summary never contains it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diary.domain.contracts.invariants import (  # noqa: E402
    numbers,
    numeral_word_digits,
    proper_names,
)
from diary.domain.contracts.nli import (  # noqa: E402
    CONTRADICTION_THRESHOLD,
    PROPOSITIONAL_FIELDS,
    SUPPORT_THRESHOLD,
    Claim,
    Judgement,
    NliJudgeError,
    claims_for,
    judge_claims,
    split_sentences,
)
from diary.domain.nli.onnx_judge import OnnxNliJudge  # noqa: E402

HARNESS_PATH = ROOT / "eval" / "run.py"

#: Inversions planted into clean claims (the source says the left, the plant says
#: the right). Deliberately tiny: the goal is a defensible negative class, not
#: linguistic coverage — the catch rate is reported per strategy.
ANTONYMS: tuple[tuple[str, str], ...] = (
    ("раньше", "позже"),
    ("до ", "после "),
    ("мало", "много"),
    ("быстрее", "медленнее"),
    ("успела", "не успела"),
    ("спал", "не спал"),
    ("легко", "тяжело"),
    ("улучшил", "ухудшил"),
)

#: Contradiction thresholds the sweep walks over.
SWEEP: tuple[float, ...] = (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

#: Labels of planted (negative) claims.
PLANTED = ("number_swap", "antonym", "foreign_claim")

#: The counterfactual of T26: fields the rule does **not** judge, which the calibration
#: judges anyway — split into sentences, against the same premise, with the
#: contradiction rule applied as if the field were propositional. The week's ``rollup``
#: is the case that exists: T26 measured two formulations for it and rejected both
#: (``docs/nli-validation-analysis.md`` §10, ``ADR-015``), so this class is how the
#: refusal stays reproducible instead of becoming a belief. The sentences are split by
#: measurability, because that is the distinction the second formulation turned on:
#: a sentence that names a number, a quantity or a name is one the days could in
#: principle be held to.
COUNTERFACTUAL_FIELDS: dict[str, str] = {"weekly_summary": "rollup"}

#: Label of a counterfactual sentence that names a number, a quantity or a name.
WEEK_MEASURABLE = "week_measurable"

#: Label of a counterfactual sentence that names none of them (the rollup's narrative).
WEEK_NARRATIVE = "week_narrative"

#: The labels whose samples are a counterfactual of the rule rather than its work.
COUNTERFACTUAL_LABELS = (WEEK_MEASURABLE, WEEK_NARRATIVE)


@dataclass
class Sample:
    """One judged claim with the label the corpus gave it.

    The two probabilities are the *claim-level aggregates* the rules compute, not
    the evidence row: ``support_probability`` decides support (with
    :data:`SUPPORT_THRESHOLD`) and ``contradiction_probability`` is what a
    contradiction threshold is compared with. Sweeping the evidence row instead was
    the second defect found in T24a's calibration: a supported claim's evidence row
    is its weakest sentence, so faithful claims were counted as flagged.
    """

    chunk_id: str
    contract: str
    field: str
    claim: str
    support_probability: float
    contradiction_probability: float
    propositional: bool
    # clean | clean_failed_checks | number_swap | antonym | foreign_claim
    # | week_measurable | week_narrative (the last two are counterfactual, see below)
    label: str
    #: Which measured summary this claim came from (1-based; 0 for a synthetic sample —
    #: a planted mutant or a foreign claim). It is what turns per-sentence rates into
    #: the layer's own answer: a rule refuses a *summary* as soon as one of its claims is
    #: condemned, so the number that decides a gate is the share of summaries with at
    #: least one condemned claim, not the share of condemned sentences.
    evaluation: int = 0

    def flagged_at(self, threshold: float) -> bool:
        """Whether the rule would condemn this claim at ``threshold``."""
        return self.propositional and self.contradiction_probability >= threshold

    @property
    def supported(self) -> bool:
        """Whether the claim is fully supported (independent of the sweep)."""
        return self.support_probability >= SUPPORT_THRESHOLD


@dataclass
class Corpus:
    """The measured corpus, plus the evaluations that produced no measurement.

    Two of them, and they are different in kind:

    * ``unverifiable`` — the claim extraction found nothing at all (a blank field, or a
      summary whose only fields are blank): the rule has no material;
    * ``unjudged`` — the harness never handed the summary to a judge, which happens when
      the layer **refused** its own output (T17b/T24b: the gate fired and stored
      nothing). There is no artifact to judge, so the evaluation is skipped.

    Both are counted and reported rather than dropped, because a corpus of ten
    summaries that quietly measured nine would make the rule look better covered than
    it is; and neither is a crash, because a measurement tool that dies on one summary
    cannot finish measuring anything. The second class is not hypothetical: a re-run of
    T26's live measurement met it (a refused day inside the weekly refresh) and the
    script then crashed on the empty premise it had built — with a *stale* payload,
    because the recorded payload was kept in a map keyed by chunk. Both halves of that
    are fixed, and one of them by construction: the payload lives in a single slot that
    is cleared before every evaluation (``collect``), and an evaluation nothing was
    asked about is skipped and counted (``capture_inputs``).
    """

    samples: list[Sample]
    unverifiable: dict[str, int]
    unjudged: dict[str, int]


@dataclass
class Recorder:
    """Judge wrapper: records the (sentence, claim) pairs the harness asked about."""

    inner: Any
    pairs: list[tuple[str, str]]

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.pairs = []

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> Sequence[Mapping[str, float]]:
        self.pairs.extend(pairs)
        return self.inner(pairs)


def load_harness() -> Any:
    """Import ``eval/run.py`` as a module (the loader the tests use too)."""
    spec = importlib.util.spec_from_file_location("eval_run", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["eval_run"] = module
    spec.loader.exec_module(module)
    return module


def _sample(
    chunk_id: str,
    claim: Claim,
    judgement: Judgement,
    label: str,
    *,
    as_propositional: bool = False,
    evaluation: int = 0,
) -> Sample:
    """One labelled sample; ``as_propositional`` forces the counterfactual's premise.

    A counterfactual claim comes from a field the rule does not judge, so
    ``claim.is_propositional`` is False and ``Sample.flagged_at`` would answer False at
    every threshold — a measured row that can never bite. The flag is therefore
    explicit here rather than derived: the counterfactual asks «what would the rule do
    to this sentence», which presupposes the rule applies to it.
    """
    return Sample(
        chunk_id=chunk_id,
        contract=claim.contract,
        field=claim.field,
        claim=claim.text,
        support_probability=judgement.support_probability,
        contradiction_probability=judgement.contradiction_probability,
        propositional=as_propositional or claim.is_propositional,
        label=label,
        evaluation=evaluation,
    )


def _synthetic_claim(text: str, index: int, *, contract: str, field: str) -> Claim:
    """A claim of the calibration corpus, keeping the contract and field it came from.

    The field matters since T24a's second review: the contradiction rule applies to
    propositional fields only, so a corpus that labelled every synthetic claim
    «captured» would silently measure a rule that never condemns anything.
    """
    return Claim(contract=contract, field=field, index=index, text=text)


def measurable(text: str) -> bool:
    """Whether a sentence names something the week's days could be held to.

    A number (``40``, ``15:30``), a quantity written as a word (``сорок``), or a proper
    name (``Аня``, ``Москва``) — the same three classes the deterministic grounding of
    :mod:`diary.domain.contracts.invariants` already checks. The predicate lives here,
    in the measurement, and not in the rules, because T26 measured the formulation that
    used it and rejected it (§10): the rules must not carry a scope nothing ships.
    """
    return bool(numbers(text) or numeral_word_digits(text) or proper_names(text))


def counterfactual_claims(
    contract: str, payload: Mapping[str, object]
) -> tuple[list[Claim], list[Claim]]:
    """The sentences of an unjudged field, split by measurability (T26).

    Returns ``(measurable, narrative)``: the sentences of ``COUNTERFACTUAL_FIELDS``
    for ``contract``, split with the same splitter the rule uses, so what is measured
    is what a weekly formulation would have judged. Empty for a contract the rule
    already judges in full — there is nothing counterfactual about it.
    """
    field = COUNTERFACTUAL_FIELDS.get(contract)
    if field is None:
        return [], []
    value = payload.get(field)
    if not isinstance(value, str):
        return [], []
    measurable_sentences: list[Claim] = []
    narrative: list[Claim] = []
    for index, piece in enumerate(split_sentences(value), 1):
        text = piece.strip()
        if not text:
            continue
        claim = Claim(contract=contract, field=field, index=index, text=text)
        (measurable_sentences if measurable(text) else narrative).append(claim)
    return measurable_sentences, narrative


def capture_inputs(
    pairs: Sequence[tuple[str, str]], payload: Mapping[str, object] | None
) -> tuple[list[str], Mapping[str, object]] | None:
    """The premise and payload of one judged evaluation, or ``None`` if it had none.

    The premise is the *whole* source — every sentence the harness asked about — so it
    is taken from the pairs it asked about. Both parts are required: an evaluation whose
    payload is missing (the layer refused its output and stored nothing) or whose pairs
    are empty (the harness never asked a judge) has nothing to measure, and judging it
    against the previous evaluation's payload or against an empty premise would invent
    a measurement — the crash and the silent corruption a re-run of T26's measurement
    produced.
    """
    if payload is None or not pairs:
        return None
    # Sentence-level on both sides: blank premises are dropped rather than handed to the
    # rule, which refuses a premise with no sentences in it.
    sentences = list(dict.fromkeys(premise for premise, _ in pairs if premise.strip()))
    if not sentences:
        return None
    return sentences, payload


def plant_number_swap(claim: str) -> str | None:
    """Replace the first integer of a claim with a different one."""
    match = re.search(r"\d+", claim)
    if match is None:
        return None
    return claim[: match.start()] + str(int(match.group()) + 31) + claim[match.end() :]


def plant_antonym(claim: str) -> str | None:
    """Replace the first matching inversion of :data:`ANTONYMS`."""
    for left, right in ANTONYMS:
        if left in claim:
            return claim.replace(left, right, 1)
    return None


def collect(source: str, runs: int, show_text: bool, only: Sequence[str] | None = None) -> Corpus:
    """Build the labelled corpus from the summary chunks of the golden set."""
    harness = load_harness()
    config = harness.load_models_config(harness.CI_MODELS_CONFIG if source == "ci" else None)
    factory = (
        harness.ci_provider_factory if source == "ci" else harness.live_provider_factory(config)
    )
    judge = OnnxNliJudge.from_settings()

    # Every (run, chunk) is its own capture: a repeated run *adds* a summary rather
    # than replacing the previous one (the first review of T24a found the opposite —
    # keying by chunk id discarded the repetitions, so `--runs 6` built a corpus of
    # three summaries while the documents claimed eighteen).
    #
    # The harness knows the *claims* (field and all); its judge only sees texts. The
    # calibration therefore records the payload the harness judged, so the corpus carries
    # the same fields the rules see. **One slot, not a map keyed by chunk**: a refused
    # evaluation records nothing at all, and a map would hand it the *previous* run's
    # summary — the silent corruption (and then the crash on an empty premise) a live
    # re-run of T26's own measurement produced. The slot is cleared before every
    # evaluation, so nothing can leak from one to the next.
    recorded: list[dict[str, object] | None] = [None]
    original_metrics = harness._nli_metrics

    def _recording_metrics(chunk: Any, payload: Any, judge_argument: Any) -> Any:
        recorded[0] = dict(payload)
        return original_metrics(chunk, payload, judge_argument)

    captures: list[tuple[str, str, list[str], list[Claim], str, int]] = []
    unverifiable: dict[str, int] = {}
    unjudged: dict[str, int] = {}
    # One number per measured summary, shared by all of its captures (the rule's own
    # view of it and the two counterfactual classes): the refusal rate is a property of
    # the summary, and counting per capture would count a summary twice.
    evaluations = 0
    harness._nli_metrics = _recording_metrics
    try:
        for run_index in range(max(1, runs)):
            for chunk in harness.load_golden_set(chunk_ids=only)[1]:
                if chunk.contract not in harness.NLI_CONTRACTS:
                    continue
                # Nothing the previous evaluation recorded may leak into this one. The
                # load-bearing part is the guard in ``capture_inputs`` (an evaluation that
                # asked no question is skipped, whatever the slot holds); clearing the slot
                # here keeps that obvious instead of implicit.
                recorded[0] = None
                recorder = Recorder(judge)
                outcome = harness.evaluate_chunk(
                    chunk, provider=factory(chunk), models_config=config, judge=recorder
                )
                label = "clean"
                if outcome.content_violations:
                    failed = [item.check for item in outcome.content_violations]
                    label = "clean_failed_checks"
                    print(
                        f"  note: {chunk.id} (run {run_index + 1}) failed {failed}",
                        file=sys.stderr,
                    )
                # The premise is the *whole* source: every sentence the harness asked
                # about, in the order it asked (the first review of T24a found the
                # opposite — taking one pair's premise left exactly one sentence, so
                # the calibration measured a rule nobody ships). An evaluation that has
                # neither is skipped and counted, not judged (see ``capture_inputs``).
                inputs = capture_inputs(recorder.pairs, recorded[0])
                if inputs is None:
                    unjudged[chunk.id] = unjudged.get(chunk.id, 0) + 1
                    print(
                        f"  note: {chunk.id} (run {run_index + 1}) was not judged "
                        "(no payload or no pairs; the layer refused or was skipped)",
                        file=sys.stderr,
                    )
                    continue
                sentences, payload = inputs
                try:
                    claims = claims_for(chunk.contract, payload)
                except NliJudgeError as exc:
                    # A rule with nothing to check (a claimless field). Counted, printed,
                    # and *not* a crash: the measurement has to survive one such summary
                    # to measure the rest.
                    unverifiable[chunk.id] = unverifiable.get(chunk.id, 0) + 1
                    print(
                        f"  note: {chunk.id} (run {run_index + 1}) has no claim to verify: {exc}",
                        file=sys.stderr,
                    )
                    continue
                evaluations += 1
                captures.append((chunk.id, chunk.contract, sentences, claims, label, evaluations))
                # The T26 counterfactual: the rule against the field it does not judge
                # (the weekly rollup), split into its own sentences and reported by
                # measurability. Same source, same instrument, one formulation further
                # out — this is how the refusal of that formulation stays reproducible.
                for counterfactual_label, counterfactual in zip(
                    (WEEK_MEASURABLE, WEEK_NARRATIVE),
                    counterfactual_claims(chunk.contract, payload),
                    strict=True,
                ):
                    if counterfactual:
                        captures.append(
                            (
                                chunk.id,
                                chunk.contract,
                                sentences,
                                counterfactual,
                                counterfactual_label,
                                evaluations,
                            )
                        )
                if show_text:
                    print(
                        f"[{chunk.id} run {run_index + 1}] source sentences: {len(sentences)} "
                        f"claims: {len(claims)}"
                    )
                    for sentence in sentences:
                        print(f"    · {sentence[:150]}")
    finally:
        harness._nli_metrics = original_metrics

    samples: list[Sample] = []

    def judge_claims_of(
        chunk_id: str,
        sentences: Sequence[str],
        claims: Sequence[Claim],
        label: str,
        *,
        as_propositional: bool | None = None,
        evaluation: int = 0,
    ) -> None:
        """Verdicts from the *rules*, so the calibration measures what ships.

        ``as_propositional`` overrides the claim's own field scope: the counterfactual
        classes are not judged by the rule today, and a row that can never be flagged
        would measure nothing (see :func:`_sample`).
        """
        judgements = judge_claims(sentences, claims, judge)
        counterfactual = (
            label in COUNTERFACTUAL_LABELS if as_propositional is None else as_propositional
        )
        samples.extend(
            _sample(
                chunk_id,
                claim,
                judgement,
                label,
                as_propositional=counterfactual,
                evaluation=evaluation,
            )
            for claim, judgement in zip(claims, judgements, strict=True)
        )

    for chunk_id, _contract, sentences, claims, label, evaluation in captures:
        judge_claims_of(chunk_id, sentences, claims, label, evaluation=evaluation)

    # A chunk's source is identical in every run, so the per-chunk source is taken
    # from its first capture; claims accumulate across runs.
    sentences_by_chunk = {chunk_id: sentences for chunk_id, _c, sentences, _k, _l, _e in captures}
    contract_by_chunk = {chunk_id: contract for chunk_id, contract, _s, _k, _l, _e in captures}

    # Planted negatives: mutants of clean *propositional* claims (a mutated label is
    # not a defect the rule can judge), against the same source.
    for chunk_id, sentences in sentences_by_chunk.items():
        contract = contract_by_chunk[chunk_id]
        # Two bases: what the rule judges today (the day's facts), and — for a contract
        # whose field the rule does not judge — what the rejected weekly formulation
        # would have judged. The second base is what makes the refusal quantitative: it
        # says what such a rule would have *caught*, next to what it would have broken.
        bases = (
            ("clean", PROPOSITIONAL_FIELDS.get(contract, ())),
            (WEEK_MEASURABLE, ("rollup",)),
        )
        for base_label, base_fields in bases:
            base = [
                item
                for item in samples
                if item.chunk_id == chunk_id
                and item.label == base_label
                and item.field in base_fields
            ]
            if not base:
                continue
            for mutation, plant_label in (
                (plant_number_swap, "number_swap"),
                (plant_antonym, "antonym"),
            ):
                planted: list[Claim] = []
                for item in base:
                    mutated = mutation(item.claim)
                    if mutated:
                        planted.append(
                            _synthetic_claim(
                                mutated, len(planted) + 1, contract=contract, field=item.field
                            )
                        )
                if planted:
                    judge_claims_of(
                        chunk_id,
                        sentences,
                        planted,
                        plant_label,
                        # A mutant of a counterfactual sentence stays a counterfactual:
                        # it measures the formulation that was rejected.
                        as_propositional=base_label in COUNTERFACTUAL_LABELS,
                    )

    # Foreign claims: a propositional claim of another chunk judged against this
    # source, so the source is silent about it. For a gate this is the dangerous
    # class: silence must not read as a contradiction.
    for chunk_id, sentences in sentences_by_chunk.items():
        contract = contract_by_chunk[chunk_id]
        propositional_fields = PROPOSITIONAL_FIELDS.get(contract, ())
        if not propositional_fields:
            # A contract with no propositional field cannot be condemned, so it has no
            # «foreign» class either — planting one would measure a rule that does not
            # apply. The week is exactly this case since T26 (its refusal is measured by
            # the counterfactual classes instead); the guard stays because a contract
            # may be added without a propositional field.
            continue
        field = propositional_fields[0]
        seen: list[str] = []
        foreign: list[Claim] = []
        for item in samples:
            if item.chunk_id == chunk_id or item.label != "clean":
                continue
            if item.field not in PROPOSITIONAL_FIELDS.get(contract_by_chunk[item.chunk_id], ()):
                continue
            if item.claim in seen:
                continue
            seen.append(item.claim)
            foreign.append(
                _synthetic_claim(item.claim, len(foreign) + 1, contract=contract, field=field)
            )
            if len(foreign) == 4:
                break
        if foreign:
            judge_claims_of(chunk_id, sentences, foreign, "foreign_claim")

    return Corpus(samples=samples, unverifiable=unverifiable, unjudged=unjudged)


def sweep(samples: Sequence[Sample]) -> list[dict[str, float]]:
    """Catch and false-alarm rates for every threshold of :data:`SWEEP`."""
    clean = [item for item in samples if item.label == "clean"]
    rows: list[dict[str, float]] = []
    for threshold in SWEEP:
        row: dict[str, float] = {"threshold": threshold}
        for label in PLANTED:
            planted = [item for item in samples if item.label == label]
            row[f"caught_{label}"] = sum(1 for item in planted if item.flagged_at(threshold))
            row[f"total_{label}"] = len(planted)
        row["flagged_clean"] = sum(1 for item in clean if item.flagged_at(threshold))
        row["total_clean"] = len(clean)
        for label in COUNTERFACTUAL_LABELS:
            counterfactual = [item for item in samples if item.label == label]
            row[f"flagged_{label}"] = sum(
                1 for item in counterfactual if item.flagged_at(threshold)
            )
            row[f"total_{label}"] = len(counterfactual)
            # Support is threshold-free, and it is half of why the refusals happen: a
            # sentence nothing entails falls through to the contradiction fallback.
            row[f"unsupported_{label}"] = sum(1 for item in counterfactual if not item.supported)
        # The layer's own answer. A rule that condemns any claim of a summary refuses
        # that summary, so the share of *summaries* with at least one condemned claim is
        # what a gate costs — and it is not the share of condemned sentences. Counted per
        # evaluation, so a summary whose measurable and narrative sentences are both
        # flagged is still one summary.
        for rule, labels in (
            ("measurable", (WEEK_MEASURABLE,)),
            ("any", COUNTERFACTUAL_LABELS),
        ):
            counterfactual = [item for item in samples if item.label in labels]
            evaluations = {item.evaluation for item in counterfactual if item.evaluation}
            refused = {
                item.evaluation
                for item in counterfactual
                if item.evaluation and item.flagged_at(threshold)
            }
            row[f"evaluations_{rule}"] = len(evaluations)
            row[f"refused_evaluations_{rule}"] = len(refused)
        row["unsupported_clean"] = sum(1 for item in clean if not item.supported)
        row["flagged_foreign_as_contradiction"] = sum(
            1 for item in samples if item.label == "foreign_claim" and item.flagged_at(threshold)
        )
        rows.append(row)
    return rows


def report(corpus: Corpus, rows: Sequence[Mapping[str, float]], show_text: bool) -> None:
    samples = corpus.samples
    by_label: dict[str, int] = defaultdict(int)
    for item in samples:
        by_label[item.label] += 1
    print("\nкорпус (число утверждений по классам):")
    for label, count in sorted(by_label.items()):
        print(f"  {label:<20} {count}")
    fielded = {label: defaultdict(int) for label in ("clean", *PLANTED)}
    for item in samples:
        if item.label in fielded:
            fielded[item.label][item.field] += 1
    for label, counts in fielded.items():
        if counts:
            print(
                f"  {label} по полям: "
                + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            )
    if corpus.unverifiable:
        # Summaries where the rule had nothing to check: a claimless field, which the
        # fail-closed rule of T24a refuses rather than passing silently.
        print(
            "  сводок без утверждений (правилу нечего проверять): "
            + ", ".join(f"{name}={count}" for name, count in sorted(corpus.unverifiable.items()))
        )
    if corpus.unjudged:
        # Summaries the harness never handed to a judge: the layer refused its own
        # output and stored nothing, so there is no artifact to measure.
        print(
            "  сводок без измерения (слой отказал, судить нечего): "
            + ", ".join(f"{name}={count}" for name, count in sorted(corpus.unjudged.items()))
        )
    # T26: what the rejected weekly formulation would have judged, and how much of the
    # rollup it covers. The refusal is a statement about these numbers, so they are
    # printed next to the corpus rather than kept in the sweep.
    for contract, field in COUNTERFACTUAL_FIELDS.items():
        measured = sum(
            1 for item in samples if item.contract == contract and item.label == WEEK_MEASURABLE
        )
        narrative = sum(
            1 for item in samples if item.contract == contract and item.label == WEEK_NARRATIVE
        )
        if measured or narrative:
            share = measured / (measured + narrative)
            unsupported = sum(
                1
                for item in samples
                if item.contract == contract
                and item.label == WEEK_MEASURABLE
                and not item.supported
            )
            print(
                f"  контрфакт {contract}.{field} (правило его не судит): предложений "
                f"{measured + narrative}, из них с числом/именем {measured} ({share:.0%}); "
                f"источник не влечёт {unsupported} из {measured} измеримых"
            )
            # The layer-level consequence, at the rule's own threshold: a summary is
            # refused as soon as one of its claims is condemned.
            for rule, labels, title in (
                ("measurable", (WEEK_MEASURABLE,), "измеримым предложениям"),
                ("any", COUNTERFACTUAL_LABELS, "любому предложению"),
            ):
                candidates = [item for item in samples if item.label in labels]
                evaluations = {item.evaluation for item in candidates if item.evaluation}
                refused = {
                    item.evaluation
                    for item in candidates
                    if item.evaluation and item.flagged_at(CONTRADICTION_THRESHOLD)
                }
                if evaluations:
                    print(
                        f"  гейт отверг бы по {title} (порог {CONTRADICTION_THRESHOLD:.2f}): "
                        f"{len(refused)} из {len(evaluations)} сводок"
                    )

    if show_text:
        print("\nутверждения и вероятности:")
        for item in samples:
            mark = "  <-- contradiction" if item.flagged_at(CONTRADICTION_THRESHOLD) else ""
            print(
                f"  {item.chunk_id:<28}{item.field:<10}{item.label:<20} "
                f"support={item.support_probability:.2f} "
                f"contra={item.contradiction_probability:.2f} "
                f"{item.claim[:60]}{mark}"
            )

    print(f"\nсвип порога противоречия (support threshold {SUPPORT_THRESHOLD:.2f}):")
    header = (
        f"{'T_c':>5} {'numswap':>12} {'antonym':>12} {'foreign->C':>12} "
        f"{'clean FP':>10} {'unsupported clean':>18} {'week meas':>11} {'week narr':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['threshold']:>5.2f} "
            f"{row['caught_number_swap']:>5.0f}/{row['total_number_swap']:<6.0f} "
            f"{row['caught_antonym']:>5.0f}/{row['total_antonym']:<6.0f} "
            f"{row['flagged_foreign_as_contradiction']:>12.0f} "
            f"{row['flagged_clean']:>4.0f}/{row['total_clean']:<5.0f} "
            f"{row['unsupported_clean']:>18.0f} "
            f"{row['flagged_' + WEEK_MEASURABLE]:>4.0f}/{row['total_' + WEEK_MEASURABLE]:<6.0f} "
            f"{row['flagged_' + WEEK_NARRATIVE]:>4.0f}/{row['total_' + WEEK_NARRATIVE]:<6.0f}"
        )
    print(
        "\nчтение: «numswap/antonym» — сколько подделок поймано; «foreign->C» — сколько "
        "посторонних утверждений названы противоречием (цена ошибки для гейта); "
        "«clean FP» — сколько корректных утверждений отвергнуто; "
        "«week meas / week narr» — контрфакт T26: сколько предложений недельного rollup-а "
        "отвергнуто бы, если бы правило его судило (посентенсно, отдельно предложения с "
        "числом или именем и нарративные). Это замер отказа от недельного правила, а не "
        "поведение продукта: правило неделю не судит (см. §10 анализа)."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the NLI thresholds (T24a).")
    parser.add_argument("--source", choices=("ci", "live"), default="ci")
    parser.add_argument("--runs", type=int, default=1, help="repeat the live layers N times")
    parser.add_argument("--show-text", action="store_true", help="print claim text (synthetic)")
    parser.add_argument("--json", type=Path, default=None, help="write a content-free summary")
    parser.add_argument(
        "--chunks",
        default=None,
        help="comma-separated golden chunk ids (default: every summary chunk)",
    )
    args = parser.parse_args(argv)

    model_dir = OnnxNliJudge.from_settings().model_path.parent
    only = [item.strip() for item in args.chunks.split(",")] if args.chunks else None
    print(f"source={args.source} runs={args.runs} model_dir={model_dir} chunks={only or 'all'}")
    corpus = collect(args.source, args.runs, args.show_text, only)
    rows = sweep(corpus.samples)
    report(corpus, rows, args.show_text)
    if args.json is not None:
        args.json.write_text(
            json.dumps({"source": args.source, "runs": args.runs, "sweep": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nсводка записана: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
