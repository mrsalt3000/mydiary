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

from diary.domain.contracts.nli import (  # noqa: E402
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    SUPPORT_THRESHOLD,
    Claim,
    Judgement,
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


@dataclass
class Sample:
    """One judged claim with the label the corpus gave it."""

    chunk_id: str
    claim: str
    p_contradiction: float
    p_entailment: float
    p_neutral: float
    label: str  # clean | clean_failed_checks | number_swap | antonym | foreign_claim

    @property
    def contradicts_at(self) -> float:
        return self.p_contradiction


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


def _sample(chunk_id: str, claim: str, judgement: Judgement, label: str) -> Sample:
    probabilities = judgement.probabilities
    return Sample(
        chunk_id=chunk_id,
        claim=claim,
        p_contradiction=float(probabilities[CONTRADICTION]),
        p_entailment=float(probabilities[ENTAILMENT]),
        p_neutral=float(probabilities[NEUTRAL]),
        label=label,
    )


def _synthetic_claim(text: str, index: int) -> Claim:
    """A claim object for the calibration corpus (the field does not affect the rule)."""
    return Claim(contract="calibration", field="captured", index=index, text=text)


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


def collect(
    source: str, runs: int, show_text: bool, only: Sequence[str] | None = None
) -> list[Sample]:
    """Build the labelled corpus from the summary chunks of the golden set."""
    harness = load_harness()
    config = harness.load_models_config(harness.CI_MODELS_CONFIG if source == "ci" else None)
    factory = (
        harness.ci_provider_factory if source == "ci" else harness.live_provider_factory(config)
    )
    judge = OnnxNliJudge.from_settings()

    # Chunk id -> (source sentences, claim texts, label of that summary).
    captured: dict[str, tuple[list[str], list[str], str]] = {}
    for run_index in range(max(1, runs)):
        for chunk in harness.load_golden_set(chunk_ids=only)[1]:
            if chunk.contract not in harness.NLI_CONTRACTS:
                continue
            recorder = Recorder(judge)
            outcome = harness.evaluate_chunk(
                chunk, provider=factory(chunk), models_config=config, judge=recorder
            )
            label = "clean"
            if outcome.content_violations:
                failed = [item.check for item in outcome.content_violations]
                label = "clean_failed_checks"
                print(f"  note: {chunk.id} (run {run_index + 1}) failed {failed}", file=sys.stderr)
            sentences = split_sentences(recorder.pairs[0][0]) if recorder.pairs else []
            hypotheses: list[str] = []
            for _sentence, hypothesis in recorder.pairs:
                if hypothesis not in hypotheses:
                    hypotheses.append(hypothesis)
            captured[chunk.id] = (sentences, hypotheses, label)
            if show_text:
                print(f"[{chunk.id} run {run_index + 1}] source sentences: {len(sentences)}")
                for sentence in sentences:
                    print(f"    · {sentence[:150]}")

    samples: list[Sample] = []

    def judge_texts(chunk_id: str, texts: Sequence[str], label: str) -> None:
        """Verdicts from the *rules*, so the calibration measures what ships."""
        sentences = captured[chunk_id][0]
        judgements = judge_claims(
            sentences,
            [_synthetic_claim(text, index) for index, text in enumerate(texts, 1)],
            judge,
        )
        samples.extend(
            _sample(chunk_id, text, judgement, label)
            for text, judgement in zip(texts, judgements, strict=True)
        )

    for chunk_id, (_sentences, claims, label) in captured.items():
        judge_texts(chunk_id, claims, label)

    # Planted negatives: mutants of clean claims, judged against the same source.
    for chunk_id in captured:
        clean = [
            item.claim for item in samples if item.chunk_id == chunk_id and item.label == "clean"
        ]
        for mutation, plant_label in (
            (plant_number_swap, "number_swap"),
            (plant_antonym, "antonym"),
        ):
            mutants = [text for text in (mutation(claim) for claim in clean) if text]
            if mutants:
                judge_texts(chunk_id, mutants, plant_label)

    # Foreign claims: a claim of another chunk judged against this source, so the
    # source is silent about it. For a gate this is the dangerous class: silence
    # must not read as a contradiction.
    for chunk_id in captured:
        foreign = [
            item.claim for item in samples if item.chunk_id != chunk_id and item.label == "clean"
        ][:4]
        if foreign:
            judge_texts(chunk_id, foreign, "foreign_claim")

    return samples


def sweep(samples: Sequence[Sample]) -> list[dict[str, float]]:
    """Catch and false-alarm rates for every threshold of :data:`SWEEP`."""
    clean = [item for item in samples if item.label == "clean"]
    rows: list[dict[str, float]] = []
    for threshold in SWEEP:
        row: dict[str, float] = {"threshold": threshold}
        for label in PLANTED:
            planted = [item for item in samples if item.label == label]
            caught = [item for item in planted if item.contradicts_at >= threshold]
            row[f"caught_{label}"] = len(caught)
            row[f"total_{label}"] = len(planted)
        row["flagged_clean"] = sum(1 for item in clean if item.contradicts_at >= threshold)
        row["total_clean"] = len(clean)
        row["unsupported_clean"] = sum(1 for item in clean if item.p_entailment < SUPPORT_THRESHOLD)
        row["flagged_foreign_as_contradiction"] = sum(
            1
            for item in samples
            if item.label == "foreign_claim" and item.contradicts_at >= threshold
        )
        rows.append(row)
    return rows


def report(samples: Sequence[Sample], rows: Sequence[Mapping[str, float]], show_text: bool) -> None:
    by_label: dict[str, int] = defaultdict(int)
    for item in samples:
        by_label[item.label] += 1
    print("\nкорпус (число утверждений по классам):")
    for label, count in sorted(by_label.items()):
        print(f"  {label:<20} {count}")

    if show_text:
        print("\nутверждения и вероятности:")
        for item in samples:
            mark = "  <-- contradiction" if item.p_contradiction >= 0.60 else ""
            print(
                f"  {item.chunk_id:<34}{item.label:<20} "
                f"entr={item.p_entailment:.2f} contr={item.p_contradiction:.2f} "
                f"{item.claim[:60]}{mark}"
            )

    print(f"\nсвип порога противоречия (support threshold {SUPPORT_THRESHOLD:.2f}):")
    header = (
        f"{'T_c':>5} {'numswap':>12} {'antonym':>12} {'foreign->C':>12} "
        f"{'clean FP':>10} {'unsupported clean':>18}"
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
            f"{row['unsupported_clean']:>18.0f}"
        )
    print(
        "\nчтение: «numswap/antonym» — сколько подделок поймано; «foreign->C» — сколько "
        "посторонних утверждений названы противоречием (цена ошибки для гейта); "
        "«clean FP» — сколько корректных утверждений отвергнуто."
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
    samples = collect(args.source, args.runs, args.show_text, only)
    rows = sweep(samples)
    report(samples, rows, args.show_text)
    if args.json is not None:
        args.json.write_text(
            json.dumps({"source": args.source, "runs": args.runs, "sweep": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nсводка записана: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
