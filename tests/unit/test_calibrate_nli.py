"""Tests for the T26 measurement inside ``scripts/calibrate_nli.py``.

T26 asked whether the weekly rollup can be judged by the contradiction rule, and the
answer came from a **counterfactual** measurement: the calibration judges the field the
rule does not judge, split into its own sentences and reported by measurability. Two
things therefore need pinning rather than trusting:

* the counterfactual actually bites — a row that can never be flagged would "measure" a
  reassuring zero forever (the samples are forced to be propositional on purpose, and
  that is asserted both ways);
* the corpus walk fills the classes and prints them, with the coverage next to them, so
  the decision they support stays reproducible.

The instrument is scripted here (no weights, no provider, no database), exactly as the
golden-set tests do it. All text is synthetic (AGENTS.md §13).
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from diary.domain.contracts.nli import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    Claim,
    judge_claims,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

NARRATIVE_TEXT = "Неделя прошла спокойно и была сфокусирована на восстановлении."
MEASURABLE_TEXT = "Плавал 40 минут трижды."


def _load_calibration() -> Any:
    """Import ``scripts/calibrate_nli.py`` by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("calibrate_nli", SCRIPTS / "calibrate_nli.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_nli"] = module
    spec.loader.exec_module(module)
    return module


calibration = _load_calibration()


def _rows(count: int, *, contradiction: float) -> list[dict[str, float]]:
    rest = round((1.0 - contradiction) / 2, 4)
    return [{ENTAILMENT: rest, CONTRADICTION: contradiction, NEUTRAL: rest} for _ in range(count)]


class _ScriptedJudge:
    """The instrument stand-in: no weights, one fixed verdict per pair."""

    def __init__(self, contradiction: float = 0.02) -> None:
        self._contradiction = contradiction

    def load(self) -> None:
        """Nothing to load."""

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        return _rows(len(pairs), contradiction=self._contradiction)


class _ScriptedFactory:
    """What ``OnnxNliJudge.from_settings()`` is replaced with in the wiring test."""

    @classmethod
    def from_settings(cls) -> _ScriptedJudge:
        return _ScriptedJudge()


def _judged(sentences: Sequence[str], claim: Claim, contradiction: float):
    return judge_claims(
        sentences, [claim], lambda pairs: _rows(len(pairs), contradiction=contradiction)
    )[0]


# ---------------------------------------------------------------------------
# what the counterfactual is
# ---------------------------------------------------------------------------


def test_a_contract_the_rule_judges_has_no_counterfactual() -> None:
    """The day is judged by the rule itself; nothing about it is counterfactual."""
    payload = {"facts": [NARRATIVE_TEXT, MEASURABLE_TEXT], "themes": ["отдых"]}

    assert calibration.counterfactual_claims("daily_summary", payload) == ([], [])


def test_the_counterfactual_splits_the_rollup_by_measurability() -> None:
    """The classes are the distinction the rejected formulation turned on.

    A sentence that names a number, a quantity or a name is one the days could be held
    to; the rest is the rollup's narrative. Coordinates keep the position in the field,
    so a class's verdict still addresses the sentence in the stored rollup.
    """
    payload = {"rollup": f"{MEASURABLE_TEXT} {NARRATIVE_TEXT} Встретился с Аней."}

    measurable_sentences, narrative = calibration.counterfactual_claims("weekly_summary", payload)

    assert [claim.coordinate for claim in measurable_sentences] == ["rollup#1", "rollup#3"]
    assert [claim.coordinate for claim in narrative] == ["rollup#2"]
    assert narrative[0].text == NARRATIVE_TEXT
    assert calibration.measurable(MEASURABLE_TEXT) is True
    assert calibration.measurable("Встретился с Аней.") is True
    assert calibration.measurable(NARRATIVE_TEXT) is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Плавал 40 минут.", True),  # a digit
        ("Плавал сорок минут.", True),  # a quantity written as a word
        ("Встретился с Аней.", True),  # a proper name
        ("День прошёл спокойно.", False),  # narrative, nothing to hold the days to
    ],
)
def test_measurable_names_the_classes_the_days_can_be_held_to(text: str, expected: bool) -> None:
    assert calibration.measurable(text) is expected


# ---------------------------------------------------------------------------
# the counterfactual bites, and it is printed
# ---------------------------------------------------------------------------


def test_a_counterfactual_sample_is_propositional_on_purpose_and_so_can_bite() -> None:
    """Without the flag the row would answer False at every threshold.

    Both halves are asserted: the sentence is outside the rule's scope *and* the sample
    built from it is flagged the way the formulation under test would flag it.
    """
    claim = Claim(contract="weekly_summary", field="rollup", index=1, text=MEASURABLE_TEXT)
    assert claim.is_propositional is False  # the rule does not judge it today
    judgement = _judged([MEASURABLE_TEXT], claim, contradiction=0.91)

    counterfactual = calibration._sample(
        "week-01", claim, judgement, calibration.WEEK_MEASURABLE, as_propositional=True
    )
    shipped = calibration._sample("week-01", claim, judgement, "clean")

    assert counterfactual.propositional is True
    assert counterfactual.contract == "weekly_summary"
    assert counterfactual.flagged_at(calibration.CONTRADICTION_THRESHOLD) is True
    assert shipped.flagged_at(calibration.CONTRADICTION_THRESHOLD) is False


def test_the_sweep_and_the_report_carry_both_counterfactual_rows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A measurement the sweep and the report do not carry cannot decide anything."""
    measurable_claim = Claim(
        contract="weekly_summary", field="rollup", index=1, text=MEASURABLE_TEXT
    )
    narrative_claim = Claim(contract="weekly_summary", field="rollup", index=2, text=NARRATIVE_TEXT)
    samples = [
        calibration._sample(
            "week-01",
            measurable_claim,
            _judged([MEASURABLE_TEXT], measurable_claim, contradiction=0.91),
            calibration.WEEK_MEASURABLE,
            as_propositional=True,
        ),
        calibration._sample(
            "week-01",
            narrative_claim,
            _judged([NARRATIVE_TEXT], narrative_claim, contradiction=0.02),
            calibration.WEEK_NARRATIVE,
            as_propositional=True,
        ),
    ]
    rows = calibration.sweep(samples)

    assert rows[0]["total_week_measurable"] == 1
    assert rows[0]["flagged_week_measurable"] == 1
    assert rows[0]["total_week_narrative"] == 1
    assert rows[0]["flagged_week_narrative"] == 0

    corpus = calibration.Corpus(samples=samples, unverifiable={}, unjudged={})
    calibration.report(corpus, rows, False)
    output = capsys.readouterr().out
    assert "week meas" in output  # the sweep column
    assert "week narr" in output
    # The counterfactual's coverage: how much of the rollup a weekly rule would judge.
    assert "контрфакт weekly_summary.rollup (правило его не судит): предложений 2, из них" in output


def test_the_report_names_the_summaries_the_rule_had_nothing_to_check_in(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A corpus quietly smaller than it looks is a misleading measurement."""
    corpus = calibration.Corpus(
        samples=[], unverifiable={"week-01-2026-w38": 2}, unjudged={"week-02-2026-w39": 1}
    )
    calibration.report(corpus, calibration.sweep([]), False)

    output = capsys.readouterr().out
    assert "сводок без утверждений (правилу нечего проверять): week-01-2026-w38=2" in output
    # The second class is different in kind and is named separately: the layer refused
    # its own output, so there is no artifact to judge at all.
    assert "сводок без измерения (слой отказал, судить нечего): week-02-2026-w39=1" in output


def test_the_refusal_rate_is_counted_per_summary_not_per_sentence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """What a gate costs is the share of *summaries* it refuses, not of sentences.

    A rule that condemns one claim refuses the whole summary (ADR-012), so a summary
    with two flagged sentences is one refusal — counting captures instead of summaries
    would double it, and the number would stop answering the question the decision is
    about. The same samples are used twice: once as sentences, once as summaries.
    """
    first = Claim(contract="weekly_summary", field="rollup", index=1, text=MEASURABLE_TEXT)
    second = Claim(contract="weekly_summary", field="rollup", index=2, text="Спал 7 часов.")
    clean = Claim(contract="weekly_summary", field="rollup", index=1, text=NARRATIVE_TEXT)
    samples = [
        calibration._sample(
            "week-01",
            first,
            _judged([MEASURABLE_TEXT], first, contradiction=0.91),
            calibration.WEEK_MEASURABLE,
            as_propositional=True,
            evaluation=1,
        ),
        calibration._sample(
            "week-02",
            second,
            _judged(["Спал 7 часов."], second, contradiction=0.95),
            calibration.WEEK_MEASURABLE,
            as_propositional=True,
            evaluation=1,  # the same summary as the claim above
        ),
        calibration._sample(
            "week-02",
            clean,
            _judged([NARRATIVE_TEXT], clean, contradiction=0.02),
            calibration.WEEK_NARRATIVE,
            as_propositional=True,
            evaluation=1,  # and the same summary's narrative sentence, unflagged
        ),
        calibration._sample(
            "week-03",
            Claim(contract="weekly_summary", field="rollup", index=1, text="Встретился с Аней."),
            _judged(["Встретился с Аней."], clean, contradiction=0.01),
            calibration.WEEK_MEASURABLE,
            as_propositional=True,
            evaluation=2,
        ),
    ]
    rows = calibration.sweep(samples)

    assert rows[0]["total_week_measurable"] == 3  # three sentences...
    assert rows[0]["flagged_week_measurable"] == 2  # two of them flagged...
    assert rows[0]["evaluations_measurable"] == 2  # ...spread over two summaries
    assert rows[0]["refused_evaluations_measurable"] == 1  # one of which is refused
    assert rows[0]["evaluations_any"] == 2
    assert rows[0]["refused_evaluations_any"] == 1

    corpus = calibration.Corpus(samples=samples, unverifiable={}, unjudged={})
    calibration.report(corpus, rows, False)
    output = capsys.readouterr().out
    assert "гейт отверг бы по измеримым предложениям (порог 0.60): 1 из 2 сводок" in output
    assert "гейт отверг бы по любому предложению (порог 0.60): 1 из 2 сводок" in output


def test_an_evaluation_nothing_was_asked_about_is_skipped_not_invented() -> None:
    """The crash a live re-run produced, and the silent corruption behind it.

    A refused evaluation stores nothing: the harness returns before it hands the summary
    to a judge. Judging it anyway meant using the *previous* run's payload (the recorder
    is keyed by chunk) against an empty premise — the measurement would have invented a
    summary that never existed, and then died. Both halves are pinned here.
    """
    payload = {"rollup": "Плавал 40 минут."}
    pairs = [("Плавал 40 минут.", "Плавал 40 минут."), ("Спал 7 часов.", "Спал 7 часов.")]

    sentences, resolved = calibration.capture_inputs(pairs, payload)
    assert sentences == ["Плавал 40 минут.", "Спал 7 часов."]  # the whole source, once each
    assert resolved is payload

    assert calibration.capture_inputs([], payload) is None  # nothing was asked
    assert calibration.capture_inputs(pairs, None) is None  # nothing was stored
    assert calibration.capture_inputs([("   ", "x")], payload) is None  # nothing usable


# ---------------------------------------------------------------------------
# the corpus walk fills the classes
# ---------------------------------------------------------------------------


class _StubChunk:
    """A golden chunk with only what ``collect`` reads from one."""

    def __init__(self, chunk_id: str, contract: str) -> None:
        self.id = chunk_id
        self.contract = contract
        self.category = "week"


class _StubOutcome:
    """What ``evaluate_chunk`` returns for a summary the harness accepted."""

    content_violations: tuple[object, ...] = ()


class _StubHarness:
    """The smallest harness ``collect`` can walk, deciding per run whether to judge.

    The real harness returns **before** it hands the summary to a judge when the layer
    refused its own output (T17b/T24b), which is the case that used to corrupt and then
    crash the corpus. This stub reproduces exactly that: a run in ``judged_runs`` records
    a payload and asks the judge; every other run records nothing.
    """

    CI_MODELS_CONFIG = "ci"
    NLI_CONTRACTS = frozenset({"weekly_summary"})

    def __init__(self, judged_runs: set[int]) -> None:
        self._judged_runs = judged_runs
        self._runs = 0

    def load_models_config(self, path: object) -> object:
        return object()

    def ci_provider_factory(self, config: object) -> object:
        return lambda chunk: object()

    def live_provider_factory(self, config: object) -> object:
        return lambda chunk: object()

    def load_golden_set(self, chunk_ids: object = None) -> tuple[object, list[_StubChunk]]:
        return object(), [_StubChunk("week-01", "weekly_summary")]

    def _nli_metrics(self, chunk: object, payload: object, judge: object) -> None:
        """What the real one does: the judge is asked about the payload's claims."""
        claims = [item for item in payload["rollup"].split(". ") if item]
        judge([(claim, claim) for claim in claims])

    def evaluate_chunk(self, chunk, provider, models_config, judge) -> _StubOutcome:
        self._runs += 1
        if self._runs in self._judged_runs:
            self._nli_metrics(chunk, {"rollup": "Спал 7 часов. Плавал 40 минут."}, judge)
        return _StubOutcome()


def test_collect_skips_an_evaluation_the_harness_did_not_judge(monkeypatch) -> None:
    """The refused evaluation must contribute nothing — not the previous run's summary.

    A live re-run of T26's measurement hit this: the layer refused inside the weekly
    refresh, the harness returned without asking a judge, and the script judged the
    *earlier* run's payload (recorded in a map keyed by chunk) against an empty premise.
    The stub below judges the first run and refuses the second, so a regression that
    reintroduces any cross-evaluation state would show up here as a sample of the second
    evaluation — or as a crash, which is what the run produced.
    """
    monkeypatch.setattr(calibration, "OnnxNliJudge", _ScriptedFactory)
    monkeypatch.setattr(calibration, "load_harness", lambda: _StubHarness(judged_runs={1}))

    corpus = calibration.collect("ci", 2, False, None)

    assert corpus.unjudged == {"week-01": 1}
    # The judged run is measured; the refused one is not, and nothing of it appears —
    # the planted mutants carry evaluation 0 and are the only synthetic samples here.
    assert {item.evaluation for item in corpus.samples if item.evaluation} == {1}
    assert not [item for item in corpus.samples if item.evaluation == 2]
    assert all(item.chunk_id == "week-01" for item in corpus.samples)


def test_collect_wires_the_counterfactual_into_the_weekly_corpus(monkeypatch) -> None:
    """The classes have to be filled by the real corpus walk, not merely computable.

    The instrument is scripted (``from_settings`` replaced), so the weekly chunk runs
    offline; what is asserted is the *composition* of the corpus: the rule's own class
    (the whole rollup, uncondemnable), the two counterfactual classes, and the planted
    mutants of the counterfactual sentences — the mutants are what make the refusal
    quantitative, because they say what such a rule would have caught. Deleting either
    wiring leaves the measurement silent, which is what this test refuses.
    """
    monkeypatch.setattr(calibration, "OnnxNliJudge", _ScriptedFactory)

    corpus = calibration.collect("ci", 1, False, ["week-01-2026-w38"])
    samples = corpus.samples
    labels: dict[str, int] = {}
    for item in samples:
        labels[item.label] = labels.get(item.label, 0) + 1

    assert labels.get("clean", 0) == 1  # the whole rollup, the rule's own view
    assert labels.get(calibration.WEEK_MEASURABLE, 0) > 0
    assert labels.get(calibration.WEEK_NARRATIVE, 0) > 0
    assert corpus.unverifiable == {}
    assert corpus.unjudged == {}
    judged = [item for item in samples if item.label == "clean"]
    assert {item.contract for item in judged} == {"weekly_summary"}
    assert {item.chunk_id for item in judged} == {"week-01-2026-w38"}

    # Measurability is exactly the line between the two counterfactual classes.
    for label in (calibration.WEEK_MEASURABLE, calibration.WEEK_NARRATIVE):
        for item in (sample for sample in samples if sample.label == label):
            assert item.propositional is True
            assert calibration.measurable(item.claim) is (label == calibration.WEEK_MEASURABLE)

    # The mutants of the counterfactual sentences can bite, and they are the rollup's.
    mutants = [
        item for item in samples if item.label in calibration.PLANTED and item.field == "rollup"
    ]
    assert mutants, "the rejected formulation's catch rate would not be measured"
    assert all(item.propositional for item in mutants)
