"""Unit tests for the runtime NLI gate (T24b).

The gate is what makes the contradiction rule binding in production: the day layer
refuses to store a summary whose facts contradict the source. These tests fix the
contract of the gate itself, without the 712 MB instrument:

* a supported payload passes and nothing is judged beyond the propositions of the
  contract (labels are exempt — a theme read as a contradiction scored 0.90 on a
  faithful summary in the T24a measurement);
* a contradicted claim becomes a violation carrying the check name and a
  **content-free** detail (coordinate and probability, never the claim);
* an unusable instrument and an unusable source are **permanent** failures, because
  «нечего проверять» and «проверять нечем» must never look like «прошло»;
* a judge that answers in a shape the rules cannot use is a permanent failure too.

All text is synthetic (AGENTS.md §13); no weights, provider or database is used.
"""

from __future__ import annotations

import pytest

from diary.domain.contracts import DailySummary
from diary.domain.contracts.nli import CONTRADICTION_CHECK
from diary.domain.nli.onnx_judge import NliUnavailableError
from diary.domain.pipelines.base import PipelinePermanentError
from diary.domain.pipelines.nli_guard import (
    GUARDED_CONTRACT,
    contradiction_guard,
    contradiction_violations,
)

#: The source a day summary is checked against: the day's two entries.
SOURCE = ["Плавал в бассейне сорок минут.", "Лёг спать раньше плана."]


class _Judge:
    """A scripted judge: answers every pair with the given probabilities."""

    def __init__(self, entailment: float, contradiction: float, neutral: float) -> None:
        self.answer = {
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
        }
        self.pairs: list[tuple[str, str]] = []

    def load(self) -> None:
        """The gate proves the instrument before generating; the double needs nothing."""

    def __call__(self, pairs):
        self.pairs.extend(pairs)
        return [dict(self.answer) for _ in pairs]


def _payload(**overrides: object) -> DailySummary:
    data: dict[str, object] = {
        "date": "2026-09-14",
        "facts": ["Плавал в бассейне сорок минут."],
        "themes": ["спорт"],
        "energy": None,
        "open_questions": [],
    }
    data.update(overrides)
    return DailySummary.model_validate(data)


def test_a_supported_summary_has_no_violations() -> None:
    judge = _Judge(entailment=0.98, contradiction=0.01, neutral=0.01)
    assert (
        contradiction_violations(
            _payload(), contract=GUARDED_CONTRACT, premise=SOURCE[0], judge=judge
        )
        == []
    )
    assert judge.pairs  # the claim was asked about


def test_only_propositions_are_judged() -> None:
    """Labels are exempt from the rule, so they are not even sent to the judge.

    The contract requires at least one fact, so the check is by exclusion: with a
    contradicting judge answering everything, only the fact produces a violation,
    and the theme never appears among the hypotheses that were asked about.
    """
    judge = _Judge(entailment=0.10, contradiction=0.90, neutral=0.0)
    payload = _payload(facts=["Не читал ничего."], themes=["чтение"], open_questions=[])
    violations = contradiction_violations(
        payload, contract=GUARDED_CONTRACT, premise=SOURCE[0], judge=judge
    )
    assert [violation.check for violation in violations] == [CONTRADICTION_CHECK]
    hypotheses = {hypothesis for _premise, hypothesis in judge.pairs}
    assert hypotheses == {"Не читал ничего."}


def test_a_contradicted_claim_is_a_violation_with_a_content_free_detail() -> None:
    judge = _Judge(entailment=0.02, contradiction=0.97, neutral=0.01)
    violations = contradiction_violations(
        _payload(), contract=GUARDED_CONTRACT, premise=SOURCE[0], judge=judge
    )
    assert [violation.check for violation in violations] == [CONTRADICTION_CHECK]
    detail = violations[0].detail
    assert "facts#1" in detail and "0.9" in detail
    assert "Плавал" not in detail  # the claim itself never leaves the process


def test_a_guard_reports_violations_and_shares_the_premise() -> None:
    judge = _Judge(entailment=0.02, contradiction=0.97, neutral=0.01)
    check = contradiction_guard(source_texts=SOURCE, judge=judge)
    violations = check(_payload())
    assert [violation.check for violation in violations] == [CONTRADICTION_CHECK]
    # The premise is the source as the layer saw it, sentence by sentence.
    assert {premise for premise, _claim in judge.pairs} == set(SOURCE)


def test_an_unusable_instrument_is_a_permanent_failure() -> None:
    class _Unavailable:
        def load(self) -> None:
            raise NliUnavailableError("/nowhere/model.onnx is missing")

    with pytest.raises(PipelinePermanentError) as error:
        contradiction_guard(source_texts=SOURCE, judge=_Unavailable())
    assert "NLI" in str(error.value)
    assert "/nowhere" not in str(error.value)  # the fix is named, not the filesystem


def test_an_empty_source_is_a_permanent_failure() -> None:
    judge = _Judge(entailment=0.98, contradiction=0.01, neutral=0.01)
    with pytest.raises(PipelinePermanentError):
        contradiction_guard(source_texts=["   ", ""], judge=judge)


def test_a_broken_judge_answer_is_a_permanent_failure() -> None:
    class _Broken:
        def load(self) -> None:
            """Nothing to load."""

        def __call__(self, pairs):
            return []  # no rows for the pairs that were asked about

    check = contradiction_guard(source_texts=SOURCE, judge=_Broken())
    with pytest.raises(PipelinePermanentError):
        check(_payload())


def test_the_default_judge_is_the_pinned_onnx_instrument() -> None:
    """The production default is the measured instrument, not a placeholder."""
    from diary.domain.nli.onnx_judge import OnnxNliJudge
    from diary.domain.pipelines.nli_guard import default_nli_judge

    assert isinstance(default_nli_judge(), OnnxNliJudge)
    assert GUARDED_CONTRACT == "daily_summary"
