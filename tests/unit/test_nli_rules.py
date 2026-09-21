"""Unit tests for the NLI rules (T24a).

The rules are the part of the NLI validation that must be provable offline: they
decide how a stored summary is split into claims, how the source is split into
sentences, how the judge's probabilities compose into a verdict, and what a
metric may carry. The model itself is not involved here — a scripted judge
answers with the numbers each test needs, which is also how the harness stays
runnable in CI (the ONNX instrument lives in
``diary.domain.nli.onnx_judge`` and is exercised only by a live run, whose
measured behaviour is recorded in ``docs/nli-validation-analysis.md``).

What the tests pin:

* the unit of comparison is a sentence pair on both sides, and the claim is a
  conjunction: contradicted if any of its sentences is, supported only if all are;
* support outvotes contradiction (the artefact of unrelated sentences must not
  condemn a claim a source sentence entails);
* an empty premise or a mis-shaped judge answer is a loud error, never a verdict;
* metrics carry counts, coordinates and probabilities — never claim text.

All text is synthetic (AGENTS.md §13); no provider, model or database is used.
"""

from __future__ import annotations

import pytest

from diary.domain.contracts.nli import (
    CONTRADICTION,
    CONTRADICTION_CHECK,
    CONTRADICTION_THRESHOLD,
    ENTAILMENT,
    NEUTRAL,
    SUPPORT_THRESHOLD,
    Claim,
    NliError,
    NliJudgeError,
    NliPremiseError,
    claims_for,
    contradiction_judgements,
    ensure_premise,
    judge_claims,
    metrics_for,
    premise_sentences,
    split_sentences,
    unsupported_judgements,
)


def _row(entailment: float, contradiction: float, neutral: float) -> dict[str, float]:
    assert abs(entailment + contradiction + neutral - 1.0) < 0.02
    return {ENTAILMENT: entailment, CONTRADICTION: contradiction, NEUTRAL: neutral}


def scripted(answers: dict[tuple[str, str], dict[str, float]]):
    """A judge answering from a fixed table, loudly failing on a missing pair."""

    def judge(pairs):
        return [answers[pair] for pair in pairs]

    return judge


def _claim(text: str, field: str = "facts", index: int = 1) -> Claim:
    return Claim(contract="daily_summary", field=field, index=index, text=text)


# ---------------------------------------------------------------------------
# claims and sentences
# ---------------------------------------------------------------------------


def test_claims_come_from_the_contract_fields_in_document_order() -> None:
    payload = {
        "facts": ["Плавал в бассейне.", "Дочитал отчёт."],
        "themes": ["спорт"],
        "open_questions": [],
        "energy": None,
    }
    claims = claims_for("daily_summary", payload)
    assert [claim.coordinate for claim in claims] == ["facts#1", "facts#2", "themes#1"]
    assert [claim.text for claim in claims] == ["Плавал в бассейне.", "Дочитал отчёт.", "спорт"]


def test_blank_and_missing_fields_are_skipped_but_a_claimless_summary_is_an_error() -> None:
    claims = claims_for("daily_summary", {"facts": ["  ", "Есть текст."], "themes": None})
    assert [claim.text for claim in claims] == ["Есть текст."]
    with pytest.raises(NliJudgeError):
        claims_for("daily_summary", {"facts": [], "themes": [], "open_questions": []})


def test_an_unknown_contract_is_an_error_not_an_empty_check() -> None:
    with pytest.raises(NliError):
        claims_for("stt_corrected", {"corrected_text": "что-то"})


def test_the_weekly_rollup_is_one_claim() -> None:
    claims = claims_for("weekly_summary", {"rollup": "Отправила отчёт раньше срока."})
    assert [claim.coordinate for claim in claims] == ["rollup#1"]


def test_sentences_split_on_punctuation_and_lines() -> None:
    assert split_sentences("Первое. Второе!\nТретье") == ["Первое.", "Второе!", "Третье"]


def test_an_empty_premise_is_a_check_error() -> None:
    with pytest.raises(NliPremiseError):
        ensure_premise("   \n  ")
    with pytest.raises(NliPremiseError):
        premise_sentences("\n\n")


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def test_a_supported_claim_is_not_flagged() -> None:
    """A source sentence that entails the claim settles it in the claim's favour."""
    claim_text = "Плавал в бассейне и дочитал отчёт."
    judge = scripted(
        {
            ("Плавал в бассейне.", claim_text): _row(0.94, 0.03, 0.03),
            ("Дочитал отчёт вчера.", claim_text): _row(0.22, 0.71, 0.07),
        }
    )
    judgements = judge_claims(
        ["Плавал в бассейне.", "Дочитал отчёт вчера."], [_claim(claim_text)], judge
    )
    assert len(judgements) == 1
    assert judgements[0].is_supported
    assert not judgements[0].contradicts
    assert judgements[0].governing_sentence == 1
    assert contradiction_judgements(judgements) == []


def test_a_contradicted_clause_condemns_the_whole_claim() -> None:
    """A claim is a conjunction: one false part makes the claim untrue."""
    judge = scripted(
        {
            ("Плавал сорок минут.", "Плавал сорок минут."): _row(0.95, 0.02, 0.03),
            ("Плавал сорок минут.", "Плавал девяносто минут."): _row(0.03, 0.95, 0.02),
        }
    )
    judgements = judge_claims(
        ["Плавал сорок минут."], [_claim("Плавал сорок минут. Плавал девяносто минут.")], judge
    )
    assert judgements[0].contradicts
    assert not judgements[0].is_supported
    assert [item.claim.coordinate for item in contradiction_judgements(judgements)] == ["facts#1"]
    # Provenance: the second sentence of the claim, against the only source sentence.
    assert judgements[0].governing_piece == 2
    assert judgements[0].governing_sentence == 1


def test_support_outvotes_an_unrelated_contradiction() -> None:
    """The measured artefact: unrelated sentences must not condemn a supported claim.

    With a multi-sentence source, a verbatim entry compared against a *different*
    entry reads as a contradiction (measured 0.79 in T24a). The rule asks every
    source sentence, and one supporting sentence is enough — even when the
    unrelated sentence is *above* the contradiction threshold, which is what makes
    this test bite: under «contradiction first» the claim would be condemned.
    """
    judge = scripted(
        {
            ("Утром плавал.", "Утром плавал в бассейне."): _row(0.93, 0.05, 0.02),
            ("Вечером читал.", "Утром плавал в бассейне."): _row(0.02, 0.95, 0.03),
        }
    )
    judgements = judge_claims(
        ["Утром плавал.", "Вечером читал."], [_claim("Утром плавал в бассейне.")], judge
    )
    assert judgements[0].is_supported
    assert not judgements[0].contradicts
    assert judgements[0].governing_sentence == 1


def test_support_is_a_conjunction_over_the_sentences_of_the_claim() -> None:
    """One entailed sentence plus one the source is silent about is not «supported».

    The second half of the composition (the first is
    :func:`test_a_contradicted_clause_condemns_the_whole_claim`): were support the
    *maximum* over the claim's sentences rather than their conjunction, a claim
    whose second half the source cannot vouch for would pass as supported.
    """
    judge = scripted(
        {
            ("Плавал в бассейне.", "Плавал в бассейне."): _row(0.93, 0.05, 0.02),
            ("Плавал в бассейне.", "День прошёл спокойно."): _row(0.40, 0.20, 0.40),
            ("Дочитал отчёт.", "Плавал в бассейне."): _row(0.10, 0.30, 0.60),
            ("Дочитал отчёт.", "День прошёл спокойно."): _row(0.30, 0.20, 0.50),
        }
    )
    judgements = judge_claims(
        ["Плавал в бассейне.", "Дочитал отчёт."],
        [_claim("Плавал в бассейне. День прошёл спокойно.")],
        judge,
    )
    assert not judgements[0].is_supported
    assert not judgements[0].contradicts
    assert judgements[0].label == NEUTRAL
    assert [item.claim.coordinate for item in unsupported_judgements(judgements)] == ["facts#1"]


def test_a_theme_label_is_judged_but_cannot_condemn() -> None:
    """A nominal label read as a contradiction must not flag a faithful summary.

    Measured in T24a: the theme «чтение» against «почитала книгу минут 40» scores
    ``p_contradiction`` 0.90 while the summary is faithful, because a topic is not a
    proposition. The same numbers on a propositional field do condemn — which is
    what keeps this test from being satisfied by switching the rule off.
    """
    judge = scripted(
        {
            ("Почитала книгу минут 40.", "чтение"): _row(0.05, 0.90, 0.05),
            ("Почитала книгу минут 40.", "Читала весь день напролёт."): _row(0.05, 0.90, 0.05),
        }
    )
    theme = Claim(contract="daily_summary", field="themes", index=1, text="чтение")
    fact = Claim(
        contract="daily_summary", field="facts", index=1, text="Читала весь день напролёт."
    )
    judgements = judge_claims(["Почитала книгу минут 40."], [theme, fact], judge)
    assert theme.is_propositional is False
    assert not judgements[0].contradicts
    assert judgements[0].label == NEUTRAL
    assert fact.is_propositional is True
    assert judgements[1].contradicts
    assert [item.claim.coordinate for item in contradiction_judgements(judgements)] == ["facts#1"]


def test_the_weekly_rollup_is_judged_but_not_condemned() -> None:
    """The week is out of the rule's scope on its own measurement, not by accident.

    A week summary is an abstraction over seven days: the sentence-level rule
    condemned **all four** correct rollups of the corrected corpus at every
    threshold (``docs/nli-validation-analysis.md`` §8.3.3). The rollup is still
    judged (the support diagnostics see it), but it cannot be condemned until the
    rule has a weekly formulation.
    """
    judge = scripted(
        {("Плавал в бассейне сорок минут.", "Ходила в бассейн ежедневно."): _row(0.05, 0.95, 0.0)}
    )
    rollup = Claim(
        contract="weekly_summary", field="rollup", index=1, text="Ходила в бассейн ежедневно."
    )
    judgements = judge_claims(["Плавал в бассейне сорок минут."], [rollup], judge)
    assert rollup.is_propositional is False
    assert not judgements[0].contradicts
    assert contradiction_judgements(judgements) == []


def test_a_contradiction_below_the_threshold_is_not_a_violation() -> None:
    """The threshold, relative to itself: reached counts, one step below does not."""
    rest = (1.0 - CONTRADICTION_THRESHOLD) / 2
    reached = scripted(
        {("Плавал сорок минут.", "Плавал много."): _row(rest, CONTRADICTION_THRESHOLD, rest)}
    )
    judgements = judge_claims(["Плавал сорок минут."], [_claim("Плавал много.")], reached)
    assert judgements[0].contradicts

    just_below = CONTRADICTION_THRESHOLD - 0.01
    remainder = (1.0 - just_below) / 2
    below = scripted(
        {("Плавал сорок минут.", "Плавал много."): _row(remainder, just_below, remainder)}
    )
    assert not judge_claims(["Плавал сорок минут."], [_claim("Плавал много.")], below)[
        0
    ].contradicts


def test_an_unsupported_claim_is_neither_supported_nor_contradicted() -> None:
    """The source staying silent is the normal case for an abstractive summary."""
    judge = scripted({("Плавал сорок минут.", "День прошёл спокойно."): _row(0.40, 0.20, 0.40)})
    judgements = judge_claims(["Плавал сорок минут."], [_claim("День прошёл спокойно.")], judge)
    assert not judgements[0].is_supported
    assert not judgements[0].contradicts
    assert judgements[0].label == NEUTRAL
    assert [item.claim.coordinate for item in unsupported_judgements(judgements)] == ["facts#1"]


def test_a_broken_judge_is_an_error_not_a_verdict() -> None:
    with pytest.raises(NliJudgeError, match="row"):
        judge_claims(["Источник."], [_claim("Утверждение.")], lambda pairs: [])
    with pytest.raises(NliJudgeError, match="omitted"):
        judge_claims(
            ["Источник."],
            [_claim("Утверждение.")],
            lambda pairs: [{ENTAILMENT: 0.5, CONTRADICTION: 0.5}],
        )
    with pytest.raises(NliJudgeError, match="sum to"):
        judge_claims(
            ["Источник."],
            [_claim("Утверждение.")],
            lambda pairs: [{ENTAILMENT: 0.5, CONTRADICTION: 0.9, NEUTRAL: 0.1}],
        )
    with pytest.raises(NliJudgeError, match="no claims"):
        judge_claims(["Источник."], [], lambda pairs: [])


def test_the_judge_is_asked_about_sentence_pairs_of_both_sides() -> None:
    """The unit of comparison, pinned: sentences of the source against sentences of the claim."""
    seen: list[tuple[str, str]] = []

    def judge(pairs):
        seen.extend(pairs)
        return [_row(0.9, 0.05, 0.05) for _ in pairs]

    claims = [_claim("Первое. Второе.")]
    judge_claims(["Источник один.", "Источник два."], claims, judge)
    assert seen == [
        ("Источник один.", "Первое."),
        ("Источник два.", "Первое."),
        ("Источник один.", "Второе."),
        ("Источник два.", "Второе."),
    ]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_metrics_carry_counts_coordinates_and_probabilities_only() -> None:
    judge = scripted(
        {
            ("Плавал сорок минут.", "Плавал девяносто минут."): _row(0.03, 0.95, 0.02),
            ("Плавал сорок минут.", "День прошёл спокойно."): _row(0.40, 0.30, 0.30),
        }
    )
    judgements = judge_claims(
        ["Плавал сорок минут."],
        [_claim("Плавал девяносто минут."), _claim("День прошёл спокойно.", index=2)],
        judge,
    )
    metrics = metrics_for(judgements)
    assert (metrics.claims, metrics.contradiction, metrics.neutral) == (2, 1, 1)
    assert metrics.contradicted_coordinates == ("facts#1",)
    # The classes are disjoint: a contradicted claim is not also «unsupported».
    assert metrics.unsupported_coordinates == ("facts#2",)
    assert metrics.max_contradiction == pytest.approx(0.95)
    assert metrics.min_entailment == pytest.approx(0.03)
    # Nothing that could carry diary text (or claim text) is part of the metrics.
    assert "Плавал" not in repr(metrics)
    assert not any(" " in coordinate for coordinate in metrics.contradicted_coordinates)


def test_the_thresholds_are_the_reviewed_values() -> None:
    """Threshold changes belong to a measurement, so they are pinned literally.

    ``CONTRADICTION_THRESHOLD`` is the middle of the flat region of the T24a sweep,
    where recall on the planted defects is total and the count of rejected correct
    claims is the same as at any neighbouring value; the measurement, its three
    corrected defects and the open question of the gate are in
    ``docs/nli-validation-analysis.md`` §8.
    """
    assert CONTRADICTION_THRESHOLD == 0.60
    assert SUPPORT_THRESHOLD == 0.80
    assert CONTRADICTION_CHECK == "no_contradiction_with_source"
