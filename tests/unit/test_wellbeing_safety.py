"""Unit tests for the deterministic wellbeing safety classifier and schema (Task 15).

The classifier is pure: no database, settings or LLM (ADR-006 — the crisis
level must be decided by code). Synthetic data only (AGENTS.md §13). Tests
cover the three levels, boundary phrasings, determinism, the fixed escalation
content, and the strict structured-output schema.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from diary.domain.pipelines.wellbeing.safety import (
    DISCLAIMER,
    ESCALATION_LINES,
    ESCALATION_TITLE,
    SafetyLevel,
    classify,
    escalation_block,
)
from diary.domain.pipelines.wellbeing.schema import WellbeingResult

# ---------------------------------------------------------------------------
# classifier: level mapping
# ---------------------------------------------------------------------------


def test_classify_normal_text_is_normal() -> None:
    text = "Работал над проектом. Вечером гулял, общался с семьёй. В целом неделя спокойная."
    assert classify(text) is SafetyLevel.normal


def test_classify_distress_text_is_needs_support() -> None:
    text = "Почти не сплю, постоянная тревога, нет сил даже на простые дела."
    assert classify(text) is SafetyLevel.needs_support


def test_classify_self_report_of_depression_is_needs_support() -> None:
    text = "Врач в прошлом году говорил про депрессию, сейчас снова тяжело."
    assert classify(text) is SafetyLevel.needs_support


def test_classify_explicit_suicidal_phrase_is_urgent() -> None:
    text = "Не знаю, что делать, иногда кажется, что я больше не хочу жить."
    assert classify(text) is SafetyLevel.urgent


def test_classify_self_harm_statement_is_urgent() -> None:
    text = "Вчера вечером снова думала о том, чтобы причинить себе вред."
    assert classify(text) is SafetyLevel.urgent


def test_classify_urgent_wins_over_needs_support() -> None:
    text = "Не сплю, тревога, а сегодня ночью поняла, что хочу покончить с собой."
    assert classify(text) is SafetyLevel.urgent


def test_classify_passive_wish_to_disappear_is_needs_support() -> None:
    """A passive 'хочется исчезнуть' is significant distress but not (by
    itself) an explicit intent statement → needs_support, never urgent."""
    text = "Всё надоело, иногда хочется просто исчезнуть."
    assert classify(text) is SafetyLevel.needs_support


def test_classify_metaphor_without_crisis_phrasing_stays_normal() -> None:
    text = "Работа была адской, но я справился и доволен результатом."
    assert classify(text) is SafetyLevel.normal


def test_classify_is_case_insensitive() -> None:
    upper = classify("Я НЕ ХОЧУ БОЛЬШЕ ЖИТЬ. Это письмо в никуда.")
    lower = classify("я не хочу больше жить. это письмо в никуда.")
    assert upper is SafetyLevel.urgent
    assert upper is lower


def test_classify_blank_text_is_normal() -> None:
    assert classify("") is SafetyLevel.normal
    assert classify("   \n\t ") is SafetyLevel.normal


def test_classify_is_deterministic() -> None:
    urgent_text = "Сказал себе: не хочу больше так жить."
    needs_text = "Подавлен и одинок всю неделю."
    normal_text = "Спокойная неделя."
    for text in (urgent_text, needs_text, normal_text):
        levels = {classify(text) for _ in range(5)}
        assert len(levels) == 1, f"classify must be deterministic for {text!r}"


# ---------------------------------------------------------------------------
# escalation content
# ---------------------------------------------------------------------------


def test_escalation_content_is_fixed_and_general() -> None:
    """The escalation guidance is static code content (never LLM text) and
    names no region-specific phone numbers (privacy-and-safety.md §6-7)."""
    assert ESCALATION_TITLE
    assert len(ESCALATION_LINES) >= 4
    joined = " ".join(ESCALATION_LINES)
    # Points the user to professional help…
    assert "врачу" in joined and "психологу" in joined
    assert "экстренной помощи" in joined
    # …but does not invent specific hotline numbers.
    assert not any(character.isdigit() for character in joined)


def test_escalation_block_returns_heading_and_bullets() -> None:
    block = escalation_block()
    assert block[0] == f"## {ESCALATION_TITLE}"
    assert block[1] == ""
    assert len(block) == 2 + len(ESCALATION_LINES)
    assert all(line.startswith("- ") for line in block[2:])


def test_disclaimer_is_present() -> None:
    lowered = DISCLAIMER.lower()
    assert "не медицинский" in lowered
    assert "не психологический диагноз" in lowered
    assert "профессиональной помощи" in lowered


# ---------------------------------------------------------------------------
# structured schema
# ---------------------------------------------------------------------------


def _result(
    *,
    observed_facts: list[str] | None = None,
    recommendations: list[str] | None = None,
) -> dict[str, object]:
    return {
        "observed_facts": observed_facts
        if observed_facts is not None
        else ["[2026-09-07] Автор пишет, что плохо спал."],
        "cautious_interpretation": ["Возможно, неделя была напряжённой."],
        "recommendations": recommendations if recommendations is not None else ["Больше отдыхать."],
        "confidence_and_limitations": ["Только записи одной недели."],
    }


def test_schema_validates_a_complete_result() -> None:
    result = WellbeingResult.model_validate(_result())
    assert result.recommendations == ["Больше отдыхать."]
    assert result.observed_facts[0].startswith("[2026-09-07]")


def test_schema_rejects_observed_fact_without_a_date_anchor() -> None:
    payload = _result(observed_facts=["Автор пишет, что плохо спал."])
    with pytest.raises(ValidationError):
        WellbeingResult.model_validate(payload)


def test_schema_rejects_empty_observed_facts() -> None:
    payload = _result(observed_facts=[])
    with pytest.raises(ValidationError):
        WellbeingResult.model_validate(payload)


def test_schema_rejects_zero_and_more_than_three_recommendations() -> None:
    with pytest.raises(ValidationError):
        WellbeingResult.model_validate(_result(recommendations=[]))
    with pytest.raises(ValidationError):
        WellbeingResult.model_validate(
            _result(recommendations=["Один.", "Два.", "Три.", "Четыре."])
        )


def test_schema_rejects_blank_list_items() -> None:
    payload = _result()
    payload["recommendations"] = ["   "]
    with pytest.raises(ValidationError):
        WellbeingResult.model_validate(payload)


def test_schema_rejects_empty_confidence_section() -> None:
    payload = _result()
    payload["confidence_and_limitations"] = []
    with pytest.raises(ValidationError):
        WellbeingResult.model_validate(payload)


def test_schema_rejects_extra_keys() -> None:
    payload = _result()
    payload["severity"] = "urgent"  # the level must never come from the LLM
    with pytest.raises(ValidationError):
        WellbeingResult.model_validate(payload)
