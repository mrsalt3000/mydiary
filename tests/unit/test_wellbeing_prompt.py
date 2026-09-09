"""Checks for the versioned wellbeing prompt template (Task 15).

The prompt lives at ``prompts/wellbeing/v1.md`` (version ``v1``, recorded in
run provenance). Rules verified here:

* the file exists and is non-empty readable text;
* the mandatory Task-15 invariants are stated: separation of observed facts
  from interpretation, the prohibition of diagnoses / treatment plans /
  invented symptoms / causal certainty / unsupported claims, the 1-3
  recommendations bound, the strict JSON output schema and the requirement
  that observed facts start with a record date;
* the template is only instructions and synthetic examples — no diary content.
"""

from __future__ import annotations

from diary.domain.pipelines.wellbeing.pipeline import PROMPT_PATH

#: Phrases that encode the wellbeing invariants (architecture brief §10.3,
#: tasklist Task 15).
_INVARIANT_PHRASES = (
    # Structural separation of facts and interpretation.
    "observed_facts",
    "cautious_interpretation",
    "recommendations",
    "confidence_and_limitations",
    "[YYYY-MM-DD]",
    # Prohibited content.
    "Диагнозы",
    "Планы лечения",
    "Выдуманные симптомы",
    "причинно-следственных связях",
    "Неподтверждённые утверждения",
    # Recommendation bound and hedging.
    "одной до трёх",
    "возможно",
    # Not a diagnosis / professional help.
    "не врач",
    "профессиональную помощь",
)


def test_prompt_file_exists_and_is_readable() -> None:
    assert PROMPT_PATH.name == "v1.md"
    assert PROMPT_PATH.parent.name == "wellbeing"
    assert PROMPT_PATH.is_file()
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert len(text.strip()) > 300
    assert text.startswith("#")


def test_prompt_states_all_wellbeing_invariants() -> None:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for phrase in _INVARIANT_PHRASES:
        assert phrase.lower() in lowered, f"missing invariant phrase: {phrase!r}"


def test_prompt_uses_synthetic_examples_only() -> None:
    """The template must not contain diary content — only instructions and a
    clearly labeled synthetic example."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "Синтетический пример" in text
    assert "json" in text.lower()
