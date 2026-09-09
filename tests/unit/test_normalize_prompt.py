"""Checks for the versioned normalization prompt template (Task 13).

The prompt lives at ``prompts/normalize/v1.md`` (version ``v1``, recorded in
run provenance). Rules verified here:

* the file exists and is non-empty readable text;
* the mandatory Task-13 invariants are stated (what is allowed and what is
  prohibited, the ``[неразборчиво]`` uncertainty rule, the fixed JSON output
  schema fields);
* the template is only instructions and synthetic examples — the dynamic
  per-chunk entries block is appended by the pipeline at request time, so the
  template itself must not contain the pipeline's entries marker line.
"""

from __future__ import annotations

from diary.domain.pipelines.normalize.pipeline import ENTRIES_HEADER, PROMPT_PATH

#: Phrases that encode the normalization invariants (architecture brief §10.1).
_INVARIANT_PHRASES = (
    # Prohibited operations.
    "добавлять факты",
    "диагнозы",
    "эмоциональные интерпретации",
    "Удалять значимую информацию",
    "менять хронологию",
    "литературно переписывать голос",
    "выдумывай",
    # Uncertainty rule.
    "[неразборчиво]",
    # Allowed operations.
    "ошибки распознавания речи",
    "слова-паразиты",
    "пунктуацию, орфографию",
    # Fixed output-schema fields.
    "normalized_text",
    "entry_count",
    "index",
    "timestamp",
    "kind",
)


def test_prompt_file_exists_and_is_readable() -> None:
    assert PROMPT_PATH.name == "v1.md"
    assert PROMPT_PATH.parent.name == "normalize"
    assert PROMPT_PATH.is_file()
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert len(text.strip()) > 300
    assert text.startswith("#")


def test_prompt_states_all_normalization_invariants() -> None:
    text = PROMPT_PATH.read_text(encoding="utf-8").lower()
    for phrase in _INVARIANT_PHRASES:
        assert phrase.lower() in text, f"missing invariant phrase: {phrase!r}"


def test_prompt_does_not_contain_the_dynamic_entries_marker() -> None:
    """The chunk-entries block is appended per request; the static template
    must not duplicate the marker the pipeline uses to separate it."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert ENTRIES_HEADER not in text
