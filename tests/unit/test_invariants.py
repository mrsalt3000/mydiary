"""Unit tests for the shared generated-text invariants (T17, finding F1).

Covers the invariants the runtime gate and the eval harness both enforce, with
the two halves the phase-2 acceptance separated:

* what the checks must **catch** — advice and value judgement, causal claims in
  assertive fields, invented numbers, numeral words and proper names, and a
  substituted quantity (the T07 finding «два созвона при трёх во входе»);
* what they must **not** report — the false positives that would have made a
  fail-closed runtime gate reject legitimate, prompt-conformant output. The
  three classes are not equally evidenced, and the difference is preserved
  here rather than flattened:

  1. a quantity the *source* spells out («плавал сорок минут»): comparing an
     output word against the input's digits alone reported the layer as
     inventing a fact — all 8 grounding failures over 43 stored artifacts of a
     real diary, the only class observed in that corpus;
  2. the constant the layer itself declares (the 7-day horizon of a coaching
     note) named outside ``experiment_7d`` — recorded by the live harness on
     the coaching chunk of the golden set, and inert on the stored corpus;
  3. an ordinary word capitalized after a colon («Наблюдение: Первый день…»)
     read as a proper name — no instance in the corpus at all; this one
     corrects a demonstrably wrong rule and is pinned by a test rather than by
     a measurement.

Every text here is synthetic (AGENTS.md §13); no provider, database or model is
touched.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from diary.domain.contracts import RawEntry
from diary.domain.contracts.invariants import (
    ADVICE_CHECK,
    CAUSAL_CHECK,
    NEW_FACTS_CHECK,
    check_generated_text,
    contract_constant_digits,
    input_facts,
    input_facts_from_models,
    violation_summary,
)

#: A week input whose quantities are spelled out as words («три», «сорок») and
#: which carries no digit for the coaching horizon — the shape of the real
#: inputs that produced the measured false positives.
WEEK_INPUT = {
    "daily_ids": ["2026-09-14", "2026-09-15"],
    "rollup": "Просыпалась три раза за ночь, плавал сорок минут.",
}


def _coach(**overrides: object) -> dict[str, object]:
    """A coaching note payload that satisfies every invariant as written."""
    payload: dict[str, object] = {
        "week_id": "2026-W38",
        "summary_id": str(uuid.uuid4()),
        "disclaimer": "Это не медицинская рекомендация.",
        "observations": ["Просыпалась три раза за ночь."],
        "hypotheses": ["Возможная связь со стрессом."],
        "experiment_7d": "Эксперимент на 7 дней: ложиться спать раньше обычного.",
    }
    payload.update(overrides)
    return payload


def _checks(payload: dict[str, object], contract: str = "coach_note", source: object = None):
    """Check names reported for ``payload`` against the week input."""
    facts = input_facts(WEEK_INPUT if source is None else source)
    return [item.check for item in check_generated_text(contract, payload, facts)]


# ---------------------------------------------------------------------------
# Grounding: a quantity counts in whichever form the source writes it
# ---------------------------------------------------------------------------


def test_source_numeral_word_grounds_the_output_word() -> None:
    """The measured false positive: the layer keeps the source's own word."""
    assert _checks(_coach(observations=["Плавал сорок минут."])) == []


def test_source_digit_grounds_the_output_word() -> None:
    """A word in the output is grounded by the digit form in the source."""
    assert _checks(_coach(observations=["Плавал 40 минут."])) == []


def test_source_numeral_word_grounds_the_output_digit() -> None:
    """And the other way round: the word in the source grounds the digit."""
    assert _checks(_coach(observations=["Просыпалась 3 раза за ночь."])) == []


def test_substituted_numeral_is_still_caught() -> None:
    """The T07 finding survives the widening: «два» where the source says «три»."""
    assert NEW_FACTS_CHECK in _checks(_coach(observations=["Просыпалась два раза за ночь."]))


def test_invented_digit_is_caught() -> None:
    """A quantity that is nowhere in the source stays a new fact."""
    assert NEW_FACTS_CHECK in _checks(_coach(observations=["Просыпалась 5 раз за ночь."]))


def test_invented_proper_name_is_caught() -> None:
    """A name that is nowhere in the source is a new fact."""
    assert NEW_FACTS_CHECK in _checks(_coach(observations=["Говорила с Оксаной."]))


def test_inflected_name_is_grounded() -> None:
    """Russian inflection of a source name is not an invented name."""
    source = {"rollup": "Говорила с Мариной о планах."}
    assert _checks(_coach(observations=["Созвон с Мариной прошёл спокойно."]), source=source) == []


# ---------------------------------------------------------------------------
# Grounding: a constant the layer declares is not a fact about the source
# ---------------------------------------------------------------------------


def test_contract_constant_is_allowed_in_its_own_field() -> None:
    assert _checks(_coach()) == []


def test_contract_constant_is_allowed_in_every_field() -> None:
    """The horizon belongs to the layer, so any field may name it (measured FP)."""
    assert _checks(_coach(observations=["Сон нарушался семь дней подряд."])) == []
    assert _checks(_coach(hypotheses=["Сон нарушался 7 дней подряд."])) == []


def test_contract_constant_does_not_leak_to_another_contract() -> None:
    """The same digit is still a new fact in a layer that does not declare it."""
    day = {"facts": ["Держала 7 встреч."], "themes": ["Работа"], "open_questions": []}
    assert NEW_FACTS_CHECK in _checks(day, contract="daily_summary")


def test_contract_constant_digits_by_field() -> None:
    assert contract_constant_digits("coach_note", "experiment_7d") == frozenset({"7"})
    assert contract_constant_digits("coach_note", "observations") == frozenset({"7"})
    assert contract_constant_digits("daily_summary", "facts") == frozenset()


# ---------------------------------------------------------------------------
# Proper names: a capital is not a name just because it is capitalized
# ---------------------------------------------------------------------------


def test_capital_after_colon_is_not_a_name_candidate() -> None:
    """«Наблюдение: Первый день…» — the measured false positive."""
    assert _checks(_coach(observations=["Наблюдение: Первый день прошёл спокойно."])) == []


def test_name_after_colon_is_not_checked() -> None:
    """Accepted loss of the colon rule: a name there is not a name candidate.

    The rule buys the removal of a whole false-positive class for a case that
    does not occur in practice (a name introduced by a colon mid-sentence).
    """
    assert _checks(_coach(observations=["Наблюдение: Оксана звонила вечером."])) == []


def test_name_inside_a_sentence_is_checked() -> None:
    assert NEW_FACTS_CHECK in _checks(_coach(observations=["Вечером звонила Оксана."]))


# ---------------------------------------------------------------------------
# Advice, judgement and causal claims
# ---------------------------------------------------------------------------


def test_advice_and_judgement_are_caught() -> None:
    assert ADVICE_CHECK in _checks(_coach(observations=["Тебе следует больше отдыхать."]))


def test_causal_claim_is_caught() -> None:
    claim = "Просыпалась ночью, что привело к усталости."
    assert CAUSAL_CHECK in _checks(_coach(observations=[claim]))


def test_causal_claim_is_allowed_in_doubt_not_assertion() -> None:
    """A question does not assert a cause, so ``open_questions`` is not checked."""
    day = {
        "facts": ["Спала плохо."],
        "themes": ["Сон"],
        "open_questions": ["Непонятно, из-за чего сбился сон."],
    }
    assert CAUSAL_CHECK not in _checks(day, contract="daily_summary")


def test_correction_layer_may_keep_the_speakers_words() -> None:
    """``stt_corrected`` is faithful rewriting: the source's lexemes are not new."""
    source = [{"id": str(uuid.uuid4()), "text": "устал из-за дороги", "ts": "2026-09-14T18:00:00Z"}]
    kept = {"raw_id": str(uuid.uuid4()), "text": "Устал из-за дороги.", "uncertain": []}
    facts = input_facts(source)
    assert [item.check for item in check_generated_text("stt_corrected", kept, facts)] == []


def test_correction_layer_may_not_introduce_them() -> None:
    source = [{"id": str(uuid.uuid4()), "text": "устал к вечеру", "ts": "2026-09-14T18:00:00Z"}]
    added = {"raw_id": str(uuid.uuid4()), "text": "Устал, потому что мало спал.", "uncertain": []}
    assert CAUSAL_CHECK in _checks(added, contract="stt_corrected", source=source)


# ---------------------------------------------------------------------------
# The runtime-facing API
# ---------------------------------------------------------------------------


def test_violation_summary_carries_no_diary_text() -> None:
    """The only form of a violation the runtime may log or raise."""
    facts = input_facts(WEEK_INPUT)
    found = check_generated_text("coach_note", _coach(observations=["Говорила с Оксаной."]), facts)
    summary = violation_summary(found)
    assert summary == f"{NEW_FACTS_CHECK} x1"
    assert "Оксаной" not in summary
    assert all(item.detail for item in found)  # the detail stays available to eval


def test_input_facts_from_models_reads_contract_models() -> None:
    entry = RawEntry(
        id=uuid.uuid4(),
        ts=datetime(2026, 9, 14, 18, 0, tzinfo=timezone.utc),
        source="text",
        text="Плавал сорок минут",
    )
    facts = input_facts_from_models([entry])
    assert "сорок" in facts.text
    assert "40" in facts.numbers  # the word is a quantity, whichever form it takes


def test_input_facts_skip_identifiers_and_audio_paths() -> None:
    """A UUID and a storage path are not facts."""
    payload = [{"id": str(uuid.uuid4()), "audio_path": "/voice/98765.ogg", "text": "День прошёл"}]
    facts = input_facts(payload)
    assert facts.numbers == frozenset()
    assert "98765" not in facts.text


# ---------------------------------------------------------------------------
# Grounding: a moment or an anchor is not a fact about the diary (T32)
# ---------------------------------------------------------------------------


def _coach_note(experiment: str, **overrides: object) -> dict[str, object]:
    """A coaching payload whose only number is the one ``experiment`` carries."""
    payload: dict[str, object] = {
        "observations": ["Спала мало.", "Работала над отчётом.", "Вечером читала."],
        "hypotheses": ["Возможно, нагрузка шла неравномерно.", "Похоже, темп менялся."],
        "experiment_7d": experiment,
    }
    payload.update(overrides)
    return payload


def test_a_timestamp_does_not_ground_a_number() -> None:
    """Before T32 the digits of ``2026-09-23T21:00`` grounded a claimed «в 21»."""
    source = {"ts": "2026-09-23T21:00:00+03:00", "text": "Лег спать раньше плана."}
    note = _coach_note("Отмечать, во сколько получилось лечь: в наблюдениях называли 21.")
    assert NEW_FACTS_CHECK in _checks(note, source=source)


def test_an_anchor_date_does_not_ground_a_number() -> None:
    """The week's own ``daily_ids`` are anchors, not what the diary says.

    The measured case: «ложиться до 23:00» was accepted on the week containing
    the 23rd and refused on the week that did not — the same invented hour, two
    verdicts, decided only by the calendar (T32 point 2).
    """
    source = {"daily_ids": ["2026-09-22", "2026-09-23"], "rollup": "Спала плохо."}
    note = _coach_note("Эксперимент на 7 дней: ложиться в 23.")
    assert NEW_FACTS_CHECK in _checks(note, source=source)


def test_record_content_still_grounds_a_number() -> None:
    """The narrowing takes only the anchor away, never the record's own words."""
    source = {"daily_ids": ["2026-09-23"], "rollup": "Лег спать в 23 часа."}
    note = _coach_note("Эксперимент на 7 дней: ложиться в 23.")
    assert _checks(note, source=source) == []


def test_input_facts_read_anchors_as_no_facts_at_all() -> None:
    """A moment, a week id, a date and an id contribute neither text nor numbers."""
    payload = {
        "ts": "2026-09-23T21:00:00+03:00",
        "date": "2026-09-23",
        "week_id": "2026-W39",
        "daily_ids": ["2026-09-22", "2026-09-23"],
        "summary_id": str(uuid.uuid4()),
        "created_at": "2026-09-24T07:00:00+00:00",
        "text": "День прошёл спокойно",
    }
    facts = input_facts(payload)
    assert facts.numbers == frozenset()
    assert "2026" not in facts.text
    assert "день прошёл спокойно" in facts.text
