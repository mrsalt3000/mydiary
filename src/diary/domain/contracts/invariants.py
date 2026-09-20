"""Runtime enforcement of the generated-text invariants (T17, finding F1).

The T10 harness (``eval/run.py``) defined three always-on invariants of every
derivational layer — no advice or judgement, no causal claims, no invented
facts — but it was the **only** place that applied them: neither the
orchestrator, nor the layers, nor the worker referenced them. In production the
guarantee therefore rested on the prompt alone, and the phase-2 acceptance
measured the consequences in stored artifacts: causal connectors, a substituted
quantity («два часа» where the source says «1 час 20 минут») and a reversal of
meaning (`docs/acceptance-log.md`, T12 items 4–5; finding F1).

This module is the single implementation of those checks, shared by the harness
and by the runtime gate (T17b), so the two cannot drift apart: a check the
runtime enforces is by construction the check the golden set measures, and a fix
to a check fixes both.

Two rules shape the API:

* **A violation may carry generated text.** :attr:`InvariantViolation.detail`
  quotes the offending token, because that is what makes a harness failure
  diagnosable. It is therefore **not log-safe**: the runtime must never log it
  or carry it into an exception (AGENTS.md §7–8 forbid diary content in
  messages, logs and exceptions). :func:`violation_summary` is the content-free
  form the runtime uses — check names and counts only;
* **Grounding is relative to the input, never to a dictionary.** A number is a
  violation only when it does not occur in the source, and for a correction
  layer (:data:`~diary.domain.contracts.golden.LEXEME_FIDELITY_CONTRACTS`) a
  lexeme already present in the source is not a violation at all. The input is
  reduced once to :class:`InputFacts` and reused by every check and every
  retry — and the harness reuses the same reduction for its own grounding
  checks (``names_and_numbers_grounded``, ``numbers_grounded``), so a fix to
  what counts as a fact changes all of them together rather than diverging.

The lexemes, field maps and constants of the checks live in
:mod:`diary.domain.contracts.golden` (the eval contract) and are imported from
there: this module adds the *logic*, not a second vocabulary.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from diary.domain.contracts.golden import (
    ADVICE_EVALUATION_LEXEMES,
    CAUSAL_CONNECTOR_LEXEMES,
    CONTRACT_CONSTANT_DIGITS,
    DEFAULT_FORBIDDEN_LEXEMES,
    LEXEME_FIDELITY_CONTRACTS,
    NON_CONTENT_INPUT_KEYS,
    NUMERAL_WORDS,
    assertive_text_fields,
    generated_text_fields,
)

#: A check of one invariant against one generated payload.
InvariantCheck = Callable[[str, Mapping[str, object], "InputFacts"], list["InvariantViolation"]]

#: Words are matched with ``ё`` as a distinct letter, so the numeral lexicon and
#: the quoted lexeme forms behave exactly as they do in the harness.
_WORD_RE = re.compile(r"[а-яёa-z]+")
_DIGIT_RE = re.compile(r"\d+")
#: Sentence split used to exclude a sentence-initial capital from the names:
#: sentence-ending punctuation followed by whitespace, a line break, or a colon
#: or dash opening a new clause. Russian orthography capitalizes after all of
#: them, so such a capital is not evidence of a proper name. This one corrects a
#: rule that was demonstrably wrong rather than an observed failure: no instance
#: of the class occurs in the corpus, so the widening is inert on real data and
#: is pinned by a unit test instead (T17). Its accepted cost: a name introduced
#: by a colon mid-sentence is no longer checked.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+|(?<=[:—–])\s+")
#: A capitalized word (Cyrillic or Latin) — candidate proper name.
_CAPITALIZED_RE = re.compile(r"[А-ЯЁ][а-яё]+|[A-Z][a-z]+")

#: Stem length used when checking that a proper name occurs in the input (long
#: enough to tolerate Russian inflection, short enough to reject a new name).
_NAME_STEM_LEN = 5

#: Names of the invariants this module enforces. They are the machine-readable
#: identifiers of the harness checks, reused verbatim so a gate failure names
#: the same check the golden set reports.
ADVICE_CHECK = "advice_and_judgement_absent"
CAUSAL_CHECK = "causal_claims_absent"
NEW_FACTS_CHECK = "no_new_facts"

#: The invariants a generated output must satisfy, in report order.
INVARIANT_CHECKS: tuple[str, ...] = (ADVICE_CHECK, CAUSAL_CHECK, NEW_FACTS_CHECK)


@dataclass(frozen=True)
class InvariantViolation:
    """One generated-text invariant violated by one field of one output.

    ``detail`` quotes the offending token and therefore **contains generated
    text**: it is safe to report inside the eval harness and unsafe to log,
    raise or store (AGENTS.md §7–8). Runtime callers use
    :func:`violation_summary`.
    """

    check: str
    detail: str


@dataclass(frozen=True)
class InputFacts:
    """The source content a grounding check compares against, reduced once.

    ``text`` is the casefolded content text of the layer input (identifiers and
    audio paths excluded); ``numbers`` are its digit tokens. Both are computed
    once per layer call and reused across retries.
    """

    text: str
    numbers: frozenset[str]


def content_strings(value: object) -> Iterator[str]:
    """String leaves of a payload, skipping non-content keys (ids, paths)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if key in NON_CONTENT_INPUT_KEYS:
                continue
            yield from content_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from content_strings(item)


def input_facts(payload: object) -> InputFacts:
    """Reduce a layer input payload to the facts a grounding check needs.

    Numbers are collected in both of their written forms. A quantity the source
    spells out («плавал сорок минут») is the same fact as one it writes as a
    digit («40 минут»), so both enter :attr:`InputFacts.numbers`. Without this,
    a layer that faithfully keeps the speaker's wording was reported as
    inventing a fact: the check normalized the output word to ``40`` and then
    looked for the *digit* ``40`` in an input that only had the word — all 8
    grounding failures over 43 real stored artifacts, at all three layers
    (measured false positive, T17).
    """
    text = " ".join(content_strings(payload)).casefold()
    digits = set(numbers(text))
    digits.update(numeral_word_digits(text))
    return InputFacts(text=text, numbers=frozenset(digits))


def input_facts_from_models(inputs: object) -> InputFacts:
    """Reduce a layer input of contract models (or plain payloads) to facts."""
    return input_facts(_as_payload(inputs))


def _as_payload(value: object) -> object:
    """JSON-compatible view of a layer input that may hold contract models."""
    if isinstance(value, (list, tuple)):
        return [_as_payload(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value


def numbers(text_value: str) -> set[str]:
    """Digit sequences of a text value (``15:30`` contributes ``15`` and ``30``)."""
    return set(_DIGIT_RE.findall(text_value))


def numeral_word_digits(text_value: str) -> dict[str, str]:
    """Map ``digit`` → the numeral word that produced it, for the output text."""
    found: dict[str, str] = {}
    for token in _WORD_RE.findall(text_value.casefold()):
        value = NUMERAL_WORDS.get(token)
        if value is not None:
            found.setdefault(str(value), token)
    return found


def proper_names(text_value: str) -> list[str]:
    """Capitalized words of ``text_value`` that are not sentence-initial.

    Sentence starts are excluded because every Russian sentence begins with a
    capital by orthography; the first capitalized word of each sentence is
    therefore not a name candidate.
    """
    names: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text_value):
        stripped = sentence.lstrip()
        skip = _CAPITALIZED_RE.match(stripped)
        skipped = skip.group(0) if skip else ""
        for token in _CAPITALIZED_RE.findall(sentence):
            if token == skipped:
                skipped = ""
                continue
            names.append(token)
    return names


def lexeme_hits(text_value: str, lexemes: Sequence[str]) -> list[str]:
    """Lexemes of ``lexemes`` occurring in ``text_value`` (case-insensitive)."""
    folded = text_value.casefold()
    return [lexeme for lexeme in lexemes if lexeme.casefold() in folded]


def name_is_grounded(name: str, input_text: str) -> bool:
    """Whether ``name`` occurs in the (casefolded) input, inflections allowed.

    A stem match tolerates Russian inflection (``Марина`` vs ``мариной``) while
    still rejecting a name that is nowhere in the input.
    """
    folded = name.casefold()
    return folded in input_text or folded[:_NAME_STEM_LEN] in input_text


def introduced_lexemes(
    contract: str, value: str, lexemes: Sequence[str], input_text: str
) -> list[str]:
    """Lexeme hits the layer introduced rather than copied from the source.

    For a correction layer (:data:`LEXEME_FIDELITY_CONTRACTS`) the generated
    text must stay faithful to the input, so a lexeme already present in the
    input is not a violation — only wording the layer added is. Generative
    layers get no exemption at all.
    """
    hits = lexeme_hits(value, lexemes)
    if contract not in LEXEME_FIDELITY_CONTRACTS:
        return hits
    return [hit for hit in hits if hit.casefold() not in input_text]


def generated_field_values(
    payload: Mapping[str, object], fields: Sequence[str]
) -> Iterator[tuple[str, str]]:
    """``(field, value)`` pairs of the generated text fields of a payload."""
    for name in fields:
        value = payload.get(name)
        if isinstance(value, str):
            yield name, value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    yield name, item


def generated_values(payload: Mapping[str, object], fields: Sequence[str]) -> Iterator[str]:
    """Values of the generated text ``fields`` of one output payload."""
    for _, value in generated_field_values(payload, fields):
        yield value


def advice_and_judgement_violations(
    contract: str, payload: Mapping[str, object], facts: InputFacts
) -> list[InvariantViolation]:
    """No advice, praise, evaluation, morality or diagnosis in the generated text.

    The vocabulary is the union of
    :data:`~diary.domain.contracts.golden.DEFAULT_FORBIDDEN_LEXEMES` — the
    baseline every layer inherits, which carries the product's first veto
    («диагноз», «лечение») — and
    :data:`~diary.domain.contracts.golden.ADVICE_EVALUATION_LEXEMES`. The
    harness applies the same union through its two checks (its
    ``forbidden_lexemes`` check adds the per-chunk list on top of the baseline,
    and a runtime call has no chunk to supply). Before T17b the diagnosis words
    reached production unguarded, because production applied no check at all.
    """
    lexemes = sorted(set(DEFAULT_FORBIDDEN_LEXEMES) | set(ADVICE_EVALUATION_LEXEMES))
    violations: list[InvariantViolation] = []
    for value in generated_values(payload, generated_text_fields(contract)):
        for hit in introduced_lexemes(contract, value, lexemes, facts.text):
            violations.append(InvariantViolation(ADVICE_CHECK, f"advice/evaluation lexeme {hit!r}"))
    return violations


def causal_claim_violations(
    contract: str, payload: Mapping[str, object], facts: InputFacts
) -> list[InvariantViolation]:
    """No causal connector in the assertive generated text fields."""
    violations: list[InvariantViolation] = []
    for value in generated_values(payload, assertive_text_fields(contract)):
        for hit in introduced_lexemes(contract, value, CAUSAL_CONNECTOR_LEXEMES, facts.text):
            violations.append(InvariantViolation(CAUSAL_CHECK, f"causal connector {hit!r}"))
    return violations


#: Union of the constant digits a contract declares for any of its fields, by
#: contract. A constant the layer declares is a parameter of the layer rather
#: than a fact about the source, so it stays legitimate in every field of that
#: contract: a coaching note may name its own 7-day horizon in ``observations``
#: or ``hypotheses``, not only in ``experiment_7d``. The instance is the one the
#: live harness recorded for the coaching chunk of the golden set (T12
#: acceptance, ``docs/acceptance-log.md``); it does not occur among the stored
#: artifacts whose input could be rebuilt, so this widening too is inert on that
#: corpus and is pinned by a unit test (T17).
CONTRACT_CONSTANT_DIGITS_ANY_FIELD: dict[str, frozenset[str]] = {
    contract: frozenset().union(
        *(digits for (name, _), digits in CONTRACT_CONSTANT_DIGITS.items() if name == contract)
    )
    for contract in {name for name, _ in CONTRACT_CONSTANT_DIGITS}
}


def contract_constant_digits(contract: str, field_name: str) -> frozenset[str]:
    """Digits ``field_name`` of ``contract`` may carry without grounding."""
    specific = CONTRACT_CONSTANT_DIGITS.get((contract, field_name), frozenset())
    return specific | CONTRACT_CONSTANT_DIGITS_ANY_FIELD.get(contract, frozenset())


def new_fact_violations(
    contract: str, payload: Mapping[str, object], facts: InputFacts
) -> list[InvariantViolation]:
    """Grounding of digits, numeral words and proper names in the input.

    Exact for the three classes it covers; free-form content words are left to
    the harness diagnostics, because a lexical check cannot tell a paraphrase
    from an invention. Digits a field is contractually required to carry
    (:data:`CONTRACT_CONSTANT_DIGITS`) are exempt.
    """
    violations: list[InvariantViolation] = []
    for field_name, value in generated_field_values(payload, generated_text_fields(contract)):
        allowed = contract_constant_digits(contract, field_name)
        for digit in sorted(numbers(value)):
            if digit not in facts.numbers and digit not in allowed:
                violations.append(
                    InvariantViolation(
                        NEW_FACTS_CHECK, f"number {digit!r} does not occur in the input"
                    )
                )
        for digit, word in sorted(numeral_word_digits(value).items()):
            if digit not in facts.numbers and digit not in allowed:
                violations.append(
                    InvariantViolation(
                        NEW_FACTS_CHECK,
                        f"numeral word {word!r} ({digit}) does not occur in the input",
                    )
                )
        for proper in proper_names(value):
            if not name_is_grounded(proper, facts.text):
                violations.append(
                    InvariantViolation(
                        NEW_FACTS_CHECK,
                        f"proper name {proper!r} does not occur in the input",
                    )
                )
    return violations


#: Every enforced invariant, by its check name.
CHECK_BY_NAME: dict[str, InvariantCheck] = {
    ADVICE_CHECK: advice_and_judgement_violations,
    CAUSAL_CHECK: causal_claim_violations,
    NEW_FACTS_CHECK: new_fact_violations,
}


def check_generated_text(
    contract: str, payload: Mapping[str, object], facts: InputFacts
) -> list[InvariantViolation]:
    """Every invariant violated by one generated output payload.

    ``contract`` names the contract the payload was validated against (the same
    string the harness uses: ``stt_corrected``, ``daily_summary``,
    ``weekly_summary``, ``coach_note``), ``payload`` is its JSON form and
    ``facts`` the reduced source it must stay grounded in. An empty list means
    the output satisfies every invariant.
    """
    violations: list[InvariantViolation] = []
    violations.extend(advice_and_judgement_violations(contract, payload, facts))
    violations.extend(causal_claim_violations(contract, payload, facts))
    violations.extend(new_fact_violations(contract, payload, facts))
    return violations


def violation_summary(violations: Sequence[InvariantViolation]) -> str:
    """Content-free description of violations: check names and counts only.

    This is the only form of a violation that may leave the process (a log
    line, an exception message, run metadata): :attr:`InvariantViolation.detail`
    quotes diary text.
    """
    counts: dict[str, int] = {}
    for violation in violations:
        counts[violation.check] = counts.get(violation.check, 0) + 1
    return ", ".join(f"{check} x{counts[check]}" for check in sorted(counts))


__all__ = [
    "ADVICE_CHECK",
    "CAUSAL_CHECK",
    "CHECK_BY_NAME",
    "CONTRACT_CONSTANT_DIGITS_ANY_FIELD",
    "INVARIANT_CHECKS",
    "NEW_FACTS_CHECK",
    "InputFacts",
    "InvariantCheck",
    "InvariantViolation",
    "advice_and_judgement_violations",
    "causal_claim_violations",
    "check_generated_text",
    "content_strings",
    "contract_constant_digits",
    "generated_field_values",
    "generated_values",
    "input_facts",
    "input_facts_from_models",
    "introduced_lexemes",
    "lexeme_hits",
    "name_is_grounded",
    "new_fact_violations",
    "numbers",
    "numeral_word_digits",
    "proper_names",
    "violation_summary",
]
