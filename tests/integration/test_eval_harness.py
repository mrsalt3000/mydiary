"""Tests for the T10 regression harness (``eval/run.py``).

Two things are proven here:

* **coverage** — every ``kind`` of the golden set has a checker, and every
  invariant field a chunk declares maps to an implemented check, so an
  invariant can never be declared and then silently skipped. The CI mode runs
  the whole set on a deterministic mock provider (no network, no GPU) and must
  pass 10/10;
* **bite** — a deliberately violating completion, fed through the same real
  layers and the same throwaway database, must fail the harness: an advice
  phrase, a causal connector, an invented number, an invented numeral word and
  an invented proper name each produce their own failed check. A check that
  cannot fail is worthless, so these negative cases are the point of the file.

The harness module lives at ``eval/run.py`` (the T10 artifact), outside the
``src`` package, so it is loaded from its path. All content is synthetic
(AGENTS.md §13).
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest
import yaml

from diary.core.models_config import load_models_config, parse_models_config
from diary.core.settings import PROJECT_ROOT
from diary.domain.contracts import DailySummary
from diary.domain.contracts.golden import (
    BUDGET_RESERVE_TOKENS,
    LAYER_PROMPT_TOKENS_EST,
    PROFILE_MAX_TOKENS,
    DailyInvariants,
    GoldenChunk,
    reserve_tokens_for,
)
from diary.domain.llm.errors import LLMStructuredOutputError

EVAL_RUN = PROJECT_ROOT / "eval" / "run.py"
EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "models.example.yaml"


def _load_harness() -> Any:
    """Import ``eval/run.py`` by path (it is not an installed package)."""
    spec = importlib.util.spec_from_file_location("eval_run", EVAL_RUN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through ``sys.modules`` at class
    # creation time, so the module must be registered before execution.
    sys.modules["eval_run"] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


@pytest.fixture(scope="module")
def models_config():
    """CI configuration: the tracked example config (no endpoint is contacted)."""
    return load_models_config(harness.CI_MODELS_CONFIG)


@pytest.fixture(scope="module")
def chunks() -> list[GoldenChunk]:
    _, loaded = harness.load_golden_set()
    return loaded


def _chunk(chunk_id: str) -> GoldenChunk:
    _, loaded = harness.load_golden_set(chunk_ids=[chunk_id])
    return loaded[0]


def _run(chunk: GoldenChunk, payload: dict[str, object], models_config) -> Any:
    """Evaluate one chunk with a scripted provider answering ``payload``."""
    return harness.evaluate_chunk(
        chunk,
        provider=harness.ScriptedProvider(payload),
        models_config=models_config,
    )


def _passing(chunk: GoldenChunk) -> dict[str, object]:
    """The CI mock completion of a chunk, as the starting point of a mutation."""
    return harness.ci_payload(chunk)


def _checks(outcome: Any) -> set[str]:
    return set(outcome.failed_checks)


def _violations_of_a_payload(
    chunk: GoldenChunk, payload: dict[str, object], check: str
) -> list[Any]:
    """Run one harness check against a payload directly, bypassing the layer.

    Since T17b a violating completion never reaches the store — the layer refuses
    it — so the end-to-end path can only prove *that* a chunk failed and on which
    check. The harness's own checks survive as a second line of defence for a
    payload that was stored, and this helper is how their precision (which token
    they caught) stays covered.

    The completion is completed with the envelope the backend owns (ADR-007 §7):
    a stored payload carries it, a raw completion does not. Only the day
    category is supported, because only its envelope is derivable from the chunk
    alone (the coaching note's anchor is an artifact id the store assigns).
    """
    if not isinstance(chunk.invariants, DailyInvariants):
        raise AssertionError("this helper only rebuilds the envelope of a day chunk")
    stored = {**payload, "date": chunk.invariants.expected_date.isoformat()}
    model, _ = harness._validate_output(chunk, stored)
    assert model is not None
    facts = harness._chunk_facts(chunk)
    ctx = harness.CheckContext(
        chunk=chunk,
        model=model,
        payload=stored,
        input_json_before=harness.canonical_input_json(chunk.input),
        input_text=facts.text,
        input_numbers=facts.numbers,
    )
    return harness.CHECKERS[check](ctx)


# ---------------------------------------------------------------------------
# coverage: every kind and every invariant field is implemented
# ---------------------------------------------------------------------------


def test_every_kind_of_the_set_has_a_checker(chunks: list[GoldenChunk]) -> None:
    assert set(harness.KIND_CHECKS) == {
        "stt_corrected",
        "daily_summary",
        "weekly_summary",
        "coach_note",
    }
    for chunk in chunks:
        assert chunk.invariants.kind in harness.KIND_CHECKS, chunk.id
        assert chunk.invariants.kind == chunk.contract


def test_every_declared_invariant_field_maps_to_an_implemented_check(
    chunks: list[GoldenChunk],
) -> None:
    """A new invariant field cannot be added and then silently skipped."""
    implemented = set(harness.CHECKERS)
    for check_name in harness.CHECK_BY_INVARIANT_FIELD.values():
        assert check_name in implemented, f"check {check_name!r} has no implementation"
    for chunk in chunks:
        declared = set(type(chunk.invariants).model_fields) - {"kind"}
        assert declared <= set(harness.CHECK_BY_INVARIANT_FIELD), chunk.id


def test_every_check_name_is_wired_into_a_run() -> None:
    """Every implemented check is actually called by a kind or the generic set."""
    wired = {*harness.GENERIC_CHECKS}
    for names in harness.KIND_CHECKS.values():
        wired.update(names)
    assert set(harness.CHECK_BY_INVARIANT_FIELD.values()) <= wired


def test_an_unknown_kind_is_a_harness_error(monkeypatch: pytest.MonkeyPatch, models_config) -> None:
    """A kind without a checker fails loudly instead of being skipped."""
    monkeypatch.delitem(harness.KIND_CHECKS, "daily_summary")
    with pytest.raises(harness.HarnessError):
        _run(_chunk("day-01-quiet-monday"), _passing(_chunk("day-01-quiet-monday")), models_config)


def test_unknown_chunk_id_is_a_harness_error() -> None:
    with pytest.raises(harness.HarnessError):
        harness.load_golden_set(chunk_ids=["no-such-chunk"])


# ---------------------------------------------------------------------------
# CI mode: the whole set passes on the deterministic mock provider
# ---------------------------------------------------------------------------


def test_ci_mode_passes_all_ten_chunks(chunks: list[GoldenChunk], models_config) -> None:
    outcomes = harness.evaluate_all(
        chunks, provider_factory=harness.ci_provider_factory, models_config=models_config
    )
    assert [outcome.chunk_id for outcome in outcomes] == [chunk.id for chunk in chunks]
    for outcome in outcomes:
        assert outcome.violations == [], (outcome.chunk_id, outcome.violations)
        assert outcome.passed
        assert outcome.requests >= 1
        # The mock reports no usage, so the measured budget check stays silent.
        assert outcome.usage is None
    assert harness.summarize(outcomes).passed == 10


def test_selecting_chunks_keeps_the_manifest_order(chunks: list[GoldenChunk]) -> None:
    _, selected = harness.load_golden_set(
        chunk_ids=["coach-02-new-project-load", "day-01-quiet-monday"]
    )
    assert [chunk.id for chunk in selected] == ["day-01-quiet-monday", "coach-02-new-project-load"]


# ---------------------------------------------------------------------------
# context budget
# ---------------------------------------------------------------------------


def test_declared_budget_holds_for_every_chunk(chunks: list[GoldenChunk]) -> None:
    for chunk in chunks:
        assert harness.check_budget(chunk) == [], chunk.id
        assert (
            chunk.budget.chunk_tokens_est + reserve_tokens_for(chunk.category)
            <= chunk.budget.num_ctx
        )


def test_reserve_is_prompt_plus_generation_cap() -> None:
    for category, reserve in BUDGET_RESERVE_TOKENS.items():
        assert reserve == LAYER_PROMPT_TOKENS_EST[category] + PROFILE_MAX_TOKENS[category]


def test_generation_caps_mirror_the_example_configuration() -> None:
    """The reserve must not drift away from the profiles it describes."""
    parsed = parse_models_config(
        yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8")),
        source=str(EXAMPLE_CONFIG),
    )
    by_profile = {
        "stt_error": parsed.diary_stt,
        "day": parsed.diary_sum,
        "week": parsed.diary_sum,
        "wellbeing": parsed.diary_coach,
    }
    for category, profile in by_profile.items():
        assert profile is not None, category
        assert PROFILE_MAX_TOKENS[category] == profile.max_tokens


def test_a_budget_overflow_is_detected(chunks: list[GoldenChunk]) -> None:
    chunk = _chunk("week-01-2026-w38")
    oversized = chunk.model_copy(
        update={
            "budget": chunk.budget.model_copy(
                update={"chunk_tokens_est": chunk.budget.num_ctx - reserve_tokens_for("week") + 1}
            )
        }
    )
    violations = harness.check_budget(oversized)
    assert [violation.check for violation in violations] == ["context_budget"]


# ---------------------------------------------------------------------------
# negative cases: a variant of the measure method
# ---------------------------------------------------------------------------


def test_an_advice_phrase_is_refused_by_the_layer(models_config) -> None:
    """The layer refuses advice before storage (T17b) and the chunk fails on it."""
    chunk = _chunk("day-01-quiet-monday")
    payload = _passing(chunk)
    payload["facts"] = ["Ты молодец, успела многое."]
    outcome = _run(chunk, payload, models_config)
    assert not outcome.passed
    assert "advice_and_judgement_absent" in _checks(outcome)
    assert "output_contract" not in _checks(outcome)  # the completion is valid JSON of the contract
    # The layer refused, so the harness never sees a stored payload to grade; its
    # own blacklist check keeps biting when handed one directly.
    violations = _violations_of_a_payload(chunk, payload, "forbidden_lexemes")
    assert violations and {violation.check for violation in violations} == {"forbidden_lexemes"}


def test_a_moralising_phrase_fails_the_chunk(models_config) -> None:
    chunk = _chunk("day-02-long-tuesday")
    payload = _passing(chunk)
    payload["open_questions"] = ["Стоило бы ложиться раньше."]
    outcome = _run(chunk, payload, models_config)
    assert "advice_and_judgement_absent" in _checks(outcome)


def test_a_causal_connector_fails_the_chunk(models_config) -> None:
    chunk = _chunk("week-01-2026-w38")
    payload = _passing(chunk)
    payload["rollup"] = "Неделя прошла вокруг отчёта, что привело к усталости."
    outcome = _run(chunk, payload, models_config)
    assert "causal_claims_absent" in _checks(outcome)
    assert "output_contract" not in _checks(outcome)


def test_a_causal_connector_is_allowed_in_an_open_question(models_config) -> None:
    """Open questions are not assertions; the golden set itself uses them."""
    chunk = _chunk("day-02-long-tuesday")
    payload = _passing(chunk)
    payload["open_questions"] = ["Непонятно, из-за чего сбился сон."]
    outcome = _run(chunk, payload, models_config)
    assert "causal_claims_absent" not in _checks(outcome)


def test_an_invented_number_is_refused_by_the_layer(models_config) -> None:
    chunk = _chunk("day-01-quiet-monday")
    payload = _passing(chunk)
    payload["facts"] = ["Закрыла 9 задач из списка."]
    outcome = _run(chunk, payload, models_config)
    assert not outcome.passed
    assert "no_new_facts" in _checks(outcome)
    grounded = _violations_of_a_payload(chunk, payload, "numbers_grounded")
    assert [violation.check for violation in grounded] == ["numbers_grounded"]


def test_an_invented_numeral_word_is_refused_by_the_layer(models_config) -> None:
    """The T07 case «два созвона при трёх во входе», normalized to digits."""
    chunk = _chunk("day-01-quiet-monday")
    payload = _passing(chunk)
    payload["facts"] = ["Успела сделать зарядку восемь раз."]
    outcome = _run(chunk, payload, models_config)
    assert "no_new_facts" in _checks(outcome)
    violations = _violations_of_a_payload(chunk, payload, "no_new_facts")
    assert any("numeral word" in violation.detail for violation in violations)


def test_an_invented_proper_name_is_refused_by_the_layer(models_config) -> None:
    chunk = _chunk("day-01-quiet-monday")
    payload = _passing(chunk)
    payload["facts"] = ["Встретилась с Ольгой и обсудила отчёт."]
    outcome = _run(chunk, payload, models_config)
    assert "no_new_facts" in _checks(outcome)
    violations = _violations_of_a_payload(chunk, payload, "no_new_facts")
    assert any("proper name" in violation.detail for violation in violations)


def test_the_contract_constant_digit_is_not_an_invented_fact(models_config) -> None:
    """The 7-day horizon is required by the contract, not invented by the model."""
    chunk = _chunk("coach-01-low-energy-tuesday")
    outcome = _run(chunk, _passing(chunk), models_config)
    assert "no_new_facts" not in _checks(outcome)


def test_a_source_faithful_correction_is_not_advice(models_config) -> None:
    """Stt-02's source says «надо было»; a faithful fix must keep it (T04 DoD)."""
    chunk = _chunk("stt-02-rainy-walk")
    outcome = _run(chunk, _passing(chunk), models_config)
    assert outcome.passed


def test_invented_advice_in_a_correction_still_fails(models_config) -> None:
    """The fidelity exemption covers source wording, not new wording."""
    chunk = _chunk("stt-03-weekend-plan")
    payload = _passing(chunk)
    payload["text"] = f"{payload['text']} Стоило бы отдохнуть."
    outcome = _run(chunk, payload, models_config)
    assert "advice_and_judgement_absent" in _checks(outcome)


def test_an_invalid_contract_shape_is_rejected_before_storage(models_config) -> None:
    """AGENTS.md §15: an invalid completion never reaches the store.

    The harness keeps its own ``output_contract`` check as a second line of
    defence; through the real layers a wrong shape fails earlier, at the layer's
    own validation, so the harness would never see such a payload.
    """
    chunk = _chunk("coach-01-low-energy-tuesday")
    payload = _passing(chunk)
    payload.pop("observations")
    with pytest.raises(LLMStructuredOutputError):
        _run(chunk, payload, models_config)
    # The harness-side check still bites when a malformed payload is inspected.
    _, violations = harness._validate_output(chunk, payload)
    assert [violation.check for violation in violations] == ["output_contract"]


def test_a_violating_result_never_reaches_the_store(models_config) -> None:
    """T17b inverts what T10 could only measure: the layer now refuses.

    Before the gate the completion was stored and the harness flagged it, which
    is why the phase acceptance could measure advice and causal claims in real
    artifacts. The refusal is the contract now, and the failure is reported
    through the checks the gate named rather than through a graded payload.
    """
    chunk = _chunk("day-01-quiet-monday")
    payload = _passing(chunk)
    payload["facts"] = ["Ты молодец."]
    outcome = _run(chunk, payload, models_config)
    assert not outcome.passed
    assert "output_contract" not in _checks(outcome)
    assert [
        violation.check
        for violation in outcome.violations
        if "refused to store" in violation.detail
    ] == ["advice_and_judgement_absent"]


def test_ungrounded_content_words_are_a_diagnostic_not_a_failure(models_config) -> None:
    chunk = _chunk("week-01-2026-w38")
    payload = _passing(chunk)
    payload["rollup"] = "Неделя прошла спокойно, несмотря на плохое самочувствие."
    outcome = _run(chunk, payload, models_config)
    assert any("самочувствие" in token for token in outcome.diagnostics)
    assert "no_new_facts" not in _checks(outcome)  # documented limit of the check


def test_the_input_is_not_mutated_by_a_run(models_config) -> None:
    chunk = _chunk("day-03-family-saturday")
    before = harness.canonical_input_json(chunk.input)
    outcome = _run(chunk, _passing(chunk), models_config)
    assert harness.canonical_input_json(chunk.input) == before
    assert "input_unchanged" not in _checks(outcome)


def test_a_mutated_input_is_detected(models_config) -> None:
    """The check itself bites when the captured input differs from the current one."""
    chunk = _chunk("day-03-family-saturday")
    payload = _passing(chunk)
    ctx = harness.CheckContext(
        chunk=chunk,
        model=DailySummary.model_validate({**payload, "date": "2026-09-19"}),
        payload=payload,
        input_json_before="something-else",
        input_text="",
        input_numbers=frozenset(),
    )
    assert [violation.check for violation in harness._check_input_unchanged(ctx)] == [
        "input_unchanged"
    ]


def test_the_report_contains_no_generated_text(models_config) -> None:
    """The report is ids, metrics and check names only (AGENTS.md §7-8)."""
    marker = "злоключениеикс"
    chunk = _chunk("week-02-2026-w39")
    payload = _passing(chunk)
    payload["rollup"] = f"Неделя прошла вокруг {marker} и привела к нагрузке."
    outcome = _run(chunk, payload, models_config)
    assert "causal_claims_absent" in _checks(outcome)
    report = harness.format_report([outcome], harness.golden_manifest())
    assert marker not in report
    assert outcome.chunk_id in report
    assert "causal_claims_absent" in report


def test_report_lists_latency_and_pass_state(chunks: list[GoldenChunk], models_config) -> None:
    outcomes = harness.evaluate_all(
        chunks[:2], provider_factory=harness.ci_provider_factory, models_config=models_config
    )
    report = harness.format_report(outcomes, harness.golden_manifest())
    assert "PASS" in report
    assert "threshold 8/10" in report
    assert "latency stt_error" in report
