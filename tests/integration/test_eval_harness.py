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
    DEFAULT_NUM_CTX,
    LAYER_PROMPT_TOKENS_EST,
    PROFILE_MAX_TOKENS,
    DailyInvariants,
    GoldenChunk,
)
from diary.domain.contracts.nli import NliPremiseError
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.nli.onnx_judge import OnnxNliJudge
from diary.domain.pipelines.base import (
    PipelineInputError,
    PipelineNotRegisteredError,
    PipelineOutputError,
    PipelinePermanentError,
    PipelineStorageError,
)
from diary.domain.pipelines.coach.pipeline import PROFILE_NAME as COACH_PROFILE_NAME
from diary.domain.pipelines.coach.pipeline import PROMPT_PATH as COACH_PROMPT_PATH
from diary.domain.pipelines.stt_fix.pipeline import PROFILE_NAME as STT_PROFILE_NAME
from diary.domain.pipelines.stt_fix.pipeline import PROMPT_PATH as STT_PROMPT_PATH
from diary.domain.pipelines.sum_daily.pipeline import NUM_CTX as SUM_DAILY_NUM_CTX
from diary.domain.pipelines.sum_daily.pipeline import PROFILE_NAME as SUM_PROFILE_NAME
from diary.domain.pipelines.sum_daily.pipeline import PROMPT_PATH as SUM_PROMPT_PATH
from diary.domain.pipelines.sum_weekly.pipeline import PROFILE_NAME as WEEK_PROFILE_NAME
from diary.domain.pipelines.sum_weekly.pipeline import PROMPT_PATH as WEEK_PROMPT_PATH

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


def _run(chunk: GoldenChunk, payload: dict[str, object], models_config, judge: Any = None) -> Any:
    """Evaluate one chunk with a scripted provider answering ``payload``."""
    return harness.evaluate_chunk(
        chunk,
        provider=harness.ScriptedProvider(payload),
        models_config=models_config,
        judge=judge,
    )


def _passing(chunk: GoldenChunk) -> dict[str, object]:
    """The CI mock completion of a chunk, as the starting point of a mutation."""
    return harness.ci_payload(chunk)


def _checks(outcome: Any) -> set[str]:
    return set(outcome.failed_checks)


class _AllClaimsContradicted:
    """A scripted NLI judge: every claim of every summary contradicts its source.

    Stateless on purpose — one judge instance serves the whole run, so a judge that
    reacted to the first thing it saw would only affect the first chunk. Exactly
    what the gate of T24b will have to catch, in the largest amount possible.
    """

    def __init__(self) -> None:
        self.rows = 0

    def __call__(self, pairs: Any) -> list[dict[str, float]]:
        self.rows += len(pairs)
        return [{"entailment": 0.03, "contradiction": 0.95, "neutral": 0.02} for _ in pairs]


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


def test_ci_mode_passes_every_chunk(chunks: list[GoldenChunk], models_config) -> None:
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
    summary = harness.summarize(outcomes)
    assert summary.invariants_passed == 11
    assert summary.fully_passed == 11
    # Nine chunks declare a latency budget (three STT, four days, two weeks);
    # the coaching note declares none and is not counted as measured (T21).
    assert summary.latency_measured == 9
    assert summary.latency_within_budget == 9


def test_selecting_chunks_keeps_the_manifest_order(chunks: list[GoldenChunk]) -> None:
    _, selected = harness.load_golden_set(
        chunk_ids=["coach-02-new-project-load", "day-01-quiet-monday"]
    )
    assert [chunk.id for chunk in selected] == ["day-01-quiet-monday", "coach-02-new-project-load"]


# ---------------------------------------------------------------------------
# context budget
# ---------------------------------------------------------------------------


def test_declared_budget_holds_for_every_chunk(chunks: list[GoldenChunk], models_config) -> None:
    """T35: the budget is checked per **call**, so a split day is graded by its parts.

    For a chunk the layer does not split the declared whole-chunk inequality is the
    call's inequality; for the long day (``chunks_planned > 1``) the whole input is
    *deliberately* larger than one call can carry, and comparing it with the window
    is exactly the false failure T35 removed. The per-part inequality is asserted
    here through the layer's own chunker, and the check itself answers with no
    violation.
    """
    for chunk in chunks:
        profile = resolve_phase2_profile(
            models_config, harness.PROFILE_NAME_BY_CATEGORY[chunk.category]
        )
        assert harness.check_budget(chunk, profile=profile) == [], chunk.id
        parts = harness._planned_call_tokens(chunk, profile=profile)
        assert len(parts) >= chunk.budget.chunks_planned, chunk.id
        for part_tokens in parts:
            assert (
                part_tokens
                + harness.prompt_tokens_est_for(chunk.category)
                + PROFILE_MAX_TOKENS[chunk.category]
                <= chunk.budget.num_ctx
            ), chunk.id
        if chunk.budget.chunks_planned == 1:
            assert (
                chunk.budget.chunk_tokens_est
                + harness.prompt_tokens_est_for(chunk.category)
                + PROFILE_MAX_TOKENS[chunk.category]
                <= chunk.budget.num_ctx
            ), chunk.id


def test_the_declared_prompt_overhead_matches_the_template_files() -> None:
    """T20: the declared constant is an expectation the prompt files must meet.

    The harness derives the reserve from the template the layer loads, so a stale
    constant can no longer hide an oversized request. This test keeps the two in
    step in **both** directions: a prompt that grew (or was re-measured) and a
    bumped constant are the only ways to change either without failing here.
    """
    for category, declared in LAYER_PROMPT_TOKENS_EST.items():
        assert harness.prompt_tokens_est_for(category) == declared, category


def test_the_harness_reads_the_prompt_template_its_layer_loads() -> None:
    """The file the harness estimates is the file the layer sends."""
    by_layer = {
        "stt_error": STT_PROMPT_PATH,
        "day": SUM_PROMPT_PATH,
        "week": WEEK_PROMPT_PATH,
        "wellbeing": COACH_PROMPT_PATH,
    }
    for category, path in by_layer.items():
        assert harness.PROMPT_PATH_BY_CATEGORY[category] == path
        assert path.read_text(encoding="utf-8").strip()


def test_the_harness_runs_the_profile_of_each_layer() -> None:
    """F6: the harness must measure the profile the layer really runs under.

    The weekly layer moved to the CPU twin ``diary_week`` in T13b while the
    harness kept sending its chunks through the day profile, so the hybrid
    routing and the tag ``diary-sum-cpu`` had no regression coverage at all.
    Pinning the mapping to the layers' own constants means a rename or a
    re-routing fails here instead of silently changing what the run measures.
    """
    by_layer = {
        "stt_error": STT_PROFILE_NAME,
        "day": SUM_PROFILE_NAME,
        "week": WEEK_PROFILE_NAME,
        "wellbeing": COACH_PROFILE_NAME,
    }
    assert harness.PROFILE_NAME_BY_CATEGORY == by_layer
    # The explicit shape of the F6 defect: the week must not be the day profile.
    assert WEEK_PROFILE_NAME == "diary_week"
    assert harness.PROFILE_NAME_BY_CATEGORY["week"] != SUM_PROFILE_NAME


def test_latency_budgets_are_declared_per_inference_route() -> None:
    """T21: a budget belongs to a route, not to the run."""
    assert harness._latency_budgets() == {"stt_error": 5.0, "day": 15.0, "week": 180.0}
    assert harness._latency_budget_for(_chunk("stt-01-colleague-call")) == 5.0
    assert harness._latency_budget_for(_chunk("day-01-quiet-monday")) == 15.0
    week = _chunk("week-01-2026-w38")
    assert harness.PROFILE_NAME_BY_CATEGORY[week.category] == "diary_week"
    assert harness._latency_budget_for(week) == 180.0
    # The coaching note declares no budget: it must not inherit the weekly one.
    assert harness._latency_budget_for(_chunk("coach-01-low-energy-tuesday")) is None


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
        "week": parsed.diary_week,
        "wellbeing": parsed.diary_coach,
    }
    for category, profile in by_profile.items():
        assert profile is not None, category
        assert PROFILE_MAX_TOKENS[category] == profile.max_tokens


def test_the_coach_generation_cap_leaves_room_for_its_prompt() -> None:
    """T20: with the honest prompt estimate the coach envelope has to close.

    The old blanket cap of 4096 made prompt and cap alone (8209) exceed the 8192
    window, so the declared check passed only because the prompt constant was
    understated.
    """
    prompt_est = harness.prompt_tokens_est_for("wellbeing")
    assert prompt_est + PROFILE_MAX_TOKENS["wellbeing"] <= 8192
    assert PROFILE_MAX_TOKENS["wellbeing"] < 4096


def test_the_prompt_and_the_cap_alone_fit_every_window() -> None:
    """The tightest form of the envelope, stated for every category (T20).

    The chunk-level check subsumes this, but stating it separately makes the
    arithmetic explicit: even a chunk with an empty input must fit, and this is
    exactly the inequality the coaching layer used to violate.
    """
    for category, cap in PROFILE_MAX_TOKENS.items():
        assert harness.prompt_tokens_est_for(category) + cap <= DEFAULT_NUM_CTX, category


def test_a_budget_overflow_is_detected(chunks: list[GoldenChunk], models_config) -> None:
    chunk = _chunk("week-01-2026-w38")
    profile = resolve_phase2_profile(models_config, harness.PROFILE_NAME_BY_CATEGORY["week"])
    reserve = harness.prompt_tokens_est_for("week") + PROFILE_MAX_TOKENS["week"]
    oversized = chunk.model_copy(
        update={
            "budget": chunk.budget.model_copy(
                update={"chunk_tokens_est": chunk.budget.num_ctx - reserve + 1}
            )
        }
    )
    violations = harness.check_budget(oversized, profile=profile)
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


def test_an_invalid_contract_shape_is_a_named_chunk_failure_not_a_dead_run(models_config) -> None:
    """AGENTS.md §15: an invalid completion never reaches the store.

    The layer's own validation fails first, so the harness never sees such a
    payload; since T32 that failure is a *chunk* failure named by its error
    class and the run continues, instead of a traceback that costs the run every
    chunk after this one. The harness keeps its own ``output_contract`` check as
    a second line of defence for a stored payload.
    """
    chunk = _chunk("coach-01-low-energy-tuesday")
    payload = _passing(chunk)
    payload.pop("observations")
    outcome = _run(chunk, payload, models_config)
    assert _checks(outcome) == {f"pipeline_error:{LLMStructuredOutputError.__name__}"}
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
    assert "threshold 8/11" in report
    assert "latency stt_error" in report
    # Each row names the profile the chunk ran under, so a route is visible (T19).
    assert "diary_stt" in report
    # A subset run cannot meet the threshold of the whole set (T18).
    assert "invariants: 2/2 passed (threshold 8/11, 2/11 chunks measured)" in report
    assert "latency: 2/2 within budget" in report


# ---------------------------------------------------------------------------
# T18: the two natures of a failure are counted, and reported, apart
# ---------------------------------------------------------------------------


def test_a_latency_overrun_is_not_a_broken_invariant(models_config) -> None:
    """The count the threshold talks about must survive a cold model load.

    This is the defect T18 closes: the run used to fold a missed latency budget
    into the same number as a violated invariant, so a slow first call read as
    «content broken» and the red gate looked like a checklist violation.
    """
    chunk = _chunk("day-01-quiet-monday")
    outcome = _run(chunk, _passing(chunk), models_config)
    assert outcome.latency_within_budget
    # A cold start: the content is clean, only the clock failed.
    outcome.latency_budget_s = 0.0
    harness._apply_latency_budget(outcome)
    assert not outcome.latency_within_budget
    assert outcome.content_violations == []
    summary = harness.summarize([outcome])
    assert summary.invariants_passed == 1
    assert summary.latency_measured == 1
    assert summary.latency_within_budget == 0
    assert summary.fully_passed == 0
    report = harness.format_report([outcome], harness.golden_manifest())
    assert "FAIL:latency" in report
    assert "FAIL:content" not in report
    assert "invariants: 1/1 passed" in report
    assert "latency: 0/1 within budget" in report


def test_a_content_violation_is_not_counted_as_latency(models_config) -> None:
    """The mirror case: the clock is fine and the content is not."""
    chunk = _chunk("day-01-quiet-monday")
    payload = _passing(chunk)
    payload["facts"] = ["Ты молодец."]
    outcome = _run(chunk, payload, models_config)
    assert outcome.latency_within_budget
    assert outcome.content_violations
    summary = harness.summarize([outcome])
    assert summary.invariants_passed == 0
    assert summary.latency_within_budget == summary.latency_measured == 1
    report = harness.format_report([outcome], harness.golden_manifest())
    assert "FAIL:content" in report
    assert "FAIL:latency" not in report
    assert "invariants: 0/1 passed" in report


def test_the_ci_run_reports_both_counts_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """The CI definition of done, read from the report the CLI prints."""
    assert harness.main(["--strict"]) == 0
    output = capsys.readouterr().out
    assert "OK: invariants 11/11 (required 11)" in output
    assert "latency 9/9 within budget" in output
    # The mock reports no usage, so the measured-token line stays absent.
    assert "measured input tokens" not in output


def test_the_cli_fails_the_run_on_a_latency_overrun(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """T18 keeps latency a gate: the run fails, and it is not a content failure.

    The budget of every chunk is forced to zero, so the only failed check is the
    clock one — exactly the shape of a cold model load. The exit code must be
    nonzero (latency is still a gate) while the invariant count stays clean and
    the message names latency, not content.
    """
    monkeypatch.setattr(harness, "_latency_budget_for", lambda chunk: 0.0)
    assert harness.main(["--chunk", "day-01-quiet-monday"]) == 1
    output = capsys.readouterr().out
    assert "FAILED: latency 1 chunk(s) over budget" in output
    assert "FAIL:latency" in output
    assert "invariants: 1/1 passed" in output


# ---------------------------------------------------------------------------
# a layer switched off is a skip, not a crash (T19 DoD 4, §9.8)
# ---------------------------------------------------------------------------


def _config_without_week(tmp_path) -> Any:
    """The tracked example config with the weekly profile removed."""
    config = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    config["profiles"].pop("diary_week")
    path = tmp_path / "models-without-week.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def test_the_example_config_resolves_every_category_profile(models_config) -> None:
    """Coverage guard: the tracked config must be able to run every category.

    Without this, a profile renamed in the example config would turn a whole
    category into skips and the CI run would still be green — the failure mode
    the skip handling has to avoid.
    """
    for category, profile_name in harness.PROFILE_NAME_BY_CATEGORY.items():
        profile = resolve_phase2_profile(models_config, profile_name)
        assert profile is not None, category


def test_a_config_without_the_weekly_profile_skips_the_week_instead_of_crashing(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """T19 DoD 4: an absent ``diary_week`` means «the layer is off», not a crash.

    The product path reads an absent phase-2 profile as ``profile_off`` and skips
    the layer; before this test the harness raised ``LLMProviderUnavailableError``
    straight out of ``main`` and died with a traceback, so the golden run could
    not be used at all on a configuration without the weekly profile.
    """
    config = _config_without_week(tmp_path)
    assert harness.main(["--models-config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "SKIP:profile_off" in output
    assert "skipped (profile off): 2 chunk(s)" in output
    # The skipped chunks are outside the counts: nine were measured, not eleven.
    assert "invariants: 9/9 passed" in output
    assert "9/11 chunks measured" in output
    assert "OK: invariants 9/9 (required 8)" in output


def test_a_run_that_measured_nothing_fails_closed(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifying nothing is not the same as verifying that everything is fine."""
    config = _config_without_week(tmp_path)
    code = harness.main(["--chunk", "week-01-2026-w38", "--models-config", str(config)])
    assert code == 1
    output = capsys.readouterr().out
    assert "FAILED: nothing was measured" in output
    assert "SKIP:profile_off" in output


def test_the_skipped_category_is_not_reported_as_measured(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`latency week: max 0.00s` next to `latency: not measured` reads as a lie."""
    config = _config_without_week(tmp_path)
    harness.main(["--models-config", str(config)])
    output = capsys.readouterr().out
    assert "latency week: n/a (skipped, layer off)" in output
    assert "latency week: max" not in output


def test_strict_is_never_softer_than_a_plain_run(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """T19 fix review: `--strict` must not pass where the plain run fails.

    With only the STT profile configured, eight chunks are skipped. The plain run
    already fails (three measured, threshold eight), and `--strict` — which demands
    full coverage — must not turn that into a green run.
    """
    config = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    # Keep the MVP profiles and the STT profile; every other phase-2 layer is off.
    config["profiles"] = {
        name: value
        for name, value in config["profiles"].items()
        if not name.startswith("diary_") or name == "diary_stt"
    }
    path = tmp_path / "models-stt-only.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    assert harness.main(["--models-config", str(path)]) == 1
    plain = capsys.readouterr().out
    assert "FAILED: invariants 3/3 measured, 8 skipped (profile off), 8 required" in plain
    assert harness.main(["--strict", "--models-config", str(path)]) == 1
    strict = capsys.readouterr().out
    assert "FAILED: invariants 3/3 measured, 8 skipped (profile off), 11 required" in strict
    # A single selected chunk that is measured still passes on the same config.
    code = harness.main(["--chunk", "stt-01-colleague-call", "--models-config", str(path)])
    assert code == 0


def test_a_skipped_chunk_is_excluded_from_both_counts() -> None:
    """The skip must not inflate the score, in either counter."""
    chunk = _chunk("week-01-2026-w38")
    skipped = harness._skipped_outcome(chunk)
    assert skipped.skipped
    assert skipped.skipped_reason == "profile_off"
    assert skipped.violations == []
    assert harness._result_label(skipped) == "SKIP:profile_off"
    summary = harness.summarize([skipped])
    assert summary.skipped == 1
    assert summary.measured == 0
    assert summary.invariants_passed == 0
    assert summary.latency_measured == 0
    assert summary.fully_passed == 0


# ---------------------------------------------------------------------------
# T24a: the NLI measurement (--nli), the premise, and fail-closed availability
# ---------------------------------------------------------------------------


def test_the_nli_measurement_is_off_unless_it_is_requested(
    chunks: list[GoldenChunk], models_config, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing is judged, nothing is loaded, and the report says nothing."""
    outcomes = harness.evaluate_all(
        chunks, provider_factory=harness.ci_provider_factory, models_config=models_config
    )
    summary = harness.summarize(outcomes)
    assert summary.nli_summaries == 0
    assert summary.nli_claims == 0
    assert "nli (" not in harness.format_report(outcomes, harness.golden_manifest())
    assert harness.main(["--strict"]) == 0
    assert "nli (" not in capsys.readouterr().out


def test_the_nli_block_measures_without_touching_the_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """T24a is a measurement: the counters of T18 and T21 must not move."""
    monkeypatch.setattr(harness, "nli_judge", _AllClaimsContradicted)
    assert harness.main(["--strict", "--nli"]) == 0
    output = capsys.readouterr().out
    assert "OK: invariants 11/11 (required 11)" in output
    assert "latency 9/9 within budget" in output
    # Six summaries, forty claims: three facts per short day, twenty-nine for the
    # long day of T35 (the mock answers a part with the facts it carries, and the
    # fold-back with all of them), and one weekly rollup per week. The judge
    # condemns every pair, yet only the thirty-eight *propositional* claims — the
    # days' facts — can be contradicted; the two weekly rollups are outside the
    # rule's scope and land in the diagnostic column instead.
    assert "nli (measurement, not a gate): summaries 6, claims 40" in output
    assert "contradictions 38, unsupported 2" in output
    assert "nli flagged contradictions: day-01-quiet-monday facts#1" in output


def test_the_metrics_of_a_scripted_judge_land_in_the_outcome(models_config) -> None:
    chunk = _chunk("day-01-quiet-monday")
    outcome = _run(chunk, _passing(chunk), models_config, judge=_AllClaimsContradicted())
    assert outcome.nli is not None
    assert outcome.nli.claims == len(outcome.nli.contradicted_coordinates)
    assert outcome.nli.contradicted_coordinates[0] == "facts#1"
    assert outcome.nli.max_contradiction == pytest.approx(0.95)
    # A chunk whose contract declares no claims is not judged at all.
    stt = _chunk("stt-01-colleague-call")
    assert _run(stt, _passing(stt), models_config, judge=_AllClaimsContradicted()).nli is None


def test_a_missing_instrument_fails_the_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The owner's decision for T24: an unavailable instrument is a failed run.

    A requested check that silently does nothing would let the report claim a
    validation nobody performed, so the run stops and the message names the fix.
    """
    missing = OnnxNliJudge(
        model_path=tmp_path / "model.onnx",
        tokenizer_path=tmp_path / "tokenizer.json",
        config_path=tmp_path / "config.json",
    )
    monkeypatch.setattr(harness, "nli_judge", lambda: missing)
    assert harness.main(["--strict", "--nli"]) == 1
    output = capsys.readouterr().out
    assert "FAILED: the NLI check could not run" in output
    # The message names whichever fix applies: the extra or the weights (CI has
    # neither, a developer usually has the extra and lacks the 712 MB).
    assert "fetch-nli-model.sh" in output or 'pip install -e ".[nli]"' in output
    # Without the flag the same missing instrument is irrelevant, as designed.
    assert "traceback" not in output.lower()


def test_a_run_that_verified_nothing_with_nli_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A requested check that checked nothing must not look like a green run.

    The first review of T24a found this hole: with a working instrument and a
    selection that contains no summary chunk (an STT chunk has no claims), the
    block was simply absent and the run exited zero — «nothing to check» reading as
    «checked».
    """
    monkeypatch.setattr(harness, "nli_judge", _AllClaimsContradicted)
    code = harness.main(["--chunk", "stt-01-colleague-call", "--nli"])
    assert code == 1
    output = capsys.readouterr().out
    assert "the NLI check verified nothing" in output


def test_an_unavailable_instrument_fails_even_without_summaries_in_scope(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The instrument is proved before the run, not lazily on the first claim."""
    missing = OnnxNliJudge(
        model_path=tmp_path / "model.onnx",
        tokenizer_path=tmp_path / "tokenizer.json",
        config_path=tmp_path / "config.json",
    )
    monkeypatch.setattr(harness, "nli_judge", lambda: missing)
    assert harness.main(["--chunk", "stt-01-colleague-call", "--nli"]) == 1
    output = capsys.readouterr().out
    assert "FAILED: the NLI check could not run" in output
    assert "verified nothing" not in output


def test_the_harness_arms_the_layer_gate_when_asked(models_config) -> None:
    """T24b: the gate judge reaches the layer, so a live run measures the gated layer.

    A live harness run passes the instrument the product uses; the check below is
    that the plumbing really gets there — a contradicting judge makes the *layer*
    refuse, and the harness grades the refusal as the failure of that check.
    """
    chunk = _chunk("day-01-quiet-monday")
    outcome = harness.evaluate_chunk(
        chunk,
        provider=harness.ScriptedProvider(_passing(chunk)),
        models_config=models_config,
        gate_judge=_AllClaimsContradicted(),
    )
    assert "no_contradiction_with_source" in outcome.failed_checks
    assert outcome.requests >= 2  # the gate regenerated before refusing

    # The same run without the gate judge stays clean: the seam is the caller's.
    assert _run(chunk, _passing(chunk), models_config).failed_checks == []


def test_a_judge_that_answers_the_wrong_shape_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(harness, "nli_judge", lambda: lambda pairs: [])
    assert harness.main(["--strict", "--nli"]) == 1
    assert "FAILED: the NLI check could not run" in capsys.readouterr().out


def test_the_nli_premise_is_the_content_of_the_layer_input(models_config) -> None:
    """Only what the layer asserted about: entry texts, no ids, no paths."""
    day = _chunk("day-01-quiet-monday")
    premise = harness._nli_premise(day)
    assert premise == "\n".join(entry.text for entry in day.input)
    assert str(day.input[0].id) not in premise

    week = _chunk("week-01-2026-w38")
    assert harness._nli_premise(week) == "\n".join(
        part
        for summary in week.input
        for part in (*summary.facts, *summary.themes, *summary.open_questions)
    )


def test_an_empty_premise_is_an_error_not_a_clean_result(models_config) -> None:
    """The measurement that broke once: an empty source must not read as «neutral»."""
    chunk = _chunk("day-01-quiet-monday")
    blank = chunk.model_copy(
        update={"input": [entry.model_copy(update={"text": ""}) for entry in chunk.input]}
    )
    with pytest.raises(NliPremiseError):
        _run(blank, _passing(blank), models_config, judge=_AllClaimsContradicted())


def test_the_report_names_the_measured_prompt_when_the_endpoint_reports_it(
    models_config,
) -> None:
    """T20 evidence in the report itself: estimate against the real count.

    The wording says «max call» because that is what the figure is since T35: the
    largest prompt among the calls the layer made, not the last one.
    """
    chunk = _chunk("day-01-quiet-monday")
    outcome = _run(chunk, _passing(chunk), models_config)
    outcome.usage = harness.LLMUsage(input_tokens=1871, output_tokens=214)
    report = harness.format_report([outcome], harness.golden_manifest())
    assert "measured input tokens (endpoint), max call per category: day 1871 (cap 4096)" in report


# ---------------------------------------------------------------------------
# a deterministic layer failure is a failed chunk, not a dead run (T32)
# ---------------------------------------------------------------------------


def _raise(error: Exception):
    """A stand-in layer that fails the way the real one can."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise error

    return boom


@pytest.mark.parametrize(
    "error_type",
    [
        PipelineInputError,
        PipelineOutputError,
        PipelinePermanentError,
        PipelineNotRegisteredError,
        PipelineStorageError,
    ],
)
def test_every_permanent_layer_error_is_isolated_under_its_own_name(
    error_type: type[PipelinePermanentError],
    monkeypatch: pytest.MonkeyPatch,
    models_config,
) -> None:
    """Every deterministic layer error names the chunk's failure, none ends the run.

    The classes are the whole ``PipelinePermanentError`` family a phase-2 layer can
    raise (``PipelineInvariantError`` has its own branch and its own test below).
    """
    chunk = _chunk("day-01-quiet-monday")
    monkeypatch.setattr(harness, "sum_daily", _raise(error_type("synthetic permanent failure")))
    outcome = _run(chunk, _passing(chunk), models_config)
    assert not outcome.passed
    assert _checks(outcome) == {f"{harness.PIPELINE_ERROR_CHECK}:{error_type.__name__}"}


def test_a_day_that_cannot_be_split_is_isolated_as_a_named_chunk_failure(models_config) -> None:
    """The T23 case: the day layer raises ``PipelinePermanentError`` on its budget.

    A generation cap that leaves no input budget makes ``chunk_entries`` refuse
    the day — the failure that used to end the run with a traceback the moment a
    budget was broken.
    """
    chunk = _chunk("day-01-quiet-monday")
    profile = resolve_phase2_profile(models_config, SUM_PROFILE_NAME)
    no_room = profile.model_copy(update={"max_tokens": SUM_DAILY_NUM_CTX})
    outcome = harness.evaluate_chunk(
        chunk,
        provider=harness.ScriptedProvider(_passing(chunk)),
        models_config=models_config,
        profile=no_room,
    )
    assert not outcome.passed
    assert _checks(outcome) == {f"pipeline_error:{PipelinePermanentError.__name__}"}


def test_a_permanent_failure_keeps_the_rest_of_the_grading(
    monkeypatch: pytest.MonkeyPatch, models_config
) -> None:
    """The branch is not a shortcut: the latency budget is still applied to it."""
    chunk = _chunk("day-01-quiet-monday")
    monkeypatch.setattr(harness, "sum_daily", _raise(PipelineInputError("the day is unusable")))
    monkeypatch.setattr(harness, "_latency_budget_for", lambda _chunk: -1.0)
    outcome = _run(chunk, _passing(chunk), models_config)
    assert _checks(outcome) == {
        f"{harness.PIPELINE_ERROR_CHECK}:{PipelineInputError.__name__}",
        harness.LATENCY_CHECK,
    }


def test_a_harness_error_stays_fatal(monkeypatch: pytest.MonkeyPatch, models_config) -> None:
    """A broken *measurement* must not be graded as a broken chunk."""
    chunk = _chunk("stt-01-colleague-call")
    monkeypatch.setattr(
        harness, "correct_stt", _raise(harness.HarnessError("the harness itself cannot run"))
    )
    with pytest.raises(harness.HarnessError):
        _run(chunk, _passing(chunk), models_config)


def test_a_permanent_error_does_not_take_the_remaining_chunks(
    chunks: list[GoldenChunk], models_config
) -> None:
    """T32: the run keeps measuring after a chunk the layer cannot produce."""
    broken = "day-02-long-tuesday"
    broken_payload = _passing(_chunk(broken))
    broken_payload.pop("facts")  # the contract requires a non-empty list

    def factory(chunk: GoldenChunk) -> Any:
        # The healthy chunks get the ordinary CI provider (which answers a split day
        # call by call, T35); only the broken one gets the malformed completion.
        if chunk.id == broken:
            return harness.ScriptedProvider(broken_payload)
        return harness.ci_provider_factory(chunk)

    outcomes = harness.evaluate_all(chunks, provider_factory=factory, models_config=models_config)
    assert [outcome.chunk_id for outcome in outcomes] == [chunk.id for chunk in chunks]
    failed = {outcome.chunk_id: _checks(outcome) for outcome in outcomes if not outcome.passed}
    assert failed == {broken: {"pipeline_error:LLMStructuredOutputError"}}
    assert harness.summarize(outcomes).invariants_passed == len(chunks) - 1


def test_strict_counts_a_permanent_failure_as_a_failure_not_a_skip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The DoD is explicit: ``--strict`` must read it as a failure, never a skip."""
    monkeypatch.setattr(harness, "sum_daily", _raise(PipelineInputError("the day is unusable")))
    assert harness.main(["--strict", "--chunk", "day-01-quiet-monday"]) == 1
    output = capsys.readouterr().out
    assert "pipeline_error:PipelineInputError" in output
    assert "FAILED:" in output
    assert "skipped" not in output


def test_an_invariant_refusal_keeps_its_own_grading(models_config) -> None:
    """The gate's refusal is not folded into the permanent-error bucket (T17b)."""
    chunk = _chunk("day-01-quiet-monday")
    payload = _passing(chunk)
    payload["facts"] = ["Ты молодец, успела многое."]
    outcome = _run(chunk, payload, models_config)
    assert _checks(outcome) == {"advice_and_judgement_absent"}
    assert not any(check.startswith("pipeline_error") for check in _checks(outcome))


def test_the_report_names_the_error_without_carrying_its_message(
    monkeypatch: pytest.MonkeyPatch, models_config
) -> None:
    """Privacy: the row names the class; the error text stays in ``detail``."""
    chunk = _chunk("week-01-2026-w38")
    monkeypatch.setattr(
        harness, "sum_weekly", _raise(PipelineInputError("синтетическая строка из теста"))
    )
    outcome = _run(chunk, _passing(chunk), models_config)
    row = harness.format_report([outcome], harness.golden_manifest())
    assert "pipeline_error:PipelineInputError" in row
    assert "синтетическая строка из теста" not in row


# ---------------------------------------------------------------------------
# grounding: a moment and an anchor are not facts about the diary (T32)
# ---------------------------------------------------------------------------


def _time_claim(chunk: GoldenChunk, hour: str) -> dict[str, object]:
    payload = _passing(chunk)
    payload["experiment_7d"] = f"В течение следующих 7 дней отмечать, что происходит в {hour}."
    return payload


def test_an_anchor_date_does_not_ground_a_number_of_the_output(models_config) -> None:
    """T32: the date ``2026-09-23`` in ``daily_ids`` must not ground a claimed hour.

    Before T32 the same invented «23» was refused on one week of the golden set
    and accepted on another purely because the other contained the 23rd — the
    weakening this check closes (numbers in §16 and ``ADR-012``).
    """
    chunk = _chunk("coach-02-new-project-load")
    assert "2026-09-23" in chunk.input.daily_ids  # the anchor really carries the digit
    outcome = _run(chunk, _time_claim(chunk, "23"), models_config)
    assert _checks(outcome) == {"no_new_facts"}


def test_an_entry_timestamp_does_not_ground_a_number_of_the_output(models_config) -> None:
    """The same for a day chunk, whose entries carry the moment in ``ts``."""
    chunk = _chunk("day-01-quiet-monday")
    content = " ".join(entry.text for entry in chunk.input)
    hours = {f"{entry.ts.hour:02d}" for entry in chunk.input} - harness.numbers(content)
    assert hours, "the probe needs an hour the entries themselves never mention"
    hour = sorted(hours)[0]
    payload = _passing(chunk)
    payload["facts"] = [*payload["facts"], f"В записях упомянуто время {hour}."]
    outcome = _run(chunk, payload, models_config)
    assert _checks(outcome) == {"no_new_facts"}


def test_record_content_still_grounds_a_number_of_the_output(models_config) -> None:
    """The narrowing must not touch numbers the entries really say."""
    chunk = _chunk("coach-01-low-energy-tuesday")
    payload = _passing(chunk)
    payload["observations"] = [*payload["observations"][:2], "В сводке упомянуто 1 километр."]
    outcome = _run(chunk, payload, models_config)
    assert outcome.passed, outcome.violations


def test_the_form_of_the_project_fixture_is_refused_on_both_weeks(models_config) -> None:
    """T32 point 2, decided by measurement: the hour is invented, the refusal is right.

    The literal form of ``tests/unit/test_contracts_schemas.py`` names a moment no
    weekly rollup of the golden set states, so the gate refuses it — on the week
    that contains the 23rd exactly as on the week that does not. Nothing is
    weakened for it; the numbers are in ``ADR-012`` and §16.
    """
    for chunk_id in ("coach-01-low-energy-tuesday", "coach-02-new-project-load"):
        chunk = _chunk(chunk_id)
        payload = _passing(chunk)
        payload["experiment_7d"] = "Попробовать 7 дней ложиться до 23:00."
        outcome = _run(chunk, payload, models_config)
        assert _checks(outcome) == {"no_new_facts"}, chunk_id
