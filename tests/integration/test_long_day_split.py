"""T35: the long golden day, the parts of a split day, and the harness budget.

The finding this file answers (T23 acceptance, recorded in ``ADR-016``
§ Consequences 4): the day layer splits a long day into parts, while the harness
counted the input budget of **one** call — ``check_budget`` compared the whole
chunk with the window and ``measured_context_budget`` read the usage of the
**last** call (the fold-back). On a long day that is a false ``context_budget``
and no measurement of the parts at all.

What is proven here, in the order of the definition of done:

* the new golden chunk ``day-04-long-wednesday`` really does not fit one window,
  and the whole-chunk comparison the harness used to make would have failed it —
  so the fixture is a regression case and not decoration;
* ``check_budget`` grades such a chunk **per part** (the layer's own chunker is the
  single implementation of the split) and reports a broken declaration as
  ``expected_parts`` instead of trusting it;
* the measured figure is the largest call of the chunk, and the report names it and
  the call count;
* the fold-back carries every part, and the summary of the parts is the summary of
  one call **for a provider that is a function of the content it is given** — the
  honest structural baseline. The quality of a real model's fold-back is **not**
  measured here and cannot be: no reference answer exists for a synthetic long day,
  and a live run is non-deterministic. That limit is stated, not hidden.

Everything is synthetic (AGENTS.md §7); the provider is deterministic and no
endpoint is contacted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from typing import Any

import pytest

from diary.core.models_config import ModelsConfig, OpenAICompatibleProfile, load_models_config
from diary.core.settings import PROJECT_ROOT
from diary.db.models import ArtifactLayer
from diary.db.results import list_layer_results
from diary.domain.contracts import DailySummary, RawEntry
from diary.domain.contracts.golden import (
    DEFAULT_NUM_CTX,
    PROFILE_MAX_TOKENS,
    GoldenChunk,
    estimate_tokens,
)
from diary.domain.llm.models import LLMRequest, LLMResponse, LLMUsage
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.pipelines.sum_daily.pipeline import (
    ENTRIES_HEADER,
    PARTS_HEADER,
    SumDailyPipeline,
)
from diary.domain.pipelines.sum_daily.pipeline import (
    NUM_CTX as SUM_DAILY_NUM_CTX,
)

EVAL_RUN = PROJECT_ROOT / "eval" / "run.py"

#: The chunk this task added: more messages than one call can carry.
LONG_DAY = "day-04-long-wednesday"


def _load_harness() -> Any:
    """Import ``eval/run.py`` by path (it is not an installed package)."""
    spec = importlib.util.spec_from_file_location("eval_run_t35", EVAL_RUN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_run_t35"] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


@pytest.fixture(scope="module")
def models_config() -> ModelsConfig:
    """CI configuration: the tracked example config (no endpoint is contacted)."""
    return load_models_config(harness.CI_MODELS_CONFIG)


@pytest.fixture(scope="module")
def long_day() -> GoldenChunk:
    _, chunks = harness.load_golden_set(chunk_ids=[LONG_DAY])
    return chunks[0]


@pytest.fixture(scope="module")
def day_profile(models_config: ModelsConfig) -> OpenAICompatibleProfile:
    """The day profile of the tracked example configuration (cap 4096)."""
    return resolve_phase2_profile(models_config, harness.PROFILE_NAME_BY_CATEGORY["day"])


def _entries(chunk: GoldenChunk) -> list[Any]:
    return list(chunk.input)


def _entry_texts(chunk: GoldenChunk) -> list[str]:
    return [entry.text for entry in _entries(chunk)]


def _parts(chunk: GoldenChunk, profile: OpenAICompatibleProfile) -> list[list[Any]]:
    return SumDailyPipeline().chunk_entries(_entries(chunk), profile=profile)


def _whole_day_request_tokens(chunk: GoldenChunk) -> int:
    """Estimated tokens of the one call the layer would make for the whole day."""
    request = SumDailyPipeline().build_request(entries=_entries(chunk))
    return estimate_tokens(len(request.messages[0].content))


def _part_payloads(requests: list[LLMRequest]) -> list[list[str]]:
    """The facts each fold-back input carried, one entry per fold-back request."""
    payloads: list[list[str]] = []
    for request in requests:
        content = request.messages[0].content
        if PARTS_HEADER not in content:
            continue
        body = json.loads(content.rsplit(PARTS_HEADER, 1)[-1].strip())
        payloads.append([fact for part in body for fact in part["facts"]])
    return payloads


def _requests_of_a_ci_run(chunk: GoldenChunk, models_config: ModelsConfig) -> list[LLMRequest]:
    """The requests of one CI run of ``chunk`` (the provider is kept by hand)."""
    provider = harness.ci_provider_factory(chunk)
    harness.evaluate_chunk(chunk, provider=provider, models_config=models_config)
    assert isinstance(provider, harness.ScriptedProvider)
    return list(provider.requests)


def _stored_summary(chunk: GoldenChunk, *, profile: OpenAICompatibleProfile) -> DailySummary:
    """Run the day layer on a throwaway database and read back what it stored."""
    provider = harness.ci_provider_factory(chunk)
    with harness.temp_database() as session:
        harness._seed_chunk_input(session, chunk)
        assert harness._collected_day_matches(session, chunk)
        harness._run_layer(session, chunk, profile=profile, provider=provider)
        session.commit()
        stored = list(list_layer_results(session, layer=ArtifactLayer.daily_summary))
        payload = harness._latest_result_payload(session, chunk)
    assert len(stored) == 1, "a split day stores one artifact, not one per part"
    return DailySummary.model_validate(dict(payload))


# ---------------------------------------------------------------------------
# the fixture: a day that really does not fit one window
# ---------------------------------------------------------------------------


def test_the_long_day_does_not_fit_one_window(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile
) -> None:
    """The chunk is longer than one call: the layer's own rule says so."""
    assert _whole_day_request_tokens(long_day) > SumDailyPipeline.input_budget_tokens(day_profile)
    parts = _parts(long_day, day_profile)
    assert len(parts) >= 2
    # The parts partition the day in order: no message is lost or duplicated.
    assert [entry.id for part in parts for entry in part] == [
        entry.id for entry in _entries(long_day)
    ]
    # The manifest declares the split as a lower bound: the harness grades that the
    # layer really makes at least this many calls (the real count follows from the
    # layer's own budget rule and may be higher).
    assert long_day.budget.chunks_planned >= 2
    # Splitting a day does not widen the window: only a weekly chunk in pieces does.
    assert long_day.budget.num_ctx == DEFAULT_NUM_CTX


def test_the_whole_chunk_comparison_would_false_fail_the_long_day(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile
) -> None:
    """T35's defect as arithmetic: the whole-chunk check says «overflow», the truth is «parts».

    Before T35 ``check_budget`` compared ``chunk_tokens_est + reserve`` with the
    window — for this chunk that sum is **larger** than the window although every
    call the layer makes fits it. This is the false ``context_budget`` the task
    removed; the check now answers with no violation for the same chunk, and each
    part fits the declared envelope.
    """
    reserve = harness.prompt_tokens_est_for("day") + PROFILE_MAX_TOKENS["day"]
    assert long_day.budget.chunk_tokens_est + reserve > long_day.budget.num_ctx
    assert harness.check_budget(long_day, profile=day_profile) == []
    parts = harness._planned_call_tokens(long_day, profile=day_profile)
    assert len(parts) == len(_parts(long_day, day_profile))
    for part_tokens in parts:
        assert part_tokens + reserve <= long_day.budget.num_ctx


def test_a_part_that_does_not_fit_the_window_is_named_as_a_context_budget_violation(
    monkeypatch: pytest.MonkeyPatch, long_day: GoldenChunk, day_profile: OpenAICompatibleProfile
) -> None:
    """The per-part branch has teeth: an oversized part is reported, and named.

    The parts are computed with the layer's own chunker, so a part larger than the
    window cannot arise from the tracked configuration (the two envelopes coincide
    there). The branch is therefore exercised directly: what is asserted is that the
    check compares **each part** with the window and names the part that failed,
    instead of summing the parts into one number.
    """
    monkeypatch.setattr(
        harness,
        "_planned_call_tokens",
        # Two parts, as the chunk declares, but each of them over the window.
        lambda _chunk, *, profile: [long_day.budget.num_ctx, long_day.budget.num_ctx],
    )
    violations = harness.check_budget(long_day, profile=day_profile)
    assert [violation.check for violation in violations] == ["context_budget", "context_budget"]
    assert "part 1" in violations[0].detail
    assert "part 2" in violations[1].detail


def test_a_declared_split_the_layer_does_not_make_is_a_violation(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile
) -> None:
    """A declaration is graded, not trusted: one declared part more than the layer makes.

    The real count belongs to the layer's own budget rule and moves with it (T39
    closed the parts of this chunk earlier, so the count is no longer the two of
    T35), and the declaration is built from that count instead of a number the
    rule can outgrow.
    """
    real = len(_parts(long_day, day_profile))
    declared = long_day.model_copy(
        update={"budget": long_day.budget.model_copy(update={"chunks_planned": real + 1})}
    )
    violations = harness.check_budget(declared, profile=day_profile)
    assert [violation.check for violation in violations] == [harness.PARTS_CHECK]
    assert str(real + 1) in violations[0].detail


def test_a_declared_split_that_cannot_be_made_is_a_violation(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile
) -> None:
    """A cap that leaves the layer no input budget: the declaration cannot hold."""
    no_room = day_profile.model_copy(update={"max_tokens": SUM_DAILY_NUM_CTX})
    violations = harness.check_budget(long_day, profile=no_room)
    assert [violation.check for violation in violations] == [harness.PARTS_CHECK]


# ---------------------------------------------------------------------------
# the run: parts, one fold-back, and the fold-back carries everything
# ---------------------------------------------------------------------------


def test_the_long_day_runs_as_parts_plus_one_fold_back(
    long_day: GoldenChunk, models_config: ModelsConfig, day_profile: OpenAICompatibleProfile
) -> None:
    """One call per part plus the fold-back, and the chunk is clean in CI."""
    provider = harness.ci_provider_factory(long_day)
    outcome = harness.evaluate_chunk(long_day, provider=provider, models_config=models_config)
    assert isinstance(provider, harness.ScriptedProvider)
    assert outcome.requests == len(_parts(long_day, day_profile)) + 1
    assert outcome.violations == []


def test_the_fold_back_carries_every_part_and_the_whole_day(
    long_day: GoldenChunk, models_config: ModelsConfig
) -> None:
    """No part is lost on the way in: the fold-back input holds the whole day's facts.

    The parts partition the day in order (the layer's own rule) and the fold-back
    request carries the facts of every part, so the summary the layer stores rests
    on all of them. The assertions read the requests the run really made — the
    messages each call carried — not a reconstruction of them.
    """
    requests = _requests_of_a_ci_run(long_day, models_config)
    texts = _entry_texts(long_day)
    part_requests = [
        request for request in requests if ENTRIES_HEADER in request.messages[0].content
    ]
    assert len(part_requests) >= 2
    assert len(_part_payloads(requests)) == 1  # exactly one fold-back request
    assert _part_payloads(requests)[0] == texts
    delivered: list[str] = []
    for request in part_requests:
        body = json.loads(request.messages[0].content.rsplit(ENTRIES_HEADER, 1)[-1].strip())
        delivered.extend(entry["text"] for entry in body)
    assert delivered == texts


def test_the_part_the_prose_estimate_accepted_is_refused_by_the_gate(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile
) -> None:
    """T39 as arithmetic on this chunk: the call the old gate let through was over budget.

    The live run of 2026-09-22 measured **4493** input tokens for the biggest part
    the prose estimate closed (``eval/golden/README.md``); ``4493 + 4096 > 8192``,
    so that call could not fit the window, and the harness named it
    (``measured_context_budget``). The regression is the same claim on the chunk's
    own input: the part the prose rule accepts is under the budget by that rule,
    over it by the layer's gate, and the day the gate splits has no such part.
    """
    pipeline = SumDailyPipeline()
    budget = SumDailyPipeline.input_budget_tokens(day_profile)
    accepted = _entries(long_day)[:22]  # the first part the prose rule closed
    request = pipeline.build_request(entries=accepted)
    content = request.messages[0].content

    assert estimate_tokens(len(content)) <= budget  # the prose rule accepted the call
    assert not pipeline.request_fits(request, profile=day_profile)  # the gate does not
    # ...because the estimate is now above the endpoint's own count for it.
    assert SumDailyPipeline.content_tokens(content) >= 4493

    parts = _parts(long_day, day_profile)
    assert len(parts) >= 3, "the gate closes the parts earlier than the prose rule did"
    for part in parts:
        assert pipeline.request_fits(pipeline.build_request(entries=part), profile=day_profile)


def test_the_correction_does_not_split_a_golden_day_that_fits(
    day_profile: OpenAICompatibleProfile,
) -> None:
    """T39 DoD 4 on the corpus: only the day that measured over the window changes.

    The three short golden days (three entries each) are one call before and after
    the correction — the coefficient must not start splitting a day that fits — and
    the long day, whose biggest part measured 4493, is split further than the
    declared two parts. The verdict per chunk is read from the layer's own chunker,
    not assumed from the coefficient.
    """
    _manifest, chunks = harness.load_golden_set()
    pipeline = SumDailyPipeline()
    day_chunks = [chunk for chunk in chunks if chunk.category == "day"]
    assert len(day_chunks) == 4  # three short days and the long one
    for chunk in day_chunks:
        entries = [item for item in chunk.input if isinstance(item, RawEntry)]
        parts = pipeline.chunk_entries(entries, profile=day_profile)
        if chunk.budget.chunks_planned == 1:
            assert len(parts) == 1, chunk.id
        else:
            assert chunk.id == LONG_DAY
            assert len(parts) >= 3, chunk.id


def test_the_summary_of_the_parts_equals_the_summary_of_one_call(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile
) -> None:
    """The structural baseline DoD 4 asks for, and the exact scope of it.

    A reference answer for a **model's** fold-back does not exist (the day is
    synthetic, a live run is non-deterministic), so the quality of summarisation
    cannot be measured. What can be measured — and is measured here — is the
    layer's structure: with a provider that answers as a function of the content it
    is given, the summary of the parts is the summary of one call. The single-call
    arm is the same day under a profile whose cap leaves room for the whole day;
    both arms store **one** ``DailySummary`` under the day anchor, with the same
    facts in the same order, resting on every entry.

    Teeth: a fold-back that lost a part, or a split that dropped a message, breaks
    the equality — the mutation is reproduced in the T35 acceptance notes.
    """
    one_call_profile = day_profile.model_copy(update={"max_tokens": 1024})
    assert len(_parts(long_day, one_call_profile)) == 1, "the baseline arm must be a single call"

    split = _stored_summary(long_day, profile=day_profile)
    single_call = _stored_summary(long_day, profile=one_call_profile)

    assert split.facts == single_call.facts
    assert split.facts == _entry_texts(long_day)
    assert split.date == single_call.date
    assert split.energy == single_call.energy


# ---------------------------------------------------------------------------
# measurements: the largest call, not the last one
# ---------------------------------------------------------------------------


class _UsageProvider:
    """A provider that reports the usage of each call from a script."""

    def __init__(self, chunk: GoldenChunk, usages: list[LLMUsage]) -> None:
        self._inner = harness.ci_provider_factory(chunk)
        self._usages = usages
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        usage = self._usages[min(self.calls, len(self._usages) - 1)]
        self.calls += 1
        return self._inner.complete(request).model_copy(update={"usage": usage})


def _outcome_with_usages(
    chunk: GoldenChunk, usages: list[LLMUsage], models_config: ModelsConfig
) -> Any:
    """Evaluate ``chunk`` through the live-shaped measurement wrapper."""
    provider = harness.MeasuringProvider(_UsageProvider(chunk, usages))
    return harness.evaluate_chunk(chunk, provider=provider, models_config=models_config)


def test_the_measured_figure_is_the_largest_call_not_the_last(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile, models_config: ModelsConfig
) -> None:
    """T35 DoD 2: the report names the largest part, not the fold-back.

    The number of calls is the layer's own (T39 made it larger for this chunk), so
    the scripted usage follows the real count: a large first part, smaller parts,
    and the smallest last call (the fold-back).
    """
    parts = len(_parts(long_day, day_profile))
    usages = [
        LLMUsage(input_tokens=3000, output_tokens=200),  # first part
        *[LLMUsage(input_tokens=1200, output_tokens=200) for _ in range(parts - 1)],
        LLMUsage(input_tokens=100, output_tokens=200),  # the fold-back
    ]
    outcome = _outcome_with_usages(long_day, usages, models_config)
    assert outcome.requests == parts + 1
    assert outcome.usage is not None and outcome.usage.input_tokens == 3000
    report = harness.format_report([outcome], harness.golden_manifest())
    assert "max call per category: day 3000 (cap 4096)" in report
    # The row names the call count, so a split day is visible as one: the columns
    # are chunk, category, profile, latency, calls, slowest, result, failed checks.
    row = next(line for line in report.splitlines() if line.startswith(LONG_DAY))
    assert row.split()[4] == str(parts + 1)


def test_a_large_part_call_is_caught_by_the_measured_budget(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile, models_config: ModelsConfig
) -> None:
    """The defect itself: a huge first call used to hide behind a small last one.

    The window is a property of one call, so the check compares the **largest**
    measured call. With the fold-back small and a part over the window, the old code
    (``last_usage``) stayed silent while a call really did overflow. The call count
    is the layer's own, as in the test above.
    """
    usages = [
        LLMUsage(input_tokens=8000, output_tokens=10),  # a part over the window
        *[
            LLMUsage(input_tokens=50, output_tokens=10)
            for _ in range(len(_parts(long_day, day_profile)))
        ],
    ]
    outcome = _outcome_with_usages(long_day, usages, models_config)
    assert outcome.failed_checks == ["measured_context_budget"]
    assert "8000" in outcome.violations[0].detail
    assert f"largest of {outcome.requests} call(s)" in outcome.violations[0].detail


# ---------------------------------------------------------------------------
# T40: the unit of the latency budget is one call
# ---------------------------------------------------------------------------


class _ScriptedClock:
    """A clock that reads one scripted run and refuses to be read twice over.

    Counting the readings is the point: a run whose clock is read more often than
    the script describes would otherwise keep a stale value and measure nothing.
    """

    def __init__(self, readings: list[float]) -> None:
        self._readings = readings
        self.read = 0

    def __call__(self) -> float:
        assert self.read < len(self._readings), (
            "the run read the clock more often than the scripted calls describe"
        )
        value = self._readings[self.read]
        self.read += 1
        return value


def _scripted_run(call_seconds: list[float], *, overhead: float = 0.0) -> _ScriptedClock:
    """A clock reading the run the harness measures: start, each call, end (T40).

    ``call_seconds`` is what the layer spends in its calls, one figure per call;
    ``overhead`` is everything around them (the store, the chunker). The harness
    reads every measurement off one clock (the chunk's total and the time of each
    call), so the readings are a sequence: the chunk starts, each call starts and
    ends, the chunk ends after the last call plus the overhead.

    Scripting the clock is what lets these tests state the T35 numbers exactly
    (parts of seconds, a fold-back of seconds) without sleeping: a test that slept
    would measure the machine under load, which is the very thing this task is
    about.
    """
    readings: list[float] = [0.0]
    now = 0.0
    for seconds in call_seconds:
        readings.extend((now, now + seconds))
        now += seconds
    readings.append(now + overhead)
    return _ScriptedClock(readings)


def _run_with_clock(
    chunk: GoldenChunk, call_seconds: list[float], *, overhead: float, models_config: ModelsConfig
) -> Any:
    """Evaluate ``chunk`` on the CI mock while its clock reads ``call_seconds``.

    ``monkeypatch`` is deliberately not used: the clock is replaced for the duration
    of one run and restored by the fixture-free helper, so a failing run cannot leave
    a scripted clock behind for the rest of the session.
    """
    original = harness._clock
    harness._clock = _scripted_run(call_seconds, overhead=overhead)
    try:
        provider = harness.ci_provider_factory(chunk)
        return harness.evaluate_chunk(chunk, provider=provider, models_config=models_config)
    finally:
        harness._clock = original


def test_a_split_day_is_graded_by_its_slowest_call_not_by_the_sum_of_them(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile, models_config: ModelsConfig
) -> None:
    """T40 DoD 1 and 2(a): the number of calls stops being a latency defect.

    The defect: the budget of 15 s was calibrated on **one** call (T21), while a
    correctly split day costs three or four (T35: 17.76 + 8.14 + 18.77 = 44.67 s of
    machine time), so every honestly split day was red by construction. Here the
    layer's own calls each stay inside the budget while their sum — the chunk total —
    is far above it: the chunk must pass, and the total must still be reported, so
    the price of splitting stays visible.
    """
    parts = len(_parts(long_day, day_profile))
    calls = [6.0] * parts + [7.0]  # every call inside 15 s, the day far outside
    outcome = _run_with_clock(long_day, calls, overhead=1.5, models_config=models_config)

    # The timed calls are exactly the requests the endpoint saw, no more and no less.
    assert outcome.requests == len(calls)
    assert [round(seconds, 2) for seconds in outcome.call_seconds] == calls
    assert outcome.slowest_call_s == pytest.approx(7.0)
    # The chunk's total is what the old rule compared with the budget: over it.
    assert outcome.latency_s > 15
    assert outcome.latency_within_budget
    assert outcome.content_violations == []

    report = harness.format_report([outcome], harness.golden_manifest())
    row = next(line for line in report.splitlines() if line.startswith(LONG_DAY))
    cells = row.split()
    assert cells[3] == f"{outcome.latency_s:.2f}s"  # the chunk's total stays printed
    assert cells[4] == str(len(calls))  # the call count (T35)
    assert cells[5] == f"{outcome.slowest_call_s:.2f}s"  # the figure the budget grades
    assert "PASS" in cells
    # Every call is named, not only the slowest, and the total is named next to them.
    expected = ", ".join(f"{seconds:.2f}s" for seconds in calls)
    assert f"{LONG_DAY} ({len(calls)} call(s), budget < 15s): {expected}" in report
    assert f"chunk total {outcome.latency_s:.2f}s" in report
    assert "latency day: slowest call 7.00s (budget < 15s), slowest chunk total" in report


def test_a_single_call_chunk_over_its_budget_stays_red(models_config: ModelsConfig) -> None:
    """T40 DoD 2(b): the unit change must not weaken an ordinary day.

    A short day makes **one** call, so its call *is* its latency: the figure the
    budget grades was the chunk before T40 and is the call now, and the verdict must
    not move. 16 s against the day budget of 15 s is red, on the clock check alone
    and not as a content failure (T18).

    «Almost» rather than «exactly», and this test is where the difference is visible:
    the chunk total is 16.4 s and the graded figure 16.0 s, because the work around
    the call (the store, the gate) is no longer inside the budget. That is the
    intended scope of the unit, not a loosening of the number — the pass/fail line
    for a one-call chunk moved down by the work outside the call; a chunk whose
    **call** misses the budget is still red (see the T40 sections of
    ``docs/device-choice.md``, ``ADR-016`` and ``eval/golden/README.md``).
    """
    chunk = harness.load_golden_set(chunk_ids=["day-01-quiet-monday"])[1][0]
    outcome = _run_with_clock(chunk, [16.0], overhead=0.4, models_config=models_config)

    assert len(outcome.call_seconds) == 1 and outcome.requests == 1
    assert outcome.slowest_call_s == pytest.approx(16.0)
    assert outcome.latency_s == pytest.approx(16.4)
    assert outcome.failed_checks == [harness.LATENCY_CHECK]
    report = harness.format_report([outcome], harness.golden_manifest())
    assert "FAIL:latency" in report
    assert "invariants: 1/1 passed" in report


def test_a_slow_call_of_a_split_day_is_still_red(
    long_day: GoldenChunk, day_profile: OpenAICompatibleProfile, models_config: ModelsConfig
) -> None:
    """T40 DoD 2(b) for a chunk that splits: one slow call still fails the chunk.

    The other side of the same axis. The budget is applied to **every** call, so a
    single call over it fails the chunk even though the *number* of calls is not what
    failed it — the two cases are separate claims, and only this test holds the
    second one. Without it a rule that quietly graded nothing but single-call chunks
    (``if len(outcome.call_seconds) > 1: return``) would pass the whole suite: that
    mutation was measured against the suite during the T40 acceptance and survived
    every test but this one.
    """
    parts = len(_parts(long_day, day_profile))
    calls = [6.0] * parts + [20.0]  # one call over the 15 s budget, the rest inside
    outcome = _run_with_clock(long_day, calls, overhead=1.5, models_config=models_config)

    assert outcome.requests == len(calls)
    assert outcome.slowest_call_s == pytest.approx(20.0)
    assert not outcome.latency_within_budget
    assert outcome.failed_checks == [harness.LATENCY_CHECK]
    assert outcome.content_violations == []
    # The report names the slowest call in the row, so the failing figure is visible
    # without reading the violation detail.
    report = harness.format_report([outcome], harness.golden_manifest())
    row = next(line for line in report.splitlines() if line.startswith(LONG_DAY))
    assert row.split()[5] == "20.00s"
    assert "FAIL:latency" in report


def test_a_chunk_whose_call_was_never_timed_is_not_read_as_fast() -> None:
    """The fallback is the chunk's own wall time, never zero (T40 DoD 2).

    A layer that refuses its input before its first request (``PipelineInputError``)
    makes no call at all, so there is nothing to time per call. Grading such a chunk
    as ``0.00s`` would turn «nothing was measured» into «fast»; the honest figure is
    the wall time the harness did measure.
    """
    outcome = harness.ChunkOutcome(
        chunk_id="day-without-a-call",
        category="day",
        kind="daily_summary",
        latency_s=0.5,
        latency_budget_s=0.0,
    )
    assert outcome.call_seconds == []
    assert outcome.slowest_call_s == pytest.approx(0.5)
    harness._apply_latency_budget(outcome)
    assert [violation.check for violation in outcome.violations] == [harness.LATENCY_CHECK]
    assert "no call was timed" in outcome.violations[0].detail
