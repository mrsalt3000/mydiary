"""Unit tests for the pure chunk planner and stitcher (T44).

Everything here is synthetic **per-frame RMS** — no audio file, no numpy, no
model, no GPU. The planner is the product code the sidecar worker calls, so
these tests are what makes "the split loses no audio" a checked property instead
of a claim: the mandatory case is the unclosed tail, the one failure a planner
built from successive bounds can actually have (the T42 acceptance finding, `r1`).
"""

from __future__ import annotations

import pytest

from diary.domain.transcription import chunking

FRAME_S = chunking.SILENCE_FRAME_S


def _constant_rms(seconds: float, value: float) -> list[float]:
    """A recording whose every frame has the same level (loud, or all silence)."""
    return [value] * int(round(seconds / FRAME_S))


def _tone_pattern(seconds: float, *, loud_s: float = 3.0, silence_s: float = 1.0) -> list[float]:
    """Alternating loud/silent blocks — the synthetic file the T42 selftest used."""
    frames: list[float] = []
    while len(frames) < seconds / FRAME_S:
        frames += [0.3] * int(round(loud_s / FRAME_S))
        frames += [0.0] * int(round(silence_s / FRAME_S))
    return frames[: int(round(seconds / FRAME_S))]


def _plan(rms: list[float], seconds: float, **overrides: float) -> list[chunking.Chunk]:
    options: dict[str, float] = {"max_chunk_s": 20.0, "overlap_s": 0.5}
    options.update(overrides)
    return chunking.plan_chunks(
        rms,
        frame_s=FRAME_S,
        total_s=seconds,
        max_chunk_s=options["max_chunk_s"],
        overlap_s=options["overlap_s"],
    )


# ---------------------------------------------------------------------------
# Planning: short recordings, silence cuts, window cuts
# ---------------------------------------------------------------------------


def test_recording_shorter_than_the_clip_is_one_whole_chunk() -> None:
    chunks = _plan(_constant_rms(9.0, 0.3), 9.0)

    assert len(chunks) == 1
    assert chunks[0].cut == "whole"
    assert (chunks[0].owned_start_s, chunks[0].owned_end_s) == (0.0, 9.0)
    # The overlap cannot widen a chunk past the recording's own ends.
    assert (chunks[0].decode_start_s, chunks[0].decode_end_s) == (0.0, 9.0)


def test_recording_exactly_one_clip_long_is_not_cut() -> None:
    chunks = _plan(_constant_rms(20.0, 0.3), 20.0)

    assert len(chunks) == 1
    assert chunks[0].cut == "whole"


def test_cut_is_placed_in_silence_when_silence_exists() -> None:
    """3 s of tone / 1 s of silence, 60 s long, 20 s clips: every cut is silent."""
    rms = _tone_pattern(60.0)
    chunks = _plan(rms, 60.0, overlap_s=0.0)

    assert len(chunks) == 4
    assert all(chunk.cut == "silence" for chunk in chunks[:-1])
    assert chunks[-1].cut == "end"
    # The last silence midpoint at or before 20 s is 19.5 s — a deterministic
    # consequence of the pattern, not a tolerance.
    assert chunks[0].owned_end_s == pytest.approx(19.5)
    # Every internal boundary sits on a silent frame of the synthetic recording.
    for previous, following in zip(chunks, chunks[1:]):
        assert previous.owned_end_s == pytest.approx(following.owned_start_s)
        assert rms[int(previous.owned_end_s / FRAME_S)] == 0.0


def test_cut_falls_back_to_a_window_when_there_is_no_silence() -> None:
    """An all-loud recording: the boundary is the window edge, labelled honestly."""
    chunks = _plan(_constant_rms(60.0, 0.3), 60.0)

    assert [chunk.cut for chunk in chunks] == ["window", "window", "end"]
    assert all(chunk.owned_seconds <= 20.0 + chunking.COVERAGE_TOLERANCE_S for chunk in chunks)


def test_no_chunk_exceeds_max_chunk_s() -> None:
    chunks = _plan(_constant_rms(95.0, 0.3), 95.0, max_chunk_s=15.0)

    assert all(chunk.owned_seconds <= 15.0 + chunking.COVERAGE_TOLERANCE_S for chunk in chunks)


def test_min_silence_is_required_for_a_cut() -> None:
    """A 20 ms gap is not silence: the planner cuts by window instead."""
    frames = int(round(60.0 / FRAME_S))
    rms = [0.3] * frames
    rms[100] = 0.0  # a single frame = one analysis frame of 20 ms

    chunks = _plan(rms, 60.0)

    assert all(chunk.cut == "window" for chunk in chunks[:-1])


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


def test_overlap_zero_makes_decode_equal_the_owned_range() -> None:
    chunks = _plan(_tone_pattern(60.0), 60.0, overlap_s=0.0)

    for chunk in chunks:
        assert chunk.decode_start_s == pytest.approx(chunk.owned_start_s)
        assert chunk.decode_end_s == pytest.approx(chunk.owned_end_s)


def test_overlap_widens_the_decode_range_and_clips_at_the_ends() -> None:
    chunks = _plan(_tone_pattern(60.0), 60.0, overlap_s=0.5)

    assert chunks[0].decode_start_s == 0.0
    assert chunks[-1].decode_end_s == pytest.approx(60.0)
    for index, chunk in enumerate(chunks):
        if index > 0:
            assert chunk.decode_start_s == pytest.approx(chunk.owned_start_s - 0.5)
        if index < len(chunks) - 1:
            assert chunk.decode_end_s == pytest.approx(chunk.owned_end_s + 0.5)


# ---------------------------------------------------------------------------
# Coverage: the mandatory rejection cases
# ---------------------------------------------------------------------------


def test_a_plan_without_its_last_bound_is_rejected() -> None:
    """The T42 `r1` finding, as product code: a lost tail is not "0.0 and off we go".

    The mutation drops the planner's final bound. Walking the plan's own
    boundaries would call it valid (there is no internal hole), so the check has
    to be made against the recording's duration — which is why the call takes it.
    """
    lost_tail = [
        chunking.Chunk(0, 0.0, 60.0, 0.0, 60.0, "silence"),
        chunking.Chunk(1, 60.0, 96.0, 60.0, 96.0, "end"),
    ]

    with pytest.raises(chunking.ChunkPlanError) as excinfo:
        chunking.assert_coverage(lost_tail, 100.0)

    message = str(excinfo.value)
    assert "uncovered tail 4.000000s" in message
    assert "100.000000s" in message


def test_the_planner_never_returns_a_plan_with_a_lost_tail() -> None:
    """The same instrument accepts the real planner's plan (or the planner fails)."""
    rms = _tone_pattern(60.0)
    chunks = _plan(rms, 60.0)

    assert chunking.assert_coverage(chunks, 60.0) == (0.0, 0.0)


def test_the_planner_validates_its_plan_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarantee is in the code path, not only in these tests.

    If the planner ever stopped calling the instrument, the plan would travel
    downstream unchecked — so the call itself is pinned: the planner must refuse
    to return when the instrument rejects, and must pass it the recording length.
    """
    seen: list[tuple[int, float]] = []

    def _reject(chunks: list[chunking.Chunk], total_s: float) -> tuple[float, float]:
        seen.append((len(chunks), total_s))
        raise chunking.ChunkPlanError("synthetic: the instrument rejected the plan")

    monkeypatch.setattr(chunking, "assert_coverage", _reject)

    with pytest.raises(chunking.ChunkPlanError, match="synthetic"):
        _plan(_tone_pattern(60.0), 60.0)

    assert seen and seen[0][1] == 60.0


def test_a_short_recording_is_validated_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole-file branch is a plan like any other: it is checked as well."""
    seen: list[tuple[int, float]] = []

    def _reject(chunks: list[chunking.Chunk], total_s: float) -> tuple[float, float]:
        seen.append((len(chunks), total_s))
        raise chunking.ChunkPlanError("synthetic: the instrument rejected the plan")

    monkeypatch.setattr(chunking, "assert_coverage", _reject)

    with pytest.raises(chunking.ChunkPlanError, match="synthetic"):
        _plan(_constant_rms(9.0, 0.3), 9.0)

    assert seen == [(1, 9.0)]


def test_an_empty_plan_is_rejected() -> None:
    with pytest.raises(chunking.ChunkPlanError, match="empty"):
        chunking.assert_coverage([], 10.0)


def test_double_ownership_is_rejected() -> None:
    overlapping = [
        chunking.Chunk(0, 0.0, 55.0, 0.0, 55.0, "silence"),
        chunking.Chunk(1, 50.0, 100.0, 50.0, 100.0, "end"),
    ]

    with pytest.raises(chunking.ChunkPlanError, match="double ownership 5.000000s") as excinfo:
        chunking.assert_coverage(overlapping, 100.0)

    assert "owned" in str(excinfo.value)


def test_a_leading_gap_is_rejected() -> None:
    with pytest.raises(chunking.ChunkPlanError, match="uncovered gap"):
        chunking.assert_coverage([chunking.Chunk(0, 5.0, 100.0, 5.0, 100.0, "end")], 100.0)


def test_an_internal_gap_is_rejected() -> None:
    holed = [
        chunking.Chunk(0, 0.0, 50.0, 0.0, 50.0, "silence"),
        chunking.Chunk(1, 55.0, 100.0, 55.0, 100.0, "end"),
    ]

    with pytest.raises(chunking.ChunkPlanError, match="uncovered gap 5.000000s"):
        chunking.assert_coverage(holed, 100.0)


def test_a_plan_that_tiles_the_recording_is_accepted() -> None:
    exact = [
        chunking.Chunk(0, 0.0, 60.0, 0.0, 60.0, "silence"),
        chunking.Chunk(1, 60.0, 100.0, 60.0, 100.0, "end"),
    ]

    assert chunking.assert_coverage(exact, 100.0) == (0.0, 0.0)


def test_coverage_against_a_non_positive_recording_is_rejected() -> None:
    with pytest.raises(chunking.ChunkPlanError, match="non-positive"):
        chunking.assert_coverage([chunking.Chunk(0, 0.0, 1.0, 0.0, 1.0, "whole")], 0.0)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"total_s": 0.0, "max_chunk_s": 20.0, "overlap_s": 0.5}, "total_s"),
        ({"total_s": 60.0, "max_chunk_s": 0.0, "overlap_s": 0.0}, "max_chunk_s"),
        ({"total_s": 60.0, "max_chunk_s": 20.0, "overlap_s": -1.0}, "not be negative"),
        ({"total_s": 60.0, "max_chunk_s": 20.0, "overlap_s": 20.0}, "smaller than max_chunk_s"),
    ],
)
def test_unusable_requests_are_refused(options: dict[str, float], match: str) -> None:
    with pytest.raises(chunking.ChunkPlanError, match=match):
        chunking.plan_chunks(
            _constant_rms(60.0, 0.3),
            frame_s=FRAME_S,
            total_s=options["total_s"],
            max_chunk_s=options["max_chunk_s"],
            overlap_s=options["overlap_s"],
        )


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------


def test_stitch_removes_the_words_the_overlap_duplicated() -> None:
    chunks = [
        chunking.Chunk(0, 0.0, 6.0, 0.0, 5.0, "silence"),
        chunking.Chunk(1, 5.0, 11.0, 5.0, 10.0, "end"),
    ]

    text, removed = chunking.stitch(
        ["раз два три четыре пять", "четыре пять шесть семь восемь"], chunks, 0.5
    )

    assert text == "раз два три четыре пять шесть семь восемь"
    assert removed == 2


def test_stitch_without_a_duplicated_seam_removes_nothing() -> None:
    chunks = [
        chunking.Chunk(0, 0.0, 6.0, 0.0, 5.0, "silence"),
        chunking.Chunk(1, 5.0, 11.0, 5.0, 10.0, "end"),
    ]

    text, removed = chunking.stitch(["альфа бета", "гамма дельта"], chunks, 0.5)

    assert text == "альфа бета гамма дельта"
    assert removed == 0


def test_stitch_does_not_remove_more_words_than_the_overlap_could_hold() -> None:
    """The removal is capped, so a genuinely repeated phrase is not eaten.

    A 0.5 s overlap may plausibly hide a few words, not an unbounded run of
    them: for a six-second chunk of seven words the cap is
    ``int(0.5 * (7 / 6) * 1.5) + 3 == 3``. The rest of the repetition stays —
    the stitch removes the *seam*, it does not deduplicate the speech.
    """
    chunks = [
        chunking.Chunk(0, 0.0, 6.0, 0.0, 5.0, "silence"),
        chunking.Chunk(1, 5.0, 11.0, 5.0, 10.0, "end"),
    ]
    repeated = "да да да да да да да"

    text, removed = chunking.stitch([repeated, repeated], chunks, 0.5)

    assert removed == 3
    assert text == " ".join(["да"] * 11)


def test_stitch_requires_one_transcript_per_chunk() -> None:
    chunks = [chunking.Chunk(0, 0.0, 6.0, 0.0, 6.0, "whole")]

    with pytest.raises(chunking.ChunkPlanError, match="pair up"):
        chunking.stitch(["один", "два"], chunks, 0.5)


def test_stitch_of_a_single_chunk_keeps_its_text() -> None:
    chunks = [chunking.Chunk(0, 0.0, 6.0, 0.0, 6.0, "whole")]

    text, removed = chunking.stitch(["  привет,   мир  "], chunks, 0.5)

    assert text == "привет, мир"
    assert removed == 0


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------


def test_silence_midpoints_are_empty_for_an_all_loud_recording() -> None:
    assert chunking.silence_midpoints(_constant_rms(10.0, 0.3), frame_s=FRAME_S) == []


def test_silence_midpoints_are_the_centre_of_each_silent_run() -> None:
    frames = int(round(10.0 / FRAME_S))
    rms = [0.3] * frames
    for frame in range(100, 150):  # 2.0 s .. 3.0 s
        rms[frame] = 0.0

    midpoints = chunking.silence_midpoints(rms, frame_s=FRAME_S)

    assert midpoints == [pytest.approx(2.5)]


def test_a_short_silent_run_is_not_a_candidate() -> None:
    frames = int(round(10.0 / FRAME_S))
    rms = [0.3] * frames
    for frame in range(100, 110):  # 200 ms, below MIN_SILENCE_S
        rms[frame] = 0.0

    assert chunking.silence_midpoints(rms, frame_s=FRAME_S) == []


def test_silence_midpoints_needs_a_positive_frame_length() -> None:
    with pytest.raises(chunking.ChunkPlanError, match="frame_s"):
        chunking.silence_midpoints([0.0, 0.0], frame_s=0.0)
