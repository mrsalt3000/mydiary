"""Pure chunk planning and stitching for recordings longer than one model clip (T44).

A long recording is cut into chunks no longer than ``max_chunk_s`` so a single
inference stays inside the VRAM the 6 GiB laptop GPU has
(``docs/stt-backends-analysis.md`` §6.5, §7.2). The cut prefers a **silence**
(a word cut in half costs more than a slightly uneven chunk) and falls back to a
fixed window when the neighbourhood of the target boundary is loud; either way
the chunk is labelled honestly.

Why this module is pure
=======================

It takes **per-frame RMS values** (a plain sequence of numbers), a frame
duration and the total duration — not an audio array — so it has no numpy and no
third-party dependency at all. That has two consequences worth naming:

* it is covered by ordinary pytest tests on synthetic RMS, instead of being
  exercised only by a live run against a real recording;
* the sidecar worker (``scripts/gigaam-stt-worker.py``) can and does call
  :func:`plan_chunks` for the audio it decoded, so the *product* planner is what
  runs in production rather than a copy of it. The worker loads this **file**
  through ``importlib`` because importing it as ``diary.…chunking`` would run
  the package ``__init__``, which pulls pydantic and SQLAlchemy — neither of
  which exists in the sidecar's virtualenv.

The coverage check is a product requirement, not a probe instrument
==================================================================

:func:`plan_chunks` refuses to return a plan that does not tile the recording
exactly once, tail included (:func:`assert_coverage`). This is the acceptance
finding of T42 written into product code: a planner built from successive bounds
cannot produce an internal hole, so an instrument that only walks the boundaries
measures nothing — the failure it *can* have is an unclosed tail, and that is
exactly what is checked (against the recorded duration, not against the plan's
own last bound). A bad plan raises :class:`ChunkPlanError` instead of returning
"0.0 and off we go".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

#: Sample rate both backends want (GigaAM is hard-wired to it); the frame
#: duration below is expressed against it and the chunk bounds are published in
#: seconds, so a caller that decoded to anything else cannot silently agree.
SAMPLE_RATE = 16000

#: Analysis frame of the energy detector, in seconds (20 ms).
SILENCE_FRAME_S = 0.02
#: A silent run shorter than this is not a candidate cut.
MIN_SILENCE_S = 0.3
#: A frame is silence when its RMS is below ``max(SILENCE_ABSOLUTE_FLOOR,
#: SILENCE_REL_THRESHOLD * p95(RMS))`` — relative to the recording's own loud
#: level, so no per-recording threshold has to be configured.
SILENCE_REL_THRESHOLD = 0.05
#: Absolute floor of that threshold, so an all-but-silent file still has silence.
SILENCE_ABSOLUTE_FLOOR = 1e-4
#: Tolerance of the coverage check, in seconds. One microsecond is far below a
#: sample (1/16000 s = 62.5 µs) and absorbs only the float rounding of frame
#: counts and durations.
COVERAGE_TOLERANCE_S = 1e-6

#: Why a chunk's *owned* range ends where it does (its ``cut``).
CutKind = Literal["start", "silence", "window", "end", "whole"]


class ChunkPlanError(ValueError):
    """The chunk plan does not tile the recording exactly once.

    Raised for an invalid request (non-positive duration, an overlap that is not
    smaller than the chunk) and for a plan that would lose, duplicate or retain
    an unowned part of the recording. It is a hard failure by design: a plan
    that drops audio must never be reported as a measurement or transcribed as
    if it covered the file.
    """


@dataclass(frozen=True)
class Chunk:
    """One decode unit.

    ``owned_*`` is the range this chunk is responsible for; the owned ranges of
    a plan tile ``[0, total_s]`` exactly once. ``decode_*`` is what is actually
    handed to the model: the owned range widened by the overlap on both sides
    (clipped at the file ends), so a word straddling a seam is fully inside one
    chunk. ``cut`` names what ended the owned range.
    """

    index: int
    decode_start_s: float
    decode_end_s: float
    owned_start_s: float
    owned_end_s: float
    cut: CutKind

    @property
    def owned_seconds(self) -> float:
        """Length of the owned range."""
        return self.owned_end_s - self.owned_start_s

    @property
    def decode_seconds(self) -> float:
        """Length of the range handed to the model."""
        return self.decode_end_s - self.decode_start_s


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation percentile of ``values`` (numpy's default method).

    Hand-written because the module has no numpy: the sidecar computes the RMS
    values, this function only has to agree with the threshold the T42 probe
    measured (``np.percentile(rms, 95)``).
    """
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def silence_midpoints(frame_rms: Sequence[float], *, frame_s: float) -> list[float]:
    """Midpoints (seconds, ascending) of the candidate silence runs.

    A run is a maximal stretch of frames whose RMS is below the relative
    threshold; runs shorter than :data:`MIN_SILENCE_S` are dropped. The midpoint
    is the cut candidate: a chunk boundary there sits in the middle of silence.
    """
    if frame_s <= 0:
        raise ChunkPlanError(f"frame_s must be positive, got {frame_s}")
    values = [float(value) for value in frame_rms]
    if len(values) < 2:
        return []
    threshold = max(SILENCE_ABSOLUTE_FLOOR, SILENCE_REL_THRESHOLD * _percentile(values, 0.95))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value < threshold:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return [
        ((begin + end) / 2) * frame_s
        for begin, end in runs
        if (end - begin) * frame_s >= MIN_SILENCE_S
    ]


def assert_coverage(chunks: Sequence[Chunk], total_s: float) -> tuple[float, float]:
    """Check that ``chunks`` tile ``[0, total_s]`` exactly once; return ``(gap, double)``.

    Three properties are required, and all three are checked here:

    * the owned ranges of consecutive chunks join up — no internal hole, no
      double ownership;
    * the last owned range ends at ``total_s`` — **this is the one failure a
      planner built from successive boundaries can actually have**, and the
      reason this check takes the file duration at all: a missing final bound
      silently drops the tail, and walking the plan's own boundaries would never
      see it;
    * the owned seconds add up to ``total_s``.

    ``gap``/``double`` are the accumulated omissions and overlaps (both zero on
    success, within :data:`COVERAGE_TOLERANCE_S`). Anything else means the split
    lost or duplicated audio and :class:`ChunkPlanError` is raised with the
    numbers.
    """
    if not chunks:
        raise ChunkPlanError("chunk plan is empty: nothing would be transcribed")
    if total_s <= 0:
        raise ChunkPlanError(
            f"chunk plan checked against a non-positive recording: total_s={total_s}"
        )

    gap = 0.0
    double = 0.0
    owned = 0.0
    cursor = 0.0
    for chunk in chunks:
        if chunk.owned_start_s > cursor:
            gap += chunk.owned_start_s - cursor
        elif chunk.owned_start_s < cursor:
            double += cursor - chunk.owned_start_s
        owned += chunk.owned_seconds
        cursor = max(cursor, chunk.owned_end_s)

    failures: list[str] = []
    if gap > COVERAGE_TOLERANCE_S:
        failures.append(f"uncovered gap {gap:.6f}s")
    if double > COVERAGE_TOLERANCE_S:
        failures.append(f"double ownership {double:.6f}s")
    tail = total_s - cursor
    if abs(tail) > COVERAGE_TOLERANCE_S:
        failures.append(
            f"uncovered tail {tail:.6f}s (plan ends at {cursor:.6f}s of {total_s:.6f}s)"
        )
    if abs(owned - total_s) > COVERAGE_TOLERANCE_S:
        failures.append(f"owned {owned:.6f}s != recording {total_s:.6f}s")
    if failures:
        raise ChunkPlanError(
            f"chunk plan does not tile the recording exactly once "
            f"(tolerance {COVERAGE_TOLERANCE_S}s): " + "; ".join(failures)
        )
    return gap, double


def plan_chunks(
    frame_rms: Sequence[float],
    *,
    frame_s: float,
    total_s: float,
    max_chunk_s: float,
    overlap_s: float,
) -> list[Chunk]:
    """Cut a recording into chunks of at most ``max_chunk_s`` owned seconds.

    The cut is placed at the midpoint of the *last* candidate silence inside the
    window ``[position + max_chunk_s / 2, position + max_chunk_s]`` (``silence``);
    with no such silence the window edge is used and the chunk is labelled
    ``window``. Owned ranges tile ``[0, total_s]`` exactly once: the plan is
    verified by :func:`assert_coverage` before it is returned, so a caller can
    never receive a plan that loses or duplicates audio.

    Raises :class:`ChunkPlanError` for an unusable request (non-positive
    duration or frame length, an overlap that is not smaller than the chunk) and
    for a plan that fails the coverage check.
    """
    if total_s <= 0:
        raise ChunkPlanError(f"total_s must be positive, got {total_s}")
    if max_chunk_s <= 0:
        raise ChunkPlanError(f"max_chunk_s must be positive, got {max_chunk_s}")
    if overlap_s < 0:
        raise ChunkPlanError(f"chunk_overlap_s must not be negative, got {overlap_s}")
    if overlap_s >= max_chunk_s:
        raise ChunkPlanError(
            f"chunk_overlap_s ({overlap_s}) must be smaller than max_chunk_s ({max_chunk_s})"
        )

    if total_s <= max_chunk_s + COVERAGE_TOLERANCE_S:
        whole = [Chunk(0, 0.0, total_s, 0.0, total_s, "whole")]
        assert_coverage(whole, total_s)
        return whole

    midpoints = silence_midpoints(frame_rms, frame_s=frame_s)
    min_chunk_s = max_chunk_s / 2

    bounds: list[tuple[float, CutKind]] = [(0.0, "start")]
    position = 0.0
    while total_s - position > max_chunk_s + COVERAGE_TOLERANCE_S:
        ideal = position + max_chunk_s
        lower = position + min_chunk_s
        candidates = [midpoint for midpoint in midpoints if lower <= midpoint <= ideal]
        if candidates:
            boundary = candidates[-1]
            bounds.append((boundary, "silence"))
            position = boundary
        else:
            bounds.append((ideal, "window"))
            position = ideal
    bounds.append((total_s, "end"))

    chunks: list[Chunk] = []
    for index in range(len(bounds) - 1):
        owned_start, _ = bounds[index]
        owned_end, cut = bounds[index + 1]
        chunks.append(
            Chunk(
                index=index,
                decode_start_s=max(0.0, owned_start - overlap_s),
                decode_end_s=min(total_s, owned_end + overlap_s),
                owned_start_s=owned_start,
                owned_end_s=owned_end,
                cut=cut,
            )
        )
    assert_coverage(chunks, total_s)
    return chunks


def stitch(parts: Sequence[str], chunks: Sequence[Chunk], overlap_s: float) -> tuple[str, int]:
    """Join chunk transcripts, dropping the words the overlap duplicated.

    The overlap exists so that a word at the seam is fully inside one chunk; the
    price is that the next chunk usually repeats the tail of the previous one.
    For each following chunk the longest word-level overlap with the accumulated
    text is removed, limited to what the overlap could plausibly contain (its
    duration times that chunk's own words-per-second, plus slack). Returns
    ``(text, removed_words)``; ``removed_words`` is what the caller publishes as
    evidence that the stitch did its job rather than the caller believing it did.
    """
    if len(parts) != len(chunks):
        raise ChunkPlanError(
            f"got {len(parts)} transcripts for {len(chunks)} chunks: they must pair up"
        )
    removed = 0
    accumulated: list[str] = []
    for index, part in enumerate(parts):
        words = part.split()
        if index == 0:
            accumulated = words
            continue
        rate = len(words) / max(chunks[index].decode_seconds, 1e-6)
        limit = max(1, int(overlap_s * rate * 1.5) + 3)
        limit = min(limit, len(words), len(accumulated))
        overlap = 0
        for size in range(limit, 0, -1):
            if accumulated[-size:] == words[:size]:
                overlap = size
                break
        removed += overlap
        accumulated.extend(words[overlap:])
    return " ".join(accumulated), removed


__all__ = [
    "COVERAGE_TOLERANCE_S",
    "MIN_SILENCE_S",
    "SAMPLE_RATE",
    "SILENCE_ABSOLUTE_FLOOR",
    "SILENCE_FRAME_S",
    "SILENCE_REL_THRESHOLD",
    "Chunk",
    "ChunkPlanError",
    "CutKind",
    "assert_coverage",
    "plan_chunks",
    "silence_midpoints",
    "stitch",
]
