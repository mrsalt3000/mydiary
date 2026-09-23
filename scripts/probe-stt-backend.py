#!/usr/bin/env python
"""Probe one STT backend and print its measurable cost (T42 spike).

This script is a *measurement*, not a product code path (T42,
``docs/tasklist-phase2.md`` §17). It exists to answer, with numbers, whether the
GigaAM-v2 CTC+LM contour of ``tts_research.md`` fits this machine (RTX 3060
Laptop, 6 GB) and whether it must run in-process or as an isolated sidecar. The
measurements and the decision live in ``docs/stt-backends-analysis.md``.

Backends (``--backend``):

* ``gigaam-lm`` — ``waveletdeboshir/gigaam-ctc-with-lm`` through
  ``trust_remote_code``: encoder logits, then ``pyctcdecode`` beam search over a
  3-gram KenLM language model (the ``language_model/`` files of the checkpoint);
* ``gigaam-greedy`` — the same checkpoint with the language model switched off:
  argmax over the logits, consecutive duplicates collapsed, blank (id 33)
  dropped. This is the "how much does the LM buy" control;
* ``whisper`` — the current production backend (faster-whisper, ``large-v3``),
  with the same profile settings as ``configs/models.yaml``. That one runs in
  the *application* virtualenv; the two GigaAM backends run in the spike
  virtualenv. The script imports each runtime lazily, so the same file works in
  both interpreters.

Audio is decoded by this script, not by the backend: PyAV (``av``) opens the
container (``webm``/``ogg``/``m4a``/``wav``) and resamples to 16 kHz mono, so
**no system ffmpeg is needed** (``docs/wsl2-deployment-brief.md``). Every
backend then receives the same float32 numpy array, which keeps the comparison
apples-to-apples and lets ``--max-seconds`` cut a prefix for a short-file shot.

Long files are cut at silence (energy-based, no extra dependency) into chunks no
longer than ``--max-chunk-s``; if no silence is found near the target cut, the
cut is a fixed window and the chunk is labelled ``window`` honestly. Chunks may
be extended by ``--chunk-overlap-s`` on both sides so a word straddling the seam
is fully inside one of them; the stitch removes the duplicated words the overlap
produced. Coverage is then *checked*, not assumed: :func:`chunk_coverage` refuses
the run unless the owned ranges tile the file exactly once **including the tail**,
and it takes the file length for that reason — a plan built from successive
boundaries cannot have an internal hole, so an instrument that only walks the
boundaries would be blind to the one failure the planner can actually have.

What is measured, per run: decoded duration, decode time, model load time,
inference time, RTF (inference / decoded seconds), chunk count and cut kinds,
peak VRAM sampled from ``nvidia-smi`` *and* (GigaAM, Torch) the allocator
counters, VRAM after ``unload()`` by both means, process RAM (``VmRSS`` and the
kernel's own ``VmHWM`` high-water mark) before the load, after the load and
after the unload, weight bytes and where they live, and a SHA-256 of the
transcript so two runs can be compared for identity without writing the text
anywhere.

Modes: ``run`` (one backend, one file), ``compare`` (agreement between
transcripts, with optional attribution of the differences to chunk seams against
the count a uniform distribution would produce), and ``selftest`` — which checks
the probe's own instruments on synthetic data, with no model, no GPU and no
audio files: the coverage cases above (lost tail, empty plan, internal gap,
leading gap, double ownership, plus the real planner) and the decoding of
``wav``/``ogg``/``m4a``/``webm`` written on the fly by PyAV. A measurement whose
instrument is untested is a belief, which is why ``selftest`` exists.

Usage::

    # spike venv, GigaAM with the LM, long file, silence-aware chunks
    HF_HOME=~/.local/share/mydiary/gigaam-spike/hf \\
      ~/.local/share/mydiary/gigaam-spike/venv/bin/python scripts/probe-stt-backend.py \\
      run --backend gigaam-lm --audio data/audio/2026/09/<uuid>.webm \\
      --device cuda --max-chunk-s 20 --chunk-overlap-s 0.5 --json /tmp/t42-run.json

    # application venv, the current production backend
    .venv/bin/python scripts/probe-stt-backend.py run --backend whisper \\
      --audio data/audio/2026/09/<uuid>.webm --device cuda --json /tmp/t42-whisper.json

    # agreement between two transcripts (no text on stdout, only the numbers)
    python scripts/probe-stt-backend.py compare --text /tmp/a.txt --text /tmp/b.txt \\
      --seams-from-json /tmp/b-run.json

    # the instruments themselves: coverage cases and container decoding, no model
    python scripts/probe-stt-backend.py selftest

``--show-text N`` prints at most N characters of the transcript, ``--emit-text
PATH`` writes it to a file; both are off by default. Refusing to write inside
the repository is deliberate: unlike ``/tmp``, the work tree is a place where
diary text can leak into a commit.

Metric in ``compare``: normalized character agreement — ``1 - Levenshtein(a, b)
/ max(len(a), len(b))`` over normalized text (lowercase, ``ё``→``е``,
punctuation dropped, whitespace collapsed). Its boundary is named in the
document: without a human transcript this is *agreement between backends*, not
accuracy. ``--seams``/``--seams-from-json`` additionally attribute the differing
characters to the nearest chunk seam in character space (a proportion, not a word
timestamp) and print, for each window width, the count a uniform distribution
would put there — with 18 seams a ±1 % window covers a third of the file, so the
observed count alone says nothing.

Privacy (``AGENTS.md`` §7–8): audio and transcript text are never logged; the
only text that leaves the process is what ``--show-text``/``--emit-text``
explicitly ask for, and the JSON carries a SHA-256 instead of the text.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16000  #: GigaAM and faster-whisper both want 16 kHz mono.
REPO_ROOT = Path(__file__).resolve().parent.parent
VRAM_INTERVAL_S = 0.05
SILENCE_FRAME_S = 0.02
MIN_SILENCE_S = 0.3
SILENCE_REL_THRESHOLD = 0.05
#: Tolerance for the coverage check, in seconds. One microsecond is far below a
#: sample (1/16000 s = 62.5 µs) and only absorbs the float rounding of sample
#: counts divided by the sample rate.
COVERAGE_TOLERANCE_S = 1e-6
#: Half-widths of the seam windows reported by :func:`seam_attribution`.
SEAM_WINDOWS = (0.005, 0.01, 0.02, 0.05)


# --------------------------------------------------------------------------- #
# VRAM sampling
# --------------------------------------------------------------------------- #


class VramSampler:
    """Sample ``nvidia-smi`` GPU memory in a background thread.

    The sampler is the backend-agnostic measurement: it works for CTranslate2
    (faster-whisper), which exposes no allocator counters, and it is the means
    by which "peak" and "after unload" are compared against the Torch allocator
    numbers of the GigaAM path.
    """

    def __init__(self, interval_s: float = VRAM_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []
        self.error: str | None = None

    @staticmethod
    def read_used_mib() -> int | None:
        import subprocess

        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except Exception:
            return None
        try:
            return int(out.stdout.strip().splitlines()[0])
        except (IndexError, ValueError):
            return None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            value = self.read_used_mib()
            if value is None:
                self.error = "nvidia-smi unavailable"
                return
            self.samples.append(value)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def peak(self) -> int | None:
        return max(self.samples) if self.samples else None

    @property
    def baseline(self) -> int | None:
        return self.samples[0] if self.samples else None


# --------------------------------------------------------------------------- #
# Audio decoding (PyAV — container support without system ffmpeg)
# --------------------------------------------------------------------------- #


def decode_audio(path: Path, max_seconds: float | None) -> tuple["object", float, float]:
    """Decode ``path`` to float32 mono 16 kHz.

    Returns ``(samples, decoded_seconds, source_seconds)``. ``source_seconds`` is
    the full container duration; ``decoded_seconds`` is what is actually fed to
    the backend after the optional ``--max-seconds`` prefix cut.

    An unreadable or empty container aborts the run with a message instead of
    handing a backend an empty array: a measurement of silence is worse than no
    measurement.
    """
    import av
    import numpy as np

    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise SystemExit("no audio stream in the container")
            source_seconds = 0.0
            if stream.duration:
                source_seconds = float(stream.duration * stream.time_base)
            elif container.duration:
                source_seconds = container.duration / av.time_base
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            parts = []
            for frame in container.decode(stream):
                parts.extend(resampler.resample(frame))
            parts.extend(resampler.resample(None))
    except SystemExit:
        raise
    except Exception as exc:
        # `av` errors carry the path and a decoder message — no audio bytes.
        raise SystemExit(f"cannot decode {path.name}: {type(exc).__name__}: {exc}") from exc

    if not parts:
        raise SystemExit("the container decoded to zero audio frames")
    joined = np.concatenate([np.asarray(p.to_ndarray()).reshape(-1) for p in parts])
    samples = joined.astype(np.float32) / 32768.0
    if max_seconds is not None:
        samples = samples[: int(max_seconds * SAMPLE_RATE)]
    return samples, len(samples) / SAMPLE_RATE, source_seconds


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    """One decode unit: an "owned" range tiling the file, widened by the overlap."""

    index: int
    decode_start_s: float
    decode_end_s: float
    owned_start_s: float
    owned_end_s: float
    cut: str  # 'start' | 'silence' | 'window' | 'end' | 'whole'

    @property
    def owned_seconds(self) -> float:
        return self.owned_end_s - self.owned_start_s

    @property
    def decode_seconds(self) -> float:
        return self.decode_end_s - self.decode_start_s


def _silence_runs(samples: "object") -> list[tuple[int, int]]:
    """Energy-based silence runs in samples (no VAD dependency).

    A frame counts as silence when its RMS is below
    ``max(1e-4, 0.05 * p95(RMS))`` — relative to the file's own loud level, so
    the detector needs no per-recording threshold. Runs shorter than
    ``MIN_SILENCE_S`` are not candidate cuts.
    """
    import numpy as np

    frame = int(SILENCE_FRAME_S * SAMPLE_RATE)
    n_frames = len(samples) // frame
    if n_frames < 2:
        return []
    frames = np.asarray(samples[: n_frames * frame], dtype=np.float32).reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    threshold = max(1e-4, SILENCE_REL_THRESHOLD * float(np.percentile(rms, 95)))
    silent = rms < threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_silent in enumerate(silent):
        if is_silent and start is None:
            start = index
        elif not is_silent and start is not None:
            runs.append((start * frame, index * frame))
            start = None
    if start is not None:
        runs.append((start * frame, n_frames * frame))
    return [(s, e) for s, e in runs if (e - s) / SAMPLE_RATE >= MIN_SILENCE_S]


def plan_chunks(
    samples: "object", max_chunk_s: float, overlap_s: float, single_pass: bool
) -> list[Chunk]:
    """Cut the file into chunks of at most ``max_chunk_s`` owned seconds.

    The cut is placed at the midpoint of the longest silence found in the second
    half of the window (``silence``); with no such silence the window edge is
    used (``window``). Owned ranges tile ``[0, n)`` exactly — the assertion in
    :func:`chunk_coverage` is what makes "no omission" a fact rather than a hope.
    """
    total = len(samples)
    if single_pass:
        return [Chunk(0, 0.0, total / SAMPLE_RATE, 0.0, total / SAMPLE_RATE, "whole")]

    max_samples = int(max_chunk_s * SAMPLE_RATE)
    min_chunk_samples = max_samples // 2
    runs = _silence_runs(samples)
    midpoints = sorted((s + e) // 2 for s, e in runs)

    bounds: list[tuple[int, str]] = [(0, "start")]
    position = 0
    while total - position > max_samples:
        ideal = position + max_samples
        lower = position + min_chunk_samples
        candidates = [m for m in midpoints if lower <= m <= ideal]
        if candidates:
            bounds.append((candidates[-1], "silence"))
            position = candidates[-1]
        else:
            bounds.append((ideal, "window"))
            position = ideal
    bounds.append((total, "end"))

    overlap = int(overlap_s * SAMPLE_RATE)
    chunks: list[Chunk] = []
    for index in range(len(bounds) - 1):
        owned_start, _ = bounds[index]
        owned_end, cut = bounds[index + 1]
        decode_start = max(0, owned_start - overlap)
        decode_end = min(total, owned_end + overlap)
        chunks.append(
            Chunk(
                index=index,
                decode_start_s=decode_start / SAMPLE_RATE,
                decode_end_s=decode_end / SAMPLE_RATE,
                owned_start_s=owned_start / SAMPLE_RATE,
                owned_end_s=owned_end / SAMPLE_RATE,
                cut=cut,
            )
        )
    if len(chunks) == 1:
        chunks[0].cut = "whole"
    return chunks


def chunk_coverage(chunks: list[Chunk], total_s: float) -> tuple[float, float]:
    """Return ``(gap_s, double_owned_s)``, or refuse a plan that loses audio.

    Three properties are required of a plan, and all three are checked here:

    * every owned range starts where the previous one ended — no internal gap,
      no double ownership;
    * the last owned range ends at ``total_s`` — **this is the one failure a plan
      built from boundaries can actually have**, and checking the boundaries
      alone would miss it: a missing final bound silently drops the tail;
    * the owned seconds add up to ``total_s``.

    The first property is a structural consequence of :func:`plan_chunks`, which
    cuts at successive boundaries — which is exactly why it must be *asserted*
    and not reported as a finding: an instrument that can only confirm what the
    planner guarantees by construction measures nothing. The tail check is the
    part with teeth, and it needs ``total_s``, so the caller must pass it.

    Anything else means the split lost or duplicated audio, and the run must be
    treated as invalid rather than reported as a measurement: this raises
    ``SystemExit`` with the numbers instead of returning a verdict the caller can
    forget to read. ``COVERAGE_TOLERANCE_S`` absorbs float rounding only.
    """
    if not chunks:
        raise SystemExit("chunk plan is empty: nothing would be transcribed")
    if total_s <= 0:
        raise SystemExit(f"chunk plan checked against a non-positive file: total_s={total_s}")

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
        failures.append(f"owned {owned:.6f}s != file {total_s:.6f}s")
    if failures:
        raise SystemExit(
            "chunk plan does not tile the file exactly once (tolerance "
            f"{COVERAGE_TOLERANCE_S}s): " + "; ".join(failures)
        )
    return gap, double


# --------------------------------------------------------------------------- #
# Stitching
# --------------------------------------------------------------------------- #


def stitch(parts: list[str], chunks: list[Chunk], overlap_s: float) -> tuple[str, int]:
    """Join chunk transcripts, dropping the words the overlap duplicated.

    The overlap exists so a word at the seam is fully inside one chunk; the price
    is that the next chunk usually repeats the tail of the previous one. For each
    following chunk the longest word-level overlap with the accumulated text is
    removed, limited to the words the overlap could plausibly contain (its
    duration times the chunk's own words-per-second, plus slack). Returns
    ``(text, removed_words)``.
    """
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


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _weights_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _weights_breakdown(snapshot: Path) -> dict[str, int]:
    """Per-part sizes of a checkpoint snapshot — what provisioning has to budget."""
    breakdown: dict[str, int] = {}
    for name in ("model.safetensors", "model.bin", "language_model", "other"):
        if name == "other":
            breakdown[name] = _weights_bytes(snapshot) - sum(breakdown.values())
        else:
            target = snapshot / name
            if target.exists():
                breakdown[name] = _weights_bytes(target)
    return breakdown


def _rss_mib(field: str = "VmRSS") -> float | None:
    """Current or peak (``VmHWM``) resident set size of this process, in MiB.

    The KenLM side is CPU-only and invisible to ``nvidia-smi``, so RAM is the
    cost that decides in-process vs sidecar; ``VmHWM`` is the kernel's own high
    water mark since process start, not a sample that can miss the peak.
    """
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def _rss_snapshot() -> tuple[float | None, float | None]:
    """``(VmRSS, VmHWM)`` of this process in MiB — current and lifetime peak.

    Sampled at three points of a run (before the model is loaded, after it is on
    the device, after unloading), because RAM is the whole argument in the
    in-process vs sidecar question and two of its parts never meet: Torch's
    allocator counters describe VRAM, ``nvidia-smi`` cannot see the 2.1 GB KenLM
    that lives on the CPU, and only ``VmRSS``/``VmHWM`` cover both.
    """
    return _rss_mib("VmRSS"), _rss_mib("VmHWM")


def _load_gigaam_model(args: argparse.Namespace) -> tuple[object, str]:
    """Load ``GigaAMCTCHF``, recording which loader actually worked.

    ``transformers`` 5.x builds every model inside a ``torch.device("meta")``
    context, and this remote code calls ``int(tensor)`` on a tensor it created in
    ``StridingSubsampling.__init__`` (``encoder.py``) — which raises
    ``RuntimeError: Tensor.item() cannot be called on meta tensors``. The wrapper
    predates that change. ``--loader from_config`` therefore builds the module
    from the config (no meta context) and fills it from the checkpoint's own
    ``model.safetensors``: same weights, same classes, one extra step, no edit to
    the third-party file. ``--loader auto`` tries ``from_pretrained`` first so the
    failure is a measured fact rather than a memory, then falls back.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    if args.loader in ("auto", "from_pretrained"):
        try:
            model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
            return model, "from_pretrained"
        except Exception as exc:
            if args.loader == "from_pretrained":
                raise
            print(
                f"from_pretrained failed ({type(exc).__name__}: {exc}); "
                "falling back to from_config + safetensors",
                file=sys.stderr,
            )

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    weights = hf_hub_download(args.model_id, "model.safetensors")
    state = load_file(weights)
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = model.to(dtype)
    report = model.load_state_dict(state, strict=False)
    if report.unexpected_keys:
        raise SystemExit(f"checkpoint has keys the model does not: {report.unexpected_keys[:5]}")
    if len(report.missing_keys) > 0:
        print(f"missing keys filled by init: {len(report.missing_keys)}", file=sys.stderr)
    return model, "from_config+safetensors"


def _processor_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Kwargs for ``AutoProcessor.from_pretrained`` — including ``cache_dir``.

    ``cache_dir`` matters more than it looks: the wrapper hands it straight to
    ``pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub``, which otherwise
    defaults to ``~/.cache/pyctcdecode`` and *ignores* ``HF_HOME``. Without it
    the 2.1 GB KenLM snapshot is fetched into that separate directory, and an
    offline run then fails with ``LocalEntryNotFoundError`` although the same
    files already sit in the HF cache. Pointing it at ``$HF_HOME/hub`` keeps a
    single copy and lets ``HF_HUB_OFFLINE=1`` hold.
    """
    kwargs: dict[str, object] = {"trust_remote_code": True}
    cache_dir = args.cache_dir
    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME")
        cache_dir = str(Path(hf_home) / "hub") if hf_home else None
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if args.alpha is not None:
        kwargs["alpha"] = args.alpha
    if args.beta is not None:
        kwargs["beta"] = args.beta
    return kwargs


def _load_frontend(args: argparse.Namespace, with_lm: bool) -> object:
    """Feature extractor + tokenizer, with the KenLM decoder only when asked for.

    ``gigaam-greedy`` must *not* pay for the LM: through ``AutoProcessor`` the
    wrapper always builds the 2.1 GB ``BeamSearchDecoderCTC``, which would make
    the greedy run an unfair picture of what greedy costs (and the earlier
    version of this probe did exactly that). Greedy therefore loads the
    feature extractor and tokenizer through their own auto classes.
    """
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

    if with_lm:
        return AutoProcessor.from_pretrained(args.model_id, **_processor_kwargs(args))
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_id, trust_remote_code=True, cache_dir=_processor_kwargs(args).get("cache_dir")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, cache_dir=_processor_kwargs(args).get("cache_dir")
    )
    return (feature_extractor, tokenizer)


def transcribe_gigaam(
    args: argparse.Namespace, samples: "object", chunks: list[Chunk]
) -> dict[str, object]:
    """GigaAM-v2 CTC through the HF wrapper: encoder on the device, LM on the CPU."""
    import numpy as np
    import torch

    device = "cuda" if args.device == "cuda" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("torch reports no CUDA device; --device cpu is the honest fallback")

    with_lm = args.backend == "gigaam-lm"
    rss_before_mib, rss_hwm_before_mib = _rss_snapshot()
    started = time.perf_counter()
    model, load_path = _load_gigaam_model(args)
    frontend = _load_frontend(args, with_lm=with_lm)
    model = model.to(device).eval()
    load_s = time.perf_counter() - started
    rss_after_load_mib, rss_hwm_after_load_mib = _rss_snapshot()

    if with_lm:
        feature_extractor = frontend.feature_extractor
        tokenizer = frontend.tokenizer
    else:
        feature_extractor, tokenizer = frontend

    texts: list[str] = []
    model_dtype = next(model.parameters()).dtype
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inference_s = 0.0
    with torch.inference_mode():
        for chunk in chunks:
            audio = np.asarray(samples)[
                int(chunk.decode_start_s * SAMPLE_RATE) : int(chunk.decode_end_s * SAMPLE_RATE)
            ]
            started = time.perf_counter()
            inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            # The feature extractor always emits float32; the model may be fp16
            # (the checkpoint's own dtype, 465 MB) or fp32 (930 MB, what the
            # wrapper's README gets when it does not pass `dtype`). The cast is
            # the adapter's job either way.
            input_features = inputs["input_features"].to(device=device, dtype=model_dtype)
            input_lengths = inputs["input_lengths"].to(device)
            # The wrapper returns a CausalLMOutput (``.logits``), not the
            # ``(logits, lengths)`` tuple of the inner ``GigaAMCTC.forward``.
            output = model(input_features=input_features, input_lengths=input_lengths)
            logits = output.logits if hasattr(output, "logits") else output[0]
            logits = logits.float().cpu().numpy()
            if args.backend == "gigaam-greedy":
                texts.append(_greedy_ctc(logits, tokenizer))
            else:
                decoded = frontend.batch_decode(
                    logits=logits,
                    beam_width=args.beam_width,
                    alpha=args.alpha,
                    beta=args.beta,
                )
                texts.append(decoded.text[0])
            inference_s += time.perf_counter() - started

    peak_alloc_mib = None
    after_unload_alloc_mib = None
    if device == "cuda":
        peak_alloc_mib = torch.cuda.max_memory_allocated() / 1024**2
    # A previous version of this probe deleted these names inside a helper,
    # which freed nothing: the *caller* still held `model` and the processor, and
    # the "after unload" allocator counter stayed at the model's own size. The
    # deletion has to happen where the references live.
    del model, frontend, feature_extractor, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        after_unload_alloc_mib = torch.cuda.memory_allocated() / 1024**2
    rss_after_unload_mib, rss_hwm_after_unload_mib = _rss_snapshot()

    return {
        "load_s": load_s,
        "load_path": load_path,
        "inference_s": inference_s,
        "parts": texts,
        "peak_vram_alloc_mib": peak_alloc_mib,
        "vram_after_unload_alloc_mib": after_unload_alloc_mib,
        "rss_before_mib": rss_before_mib,
        "rss_hwm_before_mib": rss_hwm_before_mib,
        "rss_after_load_mib": rss_after_load_mib,
        "rss_hwm_after_load_mib": rss_hwm_after_load_mib,
        "rss_after_unload_mib": rss_after_unload_mib,
        "rss_hwm_after_unload_mib": rss_hwm_after_unload_mib,
        "unloaded": True,
    }


def _greedy_ctc(logits: "object", tokenizer: "object") -> str:
    """Argmax CTC: drop consecutive duplicates, then drop the blank token (id 33)."""
    import numpy as np

    ids = np.argmax(np.asarray(logits)[0], axis=-1).tolist()
    blank_id = 33
    pieces: list[str] = []
    previous = None
    for token_id in ids:
        if token_id != previous and token_id != blank_id:
            pieces.append(str(tokenizer.convert_ids_to_tokens(token_id)))
        previous = token_id
    return "".join(pieces).strip()


def transcribe_whisper(args: argparse.Namespace, samples: "object") -> dict[str, object]:
    """The current production backend, with the live profile's own settings."""
    from faster_whisper import WhisperModel

    device = args.device
    rss_before_mib, rss_hwm_before_mib = _rss_snapshot()
    started = time.perf_counter()
    model = WhisperModel(args.whisper_model, device=device, compute_type=args.compute_type)
    load_s = time.perf_counter() - started
    rss_after_load_mib, rss_hwm_after_load_mib = _rss_snapshot()

    started = time.perf_counter()
    segments, _info = model.transcribe(samples, language=args.language, beam_size=args.beam_size)
    text = "".join(segment.text for segment in segments).strip()
    inference_s = time.perf_counter() - started

    unloaded = False
    unload_hook = getattr(getattr(model, "model", None), "unload_model", None)
    if callable(unload_hook):
        unload_hook()
        unloaded = True
    del model
    gc.collect()
    rss_after_unload_mib, rss_hwm_after_unload_mib = _rss_snapshot()
    return {
        "load_s": load_s,
        "inference_s": inference_s,
        "parts": [text],
        "peak_vram_alloc_mib": None,
        "vram_after_unload_alloc_mib": None,
        "rss_before_mib": rss_before_mib,
        "rss_hwm_before_mib": rss_hwm_before_mib,
        "rss_after_load_mib": rss_after_load_mib,
        "rss_hwm_after_load_mib": rss_hwm_after_load_mib,
        "rss_after_unload_mib": rss_after_unload_mib,
        "rss_hwm_after_unload_mib": rss_hwm_after_unload_mib,
        "unloaded": unloaded,
    }


# --------------------------------------------------------------------------- #
# Text metrics
# --------------------------------------------------------------------------- #


def normalize_text(text: str) -> str:
    """Lowercase, ``ё``→``е``, letters/digits/space only, single spaces."""
    folded = text.lower().replace("ё", "е")
    kept = [ch if ch.isalnum() or ch.isspace() else " " for ch in folded]
    return " ".join("".join(kept).split())


def levenshtein(a: str, b: str) -> int:
    """Edit distance over characters, one vectorized ``numpy`` row per character.

    ``D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+cost)`` — the left
    dependency (``D[i][j-1]+1``) is not a plain elementwise minimum, so it is
    folded into a running minimum: ``D[i][j] = min_k(raw[k] - k) + j``.
    Doing it by elementwise ``minimum`` over overlapping views instead produced
    *negative* distances (a silently wrong number, caught by a comparison
    against a pure-Python reference on random strings).
    """
    import numpy as np

    if not a:
        return len(b)
    if not b:
        return len(a)
    codes = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    size = len(b)
    index = np.arange(size + 1, dtype=np.int64)
    previous = index.copy()
    raw = np.empty(size + 1, dtype=np.int64)
    for position, char in enumerate(a):
        cost = (codes != ord(char)).astype(np.int64)
        raw[0] = position + 1
        np.minimum(previous[1:] + 1, previous[:-1] + cost, out=raw[1:])
        previous = np.minimum.accumulate(raw - index) + index
    return int(previous[-1])


def agreement(a: str, b: str) -> dict[str, float | int]:
    """Normalized character agreement ``1 - edit / max(len)`` plus raw counts."""
    na, nb = normalize_text(a), normalize_text(b)
    distance = levenshtein(na, nb)
    longest = max(len(na), len(nb))
    return {
        "chars_a": len(na),
        "chars_b": len(nb),
        "distance": distance,
        "agreement": 1.0 - (distance / longest if longest else 0.0),
    }


def _window_measure(seams: list[float], half_width: float) -> float:
    """Share of the positional axis inside ``[s-h, s+h] ∩ [0, 1]`` for any seam.

    This is the number that turns "the differences sit at the seams" into a
    checkable statement: with the 18 seams of the 20 s plan, a ±1 % window already
    covers about a third of the file, so a raw count of "near-seam" characters
    means nothing on its own — it has to be compared with what a uniform
    distribution of the same differences would put there. Overlapping intervals
    are merged, and the window is clipped at both ends of the file.
    """
    intervals = sorted((max(0.0, s - half_width), min(1.0, s + half_width)) for s in seams)
    measure = 0.0
    reach = 0.0
    for start, stop in intervals:
        if stop <= reach:
            continue
        measure += stop - max(start, reach)
        reach = stop
    return measure


def seam_attribution(a: str, b: str, seams: list[float]) -> dict[str, object]:
    """How many differing characters sit near a chunk seam, against expectation.

    Character positions are mapped to time by proportion — good enough to say
    whether the damage clusters at the seams or is spread over the file, and
    explicitly *not* a word timestamp. Positions are read from whichever text is
    longer, so swapping the two arguments cannot change the numbers (a metric that
    reports 107 for one order and 105 for the other is not a measurement). For
    each half-width in ``SEAM_WINDOWS`` the report carries both the observed count
    and the count a uniform placement of the same differing characters would
    produce: ``observed / expected`` is the only readable form of this metric,
    because the 18 seams of the 20 s plan put a ±1 % window over a third of the
    file.
    """
    na, nb = normalize_text(a), normalize_text(b)
    # Canonical, order-independent alignment: longer text first, ties broken by the
    # lexicographically larger one. ``difflib`` aligns its first argument to its
    # second, and the opcodes it produces are not symmetric — the same pair gave
    # 107 differing characters when read one way and 105 the other. A metric that
    # answers differently depending on the argument order is not a measurement, so
    # the order is fixed here and `--text` order stops mattering.
    first, second = sorted((na, nb), key=lambda text: (-len(text), text))
    longest = len(first)

    def opcodes(x: str, y: str) -> list[tuple[int, int, int, int]]:
        import difflib

        matcher = difflib.SequenceMatcher(a=x, b=y, autojunk=False)
        return [op for op in matcher.get_opcodes() if op[0] != "equal"]

    regions = opcodes(first, second)
    different: list[float] = []
    worst = 0.0
    for _tag, i1, i2, j1, j2 in regions:
        position = (i1 + i2) / 2 / max(longest, 1)
        distance = min((abs(position - s) for s in seams), default=1.0)
        worst = max(worst, distance)
        # One entry per differing character in the region, so a region's width
        # counts the way the edit metric counts it.
        different.extend([distance] * max(i2 - i1, j2 - j1))

    windows: dict[str, dict[str, float | int]] = {}
    for half_width in SEAM_WINDOWS:
        near = sum(1 for distance in different if distance <= half_width)
        # What a uniform placement of the *same number of differing characters*
        # would put inside the windows — the only comparison that means anything.
        expected = len(different) * _window_measure(seams, half_width)
        windows[f"{half_width:g}"] = {
            "observed_chars": near,
            "expected_uniform_chars": round(expected, 1),
            "observed_over_expected": round(near / expected, 3) if expected else None,
        }
    return {
        "diff_regions": len(regions),
        "diff_chars": len(different),
        "chars_first_text": len(first),
        "chars_second_text": len(second),
        "diff_max_distance_frac": worst,
        "seam_windows": windows,
    }


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    backend: str
    audio: str
    duration_s: float
    source_duration_s: float
    decode_s: float
    load_s: float
    load_path: str
    inference_s: float
    rtf: float
    n_chunks: int
    cut_kinds: dict[str, int] = field(default_factory=dict)
    dedup_words: int = 0
    coverage_gap_s: float = 0.0
    coverage_double_s: float = 0.0
    vram_before_mib: int | None = None
    peak_vram_mib: int | None = None
    vram_after_unload_mib: int | None = None
    peak_vram_alloc_mib: float | None = None
    vram_after_unload_alloc_mib: float | None = None
    rss_before_mib: float | None = None
    rss_hwm_before_mib: float | None = None
    rss_after_load_mib: float | None = None
    rss_hwm_after_load_mib: float | None = None
    rss_after_unload_mib: float | None = None
    rss_hwm_after_unload_mib: float | None = None
    unloaded: bool = False
    weights_bytes: int | None = None
    weights_breakdown: dict[str, int] = field(default_factory=dict)
    weights_path: str | None = None
    text_chars: int = 0
    text_sha256: str = ""
    seams_frac: list[float] = field(default_factory=list)
    show_text: str | None = None


def _guard_repo_path(path: Path, what: str) -> Path:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SystemExit(
            f"refusing to write {what} inside the repository ({resolved}); use a path under /tmp"
        )
    return resolved


def run_backend(args: argparse.Namespace) -> RunResult:
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"audio not found: {audio_path}")

    sampler = VramSampler()
    sampler.start()
    baseline = VramSampler.read_used_mib()
    if baseline is not None:
        sampler.samples.append(baseline)

    started = time.perf_counter()
    samples, duration_s, source_duration_s = decode_audio(audio_path, args.max_seconds)
    decode_s = time.perf_counter() - started

    if args.backend == "whisper":
        # faster-whisper does its own windowing internally (30 s windows, VAD
        # off by default in the live profile), so *this* script does not cut the
        # file for it; claiming its chunk count from our own plan would be a
        # measurement of nothing. Report one whole-file unit instead.
        chunks = [Chunk(0, 0.0, duration_s, 0.0, duration_s, "whole")]
    else:
        chunks = plan_chunks(samples, args.max_chunk_s, args.chunk_overlap_s, args.single_pass)
    # Refuses (SystemExit) unless the owned ranges tile the whole file exactly
    # once, the tail included — see `chunk_coverage` for why the tail needs the
    # file length passed in.
    gap, double = chunk_coverage(chunks, duration_s)

    if args.backend == "whisper":
        outcome = transcribe_whisper(args, samples)
    else:
        outcome = transcribe_gigaam(args, samples, chunks)

    text, dedup_words = stitch(list(outcome["parts"]), chunks, args.chunk_overlap_s)
    if args.backend == "whisper":
        text = str(outcome["parts"][0])
        dedup_words = 0

    unload_used = VramSampler.read_used_mib()
    sampler.stop()
    peak = sampler.peak
    if peak is None:
        peak = unload_used

    weights_path = None
    weights_bytes = None
    weights_breakdown: dict[str, int] = {}
    if args.backend.startswith("gigaam"):
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub = hf_home / "hub" / ("models--" + args.model_id.replace("/", "--"))
        if hub.exists():
            snapshots = sorted((hub / "snapshots").iterdir())
            if snapshots:
                weights_path = str(snapshots[-1])
                weights_bytes = _weights_bytes(snapshots[-1])
                weights_breakdown = _weights_breakdown(snapshots[-1])
    else:
        hub = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / ("models--Systran--faster-whisper-" + args.whisper_model)
        )
        if hub.exists():
            snapshots = sorted((hub / "snapshots").iterdir())
            if snapshots:
                weights_path = str(snapshots[-1])
                weights_bytes = _weights_bytes(snapshots[-1])
                weights_breakdown = _weights_breakdown(snapshots[-1])

    result = RunResult(
        backend=args.backend,
        audio=str(audio_path),
        duration_s=duration_s,
        source_duration_s=source_duration_s,
        decode_s=decode_s,
        load_s=float(outcome["load_s"]),
        load_path=str(outcome.get("load_path", "n/a")),
        inference_s=float(outcome["inference_s"]),
        rtf=float(outcome["inference_s"]) / duration_s if duration_s else 0.0,
        n_chunks=len(chunks),
        cut_kinds={
            kind: sum(1 for c in chunks if c.cut == kind)
            for kind in sorted({c.cut for c in chunks})
        },
        dedup_words=dedup_words,
        coverage_gap_s=gap,
        coverage_double_s=double,
        vram_before_mib=baseline,
        peak_vram_mib=peak,
        vram_after_unload_mib=unload_used,
        peak_vram_alloc_mib=outcome["peak_vram_alloc_mib"],
        vram_after_unload_alloc_mib=outcome["vram_after_unload_alloc_mib"],
        rss_before_mib=outcome["rss_before_mib"],
        rss_hwm_before_mib=outcome["rss_hwm_before_mib"],
        rss_after_load_mib=outcome["rss_after_load_mib"],
        rss_hwm_after_load_mib=outcome["rss_hwm_after_load_mib"],
        rss_after_unload_mib=outcome["rss_after_unload_mib"],
        rss_hwm_after_unload_mib=outcome["rss_hwm_after_unload_mib"],
        unloaded=bool(outcome["unloaded"]),
        weights_bytes=weights_bytes,
        weights_breakdown=weights_breakdown,
        weights_path=weights_path,
        text_chars=len(text),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        seams_frac=[c.owned_end_s / duration_s for c in chunks[:-1]] if duration_s else [],
        show_text=text[: args.show_text] if args.show_text else None,
    )

    if args.emit_text:
        target = _guard_repo_path(Path(args.emit_text), "a transcript")
        target.write_text(text, encoding="utf-8")
        print(f"transcript written to {target} (delete after use)", file=sys.stderr)

    if args.json:
        target = _guard_repo_path(Path(args.json), "the metrics JSON")
        target.write_text(json.dumps(result.__dict__, ensure_ascii=False, indent=2), "utf-8")
        print(f"metrics written to {target}", file=sys.stderr)

    return result


def print_run(result: RunResult) -> None:
    def mib(value: float | None) -> str | None:
        return None if value is None else f"{value:.1f}"

    rows: list[tuple[str, object]] = [
        ("backend", result.backend),
        ("audio", result.audio),
        ("source_duration_s", f"{result.source_duration_s:.3f}"),
        ("decoded_duration_s", f"{result.duration_s:.3f}"),
        ("decode_audio_s", f"{result.decode_s:.3f}"),
        ("model_load_s", f"{result.load_s:.3f}"),
        ("load_path", result.load_path),
        ("inference_s", f"{result.inference_s:.3f}"),
        ("rtf", f"{result.rtf:.4f}"),
        ("n_chunks", result.n_chunks),
        ("cut_kinds", json.dumps(result.cut_kinds)),
        ("dedup_words", result.dedup_words),
        ("coverage_gap_s", f"{result.coverage_gap_s:.3f}"),
        ("coverage_double_s", f"{result.coverage_double_s:.3f}"),
        ("vram_before_mib", result.vram_before_mib),
        ("peak_vram_mib", result.peak_vram_mib),
        ("vram_after_unload_mib", result.vram_after_unload_mib),
        ("peak_vram_alloc_mib", mib(result.peak_vram_alloc_mib)),
        ("vram_after_unload_alloc_mib", mib(result.vram_after_unload_alloc_mib)),
        ("rss_before_mib", mib(result.rss_before_mib)),
        ("rss_hwm_before_mib", mib(result.rss_hwm_before_mib)),
        ("rss_after_load_mib", mib(result.rss_after_load_mib)),
        ("rss_hwm_after_load_mib", mib(result.rss_hwm_after_load_mib)),
        ("rss_after_unload_mib", mib(result.rss_after_unload_mib)),
        ("rss_hwm_after_unload_mib", mib(result.rss_hwm_after_unload_mib)),
        ("unloaded", result.unloaded),
        ("weights_bytes", result.weights_bytes),
        ("weights_breakdown", json.dumps(result.weights_breakdown)),
        ("weights_path", result.weights_path),
        ("text_chars", result.text_chars),
        ("text_sha256", result.text_sha256),
        ("seams_frac", ",".join(f"{s:.4f}" for s in result.seams_frac)),
    ]
    width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}}  {value}")
    if result.show_text is not None:
        print(f"text_preview  {result.show_text}")


def mode_compare(args: argparse.Namespace) -> None:
    labels = args.label or []
    texts: list[tuple[str, str]] = []
    for index, path in enumerate(args.text):
        label = labels[index] if index < len(labels) else Path(path).stem
        texts.append((label, Path(path).read_text(encoding="utf-8")))

    # Seams come from a run's metrics JSON whenever possible: a hand-typed list is
    # one transcription slip away from reporting a different number for the same
    # texts, and the seam positions are already recorded in the JSON by the run
    # that produced the text.
    if args.seams_from_json:
        for path in args.seams_from_json:
            text = Path(path).read_text(encoding="utf-8")
            seams = [float(value) for value in json.loads(text)["seams_frac"]]
            print(f"seams from {path}: {len(seams)} values", file=sys.stderr)
            break
        if not seams:
            raise SystemExit("the metrics JSON has an empty seams_frac; nothing to attribute")
    else:
        seams = [float(part) for part in args.seams.split(",")] if args.seams else []
    if seams and (min(seams) < 0.0 or max(seams) > 1.0):
        raise SystemExit(
            f"seams must be proportions in [0, 1], got min={min(seams)} max={max(seams)}"
        )
    if seams:
        print(
            f"seam windows (half-width): {', '.join(f'{w:g}' for w in SEAM_WINDOWS)}; "
            f"±1 % covers {_window_measure(seams, 0.01) * 100:.1f} % of the file",
            file=sys.stderr,
        )

    report: dict[str, object] = {}
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a_label, a_text = texts[i]
            b_label, b_text = texts[j]
            key = f"{a_label}__vs__{b_label}"
            entry: dict[str, object] = agreement(a_text, b_text)
            if seams:
                entry.update(seam_attribution(a_text, b_text, seams))
            report[key] = entry
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        target = _guard_repo_path(Path(args.json), "the comparison JSON")
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


# --------------------------------------------------------------------------- #
# Self-check (no model, no GPU, no audio files)
# --------------------------------------------------------------------------- #


def _selftest_case(name: str, chunks: list[Chunk], total_s: float, must_reject: bool) -> bool:
    """Run one coverage case and report whether the instrument behaved."""
    try:
        gap, double = chunk_coverage(chunks, total_s)
    except SystemExit as exc:
        accepted: bool | str = False
        note = str(exc)
    else:
        accepted = True
        note = f"accepted with gap={gap}s double={double}s"

    ok = accepted is not must_reject
    verdict = "rejected" if not accepted else "accepted"
    print(f"{'ok  ' if ok else 'FAIL'} {name:<34} {verdict:<8} {note}")
    return ok


def _selftest_coverage() -> bool:
    """Feed the coverage instrument the failures it is supposed to catch.

    The mutation this guards against is a plan that simply stops early: the
    boundary-walking check would call it valid, because a plan built from
    successive bounds cannot have an internal hole. The tail case below is that
    exact mutation, and `plan_chunks` on synthetic audio checks the real planner
    is accepted by the same instrument.
    """
    import numpy as np

    print("== coverage of the chunk plan ==")
    results = [
        _selftest_case(
            "exact tiling of 100 s",
            [Chunk(0, 0.0, 60.0, 0.0, 60.0, "silence"), Chunk(1, 60.0, 100.0, 60.0, 100.0, "end")],
            100.0,
            must_reject=False,
        ),
        _selftest_case(
            "lost tail (mutation)",
            [Chunk(0, 0.0, 60.0, 0.0, 60.0, "silence"), Chunk(1, 60.0, 96.0, 60.0, 96.0, "end")],
            100.0,
            must_reject=True,
        ),
        _selftest_case("empty plan", [], 100.0, must_reject=True),
        _selftest_case(
            "internal gap",
            [Chunk(0, 0.0, 50.0, 0.0, 50.0, "silence"), Chunk(1, 55.0, 100.0, 55.0, 100.0, "end")],
            100.0,
            must_reject=True,
        ),
        _selftest_case(
            "leading gap",
            [Chunk(0, 5.0, 100.0, 5.0, 100.0, "end")],
            100.0,
            must_reject=True,
        ),
        _selftest_case(
            "double ownership",
            [Chunk(0, 0.0, 60.0, 0.0, 60.0, "silence"), Chunk(1, 55.0, 100.0, 55.0, 100.0, "end")],
            100.0,
            must_reject=True,
        ),
    ]

    # The real planner on synthetic audio: 100 s of tone with silence once a
    # second, cut at 20 s. This is the case that would have passed before the fix
    # even if the planner dropped its last bound, so it must be the same
    # instrument that rejects the mutation above.
    seconds = 100.0
    samples = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    for start in range(0, int(seconds), 4):
        tone = np.sin(2 * np.pi * 220 * np.arange(int(3 * SAMPLE_RATE)) / SAMPLE_RATE)
        samples[start * SAMPLE_RATE : start * SAMPLE_RATE + len(tone)] = (tone * 0.3).astype(
            np.float32
        )
    planned = plan_chunks(samples, 20.0, 0.5, single_pass=False)
    results.append(
        _selftest_case(f"plan_chunks on 100 s ({len(planned)} chunks)", planned, seconds, False)
    )
    print(f"     plan: cuts={[c.cut for c in planned]}")
    return all(results)


def _write_tone(path: Path, container_format: str, codec: str, samples: "object") -> str:
    """Mux a synthetic mono tone into ``path`` and return the codec actually used.

    Falls back to the next candidate codec when the bundled FFmpeg has no encoder
    for the first one (PyAV ships ``libopus`` but not ``libvorbis``, and the
    native ``vorbis`` encoder is experimental, so it needs ``-strict -2``). The
    container is what this check is about; which encoder muxed it is reported.
    """
    import av
    import numpy as np

    options = {"strict": "-2"} if codec == "vorbis" else {}
    with av.open(str(path), mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=SAMPLE_RATE, options=options)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            (np.asarray(samples) * 32767).astype(np.int16).reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = SAMPLE_RATE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return codec


def _selftest_decode() -> bool:
    """Decode ogg/m4a/wav/webm written by PyAV with no ffmpeg on the probe's PATH.

    The point is not that this machine lacks ffmpeg (it has one, `/usr/bin/ffmpeg`)
    but that the probe never calls it: PyAV carries its own FFmpeg libraries, so
    these containers must open with an empty PATH, and the check prints what
    ``shutil.which("ffmpeg")`` saw so the reader can tell which situation they are
    in. The audio is a synthetic tone — nothing from the diary is involved.
    """
    import shutil
    import tempfile

    import numpy as np

    print("== decoding containers without system ffmpeg ==")
    ffmpeg = shutil.which("ffmpeg")
    print(f"     shutil.which('ffmpeg') -> {ffmpeg!r} (checking the PATH this process sees)")

    seconds = 2.0
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    samples_in = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    results = []
    with tempfile.TemporaryDirectory(prefix="t42-decode-") as tmp:
        # (extension, container format, candidate encoders) — the muxer is named
        # explicitly because the extension alone is ambiguous for m4a.
        for extension, container_format, codecs in (
            ("wav", "wav", ("pcm_s16le",)),
            ("ogg", "ogg", ("vorbis", "libopus")),
            ("m4a", "mp4", ("aac",)),
            ("webm", "webm", ("libopus",)),
        ):
            path = Path(tmp) / f"tone.{extension}"
            used = None
            failure = ""
            for codec in codecs:
                try:
                    used = _write_tone(path, container_format, codec, samples_in)
                    break
                except Exception as exc:  # noqa: BLE001 — the report says which codec failed
                    failure = f"{codec}: {type(exc).__name__}: {exc}"
            if used is None:
                results.append(False)
                print(f"FAIL {extension:<5} no encoder available ({failure})")
                continue
            decoded, duration_s, _source = decode_audio(path, None)
            # AAC in mp4 carries encoder priming/padding, so the decoded length can
            # exceed the source by ~0.11 s; the other containers are sample exact.
            ok = (
                decoded.dtype == np.float32
                and abs(duration_s - seconds) < 0.25
                and float(np.max(np.abs(decoded))) > 0.05
            )
            results.append(ok)
            print(
                f"{'ok  ' if ok else 'FAIL'} {extension:<5} codec={used:<10} "
                f"decoded={duration_s:.3f}s (Δ{duration_s - seconds:+.3f}) "
                f"dtype={decoded.dtype} peak={float(np.max(np.abs(decoded))):.3f}"
            )
        # A real decoder failure must stay a failure, not turn into silence.
        broken = Path(tmp) / "broken.webm"
        broken.write_bytes(b"not a container")
        try:
            decode_audio(broken, None)
        except SystemExit as exc:
            results.append(True)
            print(f"ok   garbage input rejected           SystemExit: {exc}")
        else:
            results.append(False)
            print("FAIL garbage input decoded")
    return all(results)


def mode_selftest() -> None:
    """Check the instruments themselves (D1 class of defect: a blind instrument)."""
    coverage = _selftest_coverage()
    decode = _selftest_decode()
    print(
        f"\nselftest: coverage={'ok' if coverage else 'FAILED'} "
        f"decode={'ok' if decode else 'FAILED'}"
    )
    if not (coverage and decode):
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure one STT backend for the T42 spike (see the module docstring)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run", help="measure one backend on one audio file")
    run.add_argument("--backend", choices=["gigaam-lm", "gigaam-greedy", "whisper"], required=True)
    run.add_argument("--audio", required=True, help="path to webm/ogg/m4a/wav")
    run.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run.add_argument("--max-chunk-s", type=float, default=20.0)
    run.add_argument("--chunk-overlap-s", type=float, default=0.5)
    run.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="decode only this prefix (the short-file shot)",
    )
    run.add_argument(
        "--single-pass",
        action="store_true",
        help="no cutting at all; use to find the positional-embedding limit",
    )
    run.add_argument("--model-id", default="waveletdeboshir/gigaam-ctc-with-lm")
    run.add_argument(
        "--loader",
        choices=["auto", "from_pretrained", "from_config"],
        default="auto",
        help="auto = try from_pretrained, fall back to from_config+safetensors",
    )
    run.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    run.add_argument(
        "--cache-dir",
        default=None,
        help="HF cache dir handed to the wrapper; default = $HF_HOME/hub",
    )
    run.add_argument("--beam-width", type=int, default=64)
    run.add_argument(
        "--alpha", type=float, default=None, help="LM weight; default = attrs.json of the LM"
    )
    run.add_argument("--beta", type=float, default=None, help="word-insertion bonus")
    run.add_argument("--whisper-model", default="large-v3")
    run.add_argument("--compute-type", default="int8_float16")
    run.add_argument("--language", default="ru")
    run.add_argument("--beam-size", type=int, default=5)
    run.add_argument("--show-text", type=int, default=0, help="print at most N characters")
    run.add_argument("--emit-text", default=None, help="write the transcript (outside the repo)")
    run.add_argument("--json", default=None, help="write the metrics JSON (outside the repo)")

    compare = sub.add_parser("compare", help="agreement between two or more transcripts")
    compare.add_argument("--text", action="append", required=True)
    compare.add_argument("--label", action="append")
    compare.add_argument("--seams", default=None, help="comma-separated seam proportions")
    compare.add_argument(
        "--seams-from-json",
        action="append",
        default=None,
        help="take seams_frac from a run's metrics JSON (preferred over --seams)",
    )
    compare.add_argument("--json", default=None)

    sub.add_parser(
        "selftest",
        help="check the probe's own instruments on synthetic data (no model, no GPU)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "compare":
        mode_compare(args)
        return 0
    if args.mode == "selftest":
        mode_selftest()
        return 0
    print_run(run_backend(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
