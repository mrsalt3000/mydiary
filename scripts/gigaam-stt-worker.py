#!/usr/bin/env python
"""GigaAM-v2 CTC+LM sidecar worker (T44).

One invocation transcribes **one** audio file and prints **one JSON object** as
its last stdout line; diagnostics go to stderr. The application
(:mod:`diary.domain.transcription.gigaam`) starts it in the GigaAM virtualenv,
so this script is the only place in the product that imports ``torch``,
``transformers``, ``av`` and the KenLM decoder — none of them is a dependency of
the application or of CI. The wire contract (arguments, exit codes, JSON keys)
is :mod:`diary.domain.transcription.sidecar_protocol`.

Why this file has to be imported through ``importlib``
=====================================================

The chunk planner is **product code**, not a copy
(:mod:`diary.domain.transcription.chunking`), and this script loads that file by
path. A plain ``from diary.domain.transcription.chunking import plan_chunks``
cannot work here: importing a submodule runs the package ``__init__``, which
imports the service layer and, with it, pydantic and SQLAlchemy — neither of
which exists in the sidecar virtualenv (the runtime is deliberately isolated,
``docs/stt-backends-analysis.md`` §10).

What it does, and what was measured before it
=============================================

* decodes the container with PyAV and resamples to 16 kHz mono — no system
  ffmpeg (T42 §5 of the analysis document);
* computes per-frame RMS and lets :func:`plan_chunks` cut the recording at
  silence, up to ``--max-chunk-s``, with ``--chunk-overlap-s`` of overlap;
* loads ``model_id`` **at the pinned revision** through
  ``AutoConfig.from_pretrained`` + ``AutoModel.from_config`` + the checkpoint's
  ``model.safetensors``. The obvious ``AutoModel.from_pretrained`` is not used
  because transformers 5.x builds the model in a ``meta`` device context and
  this wrapper calls ``int(tensor)`` during construction (measured: ``RuntimeError:
  Tensor.item() cannot be called on meta tensors``; the workaround was verified
  to load all 600 tensors with no unexpected and no missing keys, analysis §11);
* decodes the encoder logits with the CTC+LM beam search, or with plain greedy
  CTC under ``--greedy`` (a debugging path: the language model is what the
  backend is chosen for);
* stitches the chunk transcripts and reports how many words the overlap
  duplicated, so the number is evidence rather than a belief.

Privacy (AGENTS.md §7–8): recognised text and audio samples never reach stderr
or any file. The only text that leaves this process is the ``text`` field of the
success JSON on stdout, which the application writes into the
``transcriptions`` table; the failure path prints nothing but the *class name*
of the error as the last stderr line.

Usage::

    HF_HOME=<sidecar-data>/hf HF_HUB_OFFLINE=1 <sidecar-python> \\
        scripts/gigaam-stt-worker.py <audio> \\
        --model-id waveletdeboshir/gigaam-ctc-with-lm \\
        --revision 07d97cf8a4f178b68679cafae0600c8487877301 \\
        --device cuda --beam-width 64 --max-chunk-s 20 --chunk-overlap-s 0.5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

#: Repository root, derived from this file's location (``<root>/scripts/…``).
REPO_ROOT = Path(__file__).resolve().parent.parent
#: Pure product modules this worker shares with the application.
_TRANSCRIPTION_DIR = REPO_ROOT / "src" / "diary" / "domain" / "transcription"

# The sample rate and the analysis frame are **the product planner's** values
# (``chunking.SAMPLE_RATE`` / ``chunking.SILENCE_FRAME_S``), and this script does
# not declare copies of them: it decodes and computes RMS with exactly the numbers
# the planner was measured against, so the two cannot drift apart (T45, closing the
# finding of the T44 acceptance). The module arrives as :func:`_load_pure_module`'s
# result and is threaded into the helpers below instead of being read from a
# module-level global — the worker has no import of the application package (see
# the module docstring), and a global loaded at import time would turn "the file is
# missing" into an ImportError before :func:`main` can report it through the
# protocol.

#: Exit code used when the *protocol module itself* cannot be loaded — the one
#: situation in which its constant is unavailable.
_FALLBACK_UNAVAILABLE_EXIT = 3


class UnavailableError(Exception):
    """A condition that repeats identically on this machine or with this input.

    Raised when the runtime or the weights are not usable here, or when the
    recording cannot be planned. Carries the *name* of the underlying failure, so
    the application's message stays informative without this script ever echoing
    a message (which could carry a path or a decoder payload).
    """

    def __init__(self, reported_class: str) -> None:
        super().__init__(reported_class)
        self.reported_class = reported_class


def _load_pure_module(name: str) -> ModuleType:
    """Load a stdlib-only product module from its file (see the module docstring)."""
    path = _TRANSCRIPTION_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gigaam_sidecar_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - requires a broken tree
        raise ImportError(f"cannot load the sidecar module {name!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fail(exc: BaseException, code: int) -> int:
    """Report the failure class as the last stderr line and return ``code``."""
    name = exc.reported_class if isinstance(exc, UnavailableError) else type(exc).__name__
    sys.stderr.write(f"{name}\n")
    sys.stderr.flush()
    return code


def build_parser() -> argparse.ArgumentParser:
    """The command line the application builds in ``gigaam._build_argv``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", help="path to the recording to transcribe")
    parser.add_argument("--model-id", required=True, help="Hugging Face repository of the wrapper")
    parser.add_argument("--revision", required=True, help="immutable revision (commit sha or tag)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument(
        "--alpha", type=float, default=None, help="LM weight; omitted = the checkpoint's own"
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="word bonus; omitted = the checkpoint's own"
    )
    parser.add_argument("--max-chunk-s", type=float, default=20.0)
    parser.add_argument("--chunk-overlap-s", type=float, default=0.5)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="argmax CTC without the language model (debugging path)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache root (default: $HF_HOME/hub)",
    )
    return parser


def _cache_dir(args: argparse.Namespace) -> str | None:
    """The hub cache the wrapper must be pointed at explicitly.

    ``cache_dir`` matters more than it looks: the wrapper hands it straight to
    ``pyctcdecode``'s ``load_from_hf_hub``, which otherwise defaults to
    ``~/.cache/pyctcdecode`` and ignores ``HF_HOME`` — the 2.1 GB KenLM snapshot
    would be fetched into that separate directory, and an offline run would then
    fail although the same files sit in the HF cache (measured, T42 §4.5).
    """
    if args.cache_dir:
        return str(args.cache_dir)
    hf_home = os.environ.get("HF_HOME")
    return str(Path(hf_home) / "hub") if hf_home else None


def decode_audio(path: Path, chunking: ModuleType) -> tuple[object, float, float]:
    """Decode ``path`` to float32 mono 16 kHz; return ``(samples, seconds, source_seconds)``.

    PyAV carries its own FFmpeg libraries, so no system ffmpeg is involved
    (``docs/wsl2-deployment-brief.md``). The target rate is the planner's
    ``chunking.SAMPLE_RATE``, not a number of this script's own. An unreadable or
    empty container is an error: a transcript of nothing is worse than no
    transcript.
    """
    import av
    import numpy as np

    with av.open(str(path)) as container:
        stream = next(
            (candidate for candidate in container.streams if candidate.type == "audio"), None
        )
        if stream is None:
            raise RuntimeError("no audio stream in the container")
        source_seconds = 0.0
        if stream.duration:
            source_seconds = float(stream.duration * stream.time_base)
        elif container.duration:
            source_seconds = container.duration / av.time_base
        resampler = av.AudioResampler(format="s16", layout="mono", rate=chunking.SAMPLE_RATE)
        parts = []
        for frame in container.decode(stream):
            parts.extend(resampler.resample(frame))
        parts.extend(resampler.resample(None))

    if not parts:
        raise RuntimeError("the container decoded to zero audio frames")
    joined = np.concatenate([np.asarray(part.to_ndarray()).reshape(-1) for part in parts])
    samples = joined.astype(np.float32) / 32768.0
    return samples, len(samples) / chunking.SAMPLE_RATE, source_seconds


def frame_rms(samples: object, chunking: ModuleType) -> list[float]:
    """Per-frame RMS of the whole frames in ``samples`` (see ``chunking``)."""
    import numpy as np

    frame = int(chunking.SILENCE_FRAME_S * chunking.SAMPLE_RATE)
    n_frames = len(samples) // frame  # type: ignore[arg-type]
    if n_frames < 2:
        return []
    frames = np.asarray(samples[: n_frames * frame], dtype=np.float32).reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return [float(value) for value in rms]


def _load_unavailable_types() -> tuple[type[BaseException], ...]:
    """Exceptions that, **during loading**, mean "the weights are not on this machine".

    ``OSError`` is the one that actually fires: measured 2026-09-24, an offline
    cache miss from this wrapper comes out as a bare
    ``OSError("We couldn't connect to 'https://huggingface.co' … couldn't find
    them in the cached files")``, not as the ``LocalEntryNotFoundError`` its name
    suggests (that class is still accepted for versions that raise it).
    Classification is applied to the **load** step only: there an OSError can
    only mean missing files, while the same exception during inference is a
    runtime failure and must stay transient.
    """
    types: list[type[BaseException]] = [ImportError, OSError]
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError

        types.append(LocalEntryNotFoundError)
    except Exception:  # pragma: no cover - huggingface_hub ships with transformers
        pass
    return tuple(types)


def load_backend(args: argparse.Namespace) -> tuple[object, object]:
    """Load the model and its frontend, reporting missing files as *unavailable*."""
    try:
        model = load_model(args)
        frontend = load_frontend(args, with_lm=not args.greedy)
    except _load_unavailable_types() as exc:
        raise UnavailableError(type(exc).__name__) from exc
    return model, frontend


def load_model(args: argparse.Namespace) -> object:
    """Load the checkpoint at the pinned revision and return the model.

    See the module docstring for why the weights are loaded from
    ``model.safetensors`` instead of through ``AutoModel.from_pretrained``.

    ``code_revision`` is the pin again, and that is not decoration: with
    ``trust_remote_code=True`` transformers fetches the checkpoint's *own* module
    file (``gigaam_transformers.py``) through the dynamic-module loader, and
    without an explicit revision that loader asks for the repository's default
    revision (``main``). A cache provisioned at a pinned commit deliberately has no
    ``refs/main`` entry — the pin is the sha, and the library writes no ref when the
    revision *is* the commit — so the load failed with ``LocalEntryNotFoundError``
    and the application reported "runtime is not provisioned" while every file was
    in fact present (measured 2026-09-24 on the provisioned cache: all 15 files
    verified by sha256, ``AutoTokenizer``, ``AutoFeatureExtractor`` and
    ``AutoProcessor`` loaded fine, only this call raised).

    ``code_revision`` rather than ``revision`` is deliberate, and it is the
    supported knob for this: ``from_config`` forwards leftover keyword arguments to
    **both** the module lookup and the model constructor, so ``revision=…`` here
    would reach ``GigaAMCTCHF.__init__`` and raise ``TypeError`` (measured), while
    ``code_revision`` is consumed by the loader (transformers pops it for exactly
    this purpose). The cache directory is not passed for the same reason — the child
    runs with ``HF_HOME`` pointing at the provisioned cache
    (``src/diary/domain/transcription/gigaam.py``), which *is* the default
    ``HF_HUB_CACHE`` — and the revision the code is read from is the same pin whose
    weights and ``sha256`` are verified at provisioning time.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=_cache_dir(args),
    )
    model = AutoModel.from_config(config, trust_remote_code=True, code_revision=args.revision)
    weights = hf_hub_download(
        args.model_id,
        "model.safetensors",
        revision=args.revision,
        cache_dir=_cache_dir(args),
    )
    report = model.load_state_dict(load_file(weights), strict=False)
    if report.unexpected_keys:
        raise RuntimeError(f"the checkpoint has {len(report.unexpected_keys)} unexpected keys")
    if report.missing_keys:
        print(f"keys filled by the model's own init: {len(report.missing_keys)}", file=sys.stderr)
    # The checkpoint's native dtype is fp16; fp32 would only cost more VRAM for
    # the same result (``GigaamCtcLmProfile`` deliberately has no dtype field).
    return model.to(torch.float16)


def load_frontend(args: argparse.Namespace, *, with_lm: bool) -> object:
    """Feature extractor + tokenizer, with the KenLM decoder only when asked for.

    ``--greedy`` must not pay for the language model: through ``AutoProcessor``
    the wrapper always builds its ``BeamSearchDecoderCTC``, which would make the
    greedy path an unfair picture of what greedy costs.
    """
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

    kwargs: dict[str, object] = {"trust_remote_code": True, "revision": args.revision}
    cache_dir = _cache_dir(args)
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if with_lm:
        if args.alpha is not None:
            kwargs["alpha"] = args.alpha
        if args.beta is not None:
            kwargs["beta"] = args.beta
        return AutoProcessor.from_pretrained(args.model_id, **kwargs)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **kwargs)
    return feature_extractor, tokenizer


def _greedy_ctc(logits: object, tokenizer: object) -> str:
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


def transcribe_chunks(
    args: argparse.Namespace, samples: object, chunks: list[object], chunking: ModuleType
) -> tuple[list[str], float]:
    """Run the model over ``chunks`` and return ``(per_chunk_texts, inference_seconds)``."""
    import numpy as np
    import torch

    with_lm = not args.greedy
    device = "cuda" if args.device == "cuda" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch reports no CUDA device; --device cpu is the honest fallback")

    model, frontend = load_backend(args)
    model = model.to(device).eval()
    if with_lm:
        feature_extractor = frontend.feature_extractor
        tokenizer = frontend.tokenizer
    else:
        feature_extractor, tokenizer = frontend
    print(f"model loaded on {device} ({args.model_id}@{args.revision})", file=sys.stderr)

    model_dtype = next(model.parameters()).dtype
    texts: list[str] = []
    inference_s = 0.0
    with torch.inference_mode():
        for chunk in chunks:
            audio = np.asarray(samples)[
                int(chunk.decode_start_s * chunking.SAMPLE_RATE) : int(
                    chunk.decode_end_s * chunking.SAMPLE_RATE
                )
            ]
            started = time.perf_counter()
            inputs = feature_extractor(
                audio, sampling_rate=chunking.SAMPLE_RATE, return_tensors="pt"
            )
            # The feature extractor always emits float32; the model is fp16.
            input_features = inputs["input_features"].to(device=device, dtype=model_dtype)
            input_lengths = inputs["input_lengths"].to(device)
            output = model(input_features=input_features, input_lengths=input_lengths)
            logits = output.logits if hasattr(output, "logits") else output[0]
            logits = logits.float().cpu().numpy()
            if with_lm:
                decoded = frontend.batch_decode(
                    logits=logits,
                    beam_width=args.beam_width,
                    alpha=args.alpha,
                    beta=args.beta,
                )
                texts.append(decoded.text[0])
            else:
                texts.append(_greedy_ctc(logits, tokenizer))
            inference_s += time.perf_counter() - started
    return texts, inference_s


def run(
    args: argparse.Namespace, chunking_module: ModuleType, protocol_module: ModuleType
) -> dict[str, object]:
    """Decode, plan, transcribe, stitch — and return the JSON payload."""
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise UnavailableError("FileNotFoundError")

    started = time.perf_counter()
    samples, duration_s, source_duration_s = decode_audio(audio_path, chunking_module)
    decode_s = time.perf_counter() - started

    rms = frame_rms(samples, chunking_module)
    try:
        chunks = chunking_module.plan_chunks(
            rms,
            frame_s=chunking_module.SILENCE_FRAME_S,
            total_s=duration_s,
            max_chunk_s=args.max_chunk_s,
            overlap_s=args.chunk_overlap_s,
        )
        gap_s, double_s = chunking_module.assert_coverage(chunks, duration_s)
    except chunking_module.ChunkPlanError as exc:
        # The plan is deterministic: the same file and settings repeat it, so
        # retrying the job would never help.
        raise UnavailableError(type(exc).__name__) from exc

    started = time.perf_counter()
    # ``transcribe_chunks`` loads the model through :func:`load_backend`, which
    # reports missing files as :class:`UnavailableError`; a failure after that is
    # a runtime one and stays transient (exit 1).
    parts, inference_s = transcribe_chunks(args, samples, chunks, chunking_module)
    load_and_inference_s = time.perf_counter() - started

    text, removed_words = chunking_module.stitch(parts, chunks, args.chunk_overlap_s)
    payload: dict[str, object] = {
        protocol_module.RESULT_KEY_TEXT: text,
        "chunks": [
            {
                "index": chunk.index,
                "owned_start_s": chunk.owned_start_s,
                "owned_end_s": chunk.owned_end_s,
                "decode_start_s": chunk.decode_start_s,
                "decode_end_s": chunk.decode_end_s,
                "cut": chunk.cut,
            }
            for chunk in chunks
        ],
        "n_chunks": len(chunks),
        "cut_kinds": {
            kind: sum(1 for chunk in chunks if chunk.cut == kind)
            for kind in sorted({chunk.cut for chunk in chunks})
        },
        "duration_s": duration_s,
        "source_duration_s": source_duration_s,
        "decode_s": decode_s,
        "transcribe_s": load_and_inference_s,
        "inference_s": inference_s,
        "removed_words": removed_words,
        protocol_module.RESULT_KEY_COVERAGE: {
            protocol_module.COVERAGE_KEY_GAP_S: gap_s,
            protocol_module.COVERAGE_KEY_DOUBLE_S: double_s,
        },
        "decode_mode": "greedy" if args.greedy else "ctc_lm",
        "device": args.device,
        "model_id": args.model_id,
        "revision": args.revision,
    }
    print(
        "decoded {duration:.2f}s in {decode:.2f}s; {chunks} chunks ({kinds}); "
        "load+inference {load:.2f}s ({inference:.2f}s inference); "
        "coverage gap {gap:.6f}s double {double:.6f}s; duplicate words removed {removed}; "
        "text {chars} chars".format(
            duration=duration_s,
            decode=decode_s,
            chunks=len(chunks),
            kinds=",".join(f"{chunk.cut}" for chunk in chunks[:3])
            + ("…" if len(chunks) > 3 else ""),
            load=load_and_inference_s,
            inference=inference_s,
            gap=gap_s,
            double=double_s,
            removed=removed_words,
            chars=len(text),
        ),
        file=sys.stderr,
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    """Entry point: run one transcription and return the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        chunking_module = _load_pure_module("chunking")
        protocol_module = _load_pure_module("sidecar_protocol")
    except Exception as exc:  # the protocol constant is what could not be loaded
        return _fail(exc, _FALLBACK_UNAVAILABLE_EXIT)
    try:
        payload = run(args, chunking_module, protocol_module)
    except UnavailableError as exc:
        # The runtime or the weights are not usable here, or the plan cannot be
        # built: the identical request repeats the identical outcome.
        return _fail(exc, protocol_module.SIDECAR_EXIT_UNAVAILABLE)
    except ImportError as exc:
        return _fail(exc, protocol_module.SIDECAR_EXIT_UNAVAILABLE)
    except Exception as exc:
        return _fail(exc, protocol_module.SIDECAR_EXIT_FAILED)
    # The last stdout line is the result; nothing else is printed to stdout.
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return protocol_module.SIDECAR_EXIT_OK


if __name__ == "__main__":
    # os._exit keeps the protocol exact: no atexit handler can append a line to
    # stderr after the error class name that the application reads.
    os._exit(main())
