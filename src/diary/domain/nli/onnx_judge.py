"""ONNX Runtime judge for the NLI rules (T24a).

The rules live in :mod:`diary.domain.contracts.nli` and know nothing about models;
this module is the production adapter that turns a (premise, claims) pair into the
three probabilities the rules consume, using a pinned Russian NLI cross-encoder
through ONNX Runtime — no ``torch``, no ``transformers``, no VRAM (CPU execution
provider only, so the veto «STT and LLM never share VRAM» is untouched).

Everything about availability is explicit and **fail-closed** (the owner's
decision for T24: «если NLI недоступен — Fail»):

* the optional extra ``nli`` is not installed — :class:`NliUnavailableError`
  naming the extra;
* the weights or the tokenizer are absent, or the model does not hash to the
  pinned value — :class:`NliUnavailableError` naming the file and
  ``scripts/fetch-nli-model.sh``;
* the label order of the converted model is not the expected
  entailment / contradiction / neutral — :class:`NliUnavailableError`, because a
  silently reordered model would invert every verdict.

Nothing here is loaded at import time and nothing is downloaded at run time:
weights are fetched once by ``scripts/fetch-nli-model.sh`` into
``APP_NLI_MODEL_DIR`` (default ``data/models/nli``), which is the same principle as
the pinned Ollama runtime and the Whisper cache — model files never enter the
repository, and the application never reaches the network for them
(``HF_HUB_OFFLINE=1``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diary.core.settings import (
    PINNED_NLI_MODEL_SHA256,
    AppSettings,
    get_settings,
)
from diary.domain.contracts.nli import LABELS, NliError

__all__ = [
    "DEFAULT_CONFIG_FILENAME",
    "DEFAULT_MODEL_FILENAME",
    "DEFAULT_TOKENIZER_FILENAME",
    "LABEL_ORDER",
    "NliUnavailableError",
    "OnnxNliJudge",
    "PINNED_MODEL_SHA256",
]

#: File names expected inside ``APP_NLI_MODEL_DIR``.
DEFAULT_MODEL_FILENAME = "model.onnx"
DEFAULT_TOKENIZER_FILENAME = "tokenizer.json"
DEFAULT_CONFIG_FILENAME = "config.json"

#: Label order the converted model must declare (``config.json`` → ``id2label``).
LABEL_ORDER: tuple[str, str, str] = LABELS

#: sha256 of the fp32 conversion this project measured and pinned; the settings
#: module owns the value and ``scripts/fetch-nli-model.sh`` carries the same one.
PINNED_MODEL_SHA256 = PINNED_NLI_MODEL_SHA256

#: Longest tokenized pair sent to the model (the model itself was trained on 512).
MAX_PAIR_TOKENS = 512


class NliUnavailableError(NliError):
    """The NLI instrument is requested but cannot be used.

    Raised instead of degrading: a requested check that silently does nothing is
    worse than a loud failure, because the report would claim a validation nobody
    performed.
    """


@dataclass
class OnnxNliJudge:
    """Callable judge backed by an ONNX cross-encoder on the CPU.

    Called with a batch of (premise, hypothesis) pairs — the shape the rules
    produce — and answering one three-way distribution per pair. The instance is
    cheap until it is first called: the session and the tokenizer are built
    lazily, so a run that never asks for a verdict never pays for loading 712 MB
    of weights. ``expected_sha256`` is verified at that first load (an empty value
    disables the pin, which the fetch script never does).
    """

    model_path: Path
    tokenizer_path: Path
    config_path: Path
    expected_sha256: str = PINNED_MODEL_SHA256
    _session: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _id_to_label: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> OnnxNliJudge:
        """Build the judge described by the application settings.

        The files are not touched here — a missing directory surfaces on the first
        verdict as :class:`NliUnavailableError`, which is where the caller can
        report it next to the run it belongs to.
        """
        resolved = settings if settings is not None else get_settings()
        directory = Path(resolved.nli_model_dir)
        return cls(
            model_path=directory / DEFAULT_MODEL_FILENAME,
            tokenizer_path=directory / DEFAULT_TOKENIZER_FILENAME,
            config_path=directory / DEFAULT_CONFIG_FILENAME,
            expected_sha256=resolved.nli_model_sha256,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _sha256(self) -> str:
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _require_file(self, path: Path, *, extra: str = "nli") -> None:
        if not path.is_file():
            raise NliUnavailableError(
                f"{path} is missing: run scripts/fetch-nli-model.sh "
                f'(and install the optional extra with pip install -e ".[{extra}]")'
            )

    def load(self) -> None:
        """Build the session and the tokenizer, verifying the pinned weights."""
        if self._session is not None:
            return
        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise NliUnavailableError(
                "the optional NLI extra is not installed: "
                'pip install -e ".[nli]" (onnxruntime, tokenizers)'
            ) from exc
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise NliUnavailableError(
                'the optional NLI extra is not installed: pip install -e ".[nli]"'
            ) from exc

        self._require_file(self.model_path)
        self._require_file(self.tokenizer_path)
        self._require_file(self.config_path)

        if self.expected_sha256:
            actual = self._sha256()
            if actual != self.expected_sha256:
                raise NliUnavailableError(
                    f"{self.model_path} does not match the pinned sha256 "
                    f"(expected {self.expected_sha256}, got {actual}): re-run "
                    "scripts/fetch-nli-model.sh"
                )

        self._id_to_label = self._read_labels()
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        tokenizer.enable_truncation(max_length=MAX_PAIR_TOKENS)
        tokenizer.enable_padding()
        self._tokenizer = tokenizer
        self._session = onnxruntime.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )

    def _read_labels(self) -> dict[int, str]:
        """Read ``id2label`` and refuse a model whose label order we do not know."""
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NliUnavailableError(f"{self.config_path} is not readable JSON") from exc
        raw = config.get("id2label")
        if not isinstance(raw, dict):
            raise NliUnavailableError(f"{self.config_path} declares no id2label mapping")
        mapping = {int(key): str(value) for key, value in raw.items()}
        if tuple(mapping[key] for key in sorted(mapping)) != LABEL_ORDER:
            raise NliUnavailableError(
                "the NLI model declares labels "
                f"{tuple(mapping[key] for key in sorted(mapping))!r}, expected {LABEL_ORDER!r}: "
                "a reordered model would invert every verdict"
            )
        return mapping

    # ------------------------------------------------------------------
    # Judging
    # ------------------------------------------------------------------

    @property
    def input_names(self) -> Iterable[str]:
        """Names of the model inputs (available after :meth:`load`)."""
        self.load()
        return [item.name for item in self._session.get_inputs()]

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> Sequence[Mapping[str, float]]:
        """Judge (premise, hypothesis) pairs in one batched run through the model.

        The unit of comparison is a *sentence pair* — what the model was trained
        on — so the rules split the source into sentences and hand the whole
        cross-product over in one call; ONNX Runtime runs the batch at once, which
        keeps a week's worth of pairs in the order of a second.
        """
        import numpy as np

        self.load()
        if not pairs:
            return []
        encodings = self._tokenizer.encode_batch([(premise, claim) for premise, claim in pairs])
        feeds: dict[str, Any] = {
            "input_ids": np.array([item.ids for item in encodings], dtype=np.int64),
            "attention_mask": np.array([item.attention_mask for item in encodings], dtype=np.int64),
        }
        for name in self.input_names:
            if name == "token_type_ids":
                feeds[name] = np.array([item.type_ids for item in encodings], dtype=np.int64)
        logits = self._session.run(None, feeds)[0]
        rows: list[dict[str, float]] = []
        for row in logits:
            shifted = np.exp(row - row.max())
            probabilities = shifted / shifted.sum()
            # The label order comes from the model's own config, validated at load.
            rows.append(
                {label: float(probabilities[index]) for index, label in self._id_to_label.items()}
            )
        return rows
