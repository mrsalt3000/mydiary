"""Unit tests for the ONNX NLI adapter's guard rails (T24a).

The adapter is the only place the project touches the optional ``nli`` extra, and
it must fail *loudly* wherever the instrument is unusable — that is the owner's
decision for T24 (an unavailable instrument is a failure, not a silent skip). The
loading path itself needs the 712 MB of weights and the extra, so it is exercised
by live runs whose measurements are recorded in
``docs/nli-validation-analysis.md``; what is provable here without either is:

* the pin: the weights' sha256 in the settings, in the adapter and in the fetch
  script are one value, and a mismatch is refused before the model is used;
* the label order: a model whose own ``config.json`` declares a different order is
  refused, because a reordered model would invert every verdict;
* the lazy import: importing the adapter must not import the extra, so CI (which
  installs neither ``onnxruntime`` nor ``tokenizers``) keeps working and a missing
  extra surfaces as a typed error at use time;
* the error type: everything above raises ``NliUnavailableError``, which the
  harness reports as a failed run.

All text and files are synthetic; no model is loaded and no network is used.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from diary.core.settings import PINNED_NLI_MODEL_SHA256
from diary.domain.nli.onnx_judge import (
    LABEL_ORDER,
    PINNED_MODEL_SHA256,
    NliUnavailableError,
    OnnxNliJudge,
)

FETCH_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch-nli-model.sh"


def _judge(tmp_path: Path, *, sha: str = "") -> OnnxNliJudge:
    return OnnxNliJudge(
        model_path=tmp_path / "model.onnx",
        tokenizer_path=tmp_path / "tokenizer.json",
        config_path=tmp_path / "config.json",
        expected_sha256=sha,
    )


def test_the_pin_is_one_value_across_the_settings_the_adapter_and_the_script() -> None:
    """Three copies of a hash is two too many; the test is what keeps them honest."""
    assert PINNED_MODEL_SHA256 == PINNED_NLI_MODEL_SHA256
    script = FETCH_SCRIPT.read_text(encoding="utf-8")
    assert f'SHA_FP32="{PINNED_MODEL_SHA256}"' in script
    # The lighter conversion is pinned too, so a swap of files cannot go unnoticed.
    assert 'SHA_FP16="e838a882102d328e74eef45b52700755ff63545cb39ad132b717c44ce714a48e"' in script


def test_importing_the_adapter_does_not_import_the_optional_extra() -> None:
    """CI installs neither package, and importing the module must still work.

    Checked in a fresh interpreter instead of by looking at this process: another
    test may legitimately have loaded the extra already, and that would make the
    property untestable rather than false.
    """
    code = (
        "import sys, diary.domain.nli.onnx_judge as module;"
        "print('onnxruntime' in sys.modules, 'tokenizers' in sys.modules);"
        "print(module.LABEL_ORDER)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    first, second = completed.stdout.strip().splitlines()
    assert first == "False False", completed.stdout
    assert second == str(LABEL_ORDER)


def test_missing_files_are_refused_with_the_fix_named(tmp_path: Path) -> None:
    judge = _judge(tmp_path)
    with pytest.raises(NliUnavailableError) as error:
        judge.load()
    message = str(error.value)
    # Either the extra or the weights: whichever is missing first, the message
    # names what to do about it.
    assert "fetch-nli-model.sh" in message or 'pip install -e ".[nli]"' in message


def test_the_settings_default_directory_is_where_the_fetch_script_writes() -> None:
    from diary.core.settings import AppSettings

    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.nli_model_dir.name == "nli"
    assert settings.nli_model_dir.parent.name == "models"
    assert settings.nli_model_sha256 == PINNED_MODEL_SHA256
    assert 'TARGET="${APP_NLI_MODEL_DIR:-$ROOT/data/models/nli}"' in FETCH_SCRIPT.read_text(
        encoding="utf-8"
    )


def test_a_different_label_order_is_refused(tmp_path: Path) -> None:
    """A reordered model would invert entailment and contradiction silently."""
    judge = _judge(tmp_path)
    judge._session = object()  # pretend the model is loaded; only labels are read
    (tmp_path / "config.json").write_text(
        json.dumps({"id2label": {"0": "neutral", "1": "entailment", "2": "contradiction"}}),
        encoding="utf-8",
    )
    with pytest.raises(NliUnavailableError, match="reordered"):
        judge._read_labels()


def test_the_expected_label_order_matches_the_rules() -> None:
    from diary.domain.contracts.nli import LABELS

    assert LABEL_ORDER == LABELS
