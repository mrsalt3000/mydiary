"""Unit tests for the Ollama runtime assets (phase 2, T01 and T02, role A1).

The runtime artifacts are text/repo-level, so these tests are hermetic: no GPU,
no network and no ``ollama`` invocation. They lock the parts of T01/T02 that
later tasks depend on:

* ``ollama/Modelfile`` — base model ``qwen3.5:4b`` and exactly one documented
  ``PARAMETER`` (``num_ctx 8192``). ``num_ctx`` lives in the Modelfile and
  nowhere else, because the OpenAI-compatible Ollama API does not accept it
  (docs/tasklist-phase2.md §5, §9.9). Temperature is deliberately absent — that
  is T02.
* ``ollama/Modelfile.stt`` / ``.sum`` / ``.coach`` (T02) — one tag per phase-2
  layer: same base model, one ``num_ctx 8192`` shared by all three, and exactly
  one ``temperature`` each, the three values distinct and equal to the decided
  numbers (0.15 / 0.25 / 0.4). ``presence_penalty``/``top_k``/``top_p`` and the
  template machinery are inherited from the base model and must not be
  re-declared here (T02 decision, docs/tasklist-phase2.md §9).
* ``scripts/serve-ollama.sh`` — exists, is executable, and parses as bash.

The YAML ⇄ Modelfile agreement (temperature must match the phase-2 profile in
``configs/models.example.yaml``) is asserted in
``tests/unit/test_models_config.py``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diary.core.settings import PROJECT_ROOT

MODELFILE = PROJECT_ROOT / "ollama" / "Modelfile"
MODELFILE_DIR = PROJECT_ROOT / "ollama"
SERVE_SCRIPT = PROJECT_ROOT / "scripts" / "serve-ollama.sh"

BASE_MODEL = "qwen3.5:4b"
NUM_CTX = "8192"

#: Phase-2 profile tags (T02): tag → (paired Modelfile name, decided temperature).
#: The numbers come from the T02 decision; the brief ranges (0.1–0.2 / 0.2–0.3 /
#: 0.4) are the bounds of the choice, one number each lands in the artifacts.
PHASE2_TAGS: dict[str, tuple[str, float]] = {
    "diary-stt": ("Modelfile.stt", 0.15),
    "diary-sum": ("Modelfile.sum", 0.25),
    "diary-coach": ("Modelfile.coach", 0.4),
}

#: Parameters documented for a Modelfile (docs.ollama.com/modelfile). Server
#: variables such as ``flash_attn``, ``num_gpu`` or ``f16_kv`` are *not*
#: Modelfile parameters and must not appear here.
DOCUMENTED_PARAMETERS = frozenset(
    {
        "num_ctx",
        "repeat_last_n",
        "repeat_penalty",
        "temperature",
        "seed",
        "stop",
        "num_predict",
        "top_k",
        "top_p",
        "min_p",
    }
)


def directives(path: Path) -> list[tuple[str, str]]:
    """Return ``(KEYWORD, value)`` for every meaningful Modelfile line."""
    parsed: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, value = line.partition(" ")
        parsed.append((keyword.upper(), value.strip()))
    return parsed


def parameters(parsed: list[tuple[str, str]]) -> list[str]:
    """Parameter values from ``parsed``, e.g. ``["num_ctx 8192"]``."""
    return [value for keyword, value in parsed if keyword == "PARAMETER"]


@pytest.fixture(scope="module")
def modelfile() -> list[tuple[str, str]]:
    assert MODELFILE.is_file(), f"missing {MODELFILE}"
    return directives(MODELFILE)


def test_modelfile_is_present() -> None:
    assert MODELFILE.is_file(), f"missing {MODELFILE}"


def test_first_directive_selects_base_model(modelfile: list[tuple[str, str]]) -> None:
    keyword, value = modelfile[0]
    assert keyword == "FROM"
    assert value == BASE_MODEL


def test_num_ctx_is_declared_exactly_once(modelfile: list[tuple[str, str]]) -> None:
    declared = parameters(modelfile)
    assert declared.count(f"num_ctx {NUM_CTX}") == 1
    assert [value for value in declared if value.startswith("num_ctx ")] == [f"num_ctx {NUM_CTX}"]


def test_only_documented_parameters_are_used(modelfile: list[tuple[str, str]]) -> None:
    used = {value.split()[0] for value in parameters(modelfile)}
    assert used, "Modelfile declares no PARAMETER — num_ctx is expected"
    assert used <= DOCUMENTED_PARAMETERS


def test_serve_script_is_present_and_executable() -> None:
    assert SERVE_SCRIPT.is_file(), f"missing {SERVE_SCRIPT}"
    assert SERVE_SCRIPT.stat().st_mode & 0o111, f"{SERVE_SCRIPT} is not executable"


def test_serve_script_parses_as_bash() -> None:
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - bash is the target environment
        pytest.skip("bash is not available")
    result = subprocess.run(
        [bash, "-n", str(SERVE_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_serve_script_documents_create_phase2() -> None:
    """The usage block lists the T02 command and its single-tag form."""
    text = SERVE_SCRIPT.read_text(encoding="utf-8")
    assert "create-phase2" in text
    assert "create <tag> [<modelfile>]" in text


# ---------------------------------------------------------------------------
# T02 — three phase-2 profile tags
# ---------------------------------------------------------------------------

PHASE2_IDS = sorted(PHASE2_TAGS)


def _phase2_directives(tag: str) -> list[tuple[str, str]]:
    modelfile = MODELFILE_DIR / PHASE2_TAGS[tag][0]
    assert modelfile.is_file(), f"missing {modelfile}"
    return directives(modelfile)


def _declared_parameter(parsed: list[tuple[str, str]], name: str) -> list[str]:
    return [value for value in parameters(parsed) if value.split()[0] == name]


@pytest.mark.parametrize("tag", PHASE2_IDS)
def test_phase2_modelfile_exists(tag: str) -> None:
    path = MODELFILE_DIR / PHASE2_TAGS[tag][0]
    assert path.is_file(), f"missing {path}"


@pytest.mark.parametrize("tag", PHASE2_IDS)
def test_phase2_modelfile_selects_base_model_first(tag: str) -> None:
    keyword, value = _phase2_directives(tag)[0]
    assert keyword == "FROM"
    assert value == BASE_MODEL


@pytest.mark.parametrize("tag", PHASE2_IDS)
def test_phase2_modelfile_num_ctx_is_declared_exactly_once(tag: str) -> None:
    declared = _declared_parameter(_phase2_directives(tag), "num_ctx")
    assert declared == [f"num_ctx {NUM_CTX}"]


@pytest.mark.parametrize("tag", PHASE2_IDS)
def test_phase2_modelfile_temperature_is_declared_exactly_once(tag: str) -> None:
    expected = PHASE2_TAGS[tag][1]
    declared = _declared_parameter(_phase2_directives(tag), "temperature")
    assert len(declared) == 1, f"{tag}: expected exactly one temperature, got {declared}"
    assert float(declared[0].split()[1]) == pytest.approx(expected)


def test_phase2_temperatures_are_distinct_and_expected() -> None:
    temperatures = {
        tag: float(_declared_parameter(_phase2_directives(tag), "temperature")[0].split()[1])
        for tag in PHASE2_IDS
    }
    assert len(set(temperatures.values())) == len(PHASE2_IDS)
    assert temperatures == {tag: expected for tag, (_, expected) in PHASE2_TAGS.items()}


@pytest.mark.parametrize("tag", PHASE2_IDS)
def test_phase2_modelfile_declares_only_documented_parameters(tag: str) -> None:
    """``num_ctx`` + ``temperature`` only: everything else is inherited.

    ``presence_penalty 1.5``, ``top_k 20`` and ``top_p 0.95`` come from the base
    model on purpose (T02 decision — the repeat penalty is a named risk, but it
    is not overridden here).
    """
    used = [value.split()[0] for value in parameters(_phase2_directives(tag))]
    assert used == ["num_ctx", "temperature"]
    assert set(used) <= DOCUMENTED_PARAMETERS


@pytest.mark.parametrize("tag", PHASE2_IDS)
def test_phase2_modelfile_declares_no_template_machinery(tag: str) -> None:
    """TEMPLATE/RENDERER/PARSER and other directives must be inherited."""
    keywords = {keyword for keyword, _ in _phase2_directives(tag)}
    assert keywords <= {"FROM", "PARAMETER"}
