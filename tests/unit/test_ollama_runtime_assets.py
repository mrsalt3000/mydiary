"""Unit tests for the Ollama runtime assets (phase 2, T01, role A1).

The runtime artifacts are text/repo-level, so these tests are hermetic: no GPU,
no network and no ``ollama`` invocation. They lock the parts of T01 that later
tasks depend on:

* ``ollama/Modelfile`` — base model ``qwen3.5:4b`` and exactly one documented
  ``PARAMETER`` (``num_ctx 8192``). ``num_ctx`` lives in the Modelfile and
  nowhere else, because the OpenAI-compatible Ollama API does not accept it
  (docs/tasklist-phase2.md §5, §9.9). Temperature is deliberately absent — that
  is T02.
* ``scripts/serve-ollama.sh`` — exists, is executable, and parses as bash.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diary.core.settings import PROJECT_ROOT

MODELFILE = PROJECT_ROOT / "ollama" / "Modelfile"
SERVE_SCRIPT = PROJECT_ROOT / "scripts" / "serve-ollama.sh"

BASE_MODEL = "qwen3.5:4b"
NUM_CTX = "8192"

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
