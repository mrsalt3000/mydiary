"""Unit tests for the CPU twins of the phase-2 Ollama tags (T13, role A1).

T13 asks a measurement question — "LLM on the GPU or on the CPU/RAM?" — and the
measurement lives in ``docs/device-choice.md``. This module locks the *assets*
the measurement introduced, so a later edit cannot silently change what was
measured:

* ``ollama/Modelfile.stt-cpu`` / ``.sum-cpu`` / ``.coach-cpu`` — one CPU twin
  per phase-2 tag. Compared with the GPU twin the file must differ in exactly
  one declared parameter: ``num_gpu 0`` (no layer on the GPU). ``num_ctx 8192``
  and ``temperature`` stay identical, because a comparison across devices that
  also changes the context or the sampler would not be a comparison.
* the GPU Modelfiles themselves are **not** touched by T13: the twins are new
  files with new tags (``diary-*-cpu``).
* ``configs/models.cpu.example.yaml`` — the tracked CPU variant of the example
  config. Its phase-2 profiles differ from ``configs/models.example.yaml`` only
  in ways the file documents (the CPU tag, and the transport ``timeout_s``
  headroom); temperature, ``max_tokens``, ``reasoning_effort`` and the four MVP
  profiles are identical.
* ``scripts/serve-ollama.sh create-phase2-cpu`` — the reproducible way to build
  the three CPU tags from those Modelfiles.

Hermetic: no GPU, no network, no ``ollama`` invocation, nothing is executed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from diary.core.settings import PROJECT_ROOT

MODELFILE_DIR = PROJECT_ROOT / "ollama"
SERVE_SCRIPT = PROJECT_ROOT / "scripts" / "serve-ollama.sh"
GPU_EXAMPLE = PROJECT_ROOT / "configs" / "models.example.yaml"
CPU_EXAMPLE = PROJECT_ROOT / "configs" / "models.cpu.example.yaml"

BASE_MODEL = "qwen3.5:4b"
NUM_CTX = "8192"

#: CPU tag → (paired CPU Modelfile, temperature it must declare). The
#: temperatures are the T02 decision (0.15 / 0.25 / 0.4) — identical to the GPU
#: twins on purpose (``docs/tasklist-phase2.md`` §5).
CPU_TAGS: dict[str, tuple[str, float]] = {
    "diary-stt-cpu": ("Modelfile.stt-cpu", 0.15),
    "diary-sum-cpu": ("Modelfile.sum-cpu", 0.25),
    "diary-coach-cpu": ("Modelfile.coach-cpu", 0.4),
}

#: GPU twin of every CPU tag (the file T13 must not change).
GPU_TWINS: dict[str, str] = {
    "diary-stt-cpu": "Modelfile.stt",
    "diary-sum-cpu": "Modelfile.sum",
    "diary-coach-cpu": "Modelfile.coach",
}

#: Phase-2 profile names of the example configs.
PHASE2_PROFILES = ("diary_stt", "diary_sum", "diary_coach")

#: Keys the CPU example is allowed to differ in — both named in its own header:
#: the model tag (``diary-*-cpu``) and the transport timeout (600 s of headroom
#: for a CPU call, ``docs/device-choice.md``).
ALLOWED_CPU_EXAMPLE_DIFFERENCES = frozenset({"model", "timeout_s"})

CPU_TAG_IDS = sorted(CPU_TAGS)


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


def declared_parameters(parsed: list[tuple[str, str]]) -> list[str]:
    """Parameter values from ``parsed``, e.g. ``["num_ctx 8192"]``."""
    return [value for keyword, value in parsed if keyword == "PARAMETER"]


def parameter(parsed: list[tuple[str, str]], name: str) -> list[str]:
    """Every declared value of the parameter ``name``."""
    return [value for value in declared_parameters(parsed) if value.split()[0] == name]


def cpu_twin(tag: str) -> list[tuple[str, str]]:
    path = MODELFILE_DIR / CPU_TAGS[tag][0]
    assert path.is_file(), f"missing {path}"
    return directives(path)


def load_example(path: Path) -> dict[str, object]:
    assert path.is_file(), f"missing {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The CPU Modelfiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tag", CPU_TAG_IDS)
def test_cpu_modelfile_exists(tag: str) -> None:
    path = MODELFILE_DIR / CPU_TAGS[tag][0]
    assert path.is_file(), f"missing {path}"


@pytest.mark.parametrize("tag", CPU_TAG_IDS)
def test_cpu_modelfile_selects_base_model_first(tag: str) -> None:
    keyword, value = cpu_twin(tag)[0]
    assert keyword == "FROM"
    assert value == BASE_MODEL


@pytest.mark.parametrize("tag", CPU_TAG_IDS)
def test_cpu_modelfile_keeps_num_ctx(tag: str) -> None:
    """Same context as the GPU twin: only the device may differ."""
    assert parameter(cpu_twin(tag), "num_ctx") == [f"num_ctx {NUM_CTX}"]
    assert parameter(directives(MODELFILE_DIR / GPU_TWINS[tag]), "num_ctx") == [
        f"num_ctx {NUM_CTX}"
    ]


@pytest.mark.parametrize("tag", CPU_TAG_IDS)
def test_cpu_modelfile_keeps_temperature(tag: str) -> None:
    expected = CPU_TAGS[tag][1]
    declared = parameter(cpu_twin(tag), "temperature")
    assert len(declared) == 1, f"{tag}: expected exactly one temperature, got {declared}"
    assert float(declared[0].split()[1]) == pytest.approx(expected)
    assert declared == parameter(directives(MODELFILE_DIR / GPU_TWINS[tag]), "temperature")


@pytest.mark.parametrize("tag", CPU_TAG_IDS)
def test_cpu_modelfile_declares_num_gpu_zero(tag: str) -> None:
    """``num_gpu 0`` is the whole point: no layer is offloaded to the GPU."""
    assert parameter(cpu_twin(tag), "num_gpu") == ["num_gpu 0"]


@pytest.mark.parametrize("tag", CPU_TAG_IDS)
def test_cpu_modelfile_differs_from_gpu_twin_only_in_num_gpu(tag: str) -> None:
    """A comparison across devices must not change anything else."""
    cpu = declared_parameters(cpu_twin(tag))
    gpu = declared_parameters(directives(MODELFILE_DIR / GPU_TWINS[tag]))
    assert sorted(cpu) == sorted([*gpu, "num_gpu 0"])


@pytest.mark.parametrize("tag", CPU_TAG_IDS)
def test_cpu_modelfile_declares_no_template_machinery(tag: str) -> None:
    keywords = {keyword for keyword, _ in cpu_twin(tag)}
    assert keywords <= {"FROM", "PARAMETER"}


# ---------------------------------------------------------------------------
# The CPU example config
# ---------------------------------------------------------------------------


def test_cpu_example_config_is_tracked_by_gitignore_pattern() -> None:
    """The file must match ``!configs/*.example.yaml``, or it cannot be committed.

    ``.gitignore`` ignores ``configs/*.yaml`` and re-includes
    ``configs/*.example.yaml``; a name such as ``models.cpu-example.yaml``
    would silently stay untracked (checked at T13).
    """
    assert CPU_EXAMPLE.name.endswith(".example.yaml")


def test_cpu_example_config_declares_phase2_profiles() -> None:
    config = load_example(CPU_EXAMPLE)
    profiles = config["profiles"]
    for name in PHASE2_PROFILES:
        assert name in profiles, f"{name} missing from {CPU_EXAMPLE.name}"


@pytest.mark.parametrize("name", PHASE2_PROFILES)
def test_cpu_example_uses_cpu_tags(name: str) -> None:
    cpu = load_example(CPU_EXAMPLE)["profiles"][name]
    gpu = load_example(GPU_EXAMPLE)["profiles"][name]
    assert cpu["model"] == f"{gpu['model']}-cpu"
    assert cpu["model"] in CPU_TAGS


@pytest.mark.parametrize("name", PHASE2_PROFILES)
def test_cpu_example_differs_only_in_documented_keys(name: str) -> None:
    """``model`` and ``timeout_s`` (documented); everything else identical."""
    cpu = load_example(CPU_EXAMPLE)["profiles"][name]
    gpu = load_example(GPU_EXAMPLE)["profiles"][name]
    differing = {key for key in set(cpu) | set(gpu) if cpu.get(key) != gpu.get(key)}
    assert differing <= ALLOWED_CPU_EXAMPLE_DIFFERENCES, f"{name}: {sorted(differing)}"
    assert cpu["temperature"] == gpu["temperature"]
    assert cpu["max_tokens"] == gpu["max_tokens"]
    assert cpu["reasoning_effort"] == gpu["reasoning_effort"]


def test_cpu_example_keeps_mvp_profiles_identical() -> None:
    cpu = load_example(CPU_EXAMPLE)["profiles"]
    gpu = load_example(GPU_EXAMPLE)["profiles"]
    for name in ("transcription", "normalization", "summarization", "wellbeing_analysis"):
        assert cpu[name] == gpu[name], f"MVP profile {name} was changed by T13"


# ---------------------------------------------------------------------------
# Reproducibility: the script that builds the CPU tags
# ---------------------------------------------------------------------------


def test_serve_script_exposes_create_phase2_cpu() -> None:
    text = SERVE_SCRIPT.read_text(encoding="utf-8")
    assert "create-phase2-cpu" in text
    assert "cmd_create_phase2_cpu" in text
    for tag, (modelfile, _) in CPU_TAGS.items():
        assert tag in text, f"{tag} is not listed in {SERVE_SCRIPT.name}"
        assert modelfile in text, f"{modelfile} is not listed in {SERVE_SCRIPT.name}"
