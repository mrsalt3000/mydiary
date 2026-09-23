"""Provisioning of the GigaAM sidecar runtime (T45).

Two shell scripts do the work — ``scripts/setup-gigaam-venv.sh`` (the isolated
interpreter, ``docs/stt-backends-analysis.md`` §10) and
``scripts/fetch-gigaam-model.sh`` (the pinned weights, outside the repository) —
and a shell script cannot import Python. So what is guarded here is exactly the
part the acceptance found worth guarding:

* the **pin agrees with the profile**: the revision the weights are fetched at is
  the revision both example configs hand to the provider. A pin living in two
  files that can drift apart silently is the class of defect that costs rounds on
  this project (T38, T43), and the configs carry the GigaAM block commented out,
  so the check reads the text rather than the parsed YAML;
* the **file list is the measured one**: every file the script downloads is
  pinned by byte size and ``sha256``, and the sizes add up to the 2 637 136 001 B
  the snapshot was measured at in T42 §4.5;
* ``--check`` **reports instead of downloading**: it names what is missing, creates
  nothing and needs no network;
* a file that is present but wrong is reported as a **mismatch**, not as missing —
  which is also what proves the script looks where the Hugging Face cache really
  keeps a snapshot (``…/snapshots/<revision>/…``);
* the venv script **refuses a prefix inside the repository** (the runtime lives
  outside it) and reports a missing interpreter instead of creating one;
* the worker **pins every load call**: it passes the pinned revision (and the
  provisioned cache) to every library call that can reach the Hub, so a load can
  never fall back to the repository's default revision. Measured 2026-09-24: with
  that missing, the dynamic-module loader asked for ``main``, which a cache
  provisioned at a pinned commit does not carry, and the load failed with
  ``LocalEntryNotFoundError`` while all 15 files were present and verified;
* both scripts are syntactically valid and executable.

Nothing here installs a package, downloads a weight or starts an interpreter: the
whole point of the provisioning design is that none of that happens implicitly.
The worker is exercised with **fake** ``transformers``/``torch``/``safetensors``
modules, so its load path can be checked in an environment that has none of them.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch-gigaam-model.sh"
VENV_SCRIPT = PROJECT_ROOT / "scripts" / "setup-gigaam-venv.sh"
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "gigaam-stt-worker.py"
EXAMPLE_CONFIGS = (
    PROJECT_ROOT / "configs" / "models.example.yaml",
    PROJECT_ROOT / "configs" / "models.cpu.example.yaml",
)
MODEL_ID = "waveletdeboshir/gigaam-ctc-with-lm"
#: The pinned revision of the model wrapper — the T42 measurement revision, and
#: the value both example configs put in ``revision:``.
REVISION = "07d97cf8a4f178b68679cafae0600c8487877301"
#: Large files with the byte size and sha256 measured in T42 §4.5 (2026-09-23).
PINNED_LARGE_FILES: dict[str, tuple[int, str]] = {
    "model.safetensors": (
        465343856,
        "c7bd7e277cb601bf55036251be654dd374c455313edefaa69d32e6ec1f9c7161",
    ),
    "language_model/ru_3gram.bin": (
        2142431571,
        "9c2a8143d33c234f881f0f7072bd0ed12c5dd2d697328410c317ed50892d70ee",
    ),
    "language_model/unigrams.txt": (
        29321792,
        "6c5edc3b6134e2a64b04696a1c00abccb660a86776b8016c43108f80f35d2fae",
    ),
}
#: What the whole snapshot weighs: sum of the per-file sizes, equal to the
#: measured ``du -sbL`` of the snapshot in T42 §4.5 (2.46 GiB).
SNAPSHOT_BYTES = 2637136001

_SNAPSHOT_SUFFIX = Path("hub") / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / REVISION


def _run(
    script: Path, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *arguments],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _fetch_check(target: Path) -> subprocess.CompletedProcess[str]:
    """Run ``fetch-gigaam-model.sh --check`` against ``target`` (no network)."""
    environment = {**os.environ, "APP_GIGAAM_MODEL_DIR": str(target)}
    return _run(FETCH_SCRIPT, "--check", env=environment)


def _revision_in_script() -> str:
    text = FETCH_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^REVISION=(\S+)$", text, flags=re.MULTILINE)
    assert match is not None, "scripts/fetch-gigaam-model.sh declares no REVISION="
    return match.group(1).strip('"')


def _revisions_in_configs() -> dict[str, str]:
    revisions: dict[str, str] = {}
    for config_path in EXAMPLE_CONFIGS:
        # The GigaAM block is commented out on purpose (T43), so the revision is
        # read from the text, not from a parsed mapping.
        match = re.search(r"revision:\s*([0-9a-f]{40})", config_path.read_text(encoding="utf-8"))
        assert match is not None, f"{config_path.name} carries no pinned revision"
        revisions[config_path.name] = match.group(1)
    return revisions


def _default_prefix_in_script(script: Path) -> str:
    match = re.search(
        r'^DEFAULT_PREFIX="([^"]+)"$', script.read_text(encoding="utf-8"), flags=re.MULTILINE
    )
    assert match is not None, f"{script.name} declares no DEFAULT_PREFIX="
    return match.group(1)


@pytest.mark.parametrize("script", [FETCH_SCRIPT, VENV_SCRIPT])
def test_the_provisioning_scripts_are_valid_and_executable(script: Path) -> None:
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
    assert os.access(script, os.X_OK), f"{script} is not executable"


def test_the_two_scripts_agree_on_the_provisioning_prefix() -> None:
    """The weights script looks for the interpreter exactly where the venv script puts it."""
    assert _default_prefix_in_script(FETCH_SCRIPT) == _default_prefix_in_script(VENV_SCRIPT)


def test_the_weight_pin_is_the_revision_the_profiles_use() -> None:
    """One revision, three files: the script's pin and both example configs."""
    assert _revision_in_script() == REVISION
    assert set(_revisions_in_configs().values()) == {REVISION}


def test_the_script_pins_the_documented_weight_files() -> None:
    """Every downloaded file carries a size and a hash; the large ones are T42's."""
    text = FETCH_SCRIPT.read_text(encoding="utf-8")
    for name, (size, sha) in PINNED_LARGE_FILES.items():
        assert f'"{name}|{size}|{sha}"' in text, f"{name} is not pinned as measured in T42 §4.5"

    sizes = [int(match) for match in re.findall(r'^\s+"[^"|]+\|(\d+)\|', text, flags=re.MULTILINE)]
    assert sum(sizes) == SNAPSHOT_BYTES, "the pinned file list is not the measured snapshot"


def test_check_reports_missing_weights_without_writing_anything(tmp_path: Path) -> None:
    """``--check`` is the honest half of provisioning: it names what is absent."""
    target = tmp_path / "weights"

    result = _fetch_check(target)

    assert result.returncode == 1
    assert "нет: model.safetensors" in result.stdout
    assert "scripts/fetch-gigaam-model.sh" in result.stderr
    assert not target.exists()


def test_check_reports_a_present_but_wrong_file_as_a_mismatch(tmp_path: Path) -> None:
    """Found-and-wrong is not missing: this is what pins the snapshot layout."""
    target = tmp_path / "weights"
    snapshot = target / _SNAPSHOT_SUFFIX
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")

    result = _fetch_check(target)

    assert result.returncode == 1
    assert "нет: config.json" not in result.stdout  # it was found
    assert "sha256 не совпал" in result.stderr
    assert "config.json" in result.stderr


def test_the_venv_script_refuses_a_prefix_inside_the_repository() -> None:
    """The sidecar's venv never lives in the repository (§10 of the analysis)."""
    inside = PROJECT_ROOT / "data" / "models" / "gigaam-venv"

    result = _run(VENV_SCRIPT, "--prefix", str(inside))

    assert result.returncode == 2
    assert "внутри репозитория" in result.stderr
    assert not inside.exists()


def test_the_venv_check_reports_a_missing_interpreter(tmp_path: Path) -> None:
    """``--check`` says how to build it; it does not build anything by surprise."""
    prefix = tmp_path / "absent"

    result = _run(VENV_SCRIPT, "--check", "--prefix", str(prefix))

    assert result.returncode == 1
    assert "scripts/setup-gigaam-venv.sh" in result.stderr
    assert not prefix.exists()


def test_the_weight_script_needs_the_sidecar_interpreter(tmp_path: Path) -> None:
    """The downloader runs in the sidecar venv (T45): without it, say so and stop.

    No interpreter — no download attempt, no half-populated cache: the failure names
    the script that builds one. Measured reason for using that interpreter at all —
    the reader is ``huggingface_hub``, so the writer has to be the same library
    (``AutoProcessor`` loads the language model through ``snapshot_download``, which
    needs the cache the library itself wrote).
    """
    target = tmp_path / "weights"
    environment = {
        **os.environ,
        "APP_GIGAAM_MODEL_DIR": str(target),
        "APP_GIGAAM_PYTHON": str(tmp_path / "absent-venv" / "bin" / "python"),
    }

    result = _run(FETCH_SCRIPT, env=environment)

    assert result.returncode == 2
    assert "scripts/setup-gigaam-venv.sh" in result.stderr
    assert not target.exists()


# ---------------------------------------------------------------------------
# The worker's own load calls must carry the pin
# ---------------------------------------------------------------------------


class _FakeStateDictReport:
    unexpected_keys: list[str] = []
    missing_keys: list[str] = []


class _FakeModel:
    """A model that records the dtype it was cast to and never touches a GPU."""

    def __init__(self, config: object, **kwargs: object) -> None:
        # A real remote-code model class takes the config and nothing else; extra
        # keyword arguments raise, which is what makes ``revision=`` in
        # ``from_config`` a live-only failure (measured 2026-09-24).
        if kwargs:
            raise TypeError(
                f"GigaAMCTCHF.__init__() got an unexpected keyword argument {next(iter(kwargs))!r}"
            )
        self.config = config

    def load_state_dict(self, weights: object, strict: bool = False) -> _FakeStateDictReport:
        return _FakeStateDictReport()

    def to(self, dtype: object) -> _FakeModel:
        self.dtype = dtype
        return self


def _install_fake_stack(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, dict]]) -> None:
    """Fake ``transformers``/``torch``/``safetensors``/``huggingface_hub`` modules.

    The fakes mimic the two behaviours that matter here: ``from_config`` consumes
    ``code_revision`` for the module lookup and forwards everything else to the
    model constructor (so ``revision=`` raises), and every call that could reach
    the Hub is recorded with its keyword arguments.
    """

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            calls.append(("AutoConfig.from_pretrained", dict(kwargs)))
            return object()

    class AutoModel:
        @staticmethod
        def from_config(config: object, **kwargs: object) -> _FakeModel:
            calls.append(("AutoModel.from_config", dict(kwargs)))
            kwargs.pop("trust_remote_code", None)  # transformers consumes both of these
            kwargs.pop("code_revision", None)
            return _FakeModel(config, **kwargs)

    class _FromPretrained:
        def __init__(self, name: str) -> None:
            self._name = name

        def from_pretrained(self, *args: object, **kwargs: object) -> object:
            calls.append((f"{self._name}.from_pretrained", dict(kwargs)))
            return object()

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = AutoConfig  # type: ignore[attr-defined]
    transformers.AutoModel = AutoModel  # type: ignore[attr-defined]
    transformers.AutoTokenizer = _FromPretrained("AutoTokenizer")  # type: ignore[attr-defined]
    transformers.AutoFeatureExtractor = _FromPretrained("AutoFeatureExtractor")  # type: ignore[attr-defined]
    transformers.AutoProcessor = _FromPretrained("AutoProcessor")  # type: ignore[attr-defined]

    def _hf_hub_download(*args: object, **kwargs: object) -> str:
        calls.append(("hf_hub_download", dict(kwargs)))
        return "/dev/null/model.safetensors"

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.hf_hub_download = _hf_hub_download  # type: ignore[attr-defined]

    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda path: {}  # type: ignore[attr-defined]
    safetensors = types.ModuleType("safetensors")
    safetensors.torch = safetensors_torch  # type: ignore[attr-defined]

    torch = types.ModuleType("torch")
    torch.float16 = "float16"  # type: ignore[attr-defined]

    for name, module in (
        ("transformers", transformers),
        ("huggingface_hub", huggingface_hub),
        ("safetensors", safetensors),
        ("safetensors.torch", safetensors_torch),
        ("torch", torch),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _load_worker() -> types.ModuleType:
    """Load ``scripts/gigaam-stt-worker.py`` by path (a dash keeps it out of packages)."""
    spec = importlib.util.spec_from_file_location("t45_worker_under_test", WORKER_SCRIPT)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    return worker


def test_the_worker_pins_every_load_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No load call may fall back to the repository's default revision.

    A cache provisioned at a pinned commit has no ``refs/main`` — so a call that
    asks for ``main`` fails offline even though every file is there (measured
    2026-09-24, ``LocalEntryNotFoundError`` → "runtime is not provisioned"). This
    test fails if any of the worker's calls loses its revision.
    """
    calls: list[tuple[str, dict]] = []
    _install_fake_stack(monkeypatch, calls)
    worker = _load_worker()
    args = worker.build_parser().parse_args(
        [
            str(tmp_path / "audio.webm"),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--device",
            "cuda",
            "--cache-dir",
            str(tmp_path / "weights" / "hub"),
        ]
    )

    worker.load_model(args)
    worker.load_frontend(args, with_lm=True)

    # The code fetch is pinned through ``code_revision`` (``revision=`` would be
    # forwarded to the model constructor and raise), everything else through
    # ``revision`` — and every call names the provisioned cache, except the code
    # fetch, which cannot: ``from_config`` forwards ``cache_dir`` to the model
    # constructor, and the child's ``HF_HOME`` *is* the default ``HF_HUB_CACHE``.
    assert [name for name, _ in calls] == [
        "AutoConfig.from_pretrained",
        "AutoModel.from_config",
        "hf_hub_download",
        "AutoProcessor.from_pretrained",
    ]
    assert calls[1][1] == {"trust_remote_code": True, "code_revision": REVISION}
    for name, kwargs in calls:
        assert kwargs.get("revision") or kwargs.get("code_revision") == REVISION, name
        if name == "AutoModel.from_config":
            assert "cache_dir" not in kwargs, name
        else:
            assert str(kwargs.get("cache_dir", "")) == str(tmp_path / "weights" / "hub"), name
