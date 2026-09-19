#!/usr/bin/env python3
"""Generate JSON Schema files for the phase-2 domain contracts (task T00).

The Pydantic models in ``src/diary/domain/contracts/`` are the source of truth:
each contract's JSON Schema is derived from it and written to
``schemas/<contract_name>.json``. The output is deterministic — models are
rendered with ``ensure_ascii=False, indent=2, sort_keys=True`` and a single
trailing newline — so a regenerated file is byte-for-byte identical unless a
contract actually changed, which makes the files reviewable in a pull request.

Usage::

    .venv/bin/python scripts/generate_contract_schemas.py            # write files
    .venv/bin/python scripts/generate_contract_schemas.py --check    # verify only

``--check`` writes nothing and exits with status 1 when a file is missing or
differs from the current models; ``make contracts`` regenerates, and the T00
test asserts the same equality.

Dependencies: standard library plus Pydantic only — no additional package and
no external schema validator is involved (AGENTS.md §10).
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

# Running this file directly puts ``scripts/`` (not the repository root) on
# sys.path, so the import below is absolute and relies on the installed
# package. It keeps the script runnable as ``python scripts/<name>.py`` from any
# directory.
from diary.core.settings import PROJECT_ROOT
from diary.domain.contracts.coach_note import CoachNote
from diary.domain.contracts.daily_summary import DailySummary
from diary.domain.contracts.raw_entry import RawEntry
from diary.domain.contracts.stt_corrected import SttCorrection
from diary.domain.contracts.weekly_summary import WeeklySummary

#: Directory holding the generated JSON Schema files.
SCHEMAS_DIR = PROJECT_ROOT / "schemas"

#: Contract name → the model its schema is generated from, in the order the
#: files are reported.
CONTRACT_MODELS: Mapping[str, type[BaseModel]] = {
    RawEntry.contract_name: RawEntry,
    SttCorrection.contract_name: SttCorrection,
    DailySummary.contract_name: DailySummary,
    WeeklySummary.contract_name: WeeklySummary,
    CoachNote.contract_name: CoachNote,
}


def render_schema(model: type[BaseModel]) -> str:
    """Deterministic JSON Schema text of one contract model."""
    schema = model.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def schema_path(contract_name: str, schemas_dir: Path = SCHEMAS_DIR) -> Path:
    """Path of the generated schema file of ``contract_name``."""
    return schemas_dir / f"{contract_name}.json"


def _display(path: Path) -> str:
    """Path relative to the repository when possible, absolute otherwise."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _render_all(schemas_dir: Path) -> dict[Path, str]:
    return {
        schema_path(contract_name, schemas_dir): render_schema(model)
        for contract_name, model in CONTRACT_MODELS.items()
    }


def _first_difference(expected: str, actual: str) -> str:
    diff = difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile="on disk",
        tofile="regenerated",
        n=2,
    )
    return "".join(list(diff)[:40])


def write_schemas(schemas_dir: Path = SCHEMAS_DIR) -> list[Path]:
    """Write every schema file; return the paths written (sorted)."""
    schemas_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path, content in sorted(_render_all(schemas_dir).items()):
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def check_schemas(schemas_dir: Path = SCHEMAS_DIR) -> list[str]:
    """Return one problem description per file that is missing or stale."""
    problems: list[str] = []
    for path, expected in sorted(_render_all(schemas_dir).items()):
        if not path.is_file():
            problems.append(f"missing: {_display(path)}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            problems.append(f"stale: {_display(path)}\n{_first_difference(expected, actual)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the files are up to date without writing them",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCHEMAS_DIR,
        help=f"directory for the generated files (default: {SCHEMAS_DIR})",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = check_schemas(args.output_dir)
        if problems:
            print("contract schemas are out of date; run `make contracts`:", file=sys.stderr)
            for problem in problems:
                print(f"- {problem}", file=sys.stderr)
            return 1
        print(f"contract schemas are up to date ({len(CONTRACT_MODELS)} files)")
        return 0

    for path in write_schemas(args.output_dir):
        print(f"wrote {_display(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
