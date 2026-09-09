"""Unit tests for the backup manifest and the data-root path helpers (Task 18).

Covers the reusable path-safety contract (AGENTS.md §14): lexical rejection
of absolute paths, ``..``/``.`` segments, backslashes, drive letters and NUL
bytes, and the canonical ``resolve()`` + ``is_relative_to`` checks that also
reject symlinked ancestors escaping the data root. Also covers manifest
serialization/parsing and structural validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from diary.domain.backup.manifest import (
    BackupManifest,
    ManifestEntry,
    UnsafePathError,
    is_safe_relative_path,
    iso_utc,
    manifest_problems,
    manifest_to_bytes,
    parse_manifest,
    resolve_data_path,
    sha256_file,
)

_ENTRY = {"path": "db/diary.db", "sha256": "a" * 64, "size": 5, "mtime": "2026-09-09T10:00:00Z"}


def _manifest(**overrides: object) -> BackupManifest:
    data: dict[str, object] = {
        "format": "local-diary-backup",
        "format_version": 1,
        "created_at": "2026-09-09T10:00:00+00:00",
        "mode": "text",
        "timezone": "Europe/Moscow",
        "db_included": True,
        "files": [],
    }
    data.update(overrides)
    return BackupManifest.model_validate(data)


# ---------------------------------------------------------------------------
# is_safe_relative_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "db/diary.db",
        "exports/01_original/2026-09-13_original_0001_r1.txt",
        "audio/2026/09/abc-123.webm",
        "a/b/c",
    ],
)
def test_safe_relative_paths_are_accepted(value: str) -> None:
    assert is_safe_relative_path(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/etc/passwd",
        "db/../escape.txt",
        "../escape.txt",
        "a/./b",
        "a//b",
        "a/b/",
        "a\\b",
        "..\\escape",
        "C:evil",
        "c:/windows",
        "a\x00b",
        "db/..",
        "..",
        ".",
    ],
)
def test_unsafe_relative_paths_are_rejected(value: str) -> None:
    assert not is_safe_relative_path(value)


# ---------------------------------------------------------------------------
# resolve_data_path
# ---------------------------------------------------------------------------


def test_resolve_data_path_joins_within_root(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    target = resolve_data_path(root, "exports/01_original/file.txt")
    assert target == (root / "exports" / "01_original" / "file.txt").resolve()
    assert target.is_relative_to(root.resolve())


@pytest.mark.parametrize(
    "value",
    ["../escape.txt", "/etc/passwd", "a/../../escape", "C:evil"],
)
def test_resolve_data_path_rejects_traversal(tmp_path: Path, value: str) -> None:
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(UnsafePathError):
        resolve_data_path(root, value)


def test_resolve_data_path_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink inside the data root pointing outside is a traversal vector."""
    root = tmp_path / "data"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside", encoding="utf-8")
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafePathError):
        resolve_data_path(root, "link/secret.txt")


def test_resolve_data_path_accepts_symlink_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    inner = root / "inner"
    inner.mkdir()
    (root / "link").symlink_to(inner, target_is_directory=True)
    target = resolve_data_path(root, "link/file.txt")
    assert target == (inner / "file.txt").resolve()
    assert target.is_relative_to(root.resolve())


def test_sha256_file_matches_read_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"0123456789" * 1000)
    import hashlib

    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Manifest model, parsing, structure
# ---------------------------------------------------------------------------


def test_manifest_roundtrip() -> None:
    manifest = _manifest(files=[ManifestEntry(**dict(_ENTRY))])
    parsed = parse_manifest(manifest_to_bytes(manifest))
    assert parsed == manifest
    assert parsed.files[0].path == "db/diary.db"


def test_manifest_rejects_unknown_format() -> None:
    with pytest.raises(ValidationError, match="unsupported manifest format"):
        _manifest(format="other")


def test_manifest_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError, match="unsupported backup mode"):
        _manifest(mode="everything")


def test_manifest_rejects_bad_created_at() -> None:
    with pytest.raises(ValidationError, match="created_at"):
        _manifest(created_at="not-a-date")


def test_manifest_entry_rejects_unsafe_path() -> None:
    with pytest.raises(ValidationError, match="relative and confined"):
        ManifestEntry(**{**_ENTRY, "path": "../escape.txt"})


def test_manifest_entry_rejects_bad_sha_and_size() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        ManifestEntry(**{**_ENTRY, "sha256": "xyz"})
    with pytest.raises(ValidationError, match="size"):
        ManifestEntry(**{**_ENTRY, "size": -1})


def test_manifest_problems_detect_duplicates() -> None:
    manifest = _manifest(files=[ManifestEntry(**dict(_ENTRY)), ManifestEntry(**dict(_ENTRY))])
    problems = manifest_problems(manifest)
    assert len(problems) == 1
    assert "duplicate path" in problems[0]
    assert manifest_problems(_manifest()) == []


def test_parse_manifest_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_manifest(b"{not json")


def test_iso_utc_produces_parseable_timestamps() -> None:
    assert iso_utc().endswith("+00:00") or "+00:00" in iso_utc()
