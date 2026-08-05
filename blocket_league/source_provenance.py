"""Canonical, Git-independent source-tree sealing for Experiment F.

The manifest is the authority for executable provenance.  Git metadata is
deliberately excluded: ``.git`` is mutable and is not mounted into every
contained stage.  Instead we hash the complete Python package, experiment
scripts, tests, preregistration, and locked Python dependency inputs by
relative path, byte length, and file SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SOURCE_MANIFEST_KIND = "direct_experiment_f_source_tree"
SOURCE_MANIFEST_SCHEMA_VERSION = 1
_INCLUDED_DIRECTORIES = ("blocket_league", "scripts/mesohelios", "tests")
_INCLUDED_FILES = (
    "docs/direct_jacobian_poisson_ph_experiment.md",
    "pyproject.toml",
    "uv.lock",
)
_IGNORED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def default_source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(root: Path) -> tuple[Path, ...]:
    root = root.resolve(strict=True)
    selected: list[Path] = []
    for relative in _INCLUDED_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"source directory {relative!r} is missing or symbolic")
        for path in directory.rglob("*"):
            if any(part in _IGNORED_PARTS for part in path.relative_to(root).parts):
                continue
            if path.is_symlink():
                raise ValueError(
                    f"source manifest refuses symbolic path {path.relative_to(root)}"
                )
            if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
                selected.append(path)
    for relative in _INCLUDED_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"source file {relative!r} is missing or symbolic")
        selected.append(path)
    ordered = tuple(sorted(set(selected), key=lambda item: item.relative_to(root).as_posix()))
    if not ordered:
        raise ValueError("source manifest selected no files")
    return ordered


def build_source_manifest(root: Path | None = None) -> dict[str, Any]:
    """Hash the exact registered source tree without consulting Git."""

    source_root = default_source_root() if root is None else root.resolve(strict=True)
    files = []
    for path in _source_paths(source_root):
        relative = path.relative_to(source_root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    unsigned = {
        "kind": SOURCE_MANIFEST_KIND,
        "schemaVersion": SOURCE_MANIFEST_SCHEMA_VERSION,
        "files": files,
    }
    return {
        **unsigned,
        "treeSha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def validate_source_manifest_schema(manifest: Any) -> str:
    """Validate a standalone manifest and return its canonical tree digest."""

    if type(manifest) is not dict or set(manifest) != {
        "kind",
        "schemaVersion",
        "files",
        "treeSha256",
    }:
        raise ValueError("source manifest top-level schema is not exact")
    if (
        manifest["kind"] != SOURCE_MANIFEST_KIND
        or type(manifest["schemaVersion"]) is not int
        or manifest["schemaVersion"] != SOURCE_MANIFEST_SCHEMA_VERSION
        or type(manifest["files"]) is not list
        or not manifest["files"]
        or type(manifest["treeSha256"]) is not str
        or _SHA256.fullmatch(manifest["treeSha256"]) is None
    ):
        raise ValueError("source manifest identity is invalid")
    paths: list[str] = []
    for item in manifest["files"]:
        if type(item) is not dict or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("source manifest file schema is not exact")
        path = item["path"]
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
            or type(item["sha256"]) is not str
            or _SHA256.fullmatch(item["sha256"]) is None
        ):
            raise ValueError("source manifest file entry is invalid")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("source manifest paths are not sorted and unique")
    unsigned = {
        "kind": manifest["kind"],
        "schemaVersion": manifest["schemaVersion"],
        "files": manifest["files"],
    }
    observed = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if observed != manifest["treeSha256"]:
        raise ValueError("source manifest canonical tree SHA-256 mismatch")
    return observed


def verify_source_manifest(
    manifest: Any,
    root: Path | None = None,
) -> str:
    """Recompute the current registered tree and require exact equality."""

    expected_sha256 = validate_source_manifest_schema(manifest)
    observed = build_source_manifest(root)
    if observed != manifest:
        expected_files = {
            item["path"]: (item["bytes"], item["sha256"])
            for item in manifest["files"]
        }
        observed_files = {
            item["path"]: (item["bytes"], item["sha256"])
            for item in observed["files"]
        }
        changed = sorted(
            path
            for path in set(expected_files).union(observed_files)
            if expected_files.get(path) != observed_files.get(path)
        )
        preview = changed[:8]
        raise ValueError(
            "current source tree differs from sealed manifest at "
            f"{preview!r}" + (" ..." if len(changed) > len(preview) else "")
        )
    return expected_sha256


def load_source_manifest(path: Path, *, verify_current_tree: bool = True) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    validate_source_manifest_schema(raw)
    if verify_current_tree:
        verify_source_manifest(raw)
    return raw


def write_or_verify_source_manifest(path: Path, root: Path | None = None) -> dict[str, Any]:
    manifest = build_source_manifest(root)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        validate_source_manifest_schema(observed)
        if observed != manifest:
            raise ValueError("existing launch source manifest differs from current tree")
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=default_source_root())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        manifest = write_or_verify_source_manifest(args.manifest, args.repo_root)
    else:
        manifest = load_source_manifest(args.manifest, verify_current_tree=False)
        verify_source_manifest(manifest, args.repo_root)
    print(manifest["treeSha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_MANIFEST_KIND",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_source_manifest",
    "default_source_root",
    "load_source_manifest",
    "validate_source_manifest_schema",
    "verify_source_manifest",
    "write_or_verify_source_manifest",
]
