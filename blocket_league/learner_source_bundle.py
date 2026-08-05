"""Build and attest the exact simulator-free Experiment F learner bundle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

from .source_provenance import (
    load_source_manifest,
    validate_source_manifest_schema,
    verify_source_manifest,
)


LEARNER_BUNDLE_KIND = "direct_experiment_f_learner_source_bundle"
LEARNER_BUNDLE_SCHEMA_VERSION = 1
LEARNER_ENTRY_MODULE = "direct_distributed_training"
LEARNER_MODULE_ALLOWLIST = frozenset(
    {
        "action_free_latent_effort",
        "cotangent_jacobian_ports",
        "direct_action_free_data",
        "direct_activation_lens",
        "direct_cotangent_bridge",
        "direct_distributed_training",
        "direct_experiment_training",
        "direct_jacobian_port_extractor",
        "direct_jacobian_port_precompute",
        "direct_ph_ablation_cores",
        "direct_pixels_io",
        "direct_poisson_ph",
        "direct_unstructured_postfreeze",
        "direct_unstructured_training",
        "direct_unstructured_world_model",
        "direct_visual_poisson_ph",
        "experiment_f_contract",
        "factorized_transformer",
        "latent_patch_renderer",
        "learner_source_bundle",
        "pixel_direct_model",
        "pixel_palette",
        "runtime_firewall_trace",
        "source_provenance",
        "tensor_provenance",
    }
)
FORBIDDEN_LEARNER_MODULES = frozenset(
    {
        "action_free_excitation",
        "action_port_pixel_experiment",
        "data",
        "direct_experiment_f_producer",
        "direct_jacobian_poisson_ph_experiment",
        "end_to_end_ph_experiment",
        "env",
        "passive_control_systems",
        "passive_jacobian_ph_model",
        "pixel_only_ph_experiment",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_NAME = "learner-source-manifest.json"
_FULL_MANIFEST_NAME = "source-manifest.json"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular_file(path: Path) -> bytes:
    """Read one same-inode regular file without following symbolic links."""

    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"bundle source must be a regular file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError(f"bundle source changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        value = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino, opened.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or len(value) != after.st_size
        ):
            raise ValueError(f"bundle source changed while reading: {path}")
        return value
    finally:
        os.close(descriptor)


def _module_path(source_root: Path, module: str) -> Path:
    return source_root / "blocket_league" / (module.replace(".", "/") + ".py")


def _resolve_relative_import(
    current_module: str, level: int, imported: str | None
) -> str:
    package = current_module.rsplit(".", 1)[0] if "." in current_module else ""
    pieces = package.split(".") if package else []
    upward = level - 1
    if upward > len(pieces):
        raise ValueError(f"relative import escapes package in {current_module}")
    base = pieces[: len(pieces) - upward] if upward else pieces
    suffix = imported.split(".") if imported else []
    return ".".join((*base, *suffix))


def _local_imports(module: str, source: bytes) -> frozenset[str]:
    try:
        tree = ast.parse(source, filename=f"blocket_league/{module}.py")
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError(f"learner module {module!r} is not valid Python") from error
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id == "__import__":
                raise ValueError(f"dynamic import is forbidden in learner module {module}")
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "import_module"
                and isinstance(function.value, ast.Name)
                and function.value.id == "importlib"
            ):
                raise ValueError(f"dynamic import is forbidden in learner module {module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "blocket_league":
                    raise ValueError(
                        f"package-wide import is forbidden in learner module {module}"
                    )
                if alias.name.startswith("blocket_league."):
                    imports.add(alias.name.removeprefix("blocket_league."))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _resolve_relative_import(module, node.level, node.module)
                if node.module:
                    imports.add(base)
                else:
                    imports.update(
                        f"{base}.{alias.name}" if base else alias.name
                        for alias in node.names
                    )
            elif node.module == "blocket_league":
                imports.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith("blocket_league."):
                imports.add(node.module.removeprefix("blocket_league."))
    return frozenset(imports)


def learner_import_closure(
    source_root: Path,
    *,
    entry_module: str = LEARNER_ENTRY_MODULE,
) -> frozenset[str]:
    package_root = source_root / "blocket_league"
    if not package_root.is_dir() or package_root.is_symlink():
        raise ValueError("learner package root is missing or symbolic")
    closure: set[str] = set()
    pending = [entry_module]
    while pending:
        module = pending.pop()
        if module in closure:
            continue
        path = _module_path(source_root, module)
        source = _read_regular_file(path)
        closure.add(module)
        for imported in _local_imports(module, source):
            if imported in FORBIDDEN_LEARNER_MODULES:
                raise ValueError(f"forbidden learner import: blocket_league.{imported}")
            if _module_path(source_root, imported).is_file():
                pending.append(imported)
    return frozenset(closure)


def _validate_allowlisted_closure(source_root: Path) -> None:
    closure = learner_import_closure(source_root)
    if closure != LEARNER_MODULE_ALLOWLIST:
        missing = sorted(LEARNER_MODULE_ALLOWLIST - closure)
        unexpected = sorted(closure - LEARNER_MODULE_ALLOWLIST)
        raise ValueError(
            "learner AST closure differs from reviewed allowlist; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def build_learner_source_manifest(
    source_root: Path,
    full_source_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Create a canonical manifest and in-memory payload for the reviewed closure."""

    source_root = source_root.resolve(strict=True)
    full_tree_sha256 = verify_source_manifest(full_source_manifest, source_root)
    _validate_allowlisted_closure(source_root)
    payload: dict[str, bytes] = {
        "blocket_league/__init__.py": _read_regular_file(
            source_root / "blocket_league" / "__init__.py"
        ),
        _FULL_MANIFEST_NAME: json.dumps(
            dict(full_source_manifest), indent=2, allow_nan=False
        ).encode("utf-8"),
    }
    for module in sorted(LEARNER_MODULE_ALLOWLIST):
        relative = f"blocket_league/{module.replace('.', '/')}.py"
        payload[relative] = _read_regular_file(source_root / relative)
    files = [
        {"path": path, "bytes": len(value), "sha256": _sha256(value)}
        for path, value in sorted(payload.items())
    ]
    unsigned = {
        "kind": LEARNER_BUNDLE_KIND,
        "schemaVersion": LEARNER_BUNDLE_SCHEMA_VERSION,
        "fullSourceTreeSha256": full_tree_sha256,
        "entryModule": LEARNER_ENTRY_MODULE,
        "modules": sorted(LEARNER_MODULE_ALLOWLIST),
        "files": files,
    }
    return {
        **unsigned,
        "treeSha256": _sha256(_canonical_json(unsigned)),
    }, payload


def validate_learner_source_manifest(manifest: Any) -> str:
    keys = {
        "kind",
        "schemaVersion",
        "fullSourceTreeSha256",
        "entryModule",
        "modules",
        "files",
        "treeSha256",
    }
    if type(manifest) is not dict or set(manifest) != keys:
        raise ValueError("learner source manifest top-level schema is not exact")
    if (
        manifest["kind"] != LEARNER_BUNDLE_KIND
        or manifest["schemaVersion"] != LEARNER_BUNDLE_SCHEMA_VERSION
        or type(manifest["schemaVersion"]) is not int
        or type(manifest["fullSourceTreeSha256"]) is not str
        or _SHA256.fullmatch(manifest["fullSourceTreeSha256"]) is None
        or manifest["entryModule"] != LEARNER_ENTRY_MODULE
        or manifest["modules"] != sorted(LEARNER_MODULE_ALLOWLIST)
        or type(manifest["treeSha256"]) is not str
        or _SHA256.fullmatch(manifest["treeSha256"]) is None
    ):
        raise ValueError("learner source manifest identity is invalid")
    files = manifest["files"]
    if type(files) is not list or not files:
        raise ValueError("learner source manifest has no file table")
    paths: list[str] = []
    for item in files:
        if type(item) is not dict or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("learner source file schema is not exact")
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
            raise ValueError("learner source file entry is invalid")
        paths.append(path)
    expected_paths = {
        "blocket_league/__init__.py",
        _FULL_MANIFEST_NAME,
        *(
            f"blocket_league/{module.replace('.', '/')}.py"
            for module in LEARNER_MODULE_ALLOWLIST
        ),
    }
    if paths != sorted(paths) or len(paths) != len(set(paths)) or set(paths) != expected_paths:
        raise ValueError("learner source file table is not exact, sorted, and unique")
    unsigned = {key: manifest[key] for key in keys - {"treeSha256"}}
    observed = _sha256(_canonical_json(unsigned))
    if observed != manifest["treeSha256"]:
        raise ValueError("learner source canonical tree SHA-256 mismatch")
    return observed


def verify_learner_source_bundle(
    bundle_root: Path,
    *,
    expected_full_source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify manifest, every byte, exact contents, and the AST closure."""

    if bundle_root.is_symlink():
        raise ValueError("learner bundle root is missing or symbolic")
    bundle_root = bundle_root.resolve(strict=True)
    if not bundle_root.is_dir():
        raise ValueError("learner bundle root is missing or symbolic")
    manifest_path = bundle_root / _MANIFEST_NAME
    try:
        manifest = json.loads(_read_regular_file(manifest_path))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("learner source manifest is not valid JSON") from error
    validate_learner_source_manifest(manifest)
    if (
        expected_full_source_tree_sha256 is not None
        and manifest["fullSourceTreeSha256"] != expected_full_source_tree_sha256
    ):
        raise ValueError("learner bundle is anchored to another full source tree")
    expected = {
        item["path"]: (item["bytes"], item["sha256"])
        for item in manifest["files"]
    }
    observed_paths: set[str] = set()
    observed_directories: set[str] = set()
    for path in bundle_root.rglob("*"):
        relative = path.relative_to(bundle_root).as_posix()
        if path.is_symlink():
            raise ValueError(f"learner bundle contains symbolic path {relative!r}")
        if path.is_file():
            observed_paths.add(relative)
        elif path.is_dir():
            observed_directories.add(relative)
        else:
            raise ValueError(f"learner bundle contains special path {relative!r}")
    if observed_paths != set(expected) | {_MANIFEST_NAME}:
        extra = sorted(observed_paths - set(expected) - {_MANIFEST_NAME})
        missing = sorted(set(expected) - observed_paths)
        raise ValueError(
            f"learner bundle contents are not exact; extra={extra!r}, missing={missing!r}"
        )
    expected_directories = {
        parent.as_posix()
        for relative in expected
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    expected_directories.add("blocket_league")
    if observed_directories != expected_directories:
        extra = sorted(observed_directories - expected_directories)
        missing = sorted(expected_directories - observed_directories)
        raise ValueError(
            "learner bundle directory contents are not exact; "
            f"extra={extra!r}, missing={missing!r}"
        )
    for relative, (expected_bytes, expected_sha256) in expected.items():
        value = _read_regular_file(bundle_root / relative)
        if len(value) != expected_bytes or _sha256(value) != expected_sha256:
            raise ValueError(f"learner bundle file hash mismatch: {relative!r}")
    full_manifest = json.loads(_read_regular_file(bundle_root / _FULL_MANIFEST_NAME))
    if (
        validate_source_manifest_schema(full_manifest)
        != manifest["fullSourceTreeSha256"]
    ):
        raise ValueError("embedded full source manifest anchor is invalid")
    _validate_allowlisted_closure(bundle_root)
    return manifest


def validate_code_free_cache(cache_root: Path) -> dict[str, Any]:
    """Reject cache paths capable of injecting Python learner source."""

    if cache_root.is_symlink():
        raise ValueError("learner cache root cannot be symbolic")
    cache_root = cache_root.resolve(strict=True)
    if not cache_root.is_dir():
        raise ValueError("learner cache root must be a directory")
    entries: list[dict[str, Any]] = []
    forbidden: list[str] = []
    symbolic: list[str] = []
    special: list[str] = []
    for path in sorted(
        cache_root.rglob("*"),
        key=lambda item: item.relative_to(cache_root).as_posix(),
    ):
        relative = path.relative_to(cache_root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            symbolic.append(relative)
            entries.append({"path": relative, "kind": "symlink"})
        elif stat.S_ISDIR(metadata.st_mode):
            entries.append({"path": relative, "kind": "directory"})
        elif stat.S_ISREG(metadata.st_mode):
            value = _read_regular_file(path)
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": len(value),
                    "sha256": _sha256(value),
                }
            )
            if path.suffix in {".py", ".pyc", ".pyo", ".pth"}:
                forbidden.append(relative)
        else:
            special.append(relative)
            entries.append({"path": relative, "kind": "special"})
    if forbidden or symbolic or special:
        raise ValueError(
            "learner cache is not code-free and regular; "
            f"python={forbidden!r}, symlinks={symbolic!r}, special={special!r}"
        )
    return {
        "entries": entries,
        "entriesSha256": _sha256(_canonical_json({"entries": entries})),
        "pythonCodeFiles": forbidden,
        "symbolicPaths": symbolic,
        "specialPaths": special,
    }


def build_learner_source_bundle(
    source_root: Path,
    full_source_manifest: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Publish an immutable learner bundle with one same-filesystem rename."""

    manifest, payload = build_learner_source_manifest(
        source_root, full_source_manifest
    )
    destination = destination.absolute()
    if destination.exists():
        observed = verify_learner_source_bundle(
            destination,
            expected_full_source_tree_sha256=manifest["fullSourceTreeSha256"],
        )
        if observed != manifest:
            raise ValueError("existing learner bundle differs from current source")
        return observed
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for relative, value in payload.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
            target.chmod(0o444)
        manifest_path = temporary / _MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        manifest_path.chmod(0o444)
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        verify_learner_source_bundle(
            temporary,
            expected_full_source_tree_sha256=manifest["fullSourceTreeSha256"],
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.chmod(0o755)
            for directory in temporary.rglob("*"):
                if directory.is_dir():
                    directory.chmod(0o755)
            shutil.rmtree(temporary)
    return verify_learner_source_bundle(
        destination,
        expected_full_source_tree_sha256=manifest["fullSourceTreeSha256"],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--expected-full-source-tree-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        if args.repo_root is None or args.source_manifest is None:
            raise ValueError("build requires --repo-root and --source-manifest")
        full_manifest = load_source_manifest(
            args.source_manifest, verify_current_tree=False
        )
        manifest = build_learner_source_bundle(
            args.repo_root, full_manifest, args.destination
        )
    else:
        manifest = verify_learner_source_bundle(
            args.destination,
            expected_full_source_tree_sha256=(
                args.expected_full_source_tree_sha256
            ),
        )
    print(manifest["treeSha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORBIDDEN_LEARNER_MODULES",
    "LEARNER_BUNDLE_KIND",
    "LEARNER_BUNDLE_SCHEMA_VERSION",
    "LEARNER_ENTRY_MODULE",
    "LEARNER_MODULE_ALLOWLIST",
    "build_learner_source_bundle",
    "build_learner_source_manifest",
    "learner_import_closure",
    "validate_learner_source_manifest",
    "validate_code_free_cache",
    "verify_learner_source_bundle",
]
