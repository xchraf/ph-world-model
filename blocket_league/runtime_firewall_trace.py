"""Append-only, hash-chained runtime firewall traces for Experiment F."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

import torch


TRACE_KIND = "experiment_f_runtime_firewall_trace_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_SUBMIT_DIR",
)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _recursive_manifest(root: Path) -> list[dict[str, Any]]:
    """Inventory exact bytes below ``root`` without following symlinks."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("recursive manifest root must be a nonsymbolic directory")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            entries.append({"path": relative, "kind": "symlink"})
        elif stat.S_ISDIR(metadata.st_mode):
            entries.append({"path": relative, "kind": "directory"})
        elif stat.S_ISREG(metadata.st_mode):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (metadata.st_dev, metadata.st_ino)
                    != (opened.st_dev, opened.st_ino)
                ):
                    raise ValueError(
                        f"recursive manifest file changed while opening: {relative}"
                    )
                digest = _descriptor_sha256(descriptor)
                after = os.fstat(descriptor)
                if (
                    opened.st_size != after.st_size
                    or (opened.st_dev, opened.st_ino)
                    != (after.st_dev, after.st_ino)
                ):
                    raise ValueError(
                        f"recursive manifest file changed while reading: {relative}"
                    )
                entries.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "bytes": after.st_size,
                        "sha256": digest,
                    }
                )
            finally:
                os.close(descriptor)
        else:
            entries.append({"path": relative, "kind": "special"})
    return entries


def _mount_info() -> tuple[str, int, list[dict[str, Any]]]:
    path = Path("/proc/self/mountinfo")
    if not path.is_file():
        return hashlib.sha256(b"").hexdigest(), 0, []
    content = path.read_bytes()
    mounts: list[dict[str, Any]] = []
    for raw_line in content.decode("utf-8", errors="strict").splitlines():
        before, separator, after = raw_line.partition(" - ")
        left = before.split()
        right = after.split()
        if not separator or len(left) < 6 or len(right) < 3:
            raise ValueError("/proc/self/mountinfo contains a malformed record")
        mounts.append(
            {
                "mountId": left[0],
                "parentId": left[1],
                "device": left[2],
                "root": left[3],
                "mountPoint": left[4],
                "mountOptions": left[5],
                "optionalFields": left[6:],
                "fileSystem": right[0],
                "source": right[1],
                "superOptions": right[2:],
            }
        )
    mounts.sort(
        key=lambda item: (
            item["mountPoint"], item["source"], item["mountId"]
        )
    )
    return hashlib.sha256(content).hexdigest(), len(content.splitlines()), mounts


def _tensor_schema(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if type(tensors) is not dict or not tensors:
        raise ValueError("runtime gradient batch must be a non-empty plain dictionary")
    result = {}
    for name, tensor in tensors.items():
        if type(name) is not str or not name or type(tensor) is not torch.Tensor:
            raise ValueError("runtime gradient batch keys/tensors are invalid")
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requiresGrad": bool(tensor.requires_grad),
        }
    return result


def _payload_schema(values: Mapping[str, object]) -> dict[str, Any]:
    if type(values) is not dict or not values:
        raise ValueError("runtime payload must be a non-empty plain dictionary")
    result: dict[str, Any] = {}
    for name, value in values.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if type(name) is not str or not name or shape is None or dtype is None:
            raise ValueError("runtime payload key/value schema is invalid")
        result[name] = {
            "shape": [int(item) for item in shape],
            "dtype": str(dtype),
            "pythonType": f"{type(value).__module__}.{type(value).__qualname__}",
            "device": str(value.device) if type(value) is torch.Tensor else "cpu",
            "requiresGrad": bool(value.requires_grad)
            if type(value) is torch.Tensor
            else False,
        }
    return result


@dataclass(frozen=True)
class RuntimeTraceSeal:
    path: str
    events: int
    head_sha256: str
    file_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "events": self.events,
            "headSha256": self.head_sha256,
            "fileSha256": self.file_sha256,
        }


def _read_and_validate(path: Path) -> tuple[int, str, list[dict[str, Any]]]:
    previous = "0" * 64
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for expected_sequence, line in enumerate(source):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("runtime firewall trace contains invalid JSON") from error
            if type(record) is not dict or set(record) != {
                "kind",
                "sequence",
                "previousSha256",
                "event",
                "payload",
                "eventSha256",
            }:
                raise ValueError("runtime firewall trace event schema is not exact")
            unsigned = {name: record[name] for name in record if name != "eventSha256"}
            observed = hashlib.sha256(_canonical(unsigned)).hexdigest()
            if (
                record["kind"] != TRACE_KIND
                or record["sequence"] != expected_sequence
                or record["previousSha256"] != previous
                or type(record["event"]) is not str
                or not record["event"]
                or type(record["payload"]) is not dict
                or record["eventSha256"] != observed
            ):
                raise ValueError("runtime firewall trace hash chain is invalid")
            previous = observed
            records.append(record)
    if not records:
        raise ValueError("runtime firewall trace is empty")
    return len(records), previous, records


class RuntimeFirewallTrace:
    """Append events through ``O_APPEND`` and expose immutable snapshots."""

    def __init__(self, path: Path, *, stage: str, source_tree_sha256: str) -> None:
        if (
            type(stage) is not str
            or not stage
            or _SHA256.fullmatch(source_tree_sha256) is None
        ):
            raise ValueError("runtime trace stage/source identity is invalid")
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("runtime trace path cannot be symbolic")
        if path.exists():
            events, head, _ = _read_and_validate(path)
            self._sequence = events
            self._previous = head
        else:
            self._sequence = 0
            self._previous = "0" * 64
        self._descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        mount_sha256, mount_lines, mounts = _mount_info()
        self.append(
            "stage_boundary",
            {
                "boundary": "resume" if self._sequence else "start",
                "stage": stage,
                "sourceTreeSha256": source_tree_sha256,
                "pid": os.getpid(),
                "cwd": str(Path.cwd().resolve()),
                "slurm": {key: os.environ[key] for key in _SLURM_KEYS if key in os.environ},
                "mountInfoSha256": mount_sha256,
                "mountInfoLines": mount_lines,
                "mounts": mounts,
            },
        )

    def append(self, event: str, payload: Mapping[str, Any]) -> None:
        if type(event) is not str or not event or type(payload) is not dict:
            raise ValueError("runtime firewall event is invalid")
        unsigned = {
            "kind": TRACE_KIND,
            "sequence": self._sequence,
            "previousSha256": self._previous,
            "event": event,
            "payload": dict(payload),
        }
        event_sha256 = hashlib.sha256(_canonical(unsigned)).hexdigest()
        encoded = json.dumps(
            {**unsigned, "eventSha256": event_sha256},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        written = os.write(self._descriptor, encoded)
        if written != len(encoded):  # pragma: no cover - regular-file invariant
            raise OSError("short append to runtime firewall trace")
        self._sequence += 1
        self._previous = event_sha256

    def record_file_read(
        self,
        path: Path,
        *,
        role: str,
        serialized_keys: tuple[str, ...] | None = None,
        semantic_sha256: str | None = None,
    ) -> None:
        if type(role) is not str or not role:
            raise ValueError("runtime file-read role is invalid")
        if semantic_sha256 is not None and _SHA256.fullmatch(semantic_sha256) is None:
            raise ValueError("runtime file-read semantic digest is invalid")
        keys = tuple(serialized_keys or ())
        if any(type(key) is not str or not key for key in keys):
            raise ValueError("runtime file-read serialized keys are invalid")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_stat.st_mode):
                raise ValueError("runtime file read target is not a regular file")
            content_sha256 = _descriptor_sha256(descriptor)
            proc_descriptor = Path(f"/proc/self/fd/{descriptor}")
            if proc_descriptor.exists():
                resolved_path = os.readlink(proc_descriptor)
                if resolved_path.endswith(" (deleted)"):
                    raise ValueError("runtime file read target was unlinked during audit")
            else:
                # macOS has no /proc; the O_NOFOLLOW descriptor identity and
                # fstat fields remain authoritative for the content read.
                resolved_path = str(path.resolve(strict=True))
            self.append(
                "file_read",
                {
                    "role": role,
                    "path": str(path.absolute()),
                    "resolvedPath": resolved_path,
                    "device": descriptor_stat.st_dev,
                    "inode": descriptor_stat.st_ino,
                    "bytes": descriptor_stat.st_size,
                    "contentSha256": content_sha256,
                    "semanticSha256": semantic_sha256,
                    "serializedKeys": list(keys),
                },
            )
        finally:
            os.close(descriptor)

    def record_mount_manifest(self, root: Path, *, role: str) -> None:
        resolved = root.resolve(strict=True)
        if not resolved.is_dir() or type(role) is not str or not role:
            raise ValueError("runtime mount manifest root/role is invalid")
        stat = resolved.stat()
        entries = sorted(path.name for path in resolved.iterdir())
        self.append(
            "mount_manifest",
            {
                "role": role,
                "path": str(root.absolute()),
                "resolvedPath": str(resolved),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "entries": entries,
                "entriesSha256": hashlib.sha256(
                    json.dumps(entries, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
        )

    def record_recursive_manifest(
        self,
        root: Path,
        *,
        role: str,
        python_code_suffixes: tuple[str, ...] = (".py", ".pyc", ".pyo", ".pth"),
    ) -> None:
        """Record every visible entry and explicitly count injectable code files."""

        if type(role) is not str or not role:
            raise ValueError("runtime recursive-manifest role is invalid")
        if (
            type(python_code_suffixes) is not tuple
            or any(
                type(suffix) is not str or not suffix.startswith(".")
                for suffix in python_code_suffixes
            )
        ):
            raise ValueError("runtime recursive-manifest suffix policy is invalid")
        if root.is_symlink():
            raise ValueError("runtime recursive-manifest root cannot be symbolic")
        resolved = root.resolve(strict=True)
        entries = _recursive_manifest(resolved)
        code_files = sorted(
            entry["path"]
            for entry in entries
            if entry["kind"] == "file"
            and Path(entry["path"]).suffix in python_code_suffixes
        )
        symbolic_paths = sorted(
            entry["path"] for entry in entries if entry["kind"] == "symlink"
        )
        special_paths = sorted(
            entry["path"] for entry in entries if entry["kind"] == "special"
        )
        self.append(
            "recursive_manifest",
            {
                "role": role,
                "path": str(root.absolute()),
                "resolvedPath": str(resolved),
                "entries": entries,
                "entriesSha256": hashlib.sha256(
                    json.dumps(
                        entries,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
                "pythonCodeSuffixes": list(python_code_suffixes),
                "pythonCodeFiles": code_files,
                "pythonCodeFileCount": len(code_files),
                "symbolicPaths": symbolic_paths,
                "specialPaths": special_paths,
            },
        )

    def record_gradient_batch(
        self,
        *,
        phase: str,
        step: int,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        self.append(
            "gradient_batch",
            {
                "phase": phase,
                "step": step,
                "tensors": _tensor_schema(dict(tensors)),
            },
        )

    def record_tensor_payload(
        self,
        *,
        phase: str,
        role: str,
        tensors: Mapping[str, object],
    ) -> None:
        if type(role) is not str or not role:
            raise ValueError("runtime tensor-payload role is invalid")
        self.append(
            "tensor_payload",
            {
                "phase": phase,
                "role": role,
                "tensors": _payload_schema(dict(tensors)),
            },
        )

    def record_optimizer(
        self,
        *,
        phase: str,
        named_parameters: Mapping[str, torch.nn.Parameter],
        protected_parameters: Mapping[str, torch.nn.Parameter] | None = None,
    ) -> None:
        if type(named_parameters) is not dict or not named_parameters:
            raise ValueError("optimizer trace requires a non-empty named mapping")
        if any(
            type(name) is not str
            or not name
            or type(parameter) is not torch.nn.Parameter
            for name, parameter in named_parameters.items()
        ):
            raise ValueError("optimizer trace parameter mapping is invalid")
        protected = dict(protected_parameters or {})
        if any(
            type(name) is not str
            or not name
            or type(parameter) is not torch.nn.Parameter
            for name, parameter in protected.items()
        ):
            raise ValueError("optimizer trace protected mapping is invalid")
        parameter_ids = {id(parameter) for parameter in named_parameters.values()}
        protected_ids = {id(parameter) for parameter in protected.values()}
        self.append(
            "optimizer_constructed",
            {
                "phase": phase,
                "parameters": [
                    {
                        "name": name,
                        "pythonId": id(parameter),
                        "dataPointer": parameter.data_ptr(),
                        "shape": list(parameter.shape),
                        "numel": parameter.numel(),
                    }
                    for name, parameter in named_parameters.items()
                ],
                "protectedParameters": [
                    {
                        "name": name,
                        "pythonId": id(parameter),
                        "dataPointer": parameter.data_ptr(),
                    }
                    for name, parameter in protected.items()
                ],
                "protectedOverlap": bool(parameter_ids & protected_ids),
            },
        )

    def record_backbone_boundary(
        self, *, phase: str, boundary: str, sha256: str
    ) -> None:
        if _SHA256.fullmatch(sha256) is None:
            raise ValueError("runtime backbone hash is invalid")
        self.append(
            "backbone_boundary",
            {"phase": phase, "boundary": boundary, "sha256": sha256},
        )

    def snapshot(self) -> RuntimeTraceSeal:
        os.fsync(self._descriptor)
        events, head, _ = _read_and_validate(self.path)
        if events != self._sequence or head != self._previous:
            raise ValueError("runtime firewall trace changed outside append writer")
        return RuntimeTraceSeal(
            path=self.path.name,
            events=events,
            head_sha256=head,
            file_sha256=_file_sha256(self.path),
        )

    def close(self) -> None:
        if getattr(self, "_descriptor", None) is not None:
            os.fsync(self._descriptor)
            os.close(self._descriptor)
            self._descriptor = None

    def __enter__(self) -> "RuntimeFirewallTrace":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def verify_runtime_trace(path: Path, seal: Mapping[str, Any]) -> list[dict[str, Any]]:
    if type(seal) is not dict or set(seal) != {
        "path",
        "events",
        "headSha256",
        "fileSha256",
    }:
        raise ValueError("runtime trace seal schema is not exact")
    events, head, records = _read_and_validate(path)
    if (
        seal["path"] != path.name
        or seal["events"] != events
        or seal["headSha256"] != head
        or seal["fileSha256"] != _file_sha256(path)
    ):
        raise ValueError("runtime firewall trace differs from its sealed snapshot")
    return records


__all__ = [
    "RuntimeFirewallTrace",
    "RuntimeTraceSeal",
    "TRACE_KIND",
    "verify_runtime_trace",
]
