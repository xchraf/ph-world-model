from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from blocket_league.runtime_firewall_trace import (
    RuntimeFirewallTrace,
    verify_runtime_trace,
)


SOURCE = "a" * 64


def _sealed_trace(root: Path) -> tuple[Path, dict[str, object]]:
    archive = root / "pixels.pt"
    archive.write_bytes(b"pixels-only")
    parameter = torch.nn.Parameter(torch.ones(2))
    protected = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    path = root / "firewall-trace.jsonl"
    trace = RuntimeFirewallTrace(path, stage="direct:full", source_tree_sha256=SOURCE)
    trace.record_mount_manifest(root, role="test_mount")
    trace.record_file_read(
        archive,
        role="trainer_archive:fit",
        serialized_keys=("manifest", "pixels"),
    )
    trace.record_tensor_payload(
        phase="producer",
        role="raw_video",
        tensors={"frames": np.zeros((2, 4, 4, 3), dtype=np.uint8)},
    )
    trace.record_optimizer(
        phase="direct:full",
        named_parameters={"head.weight": parameter},
        protected_parameters={"encoder.backbone.weight": protected},
    )
    trace.record_gradient_batch(
        phase="direct:full", step=1, tensors={"frames": torch.zeros(2, 4, 4)}
    )
    trace.record_backbone_boundary(
        phase="direct:full", boundary="selected_checkpoint", sha256="b" * 64
    )
    seal = trace.snapshot().to_dict()
    trace.close()
    return path, seal


def test_runtime_trace_records_real_inode_digest_optimizer_and_payload(tmp_path: Path) -> None:
    path, seal = _sealed_trace(tmp_path)
    records = verify_runtime_trace(path, seal)
    file_record = next(record for record in records if record["event"] == "file_read")
    archive = tmp_path / "pixels.pt"
    assert file_record["payload"]["inode"] == archive.stat().st_ino
    assert file_record["payload"]["contentSha256"] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    optimizer = next(
        record for record in records if record["event"] == "optimizer_constructed"
    )["payload"]
    assert optimizer["protectedOverlap"] is False
    assert optimizer["parameters"][0]["name"] == "head.weight"
    payload = next(record for record in records if record["event"] == "tensor_payload")
    assert set(payload["payload"]["tensors"]) == {"frames"}


@pytest.mark.parametrize("tamper", ("event", "chain", "seal"))
def test_runtime_trace_tampering_is_fail_closed(tmp_path: Path, tamper: str) -> None:
    path, seal = _sealed_trace(tmp_path)
    if tamper == "seal":
        seal = dict(seal)
        seal["events"] = int(seal["events"]) + 1
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[2])
        if tamper == "event":
            record["payload"]["role"] = "heldout"
        else:
            record["previousSha256"] = "f" * 64
        lines[2] = json.dumps(record, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_runtime_trace(path, seal)


def test_record_file_read_refuses_symbolic_final_component(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"safe")
    link = tmp_path / "link"
    link.symlink_to(target)
    trace = RuntimeFirewallTrace(
        tmp_path / "trace.jsonl", stage="backbone", source_tree_sha256=SOURCE
    )
    with pytest.raises(OSError):
        trace.record_file_read(link, role="trainer_archive:fit")
    trace.close()
