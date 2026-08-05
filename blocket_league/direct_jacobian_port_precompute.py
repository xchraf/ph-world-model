"""Pixels-only pre-pH construction of the frozen empirical port tangent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from .direct_activation_lens import FrozenSoftPixelActivationLens
from .direct_jacobian_port_extractor import (
    EmpiricalTangentAccumulator,
    EmpiricalTangentArtifact,
    EmpiricalTangentConfig,
)
from .pixel_direct_model import DirectPixelTransformer
from .runtime_firewall_trace import RuntimeFirewallTrace
from .tensor_provenance import module_tensor_hash


@dataclass(frozen=True)
class JacobianPortPrecomputeConfig:
    """Locked action-free sample budget before any pH module is constructed."""

    contexts: int = 4_096
    batch_size: int = 4
    lens_block: int = 4
    horizons: tuple[int, ...] = (1, 2, 4)
    channel_rank: int = 16
    neighbors: int = 32
    support_floor_ratio: float = 0.02

    def __post_init__(self) -> None:
        for name in ("contexts", "batch_size", "channel_rank", "neighbors"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.lens_block) is not int or self.lens_block < 0:
            raise ValueError("lens_block must be a non-negative integer")
        if (
            type(self.horizons) is not tuple
            or not self.horizons
            or tuple(sorted(set(self.horizons))) != self.horizons
            or any(type(value) is not int or value < 1 for value in self.horizons)
        ):
            raise ValueError("horizons must be a sorted unique positive tuple")
        if (
            type(self.support_floor_ratio) not in (int, float)
            or not math.isfinite(float(self.support_floor_ratio))
            or not 0.0 <= self.support_floor_ratio < 1.0
        ):
            raise ValueError("support_floor_ratio must lie in [0,1)")

    @property
    def tangent(self) -> EmpiricalTangentConfig:
        return EmpiricalTangentConfig(
            channel_rank=self.channel_rank,
            neighbors=self.neighbors,
            support_floor_ratio=self.support_floor_ratio,
        )


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_json_save(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _selected_transition_rows(
    suite: Mapping[str, torch.Tensor],
    contexts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if type(suite) is not dict or set(suite) != {"pixelContexts", "frames"}:
        raise ValueError("port precompute suite must be exactly pixels-only")
    pixel_contexts = suite["pixelContexts"]
    frames = suite["frames"]
    if (
        type(pixel_contexts) is not torch.Tensor
        or type(frames) is not torch.Tensor
        or pixel_contexts.dtype != torch.uint8
        or frames.dtype != torch.uint8
        or pixel_contexts.ndim != 5
        or frames.ndim != 4
        or pixel_contexts.shape[:2] != frames.shape[:2]
        or pixel_contexts.shape[1] < 2
    ):
        raise ValueError("port precompute pixel tensors are malformed")
    source = pixel_contexts[:, :-1].flatten(0, 1)
    successor = pixel_contexts[:, 1:].flatten(0, 1)
    available = source.shape[0]
    if available < contexts:
        raise ValueError(
            f"registered port precompute needs {contexts} transitions, got {available}"
        )
    indices = torch.linspace(0, available - 1, contexts).round().long()
    if int(torch.unique(indices).numel()) != contexts:
        raise AssertionError("response-blind transition selection produced duplicates")
    return source[indices].contiguous(), successor[indices].contiguous(), indices


def build_empirical_tangent_from_pixels(
    backbone: DirectPixelTransformer,
    fit_suite: Mapping[str, torch.Tensor],
    *,
    system: str,
    fit_sanitized_tensor_sha256: str,
    output_dir: Path,
    device: torch.device,
    config: JacobianPortPrecomputeConfig = JacobianPortPrecomputeConfig(),
    runtime_trace: RuntimeFirewallTrace | None = None,
    source_tree_sha256: str,
) -> tuple[EmpiricalTangentArtifact, dict[str, Any]]:
    """Build and atomically seal the fit-only tangent before pH construction."""

    if not _valid_sha256(fit_sanitized_tensor_sha256):
        raise ValueError("fit sanitized tensor hash is malformed")
    if not _valid_sha256(source_tree_sha256):
        raise ValueError("source tree hash is malformed")
    if not isinstance(system, str) or not system:
        raise ValueError("system must be a non-empty string")
    if not 0 <= config.lens_block < len(backbone.blocks):
        raise ValueError("registered lens block is outside the backbone")
    if config.channel_rank > backbone.config.hidden_size:
        raise ValueError("channel rank exceeds the backbone hidden size")
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("port precompute requires a frozen backbone")
    backbone.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    owns_trace = runtime_trace is None
    if runtime_trace is None:
        runtime_trace = RuntimeFirewallTrace(
            output_dir / "firewall-trace.jsonl",
            stage=f"jacobian-port-precompute:{system}",
            source_tree_sha256=source_tree_sha256,
        )
    source, successor, selected_indices = _selected_transition_rows(
        fit_suite, config.contexts
    )
    started = time.perf_counter()
    runtime_trace.record_tensor_payload(
        phase="jacobian_port_precompute_no_optimizer",
        role="response_blind_pixels_only_port_fit",
        tensors={"pixelContexts": source, "frames": fit_suite["frames"][:1]},
    )
    lens = FrozenSoftPixelActivationLens(
        backbone,
        intervention_block=config.lens_block,
        horizons=config.horizons,
    ).to(device).eval().requires_grad_(False)
    accumulator = EmpiricalTangentAccumulator(config.tangent)
    for start in range(0, config.contexts, config.batch_size):
        stop = min(start + config.batch_size, config.contexts)
        source_batch = source[start:stop].to(device, non_blocking=True).long()
        successor_batch = successor[start:stop].to(device, non_blocking=True).long()
        with torch.no_grad():
            source_activation = lens.soft_prefix_activation(source_batch).detach()
            observed_activation = lens.soft_prefix_activation(successor_batch).detach()
            source_probabilities = lens.pixel_probabilities(source_batch)
            predicted_frame = lens.soft_forward(source_probabilities)[:, -1]
            predicted_context = torch.cat(
                (source_probabilities[:, 1:], predicted_frame[:, None]), dim=1
            )
            predicted_activation = lens.soft_prefix_activation(
                predicted_context
            ).detach()
        accumulator.update(
            source_activation, observed_activation, predicted_activation
        )
    artifact = accumulator.finalize()
    backbone_hash = module_tensor_hash(backbone)
    payload: dict[str, Any] = {
        "kind": "frozen_empirical_jacobian_tangent_v1",
        "system": system,
        "actionChannels": 0,
        "physicalStateChannels": 0,
        "sourceSchema": ["pixelContexts", "frames"],
        "config": asdict(config),
        "fitSanitizedTensorSha256": fit_sanitized_tensor_sha256,
        "sourceTreeSha256": source_tree_sha256,
        "backboneHash": backbone_hash,
        "selectedTransitionIndices": selected_indices,
        "selectedTransitionIndicesSha256": _tensor_sha256(selected_indices),
        "channelBasis": artifact.channel_basis,
        "featureLocations": artifact.feature_locations,
        "featureMean": artifact.feature_mean,
        "featureScale": artifact.feature_scale,
        "innovationSupport": artifact.innovation_support,
        "innovationChannelEigenvalues": artifact.innovation_channel_eigenvalues,
        "sourceActivationTensorSha256": artifact.source_tensor_sha256,
    }
    artifact_path = output_dir / "empirical-tangent.pt"
    _atomic_torch_save(artifact_path, payload)
    trace = runtime_trace.snapshot().to_dict()
    if owns_trace:
        runtime_trace.close()
    summary = {
        "kind": "frozen_empirical_jacobian_tangent_summary_v1",
        "system": system,
        "contexts": config.contexts,
        "seconds": float(time.perf_counter() - started),
        "backboneHash": backbone_hash,
        "fitSanitizedTensorSha256": fit_sanitized_tensor_sha256,
        "selectedTransitionIndicesSha256": payload[
            "selectedTransitionIndicesSha256"
        ],
        "sourceActivationTensorSha256": artifact.source_tensor_sha256,
        # Stable across the learner (/output) and post-freeze (/training)
        # bind points. The completion seal carries the full artifact hash.
        "artifact": artifact_path.name,
        "runtimeTrace": trace,
    }
    _atomic_json_save(output_dir / "summary.json", summary)
    return artifact, summary


_ARTIFACT_KEYS = {
    "kind",
    "system",
    "actionChannels",
    "physicalStateChannels",
    "sourceSchema",
    "config",
    "fitSanitizedTensorSha256",
    "sourceTreeSha256",
    "backboneHash",
    "selectedTransitionIndices",
    "selectedTransitionIndicesSha256",
    "channelBasis",
    "featureLocations",
    "featureMean",
    "featureScale",
    "innovationSupport",
    "innovationChannelEigenvalues",
    "sourceActivationTensorSha256",
}


def load_empirical_tangent_artifact(
    path: Path,
    *,
    expected_system: str,
    expected_fit_sanitized_tensor_sha256: str,
    expected_source_tree_sha256: str,
    expected_backbone_hash: str,
    expected_config: JacobianPortPrecomputeConfig = JacobianPortPrecomputeConfig(),
    runtime_trace: RuntimeFirewallTrace | None = None,
) -> EmpiricalTangentArtifact:
    """Fail closed when loading the zero-action pre-pH tangent artifact."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if runtime_trace is not None:
        runtime_trace.record_file_read(
            path,
            role="frozen_empirical_jacobian_tangent",
            serialized_keys=tuple(sorted(payload)) if type(payload) is dict else (),
        )
    if type(payload) is not dict or set(payload) != _ARTIFACT_KEYS:
        raise ValueError("empirical tangent artifact schema is not exact")
    scalar_expected = {
        "kind": "frozen_empirical_jacobian_tangent_v1",
        "system": expected_system,
        "actionChannels": 0,
        "physicalStateChannels": 0,
        "sourceSchema": ["pixelContexts", "frames"],
        "config": asdict(expected_config),
        "fitSanitizedTensorSha256": expected_fit_sanitized_tensor_sha256,
        "sourceTreeSha256": expected_source_tree_sha256,
        "backboneHash": expected_backbone_hash,
    }
    for name, expected in scalar_expected.items():
        if payload[name] != expected:
            raise ValueError(f"empirical tangent artifact {name} mismatch")
    if not _valid_sha256(payload["sourceActivationTensorSha256"]):
        raise ValueError("empirical tangent activation hash is malformed")
    indices = payload["selectedTransitionIndices"]
    if (
        type(indices) is not torch.Tensor
        or indices.dtype != torch.long
        or indices.ndim != 1
        or indices.shape[0] != expected_config.contexts
        or _tensor_sha256(indices) != payload["selectedTransitionIndicesSha256"]
    ):
        raise ValueError("empirical tangent transition selection is malformed")
    artifact = EmpiricalTangentArtifact(
        channel_basis=payload["channelBasis"],
        feature_locations=payload["featureLocations"],
        feature_mean=payload["featureMean"],
        feature_scale=payload["featureScale"],
        innovation_support=payload["innovationSupport"],
        innovation_channel_eigenvalues=payload[
            "innovationChannelEigenvalues"
        ],
        source_tensor_sha256=payload["sourceActivationTensorSha256"],
    )
    # Constructing the frozen module is the canonical full shape/schema check.
    from .direct_jacobian_port_extractor import FrozenEmpiricalJacobianActivationPort

    FrozenEmpiricalJacobianActivationPort(
        artifact,
        history_frames=artifact.innovation_support.shape[1],
        patch_count=artifact.innovation_support.shape[2],
        hidden_size=artifact.channel_basis.shape[0],
        port_size=1,
        config=expected_config.tangent,
    ).assert_frozen_parameter_free()
    return artifact


__all__ = [
    "JacobianPortPrecomputeConfig",
    "build_empirical_tangent_from_pixels",
    "load_empirical_tangent_artifact",
]
