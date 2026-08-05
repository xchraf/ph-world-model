"""Safe typed persistence for shared physical evidence and control shards.

The payloads contain primitive values and tensors only, so they can be loaded
with ``torch.load(..., weights_only=True)``.  Every field is included in a
canonical digest.  Gate decisions are never persisted as authoritative input;
they are derived later from the calibrated responses or replayed command traces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import torch

from .direct_physical_evaluation import (
    CalibrationResult,
    ControlResult,
    DOptimalSelection,
    LinearInterfaceProtocol,
    RealizabilityMetrics,
    control_trace_sha256,
)

if TYPE_CHECKING:  # pragma: no cover
    from .direct_postfreeze_runner import (
        ControlShard,
        LoadedPostFreezeSystem,
        PhysicalRealizabilityResult,
    )


REGISTERED_CONTROL_EPISODE_SEED = 151_910_737 + 80_000
REGISTERED_CONTROL_PLANNER_SEED = 151_910_737 + 90_000
REGISTERED_CONTROL_NAMES = (
    "structured",
    "unstructured",
    "activation",
    "no_jacobian",
    "shuffled_lens",
    "coast",
    "random",
)


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
    if value is None:
        digest.update(b"none")
    elif type(value) is bool:
        digest.update(b"bool:1" if value else b"bool:0")
    elif type(value) is int:
        digest.update(f"int:{value}".encode("ascii"))
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical evidence cannot contain a non-finite float")
        digest.update(b"float:")
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    elif type(value) is str:
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.requires_grad or tensor.grad_fn is not None:
            raise ValueError("canonical evidence tensor is attached to autograd")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("canonical evidence tensor is non-finite")
        digest.update(b"tensor:")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif type(value) in (tuple, list):
        digest.update(("tuple:" if type(value) is tuple else "list:").encode("ascii"))
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _update_digest(digest, item)
    elif type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("canonical evidence mapping keys must be strings")
        digest.update(f"dict:{len(value)}".encode("ascii"))
        for key in sorted(value):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    else:
        raise TypeError(f"unsupported canonical evidence type: {type(value)!r}")


def canonical_evidence_sha256(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, dict(value))
    return digest.hexdigest()


def training_lineage_sha256(loaded: "LoadedPostFreezeSystem") -> str:
    loaded.assert_frozen_and_unchanged()
    payload = {
        "kind": "experiment_f_frozen_training_lineage_v1",
        "system": loaded.system_name,
        "producerSealSha256": loaded.producer_seal_sha256,
        "backboneCheckpointSha256": loaded.backbone_checkpoint_sha256,
        "baselineCheckpointSha256": loaded.baseline_checkpoint_sha256,
        "backboneSha256": loaded.backbone_hash,
        "sourceTreeSha256": loaded.source_tree_sha256,
        "splitSha256": dict(loaded.observed_split_sha256),
        "variantCheckpointSha256": {
            name: frozen.checkpoint_sha256
            for name, frozen in loaded.variants.items()
        },
    }
    return canonical_evidence_sha256(payload)


def _selection_payload(selection: DOptimalSelection) -> dict[str, Any]:
    return {item.name: getattr(selection, item.name) for item in fields(DOptimalSelection)}


def _selection_from_payload(payload: Any) -> DOptimalSelection:
    expected = {item.name for item in fields(DOptimalSelection)}
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("physical selection payload schema is not exact")
    return DOptimalSelection(**payload)


def _protocol_payload(protocol: LinearInterfaceProtocol) -> dict[str, Any]:
    return {
        item.name: getattr(protocol, item.name)
        for item in fields(LinearInterfaceProtocol)
    }


def _protocol_from_payload(payload: Any) -> LinearInterfaceProtocol:
    expected = {item.name for item in fields(LinearInterfaceProtocol)}
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("physical interface protocol schema is not exact")
    return LinearInterfaceProtocol(**payload)


def _calibration_payload(calibration: CalibrationResult) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in fields(CalibrationResult):
        value = getattr(calibration, item.name)
        if item.name == "selection":
            value = _selection_payload(value)
        elif item.name == "physical_protocol":
            value = _protocol_payload(value)
        elif isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        elif isinstance(value, Mapping):
            value = dict(value)
        result[item.name] = value
    return result


def _calibration_from_payload(payload: Any) -> CalibrationResult:
    expected = {item.name for item in fields(CalibrationResult)}
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("physical calibration payload schema is not exact")
    values = dict(payload)
    values["selection"] = _selection_from_payload(values["selection"])
    values["physical_protocol"] = _protocol_from_payload(values["physical_protocol"])
    return CalibrationResult(**values)


def _metrics_payload(metrics: RealizabilityMetrics) -> dict[str, Any]:
    return {item.name: getattr(metrics, item.name) for item in fields(RealizabilityMetrics)}


def _metrics_from_payload(payload: Any) -> RealizabilityMetrics:
    expected = {item.name for item in fields(RealizabilityMetrics)}
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("physical realizability metric schema is not exact")
    return RealizabilityMetrics(**payload)


def physical_result_payload(physical: "PhysicalRealizabilityResult") -> dict[str, Any]:
    return {
        "kind": "experiment_f_shared_physical_realizability_v1",
        "system": physical.system_name,
        "calibrationPairsPerAxis": physical.calibration_pairs_per_axis,
        "heldoutPairsPerAxis": physical.heldout_pairs_per_axis,
        "neuralHashesBefore": dict(physical.neural_hashes_before),
        "neuralHashesAfter": dict(physical.neural_hashes_after),
        "interfaces": {
            interface_name: {
                "interface": interface.interface_name,
                "models": {
                    model_name: {
                        "calibration": _calibration_payload(result.calibration),
                        "metrics": _metrics_payload(result.metrics),
                    }
                    for model_name, result in interface.models.items()
                },
            }
            for interface_name, interface in physical.interfaces.items()
        },
    }


def physical_result_sha256(physical: "PhysicalRealizabilityResult") -> str:
    return canonical_evidence_sha256(physical_result_payload(physical))


def save_physical_result(
    path: Path,
    loaded: "LoadedPostFreezeSystem",
    physical: "PhysicalRealizabilityResult",
) -> str:
    core = physical_result_payload(physical)
    digest = canonical_evidence_sha256(core)
    wrapper = {
        "kind": "experiment_f_shared_physical_file_v1",
        "trainingLineageSha256": training_lineage_sha256(loaded),
        "physicalSha256": digest,
        "physical": core,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(wrapper, temporary)
    temporary.replace(path)
    return digest


def load_physical_result(
    path: Path,
    loaded: "LoadedPostFreezeSystem",
) -> "PhysicalRealizabilityResult":
    from .direct_postfreeze_runner import (
        InterfaceRealizabilityResult,
        ModelRealizabilityResult,
        PhysicalRealizabilityResult,
    )

    wrapper = torch.load(path, map_location="cpu", weights_only=True)
    if type(wrapper) is not dict or set(wrapper) != {
        "kind",
        "trainingLineageSha256",
        "physicalSha256",
        "physical",
    }:
        raise ValueError("shared physical file schema is not exact")
    if (
        wrapper["kind"] != "experiment_f_shared_physical_file_v1"
        or wrapper["trainingLineageSha256"] != training_lineage_sha256(loaded)
    ):
        raise ValueError("shared physical file training lineage changed")
    core = wrapper["physical"]
    if (
        type(core) is not dict
        or set(core)
        != {
            "kind",
            "system",
            "calibrationPairsPerAxis",
            "heldoutPairsPerAxis",
            "neuralHashesBefore",
            "neuralHashesAfter",
            "interfaces",
        }
        or core["kind"] != "experiment_f_shared_physical_realizability_v1"
        or core["system"] != loaded.system_name
        or canonical_evidence_sha256(core) != wrapper["physicalSha256"]
    ):
        raise ValueError("shared physical evidence hash/schema is invalid")
    interfaces_payload = core["interfaces"]
    if type(interfaces_payload) is not dict:
        raise ValueError("shared physical interface table is invalid")
    interfaces = {}
    for interface_name, interface_payload in interfaces_payload.items():
        if type(interface_payload) is not dict or set(interface_payload) != {
            "interface",
            "models",
        }:
            raise ValueError("shared physical interface payload is invalid")
        models_payload = interface_payload["models"]
        if type(models_payload) is not dict:
            raise ValueError("shared physical model table is invalid")
        models = {}
        for model_name, model_payload in models_payload.items():
            if type(model_payload) is not dict or set(model_payload) != {
                "calibration",
                "metrics",
            }:
                raise ValueError("shared physical model payload is invalid")
            models[model_name] = ModelRealizabilityResult(
                _calibration_from_payload(model_payload["calibration"]),
                _metrics_from_payload(model_payload["metrics"]),
            )
        interfaces[interface_name] = InterfaceRealizabilityResult(
            interface_payload["interface"], models
        )
    result = PhysicalRealizabilityResult(
        system_name=core["system"],
        interfaces=interfaces,
        neural_hashes_before=core["neuralHashesBefore"],
        neural_hashes_after=core["neuralHashesAfter"],
        calibration_pairs_per_axis=core["calibrationPairsPerAxis"],
        heldout_pairs_per_axis=core["heldoutPairsPerAxis"],
    )
    if physical_result_sha256(result) != wrapper["physicalSha256"]:
        raise ValueError("reconstructed physical evidence differs from its SHA-256")
    return result


def _control_result_payload(result: ControlResult) -> dict[str, Any]:
    return {
        "errors": dict(result.errors),
        "interfaceName": result.interface_name,
        "episodes": result.episodes,
        "controlSteps": result.control_steps,
        "plannerBudget": dict(result.planner_budget),
        "targetSource": result.target_source,
        "episodeIdentifiers": result.episode_identifiers,
        "interfaceCommandTraces": dict(result.interface_command_traces),
        "plannerSeedScheduleSha256": result.planner_seed_schedule_sha256,
        "physicalProtocol": (
            None
            if result.physical_protocol is None
            else _protocol_payload(result.physical_protocol)
        ),
    }


def _control_result_from_payload(payload: Any) -> ControlResult:
    if type(payload) is not dict or set(payload) != {
        "errors",
        "interfaceName",
        "episodes",
        "controlSteps",
        "plannerBudget",
        "targetSource",
        "episodeIdentifiers",
        "interfaceCommandTraces",
        "plannerSeedScheduleSha256",
        "physicalProtocol",
    }:
        raise ValueError("control result payload schema is not exact")
    errors = {
        name: tuple(float(value) for value in values)
        for name, values in payload["errors"].items()
    }
    traces = {
        name: tuple(
            tuple(tuple(float(value) for value in command) for command in episode)
            for episode in episodes
        )
        for name, episodes in payload["interfaceCommandTraces"].items()
    }
    return ControlResult(
        errors=errors,
        interface_name=payload["interfaceName"],
        episodes=payload["episodes"],
        control_steps=payload["controlSteps"],
        planner_budget=payload["plannerBudget"],
        target_source=payload["targetSource"],
        episode_identifiers=tuple(payload["episodeIdentifiers"]),
        interface_command_traces=traces,
        planner_seed_schedule_sha256=payload["plannerSeedScheduleSha256"],
        physical_protocol=(
            None
            if payload["physicalProtocol"] is None
            else _protocol_from_payload(payload["physicalProtocol"])
        ),
    )


@dataclass(frozen=True)
class AuthenticatedControlShard:
    system_name: str
    interface_name: str
    start: int
    stop: int
    total_episodes: int
    episode_seed: int
    planner_seed: int
    training_lineage_sha256: str
    physical_sha256: str
    neural_hashes_before: Mapping[str, str]
    neural_hashes_after: Mapping[str, str]
    result: ControlResult
    trace_sha256: str
    artifact_sha256: str

    def core_payload(self) -> dict[str, Any]:
        return {
            "kind": "experiment_f_authenticated_control_shard_v1",
            "system": self.system_name,
            "interface": self.interface_name,
            "start": self.start,
            "stop": self.stop,
            "totalEpisodes": self.total_episodes,
            "episodeSeed": self.episode_seed,
            "plannerSeed": self.planner_seed,
            "trainingLineageSha256": self.training_lineage_sha256,
            "physicalSha256": self.physical_sha256,
            "neuralHashesBefore": dict(self.neural_hashes_before),
            "neuralHashesAfter": dict(self.neural_hashes_after),
            "traceSha256": self.trace_sha256,
            "result": _control_result_payload(self.result),
        }

    def __post_init__(self) -> None:
        hashes = (
            self.training_lineage_sha256,
            self.physical_sha256,
            self.trace_sha256,
            self.artifact_sha256,
            *self.neural_hashes_before.values(),
            *self.neural_hashes_after.values(),
        )
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
            raise ValueError("control shard contains a non-canonical SHA-256")
        if (
            self.system_name not in {"pendulum", "blocket"}
            or self.interface_name not in {"native", "unseen"}
            or not 0 <= self.start < self.stop <= self.total_episodes
            or self.total_episodes != 64
            or self.episode_seed != REGISTERED_CONTROL_EPISODE_SEED
            or self.planner_seed != REGISTERED_CONTROL_PLANNER_SEED
            or self.result.interface_name != self.interface_name
            or self.result.episodes != self.stop - self.start
            or tuple(self.result.errors) != REGISTERED_CONTROL_NAMES
            or not self.result.interface_command_traces
            or self.result.physical_protocol is None
            or self.result.physical_protocol.system_name != self.system_name
            or self.result.physical_protocol.interface_name != self.interface_name
            or dict(self.neural_hashes_before) != dict(self.neural_hashes_after)
            or control_trace_sha256(self.result) != self.trace_sha256
            or canonical_evidence_sha256(self.core_payload()) != self.artifact_sha256
        ):
            raise ValueError("authenticated control shard provenance is invalid")

    def to_payload(self) -> dict[str, Any]:
        return {**self.core_payload(), "artifactSha256": self.artifact_sha256}


def build_authenticated_control_shard(
    loaded: "LoadedPostFreezeSystem",
    physical: "PhysicalRealizabilityResult",
    shard: "ControlShard",
    *,
    episode_seed: int = REGISTERED_CONTROL_EPISODE_SEED,
    planner_seed: int = REGISTERED_CONTROL_PLANNER_SEED,
) -> AuthenticatedControlShard:
    loaded.assert_frozen_and_unchanged()
    values = {
        "system_name": loaded.system_name,
        "interface_name": shard.interface_name,
        "start": shard.start,
        "stop": shard.stop,
        "total_episodes": shard.total_episodes,
        "episode_seed": episode_seed,
        "planner_seed": planner_seed,
        "training_lineage_sha256": training_lineage_sha256(loaded),
        "physical_sha256": physical_result_sha256(physical),
        "neural_hashes_before": dict(physical.neural_hashes_before),
        "neural_hashes_after": dict(physical.neural_hashes_after),
        "result": shard.result,
        "trace_sha256": control_trace_sha256(shard.result),
        "artifact_sha256": "0" * 64,
    }
    provisional = AuthenticatedControlShard.__new__(AuthenticatedControlShard)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    values["artifact_sha256"] = canonical_evidence_sha256(
        provisional.core_payload()
    )
    return AuthenticatedControlShard(**values)


def save_control_shard(path: Path, shard: AuthenticatedControlShard) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(shard.to_payload(), temporary)
    temporary.replace(path)


def load_control_shard(path: Path) -> AuthenticatedControlShard:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "kind",
        "system",
        "interface",
        "start",
        "stop",
        "totalEpisodes",
        "episodeSeed",
        "plannerSeed",
        "trainingLineageSha256",
        "physicalSha256",
        "neuralHashesBefore",
        "neuralHashesAfter",
        "traceSha256",
        "result",
        "artifactSha256",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("authenticated control shard file schema is not exact")
    if payload["kind"] != "experiment_f_authenticated_control_shard_v1":
        raise ValueError("authenticated control shard kind mismatch")
    return AuthenticatedControlShard(
        system_name=payload["system"],
        interface_name=payload["interface"],
        start=payload["start"],
        stop=payload["stop"],
        total_episodes=payload["totalEpisodes"],
        episode_seed=payload["episodeSeed"],
        planner_seed=payload["plannerSeed"],
        training_lineage_sha256=payload["trainingLineageSha256"],
        physical_sha256=payload["physicalSha256"],
        neural_hashes_before=payload["neuralHashesBefore"],
        neural_hashes_after=payload["neuralHashesAfter"],
        result=_control_result_from_payload(payload["result"]),
        trace_sha256=payload["traceSha256"],
        artifact_sha256=payload["artifactSha256"],
    )


__all__ = [
    "AuthenticatedControlShard",
    "REGISTERED_CONTROL_EPISODE_SEED",
    "REGISTERED_CONTROL_NAMES",
    "REGISTERED_CONTROL_PLANNER_SEED",
    "build_authenticated_control_shard",
    "canonical_evidence_sha256",
    "load_control_shard",
    "load_physical_result",
    "physical_result_sha256",
    "save_control_shard",
    "save_physical_result",
    "training_lineage_sha256",
]
