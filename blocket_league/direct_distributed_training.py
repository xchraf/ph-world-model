"""Resumable, race-free distributed stages for Experiment F training.

The monolithic reference entry point remains useful on small configurations,
but the registered six-ablation run cannot reliably fit inside one Slurm GPU
wall-time.  This module splits the exact same computation into independently
scheduled stages while preserving the one-seed lineage:

``backbone -> {six variants, independent baseline} -> finalize``.

Only the pixels-only trainer mount is accepted.  No stage in this module opens
the producer seal or held-out test archive.  The finalizer emits
``training-complete.json`` only after validating every prerequisite checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .experiment_f_contract import (
    ExperimentFConfig,
    HiddenExcitationConfig,
    REGISTERED_VARIANTS,
    Variant,
    hidden_excitation_config_sha256,
)
from .direct_action_free_data import (
    ActionFreeBackboneTrainConfig,
    PixelsOnlyManifest,
    build_validated_action_free_backbone,
    class_weights,
    make_optimization_suite,
)
from .direct_cotangent_bridge import PixelChangeProbeBank
from .direct_jacobian_port_extractor import EmpiricalTangentArtifact
from .direct_jacobian_port_precompute import (
    JacobianPortPrecomputeConfig,
    build_empirical_tangent_from_pixels,
    load_empirical_tangent_artifact,
)
from .direct_experiment_training import (
    DIRECT_SYSTEMS,
    DirectModelBundle,
    DirectTrainingConfig,
    _named_optimized_parameters,
    _validate_direct_checkpoint,
    build_direct_bundle,
    seed_everything,
    train_direct_bundle,
)
from .direct_pixels_io import (
    load_sanitized_split,
    prepare_action_free_backbone,
)
from .direct_visual_poisson_ph import DirectVideoLossConfig
from .direct_unstructured_postfreeze import (
    FrozenIndependentBaseline,
    load_frozen_independent_baseline,
)
from .direct_unstructured_training import (
    build_fresh_independent_baseline,
    train_independent_unstructured_world_model,
)
from .learner_source_bundle import (
    build_learner_source_manifest,
    validate_code_free_cache,
    validate_learner_source_manifest,
    verify_learner_source_bundle,
)
from .pixel_direct_model import DirectPixelTransformer
from .tensor_provenance import module_tensor_hash
from .source_provenance import (
    build_source_manifest,
    load_source_manifest,
    validate_source_manifest_schema,
)
from .runtime_firewall_trace import RuntimeFirewallTrace, verify_runtime_trace


_CONFIG_KEYS = {
    "kind",
    "system",
    "experimentConfig",
    "backboneConfig",
    "portConfig",
    "directConfig",
    "baselineConfig",
    "lossConfig",
    "manifests",
    "sourceManifest",
    "learnerSourceManifest",
    "actionGradientUpdates",
    "physicalStateGradientUpdates",
}
_PROBE_KEYS = {
    "kind",
    "system",
    "configSha256",
    "dataSeal",
    "sourceProbeBasis",
    "sourceProbeHash",
    "registeredProbeHash",
}
_BACKBONE_COMPLETE_KEYS = {
    "kind",
    "system",
    "configSha256",
    "backboneHash",
    "backboneCheckpointSha256",
    "sourceProbeHash",
    "registeredProbeHash",
    "summary",
    "sourceTreeSha256",
}
_PORT_COMPLETE_KEYS = {
    "kind",
    "system",
    "configSha256",
    "backboneHash",
    "fitSanitizedTensorSha256",
    "artifactSha256",
    "sourceTreeSha256",
    "summary",
}
_VARIANT_SUMMARY_KEYS = {
    "system",
    "variant",
    "bestStep",
    "bestValidation",
    "bestStructureEligible",
    "seconds",
    "backboneHashBefore",
    "backboneHashAfter",
    "actionGradientUpdates",
    "physicalStateGradientUpdates",
    "trainableParameters",
    "sourceTreeSha256",
    "runtimeTrace",
}
_BASELINE_COMPLETE_KEYS = {
    "kind",
    "system",
    "configSha256",
    "backboneHash",
    "baselineCheckpointSha256",
    "referenceInitializationSeed",
    "summary",
    "sourceTreeSha256",
}


@dataclass(frozen=True)
class SealedDistributedConfig:
    system: str
    experiment: ExperimentFConfig
    backbone: ActionFreeBackboneTrainConfig
    port: JacobianPortPrecomputeConfig
    direct: DirectTrainingConfig
    baseline: DirectTrainingConfig
    loss: DirectVideoLossConfig
    manifests: Mapping[str, PixelsOnlyManifest]
    source_manifest: Mapping[str, Any]
    learner_source_manifest: Mapping[str, Any]
    sha256: str

    @property
    def source_tree_sha256(self) -> str:
        return str(self.source_manifest["treeSha256"])

    @property
    def learner_source_tree_sha256(self) -> str:
        return str(self.learner_source_manifest["treeSha256"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _plain_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain a plain JSON object")
    return value


def _strict_config(
    value: Any,
    config_type: type,
    *,
    label: str,
    tuple_fields: Sequence[str] = (),
) -> Any:
    if type(value) is not dict or set(value) != {
        item.name for item in fields(config_type)
    }:
        raise ValueError(f"{label} schema is not exact")
    defaults = config_type()
    normalized: dict[str, Any] = {}
    for item in fields(config_type):
        observed = value[item.name]
        expected = getattr(defaults, item.name)
        if item.name in tuple_fields:
            if type(observed) is not list or type(expected) is not tuple or not expected:
                raise ValueError(f"{label}.{item.name} must be a JSON array")
            expected_type = type(expected[0])
            if any(type(element) is not expected_type for element in observed):
                raise ValueError(f"{label}.{item.name} element type is invalid")
            normalized[item.name] = tuple(observed)
        else:
            if type(observed) is not type(expected):
                raise ValueError(f"{label}.{item.name} scalar type is invalid")
            if isinstance(observed, float) and not math.isfinite(observed):
                raise ValueError(f"{label}.{item.name} must be finite")
            normalized[item.name] = observed
    try:
        parsed = config_type(**normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} values are invalid") from error
    if asdict(parsed) != normalized:
        raise ValueError(f"{label} did not survive canonical parsing")
    return parsed


def _manifest(value: Any, *, split: str) -> PixelsOnlyManifest:
    if type(value) is not dict or set(value) != {
        item.name for item in fields(PixelsOnlyManifest)
    }:
        raise ValueError(f"{split} manifest schema is not exact")
    normalized = dict(value)
    for name in ("source_schema", "optimization_schema"):
        if type(normalized[name]) is not list:
            raise ValueError(f"{split} manifest {name} must be a JSON array")
        normalized[name] = tuple(normalized[name])
    manifest = PixelsOnlyManifest(**normalized)
    if manifest.source_schema != ("frames",) or manifest.optimization_schema != (
        "pixelContexts",
        "frames",
    ):
        raise ValueError(f"{split} manifest is not pixels-only")
    return manifest


def _data_seal(
    system: str, manifests: Mapping[str, PixelsOnlyManifest]
) -> dict[str, str]:
    return {
        "system": system,
        "fitAggregateSha256": manifests["fit"].aggregate_sha256,
        "fitSanitizedTensorSha256": manifests["fit"].sanitized_tensor_sha256,
        "validationAggregateSha256": manifests["validation"].aggregate_sha256,
        "validationSanitizedTensorSha256": manifests[
            "validation"
        ].sanitized_tensor_sha256,
    }


def _variant_loss(
    base: DirectVideoLossConfig, variant: Variant
) -> DirectVideoLossConfig:
    if variant == "no_jacobian":
        return replace(
            base,
            jacobian_bridge_weight=0.0,
            oddness_weight=0.0,
            manifold_cycle_weight=0.0,
        )
    if variant == "skew_only":
        return replace(base, chart_conditioning_weight=0.0)
    return base


def _trainer_archives(
    trainer_mount: Path,
    system: str,
    *,
    runtime_trace: RuntimeFirewallTrace | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, PixelsOnlyManifest]]:
    system_root = trainer_mount / system
    if not system_root.is_dir():
        raise FileNotFoundError(f"missing trainer mount for {system!r}")
    entries = {path.name for path in system_root.iterdir()}
    if entries != {"fit-pixels.pt", "validation-pixels.pt"}:
        raise ValueError(
            "trainer system mount must expose exactly fit-pixels.pt and "
            "validation-pixels.pt"
        )
    if runtime_trace is not None:
        runtime_trace.record_mount_manifest(trainer_mount, role="trainer_mount_root")
        runtime_trace.record_mount_manifest(
            system_root, role=f"trainer_mount_system:{system}"
        )
    pixels: dict[str, torch.Tensor] = {}
    manifests: dict[str, PixelsOnlyManifest] = {}
    for split in ("fit", "validation"):
        pixels[split], manifests[split] = load_sanitized_split(
            system_root / f"{split}-pixels.pt",
            expected_system=system,
            runtime_trace=runtime_trace,
            trace_role=f"trainer_archive:{split}",
        )
    return pixels, manifests


def _record_stage_configuration_reads(
    runtime_trace: RuntimeFirewallTrace,
    output_dir: Path,
    *,
    config_sha256: str,
    source_tree_sha256: str,
) -> None:
    config_path = output_dir / "distributed-config.json"
    config_payload = _plain_json(config_path)
    runtime_trace.record_file_read(
        config_path,
        role="sealed_distributed_config",
        serialized_keys=tuple(sorted(config_payload)),
        semantic_sha256=config_sha256,
    )
    source_path = output_dir / "source-manifest.json"
    if source_path.is_file():
        source_payload = _plain_json(source_path)
        runtime_trace.record_file_read(
            source_path,
            role="sealed_source_manifest",
            serialized_keys=tuple(sorted(source_payload)),
            semantic_sha256=source_tree_sha256,
        )


def _attest_learner_runtime(
    runtime_trace: RuntimeFirewallTrace | None,
    *,
    learner_bundle_root: Path | None,
    learner_source_manifest: Mapping[str, Any],
    source_tree_sha256: str,
    learner_cache_roots: Sequence[Path] = (),
) -> None:
    """Fail closed on learner-visible source/cache paths and trace exact bytes."""

    validate_learner_source_manifest(learner_source_manifest)
    if learner_source_manifest["fullSourceTreeSha256"] != source_tree_sha256:
        raise ValueError("learner source manifest has another full-tree anchor")
    if learner_bundle_root is None:
        if learner_cache_roots:
            raise ValueError("learner caches require an attested learner bundle")
        return
    observed = verify_learner_source_bundle(
        learner_bundle_root,
        expected_full_source_tree_sha256=source_tree_sha256,
    )
    if observed != learner_source_manifest:
        raise ValueError("mounted learner bundle differs from sealed configuration")
    for cache_root in learner_cache_roots:
        validate_code_free_cache(cache_root)
    if runtime_trace is None:
        return
    runtime_trace.record_file_read(
        learner_bundle_root / "learner-source-manifest.json",
        role="learner_source_manifest",
        serialized_keys=tuple(sorted(learner_source_manifest)),
        semantic_sha256=str(learner_source_manifest["treeSha256"]),
    )
    runtime_trace.record_recursive_manifest(
        learner_bundle_root,
        role="learner_source_bundle",
    )
    for index, cache_root in enumerate(learner_cache_roots):
        runtime_trace.record_recursive_manifest(
            cache_root,
            role=f"learner_cache:{index}",
        )


def _validate_dataset_against_experiment(
    experiment: ExperimentFConfig,
    manifests: Mapping[str, PixelsOnlyManifest],
) -> None:
    expected_counts = {
        "fit": experiment.fit_trajectories,
        "validation": experiment.validation_trajectories,
    }
    if set(manifests) != set(expected_counts):
        raise ValueError("trainer manifest table is incomplete")
    for split, expected_count in expected_counts.items():
        manifest = manifests[split]
        if manifest.trajectories != expected_count:
            raise ValueError(f"{split} trajectory count differs from configuration")
        if manifest.frames_per_trajectory != experiment.cache_frames:
            raise ValueError(f"{split} cache length differs from configuration")
        if manifest.image_size != experiment.image_size:
            raise ValueError(f"{split} image size differs from configuration")


def _config_payload(
    system: str,
    experiment: ExperimentFConfig,
    backbone: ActionFreeBackboneTrainConfig,
    port: JacobianPortPrecomputeConfig,
    direct: DirectTrainingConfig,
    baseline: DirectTrainingConfig,
    loss: DirectVideoLossConfig,
    manifests: Mapping[str, PixelsOnlyManifest],
    source_manifest: Mapping[str, Any],
    learner_source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if tuple(experiment.variants) != tuple(REGISTERED_VARIANTS):
        raise ValueError("distributed Experiment F requires all six variants exactly")
    return {
        "kind": "direct_distributed_training_config",
        "system": system,
        "experimentConfig": asdict(experiment),
        "backboneConfig": asdict(backbone),
        "portConfig": asdict(port),
        "directConfig": asdict(direct),
        "baselineConfig": asdict(baseline),
        "lossConfig": asdict(loss),
        "manifests": {
            split: asdict(manifests[split]) for split in ("fit", "validation")
        },
        "sourceManifest": dict(source_manifest),
        "learnerSourceManifest": dict(learner_source_manifest),
        "actionGradientUpdates": 0,
        "physicalStateGradientUpdates": 0,
    }


def _seal_config(
    output_dir: Path,
    system: str,
    experiment: ExperimentFConfig,
    backbone: ActionFreeBackboneTrainConfig,
    port: JacobianPortPrecomputeConfig,
    direct: DirectTrainingConfig,
    baseline: DirectTrainingConfig,
    loss: DirectVideoLossConfig,
    manifests: Mapping[str, PixelsOnlyManifest],
    source_manifest: Mapping[str, Any] | None = None,
    learner_source_manifest: Mapping[str, Any] | None = None,
) -> SealedDistributedConfig:
    _validate_dataset_against_experiment(experiment, manifests)
    if baseline != direct:
        raise ValueError(
            "independent baseline must use the exact direct training schedule"
        )
    if (
        port.lens_block != DIRECT_SYSTEMS[system].lens_block
        or port.horizons != direct.lens_horizons
        or port.channel_rank != direct.port_tangent_channel_rank
        or port.neighbors != direct.port_tangent_neighbors
        or port.support_floor_ratio != direct.port_support_floor_ratio
    ):
        raise ValueError("Jacobian port precompute and direct model configs differ")
    if source_manifest is None:
        source_manifest = build_source_manifest()
    source_tree_sha256 = validate_source_manifest_schema(source_manifest)
    if learner_source_manifest is None:
        learner_source_manifest, _ = build_learner_source_manifest(
            Path(__file__).resolve().parents[1], source_manifest
        )
    validate_learner_source_manifest(learner_source_manifest)
    if learner_source_manifest["fullSourceTreeSha256"] != source_tree_sha256:
        raise ValueError("learner source bundle is anchored to another source tree")
    payload = _config_payload(
        system,
        experiment,
        backbone,
        port,
        direct,
        baseline,
        loss,
        manifests,
        source_manifest,
        learner_source_manifest,
    )
    # Round-trip once so tuples have the exact representation that is hashed
    # and later parsed from disk.
    canonical = json.loads(json.dumps(payload, allow_nan=False))
    path = output_dir / "distributed-config.json"
    if path.exists():
        if _plain_json(path) != canonical:
            raise ValueError("distributed training configuration changed on resume")
    else:
        _atomic_json(path, canonical)
    return SealedDistributedConfig(
        system,
        experiment,
        backbone,
        port,
        direct,
        baseline,
        loss,
        dict(manifests),
        dict(source_manifest),
        dict(learner_source_manifest),
        _json_sha256(canonical),
    )


def load_sealed_config(output_dir: Path) -> SealedDistributedConfig:
    raw = _plain_json(output_dir / "distributed-config.json")
    if set(raw) != _CONFIG_KEYS or raw.get("kind") != "direct_distributed_training_config":
        raise ValueError("distributed configuration top-level schema is not exact")
    system = raw.get("system")
    if type(system) is not str or system not in DIRECT_SYSTEMS:
        raise ValueError("distributed configuration system is invalid")
    if raw.get("actionGradientUpdates") != 0 or raw.get(
        "physicalStateGradientUpdates"
    ) != 0:
        raise ValueError("distributed configuration admits a forbidden channel")
    experiment = _strict_config(
        raw["experimentConfig"],
        ExperimentFConfig,
        label="experimentConfig",
        tuple_fields=("variants",),
    )
    if tuple(experiment.variants) != tuple(REGISTERED_VARIANTS):
        raise ValueError("distributed configuration does not contain all six variants")
    backbone = _strict_config(
        raw["backboneConfig"],
        ActionFreeBackboneTrainConfig,
        label="backboneConfig",
    )
    port = _strict_config(
        raw["portConfig"],
        JacobianPortPrecomputeConfig,
        label="portConfig",
        tuple_fields=("horizons",),
    )
    direct = _strict_config(
        raw["directConfig"],
        DirectTrainingConfig,
        label="directConfig",
        tuple_fields=("lens_horizons",),
    )
    baseline = _strict_config(
        raw["baselineConfig"],
        DirectTrainingConfig,
        label="baselineConfig",
        tuple_fields=("lens_horizons",),
    )
    if baseline != direct:
        raise ValueError("baselineConfig differs from directConfig")
    loss = _strict_config(
        raw["lossConfig"],
        DirectVideoLossConfig,
        label="lossConfig",
        tuple_fields=("rollout_horizons",),
    )
    manifest_table = raw.get("manifests")
    if type(manifest_table) is not dict or set(manifest_table) != {
        "fit",
        "validation",
    }:
        raise ValueError("distributed configuration manifest table is incomplete")
    manifests = {
        split: _manifest(manifest_table[split], split=split)
        for split in ("fit", "validation")
    }
    if any(item.system != system for item in manifests.values()):
        raise ValueError("distributed configuration contains another system manifest")
    _validate_dataset_against_experiment(experiment, manifests)
    source_manifest = raw.get("sourceManifest")
    source_tree_sha256 = validate_source_manifest_schema(source_manifest)
    learner_source_manifest = raw.get("learnerSourceManifest")
    validate_learner_source_manifest(learner_source_manifest)
    if learner_source_manifest["fullSourceTreeSha256"] != source_tree_sha256:
        raise ValueError("learner source bundle source-tree anchor is invalid")
    return SealedDistributedConfig(
        system,
        experiment,
        backbone,
        port,
        direct,
        baseline,
        loss,
        manifests,
        source_manifest,
        learner_source_manifest,
        _json_sha256(raw),
    )


def _assert_archives_match_config(
    observed: Mapping[str, PixelsOnlyManifest], config: SealedDistributedConfig
) -> None:
    if set(observed) != {"fit", "validation"}:
        raise ValueError("observed trainer manifest table is incomplete")
    for split in ("fit", "validation"):
        if asdict(observed[split]) != asdict(config.manifests[split]):
            raise ValueError(f"{split} trainer archive changed after config sealing")


def _write_or_validate_probes(
    output_dir: Path,
    config: SealedDistributedConfig,
    fit_pixels: torch.Tensor,
    model_config: Any,
) -> tuple[str, str]:
    suite = make_optimization_suite(
        fit_pixels, model_config, transitions=config.experiment.transitions
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.experiment.seed + 71)
        source = PixelChangeProbeBank.from_pixel_frames(
            suite["frames"],
            palette_size=model_config.palette_size,
            probe_size=DIRECT_SYSTEMS[config.system].port_size,
        )
    registered = PixelChangeProbeBank(source.basis.detach().clone())
    payload = {
        "kind": "pixels_only_shared_probe_bank",
        "system": config.system,
        "configSha256": config.sha256,
        "dataSeal": _data_seal(config.system, config.manifests),
        "sourceProbeBasis": source.basis.detach().cpu().clone(),
        "sourceProbeHash": module_tensor_hash(source),
        "registeredProbeHash": module_tensor_hash(registered),
    }
    path = output_dir / "shared" / "probes.pt"
    if path.exists():
        observed = torch.load(path, map_location="cpu", weights_only=True)
        if type(observed) is not dict or set(observed) != _PROBE_KEYS:
            raise ValueError("existing shared probe checkpoint schema is invalid")
        if any(
            observed[name] != payload[name]
            for name in _PROBE_KEYS - {"sourceProbeBasis"}
        ) or not torch.equal(observed["sourceProbeBasis"], payload["sourceProbeBasis"]):
            raise ValueError("shared pixels-only probes changed on resume")
    else:
        _atomic_torch_save(path, payload)
    return str(payload["sourceProbeHash"]), str(payload["registeredProbeHash"])


def _load_registered_probes(
    output_dir: Path,
    config: SealedDistributedConfig,
    runtime_trace: RuntimeFirewallTrace | None = None,
) -> PixelChangeProbeBank:
    path = output_dir / "shared" / "probes.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if runtime_trace is not None:
        runtime_trace.record_file_read(
            path,
            role="registered_probe_checkpoint",
            serialized_keys=tuple(sorted(payload)) if type(payload) is dict else (),
        )
    if type(payload) is not dict or set(payload) != _PROBE_KEYS:
        raise ValueError("shared probe checkpoint schema is not exact")
    if (
        payload["kind"] != "pixels_only_shared_probe_bank"
        or payload["system"] != config.system
        or payload["configSha256"] != config.sha256
        or payload["dataSeal"] != _data_seal(config.system, config.manifests)
    ):
        raise ValueError("shared probe checkpoint provenance mismatch")
    basis = payload["sourceProbeBasis"]
    if type(basis) is not torch.Tensor or basis.ndim != 4 or not bool(
        torch.isfinite(basis).all()
    ):
        raise ValueError("shared source probe basis is invalid")
    source = PixelChangeProbeBank(basis)
    source.load_state_dict({"basis": basis}, strict=True)
    if module_tensor_hash(source) != payload["sourceProbeHash"]:
        raise ValueError("shared source probe hash mismatch")
    registered = PixelChangeProbeBank(source.basis.detach().clone())
    if module_tensor_hash(registered) != payload["registeredProbeHash"]:
        raise ValueError("shared registered probe hash mismatch")
    return registered.eval().requires_grad_(False)


def _load_backbone(
    output_dir: Path,
    config: SealedDistributedConfig,
    device: torch.device,
    runtime_trace: RuntimeFirewallTrace | None = None,
) -> nn.Module:
    path = output_dir / "backbone" / "checkpoint.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if runtime_trace is not None:
        runtime_trace.record_file_read(
            path,
            role="frozen_backbone_checkpoint",
            serialized_keys=tuple(sorted(payload)) if type(payload) is dict else (),
        )
    if type(payload) is not dict or payload.get("train_config") != asdict(
        config.backbone
    ):
        raise ValueError("backbone checkpoint training configuration mismatch")
    model = build_validated_action_free_backbone(
        payload,
        expected_manifest_sha256=config.manifests["fit"].aggregate_sha256,
        expected_sanitized_tensor_sha256=config.manifests[
            "fit"
        ].sanitized_tensor_sha256,
        expected_system=config.system,
    ).to(device)
    return model.eval().requires_grad_(False)


def run_backbone_stage(
    trainer_mount: Path,
    output_dir: Path,
    *,
    system: str,
    experiment: ExperimentFConfig,
    backbone_config: ActionFreeBackboneTrainConfig,
    port_config: JacobianPortPrecomputeConfig,
    direct_config: DirectTrainingConfig,
    baseline_config: DirectTrainingConfig,
    loss_config: DirectVideoLossConfig,
    device: torch.device,
    source_manifest: Mapping[str, Any] | None = None,
    learner_source_manifest: Mapping[str, Any] | None = None,
    learner_bundle_root: Path | None = None,
    learner_cache_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    if system not in DIRECT_SYSTEMS:
        raise KeyError(f"unknown system {system!r}")
    effective_source_manifest = (
        build_source_manifest() if source_manifest is None else dict(source_manifest)
    )
    source_tree_sha256 = validate_source_manifest_schema(
        effective_source_manifest
    )
    if learner_source_manifest is None:
        learner_source_manifest, _ = build_learner_source_manifest(
            Path(__file__).resolve().parents[1], effective_source_manifest
        )
    completion_path = output_dir / "backbone-complete.json"
    runtime_trace = None
    if not completion_path.exists():
        runtime_trace = RuntimeFirewallTrace(
            output_dir / "backbone" / "firewall-trace.jsonl",
            stage=f"backbone:{system}",
            source_tree_sha256=source_tree_sha256,
        )
    _attest_learner_runtime(
        runtime_trace,
        learner_bundle_root=learner_bundle_root,
        learner_source_manifest=learner_source_manifest,
        source_tree_sha256=source_tree_sha256,
        learner_cache_roots=learner_cache_roots,
    )
    pixels, manifests = _trainer_archives(
        trainer_mount, system, runtime_trace=runtime_trace
    )
    config = _seal_config(
        output_dir,
        system,
        experiment,
        backbone_config,
        port_config,
        direct_config,
        baseline_config,
        loss_config,
        manifests,
        effective_source_manifest,
        learner_source_manifest,
    )
    if runtime_trace is not None:
        _record_stage_configuration_reads(
            runtime_trace,
            output_dir,
            config_sha256=config.sha256,
            source_tree_sha256=config.source_tree_sha256,
        )
    seed_everything(experiment.seed)
    backbone, backbone_summary = prepare_action_free_backbone(
        system,
        pixels["fit"],
        manifests["fit"],
        output_dir / "backbone",
        experiment,
        backbone_config,
        device,
        runtime_trace=runtime_trace,
        source_tree_sha256=config.source_tree_sha256,
    )
    if runtime_trace is not None:
        backbone_summary = {
            **dict(backbone_summary),
            "runtimeTrace": runtime_trace.snapshot().to_dict(),
        }
        _atomic_json(output_dir / "backbone" / "summary.json", backbone_summary)
        runtime_trace.close()
    backbone = backbone.to(device).eval().requires_grad_(False)
    source_hash, registered_hash = _write_or_validate_probes(
        output_dir, config, pixels["fit"], backbone.config
    )
    checkpoint_path = output_dir / "backbone" / "checkpoint.pt"
    verify_runtime_trace(
        output_dir / "backbone" / "firewall-trace.jsonl",
        backbone_summary.get("runtimeTrace"),
    )
    completion = {
        "kind": "direct_distributed_backbone_complete",
        "system": system,
        "configSha256": config.sha256,
        "backboneHash": module_tensor_hash(backbone),
        "backboneCheckpointSha256": _file_sha256(checkpoint_path),
        "sourceProbeHash": source_hash,
        "registeredProbeHash": registered_hash,
        "summary": backbone_summary,
        "sourceTreeSha256": config.source_tree_sha256,
    }
    path = completion_path
    if path.exists() and _plain_json(path) != completion:
        # A resumed prepare call reports only ``resumed=True``.  Preserve the
        # original completed summary, but require every immutable seal to agree.
        observed = _plain_json(path)
        if set(observed) != _BACKBONE_COMPLETE_KEYS or any(
            observed[name] != completion[name]
            for name in _BACKBONE_COMPLETE_KEYS - {"summary"}
        ):
            raise ValueError("backbone completion seal changed on resume")
        completion = observed
    else:
        _atomic_json(path, completion)
    return completion


def _load_empirical_tangent(
    output_dir: Path,
    config: SealedDistributedConfig,
    backbone: DirectPixelTransformer,
    runtime_trace: RuntimeFirewallTrace | None = None,
) -> EmpiricalTangentArtifact:
    completion = _plain_json(output_dir / "port-precompute-complete.json")
    artifact_path = output_dir / "port-precompute" / "empirical-tangent.pt"
    if (
        set(completion) != _PORT_COMPLETE_KEYS
        or completion["kind"] != "direct_empirical_jacobian_port_precompute_complete"
        or completion["system"] != config.system
        or completion["configSha256"] != config.sha256
        or completion["backboneHash"] != module_tensor_hash(backbone)
        or completion["fitSanitizedTensorSha256"]
        != config.manifests["fit"].sanitized_tensor_sha256
        or completion["artifactSha256"] != _file_sha256(artifact_path)
        or completion["sourceTreeSha256"] != config.source_tree_sha256
    ):
        raise ValueError("empirical Jacobian port completion seal is invalid")
    return load_empirical_tangent_artifact(
        artifact_path,
        expected_system=config.system,
        expected_fit_sanitized_tensor_sha256=config.manifests[
            "fit"
        ].sanitized_tensor_sha256,
        expected_source_tree_sha256=config.source_tree_sha256,
        expected_backbone_hash=module_tensor_hash(backbone),
        expected_config=config.port,
        runtime_trace=runtime_trace,
    )


def run_port_precompute_stage(
    trainer_mount: Path,
    output_dir: Path,
    *,
    system: str,
    device: torch.device,
    learner_bundle_root: Path | None = None,
    learner_cache_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Extract the zero-parameter Jacobian tangent before any pH model exists."""

    config = load_sealed_config(output_dir)
    if config.system != system:
        raise ValueError("port-precompute job system differs from distributed config")
    completion_path = output_dir / "port-precompute-complete.json"
    runtime_trace = None
    if not completion_path.exists():
        runtime_trace = RuntimeFirewallTrace(
            output_dir / "port-precompute" / "firewall-trace.jsonl",
            stage=f"jacobian-port-precompute:{system}",
            source_tree_sha256=config.source_tree_sha256,
        )
    _attest_learner_runtime(
        runtime_trace,
        learner_bundle_root=learner_bundle_root,
        learner_source_manifest=config.learner_source_manifest,
        source_tree_sha256=config.source_tree_sha256,
        learner_cache_roots=learner_cache_roots,
    )
    pixels, manifests = _trainer_archives(
        trainer_mount, system, runtime_trace=runtime_trace
    )
    _assert_archives_match_config(manifests, config)
    backbone = _load_backbone(output_dir, config, device, runtime_trace)
    if completion_path.exists():
        _load_empirical_tangent(output_dir, config, backbone, runtime_trace)
        return _plain_json(completion_path)
    assert runtime_trace is not None
    _record_stage_configuration_reads(
        runtime_trace,
        output_dir,
        config_sha256=config.sha256,
        source_tree_sha256=config.source_tree_sha256,
    )
    runtime_trace.record_backbone_boundary(
        phase=f"jacobian-port-precompute:{system}",
        boundary="selected_checkpoint",
        sha256=module_tensor_hash(backbone),
    )
    fit_suite = make_optimization_suite(
        pixels["fit"],
        backbone.config,
        transitions=config.experiment.transitions,
    )
    _, summary = build_empirical_tangent_from_pixels(
        backbone,
        fit_suite,
        system=system,
        fit_sanitized_tensor_sha256=manifests["fit"].sanitized_tensor_sha256,
        output_dir=output_dir / "port-precompute",
        device=device,
        config=config.port,
        runtime_trace=runtime_trace,
        source_tree_sha256=config.source_tree_sha256,
    )
    summary = {**summary, "runtimeTrace": runtime_trace.snapshot().to_dict()}
    _atomic_json(output_dir / "port-precompute" / "summary.json", summary)
    runtime_trace.close()
    artifact_path = output_dir / "port-precompute" / "empirical-tangent.pt"
    completion = {
        "kind": "direct_empirical_jacobian_port_precompute_complete",
        "system": system,
        "configSha256": config.sha256,
        "backboneHash": module_tensor_hash(backbone),
        "fitSanitizedTensorSha256": manifests["fit"].sanitized_tensor_sha256,
        "artifactSha256": _file_sha256(artifact_path),
        "sourceTreeSha256": config.source_tree_sha256,
        "summary": summary,
    }
    _atomic_json(completion_path, completion)
    return completion


def _validate_variant_summary(
    summary: Any,
    payload: Mapping[str, Any],
    *,
    system: str,
    variant: Variant,
    backbone_hash: str,
    expected_trainable_parameters: int,
    source_tree_sha256: str,
    trace_path: Path,
) -> dict[str, Any]:
    if type(summary) is not dict or set(summary) != _VARIANT_SUMMARY_KEYS:
        raise ValueError(f"{variant} training summary schema is not exact")
    if (
        summary["system"] != system
        or summary["variant"] != variant
        or summary["bestStep"] != payload["step"]
        or float(summary["bestValidation"]) != float(payload["bestValidation"])
        or summary["bestStructureEligible"] is not payload["bestStructureEligible"]
        or summary["backboneHashBefore"] != backbone_hash
        or summary["backboneHashAfter"] != backbone_hash
        or summary["actionGradientUpdates"] != 0
        or summary["physicalStateGradientUpdates"] != 0
        or type(summary["seconds"]) not in (int, float)
        or not math.isfinite(float(summary["seconds"]))
        or float(summary["seconds"]) < 0.0
        or type(expected_trainable_parameters) is not int
        or expected_trainable_parameters < 1
        or summary["trainableParameters"] != expected_trainable_parameters
        or summary["sourceTreeSha256"] != source_tree_sha256
    ):
        raise ValueError(f"{variant} training summary provenance mismatch")
    records = verify_runtime_trace(trace_path, summary["runtimeTrace"])
    if not records or any(
        record["payload"].get("sourceTreeSha256") != source_tree_sha256
        for record in records
        if record["event"] == "stage_boundary"
    ):
        raise ValueError(f"{variant} runtime trace source provenance mismatch")
    return dict(summary)


def _runtime_trace_entry(
    output_dir: Path,
    *,
    phase: str,
    relative_path: str,
    seal: Any,
    source_tree_sha256: str,
) -> dict[str, Any]:
    path = output_dir / relative_path
    records = verify_runtime_trace(path, seal)
    boundaries = [
        record for record in records if record["event"] == "stage_boundary"
    ]
    if not boundaries or any(
        record["payload"].get("sourceTreeSha256") != source_tree_sha256
        for record in boundaries
    ):
        raise ValueError(f"runtime trace {phase!r} has invalid source boundaries")
    return {
        "phase": phase,
        "relativePath": relative_path,
        "seal": dict(seal),
    }


def _validate_best_checkpoint(
    path: Path,
    bundle: DirectModelBundle,
    config: SealedDistributedConfig,
    variant: Variant,
    runtime_trace: RuntimeFirewallTrace | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if runtime_trace is not None:
        runtime_trace.record_file_read(
            path,
            role=f"direct_checkpoint:{variant}",
            serialized_keys=tuple(sorted(payload)) if type(payload) is dict else (),
        )
    if type(payload) is not dict:
        raise ValueError(f"{variant} best checkpoint is not a plain dictionary")
    _validate_direct_checkpoint(
        payload,
        bundle,
        variant=variant,
        system=DIRECT_SYSTEMS[config.system],
        train_config=config.direct,
        loss_config=_variant_loss(config.loss, variant),
        data_seal=_data_seal(config.system, config.manifests),
        source_tree_sha256=config.source_tree_sha256,
        include_training_state=False,
    )
    return payload


def _load_best_into_bundle(
    path: Path,
    bundle: DirectModelBundle,
    config: SealedDistributedConfig,
    variant: Variant,
    runtime_trace: RuntimeFirewallTrace | None = None,
) -> dict[str, Any]:
    payload = _validate_best_checkpoint(
        path, bundle, config, variant, runtime_trace
    )
    bundle.model.load_state_dict(payload["model"], strict=True)
    bundle.write_field.load_state_dict(payload["writeField"], strict=True)
    bundle.response_frame.load_state_dict(payload["responseFrame"], strict=True)
    bundle.cotangent_frame.load_state_dict(payload["cotangentFrame"], strict=True)
    bundle.probes.load_state_dict(payload["probes"], strict=True)
    for module in (
        bundle.model,
        bundle.write_field,
        bundle.response_frame,
        bundle.cotangent_frame,
        bundle.probes,
    ):
        module.eval().requires_grad_(False)
    bundle.model.encoder.assert_backbone_frozen()
    return payload


def _trainable_parameter_count(bundle: DirectModelBundle) -> int:
    return sum(parameter.numel() for _, parameter in _named_optimized_parameters(bundle))


def _repair_finished_variant_summary(
    output_dir: Path,
    bundle: DirectModelBundle,
    config: SealedDistributedConfig,
    variant: Variant,
    best: Mapping[str, Any],
    runtime_trace: RuntimeFirewallTrace,
) -> dict[str, Any] | None:
    last_path = output_dir / "direct" / variant / "last.pt"
    if not last_path.exists():
        return None
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    runtime_trace.record_file_read(
        last_path,
        role=f"direct_resume_checkpoint:{variant}",
        serialized_keys=tuple(sorted(last)) if type(last) is dict else (),
    )
    if type(last) is not dict or last.get("step") != config.direct.steps:
        return None
    parameters = [parameter for _, parameter in _named_optimized_parameters(bundle)]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.direct.learning_rate,
        weight_decay=config.direct.weight_decay,
    )
    _validate_direct_checkpoint(
        last,
        bundle,
        variant=variant,
        system=DIRECT_SYSTEMS[config.system],
        train_config=config.direct,
        loss_config=_variant_loss(config.loss, variant),
        data_seal=_data_seal(config.system, config.manifests),
        source_tree_sha256=config.source_tree_sha256,
        include_training_state=True,
        optimizer=optimizer,
    )
    runtime_trace.record_backbone_boundary(
        phase=f"direct:{variant}",
        boundary="repaired_selected_checkpoint",
        sha256=bundle.model.encoder.sealed_backbone_hash,
    )
    runtime_trace_seal = runtime_trace.snapshot().to_dict()
    summary = {
        "system": config.system,
        "variant": variant,
        "bestStep": int(best["step"]),
        "bestValidation": float(best["bestValidation"]),
        "bestStructureEligible": bool(best["bestStructureEligible"]),
        "seconds": 0.0,
        "backboneHashBefore": bundle.model.encoder.sealed_backbone_hash,
        "backboneHashAfter": bundle.model.encoder.sealed_backbone_hash,
        "actionGradientUpdates": 0,
        "physicalStateGradientUpdates": 0,
        "trainableParameters": _trainable_parameter_count(bundle),
        "sourceTreeSha256": config.source_tree_sha256,
        "runtimeTrace": runtime_trace_seal,
    }
    _atomic_json(output_dir / "direct" / variant / "training-summary.json", summary)
    return summary


def run_variant_stage(
    trainer_mount: Path,
    output_dir: Path,
    *,
    system: str,
    variant: Variant,
    device: torch.device,
    learner_bundle_root: Path | None = None,
    learner_cache_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    if variant not in REGISTERED_VARIANTS:
        raise KeyError(f"unknown registered variant {variant!r}")
    config = load_sealed_config(output_dir)
    if config.system != system:
        raise ValueError("variant job system differs from distributed config")
    variant_dir = output_dir / "direct" / variant
    best_path = variant_dir / "best.pt"
    summary_path = variant_dir / "training-summary.json"
    already_complete = best_path.exists() and summary_path.exists()
    runtime_trace = None
    if not already_complete:
        runtime_trace = RuntimeFirewallTrace(
            variant_dir / "firewall-trace.jsonl",
            stage=f"direct:{system}:{variant}",
            source_tree_sha256=config.source_tree_sha256,
        )
    _attest_learner_runtime(
        runtime_trace,
        learner_bundle_root=learner_bundle_root,
        learner_source_manifest=config.learner_source_manifest,
        source_tree_sha256=config.source_tree_sha256,
        learner_cache_roots=learner_cache_roots,
    )
    if runtime_trace is not None:
        _record_stage_configuration_reads(
            runtime_trace,
            output_dir,
            config_sha256=config.sha256,
            source_tree_sha256=config.source_tree_sha256,
        )
    pixels, manifests = _trainer_archives(
        trainer_mount, system, runtime_trace=runtime_trace
    )
    _assert_archives_match_config(manifests, config)
    backbone = _load_backbone(output_dir, config, device, runtime_trace)
    empirical_tangent = _load_empirical_tangent(
        output_dir, config, backbone, runtime_trace
    )
    probes = _load_registered_probes(output_dir, config, runtime_trace)
    seed_everything(config.experiment.seed + 10_003)
    bundle = build_direct_bundle(
        backbone,
        DIRECT_SYSTEMS[system],
        probes,
        config.direct,
        device,
        empirical_tangent=empirical_tangent,
        variant=variant,
    )
    if best_path.exists():
        best = _validate_best_checkpoint(
            best_path, bundle, config, variant, runtime_trace
        )
        if summary_path.exists():
            return _validate_variant_summary(
                _plain_json(summary_path),
                best,
                system=system,
                variant=variant,
                backbone_hash=module_tensor_hash(backbone),
                expected_trainable_parameters=_trainable_parameter_count(bundle),
                source_tree_sha256=config.source_tree_sha256,
                trace_path=variant_dir / "firewall-trace.jsonl",
            )
        repaired = _repair_finished_variant_summary(
            output_dir, bundle, config, variant, best, runtime_trace
        )
        if repaired is not None:
            runtime_trace.close()
            return _validate_variant_summary(
                repaired,
                best,
                system=system,
                variant=variant,
                backbone_hash=module_tensor_hash(backbone),
                expected_trainable_parameters=_trainable_parameter_count(bundle),
                source_tree_sha256=config.source_tree_sha256,
                trace_path=variant_dir / "firewall-trace.jsonl",
            )
    fit_suite = make_optimization_suite(
        pixels["fit"], backbone.config, transitions=config.experiment.transitions
    )
    validation_suite = make_optimization_suite(
        pixels["validation"],
        backbone.config,
        transitions=config.experiment.transitions,
    )
    weights = class_weights(fit_suite["frames"], backbone.config.palette_size, device)
    summary = train_direct_bundle(
        bundle,
        fit_suite,
        validation_suite,
        weights,
        DIRECT_SYSTEMS[system],
        variant_dir,
        config.direct,
        config.loss,
        variant=variant,
        data_seal=_data_seal(system, manifests),
        source_tree_sha256=config.source_tree_sha256,
        runtime_trace=runtime_trace,
    )
    runtime_trace.close()
    best = _validate_best_checkpoint(best_path, bundle, config, variant)
    return _validate_variant_summary(
        summary,
        best,
        system=system,
        variant=variant,
        backbone_hash=module_tensor_hash(backbone),
        expected_trainable_parameters=_trainable_parameter_count(bundle),
        source_tree_sha256=config.source_tree_sha256,
        trace_path=variant_dir / "firewall-trace.jsonl",
    )


def _validate_module_state(name: str, state: Any, module: nn.Module) -> None:
    reference = module.state_dict()
    if type(state) is not dict or set(state) != set(reference):
        raise ValueError(f"{name} state schema mismatch")
    for field, tensor in state.items():
        expected = reference[field]
        if (
            type(tensor) is not torch.Tensor
            or tensor.shape != expected.shape
            or tensor.dtype != expected.dtype
            or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
        ):
            raise ValueError(f"{name}.{field} tensor is invalid")



def _load_independent_baseline_artifact(
    output_dir: Path,
    config: SealedDistributedConfig,
    backbone: nn.Module,
    probes: PixelChangeProbeBank,
    device: torch.device,
) -> tuple[FrozenIndependentBaseline, dict[str, Any]]:
    baseline_dir = output_dir / "baseline-independent"
    if not (baseline_dir / "last.pt").is_file():
        raise FileNotFoundError("independent baseline last.pt is missing")
    summary = _plain_json(baseline_dir / "summary.json")
    empirical_tangent = _load_empirical_tangent(
        output_dir, config, backbone  # type: ignore[arg-type]
    )
    frozen = load_frozen_independent_baseline(
        backbone=backbone,  # type: ignore[arg-type]
        system=DIRECT_SYSTEMS[config.system],
        probes=probes,
        empirical_tangent=empirical_tangent,
        train_config=config.direct,
        loss_config=config.loss,
        checkpoint_path=baseline_dir / "best.pt",
        summary=summary,
        data_seal=_data_seal(config.system, config.manifests),
        source_tree_sha256=config.source_tree_sha256,
        reference_initialization_seed=config.experiment.seed + 10_003,
        device=device,
    )
    verify_runtime_trace(
        baseline_dir / "firewall-trace.jsonl", summary.get("runtimeTrace")
    )
    return frozen, summary


def run_baseline_stage(
    trainer_mount: Path,
    output_dir: Path,
    *,
    system: str,
    device: torch.device,
    learner_bundle_root: Path | None = None,
    learner_cache_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Train the independent pixels-only Jacobian-lens world model."""

    config = load_sealed_config(output_dir)
    if config.system != system:
        raise ValueError("baseline job system differs from distributed config")
    baseline_dir = output_dir / "baseline-independent"
    checkpoint_path = baseline_dir / "best.pt"
    summary_path = baseline_dir / "summary.json"
    completion_path = output_dir / "baseline-complete.json"
    already_complete = (
        checkpoint_path.is_file()
        and (baseline_dir / "last.pt").is_file()
        and summary_path.is_file()
    )
    runtime_trace = None
    if not already_complete:
        runtime_trace = RuntimeFirewallTrace(
            baseline_dir / "firewall-trace.jsonl",
            stage=f"baseline:{system}:independent_unstructured",
            source_tree_sha256=config.source_tree_sha256,
        )
    _attest_learner_runtime(
        runtime_trace,
        learner_bundle_root=learner_bundle_root,
        learner_source_manifest=config.learner_source_manifest,
        source_tree_sha256=config.source_tree_sha256,
        learner_cache_roots=learner_cache_roots,
    )
    if runtime_trace is not None:
        _record_stage_configuration_reads(
            runtime_trace,
            output_dir,
            config_sha256=config.sha256,
            source_tree_sha256=config.source_tree_sha256,
        )
    pixels, manifests = _trainer_archives(
        trainer_mount, system, runtime_trace=runtime_trace
    )
    _assert_archives_match_config(manifests, config)
    backbone = _load_backbone(output_dir, config, device, runtime_trace)
    empirical_tangent = _load_empirical_tangent(
        output_dir, config, backbone, runtime_trace
    )
    probes = _load_registered_probes(output_dir, config, runtime_trace)
    if not already_complete:
        bundle = build_fresh_independent_baseline(
            backbone,
            DIRECT_SYSTEMS[system],
            probes,
            config.direct,
            device,
            empirical_tangent=empirical_tangent,
            reference_initialization_seed=config.experiment.seed + 10_003,
        )
        fit_suite = make_optimization_suite(
            pixels["fit"],
            backbone.config,  # type: ignore[attr-defined]
            transitions=config.experiment.transitions,
        )
        validation_suite = make_optimization_suite(
            pixels["validation"],
            backbone.config,  # type: ignore[attr-defined]
            transitions=config.experiment.transitions,
        )
        weights = class_weights(
            fit_suite["frames"],
            backbone.config.palette_size,  # type: ignore[attr-defined]
            device,
        )
        summary = train_independent_unstructured_world_model(
            bundle,
            fit_suite,
            validation_suite,
            weights,
            DIRECT_SYSTEMS[system],
            baseline_dir,
            config.direct,
            config.loss,
            data_seal=_data_seal(system, manifests),
            pixel_archive_paths={
                split: trainer_mount / system / f"{split}-pixels.pt"
                for split in ("fit", "validation")
            },
            source_tree_sha256=config.source_tree_sha256,
            runtime_trace=runtime_trace,
        )
        if runtime_trace is None:  # pragma: no cover - construction invariant
            raise AssertionError("independent baseline trace was not created")
        runtime_trace.close()
    frozen, summary = _load_independent_baseline_artifact(
        output_dir, config, backbone, probes, device
    )
    completion = {
        "kind": "direct_distributed_independent_baseline_complete",
        "system": system,
        "configSha256": config.sha256,
        "backboneHash": module_tensor_hash(backbone),
        "baselineCheckpointSha256": frozen.checkpoint_sha256,
        "referenceInitializationSeed": config.experiment.seed + 10_003,
        "summary": summary,
        "sourceTreeSha256": config.source_tree_sha256,
    }
    if completion_path.exists() and _plain_json(completion_path) != completion:
        raise ValueError("independent baseline completion lineage changed")
    _atomic_json(completion_path, completion)
    frozen.assert_frozen_and_unchanged()
    return summary


def finalize_training_complete(
    trainer_mount: Path,
    output_dir: Path,
    *,
    system: str,
    device: torch.device = torch.device("cpu"),
    learner_bundle_root: Path | None = None,
    learner_cache_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    config = load_sealed_config(output_dir)
    _attest_learner_runtime(
        None,
        learner_bundle_root=learner_bundle_root,
        learner_source_manifest=config.learner_source_manifest,
        source_tree_sha256=config.source_tree_sha256,
        learner_cache_roots=learner_cache_roots,
    )
    if config.system != system:
        raise ValueError("finalizer system differs from distributed config")
    _, manifests = _trainer_archives(trainer_mount, system)
    _assert_archives_match_config(manifests, config)
    backbone = _load_backbone(output_dir, config, device)
    empirical_tangent = _load_empirical_tangent(output_dir, config, backbone)
    backbone_hash = module_tensor_hash(backbone)
    completion = _plain_json(output_dir / "backbone-complete.json")
    if set(completion) != _BACKBONE_COMPLETE_KEYS or (
        completion["system"] != system
        or completion["configSha256"] != config.sha256
        or completion["backboneHash"] != backbone_hash
        or completion["sourceTreeSha256"] != config.source_tree_sha256
        or completion["backboneCheckpointSha256"]
        != _file_sha256(output_dir / "backbone" / "checkpoint.pt")
    ):
        raise ValueError("backbone completion seal is invalid")
    variant_summaries: dict[str, Any] = {}
    for raw_variant in REGISTERED_VARIANTS:
        variant: Variant = raw_variant
        probes = _load_registered_probes(output_dir, config)
        seed_everything(config.experiment.seed + 10_003)
        bundle = build_direct_bundle(
            backbone,
            DIRECT_SYSTEMS[system],
            probes,
            config.direct,
            device,
            empirical_tangent=empirical_tangent,
            variant=variant,
        )
        best = _validate_best_checkpoint(
            output_dir / "direct" / variant / "best.pt", bundle, config, variant
        )
        variant_summaries[variant] = _validate_variant_summary(
            _plain_json(output_dir / "direct" / variant / "training-summary.json"),
            best,
            system=system,
            variant=variant,
            backbone_hash=backbone_hash,
            expected_trainable_parameters=_trainable_parameter_count(bundle),
            source_tree_sha256=config.source_tree_sha256,
            trace_path=output_dir / "direct" / variant / "firewall-trace.jsonl",
        )
        del bundle
    full_trainable = int(variant_summaries["full"]["trainableParameters"])
    constant_trainable = int(
        variant_summaries["constant_port"]["trainableParameters"]
    )
    if abs(full_trainable - constant_trainable) / max(full_trainable, 1) > 0.01:
        raise ValueError("constant-port end-to-end parameter gap exceeds 1%")
    probes = _load_registered_probes(output_dir, config)
    independent, baseline_summary = _load_independent_baseline_artifact(
        output_dir, config, backbone, probes, device
    )
    runtime_firewall_traces = [
        _runtime_trace_entry(
            output_dir,
            phase="backbone",
            relative_path="backbone/firewall-trace.jsonl",
            seal=completion["summary"].get("runtimeTrace"),
            source_tree_sha256=config.source_tree_sha256,
        ),
        _runtime_trace_entry(
            output_dir,
            phase=f"jacobian-port-precompute:{system}",
            relative_path="port-precompute/firewall-trace.jsonl",
            seal=_plain_json(output_dir / "port-precompute-complete.json")[
                "summary"
            ].get("runtimeTrace"),
            source_tree_sha256=config.source_tree_sha256,
        ),
        *(
            _runtime_trace_entry(
                output_dir,
                phase=f"direct:{variant}",
                relative_path=f"direct/{variant}/firewall-trace.jsonl",
                seal=variant_summaries[variant].get("runtimeTrace"),
                source_tree_sha256=config.source_tree_sha256,
            )
            for variant in REGISTERED_VARIANTS
        ),
        _runtime_trace_entry(
            output_dir,
            phase="baseline:independent_unstructured",
            relative_path="baseline-independent/firewall-trace.jsonl",
            seal=baseline_summary.get("runtimeTrace"),
            source_tree_sha256=config.source_tree_sha256,
        ),
    ]
    baseline_completion = _plain_json(output_dir / "baseline-complete.json")
    if set(baseline_completion) != _BASELINE_COMPLETE_KEYS or (
        baseline_completion["kind"]
        != "direct_distributed_independent_baseline_complete"
        or baseline_completion["system"] != system
        or baseline_completion["configSha256"] != config.sha256
        or baseline_completion["backboneHash"] != backbone_hash
        or baseline_completion["sourceTreeSha256"] != config.source_tree_sha256
        or baseline_completion["baselineCheckpointSha256"]
        != independent.checkpoint_sha256
        or baseline_completion["referenceInitializationSeed"]
        != config.experiment.seed + 10_003
        or baseline_completion["summary"] != baseline_summary
    ):
        raise ValueError("baseline completion lineage seal is invalid")
    port_completion = _plain_json(output_dir / "port-precompute-complete.json")
    stage_seconds = [
        float(completion["summary"].get("seconds", 0.0)),
        float(port_completion["summary"].get("seconds", 0.0)),
    ]
    stage_seconds.extend(
        float(variant_summaries[variant]["seconds"])
        for variant in REGISTERED_VARIANTS
    )
    stage_seconds.append(float(baseline_summary.get("seconds", 0.0)))
    if not all(math.isfinite(value) and value >= 0.0 for value in stage_seconds):
        raise ValueError("a distributed stage duration is invalid")
    excitation_config = HiddenExcitationConfig(
        frames=config.experiment.cache_frames,
        image_size=config.experiment.image_size,
    )
    summary = {
        "kind": "direct_jacobian_poisson_ph_training_complete",
        "system": system,
        "experimentConfig": asdict(config.experiment),
        "backboneConfig": asdict(config.backbone),
        "portConfig": asdict(config.port),
        "directConfig": asdict(config.direct),
        "baselineConfig": asdict(config.baseline),
        "lossConfig": asdict(config.loss),
        "manifests": {
            split: asdict(config.manifests[split])
            for split in ("fit", "validation")
        },
        "sourceManifest": dict(config.source_manifest),
        "sourceTreeSha256": config.source_tree_sha256,
        "learnerSourceManifest": dict(config.learner_source_manifest),
        "learnerSourceTreeSha256": config.learner_source_tree_sha256,
        "heldoutTestArchiveOpenedByTraining": False,
        "backbone": completion["summary"],
        "portPrecompute": port_completion["summary"],
        "backboneHash": backbone_hash,
        "variants": variant_summaries,
        "baseline": baseline_summary,
        "seconds": float(sum(stage_seconds)),
        "neuralParametersFrozenForPhysicalEvaluation": True,
        "actionGradientUpdates": 0,
        "physicalStateGradientUpdates": 0,
        "runtimeFirewallTraces": runtime_firewall_traces,
        "hiddenExcitationConfig": asdict(excitation_config),
        "hiddenExcitationConfigSha256": hidden_excitation_config_sha256(
            excitation_config
        ),
    }
    path = output_dir / "training-complete.json"
    canonical = json.loads(json.dumps(summary, allow_nan=False))
    if path.exists() and _plain_json(path) != canonical:
        raise ValueError("existing training-complete summary differs from strict aggregate")
    _atomic_json(path, canonical)
    return canonical


def _experiment_from_args(args: argparse.Namespace) -> ExperimentFConfig:
    return ExperimentFConfig(
        seed=151_910_737,
        fit_trajectories=args.fit_trajectories,
        validation_trajectories=args.validation_trajectories,
        test_trajectories=args.test_trajectories,
        history_frames=8,
        transitions=args.transitions,
        cache_frames=args.cache_frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        backbone_preset=args.backbone_preset,
        variants=REGISTERED_VARIANTS,
    )


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fit-trajectories", type=int, default=4_096)
    parser.add_argument("--validation-trajectories", type=int, default=512)
    parser.add_argument("--test-trajectories", type=int, default=512)
    parser.add_argument("--transitions", type=int, default=8)
    parser.add_argument("--cache-frames", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--backbone-preset", default="tiny")


def _add_learner_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--learner-bundle-root", type=Path, required=True)
    parser.add_argument(
        "--learner-cache",
        type=Path,
        action="append",
        default=[],
        help="code-free cache root; may be repeated",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    backbone = stages.add_parser("backbone")
    backbone.add_argument("trainer_mount", type=Path)
    backbone.add_argument("output_dir", type=Path)
    backbone.add_argument("--system", choices=tuple(DIRECT_SYSTEMS), required=True)
    backbone.add_argument("--device", default="cuda")
    backbone.add_argument("--backbone-steps", type=int, default=30_000)
    backbone.add_argument("--direct-steps", type=int, default=30_000)
    backbone.add_argument("--baseline-steps", type=int, default=30_000)
    backbone.add_argument("--micro-batch-size", type=int, default=16)
    backbone.add_argument("--lens-batch-size", type=int, default=4)
    backbone.add_argument("--implicit-iterations", type=int, default=32)
    backbone.add_argument("--port-contexts", type=int, default=4_096)
    backbone.add_argument("--port-batch-size", type=int, default=4)
    backbone.add_argument("--port-channel-rank", type=int, default=16)
    backbone.add_argument("--port-neighbors", type=int, default=32)
    backbone.add_argument("--source-manifest", type=Path, required=True)
    _add_data_arguments(backbone)
    _add_learner_runtime_arguments(backbone)

    for name in ("port", "variant", "baseline", "finalize"):
        stage = stages.add_parser(name)
        stage.add_argument("trainer_mount", type=Path)
        stage.add_argument("output_dir", type=Path)
        stage.add_argument("--system", choices=tuple(DIRECT_SYSTEMS), required=True)
        stage.add_argument("--device", default="cuda" if name != "finalize" else "cpu")
        if name == "variant":
            stage.add_argument("--variant", choices=REGISTERED_VARIANTS, required=True)
        _add_learner_runtime_arguments(stage)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "backbone":
        source_manifest = load_source_manifest(
            args.source_manifest, verify_current_tree=False
        )
        learner_source_manifest = verify_learner_source_bundle(
            args.learner_bundle_root,
            expected_full_source_tree_sha256=source_manifest["treeSha256"],
        )
        direct_config = DirectTrainingConfig(
            steps=args.direct_steps,
            micro_batch_size=args.micro_batch_size,
            lens_batch_size=args.lens_batch_size,
            implicit_iterations=args.implicit_iterations,
            port_tangent_channel_rank=args.port_channel_rank,
            port_tangent_neighbors=args.port_neighbors,
        )
        if args.baseline_steps != args.direct_steps:
            raise ValueError(
                "independent baseline and direct model must use the same step count"
            )
        result = run_backbone_stage(
            args.trainer_mount,
            args.output_dir,
            system=args.system,
            experiment=_experiment_from_args(args),
            backbone_config=ActionFreeBackboneTrainConfig(steps=args.backbone_steps),
            port_config=JacobianPortPrecomputeConfig(
                contexts=args.port_contexts,
                batch_size=args.port_batch_size,
                lens_block=DIRECT_SYSTEMS[args.system].lens_block,
                horizons=direct_config.lens_horizons,
                channel_rank=args.port_channel_rank,
                neighbors=args.port_neighbors,
                support_floor_ratio=direct_config.port_support_floor_ratio,
            ),
            direct_config=direct_config,
            baseline_config=direct_config,
            loss_config=DirectVideoLossConfig(),
            device=torch.device(args.device),
            source_manifest=source_manifest,
            learner_source_manifest=learner_source_manifest,
            learner_bundle_root=args.learner_bundle_root,
            learner_cache_roots=tuple(args.learner_cache),
        )
    elif args.stage == "port":
        result = run_port_precompute_stage(
            args.trainer_mount,
            args.output_dir,
            system=args.system,
            device=torch.device(args.device),
            learner_bundle_root=args.learner_bundle_root,
            learner_cache_roots=tuple(args.learner_cache),
        )
    elif args.stage == "variant":
        result = run_variant_stage(
            args.trainer_mount,
            args.output_dir,
            system=args.system,
            variant=args.variant,
            device=torch.device(args.device),
            learner_bundle_root=args.learner_bundle_root,
            learner_cache_roots=tuple(args.learner_cache),
        )
    elif args.stage == "baseline":
        result = run_baseline_stage(
            args.trainer_mount,
            args.output_dir,
            system=args.system,
            device=torch.device(args.device),
            learner_bundle_root=args.learner_bundle_root,
            learner_cache_roots=tuple(args.learner_cache),
        )
    else:
        result = finalize_training_complete(
            args.trainer_mount,
            args.output_dir,
            system=args.system,
            device=torch.device(args.device),
            learner_bundle_root=args.learner_bundle_root,
            learner_cache_roots=tuple(args.learner_cache),
        )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "SealedDistributedConfig",
    "finalize_training_complete",
    "load_sealed_config",
    "run_backbone_stage",
    "run_baseline_stage",
    "run_port_precompute_stage",
    "run_variant_stage",
]
