"""Fail-closed post-freeze runner for Experiment F.

This module is intentionally separated from training.  It reconstructs the
registered neural bundles from sealed checkpoints, freezes every tensor, and
then assembles detached evidence for the structural and physical audits.  It
never selects a checkpoint and exposes no optimizer or gradient update.

The expensive control experiment can be split into independently scheduled
shards.  :func:`merge_control_shards` refuses incomplete or overlapping
coverage, and :func:`compose_single_seed_outcome` can never turn missing gate
evidence into a positive outcome.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .action_free_excitation import (
    HiddenExcitationConfig,
    action_free_environment_config_sha256,
    hidden_excitation_config_sha256,
)
from .direct_action_free_data import (
    ActionFreeBackboneTrainConfig,
    PixelsOnlyManifest,
    build_validated_action_free_backbone,
    make_optimization_suite,
    sanitized_pixel_tensor_sha256,
)
from .direct_activation_lens import (
    differentiable_attention_backend,
    direct_dynamics_pulse_responses,
)
from .direct_cotangent_bridge import (
    PixelChangeProbeBank,
    activation_observable_covectors,
)
from .direct_experiment_training import (
    DIRECT_SYSTEMS,
    DirectModelBundle,
    DirectTrainingConfig,
    Variant,
    _named_optimized_parameters,
    build_direct_bundle,
)
from .direct_jacobian_port_precompute import (
    JacobianPortPrecomputeConfig,
    load_empirical_tangent_artifact,
)
from .direct_unstructured_postfreeze import (
    FrozenIndependentBaseline,
    load_frozen_independent_baseline,
)
from .direct_unstructured_training import INDEPENDENT_SUMMARY_KEYS
from .direct_unstructured_world_model import independent_evaluation_modules
from .learner_source_bundle import validate_learner_source_manifest
from .direct_jacobian_poisson_ph_experiment import (
    ExperimentFConfig,
    load_sanitized_split,
)
from .direct_ph_structure_audits import (
    ACTIVATION_SUFFIX_RETENTION_PATH,
    FirewallAuditEvidence,
    Gate3Thresholds,
    Gate4Thresholds,
    GateAuditResult,
    LensAuditEvidence,
    audit_gate_1,
    audit_gate_3,
    audit_gate_4,
    gate4_path_fingerprint_sha256,
)
from .direct_physical_evaluation import (
    CalibrationResult,
    ControlResult,
    FrozenActivationWriteWorldModel,
    FrozenEvaluationSeal,
    FrozenLatentPlannerSpec,
    RealizabilityMetrics,
    adapt_dynamics_for_evaluation,
    builtin_pixel_plant,
    calibrate_activation_interface_after_freeze,
    collect_paired_calibration_response_bank,
    collect_paired_heldout_response_bank,
    evaluate_closed_loop_controllers,
    evaluate_heldout_activation_from_response_bank,
    evaluate_heldout_realizability_from_response_bank,
    evaluation_system_from_direct_spec,
    fixed_interfaces,
    fit_interface_calibration_from_response_bank,
    make_builtin_control_episodes,
    make_builtin_probe_candidates,
    realizability_gate_metrics,
    select_shared_maximin_probe_states,
)
from .direct_visual_poisson_ph import DirectVideoLossConfig
from .env import PALETTE
from .passive_jacobian_ph_model import module_tensor_hash
from .source_provenance import build_source_manifest, verify_source_manifest
from .runtime_firewall_trace import verify_runtime_trace
from .pixel_direct_model import pixel_direct_config_for_preset


REQUIRED_POSTFREEZE_VARIANTS: tuple[Variant, ...] = (
    "full",
    "no_jacobian",
    "single_horizon",
    "shuffled_lens",
    "skew_only",
    "constant_port",
)
REGISTERED_HORIZONS = (1, 2, 4)
REGISTERED_GATE3_TRANSITIONS = 4_096
REGISTERED_GATE4_CONTEXTS = 128
REGISTERED_RANDOM_WRITES = 16
REGISTERED_REALIZABILITY_STATES_PER_AXIS = 128
REGISTERED_CONTROL_EPISODES = 64

_GATE4_PATH_SOURCE_FILES = (
    "blocket_league/direct_activation_lens.py",
    "blocket_league/direct_cotangent_bridge.py",
    "blocket_league/direct_jacobian_port_extractor.py",
    "blocket_league/direct_ph_structure_audits.py",
    "blocket_league/direct_postfreeze_runner.py",
    "blocket_league/direct_visual_poisson_ph.py",
)

_DIRECT_CHECKPOINT_KEYS = {
    "kind",
    "actionChannels",
    "physicalStateChannels",
    "optimizationTensorKeys",
    "system",
    "variant",
    "step",
    "bestValidation",
    "bestStructureEligible",
    "model",
    "writeField",
    "responseFrame",
    "cotangentFrame",
    "probes",
    "probeHash",
    "dataSeal",
    "optimizedParameterNames",
    "trainConfig",
    "lossConfig",
    "backboneHash",
    "sourceTreeSha256",
}
_DATA_SEAL_KEYS = {
    "system",
    "fitAggregateSha256",
    "fitSanitizedTensorSha256",
    "validationAggregateSha256",
    "validationSanitizedTensorSha256",
}
_TRAINING_SUMMARY_KEYS = {
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
    "sourceTreeSha256",
    "learnerSourceManifest",
    "learnerSourceTreeSha256",
    "heldoutTestArchiveOpenedByTraining",
    "backbone",
    "portPrecompute",
    "backboneHash",
    "variants",
    "baseline",
    "seconds",
    "neuralParametersFrozenForPhysicalEvaluation",
    "actionGradientUpdates",
    "physicalStateGradientUpdates",
    "runtimeFirewallTraces",
    "hiddenExcitationConfig",
    "hiddenExcitationConfigSha256",
}
_PORT_PRECOMPUTE_COMPLETE_KEYS = {
    "kind",
    "system",
    "configSha256",
    "backboneHash",
    "fitSanitizedTensorSha256",
    "artifactSha256",
    "sourceTreeSha256",
    "summary",
}
_PORT_PRECOMPUTE_SUMMARY_KEYS = {
    "kind",
    "system",
    "contexts",
    "seconds",
    "backboneHash",
    "fitSanitizedTensorSha256",
    "selectedTransitionIndicesSha256",
    "sourceActivationTensorSha256",
    "artifact",
    "runtimeTrace",
}
_DISTRIBUTED_CONFIG_KEYS = {
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
_RUNTIME_TRACE_ENTRY_KEYS = {"phase", "relativePath", "seal"}
_FORBIDDEN_RUNTIME_KEYS = (
    "action",
    "control",
    "force",
    "torque",
    "state_label",
    "simulator_state",
    "coordinate",
    "centroid",
    "velocity",
    "momentum",
    "energy",
    "object_mask",
    "entity_mask",
    "contact",
    "event",
    "reward",
    "score",
    "seed",
)
_PRIVATE_MOUNT_TERMS = ("producer-private", "heldout", "simulator-private")


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


def _split_set_sha256(values: Mapping[str, str]) -> str:
    """Canonical seal over the exact fit/validation/test sanitized tensors."""

    if set(values) != {"fit", "validation", "test"}:
        raise ValueError("split hash set is incomplete")
    digest = hashlib.sha256()
    for split in ("fit", "validation", "test"):
        value = values[split]
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{split} tensor hash is not canonical SHA-256")
        digest.update(split.encode("ascii"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _expected_data_seal(
    system_name: str,
    manifests: Mapping[str, PixelsOnlyManifest],
) -> dict[str, str]:
    if not {"fit", "validation"}.issubset(manifests):
        raise ValueError("cannot build a data seal from incomplete manifests")
    return {
        "system": system_name,
        "fitAggregateSha256": manifests["fit"].aggregate_sha256,
        "fitSanitizedTensorSha256": manifests["fit"].sanitized_tensor_sha256,
        "validationAggregateSha256": manifests["validation"].aggregate_sha256,
        "validationSanitizedTensorSha256": manifests["validation"].sanitized_tensor_sha256,
    }


def _plain_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _strict_dataclass_config(
    value: Any,
    config_type: type,
    *,
    label: str,
    tuple_fields: Sequence[str] = (),
    json_arrays: bool,
) -> Any:
    """Parse an exact primitive-only dataclass configuration.

    JSON necessarily turns tuples into arrays, while weights-only PyTorch
    checkpoints preserve tuples.  The caller declares which representation is
    expected so an accidental or hand-edited representation cannot be silently
    normalized.  Scalar types are matched to the registered defaults exactly;
    in particular booleans cannot masquerade as integers.
    """

    if type(value) is not dict:
        raise ValueError(f"{label} must be a plain dictionary")
    expected_fields = {item.name for item in fields(config_type)}
    if set(value) != expected_fields:
        difference = sorted(set(value) ^ expected_fields)
        raise ValueError(f"{label} schema is not exact: {difference}")
    tuple_field_set = set(tuple_fields)
    if not tuple_field_set.issubset(expected_fields):  # pragma: no cover
        raise AssertionError(f"unknown tuple field registered for {label}")
    defaults = config_type()
    normalized: dict[str, Any] = {}
    for item in fields(config_type):
        name = item.name
        observed = value[name]
        expected = getattr(defaults, name)
        if name in tuple_field_set:
            expected_container = list if json_arrays else tuple
            if type(observed) is not expected_container:
                representation = "JSON array" if json_arrays else "tuple"
                raise ValueError(f"{label}.{name} must be a {representation}")
            if type(expected) is not tuple or not expected:  # pragma: no cover
                raise AssertionError(f"{label}.{name} has no registered tuple type")
            element_type = type(expected[0])
            if any(type(element) is not element_type for element in observed):
                raise ValueError(f"{label}.{name} element types are invalid")
            normalized[name] = tuple(observed)
            continue
        if type(observed) is not type(expected):
            raise ValueError(f"{label}.{name} scalar type is invalid")
        if isinstance(observed, float) and not np.isfinite(observed):
            raise ValueError(f"{label}.{name} must be finite")
        normalized[name] = observed
    try:
        parsed = config_type(**normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} values are invalid") from error
    if asdict(parsed) != normalized:  # pragma: no cover - defensive dataclass seal
        raise ValueError(f"{label} was not preserved by its registered parser")
    return parsed


def _variant_loss_config(
    base: DirectVideoLossConfig,
    variant: Variant,
) -> DirectVideoLossConfig:
    """Reproduce the only registered per-ablation loss transformations."""

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


def _manifest_from_dict(value: Any, *, split: str) -> PixelsOnlyManifest:
    if not isinstance(value, dict):
        raise ValueError(f"training summary is missing the {split} manifest")
    try:
        normalized = dict(value)
        normalized["source_schema"] = tuple(normalized.get("source_schema", ()))
        normalized["optimization_schema"] = tuple(
            normalized.get("optimization_schema", ())
        )
        manifest = PixelsOnlyManifest(**normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {split} pixels-only manifest") from error
    if (
        re.fullmatch(r"[0-9a-f]{64}", manifest.aggregate_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", manifest.sanitized_tensor_sha256) is None
    ):
        raise ValueError(f"{split} manifest does not contain both SHA-256 seals")
    if tuple(manifest.source_schema) != ("frames",):
        raise ValueError(f"{split} source schema is not exactly frames-only")
    if set(manifest.optimization_schema) != {"pixelContexts", "frames"}:
        raise ValueError(f"{split} optimization schema is not pixels-only")
    return manifest


@dataclass(frozen=True)
class PostFreezePaths:
    """Registered on-disk layout for one already-trained system."""

    sanitized_root: Path
    output_dir: Path
    direct_checkpoint_name: str = "best.pt"

    def sanitized_split(self, system: str, split: str) -> Path:
        if split in {"fit", "validation"}:
            boundary = "trainer-mount"
        elif split == "test":
            boundary = "heldout"
        else:
            raise ValueError(f"unknown sanitized split {split!r}")
        return self.sanitized_root / boundary / system / f"{split}-pixels.pt"

    def producer_seal(self, system: str) -> Path:
        return self.sanitized_root / "seals" / system / "manifest.json"

    @property
    def training_summary(self) -> Path:
        return self.output_dir / "training-complete.json"

    @property
    def distributed_config(self) -> Path:
        return self.output_dir / "distributed-config.json"

    @property
    def backbone_checkpoint(self) -> Path:
        return self.output_dir / "backbone" / "checkpoint.pt"

    @property
    def empirical_tangent(self) -> Path:
        return self.output_dir / "port-precompute" / "empirical-tangent.pt"

    @property
    def port_precompute_completion(self) -> Path:
        return self.output_dir / "port-precompute-complete.json"

    def direct_checkpoint(self, variant: str) -> Path:
        return self.output_dir / "direct" / variant / self.direct_checkpoint_name

    @property
    def baseline_checkpoint(self) -> Path:
        return self.output_dir / "baseline-independent" / "best.pt"

    @property
    def baseline_summary(self) -> Path:
        return self.output_dir / "baseline-independent" / "summary.json"

    @property
    def baseline_last_checkpoint(self) -> Path:
        return self.output_dir / "baseline-independent" / "last.pt"


@dataclass(frozen=True)
class FrozenVariantBundle:
    variant: Variant
    bundle: DirectModelBundle = field(repr=False, compare=False)
    checkpoint_path: Path
    checkpoint_sha256: str
    step: int
    best_validation: float
    best_structure_eligible: bool
    optimizer_excluded_backbone: bool
    write_field_sha256: str


@dataclass(frozen=True)
class LoadedPostFreezeSystem:
    """Fully reconstructed neural state and validated held-out pixels."""

    system_name: str
    paths: PostFreezePaths
    training_summary: Mapping[str, Any]
    manifests: Mapping[str, PixelsOnlyManifest]
    observed_split_sha256: Mapping[str, str]
    test_pixels: torch.Tensor = field(repr=False, compare=False)
    backbone: nn.Module = field(repr=False, compare=False)
    backbone_hash: str
    backbone_checkpoint_sha256: str
    producer_seal_sha256: str
    empirical_tangent_artifact_sha256: str
    variants: Mapping[str, FrozenVariantBundle] = field(repr=False, compare=False)
    independent_baseline: FrozenIndependentBaseline = field(
        repr=False, compare=False
    )
    source_tree_sha256: str
    learner_source_manifest: Mapping[str, Any]
    learner_source_tree_sha256: str
    runtime_firewall_traces: Mapping[str, tuple[Mapping[str, Any], ...]] = field(
        repr=False, compare=False
    )

    @property
    def full(self) -> FrozenVariantBundle:
        return self.variants["full"]

    @property
    def baseline_checkpoint_sha256(self) -> str:
        return self.independent_baseline.checkpoint_sha256

    @property
    def device(self) -> torch.device:
        parameter = next(self.full.bundle.model.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def assert_frozen_and_unchanged(self) -> None:
        if module_tensor_hash(self.backbone) != self.backbone_hash:
            raise AssertionError("post-freeze backbone hash changed")
        for name, frozen in self.variants.items():
            modules = (
                frozen.bundle.model,
                frozen.bundle.write_field,
                frozen.bundle.response_frame,
                frozen.bundle.cotangent_frame,
                frozen.bundle.probes,
            )
            if any(module.training for module in modules):
                raise AssertionError(f"variant {name!r} left evaluation mode")
            if any(
                parameter.requires_grad
                for module in modules
                for parameter in module.parameters()
            ):
                raise AssertionError(f"variant {name!r} became trainable")
            if module_tensor_hash(frozen.bundle.write_field) != frozen.write_field_sha256:
                raise AssertionError(f"variant {name!r} empirical port changed")
            frozen.bundle.model.encoder.assert_backbone_frozen()
        self.independent_baseline.assert_frozen_and_unchanged()


def validate_direct_checkpoint_metadata(
    payload: Mapping[str, Any],
    *,
    system_name: str,
    variant: Variant,
    backbone_hash: str,
    source_tree_sha256: str,
) -> None:
    """Strictly validate non-tensor provenance before loading any weights."""

    if type(payload) is not dict:
        raise ValueError("direct checkpoint must be a plain dictionary")
    if set(payload) != _DIRECT_CHECKPOINT_KEYS:
        difference = sorted(set(payload) ^ _DIRECT_CHECKPOINT_KEYS)
        raise ValueError(f"direct checkpoint schema mismatch: {difference}")
    if (
        type(source_tree_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256) is None
        or payload.get("sourceTreeSha256") != source_tree_sha256
    ):
        raise ValueError("direct checkpoint source-tree provenance mismatch")
    if (
        type(payload.get("kind")) is not str
        or payload.get("kind") != "direct_jacobian_poisson_port_hamiltonian"
    ):
        raise ValueError("checkpoint is not a direct Jacobian-Poisson pH model")
    if (
        type(payload.get("actionChannels")) is not int
        or payload.get("actionChannels") != 0
        or type(payload.get("physicalStateChannels")) is not int
        or payload.get("physicalStateChannels") != 0
    ):
        raise ValueError("checkpoint admits a forbidden physical training channel")
    if (
        type(payload.get("optimizationTensorKeys")) is not list
        or payload.get("optimizationTensorKeys") != ["pixelContexts", "frames"]
    ):
        raise ValueError("direct optimization schema is not exactly pixels-only")
    if type(payload.get("system")) is not dict or payload.get("system") != asdict(
        DIRECT_SYSTEMS[system_name]
    ):
        raise ValueError("direct checkpoint system specification drifted")
    if type(payload.get("variant")) is not str or payload.get("variant") != variant:
        raise ValueError("direct checkpoint variant mismatch")
    if (
        type(payload.get("backboneHash")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("backboneHash"))) is None
        or payload.get("backboneHash") != backbone_hash
    ):
        raise ValueError("direct checkpoint references another video backbone")
    if type(payload.get("step")) is not int or int(payload["step"]) < 1:
        raise ValueError("direct checkpoint has no valid training step")
    if type(payload.get("bestValidation")) not in (int, float) or not np.isfinite(
        float(payload["bestValidation"])
    ):
        raise ValueError("direct checkpoint has no finite validation score")
    if type(payload.get("bestStructureEligible")) is not bool:
        raise ValueError("direct checkpoint lacks its structural-eligibility decision")
    if (
        type(payload.get("probeHash")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("probeHash"))) is None
    ):
        raise ValueError("direct checkpoint fixed-probe hash is invalid")
    data_seal = payload.get("dataSeal")
    if type(data_seal) is not dict or set(data_seal) != _DATA_SEAL_KEYS:
        raise ValueError("direct checkpoint sanitized data-seal schema is invalid")
    if any(type(value) is not str for value in data_seal.values()):
        raise ValueError("direct checkpoint sanitized data seal is not string-only")
    optimized_names = payload.get("optimizedParameterNames")
    if (
        type(optimized_names) is not list
        or not optimized_names
        or any(
            type(name) is not str
            or re.fullmatch(r"[A-Za-z0-9_.]+", name) is None
            for name in optimized_names
        )
        or len(set(optimized_names)) != len(optimized_names)
        or any(name.startswith("model.encoder.backbone.") for name in optimized_names)
    ):
        raise ValueError("optimized parameter-name seal is missing or invalid")
    try:
        train_config = payload["trainConfig"]
        loss_config = payload["lossConfig"]
        if type(train_config) is not dict or set(train_config) != {
            item.name for item in fields(DirectTrainingConfig)
        }:
            raise ValueError("trainConfig keys are not exact")
        if type(loss_config) is not dict or set(loss_config) != {
            item.name for item in fields(DirectVideoLossConfig)
        }:
            raise ValueError("lossConfig keys are not exact")
        DirectTrainingConfig(**train_config)
        DirectVideoLossConfig(**loss_config)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("direct checkpoint configuration schema drifted") from error


def _expected_nonbackbone_optimizer_parameter_names(
    bundle: DirectModelBundle,
) -> tuple[str, ...]:
    """Reproduce the exact named parameter order passed to AdamW in training."""

    backbone_ids = {id(parameter) for parameter in bundle.model.encoder.backbone.parameters()}
    names = [
        f"model.{name}"
        for name, parameter in bundle.model.named_parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    for prefix, module in (
        ("writeField", bundle.write_field),
        ("responseFrame", bundle.response_frame),
        ("cotangentFrame", bundle.cotangent_frame),
    ):
        names.extend(
            f"{prefix}.{name}"
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
    if len(names) != len(set(names)):
        raise AssertionError("optimizer parameter names are not unique")
    return tuple(names)


def _freeze_bundle(bundle: DirectModelBundle) -> None:
    for module in (
        bundle.model,
        bundle.write_field,
        bundle.lens,
        bundle.probes,
        bundle.response_frame,
        bundle.cotangent_frame,
    ):
        module.eval().requires_grad_(False)
    bundle.model.encoder.assert_backbone_frozen()


def _validate_module_state(
    field_name: str,
    state: Any,
    module: nn.Module,
) -> None:
    reference = module.state_dict()
    if type(state) is not dict or set(state) != set(reference):
        raise ValueError(f"direct checkpoint {field_name} state schema mismatch")
    for name, tensor in state.items():
        expected = reference[name]
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"direct checkpoint {field_name}.{name} is not a tensor")
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise ValueError(
                f"direct checkpoint {field_name}.{name} shape/dtype mismatch"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"direct checkpoint {field_name}.{name} is non-finite")


def _verified_runtime_firewall_traces(
    paths: PostFreezePaths,
    raw_entries: Any,
    *,
    required_phases: Sequence[str],
    source_tree_sha256: str,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Replay every sealed training trace and bind it to its declared phase."""

    if type(raw_entries) is not list or not raw_entries:
        raise ValueError("training summary runtime trace table is empty or malformed")
    result: dict[str, tuple[Mapping[str, Any], ...]] = {}
    observed_paths: set[Path] = set()
    output_root = paths.output_dir.resolve(strict=True)
    for entry in raw_entries:
        if type(entry) is not dict or set(entry) != _RUNTIME_TRACE_ENTRY_KEYS:
            raise ValueError("runtime trace entry schema is not exact")
        phase = entry["phase"]
        relative = entry["relativePath"]
        if (
            type(phase) is not str
            or not phase
            or phase in result
            or type(relative) is not str
            or not relative
        ):
            raise ValueError("runtime trace phase/path identity is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("runtime trace path escapes the training output")
        trace_path = paths.output_dir / relative_path
        if not trace_path.is_file() or trace_path.is_symlink():
            raise FileNotFoundError(f"missing nonsymbolic runtime trace {trace_path}")
        resolved = trace_path.resolve(strict=True)
        if output_root not in resolved.parents or resolved in observed_paths:
            raise ValueError("runtime trace path is outside output or duplicated")
        observed_paths.add(resolved)
        records = verify_runtime_trace(trace_path, entry["seal"])
        stage_boundaries = [
            record for record in records if record["event"] == "stage_boundary"
        ]
        gradient_events = [
            record for record in records if record["event"] == "gradient_batch"
        ]
        optimizer_events = [
            record for record in records if record["event"] == "optimizer_constructed"
        ]
        backbone_boundaries = [
            record for record in records if record["event"] == "backbone_boundary"
        ]
        is_port_precompute = phase.startswith("jacobian-port-precompute:")
        latest_stage_boundary = max(
            record["sequence"] for record in stage_boundaries
        ) if stage_boundaries else -1
        port_payloads = [
            record
            for record in records
            if record["event"] == "tensor_payload"
            and record["sequence"] > latest_stage_boundary
            and record["payload"].get("phase")
            == "jacobian_port_precompute_no_optimizer"
        ]
        common_invalid = (
            not stage_boundaries
            or not backbone_boundaries
            or any(
                record["payload"].get("sourceTreeSha256")
                != source_tree_sha256
                for record in stage_boundaries
            )
        )
        port_invalid = is_port_precompute and (
            bool(gradient_events)
            or bool(optimizer_events)
            or len(port_payloads) != 1
            or set(port_payloads[0]["payload"].get("tensors", ()))
            != {"pixelContexts", "frames"}
        )
        training_invalid = not is_port_precompute and (
            not gradient_events
            or not optimizer_events
            or any(
                record["payload"].get("phase") != phase
                for record in (*gradient_events, *optimizer_events)
            )
        )
        if common_invalid or port_invalid or training_invalid:
            raise ValueError(f"runtime trace {phase!r} lacks exact phase evidence")
        result[phase] = tuple(records)
    missing = sorted(set(required_phases) - set(result))
    if missing:
        raise ValueError(f"training summary omits required runtime traces: {missing}")
    return result


def load_postfreeze_system(
    system_name: str,
    paths: PostFreezePaths,
    device: torch.device,
) -> LoadedPostFreezeSystem:
    """Reconstruct and seal all registered post-freeze networks.

    Validation is fail-closed: every archive/model schema, tensor manifest,
    system specification, optimizer exclusion count, and backbone hash must
    agree before a module is returned.
    """

    if system_name not in DIRECT_SYSTEMS:
        raise KeyError(f"unknown direct system {system_name!r}")
    required_paths = (
        paths.training_summary,
        paths.distributed_config,
        paths.backbone_checkpoint,
        paths.empirical_tangent,
        paths.port_precompute_completion,
        paths.baseline_checkpoint,
        paths.baseline_last_checkpoint,
        paths.baseline_summary,
        paths.producer_seal(system_name),
        *(paths.sanitized_split(system_name, split) for split in ("fit", "validation", "test")),
        *(paths.direct_checkpoint(variant) for variant in REQUIRED_POSTFREEZE_VARIANTS),
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing post-freeze artifacts: {missing}")

    summary = _plain_json(paths.training_summary)
    if set(summary) != _TRAINING_SUMMARY_KEYS:
        raise ValueError("training-complete summary schema is not exact")
    if (
        type(summary.get("kind")) is not str
        or summary.get("kind") != "direct_jacobian_poisson_ph_training_complete"
    ):
        raise ValueError("training summary kind mismatch")
    if type(summary.get("system")) is not str or summary.get("system") != system_name:
        raise ValueError("training summary belongs to another system")
    source_tree_sha256 = verify_source_manifest(summary.get("sourceManifest"))
    if summary.get("sourceTreeSha256") != source_tree_sha256:
        raise ValueError("training summary source-tree SHA-256 mismatch")
    learner_source_manifest = summary.get("learnerSourceManifest")
    learner_source_tree_sha256 = validate_learner_source_manifest(
        learner_source_manifest
    )
    if (
        summary.get("learnerSourceTreeSha256") != learner_source_tree_sha256
        or learner_source_manifest.get("fullSourceTreeSha256")
        != source_tree_sha256
    ):
        raise ValueError("training summary learner-source provenance mismatch")
    distributed_config_payload = _plain_json(paths.distributed_config)
    if (
        set(distributed_config_payload) != _DISTRIBUTED_CONFIG_KEYS
        or distributed_config_payload.get("kind")
        != "direct_distributed_training_config"
        or distributed_config_payload.get("system") != system_name
        or distributed_config_payload.get("actionGradientUpdates") != 0
        or distributed_config_payload.get("physicalStateGradientUpdates") != 0
        or any(
            distributed_config_payload.get(name) != summary.get(name)
            for name in (
                "experimentConfig",
                "backboneConfig",
                "portConfig",
                "directConfig",
                "baselineConfig",
                "lossConfig",
                "manifests",
                "sourceManifest",
                "learnerSourceManifest",
            )
        )
    ):
        raise ValueError("distributed configuration and training summary differ")
    distributed_config_sha256 = _json_sha256(distributed_config_payload)
    experiment_config: ExperimentFConfig = _strict_dataclass_config(
        summary.get("experimentConfig"),
        ExperimentFConfig,
        label="training summary experimentConfig",
        tuple_fields=("variants",),
        json_arrays=True,
    )
    backbone_train_config: ActionFreeBackboneTrainConfig = _strict_dataclass_config(
        summary.get("backboneConfig"),
        ActionFreeBackboneTrainConfig,
        label="training summary backboneConfig",
        json_arrays=True,
    )
    port_precompute_config: JacobianPortPrecomputeConfig = _strict_dataclass_config(
        summary.get("portConfig"),
        JacobianPortPrecomputeConfig,
        label="training summary portConfig",
        tuple_fields=("horizons",),
        json_arrays=True,
    )
    direct_train_config: DirectTrainingConfig = _strict_dataclass_config(
        summary.get("directConfig"),
        DirectTrainingConfig,
        label="training summary directConfig",
        tuple_fields=("lens_horizons",),
        json_arrays=True,
    )
    # Parsing this exact schema matters even though the post-freeze runner can
    # no longer change baseline weights: otherwise a hand-edited summary could
    # falsely claim a different matched-budget training run.
    baseline_train_config: DirectTrainingConfig = _strict_dataclass_config(
        summary.get("baselineConfig"),
        DirectTrainingConfig,
        label="training summary baselineConfig",
        tuple_fields=("lens_horizons",),
        json_arrays=True,
    )
    if baseline_train_config != direct_train_config:
        raise ValueError("independent baseline did not use the direct schedule")
    system = DIRECT_SYSTEMS[system_name]
    if (
        port_precompute_config.lens_block != system.lens_block
        or port_precompute_config.horizons != direct_train_config.lens_horizons
        or port_precompute_config.channel_rank
        != direct_train_config.port_tangent_channel_rank
        or port_precompute_config.neighbors
        != direct_train_config.port_tangent_neighbors
        or port_precompute_config.support_floor_ratio
        != direct_train_config.port_support_floor_ratio
    ):
        raise ValueError("Jacobian port and direct-model configurations differ")
    base_loss_config: DirectVideoLossConfig = _strict_dataclass_config(
        summary.get("lossConfig"),
        DirectVideoLossConfig,
        label="training summary lossConfig",
        tuple_fields=("rollout_horizons",),
        json_arrays=True,
    )
    summarized_variants = summary.get("variants")
    if type(summarized_variants) is not dict or tuple(summarized_variants) != tuple(
        experiment_config.variants
    ):
        raise ValueError(
            "training summary variants do not exactly match experimentConfig"
        )
    if not set(REQUIRED_POSTFREEZE_VARIANTS).issubset(experiment_config.variants):
        raise ValueError("experimentConfig omits a required post-freeze variant")
    required_runtime_phases = (
        "backbone",
        f"jacobian-port-precompute:{system_name}",
        *(f"direct:{variant}" for variant in experiment_config.variants),
        "baseline:independent_unstructured",
    )
    runtime_firewall_traces = _verified_runtime_firewall_traces(
        paths,
        summary.get("runtimeFirewallTraces"),
        required_phases=required_runtime_phases,
        source_tree_sha256=source_tree_sha256,
    )
    excitation_config = HiddenExcitationConfig(
        frames=experiment_config.cache_frames,
        image_size=experiment_config.image_size,
    )
    if (
        summary.get("hiddenExcitationConfig") != asdict(excitation_config)
        or summary.get("hiddenExcitationConfigSha256")
        != hidden_excitation_config_sha256(excitation_config)
    ):
        raise ValueError("training summary hidden-excitation seal is invalid")
    if (
        type(summary.get("seconds")) is not float
        or not np.isfinite(summary["seconds"])
        or summary["seconds"] < 0.0
    ):
        raise ValueError("training summary duration is invalid")
    if (
        type(summary.get("actionGradientUpdates")) is not int
        or summary.get("actionGradientUpdates") != 0
        or type(summary.get("physicalStateGradientUpdates")) is not int
        or summary.get("physicalStateGradientUpdates") != 0
        or summary.get("neuralParametersFrozenForPhysicalEvaluation") is not True
    ):
        raise ValueError("training summary does not seal an action-free frozen evaluation")
    if summary.get("heldoutTestArchiveOpenedByTraining") is not False:
        raise ValueError("training summary does not prove test-archive isolation")
    summary_manifests = summary.get("manifests")
    if not isinstance(summary_manifests, dict) or set(summary_manifests) != {
        "fit",
        "validation",
    }:
        raise ValueError("training summary must seal exactly fit/validation manifests")
    manifests = {
        split: _manifest_from_dict(summary_manifests[split], split=split)
        for split in ("fit", "validation")
    }
    if any(manifest.system != system_name for manifest in manifests.values()):
        raise ValueError("a sealed split manifest belongs to another system")

    observed_split_sha256: dict[str, str] = {}
    fit_pixels_for_probe: torch.Tensor | None = None
    test_pixels: torch.Tensor | None = None
    for split in ("fit", "validation", "test"):
        pixels, observed_manifest = load_sanitized_split(
            paths.sanitized_split(system_name, split), expected_system=system_name
        )
        if split == "test":
            manifests["test"] = observed_manifest
        elif asdict(observed_manifest) != asdict(manifests[split]):
            raise ValueError(
                f"{split} archive manifest differs from training-complete seal"
            )
        observed_digest = sanitized_pixel_tensor_sha256(pixels)
        if observed_digest != manifests[split].sanitized_tensor_sha256:
            raise ValueError(f"{split} archive tensor differs from its sealed SHA-256")
        observed_split_sha256[split] = observed_digest
        if split == "fit":
            fit_pixels_for_probe = pixels
        elif split == "test":
            test_pixels = pixels
        else:
            del pixels
    if fit_pixels_for_probe is None or test_pixels is None:  # pragma: no cover
        raise AssertionError("fit/test pixels were not retained")

    expected_trajectory_counts = {
        "fit": experiment_config.fit_trajectories,
        "validation": experiment_config.validation_trajectories,
        "test": experiment_config.test_trajectories,
    }
    for split, manifest in manifests.items():
        if manifest.trajectories != expected_trajectory_counts[split]:
            raise ValueError(f"{split} trajectory count differs from experimentConfig")
        if manifest.frames_per_trajectory != experiment_config.cache_frames:
            raise ValueError(f"{split} frame count differs from experimentConfig")
        if manifest.image_size != experiment_config.image_size:
            raise ValueError(f"{split} image size differs from experimentConfig")

    producer_seal = _plain_json(paths.producer_seal(system_name))
    if set(producer_seal) != {
        "system",
        "splits",
        "generationEnvironmentSha256",
        "producerSeedSerialized",
        "physicalCommandsSerialized",
        "simulatorStatesSerialized",
        "sourceTreeSha256",
        "runtimeTrace",
        "hiddenExcitationConfig",
        "hiddenExcitationConfigSha256",
    }:
        raise ValueError("producer boundary seal schema is not exact")
    if (
        producer_seal.get("system") != system_name
        or producer_seal.get("generationEnvironmentSha256")
        != action_free_environment_config_sha256(
            system_name, image_size=experiment_config.image_size
        )
        or producer_seal.get("producerSeedSerialized") is not False
        or producer_seal.get("physicalCommandsSerialized") is not False
        or producer_seal.get("simulatorStatesSerialized") is not False
        or producer_seal.get("sourceTreeSha256") != source_tree_sha256
        or producer_seal.get("hiddenExcitationConfig") != asdict(excitation_config)
        or producer_seal.get("hiddenExcitationConfigSha256")
        != hidden_excitation_config_sha256(excitation_config)
    ):
        raise ValueError("producer boundary seal does not certify erased private data")
    producer_splits = producer_seal.get("splits")
    if type(producer_splits) is not dict or set(producer_splits) != {
        "fit",
        "validation",
        "test",
    }:
        raise ValueError("producer boundary seal has an incomplete split table")
    for split in ("fit", "validation", "test"):
        sealed_manifest = _manifest_from_dict(producer_splits[split], split=split)
        if asdict(sealed_manifest) != asdict(manifests[split]):
            raise ValueError(f"producer boundary {split} seal does not match archive")
    producer_trace_path = (
        paths.sanitized_root / "seals" / system_name / "firewall-trace.jsonl"
    )
    producer_records = verify_runtime_trace(
        producer_trace_path, producer_seal.get("runtimeTrace")
    )
    producer_latest_attempt = max(
        record["sequence"]
        for record in producer_records
        if record["event"] == "stage_boundary"
    )
    producer_payloads = [
        record
        for record in producer_records
        if record["event"] == "tensor_payload"
        and record["sequence"] > producer_latest_attempt
    ]
    if (
        len(producer_payloads) != sum(expected_trajectory_counts.values())
        or any(
            record["payload"].get("phase") != "producer"
            or set(record["payload"].get("tensors", {})) != {"frames"}
            for record in producer_payloads
        )
        or any(
            record["payload"].get("sourceTreeSha256") != source_tree_sha256
            for record in producer_records
            if record["event"] == "stage_boundary"
        )
    ):
        raise ValueError("producer runtime firewall trace is incomplete")
    runtime_firewall_traces["producer"] = tuple(producer_records)

    backbone_payload = torch.load(
        paths.backbone_checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(backbone_payload, dict):
        raise ValueError("backbone checkpoint is not a mapping")
    if backbone_payload.get("train_config") != asdict(backbone_train_config):
        raise ValueError(
            "backbone checkpoint training configuration differs from training-complete"
        )
    backbone = build_validated_action_free_backbone(
        backbone_payload,
        expected_manifest_sha256=manifests["fit"].aggregate_sha256,
        expected_sanitized_tensor_sha256=manifests["fit"].sanitized_tensor_sha256,
        expected_system=system_name,
    ).to(device)
    backbone.eval().requires_grad_(False)
    expected_model_config = pixel_direct_config_for_preset(
        experiment_config.backbone_preset,
        image_size=experiment_config.image_size,
        patch_size=experiment_config.patch_size,
        palette_size=len(PALETTE),
        history_frames=experiment_config.history_frames,
    )
    if backbone.config != expected_model_config:  # type: ignore[attr-defined]
        raise ValueError(
            "sealed backbone architecture differs from experimentConfig preset"
        )
    backbone_hash = module_tensor_hash(backbone)
    if summary.get("backboneHash") != backbone_hash:
        raise ValueError("training summary backbone hash mismatch")

    port_completion = _plain_json(paths.port_precompute_completion)
    port_summary = summary.get("portPrecompute")
    if (
        set(port_completion) != _PORT_PRECOMPUTE_COMPLETE_KEYS
        or port_completion.get("kind")
        != "direct_empirical_jacobian_port_precompute_complete"
        or port_completion.get("system") != system_name
        or port_completion.get("configSha256") != distributed_config_sha256
        or port_completion.get("backboneHash") != backbone_hash
        or port_completion.get("fitSanitizedTensorSha256")
        != manifests["fit"].sanitized_tensor_sha256
        or port_completion.get("artifactSha256")
        != _file_sha256(paths.empirical_tangent)
        or port_completion.get("sourceTreeSha256") != source_tree_sha256
        or type(port_summary) is not dict
        or set(port_summary) != _PORT_PRECOMPUTE_SUMMARY_KEYS
        or port_completion.get("summary") != port_summary
        or port_summary.get("kind")
        != "frozen_empirical_jacobian_tangent_summary_v1"
        or port_summary.get("system") != system_name
        or port_summary.get("contexts") != port_precompute_config.contexts
        or type(port_summary.get("seconds")) is not float
        or not np.isfinite(port_summary["seconds"])
        or port_summary["seconds"] < 0.0
        or port_summary.get("backboneHash") != backbone_hash
        or port_summary.get("fitSanitizedTensorSha256")
        != manifests["fit"].sanitized_tensor_sha256
        or port_summary.get("artifact") != paths.empirical_tangent.name
    ):
        raise ValueError("empirical Jacobian port completion lineage is invalid")
    empirical_tangent = load_empirical_tangent_artifact(
        paths.empirical_tangent,
        expected_system=system_name,
        expected_fit_sanitized_tensor_sha256=manifests[
            "fit"
        ].sanitized_tensor_sha256,
        expected_source_tree_sha256=source_tree_sha256,
        expected_backbone_hash=backbone_hash,
        expected_config=port_precompute_config,
    )

    system = DIRECT_SYSTEMS[system_name]
    fit_suite_for_probe = make_optimization_suite(
        fit_pixels_for_probe,
        backbone.config,  # type: ignore[attr-defined]
        transitions=experiment_config.transitions,
    )
    # Training resets the one master RNG immediately before deriving this
    # pixels-only PCA bank.  fork_rng makes the audit recomputation exact
    # without changing any later evaluation randomness.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(experiment_config.seed + 71)
        source_probes = PixelChangeProbeBank.from_pixel_frames(
            fit_suite_for_probe["frames"],
            palette_size=backbone.config.palette_size,  # type: ignore[attr-defined]
            probe_size=system.port_size,
        )
    # ``train_registered_system`` calls copy_probe_bank once before each
    # variant; reproduce that second QR exactly.
    registered_probes = PixelChangeProbeBank(source_probes.basis.detach().clone())
    registered_probe_hash = module_tensor_hash(registered_probes)
    del fit_suite_for_probe, fit_pixels_for_probe, source_probes
    expected_data_seal = _expected_data_seal(system_name, manifests)
    variants: dict[str, FrozenVariantBundle] = {}
    variant_trainable_parameters: dict[str, int] = {}
    for variant in REQUIRED_POSTFREEZE_VARIANTS:
        checkpoint_path = paths.direct_checkpoint(variant)
        # A post-freeze evaluator must never execute an arbitrary checkpoint
        # pickle before it has validated provenance.  Training therefore has
        # to emit a weights-only-compatible evaluation artifact; legacy resume
        # checkpoints containing NumPy/Python RNG objects fail here.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"{variant} checkpoint is not a mapping")
        validate_direct_checkpoint_metadata(
            payload,
            system_name=system_name,
            variant=variant,
            backbone_hash=backbone_hash,
            source_tree_sha256=source_tree_sha256,
        )
        checkpoint_train_config: DirectTrainingConfig = _strict_dataclass_config(
            payload["trainConfig"],
            DirectTrainingConfig,
            label=f"{variant} checkpoint trainConfig",
            tuple_fields=("lens_horizons",),
            json_arrays=False,
        )
        if checkpoint_train_config != direct_train_config:
            raise ValueError(
                f"{variant} checkpoint trainConfig differs from training-complete"
            )
        checkpoint_loss_config: DirectVideoLossConfig = _strict_dataclass_config(
            payload["lossConfig"],
            DirectVideoLossConfig,
            label=f"{variant} checkpoint lossConfig",
            tuple_fields=("rollout_horizons",),
            json_arrays=False,
        )
        expected_variant_loss = _variant_loss_config(base_loss_config, variant)
        if checkpoint_loss_config != expected_variant_loss:
            raise ValueError(
                f"{variant} checkpoint lossConfig is not the registered "
                "deterministic ablation of training-complete"
            )
        if payload["dataSeal"] != expected_data_seal:
            raise ValueError(f"{variant} checkpoint sanitized data seal does not match")
        summarized_variant = summarized_variants[variant]
        if not isinstance(summarized_variant, dict):
            raise ValueError(f"training summary for {variant} is not a mapping")
        if (
            summarized_variant.get("system") != system_name
            or summarized_variant.get("variant") != variant
            or summarized_variant.get("bestStep") != payload["step"]
            or float(summarized_variant.get("bestValidation", float("nan")))
            != float(payload["bestValidation"])
            or summarized_variant.get("bestStructureEligible")
            is not payload["bestStructureEligible"]
            or summarized_variant.get("backboneHashBefore") != backbone_hash
            or summarized_variant.get("backboneHashAfter") != backbone_hash
            or summarized_variant.get("actionGradientUpdates") != 0
            or summarized_variant.get("physicalStateGradientUpdates") != 0
            or summarized_variant.get("sourceTreeSha256") != source_tree_sha256
        ):
            raise ValueError(f"{variant} checkpoint disagrees with training-complete summary")
        probe_state = payload["probes"]
        if type(probe_state) is not dict or set(probe_state) != {"basis"}:
            raise ValueError(f"{variant} probe-bank state schema mismatch")
        basis = probe_state["basis"]
        if not isinstance(basis, torch.Tensor):
            raise ValueError(f"{variant} probe-bank basis is not a tensor")
        if (
            payload["probeHash"] != registered_probe_hash
            or not torch.equal(
                basis.detach().cpu(), registered_probes.basis.detach().cpu()
            )
        ):
            raise ValueError(
                f"{variant} fixed probes do not match pixels-only recomputation"
            )
        probe_copy = PixelChangeProbeBank(registered_probes.basis.detach().clone())
        probe_copy.load_state_dict(registered_probes.state_dict(), strict=True)
        bundle = build_direct_bundle(
            backbone,
            system,
            probe_copy,
            direct_train_config,
            device,
            empirical_tangent=empirical_tangent,
            variant=variant,
        )
        trainable_parameter_count = sum(
            parameter.numel() for _, parameter in _named_optimized_parameters(bundle)
        )
        if summarized_variant.get("trainableParameters") != trainable_parameter_count:
            raise ValueError(f"{variant} trainable-parameter count is not exact")
        variant_trainable_parameters[variant] = trainable_parameter_count
        expected_optimizer_names = _expected_nonbackbone_optimizer_parameter_names(bundle)
        observed_optimizer_names = tuple(payload["optimizedParameterNames"])
        optimizer_excluded_backbone = observed_optimizer_names == expected_optimizer_names
        if not optimizer_excluded_backbone:
            raise ValueError(
                f"{variant} optimized parameter-name seal is incompatible with backbone exclusion"
            )
        for field_name, module in (
            ("model", bundle.model),
            ("writeField", bundle.write_field),
            ("responseFrame", bundle.response_frame),
            ("cotangentFrame", bundle.cotangent_frame),
            ("probes", bundle.probes),
        ):
            _validate_module_state(field_name, payload[field_name], module)
        current_model_state = bundle.model.state_dict()
        for name, tensor in payload["model"].items():
            if name.startswith("encoder.backbone.") and not torch.equal(
                tensor.detach().cpu(), current_model_state[name].detach().cpu()
            ):
                raise ValueError(f"{variant} attempted to change the sealed backbone")
        bundle.model.load_state_dict(payload["model"], strict=True)
        bundle.write_field.load_state_dict(payload["writeField"], strict=True)
        bundle.response_frame.load_state_dict(payload["responseFrame"], strict=True)
        bundle.cotangent_frame.load_state_dict(payload["cotangentFrame"], strict=True)
        bundle.probes.load_state_dict(payload["probes"], strict=True)
        _freeze_bundle(bundle)
        if module_tensor_hash(bundle.probes) != payload["probeHash"]:
            raise ValueError(f"{variant} checkpoint fixed-probe hash does not match")
        if module_tensor_hash(bundle.model.encoder.backbone) != backbone_hash:
            raise ValueError(f"{variant} checkpoint mutated the sealed backbone")
        variants[variant] = FrozenVariantBundle(
            variant=variant,
            bundle=bundle,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=_file_sha256(checkpoint_path),
            step=int(payload["step"]),
            best_validation=float(payload["bestValidation"]),
            best_structure_eligible=bool(payload["bestStructureEligible"]),
            optimizer_excluded_backbone=optimizer_excluded_backbone,
            write_field_sha256=module_tensor_hash(bundle.write_field),
        )

    full_trainable = variant_trainable_parameters["full"]
    constant_trainable = variant_trainable_parameters["constant_port"]
    if abs(full_trainable - constant_trainable) / max(full_trainable, 1) > 0.01:
        raise ValueError("constant-port end-to-end parameter gap exceeds 1%")

    baseline_summary = summary.get("baseline")
    on_disk_baseline_summary = _plain_json(paths.baseline_summary)
    if (
        type(baseline_summary) is not dict
        or set(baseline_summary) != INDEPENDENT_SUMMARY_KEYS
        or on_disk_baseline_summary != baseline_summary
    ):
        raise ValueError("independent baseline summary schema/lineage mismatch")
    independent_baseline = load_frozen_independent_baseline(
        backbone=backbone,
        system=system,
        probes=registered_probes,
        empirical_tangent=empirical_tangent,
        train_config=direct_train_config,
        loss_config=base_loss_config,
        checkpoint_path=paths.baseline_checkpoint,
        summary=baseline_summary,
        data_seal=expected_data_seal,
        source_tree_sha256=source_tree_sha256,
        reference_initialization_seed=experiment_config.seed + 10_003,
        device=device,
    )
    loaded = LoadedPostFreezeSystem(
        system_name=system_name,
        paths=paths,
        training_summary=summary,
        manifests=manifests,
        observed_split_sha256=observed_split_sha256,
        test_pixels=test_pixels,
        backbone=backbone,
        backbone_hash=backbone_hash,
        backbone_checkpoint_sha256=_file_sha256(paths.backbone_checkpoint),
        producer_seal_sha256=_file_sha256(paths.producer_seal(system_name)),
        empirical_tangent_artifact_sha256=_file_sha256(paths.empirical_tangent),
        variants=variants,
        independent_baseline=independent_baseline,
        source_tree_sha256=source_tree_sha256,
        learner_source_manifest=learner_source_manifest,
        learner_source_tree_sha256=learner_source_tree_sha256,
        runtime_firewall_traces=runtime_firewall_traces,
    )
    loaded.assert_frozen_and_unchanged()
    return loaded


def assemble_gate1_evidence(loaded: LoadedPostFreezeSystem) -> FirewallAuditEvidence:
    """Derive Gate 1 observations exclusively from replayed runtime events."""

    loaded.assert_frozen_and_unchanged()
    test_manifest = loaded.manifests["test"]

    def latest_attempt(
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        boundaries = [
            int(record["sequence"])
            for record in records
            if record["event"] == "stage_boundary"
        ]
        if not boundaries:
            return ()
        latest = max(boundaries)
        return tuple(record for record in records if int(record["sequence"]) >= latest)

    training_records = tuple(
        record
        for phase, records in loaded.runtime_firewall_traces.items()
        if phase != "producer"
        for record in latest_attempt(records)
    )
    all_records = tuple(
        (*latest_attempt(loaded.runtime_firewall_traces.get("producer", ())),
         *training_records)
    )
    gradient_events = tuple(
        record for record in training_records if record["event"] == "gradient_batch"
    )
    observed_gradient_schemas = tuple(
        sorted(
            {
                tuple(sorted(record["payload"]["tensors"]))
                for record in gradient_events
            }
        )
    )
    producer_payloads = tuple(
        record
        for record in latest_attempt(
            loaded.runtime_firewall_traces.get("producer", ())
        )
        if record["event"] == "tensor_payload"
    )
    observed_source_schemas = {
        tuple(sorted(record["payload"]["tensors"])) for record in producer_payloads
    }
    observed_source_schema = (
        next(iter(observed_source_schemas))
        if len(observed_source_schemas) == 1
        else ()
    )
    stage_boundaries = tuple(
        record for record in all_records if record["event"] == "stage_boundary"
    )
    source_hashes = {
        record["payload"].get("sourceTreeSha256") for record in stage_boundaries
    }
    observed_source_tree_sha256 = (
        next(iter(source_hashes)) if len(source_hashes) == 1 else ""
    )
    observed_backbone_hashes: list[str] = []
    for record in training_records:
        if record["event"] != "backbone_boundary":
            continue
        payload = record["payload"]
        if payload.get("phase") == "backbone" and payload.get("boundary") != (
            "selected_checkpoint"
        ):
            continue
        observed_backbone_hashes.append(str(payload.get("sha256", "")))

    def _contains_forbidden(value: object) -> bool:
        normalized = str(value).strip().lower()
        return any(term in normalized for term in _FORBIDDEN_RUNTIME_KEYS)

    forbidden_gradient_read_count = sum(
        _contains_forbidden(key)
        for record in gradient_events
        for key in record["payload"]["tensors"]
    )
    file_events = tuple(
        record for record in training_records if record["event"] == "file_read"
    )
    # Only serialized keys from learner-visible data archives can introduce a
    # command channel.  Model checkpoint metadata such as ``actionChannels=0``
    # documents absence and is not an optimization tensor.
    nonanalytic_command_read_count = sum(
        _contains_forbidden(key)
        for record in file_events
        if str(record["payload"].get("role", "")).startswith("trainer_archive:")
        for key in record["payload"].get("serializedKeys", ())
    )
    optimizer_events = tuple(
        record
        for record in training_records
        if record["event"] == "optimizer_constructed"
        and record["payload"].get("phase") != "backbone"
    )
    backbone_in_optimizer = any(
        not record["payload"].get("protectedParameters")
        or record["payload"].get("protectedOverlap") is not False
        for record in optimizer_events
    )
    runtime_mount_inventory_count = sum(
        record["event"] == "stage_boundary"
        and type(record["payload"].get("mounts")) is list
        for record in training_records
    ) + sum(
        record["event"] in {"mount_manifest", "recursive_manifest"}
        for record in training_records
    )
    forbidden_training_mount_count = 0
    for record in training_records:
        if record["event"] == "stage_boundary":
            for mount in record["payload"].get("mounts", ()):
                serialized = " ".join(
                    str(mount.get(field, ""))
                    for field in ("root", "mountPoint", "source")
                ).lower()
                forbidden_training_mount_count += any(
                    term in serialized for term in _PRIVATE_MOUNT_TERMS
                )
        elif record["event"] == "mount_manifest":
            serialized = " ".join(
                str(record["payload"].get(field, ""))
                for field in ("path", "resolvedPath")
            ).lower()
            forbidden_training_mount_count += any(
                term in serialized for term in _PRIVATE_MOUNT_TERMS
            )
    expected_learner_attestations = len(
        tuple(
            phase
            for phase in loaded.runtime_firewall_traces
            if phase != "producer"
        )
    )
    learner_manifest_reads = tuple(
        record
        for record in training_records
        if record["event"] == "file_read"
        and record["payload"].get("role") == "learner_source_manifest"
    )
    learner_bundle_hashes = {
        str(record["payload"].get("semanticSha256", ""))
        for record in learner_manifest_reads
    }
    observed_learner_bundle_sha256 = (
        next(iter(learner_bundle_hashes))
        if len(learner_bundle_hashes) == 1
        else ""
    )
    learner_source_inventories = tuple(
        record
        for record in training_records
        if record["event"] == "recursive_manifest"
        and record["payload"].get("role") == "learner_source_bundle"
    )
    sealed_learner_files = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in loaded.learner_source_manifest["files"]
    }
    sealed_learner_directories = {
        parent.as_posix()
        for path in sealed_learner_files
        for parent in Path(path).parents
        if parent.as_posix() != "."
    }
    sealed_learner_directories.add("blocket_league")
    forbidden_learner_source_file_count = 0
    learner_source_file_mismatch_count = 0
    for record in learner_source_inventories:
        entries = record["payload"].get("entries")
        if type(entries) is not list:
            learner_source_file_mismatch_count += len(sealed_learner_files) + 1
            continue
        observed_files: dict[str, tuple[int, str]] = {}
        observed_directories: set[str] = set()
        unsafe_paths = 0
        for item in entries:
            if type(item) is not dict or type(item.get("path")) is not str:
                unsafe_paths += 1
                continue
            if item.get("kind") == "file":
                path = str(item["path"])
                if path == "learner-source-manifest.json":
                    continue
                try:
                    observed_files[path] = (
                        int(item["bytes"]),
                        str(item["sha256"]),
                    )
                except (KeyError, TypeError, ValueError):
                    unsafe_paths += 1
            elif item.get("kind") == "directory":
                observed_directories.add(str(item["path"]))
            else:
                unsafe_paths += 1
        extra = set(observed_files) - set(sealed_learner_files)
        missing = set(sealed_learner_files) - set(observed_files)
        changed = {
            path
            for path in set(observed_files).intersection(sealed_learner_files)
            if observed_files[path] != sealed_learner_files[path]
        }
        extra_directories = observed_directories - sealed_learner_directories
        missing_directories = sealed_learner_directories - observed_directories
        forbidden_learner_source_file_count += (
            len(extra) + len(extra_directories) + unsafe_paths
        )
        learner_source_file_mismatch_count += (
            len(extra)
            + len(missing)
            + len(changed)
            + len(extra_directories)
            + len(missing_directories)
            + unsafe_paths
        )
    learner_cache_inventories = tuple(
        record
        for record in training_records
        if record["event"] == "recursive_manifest"
        and str(record["payload"].get("role", "")).startswith("learner_cache:")
    )
    learner_cache_python_code_file_count = sum(
        int(record["payload"].get("pythonCodeFileCount", -1))
        for record in learner_cache_inventories
    )
    learner_cache_unsafe_path_count = sum(
        len(record["payload"].get("symbolicPaths", ()))
        + len(record["payload"].get("specialPaths", ()))
        for record in learner_cache_inventories
    )
    sealed_split_hash = _split_set_sha256(
        {
            split: loaded.manifests[split].sanitized_tensor_sha256
            for split in ("fit", "validation", "test")
        }
    )
    observed_split_hash = _split_set_sha256(loaded.observed_split_sha256)
    return FirewallAuditEvidence(
        sealed_archive_sha256=sealed_split_hash,
        observed_archive_sha256=observed_split_hash,
        sealed_source_tree_sha256=loaded.source_tree_sha256,
        observed_source_tree_sha256=observed_source_tree_sha256,
        sealed_source_schema=tuple(test_manifest.source_schema),
        observed_source_schema=observed_source_schema,
        sealed_gradient_schemas=(
            ("pixels",),
            tuple(test_manifest.optimization_schema),
        ),
        observed_gradient_schemas=observed_gradient_schemas,
        sealed_backbone_hash=loaded.backbone_hash,
        observed_backbone_hashes=tuple(observed_backbone_hashes),
        forbidden_gradient_read_count=forbidden_gradient_read_count,
        backbone_in_optimizer=backbone_in_optimizer,
        nonanalytic_command_read_count=nonanalytic_command_read_count,
        runtime_trace_event_count=len(all_records),
        runtime_gradient_batch_count=len(gradient_events),
        runtime_file_read_count=len(file_events),
        runtime_stage_boundary_count=len(stage_boundaries),
        runtime_mount_inventory_count=runtime_mount_inventory_count,
        forbidden_training_mount_count=forbidden_training_mount_count,
        sealed_learner_bundle_sha256=loaded.learner_source_tree_sha256,
        observed_learner_bundle_sha256=observed_learner_bundle_sha256,
        expected_learner_manifest_count=expected_learner_attestations,
        observed_learner_manifest_count=len(learner_manifest_reads),
        expected_learner_source_inventory_count=expected_learner_attestations,
        observed_learner_source_inventory_count=len(learner_source_inventories),
        forbidden_learner_source_file_count=(
            forbidden_learner_source_file_count
        ),
        learner_source_file_mismatch_count=learner_source_file_mismatch_count,
        expected_learner_cache_inventory_count=expected_learner_attestations,
        observed_learner_cache_inventory_count=len(learner_cache_inventories),
        learner_cache_python_code_file_count=(
            learner_cache_python_code_file_count
        ),
        learner_cache_unsafe_path_count=learner_cache_unsafe_path_count,
    )


def audit_gate1_postfreeze(loaded: LoadedPostFreezeSystem) -> GateAuditResult:
    return audit_gate_1(loaded.full.bundle.model, assemble_gate1_evidence(loaded))


@dataclass(frozen=True)
class HeldoutLatentTransitions:
    states: torch.Tensor
    efforts: torch.Tensor
    transition_count: int
    source_split: str = "test"
    source_manifest_sha256: str = ""


@torch.no_grad()
def collect_test_latent_transitions(
    loaded: LoadedPostFreezeSystem,
    *,
    minimum_transitions: int = REGISTERED_GATE3_TRANSITIONS,
    batch_size: int = 64,
) -> HeldoutLatentTransitions:
    """Encode at least 4,096 held-out visual transitions and infer efforts."""

    if minimum_transitions < REGISTERED_GATE3_TRANSITIONS:
        raise ValueError("the registered structural audit requires at least 4,096 transitions")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    loaded.assert_frozen_and_unchanged()
    experiment = loaded.training_summary.get("experimentConfig")
    if not isinstance(experiment, dict):
        raise ValueError("training summary is missing experimentConfig")
    transitions = int(experiment.get("transitions", 0))
    suite = make_optimization_suite(
        loaded.test_pixels,
        loaded.backbone.config,  # type: ignore[attr-defined]
        transitions=transitions,
    )
    present_contexts = suite["pixelContexts"][:, :-1].reshape(
        -1, *suite["pixelContexts"].shape[-3:]
    )
    successor_contexts = suite["pixelContexts"][:, 1:].reshape(
        -1, *suite["pixelContexts"].shape[-3:]
    )
    if present_contexts.shape[0] < minimum_transitions:
        raise ValueError(
            f"test split exposes only {present_contexts.shape[0]} transitions; "
            f"{minimum_transitions} are required"
        )
    present_contexts = present_contexts[:minimum_transitions]
    successor_contexts = successor_contexts[:minimum_transitions]
    model = loaded.full.bundle.model
    device = loaded.device
    state_chunks: list[torch.Tensor] = []
    effort_chunks: list[torch.Tensor] = []
    for start in range(0, minimum_transitions, batch_size):
        stop = min(start + batch_size, minimum_transitions)
        present = model.encode(present_contexts[start:stop].to(device).long())
        successor = model.encode(successor_contexts[start:stop].to(device).long())
        state_chunks.append(present.detach())
        effort_chunks.append(model.infer_latent_effort(present, successor).detach())
    loaded.assert_frozen_and_unchanged()
    return HeldoutLatentTransitions(
        states=torch.cat(state_chunks),
        efforts=torch.cat(effort_chunks),
        transition_count=minimum_transitions,
        source_manifest_sha256=loaded.manifests["test"].sanitized_tensor_sha256,
    )


def audit_gate3_postfreeze(
    loaded: LoadedPostFreezeSystem,
    *,
    batch_size: int = 64,
    audit_chunk_size: int = 32,
    thresholds: Gate3Thresholds = Gate3Thresholds(),
) -> tuple[GateAuditResult, HeldoutLatentTransitions]:
    transitions = collect_test_latent_transitions(
        loaded,
        minimum_transitions=max(REGISTERED_GATE3_TRANSITIONS, thresholds.minimum_states),
        batch_size=batch_size,
    )
    result = audit_gate_3(
        loaded.full.bundle.model.core,
        transitions.states,
        transitions.efforts,
        thresholds,
        production_step=loaded.full.bundle.model.step,
        chunk_size=audit_chunk_size,
    )
    loaded.assert_frozen_and_unchanged()
    return result, transitions


@dataclass(frozen=True)
class Gate4CollectionConfig:
    samples: int = REGISTERED_GATE4_CONTEXTS
    random_draws: int = REGISTERED_RANDOM_WRITES
    horizons: tuple[int, ...] = REGISTERED_HORIZONS
    write_amplitude: float = 0.05
    decode_amplitude: float = 0.50
    # Four keeps the differentiable 64x64 soft rollout and re-encoding JVP
    # comfortably inside an A100 while avoiding 128 serial transformer
    # launches. It changes no sample/write count.
    batch_size: int = 4
    random_seed: int = 151_910_737 + 4_004

    def __post_init__(self) -> None:
        if self.samples != REGISTERED_GATE4_CONTEXTS:
            raise ValueError("Gate 4 is locked to exactly 128 held-out contexts")
        if self.random_draws != REGISTERED_RANDOM_WRITES:
            raise ValueError("Gate 4 is locked to exactly 16 norm-matched random writes")
        if tuple(self.horizons) != REGISTERED_HORIZONS:
            raise ValueError("Gate 4 horizons are locked to 1, 2, and 4")
        if self.write_amplitude <= 0.0 or self.decode_amplitude <= 0.0:
            raise ValueError("Gate 4 intervention amplitudes must be positive")
        if self.batch_size < 1:
            raise ValueError("Gate 4 batch_size must be positive")


@dataclass(frozen=True)
class Gate4PathProvenance:
    """Independent live/sealed components of the Gate-4 code path seal."""

    path_code_sha256: str
    sealed_path_code_sha256: str
    path_backbone_sha256: str
    sealed_backbone_sha256: str
    path_extractor_sha256: str
    sealed_extractor_sha256: str
    path_source_tree_sha256: str
    sealed_source_tree_sha256: str
    path_fingerprint_sha256: str


def _gate4_code_sha256_from_manifest(manifest: Mapping[str, Any]) -> str:
    files = manifest.get("files")
    if type(files) is not list:
        raise ValueError("Gate 4 source manifest has no exact file inventory")
    by_path = {
        item.get("path"): item.get("sha256")
        for item in files
        if type(item) is dict
    }
    if any(
        path not in by_path
        or type(by_path[path]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", by_path[path]) is None
        for path in _GATE4_PATH_SOURCE_FILES
    ):
        raise ValueError("Gate 4 source manifest omits a registered path file")
    digest = hashlib.sha256()
    for path in _GATE4_PATH_SOURCE_FILES:
        encoded_path = path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(str(by_path[path]).encode("ascii"))
    return digest.hexdigest()


def _gate4_sealed_extractor_sha256(frozen: FrozenVariantBundle) -> str:
    """Hash the extractor state authenticated by the sealed direct checkpoint."""

    if _file_sha256(frozen.checkpoint_path) != frozen.checkpoint_sha256:
        raise ValueError("Gate 4 direct checkpoint changed after post-freeze loading")
    payload = torch.load(
        frozen.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if type(payload) is not dict or type(payload.get("writeField")) is not dict:
        raise ValueError("Gate 4 checkpoint has no sealed writeField state")
    sealed_extractor = copy.deepcopy(frozen.bundle.write_field).cpu()
    sealed_extractor.load_state_dict(payload["writeField"], strict=True)
    sealed_sha256 = module_tensor_hash(sealed_extractor)
    del sealed_extractor, payload
    return sealed_sha256


def _gate4_path_provenance(
    bundle: DirectModelBundle,
    *,
    sealed_source_manifest: Mapping[str, Any] | None = None,
    sealed_source_tree_sha256: str | None = None,
    sealed_backbone_sha256: str | None = None,
    sealed_extractor_sha256: str | None = None,
) -> Gate4PathProvenance:
    """Recompute live code/backbone/source hashes and bind them to their seals."""

    live_manifest = build_source_manifest()
    live_source_sha256 = str(live_manifest["treeSha256"])
    if sealed_source_manifest is None:
        sealed_source_manifest = live_manifest
    if sealed_source_tree_sha256 is None:
        sealed_source_tree_sha256 = str(sealed_source_manifest.get("treeSha256", ""))
    if sealed_backbone_sha256 is None:
        sealed_backbone_sha256 = bundle.model.encoder.sealed_backbone_hash
    live_extractor_sha256 = module_tensor_hash(bundle.write_field)
    if sealed_extractor_sha256 is None:
        sealed_extractor_sha256 = live_extractor_sha256
    live_code_sha256 = _gate4_code_sha256_from_manifest(live_manifest)
    sealed_code_sha256 = _gate4_code_sha256_from_manifest(sealed_source_manifest)
    live_backbone_sha256 = module_tensor_hash(bundle.model.encoder.backbone)
    fingerprint = gate4_path_fingerprint_sha256(
        code_sha256=live_code_sha256,
        backbone_sha256=live_backbone_sha256,
        extractor_sha256=live_extractor_sha256,
        source_tree_sha256=live_source_sha256,
        retention_path_kind=ACTIVATION_SUFFIX_RETENTION_PATH,
        horizons=REGISTERED_HORIZONS,
    )
    return Gate4PathProvenance(
        path_code_sha256=live_code_sha256,
        sealed_path_code_sha256=sealed_code_sha256,
        path_backbone_sha256=live_backbone_sha256,
        sealed_backbone_sha256=sealed_backbone_sha256,
        path_extractor_sha256=live_extractor_sha256,
        sealed_extractor_sha256=sealed_extractor_sha256,
        path_source_tree_sha256=live_source_sha256,
        sealed_source_tree_sha256=sealed_source_tree_sha256,
        path_fingerprint_sha256=fingerprint,
    )


def _cpu_unit_random_like(
    reference: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    value = torch.randn(
        reference.shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=reference.device, dtype=reference.dtype)
    flat = value.flatten(1)
    flat = flat / torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    return flat.reshape_as(value)


def _frozen_activation_rollout_states(
    bundle: DirectModelBundle,
    contexts: torch.Tensor,
    residual_write: torch.Tensor,
    horizons: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    """Exact activation-write -> frozen suffix -> soft feedback -> E path."""

    selected = tuple(sorted(int(horizon) for horizon in horizons))
    if not selected or selected[0] < 1:
        raise ValueError("Gate 4 derivative horizons must be positive")
    lens = bundle.lens
    encoder = bundle.model.encoder
    current = lens.pixel_probabilities(contexts)
    states: dict[int, torch.Tensor] = {}
    for transition in range(1, selected[-1] + 2):
        _, tokens = lens._soft_suffix_tokens(  # noqa: SLF001 - audited exact path
            current,
            residual_write=residual_write if transition == 1 else None,
        )
        completed_horizon = transition - 1
        if completed_horizon in selected:
            states[completed_horizon] = encoder.read_suffix_tokens(tokens)
        if transition <= selected[-1]:
            probabilities = torch.softmax(
                lens.backbone.unpatch_logits(tokens) / lens.temperature,
                dim=2,
            )
            current = torch.cat((current[:, 1:], probabilities[:, -1, None]), dim=1)
    return tuple(states[horizon] for horizon in selected)


def _gate4_derivative_proofs(
    bundle: DirectModelBundle,
    contexts: torch.Tensor,
    horizons: Sequence[int],
    generator: torch.Generator,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Collect numeric adjoint and explicit-Jacobian identities on heldout pixels."""

    lens = bundle.lens
    encoder = bundle.model.encoder
    zero_write = torch.zeros(
        contexts.shape[0],
        *lens.activation_shape,
        device=contexts.device,
        dtype=lens.backbone.pixel_embedding.weight.dtype,
    )
    write_tangent = _cpu_unit_random_like(zero_write, generator)

    def rollout_path(write: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _frozen_activation_rollout_states(
            bundle, contexts, write, horizons
        )

    _, rollout_jvps = torch.autograd.functional.jvp(
        rollout_path,
        zero_write,
        write_tangent,
        create_graph=False,
        strict=False,
    )
    vjp_write = zero_write.detach().requires_grad_(True)
    rollout_values = rollout_path(vjp_write)
    cotangents = tuple(
        _cpu_unit_random_like(value, generator) for value in rollout_values
    )
    jvp_products: list[torch.Tensor] = []
    vjp_products: list[torch.Tensor] = []
    jvp_norm_bounds: list[torch.Tensor] = []
    vjp_norm_bounds: list[torch.Tensor] = []
    for index, (value, directional, cotangent) in enumerate(
        zip(rollout_values, rollout_jvps, cotangents, strict=True)
    ):
        pullback = torch.autograd.grad(
            value,
            vjp_write,
            grad_outputs=cotangent,
            retain_graph=index + 1 < len(rollout_values),
            create_graph=False,
        )[0]
        jvp_products.append((directional * cotangent).flatten(1).sum(dim=-1))
        vjp_products.append((write_tangent * pullback).flatten(1).sum(dim=-1))
        jvp_norm_bounds.append(
            torch.linalg.vector_norm(directional.flatten(1), dim=-1)
            * torch.linalg.vector_norm(cotangent.flatten(1), dim=-1)
        )
        vjp_norm_bounds.append(
            torch.linalg.vector_norm(write_tangent.flatten(1), dim=-1)
            * torch.linalg.vector_norm(pullback.flatten(1), dim=-1)
        )

    prefix = lens.soft_prefix_activation(contexts).detach()
    prefix_tangent = _cpu_unit_random_like(prefix, generator)
    explicit_jacobian = encoder.state_jacobian_from_activation(
        prefix, create_graph=False
    )
    explicit_product = torch.einsum(
        "bna,ba->bn", explicit_jacobian, prefix_tangent.flatten(1)
    )
    _, independent_product = torch.autograd.functional.jvp(
        encoder.from_activation,
        prefix,
        prefix_tangent,
        create_graph=False,
        strict=False,
    )
    return (
        torch.stack(jvp_products, dim=-1).detach(),
        torch.stack(vjp_products, dim=-1).detach(),
        torch.stack(jvp_norm_bounds, dim=-1).detach(),
        torch.stack(vjp_norm_bounds, dim=-1).detach(),
        explicit_product.detach(),
        independent_product.detach(),
    )


def _expanded_registered_effects(
    bundle: DirectModelBundle,
    contexts: torch.Tensor,
    states: torch.Tensor,
    basis: torch.Tensor,
    amplitude: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lens = bundle.lens
    batch, port_size = states.shape[0], bundle.model.core.config.port_size
    expanded_context = contexts[:, None].expand(
        batch, port_size, *contexts.shape[1:]
    ).reshape(batch * port_size, *contexts.shape[1:])
    expanded_basis = basis[:, None].expand(
        batch, port_size, *basis.shape[1:]
    ).reshape(batch * port_size, *basis.shape[1:])
    pulses = amplitude * torch.eye(
        port_size, dtype=basis.dtype, device=basis.device
    )[None].expand(batch, port_size, port_size).reshape(batch * port_size, port_size)
    zero = torch.zeros_like(pulses)
    baseline = lens.rollout(expanded_context, expanded_basis, zero, horizons=(1,))[1]
    positive = lens.rollout(expanded_context, expanded_basis, pulses, horizons=(1,))[1]
    negative = lens.rollout(expanded_context, expanded_basis, -pulses, horizons=(1,))[1]
    tail = positive.shape[1:]
    return (
        positive.reshape(batch, port_size, *tail),
        negative.reshape(batch, port_size, *tail),
        baseline.reshape(batch, port_size, *tail),
    )


def _norm_matched_random_effects(
    bundle: DirectModelBundle,
    contexts: torch.Tensor,
    baseline: torch.Tensor,
    *,
    amplitude: float,
    random_draws: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, bool]:
    """Measure full-pixel effects of random ambient residual-stream writes."""

    lens = bundle.lens
    batch = contexts.shape[0]
    port_size = bundle.model.core.config.port_size
    repeat = port_size * random_draws
    expanded_contexts = contexts[:, None].expand(
        batch, repeat, *contexts.shape[1:]
    ).reshape(batch * repeat, *contexts.shape[1:])
    random_cpu = torch.randn(
        batch,
        repeat,
        *lens.activation_shape,
        generator=generator,
        dtype=torch.float32,
    )
    flat = random_cpu.flatten(2)
    flat = amplitude * flat / torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(1e-12)
    random_write = flat.reshape_as(random_cpu).to(
        device=contexts.device,
        dtype=bundle.model.encoder.backbone.pixel_embedding.weight.dtype,
    )
    random_output = lens.soft_forward(
        expanded_contexts,
        residual_write=random_write.reshape(batch * repeat, *lens.activation_shape),
    )[:, -1]
    baseline_frame = baseline[:, 0].reshape(batch, -1)
    random_flat = random_output.reshape(batch, repeat, -1)
    effects = torch.linalg.vector_norm(random_flat - baseline_frame[:, None], dim=-1)
    effects = effects.reshape(batch, port_size, random_draws)
    norms = torch.linalg.vector_norm(flat, dim=-1)
    verified = bool(
        torch.allclose(
            norms,
            torch.full_like(norms, amplitude),
            atol=1e-6,
            rtol=1e-6,
        )
    )
    return effects, verified


def _decode_reencode_directions(
    bundle: DirectModelBundle,
    contexts: torch.Tensor,
    states: torch.Tensor,
    *,
    amplitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy renderer-cycle diagnostic, never Gate-4 pass/fail evidence.

    The registered retention path is collected by
    ``state_response_jacobians`` in :func:`_gate4_batch`.  This helper remains
    available only for explicitly labelled exploratory diagnostics.
    """

    model = bundle.model
    batch = states.shape[0]
    port_size = model.core.config.port_size
    repeated_states = states[:, None].expand(batch, port_size, states.shape[-1]).reshape(
        batch * port_size, states.shape[-1]
    )
    pulses = amplitude * torch.eye(
        port_size, device=states.device, dtype=states.dtype
    )[None].expand(batch, port_size, port_size).reshape(batch * port_size, port_size)
    plus_state = model.step(repeated_states, pulses)
    minus_state = model.step(repeated_states, -pulses)
    registered = ((plus_state - minus_state) / (2.0 * amplitude)).reshape(
        batch, port_size, states.shape[-1]
    ).transpose(1, 2)

    plus_frame = model.render(plus_state).argmax(dim=1).to(contexts.dtype)
    minus_frame = model.render(minus_state).argmax(dim=1).to(contexts.dtype)
    repeated_context = contexts[:, None].expand(
        batch, port_size, *contexts.shape[1:]
    ).reshape(batch * port_size, *contexts.shape[1:])
    plus_context = torch.cat((repeated_context[:, 1:], plus_frame[:, None]), dim=1)
    minus_context = torch.cat((repeated_context[:, 1:], minus_frame[:, None]), dim=1)
    plus_cycle = model.encode(plus_context.long())
    minus_cycle = model.encode(minus_context.long())
    cycled = ((plus_cycle - minus_cycle) / (2.0 * amplitude)).reshape(
        batch, port_size, states.shape[-1]
    ).transpose(1, 2)
    return registered, cycled


def _gate4_batch(
    bundle: DirectModelBundle,
    contexts: torch.Tensor,
    config: Gate4CollectionConfig,
    generator: torch.Generator,
    path_provenance: Gate4PathProvenance | None = None,
) -> LensAuditEvidence:
    # Parameters are frozen, but JVPs still need a local autograd tape for the
    # pulse variable.  Every returned tensor is detached below.
    with torch.enable_grad():
        states = bundle.model.encode(contexts)
        activation_covectors = activation_observable_covectors(
            bundle.lens,
            contexts,
            bundle.probes,
            horizons=config.horizons,
            create_graph=False,
        )
        source_activation = bundle.model.encoder.prefix_activation(contexts).detach()
        bundle.write_field.assert_frozen_parameter_free()
        extraction = bundle.write_field(activation_covectors, source_activation)
        basis = extraction.jacobian.write_basis
        flat_basis = basis.flatten(1, 3)
        extracted_gram = flat_basis.transpose(-1, -2) @ flat_basis
        lens_response = bundle.lens.state_response_jacobians(
            contexts,
            basis,
            bundle.model.encoder.read_suffix_tokens,
            horizons=config.horizons,
            create_graph=False,
        )
        with differentiable_attention_backend(states):
            ph_response = direct_dynamics_pulse_responses(
                bundle.model.step,
                states,
                bundle.model.core.config.port_size,
                horizons=config.horizons,
                create_graph=False,
            )
        positive, negative, baseline = _expanded_registered_effects(
            bundle, contexts, states, basis, config.write_amplitude
        )
        random_norms, norm_verified = _norm_matched_random_effects(
            bundle,
            contexts,
            baseline,
            amplitude=config.write_amplitude,
            random_draws=config.random_draws,
            generator=generator,
        )
        (
            adjoint_jvp,
            adjoint_vjp,
            adjoint_jvp_bound,
            adjoint_vjp_bound,
            explicit_state_jacobian,
            independent_state_jvp,
        ) = _gate4_derivative_proofs(
            bundle,
            contexts,
            config.horizons,
            generator,
        )
    if path_provenance is None:
        path_provenance = _gate4_path_provenance(bundle)
    return LensAuditEvidence(
        lens_responses={
            horizon: value.detach() for horizon, value in lens_response.jacobians.items()
        },
        ph_responses={
            horizon: value.detach() for horizon, value in ph_response.jacobians.items()
        },
        positive_effects=positive.detach(),
        negative_effects=negative.detach(),
        baseline_effects=baseline.detach(),
        random_write_effect_norms=random_norms.detach(),
        adjoint_jvp_inner_products=adjoint_jvp.detach(),
        adjoint_vjp_inner_products=adjoint_vjp.detach(),
        adjoint_jvp_norm_bounds=adjoint_jvp_bound.detach(),
        adjoint_vjp_norm_bounds=adjoint_vjp_bound.detach(),
        explicit_state_jacobian_products=explicit_state_jacobian.detach(),
        independent_state_jvp_products=independent_state_jvp.detach(),
        extracted_port_gram_matrices=extracted_gram.detach(),
        extracted_port_singular_values=(
            extraction.jacobian.singular_values.detach()
        ),
        extracted_port_reported_orthonormality_defects=(
            extraction.jacobian.orthonormality_defect.detach()
        ),
        extracted_projected_signal_ratios=(
            extraction.projected_signal_ratio.detach()
        ),
        extracted_neighbor_indices=extraction.neighbor_indices.detach(),
        extracted_neighbor_fit_population=int(
            bundle.write_field.feature_locations.shape[0]
        ),
        path_code_sha256=path_provenance.path_code_sha256,
        sealed_path_code_sha256=path_provenance.sealed_path_code_sha256,
        path_backbone_sha256=path_provenance.path_backbone_sha256,
        sealed_backbone_sha256=path_provenance.sealed_backbone_sha256,
        path_extractor_sha256=path_provenance.path_extractor_sha256,
        sealed_extractor_sha256=path_provenance.sealed_extractor_sha256,
        path_source_tree_sha256=path_provenance.path_source_tree_sha256,
        sealed_source_tree_sha256=path_provenance.sealed_source_tree_sha256,
        path_fingerprint_sha256=path_provenance.path_fingerprint_sha256,
        random_writes_norm_matched=norm_verified,
        retention_path_kind=ACTIVATION_SUFFIX_RETENTION_PATH,
    )


def _concatenate_lens_evidence(chunks: Sequence[LensAuditEvidence]) -> LensAuditEvidence:
    if not chunks:
        raise ValueError("cannot concatenate an empty Gate 4 evidence list")
    horizons = tuple(sorted((chunks[0].lens_responses or {}).keys()))
    if horizons != REGISTERED_HORIZONS:
        raise ValueError("Gate 4 chunk horizons drifted")

    def concatenate(name: str) -> torch.Tensor:
        values = [getattr(chunk, name) for chunk in chunks]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise ValueError(f"Gate 4 chunk is missing {name}")
        return torch.cat(values, dim=0).detach()  # type: ignore[arg-type]

    def identical(name: str) -> str | None:
        values = tuple(getattr(chunk, name) for chunk in chunks)
        return values[0] if all(value == values[0] for value in values) else None

    return LensAuditEvidence(
        lens_responses={
            horizon: torch.cat(
                [(chunk.lens_responses or {})[horizon] for chunk in chunks], dim=0
            ).detach()
            for horizon in horizons
        },
        ph_responses={
            horizon: torch.cat(
                [(chunk.ph_responses or {})[horizon] for chunk in chunks], dim=0
            ).detach()
            for horizon in horizons
        },
        positive_effects=concatenate("positive_effects"),
        negative_effects=concatenate("negative_effects"),
        baseline_effects=concatenate("baseline_effects"),
        random_write_effect_norms=concatenate("random_write_effect_norms"),
        adjoint_jvp_inner_products=concatenate("adjoint_jvp_inner_products"),
        adjoint_vjp_inner_products=concatenate("adjoint_vjp_inner_products"),
        adjoint_jvp_norm_bounds=concatenate("adjoint_jvp_norm_bounds"),
        adjoint_vjp_norm_bounds=concatenate("adjoint_vjp_norm_bounds"),
        explicit_state_jacobian_products=concatenate(
            "explicit_state_jacobian_products"
        ),
        independent_state_jvp_products=concatenate(
            "independent_state_jvp_products"
        ),
        extracted_port_gram_matrices=concatenate("extracted_port_gram_matrices"),
        extracted_port_singular_values=concatenate(
            "extracted_port_singular_values"
        ),
        extracted_port_reported_orthonormality_defects=concatenate(
            "extracted_port_reported_orthonormality_defects"
        ),
        extracted_projected_signal_ratios=concatenate(
            "extracted_projected_signal_ratios"
        ),
        extracted_neighbor_indices=concatenate("extracted_neighbor_indices"),
        extracted_neighbor_fit_population=(
            chunks[0].extracted_neighbor_fit_population
            if all(
                chunk.extracted_neighbor_fit_population
                == chunks[0].extracted_neighbor_fit_population
                for chunk in chunks
            )
            else None
        ),
        path_code_sha256=identical("path_code_sha256"),
        sealed_path_code_sha256=identical("sealed_path_code_sha256"),
        path_backbone_sha256=identical("path_backbone_sha256"),
        sealed_backbone_sha256=identical("sealed_backbone_sha256"),
        path_extractor_sha256=identical("path_extractor_sha256"),
        sealed_extractor_sha256=identical("sealed_extractor_sha256"),
        path_source_tree_sha256=identical("path_source_tree_sha256"),
        sealed_source_tree_sha256=identical("sealed_source_tree_sha256"),
        path_fingerprint_sha256=identical("path_fingerprint_sha256"),
        random_writes_norm_matched=all(
            chunk.random_writes_norm_matched is True for chunk in chunks
        ),
        retention_path_kind=(
            ACTIVATION_SUFFIX_RETENTION_PATH
            if all(
                chunk.retention_path_kind == ACTIVATION_SUFFIX_RETENTION_PATH
                for chunk in chunks
            )
            else "mixed_or_unregistered_retention_path"
        ),
    )


def assemble_gate4_evidence(
    loaded: LoadedPostFreezeSystem,
    config: Gate4CollectionConfig = Gate4CollectionConfig(),
) -> LensAuditEvidence:
    """Compute the complete registered internal-port evidence after freezing."""

    loaded.assert_frozen_and_unchanged()
    experiment = loaded.training_summary.get("experimentConfig")
    if not isinstance(experiment, dict):
        raise ValueError("training summary is missing experimentConfig")
    suite = make_optimization_suite(
        loaded.test_pixels,
        loaded.backbone.config,  # type: ignore[attr-defined]
        transitions=int(experiment.get("transitions", 0)),
    )
    contexts = suite["pixelContexts"][:, 0]
    if contexts.shape[0] < config.samples:
        raise ValueError("test split has fewer than 128 independent Gate 4 contexts")
    contexts = contexts[: config.samples]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.random_seed)
    bundle = loaded.full.bundle
    source_manifest = loaded.training_summary.get("sourceManifest")
    if type(source_manifest) is not dict:
        raise ValueError("training summary is missing its sealed source manifest")
    path_provenance = _gate4_path_provenance(
        bundle,
        sealed_source_manifest=source_manifest,
        sealed_source_tree_sha256=loaded.source_tree_sha256,
        sealed_backbone_sha256=loaded.backbone_hash,
        sealed_extractor_sha256=_gate4_sealed_extractor_sha256(loaded.full),
    )
    chunks = []
    for start in range(0, config.samples, config.batch_size):
        stop = min(start + config.batch_size, config.samples)
        chunks.append(
            _gate4_batch(
                bundle,
                contexts[start:stop].to(loaded.device).long(),
                config,
                generator,
                path_provenance,
            )
        )
    evidence = _concatenate_lens_evidence(chunks)
    loaded.assert_frozen_and_unchanged()
    return evidence


def audit_gate4_postfreeze(
    loaded: LoadedPostFreezeSystem,
    config: Gate4CollectionConfig = Gate4CollectionConfig(),
    thresholds: Gate4Thresholds = Gate4Thresholds(),
) -> tuple[GateAuditResult, LensAuditEvidence]:
    if thresholds.minimum_samples != REGISTERED_GATE4_CONTEXTS:
        raise ValueError("registered Gate 4 minimum_samples cannot be changed")
    if thresholds.minimum_random_draws != REGISTERED_RANDOM_WRITES:
        raise ValueError("registered Gate 4 random-write count cannot be changed")
    evidence = assemble_gate4_evidence(loaded, config)
    return audit_gate_4(evidence, thresholds), evidence


@dataclass(frozen=True)
class ModelRealizabilityResult:
    calibration: CalibrationResult
    metrics: RealizabilityMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration.as_dict(),
            "realizability": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class InterfaceRealizabilityResult:
    interface_name: str
    models: Mapping[str, ModelRealizabilityResult]

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("interface realizability result has no models")
        calibrations = [result.calibration for result in self.models.values()]
        if any(value.interface_name != self.interface_name for value in calibrations):
            raise ValueError("interface result contains another interface calibration")
        if len({value.response_evidence_sha256 for value in calibrations}) != 1:
            raise ValueError("models did not share one physical calibration bank")
        if len({value.selection for value in calibrations}) != 1:
            raise ValueError("models did not share one calibration selection")
        if any(value.additional_environment_steps != 0 for value in calibrations):
            raise ValueError("a model performed an extra physical calibration probe")
        if len({value.environment_steps for value in calibrations}) != 1:
            raise ValueError("shared calibration-bank step accounting differs")
        metrics = [result.metrics for result in self.models.values()]
        heldout_hashes = {value.response_evidence_sha256 for value in metrics}
        if None in heldout_hashes or len(heldout_hashes) != 1:
            raise ValueError("models did not share one held-out physical response bank")
        if any(value.additional_environment_steps != 0 for value in metrics):
            raise ValueError("a model performed an extra held-out physical probe")
        if len({value.environment_steps for value in metrics}) != 1:
            raise ValueError("shared held-out-bank step accounting differs")

    def to_dict(self) -> dict[str, Any]:
        first = next(iter(self.models.values())).calibration
        first_metrics = next(iter(self.models.values())).metrics
        return {
            "interface": self.interface_name,
            "sharedPhysicalCalibrationBankSha256": first.response_evidence_sha256,
            "totalPhysicalCalibrationEnvironmentSteps": first.environment_steps,
            "additionalModelSpecificCalibrationEnvironmentSteps": 0,
            "sharedPhysicalHeldoutBankSha256": first_metrics.response_evidence_sha256,
            "totalPhysicalHeldoutEnvironmentSteps": first_metrics.environment_steps,
            "additionalModelSpecificHeldoutEnvironmentSteps": 0,
            "models": {name: value.to_dict() for name, value in self.models.items()},
        }


@dataclass(frozen=True)
class PhysicalRealizabilityResult:
    system_name: str
    interfaces: Mapping[str, InterfaceRealizabilityResult]
    neural_hashes_before: Mapping[str, str]
    neural_hashes_after: Mapping[str, str]
    calibration_pairs_per_axis: int = 4
    heldout_pairs_per_axis: int = REGISTERED_REALIZABILITY_STATES_PER_AXIS

    @property
    def neural_hashes_unchanged(self) -> bool:
        return dict(self.neural_hashes_before) == dict(self.neural_hashes_after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system_name,
            "calibrationPairsPerAxis": self.calibration_pairs_per_axis,
            "heldoutPairsPerAxis": self.heldout_pairs_per_axis,
            "neuralHashesUnchanged": self.neural_hashes_unchanged,
            "interfaces": {
                name: result.to_dict() for name, result in self.interfaces.items()
            },
        }


def build_frozen_activation_world_model(
    loaded: LoadedPostFreezeSystem,
) -> FrozenActivationWriteWorldModel:
    """Reconstruct the registered generic planner from the sealed full bundle.

    Only the action-free visual encoder, its exact empirical Jacobian
    extractor, fixed pixels-only probes, and frozen video-transformer lens are
    retained.  In particular, the pH core, latent renderer, and
    inferred-effort network are not reachable from the returned module.
    """

    loaded.assert_frozen_and_unchanged()
    bundle = loaded.full.bundle
    result = FrozenActivationWriteWorldModel(
        bundle.model.encoder,
        bundle.write_field,
        bundle.lens,
        bundle.probes,
    )
    result.assert_frozen_and_unchanged()
    return result


def _comprehensive_evaluation_seal(
    loaded: LoadedPostFreezeSystem,
    *,
    activation_world_model: FrozenActivationWriteWorldModel | None = None,
) -> FrozenEvaluationSeal:
    modules: dict[str, nn.Module] = {
        f"variant-{name}": frozen.bundle.model
        for name, frozen in loaded.variants.items()
    }
    modules.update(
        {
            **{
                f"unstructured-{name}": module
                for name, module in independent_evaluation_modules(
                    loaded.independent_baseline.bundle
                ).items()
            },
            "unstructured-lens": loaded.independent_baseline.bundle.lens,
        }
    )
    if activation_world_model is not None:
        modules["activation-world-model"] = activation_world_model
    return FrozenEvaluationSeal.capture(modules)


def run_physical_realizability(
    loaded: LoadedPostFreezeSystem,
    *,
    candidate_pool_size: int = 64,
    candidate_seed: int = 151_910_737 + 60_000,
    heldout_seed: int = 151_910_737 + 70_000,
) -> PhysicalRealizabilityResult:
    """Run native and unseen calibration/realizability after full freeze.

    Every model receives exactly four paired calibration probes per physical
    axis and exactly 128 disjoint held-out paired probes per axis.  The sole
    fitted object is its constant ridge matrix.
    """

    if candidate_pool_size < 4:
        raise ValueError("D-optimal calibration needs at least four candidates")
    if candidate_seed == heldout_seed:
        raise ValueError("calibration and held-out physical pools must be disjoint")
    loaded.assert_frozen_and_unchanged()
    system_spec = DIRECT_SYSTEMS[loaded.system_name]
    system = evaluation_system_from_direct_spec(system_spec)
    history = loaded.backbone.config.history_frames  # type: ignore[attr-defined]
    image_size = loaded.backbone.config.image_size  # type: ignore[attr-defined]
    plant = builtin_pixel_plant(system)
    calibration_candidates = make_builtin_probe_candidates(
        system,
        history_frames=history,
        count=candidate_pool_size,
        seed=candidate_seed,
        image_size=image_size,
    )
    heldout_candidates = make_builtin_probe_candidates(
        system,
        history_frames=history,
        count=REGISTERED_REALIZABILITY_STATES_PER_AXIS,
        seed=heldout_seed,
        image_size=image_size,
    )
    calibration_ids = {candidate.identifier for candidate in calibration_candidates}
    heldout_ids = {candidate.identifier for candidate in heldout_candidates}
    if calibration_ids & heldout_ids:
        raise AssertionError("physical calibration and held-out candidate pools overlap")

    activation_world_model = build_frozen_activation_world_model(loaded)
    seal = _comprehensive_evaluation_seal(
        loaded, activation_world_model=activation_world_model
    )
    model_pairs: dict[str, tuple[nn.Module, nn.Module]] = {
        name: (frozen.bundle.model.encoder, frozen.bundle.model.core)
        for name, frozen in loaded.variants.items()
    }
    model_pairs["unstructured"] = (
        loaded.independent_baseline.encoder,
        adapt_dynamics_for_evaluation(loaded.independent_baseline.dynamics),
    )
    # One response-blind selection and one raw +/- pixel bank are shared by
    # every registered model.  Normalized max-min D-optimality prevents the
    # primary pH model's chart from choosing states that are ill-conditioned
    # for a comparator.
    shared_selection = select_shared_maximin_probe_states(
        model_pairs,
        activation_world_model,
        calibration_candidates,
        system,
        seal=seal,
    )
    expected_selection_models = tuple(sorted((*model_pairs, "activation")))
    if (
        shared_selection.selection_method
        != "shared_maximin_normalized_d_optimal"
        or shared_selection.selection_model_names != expected_selection_models
        or any(
            indices != shared_selection.indices_by_axis[0]
            for indices in shared_selection.indices_by_axis
        )
    ):
        raise AssertionError("shared calibration selection fairness seal drifted")
    interface_results: dict[str, InterfaceRealizabilityResult] = {}
    for interface_name, interface in fixed_interfaces(system).items():
        response_bank = collect_paired_calibration_response_bank(
            plant,
            calibration_candidates,
            shared_selection,
            system,
            interface,
            seal=seal,
        )
        heldout_response_bank = collect_paired_heldout_response_bank(
            plant,
            heldout_candidates,
            system,
            interface,
            seal=seal,
            states_per_axis=REGISTERED_REALIZABILITY_STATES_PER_AXIS,
        )
        models: dict[str, ModelRealizabilityResult] = {}
        for model_name, (encoder, dynamics) in model_pairs.items():
            calibration = fit_interface_calibration_from_response_bank(
                encoder,
                dynamics,
                calibration_candidates,
                response_bank,
                system,
                interface,
                seal=seal,
                model_name=model_name,
            )
            if (
                calibration.paired_states_per_axis != 4
                or calibration.gradient_updates != 0
                or calibration.additional_environment_steps != 0
                or calibration.response_evidence_sha256
                != response_bank.evidence_sha256
            ):
                raise AssertionError("physical calibration query/update budget drifted")
            metrics = evaluate_heldout_realizability_from_response_bank(
                encoder,
                dynamics,
                heldout_candidates,
                system,
                interface,
                calibration,
                heldout_response_bank,
                seal=seal,
            )
            if (
                metrics.samples_per_axis != REGISTERED_REALIZABILITY_STATES_PER_AXIS
                or metrics.additional_environment_steps != 0
                or metrics.response_evidence_sha256
                != heldout_response_bank.evidence_sha256
            ):
                raise AssertionError("held-out realizability query budget drifted")
            models[model_name] = ModelRealizabilityResult(calibration, metrics)
        # The generic planner refits on the exact shared raw-pixel bank with
        # zero environment steps.  Its design is D_h E U / dt rather than B(x),
        # so finite response-frame mismatch is absorbed without a pH call.
        activation_calibration = calibrate_activation_interface_after_freeze(
            activation_world_model,
            calibration_candidates,
            response_bank,
            system,
            interface,
            seal=seal,
        )
        activation_metrics = evaluate_heldout_activation_from_response_bank(
            activation_world_model,
            heldout_candidates,
            system,
            interface,
            activation_calibration,
            heldout_response_bank,
            seal=seal,
        )
        if (
            activation_calibration.paired_states_per_axis != 4
            or activation_calibration.gradient_updates != 0
            or activation_calibration.additional_environment_steps != 0
            or activation_calibration.response_evidence_sha256
            != response_bank.evidence_sha256
            or activation_metrics.samples_per_axis
            != REGISTERED_REALIZABILITY_STATES_PER_AXIS
            or activation_metrics.additional_environment_steps != 0
            or activation_metrics.response_evidence_sha256
            != heldout_response_bank.evidence_sha256
        ):
            raise AssertionError("activation baseline query/update budget drifted")
        models["activation"] = ModelRealizabilityResult(
            activation_calibration, activation_metrics
        )
        interface_results[interface_name] = InterfaceRealizabilityResult(
            interface_name, models
        )
    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    loaded.assert_frozen_and_unchanged()
    # ``assert_unchanged`` recomputed the evaluation module hashes using the
    # seal's canonical hashing routine.  Preserve those canonical values
    # rather than mixing hash implementations in the report.
    hashes_after = dict(seal.hashes)
    return PhysicalRealizabilityResult(
        system_name=loaded.system_name,
        interfaces=interface_results,
        neural_hashes_before=dict(seal.hashes),
        neural_hashes_after=hashes_after,
    )


@dataclass(frozen=True)
class ControlShard:
    """Detached output from one distributed subset of the 64 episodes."""

    interface_name: str
    start: int
    stop: int
    total_episodes: int
    result: ControlResult

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.stop <= self.total_episodes:
            raise ValueError("invalid control-shard interval")
        if self.result.interface_name != self.interface_name:
            raise ValueError("control-shard interface mismatch")
        if self.result.episodes != self.stop - self.start:
            raise ValueError("control-shard result length mismatch")


def registered_control_shard_ranges(shard_size: int = 4) -> tuple[tuple[int, int], ...]:
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    return tuple(
        (start, min(start + shard_size, REGISTERED_CONTROL_EPISODES))
        for start in range(0, REGISTERED_CONTROL_EPISODES, shard_size)
    )


def run_control_shard(
    loaded: LoadedPostFreezeSystem,
    physical: PhysicalRealizabilityResult,
    *,
    interface_name: str,
    start: int,
    stop: int,
    episode_seed: int = 151_910_737 + 80_000,
    planner_seed: int = 151_910_737 + 90_000,
    class_weights: torch.Tensor | None = None,
) -> ControlShard:
    """Run one independently schedulable slice of the locked control suite.

    The activation planner is reconstructed internally from the sealed full
    bundle and its separately fitted activation-Jacobian calibration.  A
    caller therefore cannot silently substitute an action-conditioned model
    or omit this registered baseline.
    """

    if not 0 <= start < stop <= REGISTERED_CONTROL_EPISODES:
        raise ValueError("control shard is outside the registered 64 episodes")
    if physical.system_name != loaded.system_name:
        raise ValueError("physical calibration belongs to another system")
    interface_result = physical.interfaces.get(interface_name)
    if interface_result is None:
        raise ValueError("missing physical calibration for the requested interface")
    for required in (
        "full",
        "unstructured",
        "activation",
        "no_jacobian",
        "shuffled_lens",
    ):
        if required not in interface_result.models:
            raise ValueError(f"missing {required} calibration for control")

    loaded.assert_frozen_and_unchanged()
    activation_world_model = build_frozen_activation_world_model(loaded)
    system = evaluation_system_from_direct_spec(DIRECT_SYSTEMS[loaded.system_name])
    interfaces = fixed_interfaces(system)
    if interface_name not in interfaces:
        raise ValueError("interface is not the preregistered native or unseen mapping")
    full = loaded.full.bundle.model
    modules: dict[str, nn.Module] = {
        f"variant-{name}": frozen.bundle.model
        for name, frozen in loaded.variants.items()
    }
    modules.update(
        {
            **{
                f"unstructured-{name}": module
                for name, module in independent_evaluation_modules(
                    loaded.independent_baseline.bundle
                ).items()
            },
            "unstructured-lens": loaded.independent_baseline.bundle.lens,
            "activation-world-model": activation_world_model,
        }
    )
    seal = FrozenEvaluationSeal.capture(modules)
    episodes = make_builtin_control_episodes(
        system,
        history_frames=loaded.backbone.config.history_frames,  # type: ignore[attr-defined]
        count=REGISTERED_CONTROL_EPISODES,
        seed=episode_seed,
        image_size=loaded.backbone.config.image_size,  # type: ignore[attr-defined]
    )[start:stop]
    result = evaluate_closed_loop_controllers(
        episodes,
        system,
        builtin_pixel_plant(system),
        interfaces[interface_name],
        full.encoder,
        full.renderer,
        full.core,
        adapt_dynamics_for_evaluation(loaded.independent_baseline.dynamics),
        interface_result.models["full"].calibration,
        interface_result.models["unstructured"].calibration,
        unstructured_encoder=loaded.independent_baseline.encoder,
        unstructured_renderer=loaded.independent_baseline.renderer,
        seal=seal,
        activation_rollout=activation_world_model,
        activation_calibration=interface_result.models["activation"].calibration,
        additional_latent_planners={
            variant: FrozenLatentPlannerSpec(
                encoder=loaded.variants[variant].bundle.model.encoder,
                renderer=loaded.variants[variant].bundle.model.renderer,
                dynamics=loaded.variants[variant].bundle.model.core,
                calibration=interface_result.models[variant].calibration,
            )
            for variant in ("no_jacobian", "shuffled_lens")
        },
        seed=planner_seed,
        episode_offset=start,
        class_weights=class_weights,
    )
    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    loaded.assert_frozen_and_unchanged()
    return ControlShard(
        interface_name=interface_name,
        start=start,
        stop=stop,
        total_episodes=REGISTERED_CONTROL_EPISODES,
        result=result,
    )


def merge_control_shards(shards: Sequence[ControlShard]) -> ControlResult:
    """Merge only exact, gap-free coverage of all 64 registered episodes."""

    if not shards:
        raise ValueError("missing all registered control shards")
    ordered = sorted(shards, key=lambda shard: shard.start)
    if any(shard.total_episodes != REGISTERED_CONTROL_EPISODES for shard in ordered):
        raise ValueError("control shard total differs from the registered 64 episodes")
    interfaces = {shard.interface_name for shard in ordered}
    if len(interfaces) != 1:
        raise ValueError("cannot merge control shards from different interfaces")
    cursor = 0
    for shard in ordered:
        if shard.start != cursor:
            raise ValueError("control shard coverage has a gap or overlap")
        cursor = shard.stop
    if cursor != REGISTERED_CONTROL_EPISODES:
        raise ValueError("control shard coverage is incomplete")
    controller_names = set(ordered[0].result.errors)
    required_controllers = {
        "structured",
        "unstructured",
        "activation",
        "no_jacobian",
        "shuffled_lens",
        "coast",
        "random",
    }
    if not required_controllers.issubset(controller_names):
        raise ValueError("a control shard is missing a registered controller")
    if any(set(shard.result.errors) != controller_names for shard in ordered):
        raise ValueError("control-shard controller schemas differ")
    if any(shard.result.control_steps != ordered[0].result.control_steps for shard in ordered):
        raise ValueError("control-shard horizons differ")
    if any(dict(shard.result.planner_budget) != dict(ordered[0].result.planner_budget) for shard in ordered):
        raise ValueError("control-shard planner budgets differ")
    physical_protocols = [shard.result.physical_protocol for shard in ordered]
    if any(protocol is not None for protocol in physical_protocols) and not all(
        protocol is not None for protocol in physical_protocols
    ):
        raise ValueError("control shards mix sealed and legacy physical protocols")
    physical_protocol = physical_protocols[0]
    if physical_protocol is not None and any(
        protocol != physical_protocol for protocol in physical_protocols[1:]
    ):
        raise ValueError("control-shard physical protocols differ")
    errors = {
        name: tuple(
            value
            for shard in ordered
            for value in shard.result.errors[name]
        )
        for name in sorted(controller_names)
    }
    if any(len(values) != REGISTERED_CONTROL_EPISODES for values in errors.values()):
        raise ValueError("merged controller evidence does not contain 64 errors")
    if any(not bool(np.isfinite(np.asarray(values, dtype=np.float64)).all()) for values in errors.values()):
        raise ValueError("merged controller evidence contains a non-finite error")
    traced = [bool(shard.result.interface_command_traces) for shard in ordered]
    if any(traced) and not all(traced):
        raise ValueError("control shards mix replayable and legacy evidence")
    if all(traced):
        if physical_protocol is None:
            raise ValueError("replayable control shards lack a sealed physical protocol")
        episode_identifiers = tuple(
            identifier
            for shard in ordered
            for identifier in shard.result.episode_identifiers
        )
        if (
            len(episode_identifiers) != REGISTERED_CONTROL_EPISODES
            or len(set(episode_identifiers)) != REGISTERED_CONTROL_EPISODES
        ):
            raise ValueError("merged control episode identifiers are invalid")
        traces = {
            name: tuple(
                trace
                for shard in ordered
                for trace in shard.result.interface_command_traces[name]
            )
            for name in sorted(controller_names)
        }
        schedule_digest = hashlib.sha256()
        for shard in ordered:
            value = shard.result.planner_seed_schedule_sha256
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("control shard planner-seed seal is invalid")
            schedule_digest.update(str((shard.start, shard.stop)).encode("ascii"))
            schedule_digest.update(value.encode("ascii"))
        planner_seed_schedule_sha256: str | None = schedule_digest.hexdigest()
    else:
        episode_identifiers = ()
        traces = {}
        planner_seed_schedule_sha256 = None
    return ControlResult(
        errors=errors,
        interface_name=ordered[0].interface_name,
        episodes=REGISTERED_CONTROL_EPISODES,
        control_steps=ordered[0].result.control_steps,
        planner_budget=dict(ordered[0].result.planner_budget),
        target_source=ordered[0].result.target_source,
        episode_identifiers=episode_identifiers,
        interface_command_traces=traces,
        planner_seed_schedule_sha256=planner_seed_schedule_sha256,
        physical_protocol=physical_protocol,
    )


def physical_gate6_table(
    physical: PhysicalRealizabilityResult,
    *,
    single_horizon_metrics: Mapping[str, RealizabilityMetrics] | None = None,
) -> dict[str, Any]:
    """Build Gate 6, failing closed until single-horizon evidence is supplied."""

    gates: dict[str, Any] = {}
    for interface_name in ("native", "unseen"):
        interface = physical.interfaces.get(interface_name)
        if interface is None:
            gates[interface_name] = {
                "passed": False,
                "auditable": False,
                "failures": ["missing interface realizability evidence"],
            }
            continue
        missing = [
            name for name in ("full", "shuffled_lens") if name not in interface.models
        ]
        if missing:
            gates[interface_name] = {
                "passed": False,
                "auditable": False,
                "failures": [f"missing model realizability evidence: {missing}"],
            }
            continue
        single_horizon = (
            interface.models.get("single_horizon", None)
            if single_horizon_metrics is None
            else single_horizon_metrics.get(interface_name)
        )
        single_horizon_metric = (
            None
            if single_horizon is None
            else (
                single_horizon.metrics
                if isinstance(single_horizon, ModelRealizabilityResult)
                else single_horizon
            )
        )
        gate = realizability_gate_metrics(
            interface.models["full"].metrics,
            single_horizon_mean_cosine=(
                None
                if single_horizon_metric is None
                else single_horizon_metric.mean_cosine
            ),
            shuffled_lens_mean_cosine=interface.models["shuffled_lens"].metrics.mean_cosine,
        )
        gate["checks"]["singleHorizonAblationPresent"] = (
            single_horizon_metric is not None
        )
        gate["passed"] = all(gate["checks"].values())
        gates[interface_name] = gate
    return {
        "passed": all(value.get("passed") is True for value in gates.values()),
        "interfaces": gates,
    }


def _normalized_gate(value: Any, *, missing_reason: str) -> dict[str, Any]:
    if isinstance(value, GateAuditResult):
        return value.to_dict()
    if isinstance(value, Mapping) and "passed" in value:
        return dict(value)
    return {
        "passed": False,
        "auditable": False,
        "checks": {},
        "failures": [missing_reason],
    }


def compose_single_seed_outcome(
    system_gate_tables: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Conjoin Gates 1--8 on both systems without any partial-pass label."""

    required_systems = ("pendulum", "blocket")
    required_gates = tuple(f"gate{index}" for index in range(1, 9))
    normalized_systems: dict[str, Any] = {}
    all_passed = True
    for system in required_systems:
        supplied = system_gate_tables.get(system, {})
        gates = {}
        for gate in required_gates:
            value = _normalized_gate(
                supplied.get(gate),
                missing_reason=f"missing {gate} evidence for {system}",
            )
            gates[gate] = value
            all_passed = all_passed and value.get("passed") is True
        normalized_systems[system] = {
            "passed": all(item.get("passed") is True for item in gates.values()),
            "gates": gates,
        }
    outcome = (
        "direct_jacobian_poisson_ph_breakthrough_supported_single_seed_two_systems"
        if all_passed
        else "direct_jacobian_poisson_ph_breakthrough_not_supported_single_seed"
    )
    return {
        "outcome": outcome,
        "passed": all_passed,
        "systems": normalized_systems,
    }


def run_structural_postfreeze(
    loaded: LoadedPostFreezeSystem,
    *,
    include_gate4: bool = True,
) -> dict[str, Any]:
    """Convenience entry point for Gates 1, 3, and optionally expensive 4."""

    gate1 = audit_gate1_postfreeze(loaded)
    gate3, transitions = audit_gate3_postfreeze(loaded)
    result: dict[str, Any] = {
        "system": loaded.system_name,
        "gate1": gate1.to_dict(),
        "gate3": gate3.to_dict(),
        "gate3TransitionCount": transitions.transition_count,
        "gate3SourceManifestSha256": transitions.source_manifest_sha256,
        "checkpointSha256": {
            "producerManifest": loaded.producer_seal_sha256,
            "backbone": loaded.backbone_checkpoint_sha256,
            "baseline": loaded.baseline_checkpoint_sha256,
            **{
                name: frozen.checkpoint_sha256
                for name, frozen in loaded.variants.items()
            },
        },
    }
    if include_gate4:
        gate4, _ = audit_gate4_postfreeze(loaded)
        result["gate4"] = gate4.to_dict()
    else:
        result["gate4"] = _normalized_gate(
            None, missing_reason="Gate 4 collection was not run"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sanitized_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--system", choices=tuple(DIRECT_SYSTEMS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-gate4", action="store_true")
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--result", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_postfreeze_system(
        args.system,
        PostFreezePaths(args.sanitized_root, args.output_dir),
        torch.device(args.device),
    )
    result = run_structural_postfreeze(loaded, include_gate4=not args.skip_gate4)
    if args.physical:
        physical = run_physical_realizability(loaded)
        result["physicalRealizability"] = physical.to_dict()
        result["gate6"] = physical_gate6_table(physical)
    encoded = json.dumps(result, indent=2)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "ControlShard",
    "FrozenVariantBundle",
    "Gate4CollectionConfig",
    "HeldoutLatentTransitions",
    "InterfaceRealizabilityResult",
    "LoadedPostFreezeSystem",
    "ModelRealizabilityResult",
    "PhysicalRealizabilityResult",
    "PostFreezePaths",
    "REQUIRED_POSTFREEZE_VARIANTS",
    "assemble_gate1_evidence",
    "assemble_gate4_evidence",
    "audit_gate1_postfreeze",
    "audit_gate3_postfreeze",
    "audit_gate4_postfreeze",
    "build_frozen_activation_world_model",
    "collect_test_latent_transitions",
    "compose_single_seed_outcome",
    "load_postfreeze_system",
    "merge_control_shards",
    "physical_gate6_table",
    "registered_control_shard_ranges",
    "run_control_shard",
    "run_physical_realizability",
    "run_structural_postfreeze",
    "validate_direct_checkpoint_metadata",
]
