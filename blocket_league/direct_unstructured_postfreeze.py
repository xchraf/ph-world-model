"""Strict post-freeze reconstruction of the independent baseline artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .direct_cotangent_bridge import PixelChangeProbeBank
from .direct_jacobian_port_extractor import EmpiricalTangentArtifact
from .direct_experiment_training import DirectSystemSpec, DirectTrainingConfig
from .direct_unstructured_training import (
    INDEPENDENT_BASELINE_KIND,
    INDEPENDENT_SUMMARY_KEYS,
    build_fresh_independent_baseline,
    load_independent_evaluation_state,
    validate_independent_checkpoint,
)
from .direct_unstructured_world_model import (
    IndependentUnstructuredBundle,
    freeze_independent_bundle,
    independent_evaluation_modules,
    independent_named_parameters,
)
from .direct_visual_poisson_ph import DirectVideoLossConfig
from .tensor_provenance import module_tensor_hash
from .pixel_direct_model import DirectPixelTransformer


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenIndependentBaseline:
    bundle: IndependentUnstructuredBundle = field(repr=False, compare=False)
    checkpoint_path: Path
    checkpoint_sha256: str
    step: int
    best_validation: float
    best_lens_eligible: bool
    summary: Mapping[str, Any]
    module_hashes: Mapping[str, str]

    @property
    def encoder(self) -> nn.Module:
        return self.bundle.model.encoder

    @property
    def renderer(self) -> nn.Module:
        return self.bundle.model.renderer

    @property
    def dynamics(self) -> nn.Module:
        return self.bundle.model.dynamics

    @property
    def inference(self) -> nn.Module:
        return self.bundle.model.effort_inference

    def assert_frozen_and_unchanged(self) -> None:
        modules = {
            **independent_evaluation_modules(self.bundle),
            "lens": self.bundle.lens,
        }
        for name, module in modules.items():
            if module.training or any(value.requires_grad for value in module.parameters()):
                raise AssertionError(f"independent baseline {name} is not frozen")
            if module_tensor_hash(module) != self.module_hashes[name]:
                raise AssertionError(f"independent baseline {name} changed after freeze")
        self.bundle.model.encoder.assert_backbone_frozen()


def _validate_summary(
    summary: Any,
    checkpoint: Mapping[str, Any],
    bundle: IndependentUnstructuredBundle,
    *,
    system: DirectSystemSpec,
    train_config: DirectTrainingConfig,
    loss_config: DirectVideoLossConfig,
    source_tree_sha256: str,
) -> None:
    if type(summary) is not dict or set(summary) != INDEPENDENT_SUMMARY_KEYS:
        raise ValueError("independent baseline summary schema is not exact")
    numeric = (
        summary.get("seconds"),
        summary.get("bestValidation"),
        summary.get("relativeParameterGap"),
    )
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in numeric):
        raise ValueError("independent baseline summary contains a non-finite scalar")
    if (
        summary["kind"] != INDEPENDENT_BASELINE_KIND
        or summary["system"] != system.name
        or summary["seed"] != train_config.seed
        or summary["referenceInitializationSeed"]
        != bundle.architecture.initialization_seed
        or summary["trainConfig"]
        != json.loads(json.dumps(asdict(train_config), allow_nan=False))
        or summary["lossConfig"]
        != json.loads(json.dumps(asdict(loss_config), allow_nan=False))
        or summary["dynamicsHiddenSize"] != bundle.dynamics_hidden_size
        or summary["targetTrainableParameters"]
        != bundle.target_trainable_parameters
        or summary["trainableParameters"] != bundle.trainable_parameters
        or float(summary["relativeParameterGap"])
        != bundle.relative_parameter_gap
        or summary["homologousInitializationHashes"]
        != dict(bundle.homologous_initialization_hashes)
        or summary["bestStep"] != checkpoint["step"]
        or float(summary["bestValidation"])
        != float(checkpoint["bestValidation"])
        or summary["bestLensEligible"] is not checkpoint["bestLensEligible"]
        or summary["validationMetrics"] != checkpoint["bestMetrics"]
        or summary["backboneHashBefore"]
        != bundle.model.encoder.sealed_backbone_hash
        or summary["backboneHashAfter"]
        != bundle.model.encoder.sealed_backbone_hash
        or summary["actionGradientUpdates"] != 0
        or summary["physicalStateGradientUpdates"] != 0
        or summary["optimizationTensorKeys"] != ["pixelContexts", "frames"]
        or summary["sourceTreeSha256"] != source_tree_sha256
        or type(summary["runtimeTrace"]) is not dict
        or summary["seconds"] < 0.0
    ):
        raise ValueError("independent baseline summary/checkpoint lineage mismatch")


def load_frozen_independent_baseline(
    *,
    backbone: DirectPixelTransformer,
    system: DirectSystemSpec,
    probes: PixelChangeProbeBank,
    empirical_tangent: EmpiricalTangentArtifact,
    train_config: DirectTrainingConfig,
    loss_config: DirectVideoLossConfig,
    checkpoint_path: Path,
    summary: Mapping[str, Any],
    data_seal: Mapping[str, str],
    source_tree_sha256: str,
    reference_initialization_seed: int,
    device: torch.device,
) -> FrozenIndependentBaseline:
    """Rebuild from source/seed, validate, load, and permanently freeze."""

    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise FileNotFoundError("independent baseline checkpoint is missing or symbolic")
    bundle = build_fresh_independent_baseline(
        backbone,
        system,
        probes,
        train_config,
        device,
        empirical_tangent=empirical_tangent,
        reference_initialization_seed=reference_initialization_seed,
    )
    named = independent_named_parameters(bundle)
    optimizer = torch.optim.AdamW(
        [value for _, value in named],
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    validate_independent_checkpoint(
        checkpoint,
        bundle,
        optimizer,
        system=system,
        train_config=train_config,
        loss_config=loss_config,
        data_seal=data_seal,
        source_tree_sha256=source_tree_sha256,
        include_training_state=False,
    )
    _validate_summary(
        summary,
        checkpoint,
        bundle,
        system=system,
        train_config=train_config,
        loss_config=loss_config,
        source_tree_sha256=source_tree_sha256,
    )
    load_independent_evaluation_state(checkpoint, bundle)
    freeze_independent_bundle(bundle)
    frozen = FrozenIndependentBaseline(
        bundle=bundle,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_file_sha256(checkpoint_path),
        step=int(checkpoint["step"]),
        best_validation=float(checkpoint["bestValidation"]),
        best_lens_eligible=bool(checkpoint["bestLensEligible"]),
        summary=dict(summary),
        module_hashes={
            name: module_tensor_hash(module)
            for name, module in {
                **independent_evaluation_modules(bundle),
                "lens": bundle.lens,
            }.items()
        },
    )
    frozen.assert_frozen_and_unchanged()
    if module_tensor_hash(frozen.encoder.backbone) != checkpoint["backboneHash"]:  # type: ignore[attr-defined]
        raise ValueError("independent baseline reconstructed another backbone")
    return frozen


__all__ = ["FrozenIndependentBaseline", "load_frozen_independent_baseline"]
