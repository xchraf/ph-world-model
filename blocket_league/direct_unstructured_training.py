"""Pixels-only joint training for the independent unstructured world model.

The public training API accepts sanitized ``pixelContexts`` and ``frames``
only.  It never accepts latents produced by another model, a structured model,
a physical state, or an action.  A fresh untrained structured architecture is
reconstructed from the registered seed solely to copy homologous *initial*
values and derive the total parameter budget; it is destroyed before the
first optimization batch.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from .direct_cotangent_bridge import PixelChangeProbeBank
from .direct_jacobian_port_extractor import (
    EmpiricalTangentArtifact,
    EmpiricalTangentConfig,
)
from .direct_experiment_training import (
    DirectSystemSpec,
    DirectTrainingConfig,
    _fixed_validation_batch,
    _learning_rate,
    _restore_safe_rng_state,
    _sample_batch,
    _validate_optimizer_state,
    _validate_safe_rng_state,
    build_direct_bundle,
    seed_everything,
)
from .direct_unstructured_world_model import (
    IndependentUnstructuredArchitecture,
    IndependentUnstructuredBundle,
    build_independent_unstructured_bundle,
    capture_homologous_initialization,
    freeze_independent_bundle,
    independent_evaluation_modules,
    independent_named_parameters,
    independent_tangent_lens_terms,
    independent_video_objective,
)
from .direct_visual_poisson_ph import DirectVideoLossConfig
from .tensor_provenance import module_tensor_hash
from .pixel_direct_model import DirectPixelTransformer
from .runtime_firewall_trace import RuntimeFirewallTrace
from .source_provenance import build_source_manifest


INDEPENDENT_BASELINE_KIND = "independent_unstructured_jacobian_lens_world_model"

_DATA_SEAL_KEYS = {
    "system",
    "fitAggregateSha256",
    "fitSanitizedTensorSha256",
    "validationAggregateSha256",
    "validationSanitizedTensorSha256",
}
_MODULE_FIELDS = (
    "encoderPoolScore",
    "encoderReadout",
    "renderer",
    "dynamics",
    "effortInference",
    "writeField",
    "responseFrame",
)
_EVALUATION_KEYS = frozenset(
    {
        "kind",
        "system",
        "actionChannels",
        "physicalStateChannels",
        "optimizationTensorKeys",
        "step",
        "bestValidation",
        "bestLensEligible",
        "bestMetrics",
        *_MODULE_FIELDS,
        "architecture",
        "dynamicsHiddenSize",
        "targetTrainableParameters",
        "trainableParameters",
        "relativeParameterGap",
        "homologousInitializationHashes",
        "optimizedParameterNames",
        "trainConfig",
        "lossConfig",
        "dataSeal",
        "backboneHash",
        "sourceTreeSha256",
    }
)
_VALIDATION_KEYS = frozenset(
    {
        "reconstruction",
        "rolloutPixel",
        "rolloutLatent",
        "innovation",
        "whitening",
        "portFrameTransport",
        "portFrameHolonomy",
        "portRankOrientation",
        "lensBridge",
        "lensOddness",
        "lensManifoldCycle",
        "lensResponseAlignment",
        "lensPersistentResponseFrameAlignment",
        "lensResponseSubspace",
        "lensWriteOddness",
        "lensWriteCurrentFrameLeakage",
        "lensWriteManifoldCycle",
        "lensWriteFirstOrderSignal",
        "lensMinimumFrozenResponseSingularValue",
        "lensMinimumUnstructuredResponseSingularValue",
        "lensExtractedPortMinimumSingularValue",
        "lensExtractedPortMaximumOrthonormalityDefect",
        "lensExtractedPortMinimumProjectedSignalRatio",
        "lensValidationGroups",
        "lensValidationContexts",
        "score",
        "lensEligible",
    }
)
INDEPENDENT_SUMMARY_KEYS = frozenset(
    {
        "kind",
        "system",
        "seed",
        "referenceInitializationSeed",
        "trainConfig",
        "lossConfig",
        "dynamicsHiddenSize",
        "targetTrainableParameters",
        "trainableParameters",
        "relativeParameterGap",
        "homologousInitializationHashes",
        "bestStep",
        "bestValidation",
        "bestLensEligible",
        "validationMetrics",
        "seconds",
        "backboneHashBefore",
        "backboneHashAfter",
        "actionGradientUpdates",
        "physicalStateGradientUpdates",
        "optimizationTensorKeys",
        "sourceTreeSha256",
        "runtimeTrace",
    }
)


def _source_tree_sha256(value: str | None) -> str:
    observed = build_source_manifest()["treeSha256"] if value is None else value
    if (
        type(observed) is not str
        or len(observed) != 64
        or any(character not in "0123456789abcdef" for character in observed)
    ):
        raise ValueError("source tree SHA-256 must be 64 lowercase hex characters")
    return observed


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _rng_payload() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": {
            "algorithm": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "hasGaussian": int(numpy_state[3]),
            "cachedGaussian": float(numpy_state[4]),
        },
        "python": {
            "version": int(python_state[0]),
            "state": torch.tensor(python_state[1], dtype=torch.int64),
            "gaussian": None if python_state[2] is None else float(python_state[2]),
        },
    }


def _copy_probe_bank(probes: PixelChangeProbeBank) -> PixelChangeProbeBank:
    return PixelChangeProbeBank(probes.basis.detach().cpu().clone())


def _reference_parameter_target(reference: Any) -> int:
    backbone_ids = {
        id(value) for value in reference.model.encoder.backbone.parameters()
    }
    return sum(
        value.numel()
        for module in (
            reference.model,
            reference.write_field,
            reference.response_frame,
            reference.cotangent_frame,
        )
        for value in module.parameters()
        if value.requires_grad and id(value) not in backbone_ids
    )


def build_fresh_independent_baseline(
    backbone: DirectPixelTransformer,
    system: DirectSystemSpec,
    probes: PixelChangeProbeBank,
    config: DirectTrainingConfig,
    device: torch.device,
    *,
    empirical_tangent: EmpiricalTangentArtifact,
    reference_initialization_seed: int,
) -> IndependentUnstructuredBundle:
    """Reconstruct a fresh reference initialization, copy it, then destroy it.

    There is intentionally no argument for a structured model or checkpoint.
    Consequently neither a trained encoder nor its latent outputs can enter
    construction of the independent baseline.
    """

    if type(reference_initialization_seed) is not int:
        raise ValueError("reference initialization seed must be an integer")
    seed_everything(reference_initialization_seed)
    reference = build_direct_bundle(
        backbone,
        system,
        _copy_probe_bank(probes),
        config,
        device,
        empirical_tangent=empirical_tangent,
        variant="full",
    )
    initialization = capture_homologous_initialization(
        encoder=reference.model.encoder,
        renderer=reference.model.renderer,
        effort_inference=reference.model.effort_inference,
        write_field=reference.write_field,
        response_frame=reference.response_frame,
        reference_initialization_seed=reference_initialization_seed,
    )
    target = _reference_parameter_target(reference)
    lens_block = min(system.lens_block, len(backbone.blocks) - 1)
    architecture = IndependentUnstructuredArchitecture(
        state_size=system.state_size,
        port_size=system.port_size,
        dt=system.dt,
        lens_block=lens_block,
        state_hidden_size=config.state_hidden_size,
        renderer_hidden_size=config.renderer_hidden_size,
        renderer_depth=config.renderer_depth,
        renderer_heads=config.renderer_heads,
        dynamics_hidden_layers=config.ph_hidden_layers,
        write_hidden_size=config.write_hidden_size,
        write_hidden_layers=config.write_hidden_layers,
        lens_horizons=config.lens_horizons,
        initialization_seed=reference_initialization_seed,
    )
    independent = build_independent_unstructured_bundle(
        backbone,
        architecture,
        empirical_tangent=empirical_tangent,
        probes=_copy_probe_bank(probes),
        tangent_config=EmpiricalTangentConfig(
            channel_rank=config.port_tangent_channel_rank,
            neighbors=config.port_tangent_neighbors,
            support_floor_ratio=config.port_support_floor_ratio,
        ),
        target_trainable_parameters=target,
        homologous_initialization=initialization,
        device=device,
    )
    reference_trainable_ids = {
        id(value)
        for module in (
            reference.model,
            reference.write_field,
            reference.response_frame,
            reference.cotangent_frame,
        )
        for value in module.parameters()
        if value.requires_grad
    }
    independent_ids = {id(value) for _, value in independent_named_parameters(independent)}
    if not reference_trainable_ids.isdisjoint(independent_ids):
        raise AssertionError("fresh reference and independent baseline share trainables")
    # The reference is never returned and is not used by the training API.
    del reference
    return independent


def _validate_pixel_inputs(
    fit_suite: Mapping[str, torch.Tensor],
    validation_suite: Mapping[str, torch.Tensor],
    data_seal: Mapping[str, str],
    system: DirectSystemSpec,
) -> None:
    if type(fit_suite) is not dict or set(fit_suite) != {"pixelContexts", "frames"}:
        raise ValueError("independent baseline fit tensors are not exactly pixels-only")
    if type(validation_suite) is not dict or set(validation_suite) != {
        "pixelContexts",
        "frames",
    }:
        raise ValueError(
            "independent baseline validation tensors are not exactly pixels-only"
        )
    for label, suite in (("fit", fit_suite), ("validation", validation_suite)):
        contexts = suite["pixelContexts"]
        frames = suite["frames"]
        if (
            type(contexts) is not torch.Tensor
            or type(frames) is not torch.Tensor
            or contexts.ndim != 5
            or frames.ndim != 4
            or contexts.shape[:2] != frames.shape[:2]
            or contexts.requires_grad
            or frames.requires_grad
        ):
            raise ValueError(f"independent baseline {label} pixel tensors are invalid")
    if (
        type(data_seal) is not dict
        or set(data_seal) != _DATA_SEAL_KEYS
        or data_seal.get("system") != system.name
        or any(type(value) is not str for value in data_seal.values())
    ):
        raise ValueError("independent baseline data seal schema is not exact")


def _validation_score(
    bundle: IndependentUnstructuredBundle,
    suite: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    loss_config: DirectVideoLossConfig,
    train_config: DirectTrainingConfig,
    device: torch.device,
) -> dict[str, float | bool]:
    bundle.model.eval()
    bundle.write_field.eval()
    bundle.response_frame.eval()
    base_names = (
        "reconstruction",
        "rolloutPixel",
        "rolloutLatent",
        "innovation",
        "whitening",
        "portFrameTransport",
        "portFrameHolonomy",
        "portRankOrientation",
    )
    totals = {name: 0.0 for name in base_names}
    with torch.no_grad():
        for index in range(train_config.validation_batches):
            contexts, frames = _fixed_validation_batch(
                suite, train_config.micro_batch_size, index, device
            )
            _, metrics = independent_video_objective(
                bundle,
                contexts,
                frames,
                class_weights,
                loss_config,
                lens_terms=None,
                require_lens_terms=False,
            )
            for name in base_names:
                totals[name] += float(metrics[name])
    for name in base_names:
        totals[name] /= train_config.validation_batches

    group_count = train_config.validation_batches
    group_size = train_config.lens_batch_size
    required = group_count * group_size
    if int(suite["frames"].shape[0]) < required:
        raise ValueError(
            f"independent lens validation needs {required} distinct trajectories"
        )
    grouped: dict[str, list[float]] = {}
    for index in range(group_count):
        contexts, _ = _fixed_validation_batch(suite, group_size, index, device)
        with torch.enable_grad():
            terms, metrics = independent_tangent_lens_terms(
                bundle,
                contexts[:, 0],
                horizons=train_config.lens_horizons,
            )
        values = {
            "lensBridge": float(terms["bridge"].detach()),
            "lensOddness": float(terms["oddness"].detach()),
            "lensManifoldCycle": float(terms["manifoldCycle"].detach()),
            **{
                f"lens{name[0].upper()}{name[1:]}": float(value.detach())
                for name, value in metrics.items()
            },
        }
        if grouped and set(values) != set(grouped):
            raise AssertionError("independent lens validation schema changed")
        for name, value in values.items():
            grouped.setdefault(name, []).append(value)
    minima = {
        "lensWriteFirstOrderSignal",
        "lensMinimumFrozenResponseSingularValue",
        "lensMinimumUnstructuredResponseSingularValue",
        "lensExtractedPortMinimumSingularValue",
        "lensExtractedPortMinimumProjectedSignalRatio",
    }
    maxima = {"lensExtractedPortMaximumOrthonormalityDefect"}
    for name, values in grouped.items():
        totals[name] = (
            min(values)
            if name in minima
            else max(values)
            if name in maxima
            else sum(values) / group_count
        )
    totals["lensValidationGroups"] = float(group_count)
    totals["lensValidationContexts"] = float(required)
    totals["score"] = (
        loss_config.reconstruction_weight * totals["reconstruction"]
        + loss_config.rollout_pixel_weight * totals["rolloutPixel"]
        + loss_config.rollout_latent_weight * totals["rolloutLatent"]
        + loss_config.innovation_weight * totals["innovation"]
        + loss_config.whitening_weight * totals["whitening"]
        + loss_config.port_frame_weight
        * (totals["portFrameTransport"] + totals["portRankOrientation"])
        + loss_config.port_holonomy_weight * totals["portFrameHolonomy"]
        + loss_config.jacobian_bridge_weight * totals["lensBridge"]
        + loss_config.oddness_weight * totals["lensOddness"]
        + loss_config.manifold_cycle_weight * totals["lensManifoldCycle"]
    )
    totals["lensEligible"] = bool(
        all(math.isfinite(float(value)) for value in totals.values())
        and totals["lensWriteFirstOrderSignal"] >= 1e-7
        and totals["lensMinimumFrozenResponseSingularValue"] >= 1e-6
        and totals["lensMinimumUnstructuredResponseSingularValue"] >= 1e-6
        and totals["lensExtractedPortMinimumSingularValue"] >= 1e-8
        and totals["lensExtractedPortMaximumOrthonormalityDefect"] <= 1e-4
        and totals["lensExtractedPortMinimumProjectedSignalRatio"] >= 1e-6
    )
    if set(totals) != _VALIDATION_KEYS:
        raise AssertionError("independent validation metrics are not exact")
    bundle.model.train()
    bundle.write_field.train()
    bundle.response_frame.train()
    bundle.model.encoder.backbone.eval()
    return totals


def independent_checkpoint_payload(
    bundle: IndependentUnstructuredBundle,
    optimizer: torch.optim.Optimizer,
    *,
    system: DirectSystemSpec,
    train_config: DirectTrainingConfig,
    loss_config: DirectVideoLossConfig,
    step: int,
    best_validation: float,
    best_lens_eligible: bool,
    best_metrics: Mapping[str, float | bool],
    data_seal: Mapping[str, str],
    source_tree_sha256: str,
    include_training_state: bool,
) -> dict[str, Any]:
    modules = independent_evaluation_modules(bundle)
    payload: dict[str, Any] = {
        "kind": INDEPENDENT_BASELINE_KIND,
        "system": asdict(system),
        "actionChannels": 0,
        "physicalStateChannels": 0,
        "optimizationTensorKeys": ["pixelContexts", "frames"],
        "step": step,
        "bestValidation": best_validation,
        "bestLensEligible": best_lens_eligible,
        "bestMetrics": dict(best_metrics),
        **{
            name: {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            }
            for name, module in modules.items()
        },
        "architecture": asdict(bundle.architecture),
        "dynamicsHiddenSize": bundle.dynamics_hidden_size,
        "targetTrainableParameters": bundle.target_trainable_parameters,
        "trainableParameters": bundle.trainable_parameters,
        "relativeParameterGap": bundle.relative_parameter_gap,
        "homologousInitializationHashes": dict(
            bundle.homologous_initialization_hashes
        ),
        "optimizedParameterNames": [name for name, _ in independent_named_parameters(bundle)],
        "trainConfig": asdict(train_config),
        "lossConfig": asdict(loss_config),
        "dataSeal": dict(data_seal),
        "backboneHash": bundle.model.encoder.sealed_backbone_hash,
        "sourceTreeSha256": _source_tree_sha256(source_tree_sha256),
    }
    if include_training_state:
        payload["optimizer"] = optimizer.state_dict()
        payload["rngState"] = _rng_payload()
    return payload


def _validate_state(field: str, value: Any, module: torch.nn.Module) -> None:
    reference = module.state_dict()
    if type(value) is not dict or set(value) != set(reference):
        raise ValueError(f"independent checkpoint {field} state schema mismatch")
    if any("backbone" in name.lower() for name in value):
        raise ValueError("independent checkpoint attempted to embed a backbone tensor")
    for name, tensor in value.items():
        expected = reference[name]
        if (
            type(tensor) is not torch.Tensor
            or tensor.shape != expected.shape
            or tensor.dtype != expected.dtype
            or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
        ):
            raise ValueError(f"independent checkpoint {field}.{name} is invalid")


def validate_independent_checkpoint(
    payload: Any,
    bundle: IndependentUnstructuredBundle,
    optimizer: torch.optim.Optimizer,
    *,
    system: DirectSystemSpec,
    train_config: DirectTrainingConfig,
    loss_config: DirectVideoLossConfig,
    data_seal: Mapping[str, str],
    source_tree_sha256: str,
    include_training_state: bool,
) -> None:
    expected = set(_EVALUATION_KEYS)
    if include_training_state:
        expected.update(("optimizer", "rngState"))
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("independent checkpoint top-level schema is not exact")
    forbidden_names = {
        "encodedPixelStates",
        "fullModelHash",
        "structuredModelHash",
        "model",
        "backbone",
    }
    if forbidden_names.intersection(payload):
        raise ValueError("independent checkpoint contains a forbidden dependency")
    if (
        payload["kind"] != INDEPENDENT_BASELINE_KIND
        or payload["system"] != asdict(system)
        or payload["actionChannels"] != 0
        or payload["physicalStateChannels"] != 0
        or payload["optimizationTensorKeys"] != ["pixelContexts", "frames"]
        or payload["trainConfig"] != asdict(train_config)
        or payload["lossConfig"] != asdict(loss_config)
        or payload["dataSeal"] != dict(data_seal)
        or payload["backboneHash"] != bundle.model.encoder.sealed_backbone_hash
        or payload["sourceTreeSha256"] != _source_tree_sha256(source_tree_sha256)
    ):
        raise ValueError("independent checkpoint provenance mismatch")
    if payload["architecture"] != asdict(bundle.architecture):
        raise ValueError("independent checkpoint architecture mismatch")
    if (
        type(payload["step"]) is not int
        or not 1 <= payload["step"] <= train_config.steps
        or type(payload["bestValidation"]) not in (int, float)
        or not math.isfinite(float(payload["bestValidation"]))
        or type(payload["bestLensEligible"]) is not bool
        or type(payload["bestMetrics"]) is not dict
        or set(payload["bestMetrics"]) != _VALIDATION_KEYS
        or payload["bestMetrics"].get("lensEligible") is not payload["bestLensEligible"]
        or float(payload["bestMetrics"].get("score", math.inf))
        != float(payload["bestValidation"])
    ):
        raise ValueError("independent checkpoint selection evidence is invalid")
    numeric_metrics = {
        name: value
        for name, value in payload["bestMetrics"].items()
        if name != "lensEligible"
    }
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in numeric_metrics.values()):
        raise ValueError("independent checkpoint metrics are non-finite")
    if (
        payload["dynamicsHiddenSize"] != bundle.dynamics_hidden_size
        or payload["targetTrainableParameters"] != bundle.target_trainable_parameters
        or payload["trainableParameters"] != bundle.trainable_parameters
        or type(payload["relativeParameterGap"]) not in (int, float)
        or float(payload["relativeParameterGap"]) != bundle.relative_parameter_gap
        or bundle.relative_parameter_gap > 0.01
        or payload["homologousInitializationHashes"]
        != dict(bundle.homologous_initialization_hashes)
        or payload["optimizedParameterNames"]
        != [name for name, _ in independent_named_parameters(bundle)]
    ):
        raise ValueError("independent checkpoint capacity/initialization seal mismatch")
    modules = independent_evaluation_modules(bundle)
    for field, module in modules.items():
        _validate_state(field, payload[field], module)
    if include_training_state:
        parameters = [value for _, value in independent_named_parameters(bundle)]
        _validate_optimizer_state(
            payload["optimizer"], optimizer, parameters, checkpoint_step=payload["step"]
        )
        _validate_safe_rng_state(payload["rngState"])


def load_independent_evaluation_state(
    payload: Mapping[str, Any], bundle: IndependentUnstructuredBundle
) -> None:
    for field, module in independent_evaluation_modules(bundle).items():
        module.load_state_dict(payload[field], strict=True)


def train_independent_unstructured_world_model(
    bundle: IndependentUnstructuredBundle,
    fit_suite: dict[str, torch.Tensor],
    validation_suite: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    system: DirectSystemSpec,
    output_dir: Path,
    train_config: DirectTrainingConfig,
    loss_config: DirectVideoLossConfig,
    *,
    data_seal: Mapping[str, str],
    pixel_archive_paths: Mapping[str, Path],
    source_tree_sha256: str | None = None,
    runtime_trace: RuntimeFirewallTrace | None = None,
) -> dict[str, Any]:
    """Jointly optimize the independent E, decoder, inverse, F/B, and lens U."""

    _validate_pixel_inputs(fit_suite, validation_suite, data_seal, system)
    if train_config.lens_every != 1:
        raise ValueError("independent tangent Jacobian lens must run every step")
    if type(pixel_archive_paths) is not dict or set(pixel_archive_paths) != {
        "fit",
        "validation",
    }:
        raise ValueError("independent baseline requires exact fit/validation archives")
    resolved_source = _source_tree_sha256(source_tree_sha256)
    output_dir.mkdir(parents=True, exist_ok=True)
    owns_trace = runtime_trace is None
    if runtime_trace is None:
        runtime_trace = RuntimeFirewallTrace(
            output_dir / "firewall-trace.jsonl",
            stage=f"baseline-independent:{system.name}",
            source_tree_sha256=resolved_source,
        )
    parents = {path.parent.resolve(strict=True) for path in pixel_archive_paths.values()}
    if len(parents) != 1:
        raise ValueError("fit and validation archives must share one trainer mount")
    runtime_trace.record_mount_manifest(
        next(iter(parents)), role="independent_pixels_only_trainer_mount"
    )
    for split in ("fit", "validation"):
        runtime_trace.record_file_read(
            pixel_archive_paths[split],
            role=f"trainer_archive:{split}",
            serialized_keys=("pixels", "manifest"),
            semantic_sha256=data_seal[f"{split}SanitizedTensorSha256"],
        )
    seed_everything(train_config.seed)
    named_parameters = independent_named_parameters(bundle)
    parameters = [value for _, value in named_parameters]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    runtime_trace.record_optimizer(
        phase="baseline:independent_unstructured",
        named_parameters=dict(named_parameters),
        protected_parameters={
            f"encoder.backbone.{name}": parameter
            for name, parameter in bundle.model.encoder.backbone.named_parameters()
        },
    )
    device = next(bundle.model.renderer.parameters()).device
    initial_backbone_hash = bundle.model.encoder.sealed_backbone_hash
    runtime_trace.record_backbone_boundary(
        phase="baseline:independent_unstructured",
        boundary="start",
        sha256=initial_backbone_hash,
    )
    start_step = 1
    best_validation = math.inf
    best_eligible = False
    best_metrics: dict[str, float | bool] | None = None
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    if best_path.exists() != last_path.exists():
        raise ValueError("independent resume requires paired best.pt and last.pt")
    if last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=True)
        runtime_trace.record_file_read(
            last_path,
            role="independent_resume_checkpoint:last",
            serialized_keys=tuple(sorted(last)) if type(last) is dict else (),
        )
        validate_independent_checkpoint(
            last,
            bundle,
            optimizer,
            system=system,
            train_config=train_config,
            loss_config=loss_config,
            data_seal=data_seal,
            source_tree_sha256=resolved_source,
            include_training_state=True,
        )
        selected = torch.load(best_path, map_location=device, weights_only=True)
        runtime_trace.record_file_read(
            best_path,
            role="independent_resume_checkpoint:best",
            serialized_keys=tuple(sorted(selected)) if type(selected) is dict else (),
        )
        validate_independent_checkpoint(
            selected,
            bundle,
            optimizer,
            system=system,
            train_config=train_config,
            loss_config=loss_config,
            data_seal=data_seal,
            source_tree_sha256=resolved_source,
            include_training_state=False,
        )
        if (
            selected["step"] > last["step"]
            or selected["bestValidation"] != last["bestValidation"]
            or selected["bestMetrics"] != last["bestMetrics"]
        ):
            raise ValueError("independent best/last lineage is inconsistent")
        load_independent_evaluation_state(last, bundle)
        optimizer.load_state_dict(last["optimizer"])
        _restore_safe_rng_state(last["rngState"])
        start_step = int(last["step"]) + 1
        best_validation = float(last["bestValidation"])
        best_eligible = bool(last["bestLensEligible"])
        best_metrics = dict(last["bestMetrics"])

    started = time.perf_counter()
    log_path = output_dir / "train.jsonl"
    with log_path.open("a" if start_step > 1 else "w", encoding="utf-8") as log_file:
        for step in range(start_step, train_config.steps + 1):
            bundle.model.train()
            bundle.write_field.train()
            bundle.response_frame.train()
            bundle.model.encoder.backbone.eval()
            learning_rate = _learning_rate(step, train_config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            aggregate: dict[str, float] = {}
            counts: dict[str, int] = {}
            for accumulation in range(train_config.gradient_accumulation):
                contexts, frames = _sample_batch(
                    fit_suite, train_config.micro_batch_size, device
                )
                runtime_trace.record_gradient_batch(
                    phase="baseline:independent_unstructured",
                    step=step,
                    tensors={"pixelContexts": contexts, "frames": frames},
                )
                states = bundle.model.encode(contexts)
                lens_terms = None
                lens_metrics: dict[str, torch.Tensor] = {}
                if accumulation == 0:
                    lens_terms, lens_metrics = independent_tangent_lens_terms(
                        bundle,
                        contexts[: train_config.lens_batch_size, 0],
                        horizons=train_config.lens_horizons,
                        encoded_states=states[: train_config.lens_batch_size, 0],
                    )
                    if train_config.gradient_accumulation > 1:
                        lens_terms = {
                            name: value * train_config.gradient_accumulation
                            for name, value in lens_terms.items()
                        }
                    require_lens = True
                    micro_config = loss_config
                else:
                    require_lens = False
                    micro_config = DirectVideoLossConfig(
                        **{
                            **asdict(loss_config),
                            "jacobian_bridge_weight": 0.0,
                            "oddness_weight": 0.0,
                            "manifold_cycle_weight": 0.0,
                        }
                    )
                loss, metrics = independent_video_objective(
                    bundle,
                    contexts,
                    frames,
                    class_weights,
                    micro_config,
                    lens_terms=lens_terms,
                    require_lens_terms=require_lens,
                    encoded_states=states,
                )
                (loss / train_config.gradient_accumulation).backward()
                for name, value in {**metrics, **lens_metrics}.items():
                    aggregate[name] = aggregate.get(name, 0.0) + float(value.detach())
                    counts[name] = counts.get(name, 0) + 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, train_config.gradient_clip
            )
            optimizer.step()
            bundle.model.encoder.assert_backbone_frozen()
            if module_tensor_hash(bundle.model.encoder.backbone) != initial_backbone_hash:
                raise AssertionError("backbone changed during independent training")

            if step == 1 or step % train_config.log_every == 0 or step == train_config.steps:
                elapsed = time.perf_counter() - started
                record = {
                    "stage": "independent_unstructured_jacobian_lens_world_model",
                    "system": system.name,
                    "step": step,
                    "steps": train_config.steps,
                    "learningRate": learning_rate,
                    "gradientNorm": float(gradient_norm),
                    "seconds": elapsed,
                    "estimatedSeconds": elapsed
                    / max(step - start_step + 1, 1)
                    * (train_config.steps - start_step + 1),
                    **{
                        name: value / counts[name]
                        for name, value in aggregate.items()
                    },
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
            if step % train_config.validation_every == 0 or step == train_config.steps:
                validation = _validation_score(
                    bundle,
                    validation_suite,
                    class_weights,
                    loss_config,
                    train_config,
                    device,
                )
                score = float(validation["score"])
                eligible = bool(validation["lensEligible"])
                record = {
                    "stage": "independent_pixels_only_validation",
                    "system": system.name,
                    "step": step,
                    **validation,
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if (not eligible, score) < (not best_eligible, best_validation):
                    best_validation = score
                    best_eligible = eligible
                    best_metrics = dict(validation)
                    _atomic_torch_save(
                        independent_checkpoint_payload(
                            bundle,
                            optimizer,
                            system=system,
                            train_config=train_config,
                            loss_config=loss_config,
                            step=step,
                            best_validation=best_validation,
                            best_lens_eligible=best_eligible,
                            best_metrics=best_metrics,
                            data_seal=data_seal,
                            source_tree_sha256=resolved_source,
                            include_training_state=False,
                        ),
                        best_path,
                    )
            if step % train_config.checkpoint_every == 0 or step == train_config.steps:
                if best_metrics is None:
                    raise ValueError("independent checkpoint has no validation selection")
                _atomic_torch_save(
                    independent_checkpoint_payload(
                        bundle,
                        optimizer,
                        system=system,
                        train_config=train_config,
                        loss_config=loss_config,
                        step=step,
                        best_validation=best_validation,
                        best_lens_eligible=best_eligible,
                        best_metrics=best_metrics,
                        data_seal=data_seal,
                        source_tree_sha256=resolved_source,
                        include_training_state=True,
                    ),
                    last_path,
                )

    if not best_path.is_file():
        raise ValueError("independent training has no validation-selected checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=True)
    runtime_trace.record_file_read(
        best_path,
        role="independent_selected_checkpoint",
        serialized_keys=tuple(sorted(best)) if type(best) is dict else (),
    )
    validate_independent_checkpoint(
        best,
        bundle,
        optimizer,
        system=system,
        train_config=train_config,
        loss_config=loss_config,
        data_seal=data_seal,
        source_tree_sha256=resolved_source,
        include_training_state=False,
    )
    load_independent_evaluation_state(best, bundle)
    freeze_independent_bundle(bundle)
    runtime_trace.record_backbone_boundary(
        phase="baseline:independent_unstructured",
        boundary="selected_checkpoint",
        sha256=module_tensor_hash(bundle.model.encoder.backbone),
    )
    trace_seal = runtime_trace.snapshot().to_dict()
    if owns_trace:
        runtime_trace.close()
    summary = {
        "kind": INDEPENDENT_BASELINE_KIND,
        "system": system.name,
        "seed": train_config.seed,
        "referenceInitializationSeed": bundle.architecture.initialization_seed,
        "trainConfig": asdict(train_config),
        "lossConfig": asdict(loss_config),
        "dynamicsHiddenSize": bundle.dynamics_hidden_size,
        "targetTrainableParameters": bundle.target_trainable_parameters,
        "trainableParameters": bundle.trainable_parameters,
        "relativeParameterGap": bundle.relative_parameter_gap,
        "homologousInitializationHashes": dict(
            bundle.homologous_initialization_hashes
        ),
        "bestStep": int(best["step"]),
        "bestValidation": float(best["bestValidation"]),
        "bestLensEligible": bool(best["bestLensEligible"]),
        "validationMetrics": dict(best["bestMetrics"]),
        "seconds": time.perf_counter() - started,
        "backboneHashBefore": initial_backbone_hash,
        "backboneHashAfter": module_tensor_hash(bundle.model.encoder.backbone),
        "actionGradientUpdates": 0,
        "physicalStateGradientUpdates": 0,
        "optimizationTensorKeys": ["pixelContexts", "frames"],
        "sourceTreeSha256": resolved_source,
        "runtimeTrace": trace_seal,
    }
    _atomic_json(summary, output_dir / "summary.json")
    return summary


__all__ = [
    "INDEPENDENT_BASELINE_KIND",
    "INDEPENDENT_SUMMARY_KEYS",
    "build_fresh_independent_baseline",
    "independent_checkpoint_payload",
    "load_independent_evaluation_state",
    "train_independent_unstructured_world_model",
    "validate_independent_checkpoint",
]
