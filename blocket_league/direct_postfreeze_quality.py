"""Fail-closed held-out visual quality audit for Experiment F Gate 2.

The collector consumes only the sanitized test pixels and already-frozen
modules.  It reconstructs the current image, rolls both latent dynamics for
eight steps using their own *unlabelled* inverse heads, and repeats the direct
rollout after a deterministic cross-trajectory permutation of the inferred
innovations.  No simulator object or physical action is accepted by this API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .direct_action_free_data import (
    make_optimization_suite,
    sanitized_pixel_tensor_sha256,
    weighted_pixel_cross_entropy,
)
from .direct_ph_structure_audits import GateAuditResult
from .direct_physical_evaluation import FrozenEvaluationSeal
from .pixel_direct_model import PixelDirectConfig


REGISTERED_GATE2_SAMPLES = 512
REGISTERED_GATE2_HORIZON = 8


@dataclass(frozen=True)
class Gate2Thresholds:
    pendulum_foreground_iou: float = 0.80
    blocket_disc_iou: float = 0.70
    maximum_full_to_unstructured_error_ratio: float = 1.10
    minimum_shuffled_innovation_degradation: float = 0.10
    minimum_samples: int = REGISTERED_GATE2_SAMPLES
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.minimum_samples != REGISTERED_GATE2_SAMPLES:
            raise ValueError("Gate 2 requires exactly the registered 512 test trajectories")
        probabilities = (
            self.pendulum_foreground_iou,
            self.blocket_disc_iou,
            self.minimum_shuffled_innovation_degradation,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Gate 2 probability/fraction thresholds are invalid")
        if (
            not math.isfinite(self.maximum_full_to_unstructured_error_ratio)
            or self.maximum_full_to_unstructured_error_ratio < 1.0
            or not math.isfinite(self.epsilon)
            or self.epsilon <= 0.0
        ):
            raise ValueError("Gate 2 ratio/epsilon thresholds are invalid")


@dataclass(frozen=True)
class Gate2Evidence:
    system_name: str
    sample_count: int
    horizon: int
    current_foreground_iou: Mapping[str, float]
    full_horizon_centroid_error: Mapping[str, float]
    unstructured_horizon_centroid_error: Mapping[str, float]
    full_horizon_weighted_cross_entropy: float
    unstructured_horizon_weighted_cross_entropy: float
    shuffled_horizon_weighted_cross_entropy: float
    test_sanitized_tensor_sha256: str
    class_weights_sha256: str
    neural_hashes_before: Mapping[str, str]
    neural_hashes_after: Mapping[str, str]
    physical_action_reads: int = 0
    physical_state_reads: int = 0
    target_source: str = "sanitized_categorical_pixels_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system_name,
            "sampleCount": self.sample_count,
            "horizon": self.horizon,
            "currentForegroundIoU": dict(self.current_foreground_iou),
            "fullHorizonCentroidError": dict(self.full_horizon_centroid_error),
            "unstructuredHorizonCentroidError": dict(
                self.unstructured_horizon_centroid_error
            ),
            "fullHorizonWeightedCrossEntropy": self.full_horizon_weighted_cross_entropy,
            "unstructuredHorizonWeightedCrossEntropy": self.unstructured_horizon_weighted_cross_entropy,
            "shuffledHorizonWeightedCrossEntropy": self.shuffled_horizon_weighted_cross_entropy,
            "testSanitizedTensorSha256": self.test_sanitized_tensor_sha256,
            "classWeightsSha256": self.class_weights_sha256,
            "neuralHashesBefore": dict(self.neural_hashes_before),
            "neuralHashesAfter": dict(self.neural_hashes_after),
            "physicalActionReads": self.physical_action_reads,
            "physicalStateReads": self.physical_state_reads,
            "targetSource": self.target_source,
        }


def _gate_result(
    checks: Mapping[str, bool],
    metrics: Mapping[str, float | int | str],
    *,
    unauditable: Sequence[str] = (),
) -> GateAuditResult:
    reasons = tuple(str(value) for value in unauditable)
    failed = tuple(f"failed check: {name}" for name, passed in checks.items() if not passed)
    return GateAuditResult(
        gate=2,
        auditable=not reasons,
        passed=not reasons and not failed,
        checks=dict(checks),
        metrics=dict(metrics),
        failures=reasons + failed,
    )


def _object_groups(system_name: str) -> Mapping[str, tuple[int, ...]]:
    if system_name == "pendulum":
        return {"pendulumBob": (7, 8)}
    if system_name == "blocket":
        return {"playerDisc": (5, 6), "puckDisc": (7, 8)}
    raise ValueError(f"unknown Gate 2 system {system_name!r}")


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gate2_tensor_sha256(tensor: torch.Tensor) -> str:
    """Return the canonical tensor digest used in sealed Gate 2 evidence."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Gate 2 tensor evidence must be a torch.Tensor")
    return _tensor_sha256(tensor)


def audit_gate_2(
    evidence: Gate2Evidence,
    thresholds: Gate2Thresholds = Gate2Thresholds(),
    *,
    expected_test_sanitized_tensor_sha256: str,
    expected_class_weights_sha256: str,
    expected_neural_hashes: Mapping[str, str],
) -> GateAuditResult:
    reasons: list[str] = []
    if type(evidence.system_name) is not str or evidence.system_name not in {
        "pendulum",
        "blocket",
    }:
        reasons.append("Gate 2 evidence has an unknown system")
        expected_objects: set[str] = set()
    else:
        expected_objects = set(_object_groups(evidence.system_name))
    if (
        type(evidence.current_foreground_iou) is not dict
        or type(evidence.full_horizon_centroid_error) is not dict
        or type(evidence.unstructured_horizon_centroid_error) is not dict
    ):
        reasons.append("Gate 2 object metrics must be plain dictionaries")
        return _gate_result(
            {}, {"sample_count": evidence.sample_count}, unauditable=reasons
        )
    if type(evidence.sample_count) is not int or evidence.sample_count != thresholds.minimum_samples:
        reasons.append("Gate 2 evidence does not contain exactly 512 test trajectories")
    if type(evidence.horizon) is not int or evidence.horizon != REGISTERED_GATE2_HORIZON:
        reasons.append("Gate 2 evidence is not the registered horizon 8")
    if set(evidence.current_foreground_iou) != expected_objects:
        reasons.append("Gate 2 foreground-object schema is not exact")
    if (
        set(evidence.full_horizon_centroid_error) != expected_objects
        or set(evidence.unstructured_horizon_centroid_error) != expected_objects
    ):
        reasons.append("Gate 2 horizon-8 centroid-object schema is not exact")
    if evidence.target_source != "sanitized_categorical_pixels_only":
        reasons.append("Gate 2 targets are not sealed categorical pixels")
    if (
        type(evidence.physical_action_reads) is not int
        or type(evidence.physical_state_reads) is not int
        or evidence.physical_action_reads != 0
        or evidence.physical_state_reads != 0
    ):
        reasons.append("Gate 2 read a forbidden physical action or state")
    digest = evidence.test_sanitized_tensor_sha256
    if not _valid_sha256(digest) or not _valid_sha256(
        expected_test_sanitized_tensor_sha256
    ):
        reasons.append("Gate 2 test tensor SHA-256 is malformed")
    elif digest != expected_test_sanitized_tensor_sha256:
        reasons.append("Gate 2 test tensor SHA-256 differs from the sealed manifest")
    if not _valid_sha256(evidence.class_weights_sha256) or not _valid_sha256(
        expected_class_weights_sha256
    ):
        reasons.append("Gate 2 class-weight SHA-256 is malformed")
    elif evidence.class_weights_sha256 != expected_class_weights_sha256:
        reasons.append("Gate 2 class weights differ from the sealed fit-derived weights")
    if (
        type(expected_neural_hashes) is not dict
        or not expected_neural_hashes
        or any(
            type(name) is not str or not _valid_sha256(value)
            for name, value in expected_neural_hashes.items()
        )
    ):
        reasons.append("Gate 2 expected neural-hash seal is malformed")
    if (
        type(evidence.neural_hashes_before) is not dict
        or type(evidence.neural_hashes_after) is not dict
        or not evidence.neural_hashes_before
        or dict(evidence.neural_hashes_before) != dict(evidence.neural_hashes_after)
        or any(
            type(name) is not str
            or type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for name, value in evidence.neural_hashes_before.items()
        )
    ):
        reasons.append("Gate 2 neural hashes are missing or changed")
    elif dict(evidence.neural_hashes_before) != dict(expected_neural_hashes):
        reasons.append("Gate 2 neural hashes differ from the frozen checkpoint seal")
    numeric = (
        *evidence.current_foreground_iou.values(),
        *evidence.full_horizon_centroid_error.values(),
        *evidence.unstructured_horizon_centroid_error.values(),
        evidence.full_horizon_weighted_cross_entropy,
        evidence.unstructured_horizon_weighted_cross_entropy,
        evidence.shuffled_horizon_weighted_cross_entropy,
    )
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in numeric):
        reasons.append("Gate 2 evidence contains a non-finite metric")
    if any(float(value) < 0.0 for value in numeric):
        reasons.append("Gate 2 evidence contains a negative error/IoU")
    if any(not 0.0 <= float(value) <= 1.0 for value in evidence.current_foreground_iou.values()):
        reasons.append("Gate 2 IoU lies outside [0,1]")
    if reasons:
        return _gate_result({}, {"sample_count": evidence.sample_count}, unauditable=reasons)

    iou_threshold = (
        thresholds.pendulum_foreground_iou
        if evidence.system_name == "pendulum"
        else thresholds.blocket_disc_iou
    )
    centroid_ratios = {
        name: evidence.full_horizon_centroid_error[name]
        / max(evidence.unstructured_horizon_centroid_error[name], thresholds.epsilon)
        for name in sorted(expected_objects)
    }
    cross_entropy_ratio = evidence.full_horizon_weighted_cross_entropy / max(
        evidence.unstructured_horizon_weighted_cross_entropy, thresholds.epsilon
    )
    shuffled_degradation = (
        evidence.shuffled_horizon_weighted_cross_entropy
        - evidence.full_horizon_weighted_cross_entropy
    ) / max(evidence.full_horizon_weighted_cross_entropy, thresholds.epsilon)
    checks = {
        **{
            f"current_{name}_iou": value >= iou_threshold
            for name, value in evidence.current_foreground_iou.items()
        },
        **{
            f"horizon8_{name}_centroid_vs_unstructured": (
                ratio <= thresholds.maximum_full_to_unstructured_error_ratio
            )
            for name, ratio in centroid_ratios.items()
        },
        "horizon8_cross_entropy_vs_unstructured": (
            cross_entropy_ratio <= thresholds.maximum_full_to_unstructured_error_ratio
        ),
        "shuffled_innovations_degrade_horizon8": (
            shuffled_degradation
            >= thresholds.minimum_shuffled_innovation_degradation
        ),
        "neural_hashes_unchanged": True,
        "pixels_only_evidence": True,
    }
    metrics: dict[str, float | int | str] = {
        "sample_count": evidence.sample_count,
        "horizon": evidence.horizon,
        **{
            f"current_{name}_iou": float(value)
            for name, value in evidence.current_foreground_iou.items()
        },
        **{
            f"full_horizon8_{name}_centroid_error": (
                evidence.full_horizon_centroid_error[name]
            )
            for name in sorted(expected_objects)
        },
        **{
            f"unstructured_horizon8_{name}_centroid_error": (
                evidence.unstructured_horizon_centroid_error[name]
            )
            for name in sorted(expected_objects)
        },
        **{
            f"full_to_unstructured_{name}_centroid_ratio": ratio
            for name, ratio in centroid_ratios.items()
        },
        "full_horizon8_weighted_cross_entropy": evidence.full_horizon_weighted_cross_entropy,
        "unstructured_horizon8_weighted_cross_entropy": evidence.unstructured_horizon_weighted_cross_entropy,
        "full_to_unstructured_cross_entropy_ratio": cross_entropy_ratio,
        "shuffled_horizon8_weighted_cross_entropy": evidence.shuffled_horizon_weighted_cross_entropy,
        "shuffled_innovation_degradation": shuffled_degradation,
        "test_sanitized_tensor_sha256": evidence.test_sanitized_tensor_sha256,
    }
    return _gate_result(checks, metrics)


def _module_device(*modules: nn.Module) -> torch.device:
    for module in modules:
        parameter = next(module.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(module.buffers(), None)
        if buffer is not None:
            return buffer.device
    return torch.device("cpu")


def _latent_step(dynamics: nn.Module, state: torch.Tensor, effort: torch.Tensor) -> torch.Tensor:
    if hasattr(dynamics, "step"):
        return dynamics.step(state, effort)  # type: ignore[attr-defined]
    if hasattr(dynamics, "integrate"):
        return dynamics.integrate(state, effort)  # type: ignore[attr-defined]
    return dynamics(state, effort)


def _mask_iou(logits: torch.Tensor, targets: torch.Tensor, classes: Sequence[int]) -> torch.Tensor:
    prediction = logits.argmax(dim=1)
    predicted_mask = torch.zeros_like(prediction, dtype=torch.bool)
    target_mask = torch.zeros_like(targets, dtype=torch.bool)
    for value in classes:
        predicted_mask |= prediction.eq(value)
        target_mask |= targets.eq(value)
    intersection = (predicted_mask & target_mask).sum(dim=(-2, -1)).float()
    union = (predicted_mask | target_mask).sum(dim=(-2, -1)).float()
    return intersection / union.clamp_min(1.0)


def _soft_centroid(logits: torch.Tensor, classes: Sequence[int]) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=1)
    mass_map = probabilities[:, tuple(classes)].sum(dim=1)
    height, width = mass_map.shape[-2:]
    x = torch.arange(width, device=logits.device, dtype=torch.float32) + 0.5
    y = torch.arange(height, device=logits.device, dtype=torch.float32) + 0.5
    mass = mass_map.sum(dim=(-2, -1)).clamp_min(1e-8)
    return torch.stack(
        (
            (mass_map * x).sum(dim=(-2, -1)) / mass,
            (mass_map * y[:, None]).sum(dim=(-2, -1)) / mass,
        ),
        dim=-1,
    )


def _hard_centroid(targets: torch.Tensor, classes: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros_like(targets, dtype=torch.float32)
    for value in classes:
        mask += targets.eq(value).float()
    height, width = mask.shape[-2:]
    x = torch.arange(width, device=targets.device, dtype=torch.float32) + 0.5
    y = torch.arange(height, device=targets.device, dtype=torch.float32) + 0.5
    mass = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    return torch.stack(
        (
            (mask * x).sum(dim=(-2, -1)) / mass,
            (mask * y[:, None]).sum(dim=(-2, -1)) / mass,
        ),
        dim=-1,
    )


def _centroid_errors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    object_groups: Mapping[str, Sequence[int]],
) -> dict[str, torch.Tensor]:
    errors: dict[str, torch.Tensor] = {}
    scale = float(targets.shape[-1])
    for name, classes in object_groups.items():
        errors[name] = (
            torch.linalg.vector_norm(
                _soft_centroid(logits, classes)
                - _hard_centroid(targets, classes),
                dim=-1,
            )
            / scale
        )
    return errors


@torch.no_grad()
def collect_gate2_evidence(
    *,
    system_name: str,
    test_pixels: torch.Tensor,
    test_sanitized_tensor_sha256: str,
    model_config: PixelDirectConfig,
    encoder: nn.Module,
    renderer: nn.Module,
    structured_dynamics: nn.Module,
    structured_inference: nn.Module,
    unstructured_encoder: nn.Module,
    unstructured_renderer: nn.Module,
    unstructured_dynamics: nn.Module,
    unstructured_inference: nn.Module,
    unstructured_write_field: nn.Module,
    unstructured_response_frame: nn.Module,
    class_weights: torch.Tensor,
    batch_size: int = 16,
) -> Gate2Evidence:
    """Collect the exact 512-trajectory, horizon-8 pixels-only evidence."""

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Gate 2 batch_size must be positive")
    if type(test_pixels) is not torch.Tensor or test_pixels.ndim != 4:
        raise ValueError("Gate 2 test pixels must be uint8 [trajectory,time,height,width]")
    if test_pixels.requires_grad or test_pixels.grad_fn is not None:
        raise ValueError("Gate 2 test pixels must be detached")
    if test_pixels.shape[0] != REGISTERED_GATE2_SAMPLES:
        raise ValueError("Gate 2 requires exactly 512 sanitized test trajectories")
    observed_test_sha256 = sanitized_pixel_tensor_sha256(test_pixels)
    if observed_test_sha256 != test_sanitized_tensor_sha256:
        raise ValueError("Gate 2 test pixels differ from the sealed manifest SHA-256")
    if tuple(test_pixels.shape[-2:]) != (
        model_config.image_size,
        model_config.image_size,
    ):
        raise ValueError("Gate 2 test image size differs from model_config")
    if int(test_pixels.max()) >= model_config.palette_size:
        raise ValueError("Gate 2 test pixels contain an out-of-palette class")
    if (
        type(class_weights) is not torch.Tensor
        or class_weights.ndim != 1
        or class_weights.shape[0] != model_config.palette_size
        or not class_weights.is_floating_point()
        or class_weights.requires_grad
        or class_weights.grad_fn is not None
        or not bool(torch.isfinite(class_weights).all())
        or not bool((class_weights > 0.0).all())
    ):
        raise ValueError("Gate 2 class weights are not the detached positive palette vector")
    class_weights_sha256 = _tensor_sha256(class_weights)
    if (
        unstructured_encoder is encoder
        or unstructured_renderer is renderer
        or unstructured_dynamics is structured_dynamics
        or unstructured_inference is structured_inference
    ):
        raise ValueError(
            "Gate 2 requires an independent unstructured E/renderer/F+B/inverse"
        )
    backbone = getattr(encoder, "backbone", None)
    if backbone is not None and getattr(backbone, "config", None) != model_config:
        raise ValueError("Gate 2 model_config differs from the frozen encoder backbone")
    unstructured_backbone = getattr(unstructured_encoder, "backbone", None)
    if (
        unstructured_backbone is not None
        and getattr(unstructured_backbone, "config", None) != model_config
    ):
        raise ValueError(
            "Gate 2 model_config differs from the independent encoder backbone"
        )
    shared_backbone_ids = {
        id(value)
        for candidate in (backbone, unstructured_backbone)
        if isinstance(candidate, nn.Module)
        for value in candidate.parameters()
    }
    structured_ids = {
        id(value)
        for module in (
            encoder,
            renderer,
            structured_dynamics,
            structured_inference,
        )
        for value in module.parameters()
        if id(value) not in shared_backbone_ids
    }
    unstructured_ids = {
        id(value)
        for module in (
            unstructured_encoder,
            unstructured_renderer,
            unstructured_dynamics,
            unstructured_inference,
        )
        for value in module.parameters()
        if id(value) not in shared_backbone_ids
    }
    if structured_ids & unstructured_ids:
        raise ValueError(
            "Gate 2 structured and unstructured trainable chains share tensors"
        )
    groups = _object_groups(system_name)
    modules = {
        "structuredEncoder": encoder,
        "structuredRenderer": renderer,
        "structuredDynamics": structured_dynamics,
        "structuredInference": structured_inference,
        "unstructuredEncoder": unstructured_encoder,
        "unstructuredRenderer": unstructured_renderer,
        "unstructuredDynamics": unstructured_dynamics,
        "unstructuredInference": unstructured_inference,
        "unstructuredWriteField": unstructured_write_field,
        "unstructuredResponseFrame": unstructured_response_frame,
    }
    seal = FrozenEvaluationSeal.capture(modules)
    device = _module_device(*modules.values())
    suite = make_optimization_suite(
        test_pixels,
        model_config,
        transitions=REGISTERED_GATE2_HORIZON,
    )

    structured_state_chunks: list[torch.Tensor] = []
    unstructured_state_chunks: list[torch.Tensor] = []
    structured_effort_chunks: list[torch.Tensor] = []
    unstructured_effort_chunks: list[torch.Tensor] = []
    for start in range(0, REGISTERED_GATE2_SAMPLES, batch_size):
        stop = min(start + batch_size, REGISTERED_GATE2_SAMPLES)
        contexts = suite["pixelContexts"][start:stop].to(device).long()
        structured_states = encoder(contexts)
        unstructured_states = unstructured_encoder(contexts)
        structured_effort = structured_inference(
            structured_states[:, :-1], structured_states[:, 1:]
        )
        unstructured_effort = unstructured_inference(
            unstructured_states[:, :-1], unstructured_states[:, 1:]
        )
        if (
            not isinstance(structured_states, torch.Tensor)
            or not isinstance(unstructured_states, torch.Tensor)
            or structured_states.ndim != 3
            or unstructured_states.ndim != 3
            or structured_effort.ndim != 3
            or unstructured_effort.ndim != 3
            or structured_effort.shape[:2]
            != structured_states[:, :-1].shape[:2]
            or unstructured_effort.shape[:2]
            != unstructured_states[:, :-1].shape[:2]
            or not bool(torch.isfinite(structured_states).all())
            or not bool(torch.isfinite(unstructured_states).all())
            or not bool(torch.isfinite(structured_effort).all())
            or not bool(torch.isfinite(unstructured_effort).all())
        ):
            raise ValueError("Gate 2 encoder/inverse-head output is malformed or non-finite")
        structured_state_chunks.append(structured_states.detach().cpu())
        unstructured_state_chunks.append(unstructured_states.detach().cpu())
        structured_effort_chunks.append(structured_effort.detach().cpu())
        unstructured_effort_chunks.append(unstructured_effort.detach().cpu())
    all_structured_states = torch.cat(structured_state_chunks, dim=0)
    all_unstructured_states = torch.cat(unstructured_state_chunks, dim=0)
    all_structured_efforts = torch.cat(structured_effort_chunks, dim=0)
    all_unstructured_efforts = torch.cat(unstructured_effort_chunks, dim=0)
    # A fixed-point-free cross-trajectory permutation preserves the effort
    # distribution exactly and cannot accidentally leave any example paired.
    shuffled_efforts = all_structured_efforts.roll(1, dims=0)

    ious: dict[str, list[torch.Tensor]] = {name: [] for name in groups}
    full_centroids: dict[str, list[torch.Tensor]] = {name: [] for name in groups}
    baseline_centroids: dict[str, list[torch.Tensor]] = {
        name: [] for name in groups
    }
    full_ce_sum = 0.0
    baseline_ce_sum = 0.0
    shuffled_ce_sum = 0.0
    for start in range(0, REGISTERED_GATE2_SAMPLES, batch_size):
        stop = min(start + batch_size, REGISTERED_GATE2_SAMPLES)
        count = stop - start
        structured_states = all_structured_states[start:stop].to(device)
        unstructured_states = all_unstructured_states[start:stop].to(device)
        full_efforts = all_structured_efforts[start:stop].to(device)
        baseline_efforts = all_unstructured_efforts[start:stop].to(device)
        permuted_efforts = shuffled_efforts[start:stop].to(device)
        current_logits = renderer(structured_states[:, 0])
        current_targets = suite["frames"][start:stop, 0].to(device).long()
        for name, classes in groups.items():
            ious[name].append(_mask_iou(current_logits, current_targets, classes).cpu())

        full_state = structured_states[:, 0]
        baseline_state = unstructured_states[:, 0]
        shuffled_state = structured_states[:, 0]
        for transition in range(REGISTERED_GATE2_HORIZON):
            full_state = _latent_step(
                structured_dynamics, full_state, full_efforts[:, transition]
            )
            baseline_state = _latent_step(
                unstructured_dynamics, baseline_state, baseline_efforts[:, transition]
            )
            shuffled_state = _latent_step(
                structured_dynamics, shuffled_state, permuted_efforts[:, transition]
            )
        full_logits = renderer(full_state)
        baseline_logits = unstructured_renderer(baseline_state)
        shuffled_logits = renderer(shuffled_state)
        targets = suite["frames"][start:stop, REGISTERED_GATE2_HORIZON].to(device).long()
        for name, value in _centroid_errors(full_logits, targets, groups).items():
            full_centroids[name].append(value.cpu())
        for name, value in _centroid_errors(baseline_logits, targets, groups).items():
            baseline_centroids[name].append(value.cpu())
        weights = class_weights.to(device)
        full_ce_sum += count * float(weighted_pixel_cross_entropy(full_logits, targets, weights))
        baseline_ce_sum += count * float(
            weighted_pixel_cross_entropy(baseline_logits, targets, weights)
        )
        shuffled_ce_sum += count * float(
            weighted_pixel_cross_entropy(shuffled_logits, targets, weights)
        )
    seal.assert_unchanged()
    neural_hashes_after = dict(FrozenEvaluationSeal.capture(modules).hashes)
    denominator = float(REGISTERED_GATE2_SAMPLES)
    evidence = Gate2Evidence(
        system_name=system_name,
        sample_count=REGISTERED_GATE2_SAMPLES,
        horizon=REGISTERED_GATE2_HORIZON,
        current_foreground_iou={
            name: float(torch.cat(values).mean()) for name, values in ious.items()
        },
        full_horizon_centroid_error={
            name: float(torch.cat(values).mean())
            for name, values in full_centroids.items()
        },
        unstructured_horizon_centroid_error={
            name: float(torch.cat(values).mean())
            for name, values in baseline_centroids.items()
        },
        full_horizon_weighted_cross_entropy=full_ce_sum / denominator,
        unstructured_horizon_weighted_cross_entropy=baseline_ce_sum / denominator,
        shuffled_horizon_weighted_cross_entropy=shuffled_ce_sum / denominator,
        test_sanitized_tensor_sha256=observed_test_sha256,
        class_weights_sha256=class_weights_sha256,
        neural_hashes_before=dict(seal.hashes),
        neural_hashes_after=neural_hashes_after,
    )
    return evidence


__all__ = [
    "Gate2Evidence",
    "Gate2Thresholds",
    "REGISTERED_GATE2_HORIZON",
    "REGISTERED_GATE2_SAMPLES",
    "audit_gate_2",
    "collect_gate2_evidence",
    "gate2_tensor_sha256",
]
