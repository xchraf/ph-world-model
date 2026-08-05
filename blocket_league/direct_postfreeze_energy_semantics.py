"""Post-freeze semantic audit for the learned Hamiltonian in Experiment F.

The port-Hamiltonian power identity is structural: it can hold for a learned
scalar that has no useful physical meaning.  This module therefore asks a
strictly stronger question after every neural tensor is frozen.  Using the
same disjoint 256/128 state pools as Gate 5, it fits exactly one positive
affine map from the scalar latent Hamiltonian to mechanical energy and tests
that map on the held-out pool.

No energy value, simulator coordinate, physical command, slope, or intercept
is available to training, checkpoint selection, port construction, physical
calibration, or control.  The two affine coefficients are audit-only and can
never enter a planner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping, TYPE_CHECKING

import torch

from .action_free_excitation import experiment_f_blocket_world_config
from .direct_postfreeze_force_port import Gate5Evidence
from .passive_control_systems import PendulumConfig
from .passive_jacobian_ph_model import module_tensor_hash

if TYPE_CHECKING:  # pragma: no cover
    from .direct_postfreeze_runner import LoadedPostFreezeSystem


ENERGY_SEMANTIC_KIND = "experiment_f_physical_energy_semantics_v1"
REGISTERED_ENERGY_ALIGNMENT_SAMPLES = 256
REGISTERED_ENERGY_EVALUATION_SAMPLES = 128
REGISTERED_MAXIMUM_ENERGY_NRMSE = 0.35
REGISTERED_MINIMUM_ENERGY_R2 = 0.85
REGISTERED_MINIMUM_ENERGY_CORRELATION = 0.90
REGISTERED_MINIMUM_POSITIVE_SLOPE = 1e-8
REGISTERED_MINIMUM_LATENT_ENERGY_STD = 1e-4
REGISTERED_MINIMUM_PHYSICAL_ENERGY_STD = 1e-6
_SHA256 = re.compile(r"[0-9a-f]{64}")
ENERGY_SEMANTIC_CHECKS = frozenset(
    {
        "positiveAffineEnergyScale",
        "alignmentEnergyError",
        "heldoutEnergyError",
        "heldoutEnergyR2",
        "heldoutEnergyCorrelation",
        "latentEnergyHasVariation",
        "physicalEnergyHasVariation",
        "postfreezeNoGradientOrCommand",
    }
)
ENERGY_SEMANTIC_METRICS = frozenset(
    {
        "alignmentSamples",
        "evaluationSamples",
        "positiveAffineSlope",
        "affineIntercept",
        "fitNormalizedRMSE",
        "heldoutNormalizedRMSE",
        "heldoutR2",
        "heldoutPearson",
        "alignmentLatentEnergyStd",
        "alignmentPhysicalEnergyStd",
        "evaluationLatentEnergyStd",
        "evaluationPhysicalEnergyStd",
        "semanticMap",
    }
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checks_from_serialized_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, bool]:
    """Reconstruct every pass/fail bit from the JSON-safe metric table."""

    if type(metrics) is not dict or set(metrics) != ENERGY_SEMANTIC_METRICS:
        raise ValueError("physical-energy metric schema is not exact")
    if (
        type(metrics["alignmentSamples"]) is not int
        or metrics["alignmentSamples"] != REGISTERED_ENERGY_ALIGNMENT_SAMPLES
        or type(metrics["evaluationSamples"]) is not int
        or metrics["evaluationSamples"] != REGISTERED_ENERGY_EVALUATION_SAMPLES
        or metrics["semanticMap"]
        != "physical_energy=slope*latent_H+intercept"
    ):
        raise ValueError("physical-energy metric protocol changed")
    numeric_names = ENERGY_SEMANTIC_METRICS.difference(
        {"alignmentSamples", "evaluationSamples", "semanticMap"}
    )
    if any(
        type(metrics[name]) not in (int, float)
        or not math.isfinite(float(metrics[name]))
        for name in numeric_names
    ):
        raise ValueError("physical-energy metrics contain a non-finite value")
    return {
        "positiveAffineEnergyScale": (
            float(metrics["positiveAffineSlope"])
            >= REGISTERED_MINIMUM_POSITIVE_SLOPE
        ),
        "alignmentEnergyError": (
            float(metrics["fitNormalizedRMSE"])
            <= REGISTERED_MAXIMUM_ENERGY_NRMSE
        ),
        "heldoutEnergyError": (
            float(metrics["heldoutNormalizedRMSE"])
            <= REGISTERED_MAXIMUM_ENERGY_NRMSE
        ),
        "heldoutEnergyR2": (
            float(metrics["heldoutR2"]) >= REGISTERED_MINIMUM_ENERGY_R2
        ),
        "heldoutEnergyCorrelation": (
            float(metrics["heldoutPearson"])
            >= REGISTERED_MINIMUM_ENERGY_CORRELATION
        ),
        "latentEnergyHasVariation": (
            min(
                float(metrics["alignmentLatentEnergyStd"]),
                float(metrics["evaluationLatentEnergyStd"]),
            )
            >= REGISTERED_MINIMUM_LATENT_ENERGY_STD
        ),
        "physicalEnergyHasVariation": (
            min(
                float(metrics["alignmentPhysicalEnergyStd"]),
                float(metrics["evaluationPhysicalEnergyStd"]),
            )
            >= REGISTERED_MINIMUM_PHYSICAL_ENERGY_STD
        ),
        # The typed evidence constructor has already authenticated unchanged
        # neural hashes.  The JSON artifact separately locks both counters to
        # zero and cannot expose a caller-provided pass flag for this check.
        "postfreezeNoGradientOrCommand": True,
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EnergySemanticThresholds:
    """Preregistered physical-meaning thresholds; not tunable post-freeze."""

    maximum_fit_normalized_rmse: float = REGISTERED_MAXIMUM_ENERGY_NRMSE
    maximum_heldout_normalized_rmse: float = REGISTERED_MAXIMUM_ENERGY_NRMSE
    minimum_heldout_r2: float = REGISTERED_MINIMUM_ENERGY_R2
    minimum_heldout_pearson: float = REGISTERED_MINIMUM_ENERGY_CORRELATION
    minimum_positive_slope: float = REGISTERED_MINIMUM_POSITIVE_SLOPE
    minimum_latent_energy_std: float = REGISTERED_MINIMUM_LATENT_ENERGY_STD
    minimum_physical_energy_std: float = REGISTERED_MINIMUM_PHYSICAL_ENERGY_STD
    alignment_samples: int = REGISTERED_ENERGY_ALIGNMENT_SAMPLES
    evaluation_samples: int = REGISTERED_ENERGY_EVALUATION_SAMPLES

    def __post_init__(self) -> None:
        numeric = (
            self.maximum_fit_normalized_rmse,
            self.maximum_heldout_normalized_rmse,
            self.minimum_heldout_r2,
            self.minimum_heldout_pearson,
            self.minimum_positive_slope,
            self.minimum_latent_energy_std,
            self.minimum_physical_energy_std,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("energy-semantic thresholds must be finite")
        if (
            self.maximum_fit_normalized_rmse != REGISTERED_MAXIMUM_ENERGY_NRMSE
            or self.maximum_heldout_normalized_rmse
            != REGISTERED_MAXIMUM_ENERGY_NRMSE
            or self.minimum_heldout_r2 != REGISTERED_MINIMUM_ENERGY_R2
            or self.minimum_heldout_pearson
            != REGISTERED_MINIMUM_ENERGY_CORRELATION
            or self.minimum_positive_slope
            != REGISTERED_MINIMUM_POSITIVE_SLOPE
            or self.minimum_latent_energy_std
            != REGISTERED_MINIMUM_LATENT_ENERGY_STD
            or self.minimum_physical_energy_std
            != REGISTERED_MINIMUM_PHYSICAL_ENERGY_STD
            or self.alignment_samples != REGISTERED_ENERGY_ALIGNMENT_SAMPLES
            or self.evaluation_samples != REGISTERED_ENERGY_EVALUATION_SAMPLES
        ):
            raise ValueError("energy-semantic protocol differs from preregistration")


@dataclass(frozen=True)
class PositiveAffineEnergyCalibration:
    slope: float
    intercept: float
    fit_normalized_rmse: float
    samples: int

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(value)
                for value in (self.slope, self.intercept, self.fit_normalized_rmse)
            )
            or self.fit_normalized_rmse < 0.0
            or self.samples != REGISTERED_ENERGY_ALIGNMENT_SAMPLES
        ):
            raise ValueError("physical-energy affine calibration is invalid")


@dataclass(frozen=True)
class EnergySemanticMetrics:
    heldout_normalized_rmse: float
    heldout_r2: float
    heldout_pearson: float
    alignment_latent_std: float
    alignment_physical_std: float
    evaluation_latent_std: float
    evaluation_physical_std: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) for value in self.__dict__.values()):
            raise ValueError("physical-energy semantic metric is non-finite")
        if self.heldout_normalized_rmse < 0.0:
            raise ValueError("physical-energy normalized error cannot be negative")
        if not -1.000001 <= self.heldout_pearson <= 1.000001:
            raise ValueError("physical-energy correlation is outside [-1,1]")


def physical_energy_from_gate5_coordinates(
    system_name: str,
    standardized_coordinates: torch.Tensor,
    coordinate_mean: torch.Tensor,
    coordinate_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute mechanical energy from authenticated post-freeze coordinates."""

    if standardized_coordinates.ndim != 2:
        raise ValueError("energy audit coordinates must have shape [sample,state]")
    state_size = 2 if system_name == "pendulum" else 8 if system_name == "blocket" else -1
    if (
        standardized_coordinates.shape[-1] != state_size
        or tuple(coordinate_mean.shape) != (state_size,)
        or tuple(coordinate_scale.shape) != (state_size,)
        or bool((coordinate_scale <= 0.0).any())
    ):
        raise ValueError("energy audit coordinate schema is invalid")
    raw = (
        standardized_coordinates.double()
        * coordinate_scale.to(standardized_coordinates.device).double()
        + coordinate_mean.to(standardized_coordinates.device).double()
    )
    if system_name == "pendulum":
        config = PendulumConfig()
        angle = raw[:, 0]
        angular_momentum = raw[:, 1]
        energy = (
            angular_momentum.square() / (2.0 * config.inertia)
            + config.mass
            * config.gravity
            * config.length
            * (1.0 - torch.cos(angle))
        )
    else:
        config = experiment_f_blocket_world_config()
        player_momentum = raw[:, 4:6]
        puck_momentum = raw[:, 6:8]
        energy = (
            player_momentum.square().sum(dim=-1) / (2.0 * config.player_mass)
            + puck_momentum.square().sum(dim=-1) / (2.0 * config.puck_mass)
        )
    if (
        energy.ndim != 1
        or not bool(torch.isfinite(energy).all())
        or bool((energy < -1e-10).any())
    ):
        raise ValueError("computed post-freeze mechanical energy is invalid")
    return energy.detach()


def fit_positive_affine_energy_calibration(
    latent_energy: torch.Tensor,
    physical_energy: torch.Tensor,
) -> PositiveAffineEnergyCalibration:
    """Fit the sole two-parameter audit map by closed-form float64 OLS."""

    if (
        latent_energy.ndim != 1
        or latent_energy.shape != physical_energy.shape
        or latent_energy.numel() != REGISTERED_ENERGY_ALIGNMENT_SAMPLES
        or latent_energy.requires_grad
        or physical_energy.requires_grad
    ):
        raise ValueError("energy alignment tensors have the wrong schema")
    latent = latent_energy.detach().double()
    physical = physical_energy.detach().double()
    if not bool(torch.isfinite(latent).all() and torch.isfinite(physical).all()):
        raise ValueError("energy alignment tensors contain non-finite values")
    centered_latent = latent - latent.mean()
    centered_physical = physical - physical.mean()
    denominator = centered_latent.square().sum()
    if float(denominator) <= torch.finfo(torch.float64).eps:
        raise ValueError("latent Hamiltonian has no alignment-set variation")
    slope_tensor = (centered_latent * centered_physical).sum() / denominator
    intercept_tensor = physical.mean() - slope_tensor * latent.mean()
    prediction = slope_tensor * latent + intercept_tensor
    physical_norm = torch.linalg.vector_norm(centered_physical)
    if float(physical_norm) <= torch.finfo(torch.float64).eps:
        raise ValueError("physical energy has no alignment-set variation")
    error = torch.linalg.vector_norm(prediction - physical) / physical_norm
    return PositiveAffineEnergyCalibration(
        slope=float(slope_tensor.cpu()),
        intercept=float(intercept_tensor.cpu()),
        fit_normalized_rmse=float(error.cpu()),
        samples=latent.numel(),
    )


def evaluate_affine_energy_calibration(
    calibration: PositiveAffineEnergyCalibration,
    alignment_latent_energy: torch.Tensor,
    alignment_physical_energy: torch.Tensor,
    evaluation_latent_energy: torch.Tensor,
    evaluation_physical_energy: torch.Tensor,
) -> EnergySemanticMetrics:
    if (
        evaluation_latent_energy.ndim != 1
        or evaluation_latent_energy.shape != evaluation_physical_energy.shape
        or evaluation_latent_energy.numel() != REGISTERED_ENERGY_EVALUATION_SAMPLES
        or alignment_latent_energy.numel() != REGISTERED_ENERGY_ALIGNMENT_SAMPLES
        or alignment_latent_energy.shape != alignment_physical_energy.shape
        or any(
            value.requires_grad
            for value in (
                alignment_latent_energy,
                alignment_physical_energy,
                evaluation_latent_energy,
                evaluation_physical_energy,
            )
        )
    ):
        raise ValueError("energy evaluation tensors have the wrong schema")
    latent = evaluation_latent_energy.detach().double()
    physical = evaluation_physical_energy.detach().double()
    prediction = calibration.slope * latent + calibration.intercept
    centered_physical = physical - physical.mean()
    physical_norm = torch.linalg.vector_norm(centered_physical)
    if float(physical_norm) <= torch.finfo(torch.float64).eps:
        raise ValueError("held-out physical energy has no variation")
    normalized_rmse = torch.linalg.vector_norm(prediction - physical) / physical_norm
    centered_prediction = prediction - prediction.mean()
    prediction_norm = torch.linalg.vector_norm(centered_prediction)
    pearson = (
        (centered_prediction * centered_physical).sum()
        / (prediction_norm * physical_norm).clamp_min(torch.finfo(torch.float64).eps)
    )
    return EnergySemanticMetrics(
        heldout_normalized_rmse=float(normalized_rmse.cpu()),
        heldout_r2=float((1.0 - normalized_rmse.square()).cpu()),
        heldout_pearson=float(pearson.cpu()),
        alignment_latent_std=float(
            alignment_latent_energy.detach().double().std(unbiased=False).cpu()
        ),
        alignment_physical_std=float(
            alignment_physical_energy.detach().double().std(unbiased=False).cpu()
        ),
        evaluation_latent_std=float(latent.std(unbiased=False).cpu()),
        evaluation_physical_std=float(physical.std(unbiased=False).cpu()),
    )


def _evidence_sha256(evidence: Any) -> str:
    primitive = {
        "kind": ENERGY_SEMANTIC_KIND,
        "system": evidence.system_name,
        "gate5EvidenceSha256": evidence.gate5_evidence_sha256,
        "modelSha256": evidence.model_sha256,
        "checkpointSha256": evidence.checkpoint_sha256,
        "alignmentContextSha256": evidence.alignment_context_sha256,
        "evaluationContextSha256": evidence.evaluation_context_sha256,
        "alignmentLatentEnergySha256": _tensor_sha256(evidence.alignment_latent_energy),
        "alignmentPhysicalEnergySha256": _tensor_sha256(evidence.alignment_physical_energy),
        "evaluationLatentEnergySha256": _tensor_sha256(evidence.evaluation_latent_energy),
        "evaluationPhysicalEnergySha256": _tensor_sha256(evidence.evaluation_physical_energy),
        "calibration": asdict(evidence.calibration),
        "metrics": asdict(evidence.metrics),
        "neuralHashBefore": evidence.neural_hash_before,
        "neuralHashAfter": evidence.neural_hash_after,
        "gradientUpdates": evidence.gradient_updates,
        "physicalCommandsRead": evidence.physical_commands_read,
        "readPhase": evidence.read_phase,
    }
    return _canonical_sha256(primitive)


@dataclass(frozen=True)
class EnergySemanticEvidence:
    system_name: str
    gate5_evidence_sha256: str
    model_sha256: str
    checkpoint_sha256: str
    alignment_context_sha256: str
    evaluation_context_sha256: str
    alignment_latent_energy: torch.Tensor = field(repr=False, compare=False)
    alignment_physical_energy: torch.Tensor = field(repr=False, compare=False)
    evaluation_latent_energy: torch.Tensor = field(repr=False, compare=False)
    evaluation_physical_energy: torch.Tensor = field(repr=False, compare=False)
    calibration: PositiveAffineEnergyCalibration
    metrics: EnergySemanticMetrics
    neural_hash_before: str
    neural_hash_after: str
    evidence_sha256: str
    gradient_updates: int = 0
    physical_commands_read: int = 0
    read_phase: str = "postfreeze_gate5_coordinates_energy_audit_only"

    def __post_init__(self) -> None:
        hashes = (
            self.gate5_evidence_sha256,
            self.model_sha256,
            self.checkpoint_sha256,
            self.alignment_context_sha256,
            self.evaluation_context_sha256,
            self.neural_hash_before,
            self.neural_hash_after,
            self.evidence_sha256,
        )
        tensors = (
            self.alignment_latent_energy,
            self.alignment_physical_energy,
            self.evaluation_latent_energy,
            self.evaluation_physical_energy,
        )
        if (
            self.system_name not in {"pendulum", "blocket"}
            or any(_SHA256.fullmatch(value) is None for value in hashes)
            or tuple(self.alignment_latent_energy.shape)
            != (REGISTERED_ENERGY_ALIGNMENT_SAMPLES,)
            or tuple(self.alignment_physical_energy.shape)
            != (REGISTERED_ENERGY_ALIGNMENT_SAMPLES,)
            or tuple(self.evaluation_latent_energy.shape)
            != (REGISTERED_ENERGY_EVALUATION_SAMPLES,)
            or tuple(self.evaluation_physical_energy.shape)
            != (REGISTERED_ENERGY_EVALUATION_SAMPLES,)
            or any(
                tensor.requires_grad
                or tensor.grad_fn is not None
                or not bool(torch.isfinite(tensor).all())
                for tensor in tensors
            )
            or self.neural_hash_before != self.neural_hash_after
            or self.model_sha256 != self.neural_hash_before
            or self.gradient_updates != 0
            or self.physical_commands_read != 0
            or self.read_phase
            != "postfreeze_gate5_coordinates_energy_audit_only"
            or _evidence_sha256(self) != self.evidence_sha256
        ):
            raise ValueError("physical-energy semantic evidence is invalid")


def collect_energy_semantic_evidence(
    loaded: "LoadedPostFreezeSystem",
    gate5: Gate5Evidence,
) -> EnergySemanticEvidence:
    """Derive energy evidence only from an already frozen Gate-5 state bank."""

    loaded.assert_frozen_and_unchanged()
    model = loaded.full.bundle.model
    if (
        gate5.model is not model
        or gate5.system_name != loaded.system_name
        or gate5.model_sha256 != module_tensor_hash(model)
        or gate5.checkpoint_sha256 != loaded.full.checkpoint_sha256
    ):
        raise ValueError("energy semantics are not anchored to the frozen Gate-5 model")
    model_hash = module_tensor_hash(model)
    with torch.no_grad():
        alignment_latent = model.core.hamiltonian(
            gate5.alignment_latent.to(loaded.device).float()
        ).detach().double()
        evaluation_latent = model.core.hamiltonian(
            gate5.evaluation_latent.to(loaded.device).float()
        ).detach().double()
    alignment_physical = physical_energy_from_gate5_coordinates(
        loaded.system_name,
        gate5.alignment_coordinates,
        gate5.coordinate_mean,
        gate5.coordinate_scale,
    ).to(loaded.device)
    evaluation_physical = physical_energy_from_gate5_coordinates(
        loaded.system_name,
        gate5.evaluation_coordinates,
        gate5.coordinate_mean,
        gate5.coordinate_scale,
    ).to(loaded.device)
    calibration = fit_positive_affine_energy_calibration(
        alignment_latent, alignment_physical
    )
    metrics = evaluate_affine_energy_calibration(
        calibration,
        alignment_latent,
        alignment_physical,
        evaluation_latent,
        evaluation_physical,
    )
    loaded.assert_frozen_and_unchanged()
    values: dict[str, Any] = {
        "system_name": loaded.system_name,
        "gate5_evidence_sha256": gate5.evidence_sha256,
        "model_sha256": model_hash,
        "checkpoint_sha256": gate5.checkpoint_sha256,
        "alignment_context_sha256": gate5.alignment_context_sha256,
        "evaluation_context_sha256": gate5.evaluation_context_sha256,
        "alignment_latent_energy": alignment_latent,
        "alignment_physical_energy": alignment_physical.detach().double(),
        "evaluation_latent_energy": evaluation_latent,
        "evaluation_physical_energy": evaluation_physical.detach().double(),
        "calibration": calibration,
        "metrics": metrics,
        "neural_hash_before": model_hash,
        "neural_hash_after": module_tensor_hash(model),
        "evidence_sha256": "0" * 64,
        "gradient_updates": 0,
        "physical_commands_read": 0,
        "read_phase": "postfreeze_gate5_coordinates_energy_audit_only",
    }
    values["evidence_sha256"] = _evidence_sha256(type("Evidence", (), values)())
    return EnergySemanticEvidence(**values)


@dataclass(frozen=True)
class EnergySemanticAudit:
    system_name: str
    gate5_evidence_sha256: str
    model_sha256: str
    checkpoint_sha256: str
    alignment_context_sha256: str
    evaluation_context_sha256: str
    passed: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, float | int | str]
    evidence_sha256: str

    def __post_init__(self) -> None:
        hashes = (
            self.gate5_evidence_sha256,
            self.model_sha256,
            self.checkpoint_sha256,
            self.alignment_context_sha256,
            self.evaluation_context_sha256,
            self.evidence_sha256,
        )
        if (
            self.system_name not in {"pendulum", "blocket"}
            or any(_SHA256.fullmatch(value) is None for value in hashes)
            or type(self.checks) is not dict
            or set(self.checks) != ENERGY_SEMANTIC_CHECKS
            or any(type(value) is not bool for value in self.checks.values())
            or type(self.passed) is not bool
            or self.passed is not all(self.checks.values())
            or _SHA256.fullmatch(self.evidence_sha256) is None
            or type(self.metrics) is not dict
            or _checks_from_serialized_metrics(self.metrics) != self.checks
        ):
            raise ValueError("physical-energy audit result is not derived from its checks")

    def to_dict(self) -> dict[str, Any]:
        core = {
            "kind": ENERGY_SEMANTIC_KIND,
            "system": self.system_name,
            "gate5EvidenceSha256": self.gate5_evidence_sha256,
            "modelSha256": self.model_sha256,
            "checkpointSha256": self.checkpoint_sha256,
            "alignmentContextSha256": self.alignment_context_sha256,
            "evaluationContextSha256": self.evaluation_context_sha256,
            "passed": self.passed,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "evidenceSha256": self.evidence_sha256,
            "gradientUpdates": 0,
            "physicalCommandsRead": 0,
            "readPhase": "postfreeze_gate5_coordinates_energy_audit_only",
        }
        return {**core, "artifactSha256": _canonical_sha256(core)}


def verify_energy_semantic_artifact_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != {
        "kind",
        "system",
        "gate5EvidenceSha256",
        "modelSha256",
        "checkpointSha256",
        "alignmentContextSha256",
        "evaluationContextSha256",
        "passed",
        "checks",
        "metrics",
        "evidenceSha256",
        "gradientUpdates",
        "physicalCommandsRead",
        "readPhase",
        "artifactSha256",
    }:
        raise ValueError("physical-energy artifact schema is not exact")
    core = {name: payload[name] for name in payload if name != "artifactSha256"}
    if (
        payload["kind"] != ENERGY_SEMANTIC_KIND
        or payload["system"] not in {"pendulum", "blocket"}
        or any(
            type(payload[name]) is not str
            or _SHA256.fullmatch(payload[name]) is None
            for name in (
                "gate5EvidenceSha256",
                "modelSha256",
                "checkpointSha256",
                "alignmentContextSha256",
                "evaluationContextSha256",
                "evidenceSha256",
            )
        )
        or payload["artifactSha256"] != _canonical_sha256(core)
        or type(payload["checks"]) is not dict
        or set(payload["checks"]) != ENERGY_SEMANTIC_CHECKS
        or any(type(value) is not bool for value in payload["checks"].values())
        or type(payload["passed"]) is not bool
        or payload["passed"] is not all(payload["checks"].values())
        or _checks_from_serialized_metrics(payload["metrics"])
        != payload["checks"]
        or _SHA256.fullmatch(payload["evidenceSha256"]) is None
        or payload["gradientUpdates"] != 0
        or payload["physicalCommandsRead"] != 0
        or payload["readPhase"]
        != "postfreeze_gate5_coordinates_energy_audit_only"
    ):
        raise ValueError("physical-energy artifact provenance/digest is invalid")
    return dict(payload)


def audit_energy_semantics(
    evidence: EnergySemanticEvidence,
    thresholds: EnergySemanticThresholds = EnergySemanticThresholds(),
) -> EnergySemanticAudit:
    """Recompute the analytic fit and decide from tensor evidence, not a bool."""

    EnergySemanticEvidence(**dict(evidence.__dict__))
    expected_calibration = fit_positive_affine_energy_calibration(
        evidence.alignment_latent_energy,
        evidence.alignment_physical_energy,
    )
    expected_metrics = evaluate_affine_energy_calibration(
        expected_calibration,
        evidence.alignment_latent_energy,
        evidence.alignment_physical_energy,
        evidence.evaluation_latent_energy,
        evidence.evaluation_physical_energy,
    )
    if expected_calibration != evidence.calibration or expected_metrics != evidence.metrics:
        raise ValueError("physical-energy audit statistics were not recomputed exactly")
    checks = {
        "positiveAffineEnergyScale": (
            evidence.calibration.slope >= thresholds.minimum_positive_slope
        ),
        "alignmentEnergyError": (
            evidence.calibration.fit_normalized_rmse
            <= thresholds.maximum_fit_normalized_rmse
        ),
        "heldoutEnergyError": (
            evidence.metrics.heldout_normalized_rmse
            <= thresholds.maximum_heldout_normalized_rmse
        ),
        "heldoutEnergyR2": (
            evidence.metrics.heldout_r2 >= thresholds.minimum_heldout_r2
        ),
        "heldoutEnergyCorrelation": (
            evidence.metrics.heldout_pearson
            >= thresholds.minimum_heldout_pearson
        ),
        "latentEnergyHasVariation": (
            min(
                evidence.metrics.alignment_latent_std,
                evidence.metrics.evaluation_latent_std,
            )
            >= thresholds.minimum_latent_energy_std
        ),
        "physicalEnergyHasVariation": (
            min(
                evidence.metrics.alignment_physical_std,
                evidence.metrics.evaluation_physical_std,
            )
            >= thresholds.minimum_physical_energy_std
        ),
        "postfreezeNoGradientOrCommand": (
            evidence.gradient_updates == 0
            and evidence.physical_commands_read == 0
            and evidence.neural_hash_before == evidence.neural_hash_after
        ),
    }
    metrics: dict[str, float | int | str] = {
        "alignmentSamples": thresholds.alignment_samples,
        "evaluationSamples": thresholds.evaluation_samples,
        "positiveAffineSlope": evidence.calibration.slope,
        "affineIntercept": evidence.calibration.intercept,
        "fitNormalizedRMSE": evidence.calibration.fit_normalized_rmse,
        "heldoutNormalizedRMSE": evidence.metrics.heldout_normalized_rmse,
        "heldoutR2": evidence.metrics.heldout_r2,
        "heldoutPearson": evidence.metrics.heldout_pearson,
        "alignmentLatentEnergyStd": evidence.metrics.alignment_latent_std,
        "alignmentPhysicalEnergyStd": evidence.metrics.alignment_physical_std,
        "evaluationLatentEnergyStd": evidence.metrics.evaluation_latent_std,
        "evaluationPhysicalEnergyStd": evidence.metrics.evaluation_physical_std,
        "semanticMap": "physical_energy=slope*latent_H+intercept",
    }
    return EnergySemanticAudit(
        system_name=evidence.system_name,
        gate5_evidence_sha256=evidence.gate5_evidence_sha256,
        model_sha256=evidence.model_sha256,
        checkpoint_sha256=evidence.checkpoint_sha256,
        alignment_context_sha256=evidence.alignment_context_sha256,
        evaluation_context_sha256=evidence.evaluation_context_sha256,
        passed=all(checks.values()),
        checks=checks,
        metrics=metrics,
        evidence_sha256=evidence.evidence_sha256,
    )


__all__ = [
    "ENERGY_SEMANTIC_KIND",
    "EnergySemanticAudit",
    "EnergySemanticEvidence",
    "EnergySemanticMetrics",
    "EnergySemanticThresholds",
    "PositiveAffineEnergyCalibration",
    "audit_energy_semantics",
    "collect_energy_semantic_evidence",
    "evaluate_affine_energy_calibration",
    "fit_positive_affine_energy_calibration",
    "physical_energy_from_gate5_coordinates",
    "verify_energy_semantic_artifact_payload",
]
