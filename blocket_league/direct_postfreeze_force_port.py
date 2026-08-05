"""Authenticated post-freeze collection for Experiment F Gate 5.

The simulator coordinates used here are deliberately isolated from training,
calibration, realizability, and control.  They are read only after the complete
neural system has been reconstructed and frozen.  A single affine chart is fit
analytically from a locked alignment set; all force responses remain derivatives
of the frozen latent dynamics with respect to its unnamed latent port.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from .direct_activation_lens import direct_dynamics_pulse_responses
from .direct_ph_structure_audits import (
    AffineAuditAlignment,
    ForcePortThresholds,
    GateAuditResult,
    audit_force_port_signature,
    fit_postfreeze_affine_audit_alignment,
)
from .direct_physical_evaluation import (
    ProbeCandidate,
    make_builtin_probe_candidates,
)
from .passive_jacobian_ph_model import module_tensor_hash

if TYPE_CHECKING:  # pragma: no cover - avoids a runner/import cycle at runtime
    from .direct_postfreeze_runner import LoadedPostFreezeSystem


REGISTERED_GATE5_ALIGNMENT_SAMPLES = 256
REGISTERED_GATE5_EVALUATION_SAMPLES = 128
REGISTERED_GATE5_HORIZONS = (1, 4)
REGISTERED_GATE5_ALIGNMENT_SEED = 151_910_737 + 50_000
REGISTERED_GATE5_EVALUATION_SEED = 151_910_737 + 55_000
REGISTERED_GATE5_RIDGE = 1e-6
REGISTERED_GATE5_MIN_LOCALITY_SAMPLES = 64


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _identifier_context_sha256(candidates: Sequence[ProbeCandidate]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(candidate.identifier.encode("utf-8"))
        digest.update(_tensor_sha256(candidate.context).encode("ascii"))
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Gate5CollectionConfig:
    """Locked sampling rule; only the execution batch size is operational."""

    alignment_samples: int = REGISTERED_GATE5_ALIGNMENT_SAMPLES
    evaluation_samples: int = REGISTERED_GATE5_EVALUATION_SAMPLES
    alignment_seed: int = REGISTERED_GATE5_ALIGNMENT_SEED
    evaluation_seed: int = REGISTERED_GATE5_EVALUATION_SEED
    ridge: float = REGISTERED_GATE5_RIDGE
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.alignment_samples != REGISTERED_GATE5_ALIGNMENT_SAMPLES:
            raise ValueError("Gate 5 alignment set must contain exactly 256 states")
        if self.evaluation_samples != REGISTERED_GATE5_EVALUATION_SAMPLES:
            raise ValueError("Gate 5 evaluation set must contain exactly 128 states")
        if self.alignment_seed != REGISTERED_GATE5_ALIGNMENT_SEED:
            raise ValueError("Gate 5 alignment seed is preregistered")
        if self.evaluation_seed != REGISTERED_GATE5_EVALUATION_SEED:
            raise ValueError("Gate 5 evaluation seed is preregistered")
        if self.alignment_seed == self.evaluation_seed:
            raise ValueError("Gate 5 alignment and evaluation seeds must be disjoint")
        if self.ridge != REGISTERED_GATE5_RIDGE:
            raise ValueError("Gate 5 affine ridge is preregistered at 1e-6")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Gate 5 batch_size must be a positive integer")


def _coordinate_schema(system_name: str) -> tuple[str, ...]:
    if system_name == "pendulum":
        return ("angle", "angular_momentum")
    if system_name == "blocket":
        return (
            "player_position_x",
            "player_position_y",
            "puck_position_x",
            "puck_position_y",
            "player_momentum_x",
            "player_momentum_y",
            "puck_momentum_x",
            "puck_momentum_y",
        )
    raise KeyError(f"no registered Gate 5 coordinate chart for {system_name!r}")


def _raw_audit_coordinates(
    candidates: Sequence[ProbeCandidate], system_name: str
) -> torch.Tensor:
    """Read the sole registered simulator-state fields, after full freeze."""

    rows: list[np.ndarray] = []
    for candidate in candidates:
        environment = candidate.environment
        if system_name == "pendulum":
            state = environment.state
            inertia = float(environment.config.inertia)
            row = np.asarray(
                (float(state.angle), inertia * float(state.angular_velocity)),
                dtype=np.float32,
            )
        elif system_name == "blocket":
            state = environment.state
            config = environment.config
            row = np.asarray(
                (
                    *state.player_position,
                    *state.puck_position,
                    *(float(config.player_mass) * state.player_velocity),
                    *(float(config.puck_mass) * state.puck_velocity),
                ),
                dtype=np.float32,
            )
        else:  # pragma: no cover - guarded by _coordinate_schema
            raise KeyError(system_name)
        if row.shape != (len(_coordinate_schema(system_name)),) or not bool(
            np.isfinite(row).all()
        ):
            raise ValueError("Gate 5 simulator coordinate is malformed")
        rows.append(row)
    result = torch.from_numpy(np.stack(rows)).float()
    if result.requires_grad:  # pragma: no cover - construction is detached
        raise AssertionError("physical audit coordinates acquired an autograd graph")
    return result


def _standardize_alignment_coordinates(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw.ndim != 2 or raw.shape[0] != REGISTERED_GATE5_ALIGNMENT_SAMPLES:
        raise ValueError("Gate 5 raw alignment coordinates have the wrong shape")
    mean = raw.mean(dim=0)
    scale = raw.std(dim=0, unbiased=False)
    if bool((scale <= 1e-6).any()):
        raise ValueError("Gate 5 audit coordinate has no alignment-set variation")
    standardized = ((raw - mean) / scale).detach()
    return standardized, mean.detach(), scale.detach()


def _affine_evaluation_error(
    alignment: AffineAuditAlignment,
    latent: torch.Tensor,
    coordinates: torch.Tensor,
) -> float:
    if latent.shape != coordinates.shape or latent.ndim != 2:
        raise ValueError("Gate 5 evaluation alignment tensors must share [sample,state]")
    if latent.requires_grad or coordinates.requires_grad:
        raise ValueError("Gate 5 evaluation alignment tensors must be detached")
    with torch.no_grad():
        target = coordinates.to(device=latent.device, dtype=latent.dtype)
        prediction = latent @ alignment.weight.T + alignment.bias
        denominator = torch.linalg.vector_norm(
            target - target.mean(dim=0, keepdim=True)
        ).clamp_min(torch.finfo(target.dtype).eps)
        value = torch.linalg.vector_norm(prediction - target) / denominator
    result = float(value.cpu())
    if not np.isfinite(result):
        raise ValueError("Gate 5 evaluation alignment error is non-finite")
    return result


def _precontact_locality_mask(candidates: Sequence[ProbeCandidate]) -> torch.Tensor:
    """Conservative one-step no-contact mask fixed from Blocket state geometry."""

    selected: list[bool] = []
    for candidate in candidates:
        environment = candidate.environment
        state = environment.state
        config = environment.config
        distance = float(np.linalg.norm(state.puck_position - state.player_position))
        relative_speed = float(
            np.linalg.norm(state.puck_velocity - state.player_velocity)
        )
        # The bound includes one full maximum-thrust acceleration increment.
        one_step_closing_bound = float(config.dt) * (
            relative_speed + float(config.player_acceleration) * float(config.dt)
        )
        separation = float(config.player_radius + config.puck_radius)
        selected.append(
            state.reset_timer == 0
            and state.last_event not in {"impact", "goal"}
            and distance > separation + one_step_closing_bound
        )
    mask = torch.tensor(selected, dtype=torch.bool)
    if (
        mask.shape != (REGISTERED_GATE5_EVALUATION_SAMPLES,)
        or int(mask.sum()) < REGISTERED_GATE5_MIN_LOCALITY_SAMPLES
    ):
        raise ValueError("Gate 5 has fewer than 64 conservative pre-contact states")
    return mask


@torch.no_grad()
def _encode_contexts(
    model: nn.Module,
    candidates: Sequence[ProbeCandidate],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    contexts = torch.stack([candidate.context for candidate in candidates])
    encoded = []
    for start in range(0, contexts.shape[0], batch_size):
        encoded.append(model.encode(contexts[start : start + batch_size].to(device).long()))
    return torch.cat(encoded).detach()


def _latent_pulse_responses(
    model: nn.Module,
    states: torch.Tensor,
    *,
    port_size: int,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    chunks: dict[int, list[torch.Tensor]] = {horizon: [] for horizon in REGISTERED_GATE5_HORIZONS}
    for start in range(0, states.shape[0], batch_size):
        with torch.enable_grad():
            response = direct_dynamics_pulse_responses(
                model.step,
                states[start : start + batch_size],
                port_size,
                horizons=REGISTERED_GATE5_HORIZONS,
                create_graph=False,
            )
        for horizon in REGISTERED_GATE5_HORIZONS:
            chunks[horizon].append(response.jacobians[horizon].detach())
    return {
        horizon: torch.cat(chunks[horizon], dim=0).detach()
        for horizon in REGISTERED_GATE5_HORIZONS
    }


def _gate5_groups(
    system_name: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if system_name == "pendulum":
        return (0,), (1,), (), ()
    if system_name == "blocket":
        return (0, 1, 2, 3), (4, 5, 6, 7), (4, 5), (6, 7)
    raise KeyError(system_name)


def _gate5_evidence_sha256(evidence: "Gate5Evidence") -> str:
    primitive = {
        "schema": "experiment_f_gate5_evidence_v1",
        "system": evidence.system_name,
        "protocol": asdict(evidence.protocol),
        "modelSha256": evidence.model_sha256,
        "checkpointSha256": evidence.checkpoint_sha256,
        "backboneSha256": evidence.backbone_sha256,
        "producerSealSha256": evidence.producer_seal_sha256,
        "coordinateSchema": list(evidence.coordinate_schema),
        "alignmentIdentifiers": list(evidence.alignment_identifiers),
        "evaluationIdentifiers": list(evidence.evaluation_identifiers),
        "alignmentContextSha256": evidence.alignment_context_sha256,
        "evaluationContextSha256": evidence.evaluation_context_sha256,
        "alignmentLatentSha256": _tensor_sha256(evidence.alignment_latent),
        "alignmentCoordinatesSha256": _tensor_sha256(evidence.alignment_coordinates),
        "evaluationLatentSha256": _tensor_sha256(evidence.evaluation_latent),
        "evaluationCoordinatesSha256": _tensor_sha256(evidence.evaluation_coordinates),
        "coordinateMeanSha256": _tensor_sha256(evidence.coordinate_mean),
        "coordinateScaleSha256": _tensor_sha256(evidence.coordinate_scale),
        "alignmentWeightSha256": _tensor_sha256(evidence.alignment.weight),
        "alignmentBiasSha256": _tensor_sha256(evidence.alignment.bias),
        "alignmentEvaluationNormalizedError": (
            evidence.alignment_evaluation_normalized_error
        ),
        "responses": {
            str(horizon): _tensor_sha256(evidence.latent_responses[horizon])
            for horizon in REGISTERED_GATE5_HORIZONS
        },
        "groups": {
            "configuration": list(evidence.configuration_indices),
            "momentum": list(evidence.momentum_indices),
            "actuatedMomentum": list(evidence.actuated_momentum_indices),
            "nonactuatedMomentum": list(evidence.nonactuated_momentum_indices),
        },
        "localityMaskSha256": (
            None
            if evidence.locality_sample_mask is None
            else _tensor_sha256(evidence.locality_sample_mask)
        ),
        "neuralHashBefore": evidence.neural_hash_before,
        "neuralHashAfter": evidence.neural_hash_after,
        "gradientUpdates": evidence.gradient_updates,
        "physicalCommandsRead": evidence.physical_commands_read,
        "simulatorStateReadPhase": evidence.simulator_state_read_phase,
    }
    return _canonical_json_sha256(primitive)


@dataclass(frozen=True)
class Gate5Evidence:
    """Detached typed evidence. It is never accepted as a bare pass boolean."""

    system_name: str
    protocol: Gate5CollectionConfig
    model: nn.Module = field(repr=False, compare=False)
    model_sha256: str
    checkpoint_sha256: str
    backbone_sha256: str
    producer_seal_sha256: str
    coordinate_schema: tuple[str, ...]
    alignment_identifiers: tuple[str, ...]
    evaluation_identifiers: tuple[str, ...]
    alignment_context_sha256: str
    evaluation_context_sha256: str
    alignment_latent: torch.Tensor = field(repr=False, compare=False)
    alignment_coordinates: torch.Tensor = field(repr=False, compare=False)
    evaluation_latent: torch.Tensor = field(repr=False, compare=False)
    evaluation_coordinates: torch.Tensor = field(repr=False, compare=False)
    coordinate_mean: torch.Tensor = field(repr=False, compare=False)
    coordinate_scale: torch.Tensor = field(repr=False, compare=False)
    alignment: AffineAuditAlignment
    alignment_evaluation_normalized_error: float
    latent_responses: Mapping[int, torch.Tensor] = field(repr=False, compare=False)
    configuration_indices: tuple[int, ...]
    momentum_indices: tuple[int, ...]
    actuated_momentum_indices: tuple[int, ...]
    nonactuated_momentum_indices: tuple[int, ...]
    locality_sample_mask: torch.Tensor | None = field(repr=False, compare=False)
    neural_hash_before: str
    neural_hash_after: str
    evidence_sha256: str
    gradient_updates: int = 0
    physical_commands_read: int = 0
    simulator_state_read_phase: str = "postfreeze_gate5_affine_audit_only"

    def __post_init__(self) -> None:
        hashes = (
            self.model_sha256,
            self.checkpoint_sha256,
            self.backbone_sha256,
            self.producer_seal_sha256,
            self.neural_hash_before,
            self.neural_hash_after,
            self.evidence_sha256,
            self.alignment_context_sha256,
            self.evaluation_context_sha256,
        )
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
            raise ValueError("Gate 5 contains a non-canonical SHA-256")
        if self.coordinate_schema != _coordinate_schema(self.system_name):
            raise ValueError("Gate 5 coordinate schema is not preregistered")
        if (
            len(self.alignment_identifiers) != self.protocol.alignment_samples
            or len(self.evaluation_identifiers) != self.protocol.evaluation_samples
            or len(set(self.alignment_identifiers)) != len(self.alignment_identifiers)
            or len(set(self.evaluation_identifiers)) != len(self.evaluation_identifiers)
            or set(self.alignment_identifiers) & set(self.evaluation_identifiers)
        ):
            raise ValueError("Gate 5 candidate identifiers are incomplete or overlap")
        state_size = len(self.coordinate_schema)
        if (
            self.alignment_latent.shape
            != (self.protocol.alignment_samples, state_size)
            or self.alignment_coordinates.shape
            != (self.protocol.alignment_samples, state_size)
            or self.evaluation_latent.shape
            != (self.protocol.evaluation_samples, state_size)
            or self.evaluation_coordinates.shape
            != (self.protocol.evaluation_samples, state_size)
            or self.coordinate_mean.shape != (state_size,)
            or self.coordinate_scale.shape != (state_size,)
            or bool((self.coordinate_scale <= 0.0).any())
        ):
            raise ValueError("Gate 5 affine coordinate tensor schema is invalid")
        tensors = (
            self.alignment_latent,
            self.alignment_coordinates,
            self.evaluation_latent,
            self.evaluation_coordinates,
            self.coordinate_mean,
            self.coordinate_scale,
            *self.latent_responses.values(),
        )
        if any(
            not tensor.is_floating_point()
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not bool(torch.isfinite(tensor).all())
            for tensor in tensors
        ):
            raise ValueError("Gate 5 evidence must be finite detached floating point")
        if not np.isfinite(self.alignment_evaluation_normalized_error):
            raise ValueError("Gate 5 evaluation alignment error is non-finite")
        if (
            self.alignment_evaluation_normalized_error < 0.0
            or self.alignment.weight.shape != (state_size, state_size)
            or self.alignment.bias.shape != (state_size,)
            or self.alignment.weight.requires_grad
            or self.alignment.bias.requires_grad
            or self.alignment.weight.grad_fn is not None
            or self.alignment.bias.grad_fn is not None
            or not bool(torch.isfinite(self.alignment.weight).all())
            or not bool(torch.isfinite(self.alignment.bias).all())
            or not np.isfinite(self.alignment.normalized_fit_error)
            or self.alignment.normalized_fit_error < 0.0
            or self.model.training
            or any(parameter.requires_grad for parameter in self.model.parameters())
        ):
            raise ValueError("Gate 5 affine chart/model evidence is invalid")
        if tuple(sorted(self.latent_responses)) != REGISTERED_GATE5_HORIZONS:
            raise ValueError("Gate 5 response horizons changed")
        expected_response_shape = (
            self.protocol.evaluation_samples,
            state_size,
            len(self.configuration_indices) if self.system_name == "pendulum" else 2,
        )
        if any(
            tuple(response.shape) != expected_response_shape
            for response in self.latent_responses.values()
        ):
            raise ValueError("Gate 5 response tensor schema is invalid")
        expected_groups = _gate5_groups(self.system_name)
        if (
            self.configuration_indices,
            self.momentum_indices,
            self.actuated_momentum_indices,
            self.nonactuated_momentum_indices,
        ) != expected_groups:
            raise ValueError("Gate 5 physical coordinate groups changed")
        if self.system_name == "pendulum" and self.locality_sample_mask is not None:
            raise ValueError("pendulum Gate 5 must not contain a locality mask")
        if self.system_name == "blocket" and (
            type(self.locality_sample_mask) is not torch.Tensor
            or self.locality_sample_mask.dtype != torch.bool
            or tuple(self.locality_sample_mask.shape)
            != (self.protocol.evaluation_samples,)
            or self.locality_sample_mask.requires_grad
            or int(self.locality_sample_mask.sum())
            < REGISTERED_GATE5_MIN_LOCALITY_SAMPLES
        ):
            raise ValueError("blocket Gate 5 locality mask is invalid")
        if (
            self.gradient_updates != 0
            or self.physical_commands_read != 0
            or self.simulator_state_read_phase
            != "postfreeze_gate5_affine_audit_only"
            or self.neural_hash_before != self.neural_hash_after
            or self.model_sha256 != self.neural_hash_before
            or self.alignment.model_sha256 != self.model_sha256
            or self.alignment.sample_count != self.protocol.alignment_samples
        ):
            raise ValueError("Gate 5 post-freeze firewall/provenance is invalid")
        if _gate5_evidence_sha256(self) != self.evidence_sha256:
            raise ValueError("Gate 5 evidence SHA-256 does not match its tensors")


@dataclass(frozen=True)
class Gate5Artifact:
    """JSON-safe gate outcome whose evidence is reproducible from sealed inputs."""

    system_name: str
    protocol: Gate5CollectionConfig
    result: GateAuditResult
    evidence_sha256: str
    model_sha256: str
    checkpoint_sha256: str
    alignment_context_sha256: str
    evaluation_context_sha256: str
    alignment_identifiers_sha256: str
    evaluation_identifiers_sha256: str
    alignment_normalized_fit_error: float
    alignment_evaluation_normalized_error: float
    locality_samples: int
    neural_hashes_unchanged: bool

    def core_dict(self) -> dict[str, Any]:
        return {
            "kind": "experiment_f_gate5_force_port_v1",
            "gate": 5,
            "system": self.system_name,
            "protocol": asdict(self.protocol),
            "result": self.result.to_dict(),
            "evidenceSha256": self.evidence_sha256,
            "modelSha256": self.model_sha256,
            "checkpointSha256": self.checkpoint_sha256,
            "alignmentContextSha256": self.alignment_context_sha256,
            "evaluationContextSha256": self.evaluation_context_sha256,
            "alignmentIdentifiersSha256": self.alignment_identifiers_sha256,
            "evaluationIdentifiersSha256": self.evaluation_identifiers_sha256,
            "alignmentNormalizedFitError": self.alignment_normalized_fit_error,
            "alignmentEvaluationNormalizedError": (
                self.alignment_evaluation_normalized_error
            ),
            "localitySamples": self.locality_samples,
            "neuralHashesUnchanged": self.neural_hashes_unchanged,
            "gradientUpdates": 0,
            "physicalCommandsRead": 0,
            "simulatorStateReadPhase": "postfreeze_gate5_affine_audit_only",
        }

    def to_dict(self) -> dict[str, Any]:
        core = self.core_dict()
        return {**core, "artifactSha256": _canonical_json_sha256(core)}


def verify_gate5_artifact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify exact JSON schema/digest before a finalizer compares a rerun."""

    core_keys = {
        "kind",
        "gate",
        "system",
        "protocol",
        "result",
        "evidenceSha256",
        "modelSha256",
        "checkpointSha256",
        "alignmentContextSha256",
        "evaluationContextSha256",
        "alignmentIdentifiersSha256",
        "evaluationIdentifiersSha256",
        "alignmentNormalizedFitError",
        "alignmentEvaluationNormalizedError",
        "localitySamples",
        "neuralHashesUnchanged",
        "gradientUpdates",
        "physicalCommandsRead",
        "simulatorStateReadPhase",
    }
    if type(payload) is not dict or set(payload) != core_keys | {"artifactSha256"}:
        raise ValueError("Gate 5 artifact schema is not exact")
    core = {name: payload[name] for name in core_keys}
    result = core["result"]
    if (
        core["kind"] != "experiment_f_gate5_force_port_v1"
        or core["gate"] != 5
        or core["gradientUpdates"] != 0
        or core["physicalCommandsRead"] != 0
        or core["simulatorStateReadPhase"]
        != "postfreeze_gate5_affine_audit_only"
        or type(result) is not dict
        or set(result)
        != {"gate", "auditable", "passed", "checks", "metrics", "failures"}
        or result["gate"] != 5
        or _canonical_json_sha256(core) != payload["artifactSha256"]
    ):
        raise ValueError("Gate 5 artifact provenance/digest is invalid")
    return dict(payload)


def collect_gate5_evidence(
    loaded: "LoadedPostFreezeSystem",
    config: Gate5CollectionConfig = Gate5CollectionConfig(),
) -> Gate5Evidence:
    """Collect the complete Gate 5 evidence from a fully frozen system."""

    loaded.assert_frozen_and_unchanged()
    model = loaded.full.bundle.model
    model.eval().requires_grad_(False)
    model_hash = module_tensor_hash(model)
    state_size = int(model.core.config.state_size)
    port_size = int(model.core.config.port_size)
    schema = _coordinate_schema(loaded.system_name)
    if state_size != len(schema):
        raise ValueError("Gate 5 physical audit chart and latent dimensions differ")
    image_size = int(loaded.backbone.config.image_size)
    history_frames = int(loaded.backbone.config.history_frames)
    alignment_candidates = make_builtin_probe_candidates(
        loaded.system_name,
        history_frames=history_frames,
        count=config.alignment_samples,
        seed=config.alignment_seed,
        image_size=image_size,
    )
    evaluation_candidates = make_builtin_probe_candidates(
        loaded.system_name,
        history_frames=history_frames,
        count=config.evaluation_samples,
        seed=config.evaluation_seed,
        image_size=image_size,
    )
    alignment_ids = tuple(item.identifier for item in alignment_candidates)
    evaluation_ids = tuple(item.identifier for item in evaluation_candidates)
    if set(alignment_ids) & set(evaluation_ids):
        raise AssertionError("Gate 5 alignment and evaluation candidates overlap")
    alignment_contexts = {
        _tensor_sha256(candidate.context) for candidate in alignment_candidates
    }
    evaluation_contexts = {
        _tensor_sha256(candidate.context) for candidate in evaluation_candidates
    }
    if alignment_contexts & evaluation_contexts:
        raise ValueError("Gate 5 alignment and evaluation pixel contexts overlap")

    alignment_latent = _encode_contexts(
        model,
        alignment_candidates,
        batch_size=config.batch_size,
        device=loaded.device,
    )
    raw_coordinates = _raw_audit_coordinates(
        alignment_candidates, loaded.system_name
    )
    standardized, coordinate_mean, coordinate_scale = (
        _standardize_alignment_coordinates(raw_coordinates)
    )
    standardized = standardized.to(device=loaded.device, dtype=alignment_latent.dtype)
    alignment = fit_postfreeze_affine_audit_alignment(
        model,
        alignment_latent,
        standardized,
        ridge=config.ridge,
    )

    evaluation_latent = _encode_contexts(
        model,
        evaluation_candidates,
        batch_size=config.batch_size,
        device=loaded.device,
    )
    raw_evaluation_coordinates = _raw_audit_coordinates(
        evaluation_candidates, loaded.system_name
    )
    evaluation_coordinates = (
        (raw_evaluation_coordinates - coordinate_mean) / coordinate_scale
    ).detach().to(device=loaded.device, dtype=evaluation_latent.dtype)
    alignment_evaluation_error = _affine_evaluation_error(
        alignment,
        evaluation_latent,
        evaluation_coordinates,
    )
    responses = _latent_pulse_responses(
        model,
        evaluation_latent,
        port_size=port_size,
        batch_size=config.batch_size,
    )
    locality_mask = (
        _precontact_locality_mask(evaluation_candidates).to(loaded.device)
        if loaded.system_name == "blocket"
        else None
    )
    groups = _gate5_groups(loaded.system_name)
    loaded.assert_frozen_and_unchanged()
    neural_hash_after = module_tensor_hash(model)

    values: dict[str, Any] = {
        "system_name": loaded.system_name,
        "protocol": config,
        "model": model,
        "model_sha256": model_hash,
        "checkpoint_sha256": loaded.full.checkpoint_sha256,
        "backbone_sha256": loaded.backbone_hash,
        "producer_seal_sha256": loaded.producer_seal_sha256,
        "coordinate_schema": schema,
        "alignment_identifiers": alignment_ids,
        "evaluation_identifiers": evaluation_ids,
        "alignment_context_sha256": _identifier_context_sha256(
            alignment_candidates
        ),
        "evaluation_context_sha256": _identifier_context_sha256(
            evaluation_candidates
        ),
        "alignment_latent": alignment_latent.detach(),
        "alignment_coordinates": standardized.detach(),
        "evaluation_latent": evaluation_latent.detach(),
        "evaluation_coordinates": evaluation_coordinates.detach(),
        "coordinate_mean": coordinate_mean.detach().to(loaded.device),
        "coordinate_scale": coordinate_scale.detach().to(loaded.device),
        "alignment": alignment,
        "alignment_evaluation_normalized_error": alignment_evaluation_error,
        "latent_responses": responses,
        "configuration_indices": groups[0],
        "momentum_indices": groups[1],
        "actuated_momentum_indices": groups[2],
        "nonactuated_momentum_indices": groups[3],
        "locality_sample_mask": locality_mask,
        "neural_hash_before": model_hash,
        "neural_hash_after": neural_hash_after,
        "evidence_sha256": "0" * 64,
        "gradient_updates": 0,
        "physical_commands_read": 0,
        "simulator_state_read_phase": "postfreeze_gate5_affine_audit_only",
    }
    # Compute the digest over the final immutable fields before invoking the
    # fail-closed dataclass constructor.
    values["evidence_sha256"] = _gate5_evidence_sha256(
        SimpleNamespace(**values)  # type: ignore[arg-type]
    )
    return Gate5Evidence(**values)


def audit_gate5_evidence(
    evidence: Gate5Evidence,
    loaded: "LoadedPostFreezeSystem",
    thresholds: ForcePortThresholds = ForcePortThresholds(),
) -> Gate5Artifact:
    """Authenticate against sealed inputs, then run the Gate 5 decision rule."""

    if thresholds != ForcePortThresholds():
        raise ValueError("Gate 5 thresholds are preregistered and cannot be changed")
    loaded.assert_frozen_and_unchanged()
    loaded_model = loaded.full.bundle.model
    if (
        evidence.model is not loaded_model
        or evidence.system_name != loaded.system_name
        or evidence.checkpoint_sha256 != loaded.full.checkpoint_sha256
        or evidence.backbone_sha256 != loaded.backbone_hash
        or evidence.producer_seal_sha256 != loaded.producer_seal_sha256
        or evidence.model_sha256 != module_tensor_hash(loaded_model)
    ):
        raise ValueError("Gate 5 evidence is not anchored to the loaded sealed system")
    # Regenerate both preregistered pools and all derivatives from the sealed
    # checkpoint.  A well-formed synthetic Gate5Evidence is therefore not an
    # authentication mechanism and cannot be promoted to an artifact.
    authenticated = collect_gate5_evidence(loaded, evidence.protocol)
    if authenticated.evidence_sha256 != evidence.evidence_sha256:
        raise ValueError("Gate 5 evidence differs from the authenticated rerun")
    if module_tensor_hash(evidence.model) != evidence.model_sha256:
        raise ValueError("Gate 5 model changed after evidence collection")
    # Refit the closed-form chart from the stored detached evidence so a caller
    # cannot inject an arbitrary alignment object.
    expected_alignment = fit_postfreeze_affine_audit_alignment(
        evidence.model,
        evidence.alignment_latent,
        evidence.alignment_coordinates,
        ridge=evidence.protocol.ridge,
    )
    if (
        not torch.equal(expected_alignment.weight, evidence.alignment.weight)
        or not torch.equal(expected_alignment.bias, evidence.alignment.bias)
        or expected_alignment.normalized_fit_error
        != evidence.alignment.normalized_fit_error
    ):
        raise ValueError("Gate 5 affine alignment is not the registered closed-form fit")
    evaluation_error = _affine_evaluation_error(
        evidence.alignment,
        evidence.evaluation_latent,
        evidence.evaluation_coordinates,
    )
    if evaluation_error != evidence.alignment_evaluation_normalized_error:
        raise ValueError("Gate 5 held-out affine error is not reproducible")
    # Reconstructing re-runs every evidence invariant, including its digest.
    Gate5Evidence(**dict(evidence.__dict__))
    require_locality = evidence.system_name == "blocket"
    base_result = audit_force_port_signature(
        evidence.model,
        evidence.alignment,
        evidence.latent_responses,
        configuration_indices=evidence.configuration_indices,
        momentum_indices=evidence.momentum_indices,
        actuated_momentum_indices=(
            evidence.actuated_momentum_indices if require_locality else None
        ),
        nonactuated_momentum_indices=(
            evidence.nonactuated_momentum_indices if require_locality else None
        ),
        locality_sample_mask=evidence.locality_sample_mask,
        require_locality=require_locality,
        thresholds=thresholds,
    )
    chart_generalizes = (
        evidence.alignment_evaluation_normalized_error
        <= thresholds.maximum_affine_alignment_normalized_fit_error
    )
    checks = {**base_result.checks, "heldout_affine_audit_chart_quality": chart_generalizes}
    metrics = {
        **base_result.metrics,
        "heldout_affine_alignment_normalized_error": (
            evidence.alignment_evaluation_normalized_error
        ),
    }
    failures = base_result.failures + (
        ()
        if chart_generalizes
        else ("failed check: heldout_affine_audit_chart_quality",)
    )
    result = GateAuditResult(
        gate=5,
        auditable=base_result.auditable,
        passed=base_result.auditable and all(checks.values()),
        checks=checks,
        metrics=metrics,
        failures=failures,
    )
    return Gate5Artifact(
        system_name=evidence.system_name,
        protocol=evidence.protocol,
        result=result,
        evidence_sha256=evidence.evidence_sha256,
        model_sha256=evidence.model_sha256,
        checkpoint_sha256=evidence.checkpoint_sha256,
        alignment_context_sha256=evidence.alignment_context_sha256,
        evaluation_context_sha256=evidence.evaluation_context_sha256,
        alignment_identifiers_sha256=hashlib.sha256(
            "\n".join(evidence.alignment_identifiers).encode("utf-8")
        ).hexdigest(),
        evaluation_identifiers_sha256=hashlib.sha256(
            "\n".join(evidence.evaluation_identifiers).encode("utf-8")
        ).hexdigest(),
        alignment_normalized_fit_error=evidence.alignment.normalized_fit_error,
        alignment_evaluation_normalized_error=(
            evidence.alignment_evaluation_normalized_error
        ),
        locality_samples=(
            0
            if evidence.locality_sample_mask is None
            else int(evidence.locality_sample_mask.sum().cpu())
        ),
        neural_hashes_unchanged=(
            evidence.neural_hash_before == evidence.neural_hash_after
        ),
    )


def run_gate5_postfreeze(
    loaded: "LoadedPostFreezeSystem",
    config: Gate5CollectionConfig = Gate5CollectionConfig(),
) -> tuple[Gate5Artifact, Gate5Evidence]:
    evidence = collect_gate5_evidence(loaded, config)
    return audit_gate5_evidence(evidence, loaded), evidence


__all__ = [
    "Gate5Artifact",
    "Gate5CollectionConfig",
    "Gate5Evidence",
    "REGISTERED_GATE5_ALIGNMENT_SAMPLES",
    "REGISTERED_GATE5_EVALUATION_SAMPLES",
    "REGISTERED_GATE5_HORIZONS",
    "REGISTERED_GATE5_MIN_LOCALITY_SAMPLES",
    "audit_gate5_evidence",
    "collect_gate5_evidence",
    "run_gate5_postfreeze",
    "verify_gate5_artifact_payload",
]
