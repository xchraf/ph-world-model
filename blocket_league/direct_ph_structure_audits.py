"""Locked, post-training audits for the direct visual Poisson pH model.

The functions in this module are deliberately *audits*, not losses.  They do
not return differentiable tensors, never call ``backward``, and reject any
post-freeze evidence tensor that is still attached to an autograd graph.  This
keeps the simulator-facing affine audit alignment and every held-out
diagnostic outside all gradient-based phases.

Three helpers implement the preregistered decision logic verbatim:

``audit_gate_1``
    Pixels-only information firewall and immutable video backbone.
``audit_gate_3``
    Poisson, passivity, and numerical-power identities.
``audit_gate_4``
    Multi-horizon internal Jacobian-port evidence.

Missing, non-finite, rank-deficient, or provenance-free evidence is an
*unauditable failure*.  No check is silently skipped and an unauditable gate
can never pass.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from .direct_poisson_ph import DirectPoissonPortHamiltonian
from .direct_visual_poisson_ph import DirectVisualPoissonPH
from .passive_jacobian_ph_model import module_tensor_hash


_FORBIDDEN_SCHEMA_TERMS = (
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


@dataclass(frozen=True)
class GateAuditResult:
    """Serializable result of one locked gate.

    ``passed`` is always false when ``auditable`` is false.  ``checks`` keeps
    each conjunct visible so a top-level experiment cannot accidentally turn
    a partial pass into a positive outcome.
    """

    gate: int
    auditable: bool
    passed: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, float | int | str]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "auditable": self.auditable,
            "passed": self.passed,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "failures": list(self.failures),
        }


def _gate_result(
    gate: int,
    checks: Mapping[str, bool],
    metrics: Mapping[str, float | int | str],
    *,
    unauditable_reasons: Sequence[str] = (),
) -> GateAuditResult:
    failed_checks = tuple(name for name, passed in checks.items() if not passed)
    unauditable = tuple(str(reason) for reason in unauditable_reasons)
    failures = unauditable + tuple(f"failed check: {name}" for name in failed_checks)
    auditable = not unauditable
    return GateAuditResult(
        gate=gate,
        auditable=auditable,
        passed=auditable and not failed_checks,
        checks=dict(checks),
        metrics=dict(metrics),
        failures=failures,
    )


def _plain_schema(schema: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in schema))


def _schema_contains_forbidden_value(schemas: Sequence[Sequence[str]]) -> bool:
    for schema in schemas:
        for key in schema:
            normalized = str(key).strip().lower()
            if any(term in normalized for term in _FORBIDDEN_SCHEMA_TERMS):
                return True
    return False


@dataclass(frozen=True)
class FirewallAuditEvidence:
    """Sealed and observed evidence required by Gate 1.

    Gradient schemas normally contain ``("pixels",)`` for backbone
    pretraining and ``("pixelContexts", "frames")`` for direct fitting.  The
    audit does not hard-code those spellings: it checks the exact sealed set
    and independently rejects simulator-bearing key names.
    """

    sealed_archive_sha256: str | None = None
    observed_archive_sha256: str | None = None
    sealed_source_tree_sha256: str | None = None
    observed_source_tree_sha256: str | None = None
    sealed_source_schema: tuple[str, ...] | None = None
    observed_source_schema: tuple[str, ...] | None = None
    sealed_gradient_schemas: tuple[tuple[str, ...], ...] | None = None
    observed_gradient_schemas: tuple[tuple[str, ...], ...] | None = None
    sealed_backbone_hash: str | None = None
    observed_backbone_hashes: tuple[str, ...] | None = None
    forbidden_gradient_read_count: int | None = None
    backbone_in_optimizer: bool | None = None
    nonanalytic_command_read_count: int | None = None
    runtime_trace_event_count: int | None = None
    runtime_gradient_batch_count: int | None = None
    runtime_file_read_count: int | None = None
    runtime_stage_boundary_count: int | None = None
    runtime_mount_inventory_count: int | None = None
    forbidden_training_mount_count: int | None = None
    sealed_learner_bundle_sha256: str | None = None
    observed_learner_bundle_sha256: str | None = None
    expected_learner_manifest_count: int | None = None
    observed_learner_manifest_count: int | None = None
    expected_learner_source_inventory_count: int | None = None
    observed_learner_source_inventory_count: int | None = None
    forbidden_learner_source_file_count: int | None = None
    learner_source_file_mismatch_count: int | None = None
    expected_learner_cache_inventory_count: int | None = None
    observed_learner_cache_inventory_count: int | None = None
    learner_cache_python_code_file_count: int | None = None
    learner_cache_unsafe_path_count: int | None = None


def audit_gate_1(
    model: DirectVisualPoissonPH,
    evidence: FirewallAuditEvidence,
) -> GateAuditResult:
    """Audit the pixels-only firewall and exact frozen-backbone contract."""

    missing: list[str] = []
    for name in (
        "sealed_archive_sha256",
        "observed_archive_sha256",
        "sealed_source_tree_sha256",
        "observed_source_tree_sha256",
        "sealed_source_schema",
        "observed_source_schema",
        "sealed_gradient_schemas",
        "observed_gradient_schemas",
        "sealed_backbone_hash",
        "observed_backbone_hashes",
        "forbidden_gradient_read_count",
        "backbone_in_optimizer",
        "nonanalytic_command_read_count",
        "runtime_trace_event_count",
        "runtime_gradient_batch_count",
        "runtime_file_read_count",
        "runtime_stage_boundary_count",
        "runtime_mount_inventory_count",
        "forbidden_training_mount_count",
        "sealed_learner_bundle_sha256",
        "observed_learner_bundle_sha256",
        "expected_learner_manifest_count",
        "observed_learner_manifest_count",
        "expected_learner_source_inventory_count",
        "observed_learner_source_inventory_count",
        "forbidden_learner_source_file_count",
        "learner_source_file_mismatch_count",
        "expected_learner_cache_inventory_count",
        "observed_learner_cache_inventory_count",
        "learner_cache_python_code_file_count",
        "learner_cache_unsafe_path_count",
    ):
        if getattr(evidence, name) is None:
            missing.append(f"missing Gate 1 evidence: {name}")

    metrics: dict[str, float | int | str] = {}
    checks: dict[str, bool] = {}
    if missing:
        return _gate_result(1, checks, metrics, unauditable_reasons=missing)

    # The None cases are exhausted above.  Local aliases keep the checks
    # readable without weakening the runtime validation.
    sealed_archive = str(evidence.sealed_archive_sha256)
    observed_archive = str(evidence.observed_archive_sha256)
    sealed_source_tree = str(evidence.sealed_source_tree_sha256)
    observed_source_tree = str(evidence.observed_source_tree_sha256)
    sealed_source = _plain_schema(evidence.sealed_source_schema or ())
    observed_source = _plain_schema(evidence.observed_source_schema or ())
    sealed_gradient = tuple(
        sorted(_plain_schema(schema) for schema in evidence.sealed_gradient_schemas or ())
    )
    observed_gradient = tuple(
        sorted(_plain_schema(schema) for schema in evidence.observed_gradient_schemas or ())
    )
    sealed_backbone = str(evidence.sealed_backbone_hash)
    observed_hashes = tuple(str(value) for value in evidence.observed_backbone_hashes or ())
    sealed_learner_bundle = str(evidence.sealed_learner_bundle_sha256)
    observed_learner_bundle = str(evidence.observed_learner_bundle_sha256)

    if not sealed_archive or not observed_archive:
        missing.append("archive SHA-256 evidence is blank")
    if not sealed_source_tree or not observed_source_tree:
        missing.append("source-tree SHA-256 evidence is blank")
    if not sealed_source or not observed_source:
        missing.append("source-schema evidence is empty")
    if not sealed_gradient or not observed_gradient:
        missing.append("gradient-schema evidence is empty")
    if not sealed_backbone or not observed_hashes:
        missing.append("backbone-hash evidence is empty")
    if not sealed_learner_bundle or not observed_learner_bundle:
        missing.append("learner-bundle SHA-256 evidence is blank")
    if missing:
        return _gate_result(1, checks, metrics, unauditable_reasons=missing)

    backbone = model.encoder.backbone
    current_hash = module_tensor_hash(backbone)
    encoder_seal = model.encoder.sealed_backbone_hash
    forbidden_schema = _schema_contains_forbidden_value(
        (*observed_gradient, observed_source)
    )
    mutable_running_statistics = any(
        getattr(module, "track_running_stats", False)
        and (
            getattr(module, "running_mean", None) is not None
            or getattr(module, "running_var", None) is not None
        )
        for module in backbone.modules()
    )

    checks.update(
        {
            "archive_hash_exact": observed_archive == sealed_archive,
            "source_tree_hash_exact": observed_source_tree == sealed_source_tree,
            "source_schema_exact": observed_source == sealed_source,
            "gradient_schemas_exact": observed_gradient == sealed_gradient,
            "gradient_schema_has_no_forbidden_key": not forbidden_schema,
            "zero_forbidden_gradient_reads": evidence.forbidden_gradient_read_count == 0,
            "backbone_absent_from_optimizer": evidence.backbone_in_optimizer is False,
            "commands_read_only_by_analytic_grounding": (
                evidence.nonanalytic_command_read_count == 0
            ),
            "runtime_trace_is_nonempty": type(evidence.runtime_trace_event_count)
            is int
            and evidence.runtime_trace_event_count > 0,
            "runtime_gradient_batches_observed": type(
                evidence.runtime_gradient_batch_count
            )
            is int
            and evidence.runtime_gradient_batch_count > 0,
            "runtime_file_reads_observed": type(evidence.runtime_file_read_count)
            is int
            and evidence.runtime_file_read_count > 0,
            "runtime_stage_boundaries_observed": type(
                evidence.runtime_stage_boundary_count
            )
            is int
            and evidence.runtime_stage_boundary_count > 0,
            "runtime_mount_inventory_observed": type(
                evidence.runtime_mount_inventory_count
            )
            is int
            and evidence.runtime_mount_inventory_count > 0,
            "no_forbidden_training_mount": evidence.forbidden_training_mount_count
            == 0,
            "learner_bundle_hash_exact": (
                observed_learner_bundle == sealed_learner_bundle
            ),
            "all_learner_manifest_reads_observed": (
                evidence.observed_learner_manifest_count
                == evidence.expected_learner_manifest_count
                and type(evidence.expected_learner_manifest_count) is int
                and evidence.expected_learner_manifest_count > 0
            ),
            "all_learner_source_inventories_observed": (
                evidence.observed_learner_source_inventory_count
                == evidence.expected_learner_source_inventory_count
                and type(evidence.expected_learner_source_inventory_count) is int
                and evidence.expected_learner_source_inventory_count > 0
            ),
            "zero_forbidden_learner_source_files": (
                evidence.forbidden_learner_source_file_count == 0
            ),
            "learner_source_files_match_manifest": (
                evidence.learner_source_file_mismatch_count == 0
            ),
            "all_learner_cache_inventories_observed": (
                evidence.observed_learner_cache_inventory_count
                == evidence.expected_learner_cache_inventory_count
                and type(evidence.expected_learner_cache_inventory_count) is int
                and evidence.expected_learner_cache_inventory_count > 0
            ),
            "learner_caches_contain_no_python_code": (
                evidence.learner_cache_python_code_file_count == 0
            ),
            "learner_caches_have_no_unsafe_paths": (
                evidence.learner_cache_unsafe_path_count == 0
            ),
            "backbone_requires_no_grad": all(
                not parameter.requires_grad for parameter in backbone.parameters()
            ),
            "backbone_is_in_eval_mode": not backbone.training,
            "backbone_has_no_mutable_running_statistics": not mutable_running_statistics,
            "encoder_seal_matches_manifest": encoder_seal == sealed_backbone,
            "current_backbone_hash_exact": current_hash == sealed_backbone,
            "all_stage_backbone_hashes_exact": all(
                value == sealed_backbone for value in observed_hashes
            ),
        }
    )
    metrics.update(
        {
            "archive_sha256": observed_archive,
            "source_tree_sha256": observed_source_tree,
            "backbone_sha256": current_hash,
            "observed_backbone_hash_count": len(observed_hashes),
            "forbidden_gradient_read_count": int(
                evidence.forbidden_gradient_read_count or 0
            ),
            "nonanalytic_command_read_count": int(
                evidence.nonanalytic_command_read_count or 0
            ),
            "runtime_trace_event_count": int(
                evidence.runtime_trace_event_count or 0
            ),
            "runtime_gradient_batch_count": int(
                evidence.runtime_gradient_batch_count or 0
            ),
            "runtime_file_read_count": int(evidence.runtime_file_read_count or 0),
            "runtime_stage_boundary_count": int(
                evidence.runtime_stage_boundary_count or 0
            ),
            "runtime_mount_inventory_count": int(
                evidence.runtime_mount_inventory_count or 0
            ),
            "forbidden_training_mount_count": int(
                evidence.forbidden_training_mount_count or 0
            ),
            "learner_bundle_sha256": observed_learner_bundle,
            "expected_learner_manifest_count": int(
                evidence.expected_learner_manifest_count or 0
            ),
            "observed_learner_manifest_count": int(
                evidence.observed_learner_manifest_count or 0
            ),
            "expected_learner_source_inventory_count": int(
                evidence.expected_learner_source_inventory_count or 0
            ),
            "observed_learner_source_inventory_count": int(
                evidence.observed_learner_source_inventory_count or 0
            ),
            "forbidden_learner_source_file_count": int(
                evidence.forbidden_learner_source_file_count or 0
            ),
            "learner_source_file_mismatch_count": int(
                evidence.learner_source_file_mismatch_count or 0
            ),
            "expected_learner_cache_inventory_count": int(
                evidence.expected_learner_cache_inventory_count or 0
            ),
            "observed_learner_cache_inventory_count": int(
                evidence.observed_learner_cache_inventory_count or 0
            ),
            "learner_cache_python_code_file_count": int(
                evidence.learner_cache_python_code_file_count or 0
            ),
            "learner_cache_unsafe_path_count": int(
                evidence.learner_cache_unsafe_path_count or 0
            ),
        }
    )
    return _gate_result(1, checks, metrics)


@dataclass(frozen=True)
class Gate3Thresholds:
    """Preregistered Gate 3 thresholds."""

    maximum_normalized_skew_defect: float = 1e-7
    maximum_normalized_jacobi_defect: float = 1e-5
    minimum_resistance_eigenvalue: float = -1e-7
    minimum_port_singular_value: float = 1e-5
    maximum_relative_continuous_power_defect: float = 1e-6
    maximum_relative_discrete_power_defect: float = 1e-5
    maximum_zero_effort_energy_increase_fraction: float = 0.001
    maximum_implicit_failure_fraction: float = 0.001
    implicit_residual_tolerance: float = 1e-8
    zero_effort_energy_tolerance: float = 1e-8
    production_maximum_relative_discrete_power_defect: float = 5e-3
    production_maximum_absolute_discrete_power_defect: float = 5e-6
    production_maximum_zero_effort_energy_increase_fraction: float = 0.001
    production_maximum_implicit_failure_fraction: float = 0.001
    production_implicit_residual_tolerance: float = 5e-5
    production_zero_effort_energy_tolerance: float = 5e-6
    production_path_relative_tolerance: float = 1e-6
    production_rollout_steps: int = 16
    minimum_states: int = 4_096
    relative_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        scalar_thresholds = tuple(
            value
            for name, value in self.__dict__.items()
            if name not in {"minimum_states", "production_rollout_steps"}
        )
        if any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in scalar_thresholds
        ):
            raise ValueError("Gate 3 thresholds must be finite numeric values")
        if self.minimum_states < 1:
            raise ValueError("minimum_states must be positive")
        if type(self.production_rollout_steps) is not int or self.production_rollout_steps < 1:
            raise ValueError("production_rollout_steps must be a positive integer")
        if self.implicit_residual_tolerance <= 0.0:
            raise ValueError("implicit_residual_tolerance must be positive")
        if self.zero_effort_energy_tolerance < 0.0:
            raise ValueError("zero_effort_energy_tolerance must be non-negative")
        if self.production_implicit_residual_tolerance <= 0.0:
            raise ValueError("production implicit residual tolerance must be positive")
        if self.production_zero_effort_energy_tolerance < 0.0:
            raise ValueError("production zero-effort energy tolerance must be non-negative")
        if self.production_path_relative_tolerance < 0.0:
            raise ValueError("production path tolerance must be non-negative")
        if self.minimum_port_singular_value < 0.0:
            raise ValueError("minimum_port_singular_value must be non-negative")
        if self.relative_epsilon <= 0.0:
            raise ValueError("relative_epsilon must be positive")


@dataclass(frozen=True)
class RK2PowerAuditResult:
    """Matched explicit-midpoint ablation on the exact Gate 3 transitions."""

    auditable: bool
    passed: bool
    sample_count: int
    metrics: Mapping[str, float | int | str]
    failures: tuple[str, ...]
    source_manifest_sha256: str = ""
    core_sha256: str = ""
    states_sha256: str = ""
    efforts_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "matched_non_structure_preserving_rk2_power_audit",
            "auditable": self.auditable,
            "passed": self.passed,
            "sampleCount": self.sample_count,
            "metrics": dict(self.metrics),
            "failures": list(self.failures),
            "sourceManifestSha256": self.source_manifest_sha256,
            "coreSha256": self.core_sha256,
            "statesSha256": self.states_sha256,
            "effortsSha256": self.efforts_sha256,
        }


@dataclass(frozen=True)
class Gate3TransitionSeal:
    """Cryptographic identity of the exact samples already used by Gate 3."""

    sample_count: int
    source_manifest_sha256: str
    core_sha256: str
    states_sha256: str
    efforts_sha256: str


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _audit_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def seal_gate3_transition_samples(
    core: DirectPoissonPortHamiltonian,
    states: torch.Tensor,
    latent_efforts: torch.Tensor,
    *,
    source_manifest_sha256: str,
) -> Gate3TransitionSeal:
    """Seal a detached Gate-3 transition collection before any RK2 audit."""

    if not _valid_sha256(source_manifest_sha256):
        raise ValueError("Gate 3 source-manifest SHA-256 is malformed")
    if states.requires_grad or states.grad_fn is not None:
        raise ValueError("Gate 3 states must be detached before sealing")
    if latent_efforts.requires_grad or latent_efforts.grad_fn is not None:
        raise ValueError("Gate 3 efforts must be detached before sealing")
    if states.ndim != 2 or latent_efforts.ndim != 2 or states.shape[0] != latent_efforts.shape[0]:
        raise ValueError("Gate 3 transition tensors have incompatible shapes")
    return Gate3TransitionSeal(
        sample_count=int(states.shape[0]),
        source_manifest_sha256=source_manifest_sha256,
        core_sha256=module_tensor_hash(core),
        states_sha256=_audit_tensor_sha256(states),
        efforts_sha256=_audit_tensor_sha256(latent_efforts),
    )


def audit_matched_rk2_power_error(
    core: DirectPoissonPortHamiltonian,
    states: torch.Tensor,
    latent_efforts: torch.Tensor,
    thresholds: Gate3Thresholds = Gate3Thresholds(),
    *,
    transition_seal: Gate3TransitionSeal,
    expected_source_manifest_sha256: str,
    expected_core_sha256: str,
    chunk_size: int = 32,
) -> RK2PowerAuditResult:
    """Quantify RK2's power defect without training or replacing the pH step.

    Explicit midpoint RK2 evaluates the *same learned continuous vector field*
    and the same step size as the registered discrete-gradient integrator.  Its
    midpoint power is compared with the actual Hamiltonian change.  The
    structure-preserving defect is recomputed on the identical samples so the
    ablation cannot quietly use an easier state/effort distribution.
    """

    failures: list[str] = []
    if type(transition_seal) is not Gate3TransitionSeal:
        failures.append("RK2 audit is missing the exact Gate 3 transition seal")
    if not _valid_sha256(expected_source_manifest_sha256):
        failures.append("RK2 expected source-manifest SHA-256 is malformed")
    if not _valid_sha256(expected_core_sha256):
        failures.append("RK2 expected frozen-core SHA-256 is malformed")
    observed_core_sha256 = module_tensor_hash(core)
    if core.training:
        failures.append("RK2 audit requires the frozen core in evaluation mode")
    if any(parameter.requires_grad for parameter in core.parameters()):
        failures.append("RK2 audit requires a fully frozen core")
    if type(chunk_size) is not int or chunk_size < 1:
        failures.append("RK2 audit chunk_size must be positive")
    if states.ndim != 2 or states.shape[-1] != core.config.state_size:
        failures.append("RK2 states have the wrong shape")
    if latent_efforts.ndim != 2 or latent_efforts.shape[-1] != core.config.port_size:
        failures.append("RK2 latent efforts have the wrong shape")
    if states.ndim == 2 and latent_efforts.ndim == 2 and states.shape[0] != latent_efforts.shape[0]:
        failures.append("RK2 state/effort sample counts differ")
    sample_count = int(states.shape[0]) if states.ndim >= 1 else 0
    if sample_count < thresholds.minimum_states:
        failures.append(
            f"RK2 audit requires at least {thresholds.minimum_states} Gate 3 states"
        )
    if states.requires_grad or states.grad_fn is not None:
        failures.append("RK2 states are attached to an autograd graph")
    if latent_efforts.requires_grad or latent_efforts.grad_fn is not None:
        failures.append("RK2 efforts are attached to an autograd graph")
    if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(latent_efforts).all()):
        failures.append("RK2 inputs contain a non-finite value")
    if not states.is_floating_point() or not latent_efforts.is_floating_point():
        failures.append("RK2 state/effort inputs must be floating point")
    observed_states_sha256 = _audit_tensor_sha256(states)
    observed_efforts_sha256 = _audit_tensor_sha256(latent_efforts)
    if type(transition_seal) is Gate3TransitionSeal:
        if transition_seal.sample_count != sample_count:
            failures.append("RK2 sample count differs from the Gate 3 transition seal")
        if sample_count != thresholds.minimum_states:
            failures.append("RK2 must use exactly the registered Gate 3 sample count")
        if transition_seal.source_manifest_sha256 != expected_source_manifest_sha256:
            failures.append("RK2 source manifest differs from the sealed held-out split")
        if transition_seal.core_sha256 != expected_core_sha256:
            failures.append("RK2 core differs from the frozen Gate 3 core")
        if transition_seal.core_sha256 != observed_core_sha256:
            failures.append("RK2 observed core hash differs from the Gate 3 transition seal")
        if transition_seal.states_sha256 != observed_states_sha256:
            failures.append("RK2 states differ from the exact Gate 3 samples")
        if transition_seal.efforts_sha256 != observed_efforts_sha256:
            failures.append("RK2 efforts differ from the exact Gate 3 samples")
    if failures:
        return RK2PowerAuditResult(
            False,
            False,
            sample_count,
            {},
            tuple(failures),
            expected_source_manifest_sha256,
            observed_core_sha256,
            observed_states_sha256,
            observed_efforts_sha256,
        )

    audit_core = copy.deepcopy(core).to(
        device=states.device, dtype=torch.float64
    ).eval().requires_grad_(False)
    rk2_relative: list[torch.Tensor] = []
    structured_relative: list[torch.Tensor] = []
    rk2_zero_increases: list[torch.Tensor] = []
    finite = True
    dt = float(audit_core.config.dt)
    try:
        for start in range(0, sample_count, chunk_size):
            stop = min(start + chunk_size, sample_count)
            state = states[start:stop].detach().double()
            effort = latent_efforts[start:stop].detach().double()
            with torch.no_grad():
                first = audit_core.vector_field(state, effort)
                stage = state + 0.5 * dt * first
                second = audit_core.vector_field(stage, effort)
                rk2_next = state + dt * second
                energy_before = audit_core.hamiltonian(state)
                energy_after = audit_core.hamiltonian(rk2_next)
                _, gradient, _, resistance, port = audit_core.components(
                    stage, create_graph=False
                )
                output = torch.einsum("...im,...i->...m", port, gradient)
                expected = dt * (
                    -torch.einsum(
                        "...i,...ij,...j->...", gradient, resistance, gradient
                    )
                    + torch.einsum("...m,...m->...", effort, output)
                )
                delta = energy_after - energy_before
                rk2_relative.append(
                    (delta - expected).abs()
                    / (delta.abs() + expected.abs()).clamp_min(
                        thresholds.relative_epsilon
                    )
                )

                structured = audit_core.audited_step(state, effort)
                structured_scale = (
                    structured.energy_delta.abs()
                    + structured.dissipated_energy.abs()
                    + structured.supplied_energy.abs()
                )
                structured_relative.append(
                    structured.balance_defect.abs()
                    / structured_scale.clamp_min(thresholds.relative_epsilon)
                )

                zero = torch.zeros_like(effort)
                zero_first = audit_core.vector_field(state, zero)
                zero_stage = state + 0.5 * dt * zero_first
                zero_next = state + dt * audit_core.vector_field(zero_stage, zero)
                zero_delta = audit_core.hamiltonian(zero_next) - energy_before
                tolerance = thresholds.zero_effort_energy_tolerance * (
                    1.0 + energy_before.abs()
                )
                rk2_zero_increases.append(zero_delta > tolerance)
                finite = finite and _all_finite(
                    (
                        first,
                        second,
                        rk2_next,
                        delta,
                        expected,
                        structured.balance_defect,
                        zero_delta,
                    )
                )
    except Exception as error:
        failures.append(f"RK2 power audit failed: {type(error).__name__}: {error}")
    finally:
        del audit_core
    if failures or not finite:
        if not finite:
            failures.append("RK2 audit produced a non-finite quantity")
        return RK2PowerAuditResult(
            False,
            False,
            sample_count,
            {},
            tuple(failures),
            expected_source_manifest_sha256,
            observed_core_sha256,
            observed_states_sha256,
            observed_efforts_sha256,
        )

    if module_tensor_hash(core) != observed_core_sha256:
        return RK2PowerAuditResult(
            False,
            False,
            sample_count,
            {},
            ("frozen core changed during the RK2 audit",),
            expected_source_manifest_sha256,
            observed_core_sha256,
            observed_states_sha256,
            observed_efforts_sha256,
        )

    rk2_values = torch.cat(rk2_relative)
    structured_values = torch.cat(structured_relative)
    rk2_maximum = float(rk2_values.amax().cpu())
    structured_maximum = float(structured_values.amax().cpu())
    metrics: dict[str, float | int | str] = {
        "sample_count": sample_count,
        "step_size": dt,
        "numeric_dtype": "float64",
        "rk2_maximum_relative_power_defect": rk2_maximum,
        "rk2_mean_relative_power_defect": float(rk2_values.mean().cpu()),
        "rk2_median_relative_power_defect": float(rk2_values.median().cpu()),
        "rk2_zero_effort_energy_increase_fraction": float(
            torch.cat(rk2_zero_increases).float().mean().cpu()
        ),
        "matched_structure_preserving_maximum_relative_power_defect": structured_maximum,
        "rk2_to_structure_preserving_maximum_defect_ratio": (
            rk2_maximum / max(structured_maximum, thresholds.relative_epsilon)
        ),
    }
    passed = (
        sample_count >= thresholds.minimum_states
        and structured_maximum <= thresholds.maximum_relative_discrete_power_defect
    )
    if not passed:
        failures.append("matched structure-preserving reference failed Gate 3")
    return RK2PowerAuditResult(
        True,
        passed,
        sample_count,
        metrics,
        tuple(failures),
        expected_source_manifest_sha256,
        observed_core_sha256,
        observed_states_sha256,
        observed_efforts_sha256,
    )


def _maximum_relative_vector_defect(
    defect: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> float:
    relative = defect.abs() / scale.abs().clamp_min(epsilon)
    return float(relative.detach().amax().cpu())


def _all_finite(values: Sequence[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def audit_gate_3(
    core: DirectPoissonPortHamiltonian,
    states: torch.Tensor,
    latent_efforts: torch.Tensor,
    thresholds: Gate3Thresholds = Gate3Thresholds(),
    *,
    production_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    chunk_size: int = 32,
) -> GateAuditResult:
    """Audit Poisson, PSD, continuous-power, and discrete-power structure.

    Only latent coordinates and latent efforts are accepted.  Inputs are
    detached immediately; there is no public simulator-command argument and
    no returned tensor can be used as a loss.
    """

    reasons: list[str] = []
    if production_step is None or not callable(production_step):
        reasons.append(
            "Gate 3 requires the actual deployed float32 model.step callable"
        )
    if chunk_size < 1:
        reasons.append("chunk_size must be positive")
    if states.ndim != 2 or states.shape[-1] != core.config.state_size:
        reasons.append(
            "states must have shape [sample, core.config.state_size]"
        )
    if latent_efforts.ndim != 2 or latent_efforts.shape[-1] != core.config.port_size:
        reasons.append(
            "latent_efforts must have shape [sample, core.config.port_size]"
        )
    if states.ndim == 2 and latent_efforts.ndim == 2 and states.shape[0] != latent_efforts.shape[0]:
        reasons.append("state and latent-effort sample counts differ")
    if reasons:
        return _gate_result(3, {}, {}, unauditable_reasons=reasons)

    sample_count = int(states.shape[0])
    if sample_count < thresholds.minimum_states:
        reasons.append(
            f"Gate 3 requires at least {thresholds.minimum_states} held-out states; "
            f"received {sample_count}"
        )
    if not bool(torch.isfinite(states).all()):
        reasons.append("states contain a non-finite value")
    if not bool(torch.isfinite(latent_efforts).all()):
        reasons.append("latent efforts contain a non-finite value")
    if reasons:
        return _gate_result(
            3,
            {},
            {"sample_count": sample_count},
            unauditable_reasons=reasons,
        )

    # A 1e-8 implicit residual lies below the reliable float32 floor.  Audit a
    # deep copy of the frozen weights with the identical equations in float64;
    # this performs no fitting/projection and never mutates the sealed model.
    try:
        audit_core = copy.deepcopy(core).to(
            device=states.device, dtype=torch.float64
        ).eval().requires_grad_(False)
        production_core = copy.deepcopy(core).to(
            device=states.device, dtype=torch.float32
        ).eval().requires_grad_(False)
    except Exception as error:
        return _gate_result(
            3,
            {},
            {"sample_count": sample_count},
            unauditable_reasons=(
                f"could not construct the float64 structural audit clone: {error}",
            ),
        )
    detached_states = states.detach().to(dtype=torch.float64)
    detached_efforts = latent_efforts.detach().to(
        device=detached_states.device,
        dtype=detached_states.dtype,
    )
    skew_values: list[torch.Tensor] = []
    jacobi_values: list[torch.Tensor] = []
    resistance_minima: list[torch.Tensor] = []
    port_minima: list[torch.Tensor] = []
    continuous_values: list[torch.Tensor] = []
    discrete_values: list[torch.Tensor] = []
    zero_increases: list[torch.Tensor] = []
    implicit_failures: list[torch.Tensor] = []
    implicit_residuals: list[torch.Tensor] = []
    production_path_defects: list[torch.Tensor] = []
    production_discrete_values: list[torch.Tensor] = []
    production_absolute_discrete_values: list[torch.Tensor] = []
    production_zero_increases: list[torch.Tensor] = []
    production_implicit_failures: list[torch.Tensor] = []
    production_implicit_residuals: list[torch.Tensor] = []
    finite = True
    execution_error: str | None = None
    try:
        for start in range(0, sample_count, chunk_size):
            stop = min(start + chunk_size, sample_count)
            state = detached_states[start:stop]
            effort = detached_efforts[start:stop]
            with torch.no_grad():
                energy, gradient, interconnection, resistance, port = audit_core.components(
                    state, create_graph=False
                )
            j_norm = torch.linalg.matrix_norm(
                interconnection, ord="fro", dim=(-2, -1)
            )
            skew_norm = torch.linalg.matrix_norm(
                interconnection + interconnection.transpose(-1, -2),
                ord="fro",
                dim=(-2, -1),
            )
            skew_values.append(
                skew_norm / j_norm.clamp_min(thresholds.relative_epsilon)
            )

            # The Jacobi tensor has the scale of J*dJ.  States are whitened by
            # the registered visual objective; normalizing by max(1, ||J||^2)
            # gives a deterministic dimensionless audit without a fitted
            # coordinate rescaling at evaluation time.
            with torch.no_grad():
                jacobi = audit_core.jacobi_tensor(state, create_graph=False)
            jacobi_norm = torch.linalg.vector_norm(
                jacobi.flatten(1), dim=-1
            ) / j_norm.square().clamp_min(1.0)
            jacobi_values.append(jacobi_norm)
            resistance_minima.append(torch.linalg.eigvalsh(resistance).amin(dim=-1))
            port_minima.append(torch.linalg.svdvals(port).amin(dim=-1))

            internal = torch.einsum(
                "bij,bj->bi", interconnection - resistance, gradient
            )
            supplied = torch.einsum("bim,bm->bi", port, effort)
            energy_rate = torch.einsum("bi,bi->b", gradient, internal + supplied)
            dissipative_rate = torch.einsum(
                "bi,bij,bj->b", gradient, resistance, gradient
            )
            output = torch.einsum("bim,bi->bm", port, gradient)
            supplied_rate = torch.einsum("bm,bm->b", effort, output)
            expected_rate = -dissipative_rate + supplied_rate
            continuous_scale = energy_rate.abs() + expected_rate.abs()
            continuous_values.append(
                (energy_rate - expected_rate).abs()
                / continuous_scale.clamp_min(thresholds.relative_epsilon)
            )

            with torch.no_grad():
                controlled = audit_core.audited_step(state, effort)
                zero_effort = torch.zeros(
                    state.shape[0],
                    audit_core.config.port_size,
                    device=state.device,
                    dtype=state.dtype,
                )
                autonomous = audit_core.audited_step(state, zero_effort)
            discrete_scale = (
                controlled.energy_delta.abs()
                + controlled.dissipated_energy.abs()
                + controlled.supplied_energy.abs()
            )
            discrete_values.append(
                controlled.balance_defect.abs()
                / discrete_scale.clamp_min(thresholds.relative_epsilon)
            )
            energy_tolerance = thresholds.zero_effort_energy_tolerance * (
                1.0 + autonomous.energy_before.abs()
            )
            zero_increases.append(autonomous.energy_delta > energy_tolerance)

            controlled_failure = (
                ~torch.isfinite(controlled.implicit_residual_norm)
                | (
                    controlled.implicit_residual_norm
                    > thresholds.implicit_residual_tolerance
                )
            )
            autonomous_failure = (
                ~torch.isfinite(autonomous.implicit_residual_norm)
                | (
                    autonomous.implicit_residual_norm
                    > thresholds.implicit_residual_tolerance
                )
            )
            implicit_failures.extend((controlled_failure, autonomous_failure))
            implicit_residuals.extend(
                (controlled.implicit_residual_norm, autonomous.implicit_residual_norm)
            )
            finite = finite and _all_finite(
                (
                    energy,
                    gradient,
                    interconnection,
                    resistance,
                    port,
                    jacobi,
                    controlled.next_state,
                    controlled.balance_defect,
                    autonomous.next_state,
                    autonomous.balance_defect,
                )
            )

            # The controller does not execute the float64 clone above.  Audit
            # the literal public model.step path in float32 and independently
            # compare it with an audited float32 clone of the sealed core.
            # This catches wrappers, casts or autocast changes that a pure
            # structural clone would otherwise miss.
            production_state = states.detach()[start:stop].to(
                device=states.device, dtype=torch.float32
            )
            production_effort = latent_efforts.detach()[start:stop].to(
                device=states.device, dtype=torch.float32
            )
            with torch.no_grad():
                actual_next = production_step(production_state, production_effort)  # type: ignore[misc]
                production_controlled = production_core.audited_step(
                    production_state, production_effort
                )
            if (
                type(actual_next) is not torch.Tensor
                or actual_next.shape != production_state.shape
                or actual_next.dtype != torch.float32
                or actual_next.device != production_state.device
            ):
                raise ValueError(
                    "the deployed production step returned the wrong shape/dtype/device"
                )
            path_scale = torch.linalg.vector_norm(
                production_controlled.next_state, dim=-1
            ).clamp_min(32.0 * torch.finfo(torch.float32).eps)
            production_path_defects.append(
                torch.linalg.vector_norm(
                    actual_next - production_controlled.next_state, dim=-1
                )
                / path_scale
            )
            production_scale = (
                production_controlled.energy_delta.abs()
                + production_controlled.dissipated_energy.abs()
                + production_controlled.supplied_energy.abs()
            )
            production_discrete_values.append(
                production_controlled.balance_defect.abs()
                / production_scale.clamp_min(
                    32.0 * torch.finfo(torch.float32).eps
                )
            )
            production_absolute_discrete_values.append(
                production_controlled.balance_defect.abs()
            )
            production_implicit_residuals.append(
                production_controlled.implicit_residual_norm
            )
            production_implicit_failures.append(
                ~torch.isfinite(production_controlled.implicit_residual_norm)
                | (
                    production_controlled.implicit_residual_norm
                    > thresholds.production_implicit_residual_tolerance
                )
            )

            rollout_state = production_state
            zero_effort = torch.zeros_like(production_effort)
            for _ in range(thresholds.production_rollout_steps):
                with torch.no_grad():
                    energy_before = production_core.hamiltonian(rollout_state)
                    actual_autonomous = production_step(rollout_state, zero_effort)  # type: ignore[misc]
                    audited_autonomous = production_core.audited_step(
                        rollout_state, zero_effort
                    )
                    energy_after = production_core.hamiltonian(actual_autonomous)
                rollout_scale = torch.linalg.vector_norm(
                    audited_autonomous.next_state, dim=-1
                ).clamp_min(32.0 * torch.finfo(torch.float32).eps)
                production_path_defects.append(
                    torch.linalg.vector_norm(
                        actual_autonomous - audited_autonomous.next_state, dim=-1
                    )
                    / rollout_scale
                )
                tolerance = thresholds.production_zero_effort_energy_tolerance * (
                    1.0 + energy_before.abs()
                )
                production_zero_increases.append(
                    energy_after - energy_before > tolerance
                )
                production_implicit_residuals.append(
                    audited_autonomous.implicit_residual_norm
                )
                production_implicit_failures.append(
                    ~torch.isfinite(audited_autonomous.implicit_residual_norm)
                    | (
                        audited_autonomous.implicit_residual_norm
                        > thresholds.production_implicit_residual_tolerance
                    )
                )
                finite = finite and _all_finite(
                    (
                        actual_next,
                        production_controlled.balance_defect,
                        actual_autonomous,
                        energy_before,
                        energy_after,
                        audited_autonomous.balance_defect,
                    )
                )
                rollout_state = actual_autonomous
    except Exception as error:  # an unevaluable identity is a negative outcome
        execution_error = f"Gate 3 computation failed: {type(error).__name__}: {error}"
    finally:
        del audit_core
        del production_core

    if execution_error is not None:
        return _gate_result(
            3,
            {},
            {"sample_count": sample_count},
            unauditable_reasons=(execution_error,),
        )
    if not finite:
        return _gate_result(
            3,
            {},
            {"sample_count": sample_count},
            unauditable_reasons=("a Gate 3 structural quantity is non-finite",),
        )

    max_skew = float(torch.cat(skew_values).amax().cpu())
    max_jacobi = float(torch.cat(jacobi_values).amax().cpu())
    min_resistance = float(torch.cat(resistance_minima).amin().cpu())
    min_port_singular = float(torch.cat(port_minima).amin().cpu())
    max_continuous = float(torch.cat(continuous_values).amax().cpu())
    max_discrete = float(torch.cat(discrete_values).amax().cpu())
    zero_increase_fraction = float(torch.cat(zero_increases).float().mean().cpu())
    implicit_failure_fraction = float(
        torch.cat(implicit_failures).float().mean().cpu()
    )
    max_implicit_residual = float(torch.cat(implicit_residuals).amax().cpu())
    production_max_path_defect = float(
        torch.cat(production_path_defects).amax().cpu()
    )
    production_max_discrete = float(
        torch.cat(production_discrete_values).amax().cpu()
    )
    production_max_absolute_discrete = float(
        torch.cat(production_absolute_discrete_values).amax().cpu()
    )
    production_zero_increase_fraction = float(
        torch.cat(production_zero_increases).float().mean().cpu()
    )
    production_implicit_failure_fraction = float(
        torch.cat(production_implicit_failures).float().mean().cpu()
    )
    production_max_implicit_residual = float(
        torch.cat(production_implicit_residuals).amax().cpu()
    )
    metrics: dict[str, float | int | str] = {
        "sample_count": sample_count,
        "maximum_normalized_skew_defect": max_skew,
        "maximum_normalized_jacobi_defect": max_jacobi,
        "minimum_resistance_eigenvalue": min_resistance,
        "minimum_port_singular_value": min_port_singular,
        "maximum_relative_continuous_power_defect": max_continuous,
        "maximum_relative_discrete_power_defect": max_discrete,
        "zero_effort_energy_increase_fraction": zero_increase_fraction,
        "implicit_solver_failure_fraction": implicit_failure_fraction,
        "maximum_implicit_residual": max_implicit_residual,
        "audit_numeric_dtype": "float64",
        "production_maximum_path_relative_defect": production_max_path_defect,
        "production_maximum_relative_discrete_power_defect": production_max_discrete,
        "production_maximum_absolute_discrete_power_defect": production_max_absolute_discrete,
        "production_zero_effort_energy_increase_fraction": production_zero_increase_fraction,
        "production_implicit_solver_failure_fraction": production_implicit_failure_fraction,
        "production_maximum_implicit_residual": production_max_implicit_residual,
        "production_rollout_steps": thresholds.production_rollout_steps,
        "production_numeric_dtype": "float32",
    }
    checks = {
        "skew_defect": max_skew <= thresholds.maximum_normalized_skew_defect,
        "jacobi_defect": max_jacobi <= thresholds.maximum_normalized_jacobi_defect,
        "resistance_psd": min_resistance >= thresholds.minimum_resistance_eigenvalue,
        "port_full_column_rank": (
            min_port_singular >= thresholds.minimum_port_singular_value
        ),
        "continuous_power_identity": (
            max_continuous <= thresholds.maximum_relative_continuous_power_defect
        ),
        "controlled_discrete_power_identity": (
            max_discrete <= thresholds.maximum_relative_discrete_power_defect
        ),
        "zero_effort_energy_increase_fraction": (
            zero_increase_fraction
            <= thresholds.maximum_zero_effort_energy_increase_fraction
        ),
        "implicit_solver_failure_fraction": (
            implicit_failure_fraction <= thresholds.maximum_implicit_failure_fraction
        ),
        "deployed_float32_path_identity": (
            production_max_path_defect <= thresholds.production_path_relative_tolerance
        ),
        "deployed_float32_discrete_power_identity": (
            production_max_discrete
            <= thresholds.production_maximum_relative_discrete_power_defect
            and production_max_absolute_discrete
            <= thresholds.production_maximum_absolute_discrete_power_defect
        ),
        "deployed_float32_zero_effort_passivity": (
            production_zero_increase_fraction
            <= thresholds.production_maximum_zero_effort_energy_increase_fraction
        ),
        "deployed_float32_implicit_solver": (
            production_implicit_failure_fraction
            <= thresholds.production_maximum_implicit_failure_fraction
        ),
    }
    return _gate_result(3, checks, metrics)


@dataclass(frozen=True)
class LensAuditEvidence:
    """Detached, pixels/latent-only evidence for Gate 4.

    Response matrices have shape ``[sample, observable, port]``.  Paired
    effects have shape ``[sample, port, ...]``.  Random-write norms have shape
    ``[sample, port, random_draw]``.  ``retention_path_kind`` names the exact
    computational path used by the direction-retention check.  Those
    directions are the horizon-1 entries of ``lens_responses`` and
    ``ph_responses``; a second caller-supplied pair could let a renderer cycle
    masquerade as the registered activation-to-frozen-transformer path.
    The adjoint and explicit-Jacobian fields are numeric identities collected
    independently on that path; no caller-supplied verification flag is
    accepted.  Extractor fields expose the actual frozen empirical-Jacobian
    port used by the model, including its fit-neighbour indices.
    """

    lens_responses: Mapping[int, torch.Tensor] | None = None
    ph_responses: Mapping[int, torch.Tensor] | None = None
    positive_effects: torch.Tensor | None = None
    negative_effects: torch.Tensor | None = None
    baseline_effects: torch.Tensor | None = None
    random_write_effect_norms: torch.Tensor | None = None
    adjoint_jvp_inner_products: torch.Tensor | None = None
    adjoint_vjp_inner_products: torch.Tensor | None = None
    adjoint_jvp_norm_bounds: torch.Tensor | None = None
    adjoint_vjp_norm_bounds: torch.Tensor | None = None
    explicit_state_jacobian_products: torch.Tensor | None = None
    independent_state_jvp_products: torch.Tensor | None = None
    extracted_port_gram_matrices: torch.Tensor | None = None
    extracted_port_singular_values: torch.Tensor | None = None
    extracted_port_reported_orthonormality_defects: torch.Tensor | None = None
    extracted_projected_signal_ratios: torch.Tensor | None = None
    extracted_neighbor_indices: torch.Tensor | None = None
    extracted_neighbor_fit_population: int | None = None
    path_code_sha256: str | None = None
    sealed_path_code_sha256: str | None = None
    path_backbone_sha256: str | None = None
    sealed_backbone_sha256: str | None = None
    path_extractor_sha256: str | None = None
    sealed_extractor_sha256: str | None = None
    path_source_tree_sha256: str | None = None
    sealed_source_tree_sha256: str | None = None
    path_fingerprint_sha256: str | None = None
    random_writes_norm_matched: bool | None = None
    retention_path_kind: str | None = None


ACTIVATION_SUFFIX_RETENTION_PATH = (
    "activation_U_to_frozen_transformer_suffix_to_soft_frame_context_to_E"
    "_vs_port_hamiltonian_step_jacobian"
)


@dataclass(frozen=True)
class Gate4Thresholds:
    """Preregistered Gate 4 thresholds."""

    required_horizons: tuple[int, ...] = (1, 2, 4)
    minimum_odd_symmetry_cosine: float = 0.90
    minimum_random_write_effect_ratio: float = 3.0
    minimum_principal_cosine: float = 0.80
    maximum_normalized_response_error: float = 0.25
    minimum_retained_direction_fraction: float = 0.90
    retained_direction_cosine: float = 0.80
    minimum_samples: int = 128
    minimum_random_draws: int = 16
    required_extracted_neighbor_count: int = 32
    rank_tolerance: float = 1e-6
    maximum_adjoint_relative_defect: float = 2e-4
    maximum_explicit_jvp_relative_defect: float = 2e-4
    maximum_extracted_port_orthonormality_defect: float = 2e-5
    minimum_extracted_port_relative_singular_value: float = 1e-6
    minimum_extracted_projected_signal_ratio: float = 1e-6
    maximum_extracted_projected_signal_ratio: float = 1.0 + 2e-5
    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        if not self.required_horizons or min(self.required_horizons) < 1:
            raise ValueError("required_horizons must contain positive integers")
        if (
            self.minimum_samples < 1
            or self.minimum_random_draws < 1
            or self.required_extracted_neighbor_count < 1
        ):
            raise ValueError("minimum audit counts must be positive")
        positive = (
            self.rank_tolerance,
            self.maximum_adjoint_relative_defect,
            self.maximum_explicit_jvp_relative_defect,
            self.maximum_extracted_port_orthonormality_defect,
            self.minimum_extracted_port_relative_singular_value,
            self.minimum_extracted_projected_signal_ratio,
            self.maximum_extracted_projected_signal_ratio,
            self.epsilon,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Gate 4 numerical thresholds must be finite and positive")
        if (
            self.minimum_extracted_projected_signal_ratio
            > self.maximum_extracted_projected_signal_ratio
        ):
            raise ValueError("projected-signal ratio thresholds are inconsistent")


def gate4_path_fingerprint_sha256(
    *,
    code_sha256: str,
    backbone_sha256: str,
    extractor_sha256: str,
    source_tree_sha256: str,
    retention_path_kind: str,
    horizons: Sequence[int],
) -> str:
    """Canonical seal for the exact Gate-4 differentiation path.

    The runner obtains the component digests independently from the live code,
    the live frozen backbone, the frozen empirical-port artifact, and the
    already verified source manifest.
    Keeping the canonical composition here lets the audit recompute the seal
    rather than trusting a caller-provided provenance flag.
    """

    hashes = (
        code_sha256,
        backbone_sha256,
        extractor_sha256,
        source_tree_sha256,
    )
    if any(
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise ValueError("Gate 4 path components must be canonical SHA-256 values")
    selected = tuple(int(horizon) for horizon in horizons)
    if (
        type(retention_path_kind) is not str
        or not retention_path_kind
        or not selected
        or any(horizon < 1 for horizon in selected)
    ):
        raise ValueError("Gate 4 path identity is invalid")
    digest = hashlib.sha256()
    for label, value in (
        ("code", code_sha256),
        ("backbone", backbone_sha256),
        ("extractor", extractor_sha256),
        ("source", source_tree_sha256),
        ("path", retention_path_kind),
        ("horizons", ",".join(str(item) for item in selected)),
    ):
        encoded = value.encode("utf-8")
        digest.update(label.encode("ascii"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _reject_autograd_evidence(name: str, value: torch.Tensor, reasons: list[str]) -> None:
    if value.requires_grad or value.grad_fn is not None:
        reasons.append(f"{name} is still attached to an autograd graph")


def _cosine_rows(first: torch.Tensor, second: torch.Tensor, epsilon: float) -> torch.Tensor:
    numerator = (first * second).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(first, dim=-1)
        * torch.linalg.vector_norm(second, dim=-1)
    )
    return numerator / denominator.clamp_min(epsilon)


def _response_rank_is_full(response: torch.Tensor, tolerance: float) -> bool:
    singular = torch.linalg.svdvals(response)
    if singular.shape[-1] != response.shape[-1]:
        return False
    scale = singular[..., :1].clamp_min(torch.finfo(singular.dtype).eps)
    return bool((singular[..., -1:] > tolerance * scale).all())


def _minimum_principal_cosine(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    first_basis, _ = torch.linalg.qr(first, mode="reduced")
    second_basis, _ = torch.linalg.qr(second, mode="reduced")
    cross = first_basis.transpose(-1, -2) @ second_basis
    return torch.linalg.svdvals(cross).amin(dim=-1)


def _shared_orthogonal_response_error(
    first_by_horizon: Mapping[int, torch.Tensor],
    second_by_horizon: Mapping[int, torch.Tensor],
    horizons: Sequence[int],
    epsilon: float,
) -> tuple[float, torch.Tensor]:
    first_parts: list[torch.Tensor] = []
    second_parts: list[torch.Tensor] = []
    for horizon in horizons:
        first = first_by_horizon[horizon]
        second = second_by_horizon[horizon]
        first_parts.append(first.reshape(-1, first.shape[-1]))
        second_parts.append(second.reshape(-1, second.shape[-1]))
    first_stacked = torch.cat(first_parts, dim=0)
    second_stacked = torch.cat(second_parts, dim=0)
    # Normalize exactly once over every held-out state and every horizon.  A
    # separate per-state normalization would erase a local gain c(x), allowing
    # a state-dependent port gauge to pass this supposedly post-freeze audit.
    # One global positive scale remains irrelevant, as required by the single
    # constant analytic interface calibration used at deployment.
    first_stacked = first_stacked / torch.linalg.matrix_norm(
        first_stacked, ord="fro"
    ).clamp_min(epsilon)
    second_stacked = second_stacked / torch.linalg.matrix_norm(
        second_stacked, ord="fro"
    ).clamp_min(epsilon)
    left, _, right = torch.linalg.svd(
        first_stacked.T @ second_stacked,
        full_matrices=False,
    )
    shared_basis = left @ right
    residual = first_stacked @ shared_basis - second_stacked
    error = float(
        (
            torch.linalg.vector_norm(residual)
            / torch.linalg.vector_norm(second_stacked).clamp_min(epsilon)
        ).cpu()
    )
    return error, shared_basis


def audit_gate_4(
    evidence: LensAuditEvidence,
    thresholds: Gate4Thresholds = Gate4Thresholds(),
) -> GateAuditResult:
    """Audit the internal multi-horizon Jacobian port after model freeze."""

    reasons: list[str] = []
    required_fields = (
        "lens_responses",
        "ph_responses",
        "positive_effects",
        "negative_effects",
        "baseline_effects",
        "random_write_effect_norms",
        "adjoint_jvp_inner_products",
        "adjoint_vjp_inner_products",
        "adjoint_jvp_norm_bounds",
        "adjoint_vjp_norm_bounds",
        "explicit_state_jacobian_products",
        "independent_state_jvp_products",
        "extracted_port_gram_matrices",
        "extracted_port_singular_values",
        "extracted_port_reported_orthonormality_defects",
        "extracted_projected_signal_ratios",
        "extracted_neighbor_indices",
        "extracted_neighbor_fit_population",
        "path_code_sha256",
        "sealed_path_code_sha256",
        "path_backbone_sha256",
        "sealed_backbone_sha256",
        "path_extractor_sha256",
        "sealed_extractor_sha256",
        "path_source_tree_sha256",
        "sealed_source_tree_sha256",
        "path_fingerprint_sha256",
        "random_writes_norm_matched",
        "retention_path_kind",
    )
    for name in required_fields:
        if getattr(evidence, name) is None:
            reasons.append(f"missing Gate 4 evidence: {name}")
    if reasons:
        return _gate_result(4, {}, {}, unauditable_reasons=reasons)

    lens = dict(evidence.lens_responses or {})
    ph = dict(evidence.ph_responses or {})
    required_horizons = tuple(thresholds.required_horizons)
    if tuple(sorted(lens)) != tuple(sorted(required_horizons)):
        reasons.append("lens responses do not contain exactly the registered horizons")
    if tuple(sorted(ph)) != tuple(sorted(required_horizons)):
        reasons.append("pH responses do not contain exactly the registered horizons")
    if evidence.random_writes_norm_matched is not True:
        reasons.append("random writes are not verified norm matched")
    if evidence.retention_path_kind != ACTIVATION_SUFFIX_RETENTION_PATH:
        reasons.append(
            "direction retention was not collected through exact activation U_J, the "
            "frozen transformer suffix, soft frame/context feedback, and E"
        )
    hash_fields = (
        "path_code_sha256",
        "sealed_path_code_sha256",
        "path_backbone_sha256",
        "sealed_backbone_sha256",
        "path_extractor_sha256",
        "sealed_extractor_sha256",
        "path_source_tree_sha256",
        "sealed_source_tree_sha256",
        "path_fingerprint_sha256",
    )
    for name in hash_fields:
        value = getattr(evidence, name)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            reasons.append(f"{name} is not a canonical SHA-256 value")
    if reasons:
        return _gate_result(4, {}, {}, unauditable_reasons=reasons)

    try:
        expected_path_fingerprint = gate4_path_fingerprint_sha256(
            code_sha256=evidence.path_code_sha256,  # type: ignore[arg-type]
            backbone_sha256=evidence.path_backbone_sha256,  # type: ignore[arg-type]
            extractor_sha256=evidence.path_extractor_sha256,  # type: ignore[arg-type]
            source_tree_sha256=evidence.path_source_tree_sha256,  # type: ignore[arg-type]
            retention_path_kind=evidence.retention_path_kind,  # type: ignore[arg-type]
            horizons=required_horizons,
        )
    except ValueError as error:
        return _gate_result(4, {}, {}, unauditable_reasons=(str(error),))

    tensors: dict[str, torch.Tensor] = {
        "positive_effects": evidence.positive_effects,  # type: ignore[dict-item]
        "negative_effects": evidence.negative_effects,  # type: ignore[dict-item]
        "baseline_effects": evidence.baseline_effects,  # type: ignore[dict-item]
        "random_write_effect_norms": evidence.random_write_effect_norms,  # type: ignore[dict-item]
        "adjoint_jvp_inner_products": (
            evidence.adjoint_jvp_inner_products  # type: ignore[dict-item]
        ),
        "adjoint_vjp_inner_products": (
            evidence.adjoint_vjp_inner_products  # type: ignore[dict-item]
        ),
        "adjoint_jvp_norm_bounds": evidence.adjoint_jvp_norm_bounds,  # type: ignore[dict-item]
        "adjoint_vjp_norm_bounds": evidence.adjoint_vjp_norm_bounds,  # type: ignore[dict-item]
        "explicit_state_jacobian_products": (
            evidence.explicit_state_jacobian_products  # type: ignore[dict-item]
        ),
        "independent_state_jvp_products": (
            evidence.independent_state_jvp_products  # type: ignore[dict-item]
        ),
        "extracted_port_gram_matrices": (
            evidence.extracted_port_gram_matrices  # type: ignore[dict-item]
        ),
        "extracted_port_singular_values": (
            evidence.extracted_port_singular_values  # type: ignore[dict-item]
        ),
        "extracted_port_reported_orthonormality_defects": (
            evidence.extracted_port_reported_orthonormality_defects  # type: ignore[dict-item]
        ),
        "extracted_projected_signal_ratios": (
            evidence.extracted_projected_signal_ratios  # type: ignore[dict-item]
        ),
    }
    for horizon in required_horizons:
        tensors[f"lens_response_h{horizon}"] = lens[horizon]
        tensors[f"ph_response_h{horizon}"] = ph[horizon]
    for name, value in tensors.items():
        _reject_autograd_evidence(name, value, reasons)
        if not value.is_floating_point():
            reasons.append(f"{name} must be floating point")
        elif not bool(torch.isfinite(value).all()):
            reasons.append(f"{name} contains a non-finite value")

    first_response = lens[required_horizons[0]]
    if first_response.ndim != 3:
        reasons.append("response matrices must have shape [sample, observable, port]")
        sample_count = 0
        port_size = 0
    else:
        sample_count = int(first_response.shape[0])
        port_size = int(first_response.shape[-1])
    if sample_count < thresholds.minimum_samples:
        reasons.append(
            f"Gate 4 requires at least {thresholds.minimum_samples} held-out samples; "
            f"received {sample_count}"
        )
    if port_size < 1:
        reasons.append("response matrices have no port column")
    for horizon in required_horizons:
        left = lens[horizon]
        right = ph[horizon]
        if left.ndim != 3 or right.ndim != 3:
            reasons.append(f"horizon {horizon} response is not a rank-3 tensor")
            continue
        if left.shape != right.shape:
            reasons.append(f"horizon {horizon} lens and pH response shapes differ")
        if left.shape[0] != sample_count or left.shape[-1] != port_size:
            reasons.append(f"horizon {horizon} response sample/port shape changed")
        if left.shape[-2] < port_size:
            reasons.append(f"horizon {horizon} observable dimension is below port rank")

    positive = tensors["positive_effects"]
    negative = tensors["negative_effects"]
    baseline = tensors["baseline_effects"]
    random_norms = tensors["random_write_effect_norms"]
    adjoint_jvp = tensors["adjoint_jvp_inner_products"]
    adjoint_vjp = tensors["adjoint_vjp_inner_products"]
    adjoint_jvp_bound = tensors["adjoint_jvp_norm_bounds"]
    adjoint_vjp_bound = tensors["adjoint_vjp_norm_bounds"]
    explicit_products = tensors["explicit_state_jacobian_products"]
    independent_products = tensors["independent_state_jvp_products"]
    extracted_gram = tensors["extracted_port_gram_matrices"]
    extracted_singular = tensors["extracted_port_singular_values"]
    extracted_reported_orthonormality = tensors[
        "extracted_port_reported_orthonormality_defects"
    ]
    extracted_projected_signal = tensors["extracted_projected_signal_ratios"]
    extracted_neighbors = evidence.extracted_neighbor_indices
    extracted_fit_population = evidence.extracted_neighbor_fit_population
    if not (positive.shape == negative.shape == baseline.shape):
        reasons.append("paired positive, negative, and baseline effects must match")
    if positive.ndim < 3 or positive.shape[:2] != (sample_count, port_size):
        reasons.append("paired effects must have shape [sample, port, ...]")
    if random_norms.ndim != 3 or random_norms.shape[:2] != (sample_count, port_size):
        reasons.append("random-write norms must have shape [sample, port, draw]")
    elif random_norms.shape[-1] < thresholds.minimum_random_draws:
        reasons.append(
            f"Gate 4 needs at least {thresholds.minimum_random_draws} random writes per port"
        )
    elif bool((random_norms < 0).any()):
        reasons.append("random-write effect norms cannot be negative")
    if adjoint_jvp.shape != (sample_count, len(required_horizons)):
        reasons.append("adjoint JVP products must have shape [sample,horizon]")
    if adjoint_vjp.shape != adjoint_jvp.shape:
        reasons.append("adjoint VJP products must match the JVP products")
    if (
        adjoint_jvp_bound.shape != adjoint_jvp.shape
        or adjoint_vjp_bound.shape != adjoint_jvp.shape
        or bool((adjoint_jvp_bound < 0).any())
        or bool((adjoint_vjp_bound < 0).any())
    ):
        reasons.append("adjoint norm bounds must be nonnegative [sample,horizon] tensors")
    if (
        explicit_products.ndim != 2
        or explicit_products.shape[0] != sample_count
        or independent_products.shape != explicit_products.shape
    ):
        reasons.append("explicit D_hE and independent JVP products must match [sample,state]")
    if extracted_gram.shape != (sample_count, port_size, port_size):
        reasons.append("extracted-port Gram matrices must have shape [sample,port,port]")
    if extracted_singular.shape != (sample_count, port_size):
        reasons.append("extracted-port singular values must have shape [sample,port]")
    elif bool((extracted_singular < 0).any()):
        reasons.append("extracted-port singular values cannot be negative")
    if extracted_reported_orthonormality.shape != (sample_count,):
        reasons.append(
            "reported extracted-port orthonormality must have shape [sample]"
        )
    elif bool((extracted_reported_orthonormality < 0).any()):
        reasons.append("reported extracted-port orthonormality cannot be negative")
    if extracted_projected_signal.shape != (sample_count,):
        reasons.append("projected-signal ratios must have shape [sample]")
    elif bool((extracted_projected_signal < 0).any()):
        reasons.append("projected-signal ratios cannot be negative")
    if (
        type(extracted_neighbors) is not torch.Tensor
        or extracted_neighbors.dtype != torch.long
        or extracted_neighbors.ndim != 2
        or extracted_neighbors.shape[0] != sample_count
        or extracted_neighbors.shape[1]
        != thresholds.required_extracted_neighbor_count
    ):
        reasons.append(
            "extractor neighbours must be int64 [sample,registered_neighbour]"
        )
    else:
        _reject_autograd_evidence("extracted_neighbor_indices", extracted_neighbors, reasons)
        if bool((extracted_neighbors < 0).any()):
            reasons.append("extractor neighbour indices cannot be negative")
        if any(
            torch.unique(row).numel() != row.numel()
            for row in extracted_neighbors
        ):
            reasons.append("extractor neighbour rows must contain unique fit indices")
    if type(extracted_fit_population) is not int or extracted_fit_population < 1:
        reasons.append("extractor fit population must be a positive integer")
    elif (
        type(extracted_neighbors) is torch.Tensor
        and extracted_neighbors.numel()
        and bool((extracted_neighbors >= extracted_fit_population).any())
    ):
        reasons.append("an extractor neighbour lies outside the sealed fit population")
    if reasons:
        return _gate_result(
            4,
            {},
            {"sample_count": sample_count, "port_size": port_size},
            unauditable_reasons=reasons,
        )

    rank_failures: list[str] = []
    for horizon in required_horizons:
        if not _response_rank_is_full(lens[horizon], thresholds.rank_tolerance):
            rank_failures.append(f"rank-deficient lens response at horizon {horizon}")
        if not _response_rank_is_full(ph[horizon], thresholds.rank_tolerance):
            rank_failures.append(f"rank-deficient pH response at horizon {horizon}")
    if rank_failures:
        return _gate_result(
            4,
            {},
            {"sample_count": sample_count, "port_size": port_size},
            unauditable_reasons=rank_failures,
        )

    positive_delta = (positive - baseline).reshape(sample_count, port_size, -1)
    negative_delta = (negative - baseline).reshape(sample_count, port_size, -1)
    positive_norm = torch.linalg.vector_norm(positive_delta, dim=-1)
    negative_norm = torch.linalg.vector_norm(negative_delta, dim=-1)
    if bool(
        ((positive_norm <= thresholds.epsilon) | (negative_norm <= thresholds.epsilon)).any()
    ):
        return _gate_result(
            4,
            {},
            {"sample_count": sample_count, "port_size": port_size},
            unauditable_reasons=("a paired write has zero measurable effect",),
        )
    odd_cosines = _cosine_rows(
        positive_delta,
        -negative_delta,
        thresholds.epsilon,
    )
    mean_odd_cosine = float(odd_cosines.mean().cpu())

    registered_effect_norm = 0.5 * (positive_norm + negative_norm)
    median_random_norm = random_norms.median(dim=-1).values
    random_ratio = registered_effect_norm / median_random_norm.clamp_min(
        thresholds.epsilon
    )
    median_random_ratio = float(random_ratio.median().cpu())

    adjoint_denominator = (
        adjoint_jvp_bound + adjoint_vjp_bound
    ).clamp_min(thresholds.epsilon)
    maximum_adjoint_defect = float(
        ((adjoint_jvp - adjoint_vjp).abs() / adjoint_denominator).amax().cpu()
    )
    explicit_denominator = (
        torch.linalg.vector_norm(explicit_products, dim=-1)
        + torch.linalg.vector_norm(independent_products, dim=-1)
    ).clamp_min(thresholds.epsilon)
    maximum_explicit_jvp_defect = float(
        (
            torch.linalg.vector_norm(explicit_products - independent_products, dim=-1)
            / explicit_denominator
        ).amax().cpu()
    )
    identity = torch.eye(
        port_size, dtype=extracted_gram.dtype, device=extracted_gram.device
    ).expand_as(extracted_gram)
    maximum_extracted_orthonormality_defect = float(
        torch.linalg.matrix_norm(
            extracted_gram - identity, ord="fro", dim=(-2, -1)
        ).amax().cpu()
    )
    maximum_reported_orthonormality_disagreement = float(
        (
            extracted_reported_orthonormality
            - torch.linalg.matrix_norm(
                extracted_gram - identity, ord="fro", dim=(-2, -1)
            )
        ).abs().amax().cpu()
    )
    extracted_relative_singular = extracted_singular[..., -1] / extracted_singular[
        ..., 0
    ].clamp_min(thresholds.epsilon)
    minimum_extracted_relative_singular = float(
        extracted_relative_singular.amin().cpu()
    )
    minimum_extracted_projected_signal = float(
        extracted_projected_signal.amin().cpu()
    )
    maximum_extracted_projected_signal = float(
        extracted_projected_signal.amax().cpu()
    )

    principal_cosines = []
    for horizon in required_horizons:
        principal_cosines.append(
            _minimum_principal_cosine(lens[horizon], ph[horizon])
        )
    minimum_principal_cosine = float(torch.stack(principal_cosines).amin().cpu())
    normalized_response_error, shared_response_basis = _shared_orthogonal_response_error(
        lens,
        ph,
        required_horizons,
        thresholds.epsilon,
    )

    # At horizon 1, ``lens`` is the JVP through U_J(context) -> frozen suffix ->
    # predicted soft frame -> shifted context -> E, while ``ph`` is the direct
    # pH-step Jacobian.  No learned renderer is on this pass/fail path.
    # Apply the *same single* orthogonal Procrustes frame fitted jointly over
    # every state and every registered horizon above.  This removes only the
    # allowed constant port gauge; no Q(x), per-state, or per-horizon fit can
    # enter the retention statistic.
    activation_directions = lens[required_horizons[0]] @ shared_response_basis
    ph_directions = ph[required_horizons[0]]
    activation_columns = activation_directions.transpose(-1, -2).reshape(
        -1, activation_directions.shape[-2]
    )
    ph_columns = ph_directions.transpose(-1, -2).reshape(
        -1, ph_directions.shape[-2]
    )
    direction_cosines = _cosine_rows(
        activation_columns,
        ph_columns,
        thresholds.epsilon,
    )
    if bool(
        (
            (torch.linalg.vector_norm(activation_columns, dim=-1) <= thresholds.epsilon)
            | (torch.linalg.vector_norm(ph_columns, dim=-1) <= thresholds.epsilon)
        ).any()
    ):
        return _gate_result(
            4,
            {},
            {"sample_count": sample_count, "port_size": port_size},
            unauditable_reasons=(
                "an activation-suffix or pH Jacobian direction has zero norm",
            ),
        )
    retained_fraction = float(
        (direction_cosines >= thresholds.retained_direction_cosine).float().mean().cpu()
    )

    metrics: dict[str, float | int | str] = {
        "sample_count": sample_count,
        "port_size": port_size,
        "mean_odd_symmetry_cosine": mean_odd_cosine,
        "median_norm_matched_random_write_ratio": median_random_ratio,
        "minimum_principal_cosine": minimum_principal_cosine,
        "normalized_multi_horizon_response_error": normalized_response_error,
        "activation_suffix_retained_direction_fraction": retained_fraction,
        "maximum_adjoint_jvp_vjp_relative_defect": maximum_adjoint_defect,
        "maximum_explicit_state_jacobian_jvp_relative_defect": (
            maximum_explicit_jvp_defect
        ),
        "maximum_extracted_port_orthonormality_defect": (
            maximum_extracted_orthonormality_defect
        ),
        "minimum_extracted_port_relative_singular_value": (
            minimum_extracted_relative_singular
        ),
        "maximum_reported_orthonormality_disagreement": (
            maximum_reported_orthonormality_disagreement
        ),
        "minimum_extracted_projected_signal_ratio": (
            minimum_extracted_projected_signal
        ),
        "maximum_extracted_projected_signal_ratio": (
            maximum_extracted_projected_signal
        ),
        "extracted_neighbor_count": int(extracted_neighbors.shape[1]),
        "extracted_neighbor_fit_population": extracted_fit_population,
        "gate4_path_fingerprint_sha256": evidence.path_fingerprint_sha256,  # type: ignore[dict-item]
        "gate4_extractor_sha256": evidence.path_extractor_sha256,  # type: ignore[dict-item]
        "retention_path_kind": ACTIVATION_SUFFIX_RETENTION_PATH,
    }
    checks = {
        "odd_symmetry": mean_odd_cosine >= thresholds.minimum_odd_symmetry_cosine,
        "norm_matched_random_write_ratio": (
            median_random_ratio >= thresholds.minimum_random_write_effect_ratio
        ),
        "multi_horizon_principal_cosine": (
            minimum_principal_cosine >= thresholds.minimum_principal_cosine
        ),
        "multi_horizon_response_error": (
            normalized_response_error <= thresholds.maximum_normalized_response_error
        ),
        "activation_suffix_direction_retention": (
            retained_fraction >= thresholds.minimum_retained_direction_fraction
        ),
        "frozen_rollout_adjoint_identity": (
            maximum_adjoint_defect <= thresholds.maximum_adjoint_relative_defect
        ),
        "explicit_state_jacobian_matches_independent_jvp": (
            maximum_explicit_jvp_defect
            <= thresholds.maximum_explicit_jvp_relative_defect
        ),
        "extracted_port_orthonormality": (
            maximum_extracted_orthonormality_defect
            <= thresholds.maximum_extracted_port_orthonormality_defect
            and maximum_reported_orthonormality_disagreement
            <= thresholds.maximum_extracted_port_orthonormality_defect
        ),
        "extracted_port_full_rank": (
            minimum_extracted_relative_singular
            >= thresholds.minimum_extracted_port_relative_singular_value
        ),
        "extracted_port_inside_empirical_tangent": (
            minimum_extracted_projected_signal
            >= thresholds.minimum_extracted_projected_signal_ratio
            and maximum_extracted_projected_signal
            <= thresholds.maximum_extracted_projected_signal_ratio
        ),
        "extracted_port_neighbors_valid": True,
        "gate4_code_matches_seal": (
            evidence.path_code_sha256 == evidence.sealed_path_code_sha256
        ),
        "gate4_backbone_matches_seal": (
            evidence.path_backbone_sha256 == evidence.sealed_backbone_sha256
        ),
        "gate4_extractor_matches_seal": (
            evidence.path_extractor_sha256 == evidence.sealed_extractor_sha256
        ),
        "gate4_source_tree_matches_seal": (
            evidence.path_source_tree_sha256 == evidence.sealed_source_tree_sha256
        ),
        "gate4_path_fingerprint": (
            evidence.path_fingerprint_sha256 == expected_path_fingerprint
        ),
    }
    return _gate_result(4, checks, metrics)


@dataclass(frozen=True)
class AffineAuditAlignment:
    """One affine audit chart fitted only after all neural tensors are frozen."""

    weight: torch.Tensor
    bias: torch.Tensor
    model_sha256: str
    sample_count: int
    normalized_fit_error: float

    def transform_responses(self, latent_responses: torch.Tensor) -> torch.Tensor:
        """Push response vectors through the affine chart's linear part."""

        if latent_responses.shape[-2] != self.weight.shape[-1]:
            raise ValueError("latent response dimension does not match the alignment")
        return torch.einsum("an,...nm->...am", self.weight, latent_responses)


def fit_postfreeze_affine_audit_alignment(
    model: nn.Module,
    latent_coordinates: torch.Tensor,
    audit_coordinates: torch.Tensor,
    *,
    ridge: float = 1e-6,
) -> AffineAuditAlignment:
    """Fit the sole affine chart allowed by the force-port audit.

    The helper refuses trainable models and graph-attached inputs.  It uses a
    closed-form ridge solve under ``no_grad`` and cannot update a neural
    tensor.  ``audit_coordinates`` may be supplied by the post-freeze audit
    suite; it is intentionally unavailable to every loss in this module.
    """

    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("the neural model must be frozen before affine audit alignment")
    reasons: list[str] = []
    _reject_autograd_evidence("latent_coordinates", latent_coordinates, reasons)
    _reject_autograd_evidence("audit_coordinates", audit_coordinates, reasons)
    if reasons:
        raise RuntimeError("; ".join(reasons))
    if latent_coordinates.ndim != 2 or audit_coordinates.ndim != 2:
        raise ValueError("alignment coordinates must both have shape [sample, dimension]")
    if latent_coordinates.shape[0] != audit_coordinates.shape[0]:
        raise ValueError("alignment coordinate sample counts differ")
    if latent_coordinates.shape[0] < latent_coordinates.shape[1] + 1:
        raise ValueError("too few samples to audit an affine latent chart")
    if not bool(torch.isfinite(latent_coordinates).all()) or not bool(
        torch.isfinite(audit_coordinates).all()
    ):
        raise ValueError("alignment coordinates must be finite")

    with torch.no_grad():
        source = latent_coordinates.detach()
        target = audit_coordinates.detach().to(device=source.device, dtype=source.dtype)
        design = torch.cat((source, torch.ones_like(source[:, :1])), dim=-1)
        regularizer = torch.eye(
            design.shape[-1], device=design.device, dtype=design.dtype
        ) * ridge
        regularizer[-1, -1] = 0.0
        coefficients = torch.linalg.solve(
            design.T @ design + regularizer,
            design.T @ target,
        )
        prediction = design @ coefficients
        fit_error = torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(
            target - target.mean(dim=0, keepdim=True)
        ).clamp_min(torch.finfo(target.dtype).eps)
        weight = coefficients[:-1].T.contiguous()
        bias = coefficients[-1].contiguous()
    return AffineAuditAlignment(
        weight=weight,
        bias=bias,
        model_sha256=module_tensor_hash(model),
        sample_count=int(source.shape[0]),
        normalized_fit_error=float(fit_error.cpu()),
    )


@dataclass(frozen=True)
class ForcePortThresholds:
    maximum_affine_alignment_normalized_fit_error: float = 0.35
    maximum_immediate_configuration_to_momentum_ratio: float = 0.35
    minimum_configuration_horizon_growth: float = 1.50
    maximum_nonactuated_to_actuated_momentum_ratio: float = 0.25
    minimum_samples: int = 128
    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        numeric = (
            self.maximum_affine_alignment_normalized_fit_error,
            self.maximum_immediate_configuration_to_momentum_ratio,
            self.minimum_configuration_horizon_growth,
            self.maximum_nonactuated_to_actuated_momentum_ratio,
            self.epsilon,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
            raise ValueError("force-port thresholds must be finite and positive")
        if type(self.minimum_samples) is not int or self.minimum_samples < 1:
            raise ValueError("force-port minimum_samples must be positive")


def audit_force_port_signature(
    model: nn.Module,
    alignment: AffineAuditAlignment,
    latent_responses: Mapping[int, torch.Tensor],
    *,
    configuration_indices: Sequence[int],
    momentum_indices: Sequence[int],
    actuated_momentum_indices: Sequence[int] | None = None,
    nonactuated_momentum_indices: Sequence[int] | None = None,
    locality_sample_mask: torch.Tensor | None = None,
    require_locality: bool = False,
    thresholds: ForcePortThresholds = ForcePortThresholds(),
) -> GateAuditResult:
    """Audit relative degree/locality in one frozen affine audit chart.

    This is Gate 5 rather than a training objective.  Every response and the
    optional pre-event mask must be detached, and the complete model hash must
    still equal the hash sealed by :func:`fit_postfreeze_affine_audit_alignment`.
    """

    reasons: list[str] = []
    if any(parameter.requires_grad for parameter in model.parameters()):
        reasons.append("the force-port audit model is not fully frozen")
    current_hash = module_tensor_hash(model)
    if current_hash != alignment.model_sha256:
        reasons.append("the model hash changed after affine audit alignment")
    if tuple(sorted(latent_responses)) != (1, 4):
        reasons.append("force-port responses must contain exactly horizons 1 and 4")
    for horizon, response in latent_responses.items():
        _reject_autograd_evidence(f"latent_response_h{horizon}", response, reasons)
        if response.ndim != 3:
            reasons.append(f"latent response at horizon {horizon} is not [sample, latent, port]")
        elif response.shape[-2] != alignment.weight.shape[-1]:
            reasons.append(f"latent response at horizon {horizon} has the wrong latent size")
        if not response.is_floating_point() or not bool(torch.isfinite(response).all()):
            reasons.append(f"latent response at horizon {horizon} is not finite floating point")
    if reasons:
        return _gate_result(5, {}, {}, unauditable_reasons=reasons)

    immediate = latent_responses[1]
    delayed = latent_responses[4]
    if immediate.shape != delayed.shape:
        reasons.append("horizon-1 and horizon-4 latent responses differ in shape")
    sample_count = int(immediate.shape[0])
    if sample_count < thresholds.minimum_samples:
        reasons.append(
            f"force-port audit requires at least {thresholds.minimum_samples} samples; "
            f"received {sample_count}"
        )
    aligned_size = int(alignment.weight.shape[0])

    def validate_indices(name: str, indices: Sequence[int] | None) -> tuple[int, ...]:
        if indices is None:
            reasons.append(f"missing {name}")
            return ()
        result = tuple(int(index) for index in indices)
        if not result:
            reasons.append(f"{name} is empty")
        if len(set(result)) != len(result) or any(
            index < 0 or index >= aligned_size for index in result
        ):
            reasons.append(f"{name} is invalid for the affine audit chart")
        return result

    configuration = validate_indices("configuration_indices", configuration_indices)
    momentum = validate_indices("momentum_indices", momentum_indices)
    if set(configuration) & set(momentum):
        reasons.append("configuration and momentum audit groups overlap")

    actuated: tuple[int, ...] = ()
    nonactuated: tuple[int, ...] = ()
    if require_locality:
        actuated = validate_indices("actuated_momentum_indices", actuated_momentum_indices)
        nonactuated = validate_indices(
            "nonactuated_momentum_indices", nonactuated_momentum_indices
        )
        if not set(actuated).issubset(momentum) or not set(nonactuated).issubset(momentum):
            reasons.append("locality groups must be subsets of momentum_indices")
        if set(actuated) & set(nonactuated):
            reasons.append("actuated and nonactuated locality groups overlap")
        if locality_sample_mask is None:
            reasons.append("missing locality_sample_mask")
        else:
            _reject_autograd_evidence("locality_sample_mask", locality_sample_mask, reasons)
            if locality_sample_mask.dtype != torch.bool or locality_sample_mask.shape != (sample_count,):
                reasons.append("locality_sample_mask must be boolean with one entry per sample")
            elif not bool(locality_sample_mask.any()):
                reasons.append("locality_sample_mask selects no pre-event sample")
    if reasons:
        return _gate_result(
            5,
            {},
            {"sample_count": sample_count},
            unauditable_reasons=reasons,
        )

    with torch.no_grad():
        aligned_immediate = alignment.transform_responses(immediate.detach())
        aligned_delayed = alignment.transform_responses(delayed.detach())

        def mean_response_norm(value: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
            selected = value[:, tuple(indices), :]
            return torch.linalg.vector_norm(selected, dim=1).mean()

        immediate_configuration = mean_response_norm(aligned_immediate, configuration)
        immediate_momentum = mean_response_norm(aligned_immediate, momentum)
        delayed_configuration = mean_response_norm(aligned_delayed, configuration)
        immediate_ratio = immediate_configuration / immediate_momentum.clamp_min(
            thresholds.epsilon
        )
        configuration_growth = delayed_configuration / immediate_configuration.clamp_min(
            thresholds.epsilon
        )
        locality_ratio = torch.tensor(float("nan"), device=immediate.device)
        if require_locality:
            assert locality_sample_mask is not None
            selected = aligned_immediate[locality_sample_mask]
            actuated_norm = torch.linalg.vector_norm(
                selected[:, actuated, :], dim=1
            ).mean()
            nonactuated_norm = torch.linalg.vector_norm(
                selected[:, nonactuated, :], dim=1
            ).mean()
            locality_ratio = nonactuated_norm / actuated_norm.clamp_min(thresholds.epsilon)

    metrics: dict[str, float | int | str] = {
        "sample_count": sample_count,
        "affine_alignment_samples": alignment.sample_count,
        "affine_alignment_normalized_fit_error": alignment.normalized_fit_error,
        "immediate_configuration_to_momentum_ratio": float(immediate_ratio.cpu()),
        "configuration_horizon_4_to_1_ratio": float(configuration_growth.cpu()),
    }
    checks = {
        "affine_audit_chart_quality": (
            alignment.normalized_fit_error
            <= thresholds.maximum_affine_alignment_normalized_fit_error
        ),
        "immediate_force_relative_degree": (
            float(immediate_ratio.cpu())
            <= thresholds.maximum_immediate_configuration_to_momentum_ratio
        ),
        "configuration_effect_grows_by_horizon_4": (
            float(configuration_growth.cpu())
            >= thresholds.minimum_configuration_horizon_growth
        ),
    }
    if require_locality:
        metrics["nonactuated_to_actuated_immediate_momentum_ratio"] = float(
            locality_ratio.cpu()
        )
        checks["pre_event_actuator_locality"] = (
            float(locality_ratio.cpu())
            <= thresholds.maximum_nonactuated_to_actuated_momentum_ratio
        )
    if not all(math.isfinite(float(value)) for key, value in metrics.items() if key not in {"sample_count", "affine_alignment_samples"}):
        return _gate_result(
            5,
            {},
            metrics,
            unauditable_reasons=("a force-port audit metric is non-finite",),
        )
    return _gate_result(5, checks, metrics)


__all__ = [
    "ACTIVATION_SUFFIX_RETENTION_PATH",
    "AffineAuditAlignment",
    "FirewallAuditEvidence",
    "ForcePortThresholds",
    "Gate3Thresholds",
    "Gate3TransitionSeal",
    "Gate4Thresholds",
    "GateAuditResult",
    "LensAuditEvidence",
    "RK2PowerAuditResult",
    "audit_matched_rk2_power_error",
    "audit_force_port_signature",
    "audit_gate_1",
    "audit_gate_3",
    "audit_gate_4",
    "fit_postfreeze_affine_audit_alignment",
    "gate4_path_fingerprint_sha256",
    "seal_gate3_transition_samples",
]
