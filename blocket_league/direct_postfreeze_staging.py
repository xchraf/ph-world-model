"""Authenticated, resumable staging for Experiment F post-freeze evidence.

The expensive neural evidence is collected once by ``prepare-system``.  The
finalizer reconstructs Gates 1--4 from the detached cached tensors and the
sealed checkpoints; it never reruns a physical probe or a CEM optimizer.
Gate 5's separately authenticated artifact is tied to the same checkpoint and
verified with its strict public verifier.  Its Hamiltonian-energy semantic
sub-audit is stored independently and must share the exact Gate-5 evidence and
context lineage before the combined gate can pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch

from .direct_action_free_data import class_weights, make_optimization_suite
from .direct_jacobian_poisson_ph_experiment import load_sanitized_split
from .direct_ph_structure_audits import (
    FirewallAuditEvidence,
    Gate3Thresholds,
    LensAuditEvidence,
    audit_gate_1,
    audit_gate_3,
    audit_gate_4,
    audit_matched_rk2_power_error,
    seal_gate3_transition_samples,
)
from .direct_physical_evaluation import FrozenEvaluationSeal
from .direct_postfreeze_evidence_io import (
    canonical_evidence_sha256,
    physical_result_sha256,
    training_lineage_sha256,
)
from .direct_postfreeze_force_port import verify_gate5_artifact_payload
from .direct_postfreeze_energy_semantics import (
    verify_energy_semantic_artifact_payload,
)
from .direct_postfreeze_quality import Gate2Evidence, audit_gate_2, gate2_tensor_sha256
from .direct_postfreeze_runner import (
    LoadedPostFreezeSystem,
    PhysicalRealizabilityResult,
    assemble_gate1_evidence,
)
from .passive_jacobian_ph_model import module_tensor_hash


PREPARED_KIND = "experiment_f_prepared_postfreeze_system_v1"


def _dataclass_payload(value: Any) -> dict[str, Any]:
    return {item.name: getattr(value, item.name) for item in fields(value)}


def _gate4_payload(evidence: LensAuditEvidence) -> dict[str, Any]:
    return {
        "lensResponses": {
            str(horizon): tensor.detach().cpu()
            for horizon, tensor in (evidence.lens_responses or {}).items()
        },
        "phResponses": {
            str(horizon): tensor.detach().cpu()
            for horizon, tensor in (evidence.ph_responses or {}).items()
        },
        "positiveEffects": evidence.positive_effects.detach().cpu(),
        "negativeEffects": evidence.negative_effects.detach().cpu(),
        "baselineEffects": evidence.baseline_effects.detach().cpu(),
        "randomWriteEffectNorms": evidence.random_write_effect_norms.detach().cpu(),
        "adjointJvpInnerProducts": evidence.adjoint_jvp_inner_products.detach().cpu(),
        "adjointVjpInnerProducts": evidence.adjoint_vjp_inner_products.detach().cpu(),
        "adjointJvpNormBounds": evidence.adjoint_jvp_norm_bounds.detach().cpu(),
        "adjointVjpNormBounds": evidence.adjoint_vjp_norm_bounds.detach().cpu(),
        "explicitStateJacobianProducts": (
            evidence.explicit_state_jacobian_products.detach().cpu()
        ),
        "independentStateJvpProducts": (
            evidence.independent_state_jvp_products.detach().cpu()
        ),
        "extractedPortGramMatrices": (
            evidence.extracted_port_gram_matrices.detach().cpu()
        ),
        "extractedPortSingularValues": (
            evidence.extracted_port_singular_values.detach().cpu()
        ),
        "extractedPortReportedOrthonormalityDefects": (
            evidence.extracted_port_reported_orthonormality_defects.detach().cpu()
        ),
        "extractedProjectedSignalRatios": (
            evidence.extracted_projected_signal_ratios.detach().cpu()
        ),
        "extractedNeighborIndices": (
            evidence.extracted_neighbor_indices.detach().cpu()
        ),
        "extractedNeighborFitPopulation": (
            evidence.extracted_neighbor_fit_population
        ),
        "pathCodeSha256": evidence.path_code_sha256,
        "sealedPathCodeSha256": evidence.sealed_path_code_sha256,
        "pathBackboneSha256": evidence.path_backbone_sha256,
        "sealedBackboneSha256": evidence.sealed_backbone_sha256,
        "pathExtractorSha256": evidence.path_extractor_sha256,
        "sealedExtractorSha256": evidence.sealed_extractor_sha256,
        "pathSourceTreeSha256": evidence.path_source_tree_sha256,
        "sealedSourceTreeSha256": evidence.sealed_source_tree_sha256,
        "pathFingerprintSha256": evidence.path_fingerprint_sha256,
        "randomWritesNormMatched": evidence.random_writes_norm_matched,
        "retentionPathKind": evidence.retention_path_kind,
    }


def _gate4_from_payload(payload: Any, device: torch.device) -> LensAuditEvidence:
    expected = {
        "lensResponses",
        "phResponses",
        "positiveEffects",
        "negativeEffects",
        "baselineEffects",
        "randomWriteEffectNorms",
        "adjointJvpInnerProducts",
        "adjointVjpInnerProducts",
        "adjointJvpNormBounds",
        "adjointVjpNormBounds",
        "explicitStateJacobianProducts",
        "independentStateJvpProducts",
        "extractedPortGramMatrices",
        "extractedPortSingularValues",
        "extractedPortReportedOrthonormalityDefects",
        "extractedProjectedSignalRatios",
        "extractedNeighborIndices",
        "extractedNeighborFitPopulation",
        "pathCodeSha256",
        "sealedPathCodeSha256",
        "pathBackboneSha256",
        "sealedBackboneSha256",
        "pathExtractorSha256",
        "sealedExtractorSha256",
        "pathSourceTreeSha256",
        "sealedSourceTreeSha256",
        "pathFingerprintSha256",
        "randomWritesNormMatched",
        "retentionPathKind",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("prepared Gate 4 evidence schema is not exact")

    def responses(name: str) -> dict[int, torch.Tensor]:
        values = payload[name]
        if type(values) is not dict or set(values) != {"1", "2", "4"}:
            raise ValueError("prepared Gate 4 response horizons changed")
        return {int(key): value.to(device).detach() for key, value in values.items()}

    def tensor(name: str) -> torch.Tensor:
        value = payload[name]
        if type(value) is not torch.Tensor:
            raise ValueError(f"prepared Gate 4 {name} is not a tensor")
        return value.to(device).detach()

    return LensAuditEvidence(
        lens_responses=responses("lensResponses"),
        ph_responses=responses("phResponses"),
        positive_effects=tensor("positiveEffects"),
        negative_effects=tensor("negativeEffects"),
        baseline_effects=tensor("baselineEffects"),
        random_write_effect_norms=tensor("randomWriteEffectNorms"),
        adjoint_jvp_inner_products=tensor("adjointJvpInnerProducts"),
        adjoint_vjp_inner_products=tensor("adjointVjpInnerProducts"),
        adjoint_jvp_norm_bounds=tensor("adjointJvpNormBounds"),
        adjoint_vjp_norm_bounds=tensor("adjointVjpNormBounds"),
        explicit_state_jacobian_products=tensor("explicitStateJacobianProducts"),
        independent_state_jvp_products=tensor("independentStateJvpProducts"),
        extracted_port_gram_matrices=tensor("extractedPortGramMatrices"),
        extracted_port_singular_values=tensor("extractedPortSingularValues"),
        extracted_port_reported_orthonormality_defects=tensor(
            "extractedPortReportedOrthonormalityDefects"
        ),
        extracted_projected_signal_ratios=tensor(
            "extractedProjectedSignalRatios"
        ),
        extracted_neighbor_indices=tensor("extractedNeighborIndices"),
        extracted_neighbor_fit_population=payload[
            "extractedNeighborFitPopulation"
        ],
        path_code_sha256=payload["pathCodeSha256"],
        sealed_path_code_sha256=payload["sealedPathCodeSha256"],
        path_backbone_sha256=payload["pathBackboneSha256"],
        sealed_backbone_sha256=payload["sealedBackboneSha256"],
        path_extractor_sha256=payload["pathExtractorSha256"],
        sealed_extractor_sha256=payload["sealedExtractorSha256"],
        path_source_tree_sha256=payload["pathSourceTreeSha256"],
        sealed_source_tree_sha256=payload["sealedSourceTreeSha256"],
        path_fingerprint_sha256=payload["pathFingerprintSha256"],
        random_writes_norm_matched=payload["randomWritesNormMatched"],
        retention_path_kind=payload["retentionPathKind"],
    )


@dataclass(frozen=True)
class PreparedSystemArtifact:
    system_name: str
    training_lineage_sha256: str
    physical_sha256: str
    evidence: Mapping[str, Any]
    artifact_sha256: str

    def core_payload(self) -> dict[str, Any]:
        return {
            "kind": PREPARED_KIND,
            "system": self.system_name,
            "trainingLineageSha256": self.training_lineage_sha256,
            "physicalSha256": self.physical_sha256,
            "evidence": dict(self.evidence),
        }

    def __post_init__(self) -> None:
        if self.system_name not in {"pendulum", "blocket"}:
            raise ValueError("prepared evidence has an unknown system")
        if set(self.evidence) != {
            "gate1",
            "gate2",
            "gate3",
            "gate4",
            "gate5",
            "gate5Energy",
        }:
            raise ValueError(
                "prepared evidence does not contain exactly Gates 1--5 "
                "and the Gate-5 energy sub-audit"
            )
        if canonical_evidence_sha256(self.core_payload()) != self.artifact_sha256:
            raise ValueError("prepared evidence digest is invalid")

    def to_payload(self) -> dict[str, Any]:
        return {**self.core_payload(), "artifactSha256": self.artifact_sha256}


def build_prepared_system_artifact(
    loaded: LoadedPostFreezeSystem,
    physical: PhysicalRealizabilityResult,
    *,
    gate1_evidence: FirewallAuditEvidence,
    gate2_evidence: Gate2Evidence,
    gate3_states: torch.Tensor,
    gate3_efforts: torch.Tensor,
    gate3_source_manifest_sha256: str,
    gate4_evidence: LensAuditEvidence,
    gate5_artifact: Mapping[str, Any],
    gate5_energy_artifact: Mapping[str, Any],
) -> PreparedSystemArtifact:
    """Build a tensor-only artifact after all registered collectors finish."""

    loaded.assert_frozen_and_unchanged()
    verify_gate5_artifact_payload(gate5_artifact)
    verify_energy_semantic_artifact_payload(gate5_energy_artifact)
    if gate5_artifact["system"] != loaded.system_name:
        raise ValueError("Gate 5 artifact belongs to another system")
    if (
        gate5_energy_artifact["system"] != loaded.system_name
        or gate5_energy_artifact["gate5EvidenceSha256"]
        != gate5_artifact["evidenceSha256"]
        or gate5_energy_artifact["modelSha256"]
        != gate5_artifact["modelSha256"]
        or gate5_energy_artifact["checkpointSha256"]
        != gate5_artifact["checkpointSha256"]
        or gate5_energy_artifact["alignmentContextSha256"]
        != gate5_artifact["alignmentContextSha256"]
        or gate5_energy_artifact["evaluationContextSha256"]
        != gate5_artifact["evaluationContextSha256"]
    ):
        raise ValueError("Gate-5 energy semantics differ from force-port lineage")
    if gate3_states.requires_grad or gate3_efforts.requires_grad:
        raise ValueError("prepared Gate 3 tensors must be detached")
    evidence: dict[str, Any] = {
        "gate1": _dataclass_payload(gate1_evidence),
        "gate2": _dataclass_payload(gate2_evidence),
        "gate3": {
            "states": gate3_states.detach().cpu(),
            "efforts": gate3_efforts.detach().cpu(),
            "sourceManifestSha256": gate3_source_manifest_sha256,
        },
        "gate4": _gate4_payload(gate4_evidence),
        "gate5": dict(gate5_artifact),
        "gate5Energy": dict(gate5_energy_artifact),
    }
    core = {
        "kind": PREPARED_KIND,
        "system": loaded.system_name,
        "trainingLineageSha256": training_lineage_sha256(loaded),
        "physicalSha256": physical_result_sha256(physical),
        "evidence": evidence,
    }
    return PreparedSystemArtifact(
        system_name=loaded.system_name,
        training_lineage_sha256=core["trainingLineageSha256"],
        physical_sha256=core["physicalSha256"],
        evidence=evidence,
        artifact_sha256=canonical_evidence_sha256(core),
    )


def save_prepared_system_artifact(path: Path, artifact: PreparedSystemArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(artifact.to_payload(), temporary)
    temporary.replace(path)


def load_prepared_system_artifact(
    path: Path,
    loaded: LoadedPostFreezeSystem,
    physical: PhysicalRealizabilityResult,
) -> PreparedSystemArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "kind",
        "system",
        "trainingLineageSha256",
        "physicalSha256",
        "evidence",
        "artifactSha256",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("prepared system artifact schema is not exact")
    if payload["kind"] != PREPARED_KIND:
        raise ValueError("prepared system artifact kind changed")
    artifact = PreparedSystemArtifact(
        system_name=payload["system"],
        training_lineage_sha256=payload["trainingLineageSha256"],
        physical_sha256=payload["physicalSha256"],
        evidence=payload["evidence"],
        artifact_sha256=payload["artifactSha256"],
    )
    if (
        artifact.system_name != loaded.system_name
        or artifact.training_lineage_sha256 != training_lineage_sha256(loaded)
        or artifact.physical_sha256 != physical_result_sha256(physical)
    ):
        raise ValueError("prepared evidence differs from frozen/physical lineage")
    gate5 = artifact.evidence["gate5"]
    verify_gate5_artifact_payload(gate5)
    gate5_energy = artifact.evidence["gate5Energy"]
    verify_energy_semantic_artifact_payload(gate5_energy)
    if (
        gate5["system"] != loaded.system_name
        or gate5["checkpointSha256"] != loaded.full.checkpoint_sha256
        or gate5["modelSha256"] != module_tensor_hash(loaded.full.bundle.model)
        or gate5_energy["system"] != loaded.system_name
        or gate5_energy["checkpointSha256"] != loaded.full.checkpoint_sha256
        or gate5_energy["modelSha256"]
        != module_tensor_hash(loaded.full.bundle.model)
        or gate5_energy["gate5EvidenceSha256"] != gate5["evidenceSha256"]
        or gate5_energy["alignmentContextSha256"]
        != gate5["alignmentContextSha256"]
        or gate5_energy["evaluationContextSha256"]
        != gate5["evaluationContextSha256"]
    ):
        raise ValueError(
            "prepared Gate 5/energy semantics are not tied to the loaded checkpoint"
        )
    return artifact


def _gate2_from_payload(payload: Any) -> Gate2Evidence:
    expected = {item.name for item in fields(Gate2Evidence)}
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("prepared Gate 2 evidence schema is not exact")
    return Gate2Evidence(**payload)


def _gate2_expected_inputs(
    loaded: LoadedPostFreezeSystem,
) -> tuple[str, str, Mapping[str, str]]:
    fit_pixels, fit_manifest = load_sanitized_split(
        loaded.paths.sanitized_split(loaded.system_name, "fit"),
        expected_system=loaded.system_name,
    )
    if asdict(fit_manifest) != asdict(loaded.manifests["fit"]):
        raise ValueError("prepared Gate 2 fit archive changed")
    experiment = loaded.training_summary.get("experimentConfig")
    if type(experiment) is not dict or type(experiment.get("transitions")) is not int:
        raise ValueError("prepared Gate 2 transition count is absent")
    suite = make_optimization_suite(
        fit_pixels,
        loaded.backbone.config,  # type: ignore[attr-defined]
        transitions=int(experiment["transitions"]),
    )
    weights = class_weights(
        suite["frames"], loaded.backbone.config.palette_size, loaded.device  # type: ignore[attr-defined]
    ).detach()
    full = loaded.full.bundle.model
    modules = {
        "encoder": full.encoder,
        "renderer": full.renderer,
        "structuredDynamics": full.core,
        "structuredInference": full.effort_inference,
        "unstructuredEncoder": loaded.independent_baseline.encoder,
        "unstructuredRenderer": loaded.independent_baseline.renderer,
        "unstructuredDynamics": loaded.independent_baseline.dynamics,
        "unstructuredInference": loaded.independent_baseline.inference,
        "unstructuredWriteField": loaded.independent_baseline.bundle.write_field,
        "unstructuredResponseFrame": loaded.independent_baseline.bundle.response_frame,
    }
    hashes = dict(FrozenEvaluationSeal.capture(modules).hashes)
    return (
        loaded.manifests["test"].sanitized_tensor_sha256,
        gate2_tensor_sha256(weights),
        hashes,
    )


def audit_prepared_structural_gates(
    artifact: PreparedSystemArtifact,
    loaded: LoadedPostFreezeSystem,
) -> dict[str, Any]:
    """Recompute Gates 1--4 from cached raw evidence, without recollection."""

    if artifact.training_lineage_sha256 != training_lineage_sha256(loaded):
        raise ValueError("prepared structural lineage changed")
    stored_gate1 = artifact.evidence["gate1"]
    expected_gate1 = _dataclass_payload(assemble_gate1_evidence(loaded))
    if stored_gate1 != expected_gate1:
        raise ValueError("prepared Gate 1 evidence differs from sealed archives")
    gate1 = audit_gate_1(
        loaded.full.bundle.model, FirewallAuditEvidence(**stored_gate1)
    ).to_dict()

    gate2_evidence = _gate2_from_payload(artifact.evidence["gate2"])
    test_sha, weights_sha, neural_hashes = _gate2_expected_inputs(loaded)
    gate2 = audit_gate_2(
        gate2_evidence,
        expected_test_sanitized_tensor_sha256=test_sha,
        expected_class_weights_sha256=weights_sha,
        expected_neural_hashes=neural_hashes,
    ).to_dict()

    gate3_payload = artifact.evidence["gate3"]
    if type(gate3_payload) is not dict or set(gate3_payload) != {
        "states", "efforts", "sourceManifestSha256"
    }:
        raise ValueError("prepared Gate 3 evidence schema is not exact")
    states = gate3_payload["states"]
    efforts = gate3_payload["efforts"]
    source_sha = gate3_payload["sourceManifestSha256"]
    if type(states) is not torch.Tensor or type(efforts) is not torch.Tensor:
        raise ValueError("prepared Gate 3 evidence is not tensor-valued")
    if source_sha != loaded.manifests["test"].sanitized_tensor_sha256:
        raise ValueError("prepared Gate 3 source manifest changed")
    states = states.to(loaded.device).detach()
    efforts = efforts.to(loaded.device).detach()
    thresholds = Gate3Thresholds()
    gate3_audit = audit_gate_3(
        loaded.full.bundle.model.core,
        states,
        efforts,
        thresholds,
        production_step=loaded.full.bundle.model.step,
        chunk_size=32,
    )
    transition_seal = seal_gate3_transition_samples(
        loaded.full.bundle.model.core,
        states,
        efforts,
        source_manifest_sha256=source_sha,
    )
    rk2 = audit_matched_rk2_power_error(
        loaded.full.bundle.model.core,
        states,
        efforts,
        thresholds,
        transition_seal=transition_seal,
        expected_source_manifest_sha256=source_sha,
        expected_core_sha256=module_tensor_hash(loaded.full.bundle.model.core),
        chunk_size=32,
    )
    gate3 = {
        "gate": 3,
        "auditable": gate3_audit.auditable and rk2.auditable,
        "passed": gate3_audit.passed and rk2.passed,
        "checks": {
            "portHamiltonianAudit": gate3_audit.passed,
            "matchedRK2ReferenceAdmissible": rk2.passed,
        },
        "portHamiltonian": gate3_audit.to_dict(),
        "matchedRK2": rk2.to_dict(),
        "transitionCount": int(states.shape[0]),
        "sourceManifestSha256": source_sha,
    }

    gate4_evidence = _gate4_from_payload(artifact.evidence["gate4"], loaded.device)
    gate4 = audit_gate_4(gate4_evidence).to_dict()

    gate5_artifact = verify_gate5_artifact_payload(artifact.evidence["gate5"])
    gate5_energy = verify_energy_semantic_artifact_payload(
        artifact.evidence["gate5Energy"]
    )
    if (
        gate5_energy["gate5EvidenceSha256"] != gate5_artifact["evidenceSha256"]
        or gate5_energy["modelSha256"] != gate5_artifact["modelSha256"]
        or gate5_energy["checkpointSha256"]
        != gate5_artifact["checkpointSha256"]
        or gate5_energy["alignmentContextSha256"]
        != gate5_artifact["alignmentContextSha256"]
        or gate5_energy["evaluationContextSha256"]
        != gate5_artifact["evaluationContextSha256"]
    ):
        raise ValueError("prepared Gate-5 sub-audits have different evidence lineage")
    gate5 = {
        "passed": (
            gate5_artifact["result"]["passed"] and gate5_energy["passed"]
        ),
        "auditable": (
            gate5_artifact["result"]["auditable"]
            and gate5_energy["gradientUpdates"] == 0
            and gate5_energy["physicalCommandsRead"] == 0
        ),
        "checks": {
            "forcePortSignature": gate5_artifact["result"]["passed"],
            "physicalEnergySemantics": gate5_energy["passed"],
        },
        "forcePortArtifact": gate5_artifact,
        "energySemanticArtifact": gate5_energy,
    }
    loaded.assert_frozen_and_unchanged()
    return {
        "gate1": gate1,
        "gate2": gate2,
        "gate3": gate3,
        "gate4": gate4,
        "gate5": gate5,
    }


__all__ = [
    "PREPARED_KIND",
    "PreparedSystemArtifact",
    "audit_prepared_structural_gates",
    "build_prepared_system_artifact",
    "load_prepared_system_artifact",
    "save_prepared_system_artifact",
]
