"""Typed Gate 2 and matched-RK2 collection from one sealed post-freeze system.

This module is intentionally separate from the physical evaluators.  Gate 2
opens only the pixels-only fit/test archives, and the RK2 audit consumes the
exact detached transition tensors already used by Gate 3.  Callers receive
typed evidence rather than a user-supplied ``{"passed": true}`` mapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .direct_action_free_data import class_weights, make_optimization_suite
from .direct_jacobian_poisson_ph_experiment import load_sanitized_split
from .direct_ph_structure_audits import (
    GateAuditResult,
    RK2PowerAuditResult,
    audit_matched_rk2_power_error,
    seal_gate3_transition_samples,
)
from .direct_physical_evaluation import FrozenEvaluationSeal
from .direct_postfreeze_quality import (
    Gate2Evidence,
    audit_gate_2,
    collect_gate2_evidence,
    gate2_tensor_sha256,
)
from .direct_postfreeze_runner import (
    HeldoutLatentTransitions,
    LoadedPostFreezeSystem,
    audit_gate3_postfreeze,
)
from .passive_jacobian_ph_model import module_tensor_hash


@dataclass(frozen=True)
class Gate2PostFreezeResult:
    evidence: Gate2Evidence
    audit: GateAuditResult

    def to_dict(self) -> dict[str, Any]:
        return {"evidence": self.evidence.to_dict(), "audit": self.audit.to_dict()}


@dataclass(frozen=True)
class Gate3RK2PostFreezeResult:
    gate3: GateAuditResult
    rk2: RK2PowerAuditResult
    transitions: HeldoutLatentTransitions

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate3": self.gate3.to_dict(),
            "matchedRK2": self.rk2.to_dict(),
            "transitionCount": self.transitions.transition_count,
            "sourceManifestSha256": self.transitions.source_manifest_sha256,
        }


def collect_gate2_postfreeze(
    loaded: LoadedPostFreezeSystem,
    *,
    batch_size: int = 16,
) -> Gate2PostFreezeResult:
    """Recompute fit weights and collect the registered test-pixel Gate 2."""

    loaded.assert_frozen_and_unchanged()
    fit_pixels, fit_manifest = load_sanitized_split(
        loaded.paths.sanitized_split(loaded.system_name, "fit"),
        expected_system=loaded.system_name,
    )
    if asdict(fit_manifest) != asdict(loaded.manifests["fit"]):
        raise ValueError("Gate 2 fit archive differs from the sealed training manifest")
    model_config = loaded.backbone.config  # type: ignore[attr-defined]
    experiment = loaded.training_summary.get("experimentConfig")
    if type(experiment) is not dict or type(experiment.get("transitions")) is not int:
        raise ValueError("Gate 2 training summary has no exact transition count")
    fit_suite = make_optimization_suite(
        fit_pixels,
        model_config,
        transitions=int(experiment["transitions"]),
    )
    weights = class_weights(
        fit_suite["frames"], model_config.palette_size, loaded.device
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
    expected_neural_hashes = dict(FrozenEvaluationSeal.capture(modules).hashes)
    evidence = collect_gate2_evidence(
        system_name=loaded.system_name,
        test_pixels=loaded.test_pixels,
        test_sanitized_tensor_sha256=loaded.manifests[
            "test"
        ].sanitized_tensor_sha256,
        model_config=model_config,
        encoder=full.encoder,
        renderer=full.renderer,
        structured_dynamics=full.core,
        structured_inference=full.effort_inference,
        unstructured_encoder=loaded.independent_baseline.encoder,
        unstructured_renderer=loaded.independent_baseline.renderer,
        unstructured_dynamics=loaded.independent_baseline.dynamics,
        unstructured_inference=loaded.independent_baseline.inference,
        unstructured_write_field=loaded.independent_baseline.bundle.write_field,
        unstructured_response_frame=loaded.independent_baseline.bundle.response_frame,
        class_weights=weights,
        batch_size=batch_size,
    )
    audit = audit_gate_2(
        evidence,
        expected_test_sanitized_tensor_sha256=loaded.manifests[
            "test"
        ].sanitized_tensor_sha256,
        expected_class_weights_sha256=gate2_tensor_sha256(weights),
        expected_neural_hashes=expected_neural_hashes,
    )
    loaded.assert_frozen_and_unchanged()
    return Gate2PostFreezeResult(evidence=evidence, audit=audit)


def collect_gate3_and_rk2_postfreeze(
    loaded: LoadedPostFreezeSystem,
    *,
    batch_size: int = 64,
    audit_chunk_size: int = 32,
) -> Gate3RK2PostFreezeResult:
    """Run Gate 3 and RK2 on one cryptographically identical transition set."""

    gate3, transitions = audit_gate3_postfreeze(
        loaded,
        batch_size=batch_size,
        audit_chunk_size=audit_chunk_size,
    )
    core = loaded.full.bundle.model.core
    core_sha256 = module_tensor_hash(core)
    transition_seal = seal_gate3_transition_samples(
        core,
        transitions.states,
        transitions.efforts,
        source_manifest_sha256=transitions.source_manifest_sha256,
    )
    rk2 = audit_matched_rk2_power_error(
        core,
        transitions.states,
        transitions.efforts,
        transition_seal=transition_seal,
        expected_source_manifest_sha256=transitions.source_manifest_sha256,
        expected_core_sha256=core_sha256,
        chunk_size=audit_chunk_size,
    )
    loaded.assert_frozen_and_unchanged()
    return Gate3RK2PostFreezeResult(
        gate3=gate3,
        rk2=rk2,
        transitions=transitions,
    )


__all__ = [
    "Gate2PostFreezeResult",
    "Gate3RK2PostFreezeResult",
    "collect_gate2_postfreeze",
    "collect_gate3_and_rk2_postfreeze",
]
