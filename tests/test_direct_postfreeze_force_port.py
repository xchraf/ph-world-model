from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from blocket_league.direct_ph_structure_audits import (
    fit_postfreeze_affine_audit_alignment,
)
from blocket_league.direct_physical_evaluation import ProbeCandidate
from blocket_league.direct_postfreeze_force_port import (
    Gate5CollectionConfig,
    Gate5Evidence,
    _affine_evaluation_error,
    _gate5_evidence_sha256,
    _precontact_locality_mask,
    _raw_audit_coordinates,
    audit_gate5_evidence,
    verify_gate5_artifact_payload,
)
from blocket_league.passive_jacobian_ph_model import module_tensor_hash


class Gate5CollectorTests(unittest.TestCase):
    def test_protocol_counts_seeds_and_ridge_are_locked(self) -> None:
        Gate5CollectionConfig(batch_size=7)
        with self.assertRaisesRegex(ValueError, "256"):
            Gate5CollectionConfig(alignment_samples=255)
        with self.assertRaisesRegex(ValueError, "128"):
            Gate5CollectionConfig(evaluation_samples=127)
        with self.assertRaisesRegex(ValueError, "seed"):
            Gate5CollectionConfig(alignment_seed=1)
        with self.assertRaisesRegex(ValueError, "1e-6"):
            Gate5CollectionConfig(ridge=1e-5)

    def test_bloket_coordinates_are_q_then_canonical_momenta(self) -> None:
        state = SimpleNamespace(
            player_position=np.asarray((0.2, 0.3), dtype=np.float32),
            player_velocity=np.asarray((0.4, -0.5), dtype=np.float32),
            puck_position=np.asarray((0.7, 0.8), dtype=np.float32),
            puck_velocity=np.asarray((-0.1, 0.2), dtype=np.float32),
        )
        environment = SimpleNamespace(
            state=state,
            config=SimpleNamespace(player_mass=1.8, puck_mass=1.0),
        )
        candidate = ProbeCandidate(
            "fixed", torch.zeros(2, 8, 8, dtype=torch.uint8), environment
        )
        observed = _raw_audit_coordinates((candidate,), "blocket")
        expected = torch.tensor(
            ((0.2, 0.3, 0.7, 0.8, 0.72, -0.90, -0.1, 0.2),)
        )
        torch.testing.assert_close(observed, expected)

        pendulum = ProbeCandidate(
            "pendulum",
            torch.zeros(2, 8, 8, dtype=torch.uint8),
            SimpleNamespace(
                state=SimpleNamespace(angle=0.4, angular_velocity=-0.5),
                config=SimpleNamespace(inertia=2.0),
            ),
        )
        torch.testing.assert_close(
            _raw_audit_coordinates((pendulum,), "pendulum"),
            torch.tensor(((0.4, -1.0),)),
        )

    def test_precontact_mask_is_geometric_and_requires_a_large_subset(self) -> None:
        candidates = []
        for index in range(128):
            state = SimpleNamespace(
                player_position=np.asarray((0.2, 0.5), dtype=np.float32),
                puck_position=np.asarray((0.8, 0.5), dtype=np.float32),
                player_velocity=np.zeros(2, dtype=np.float32),
                puck_velocity=np.zeros(2, dtype=np.float32),
                reset_timer=0,
                last_event="coast",
            )
            if index == 0:
                state.puck_position = np.asarray((0.25, 0.5), dtype=np.float32)
            elif index == 1:
                state.last_event = "impact"
            elif index == 2:
                state.reset_timer = 1
            environment = SimpleNamespace(
                state=state,
                config=SimpleNamespace(
                    dt=0.05,
                    player_acceleration=3.0,
                    player_radius=0.06,
                    puck_radius=0.045,
                ),
            )
            candidates.append(
                ProbeCandidate(
                    f"candidate-{index}",
                    torch.zeros(2, 8, 8, dtype=torch.uint8),
                    environment,
                )
            )
        mask = _precontact_locality_mask(candidates)
        self.assertEqual(int(mask.sum()), 125)
        self.assertFalse(bool(mask[:3].any()))

    @staticmethod
    def _passing_pendulum_evidence() -> Gate5Evidence:
        torch.manual_seed(404)
        model = nn.Linear(2, 2, bias=False).eval().requires_grad_(False)
        model_hash = module_tensor_hash(model)
        alignment_latent = torch.randn(256, 2)
        alignment_coordinates = alignment_latent + torch.tensor((0.3, -0.2))
        alignment = fit_postfreeze_affine_audit_alignment(
            model, alignment_latent, alignment_coordinates
        )
        evaluation_latent = torch.randn(128, 2)
        evaluation_coordinates = evaluation_latent + torch.tensor((0.3, -0.2))
        immediate = (
            torch.tensor((0.10, 1.00)).reshape(1, 2, 1).expand(128, 2, 1).clone()
        )
        delayed = (
            torch.tensor((0.20, 0.70)).reshape(1, 2, 1).expand(128, 2, 1).clone()
        )
        config = Gate5CollectionConfig()
        values = {
            "system_name": "pendulum",
            "protocol": config,
            "model": model,
            "model_sha256": model_hash,
            "checkpoint_sha256": "a" * 64,
            "backbone_sha256": "b" * 64,
            "producer_seal_sha256": "c" * 64,
            "coordinate_schema": ("angle", "angular_momentum"),
            "alignment_identifiers": tuple(f"alignment-{index}" for index in range(256)),
            "evaluation_identifiers": tuple(f"evaluation-{index}" for index in range(128)),
            "alignment_context_sha256": "d" * 64,
            "evaluation_context_sha256": "e" * 64,
            "alignment_latent": alignment_latent,
            "alignment_coordinates": alignment_coordinates,
            "evaluation_latent": evaluation_latent,
            "evaluation_coordinates": evaluation_coordinates,
            "coordinate_mean": torch.zeros(2),
            "coordinate_scale": torch.ones(2),
            "alignment": alignment,
            "alignment_evaluation_normalized_error": _affine_evaluation_error(
                alignment, evaluation_latent, evaluation_coordinates
            ),
            "latent_responses": {1: immediate, 4: delayed},
            "configuration_indices": (0,),
            "momentum_indices": (1,),
            "actuated_momentum_indices": (),
            "nonactuated_momentum_indices": (),
            "locality_sample_mask": None,
            "neural_hash_before": model_hash,
            "neural_hash_after": model_hash,
            "evidence_sha256": "0" * 64,
            "gradient_updates": 0,
            "physical_commands_read": 0,
            "simulator_state_read_phase": "postfreeze_gate5_affine_audit_only",
        }
        values["evidence_sha256"] = _gate5_evidence_sha256(SimpleNamespace(**values))
        return Gate5Evidence(**values)

    @staticmethod
    def _loaded(evidence: Gate5Evidence, *, checkpoint_sha256: str | None = None):
        return SimpleNamespace(
            system_name=evidence.system_name,
            full=SimpleNamespace(
                bundle=SimpleNamespace(model=evidence.model),
                checkpoint_sha256=(
                    evidence.checkpoint_sha256
                    if checkpoint_sha256 is None
                    else checkpoint_sha256
                ),
            ),
            backbone_hash=evidence.backbone_sha256,
            producer_seal_sha256=evidence.producer_seal_sha256,
            assert_frozen_and_unchanged=lambda: None,
        )

    def test_typed_evidence_is_audited_and_tensor_tampering_fails_closed(self) -> None:
        evidence = self._passing_pendulum_evidence()
        loaded = self._loaded(evidence)
        with patch(
            "blocket_league.direct_postfreeze_force_port.collect_gate5_evidence",
            return_value=evidence,
        ):
            artifact = audit_gate5_evidence(evidence, loaded)
        self.assertTrue(artifact.result.auditable, artifact.result.failures)
        self.assertTrue(artifact.result.passed, artifact.result.to_dict())
        serialized = artifact.to_dict()
        self.assertEqual(serialized["gate"], 5)
        self.assertEqual(len(serialized["artifactSha256"]), 64)
        self.assertEqual(serialized["gradientUpdates"], 0)
        self.assertEqual(
            verify_gate5_artifact_payload(serialized)["artifactSha256"],
            serialized["artifactSha256"],
        )
        tampered_artifact = dict(serialized)
        tampered_artifact["localitySamples"] = 1
        with self.assertRaisesRegex(ValueError, "digest"):
            verify_gate5_artifact_payload(tampered_artifact)
        with self.assertRaisesRegex(ValueError, "anchored"):
            audit_gate5_evidence(
                evidence,
                self._loaded(evidence, checkpoint_sha256="f" * 64),
            )
        evidence.latent_responses[1][0, 0, 0] += 0.01
        with patch(
            "blocket_league.direct_postfreeze_force_port.collect_gate5_evidence",
            return_value=evidence,
        ), self.assertRaisesRegex(ValueError, "evidence SHA-256"):
            audit_gate5_evidence(evidence, loaded)

    def test_heldout_affine_chart_quality_is_a_required_conjunct(self) -> None:
        evidence = self._passing_pendulum_evidence()
        values = dict(evidence.__dict__)
        values["evaluation_coordinates"] = torch.randn_like(
            evidence.evaluation_coordinates
        )
        values["alignment_evaluation_normalized_error"] = _affine_evaluation_error(
            evidence.alignment,
            values["evaluation_latent"],
            values["evaluation_coordinates"],
        )
        self.assertGreater(values["alignment_evaluation_normalized_error"], 0.35)
        values["evidence_sha256"] = "0" * 64
        values["evidence_sha256"] = _gate5_evidence_sha256(
            SimpleNamespace(**values)
        )
        failing = Gate5Evidence(**values)
        loaded = self._loaded(failing)
        with patch(
            "blocket_league.direct_postfreeze_force_port.collect_gate5_evidence",
            return_value=failing,
        ):
            artifact = audit_gate5_evidence(failing, loaded)
        self.assertTrue(artifact.result.auditable, artifact.result.failures)
        self.assertFalse(artifact.result.passed)
        self.assertFalse(
            artifact.result.checks["heldout_affine_audit_chart_quality"]
        )


if __name__ == "__main__":
    unittest.main()
