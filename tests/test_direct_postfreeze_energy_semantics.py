from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from blocket_league.direct_postfreeze_energy_semantics import (
    EnergySemanticEvidence,
    EnergySemanticMetrics,
    PositiveAffineEnergyCalibration,
    audit_energy_semantics,
    evaluate_affine_energy_calibration,
    fit_positive_affine_energy_calibration,
    physical_energy_from_gate5_coordinates,
    verify_energy_semantic_artifact_payload,
)


class PostfreezeEnergySemanticTests(unittest.TestCase):
    def test_physical_energy_formulas_use_only_gate5_coordinates(self) -> None:
        pendulum = torch.tensor(((0.0, 0.0), (torch.pi, 0.0), (0.0, 0.25)))
        pendulum_energy = physical_energy_from_gate5_coordinates(
            "pendulum", pendulum, torch.zeros(2), torch.ones(2)
        )
        self.assertEqual(float(pendulum_energy[0]), 0.0)
        self.assertGreater(float(pendulum_energy[1]), 0.0)
        self.assertGreater(float(pendulum_energy[2]), 0.0)

        blocket = torch.zeros(3, 8)
        blocket[1, 4] = 1.8
        blocket[2, 6] = 1.0
        blocket_energy = physical_energy_from_gate5_coordinates(
            "blocket", blocket, torch.zeros(8), torch.ones(8)
        )
        torch.testing.assert_close(
            blocket_energy, torch.tensor((0.0, 0.9, 0.5), dtype=torch.float64)
        )

    @staticmethod
    def _good_tensors():
        alignment_h = torch.linspace(-2.0, 2.0, 256, dtype=torch.float64)
        evaluation_h = torch.linspace(-1.95, 2.05, 128, dtype=torch.float64)
        alignment_e = 2.75 * alignment_h + 6.0
        evaluation_e = 2.75 * evaluation_h + 6.0
        return alignment_h, alignment_e, evaluation_h, evaluation_e

    def test_positive_affine_semantics_generalize_exactly(self) -> None:
        alignment_h, alignment_e, evaluation_h, evaluation_e = self._good_tensors()
        calibration = fit_positive_affine_energy_calibration(
            alignment_h, alignment_e
        )
        metrics = evaluate_affine_energy_calibration(
            calibration,
            alignment_h,
            alignment_e,
            evaluation_h,
            evaluation_e,
        )
        self.assertAlmostEqual(calibration.slope, 2.75, places=12)
        self.assertLess(metrics.heldout_normalized_rmse, 1e-12)
        self.assertGreater(metrics.heldout_r2, 0.999999)
        self.assertGreater(metrics.heldout_pearson, 0.999999)

    def test_negative_or_nonsemantic_hamiltonian_cannot_pass(self) -> None:
        alignment_h, alignment_e, evaluation_h, evaluation_e = self._good_tensors()
        negative = fit_positive_affine_energy_calibration(-alignment_h, alignment_e)
        self.assertLess(negative.slope, 0.0)

        nonlinear_evaluation = evaluation_e + 8.0 * evaluation_h.square()
        calibration = fit_positive_affine_energy_calibration(
            alignment_h, alignment_e
        )
        metrics = evaluate_affine_energy_calibration(
            calibration,
            alignment_h,
            alignment_e,
            evaluation_h,
            nonlinear_evaluation,
        )
        self.assertGreater(metrics.heldout_normalized_rmse, 0.35)

    def test_evidence_digest_and_statistics_are_tamper_evident(self) -> None:
        alignment_h, alignment_e, evaluation_h, evaluation_e = self._good_tensors()
        calibration = fit_positive_affine_energy_calibration(
            alignment_h, alignment_e
        )
        metrics = evaluate_affine_energy_calibration(
            calibration,
            alignment_h,
            alignment_e,
            evaluation_h,
            evaluation_e,
        )
        # Build the digest through a minimal first rejected shell, then import
        # the module-private deterministic digest only to construct authentic
        # typed test evidence.  Decision checks still recompute every value.
        from blocket_league.direct_postfreeze_energy_semantics import (
            _canonical_sha256,
            _evidence_sha256,
        )

        values = {
            "system_name": "pendulum",
            "gate5_evidence_sha256": "1" * 64,
            "model_sha256": "2" * 64,
            "checkpoint_sha256": "3" * 64,
            "alignment_context_sha256": "4" * 64,
            "evaluation_context_sha256": "5" * 64,
            "alignment_latent_energy": alignment_h,
            "alignment_physical_energy": alignment_e,
            "evaluation_latent_energy": evaluation_h,
            "evaluation_physical_energy": evaluation_e,
            "calibration": calibration,
            "metrics": metrics,
            "neural_hash_before": "2" * 64,
            "neural_hash_after": "2" * 64,
            "evidence_sha256": "0" * 64,
            "gradient_updates": 0,
            "physical_commands_read": 0,
            "read_phase": "postfreeze_gate5_coordinates_energy_audit_only",
        }
        values["evidence_sha256"] = _evidence_sha256(type("Evidence", (), values)())
        evidence = EnergySemanticEvidence(**values)
        result = audit_energy_semantics(evidence)
        self.assertTrue(result.passed)
        verified = verify_energy_semantic_artifact_payload(result.to_dict())
        self.assertEqual(verified["system"], "pendulum")
        self.assertEqual(verified["gate5EvidenceSha256"], "1" * 64)
        self.assertEqual(verified["modelSha256"], "2" * 64)
        self.assertEqual(verified["checkpointSha256"], "3" * 64)

        artifact = result.to_dict()
        with self.assertRaises(ValueError):
            verify_energy_semantic_artifact_payload(
                {**artifact, "passed": False}
            )
        with self.assertRaises(ValueError):
            verify_energy_semantic_artifact_payload(
                {**artifact, "checkpointSha256": "9" * 64}
            )
        forged_metrics = {
            **artifact["metrics"],
            "heldoutR2": -100.0,
        }
        forged_core = {
            **{name: value for name, value in artifact.items() if name != "artifactSha256"},
            "metrics": forged_metrics,
        }
        with self.assertRaises(ValueError):
            verify_energy_semantic_artifact_payload(
                {
                    **forged_core,
                    "artifactSha256": _canonical_sha256(forged_core),
                }
            )

        with self.assertRaises(ValueError):
            EnergySemanticEvidence(
                **{
                    **values,
                    "evaluation_physical_energy": evaluation_e + 0.1,
                }
            )
        with self.assertRaises(ValueError):
            audit_energy_semantics(
                replace(
                    evidence,
                    metrics=EnergySemanticMetrics(
                        **{
                            **metrics.__dict__,
                            "heldout_r2": metrics.heldout_r2 - 0.1,
                        }
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
