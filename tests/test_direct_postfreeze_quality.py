from __future__ import annotations

import unittest

import torch
from torch import nn

from blocket_league.direct_action_free_data import sanitized_pixel_tensor_sha256
from blocket_league.direct_postfreeze_quality import (
    Gate2Evidence,
    audit_gate_2,
    collect_gate2_evidence,
)
from blocket_league.pixel_direct_model import PixelDirectConfig


TEST_SHA256 = "a" * 64
CLASS_WEIGHTS_SHA256 = "c" * 64
NEURAL_HASHES = {"model": "b" * 64}


class _ToyEncoder(nn.Module):
    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        value = contexts.float().mean(dim=(-3, -2, -1))
        return torch.stack((value, torch.zeros_like(value)), dim=-1)


class _ToyInference(nn.Module):
    def forward(self, current: torch.Tensor, successor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*current.shape[:-1], 1, device=current.device)


class _ToyDynamics(nn.Module):
    def step(self, state: torch.Tensor, effort: torch.Tensor) -> torch.Tensor:
        return state + 0.0 * effort[..., :1]


class _ToyRenderer(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(state.shape[0], 9, 4, 4, device=state.device)
        logits[:, 7] = 8.0
        return logits


def _evidence(system: str = "blocket") -> Gate2Evidence:
    iou = (
        {"pendulumBob": 0.86}
        if system == "pendulum"
        else {"playerDisc": 0.76, "puckDisc": 0.74}
    )
    centroid_full = {name: 0.09 for name in iou}
    centroid_baseline = {name: 0.10 for name in iou}
    return Gate2Evidence(
        system_name=system,
        sample_count=512,
        horizon=8,
        current_foreground_iou=iou,
        full_horizon_centroid_error=centroid_full,
        unstructured_horizon_centroid_error=centroid_baseline,
        full_horizon_weighted_cross_entropy=0.80,
        unstructured_horizon_weighted_cross_entropy=0.82,
        shuffled_horizon_weighted_cross_entropy=0.90,
        test_sanitized_tensor_sha256=TEST_SHA256,
        class_weights_sha256=CLASS_WEIGHTS_SHA256,
        neural_hashes_before=NEURAL_HASHES,
        neural_hashes_after=NEURAL_HASHES,
    )


def _audit(evidence: Gate2Evidence):
    return audit_gate_2(
        evidence,
        expected_test_sanitized_tensor_sha256=TEST_SHA256,
        expected_class_weights_sha256=CLASS_WEIGHTS_SHA256,
        expected_neural_hashes=NEURAL_HASHES,
    )


class Gate2QualityTests(unittest.TestCase):
    def test_gate2_passes_only_the_complete_registered_conjunction(self) -> None:
        for system in ("pendulum", "blocket"):
            with self.subTest(system=system):
                result = _audit(_evidence(system))
                self.assertTrue(result.auditable, result.failures)
                self.assertTrue(result.passed, result.to_dict())

    def test_gate2_rejects_missing_object_and_changed_network(self) -> None:
        evidence = _evidence()
        missing = Gate2Evidence(
            **{**evidence.__dict__, "current_foreground_iou": {"playerDisc": 0.8}}
        )
        self.assertFalse(_audit(missing).auditable)
        changed = Gate2Evidence(
            **{**evidence.__dict__, "neural_hashes_after": {"model": "c" * 64}}
        )
        self.assertFalse(_audit(changed).auditable)

    def test_gate2_fails_each_quantitative_ablation(self) -> None:
        evidence = _evidence()
        cases = (
            {"current_foreground_iou": {"playerDisc": 0.69, "puckDisc": 0.74}},
            {
                "full_horizon_centroid_error": {
                    "playerDisc": 0.12,
                    "puckDisc": 0.09,
                }
            },
            {"full_horizon_weighted_cross_entropy": 0.92},
            {"shuffled_horizon_weighted_cross_entropy": 0.87},
        )
        for replacement in cases:
            with self.subTest(replacement=replacement):
                candidate = Gate2Evidence(**{**evidence.__dict__, **replacement})
                result = _audit(candidate)
                self.assertTrue(result.auditable, result.failures)
                self.assertFalse(result.passed)

    def test_gate2_rejects_forged_counts_and_nonfinite_values(self) -> None:
        evidence = _evidence()
        wrong_count = Gate2Evidence(**{**evidence.__dict__, "sample_count": 511})
        self.assertFalse(_audit(wrong_count).auditable)
        nonfinite = Gate2Evidence(
            **{
                **evidence.__dict__,
                "full_horizon_centroid_error": {
                    "playerDisc": float("nan"),
                    "puckDisc": 0.09,
                },
            }
        )
        self.assertFalse(_audit(nonfinite).auditable)

    def test_gate2_rejects_well_formed_but_forged_hashes(self) -> None:
        evidence = _evidence()
        forged_test = Gate2Evidence(
            **{**evidence.__dict__, "test_sanitized_tensor_sha256": "d" * 64}
        )
        self.assertFalse(_audit(forged_test).auditable)
        forged_weights = Gate2Evidence(
            **{**evidence.__dict__, "class_weights_sha256": "e" * 64}
        )
        self.assertFalse(_audit(forged_weights).auditable)
        forged_neural = Gate2Evidence(
            **{
                **evidence.__dict__,
                "neural_hashes_before": {"model": "f" * 64},
                "neural_hashes_after": {"model": "f" * 64},
            }
        )
        self.assertFalse(_audit(forged_neural).auditable)

    def test_collector_recomputes_test_hash_and_rejects_attached_pixels(self) -> None:
        pixels = torch.zeros(512, 10, 4, 4, dtype=torch.uint8)
        config = PixelDirectConfig(
            image_size=4,
            patch_size=2,
            palette_size=3,
            history_frames=2,
            pixel_embedding_size=2,
            hidden_size=4,
            depth=1,
            heads=1,
            mlp_ratio=1.0,
        )
        frozen = nn.Identity().eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
            collect_gate2_evidence(
                system_name="pendulum",
                test_pixels=pixels,
                test_sanitized_tensor_sha256="0" * 64,
                model_config=config,
                encoder=frozen,
                renderer=frozen,
                structured_dynamics=frozen,
                structured_inference=frozen,
                unstructured_encoder=nn.Identity().eval().requires_grad_(False),
                unstructured_renderer=nn.Identity().eval().requires_grad_(False),
                unstructured_dynamics=frozen,
                unstructured_inference=frozen,
                unstructured_write_field=nn.Identity().eval().requires_grad_(False),
                unstructured_response_frame=nn.Identity().eval().requires_grad_(False),
                class_weights=torch.ones(3),
            )

    def test_collector_captures_real_after_hashes_and_exact_shapes(self) -> None:
        pixels = torch.full((512, 10, 4, 4), 7, dtype=torch.uint8)
        config = PixelDirectConfig(
            image_size=4,
            patch_size=2,
            palette_size=9,
            history_frames=2,
            pixel_embedding_size=2,
            hidden_size=4,
            depth=1,
            heads=1,
            mlp_ratio=1.0,
        )
        modules = (
            _ToyEncoder(),
            _ToyRenderer(),
            _ToyDynamics(),
            _ToyInference(),
            _ToyEncoder(),
            _ToyRenderer(),
            _ToyDynamics(),
            _ToyInference(),
            nn.Identity(),
            nn.Identity(),
        )
        for module in modules:
            module.eval().requires_grad_(False)
        evidence = collect_gate2_evidence(
            system_name="pendulum",
            test_pixels=pixels,
            test_sanitized_tensor_sha256=sanitized_pixel_tensor_sha256(pixels),
            model_config=config,
            encoder=modules[0],
            renderer=modules[1],
            structured_dynamics=modules[2],
            structured_inference=modules[3],
            unstructured_encoder=modules[4],
            unstructured_renderer=modules[5],
            unstructured_dynamics=modules[6],
            unstructured_inference=modules[7],
            unstructured_write_field=modules[8],
            unstructured_response_frame=modules[9],
            class_weights=torch.ones(9),
            batch_size=64,
        )
        self.assertEqual(evidence.sample_count, 512)
        self.assertEqual(evidence.neural_hashes_before, evidence.neural_hashes_after)
        result = audit_gate_2(
            evidence,
            expected_test_sanitized_tensor_sha256=evidence.test_sanitized_tensor_sha256,
            expected_class_weights_sha256=evidence.class_weights_sha256,
            expected_neural_hashes=dict(evidence.neural_hashes_before),
        )
        self.assertTrue(result.auditable, result.failures)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["shuffled_innovations_degrade_horizon8"])
        with self.assertRaisesRegex(ValueError, "independent unstructured"):
            collect_gate2_evidence(
                system_name="pendulum",
                test_pixels=pixels,
                test_sanitized_tensor_sha256=sanitized_pixel_tensor_sha256(pixels),
                model_config=config,
                encoder=modules[0],
                renderer=modules[1],
                structured_dynamics=modules[2],
                structured_inference=modules[3],
                unstructured_encoder=modules[0],
                unstructured_renderer=modules[1],
                unstructured_dynamics=modules[2],
                unstructured_inference=modules[3],
                unstructured_write_field=modules[8],
                unstructured_response_frame=modules[9],
                class_weights=torch.ones(9),
            )
        frozen = nn.Identity().eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "detached"):
            collect_gate2_evidence(
                system_name="pendulum",
                test_pixels=torch.zeros(
                    512, 10, 4, 4, dtype=torch.float32, requires_grad=True
                ),
                test_sanitized_tensor_sha256="0" * 64,
                model_config=config,
                encoder=frozen,
                renderer=frozen,
                structured_dynamics=frozen,
                structured_inference=frozen,
                unstructured_encoder=nn.Identity().eval().requires_grad_(False),
                unstructured_renderer=nn.Identity().eval().requires_grad_(False),
                unstructured_dynamics=frozen,
                unstructured_inference=frozen,
                unstructured_write_field=nn.Identity().eval().requires_grad_(False),
                unstructured_response_frame=nn.Identity().eval().requires_grad_(False),
                class_weights=torch.ones(3),
            )


if __name__ == "__main__":
    unittest.main()
