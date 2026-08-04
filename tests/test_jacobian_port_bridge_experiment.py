from __future__ import annotations

import unittest

import torch

from blocket_league.jacobian_port_bridge_experiment import (
    _alignment_summary,
    encoder_jacobian,
    lift_player_write,
)
from blocket_league.neural_ph_experiment import NeuralPHBranch


class JacobianPortBridgeTests(unittest.TestCase):
    def _branch(self) -> NeuralPHBranch:
        torch.manual_seed(19_077)
        return NeuralPHBranch(
            torch.randn(6),
            torch.rand(6) + 0.2,
            torch.randn(10),
            torch.rand(10) + 0.2,
            hidden_size=8,
            hidden_layers=1,
            integration_method="midpoint",
            integration_substeps=1,
            resistance_floor=1e-5,
            structured=True,
        )

    def test_exact_encoder_jacobian_matches_autograd(self) -> None:
        branch = self._branch()
        features = torch.randn(3, 6, requires_grad=True)
        analytic = encoder_jacobian(branch)
        rows = []
        encoded = branch.encode(features)
        for output in range(encoded.shape[1]):
            rows.append(
                torch.autograd.grad(
                    encoded[:, output].sum(), features, retain_graph=True,
                )[0][0]
            )
        actual = torch.stack(rows)
        torch.testing.assert_close(analytic, actual)

    def test_player_write_lift_respects_shared_patch(self) -> None:
        directions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        lifted = lift_player_write(directions, torch.tensor([False, True]))
        torch.testing.assert_close(lifted[0, :, :2], directions)
        torch.testing.assert_close(lifted[0, :, 2:], torch.zeros_like(directions))
        torch.testing.assert_close(lifted[1, :, :2], directions)
        torch.testing.assert_close(lifted[1, :, 2:], directions)

    def test_alignment_summary_detects_correct_axes(self) -> None:
        port = torch.zeros(5, 2, 8)
        port[:, 0, 4] = 1.0
        port[:, 1, 5] = 1.0
        summary = _alignment_summary(
            port.clone(), port, target_coordinates=(4, 5)
        )
        self.assertAlmostEqual(
            summary["matchedCosine"]["pooled"]["mean"], 1.0, places=6
        )
        self.assertAlmostEqual(
            summary["absoluteSwappedAxisCosine"]["mean"], 0.0, places=6
        )
        self.assertEqual(summary["targetCoordinateSignFraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
