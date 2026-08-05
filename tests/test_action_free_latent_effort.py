from __future__ import annotations

import unittest

import torch

from blocket_league.action_free_latent_effort import (
    LatentEffortConfig,
    LatentEffortInference,
    UnstructuredLatentEffortDynamics,
    latent_effort_statistics,
)


class LatentEffortTests(unittest.TestCase):
    def test_identity_transition_has_exactly_zero_effort(self) -> None:
        model = LatentEffortInference(
            LatentEffortConfig(state_size=4, effort_size=2, hidden_size=16)
        )
        state = torch.randn(7, 4)
        torch.testing.assert_close(model(state, state), torch.zeros(7, 2))

    def test_transition_inference_is_differentiable_without_labels(self) -> None:
        model = LatentEffortInference(
            LatentEffortConfig(state_size=4, effort_size=2, hidden_size=16)
        )
        present = torch.randn(5, 4, requires_grad=True)
        successor = torch.randn(5, 4, requires_grad=True)
        value = model(present, successor)
        value.square().sum().backward()
        self.assertTrue(torch.isfinite(present.grad).all())
        self.assertTrue(torch.isfinite(successor.grad).all())
        self.assertGreater(float(successor.grad.abs().sum()), 0.0)

    def test_statistics_fix_mean_variance_and_mixing_gauge(self) -> None:
        latent = torch.randn(8, 5, 2, requires_grad=True)
        statistics = latent_effort_statistics(latent)
        self.assertEqual(
            set(statistics),
            {"mean", "variance", "decorrelation", "temporal", "total"},
        )
        statistics["total"].backward()
        self.assertTrue(torch.isfinite(latent.grad).all())

    def test_unstructured_baseline_has_separate_drift_and_port(self) -> None:
        model = UnstructuredLatentEffortDynamics(4, 2, 12, dt=0.05)
        state = torch.randn(3, 4)
        zero = torch.zeros(3, 2)
        torch.testing.assert_close(model.vector_field(state, zero), model.drift(state))
        self.assertEqual(model.port(state).shape, (3, 4, 2))


if __name__ == "__main__":
    unittest.main()
