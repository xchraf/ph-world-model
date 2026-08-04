from __future__ import annotations

import unittest

import torch

from blocket_league.neural_port_hamiltonian import NeuralPortHamiltonianConfig
from blocket_league.pixel_only_ph_experiment import (
    GenericSpatialRenderer,
    GenericStateEncoder,
    PixelOnlyDynamicsBranch,
    PixelOnlyPHConfig,
    _concatenate_training_suites,
    _counterfactual_truth,
    _fit_ridge,
    _ridge_jacobian,
    pixel_only_branch_loss,
)


class PixelOnlyPHExperimentTests(unittest.TestCase):
    def _branch(self, *, structured: bool = True) -> PixelOnlyDynamicsBranch:
        encoder = GenericStateEncoder(
            torch.zeros(12), torch.ones(12), state_size=8, hidden_size=16
        )
        renderer = GenericSpatialRenderer(
            state_size=8, image_size=8, palette_size=9, hidden_size=16
        )
        return PixelOnlyDynamicsBranch(
            encoder,
            renderer,
            core_config=NeuralPortHamiltonianConfig(
                state_size=8,
                input_size=2,
                hidden_size=16,
                hidden_layers=1,
                dt=0.05,
            ),
            structured=structured,
        )

    def test_training_suite_rejects_any_physical_label_tensor(self) -> None:
        suite = {
            "features": torch.randn(2, 9, 12),
            "frames": torch.zeros(2, 9, 8, 8, dtype=torch.uint8),
            "actions": torch.zeros(2, 8, dtype=torch.long),
            "actionVectors": torch.zeros(2, 8, 2),
        }
        merged = _concatenate_training_suites(suite, suite)
        self.assertEqual(set(merged), set(suite))
        contaminated = {**suite, "worldStates": torch.randn(2, 9, 10)}
        with self.assertRaises(AssertionError):
            _concatenate_training_suites(contaminated)

    def test_generic_renderer_has_no_semantic_state_slots_and_is_differentiable(self) -> None:
        renderer = GenericSpatialRenderer(8, 8, 9, 16)
        state = torch.randn(3, 8, requires_grad=True)
        logits = renderer(state)
        self.assertEqual(logits.shape, (3, 9, 8, 8))
        logits.square().mean().backward()
        self.assertGreater(float(state.grad.abs().sum()), 0.0)

    def test_pixel_action_only_loss_updates_all_structured_functions(self) -> None:
        torch.manual_seed(23_071)
        branch = self._branch()
        features = torch.randn(2, 9, 12)
        frames = torch.randint(0, 9, (2, 9, 8, 8))
        actions = torch.randn(2, 8, 2).clamp(-1, 1)
        loss, terms = pixel_only_branch_loss(
            branch,
            features,
            frames,
            actions,
            torch.ones(9),
            PixelOnlyPHConfig(
                fit_policy_trajectories=1,
                fit_cardinal_trajectories=1,
                test_policy_trajectories=1,
                test_diagonal_trajectories=1,
                test_reversal_trajectories=1,
                audit_trajectories=2,
                state_size=8,
                dynamics_batch_size=2,
            ),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(
            set(terms),
            {
                "reconstruction",
                "teacherLatent",
                "rolloutLatent",
                "rolloutPixel",
                "actionContrast",
                "whitening",
                "energyGauge",
            },
        )
        loss.backward()
        for name in (
            "energy_network",
            "interconnection_network",
            "resistance_network",
            "port_network",
        ):
            gradients = [
                parameter.grad for parameter in getattr(branch.core, name).parameters()
            ]
            self.assertTrue(any(gradient is not None for gradient in gradients), name)
            self.assertTrue(
                any(
                    float(gradient.abs().sum()) > 0
                    for gradient in gradients
                    if gradient is not None
                ),
                name,
            )

    def test_ridge_jacobian_matches_affine_autograd(self) -> None:
        torch.manual_seed(23_077)
        inputs = torch.randn(64, 8)
        targets = inputs @ torch.randn(8, 8) + torch.randn(8)
        fit = _fit_ridge(inputs, targets)
        jacobian = _ridge_jacobian(fit)
        probe = torch.randn(1, 8, requires_grad=True)
        from blocket_league.pixel_only_ph_experiment import _ridge_predict

        prediction = _ridge_predict(fit, probe)
        actual = torch.stack(
            [
                torch.autograd.grad(
                    prediction[:, output].sum(), probe, retain_graph=True
                )[0][0]
                for output in range(8)
            ]
        )
        torch.testing.assert_close(jacobian, actual)

    def test_simulator_counterfactual_axes_have_expected_player_momentum_sign(self) -> None:
        state = torch.tensor(
            [[0.30, 0.30, 0.0, 0.0, 0.70, 0.70, 0.0, 0.0, 0.0, 0.0]]
        )
        effect, plus, minus = _counterfactual_truth(state)
        self.assertEqual(effect.shape, (1, 2, 8))
        self.assertEqual(plus.shape, (1, 2, 64, 64))
        self.assertEqual(minus.shape, (1, 2, 64, 64))
        self.assertGreater(float(effect[0, 0, 4]), 0.0)
        self.assertGreater(float(effect[0, 1, 5]), 0.0)


if __name__ == "__main__":
    unittest.main()
