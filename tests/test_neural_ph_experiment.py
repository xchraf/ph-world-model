from __future__ import annotations

import unittest

import numpy as np
import torch

from blocket_league.data import make_excitation_clip
from blocket_league.neural_ph_experiment import (
    NeuralPHBranch,
    NeuralPHExperimentConfig,
    _energy_gauge_loss,
    _matched_control_hidden_size,
    _parameter_count,
)
from blocket_league.neural_port_hamiltonian import NeuralODE
from blocket_league.action_port_pixel_experiment import _branch_loss


class NeuralPHExperimentTests(unittest.TestCase):
    def test_excitation_clips_are_reproducible_and_family_specific(self) -> None:
        cardinal_a = make_excitation_clip(
            17_071, context_frames=4, future_frames=8, image_size=16,
            action_family="cardinal",
        )
        cardinal_b = make_excitation_clip(
            17_071, context_frames=4, future_frames=8, image_size=16,
            action_family="cardinal",
        )
        for name in ("frames", "all_state", "all_actions", "all_events"):
            np.testing.assert_array_equal(cardinal_a[name], cardinal_b[name])
        self.assertTrue(set(cardinal_a["all_actions"]).issubset({0, 1, 3, 5, 7}))
        diagonal = make_excitation_clip(
            17_071, context_frames=4, future_frames=8, image_size=16,
            action_family="diagonal",
        )
        self.assertTrue(set(diagonal["all_actions"]).issubset({0, 2, 4, 6, 8}))
        reversal = make_excitation_clip(
            17_071, context_frames=4, future_frames=8, image_size=16,
            action_family="reversal",
        )
        nonzero = reversal["all_actions"][1:]
        self.assertTrue(bool(np.all(nonzero[1:] != nonzero[:-1])))

    def test_generic_branch_loss_updates_all_four_structured_functions(self) -> None:
        torch.manual_seed(17_077)
        branch = NeuralPHBranch(
            torch.zeros(12),
            torch.ones(12),
            torch.zeros(10),
            torch.ones(10),
            hidden_size=16,
            hidden_layers=2,
            integration_method="midpoint",
            integration_substeps=1,
            resistance_floor=1e-5,
            structured=True,
        )
        features = torch.randn(4, 9, 12)
        targets = torch.randn(4, 9, 10)
        actions = torch.randn(4, 8, 2)
        labels = torch.randint(0, 6, (4, 8))
        config = NeuralPHExperimentConfig(
            fit_policy_trajectories=2,
            fit_cardinal_trajectories=2,
            test_policy_trajectories=1,
            test_diagonal_trajectories=1,
            test_reversal_trajectories=1,
            dynamics_batch_size=4,
            energy_probe_size=4,
        )
        loss, _ = _branch_loss(
            branch,
            features,
            targets,
            actions,
            labels,
            torch.ones(6),
            config,
        )
        loss = loss + config.energy_gradient_weight * _energy_gauge_loss(
            branch, branch.encode(features[:, 0])[:, :8]
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
                any(float(gradient.abs().sum()) > 0 for gradient in gradients if gradient is not None),
                name,
            )

    def test_control_capacity_matching_is_within_one_percent(self) -> None:
        branch = NeuralPHBranch(
            torch.zeros(12),
            torch.ones(12),
            torch.zeros(10),
            torch.ones(10),
            hidden_size=32,
            hidden_layers=2,
            integration_method="midpoint",
            integration_substeps=1,
            resistance_floor=1e-5,
            structured=True,
        )
        target = _parameter_count(branch.core)
        hidden = _matched_control_hidden_size(target, branch.core.config)
        control = NeuralODE(
            branch.core.config.__class__(
                **{**branch.core.config.__dict__, "hidden_size": hidden}
            )
        )
        relative_gap = abs(_parameter_count(control) - target) / target
        self.assertLess(relative_gap, 0.01)


if __name__ == "__main__":
    unittest.main()
