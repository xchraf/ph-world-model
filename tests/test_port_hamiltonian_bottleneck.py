from __future__ import annotations

import unittest

import torch

from blocket_league.port_hamiltonian_bottleneck import (
    BottleneckExperimentConfig,
    CausalBottleneckBranch,
    PortHamiltonianFreeCore,
    SignFreeMatchedCore,
    _branch_loss,
    bottleneck_state,
    regime_labels,
)


class PortHamiltonianBottleneckTests(unittest.TestCase):
    def test_bottleneck_state_contains_canonical_state_and_hybrid_mode(self) -> None:
        state = torch.tensor(
            [[0.2, 0.3, 0.4, -0.5, 0.7, 0.8, -0.2, 0.1, 6.0, 3.5]]
        )
        result = bottleneck_state(state)
        expected = torch.tensor(
            [[0.2, 0.3, 0.7, 0.8, 0.72, -0.9, -0.2, 0.1, 0.25, 0.5]]
        )
        self.assertTrue(torch.allclose(result, expected))

    def test_regime_labels_distinguish_goal_entry_and_pause(self) -> None:
        states_t = torch.zeros(6, 10)
        states_tp1 = torch.zeros(6, 10)
        events = torch.tensor([0, 2, 3, 4, 4, 5])
        states_tp1[3, 8] = 1
        states_t[4, 9] = 3
        states_tp1[4, 9] = 2
        self.assertEqual(regime_labels(states_t, states_tp1, events).tolist(), list(range(6)))

    def test_passive_core_cannot_increase_unforced_momentum_energy(self) -> None:
        core = PortHamiltonianFreeCore()
        state = torch.randn(128, 8, generator=torch.Generator().manual_seed(71))
        following = core(state)
        gain, decay = core.coefficients()
        self.assertTrue(bool((gain > 0).all()))
        self.assertTrue(bool((decay >= 0).all() and (decay <= 1).all()))
        self.assertTrue(
            bool(
                (
                    following[:, 4:].square().sum(dim=-1)
                    <= state[:, 4:].square().sum(dim=-1) + 1e-6
                ).all()
            )
        )

    def test_external_port_has_direct_canonical_momentum_response(self) -> None:
        core = PortHamiltonianFreeCore()
        state = torch.randn(16, 8, generator=torch.Generator().manual_seed(73))
        port = torch.zeros(16, 4)
        port[:, 2] = 0.05
        response = core(state, port) - core(state)
        self.assertTrue(torch.allclose(response[:, 4:8], port, atol=1e-7))
        self.assertTrue(torch.allclose(response[:, :4], torch.zeros_like(response[:, :4])))

    def test_sign_free_control_has_equal_core_capacity_and_initial_map(self) -> None:
        structured = PortHamiltonianFreeCore()
        control = SignFreeMatchedCore()
        self.assertEqual(
            sum(parameter.numel() for parameter in structured.parameters()),
            sum(parameter.numel() for parameter in control.parameters()),
        )
        state = torch.randn(32, 8, generator=torch.Generator().manual_seed(79))
        self.assertTrue(torch.allclose(structured(state), control(state), atol=2e-6))

    def test_branches_have_equal_capacity_and_support_hard_rollout(self) -> None:
        feature_mean = torch.zeros(12)
        feature_scale = torch.ones(12)
        state_mean = torch.zeros(10)
        state_scale = torch.ones(10)
        structured = CausalBottleneckBranch(
            feature_mean,
            feature_scale,
            state_mean,
            state_scale,
            hidden_size=16,
            structured=True,
        )
        control = CausalBottleneckBranch(
            feature_mean,
            feature_scale,
            state_mean,
            state_scale,
            hidden_size=16,
            structured=False,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in structured.parameters()),
            sum(parameter.numel() for parameter in control.parameters()),
        )
        features = torch.randn(4, 12, generator=torch.Generator().manual_seed(83))
        current = structured.encode(features)
        for _ in range(8):
            current, logits, jump, gate = structured.step(current)
        self.assertEqual(current.shape, (4, 10))
        self.assertEqual(logits.shape, (4, 6))
        self.assertEqual(jump.shape, (4, 10))
        self.assertEqual(gate.shape, (4, 1))
        current.square().mean().backward()
        self.assertIsNotNone(structured.encoder.weight.grad)

    def test_paired_training_loss_is_finite(self) -> None:
        feature_mean = torch.zeros(12)
        feature_scale = torch.ones(12)
        state_mean = torch.zeros(10)
        state_scale = torch.ones(10)
        branch = CausalBottleneckBranch(
            feature_mean,
            feature_scale,
            state_mean,
            state_scale,
            hidden_size=16,
            structured=True,
        )
        generator = torch.Generator().manual_seed(89)
        features = torch.randn(8, 5, 12, generator=generator)
        targets = torch.randn(8, 5, 10, generator=generator)
        labels = torch.randint(0, 6, (8, 4), generator=generator)
        loss, terms = _branch_loss(
            branch,
            features,
            targets,
            labels,
            torch.ones(6),
            BottleneckExperimentConfig(transitions_per_trajectory=4),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(set(terms), {"state", "teacher", "rollout", "event", "freePort"})
        loss.backward()
        self.assertIsNotNone(branch.hybrid_port.network[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
