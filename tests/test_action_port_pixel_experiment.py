from __future__ import annotations

import unittest

import torch

from blocket_league.action_port_pixel_experiment import (
    ActionCausalBranch,
    ActionPortHamiltonianCore,
    ActionPortPixelConfig,
    ImplicitStateRenderer,
    TangentMatchedActionCore,
    _branch_loss,
    _entity_dice_loss,
    action_vectors,
)
from blocket_league.data import make_clip
from blocket_league.env import BlocketLeagueEnv, WorldConfig
from blocket_league.port_hamiltonian_bottleneck import bottleneck_state


class ActionPortPixelExperimentTests(unittest.TestCase):
    def test_full_clip_arrays_preserve_existing_target_alignment(self) -> None:
        context_frames = 4
        clip = make_clip(
            104_729,
            context_frames=context_frames,
            future_frames=8,
            image_size=16,
        )
        self.assertEqual(clip["all_state"].shape[0], clip["frames"].shape[0])
        self.assertEqual(clip["all_actions"].shape[0], clip["frames"].shape[0])
        self.assertEqual(clip["all_events"].shape[0], clip["frames"].shape[0])
        self.assertTrue(
            torch.equal(
                torch.from_numpy(clip["all_state"][context_frames:]),
                torch.from_numpy(clip["state"]),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.from_numpy(clip["all_actions"][context_frames:]),
                torch.from_numpy(clip["actions"]),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.from_numpy(clip["all_events"][context_frames:]),
                torch.from_numpy(clip["events"]),
            )
        )

    def test_action_vectors_normalize_diagonal_controls(self) -> None:
        actions = action_vectors(torch.tensor([0, 1, 2, 3]))
        expected = torch.tensor(
            [[0.0, 0.0], [0.0, -1.0], [2**-0.5, -(2**-0.5)], [1.0, 0.0]]
        )
        self.assertTrue(torch.allclose(actions, expected, atol=1e-7))

    def test_structured_core_matches_smooth_simulator_transition(self) -> None:
        env = BlocketLeagueEnv(seed=17, config=WorldConfig(image_size=16))
        env.state.player_position[:] = (0.24, 0.28)
        env.state.player_velocity[:] = (0.08, -0.04)
        env.state.puck_position[:] = (0.72, 0.70)
        env.state.puck_velocity[:] = (-0.03, 0.02)
        before = bottleneck_state(torch.from_numpy(env.state.vector())[None])[:, :8]
        env.step(3)
        after = bottleneck_state(torch.from_numpy(env.state.vector())[None])[:, :8]
        predicted = ActionPortHamiltonianCore()(before, torch.tensor([[1.0, 0.0]]))
        self.assertTrue(torch.allclose(predicted, after, atol=2e-6))

    def test_tangent_control_matches_initial_value_capacity_and_jacobian(self) -> None:
        structured = ActionPortHamiltonianCore()
        control = TangentMatchedActionCore()
        self.assertEqual(
            sum(parameter.numel() for parameter in structured.parameters()),
            sum(parameter.numel() for parameter in control.parameters()),
        )
        generator = torch.Generator().manual_seed(113)
        state = torch.randn(7, 8, generator=generator)
        action = torch.randn(7, 2, generator=generator)
        weights = torch.randn(7, 8, generator=generator)
        structured_score = (structured(state, action) * weights).sum()
        control_score = (control(state, action) * weights).sum()
        self.assertTrue(
            torch.allclose(
                structured(state, action),
                control(state, action),
                atol=2e-6,
            )
        )
        structured_gradients = torch.autograd.grad(
            structured_score,
            tuple(structured.parameters()),
        )
        control_gradients = torch.autograd.grad(
            control_score,
            tuple(control.parameters()),
        )
        for structured_gradient, control_gradient in zip(
            structured_gradients,
            control_gradients,
        ):
            self.assertTrue(
                torch.allclose(structured_gradient, control_gradient, atol=2e-6)
            )

    def test_structured_domain_is_hard_but_tangent_control_can_leave_it(self) -> None:
        structured = ActionPortHamiltonianCore()
        control = TangentMatchedActionCore()
        with torch.no_grad():
            structured.raw_decay.add_(100.0)
            control.raw_decay.add_(100.0)
        self.assertTrue(bool((structured.coefficients()[1] <= 1.0).all()))
        self.assertTrue(bool((control.coefficients()[1] > 1.0).any()))

    def test_action_port_has_no_direct_puck_cross_talk(self) -> None:
        core = ActionPortHamiltonianCore()
        state = torch.randn(16, 8, generator=torch.Generator().manual_seed(127))
        action = torch.tensor([[1.0, -1.0]]).expand(16, -1)
        response = core(state, action) - core(state, torch.zeros_like(action))
        self.assertGreater(float(response[:, 0:2].abs().sum()), 0.0)
        self.assertGreater(float(response[:, 4:6].abs().sum()), 0.0)
        self.assertTrue(
            torch.equal(response[:, 2:4], torch.zeros_like(response[:, 2:4]))
        )
        self.assertTrue(
            torch.equal(response[:, 6:8], torch.zeros_like(response[:, 6:8]))
        )

    def test_implicit_renderer_depends_differentiably_on_state_only(self) -> None:
        renderer = ImplicitStateRenderer(
            image_size=16,
            palette_size=9,
            state_mean=torch.zeros(10),
            state_scale=torch.ones(10),
            hidden_size=16,
        )
        state = torch.randn(3, 10, generator=torch.Generator().manual_seed(131))
        state.requires_grad_(True)
        logits = renderer(state)
        self.assertEqual(logits.shape, (3, 9, 16, 16))
        logits.square().mean().backward()
        self.assertIsNotNone(state.grad)
        self.assertGreater(float(state.grad.abs().sum()), 0.0)

    def test_entity_dice_loss_rewards_correct_entity_masks(self) -> None:
        targets = torch.ones(2, 8, 8, dtype=torch.long)
        targets[:, 1:4, 1:4] = 5
        targets[:, 5:7, 5:7] = 7
        wrong = torch.zeros(2, 9, 8, 8)
        correct = wrong.clone()
        correct[:, 5, 1:4, 1:4] = 10.0
        correct[:, 7, 5:7, 5:7] = 10.0
        self.assertLess(
            float(_entity_dice_loss(correct, targets)),
            float(_entity_dice_loss(wrong, targets)),
        )

    def test_action_conditioned_branch_loss_is_finite(self) -> None:
        branch = ActionCausalBranch(
            torch.zeros(12),
            torch.ones(12),
            torch.zeros(10),
            torch.ones(10),
            hidden_size=16,
            structured=True,
        )
        generator = torch.Generator().manual_seed(137)
        features = torch.randn(8, 9, 12, generator=generator)
        targets = torch.randn(8, 9, 10, generator=generator)
        actions = torch.randn(8, 8, 2, generator=generator)
        labels = torch.randint(0, 6, (8, 8), generator=generator)
        loss, terms = _branch_loss(
            branch,
            features,
            targets,
            actions,
            labels,
            torch.ones(6),
            ActionPortPixelConfig(trajectories=8),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(
            set(terms),
            {"state", "teacher", "rollout", "event", "freePort"},
        )
        loss.backward()
        self.assertIsNotNone(branch.core.raw_action_gain.grad)
        self.assertIsNotNone(branch.hybrid_port.network[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
