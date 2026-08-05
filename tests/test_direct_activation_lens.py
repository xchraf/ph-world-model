from __future__ import annotations

import inspect
import math
import unittest

import torch

from blocket_league.direct_activation_lens import (
    ActivationWriteFieldConfig,
    FrozenSoftPixelActivationLens,
    StateConditionedActivationWriteField,
    basis_invariant_response_loss,
    direct_dynamics_pulse_responses,
    grassmann_response_loss,
    odd_symmetry_loss,
)
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class DirectActivationLensTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13_071)

    @staticmethod
    def _backbone() -> DirectPixelTransformer:
        return DirectPixelTransformer(
            PixelDirectConfig(
                image_size=4,
                patch_size=2,
                palette_size=3,
                history_frames=2,
                pixel_embedding_size=2,
                hidden_size=8,
                depth=2,
                heads=2,
                mlp_ratio=2.0,
            )
        )

    @staticmethod
    def _field() -> StateConditionedActivationWriteField:
        return StateConditionedActivationWriteField(
            ActivationWriteFieldConfig(
                latent_size=4,
                port_size=2,
                history_frames=2,
                patch_count=4,
                hidden_size=8,
                network_hidden_size=12,
            )
        )

    def test_hard_and_one_hot_soft_paths_are_the_same_frozen_transformer(self) -> None:
        backbone = self._backbone().eval()
        lens = FrozenSoftPixelActivationLens(backbone, intervention_block=0)
        pixels = torch.randint(0, 3, (2, 2, 4, 4))
        expected = backbone(pixels)
        actual = lens.soft_forward(pixels, return_logits=True)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))

    def test_state_conditioned_write_spans_every_token_and_is_orthonormal(self) -> None:
        field = self._field()
        latent = torch.randn(3, 4, requires_grad=True)
        basis = field(latent)
        self.assertEqual(basis.shape, (3, 2, 4, 8, 2))
        flat = basis.reshape(3, -1, 2)
        torch.testing.assert_close(
            flat.transpose(-1, -2) @ flat,
            torch.eye(2).expand(3, 2, 2),
            atol=2e-6,
            rtol=2e-6,
        )
        weighted = torch.linspace(0.1, 1.0, flat.shape[-2])[None, :, None]
        (flat * weighted).square().sum().backward()
        self.assertIsNotNone(latent.grad)
        self.assertTrue(bool(torch.isfinite(latent.grad).all()))
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)

    def test_soft_horizon_four_rollout_has_no_argmax_gradient_break(self) -> None:
        lens = FrozenSoftPixelActivationLens(
            self._backbone(),
            intervention_block=0,
            horizons=(1, 2, 4),
        )
        pixels = torch.randint(0, 3, (2, 2, 4, 4))
        basis = self._field()(torch.randn(2, 4))
        pulse = torch.randn(2, 2, requires_grad=True) * 0.03
        rollout = lens.rollout(pixels, basis, pulse)
        self.assertEqual(tuple(rollout), (1, 2, 4))
        self.assertEqual(rollout[4].shape, (2, 3, 4, 4))
        gradient = torch.autograd.grad(rollout[4][:, 0].mean(), pulse)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_response_jacobian_matches_central_difference_and_trains_field(self) -> None:
        lens = FrozenSoftPixelActivationLens(
            self._backbone(),
            intervention_block=0,
            horizons=(1,),
        )
        field = self._field()
        pixels = torch.randint(0, 3, (1, 2, 4, 4))
        basis = field(torch.randn(1, 4))

        response = lambda probabilities: probabilities.mean(dim=(-1, -2))
        result = lens.response_jacobians(
            pixels,
            basis,
            response=response,
            create_graph=True,
        )
        self.assertEqual(result.jacobians[1].shape, (1, 3, 2))
        epsilon = 2e-3
        direction = torch.tensor([[epsilon, 0.0]])
        plus = response(lens.rollout(pixels, basis, direction, horizons=(1,))[1])
        minus = response(lens.rollout(pixels, basis, -direction, horizons=(1,))[1])
        finite_difference = (plus - minus) / (2.0 * epsilon)
        torch.testing.assert_close(
            result.jacobians[1][..., 0],
            finite_difference,
            atol=2e-4,
            rtol=2e-2,
        )
        result.jacobians[1].square().sum().backward()
        final_weight = field.network[-1].weight
        self.assertIsNotNone(final_weight.grad)
        self.assertTrue(bool(torch.isfinite(final_weight.grad).all()))

    def test_direct_dynamics_responses_propagate_one_pulse_to_all_horizons(self) -> None:
        transition = torch.tensor([[0.8, 0.2], [0.0, 0.5]], dtype=torch.float64)
        port = torch.tensor([[1.0], [0.4]], dtype=torch.float64)

        def stepper(latent: torch.Tensor, effort: torch.Tensor) -> torch.Tensor:
            return latent @ transition.T + effort @ port.T

        initial = torch.randn(3, 2, dtype=torch.float64)
        result = direct_dynamics_pulse_responses(
            stepper,
            initial,
            1,
            horizons=(1, 2, 4),
        )
        for horizon in (1, 2, 4):
            expected = torch.linalg.matrix_power(transition, horizon - 1) @ port
            torch.testing.assert_close(
                result.jacobians[horizon],
                expected.expand(3, 2, 1),
                atol=2e-12,
                rtol=2e-12,
            )

    def test_response_losses_are_invariant_to_orthogonal_port_gauges(self) -> None:
        first = torch.randn(4, 7, 2, dtype=torch.float64)
        angle_a, angle_b = 0.31, -0.72
        rotation_a = torch.tensor(
            [[math.cos(angle_a), -math.sin(angle_a)],
             [math.sin(angle_a), math.cos(angle_a)]],
            dtype=torch.float64,
        )
        rotation_b = torch.tensor(
            [[math.cos(angle_b), -math.sin(angle_b)],
             [math.sin(angle_b), math.cos(angle_b)]],
            dtype=torch.float64,
        )
        left = first @ rotation_a
        right = 5.7 * first @ rotation_b
        torch.testing.assert_close(
            basis_invariant_response_loss(left, right),
            torch.zeros((), dtype=torch.float64),
            atol=2e-12,
            rtol=0.0,
        )
        torch.testing.assert_close(
            grassmann_response_loss(left, right),
            torch.zeros((), dtype=torch.float64),
            atol=2e-12,
            rtol=0.0,
        )

    def test_odd_symmetry_and_pixels_only_chart_proxies_are_finite(self) -> None:
        baseline = torch.randn(2, 3, 4, 4)
        increment = torch.randn_like(baseline)
        torch.testing.assert_close(
            odd_symmetry_loss(baseline + increment, baseline - increment, baseline),
            torch.zeros(()),
            atol=1e-12,
            rtol=0.0,
        )

        lens = FrozenSoftPixelActivationLens(self._backbone(), intervention_block=0)
        pixels = torch.randint(0, 3, (1, 2, 4, 4))
        basis = self._field()(torch.randn(1, 4))
        proxies = lens.intervention_proxies(pixels, basis, amplitude=0.02)
        for value in (
            proxies.odd_symmetry,
            proxies.current_frame_leakage,
            proxies.manifold_cycle,
            proxies.first_order_signal,
        ):
            self.assertTrue(bool(torch.isfinite(value)))
            self.assertGreaterEqual(float(value), 0.0)

    def test_public_interfaces_do_not_accept_external_labels_or_token_masks(self) -> None:
        forbidden = {"action", "control", "physical_state", "entity_mask", "object_mask"}
        callables = (
            StateConditionedActivationWriteField.forward,
            FrozenSoftPixelActivationLens.soft_forward,
            FrozenSoftPixelActivationLens.rollout,
            FrozenSoftPixelActivationLens.response_jacobians,
            direct_dynamics_pulse_responses,
        )
        for callable_object in callables:
            parameters = set(inspect.signature(callable_object).parameters)
            self.assertTrue(parameters.isdisjoint(forbidden), parameters & forbidden)


if __name__ == "__main__":
    unittest.main()
