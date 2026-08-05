from __future__ import annotations

import math
import unittest

import torch

from blocket_league.cotangent_jacobian_ports import (
    cotangent_pullback_solve,
    grassmannian_loss,
    orthonormal_subspace_basis,
    poisson_sharp,
    principal_angles,
    pullback_compatibility_residual,
    subspace_projector,
)


class CotangentJacobianPortTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8_205)

    def test_canonical_poisson_sharp_maps_dq_to_positive_p(self) -> None:
        canonical_poisson = torch.tensor(
            [[0.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float64,
        )
        dq = torch.tensor([1.0, 0.0], dtype=torch.float64)
        force_direction = poisson_sharp(canonical_poisson, dq)
        torch.testing.assert_close(
            force_direction,
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            atol=0.0,
            rtol=0.0,
        )

    def test_pullback_and_poisson_sharp_are_coordinate_covariant(self) -> None:
        dtype = torch.float64
        jacobian = torch.randn(3, 6, dtype=dtype)
        activation_covector = torch.randn(6, dtype=dtype)
        regularizer_metric = torch.diag(torch.tensor([0.8, 1.3, 2.1], dtype=dtype))
        ridge = 0.07

        state_covector = cotangent_pullback_solve(
            jacobian,
            activation_covector,
            ridge=ridge,
            regularizer_metric=regularizer_metric,
        )

        coordinate_change = torch.tensor(
            [[1.2, 0.2, -0.1], [0.1, 0.9, 0.3], [0.0, -0.2, 1.1]],
            dtype=dtype,
        )
        transformed_jacobian = coordinate_change @ jacobian
        transformed_metric = (
            coordinate_change @ regularizer_metric @ coordinate_change.T
        )
        transformed_covector = cotangent_pullback_solve(
            transformed_jacobian,
            activation_covector,
            ridge=ridge,
            regularizer_metric=transformed_metric,
        )
        expected_covector = torch.linalg.solve(coordinate_change.T, state_covector)
        torch.testing.assert_close(
            transformed_covector,
            expected_covector,
            atol=2e-11,
            rtol=2e-11,
        )

        raw = torch.randn(3, 3, dtype=dtype)
        poisson = raw - raw.T
        transformed_poisson = coordinate_change @ poisson @ coordinate_change.T
        original_port = poisson_sharp(poisson, state_covector)
        transformed_port = poisson_sharp(transformed_poisson, transformed_covector)
        torch.testing.assert_close(
            transformed_port,
            coordinate_change @ original_port,
            atol=2e-11,
            rtol=2e-11,
        )

        original_residual = pullback_compatibility_residual(
            jacobian,
            state_covector,
            activation_covector,
        )
        transformed_residual = pullback_compatibility_residual(
            transformed_jacobian,
            transformed_covector,
            activation_covector,
        )
        torch.testing.assert_close(
            transformed_residual,
            original_residual,
            atol=2e-12,
            rtol=2e-12,
        )

    def test_ridge_is_stable_for_a_nearly_rank_deficient_jacobian(self) -> None:
        dtype = torch.float64
        jacobian = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1e-14, 0.0]],
            dtype=dtype,
        )
        activation_covector = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        state_covector = cotangent_pullback_solve(
            jacobian,
            activation_covector,
            ridge=1e-4,
        )
        self.assertTrue(bool(torch.isfinite(state_covector).all()))
        self.assertLess(float(state_covector.norm()), 2.0)

        expected = torch.linalg.solve(
            jacobian @ jacobian.T + 1e-4 * torch.eye(2, dtype=dtype),
            jacobian @ activation_covector,
        )
        torch.testing.assert_close(state_covector, expected, atol=1e-14, rtol=1e-14)

        residual = pullback_compatibility_residual(
            jacobian,
            state_covector,
            activation_covector,
        )
        self.assertTrue(bool(torch.isfinite(residual)))
        self.assertGreater(float(residual), 0.8)

    def test_batched_covector_families_preserve_shape_and_exact_pullback(self) -> None:
        dtype = torch.float64
        jacobian = torch.randn(4, 3, 7, dtype=dtype)
        true_state_covectors = torch.randn(4, 3, 2, dtype=dtype)
        activation_covectors = jacobian.transpose(-1, -2) @ true_state_covectors
        recovered = cotangent_pullback_solve(
            jacobian,
            activation_covectors,
            ridge=0.0,
        )
        self.assertEqual(recovered.shape, (4, 3, 2))
        torch.testing.assert_close(recovered, true_state_covectors, atol=2e-11, rtol=2e-11)
        residual = pullback_compatibility_residual(
            jacobian,
            recovered,
            activation_covectors,
        )
        self.assertEqual(residual.shape, (4, 2))
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            atol=2e-14,
            rtol=0.0,
        )

    def test_subspace_tools_are_basis_invariant_and_recover_known_angle(self) -> None:
        dtype = torch.float64
        first = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=dtype,
        )
        basis_change = torch.tensor([[2.0, -0.3], [0.4, 1.7]], dtype=dtype)
        same_span = first @ basis_change
        torch.testing.assert_close(
            subspace_projector(first),
            subspace_projector(same_span),
            atol=2e-12,
            rtol=2e-12,
        )
        torch.testing.assert_close(
            grassmannian_loss(first, same_span),
            torch.zeros((), dtype=dtype),
            atol=2e-14,
            rtol=0.0,
        )

        angle = 0.37
        second = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, math.cos(angle)],
                [0.0, math.sin(angle)],
                [0.0, 0.0],
            ],
            dtype=dtype,
        )
        measured = principal_angles(first, second)
        torch.testing.assert_close(
            measured,
            torch.tensor([0.0, angle], dtype=dtype),
            atol=2e-12,
            rtol=2e-12,
        )
        expected_loss = torch.tensor(math.sin(angle) ** 2 / 2.0, dtype=dtype)
        torch.testing.assert_close(
            grassmannian_loss(first, second),
            expected_loss,
            atol=2e-12,
            rtol=2e-12,
        )

        basis = orthonormal_subspace_basis(torch.randn(2, 7, 3, dtype=dtype))
        identity = basis.transpose(-1, -2) @ basis
        torch.testing.assert_close(
            identity,
            torch.eye(3, dtype=dtype).expand(2, 3, 3),
            atol=2e-12,
            rtol=2e-12,
        )

    def test_all_training_primitives_have_finite_gradients(self) -> None:
        dtype = torch.float64
        jacobian = torch.randn(2, 3, 6, dtype=dtype, requires_grad=True)
        activation_covectors = torch.randn(2, 6, 2, dtype=dtype, requires_grad=True)
        raw_poisson = torch.randn(2, 3, 3, dtype=dtype, requires_grad=True)
        first_span = torch.randn(2, 5, 2, dtype=dtype, requires_grad=True)
        second_span = torch.randn(2, 5, 2, dtype=dtype, requires_grad=True)

        state_covectors = cotangent_pullback_solve(
            jacobian,
            activation_covectors,
            ridge=3e-3,
        )
        poisson = raw_poisson - raw_poisson.transpose(-1, -2)
        ports = poisson_sharp(poisson, state_covectors)
        compatibility = pullback_compatibility_residual(
            jacobian,
            state_covectors,
            activation_covectors,
        )
        loss = (
            ports.square().mean()
            + compatibility.square().mean()
            + grassmannian_loss(first_span, second_span)
        )
        loss.backward()

        for tensor in (
            jacobian,
            activation_covectors,
            raw_poisson,
            first_span,
            second_span,
        ):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
