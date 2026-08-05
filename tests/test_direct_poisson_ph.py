from __future__ import annotations

import inspect
import unittest

import torch

from blocket_league.direct_poisson_ph import (
    DirectPoissonPHConfig,
    DirectPoissonPortHamiltonian,
    LatentEffortEncoder,
    LatentEffortEncoderConfig,
)


class DirectPoissonPortHamiltonianTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(84_051)
        self.model = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=4,
                port_size=2,
                hidden_size=12,
                hidden_layers=1,
                coupling_layers=4,
                dt=0.015,
                implicit_iterations=48,
                implicit_relaxation=0.85,
                implicit_tolerance=1e-13,
                discrete_gradient_epsilon=1e-18,
            )
        ).double()
        self.state = 0.25 * torch.randn(3, 4, dtype=torch.float64)
        self.latent_effort = 0.2 * torch.randn(3, 2, dtype=torch.float64)

    def test_chart_is_invertible_and_components_have_structural_guarantees(self) -> None:
        base = torch.randn(5, 4, dtype=torch.float64)
        image = self.model.coordinate_map(base)
        torch.testing.assert_close(
            self.model.coordinate_map.inverse(image), base, atol=2e-12, rtol=2e-12
        )

        energy, gradient, interconnection, resistance, port = self.model.components(
            self.state, create_graph=False
        )
        self.assertEqual(energy.shape, (3,))
        self.assertEqual(gradient.shape, (3, 4))
        self.assertEqual(interconnection.shape, (3, 4, 4))
        self.assertEqual(resistance.shape, (3, 4, 4))
        self.assertEqual(port.shape, (3, 4, 2))
        torch.testing.assert_close(
            interconnection + interconnection.transpose(-1, -2),
            torch.zeros_like(interconnection),
            atol=2e-13,
            rtol=2e-13,
        )
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(resistance).min()), -2e-13)

        shifted = self.state + 0.7
        shifted_components = self.model.components(shifted, create_graph=False)
        for index, name in ((0, "H"), (2, "J"), (3, "R"), (4, "B")):
            difference = (shifted_components[index] - (energy, gradient, interconnection, resistance, port)[index]).abs().max()
            self.assertGreater(float(difference), 1e-9, name)

    def test_pushforward_tensor_satisfies_jacobi_tightly(self) -> None:
        jacobi = self.model.jacobi_tensor(self.state[:1], create_graph=False)
        self.assertTrue(bool(torch.isfinite(jacobi).all()))
        self.assertLess(float(jacobi.abs().max()), 2e-9)

    def test_degenerate_poisson_supports_odd_states_and_casimirs(self) -> None:
        model = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=5,
                port_size=2,
                poisson_rank=4,
                hidden_size=10,
                hidden_layers=1,
                coupling_layers=4,
            )
        ).double()
        states = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
        interconnection = model.interconnection(states)
        self.assertEqual(model.config.poisson_rank, 4)
        self.assertTrue(bool((torch.linalg.matrix_rank(interconnection) == 4).all()))
        jacobi = model.jacobi_tensor(states[:1], create_graph=False)
        self.assertLess(float(jacobi.abs().max()), 3e-9)
        base = model.coordinate_map.inverse(states)
        # The final canonical coordinate is a Casimir.  Its pullback gradient
        # is annihilated by the transported Poisson sharp map.
        casimir = model.coordinate_map.inverse(states)[..., -1]
        casimir_gradient = torch.autograd.grad(casimir.sum(), states)[0]
        annihilated = torch.einsum("bij,bj->bi", interconnection, casimir_gradient)
        torch.testing.assert_close(
            annihilated, torch.zeros_like(annihilated), atol=2e-9, rtol=2e-9
        )
        self.assertEqual(base.shape, states.shape)

    def test_resistance_can_represent_exact_zero_dissipation_modes(self) -> None:
        with torch.no_grad():
            for parameter in self.model.resistance_network.parameters():
                parameter.zero_()
        resistance = self.model.resistance(self.state)
        torch.testing.assert_close(
            resistance, torch.zeros_like(resistance), atol=0.0, rtol=0.0
        )

    def _assert_discrete_balance(self, latent_effort: torch.Tensor) -> None:
        result = self.model.audited_step(self.state, latent_effort)
        self.assertTrue(bool(torch.isfinite(result.next_state).all()))
        self.assertLess(float(result.chain_rule_defect.abs().max()), 2e-11)
        self.assertLess(float(result.implicit_residual_norm.max()), 2e-10)
        self.assertLess(float(result.balance_defect.abs().max()), 2e-10)
        self.assertGreaterEqual(float(result.dissipated_energy.min()), -2e-13)

    def test_discrete_energy_balance_without_latent_effort(self) -> None:
        zeros = torch.zeros_like(self.latent_effort)
        self._assert_discrete_balance(zeros)
        result = self.model.audited_step(self.state, zeros)
        torch.testing.assert_close(
            result.supplied_energy,
            torch.zeros_like(result.supplied_energy),
            atol=0.0,
            rtol=0.0,
        )
        self.assertLessEqual(float(result.energy_delta.max()), 2e-10)

    def test_discrete_energy_balance_with_latent_effort(self) -> None:
        self._assert_discrete_balance(self.latent_effort)

    def test_no_grad_inference_remains_available(self) -> None:
        with torch.no_grad():
            result = self.model.audited_step(self.state, self.latent_effort)
        self.assertFalse(result.next_state.requires_grad)
        self.assertTrue(bool(torch.isfinite(result.next_state).all()))
        self.assertLess(float(result.balance_defect.abs().max()), 2e-10)

    def test_all_structural_networks_receive_finite_gradients(self) -> None:
        result = self.model.audited_step(self.state, self.latent_effort)
        loss = result.next_state.square().mean() + 0.1 * result.energy_after.mean()
        loss.backward()
        groups = {
            "H": [self.model.energy_curvature, *self.model.energy_network.parameters()],
            "J": list(self.model.coordinate_map.parameters()),
            "R": list(self.model.resistance_network.parameters()),
            "B": list(self.model.port_network.parameters()),
        }
        for name, parameters in groups.items():
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients), name)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0, name)

    def test_feature_only_effort_encoder_and_public_inference_names(self) -> None:
        encoder = LatentEffortEncoder(
            LatentEffortEncoderConfig(
                feature_size=7,
                port_size=2,
                hidden_size=11,
                hidden_layers=1,
            )
        ).double()
        previous = torch.randn(6, 7, dtype=torch.float64, requires_grad=True)
        following = torch.randn(6, 7, dtype=torch.float64, requires_grad=True)
        posterior = encoder(previous, following)
        self.assertEqual(posterior.mean.shape, (6, 2))
        self.assertEqual(posterior.log_scale.shape, (6, 2))
        sample = posterior.rsample(torch.zeros_like(posterior.mean))
        (sample.square().mean() + posterior.standard_normal_kl().mean()).backward()
        self.assertIsNotNone(previous.grad)
        self.assertIsNotNone(following.grad)

        public_methods = (
            self.model.forward,
            self.model.vector_field,
            self.model.step,
            self.model.audited_step,
            encoder.forward,
        )
        for method in public_methods:
            parameter_names = tuple(inspect.signature(method).parameters)
            for forbidden in ("action", "control"):
                self.assertTrue(
                    all(forbidden not in name.lower() for name in parameter_names),
                    f"{method.__qualname__} exposes {forbidden!r}: {parameter_names}",
                )


if __name__ == "__main__":
    unittest.main()
