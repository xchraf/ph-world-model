from __future__ import annotations

import unittest

import torch

from blocket_league.neural_port_hamiltonian import (
    NeuralODE,
    NeuralPortHamiltonian,
    NeuralPortHamiltonianConfig,
)


class NeuralPortHamiltonianTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4_219)
        self.config = NeuralPortHamiltonianConfig(
            state_size=6,
            input_size=3,
            hidden_size=20,
            dt=0.07,
        )
        self.model = NeuralPortHamiltonian(
            self.config,
            state_mean=torch.linspace(-0.2, 0.3, 6),
            state_scale=torch.linspace(0.7, 1.3, 6),
        )
        self.state = torch.randn(5, 6)
        self.control = torch.randn(5, 3)

    def test_generic_component_shapes_and_constraints(self) -> None:
        energy, gradient, interconnection, resistance, port = self.model.components(
            self.state, create_graph=False
        )
        self.assertEqual(energy.shape, (5,))
        self.assertEqual(gradient.shape, (5, 6))
        self.assertEqual(interconnection.shape, (5, 6, 6))
        self.assertEqual(resistance.shape, (5, 6, 6))
        self.assertEqual(port.shape, (5, 6, 3))
        torch.testing.assert_close(
            interconnection + interconnection.transpose(-1, -2),
            torch.zeros_like(interconnection),
            atol=0.0,
            rtol=0.0,
        )
        eigenvalues = torch.linalg.eigvalsh(resistance)
        self.assertGreaterEqual(float(eigenvalues.min()), -1e-6)

    def test_continuous_power_balance_is_an_identity(self) -> None:
        terms = self.model.power_terms(self.state, self.control, create_graph=False)
        torch.testing.assert_close(
            terms["balanceDefect"],
            torch.zeros_like(terms["balanceDefect"]),
            atol=2e-7,
            rtol=2e-5,
        )
        self.assertGreaterEqual(float(terms["dissipation"].min()), -1e-7)

    def test_every_learned_function_receives_gradient(self) -> None:
        prediction = self.model(self.state, self.control)
        loss = prediction.square().mean()
        loss.backward()
        for name in (
            "energy_network",
            "interconnection_network",
            "resistance_network",
            "port_network",
        ):
            parameters = list(getattr(self.model, name).parameters())
            self.assertTrue(any(parameter.grad is not None for parameter in parameters), name)
            self.assertTrue(
                any(float(parameter.grad.abs().sum()) > 0.0 for parameter in parameters),
                name,
            )

    def test_outputs_are_state_dependent(self) -> None:
        first = self.state[:1]
        second = first + 0.8
        first_components = self.model.components(first, create_graph=False)
        second_components = self.model.components(second, create_graph=False)
        for index, name in ((0, "H"), (2, "J"), (3, "R"), (4, "B")):
            difference = (first_components[index] - second_components[index]).abs().max()
            self.assertGreater(float(difference), 1e-8, name)

    def test_jacobi_tensor_is_finite_and_antisymmetric_in_triples(self) -> None:
        tensor = self.model.jacobi_tensor(self.state[:2])
        self.assertEqual(tensor.shape, (2, 6, 6, 6))
        self.assertTrue(bool(torch.isfinite(tensor).all()))
        torch.testing.assert_close(
            tensor + tensor.transpose(-1, -2),
            torch.zeros_like(tensor),
            atol=2e-6,
            rtol=2e-5,
        )

    def test_all_integrators_and_unstructured_control_are_dimension_generic(self) -> None:
        for model in (self.model, NeuralODE(self.config)):
            for method in ("euler", "midpoint", "rk4"):
                result = model.integrate(
                    self.state,
                    self.control,
                    method=method,
                    substeps=2,
                )
                self.assertEqual(result.shape, self.state.shape)
                self.assertTrue(bool(torch.isfinite(result).all()))


if __name__ == "__main__":
    unittest.main()
