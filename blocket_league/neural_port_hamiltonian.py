from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


IntegrationMethod = Literal["euler", "midpoint", "rk4"]


@dataclass(frozen=True)
class NeuralPortHamiltonianConfig:
    """Dimension-independent configuration for a neural pH vector field."""

    state_size: int
    input_size: int
    hidden_size: int = 64
    hidden_layers: int = 2
    dt: float = 1.0
    integration_method: IntegrationMethod = "midpoint"
    integration_substeps: int = 1
    resistance_floor: float = 1e-5

    def __post_init__(self) -> None:
        if self.state_size < 1:
            raise ValueError("state_size must be positive")
        if self.input_size < 1:
            raise ValueError("input_size must be positive")
        if self.hidden_size < 1 or self.hidden_layers < 1:
            raise ValueError("the MLP must contain positive-sized hidden layers")
        if self.dt <= 0.0 or self.integration_substeps < 1:
            raise ValueError("dt and integration_substeps must be positive")
        if self.resistance_floor < 0.0:
            raise ValueError("resistance_floor must be non-negative")


class _SmoothMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        hidden_layers: int,
        *,
        final_scale: float,
        final_bias: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_size = input_size
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current_size, hidden_size), nn.Tanh()))
            current_size = hidden_size
        final = nn.Linear(current_size, output_size)
        layers.append(final)
        self.network = nn.Sequential(*layers)
        nn.init.normal_(final.weight, mean=0.0, std=final_scale)
        nn.init.constant_(final.bias, final_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class NeuralPortHamiltonian(nn.Module):
    r"""Learn ``H(x)``, ``J(x)``, ``R(x)`` and ``B(x)`` by construction.

    The continuous vector field is

    ``dx/dt = (J(x) - R(x)) grad(H(x)) + B(x) u``.

    ``J`` is exactly skew-symmetric and ``R`` is exactly positive
    semi-definite for every state.  No mechanical block structure, control
    incidence graph, or state-independent coefficient is assumed.
    """

    def __init__(
        self,
        config: NeuralPortHamiltonianConfig,
        *,
        state_mean: torch.Tensor | None = None,
        state_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        n = config.state_size
        m = config.input_size
        if state_mean is None:
            state_mean = torch.zeros(n)
        if state_scale is None:
            state_scale = torch.ones(n)
        if state_mean.shape != (n,) or state_scale.shape != (n,):
            raise ValueError("state_mean and state_scale must match state_size")
        if bool((state_scale <= 0.0).any()):
            raise ValueError("state_scale must be strictly positive")
        self.register_buffer("state_mean", state_mean.detach().float().clone())
        self.register_buffer("state_scale", state_scale.detach().float().clone())

        skew_size = n * (n - 1) // 2
        triangular_size = n * (n + 1) // 2
        common = (n, config.hidden_size, config.hidden_layers)
        self.energy_network = _SmoothMLP(
            common[0], 1, common[1], common[2], final_scale=0.08
        )
        self.interconnection_network = _SmoothMLP(
            common[0], skew_size, common[1], common[2], final_scale=0.015
        )
        self.resistance_network = _SmoothMLP(
            common[0], triangular_size, common[1], common[2],
            final_scale=0.006,
            final_bias=0.0,
        )
        self.port_network = _SmoothMLP(
            common[0], n * m, common[1], common[2], final_scale=0.015
        )

        skew_rows, skew_columns = torch.triu_indices(n, n, offset=1)
        lower_rows, lower_columns = torch.tril_indices(n, n, offset=0)
        resistance_bias = self.resistance_network.network[-1].bias
        with torch.no_grad():
            resistance_bias[lower_rows == lower_columns] = -4.0
        self.register_buffer("skew_rows", skew_rows)
        self.register_buffer("skew_columns", skew_columns)
        self.register_buffer("lower_rows", lower_rows)
        self.register_buffer("lower_columns", lower_columns)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_scale

    def hamiltonian(self, state: torch.Tensor) -> torch.Tensor:
        return self.energy_network(self.normalize_state(state)).squeeze(-1)

    def interconnection(self, state: torch.Tensor) -> torch.Tensor:
        n = self.config.state_size
        packed = self.interconnection_network(self.normalize_state(state))
        matrix = packed.new_zeros((*packed.shape[:-1], n, n))
        matrix[..., self.skew_rows, self.skew_columns] = packed
        matrix[..., self.skew_columns, self.skew_rows] = -packed
        return matrix

    def resistance(self, state: torch.Tensor) -> torch.Tensor:
        n = self.config.state_size
        packed = self.resistance_network(self.normalize_state(state))
        factor = packed.new_zeros((*packed.shape[:-1], n, n))
        factor[..., self.lower_rows, self.lower_columns] = packed
        diagonal = torch.arange(n, device=state.device)
        factor[..., diagonal, diagonal] = F.softplus(
            factor[..., diagonal, diagonal]
        ) + self.config.resistance_floor
        return factor @ factor.transpose(-1, -2)

    def port(self, state: torch.Tensor) -> torch.Tensor:
        values = self.port_network(self.normalize_state(state))
        return values.reshape(*state.shape[:-1], self.config.state_size, self.config.input_size)

    def components(
        self,
        state: torch.Tensor,
        *,
        create_graph: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``H, grad(H), J, R, B`` while remaining usable in no-grad eval."""

        if create_graph is None:
            create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            differentiable_state = state
            if not differentiable_state.requires_grad:
                differentiable_state = state.detach().requires_grad_(True)
            energy = self.hamiltonian(differentiable_state)
            gradient = torch.autograd.grad(
                energy.sum(),
                differentiable_state,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
        interconnection = self.interconnection(differentiable_state)
        resistance = self.resistance(differentiable_state)
        port = self.port(differentiable_state)
        return energy, gradient, interconnection, resistance, port

    def vector_field(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        _, gradient, interconnection, resistance, port = self.components(state)
        internal = torch.einsum(
            "...ij,...j->...i", interconnection - resistance, gradient
        )
        supplied = torch.einsum("...im,...m->...i", port, control)
        return internal + supplied

    def power_terms(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        *,
        create_graph: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        energy, gradient, interconnection, resistance, port = self.components(
            state, create_graph=create_graph
        )
        vector_field = torch.einsum(
            "...ij,...j->...i", interconnection - resistance, gradient
        ) + torch.einsum("...im,...m->...i", port, control)
        storage_rate = torch.einsum("...i,...i->...", gradient, vector_field)
        dissipation = torch.einsum(
            "...i,...ij,...j->...", gradient, resistance, gradient
        )
        output = torch.einsum("...im,...i->...m", port, gradient)
        supply = torch.einsum("...m,...m->...", control, output)
        return {
            "energy": energy,
            "gradient": gradient,
            "output": output,
            "storageRate": storage_rate,
            "dissipation": dissipation,
            "supply": supply,
            "balanceDefect": storage_rate + dissipation - supply,
        }

    def jacobi_tensor(
        self,
        state: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        r"""Return the Poisson Jacobi tensor for the learned ``J(x)``.

        Skew symmetry is sufficient for the power identity.  A state-dependent
        interconnection is a Poisson tensor only when this additional tensor
        vanishes; exposing it prevents silently conflating the two claims.
        """

        with torch.enable_grad():
            differentiable_state = state
            if not differentiable_state.requires_grad:
                differentiable_state = state.detach().requires_grad_(True)
            interconnection = self.interconnection(differentiable_state)
            derivatives = []
            n = self.config.state_size
            for row in range(n):
                row_derivatives = []
                for column in range(n):
                    derivative = torch.autograd.grad(
                        interconnection[..., row, column].sum(),
                        differentiable_state,
                        create_graph=create_graph,
                        retain_graph=True,
                    )[0]
                    row_derivatives.append(derivative)
                derivatives.append(torch.stack(row_derivatives, dim=-2))
            derivative_tensor = torch.stack(derivatives, dim=-3)
            return (
                torch.einsum("...il,...jkl->...ijk", interconnection, derivative_tensor)
                + torch.einsum("...jl,...kil->...ijk", interconnection, derivative_tensor)
                + torch.einsum("...kl,...ijl->...ijk", interconnection, derivative_tensor)
            )

    def integrate(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        *,
        dt: float | None = None,
        method: IntegrationMethod | None = None,
        substeps: int | None = None,
    ) -> torch.Tensor:
        dt = self.config.dt if dt is None else dt
        method = self.config.integration_method if method is None else method
        substeps = self.config.integration_substeps if substeps is None else substeps
        if dt <= 0.0 or substeps < 1:
            raise ValueError("dt and substeps must be positive")
        step_size = dt / substeps
        current = state
        for _ in range(substeps):
            if method == "euler":
                current = current + step_size * self.vector_field(current, control)
            elif method == "midpoint":
                first = self.vector_field(current, control)
                midpoint = current + 0.5 * step_size * first
                current = current + step_size * self.vector_field(midpoint, control)
            elif method == "rk4":
                first = self.vector_field(current, control)
                second = self.vector_field(current + 0.5 * step_size * first, control)
                third = self.vector_field(current + 0.5 * step_size * second, control)
                fourth = self.vector_field(current + step_size * third, control)
                current = current + step_size * (
                    first + 2.0 * second + 2.0 * third + fourth
                ) / 6.0
            else:
                raise ValueError(f"unknown integration method {method!r}")
        return current

    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return self.integrate(state, control)


class NeuralODE(nn.Module):
    """Unstructured continuous-time control with the same integration API."""

    def __init__(
        self,
        config: NeuralPortHamiltonianConfig,
        *,
        state_mean: torch.Tensor | None = None,
        state_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        n = config.state_size
        if state_mean is None:
            state_mean = torch.zeros(n)
        if state_scale is None:
            state_scale = torch.ones(n)
        self.register_buffer("state_mean", state_mean.detach().float().clone())
        self.register_buffer("state_scale", state_scale.detach().float().clone())
        self.network = _SmoothMLP(
            n + config.input_size,
            n,
            config.hidden_size,
            config.hidden_layers,
            final_scale=0.015,
        )

    def vector_field(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        normalized = (state - self.state_mean) / self.state_scale
        return self.network(torch.cat((normalized, control), dim=-1))

    def integrate(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        *,
        dt: float | None = None,
        method: IntegrationMethod | None = None,
        substeps: int | None = None,
    ) -> torch.Tensor:
        dt = self.config.dt if dt is None else dt
        method = self.config.integration_method if method is None else method
        substeps = self.config.integration_substeps if substeps is None else substeps
        if dt <= 0.0 or substeps < 1:
            raise ValueError("dt and substeps must be positive")
        step_size = dt / substeps
        current = state
        for _ in range(substeps):
            if method == "euler":
                current = current + step_size * self.vector_field(current, control)
            elif method == "midpoint":
                first = self.vector_field(current, control)
                current = current + step_size * self.vector_field(
                    current + 0.5 * step_size * first, control
                )
            elif method == "rk4":
                first = self.vector_field(current, control)
                second = self.vector_field(current + 0.5 * step_size * first, control)
                third = self.vector_field(current + 0.5 * step_size * second, control)
                fourth = self.vector_field(current + step_size * third, control)
                current = current + step_size * (
                    first + 2.0 * second + 2.0 * third + fourth
                ) / 6.0
            else:
                raise ValueError(f"unknown integration method {method!r}")
        return current

    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return self.integrate(state, control)
