"""Matched structural ablations for Experiment F.

These classes are never candidates for the registered positive outcome.  They
exist only to isolate the Jacobi and state-dependent-port contributions while
retaining the same direct visual training path and discrete-gradient solver.
"""

from __future__ import annotations

import torch
from torch import nn

from .direct_poisson_ph import DirectPoissonPHConfig, DirectPoissonPortHamiltonian


class _SkewNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = state_size
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current, hidden_size), nn.Tanh()))
            current = hidden_size
        final = nn.Linear(current, state_size * state_size)
        nn.init.normal_(final.weight, std=0.01)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)
        self.state_size = state_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.network(state).reshape(
            *state.shape[:-1], self.state_size, self.state_size
        )
        return raw - raw.transpose(-1, -2)


class SkewOnlyPortHamiltonian(DirectPoissonPortHamiltonian):
    """State-dependent skew ``J`` with no Jacobi-by-construction guarantee."""

    def __init__(self, config: DirectPoissonPHConfig) -> None:
        super().__init__(config)
        # Remove the unused Poisson chart so parameter counting is honest.
        del self.coordinate_map
        # The ablations are rebuilt after resetting the same master seed.  Do
        # not let their variant-specific parameters advance the global RNG,
        # otherwise the downstream effort head and persistent port frames no
        # longer have matched initializations despite that reset.
        with torch.random.fork_rng(devices=[]):
            self.skew_network = _SkewNetwork(
                config.state_size, config.hidden_size, config.hidden_layers
            )

    def interconnection(self, state: torch.Tensor) -> torch.Tensor:
        return self.skew_network(state)

    def jacobi_tensor(
        self,
        state: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        # Reuse the contraction implementation while dispatching to this
        # class's unconstrained interconnection method.
        with torch.enable_grad():
            differentiable = state
            if not differentiable.requires_grad:
                differentiable = state.detach().requires_grad_(True)
            interconnection = self.interconnection(differentiable)
            n = self.config.state_size
            derivatives = []
            for row in range(n):
                row_derivatives = []
                for column in range(n):
                    row_derivatives.append(
                        torch.autograd.grad(
                            interconnection[..., row, column].sum(),
                            differentiable,
                            create_graph=create_graph,
                            retain_graph=True,
                        )[0]
                    )
                derivatives.append(torch.stack(row_derivatives, dim=-2))
            derivative_tensor = torch.stack(derivatives, dim=-3)
            return (
                torch.einsum("...il,...jkl->...ijk", interconnection, derivative_tensor)
                + torch.einsum("...jl,...kil->...ijk", interconnection, derivative_tensor)
                + torch.einsum("...kl,...ijl->...ijk", interconnection, derivative_tensor)
            )


class ConstantPortHamiltonian(DirectPoissonPortHamiltonian):
    """Poisson core whose port matrix is deliberately independent of state."""

    def __init__(self, config: DirectPoissonPHConfig) -> None:
        super().__init__(config)
        del self.port_network
        with torch.random.fork_rng(devices=[]):
            self.constant_port = nn.Parameter(
                0.02 * torch.randn(config.state_size, config.port_size)
            )

    def port(self, state: torch.Tensor) -> torch.Tensor:
        return self.constant_port.expand(
            *state.shape[:-1], self.config.state_size, self.config.port_size
        )


__all__ = ["ConstantPortHamiltonian", "SkewOnlyPortHamiltonian"]
