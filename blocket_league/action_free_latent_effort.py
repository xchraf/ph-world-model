from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentEffortConfig:
    state_size: int
    effort_size: int
    hidden_size: int = 96
    hidden_layers: int = 2
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if self.state_size < 1 or self.effort_size < 1:
            raise ValueError("state_size and effort_size must be positive")
        if self.hidden_size < 1 or self.hidden_layers < 1:
            raise ValueError("the inference network must have positive size")
        if self.maximum <= 0.0:
            raise ValueError("maximum must be positive")


class LatentEffortInference(nn.Module):
    """Infer an unlabelled transition cause from two visual latent states.

    This module has no physical command input.  Subtracting the same network at
    zero displacement makes an identity transition map to exactly zero effort,
    independent of initialization and without a labelled coast example.
    """

    def __init__(self, config: LatentEffortConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = [nn.LayerNorm(2 * config.state_size)]
        current = 2 * config.state_size
        for _ in range(config.hidden_layers):
            layers.extend((nn.Linear(current, config.hidden_size), nn.SiLU()))
            current = config.hidden_size
        final = nn.Linear(current, config.effort_size)
        nn.init.normal_(final.weight, std=0.02)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def _raw(self, present: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((present, displacement), dim=-1))

    def forward(self, present: torch.Tensor, successor: torch.Tensor) -> torch.Tensor:
        if present.shape != successor.shape or present.shape[-1] != self.config.state_size:
            raise ValueError("present and successor must have the configured state shape")
        displacement = successor - present
        raw = self._raw(present, displacement)
        identity = self._raw(present, torch.zeros_like(displacement))
        return self.config.maximum * torch.tanh(raw - identity)


def latent_effort_statistics(
    latent_effort: torch.Tensor,
    *,
    target_variance: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Return label-free gauge fixing and temporal regularizers.

    Conditional effects identify a port basis only up to an invertible mixing.
    Zero mean, finite fixed variance and decorrelation choose one reproducible
    global gauge while leaving the physical interface to the later probe-only
    calibration.
    """

    if latent_effort.ndim < 2:
        raise ValueError("latent_effort must include sample and feature dimensions")
    if target_variance <= 0.0:
        raise ValueError("target_variance must be positive")
    flat = latent_effort.reshape(-1, latent_effort.shape[-1])
    mean = flat.mean(dim=0)
    centered = flat - mean
    denominator = max(flat.shape[0] - 1, 1)
    covariance = centered.transpose(0, 1) @ centered / denominator
    diagonal = covariance.diagonal()
    off_diagonal = covariance - torch.diag_embed(diagonal)
    mean_loss = mean.square().mean()
    variance_loss = (diagonal - target_variance).square().mean()
    decorrelation_loss = off_diagonal.square().mean()
    if latent_effort.ndim >= 3 and latent_effort.shape[-2] > 1:
        temporal_loss = (
            latent_effort[..., 1:, :] - latent_effort[..., :-1, :]
        ).square().mean()
    else:
        temporal_loss = flat.new_zeros(())
    return {
        "mean": mean_loss,
        "variance": variance_loss,
        "decorrelation": decorrelation_loss,
        "temporal": temporal_loss,
        "total": mean_loss + variance_loss + decorrelation_loss,
    }


class UnstructuredLatentEffortDynamics(nn.Module):
    """Matched black-box drift plus state-dependent latent-effort field."""

    def __init__(
        self,
        state_size: int,
        effort_size: int,
        hidden_size: int,
        *,
        dt: float,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.state_size = state_size
        self.effort_size = effort_size
        self.dt = dt

        def mlp(output: int) -> nn.Sequential:
            modules: list[nn.Module] = []
            current = state_size
            for _ in range(hidden_layers):
                modules.extend((nn.Linear(current, hidden_size), nn.Tanh()))
                current = hidden_size
            modules.append(nn.Linear(current, output))
            return nn.Sequential(*modules)

        self.drift_network = mlp(state_size)
        self.port_network = mlp(state_size * effort_size)

    def drift(self, state: torch.Tensor) -> torch.Tensor:
        return self.drift_network(state)

    def port(self, state: torch.Tensor) -> torch.Tensor:
        return self.port_network(state).reshape(
            *state.shape[:-1], self.state_size, self.effort_size
        )

    def vector_field(
        self,
        state: torch.Tensor,
        latent_effort: torch.Tensor,
    ) -> torch.Tensor:
        return self.drift(state) + torch.einsum(
            "...im,...m->...i", self.port(state), latent_effort
        )

    def integrate(
        self,
        state: torch.Tensor,
        latent_effort: torch.Tensor,
    ) -> torch.Tensor:
        first = self.vector_field(state, latent_effort)
        midpoint = state + 0.5 * self.dt * first
        return state + self.dt * self.vector_field(midpoint, latent_effort)

    def forward(
        self,
        state: torch.Tensor,
        latent_effort: torch.Tensor,
    ) -> torch.Tensor:
        return self.integrate(state, latent_effort)
