from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .latent_patch_renderer import LatentPatchTransformerRenderer
from .neural_port_hamiltonian import NeuralPortHamiltonian
from .pixel_direct_model import DirectPixelTransformer
from .tensor_provenance import module_tensor_hash, parameter_count


class FrozenTransformerStateAdapter(nn.Module):
    """A trainable state readout whose video transformer is permanently frozen."""

    def __init__(
        self,
        backbone: DirectPixelTransformer,
        state_size: int,
        hidden_size: int,
        *,
        lens_block: int,
    ) -> None:
        super().__init__()
        if not 0 <= lens_block < len(backbone.blocks):
            raise ValueError("lens_block is outside the transformer")
        self.backbone = backbone.eval().requires_grad_(False)
        self.state_size = state_size
        self.lens_block = lens_block
        token_size = backbone.config.hidden_size
        self.pool_score = nn.Linear(token_size, 1)
        self.readout = nn.Sequential(
            nn.LayerNorm(3 * token_size),
            nn.Linear(3 * token_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, state_size),
        )
        nn.init.normal_(self.readout[-1].weight, std=0.03)
        nn.init.zeros_(self.readout[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _tokens(
        self,
        contexts: torch.Tensor,
        *,
        intervention: torch.Tensor | None = None,
        intervention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self.backbone
        tokens = (
            model.patch_projection(model.patch_tokens(contexts))
            + model.spatial_position
            + model.temporal_position[:, : contexts.shape[1]]
        )
        for index, block in enumerate(model.blocks):
            tokens = block(tokens)
            if intervention is not None and index == self.lens_block:
                write = intervention
                if write.ndim == 2:
                    write = write[:, None, None]
                if intervention_mask is None:
                    raise ValueError("an intervention requires a token mask")
                tokens = tokens + write * intervention_mask[..., None]
        return tokens

    def _read(self, tokens: torch.Tensor) -> torch.Tensor:
        latest = tokens[:, -1]
        attention = self.pool_score(latest).squeeze(-1).softmax(dim=-1)
        attended = torch.einsum("bp,bph->bh", attention, latest)
        features = torch.cat(
            (latest.mean(dim=1), latest.std(dim=1, unbiased=False), attended), dim=-1
        )
        return self.readout(features.float())

    def forward(
        self,
        contexts: torch.Tensor,
        *,
        intervention: torch.Tensor | None = None,
        intervention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (
            self.backbone.config.history_frames,
            self.backbone.config.image_size,
            self.backbone.config.image_size,
        )
        if contexts.shape[-3:] != expected:
            raise ValueError(f"expected contexts ending in {expected}")
        leading = contexts.shape[:-3]
        flat = contexts.reshape(-1, *expected)
        if intervention is not None and flat.shape[0] != intervention.shape[0]:
            raise ValueError("interventions only support the flattened context batch")
        states = self._read(
            self._tokens(
                flat,
                intervention=intervention,
                intervention_mask=intervention_mask,
            )
        )
        return states.reshape(*leading, self.state_size)


class SmoothMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        hidden_layers: int = 2,
        *,
        final_scale: float = 0.02,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_size
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current, hidden_size), nn.Tanh()))
            current = hidden_size
        final = nn.Linear(current, output_size)
        nn.init.normal_(final.weight, std=final_scale)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


@dataclass(frozen=True)
class UnstructuredPortConfig:
    state_size: int
    input_size: int
    hidden_size: int
    hidden_layers: int = 2
    dt: float = 0.05
    substeps: int = 1


class UnstructuredPortDynamics(nn.Module):
    """Generic autonomous latent ODE with a separately auditable port field."""

    def __init__(
        self,
        config: UnstructuredPortConfig,
        *,
        state_mean: torch.Tensor | None = None,
        state_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        n, m = config.state_size, config.input_size
        self.register_buffer(
            "state_mean", torch.zeros(n) if state_mean is None else state_mean.detach().clone()
        )
        self.register_buffer(
            "state_scale", torch.ones(n) if state_scale is None else state_scale.detach().clone()
        )
        self.drift_network = SmoothMLP(n, n, config.hidden_size, config.hidden_layers)
        self.port_network = SmoothMLP(n, n * m, config.hidden_size, config.hidden_layers)

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_scale.clamp_min(1e-6)

    def drift(self, state: torch.Tensor) -> torch.Tensor:
        return self.drift_network(self.normalize(state))

    def port(self, state: torch.Tensor) -> torch.Tensor:
        values = self.port_network(self.normalize(state))
        return values.reshape(*state.shape[:-1], self.config.state_size, self.config.input_size)

    def vector_field(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return self.drift(state) + torch.einsum("...im,...m->...i", self.port(state), control)

    def integrate(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        step = self.config.dt / self.config.substeps
        current = state
        for _ in range(self.config.substeps):
            first = self.vector_field(current, control)
            middle = current + 0.5 * step * first
            current = current + step * self.vector_field(middle, control)
        return current

    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return self.integrate(state, control)


class PassiveVisualPHModel(nn.Module):
    def __init__(
        self,
        adapter: FrozenTransformerStateAdapter,
        renderer: LatentPatchTransformerRenderer,
        core: NeuralPortHamiltonian,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.renderer = renderer
        self.core = core

    def encode(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.adapter(contexts)

    def step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=state.device.type, enabled=False):
            return self.core(state.float(), control.float())


def matched_unstructured_hidden_size(
    target_parameters: int,
    *,
    state_size: int,
    input_size: int,
    hidden_layers: int,
    dt: float,
) -> int:
    candidates = range(4, 512)
    return min(
        candidates,
        key=lambda hidden: abs(
            parameter_count(
                UnstructuredPortDynamics(
                    UnstructuredPortConfig(
                        state_size=state_size,
                        input_size=input_size,
                        hidden_size=hidden,
                        hidden_layers=hidden_layers,
                        dt=dt,
                    )
                )
            )
            - target_parameters
        ),
    )
