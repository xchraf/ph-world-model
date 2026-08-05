from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .codec import RepresentationCodec, build_codec_from_checkpoint
from .factorized_transformer import DirectFactorizedBlock


@dataclass(frozen=True)
class DirectWorldModelConfig:
    latent_dim: int = 32
    latent_grid_size: int = 8
    temporal_downsample: int = 2
    history_latents: int = 8
    hidden_size: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    action_count: int = 9

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DIRECT_MODEL_PRESETS: dict[str, dict[str, int]] = {
    "micro": {"hidden_size": 128, "depth": 4, "heads": 4},
    "tiny": {"hidden_size": 192, "depth": 6, "heads": 6},
    "small": {"hidden_size": 256, "depth": 8, "heads": 8},
}


def direct_config_for_preset(name: str, **overrides: object) -> DirectWorldModelConfig:
    if name not in DIRECT_MODEL_PRESETS:
        raise ValueError(f"Unknown direct model preset {name!r}")
    return DirectWorldModelConfig(**{**DIRECT_MODEL_PRESETS[name], **overrides})


class DirectLatentTransformer(nn.Module):
    """One-pass autoregressive transformer predicting the next codec latent delta."""

    def __init__(self, config: DirectWorldModelConfig, delta_std: torch.Tensor) -> None:
        super().__init__()
        if config.hidden_size % config.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if tuple(delta_std.shape) != (config.latent_dim,):
            raise ValueError(f"Expected delta_std [{config.latent_dim}], got {tuple(delta_std.shape)}")
        self.config = config
        self.input_projection = nn.Linear(config.latent_dim, config.hidden_size)
        self.action_embedding = nn.Embedding(config.action_count, config.hidden_size)
        self.action_projection = nn.Linear(
            config.hidden_size * config.temporal_downsample,
            config.hidden_size,
        )
        self.spatial_position = nn.Parameter(
            torch.randn(1, 1, config.latent_grid_size**2, config.hidden_size) * 0.02
        )
        self.temporal_position = nn.Parameter(
            torch.randn(1, config.history_latents, 1, config.hidden_size) * 0.02
        )
        self.blocks = nn.ModuleList(
            DirectFactorizedBlock(config.hidden_size, config.heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.latent_dim)
        self.register_buffer("delta_std", delta_std.float().clamp_min(1e-3))
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        latents: torch.Tensor,
        action_pairs: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        batch, frames, channels, height, width = latents.shape
        config = self.config
        expected = (config.latent_dim, config.latent_grid_size, config.latent_grid_size)
        if (channels, height, width) != expected:
            raise ValueError(f"Expected latent tail {expected}, got {(channels, height, width)}")
        if frames > config.history_latents:
            raise ValueError(f"Sequence has {frames} latents; maximum is {config.history_latents}")
        if action_pairs.shape != (batch, frames, config.temporal_downsample):
            raise ValueError(
                f"Expected actions {(batch, frames, config.temporal_downsample)}, "
                f"got {tuple(action_pairs.shape)}"
            )

        tokens = latents.permute(0, 1, 3, 4, 2).reshape(
            batch, frames, height * width, channels
        )
        action_condition = self.action_projection(
            self.action_embedding(action_pairs).reshape(
                batch,
                frames,
                config.temporal_downsample * config.hidden_size,
            )
        )
        tokens = (
            self.input_projection(tokens)
            + action_condition[:, :, None]
            + self.spatial_position
            + self.temporal_position[:, :frames]
        )
        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            tokens = block(tokens)
            if return_hidden:
                hidden_states.append(tokens)
        normalized_delta = self.output_projection(self.output_norm(tokens))
        normalized_delta = normalized_delta.reshape(
            batch, frames, height, width, channels
        ).permute(0, 1, 4, 2, 3)
        if return_hidden:
            return normalized_delta, hidden_states
        return normalized_delta

    def next_latent(self, latents: torch.Tensor, action_pairs: torch.Tensor) -> torch.Tensor:
        normalized_delta = self(latents, action_pairs)
        if not isinstance(normalized_delta, torch.Tensor):
            normalized_delta = normalized_delta[0]
        return latents[:, -1] + normalized_delta[:, -1] * self.delta_std[None, :, None, None]


def build_direct_pipeline_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[DirectLatentTransformer, RepresentationCodec, torch.Tensor, torch.Tensor]:
    if checkpoint.get("kind") != "direct_latent_world_model":
        raise ValueError("Checkpoint is not a direct latent world model")
    config = DirectWorldModelConfig(**checkpoint["model_config"])
    delta_std = torch.as_tensor(checkpoint["delta_std"]).float()
    model = DirectLatentTransformer(config, delta_std)
    model.load_state_dict(checkpoint["model"])
    codec_checkpoint = checkpoint["codec_checkpoint"]
    codec = build_codec_from_checkpoint(codec_checkpoint)
    mean = torch.as_tensor(codec_checkpoint["latent_mean"]).float()
    std = torch.as_tensor(codec_checkpoint["latent_std"]).float()
    return model, codec, mean, std
