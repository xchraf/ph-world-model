from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .factorized_transformer import DirectFactorizedBlock


@dataclass(frozen=True)
class PixelDirectConfig:
    image_size: int = 64
    patch_size: int = 4
    palette_size: int = 9
    history_frames: int = 8
    pixel_embedding_size: int = 8
    hidden_size: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PIXEL_DIRECT_PRESETS: dict[str, dict[str, int]] = {
    "nano": {"hidden_size": 72, "depth": 4, "heads": 4},
    "micro": {"hidden_size": 128, "depth": 4, "heads": 4},
    "tiny": {"hidden_size": 192, "depth": 6, "heads": 6},
    "small": {"hidden_size": 256, "depth": 8, "heads": 8},
}


def pixel_direct_config_for_preset(name: str, **overrides: object) -> PixelDirectConfig:
    if name not in PIXEL_DIRECT_PRESETS:
        raise ValueError(f"Unknown pixel-direct preset {name!r}")
    return PixelDirectConfig(**{**PIXEL_DIRECT_PRESETS[name], **overrides})


class DirectPixelTransformer(nn.Module):
    """Causal transformer mapping passive raw-pixel histories to the next frame."""

    def __init__(self, config: PixelDirectConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.config = config
        patch_features = config.patch_size**2 * config.pixel_embedding_size
        self.pixel_embedding = nn.Embedding(config.palette_size, config.pixel_embedding_size)
        self.patch_projection = nn.Linear(patch_features, config.hidden_size)
        self.spatial_position = nn.Parameter(
            torch.randn(1, 1, config.grid_size**2, config.hidden_size) * 0.02
        )
        self.temporal_position = nn.Parameter(
            torch.randn(1, config.history_frames, 1, config.hidden_size) * 0.02
        )
        self.blocks = nn.ModuleList(
            DirectFactorizedBlock(config.hidden_size, config.heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.patch_size**2 * config.palette_size,
        )

    def patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        config = self.config
        batch, time, height, width = frames.shape
        if (height, width) != (config.image_size, config.image_size):
            raise ValueError(f"Expected {config.image_size}x{config.image_size} frames")
        embedded = self.pixel_embedding(frames.long())
        patches = embedded.reshape(
            batch,
            time,
            config.grid_size,
            config.patch_size,
            config.grid_size,
            config.patch_size,
            config.pixel_embedding_size,
        ).permute(0, 1, 2, 4, 3, 5, 6)
        return patches.reshape(batch, time, config.grid_size**2, -1)

    def unpatch_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        config = self.config
        batch, time = tokens.shape[:2]
        logits = self.output_projection(self.output_norm(tokens)).reshape(
            batch,
            time,
            config.grid_size,
            config.grid_size,
            config.patch_size,
            config.patch_size,
            config.palette_size,
        )
        return logits.permute(0, 1, 6, 2, 4, 3, 5).reshape(
            batch,
            time,
            config.palette_size,
            config.image_size,
            config.image_size,
        )

    def forward(
        self,
        frames: torch.Tensor,
        *,
        return_hidden: bool = False,
        intervention_block: int | None = None,
        intervention: torch.Tensor | None = None,
        intervention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        batch, time = frames.shape[:2]
        if time > self.config.history_frames:
            raise ValueError(f"Sequence has {time} frames; maximum is {self.config.history_frames}")
        tokens = (
            self.patch_projection(self.patch_tokens(frames))
            + self.spatial_position
            + self.temporal_position[:, :time]
        )
        hidden_states: list[torch.Tensor] = []
        if (intervention_block is None) != (intervention is None):
            raise ValueError("intervention_block and intervention must be provided together")
        for block_index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if block_index == intervention_block:
                write = intervention
                if write is None:
                    raise AssertionError("intervention unexpectedly missing")
                if write.ndim == 1:
                    write = write[None, None, None]
                elif write.ndim == 2:
                    write = write[:, None, None]
                if intervention_mask is not None:
                    write = write * intervention_mask[..., None]
                tokens = tokens + write
            if return_hidden:
                hidden_states.append(tokens)
        logits = self.unpatch_logits(tokens)
        if return_hidden:
            return logits, hidden_states
        return logits

    @torch.no_grad()
    def next_frame(self, frames: torch.Tensor) -> torch.Tensor:
        logits = self(frames)
        if not isinstance(logits, torch.Tensor):
            logits = logits[0]
        return logits[:, -1].argmax(dim=1)


def build_pixel_direct_from_checkpoint(
    checkpoint: dict[str, Any],
) -> DirectPixelTransformer:
    if checkpoint.get("kind") != "passive_direct_pixel_world_model":
        raise ValueError("Checkpoint is not a passive direct pixel world model")
    model = DirectPixelTransformer(PixelDirectConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model
