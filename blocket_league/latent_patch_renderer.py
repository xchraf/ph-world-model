"""Simulator-independent latent-state-to-pixel transformer renderer."""

from __future__ import annotations

import torch
from torch import nn

from .factorized_transformer import DirectFactorizedBlock


class LatentPatchTransformerRenderer(nn.Module):
    """Decode a latent state through spatial tokens and transformer blocks."""

    def __init__(
        self,
        state_size: int,
        *,
        image_size: int,
        patch_size: int,
        palette_size: int,
        hidden_size: int,
        depth: int,
        heads: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.state_size = state_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.palette_size = palette_size
        self.grid_size = image_size // patch_size
        self.latent_projection = nn.Linear(state_size, hidden_size)
        self.spatial_position = nn.Parameter(
            torch.randn(1, 1, self.grid_size**2, hidden_size) * 0.02
        )
        self.blocks = nn.ModuleList(
            DirectFactorizedBlock(hidden_size, heads, 4.0) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(
            hidden_size, patch_size**2 * palette_size
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        leading = state.shape[:-1]
        flat = state.reshape(-1, self.state_size)
        tokens = self.latent_projection(flat)[:, None, None] + self.spatial_position
        for block in self.blocks:
            tokens = block(tokens)
        logits = self.output_projection(self.output_norm(tokens)).reshape(
            flat.shape[0],
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.palette_size,
        )
        logits = logits.permute(0, 5, 1, 3, 2, 4).reshape(
            flat.shape[0], self.palette_size, self.image_size, self.image_size
        )
        return logits.reshape(
            *leading, self.palette_size, self.image_size, self.image_size
        )


__all__ = ["LatentPatchTransformerRenderer"]
