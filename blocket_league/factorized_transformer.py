"""Simulator-independent spatiotemporal transformer building block."""

from __future__ import annotations

import torch
from torch import nn


class DirectFactorizedBlock(nn.Module):
    """Spatial attention plus causal temporal attention over patch tokens."""

    def __init__(self, hidden_size: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(hidden_size)
        self.spatial_attention = nn.MultiheadAttention(
            hidden_size, heads, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_size)
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size, heads, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden_size),
        )
        self.spatial_scale = nn.Parameter(torch.full((hidden_size,), 1e-2))
        self.temporal_scale = nn.Parameter(torch.full((hidden_size,), 1e-2))
        self.mlp_scale = nn.Parameter(torch.full((hidden_size,), 1e-2))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, hidden = tokens.shape
        spatial = self.spatial_norm(tokens).reshape(
            batch * frames, patches, hidden
        )
        spatial = self.spatial_attention(
            spatial, spatial, spatial, need_weights=False
        )[0]
        tokens = (
            tokens
            + spatial.reshape(batch, frames, patches, hidden) * self.spatial_scale
        )

        temporal = self.temporal_norm(tokens).permute(0, 2, 1, 3)
        temporal = temporal.reshape(batch * patches, frames, hidden)
        causal_mask = torch.ones(
            frames, frames, device=tokens.device, dtype=torch.bool
        ).triu(1)
        temporal = self.temporal_attention(
            temporal,
            temporal,
            temporal,
            attn_mask=causal_mask,
            need_weights=False,
        )[0]
        temporal = temporal.reshape(batch, patches, frames, hidden).permute(
            0, 2, 1, 3
        )
        tokens = tokens + temporal * self.temporal_scale
        return tokens + self.mlp(self.mlp_norm(tokens)) * self.mlp_scale


__all__ = ["DirectFactorizedBlock"]
