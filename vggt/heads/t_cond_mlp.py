from __future__ import annotations

from typing import Optional

import torch


class AdaptiveLayerNorm1D(torch.nn.Module):
    """LayerNorm whose affine parameters are conditioned on a per-sample vector.

    x: (..., dim)
    cond: (B, cond_dim)
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {cond_dim}")

        self.norm = torch.nn.LayerNorm(dim)
        self.to_scale_shift = torch.nn.Linear(cond_dim, 2 * dim)

        # Start as identity transform
        torch.nn.init.zeros_(self.to_scale_shift.weight)
        torch.nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)

        # Broadcast to match x dims (keep batch dim aligned)
        if x.dim() > 2:
            view_shape = (scale.shape[0],) + (1,) * (x.dim() - 2) + (scale.shape[1],)
            scale = scale.view(view_shape)
            shift = shift.view(view_shape)

        return x * (1.0 + scale) + shift


def normalization_layer(norm: Optional[str], dim: int, norm_cond_dim: int = -1):
    if norm == "batch":
        return torch.nn.BatchNorm1d(dim)
    if norm == "layer":
        return torch.nn.LayerNorm(dim)
    if norm == "ada":
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive for ada norm, got {norm_cond_dim}")
        return AdaptiveLayerNorm1D(dim, norm_cond_dim)
    if norm is None:
        return torch.nn.Identity()
    raise ValueError(f"Unknown norm: {norm}")


class FrequencyEmbedder(torch.nn.Module):
    """Sine/cosine frequency embedding for scalar or vector inputs."""

    def __init__(self, num_frequencies: int, max_freq_log2: float):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,) or (N, D)
        if x.dim() == 1:
            x = x.unsqueeze(1)

        n = x.shape[0]
        x_unsqueezed = x.unsqueeze(-1)  # (N, D, 1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed  # (N, D, F)
        s = torch.sin(scaled)
        c = torch.cos(scaled)

        # (N, D * (2F + 1))
        return torch.cat([s, c, x_unsqueezed], dim=-1).reshape(n, -1)
