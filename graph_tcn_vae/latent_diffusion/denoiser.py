"""Conditional denoiser used by latent residual diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Embed integer diffusion timesteps with fixed sinusoidal features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("time embedding dimension must be >= 2")
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [batch]")
        half = self.dim // 2
        if half == 1:
            frequencies = torch.ones(1, device=timesteps.device, dtype=torch.float32)
        else:
            exponent = -math.log(10_000.0) * torch.arange(
                half, device=timesteps.device, dtype=torch.float32
            ) / float(half - 1)
            frequencies = torch.exp(exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.dim - embedding.shape[1]))
        return embedding


class ResidualMLPBlock(nn.Module):
    """A small residual block for latent-space denoising."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(self.norm(value))


class ConditionalResidualDenoiser(nn.Module):
    """Predict diffusion noise for a latent correction conditioned on observed context."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int,
        time_embedding_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or condition_dim < 1:
            raise ValueError("latent_dim and condition_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.input_projection = nn.Linear(
            latent_dim + condition_dim + time_embedding_dim,
            hidden_dim,
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_delta: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_delta.ndim != 2 or noisy_delta.shape[1] != self.latent_dim:
            raise ValueError(
                f"noisy_delta must have shape [batch, {self.latent_dim}]"
            )
        if condition.ndim != 2 or condition.shape != (
            noisy_delta.shape[0],
            self.condition_dim,
        ):
            raise ValueError(
                "condition must have shape "
                f"[batch, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if timesteps.shape != (noisy_delta.shape[0],):
            raise ValueError("timesteps must have shape [batch]")

        time_features = self.time_embedding(timesteps).to(dtype=noisy_delta.dtype)
        hidden = self.input_projection(
            torch.cat([noisy_delta, condition, time_features], dim=1)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.output_norm(hidden))
