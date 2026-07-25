"""Learned time-of-day/day-of-week/month embeddings for the graph VAE."""

import numpy as np
import torch
import torch.nn as nn


class TimeHybridEncoder(nn.Module):
    """Hybrid time encoder: learnable embeddings + cyclical sin/cos features.

    Expects per-timestep cyclical time channels in this order:
    ``[hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos]``.
    """

    def __init__(
        self,
        out_dim=6,
        hour_embed_dim=8,
        dow_embed_dim=4,
        month_embed_dim=4,
        dropout=0.1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.hour_embed = nn.Embedding(24, hour_embed_dim)
        self.dow_embed = nn.Embedding(7, dow_embed_dim)
        self.month_embed = nn.Embedding(12, month_embed_dim)

        in_dim = hour_embed_dim + dow_embed_dim + month_embed_dim + 6
        hidden_dim = max(in_dim, out_dim * 2)
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    @staticmethod
    def _cyc_to_index(sin_val, cos_val, period):
        angle = torch.atan2(sin_val, cos_val)
        angle = torch.remainder(angle + 2 * np.pi, 2 * np.pi)
        idx_float = (angle / (2 * np.pi)) * period
        return torch.round(idx_float).long() % period

    def forward(self, time_cyc):
        hour_idx = self._cyc_to_index(time_cyc[..., 0], time_cyc[..., 1], 24)
        dow_idx = self._cyc_to_index(time_cyc[..., 2], time_cyc[..., 3], 7)
        month_idx = self._cyc_to_index(time_cyc[..., 4], time_cyc[..., 5], 12)

        hour_emb = self.hour_embed(hour_idx)
        dow_emb = self.dow_embed(dow_idx)
        month_emb = self.month_embed(month_idx)
        return self.fuse(torch.cat([time_cyc, hour_emb, dow_emb, month_emb], dim=-1))
