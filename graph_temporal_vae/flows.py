"""Normalizing-flow layers used by Graph-enhanced Temporal-VAE variants."""

import torch
import torch.nn as nn


class AffineCouplingLayer(nn.Module):
    """RealNVP affine coupling layer."""

    def __init__(self, dim, mask, hidden_dim=64):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale_limit = 1.0

    def forward(self, x):
        x_masked = x * self.mask
        s, t = self.net(x_masked).chunk(2, dim=-1)
        s = self.scale_limit * torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_j = (s * (1.0 - self.mask)).sum(dim=-1)
        return z, log_det_j


class ReverseLayer(nn.Module):
    """Dimension-reversal permutation with zero log determinant."""

    def forward(self, x):
        return x.flip(-1), torch.zeros(x.shape[0], device=x.device)


class RealNVP(nn.Module):
    """Alternating affine coupling and dimension-reversal layers."""

    def __init__(self, dim, n_layers=4, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList()
        mask1 = torch.zeros(dim)
        mask1[::2] = 1.0
        mask2 = 1.0 - mask1
        masks = [mask1, mask2] * (n_layers // 2)
        if n_layers % 2:
            masks.append(mask1)
        for index, mask in enumerate(masks):
            self.layers.append(AffineCouplingLayer(dim, mask, hidden_dim))
            if index < len(masks) - 1:
                self.layers.append(ReverseLayer())

    def forward(self, x):
        log_det_j_sum = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det_j = layer(x)
            log_det_j_sum += log_det_j
        return x, log_det_j_sum
