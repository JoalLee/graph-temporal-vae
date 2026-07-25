"""Local chunked graph-attention branch."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_temporal_vae.graph_layers.token_graph import TokenGraphSelfBlock


class LocalChunkGraphBranch(nn.Module):
    def __init__(
        self,
        n_features,
        window_size,
        chunk_size=6,
        d_model=128,
        n_heads=4,
        dropout=0.1,
        gate_init=-2.0,
        ffn_mult=4,
        use_mask_embed=False,
        out_proj_init_std=0.0,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.chunk_size = int(chunk_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.use_mask_embed = bool(use_mask_embed)
        self.out_proj_init_std = float(out_proj_init_std)
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"local_chunk_graph_dim={self.d_model} must be divisible by "
                f"local_chunk_graph_heads={self.n_heads}"
            )

        self.token_embed = nn.Linear(self.chunk_size, self.d_model)
        self.mask_embed = None
        self.ratio_embed = None
        if self.use_mask_embed:
            self.mask_embed = nn.Linear(self.chunk_size, self.d_model)
            self.ratio_embed = nn.Linear(1, self.d_model)
            nn.init.zeros_(self.mask_embed.weight)
            nn.init.zeros_(self.mask_embed.bias)
            nn.init.zeros_(self.ratio_embed.weight)
            nn.init.zeros_(self.ratio_embed.bias)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.block = TokenGraphSelfBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=dropout,
            ffn_mult=ffn_mult,
        )
        self.out_proj = nn.Linear(self.d_model, self.chunk_size)
        if self.out_proj_init_std > 0:
            nn.init.normal_(
                self.out_proj.weight, mean=0.0, std=self.out_proj_init_std
            )
        else:
            nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.out_gate = nn.Parameter(
            torch.full((1, self.n_features, 1), float(gate_init))
        )
        self.last_gate = None
        self.last_out_proj_weight_norm = None
        self.last_obs_ratio_mean = None

    def forward(self, x, chunk_obs_mask=None):
        batch_size, channels, window = x.shape
        if channels != self.n_features:
            raise ValueError(
                f"LocalChunkGraphBranch expected {self.n_features} "
                f"features, got {channels}"
            )

        pad_len = (self.chunk_size - (window % self.chunk_size)) % self.chunk_size
        x_padded = F.pad(x, (0, pad_len)) if pad_len else x
        padded_window = x_padded.shape[-1]
        n_chunks = padded_window // self.chunk_size
        x_chunks = (
            x_padded.view(
                batch_size, channels, n_chunks, self.chunk_size
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        tok = self.token_embed(x_chunks)
        self.last_obs_ratio_mean = None
        if self.use_mask_embed and chunk_obs_mask is not None:
            mask_padded = (
                F.pad(chunk_obs_mask, (0, pad_len))
                if pad_len
                else chunk_obs_mask
            )
            m_chunks = (
                mask_padded.view(
                    batch_size, channels, n_chunks, self.chunk_size
                )
                .permute(0, 2, 1, 3)
                .contiguous()
                .to(dtype=x_chunks.dtype)
            )
            obs_ratio = m_chunks.mean(dim=-1, keepdim=True)
            tok = tok + self.mask_embed(m_chunks) + self.ratio_embed(obs_ratio)
            self.last_obs_ratio_mean = float(
                obs_ratio.detach().mean().item()
            )
        tok = self.token_norm(tok).view(
            batch_size * n_chunks, channels, self.d_model
        )
        tok_out, _ = self.block(tok, need_weights=False)
        tok_out = tok_out.view(
            batch_size, n_chunks, channels, self.d_model
        )
        delta = (
            self.out_proj(tok_out)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, channels, padded_window)
        )
        if pad_len:
            delta = delta[:, :, :window]
        gate = torch.sigmoid(self.out_gate)
        self.last_gate = float(gate.detach().mean().item())
        self.last_out_proj_weight_norm = float(
            self.out_proj.weight.detach().norm().item()
        )
        return x + gate * delta
