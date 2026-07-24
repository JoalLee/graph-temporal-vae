"""Decoder-side attention blocks."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedTemporalCrossAttention(nn.Module):
    """Let decoder states query unpooled encoder temporal anchors."""

    def __init__(self, dec_dim, enc_dim=None, n_heads=4, dropout=0.1):
        super().__init__()
        enc_dim = enc_dim if enc_dim is not None else dec_dim
        self.k_proj = nn.Conv1d(enc_dim, dec_dim, 1)
        self.v_proj = nn.Conv1d(enc_dim, dec_dim, 1)
        self.attn = nn.MultiheadAttention(
            dec_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.out_norm = nn.LayerNorm(dec_dim)

    def forward(self, h_dec, h_enc, target_obs_mask):
        q = h_dec.permute(0, 2, 1)
        k = self.k_proj(h_enc).permute(0, 2, 1)
        v = self.v_proj(h_enc).permute(0, 2, 1)

        if target_obs_mask is None:
            key_pad_mask = None
        elif target_obs_mask.dim() == 3:
            key_pad_mask = (target_obs_mask == 0).all(dim=-1)
        else:
            key_pad_mask = target_obs_mask == 0

        if key_pad_mask is not None:
            all_masked = key_pad_mask.all(dim=-1, keepdim=True)
            key_pad_mask = key_pad_mask.masked_fill(all_masked, False)

        out, _ = self.attn(q, k, v, key_padding_mask=key_pad_mask)
        return self.out_norm(out).permute(0, 2, 1)


class LocalContextMemoryAttention(nn.Module):
    """Support-aware local-memory cross-attention for decoder fusion."""

    def __init__(
        self,
        dec_dim,
        ctx_dim,
        n_heads=4,
        window_tokens=1,
        gate_init=-2.0,
        support_bias_scale=2.0,
        gate_support_power=0.0,
        gate_support_floor=0.0,
        dropout=0.1,
    ):
        super().__init__()
        if dec_dim % n_heads != 0:
            raise ValueError(
                f"dec_dim={dec_dim} must be divisible by n_heads={n_heads}"
            )

        self.dec_dim = int(dec_dim)
        self.ctx_dim = int(ctx_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.dec_dim // self.n_heads
        self.window_tokens = max(0, int(window_tokens))
        self.support_bias_scale = float(support_bias_scale)
        self.gate_support_power = max(0.0, float(gate_support_power))
        self.gate_support_floor = min(1.0, max(0.0, float(gate_support_floor)))
        self.dropout = float(dropout)

        self.q_proj = nn.Conv1d(self.dec_dim, self.dec_dim, 1)
        self.k_proj = nn.Conv1d(self.ctx_dim, self.dec_dim, 1)
        self.v_proj = nn.Conv1d(self.ctx_dim, self.dec_dim, 1)
        self.out_proj = nn.Conv1d(self.dec_dim, self.dec_dim, 1)
        self.out_norm = nn.LayerNorm(self.dec_dim)
        self.gate_proj = nn.Conv1d(self.dec_dim + 1, 1, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_init))

        self.last_attn_entropy = None
        self.last_attn_center_distance = None
        self.last_attn_support_mean = None
        self.last_attn_high_support_mass = None
        self.last_gate_mean = None
        self.last_gate_low_support_mean = None
        self.last_gate_high_support_mean = None

    def _local_window_mask(self, q_len, kv_len, device):
        q_pos = torch.arange(q_len, device=device)
        kv_pos = torch.arange(kv_len, device=device)
        mapped = torch.div(q_pos * kv_len, q_len, rounding_mode="floor")
        allowed = (
            kv_pos.unsqueeze(0) - mapped.unsqueeze(1)
        ).abs() <= self.window_tokens
        return ~allowed

    def forward(self, h_dec, local_ctx, support_tokens, support_high):
        batch_size, _, q_len = h_dec.shape
        _, _, kv_len = local_ctx.shape

        q = (
            self.q_proj(h_dec)
            .transpose(1, 2)
            .reshape(batch_size, q_len, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(local_ctx)
            .transpose(1, 2)
            .reshape(batch_size, kv_len, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(local_ctx)
            .transpose(1, 2)
            .reshape(batch_size, kv_len, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        local_mask = self._local_window_mask(q_len, kv_len, device=h_dec.device)
        scores = scores.masked_fill(
            local_mask.view(1, 1, q_len, kv_len), -1e4
        )

        if support_tokens is not None:
            support_bias = (
                self.support_bias_scale
                * (support_tokens.clamp(0.0, 1.0) - 0.5)
                * 2.0
            )
            scores = scores + support_bias.unsqueeze(1)

        attn = torch.softmax(scores, dim=-1)
        attn_summary = attn.detach()
        attn_mean = attn_summary.mean(dim=1)
        entropy = -(
            attn_summary * attn_summary.clamp_min(1e-8).log()
        ).sum(dim=-1)
        q_pos = torch.arange(q_len, device=h_dec.device)
        kv_pos = torch.arange(kv_len, device=h_dec.device)
        mapped = torch.div(q_pos * kv_len, q_len, rounding_mode="floor")
        distance = (
            kv_pos.unsqueeze(0) - mapped.unsqueeze(1)
        ).abs().to(attn_summary.dtype)
        self.last_attn_entropy = float(entropy.mean().item())
        self.last_attn_center_distance = float(
            (attn_mean * distance.unsqueeze(0)).sum(dim=-1).mean().item()
        )
        if support_tokens is not None:
            support = support_tokens.detach().clamp(0.0, 1.0).squeeze(1)
            self.last_attn_support_mean = float(
                (attn_mean * support.unsqueeze(1)).sum(dim=-1).mean().item()
            )
            high_support = (support >= 0.75).to(attn_mean.dtype)
            self.last_attn_high_support_mass = float(
                (attn_mean * high_support.unsqueeze(1)).sum(dim=-1).mean().item()
            )
        else:
            self.last_attn_support_mean = None
            self.last_attn_high_support_mass = None

        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = (
            out.permute(0, 2, 1, 3)
            .reshape(batch_size, q_len, self.dec_dim)
            .transpose(1, 2)
        )
        out = self.out_proj(out)
        out = self.out_norm(out.transpose(1, 2)).transpose(1, 2)

        if support_high is None:
            support_high = torch.ones(
                batch_size,
                1,
                q_len,
                device=h_dec.device,
                dtype=h_dec.dtype,
            )
        gate_in = torch.cat([h_dec, support_high], dim=1)
        gate = torch.sigmoid(self.gate_proj(gate_in))
        if self.gate_support_power > 0.0:
            support_gate = support_high.clamp(0.0, 1.0).pow(
                self.gate_support_power
            )
            if self.gate_support_floor > 0.0:
                support_gate = self.gate_support_floor + (
                    1.0 - self.gate_support_floor
                ) * support_gate
            gate = gate * support_gate

        gate_det = gate.detach()
        self.last_gate_mean = float(gate_det.mean().item())
        support_det = support_high.detach()
        low = support_det <= 0.50
        high = support_det >= 0.90
        self.last_gate_low_support_mean = (
            float(gate_det[low].mean().item()) if low.any() else None
        )
        self.last_gate_high_support_mean = (
            float(gate_det[high].mean().item()) if high.any() else None
        )
        return gate * out, gate
