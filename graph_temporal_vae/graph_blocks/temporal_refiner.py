"""Post-TCN temporal refinement attention block."""

import torch
import torch.nn as nn

from graph_temporal_vae.graph_blocks.attention import RotarySelfAttention


class TemporalObservationRefiner(nn.Module):
    """Observation-aware temporal self-attention refiner for encoder states."""

    def __init__(
        self,
        hidden_dim,
        window_size,
        attn_dim=128,
        n_heads=4,
        gate_init=-2.0,
        fixed_gate=None,
        obs_bias_init=1.0,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.window_size = int(window_size)
        self.attn_dim = int(attn_dim)
        self.n_heads = int(n_heads)

        self.in_proj = nn.Conv1d(hidden_dim, self.attn_dim, 1)
        self.in_norm = nn.LayerNorm(self.attn_dim)
        self.attn = RotarySelfAttention(
            embed_dim=self.attn_dim,
            num_heads=self.n_heads,
            dropout=dropout,
        )
        self.out_proj = nn.Conv1d(self.attn_dim, hidden_dim, 1)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.fixed_gate = None if fixed_gate is None else float(fixed_gate)
        if self.fixed_gate is None:
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.register_parameter("gate", None)
        self.obs_bias_scale = nn.Parameter(torch.tensor(float(obs_bias_init)))

        self.last_gate = None
        self.last_missing_query_attn_entropy = None
        self.last_observed_query_attn_entropy = None

    def forward(self, h, target_obs_mask=None):
        h_res = h
        x = self.in_proj(h).transpose(1, 2)
        x = self.in_norm(x)

        attn_mask = None
        if target_obs_mask is not None:
            obs_rate = target_obs_mask.float().mean(dim=-1)
            fully_missing = obs_rate <= 0.0
            if fully_missing.any():
                all_missing = fully_missing.all(dim=1, keepdim=True)
                fully_missing = fully_missing.masked_fill(all_missing, False)

            key_bias = (
                self.obs_bias_scale.to(x.dtype)
                * obs_rate.to(x.dtype).unsqueeze(1).expand(-1, x.size(1), -1)
            )
            if fully_missing.any():
                hard_bias = (
                    fully_missing.to(x.dtype).unsqueeze(1).expand(-1, x.size(1), -1)
                    * (-1e9)
                )
                key_bias = key_bias + hard_bias

            attn_mask = key_bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            attn_mask = attn_mask.reshape(
                x.size(0) * self.n_heads, x.size(1), x.size(1)
            )

        attn_out, attn_weights = self.attn(
            x,
            attn_mask=attn_mask,
            need_weights=True,
        )

        if self.fixed_gate is None:
            gate = torch.sigmoid(self.gate)
        else:
            gate = x.new_tensor(max(0.0, min(1.0, self.fixed_gate)))
        self.last_gate = float(gate.detach().item())

        if attn_weights is not None:
            attn_mean = attn_weights.detach().mean(dim=1)
            entropy = -(
                attn_mean.clamp_min(1e-8) * attn_mean.clamp_min(1e-8).log()
            ).sum(dim=-1)
            if target_obs_mask is not None:
                query_missing = (target_obs_mask == 0).all(dim=-1)
                query_observed = ~query_missing
                self.last_missing_query_attn_entropy = (
                    float(entropy[query_missing].mean().item())
                    if query_missing.any()
                    else None
                )
                self.last_observed_query_attn_entropy = (
                    float(entropy[query_observed].mean().item())
                    if query_observed.any()
                    else None
                )
            else:
                self.last_missing_query_attn_entropy = None
                self.last_observed_query_attn_entropy = float(entropy.mean().item())
        else:
            self.last_missing_query_attn_entropy = None
            self.last_observed_query_attn_entropy = None

        attn_out = self.out_proj(attn_out.transpose(1, 2))
        return self.out_norm(
            (h_res + gate * attn_out).transpose(1, 2)
        ).transpose(1, 2)
