"""Cross-attention from target features to auxiliary features."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_tcn_vae.graph_blocks.tcn import WindowTokenFFN


class CrossModalGraphLayer(nn.Module):
    """Target-query / auxiliary-key-value cross-attention."""

    def __init__(
        self,
        target_dim,
        aux_dim,
        window_size,
        n_heads=4,
        head_dim=64,
        dropout=0.1,
        use_temporal_cnn=True,
        disable_aux_bias=False,
        use_ffn=False,
        ffn_mult=4,
        aux_bias_init_mode="legacy_zero",
        query_gate_mode="legacy_hard",
    ):
        super().__init__()
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = n_heads * head_dim
        self.aux_bias_init_mode = str(aux_bias_init_mode)
        if self.aux_bias_init_mode not in {"legacy_zero", "one_sided_zero"}:
            raise ValueError(
                "aux_bias_init_mode must be 'legacy_zero' or 'one_sided_zero'"
            )
        self.query_gate_mode = str(query_gate_mode)
        if self.query_gate_mode not in {"legacy_hard", "soft", "none"}:
            raise ValueError(
                "query_gate_mode must be 'legacy_hard', 'soft', or 'none'"
            )

        self.tau_param = nn.Parameter(torch.zeros(1))
        self.prior_beta = nn.Parameter(torch.ones(1) * 0.5)
        self.aux_rank = 4
        if aux_dim > 0 and not disable_aux_bias:
            def make_aux_bias_modules():
                return (
                    nn.Parameter(torch.randn(target_dim, self.aux_rank) * 0.02),
                    nn.Parameter(torch.randn(aux_dim, self.aux_rank) * 0.02),
                    nn.Sequential(
                        nn.Linear(aux_dim * 2, 32),
                        nn.SiLU(),
                        nn.Linear(32, 2 * n_heads * self.aux_rank),
                    ),
                )

            if self.aux_bias_init_mode == "one_sided_zero":
                with torch.random.fork_rng(devices=[]):
                    modules = make_aux_bias_modules()
            else:
                modules = make_aux_bias_modules()
            self.node_embed_t, self.node_embed_a, self.aux_bias_net = modules
            nn.init.zeros_(self.aux_bias_net[-1].weight)
            nn.init.zeros_(self.aux_bias_net[-1].bias)

        self.q_proj = nn.Linear(window_size, self.d_model)
        self.k_proj = nn.Linear(window_size, self.d_model)
        self.v_proj = nn.Linear(window_size, self.d_model)
        self.out_proj = nn.Linear(self.d_model, window_size)
        self.layer_norm = nn.LayerNorm(window_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = (
            WindowTokenFFN(window_size, mult=ffn_mult, dropout=dropout)
            if use_ffn
            else None
        )
        self.last_attention_weights = None
        self.last_attention_weights_heads = None
        self.last_attention_weights_heads_batch = None
        self.record_diagnostics = False

    def forward(self, x_target, x_aux, target_mask=None):
        batch_size, target_channels, window = x_target.shape
        aux_channels = x_aux.shape[1]
        if target_mask is None:
            target_mask_t = torch.ones(
                batch_size,
                target_channels,
                window,
                device=x_target.device,
            )
        else:
            target_mask_t = target_mask.permute(0, 2, 1).float()

        q = self.q_proj(x_target).view(
            batch_size,
            target_channels,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = self.k_proj(x_aux).view(
            batch_size, aux_channels, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x_aux).view(
            batch_size, aux_channels, self.n_heads, self.head_dim
        ).transpose(1, 2)

        if target_mask is None:
            obs_rate = torch.ones(
                batch_size, target_channels, device=x_target.device
            )
        else:
            obs_rate = target_mask_t.sum(dim=2) / window
        obs_rate_expanded = obs_rate.unsqueeze(1).unsqueeze(-1).expand(
            -1, self.n_heads, -1, aux_channels
        )

        temperature = 0.5 + 0.5 * torch.sigmoid(self.tau_param)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / (
            np.sqrt(self.head_dim) * temperature
        )

        if hasattr(self, "aux_bias_net") and x_aux is not None:
            aux_ctx = torch.cat(
                [x_aux.mean(dim=-1), x_aux.std(dim=-1).clamp(min=1e-6)],
                dim=-1,
            )
            u_scale, v_scale = self.aux_bias_net(aux_ctx).chunk(2, dim=-1)
            if self.aux_bias_init_mode == "one_sided_zero":
                u_scale = 1.0 + u_scale
            rank = self.aux_rank
            u_scale = u_scale.view(batch_size, self.n_heads, 1, rank)
            v_scale = v_scale.view(batch_size, self.n_heads, 1, rank)
            bias_u = self.node_embed_t.unsqueeze(0).unsqueeze(0) * u_scale
            bias_v = self.node_embed_a.unsqueeze(0).unsqueeze(0) * v_scale
            attn_logits = attn_logits + torch.matmul(
                bias_u, bias_v.transpose(-1, -2)
            )

        if self.query_gate_mode == "soft":
            soft_gate = torch.sigmoid(20.0 * (obs_rate_expanded - 0.05))
            attn_weights = F.softmax(attn_logits, dim=-1) * soft_gate
        elif self.query_gate_mode == "legacy_hard":
            threshold = 0.1
            attn_logits = attn_logits.masked_fill(
                obs_rate_expanded < threshold, -1e4
            )
            attn_weights = F.softmax(attn_logits, dim=-1)
            valid_row_mask = (
                obs_rate_expanded >= threshold
            ).any(dim=-1, keepdim=True)
            attn_weights = attn_weights * valid_row_mask.float()
        else:
            attn_weights = F.softmax(attn_logits, dim=-1)

        attn = self.dropout(attn_weights)
        detached = attn_weights.detach()
        attn_avg = detached.mean(dim=1)
        self.last_attention_weights = attn_avg.mean(dim=0)
        self.last_attention_weights_heads = detached.mean(dim=0)
        self.last_attention_weights_heads_batch = (
            detached if self.record_diagnostics else None
        )

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(
            batch_size, target_channels, self.d_model
        )
        out = self.out_proj(out)
        out = self.layer_norm(out + x_target)
        if self.ffn is not None:
            out = self.ffn(out)
        return out, attn_avg


AuxiliaryConditionCrossAttention = CrossModalGraphLayer
