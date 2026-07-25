"""Dynamic feature self-attention graph layer."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_temporal_vae.graph_blocks.tcn import WindowTokenFFN


class InputGraphLayer(nn.Module):
    """Relation-aware or homogeneous feature graph over target channels."""

    def __init__(
        self,
        n_features,
        window_size,
        n_heads=4,
        head_dim=64,
        dropout=0.1,
        use_temporal_cnn=True,
        aux_dim=0,
        n_chem=0,
        enable_cross_modal_floor=False,
        disable_rel_scale=False,
        disable_prior_bias=False,
        disable_aux_bias=False,
        use_homogeneous=False,
        use_ffn=False,
        ffn_mult=4,
        aux_bias_init_mode="legacy_zero",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_chem = n_chem
        self.n_psd = n_features - n_chem
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = head_dim
        self.d_model = n_heads * head_dim
        self.aux_dim = aux_dim
        self.use_homogeneous = use_homogeneous
        self.disable_rel_scale = disable_rel_scale
        self.disable_prior_bias = disable_prior_bias
        self.disable_aux_bias = disable_aux_bias
        self.aux_bias_init_mode = str(aux_bias_init_mode)
        if self.aux_bias_init_mode not in {"legacy_zero", "one_sided_zero"}:
            raise ValueError(
                "aux_bias_init_mode must be 'legacy_zero' or 'one_sided_zero'"
            )

        self.q_proj = nn.Linear(window_size, self.d_model)
        self.k_proj = nn.Linear(window_size, self.d_model)
        self.v_proj = nn.Linear(window_size, self.d_model)
        # Preserve the public monolith's state-dict contract: these parameters
        # are created even in homogeneous mode, although that forward path does
        # not consume relation-specific terms.
        if not disable_rel_scale:
            self.rel_log_scale = nn.Parameter(torch.zeros(4, n_heads))
        self.tau_param = nn.Parameter(torch.zeros(1))
        if not disable_prior_bias:
            self.prior_beta = nn.Parameter(torch.ones(1) * 0.5)
            self.prior_beta_cross = nn.Parameter(torch.ones(1) * 0.1)

        self.aux_rank = 4
        if aux_dim > 0 and not disable_aux_bias:
            def make_aux_bias_modules():
                return (
                    nn.Parameter(torch.randn(n_chem, self.aux_rank) * 0.02),
                    nn.Parameter(torch.randn(self.n_psd, self.aux_rank) * 0.02),
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
            self.node_embed_c, self.node_embed_p, self.aux_bias_net = modules
            nn.init.zeros_(self.aux_bias_net[-1].weight)
            nn.init.zeros_(self.aux_bias_net[-1].bias)

        if use_homogeneous:
            self.out_proj = nn.Linear(self.d_model, window_size)
            self.norm = nn.LayerNorm(window_size)
        else:
            self.out_chem = nn.Linear(self.d_model, window_size)
            self.out_psd = nn.Linear(self.d_model, window_size)
            self.norm_chem = nn.LayerNorm(window_size)
            self.norm_psd = nn.LayerNorm(window_size)
        self.ffn = (
            WindowTokenFFN(window_size, mult=ffn_mult, dropout=dropout)
            if use_ffn
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights = None
        self.last_attention_weights_heads = None
        self.last_attention_weights_heads_batch = None
        self.record_diagnostics = False

    def _to_heads(self, tensor, n_nodes, batch_size):
        return tensor.view(
            batch_size, n_nodes, self.n_heads, self.head_dim
        ).transpose(1, 2)

    def _from_heads(self, tensor, n_nodes, batch_size):
        return tensor.transpose(1, 2).contiguous().view(
            batch_size, n_nodes, self.d_model
        )

    def forward(self, x, obs_mask=None, x_aux=None):
        batch_size, channels, window = x.shape
        if self.use_homogeneous:
            q = self._to_heads(self.q_proj(x), channels, batch_size)
            k = self._to_heads(self.k_proj(x), channels, batch_size)
            v = self._to_heads(self.v_proj(x), channels, batch_size)
            scale = np.sqrt(self.head_dim) * (
                0.5 + 0.5 * torch.sigmoid(self.tau_param)
            )
            logits = torch.matmul(q, k.transpose(-1, -2)) / scale
            attn = self.dropout(F.softmax(logits, dim=-1))
            out = torch.matmul(attn, v)
            out = self._from_heads(out, channels, batch_size)
            out = self.norm(self.out_proj(out) + x)
            if self.ffn is not None:
                out = self.ffn(out)
            detached = attn.detach()
            attn_avg = detached.mean(dim=1)
            self.last_attention_weights = attn_avg.mean(dim=0)
            self.last_attention_weights_heads = detached.mean(dim=0)
            self.last_attention_weights_heads_batch = (
                detached if self.record_diagnostics else None
            )
            return out, attn_avg

        n_chem, n_psd = self.n_chem, self.n_psd
        x_chem = x[:, :n_chem, :]
        x_psd = x[:, n_chem:, :]
        if obs_mask is None:
            mask_chem = torch.ones(
                batch_size, n_chem, window, device=x.device
            )
            mask_psd = torch.ones(
                batch_size, n_psd, window, device=x.device
            )
        else:
            mask = obs_mask.permute(0, 2, 1).float()
            mask_chem = mask[:, :n_chem, :]
            mask_psd = mask[:, n_chem:, :]

        q_chem = self._to_heads(
            self.q_proj(x_chem), n_chem, batch_size
        )
        q_psd = self._to_heads(self.q_proj(x_psd), n_psd, batch_size)
        k_chem = self._to_heads(
            self.k_proj(x_chem), n_chem, batch_size
        )
        k_psd = self._to_heads(self.k_proj(x_psd), n_psd, batch_size)
        v_chem = self._to_heads(
            self.v_proj(x_chem), n_chem, batch_size
        )
        v_psd = self._to_heads(self.v_proj(x_psd), n_psd, batch_size)
        scale = np.sqrt(self.head_dim) * (
            0.5 + 0.5 * torch.sigmoid(self.tau_param)
        )
        logits_cc = torch.matmul(q_chem, k_chem.transpose(-1, -2)) / scale
        logits_pp = torch.matmul(q_psd, k_psd.transpose(-1, -2)) / scale
        logits_cp = torch.matmul(q_chem, k_psd.transpose(-1, -2)) / scale
        logits_pc = torch.matmul(q_psd, k_chem.transpose(-1, -2)) / scale

        if not self.disable_rel_scale:
            def relation(index):
                return self.rel_log_scale[index].view(
                    1, self.n_heads, 1, 1
                )
            logits_cc = logits_cc + relation(0)
            logits_pp = logits_pp + relation(1)
            logits_cp = logits_cp + relation(2)
            logits_pc = logits_pc + relation(3)

        if not self.disable_prior_bias:
            eps = 1e-6
            co_cc = (
                torch.matmul(mask_chem, mask_chem.transpose(1, 2)) / window
            ).unsqueeze(1)
            co_pp = (
                torch.matmul(mask_psd, mask_psd.transpose(1, 2)) / window
            ).unsqueeze(1)
            co_cp = (
                torch.matmul(mask_chem, mask_psd.transpose(1, 2)) / window
            ).unsqueeze(1)
            co_pc = co_cp.transpose(-1, -2)
            logits_cc = logits_cc + self.prior_beta * torch.log(co_cc + eps)
            logits_pp = logits_pp + self.prior_beta * torch.log(co_pp + eps)
            logits_cp = logits_cp + self.prior_beta_cross * torch.log(co_cp + eps)
            logits_pc = logits_pc + self.prior_beta_cross * torch.log(co_pc + eps)

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
            node_chem = self.node_embed_c.unsqueeze(0).unsqueeze(0)
            node_psd = self.node_embed_p.unsqueeze(0).unsqueeze(0)
            bias_u_chem = node_chem * u_scale
            bias_v_chem = node_chem * v_scale
            bias_u_psd = node_psd * u_scale
            bias_v_psd = node_psd * v_scale
            logits_cc = logits_cc + torch.matmul(
                bias_u_chem, bias_v_chem.transpose(-1, -2)
            )
            logits_pp = logits_pp + torch.matmul(
                bias_u_psd, bias_v_psd.transpose(-1, -2)
            )
            logits_cp = logits_cp + torch.matmul(
                bias_u_chem, bias_v_psd.transpose(-1, -2)
            )
            logits_pc = logits_pc + torch.matmul(
                bias_u_psd, bias_v_chem.transpose(-1, -2)
            )

        eye_chem = torch.eye(
            n_chem, device=x.device, dtype=torch.bool
        ).view(1, 1, n_chem, n_chem)
        eye_psd = torch.eye(
            n_psd, device=x.device, dtype=torch.bool
        ).view(1, 1, n_psd, n_psd)
        logits_cc = logits_cc.masked_fill(eye_chem, -1e4)
        logits_pp = logits_pp.masked_fill(eye_psd, -1e4)

        attn_chem = self.dropout(
            F.softmax(torch.cat([logits_cc, logits_cp], dim=-1), dim=-1)
        )
        attn_psd = self.dropout(
            F.softmax(torch.cat([logits_pc, logits_pp], dim=-1), dim=-1)
        )
        attn_cc, attn_cp = (
            attn_chem[:, :, :, :n_chem],
            attn_chem[:, :, :, n_chem:],
        )
        attn_pc, attn_pp = (
            attn_psd[:, :, :, :n_chem],
            attn_psd[:, :, :, n_chem:],
        )
        out_chem = torch.matmul(attn_cc, v_chem) + torch.matmul(
            attn_cp, v_psd
        )
        out_psd = torch.matmul(attn_pc, v_chem) + torch.matmul(
            attn_pp, v_psd
        )
        out_chem = self.norm_chem(
            self.out_chem(
                self._from_heads(out_chem, n_chem, batch_size)
            )
            + x_chem
        )
        out_psd = self.norm_psd(
            self.out_psd(self._from_heads(out_psd, n_psd, batch_size))
            + x_psd
        )
        out = torch.cat([out_chem, out_psd], dim=1)
        if self.ffn is not None:
            out = self.ffn(out)

        attn_full = torch.zeros(
            batch_size,
            self.n_heads,
            channels,
            channels,
            device=x.device,
        )
        attn_full[:, :, :n_chem, :n_chem] = attn_cc
        attn_full[:, :, :n_chem, n_chem:] = attn_cp
        attn_full[:, :, n_chem:, :n_chem] = attn_pc
        attn_full[:, :, n_chem:, n_chem:] = attn_pp
        detached = attn_full.detach()
        attn_avg = detached.mean(dim=1)
        self.last_attention_weights = attn_avg.mean(dim=0)
        self.last_attention_weights_heads = detached.mean(dim=0)
        self.last_attention_weights_heads_batch = (
            detached if self.record_diagnostics else None
        )
        return out, attn_avg
