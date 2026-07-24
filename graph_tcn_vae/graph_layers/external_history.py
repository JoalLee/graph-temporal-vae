"""Cross-window external-history retrieval context."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalHistoryContext(nn.Module):
    """Coarse leakage-safe history memory for external chunk context."""

    def __init__(
        self,
        target_dim,
        cond_dim,
        context_dim,
        context_steps,
        window_size,
        history_chunk_size=24,
        history_num_chunks=28,
        history_support_dim=6,
        hidden_dim=128,
        n_heads=4,
        dropout=0.1,
        gate_init=-2.0,
        use_retrieval_bias=False,
        time_decay=0.0,
        support_bias=0.0,
        null_penalty=0.0,
    ):
        super().__init__()
        self.target_dim = int(target_dim)
        self.cond_dim = int(cond_dim)
        self.context_dim = int(context_dim)
        self.context_steps = int(context_steps)
        self.window_size = int(window_size)
        self.history_chunk_size = int(history_chunk_size)
        self.history_num_chunks = int(history_num_chunks)
        self.history_support_dim = int(history_support_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        if self.history_support_dim < 6:
            raise ValueError(
                "history_support_dim must be at least 6 because field 5 is the null_flag"
            )
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"external_history_dim={self.hidden_dim} must be divisible by "
                f"external_history_heads={self.n_heads}"
            )
        if self.hidden_dim % 2 != 0:
            raise ValueError(
                f"external_history_dim={self.hidden_dim} must be even"
            )
        self.use_retrieval_bias = bool(use_retrieval_bias)
        self.time_decay = float(time_decay)
        self.support_bias = float(support_bias)
        self.null_penalty = float(null_penalty)

        self.timestep_proj = nn.Sequential(
            nn.Linear(self.target_dim * 2 + self.cond_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.support_proj = nn.Sequential(
            nn.Linear(self.history_support_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.null_token = nn.Parameter(torch.zeros(self.hidden_dim))
        self.inter_gru = nn.GRU(
            self.hidden_dim,
            self.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.query_proj = nn.Sequential(
            nn.Linear(self.cond_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.context_dim),
        )
        self.out_gate = nn.Parameter(torch.tensor(float(gate_init)))
        for name in (
            "last_gate",
            "last_attn_entropy",
            "last_valid_fraction",
            "last_null_fraction",
            "last_top1_mass",
            "last_top3_mass",
            "last_attended_time_dist",
            "last_attended_support",
            "last_attended_null_fraction",
        ):
            setattr(self, name, None)

    def _current_queries(self, cond, obs_mask):
        batch_size, window, _ = cond.shape
        chunk = self.history_chunk_size
        pad_len = (chunk - (window % chunk)) % chunk
        if pad_len:
            cond_pad = F.pad(cond.transpose(1, 2), (0, pad_len)).transpose(1, 2)
            obs_pad = F.pad(obs_mask.transpose(1, 2), (0, pad_len)).transpose(1, 2)
        else:
            cond_pad = cond
            obs_pad = obs_mask
        n_chunks = cond_pad.shape[1] // chunk
        cond_chunks = cond_pad.reshape(
            batch_size, n_chunks, chunk, self.cond_dim
        ).mean(dim=2)
        support_chunks = obs_pad.reshape(
            batch_size, n_chunks, chunk, obs_mask.shape[-1]
        ).float().mean(dim=(2, 3))
        return self.query_proj(
            torch.cat([cond_chunks, support_chunks.unsqueeze(-1)], dim=-1)
        )

    def _manual_cross_attention(
        self, query, memory, support, time_dist, null_flag, valid
    ):
        logits = torch.matmul(query, memory.transpose(1, 2)) / math.sqrt(
            float(self.hidden_dim)
        )
        if self.use_retrieval_bias:
            support_quality = support[..., 0].clamp(0.0, 1.0)
            support_quality = support_quality.masked_fill(valid <= 0.0, 0.0)
            bias = self.support_bias * support_quality.unsqueeze(1)
            bias = bias - self.time_decay * time_dist.clamp_min(0.0).unsqueeze(1)
            bias = bias - self.null_penalty * null_flag.to(query.dtype).unsqueeze(1)
            logits = logits + bias
        logits = logits.masked_fill(valid.unsqueeze(1) <= 0.0, -1e4)
        attn_weights = torch.softmax(logits, dim=-1)
        return torch.matmul(attn_weights, memory), attn_weights

    def forward(self, history, cond, obs_mask):
        target = history["history_target"]
        h_obs = history["history_obs_mask"].to(dtype=target.dtype)
        h_cond = history["history_cond"].to(dtype=target.dtype)
        support = history["history_support"].to(dtype=target.dtype)
        if support.shape[-1] != self.history_support_dim:
            raise ValueError(
                f"history_support has {support.shape[-1]} fields, "
                f"expected {self.history_support_dim}"
            )
        time_dist = history["history_time_dist"].to(dtype=target.dtype)
        valid = history["history_chunk_valid"].to(dtype=target.dtype)

        batch_size, chunks, steps, _ = target.shape
        step_in = torch.cat([target, h_obs, h_cond], dim=-1)
        step_tok = self.timestep_proj(
            step_in.reshape(batch_size * chunks * steps, -1)
        ).view(batch_size, chunks, steps, self.hidden_dim)
        chunk_tok = step_tok.mean(dim=2)
        time_norm = (
            time_dist / max(float(self.history_num_chunks), 1.0)
        ).unsqueeze(-1)
        meta_tok = self.support_proj(
            torch.cat([support, time_norm], dim=-1)
        )
        null_flag = (support[..., 5] > 0.5) | (valid <= 0.0)
        null_tok = self.null_token.view(1, 1, -1).expand(
            batch_size, chunks, -1
        )
        chunk_tok = torch.where(
            null_flag.unsqueeze(-1), null_tok, chunk_tok
        ) + meta_tok
        memory, _ = self.inter_gru(chunk_tok)
        query = self._current_queries(cond, obs_mask)
        if self.use_retrieval_bias:
            attn_out, attn_weights = self._manual_cross_attention(
                query, memory, support, time_dist, null_flag, valid
            )
        else:
            attn_out, attn_weights = self.cross_attn(
                query,
                memory,
                memory,
                need_weights=True,
                average_attn_weights=True,
            )
        ctx = self.out_proj(attn_out).transpose(1, 2)
        if ctx.shape[-1] != self.context_steps:
            ctx = F.interpolate(
                ctx,
                size=self.context_steps,
                mode="linear",
                align_corners=False,
            )
        gate = torch.sigmoid(self.out_gate)
        self.last_gate = float(gate.detach().item())
        with torch.no_grad():
            p = attn_weights.detach().clamp_min(1e-8)
            self.last_attn_entropy = float(
                (-(p * p.log()).sum(dim=-1)).mean().item()
            )
            self.last_valid_fraction = float((valid > 0).float().mean().item())
            self.last_null_fraction = float(null_flag.float().mean().item())
            topk = torch.topk(
                p, k=min(3, p.shape[-1]), dim=-1
            ).values
            self.last_top1_mass = float(topk[..., 0].mean().item())
            self.last_top3_mass = float(topk.sum(dim=-1).mean().item())
            self.last_attended_time_dist = float(
                (p * time_dist.unsqueeze(1)).sum(dim=-1).mean().item()
            )
            self.last_attended_support = float(
                (p * support[..., 0].unsqueeze(1)).sum(dim=-1).mean().item()
            )
            self.last_attended_null_fraction = float(
                (p * null_flag.float().unsqueeze(1))
                .sum(dim=-1)
                .mean()
                .item()
            )
        return gate * ctx
