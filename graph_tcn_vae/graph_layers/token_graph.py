"""Persistent d_model token-graph self/cross-attention blocks."""

import torch.nn as nn


class TokenGraphFFN(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.1):
        super().__init__()
        hidden = int(mult) * int(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TokenGraphSelfBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = TokenGraphFFN(d_model, mult=ffn_mult, dropout=dropout)
        self.last_attention_weights = None
        self.last_attention_weights_heads = None
        self.last_attention_weights_heads_batch = None
        self.record_diagnostics = False

    def forward(self, x, need_weights=True):
        h = self.attn_norm(x)
        attn_out, attn_weights = self.attn(
            h,
            h,
            h,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        attn_avg = None
        if attn_weights is not None:
            detached = attn_weights.detach()
            attn_avg = detached.mean(dim=1)
            self.last_attention_weights = attn_avg.mean(dim=0)
            self.last_attention_weights_heads = detached.mean(dim=0)
            self.last_attention_weights_heads_batch = (
                detached if self.record_diagnostics else None
            )
        else:
            self.last_attention_weights = None
            self.last_attention_weights_heads = None
            self.last_attention_weights_heads_batch = None
        return x, attn_avg


class TokenGraphCrossBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = TokenGraphFFN(d_model, mult=ffn_mult, dropout=dropout)
        self.last_attention_weights = None
        self.last_attention_weights_heads = None
        self.last_attention_weights_heads_batch = None
        self.record_diagnostics = False

    def forward(self, x_query, x_kv, need_weights=True):
        q = self.query_norm(x_query)
        kv = self.kv_norm(x_kv)
        attn_out, attn_weights = self.attn(
            q,
            kv,
            kv,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x_query + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        attn_avg = None
        if attn_weights is not None:
            detached = attn_weights.detach()
            attn_avg = detached.mean(dim=1)
            self.last_attention_weights = attn_avg.mean(dim=0)
            self.last_attention_weights_heads = detached.mean(dim=0)
            self.last_attention_weights_heads_batch = (
                detached if self.record_diagnostics else None
            )
        else:
            self.last_attention_weights = None
            self.last_attention_weights_heads = None
            self.last_attention_weights_heads_batch = None
        return x, attn_avg
