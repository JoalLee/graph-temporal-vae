"""Reusable attention blocks used by the graph encoder."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotarySelfAttention(nn.Module):
    """Batch-first self-attention with rotary position encoding on Q/K."""

    def __init__(self, embed_dim, num_heads, dropout=0.1, rope_base=10000.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires an even head_dim, got "
                f"embed_dim={embed_dim}, num_heads={num_heads}"
            )
        self.rope_base = float(rope_base)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _to_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _from_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

    def _rope_cache_from_positions(self, positions, device, dtype):
        inv_freq = 1.0 / (
            self.rope_base
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    device=device,
                    dtype=torch.float32,
                )
                / self.head_dim
            )
        )
        pos = positions.to(device=device, dtype=torch.float32)
        freqs = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        cos = freqs.cos().to(dtype=dtype).unsqueeze(1)
        sin = freqs.sin().to(dtype=dtype).unsqueeze(1)
        return cos, sin

    @staticmethod
    def _apply_rope(x, cos, sin):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos],
            dim=-1,
        )
        return x_rot.flatten(start_dim=-2)

    def _build_attn_bias(
        self,
        batch_size,
        q_len,
        k_len,
        device,
        dtype,
        key_padding_mask=None,
        attn_mask=None,
    ):
        attn_bias = None
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                if attn_mask.shape[0] == batch_size * self.num_heads:
                    attn_bias = attn_mask.view(
                        batch_size, self.num_heads, q_len, k_len
                    )
                elif attn_mask.shape[0] == batch_size:
                    attn_bias = attn_mask.unsqueeze(1)
                else:
                    raise ValueError(
                        f"Unexpected attn_mask shape {tuple(attn_mask.shape)} for "
                        f"batch={batch_size}, heads={self.num_heads}, "
                        f"q_len={q_len}, k_len={k_len}"
                    )
            else:
                attn_bias = attn_mask
            attn_bias = attn_bias.to(device=device, dtype=dtype)

        if key_padding_mask is not None:
            key_bias = torch.zeros(
                (batch_size, 1, 1, k_len), device=device, dtype=dtype
            )
            key_bias = key_bias.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(dtype).min,
            )
            attn_bias = key_bias if attn_bias is None else attn_bias + key_bias
        return attn_bias

    def forward_qkv(
        self,
        q_in,
        k_in,
        v_in,
        q_pos=None,
        kv_pos=None,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=False,
    ):
        batch_size, q_len, _ = q_in.shape
        _, k_len, _ = k_in.shape

        q = self._to_heads(self.q_proj(q_in))
        k = self._to_heads(self.k_proj(k_in))
        v = self._to_heads(self.v_proj(v_in))

        if q_pos is None:
            q_pos = torch.arange(
                q_len, device=q_in.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
        if kv_pos is None:
            kv_pos = torch.arange(
                k_len, device=k_in.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

        cos_q, sin_q = self._rope_cache_from_positions(
            q_pos, q_in.device, q.dtype
        )
        cos_k, sin_k = self._rope_cache_from_positions(
            kv_pos, k_in.device, k.dtype
        )
        q = self._apply_rope(q, cos_q, sin_q)
        k = self._apply_rope(k, cos_k, sin_k)

        attn_bias = self._build_attn_bias(
            batch_size=batch_size,
            q_len=q_len,
            k_len=k_len,
            device=q_in.device,
            dtype=q.dtype,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        if not need_weights:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            return self.out_proj(self._from_heads(out)), None

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        if attn_bias is not None:
            logits = logits + attn_bias.to(dtype=logits.dtype)
        attn_probs = self.dropout(F.softmax(logits, dim=-1))
        out = torch.matmul(attn_probs, v)
        return self.out_proj(self._from_heads(out)), attn_probs

    def forward(
        self,
        x,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=False,
    ):
        return self.forward_qkv(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )


class TemporalAttentionPool(nn.Module):
    """Learned observation-aware pooling across the window axis."""

    def __init__(self, input_dim):
        super().__init__()
        hidden = max(input_dim // 4, 16)
        self.score_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, obs_mask=None):
        x_t = x.permute(0, 2, 1)
        scores = self.score_net(x_t)
        if obs_mask is not None:
            any_obs = obs_mask.any(dim=-1, keepdim=True).float()
            scores = scores.masked_fill(any_obs == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        return (x_t * weights).sum(dim=1)


class AxialObservedAttentionBlock(nn.Module):
    """Observed-key axial attention that updates only missing target positions."""

    def __init__(
        self,
        n_features,
        window_size,
        attn_dim=64,
        n_heads=4,
        dropout=0.1,
        time_gate_init=0.0,
        cross_gate_init=0.0,
        cross_time_chunk=4,
        null_output=False,
        n_chem=0,
    ):
        super().__init__()
        if attn_dim % n_heads != 0:
            raise ValueError(
                f"axial attn_dim={attn_dim} must be divisible by n_heads={n_heads}"
            )
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.attn_dim = int(attn_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.attn_dim // self.n_heads
        self.cross_time_chunk = max(1, int(cross_time_chunk))
        self.null_output = bool(null_output)
        self.n_chem = int(n_chem)

        self.value_proj = nn.Linear(1, self.attn_dim)
        self.feature_embed = nn.Embedding(self.n_features, self.attn_dim)
        self.input_norm = nn.LayerNorm(self.attn_dim)
        self.time_attn = RotarySelfAttention(
            embed_dim=self.attn_dim,
            num_heads=self.n_heads,
            dropout=dropout,
        )
        self.time_scalar_out = nn.Linear(self.attn_dim, 1)
        self.cross_q = nn.Linear(self.attn_dim, self.attn_dim)
        self.cross_k = nn.Linear(self.attn_dim, self.attn_dim)
        self.cross_v = nn.Linear(self.attn_dim, self.attn_dim)
        self.cross_out = nn.Linear(self.attn_dim, self.attn_dim)
        self.cross_scalar_out = nn.Linear(self.attn_dim, 1)
        self.cross_dropout = nn.Dropout(dropout)
        self.time_gate = nn.Parameter(
            torch.full((self.n_features,), float(time_gate_init))
        )
        self.cross_gate = nn.Parameter(
            torch.full((self.n_features,), float(cross_gate_init))
        )

        for name in (
            "last_time_gate_mean",
            "last_time_gate_chem_mean",
            "last_time_gate_psd_mean",
            "last_cross_gate_mean",
            "last_cross_gate_chem_mean",
            "last_cross_gate_psd_mean",
            "last_time_no_key_fraction",
            "last_cross_no_key_fraction",
            "last_cross_valid_query_fraction",
            "last_cross_entropy_missing",
            "last_cross_top1_mass",
            "last_cross_top3_mass",
            "last_psd_to_chem_mass",
            "last_psd_to_psd_mass",
        ):
            setattr(self, name, None)

    def _split_gate_means(self, gate):
        if self.n_chem > 0 and self.n_features > self.n_chem:
            chem = float(gate[: self.n_chem].detach().mean().item())
            psd = float(gate[self.n_chem :].detach().mean().item())
        else:
            chem = None
            psd = None
        return float(gate.detach().mean().item()), chem, psd

    def _to_heads(self, x):
        batch_size, time_steps, channels, _ = x.shape
        return x.view(
            batch_size,
            time_steps,
            channels,
            self.n_heads,
            self.head_dim,
        ).permute(0, 3, 1, 2, 4)

    def _from_heads(self, x):
        batch_size, _, time_steps, channels, _ = x.shape
        return (
            x.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(batch_size, time_steps, channels, self.attn_dim)
        )

    def forward(self, x, target_obs_mask=None):
        batch_size, channels, window = x.shape
        if channels != self.n_features:
            raise ValueError(
                f"AxialObservedAttentionBlock expected {self.n_features} "
                f"features, got {channels}"
            )

        if target_obs_mask is None:
            obs = torch.ones(
                batch_size,
                channels,
                window,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            obs = target_obs_mask.permute(0, 2, 1).to(
                device=x.device, dtype=x.dtype
            )
        missing = 1.0 - obs

        feat_ids = torch.arange(channels, device=x.device)
        feat_emb = self.feature_embed(feat_ids).view(
            1, channels, 1, self.attn_dim
        )
        tok0 = self.input_norm(self.value_proj(x.unsqueeze(-1)) + feat_emb)

        time_gate_raw = torch.sigmoid(self.time_gate)
        cross_gate_raw = torch.sigmoid(self.cross_gate)
        time_gate = time_gate_raw.to(dtype=x.dtype).view(1, channels, 1)
        cross_gate = cross_gate_raw.to(dtype=x.dtype).view(1, channels, 1)
        (
            self.last_time_gate_mean,
            self.last_time_gate_chem_mean,
            self.last_time_gate_psd_mean,
        ) = self._split_gate_means(time_gate_raw)
        (
            self.last_cross_gate_mean,
            self.last_cross_gate_chem_mean,
            self.last_cross_gate_psd_mean,
        ) = self._split_gate_means(cross_gate_raw)

        time_in = tok0.reshape(batch_size * channels, window, self.attn_dim)
        feature_obs = obs.reshape(batch_size * channels, window) > 0.0
        no_time_key = ~feature_obs.any(dim=1, keepdim=True)
        key_padding_mask = (~feature_obs).masked_fill(no_time_key, False)
        self.last_time_no_key_fraction = float(
            no_time_key.float().mean().detach().item()
        )
        time_out, _ = self.time_attn(
            time_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        time_out = time_out.reshape(
            batch_size, channels, window, self.attn_dim
        )
        has_time_key = (~no_time_key).reshape(
            batch_size, channels, 1
        ).to(dtype=x.dtype)
        missing_time = missing * has_time_key
        time_delta = (
            self.time_scalar_out(time_out).squeeze(-1) * missing_time
        )
        h_time = (
            tok0
            + time_gate.unsqueeze(-1)
            * time_out
            * missing_time.unsqueeze(-1)
        )

        h_btcd = h_time.permute(0, 2, 1, 3).contiguous()
        obs_btc = obs.permute(0, 2, 1).contiguous() > 0.0
        missing_btc = missing.permute(0, 2, 1).contiguous() > 0.0
        cross_delta_btc = x.new_zeros((batch_size, window, channels))

        total_missing_queries = int(missing_btc.sum().detach().item())
        valid_query_sum = 0
        entropy_sum = 0.0
        entropy_count = 0
        top1_sum = 0.0
        top3_sum = 0.0
        psd_chem_sum = 0.0
        psd_psd_sum = 0.0
        psd_count = 0
        no_key_sum = 0.0
        no_key_count = 0

        scale = math.sqrt(self.head_dim)
        feature_idx = torch.arange(channels, device=x.device)
        psd_query_feature = (
            feature_idx >= self.n_chem
            if self.n_chem > 0
            else torch.zeros(
                channels, device=x.device, dtype=torch.bool
            )
        )

        for start in range(0, window, self.cross_time_chunk):
            end = min(window, start + self.cross_time_chunk)
            h_chunk = h_btcd[:, start:end]
            obs_chunk = obs_btc[:, start:end]
            miss_chunk = missing_btc[:, start:end]
            no_key = ~obs_chunk.any(dim=-1)
            no_key_sum += float(no_key.float().sum().detach().item())
            no_key_count += int(no_key.numel())

            key_valid = obs_chunk | no_key.unsqueeze(-1)
            q = self._to_heads(self.cross_q(h_chunk))
            k = self._to_heads(self.cross_k(h_chunk))
            v = self._to_heads(self.cross_v(h_chunk))
            scores = torch.matmul(q, k.transpose(-1, -2)) / scale
            scores = scores.masked_fill(
                ~key_valid[:, None, :, None, :], -1e4
            )
            attn = self.cross_dropout(F.softmax(scores, dim=-1))
            out = torch.matmul(attn, v)
            out = self.cross_out(self._from_heads(out))
            scalar = self.cross_scalar_out(out).squeeze(-1)

            valid_query = miss_chunk & (~no_key.unsqueeze(-1))
            valid_query_sum += int(valid_query.sum().detach().item())
            cross_delta_btc[:, start:end] = scalar * valid_query.to(
                dtype=scalar.dtype
            )

            with torch.no_grad():
                if valid_query.any():
                    attn_mean = attn.detach().mean(dim=1).clamp_min(1e-8)
                    entropy = -(
                        attn_mean * attn_mean.log()
                    ).sum(dim=-1)
                    entropy_sum += float(entropy[valid_query].sum().item())
                    entropy_count += int(valid_query.sum().item())
                    sorted_mass = torch.sort(
                        attn_mean, dim=-1, descending=True
                    ).values
                    top1_sum += float(
                        sorted_mass[..., 0][valid_query].sum().item()
                    )
                    top3_sum += float(
                        sorted_mass[..., : min(3, channels)]
                        .sum(dim=-1)[valid_query]
                        .sum()
                        .item()
                    )
                    if self.n_chem > 0 and self.n_chem < channels:
                        psd_query = valid_query & psd_query_feature.view(
                            1, 1, channels
                        )
                        if psd_query.any():
                            chem_mass = attn_mean[..., : self.n_chem].sum(
                                dim=-1
                            )
                            psd_mass = attn_mean[..., self.n_chem :].sum(
                                dim=-1
                            )
                            psd_chem_sum += float(
                                chem_mass[psd_query].sum().item()
                            )
                            psd_psd_sum += float(
                                psd_mass[psd_query].sum().item()
                            )
                            psd_count += int(psd_query.sum().item())

        self.last_cross_no_key_fraction = (
            no_key_sum / no_key_count if no_key_count > 0 else None
        )
        self.last_cross_valid_query_fraction = (
            valid_query_sum / total_missing_queries
            if total_missing_queries > 0
            else None
        )
        self.last_cross_entropy_missing = (
            entropy_sum / entropy_count if entropy_count > 0 else None
        )
        self.last_cross_top1_mass = (
            top1_sum / entropy_count if entropy_count > 0 else None
        )
        self.last_cross_top3_mass = (
            top3_sum / entropy_count if entropy_count > 0 else None
        )
        self.last_psd_to_chem_mass = (
            psd_chem_sum / psd_count if psd_count > 0 else None
        )
        self.last_psd_to_psd_mass = (
            psd_psd_sum / psd_count if psd_count > 0 else None
        )

        cross_delta = cross_delta_btc.permute(0, 2, 1).contiguous()
        if self.null_output:
            return x
        return x + time_gate * time_delta + cross_gate * cross_delta


class PreGraphPerFeatureTemporalAttention(nn.Module):
    """Same-feature observed-key temporal retrieval before graph mixing."""

    def __init__(
        self,
        window_size,
        attn_dim=64,
        n_heads=4,
        gate_init=-1.0,
        dropout=0.1,
        chunk_size=256,
        record_weights=False,
        mode="dense",
        bucket_bounds=(4, 8, 16, 32, 64),
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.attn_dim = int(attn_dim)
        self.n_heads = int(n_heads)
        self.chunk_size = int(chunk_size)
        self.record_weights = bool(record_weights)
        self.mode = str(mode)
        self.bucket_bounds = tuple(sorted(int(b) for b in bucket_bounds))
        valid_modes = {"dense", "bucketed_missing_query_only"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Unsupported pregraph temporal attention mode: {self.mode}"
            )

        self.in_proj = nn.Linear(1, self.attn_dim)
        self.in_norm = nn.LayerNorm(self.attn_dim)
        self.attn = RotarySelfAttention(
            embed_dim=self.attn_dim,
            num_heads=self.n_heads,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(self.attn_dim, 1)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.last_gate = None
        self.last_missing_query_attn_entropy = None
        self.last_observed_query_attn_entropy = None

    def _missing_bucket_upper(self, missing_count):
        for upper in self.bucket_bounds:
            if missing_count <= upper:
                return upper
        return self.window_size

    def _forward_dense(self, x, tok, feature_obs=None):
        batch_size, channels, window = x.shape
        query_missing = None
        query_observed = None
        valid_entropy_rows = None
        key_padding_mask = None
        missing_mask = None
        if feature_obs is not None:
            key_padding_mask = feature_obs <= 0.0
            valid_entropy_rows = torch.ones(
                batch_size * channels,
                dtype=torch.bool,
                device=x.device,
            )
            if key_padding_mask.any():
                all_missing = key_padding_mask.all(dim=1, keepdim=True)
                valid_entropy_rows = ~all_missing.squeeze(1)
                key_padding_mask = key_padding_mask.masked_fill(
                    all_missing, False
                )
            query_missing = feature_obs <= 0.0
            query_observed = ~query_missing
            missing_mask = query_missing.reshape(
                batch_size, channels, window
            ).to(x.dtype)

        gate = torch.sigmoid(self.gate)
        self.last_gate = float(gate.detach().item())
        out_chunks = []
        missing_entropy_sum = 0.0
        missing_entropy_count = 0
        observed_entropy_sum = 0.0
        observed_entropy_count = 0

        total = batch_size * channels
        chunk_size = total if self.chunk_size <= 0 else self.chunk_size
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            tok_chunk = tok[start:end]
            key_mask_chunk = (
                key_padding_mask[start:end]
                if key_padding_mask is not None
                else None
            )
            attn_out, attn_weights = self.attn(
                tok_chunk,
                key_padding_mask=key_mask_chunk,
                need_weights=self.record_weights,
            )
            if self.record_weights and attn_weights is not None:
                attn_probs = attn_weights.mean(dim=1).clamp_min(1e-8)
                entropy = -(
                    attn_probs * attn_probs.log()
                ).sum(dim=-1)
                if query_missing is not None:
                    missing_chunk = query_missing[start:end]
                    observed_chunk = query_observed[start:end]
                    if valid_entropy_rows is not None:
                        valid_rows_chunk = valid_entropy_rows[
                            start:end
                        ].unsqueeze(1)
                        missing_chunk = missing_chunk & valid_rows_chunk
                        observed_chunk = observed_chunk & valid_rows_chunk
                    if missing_chunk.any():
                        missing_entropy_sum += entropy[
                            missing_chunk
                        ].sum().item()
                        missing_entropy_count += int(missing_chunk.sum().item())
                    if observed_chunk.any():
                        observed_entropy_sum += entropy[
                            observed_chunk
                        ].sum().item()
                        observed_entropy_count += int(observed_chunk.sum().item())
            out_chunks.append(attn_out)

        self.last_missing_query_attn_entropy = (
            missing_entropy_sum / missing_entropy_count
            if missing_entropy_count > 0
            else None
        )
        self.last_observed_query_attn_entropy = (
            observed_entropy_sum / observed_entropy_count
            if observed_entropy_count > 0
            else None
        )
        delta = (
            self.out_proj(torch.cat(out_chunks, dim=0))
            .squeeze(-1)
            .reshape(batch_size, channels, window)
        )
        if missing_mask is not None:
            delta = delta * missing_mask
        return x + gate * delta

    def _forward_bucketed_missing_query_only(self, x, tok, feature_obs):
        batch_size, channels, window = x.shape
        total = batch_size * channels
        gate = torch.sigmoid(self.gate)
        self.last_gate = float(gate.detach().item())

        feature_obs_cpu = feature_obs.detach().to(
            device="cpu", dtype=torch.bool
        )
        buckets = {}
        for seq_idx in range(total):
            obs_idx_cpu = torch.nonzero(
                feature_obs_cpu[seq_idx], as_tuple=False
            ).flatten()
            miss_idx_cpu = torch.nonzero(
                ~feature_obs_cpu[seq_idx], as_tuple=False
            ).flatten()
            if miss_idx_cpu.numel() == 0 or obs_idx_cpu.numel() == 0:
                continue
            bucket_key = self._missing_bucket_upper(
                int(miss_idx_cpu.numel())
            )
            buckets.setdefault(bucket_key, []).append(
                (seq_idx, miss_idx_cpu, obs_idx_cpu)
            )

        delta_flat = torch.zeros(
            (total, window), device=x.device, dtype=x.dtype
        )
        missing_entropy_sum = 0.0
        missing_entropy_count = 0
        self.last_observed_query_attn_entropy = None

        for _, entries in sorted(buckets.items(), key=lambda item: item[0]):
            n_bucket = len(entries)
            max_m = max(
                int(miss_idx.numel()) for _, miss_idx, _ in entries
            )
            max_o = max(
                int(obs_idx.numel()) for _, _, obs_idx in entries
            )
            q_batch = tok.new_zeros((n_bucket, max_m, self.attn_dim))
            k_batch = tok.new_zeros((n_bucket, max_o, self.attn_dim))
            v_batch = tok.new_zeros((n_bucket, max_o, self.attn_dim))
            q_pos = torch.zeros(
                (n_bucket, max_m), device=x.device, dtype=torch.long
            )
            kv_pos = torch.zeros(
                (n_bucket, max_o), device=x.device, dtype=torch.long
            )
            q_valid = torch.zeros(
                (n_bucket, max_m), device=x.device, dtype=torch.bool
            )
            kv_valid = torch.zeros(
                (n_bucket, max_o), device=x.device, dtype=torch.bool
            )

            for row_idx, (seq_idx, miss_idx_cpu, obs_idx_cpu) in enumerate(
                entries
            ):
                miss_idx = torch.as_tensor(
                    miss_idx_cpu, device=x.device, dtype=torch.long
                )
                obs_idx = torch.as_tensor(
                    obs_idx_cpu, device=x.device, dtype=torch.long
                )
                m_i = int(miss_idx.numel())
                o_i = int(obs_idx.numel())
                q_batch[row_idx, :m_i] = tok[seq_idx, miss_idx]
                q_pos[row_idx, :m_i] = miss_idx
                q_valid[row_idx, :m_i] = True
                kv_tokens = tok[seq_idx, obs_idx]
                k_batch[row_idx, :o_i] = kv_tokens
                v_batch[row_idx, :o_i] = kv_tokens
                kv_pos[row_idx, :o_i] = obs_idx
                kv_valid[row_idx, :o_i] = True

            attn_out, attn_weights = self.attn.forward_qkv(
                q_batch,
                k_batch,
                v_batch,
                q_pos=q_pos,
                kv_pos=kv_pos,
                key_padding_mask=~kv_valid,
                need_weights=self.record_weights,
            )
            attn_out = attn_out * q_valid.unsqueeze(-1).to(attn_out.dtype)

            if self.record_weights and attn_weights is not None:
                attn_probs = attn_weights.mean(dim=1).clamp_min(1e-8)
                entropy = -(
                    attn_probs * attn_probs.log()
                ).sum(dim=-1)
                if q_valid.any():
                    missing_entropy_sum += entropy[q_valid].sum().item()
                    missing_entropy_count += int(q_valid.sum().item())

            proj = self.out_proj(attn_out).squeeze(-1)
            for row_idx, (seq_idx, miss_idx_cpu, _) in enumerate(entries):
                miss_idx = torch.as_tensor(
                    miss_idx_cpu, device=x.device, dtype=torch.long
                )
                m_i = int(miss_idx.numel())
                delta_flat[seq_idx, miss_idx] = proj[row_idx, :m_i]

        self.last_missing_query_attn_entropy = (
            missing_entropy_sum / missing_entropy_count
            if missing_entropy_count > 0
            else None
        )
        delta = delta_flat.reshape(batch_size, channels, window)
        return x + gate * delta

    def forward(self, x, target_obs_mask=None):
        batch_size, channels, window = x.shape
        x_flat = x.reshape(batch_size * channels, window, 1)
        tok = self.in_norm(self.in_proj(x_flat))
        feature_obs = None
        if target_obs_mask is not None:
            feature_obs = (
                target_obs_mask.permute(0, 2, 1)
                .reshape(batch_size * channels, window)
                .float()
            )
        if (
            self.mode == "bucketed_missing_query_only"
            and feature_obs is not None
        ):
            return self._forward_bucketed_missing_query_only(
                x, tok, feature_obs
            )
        return self._forward_dense(x, tok, feature_obs=feature_obs)
