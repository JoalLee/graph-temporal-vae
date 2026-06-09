"""
Dynamic Graph UQ-VAE Model
==========================
Integrates Dynamic Graph Learning into UQ-VAE for spatial-temporal modeling.
Learns physical/chemical relationships between species using feature-wise self-attention.

Architecture:
    Input → InputGraphLayer → TCN → Attention Pool → μ,σ² → Decoder
"""
import math
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import parametrizations as param


class TimeHybridEncoder(nn.Module):
    """Hybrid time encoder: learnable embeddings + cyclical sin/cos features.

    Expects per-timestep cyclical time channels in this order:
    [hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos].
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
        angle = torch.atan2(sin_val, cos_val)  # [-pi, pi]
        angle = torch.remainder(angle + 2 * np.pi, 2 * np.pi)  # [0, 2pi)
        idx_float = (angle / (2 * np.pi)) * period
        idx = torch.round(idx_float).long() % period
        return idx

    def forward(self, time_cyc):
        """Encode cyclical channels into hybrid time features.

        Args:
            time_cyc: [B, W, 6]
        Returns:
            [B, W, out_dim]
        """
        hour_idx = self._cyc_to_index(time_cyc[..., 0], time_cyc[..., 1], 24)
        dow_idx = self._cyc_to_index(time_cyc[..., 2], time_cyc[..., 3], 7)
        month_idx = self._cyc_to_index(time_cyc[..., 4], time_cyc[..., 5], 12)

        hour_emb = self.hour_embed(hour_idx)
        dow_emb = self.dow_embed(dow_idx)
        month_emb = self.month_embed(month_idx)

        fused_in = torch.cat([time_cyc, hour_emb, dow_emb, month_emb], dim=-1)
        return self.fuse(fused_in)


class RotarySelfAttention(nn.Module):
    """Batch-first self-attention with RoPE applied to Q/K."""

    def __init__(self, embed_dim, num_heads, dropout=0.1, rope_base=10000.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim, got embed_dim={embed_dim}, num_heads={num_heads}"
            )
        self.rope_base = float(rope_base)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _to_heads(self, x):
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _from_heads(self, x):
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)

    def _rope_cache(self, seq_len, device, dtype):
        inv_freq = 1.0 / (
            self.rope_base ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim
            )
        )
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        cos = freqs.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def _rope_cache_from_positions(self, positions, device, dtype):
        """
        Build RoPE cos/sin cache for arbitrary integer positions.

        Args:
            positions: [B, L] integer positions
        Returns:
            cos, sin: [B, 1, L, head_dim/2]
        """
        inv_freq = 1.0 / (
            self.rope_base ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim
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

    def _build_attn_bias(self, bsz, q_len, k_len, device, dtype,
                         key_padding_mask=None, attn_mask=None):
        attn_bias = None

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                if attn_mask.shape[0] == bsz * self.num_heads:
                    attn_bias = attn_mask.view(bsz, self.num_heads, q_len, k_len)
                elif attn_mask.shape[0] == bsz:
                    attn_bias = attn_mask.unsqueeze(1)
                else:
                    raise ValueError(
                        f"Unexpected attn_mask shape {tuple(attn_mask.shape)} for "
                        f"batch={bsz}, heads={self.num_heads}, q_len={q_len}, k_len={k_len}"
                    )
            else:
                attn_bias = attn_mask
            attn_bias = attn_bias.to(device=device, dtype=dtype)

        if key_padding_mask is not None:
            key_bias = torch.zeros((bsz, 1, 1, k_len), device=device, dtype=dtype)
            key_bias = key_bias.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(dtype).min,
            )
            attn_bias = key_bias if attn_bias is None else (attn_bias + key_bias)

        return attn_bias

    def forward_qkv(self, q_in, k_in, v_in, q_pos=None, kv_pos=None,
                    key_padding_mask=None, attn_mask=None, need_weights=False):
        """
        Args:
            q_in: [B, Lq, D]
            k_in: [B, Lk, D]
            v_in: [B, Lk, D]
            q_pos: [B, Lq] integer positions or None -> arange(Lq)
            kv_pos: [B, Lk] integer positions or None -> arange(Lk)
            key_padding_mask: [B, Lk] bool, True = masked key/value
            attn_mask: additive float mask [B*h, Lq, Lk] or [B, h, Lq, Lk]
        """
        bsz, q_len, _ = q_in.shape
        _, k_len, _ = k_in.shape

        q = self._to_heads(self.q_proj(q_in))
        k = self._to_heads(self.k_proj(k_in))
        v = self._to_heads(self.v_proj(v_in))

        if q_pos is None:
            q_pos = torch.arange(q_len, device=q_in.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        if kv_pos is None:
            kv_pos = torch.arange(k_len, device=k_in.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        cos_q, sin_q = self._rope_cache_from_positions(q_pos, q_in.device, q.dtype)
        cos_k, sin_k = self._rope_cache_from_positions(kv_pos, k_in.device, k.dtype)
        q = self._apply_rope(q, cos_q, sin_q)
        k = self._apply_rope(k, cos_k, sin_k)

        attn_bias = self._build_attn_bias(
            bsz=bsz,
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
            out = self.out_proj(self._from_heads(out))
            return out, None

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            logits = logits + attn_bias.to(dtype=logits.dtype)

        attn_probs = F.softmax(logits, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = self.out_proj(self._from_heads(out))
        return out, (attn_probs if need_weights else None)

    def forward(self, x, key_padding_mask=None, attn_mask=None, need_weights=False):
        """
        Args:
            x: [B, L, D]
            key_padding_mask: [B, L] bool, True = masked key/value
            attn_mask: additive float mask [B*h, L, L] or [B, h, L, L]
        """
        return self.forward_qkv(
            x, x, x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )


# ==============================================================================
# Normalizing Flows (RealNVP)
# ==============================================================================
class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling Layer for RealNVP.
    Splits input into two parts based on mask. Part 1 remains unchanged.
    Part 2 undergoes an affine transformation parameterized by a neural network on Part 1.
    """
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
            nn.Linear(hidden_dim, dim * 2)
        )
        # Initialize the last layer to output 0 so the initial transformation is identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # x: [B, D]
        x_masked = x * self.mask
        out = self.net(x_masked)
        s, t = out.chunk(2, dim=-1)
        
        # Scale parameters bounds: use tanh to prevent numerical instability,
        # but allow a max scale of exp(2.0) ~ 7.3x instead of just exp(1.0) ~ 2.7x
        # to give the flow more ability to represent extreme heavy tails.
        self.scale_limit = 1.0
        s = self.scale_limit * torch.tanh(s) * (1.0 - self.mask) # only apply transformation to unmasked parts
        t = t * (1.0 - self.mask)
        
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        # Explicit mask: only unmasked dims contribute to log |J|
        log_det_J = (s * (1.0 - self.mask)).sum(dim=-1)
        return z, log_det_J

class ReverseLayer(nn.Module):
    """
    Dimension-reversal permutation layer for Normalizing Flows.
    No learnable parameters. log|det J| = 0 (permutation matrix has |det|=1).
    Inserted between AffineCouplingLayers so every coupling layer sees a
    different partition of the latent dimensions, improving mixing.
    """
    def forward(self, x):
        return x.flip(-1), torch.zeros(x.shape[0], device=x.device)


class RealNVP(nn.Module):
    """
    Stack of Affine Coupling Layers with Reverse permutation between each layer.
    Pattern: CouplingLayer → Reverse → CouplingLayer → Reverse → ...
    The Reverse ensures all dimension pairs interact across layers
    without any additional ELBO computation cost (log_det contribution = 0).
    """
    def __init__(self, dim, n_layers=4, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList()

        # Create alternating masks
        mask1 = torch.zeros(dim)
        mask1[::2] = 1.0  # Even indices are 1
        mask2 = 1.0 - mask1  # Odd indices are 1

        masks = [mask1, mask2] * (n_layers // 2)
        if n_layers % 2 != 0:
            masks.append(mask1)

        for i, mask in enumerate(masks):
            self.layers.append(AffineCouplingLayer(dim, mask, hidden_dim))
            # Insert Reverse between every coupling layer (not after the last one)
            if i < len(masks) - 1:
                self.layers.append(ReverseLayer())

    def forward(self, x):
        log_det_J_sum = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det_J = layer(x)
            log_det_J_sum += log_det_J
        return x, log_det_J_sum


# ==============================================================================
# Vanilla VAE (MLP Baseline for Ablation)
# ==============================================================================
class VanillaVAE(nn.Module):
    """
    Simple MLP-based VAE without temporal or graph structure.
    Flattens window input and processes as a vector.
    Used as the absolute baseline (B1) in ablation study.
    """
    
    def __init__(self, input_dim, window_size, latent_dim, hidden_dims,
                 target_dim=None, chem_dim=31, psd_dim=230,
                 var_min=1e-4, var_max=10.0, heteroscedastic=True,
                 use_realnvp=False, realnvp_layers=4):
        super().__init__()
        
        self.use_realnvp = use_realnvp
        if use_realnvp:
            self.flow = RealNVP(latent_dim, n_layers=realnvp_layers, hidden_dim=max(64, latent_dim))
        else:
            self.flow = None
        
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.target_dim = target_dim if target_dim is not None else input_dim
        self.chem_dim = chem_dim
        self.psd_dim = psd_dim
        self.var_min = var_min
        self.var_max = var_max
        self.heteroscedastic = heteroscedastic
        
        # Vectorized mask embedding: single lookup gives per-feature offset
        # Each feature gets its own learned (missing, observed) pair
        self.mask_embed = nn.Embedding(2, target_dim)
        nn.init.normal_(self.mask_embed.weight, mean=0.0, std=0.01)
        
        flat_dim = input_dim * window_size
        
        # Encoder MLP
        encoder_layers = []
        in_dim = flat_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder MLP
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output heads
        out_flat = self.target_dim * window_size
        self.recon_head = nn.Linear(hidden_dims[0], out_flat)
        
        if heteroscedastic:
            self.logvar_chem_head = nn.Linear(hidden_dims[0], chem_dim * window_size)
            self.logvar_psd_head = nn.Linear(hidden_dims[0], psd_dim * window_size)
        else:
            self.logvar_head = nn.Linear(hidden_dims[0], out_flat)
        
        # Placeholders for compatibility
        self.last_graph_attention = None
        self.last_graph_attention_heads = None
        self.last_cross_modal_attention = None
        self.last_cross_modal_attention_heads = None
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, cond=None, mask=None, aux_mask=None, sample_latent=True):
        """
        Args:
            x: [B, W, target_dim] target features
            cond: [B, W, cond_dim] conditioning features (aux + hour + time)
            mask: [B, W, target_dim] observation mask (1=observed)
        Returns:
            recon_mu, recon_logvar, mu, logvar, None (for graph attention compat)
        """
        B, W, target_dim = x.shape
        
        # Apply vectorized mask embedding
        # mask is [B, W, D] with 0/1 values — but Embedding expects same index across last dim
        # We use a trick: lookup with a scalar index (0 or 1) and get a D-dim vector
        # For per-feature behavior, we need per-position lookups
        mask_int = mask.long()  # [B, W, D]
        # Gather per-feature embeddings: for each (b, w, d), lookup mask_int[b,w,d] and take dim d
        embed_0 = self.mask_embed(torch.zeros(1, dtype=torch.long, device=x.device))  # [1, D]
        embed_1 = self.mask_embed(torch.ones(1, dtype=torch.long, device=x.device))   # [1, D]
        # Broadcast: mask_int [B,W,D] selects between embed_0 and embed_1 per feature
        embed_offset = embed_0 + mask.float() * (embed_1 - embed_0)  # [B, W, D]
        x_with_embed = x + embed_offset
        
        # Transpose to [B, C, W] format
        x_with_embed = x_with_embed.transpose(1, 2)  # [B, target_dim, W]
        
        # Concatenate: target (with mask embedding) + cond (NO separate mask channel)
        if cond is not None:
            cond = cond.transpose(1, 2)  # [B, cond_dim, W]
            x_with_embed = torch.cat([x_with_embed, cond], dim=1)
        
        # Now x is [B, input_dim, W] where input_dim = target + cond (mask embedded in target)
        B, C, W = x_with_embed.shape
        
        # Flatten
        x_flat = x_with_embed.view(B, -1)  # [B, C*W]
        
        # Sanity check: verify dimension matches model initialization
        expected_flat = self.input_dim * self.window_size
        if x_flat.shape[1] != expected_flat:
            raise RuntimeError(
                f"VanillaVAE forward() dimension mismatch!\n"
                f"  Model initialized with input_dim={self.input_dim} × window={self.window_size} = {expected_flat}\n"
                f"  Forward() received: C={C} channels × W={W} window = {x_flat.shape[1]}\n"
                f"  target_dim={target_dim}, cond_dim={cond.shape[1] if cond is not None else 0}\n"
                f"  Hint: Ensure total_aux_dim in train_ablation.py matches actual cond dimension"
            )
        
        # Encode
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Sample latent during normal VAE inference/training.  Inference
        # ablations can set sample_latent=False to isolate likelihood-only
        # predictive spread without changing the training path.
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        
        # Flow
        self.last_log_det_J = None
        self.last_z0 = z
        self.last_zK = z
        if self.use_realnvp:
            z, self.last_log_det_J = self.flow(z)
            self.last_zK = z
            
        # Decode
        h_dec = self.decoder(z)
        
        # Reconstruct
        recon_flat = self.recon_head(h_dec)  # [B, target_dim * W]
        recon_mu = recon_flat.view(B, self.target_dim, W)
        
        # Variance
        if self.heteroscedastic:
            logvar_chem = self.logvar_chem_head(h_dec).view(B, self.chem_dim, W)
            logvar_psd = self.logvar_psd_head(h_dec).view(B, self.psd_dim, W)
            recon_logvar = torch.cat([logvar_chem, logvar_psd], dim=1)
        else:
            recon_logvar = self.logvar_head(h_dec).view(B, self.target_dim, W)
        
        # Clamp variance
        recon_logvar = torch.clamp(recon_logvar, 
                                   np.log(self.var_min), 
                                   np.log(self.var_max))
        
        # Transpose output back to [B, W, C] format for Trainer compatibility
        recon_mu = recon_mu.transpose(1, 2)  # [B, W, target_dim]
        recon_logvar = recon_logvar.transpose(1, 2)  # [B, W, target_dim]
        
        # Return 5 values (None for graph attention) for API compatibility
        return recon_mu, recon_logvar, mu, logvar, None
    
    def _enable_mc_dropout(self):
        """Enable MC Dropout by setting only dropout modules to train mode."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def compute_uncertainty(self, x, cond, mask, n_samples=50, dist_type='gaussian', history=None,
                            return_extra_quantiles=False, return_samples=False,
                            enable_mc_dropout=True, sample_latent=True,
                            sample_likelihood=True):
        """
        Compute epistemic and aleatoric uncertainty via MC Dropout + z-sampling for VanillaVAE.

        With `return_samples=True`, the raw generative samples and per-MC means
        tensors are appended to the result tuple (shapes [N, B, W, D]) for use
        by the sample-level overlap-add aggregator.
        """
        self.eval()
        if enable_mc_dropout:
            self._enable_mc_dropout()
        
        all_means = []
        all_logvars = []
        all_samples = []
        
        for i in range(n_samples):
            with torch.no_grad():
                outputs = self.forward(x, cond, mask, sample_latent=sample_latent)
                recon_mean, recon_logvar = outputs[0], outputs[1]
                all_means.append(recon_mean)
                
                if recon_logvar is not None:
                    all_logvars.append(recon_logvar)

                    if not sample_likelihood:
                        y_sample = recon_mean
                    elif dist_type == 'student_t':
                        # logvar = log(variance); sigma^2 = variance * (df-2)/df
                        df = 3.0
                        variance = torch.exp(recon_logvar)
                        sigma = torch.sqrt((variance * (df - 2.0) / df).clamp(min=1e-10))
                        chi2 = torch.distributions.Chi2(df).sample(recon_mean.shape).to(x.device)
                        t_eps = torch.randn_like(recon_mean) * torch.sqrt(df / chi2)
                        y_sample = recon_mean + sigma * t_eps
                    else:
                        std = torch.exp(0.5 * recon_logvar)
                        y_sample = recon_mean + std * torch.randn_like(recon_mean)
                    all_samples.append(y_sample)
                else:
                    all_samples.append(recon_mean)
        
        self.eval()
        
        means = torch.stack(all_means, dim=0)
        samples = torch.stack(all_samples, dim=0)
        
        # Epistemic: variance and quantiles across predicted means
        epistemic_var = means.var(dim=0)
        pred_mean = means.mean(dim=0)
        epi_q05 = torch.quantile(means, 0.05, dim=0)
        epi_q95 = torch.quantile(means, 0.95, dim=0)
        epi_q025 = torch.quantile(means, 0.025, dim=0)
        epi_q975 = torch.quantile(means, 0.975, dim=0)
        
        # Total Prediction Intervals (Generative: incorporates aleatoric + epistemic)
        pred_q05 = torch.quantile(samples, 0.05, dim=0)
        pred_q95 = torch.quantile(samples, 0.95, dim=0)
        pred_q025 = torch.quantile(samples, 0.025, dim=0)
        pred_q975 = torch.quantile(samples, 0.975, dim=0)
        
        aleatoric_var = None
        if all_logvars:
            logvars = torch.stack(all_logvars, dim=0)
            # logvar = log(variance) for both Gaussian and Student-t
            aleatoric_var = torch.exp(logvars).mean(dim=0)
        
        total_var = epistemic_var + (aleatoric_var if aleatoric_var is not None else 0)
        
        result = (pred_mean, epistemic_var, aleatoric_var, total_var, None, pred_q05, pred_q95, epi_q05, epi_q95)
        if return_extra_quantiles:
            result = result + (pred_q025, pred_q975, epi_q025, epi_q975)
        if return_samples:
            result = result + (samples, means)
        return result

    def get_learned_graph(self):
        return None
    
    def get_learned_graph_heads(self):
        return None
    
    def get_cross_modal_graph(self):
        return None
    
    def get_cross_modal_graph_heads(self):
        return None



# ==============================================================================
# Depthwise Temporal Convolutional Network (Multi-Scale)
# ==============================================================================
class DepthwiseTCN(nn.Module):
    """
    Decoupled temporal feature extractor that operates on each channel independently.
    Uses groups=channels to prevent mixing channel semantics before graph attention.
    """
    def __init__(self, channels, num_layers=3, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, padding=padding,
                dilation=dilation, groups=channels, bias=True
            ))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# Cross-Modal Graph Layer (Target-Aux Attention)
# ==============================================================================
class WindowTokenFFN(nn.Module):
    """
    Lightweight token-wise FFN in window space.

    The graph blocks in this project keep token states in R^W rather than a
    persistent d_model space, so the FFN is also applied in R^W:
        W -> (mult * W) -> W
    with residual + LayerNorm.
    """
    def __init__(self, width, mult=4, dropout=0.1):
        super().__init__()
        width = int(width)
        hidden = int(mult) * width
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TokenGraphFFN(nn.Module):
    """Standard Transformer-style FFN operating in persistent d_model space."""

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
    """Pre-LN self-attention block over feature tokens in d_model space."""

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
            attn_avg = attn_weights.detach().mean(dim=1)  # [B, C, C]
            self.last_attention_weights = attn_avg.mean(dim=0)  # [C, C]
            self.last_attention_weights_heads = attn_weights.detach().mean(dim=0)  # [h, C, C]
            self.last_attention_weights_heads_batch = attn_weights.detach()  # [B, h, C, C]
        else:
            self.last_attention_weights = None
            self.last_attention_weights_heads = None
            self.last_attention_weights_heads_batch = None

        return x, attn_avg


class TokenGraphCrossBlock(nn.Module):
    """Pre-LN cross-attention block: target feature tokens attend to aux tokens."""

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
            attn_avg = attn_weights.detach().mean(dim=1)  # [B, C_t, C_a]
            self.last_attention_weights = attn_avg.mean(dim=0)  # [C_t, C_a]
            self.last_attention_weights_heads = attn_weights.detach().mean(dim=0)  # [h, C_t, C_a]
            self.last_attention_weights_heads_batch = attn_weights.detach()  # [B, h, C_t, C_a]
        else:
            self.last_attention_weights = None
            self.last_attention_weights_heads = None
            self.last_attention_weights_heads_batch = None

        return x, attn_avg


class LocalChunkGraphBranch(nn.Module):
    """Local chunk-wise feature graph branch over short temporal slices."""

    def __init__(self, n_features, window_size, chunk_size=6, d_model=128,
                 n_heads=4, dropout=0.1, gate_init=-2.0, ffn_mult=4,
                 use_mask_embed=False, out_proj_init_std=0.0):
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
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.out_proj_init_std)
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
        """
        Args:
            x: [B, C, W]
            chunk_obs_mask: optional [B, C, W] float/bool, 1=observed, 0=missing
        Returns:
            x_local: [B, C, W]
        """
        B, C, W = x.shape
        if C != self.n_features:
            raise ValueError(
                f"LocalChunkGraphBranch expected {self.n_features} features, got {C}"
            )

        pad_len = (self.chunk_size - (W % self.chunk_size)) % self.chunk_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, pad_len))
        else:
            x_padded = x

        W_pad = x_padded.shape[-1]
        n_chunks = W_pad // self.chunk_size

        # [B, C, W_pad] -> [B, T, C, chunk]
        x_chunks = x_padded.view(B, C, n_chunks, self.chunk_size).permute(0, 2, 1, 3).contiguous()
        tok = self.token_embed(x_chunks)
        self.last_obs_ratio_mean = None
        if self.use_mask_embed and chunk_obs_mask is not None:
            if pad_len > 0:
                mask_padded = F.pad(chunk_obs_mask, (0, pad_len))
            else:
                mask_padded = chunk_obs_mask
            m_chunks = (
                mask_padded.view(B, C, n_chunks, self.chunk_size)
                .permute(0, 2, 1, 3)
                .contiguous()
                .to(dtype=x_chunks.dtype)
            )
            obs_ratio = m_chunks.mean(dim=-1, keepdim=True)
            tok = tok + self.mask_embed(m_chunks) + self.ratio_embed(obs_ratio)
            self.last_obs_ratio_mean = float(obs_ratio.detach().mean().item())
        tok = self.token_norm(tok)
        tok = tok.view(B * n_chunks, C, self.d_model)

        tok_out, _ = self.block(tok, need_weights=False)
        tok_out = tok_out.view(B, n_chunks, C, self.d_model)

        delta_chunks = self.out_proj(tok_out)  # [B, T, C, chunk]
        delta = (
            delta_chunks.permute(0, 2, 1, 3)
            .contiguous()
            .view(B, C, W_pad)
        )
        if pad_len > 0:
            delta = delta[:, :, :W]

        gate = torch.sigmoid(self.out_gate)
        self.last_gate = float(gate.detach().mean().item())
        self.last_out_proj_weight_norm = float(self.out_proj.weight.detach().norm().item())
        return x + gate * delta


class ExternalHistoryContext(nn.Module):
    """Coarse leakage-safe history memory for 24d external chunk context."""

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
                f"external_history_dim={self.hidden_dim} must be even because "
                "the bidirectional history GRU returns 2 * (hidden_dim // 2) channels"
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

        self.last_gate = None
        self.last_attn_entropy = None
        self.last_valid_fraction = None
        self.last_null_fraction = None
        self.last_top1_mass = None
        self.last_top3_mass = None
        self.last_attended_time_dist = None
        self.last_attended_support = None
        self.last_attended_null_fraction = None

    def _current_queries(self, cond, obs_mask):
        B, W, _ = cond.shape
        L = self.history_chunk_size
        pad_len = (L - (W % L)) % L
        if pad_len > 0:
            cond_pad = F.pad(cond.transpose(1, 2), (0, pad_len)).transpose(1, 2)
            obs_pad = F.pad(obs_mask.transpose(1, 2), (0, pad_len)).transpose(1, 2)
        else:
            cond_pad = cond
            obs_pad = obs_mask

        W_pad = cond_pad.shape[1]
        n_chunks = W_pad // L
        cond_chunks = cond_pad.reshape(B, n_chunks, L, self.cond_dim).mean(dim=2)
        support_chunks = obs_pad.reshape(B, n_chunks, L, obs_mask.shape[-1]).float().mean(dim=(2, 3), keepdim=False)
        q_in = torch.cat([cond_chunks, support_chunks.unsqueeze(-1)], dim=-1)
        return self.query_proj(q_in)

    def _manual_cross_attention(self, query, memory, support, time_dist, null_flag, valid):
        logits = torch.matmul(query, memory.transpose(1, 2)) / math.sqrt(float(self.hidden_dim))
        if self.use_retrieval_bias:
            support_quality = support[..., 0].clamp(0.0, 1.0)
            support_quality = support_quality.masked_fill(valid <= 0.0, 0.0)
            bias = self.support_bias * support_quality.unsqueeze(1)
            bias = bias - self.time_decay * time_dist.clamp_min(0.0).unsqueeze(1)
            bias = bias - self.null_penalty * null_flag.to(query.dtype).unsqueeze(1)
            logits = logits + bias
        logits = logits.masked_fill(valid.unsqueeze(1) <= 0.0, -1e4)
        attn_weights = torch.softmax(logits, dim=-1)
        attn_out = torch.matmul(attn_weights, memory)
        return attn_out, attn_weights

    def forward(self, history, cond, obs_mask):
        """
        Args:
            history: dict with history_target/history_obs_mask/history_cond/
                history_chunk_valid/history_time_dist/history_support.
                history_cond is injected by ImputationVAE_Graph.forward from
                history_aux + history_hour after condition encoding.
            cond: [B, W, cond_dim]
            obs_mask: [B, W, target_dim], 1=observed
        Returns:
            [B, context_dim, context_steps] gated external context.
        """
        target = history['history_target']
        h_obs = history['history_obs_mask'].to(dtype=target.dtype)
        h_cond = history['history_cond'].to(dtype=target.dtype)
        support = history['history_support'].to(dtype=target.dtype)
        if support.shape[-1] != self.history_support_dim:
            raise ValueError(
                f"history_support has {support.shape[-1]} fields, expected {self.history_support_dim}"
            )
        time_dist = history['history_time_dist'].to(dtype=target.dtype)
        valid = history['history_chunk_valid'].to(dtype=target.dtype)

        B, K, L, D = target.shape
        step_in = torch.cat([target, h_obs, h_cond], dim=-1)
        step_tok = self.timestep_proj(step_in.reshape(B * K * L, -1)).view(B, K, L, self.hidden_dim)
        chunk_tok = step_tok.mean(dim=2)

        time_norm = (time_dist / max(float(self.history_num_chunks), 1.0)).unsqueeze(-1)
        meta_tok = self.support_proj(torch.cat([support, time_norm], dim=-1))
        null_flag = (support[..., 5] > 0.5) | (valid <= 0.0)
        null_tok = self.null_token.view(1, 1, -1).expand(B, K, -1)
        chunk_tok = torch.where(null_flag.unsqueeze(-1), null_tok, chunk_tok)
        chunk_tok = chunk_tok + meta_tok

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
            ctx = F.interpolate(ctx, size=self.context_steps, mode='linear', align_corners=False)

        gate = torch.sigmoid(self.out_gate)
        self.last_gate = float(gate.detach().item())
        with torch.no_grad():
            p = attn_weights.detach().clamp_min(1e-8)
            self.last_attn_entropy = float((-(p * p.log()).sum(dim=-1)).mean().item())
            self.last_valid_fraction = float((valid > 0).float().mean().item())
            self.last_null_fraction = float(null_flag.float().mean().item())
            topk = torch.topk(p, k=min(3, p.shape[-1]), dim=-1).values
            self.last_top1_mass = float(topk[..., 0].mean().item())
            self.last_top3_mass = float(topk.sum(dim=-1).mean().item())
            self.last_attended_time_dist = float((p * time_dist.unsqueeze(1)).sum(dim=-1).mean().item())
            self.last_attended_support = float((p * support[..., 0].unsqueeze(1)).sum(dim=-1).mean().item())
            self.last_attended_null_fraction = float((p * null_flag.float().unsqueeze(1)).sum(dim=-1).mean().item())
        return gate * ctx


class CrossModalGraphLayer(nn.Module):
    """
    Cross-modal attention between target features and auxiliary features.
    Query: target features, Key/Value: auxiliary features.
    Learns which auxiliary factors influence which target species.
    """
    
    def __init__(self, target_dim, aux_dim, window_size, n_heads=4, head_dim=64, dropout=0.1, 
                 use_temporal_cnn=True, disable_aux_bias=False,
                 use_ffn=False, ffn_mult=4):
        super().__init__()
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = n_heads * head_dim

        # Soft gate toggle: when True, replaces hard 10% threshold with smooth
        # sigmoid gate to allow gradient flow at any obs_rate (curriculum masking).
        self.use_soft_gate = False
        
        # Learnable Temperature
        self.tau_param = nn.Parameter(torch.zeros(1))
        
        # Learnable Prior Bias Coefficient (beta)
        self.prior_beta = nn.Parameter(torch.ones(1) * 0.5)

        # [NEW] Low-Rank Bias to fix the softmax-constant issue in cross-modal attention
        self.aux_rank = 4
        if aux_dim > 0 and not disable_aux_bias:
            # Query-side (target) and Key-side (aux) embeddings for pair-specific bias
            self.node_embed_t = nn.Parameter(torch.randn(target_dim, self.aux_rank) * 0.02)
            self.node_embed_a = nn.Parameter(torch.randn(aux_dim, self.aux_rank) * 0.02)
            self.aux_bias_net = nn.Sequential(
                nn.Linear(aux_dim * 2, 32),   # context: mean + std
                nn.SiLU(),
                nn.Linear(32, 2 * n_heads * self.aux_rank) # output scales for both sides
            )
            nn.init.zeros_(self.aux_bias_net[-1].weight)
            nn.init.zeros_(self.aux_bias_net[-1].bias)
        
        # Temporal Feature Extracts are now handled by Multi-Scale Depthwise TCNs 
        # prior to entering this CrossModalGraphLayer.
        
        # Project to d_model space, then split into heads
        self.q_proj = nn.Linear(window_size, self.d_model)  # Query from target
        self.k_proj = nn.Linear(window_size, self.d_model)  # Key from aux
        self.v_proj = nn.Linear(window_size, self.d_model)  # Value from aux
        self.out_proj = nn.Linear(self.d_model, window_size)
        
        self.layer_norm = nn.LayerNorm(window_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = WindowTokenFFN(window_size, mult=ffn_mult, dropout=dropout) if use_ffn else None
        
        self.last_attention_weights = None
        self.last_attention_weights_heads = None
        self.last_attention_weights_heads_batch = None
    
    def forward(self, x_target, x_aux, target_mask=None):
        """
        Args:
            x_target: [B, C_target, W] - target features
            x_aux: [B, C_aux, W] - auxiliary features (always observed)
            target_mask: [B, W, C_target] - observation mask for target
        
        Returns:
            out: [B, C_target, W] - enhanced target features
            attn_avg: [B, C_target, C_aux] - average cross-attention
        """
        B, C_t, W = x_target.shape
        C_a = x_aux.shape[1]
        
        if target_mask is None:
            target_mask_t = torch.ones(B, C_t, W, device=x_target.device)
        else:
            target_mask_t = target_mask.permute(0, 2, 1).float()
        
        # Normalization and depthwise TCN are now handled by GraphEncoder
        # prior to entering the graph layers. x_target and x_aux are fully processed.
        x_target_normed = x_target
        x_aux_normed = x_aux
            
        # Project to d_model space: [B, C, W] -> [B, C, d_model]
        Q = self.q_proj(x_target_normed)  # [B, C_t, d_model]
        K = self.k_proj(x_aux_normed)     # [B, C_a, d_model]
        V = self.v_proj(x_aux_normed)     # [B, C_a, d_model]
        
        # Multi-head reshape: [B, C, d_model] -> [B, C, n_heads, head_dim] -> [B, n_heads, C, head_dim]
        Q = Q.view(B, C_t, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, C_t, head_dim]
        K = K.view(B, C_a, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, C_a, head_dim]
        V = V.view(B, C_a, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, C_a, head_dim]
        
        # --- Observation-rate prior ---
        # Since aux is always fully observed, the old pairwise_ratio[i,j] = obs_rate[i] for ALL j
        # — adding the same constant to every key logit in a row, which leaves softmax unchanged.
        # We replace it with target observation rate as a per-query scalar prior:
        #   obs_rate[i] = fraction of W that target feature i is observed
        # This correctly gates/penalizes target queries with sparse observations.
        if target_mask is None:
            obs_rate = torch.ones(B, C_t, device=x_target.device)
        else:
            target_mask_t_sum = target_mask_t.sum(dim=2)  # [B, C_t]
            obs_rate = target_mask_t_sum / W              # [B, C_t] ∈ [0, 1]

        # Expand to [B, h, C_t, C_a] for broadcast over all aux keys
        obs_rate_expanded = obs_rate.unsqueeze(1).unsqueeze(-1).expand(
            -1, self.n_heads, -1, C_a
        )  # [B, h, C_t, C_a]

        # 3. Cross-attention logits
        scale_base = np.sqrt(self.head_dim)
        temperature = 0.5 + 0.5 * torch.sigmoid(self.tau_param)
        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / (scale_base * temperature)

        # 4. [NEW] Aux Conditioning: Low-Rank Pair-Specific Bias (survives softmax)
        if hasattr(self, 'aux_bias_net') and x_aux is not None:
            # Mean = background regime; std = event intensity
            aux_mean = x_aux.mean(dim=-1)                                # [B, aux_dim]
            aux_std  = x_aux.std(dim=-1).clamp(min=1e-6)                 # [B, aux_dim]
            aux_ctx  = torch.cat([aux_mean, aux_std], dim=-1)             # [B, 2*aux_dim]

            uv = self.aux_bias_net(aux_ctx)                               # [B, 2*h*r]
            u_scale, v_scale = uv.chunk(2, dim=-1)
            r = self.aux_rank
            
            u_scale = u_scale.view(B, self.n_heads, 1, r)                 # [B, h, 1, r]
            v_scale = v_scale.view(B, self.n_heads, 1, r)                 # [B, h, 1, r]

            # Modulate static node embeddings with weather-dependent importance directions
            node_t_emb = self.node_embed_t.unsqueeze(0).unsqueeze(0)      # [1, 1, C_t, r]
            node_a_emb = self.node_embed_a.unsqueeze(0).unsqueeze(0)      # [1, 1, C_a, r]
            bias_U = node_t_emb * u_scale   # [B, h, C_t, r]
            bias_V = node_a_emb * v_scale   # [B, h, C_a, r]

            # Pair-specific logit bias [B, h, C_t, C_a]
            attn_logits = attn_logits + torch.matmul(bias_U, bias_V.transpose(-1, -2))

        # 5. Gating: suppress queries from sparsely-observed target features.
        # NOTE: obs_rate_expanded is row-constant [B, h, C_t, 1] relative to keys.
        # We must use a multiplicative gate AFTER softmax (or hard masking before)
        # to ensure the total information uptake is reduced for sparse queries.
        if self.use_soft_gate:
            # Soft sigmoid gate: allows gradient flow even at obs_rate=0
            soft_gate = torch.sigmoid(20.0 * (obs_rate_expanded - 0.05))
            attn_weights = F.softmax(attn_logits, dim=-1) * soft_gate
        else:
            # Original hard threshold (backward compatible)
            threshold = 0.1
            attn_logits = attn_logits.masked_fill(obs_rate_expanded < threshold, -1e4)
            attn_weights = F.softmax(attn_logits, dim=-1)
            # Final verification: zero out fully-masked rows
            valid_row_mask = (obs_rate_expanded >= threshold).any(dim=-1, keepdim=True)
            attn_weights = attn_weights * valid_row_mask.float()
        
        attn = self.dropout(attn_weights)
        
        # Store for interpretability
        attn_avg = attn_weights.detach().mean(dim=1)  # [B, C_t, C_a]
        self.last_attention_weights = attn_avg.mean(dim=0)  # [C_t, C_a]
        self.last_attention_weights_heads = attn_weights.detach().mean(dim=0)  # [h, C_t, C_a]
        self.last_attention_weights_heads_batch = attn_weights.detach()  # [B, h, C_t, C_a]
        
        # Apply attention
        out = torch.matmul(attn, V)  # [B, h, C_t, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, C_t, self.d_model)  # [B, C_t, d_model]
        out = self.out_proj(out)  # [B, C_t, W]
        
        # Residual + LayerNorm
        out = self.layer_norm(out + x_target)
        if self.ffn is not None:
            out = self.ffn(out)
        
        return out, attn_avg


# ==============================================================================
# Input Graph Layer — Relation-only Heterogeneous Graph (RoHG)
# Shared Q/K/V projections; heterogeneity lives entirely in rel_log_scale [4,h]
# and type-specific output projections.  Full HGT ablation → val 0.2915 (worse
# than homogeneous 0.2800); this variant tests whether W_R alone is the signal.
# ==============================================================================
class InputGraphLayer(nn.Module):
    """
    Relation-only Heterogeneous Graph (RoHG) at input level.

    Node types  : Chem (first n_chem features), PSD (remaining n_psd features)
    Edge types  : CC  Chem→Chem   (photochemistry)
                  PP  PSD→PSD     (aerosol microphysics)
                  CP  Chem→PSD    (gas-to-particle conversion)
                  PC  PSD→Chem    (heterogeneous surface reactions)

    Design rationale (fallback from full HGT)
    ------------------------------------------
    Full HGT (type-specific Q/K/V) doubled projection parameters vs. the
    homogeneous predecessor without improving held-out val_loss (0.2915 vs 0.2800),
    suggesting Chem and PSD do not need different feature-extraction languages —
    both are concentration time series driven by the same meteorological forcing.

    This version tests whether the relation-type heterogeneity (W_R per edge type)
    is the genuinely useful component:
    * Shared W_Q / W_K / W_V across both node types  ← key change
    * Relation-specific log-scale bias  W_R  [4, n_heads]  (init=0 → neutral)
    * prior_bias: intra-modal (CC, PP) full strength; cross-modal (CP, PC) weaker
    * Joint softmax across all edge types per query node
    * Type-specific output projections + layer norms (kept for interpretability)
    * Aux conditioning: additive bias to logits per relation per head
      (correct replacement for FiLM-on-Q-and-K)

    Parameter count: matches original homogeneous model (~same as predecessor).

    Backward-compatible __init__ signature — enable_cross_modal_floor is silently
    ignored; use_temporal_cnn is still unused (upstream DepthwiseTCN handles it).
    """

    def __init__(
        self,
        n_features,
        window_size,
        n_heads=4,
        head_dim=64,
        dropout=0.1,
        use_temporal_cnn=True,   # kept for call-site compat, unused here
        aux_dim=0,
        n_chem=0,
        enable_cross_modal_floor=False,  # kept for compat, ignored
        disable_rel_scale=False,   # ablation: freeze rel_log_scale at 0
        disable_prior_bias=False,  # ablation: freeze prior_beta at 0
        disable_aux_bias=False,    # ablation: skip low-rank aux bias
        use_homogeneous=False,     # standard self-attention (no CC/PP/CP/PC blocks)
        use_ffn=False,
        ffn_mult=4,
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

        # ── Shared Q/K/V projections (single set for all node types) ──────
        # Both Chem and PSD features are concentration time-series driven by
        # the same meteorological forcing; separate projection languages did
        # not improve held-out performance in ablation.
        self.q_proj = nn.Linear(window_size, self.d_model)
        self.k_proj = nn.Linear(window_size, self.d_model)
        self.v_proj = nn.Linear(window_size, self.d_model)

        # ── Relation-specific log-scale bias W_R ──────────────────────────
        # Shape [4, n_heads]; index order: [CC, PP, CP, PC]
        # Additive to logits in log-space → multiplicative on attention weights.
        # init=0 → all relations start equally weighted; model learns divergence.
        if not disable_rel_scale:
            self.rel_log_scale = nn.Parameter(torch.zeros(4, n_heads))

        # ── Learnable temperature ──────────────────────────────────────────
        # Constrained to [0.5, 1.0] via sigmoid
        self.tau_param = nn.Parameter(torch.zeros(1))

        # ── Learnable prior-bias coefficients ─────────────────────────────
        if not disable_prior_bias:
            # Intra-modal (CC, PP): full strength — co-observation is a reliable signal
            self.prior_beta = nn.Parameter(torch.ones(1) * 0.5)
            # Cross-modal (CP, PC): weaker — penalise over-fitting to training-specific
            # Chem-PSD correlations without completely silencing the bridge when
            # PSD instruments are down.  Init at 0.1 (≈ 1/5 of intra-modal strength).
            self.prior_beta_cross = nn.Parameter(torch.ones(1) * 0.1)

        # ── Aux conditioning: Low-Rank Bias (Plan A, Shared MLP) ───────────
        # Replaces the ineffective [B, h, 1, 1] scalar bias (which is a constant
        # added to all (i,j) pairs and thus FULLY cancelled by softmax).
        #
        # Design:
        #   aux_ctx = [mean, std] of aux features → captures regime + variability
        #   shared MLP → u_scale, v_scale [B, h, r]  (chunked from same output)
        #   node_embed_{c,p} [n_i, r]                 (learnable per-species embeddings)
        #   U_c = node_embed_c * u_scale  [B, h, n_c, r]
        #   bias_CC = U_c @ V_c^T         [B, h, n_c, n_c]   varies per (i,j) → survives softmax
        #
        # Note: shared MLP → U and V are correlated → bias ≈ symmetric.
        # This is acceptable for Plan A; Plan B (separate nets) can be upgraded
        # later if asymmetric chemical relationships need to be captured.
        self.aux_rank = 4
        if aux_dim > 0 and not disable_aux_bias:
            self.node_embed_c = nn.Parameter(torch.randn(n_chem, self.aux_rank) * 0.02)
            self.node_embed_p = nn.Parameter(torch.randn(self.n_psd, self.aux_rank) * 0.02)
            self.aux_bias_net = nn.Sequential(
                nn.Linear(aux_dim * 2, 32),   # mean + std → richer weather state
                nn.SiLU(),
                nn.Linear(32, 2 * n_heads * self.aux_rank),  # → u_scale ++ v_scale
            )
            # Zero-init: bias starts at 0 → training starts from clean QK attention
            nn.init.zeros_(self.aux_bias_net[-1].weight)
            nn.init.zeros_(self.aux_bias_net[-1].bias)

        # ── Output projections ─────────────────────────────────────────────
        if use_homogeneous:
            self.out_proj = nn.Linear(self.d_model, window_size)
            self.norm = nn.LayerNorm(window_size)
        else:
            self.out_chem = nn.Linear(self.d_model, window_size)
            self.out_psd = nn.Linear(self.d_model, window_size)
            self.norm_chem = nn.LayerNorm(window_size)
            self.norm_psd = nn.LayerNorm(window_size)
        self.ffn = WindowTokenFFN(window_size, mult=ffn_mult, dropout=dropout) if use_ffn else None

        self.dropout = nn.Dropout(dropout)

        # Interpretability cache (same attribute names as predecessor)
        self.last_attention_weights = None
        self.last_attention_weights_heads = None
        self.last_attention_weights_heads_batch = None

    # ── helpers ───────────────────────────────────────────────────────────
    def _to_heads(self, t, N, B):
        """[B, N, d_model] → [B, n_heads, N, head_dim]"""
        return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def _from_heads(self, t, N, B):
        """[B, n_heads, N, head_dim] → [B, N, d_model]"""
        return t.transpose(1, 2).contiguous().view(B, N, self.d_model)

    # ── forward ───────────────────────────────────────────────────────────
    def forward(self, x, obs_mask=None, x_aux=None):
        """
        Args
        ----
        x        : [B, C, W]        input features (Chem first, PSD after)
        obs_mask : [B, W, C]        observation mask (1=observed); None → all observed
        x_aux    : [B, aux_dim, W]  auxiliary context for logit bias conditioning

        Returns
        -------
        out      : [B, C, W]
        attn_avg : [B, C, C]        attention matrix averaged over heads
        """
        B, C, W = x.shape

        # ══════════════════════════════════════════════════════════════════
        # Homogeneous path: standard multi-head self-attention over all C
        # features jointly.  No CC/PP/CP/PC block structure, no relation
        # biases.  This is the architecture that achieved val 0.2800.
        # ══════════════════════════════════════════════════════════════════
        if self.use_homogeneous:
            Q = self._to_heads(self.q_proj(x), C, B)     # [B, h, C, head_dim]
            K = self._to_heads(self.k_proj(x), C, B)
            V = self._to_heads(self.v_proj(x), C, B)

            scale = np.sqrt(self.head_dim) * (0.5 + 0.5 * torch.sigmoid(self.tau_param))
            logits = torch.matmul(Q, K.transpose(-1, -2)) / scale  # [B, h, C, C]

            attn = F.softmax(logits, dim=-1)              # [B, h, C, C]
            attn = self.dropout(attn)

            out = torch.matmul(attn, V)                   # [B, h, C, head_dim]
            out = self._from_heads(out, C, B)              # [B, C, d_model]
            out = self.norm(self.out_proj(out) + x)        # [B, C, W] residual + norm
            if self.ffn is not None:
                out = self.ffn(out)

            attn_avg = attn.detach().mean(dim=1)           # [B, C, C]
            self.last_attention_weights = attn_avg.mean(dim=0)
            self.last_attention_weights_heads = attn.detach().mean(dim=0)
            self.last_attention_weights_heads_batch = attn.detach()

            return out, attn_avg

        # ══════════════════════════════════════════════════════════════════
        # RoHG path: CC/PP/CP/PC block structure with relation biases
        # ══════════════════════════════════════════════════════════════════
        n_c, n_p = self.n_chem, self.n_psd

        # ── Split by node type ────────────────────────────────────────────
        x_c = x[:, :n_c, :]   # [B, n_chem, W]
        x_p = x[:, n_c:, :]   # [B, n_psd,  W]

        # ── Observation masks per type ────────────────────────────────────
        if obs_mask is None:
            mask_c = torch.ones(B, n_c, W, device=x.device)
            mask_p = torch.ones(B, n_p, W, device=x.device)
        else:
            mask = obs_mask.permute(0, 2, 1).float()   # [B, C, W]
            mask_c = mask[:, :n_c, :]
            mask_p = mask[:, n_c:, :]

        # ── Shared projections applied per node type ──────────────────────
        # q_proj/k_proj/v_proj weights are shared; node-type structure enters
        # only through rel_log_scale, prior_bias, and output projections.
        Q_c = self._to_heads(self.q_proj(x_c), n_c, B)  # [B, h, n_chem, head_dim]
        Q_p = self._to_heads(self.q_proj(x_p), n_p, B)
        K_c = self._to_heads(self.k_proj(x_c), n_c, B)
        K_p = self._to_heads(self.k_proj(x_p), n_p, B)
        V_c = self._to_heads(self.v_proj(x_c), n_c, B)
        V_p = self._to_heads(self.v_proj(x_p), n_p, B)

        # ── Base attention logits (4 blocks) ──────────────────────────────
        scale_temp = np.sqrt(self.head_dim) * (0.5 + 0.5 * torch.sigmoid(self.tau_param))

        logits_CC = torch.matmul(Q_c, K_c.transpose(-1, -2)) / scale_temp  # [B, h, n_c, n_c]
        logits_PP = torch.matmul(Q_p, K_p.transpose(-1, -2)) / scale_temp  # [B, h, n_p, n_p]
        logits_CP = torch.matmul(Q_c, K_p.transpose(-1, -2)) / scale_temp  # [B, h, n_c, n_p]
        logits_PC = torch.matmul(Q_p, K_c.transpose(-1, -2)) / scale_temp  # [B, h, n_p, n_c]

        # ── Relation-specific log-scale bias W_R ──────────────────────────
        if not self.disable_rel_scale:
            # [4, h] → broadcast shape [B, h, 1, 1] per block
            def _rel(idx):
                return self.rel_log_scale[idx].view(1, self.n_heads, 1, 1)

            logits_CC = logits_CC + _rel(0)
            logits_PP = logits_PP + _rel(1)
            logits_CP = logits_CP + _rel(2)
            logits_PC = logits_PC + _rel(3)

        # ── Prior bias ─────────────────────────────────────────────────────
        if not self.disable_prior_bias:
            eps = 1e-6
            co_cc = (torch.matmul(mask_c, mask_c.transpose(1, 2)) / W).unsqueeze(1)  # [B,1,n_c,n_c]
            co_pp = (torch.matmul(mask_p, mask_p.transpose(1, 2)) / W).unsqueeze(1)  # [B,1,n_p,n_p]
            co_cp = (torch.matmul(mask_c, mask_p.transpose(1, 2)) / W).unsqueeze(1)  # [B,1,n_c,n_p]
            co_pc = co_cp.transpose(-1, -2)                                           # [B,1,n_p,n_c]

            # Intra-modal: full prior_beta — co-observation is a reliable signal
            logits_CC = logits_CC + self.prior_beta * torch.log(co_cc + eps)
            logits_PP = logits_PP + self.prior_beta * torch.log(co_pp + eps)
            # Cross-modal: weaker prior_beta_cross — adds regularisation cost to
            # Chem-PSD learning without killing the bridge when PSD is fully missing.
            # When PSD all absent: bias ≈ 0.1 × log(ε) ≈ −1.4, soft suppression only.
            logits_CP = logits_CP + self.prior_beta_cross * torch.log(co_cp + eps)
            logits_PC = logits_PC + self.prior_beta_cross * torch.log(co_pc + eps)

        # ── Aux conditioning: Low-Rank Bias (Plan A) ─────────────────────
        if hasattr(self, 'aux_bias_net') and x_aux is not None:
            # Mean captures background regime; std captures event intensity.
            # Together they give a richer weather-state description than mean alone.
            aux_mean = x_aux.mean(dim=-1)                            # [B, aux_dim]
            aux_std  = x_aux.std(dim=-1).clamp(min=1e-6)             # [B, aux_dim]
            aux_ctx  = torch.cat([aux_mean, aux_std], dim=-1)         # [B, 2*aux_dim]

            uv = self.aux_bias_net(aux_ctx)                           # [B, 2*h*r]
            u_scale, v_scale = uv.chunk(2, dim=-1)                    # each [B, h*r]
            r = self.aux_rank
            u_scale = u_scale.view(B, self.n_heads, 1, r)             # [B, h, 1, r]
            v_scale = v_scale.view(B, self.n_heads, 1, r)             # [B, h, 1, r]

            # Expand node embeddings with weather-conditioned scale directions.
            # NOTE: use bias_U/V prefix to avoid shadowing V_c/V_p from v_proj above.
            # [1, 1, n_i, r] * [B, h, 1, r] → [B, h, n_i, r]
            node_c_emb = self.node_embed_c.unsqueeze(0).unsqueeze(0)  # [1, 1, n_c, r]
            node_p_emb = self.node_embed_p.unsqueeze(0).unsqueeze(0)  # [1, 1, n_p, r]
            bias_U_c = node_c_emb * u_scale   # [B, h, n_c, r]
            bias_V_c = node_c_emb * v_scale   # [B, h, n_c, r]
            bias_U_p = node_p_emb * u_scale   # [B, h, n_p, r]
            bias_V_p = node_p_emb * v_scale   # [B, h, n_p, r]

            # Outer product → [B, h, n_i, n_j]: pair-specific, survives softmax
            logits_CC = logits_CC + torch.matmul(bias_U_c, bias_V_c.transpose(-1, -2))
            logits_PP = logits_PP + torch.matmul(bias_U_p, bias_V_p.transpose(-1, -2))
            logits_CP = logits_CP + torch.matmul(bias_U_c, bias_V_p.transpose(-1, -2))
            logits_PC = logits_PC + torch.matmul(bias_U_p, bias_V_c.transpose(-1, -2))

        # ── Self-masking on intra-modal diagonals ─────────────────────────
        eye_c = torch.eye(n_c, device=x.device, dtype=torch.bool).view(1, 1, n_c, n_c)
        eye_p = torch.eye(n_p, device=x.device, dtype=torch.bool).view(1, 1, n_p, n_p)
        logits_CC = logits_CC.masked_fill(eye_c, -1e4)
        logits_PP = logits_PP.masked_fill(eye_p, -1e4)

        # ── Joint softmax across all edge types per query node ────────────
        # Chem query normalises over [n_chem CC-keys | n_psd CP-keys]
        # PSD  query normalises over [n_chem PC-keys | n_psd PP-keys]
        attn_C = F.softmax(torch.cat([logits_CC, logits_CP], dim=-1), dim=-1)  # [B,h,n_c,n_c+n_p]
        attn_P = F.softmax(torch.cat([logits_PC, logits_PP], dim=-1), dim=-1)  # [B,h,n_p,n_c+n_p]

        attn_C = self.dropout(attn_C)
        attn_P = self.dropout(attn_P)

        # ── Split back into per-block weights ─────────────────────────────
        attn_CC, attn_CP = attn_C[:, :, :, :n_c], attn_C[:, :, :, n_c:]
        attn_PC, attn_PP = attn_P[:, :, :, :n_c], attn_P[:, :, :, n_c:]

        # ── Aggregate: weighted sum of type-specific V ────────────────────
        out_c = torch.matmul(attn_CC, V_c) + torch.matmul(attn_CP, V_p)  # [B,h,n_c,head_dim]
        out_p = torch.matmul(attn_PC, V_c) + torch.matmul(attn_PP, V_p)  # [B,h,n_p,head_dim]

        # ── Type-specific output projections + residual + norm ────────────
        out_c = self.norm_chem(self.out_chem(self._from_heads(out_c, n_c, B)) + x_c)
        out_p = self.norm_psd( self.out_psd( self._from_heads(out_p, n_p, B)) + x_p)

        out = torch.cat([out_c, out_p], dim=1)  # [B, C, W]
        if self.ffn is not None:
            out = self.ffn(out)

        # ── Interpretability: reconstruct full [B, C, C] matrix ───────────
        attn_full = torch.zeros(B, self.n_heads, C, C, device=x.device)
        attn_full[:, :, :n_c, :n_c] = attn_CC
        attn_full[:, :, :n_c, n_c:] = attn_CP
        attn_full[:, :, n_c:, :n_c] = attn_PC
        attn_full[:, :, n_c:, n_c:] = attn_PP

        attn_avg = attn_full.detach().mean(dim=1)          # [B, C, C]
        self.last_attention_weights = attn_avg.mean(dim=0) # [C, C]
        self.last_attention_weights_heads = attn_full.detach().mean(dim=0)  # [h, C, C]
        self.last_attention_weights_heads_batch = attn_full.detach()        # [B, h, C, C]

        return out, attn_avg


# ==============================================================================
# Temporal Attention Pooling
# ==============================================================================
class TemporalAttentionPool(nn.Module):
    """
    Temporal Attention Pooling to capture key time steps.
    Learns to weigh important time steps rather than simple averaging.
    """
    def __init__(self, input_dim):
        super().__init__()
        # Lightweight attention network
        self.score_net = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 4, 16)),
            nn.Tanh(),
            nn.Linear(max(input_dim // 4, 16), 1)
        )
        
    def forward(self, x, obs_mask=None):
        """
        Args:
            x: [Batch, Channels, Window]
            obs_mask: [Batch, Window, Feature] - target observation mask (1=observed).
                      Must be target-only (not full_obs_mask which includes always-observed
                      aux channels, making any_obs permanently True = a no-op).
        Returns:
            out: [Batch, Channels]
        """
        # Permute to [B, W, C] for linear interactions
        x_t = x.permute(0, 2, 1) # [B, W, C]

        # Calculate scores
        scores = self.score_net(x_t) # [B, W, 1]

        # Observation-aware masking: suppress timesteps where ALL target features
        # are missing. These carry only zero-fill artefacts after input masking,
        # so they should not influence the pooled latent representation.
        if obs_mask is not None:
            # any_obs: True if at least one target feature is observed at that timestep
            any_obs = obs_mask.any(dim=-1, keepdim=True).float()  # [B, W, 1]
            scores = scores.masked_fill(any_obs == 0, -1e9)

        # Softmax over time dimension
        weights = F.softmax(scores, dim=1) # [B, W, 1]

        # Weighted sum: [B, W, C] * [B, W, 1] -> [B, W, C] -> sum(dim=1) -> [B, C]
        out = (x_t * weights).sum(dim=1)

        return out


class AxialObservedAttentionBlock(nn.Module):
    """
    Observed-aware axial attention before feature graph mixing.

    The block is an imputation operator, not a generic representation mixer:
    missing queries may attend observed keys, but missing keys are never treated
    as evidence. Residual updates are applied only at missing target positions.
    """

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

        self.time_gate = nn.Parameter(torch.full((self.n_features,), float(time_gate_init)))
        self.cross_gate = nn.Parameter(torch.full((self.n_features,), float(cross_gate_init)))

        self.last_time_gate_mean = None
        self.last_time_gate_chem_mean = None
        self.last_time_gate_psd_mean = None
        self.last_cross_gate_mean = None
        self.last_cross_gate_chem_mean = None
        self.last_cross_gate_psd_mean = None
        self.last_time_no_key_fraction = None
        self.last_cross_no_key_fraction = None
        self.last_cross_valid_query_fraction = None
        self.last_cross_entropy_missing = None
        self.last_cross_top1_mass = None
        self.last_cross_top3_mass = None
        self.last_psd_to_chem_mass = None
        self.last_psd_to_psd_mass = None

    def _split_gate_means(self, gate):
        if self.n_chem > 0 and self.n_features > self.n_chem:
            chem = float(gate[:self.n_chem].detach().mean().item())
            psd = float(gate[self.n_chem:].detach().mean().item())
        else:
            chem = None
            psd = None
        return float(gate.detach().mean().item()), chem, psd

    def _to_heads(self, x):
        # x: [B, T, C, D] -> [B, H, T, C, head_dim]
        B, T, C, _ = x.shape
        return x.view(B, T, C, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

    def _from_heads(self, x):
        # x: [B, H, T, C, head_dim] -> [B, T, C, D]
        B, _, T, C, _ = x.shape
        return x.permute(0, 2, 3, 1, 4).contiguous().view(B, T, C, self.attn_dim)

    def forward(self, x, target_obs_mask=None):
        """
        Args:
            x: [B, C, W]
            target_obs_mask: [B, W, C], 1=observed, 0=missing
        """
        B, C, W = x.shape
        if C != self.n_features:
            raise ValueError(
                f"AxialObservedAttentionBlock expected {self.n_features} features, got {C}"
            )

        if target_obs_mask is None:
            obs = torch.ones(B, C, W, device=x.device, dtype=x.dtype)
        else:
            obs = target_obs_mask.permute(0, 2, 1).to(device=x.device, dtype=x.dtype)
        missing = 1.0 - obs

        feat_ids = torch.arange(C, device=x.device)
        feat_emb = self.feature_embed(feat_ids).view(1, C, 1, self.attn_dim)
        tok0 = self.input_norm(self.value_proj(x.unsqueeze(-1)) + feat_emb)

        time_gate = torch.sigmoid(self.time_gate).to(dtype=x.dtype).view(1, C, 1)
        cross_gate = torch.sigmoid(self.cross_gate).to(dtype=x.dtype).view(1, C, 1)
        (
            self.last_time_gate_mean,
            self.last_time_gate_chem_mean,
            self.last_time_gate_psd_mean,
        ) = self._split_gate_means(torch.sigmoid(self.time_gate))
        (
            self.last_cross_gate_mean,
            self.last_cross_gate_chem_mean,
            self.last_cross_gate_psd_mean,
        ) = self._split_gate_means(torch.sigmoid(self.cross_gate))

        # 1) Same-feature time-axis observed-key attention.
        time_in = tok0.reshape(B * C, W, self.attn_dim)
        feature_obs = obs.reshape(B * C, W) > 0.0
        no_time_key = ~feature_obs.any(dim=1, keepdim=True)
        key_padding_mask = (~feature_obs).masked_fill(no_time_key, False)
        self.last_time_no_key_fraction = float(no_time_key.float().mean().detach().item())

        time_out, _ = self.time_attn(
            time_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        time_out = time_out.reshape(B, C, W, self.attn_dim)
        has_time_key = (~no_time_key).reshape(B, C, 1).to(dtype=x.dtype)
        missing_time = missing * has_time_key

        time_delta = self.time_scalar_out(time_out).squeeze(-1) * missing_time
        h_time = tok0 + time_gate.unsqueeze(-1) * time_out * missing_time.unsqueeze(-1)

        # 2) Same-timestep cross-feature observed-key attention.
        h_btcd = h_time.permute(0, 2, 1, 3).contiguous()  # [B, W, C, D]
        obs_btc = obs.permute(0, 2, 1).contiguous() > 0.0
        missing_btc = missing.permute(0, 2, 1).contiguous() > 0.0
        cross_delta_btc = x.new_zeros((B, W, C))

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
        feature_idx = torch.arange(C, device=x.device)
        psd_query_feature = feature_idx >= self.n_chem if self.n_chem > 0 else torch.zeros(C, device=x.device, dtype=torch.bool)

        for start in range(0, W, self.cross_time_chunk):
            end = min(W, start + self.cross_time_chunk)
            h_chunk = h_btcd[:, start:end]  # [B, Tc, C, D]
            obs_chunk = obs_btc[:, start:end]  # [B, Tc, C]
            miss_chunk = missing_btc[:, start:end]  # [B, Tc, C]
            no_key = ~obs_chunk.any(dim=-1)  # [B, Tc]
            no_key_sum += float(no_key.float().sum().detach().item())
            no_key_count += int(no_key.numel())

            key_valid = obs_chunk | no_key.unsqueeze(-1)
            q = self._to_heads(self.cross_q(h_chunk))
            k = self._to_heads(self.cross_k(h_chunk))
            v = self._to_heads(self.cross_v(h_chunk))

            scores = torch.matmul(q, k.transpose(-1, -2)) / scale
            scores = scores.masked_fill(~key_valid[:, None, :, None, :], -1e4)
            attn = F.softmax(scores, dim=-1)
            attn = self.cross_dropout(attn)

            out = torch.matmul(attn, v)
            out = self.cross_out(self._from_heads(out))
            scalar = self.cross_scalar_out(out).squeeze(-1)  # [B, Tc, C]

            valid_query = miss_chunk & (~no_key.unsqueeze(-1))
            valid_query_sum += int(valid_query.sum().detach().item())
            cross_delta_btc[:, start:end] = scalar * valid_query.to(dtype=scalar.dtype)

            with torch.no_grad():
                if valid_query.any():
                    attn_mean = attn.detach().mean(dim=1).clamp_min(1e-8)  # [B, Tc, Cq, Ck]
                    entropy = -(attn_mean * attn_mean.log()).sum(dim=-1)
                    entropy_sum += float(entropy[valid_query].sum().item())
                    entropy_count += int(valid_query.sum().item())

                    sorted_mass = torch.sort(attn_mean, dim=-1, descending=True).values
                    top1_sum += float(sorted_mass[..., 0][valid_query].sum().item())
                    top3_sum += float(sorted_mass[..., : min(3, C)].sum(dim=-1)[valid_query].sum().item())

                    if self.n_chem > 0 and self.n_chem < C:
                        psd_query = valid_query & psd_query_feature.view(1, 1, C)
                        if psd_query.any():
                            chem_mass = attn_mean[..., :self.n_chem].sum(dim=-1)
                            psd_mass = attn_mean[..., self.n_chem:].sum(dim=-1)
                            psd_chem_sum += float(chem_mass[psd_query].sum().item())
                            psd_psd_sum += float(psd_mass[psd_query].sum().item())
                            psd_count += int(psd_query.sum().item())

        self.last_cross_no_key_fraction = (
            no_key_sum / no_key_count if no_key_count > 0 else None
        )
        self.last_cross_valid_query_fraction = (
            valid_query_sum / total_missing_queries if total_missing_queries > 0 else None
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
    """
    Missing-aware per-feature temporal attention applied after the depthwise TCN
    and before graph mixing.

    Each target feature attends only over its own time history. Missing query
    positions are updated from observed keys/values of the same feature, which
    is more aligned with "same-species temporal retrieval" than the later
    hidden-space shared refiner.
    """
    def __init__(self, window_size, attn_dim=64, n_heads=4, gate_init=-1.0,
                 dropout=0.1, chunk_size=256, record_weights=False,
                 mode='dense', bucket_bounds=(4, 8, 16, 32, 64)):
        super().__init__()
        self.window_size = int(window_size)
        self.attn_dim = int(attn_dim)
        self.n_heads = int(n_heads)
        self.chunk_size = int(chunk_size)
        self.record_weights = bool(record_weights)
        self.mode = str(mode)
        self.bucket_bounds = tuple(sorted(int(b) for b in bucket_bounds))
        valid_modes = {'dense', 'bucketed_missing_query_only'}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported pregraph temporal attention mode: {self.mode}")

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

    def _missing_bucket_upper(self, missing_count: int) -> int:
        for upper in self.bucket_bounds:
            if missing_count <= upper:
                return upper
        return self.window_size

    def _forward_dense(self, x, tok, feature_obs=None):
        B, C, W = x.shape
        query_missing = None
        query_observed = None
        valid_entropy_rows = None
        key_padding_mask = None
        missing_mask = None
        if feature_obs is not None:
            key_padding_mask = feature_obs <= 0.0
            valid_entropy_rows = torch.ones(B * C, dtype=torch.bool, device=x.device)
            if key_padding_mask.any():
                # Avoid rows where every key is masked, which would otherwise
                # produce NaNs inside attention.
                all_missing = key_padding_mask.all(dim=1, keepdim=True)
                valid_entropy_rows = (~all_missing.squeeze(1))
                key_padding_mask = key_padding_mask.masked_fill(all_missing, False)

            query_missing = feature_obs <= 0.0
            query_observed = ~query_missing
            missing_mask = query_missing.reshape(B, C, W).to(x.dtype)

        gate = torch.sigmoid(self.gate)
        self.last_gate = float(gate.detach().item())
        out_chunks = []
        missing_entropy_sum = 0.0
        missing_entropy_count = 0
        observed_entropy_sum = 0.0
        observed_entropy_count = 0

        total = B * C
        chunk_size = total if self.chunk_size <= 0 else self.chunk_size
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            tok_chunk = tok[start:end]
            key_mask_chunk = key_padding_mask[start:end] if key_padding_mask is not None else None

            attn_out, attn_weights = self.attn(
                tok_chunk,
                key_padding_mask=key_mask_chunk,
                need_weights=self.record_weights,
            )
            if self.record_weights and attn_weights is not None:
                attn_probs = attn_weights.mean(dim=1).clamp_min(1e-8)
                entropy = -(attn_probs * attn_probs.log()).sum(dim=-1)
                if query_missing is not None:
                    missing_chunk = query_missing[start:end]
                    observed_chunk = query_observed[start:end]
                    if valid_entropy_rows is not None:
                        valid_rows_chunk = valid_entropy_rows[start:end].unsqueeze(1)
                        missing_chunk = missing_chunk & valid_rows_chunk
                        observed_chunk = observed_chunk & valid_rows_chunk
                    if missing_chunk.any():
                        missing_entropy_sum += entropy[missing_chunk].sum().item()
                        missing_entropy_count += int(missing_chunk.sum().item())
                    if observed_chunk.any():
                        observed_entropy_sum += entropy[observed_chunk].sum().item()
                        observed_entropy_count += int(observed_chunk.sum().item())
            out_chunks.append(attn_out)
        self.last_missing_query_attn_entropy = (
            missing_entropy_sum / missing_entropy_count
            if missing_entropy_count > 0 else None
        )
        self.last_observed_query_attn_entropy = (
            observed_entropy_sum / observed_entropy_count
            if observed_entropy_count > 0 else None
        )

        delta = self.out_proj(torch.cat(out_chunks, dim=0)).squeeze(-1).reshape(B, C, W)
        if missing_mask is not None:
            delta = delta * missing_mask

        return x + gate * delta

    def _forward_bucketed_missing_query_only(self, x, tok, feature_obs):
        B, C, W = x.shape
        total = B * C
        gate = torch.sigmoid(self.gate)
        self.last_gate = float(gate.detach().item())

        # Build buckets on CPU to avoid repeated GPU sync from many tiny index ops.
        feature_obs_cpu = feature_obs.detach().to(device='cpu', dtype=torch.bool)
        buckets = {}
        for seq_idx in range(total):
            obs_idx_cpu = torch.nonzero(feature_obs_cpu[seq_idx], as_tuple=False).flatten()
            miss_idx_cpu = torch.nonzero(~feature_obs_cpu[seq_idx], as_tuple=False).flatten()
            if miss_idx_cpu.numel() == 0:
                continue
            if obs_idx_cpu.numel() == 0:
                continue
            bucket_key = self._missing_bucket_upper(int(miss_idx_cpu.numel()))
            buckets.setdefault(bucket_key, []).append((seq_idx, miss_idx_cpu, obs_idx_cpu))

        delta_flat = torch.zeros((total, W), device=x.device, dtype=x.dtype)
        missing_entropy_sum = 0.0
        missing_entropy_count = 0
        self.last_observed_query_attn_entropy = None

        for _, entries in sorted(buckets.items(), key=lambda kv: kv[0]):
            n_bucket = len(entries)
            max_m = max(int(miss_idx.numel()) for _, miss_idx, _ in entries)
            max_o = max(int(obs_idx.numel()) for _, _, obs_idx in entries)

            q_batch = tok.new_zeros((n_bucket, max_m, self.attn_dim))
            k_batch = tok.new_zeros((n_bucket, max_o, self.attn_dim))
            v_batch = tok.new_zeros((n_bucket, max_o, self.attn_dim))
            q_pos = torch.zeros((n_bucket, max_m), device=x.device, dtype=torch.long)
            kv_pos = torch.zeros((n_bucket, max_o), device=x.device, dtype=torch.long)
            q_valid = torch.zeros((n_bucket, max_m), device=x.device, dtype=torch.bool)
            kv_valid = torch.zeros((n_bucket, max_o), device=x.device, dtype=torch.bool)

            for row_idx, (seq_idx, miss_idx_cpu, obs_idx_cpu) in enumerate(entries):
                miss_idx = torch.as_tensor(miss_idx_cpu, device=x.device, dtype=torch.long)
                obs_idx = torch.as_tensor(obs_idx_cpu, device=x.device, dtype=torch.long)
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
                attn_probs = attn_weights.mean(dim=1).clamp_min(1e-8)  # [N, M, O]
                entropy = -(attn_probs * attn_probs.log()).sum(dim=-1)
                if q_valid.any():
                    missing_entropy_sum += entropy[q_valid].sum().item()
                    missing_entropy_count += int(q_valid.sum().item())

            proj = self.out_proj(attn_out).squeeze(-1)
            for row_idx, (seq_idx, miss_idx_cpu, _) in enumerate(entries):
                miss_idx = torch.as_tensor(miss_idx_cpu, device=x.device, dtype=torch.long)
                m_i = int(miss_idx.numel())
                delta_flat[seq_idx, miss_idx] = proj[row_idx, :m_i]

        self.last_missing_query_attn_entropy = (
            missing_entropy_sum / missing_entropy_count
            if missing_entropy_count > 0 else None
        )
        delta = delta_flat.reshape(B, C, W)
        return x + gate * delta

    def forward(self, x, target_obs_mask=None):
        """
        Args:
            x: [B, C, W]
            target_obs_mask: [B, W, C] with 1=observed, 0=missing
        Returns:
            [B, C, W]
        """
        B, C, W = x.shape
        x_flat = x.reshape(B * C, W, 1)
        tok = self.in_proj(x_flat)
        tok = self.in_norm(tok)
        feature_obs = None
        if target_obs_mask is not None:
            feature_obs = target_obs_mask.permute(0, 2, 1).reshape(B * C, W).float()

        if self.mode == 'bucketed_missing_query_only' and feature_obs is not None:
            return self._forward_bucketed_missing_query_only(x, tok, feature_obs)
        return self._forward_dense(x, tok, feature_obs=feature_obs)


class TemporalObservationRefiner(nn.Module):
    """
    Observation-aware temporal self-attention refiner for encoder hidden states.

    The first version is intentionally conservative:
    - one residual block
    - small learnable residual gate
    - fully-missing target timesteps cannot serve as keys
    - higher target observation-rate timesteps receive a soft positive key bias
    """
    def __init__(self, hidden_dim, window_size, attn_dim=128, n_heads=4,
                 gate_init=-2.0, fixed_gate=None, obs_bias_init=1.0, dropout=0.1):
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
            self.register_parameter('gate', None)
        self.obs_bias_scale = nn.Parameter(torch.tensor(float(obs_bias_init)))

        self.last_gate = None
        self.last_missing_query_attn_entropy = None
        self.last_observed_query_attn_entropy = None

    def forward(self, h, target_obs_mask=None):
        """
        Args:
            h: [B, H, W]
            target_obs_mask: [B, W, D_target] with 1=observed, 0=missing
        Returns:
            [B, H, W]
        """
        h_res = h
        x = self.in_proj(h).transpose(1, 2)  # [B, W, d_attn]
        x = self.in_norm(x)

        attn_mask = None
        if target_obs_mask is not None:
            obs_rate = target_obs_mask.float().mean(dim=-1)  # [B, W], target-only
            fully_missing = obs_rate <= 0.0
            if fully_missing.any():
                all_missing = fully_missing.all(dim=1, keepdim=True)
                fully_missing = fully_missing.masked_fill(all_missing, False)

            # Use a single additive float mask so PyTorch does not need to
            # reconcile mixed bool key_padding_mask + float attn_mask types.
            key_bias = self.obs_bias_scale.to(x.dtype) * obs_rate.to(x.dtype).unsqueeze(1).expand(-1, x.size(1), -1)
            if fully_missing.any():
                hard_bias = fully_missing.to(x.dtype).unsqueeze(1).expand(-1, x.size(1), -1) * (-1e9)
                key_bias = key_bias + hard_bias

            attn_mask = key_bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            attn_mask = attn_mask.reshape(x.size(0) * self.n_heads, x.size(1), x.size(1))

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
            attn_mean = attn_weights.detach().mean(dim=1)  # [B, W, W]
            entropy = -(attn_mean.clamp_min(1e-8) * attn_mean.clamp_min(1e-8).log()).sum(dim=-1)
            if target_obs_mask is not None:
                query_missing = (target_obs_mask == 0).all(dim=-1)
                query_observed = ~query_missing
                self.last_missing_query_attn_entropy = (
                    float(entropy[query_missing].mean().item()) if query_missing.any() else None
                )
                self.last_observed_query_attn_entropy = (
                    float(entropy[query_observed].mean().item()) if query_observed.any() else None
                )
            else:
                self.last_missing_query_attn_entropy = None
                self.last_observed_query_attn_entropy = float(entropy.mean().item())
        else:
            self.last_missing_query_attn_entropy = None
            self.last_observed_query_attn_entropy = None

        attn_out = self.out_proj(attn_out.transpose(1, 2))
        return self.out_norm((h_res + gate * attn_out).transpose(1, 2)).transpose(1, 2)

# ==============================================================================
# Graph-Enhanced Encoder
# ==============================================================================
class GraphEncoder(nn.Module):
    """
    TCN Encoder with Input Graph Layer for learning feature relationships.
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim, window_size, num_layers=5,
                 kernel_size=3, dropout=0.1, n_graph_heads=4, target_dim=None,
                 aux_dim=None, use_input_graph_layer=True, use_cross_modal_graph=True,
                 use_tcn=True, n_input_graph_layers=1, use_parallel_graph=False,
                 use_temporal_cnn=True, n_chem=0, enable_cross_modal_floor=False,
                 disable_rel_scale=False, disable_prior_bias=False, disable_aux_bias=False,
                 use_homogeneous=False, ignore_obs_mask=False,
                 use_graph_ffn=False, graph_ffn_mult=4,
                 use_token_graph_trunk=False, token_graph_dim=None,
                 token_graph_out_gate_init=-1.0,
                 use_latent_pooled_norm=False, latent_logvar_min=None,
                 latent_logvar_max=None,
                 use_local_context_map=False, local_context_dim=32,
                 local_context_steps=None,
                 local_context_observe_aware=False,
                 local_context_observe_aware_blend_gate_init=None,
                 use_pregraph_feature_temporal_attn=False,
                 use_pregraph_depthwise_tcn=True,
                 use_axial_observed_attn=False,
                 axial_attn_dim=64,
                 axial_attn_heads=4,
                 axial_time_gate_init=0.0,
                 axial_cross_gate_init=0.0,
                 axial_cross_time_chunk=4,
                 axial_null_output=False,
                 pregraph_feature_temporal_attn_dim=64,
                 pregraph_feature_temporal_attn_heads=4,
                 pregraph_feature_temporal_attn_gate_init=-1.0,
                 pregraph_feature_temporal_attn_chunk_size=256,
                 pregraph_feature_temporal_attn_record_weights=False,
                 pregraph_feature_temporal_attn_mode='dense',
                 use_local_chunk_graph=False,
                 local_chunk_graph_mode='parallel',
                 local_chunk_graph_chunk_size=6,
                 local_chunk_graph_dim=128,
                 local_chunk_graph_heads=4,
                 local_chunk_graph_gate_init=-2.0,
                 local_chunk_graph_ffn_mult=4,
                 local_chunk_graph_use_mask_embed=False,
                 local_chunk_graph_out_proj_init_std=0.0,
                 use_temporal_refiner=False, temporal_refiner_dim=128,
                 temporal_refiner_heads=4, temporal_refiner_gate_init=-2.0,
                 temporal_refiner_fixed_gate=None):
        super().__init__()

        self.ignore_obs_mask = ignore_obs_mask

        self.use_parallel_graph = use_parallel_graph
        self.use_temporal_cnn = use_temporal_cnn
        self.n_chem = n_chem

        self.use_tcn = use_tcn
        self.use_graph_ffn = bool(use_graph_ffn)
        self.graph_ffn_mult = int(graph_ffn_mult)
        self.use_token_graph_trunk = bool(use_token_graph_trunk)
        self.token_graph_dim = (
            None if token_graph_dim is None else int(token_graph_dim)
        )
        self.token_graph_out_gate_init = float(token_graph_out_gate_init)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.use_latent_pooled_norm = bool(use_latent_pooled_norm)
        self.latent_logvar_min = latent_logvar_min
        self.latent_logvar_max = latent_logvar_max
        self.use_local_context_map = bool(use_local_context_map)
        self.local_context_dim = int(local_context_dim)
        self.local_context_steps = (
            None if local_context_steps is None else int(local_context_steps)
        )
        self.local_context_observe_aware = bool(local_context_observe_aware)
        self.local_context_observe_aware_blend_gate_init = (
            None
            if local_context_observe_aware_blend_gate_init is None
            else float(local_context_observe_aware_blend_gate_init)
        )
        if self.local_context_observe_aware_blend_gate_init is None:
            self.local_context_observe_aware_blend_gate = None
        else:
            self.local_context_observe_aware_blend_gate = nn.Parameter(
                torch.tensor(self.local_context_observe_aware_blend_gate_init)
            )
        self.use_pregraph_feature_temporal_attn = bool(use_pregraph_feature_temporal_attn)
        self.use_pregraph_depthwise_tcn = bool(use_pregraph_depthwise_tcn)
        self.use_axial_observed_attn = bool(use_axial_observed_attn)
        self.axial_attn_dim = int(axial_attn_dim)
        self.axial_attn_heads = int(axial_attn_heads)
        self.axial_time_gate_init = float(axial_time_gate_init)
        self.axial_cross_gate_init = float(axial_cross_gate_init)
        self.axial_cross_time_chunk = int(axial_cross_time_chunk)
        self.axial_null_output = bool(axial_null_output)
        self.pregraph_feature_temporal_attn_dim = int(pregraph_feature_temporal_attn_dim)
        self.pregraph_feature_temporal_attn_heads = int(pregraph_feature_temporal_attn_heads)
        self.pregraph_feature_temporal_attn_gate_init = float(pregraph_feature_temporal_attn_gate_init)
        self.pregraph_feature_temporal_attn_chunk_size = int(pregraph_feature_temporal_attn_chunk_size)
        self.pregraph_feature_temporal_attn_record_weights = bool(pregraph_feature_temporal_attn_record_weights)
        self.pregraph_feature_temporal_attn_mode = str(pregraph_feature_temporal_attn_mode)
        self.use_local_chunk_graph = bool(use_local_chunk_graph)
        self.local_chunk_graph_mode = str(local_chunk_graph_mode)
        self.local_chunk_graph_chunk_size = int(local_chunk_graph_chunk_size)
        self.local_chunk_graph_dim = int(local_chunk_graph_dim)
        self.local_chunk_graph_heads = int(local_chunk_graph_heads)
        self.local_chunk_graph_gate_init = float(local_chunk_graph_gate_init)
        self.local_chunk_graph_ffn_mult = int(local_chunk_graph_ffn_mult)
        self.local_chunk_graph_use_mask_embed = bool(local_chunk_graph_use_mask_embed)
        self.local_chunk_graph_out_proj_init_std = float(local_chunk_graph_out_proj_init_std)
        self.use_temporal_refiner = bool(use_temporal_refiner)
        self.temporal_refiner_dim = int(temporal_refiner_dim)
        self.temporal_refiner_heads = int(temporal_refiner_heads)
        self.temporal_refiner_gate_init = float(temporal_refiner_gate_init)
        self.temporal_refiner_fixed_gate = (
            None if temporal_refiner_fixed_gate is None else float(temporal_refiner_fixed_gate)
        )

        self.graph_target_dim = target_dim
        self.graph_aux_dim = aux_dim

        if self.use_token_graph_trunk:
            token_dim = self.token_graph_dim
            if token_dim is None:
                token_dim = int(n_graph_heads) * 64
            if token_dim % int(n_graph_heads) != 0:
                raise ValueError(
                    f"token_graph_dim={token_dim} must be divisible by n_graph_heads={n_graph_heads}"
                )
            self.token_graph_dim = token_dim
            self.n_input_graph_layers = 0
            self.input_graph_layers = None
            self.cross_modal_graph_layer = None
            self._has_parallel_gate = False
            self.parallel_gate = None
            self.gate_norm = None
            self.aux_gate_proj = None
            self.target_gate_proj = None

            self.token_target_embed = nn.Linear(window_size, token_dim)
            self.token_target_embed_norm = nn.LayerNorm(token_dim)
            self.token_shared_self_block = TokenGraphSelfBlock(
                d_model=token_dim,
                n_heads=n_graph_heads,
                dropout=dropout,
                ffn_mult=self.graph_ffn_mult,
            )
            self.token_branch_self_block = TokenGraphSelfBlock(
                d_model=token_dim,
                n_heads=n_graph_heads,
                dropout=dropout,
                ffn_mult=self.graph_ffn_mult,
            )

            self.token_aux_embed = None
            self.token_aux_embed_norm = None
            self.token_branch_cross_block = None
            if use_cross_modal_graph and target_dim is not None and aux_dim is not None and aux_dim > 0:
                self.token_aux_embed = nn.Linear(window_size, token_dim)
                self.token_aux_embed_norm = nn.LayerNorm(token_dim)
                self.token_branch_cross_block = TokenGraphCrossBlock(
                    d_model=token_dim,
                    n_heads=n_graph_heads,
                    dropout=dropout,
                    ffn_mult=self.graph_ffn_mult,
                )

            self.token_branch_gate_proj = nn.Linear(token_dim * 3, 1)
            nn.init.zeros_(self.token_branch_gate_proj.weight)
            nn.init.zeros_(self.token_branch_gate_proj.bias)
            self.token_fuse_norm = nn.LayerNorm(token_dim)

            self.token_out_proj = nn.Linear(token_dim, window_size)
            nn.init.zeros_(self.token_out_proj.weight)
            nn.init.zeros_(self.token_out_proj.bias)
            self.token_out_gate = nn.Parameter(torch.tensor(self.token_graph_out_gate_init))
            self.token_out_norm = nn.LayerNorm(window_size)
        else:
            # Target-Target Self-Attention (stacked for multi-hop relationships)
            self.n_input_graph_layers = n_input_graph_layers if use_input_graph_layer else 0
            if use_input_graph_layer and n_input_graph_layers > 0:
                n_graph_features = target_dim if target_dim is not None else input_dim
                self.input_graph_layers = nn.ModuleList([
                    InputGraphLayer(
                        n_features=n_graph_features,
                        window_size=window_size,
                        n_heads=n_graph_heads,
                        head_dim=64,
                        dropout=dropout,
                        use_temporal_cnn=use_temporal_cnn,
                        aux_dim=aux_dim if aux_dim is not None else 0,
                        n_chem=n_chem,
                        enable_cross_modal_floor=enable_cross_modal_floor,
                        disable_rel_scale=disable_rel_scale,
                        disable_prior_bias=disable_prior_bias,
                        disable_aux_bias=disable_aux_bias,
                        use_homogeneous=use_homogeneous,
                        use_ffn=self.use_graph_ffn,
                        ffn_mult=self.graph_ffn_mult,
                    )
                    for _ in range(n_input_graph_layers)
                ])
            else:
                self.input_graph_layers = None

            # Target-Aux Cross-Attention
            if use_cross_modal_graph and target_dim is not None and aux_dim is not None and aux_dim > 0:
                self.cross_modal_graph_layer = CrossModalGraphLayer(
                    target_dim=target_dim,
                    aux_dim=aux_dim,
                    window_size=window_size,
                    n_heads=n_graph_heads,
                    head_dim=64,
                    dropout=dropout,
                    use_temporal_cnn=use_temporal_cnn,
                    disable_aux_bias=disable_aux_bias,
                    use_ffn=self.use_graph_ffn,
                    ffn_mult=self.graph_ffn_mult,
                )
            else:
                self.cross_modal_graph_layer = None

            # Parallel gating: learned gate to fuse Self-Attn and Cross-Attn outputs
            # Only created when BOTH branches are active AND parallel mode is on
            self._has_parallel_gate = (
                use_parallel_graph
                and use_input_graph_layer and n_input_graph_layers > 0
                and use_cross_modal_graph and target_dim is not None and aux_dim is not None and aux_dim > 0
            )
            if self._has_parallel_gate:
                # Gate fusion for parallel mode.
                # Design: gate = Sigmoid(aux_gate_proj(x_aux_normed) + target_gate_proj(x_target_normed))
                #
                # aux_gate_proj  (4→262, full Conv1d):       weather regime switching
                # target_gate_proj (262→262, depthwise):     aerosol current state (missingness via mask_embed
                #                                            is already encoded in x_target_normed)
                # Both consume _normed features to BYPASS TCN dropout and guarantee stable gating.
                #
                # Both projections are initialized to 0 so gate = sigmoid(0) = 0.5 at start.
                # Direct Sigmoid gives valid gradient flow (no dead-w issue from Conv1d wrapper).
                self.parallel_gate = nn.Sigmoid()
                self.gate_norm = nn.LayerNorm(window_size)
                if aux_dim is not None and aux_dim > 0:
                    # aux_gate_proj: weather → species gate signal
                    self.aux_gate_proj = nn.Conv1d(aux_dim, target_dim, 1)
                    nn.init.zeros_(self.aux_gate_proj.weight)
                    nn.init.zeros_(self.aux_gate_proj.bias)
                    # target_gate_proj: aerosol state → species gate signal (depthwise = species-independent)
                    self.target_gate_proj = nn.Conv1d(target_dim, target_dim, 1, groups=target_dim)
                    nn.init.zeros_(self.target_gate_proj.weight)
                    nn.init.zeros_(self.target_gate_proj.bias)
                else:
                    self.aux_gate_proj = None
                    self.target_gate_proj = None
            else:
                self.parallel_gate = None
                self.gate_norm = None
                self.aux_gate_proj = None
                self.target_gate_proj = None
        
        # Unified Depthwise Temporal Feature Extractor
        # Calculates required depth to cover the window size.
        # Receptive field of TCN with kernel 3 and dilation 2^i is roughly 2^num_layers
        # 2^layers >= window_size -> layers >= log2(window_size)
        required_layers = int(np.ceil(np.log2(window_size)))
        
        if self.use_temporal_cnn and self.use_pregraph_depthwise_tcn and target_dim is not None:
            self.target_tcn = DepthwiseTCN(target_dim, num_layers=required_layers, kernel_size=3)
            self.aux_tcn = DepthwiseTCN(aux_dim, num_layers=required_layers, kernel_size=3) if aux_dim is not None and aux_dim > 0 else nn.Identity()
        else:
            self.target_tcn = nn.Identity()
            self.aux_tcn = nn.Identity()

        self.axial_observed_attn = None
        if self.use_axial_observed_attn and target_dim is not None:
            self.axial_observed_attn = AxialObservedAttentionBlock(
                n_features=target_dim,
                window_size=window_size,
                attn_dim=self.axial_attn_dim,
                n_heads=self.axial_attn_heads,
                dropout=dropout,
                time_gate_init=self.axial_time_gate_init,
                cross_gate_init=self.axial_cross_gate_init,
                cross_time_chunk=self.axial_cross_time_chunk,
                null_output=self.axial_null_output,
                n_chem=n_chem,
            )

        self.pregraph_feature_temporal_attn = None
        if self.use_pregraph_feature_temporal_attn and target_dim is not None:
            self.pregraph_feature_temporal_attn = PreGraphPerFeatureTemporalAttention(
                window_size=window_size,
                attn_dim=self.pregraph_feature_temporal_attn_dim,
                n_heads=self.pregraph_feature_temporal_attn_heads,
                gate_init=self.pregraph_feature_temporal_attn_gate_init,
                dropout=dropout,
                chunk_size=self.pregraph_feature_temporal_attn_chunk_size,
                record_weights=self.pregraph_feature_temporal_attn_record_weights,
                mode=self.pregraph_feature_temporal_attn_mode,
            )

        self.local_chunk_graph = None
        self.local_chunk_parallel_norm = None
        if self.use_local_chunk_graph and target_dim is not None:
            valid_local_chunk_modes = {'parallel', 'sequential_pre'}
            if self.local_chunk_graph_mode not in valid_local_chunk_modes:
                raise ValueError(
                    f"Unsupported local_chunk_graph_mode={self.local_chunk_graph_mode}; "
                    f"expected one of {sorted(valid_local_chunk_modes)}"
                )
            self.local_chunk_graph = LocalChunkGraphBranch(
                n_features=target_dim,
                window_size=window_size,
                chunk_size=self.local_chunk_graph_chunk_size,
                d_model=self.local_chunk_graph_dim,
                n_heads=self.local_chunk_graph_heads,
                dropout=dropout,
                gate_init=self.local_chunk_graph_gate_init,
                ffn_mult=self.local_chunk_graph_ffn_mult,
                use_mask_embed=self.local_chunk_graph_use_mask_embed,
                out_proj_init_std=self.local_chunk_graph_out_proj_init_std,
            )
            if self.local_chunk_graph_mode == 'parallel':
                self.local_chunk_parallel_norm = nn.LayerNorm(window_size)
            
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dims[0], 1)
        
        # TCN layers with residual connections
        self.tcn_layers = nn.ModuleList()
        self.tcn_downsample = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dims[min(i, len(hidden_dims)-1)]
            out_ch = hidden_dims[min(i+1, len(hidden_dims)-1)] if i < num_layers-1 else hidden_dims[-1]
            dilation = 2 ** i
            
            self.tcn_layers.append(
                nn.Sequential(
                    param.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, 
                                                padding=(kernel_size-1)*dilation//2, 
                                                dilation=dilation)),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
            
            # Downsample for residual connection if dimensions change
            if in_ch != out_ch:
                self.tcn_downsample.append(nn.Conv1d(in_ch, out_ch, 1))
            else:
                self.tcn_downsample.append(None)
        
        # Global pooling (mean) instead of attention pooling for simplicity
        # since we already did graph attention at input
        # self.avg_pool = nn.AdaptiveAvgPool1d(1) # [REPLACED]
        
        # Attention Pooling (New)
        self.attn_pool = TemporalAttentionPool(hidden_dims[-1])
        self.latent_pooled_norm = (
            nn.LayerNorm(hidden_dims[-1]) if self.use_latent_pooled_norm else nn.Identity()
        )
        
        # Latent projection (mu and logvar)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.temporal_refiner = None
        if self.use_temporal_refiner:
            self.temporal_refiner = TemporalObservationRefiner(
                hidden_dim=hidden_dims[-1],
                window_size=window_size,
                attn_dim=self.temporal_refiner_dim,
                n_heads=self.temporal_refiner_heads,
                gate_init=self.temporal_refiner_gate_init,
                fixed_gate=self.temporal_refiner_fixed_gate,
                obs_bias_init=1.0,
                dropout=dropout,
            )
        self.local_context_proj = None
        self.local_context_pool = None
        if self.use_local_context_map:
            local_steps = self.local_context_steps if self.local_context_steps is not None else 12
            local_steps = max(1, min(window_size, local_steps))
            self.local_context_steps = local_steps
            self.local_context_proj = nn.Sequential(
                nn.Conv1d(hidden_dims[-1], self.local_context_dim, 1),
                nn.GELU(),
            )
            self.local_context_pool = nn.AdaptiveAvgPool1d(self.local_context_steps)
        
        self.last_input_graph_attention = None
        self.last_input_graph_attention_heads = None
        self.last_input_graph_attention_batch = None
        self.last_input_graph_attention_heads_batch = None
        self.last_input_graph_attention_per_layer = []  # Store attention from each stacked layer
        self.last_cross_modal_attention = None
        self.last_cross_modal_attention_heads = None
        self.last_cross_modal_attention_batch = None
        self.last_cross_modal_attention_heads_batch = None
        self.last_parallel_gate = None
        self.last_pregraph_feature_temporal_attn_gate = None
        self.last_pregraph_feature_temporal_attn_entropy_missing = None
        self.last_pregraph_feature_temporal_attn_entropy_observed = None
        self.last_axial_time_gate_mean = None
        self.last_axial_time_gate_chem_mean = None
        self.last_axial_time_gate_psd_mean = None
        self.last_axial_cross_gate_mean = None
        self.last_axial_cross_gate_chem_mean = None
        self.last_axial_cross_gate_psd_mean = None
        self.last_axial_time_no_key_fraction = None
        self.last_axial_cross_no_key_fraction = None
        self.last_axial_cross_valid_query_fraction = None
        self.last_axial_cross_entropy_missing = None
        self.last_axial_cross_top1_mass = None
        self.last_axial_cross_top3_mass = None
        self.last_axial_psd_to_chem_mass = None
        self.last_axial_psd_to_psd_mass = None
        self.last_local_chunk_graph_gate = None
        self.last_local_chunk_graph_out_proj_norm = None
        self.last_local_chunk_graph_obs_ratio_mean = None
        self.last_temporal_refiner_gate = None
        self.last_temporal_refiner_attn_entropy_missing = None
        self.last_temporal_refiner_attn_entropy_observed = None
        self.last_local_context_attn_entropy = None
        self.last_local_context_attn_center_distance = None
        self.last_local_context_attn_support_mean = None
        self.last_local_context_attn_high_support_mass = None
        self.last_local_context_generation_support_mean = None
        self.last_local_context_observe_aware_blend_gate = None
        self.last_local_context_gate_low_support_mean = None
        self.last_local_context_gate_high_support_mean = None
    
    def forward(self, x, obs_mask=None, embed_offset=None):
        """
        Args:
            x: [Batch, Input_dim, Window]
            obs_mask: [Batch, Window, Input_dim] - observation mask (1=observed, 0=missing)
            embed_offset: [Batch, Target_dim, Window] - learned missingness embeddings
            
        Returns:
            mu: [Batch, Latent_dim]
            logvar: [Batch, Latent_dim]
            attn_weights: [Batch, C, C] - Learned input feature relationships
        """
        # 1. Input-level graph learning (BEFORE TCN)
        # Apply graph only on target features to avoid mixing mask/aux channels.
        attn_weights = None
        cross_attn = None
        
        # Extract dimensions
        if self.graph_target_dim is None or self.graph_target_dim >= x.shape[1]:
            # Fallback: treat all as target
            x_target = x
            x_aux = None
            target_mask = obs_mask
        else:
            # Split input: [target, aux] (mask channels excluded from input)
            x_target = x[:, :self.graph_target_dim, :]
            target_mask = obs_mask[:, :, :self.graph_target_dim] if obs_mask is not None else None
            
            # Aux comes after target
            if self.graph_aux_dim is not None and self.graph_aux_dim > 0:
                aux_start = self.graph_target_dim
                aux_end = self.graph_target_dim + self.graph_aux_dim
                x_aux = x[:, aux_start:aux_end, :]
            else:
                x_aux = None
                
        # ================================================================
        # Pre-Graph Data Processing & Multi-Scale Temporal Extraction
        # ================================================================
        x_processed_list = []
        if self.graph_target_dim is not None:
            # Transpose target_mask to [B, C, W] to match x_target
            target_mask_t = target_mask.permute(0, 2, 1).float() if target_mask is not None else torch.ones_like(x_target)
            
            # Masked normalization for target features
            x_target_masked = x_target * target_mask_t
            obs_count = target_mask_t.sum(dim=2, keepdim=True).clamp(min=1)
            x_target_mean = x_target_masked.sum(dim=2, keepdim=True) / obs_count
            x_target_centered = (x_target - x_target_mean) * target_mask_t
            x_target_var = (x_target_centered ** 2).sum(dim=2, keepdim=True) / obs_count.clamp(min=2)
            x_target_std = torch.sqrt(x_target_var + 1e-8)
            x_target_normed = (x_target_centered / x_target_std) * target_mask_t
            
            # --- CRITICAL BUG FIX ---
            # Add mask embedding AFTER instance normalization.
            # Previously it was added before normalization, meaning `x_target_centered` cancelled out embed_1,
            # and the `* target_mask_t` zeroed out embed_0. So missingness information was mathematically deleted!
            if embed_offset is not None:
                x_target_normed = x_target_normed + embed_offset
            
            # Extract depthwise unmixed trends uniformly
            x_target_processed = self.target_tcn(x_target_normed)
            if self.axial_observed_attn is not None:
                x_target_processed = self.axial_observed_attn(
                    x_target_processed,
                    target_obs_mask=target_mask if not self.ignore_obs_mask else None,
                )
                self.last_pregraph_feature_temporal_attn_gate = None
                self.last_pregraph_feature_temporal_attn_entropy_missing = None
                self.last_pregraph_feature_temporal_attn_entropy_observed = None
                self.last_axial_time_gate_mean = self.axial_observed_attn.last_time_gate_mean
                self.last_axial_time_gate_chem_mean = self.axial_observed_attn.last_time_gate_chem_mean
                self.last_axial_time_gate_psd_mean = self.axial_observed_attn.last_time_gate_psd_mean
                self.last_axial_cross_gate_mean = self.axial_observed_attn.last_cross_gate_mean
                self.last_axial_cross_gate_chem_mean = self.axial_observed_attn.last_cross_gate_chem_mean
                self.last_axial_cross_gate_psd_mean = self.axial_observed_attn.last_cross_gate_psd_mean
                self.last_axial_time_no_key_fraction = self.axial_observed_attn.last_time_no_key_fraction
                self.last_axial_cross_no_key_fraction = self.axial_observed_attn.last_cross_no_key_fraction
                self.last_axial_cross_valid_query_fraction = self.axial_observed_attn.last_cross_valid_query_fraction
                self.last_axial_cross_entropy_missing = self.axial_observed_attn.last_cross_entropy_missing
                self.last_axial_cross_top1_mass = self.axial_observed_attn.last_cross_top1_mass
                self.last_axial_cross_top3_mass = self.axial_observed_attn.last_cross_top3_mass
                self.last_axial_psd_to_chem_mass = self.axial_observed_attn.last_psd_to_chem_mass
                self.last_axial_psd_to_psd_mass = self.axial_observed_attn.last_psd_to_psd_mass
            elif self.pregraph_feature_temporal_attn is not None:
                x_target_processed = self.pregraph_feature_temporal_attn(
                    x_target_processed,
                    target_obs_mask=target_mask if not self.ignore_obs_mask else None,
                )
                self.last_pregraph_feature_temporal_attn_gate = (
                    self.pregraph_feature_temporal_attn.last_gate
                )
                self.last_pregraph_feature_temporal_attn_entropy_missing = (
                    self.pregraph_feature_temporal_attn.last_missing_query_attn_entropy
                )
                self.last_pregraph_feature_temporal_attn_entropy_observed = (
                    self.pregraph_feature_temporal_attn.last_observed_query_attn_entropy
                )
                self.last_axial_time_gate_mean = None
                self.last_axial_time_gate_chem_mean = None
                self.last_axial_time_gate_psd_mean = None
                self.last_axial_cross_gate_mean = None
                self.last_axial_cross_gate_chem_mean = None
                self.last_axial_cross_gate_psd_mean = None
                self.last_axial_time_no_key_fraction = None
                self.last_axial_cross_no_key_fraction = None
                self.last_axial_cross_valid_query_fraction = None
                self.last_axial_cross_entropy_missing = None
                self.last_axial_cross_top1_mass = None
                self.last_axial_cross_top3_mass = None
                self.last_axial_psd_to_chem_mass = None
                self.last_axial_psd_to_psd_mass = None
            else:
                self.last_pregraph_feature_temporal_attn_gate = None
                self.last_pregraph_feature_temporal_attn_entropy_missing = None
                self.last_pregraph_feature_temporal_attn_entropy_observed = None
                self.last_axial_time_gate_mean = None
                self.last_axial_time_gate_chem_mean = None
                self.last_axial_time_gate_psd_mean = None
                self.last_axial_cross_gate_mean = None
                self.last_axial_cross_gate_chem_mean = None
                self.last_axial_cross_gate_psd_mean = None
                self.last_axial_time_no_key_fraction = None
                self.last_axial_cross_no_key_fraction = None
                self.last_axial_cross_valid_query_fraction = None
                self.last_axial_cross_entropy_missing = None
                self.last_axial_cross_top1_mass = None
                self.last_axial_cross_top3_mass = None
                self.last_axial_psd_to_chem_mass = None
                self.last_axial_psd_to_psd_mass = None
            x_processed_list.append(x_target_processed)
            
            # Normalization and extraction for aux
            if x_aux is not None:
                x_aux_mean = x_aux.mean(dim=2, keepdim=True)
                x_aux_std = x_aux.std(dim=2, keepdim=True) + 1e-8
                x_aux_normed = (x_aux - x_aux_mean) / x_aux_std
                
                feat_aux = self.aux_tcn(x_aux_normed)
                # Keep separate for cross-modal routing
                x_aux_processed = feat_aux
                x_processed_list.append(feat_aux)
            else:
                x_aux_processed = None
        else:
            x_target_processed = x_target
            x_aux_processed = x_aux
            self.last_pregraph_feature_temporal_attn_gate = None
            self.last_pregraph_feature_temporal_attn_entropy_missing = None
            self.last_pregraph_feature_temporal_attn_entropy_observed = None
            self.last_axial_time_gate_mean = None
            self.last_axial_time_gate_chem_mean = None
            self.last_axial_time_gate_psd_mean = None
            self.last_axial_cross_gate_mean = None
            self.last_axial_cross_gate_chem_mean = None
            self.last_axial_cross_gate_psd_mean = None
            self.last_axial_time_no_key_fraction = None
            self.last_axial_cross_no_key_fraction = None
            self.last_axial_cross_valid_query_fraction = None
            self.last_axial_cross_entropy_missing = None
            self.last_axial_cross_top1_mass = None
            self.last_axial_cross_top3_mass = None
            self.last_axial_psd_to_chem_mass = None
            self.last_axial_psd_to_psd_mass = None
            
        # Re-concatenate for final bypass (e.g. into standard TCN layers) if needed
        if len(x_processed_list) > 0:
            x_flat_processed = torch.cat(x_processed_list, dim=1)
        else:
            x_flat_processed = x

        x_graph_base = x_target_processed
        x_local_chunk = None
        self.last_local_chunk_graph_gate = None
        self.last_local_chunk_graph_out_proj_norm = None
        self.last_local_chunk_graph_obs_ratio_mean = None
        if self.local_chunk_graph is not None:
            x_local_chunk = self.local_chunk_graph(
                x_target_processed,
                chunk_obs_mask=target_mask_t,
            )
            self.last_local_chunk_graph_gate = self.local_chunk_graph.last_gate
            self.last_local_chunk_graph_out_proj_norm = (
                self.local_chunk_graph.last_out_proj_weight_norm
            )
            self.last_local_chunk_graph_obs_ratio_mean = (
                self.local_chunk_graph.last_obs_ratio_mean
            )
            if self.local_chunk_graph_mode == 'sequential_pre':
                x_graph_base = x_local_chunk

        # Graph Attention: old parallel/sequential W-space path OR new d_model token-graph trunk
        # ================================================================
        if self.use_token_graph_trunk:
            target_tokens = self.token_target_embed_norm(self.token_target_embed(x_graph_base))

            shared_tokens, shared_attn = self.token_shared_self_block(target_tokens, need_weights=True)
            self_tokens, self_attn = self.token_branch_self_block(shared_tokens, need_weights=True)

            self.last_input_graph_attention_per_layer = []
            for block, attn in (
                (self.token_shared_self_block, shared_attn),
                (self.token_branch_self_block, self_attn),
            ):
                self.last_input_graph_attention_per_layer.append({
                    'avg': attn.detach().mean(dim=0) if attn is not None else None,
                    'batch': attn.detach() if attn is not None else None,
                    'heads': block.last_attention_weights_heads,
                    'heads_batch': block.last_attention_weights_heads_batch,
                })

            attn_weights = self_attn if self_attn is not None else shared_attn
            self.last_input_graph_attention = (
                attn_weights.detach().mean(dim=0) if attn_weights is not None else None
            )
            self.last_input_graph_attention_batch = (
                attn_weights.detach() if attn_weights is not None else None
            )
            self.last_input_graph_attention_heads = self.token_branch_self_block.last_attention_weights_heads
            self.last_input_graph_attention_heads_batch = (
                self.token_branch_self_block.last_attention_weights_heads_batch
            )

            x_cross_tokens = None
            cross_attn = None
            self.last_cross_modal_attention = None
            self.last_cross_modal_attention_batch = None
            self.last_cross_modal_attention_heads = None
            self.last_cross_modal_attention_heads_batch = None
            self.last_parallel_gate = None

            if self.token_branch_cross_block is not None and x_aux_processed is not None:
                aux_tokens = self.token_aux_embed_norm(self.token_aux_embed(x_aux_processed))
                x_cross_tokens, cross_attn = self.token_branch_cross_block(
                    shared_tokens, aux_tokens, need_weights=True
                )
                self.last_cross_modal_attention = cross_attn.detach().mean(dim=0)
                self.last_cross_modal_attention_batch = cross_attn.detach()
                self.last_cross_modal_attention_heads = (
                    self.token_branch_cross_block.last_attention_weights_heads
                )
                self.last_cross_modal_attention_heads_batch = (
                    self.token_branch_cross_block.last_attention_weights_heads_batch
                )

                gate_input = torch.cat([shared_tokens, self_tokens, x_cross_tokens], dim=-1)
                gate = torch.sigmoid(self.token_branch_gate_proj(gate_input))  # [B, C, 1]
                fused_tokens = self.token_fuse_norm(
                    shared_tokens
                    + gate * (self_tokens - shared_tokens)
                    + (1.0 - gate) * (x_cross_tokens - shared_tokens)
                )
                self.last_parallel_gate = gate.detach().mean(dim=(0, 2))  # [C]
            else:
                fused_tokens = self_tokens

            delta_w = self.token_out_proj(fused_tokens)
            out_gate = torch.sigmoid(self.token_out_gate)
            x_target_enhanced = self.token_out_norm(
                x_graph_base + out_gate * delta_w
            )
        else:
            has_self = self.input_graph_layers is not None
            has_cross = self.cross_modal_graph_layer is not None and x_aux is not None

            # --- Self-Attention branch ---
            x_self = None
            if has_self:
                x_self_input = x_graph_base  # Start from graph-stage base representation
                self.last_input_graph_attention_per_layer = []
                for layer_idx, graph_layer in enumerate(self.input_graph_layers):
                    # Pass target_mask to ALL layers so pairwise threshold remains valid
                    # in every hop (previously only layer 0 received the mask).
                    x_self_input, attn_weights = graph_layer(x_self_input, obs_mask=target_mask, x_aux=x_aux_processed)
                    self.last_input_graph_attention_per_layer.append({
                        'avg': attn_weights.detach().mean(dim=0),
                        'batch': attn_weights.detach(),
                        'heads': graph_layer.last_attention_weights_heads if hasattr(graph_layer, 'last_attention_weights_heads') else None,
                        'heads_batch': graph_layer.last_attention_weights_heads_batch if hasattr(graph_layer, 'last_attention_weights_heads_batch') else None,
                    })
                last_layer = self.input_graph_layers[-1]
                self.last_input_graph_attention = attn_weights.detach().mean(dim=0)
                self.last_input_graph_attention_batch = attn_weights.detach()
                if hasattr(last_layer, 'last_attention_weights_heads'):
                    self.last_input_graph_attention_heads = last_layer.last_attention_weights_heads
                if hasattr(last_layer, 'last_attention_weights_heads_batch'):
                    self.last_input_graph_attention_heads_batch = last_layer.last_attention_weights_heads_batch
                x_self = x_self_input  # post-LN output

            # --- Cross-Attention branch ---
            # In parallel mode: query = x_target_processed (independent of self branch)
            # In sequential mode: query = x_self (self-attn output feeds cross-attn)
            # NOTE: we run cross_modal_graph_layer ONCE here regardless of mode,
            # choosing the correct query upfront to avoid the previous double-call bug.
            x_cross = None
            if has_cross:
                if self._has_parallel_gate and has_self:
                    # Parallel: both branches start from preprocessed independently
                    cross_query = x_graph_base
                else:
                    # Sequential: cross-modal receives self-attn output as query
                    cross_query = x_self if x_self is not None else x_graph_base

                x_cross_output, cross_attn = self.cross_modal_graph_layer(
                    cross_query, x_aux_processed, target_mask
                )
                self.last_cross_modal_attention = cross_attn.detach().mean(dim=0)
                self.last_cross_modal_attention_batch = cross_attn.detach()
                if hasattr(self.cross_modal_graph_layer, 'last_attention_weights_heads'):
                    self.last_cross_modal_attention_heads = self.cross_modal_graph_layer.last_attention_weights_heads
                if hasattr(self.cross_modal_graph_layer, 'last_attention_weights_heads_batch'):
                    self.last_cross_modal_attention_heads_batch = self.cross_modal_graph_layer.last_attention_weights_heads_batch
                x_cross = x_cross_output  # post-LN output

            # --- Fusion ---
            if self._has_parallel_gate and has_self and has_cross:
                delta_self  = x_self  - x_graph_base  # [B, D, W]
                delta_cross = x_cross - x_graph_base  # [B, D, W]

                # --- Atmosphere-physics gate ---
                # gate = Sigmoid(
                #   aux_gate_proj(x_aux)           ← weather regime (direct physical forcing vs thermodynamics)
                #   + target_gate_proj(x_target)   ← aerosol state encoded via mask_embed + TCN
                # )
                # No explicit x_self * mask: value-dependent gating eliminated.
                # Missingness is already in x_target_processed via embed_offset (mask embedding).
                # Both projections init to 0 → gate = 0.5 at start → unbiased equal weighting.
                if (hasattr(self, 'aux_gate_proj') and self.aux_gate_proj is not None
                        and x_aux_normed is not None):
                    # Use raw _normed features to bypass TCN dropout!
                    # If we use _processed, the 10% TCN dropout randomly zeroes features,
                    # causing the gate to wildly flip between 0 and 1 during MC Dropout inference,
                    # which explodes the Epistemic Variance (CRPS degradation).
                    aux_signal    = self.aux_gate_proj(x_aux_normed)          # [B, D, W] stable weather signal
                    target_signal = self.target_gate_proj(x_target_normed)    # [B, D, W] stable aerosol state signal
                    gate_input = aux_signal + target_signal
                else:
                    # Fallback: unbiased (sigmoid(0) = 0.5 for all species)
                    gate_input = torch.zeros_like(x_graph_base)

                gate = self.parallel_gate(gate_input)  # Sigmoid → [B, D, W]
                x_target_enhanced = self.gate_norm(
                    x_graph_base + gate * delta_self + (1 - gate) * delta_cross
                )
                # Store gate values for interpretability
                self.last_parallel_gate = gate.detach().mean(dim=(0, 2))  # [D] avg gate per feature
            elif has_self and has_cross:
                # Sequential: x_cross already used x_self as query; use its output directly
                x_target_enhanced = x_cross
            elif has_self:
                x_target_enhanced = x_self
            elif has_cross:
                x_target_enhanced = x_cross
            else:
                x_target_enhanced = x_graph_base

        if x_local_chunk is not None and self.local_chunk_graph_mode == 'parallel':
            local_delta = x_local_chunk - x_target_processed
            if self.local_chunk_parallel_norm is not None:
                x_target_enhanced = self.local_chunk_parallel_norm(
                    x_target_enhanced + local_delta
                )
            else:
                x_target_enhanced = x_target_enhanced + local_delta
        
        # Reconstruct full feature tensor: [Enhanced Target, Aux]
        final_features = []
        if self.graph_target_dim is not None and x_target_enhanced is not None:
            final_features.append(x_target_enhanced)
            if x_aux_processed is not None:
                final_features.append(x_aux_processed)
            
            x_fused = torch.cat(final_features, dim=1)
        else:
            x_fused = x_flat_processed
            
        # Overwrite x so the downstream network captures the processed graphs
        x = x_fused
        
        # 2. Input projection
        h = self.input_proj(x)  # [B, hidden, W]
        
        # 3. TCN layers with residual connections (no norm — preserves scale)
        if self.use_tcn:
            for layer, downsample in zip(self.tcn_layers, self.tcn_downsample):
                h_residual = h
                h = layer(h)
                if downsample is not None:
                    h_residual = downsample(h_residual)
                h = h + h_residual
        
        h_base = h
        h_refined = h_base
        if self.temporal_refiner is not None:
            h_refined = self.temporal_refiner(
                h_base,
                target_obs_mask=target_mask if not self.ignore_obs_mask else None
            )
            self.last_temporal_refiner_gate = self.temporal_refiner.last_gate
            self.last_temporal_refiner_attn_entropy_missing = (
                self.temporal_refiner.last_missing_query_attn_entropy
            )
            self.last_temporal_refiner_attn_entropy_observed = (
                self.temporal_refiner.last_observed_query_attn_entropy
            )
        else:
            self.last_temporal_refiner_gate = None
            self.last_temporal_refiner_attn_entropy_missing = None
            self.last_temporal_refiner_attn_entropy_observed = None

        # 4. Pooling (observation-aware: suppress fully-missing timesteps)
        h_pooled = self.attn_pool(h_base, obs_mask=target_mask if not self.ignore_obs_mask else None)
        h_pooled = self.latent_pooled_norm(h_pooled)
        
        # 5. Latent projection
        mu = self.fc_mu(h_pooled)
        logvar = self.fc_logvar(h_pooled)
        if self.latent_logvar_min is not None or self.latent_logvar_max is not None:
            logvar = torch.clamp(logvar, min=self.latent_logvar_min, max=self.latent_logvar_max)
        
        local_context = None
        if self.use_local_context_map:
            local_features = self.local_context_proj(h_refined)
            local_context_raw = self.local_context_pool(local_features)
            self.last_local_context_generation_support_mean = None
            self.last_local_context_observe_aware_blend_gate = None
            if (
                self.local_context_observe_aware
                and target_mask is not None
                and not self.ignore_obs_mask
            ):
                support = target_mask.to(local_features.dtype).mean(dim=-1).unsqueeze(1)
                support_pooled = self.local_context_pool(support)
                weighted = self.local_context_pool(local_features * support)
                local_context_weighted = weighted / support_pooled.clamp_min(1e-6)
                local_context = torch.where(
                    support_pooled > 1e-6,
                    local_context_weighted,
                    local_context_raw,
                )
                if self.local_context_observe_aware_blend_gate is not None:
                    blend_gate = torch.sigmoid(
                        self.local_context_observe_aware_blend_gate
                    ).to(local_context_raw.dtype)
                    local_context = local_context_raw + blend_gate * (
                        local_context - local_context_raw
                    )
                    self.last_local_context_observe_aware_blend_gate = float(
                        blend_gate.detach().item()
                    )
                self.last_local_context_generation_support_mean = float(
                    support_pooled.detach().mean().item()
                )
            else:
                local_context = local_context_raw

        # P0: Attention-weighted cross-feature support map.
        # For each target feature f and timestep t, compute the sum of
        # attn_weight(f'→f) × obs_mask[f', t] over all source features f'.
        # This gives a per-feature, per-timestep "importance-weighted support"
        # signal: high when the features that f most attends to are observed.
        # attn_weights shape varies by path:
        #   InputGraphLayer (homogeneous): [B, C, C]  — heads already averaged
        #   Token graph trunk:             [B, h, C, C] — heads not yet averaged
        # target_mask: [B, W, C].
        # Result:      [B, W, C], values in [0, 1] (softmax-normalized weights).
        # Detach prevents variance-path NLL from backpropping into encoder attention.
        attn_weighted_support_t = None
        if attn_weights is not None and target_mask is not None:
            aw = attn_weights.detach()
            if aw.dim() == 4:
                attn_w_avg = aw.mean(dim=1)  # [B, h, C, C] → [B, C, C]
            else:
                attn_w_avg = aw              # [B, C, C] already
            # attn_w_avg[b, f_tgt, f_src]: how much f_tgt attends to f_src
            attn_weighted_support_t = torch.einsum(
                'bfc,btc->btf', attn_w_avg, target_mask.float()
            )  # [B, W, C]

        return mu, logvar, attn_weights, h_base, local_context, attn_weighted_support_t


class MaskedTemporalCrossAttention(nn.Module):
    """
    Decoder Cross-Attention module to allow the Decoder to query local temporal 
    anchors from the Encoder's un-pooled hidden sequence. Uses the combined 
    observational mask (where True means the value is entirely missing) as a 
    KeyPaddingMask to prevent Posterior Collapse.
    """
    def __init__(self, dec_dim, enc_dim=None, n_heads=4, dropout=0.1):
        super().__init__()
        enc_dim = enc_dim if enc_dim is not None else dec_dim
        self.k_proj = nn.Conv1d(enc_dim, dec_dim, 1)
        self.v_proj = nn.Conv1d(enc_dim, dec_dim, 1)
        self.attn = nn.MultiheadAttention(dec_dim, n_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(dec_dim)

    def forward(self, h_dec, h_enc, target_obs_mask):
        """
        Args:
            h_dec: [B, dec_dim, W]
            h_enc: [B, enc_dim, W]
            target_obs_mask: [B, W, target_dim] where True generally means missing.
        """
        B, _, W = h_dec.shape
        Q = h_dec.permute(0, 2, 1)
        K = self.k_proj(h_enc).permute(0, 2, 1)
        V = self.v_proj(h_enc).permute(0, 2, 1)

        # key_pad_mask should be True for items that are ignored (missing).
        # When ignore_obs_mask is enabled (mask=None), we attend to everything.
        if target_obs_mask is None:
            key_pad_mask = None
        elif target_obs_mask.dim() == 3:
            # Key padding mask: True for items to be ignored (missing).
            # When target_obs_mask is [B, W, D], we check if all features are missing at time t.
            key_pad_mask = (target_obs_mask == 0).all(dim=-1)  # [B, W]
        else:
            key_pad_mask = (target_obs_mask == 0)
            
        # Fallback to unmasked if all missing (or if mask is None) to avoid NaNs
        if key_pad_mask is not None:
            all_masked = key_pad_mask.all(dim=-1, keepdim=True) 
            key_pad_mask = key_pad_mask.masked_fill(all_masked, False) 

        out, _ = self.attn(Q, K, V, key_padding_mask=key_pad_mask)
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
            raise ValueError(f"dec_dim={dec_dim} must be divisible by n_heads={n_heads}")

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
        mapped = torch.div(q_pos * kv_len, q_len, rounding_mode='floor')
        allowed = (kv_pos.unsqueeze(0) - mapped.unsqueeze(1)).abs() <= self.window_tokens
        return ~allowed

    def forward(self, h_dec, local_ctx, support_tokens, support_high):
        """
        Args:
            h_dec: [B, hidden, W]
            local_ctx: [B, ctx_dim, S]
            support_tokens: [B, 1, S]
            support_high: [B, 1, W]
        Returns:
            delta: [B, hidden, W]
            gate: [B, 1, W]
        """
        B, _, q_len = h_dec.shape
        _, _, kv_len = local_ctx.shape

        q = self.q_proj(h_dec).transpose(1, 2).reshape(B, q_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(local_ctx).transpose(1, 2).reshape(B, kv_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(local_ctx).transpose(1, 2).reshape(B, kv_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        local_mask = self._local_window_mask(q_len, kv_len, device=h_dec.device)
        scores = scores.masked_fill(local_mask.view(1, 1, q_len, kv_len), -1e4)

        if support_tokens is not None:
            support_bias = self.support_bias_scale * (support_tokens.clamp(0.0, 1.0) - 0.5) * 2.0
            scores = scores + support_bias.unsqueeze(1)

        attn = torch.softmax(scores, dim=-1)
        attn_summary = attn.detach()
        attn_mean = attn_summary.mean(dim=1)  # [B, W, S]
        entropy = -(attn_summary * (attn_summary.clamp_min(1e-8).log())).sum(dim=-1)
        q_pos = torch.arange(q_len, device=h_dec.device)
        kv_pos = torch.arange(kv_len, device=h_dec.device)
        mapped = torch.div(q_pos * kv_len, q_len, rounding_mode='floor')
        distance = (kv_pos.unsqueeze(0) - mapped.unsqueeze(1)).abs().to(attn_summary.dtype)
        self.last_attn_entropy = float(entropy.mean().item())
        self.last_attn_center_distance = float((attn_mean * distance.unsqueeze(0)).sum(dim=-1).mean().item())
        if support_tokens is not None:
            support = support_tokens.detach().clamp(0.0, 1.0).squeeze(1)
            self.last_attn_support_mean = float((attn_mean * support.unsqueeze(1)).sum(dim=-1).mean().item())
            high_support = (support >= 0.75).to(attn_mean.dtype)
            self.last_attn_high_support_mass = float((attn_mean * high_support.unsqueeze(1)).sum(dim=-1).mean().item())
        else:
            self.last_attn_support_mean = None
            self.last_attn_high_support_mass = None
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, q_len, self.dec_dim).transpose(1, 2)
        out = self.out_proj(out)
        out = self.out_norm(out.transpose(1, 2)).transpose(1, 2)

        if support_high is None:
            support_high = torch.ones(B, 1, q_len, device=h_dec.device, dtype=h_dec.dtype)
        gate_in = torch.cat([h_dec, support_high], dim=1)
        gate = torch.sigmoid(self.gate_proj(gate_in))
        if self.gate_support_power > 0.0:
            support_gate = support_high.clamp(0.0, 1.0).pow(self.gate_support_power)
            if self.gate_support_floor > 0.0:
                support_gate = self.gate_support_floor + (1.0 - self.gate_support_floor) * support_gate
            gate = gate * support_gate
        gate_det = gate.detach()
        self.last_gate_mean = float(gate_det.mean().item())
        if support_high is not None:
            support_det = support_high.detach()
            low = support_det <= 0.50
            high = support_det >= 0.90
            self.last_gate_low_support_mean = (
                float(gate_det[low].mean().item()) if low.any() else None
            )
            self.last_gate_high_support_mean = (
                float(gate_det[high].mean().item()) if high.any() else None
            )
        else:
            self.last_gate_low_support_mean = None
            self.last_gate_high_support_mean = None
        return gate * out, gate


# ==============================================================================
# Graph-Enhanced Decoder
# ==============================================================================
class GraphDecoder(nn.Module):
    """
    TCN Decoder with optional progressive upsampling, decoder cross-attention,
    and FiLM conditioning.
    Supports separate variance projections for different feature groups.
    
    Progressive mode (use_progressive_decoder=True):
      - Compact latent projection: latent → hidden × initial_steps
      - Upsample blocks: initial_steps → ... → window_size
      - Optional decoder cross-attention + learned fusion before TCN refinement
      - FiLM conditioning at every TCN layer from cond and z
    
    Legacy mode (use_progressive_decoder=False):
      - One-shot latent projection: latent → hidden × window_size
      - Single additive condition injection before TCN
    """
    
    def __init__(self, latent_dim, aux_dim, hidden_dims, output_dim, window_size,
                 num_layers=5, kernel_size=3, dropout=0.1, heteroscedastic=True,
                 n_chem=0, use_tcn=True, use_progressive_decoder=False,
                 decoder_initial_steps=12,
                 cond_film_last_n=None,
                 cond_film_gamma_scale=0.5,
                 use_decoder_cross_attn=False, n_cross_attn_heads=4,
                 decoder_cross_attn_missing_only=False,
                 var_min=1e-3, var_max=10.0, film_kernel_size=1,
                 film_gamma_kernel_size=None, film_beta_kernel_size=None,
                 film_temporal_last_n=0, film_temporal_last_kernel_size=3,
                 z_film_alpha_init=-2.0, z_skip_gate_init=-2.0,
                 use_z_skip=True,
                 decoder_cross_attn_gate_init=-1.5,
                 use_dual_output_heads=False, output_head_hidden_dim=None,
                 use_detached_variance_pathway=False,
                 variance_detach_start_epoch=None,
                 variance_path_use_latent=True,
                 variance_path_detach_latent=False,
                 variance_head_hidden_dim=None,
                 variance_path_use_mask=False,
                 variance_mask_dim=32,
                 variance_use_grouped_conv=False,
                 use_local_context_map=False,
                 local_context_dim=32,
                 local_context_steps=None,
                 local_context_gate_init=-2.0,
                 local_context_observe_aware=False,
                 local_context_injection_mode='seed',
                 local_context_fusion_mode='add',
                 local_context_attn_heads=4,
                 local_context_attn_window_tokens=1,
                 local_context_attn_gate_init=-2.0,
                 local_context_attn_after_tcn_layers=None,
                 local_context_attn_location='mid_tcn',
                 local_context_attn_support_bias_scale=2.0,
                 local_context_attn_gate_support_power=0.0,
                 local_context_attn_gate_support_floor=0.0,
                 local_context_attn_logvar_support_boost=0.0,
                 use_variance_attn_support=False,
                 variance_attn_support_n_features=0,
                 variance_attn_support_dim=32,
                 use_support_logvar_residual=False,
                 support_logvar_hidden_dim=32,
                 support_logvar_missing_only=True,
                 support_logvar_monotone=False,
                 support_logvar_monotone_init=-2.0,
                 support_logvar_use_anchor=False,
                 support_logvar_anchor_init=-2.0,
                 use_feature_logvar_bias=False,
                 feature_logvar_bias_scope='psd',
                 feature_logvar_bias_init=0.0,
                 feature_logvar_bias_constraint='none',
                 use_decoder_final_norm=False,
                 ignore_obs_mask=False):
        super().__init__()

        self.window_size = window_size
        self.output_dim = output_dim
        self.heteroscedastic = heteroscedastic
        self.n_chem = n_chem
        self.n_psd = output_dim - n_chem
        self.use_tcn = use_tcn
        self.use_progressive_decoder = use_progressive_decoder
        self.use_decoder_cross_attn = use_decoder_cross_attn
        self.decoder_cross_attn_missing_only = decoder_cross_attn_missing_only
        self.ignore_obs_mask = ignore_obs_mask
        self.cond_film_gamma_scale = float(cond_film_gamma_scale)
        self.var_min = var_min
        self.var_max = var_max
        self.film_kernel_size = film_kernel_size
        self.film_gamma_kernel_size = film_gamma_kernel_size if film_gamma_kernel_size is not None else film_kernel_size
        self.film_beta_kernel_size = film_beta_kernel_size if film_beta_kernel_size is not None else film_kernel_size
        self.film_temporal_last_n = max(0, int(film_temporal_last_n))
        self.film_temporal_last_kernel_size = int(film_temporal_last_kernel_size)
        self.use_dual_output_heads = use_dual_output_heads
        self.use_detached_variance_pathway = bool(use_detached_variance_pathway)
        self.variance_detach_start_epoch = (
            None if variance_detach_start_epoch is None else int(variance_detach_start_epoch)
        )
        self.use_variance_attn_support = bool(use_variance_attn_support)
        self.variance_attn_support_dim = int(variance_attn_support_dim)
        self.variance_attn_support_n_features = int(variance_attn_support_n_features)
        self.use_support_logvar_residual = bool(use_support_logvar_residual)
        self.support_logvar_hidden_dim = max(4, int(support_logvar_hidden_dim))
        self.support_logvar_missing_only = bool(support_logvar_missing_only)
        self.support_logvar_monotone = bool(support_logvar_monotone)
        self.support_logvar_monotone_init = float(support_logvar_monotone_init)
        self.support_logvar_use_anchor = bool(support_logvar_use_anchor)
        self.support_logvar_anchor_init = float(support_logvar_anchor_init)
        self.use_feature_logvar_bias = bool(use_feature_logvar_bias)
        self.feature_logvar_bias_scope = str(feature_logvar_bias_scope)
        if self.feature_logvar_bias_scope not in {'all', 'chem', 'psd'}:
            raise ValueError("feature_logvar_bias_scope must be one of {'all', 'chem', 'psd'}")
        self.feature_logvar_bias_init = float(feature_logvar_bias_init)
        self.feature_logvar_bias_constraint = str(feature_logvar_bias_constraint)
        if self.feature_logvar_bias_constraint not in {'none', 'nonnegative'}:
            raise ValueError(
                "feature_logvar_bias_constraint must be one of {'none', 'nonnegative'}"
            )
        self.variance_path_use_latent = bool(variance_path_use_latent)
        self.variance_path_detach_latent = bool(variance_path_detach_latent)
        self.variance_path_use_mask = bool(variance_path_use_mask)
        self.variance_mask_dim = int(variance_mask_dim)
        self.variance_use_grouped_conv = bool(variance_use_grouped_conv)
        self.use_local_context_map = bool(use_local_context_map)
        self.local_context_dim = int(local_context_dim)
        self.local_context_steps = (
            None if local_context_steps is None else int(local_context_steps)
        )
        self.local_context_gate_init = float(local_context_gate_init)
        self.local_context_injection_mode = str(local_context_injection_mode)
        self.local_context_fusion_mode = str(local_context_fusion_mode)
        self.local_context_attn_heads = int(local_context_attn_heads)
        self.local_context_attn_window_tokens = int(local_context_attn_window_tokens)
        self.local_context_attn_gate_init = float(local_context_attn_gate_init)
        self.local_context_attn_location = str(local_context_attn_location)
        self.local_context_attn_support_bias_scale = float(local_context_attn_support_bias_scale)
        self.local_context_attn_gate_support_power = max(
            0.0, float(local_context_attn_gate_support_power)
        )
        self.local_context_attn_gate_support_floor = min(
            1.0, max(0.0, float(local_context_attn_gate_support_floor))
        )
        self.local_context_attn_logvar_support_boost = max(
            0.0, float(local_context_attn_logvar_support_boost)
        )
        self.use_decoder_final_norm = bool(use_decoder_final_norm)
        self.z_film_alpha_init = float(z_film_alpha_init)
        self.z_skip_gate_init = float(z_skip_gate_init)
        self.use_z_skip = bool(use_z_skip)
        self.decoder_cross_attn_gate_init = float(decoder_cross_attn_gate_init)
        self.current_epoch = -1
        if self.local_context_injection_mode not in {'seed', 'post_upsample', 'both'}:
            raise ValueError(
                "local_context_injection_mode must be one of "
                "{'seed', 'post_upsample', 'both'}"
            )
        if self.local_context_fusion_mode not in {'add', 'attn'}:
            raise ValueError("local_context_fusion_mode must be one of {'add', 'attn'}")
        if self.local_context_fusion_mode == 'attn' and not use_progressive_decoder:
            raise ValueError("local_context_fusion_mode='attn' requires use_progressive_decoder=True")
        if self.local_context_attn_location not in {'mid_tcn', 'upsample', 'both'}:
            raise ValueError(
                "local_context_attn_location must be one of "
                "{'mid_tcn', 'upsample', 'both'}"
            )
        
        hidden = hidden_dims[-1]
        self.hidden_dim = hidden
        self.aux_dim = int(aux_dim)
        
        if use_progressive_decoder:
            # --- Progressive Upsampling ---
            # Compact initial projection: latent → hidden × initial_steps
            self.initial_steps = int(decoder_initial_steps)
            if self.initial_steps < 1 or self.initial_steps > window_size:
                raise ValueError(
                    f"decoder_initial_steps must be in [1, {window_size}], got {self.initial_steps}"
                )
            self.latent_proj = nn.Linear(latent_dim, hidden * self.initial_steps)
            if self.local_context_steps is None:
                self.local_context_steps = self.initial_steps
            self.local_context_steps = max(1, self.local_context_steps)
            if cond_film_last_n is None:
                self.cond_film_last_n = num_layers
            else:
                self.cond_film_last_n = max(0, min(num_layers, int(cond_film_last_n)))
            
            # Use AdaptiveUpsample blocks so it works for ANY window_size
            # Start at initial_steps, then progressively upsample in powers of 2 until we hit/exceed window_size
            self.upsample_blocks = nn.ModuleList()
            current_steps = self.initial_steps
            while current_steps < window_size:
                self.upsample_blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        param.weight_norm(nn.Conv1d(hidden, hidden, 3, padding=1)),
                        nn.GELU(),
                    )
                )
                current_steps *= 2
            
            # Final adjustment layer to match exact window_size using linear interpolation
            self.final_upsample = nn.Upsample(size=window_size, mode='linear', align_corners=False)
            
            
            # Condition projection (Bottleneck MLP: compress → activate → expand).
            # Some modality ablations intentionally remove all auxiliary/time
            # conditioning.  In that case cond_h is a zero tensor in forward and
            # FiLM starts/behaves as an identity path.
            if self.aux_dim > 0:
                self.cond_proj = nn.Sequential(
                    nn.Conv1d(self.aux_dim, 64, 1),  # Compress: 11 → 64 (bottleneck)
                    nn.GELU(),                       # Nonlinearity for meteorological thresholds
                    nn.Conv1d(64, hidden, 1)         # Expand: 64 → 384 (output hidden)
                )
            else:
                self.cond_proj = None
            # FiLM layers: per-TCN-layer gamma and beta
            self.film_gammas = nn.ModuleList()
            self.film_betas = nn.ModuleList()
            self.z_film_gammas = nn.ModuleList()
            self.z_film_betas = nn.ModuleList()
            
            # [NEW] Pre-LayerNorm list (elementwise_affine=False to let FiLM handle coefficients)
            self.dec_norms = nn.ModuleList([
                nn.LayerNorm(hidden, elementwise_affine=False) for _ in range(num_layers)
            ])
            
            # Learnable per-layer scale for z-FiLM branch weighting (sigmoid-scaled in forward)
            self.z_film_alpha = nn.Parameter(torch.full((num_layers,), self.z_film_alpha_init))
            
            # [NEW] Learnable residual scaling alpha; initial ~0.2 via sigmoid(-1.38)
            # forces model to rely on identity path during early training.
            self.residual_alpha = nn.Parameter(torch.full((num_layers,), -1.38))
            if local_context_attn_after_tcn_layers is None:
                self.local_context_attn_after_tcn_layers = num_layers // 2
            else:
                self.local_context_attn_after_tcn_layers = int(local_context_attn_after_tcn_layers)
            self.local_context_attn_after_tcn_layers = max(
                0, min(num_layers, self.local_context_attn_after_tcn_layers)
            )

            for i in range(num_layers):
                gamma_k = self.film_gamma_kernel_size
                beta_k = self.film_beta_kernel_size
                if self.film_temporal_last_n > 0 and i >= (num_layers - self.film_temporal_last_n):
                    gamma_k = self.film_temporal_last_kernel_size
                    beta_k = self.film_temporal_last_kernel_size

                gamma_padding = gamma_k // 2
                beta_padding = beta_k // 2
                gamma = nn.Conv1d(
                    hidden, hidden, gamma_k,
                    padding=gamma_padding, padding_mode='reflect'
                )
                beta = nn.Conv1d(
                    hidden, hidden, beta_k,
                    padding=beta_padding, padding_mode='reflect'
                )
                z_gamma = nn.Linear(latent_dim, hidden)
                z_beta = nn.Linear(latent_dim, hidden)
                # Initialize gamma_proj output ≈ 0 → total gamma = 1.0 + 0 = 1.0 (identity at start)
                nn.init.zeros_(gamma.bias)
                nn.init.zeros_(gamma.weight)
                nn.init.zeros_(beta.bias)
                nn.init.zeros_(beta.weight)
                # Conservative init so z-FiLM starts near disabled and learns to activate.
                nn.init.zeros_(z_gamma.bias)
                nn.init.zeros_(z_gamma.weight)
                nn.init.zeros_(z_beta.bias)
                nn.init.zeros_(z_beta.weight)
                self.film_gammas.append(gamma)
                self.film_betas.append(beta)
                self.z_film_gammas.append(z_gamma)
                self.z_film_betas.append(z_beta)
            self.local_context_seed_proj = None
            self.local_context_seed_resize = None
            self.local_context_gate = None
            self.local_context_post_proj = None
            self.local_context_post_gate = None
            self.local_context_attn_fusion = None
            self.local_context_upsample_attn_fusions = None
            self.local_context_upsample_stage_lengths = ()
            self.last_local_context_gate = None
            if self.use_local_context_map:
                if self.local_context_fusion_mode == 'attn':
                    if self.local_context_attn_location in {'mid_tcn', 'both'}:
                        self.local_context_attn_fusion = LocalContextMemoryAttention(
                            dec_dim=hidden,
                            ctx_dim=self.local_context_dim,
                            n_heads=self.local_context_attn_heads,
                            window_tokens=self.local_context_attn_window_tokens,
                            gate_init=self.local_context_attn_gate_init,
                            support_bias_scale=self.local_context_attn_support_bias_scale,
                            gate_support_power=self.local_context_attn_gate_support_power,
                            gate_support_floor=self.local_context_attn_gate_support_floor,
                            dropout=dropout,
                        )
                    if self.local_context_attn_location in {'upsample', 'both'}:
                        stage_lengths = [self.initial_steps]
                        current_steps = self.initial_steps
                        while current_steps < window_size:
                            current_steps *= 2
                            if current_steps <= window_size:
                                stage_lengths.append(current_steps)
                        if stage_lengths[-1] != window_size:
                            stage_lengths.append(window_size)
                        self.local_context_upsample_stage_lengths = tuple(stage_lengths)
                        self.local_context_upsample_attn_fusions = nn.ModuleList([
                            LocalContextMemoryAttention(
                                dec_dim=hidden,
                                ctx_dim=self.local_context_dim,
                                n_heads=self.local_context_attn_heads,
                                window_tokens=self.local_context_attn_window_tokens,
                                gate_init=self.local_context_attn_gate_init,
                                support_bias_scale=self.local_context_attn_support_bias_scale,
                                gate_support_power=self.local_context_attn_gate_support_power,
                                gate_support_floor=self.local_context_attn_gate_support_floor,
                                dropout=dropout,
                            )
                            for _ in self.local_context_upsample_stage_lengths
                        ])
                else:
                    if self.local_context_injection_mode in {'seed', 'both'}:
                        self.local_context_seed_proj = nn.Conv1d(self.local_context_dim, hidden, 1)
                        nn.init.zeros_(self.local_context_seed_proj.weight)
                        nn.init.zeros_(self.local_context_seed_proj.bias)
                        if self.local_context_steps != self.initial_steps:
                            self.local_context_seed_resize = nn.AdaptiveAvgPool1d(self.initial_steps)
                        self.local_context_gate = nn.Parameter(torch.tensor(self.local_context_gate_init))
                    if self.local_context_injection_mode in {'post_upsample', 'both'}:
                        self.local_context_post_proj = nn.Conv1d(self.local_context_dim, hidden, 1)
                        nn.init.zeros_(self.local_context_post_proj.weight)
                        nn.init.zeros_(self.local_context_post_proj.bias)
                        self.local_context_post_gate = nn.Parameter(torch.tensor(self.local_context_gate_init))
        else:
            # --- Legacy: one-shot projection ---
            self.latent_proj = nn.Linear(latent_dim, hidden * window_size)
            self.cond_proj = nn.Conv1d(self.aux_dim, hidden, 1) if self.aux_dim > 0 else None
            self.local_context_seed_proj = None
            self.local_context_seed_resize = None
            self.local_context_gate = None
            self.local_context_post_proj = None
            self.local_context_post_gate = None
            self.local_context_attn_fusion = None
            self.local_context_upsample_attn_fusions = None
            self.local_context_upsample_stage_lengths = ()
            self.last_local_context_gate = None
            self.local_context_attn_after_tcn_layers = 0

        # Latent skip path to preserve z influence near output heads.
        self.z_skip_proj = None
        self.z_skip_gate = None
        if self.use_z_skip:
            self.z_skip_proj = nn.Linear(latent_dim, hidden)
            self.z_skip_gate = nn.Linear(latent_dim, hidden)
            nn.init.zeros_(self.z_skip_proj.weight)
            nn.init.zeros_(self.z_skip_proj.bias)
            nn.init.zeros_(self.z_skip_gate.weight)
            nn.init.constant_(self.z_skip_gate.bias, self.z_skip_gate_init)
        
        if self.use_decoder_cross_attn:
            self.dec_cross_attn = MaskedTemporalCrossAttention(
                dec_dim=hidden, enc_dim=hidden_dims[-1], n_heads=n_cross_attn_heads, dropout=dropout
            )
            # Global per-channel gate keeps encoder retrieval as a refinement path,
            # not a hard replacement of the decoder scaffold.
            self.cross_attn_gate = nn.Parameter(torch.full((hidden,), self.decoder_cross_attn_gate_init))
            self.cross_attn_fuse_gate = nn.Conv1d(hidden * 2, hidden, 1)
            self.cross_attn_fuse_proj = nn.Conv1d(hidden * 2, hidden, 1)
            nn.init.zeros_(self.cross_attn_fuse_gate.weight)
            nn.init.constant_(self.cross_attn_fuse_gate.bias, -2.0)
            nn.init.zeros_(self.cross_attn_fuse_proj.weight)
            nn.init.zeros_(self.cross_attn_fuse_proj.bias)
        
        # TCN decoder layers (no residual connections — forces reliance on z)
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (num_layers - 1 - i)
            self.tcn_layers.append(
                nn.Sequential(
                    param.weight_norm(nn.Conv1d(hidden, hidden, kernel_size,
                                                padding=(kernel_size-1)*dilation//2,
                                                dilation=dilation)),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output projection (mean)
        self.output_proj = nn.Conv1d(hidden, output_dim, 1)
        self.final_output_norm = (
            nn.LayerNorm(hidden) if self.use_decoder_final_norm else nn.Identity()
        )
        self.mean_head = None
        self.logvar_head = None
        if self.use_dual_output_heads:
            head_hidden = output_head_hidden_dim
            if head_hidden is None:
                head_hidden = max(64, hidden // 2)
            head_hidden = max(16, min(hidden, int(head_hidden)))
            self.output_head_hidden_dim = head_hidden

            self.mean_head = nn.Sequential(
                nn.Conv1d(hidden, head_hidden, 1),
                nn.GELU(),
                nn.Conv1d(head_hidden, hidden, 1),
            )
            # Residual head starts from near-identity, then learns task-specific
            # refinements for mean vs uncertainty.
            nn.init.zeros_(self.mean_head[-1].weight)
            nn.init.zeros_(self.mean_head[-1].bias)
            use_shared_logvar_head = (
                (not self.use_detached_variance_pathway)
                or (self.variance_detach_start_epoch is not None)
            )
            if use_shared_logvar_head:
                self.logvar_head = nn.Sequential(
                    nn.Conv1d(hidden, head_hidden, 1),
                    nn.GELU(),
                    nn.Conv1d(head_hidden, hidden, 1),
                )
                nn.init.zeros_(self.logvar_head[-1].weight)
                nn.init.zeros_(self.logvar_head[-1].bias)

        self.variance_input_norm = None
        self.variance_stem = None
        self.variance_mask_proj = None
        self.variance_attn_support_proj = None
        self.variance_refine_chem = None
        self.variance_refine_psd = None
        self.variance_refine = None
        self.support_logvar_mlp = None
        self.support_logvar_monotone_beta = None
        self.support_logvar_anchor_beta = None
        self.feature_logvar_bias = None
        self.last_support_logvar_residual_mean = None
        self.last_support_logvar_residual_missing_mean = None
        self.last_support_logvar_residual_psd_low_support_mean = None
        self.last_support_logvar_residual_psd_high_support_mean = None
        self.last_support_logvar_monotone_beta = None
        self.last_support_logvar_anchor_beta = None
        self.last_feature_logvar_bias_mean = None
        self.last_feature_logvar_bias_chem_mean = None
        self.last_feature_logvar_bias_psd_mean = None
        self.last_feature_logvar_bias_psd_min = None
        self.last_feature_logvar_bias_psd_max = None
        self.last_feature_logvar_bias_added_pre_clamp = None
        self.last_support_logvar_residual_added_pre_clamp = None
        self.last_logvar_diagnostics = None
        if self.use_detached_variance_pathway:
            var_hidden = variance_head_hidden_dim
            if var_hidden is None:
                var_hidden = max(128, hidden)
            var_hidden = max(32, int(var_hidden))
            self.variance_head_hidden_dim = var_hidden
            mask_dim = self.variance_mask_dim if self.variance_path_use_mask else 0
            attn_support_dim = self.variance_attn_support_dim if self.use_variance_attn_support else 0
            var_in_dim = hidden + (latent_dim if self.variance_path_use_latent else 0) + mask_dim + attn_support_dim
            self.variance_input_norm = nn.LayerNorm(hidden)
            if self.variance_path_use_mask:
                self.variance_mask_proj = nn.Sequential(
                    nn.Conv1d(output_dim, self.variance_mask_dim, 1),
                    nn.GELU(),
                )
            self.variance_attn_support_proj = None
            if self.use_variance_attn_support and variance_attn_support_n_features > 0:
                self.variance_attn_support_proj = nn.Conv1d(
                    variance_attn_support_n_features, self.variance_attn_support_dim, 1
                )
                nn.init.zeros_(self.variance_attn_support_proj.weight)
                nn.init.zeros_(self.variance_attn_support_proj.bias)
            grouped_conv_groups = 1
            if self.variance_use_grouped_conv:
                grouped_conv_groups = max(1, hidden // 4)
                grouped_conv_groups = math.gcd(var_hidden, grouped_conv_groups)
                grouped_conv_groups = max(1, grouped_conv_groups)
            # A deeper, detached variance stem keeps uncertainty learning from
            # being dominated by the mean-optimized decoder trunk.
            self.variance_stem = nn.Sequential(
                nn.Conv1d(var_in_dim, var_hidden, 1),
                nn.GELU(),
                nn.Conv1d(var_hidden, var_hidden, 3, padding=1, groups=grouped_conv_groups),
                nn.GELU(),
                nn.Conv1d(var_hidden, hidden, 1),
            )

            refine_hidden = max(32, min(hidden, var_hidden // 2))
            if n_chem > 0 and self.n_psd > 0:
                self.variance_refine_chem = nn.Sequential(
                    nn.Conv1d(hidden, refine_hidden, 1),
                    nn.GELU(),
                    nn.Conv1d(refine_hidden, hidden, 1),
                )
                self.variance_refine_psd = nn.Sequential(
                    nn.Conv1d(hidden, refine_hidden, 1),
                    nn.GELU(),
                    nn.Conv1d(refine_hidden, hidden, 1),
                )
                nn.init.zeros_(self.variance_refine_chem[-1].weight)
                nn.init.zeros_(self.variance_refine_chem[-1].bias)
                nn.init.zeros_(self.variance_refine_psd[-1].weight)
                nn.init.zeros_(self.variance_refine_psd[-1].bias)
            else:
                self.variance_refine = nn.Sequential(
                    nn.Conv1d(hidden, refine_hidden, 1),
                    nn.GELU(),
                    nn.Conv1d(refine_hidden, hidden, 1),
                )
                nn.init.zeros_(self.variance_refine[-1].weight)
                nn.init.zeros_(self.variance_refine[-1].bias)

        if self.use_support_logvar_residual:
            # 7 support/provenance channels built from the *visible* decoder
            # obs_mask. The free MLP's final projection is zero-initialized so it
            # is exactly inert at initialization. The monotone fallback starts as
            # a mild positive low-support prior with usable gradient.
            if self.support_logvar_monotone:
                self.support_logvar_monotone_beta = nn.Parameter(
                    torch.tensor(self.support_logvar_monotone_init)
                )
                if self.support_logvar_use_anchor:
                    self.support_logvar_anchor_beta = nn.Parameter(
                        torch.tensor(self.support_logvar_anchor_init)
                    )
            else:
                self.support_logvar_mlp = nn.Sequential(
                    nn.Conv2d(7, self.support_logvar_hidden_dim, 1),
                    nn.GELU(),
                    nn.Conv2d(self.support_logvar_hidden_dim, 1, 1),
                )
                nn.init.zeros_(self.support_logvar_mlp[-1].weight)
                nn.init.zeros_(self.support_logvar_mlp[-1].bias)

        if self.use_feature_logvar_bias:
            self.feature_logvar_bias = nn.Parameter(
                torch.full((output_dim,), self.feature_logvar_bias_init, dtype=torch.float32)
            )
        
        # Variance projection (heteroscedastic) - SEPARATE for Chem and PSD
        if heteroscedastic:
            if n_chem > 0 and self.n_psd > 0:
                self.logvar_proj_chem = nn.Conv1d(hidden, n_chem, 1)
                self.logvar_proj_psd = nn.Conv1d(hidden, self.n_psd, 1)
            else:
                self.logvar_proj = nn.Conv1d(hidden, output_dim, 1)
    
    def _apply_decoder_cross_attn_fusion(self, h, enc_h_seq=None, obs_mask=None):
        """Retrieve encoder context and fuse it into decoder states before TCN."""
        if not (self.use_decoder_cross_attn and enc_h_seq is not None):
            return h

        ca_out = self.dec_cross_attn(h, enc_h_seq, obs_mask if not self.ignore_obs_mask else None)
        fuse_in = torch.cat([h, ca_out], dim=1)
        local_gate = torch.sigmoid(self.cross_attn_fuse_gate(fuse_in))
        fused_delta = self.cross_attn_fuse_proj(fuse_in)
        global_gate = torch.sigmoid(self.cross_attn_gate).unsqueeze(0).unsqueeze(-1)

        if self.decoder_cross_attn_missing_only and obs_mask is not None:
            # Only missing-query timesteps are updated by decoder cross-attention;
            # observed queries keep their local latent/upsample scaffold.
            query_missing = (obs_mask == 0).any(dim=-1, keepdim=True).permute(0, 2, 1).to(h.dtype)
            local_gate = local_gate * query_missing
            fused_delta = fused_delta * query_missing

        return h + global_gate * local_gate * fused_delta

    def _apply_decoder_tcn_layers(self, h, cond_h, z, start_idx=0, end_idx=None):
        """Run a slice of the FiLM-conditioned decoder TCN stack."""
        if not self.use_tcn:
            return h

        total_layers = len(self.tcn_layers)
        if end_idx is None:
            end_idx = total_layers
        start_idx = max(0, min(total_layers, int(start_idx)))
        end_idx = max(start_idx, min(total_layers, int(end_idx)))

        for i in range(start_idx, end_idx):
            layer = self.tcn_layers[i]
            gamma_proj = self.film_gammas[i]
            beta_proj = self.film_betas[i]
            z_gamma_proj = self.z_film_gammas[i]
            z_beta_proj = self.z_film_betas[i]
            norm = self.dec_norms[i]

            h_res = h
            h_norm = norm(h.transpose(1, 2)).transpose(1, 2)
            h_conv = layer(h_norm)

            use_cond_film = i >= (len(self.tcn_layers) - self.cond_film_last_n)
            if use_cond_film:
                gamma_c = 1.0 + self.cond_film_gamma_scale * torch.tanh(gamma_proj(cond_h))
                beta_c = beta_proj(cond_h)
                h_cond = gamma_c * h_conv + beta_c
            else:
                h_cond = h_conv

            alpha_z = torch.sigmoid(self.z_film_alpha[i])
            gamma_z = torch.tanh(z_gamma_proj(z)).unsqueeze(-1)
            beta_z = z_beta_proj(z).unsqueeze(-1)
            h_film = (1.0 + alpha_z * gamma_z) * h_cond + alpha_z * beta_z

            alpha_res = torch.sigmoid(self.residual_alpha[i])
            h = h_res + alpha_res * h_film

        return h

    def _apply_local_context_attn_fusion(self, h, local_context=None, obs_mask=None, attn_module=None):
        """Decoder pulls from local-context memory with support-aware local attention."""
        if attn_module is None:
            attn_module = self.local_context_attn_fusion
        if attn_module is None or local_context is None:
            return h

        if obs_mask is None:
            support_full = torch.ones(
                h.shape[0], 1, self.window_size, device=h.device, dtype=h.dtype
            )
        else:
            support_full = obs_mask.to(h.dtype).mean(dim=-1, keepdim=False).unsqueeze(1)
        support_tokens = F.adaptive_avg_pool1d(support_full, local_context.shape[-1])
        support_high = F.adaptive_avg_pool1d(support_full, h.shape[-1])
        attn_delta, gate = attn_module(
            h,
            local_context,
            support_tokens=support_tokens,
            support_high=support_high,
        )
        self.last_local_context_gate = attn_module.last_gate_mean
        return h + attn_delta

    def _apply_local_context_upsample_attn_fusion(self, h, stage_idx, local_context=None, obs_mask=None):
        """Stage-wise local-context memory attention during coarse-to-fine upsampling."""
        if self.local_context_upsample_attn_fusions is None:
            return h
        if stage_idx < 0 or stage_idx >= len(self.local_context_upsample_attn_fusions):
            return h
        attn_module = self.local_context_upsample_attn_fusions[stage_idx]
        return self._apply_local_context_attn_fusion(
            h,
            local_context=local_context,
            obs_mask=obs_mask,
            attn_module=attn_module,
        )

    def _use_detached_variance_now(self):
        """Whether the variance branch should run in detached mode this forward."""
        if not self.use_detached_variance_pathway:
            return False
        if self.variance_detach_start_epoch is None:
            return True
        if self.current_epoch < 0:
            # Fresh checkpoint loaded for inference/eval should use final detached mode.
            return True
        return self.current_epoch >= self.variance_detach_start_epoch

    def _support_logvar_features(self, obs_mask):
        """Build visible-support features for support-to-logvar routing.

        Args:
            obs_mask: [B, W, C], 1 = value visible to the model.

        Returns:
            features: [B, 7, C, W]
            low_support_score: [B, C, W]
            same_feature_window_support: [B, C, W]
        """
        obs = obs_mask.permute(0, 2, 1).to(dtype=torch.float32)  # [B, C, W]
        B, C, W = obs.shape
        missing = 1.0 - obs

        same_feature_window = obs.mean(dim=-1, keepdim=True).expand(-1, -1, W)

        pos = torch.arange(W, device=obs.device, dtype=torch.long).view(1, 1, W)
        obs_bool = obs > 0.5
        neg_large = torch.full((1, 1, W), -W - 1, device=obs.device, dtype=torch.long)
        pos_large = torch.full((1, 1, W), W + 1, device=obs.device, dtype=torch.long)
        left_candidates = torch.where(obs_bool, pos, neg_large)
        left_idx = torch.cummax(left_candidates, dim=-1).values
        left_dist = (pos - left_idx).clamp(min=0, max=W + 1)
        left_dist = torch.where(left_idx < 0, torch.full_like(left_dist, W + 1), left_dist)
        right_candidates = torch.where(obs_bool, pos, pos_large)
        right_idx = torch.flip(
            torch.cummin(torch.flip(right_candidates, dims=[-1]), dim=-1).values,
            dims=[-1],
        )
        right_dist = (right_idx - pos).clamp(min=0, max=W + 1)
        right_dist = torch.where(right_idx > W, torch.full_like(right_dist, W + 1), right_dist)
        nearest_dist = torch.minimum(left_dist, right_dist).clamp(max=W).to(obs.dtype) / max(1, W)

        if self.n_chem > 0 and self.n_psd > 0 and C == self.output_dim:
            chem_obs = obs[:, :self.n_chem, :]
            psd_obs = obs[:, self.n_chem:, :]
            chem_t = chem_obs.mean(dim=1, keepdim=True)
            psd_t = psd_obs.mean(dim=1, keepdim=True)
            chem_w = chem_obs.mean(dim=(1, 2), keepdim=True)
            psd_w = psd_obs.mean(dim=(1, 2), keepdim=True)
            same_family_t = torch.cat([
                chem_t.expand(-1, self.n_chem, -1),
                psd_t.expand(-1, self.n_psd, -1),
            ], dim=1)
            other_family_t = torch.cat([
                psd_t.expand(-1, self.n_chem, -1),
                chem_t.expand(-1, self.n_psd, -1),
            ], dim=1)
            same_family_w = torch.cat([
                chem_w.expand(-1, self.n_chem, W),
                psd_w.expand(-1, self.n_psd, W),
            ], dim=1)
            other_family_w = torch.cat([
                psd_w.expand(-1, self.n_chem, W),
                chem_w.expand(-1, self.n_psd, W),
            ], dim=1)
        else:
            family_t = obs.mean(dim=1, keepdim=True).expand(-1, C, -1)
            family_w = obs.mean(dim=(1, 2), keepdim=True).expand(-1, C, W)
            same_family_t = family_t
            other_family_t = family_t
            same_family_w = family_w
            other_family_w = family_w

        features = torch.stack([
            missing,
            same_feature_window,
            nearest_dist,
            same_family_t,
            other_family_t,
            same_family_w,
            other_family_w,
        ], dim=1)
        low_support_score = missing * (1.0 - same_feature_window).clamp(0.0, 1.0)
        return features, low_support_score, same_feature_window

    def _apply_support_logvar_residual(self, logvar, obs_mask):
        """Add an explicit support-derived residual to decoder log variance."""
        self.last_support_logvar_residual_mean = None
        self.last_support_logvar_residual_missing_mean = None
        self.last_support_logvar_residual_psd_low_support_mean = None
        self.last_support_logvar_residual_psd_high_support_mean = None
        self.last_support_logvar_monotone_beta = None
        self.last_support_logvar_anchor_beta = None
        self.last_support_logvar_residual_added_pre_clamp = None

        if (not self.use_support_logvar_residual) or obs_mask is None or logvar is None:
            return logvar

        features, low_support_score, same_feature_window = self._support_logvar_features(obs_mask)
        features = features.to(dtype=logvar.dtype)
        low_support_score = low_support_score.to(dtype=logvar.dtype)

        if self.support_logvar_monotone:
            beta = F.softplus(self.support_logvar_monotone_beta).to(dtype=logvar.dtype)
            residual = beta * low_support_score
            self.last_support_logvar_monotone_beta = float(beta.detach().item())
            if self.support_logvar_use_anchor and self.support_logvar_anchor_beta is not None:
                anchor_beta = F.softplus(self.support_logvar_anchor_beta).to(dtype=logvar.dtype)
                nearest_anchor = features[:, 2, :, :].to(dtype=logvar.dtype)
                missing_gate = features[:, 0, :, :].to(dtype=logvar.dtype)
                residual = residual + anchor_beta * nearest_anchor * missing_gate
                self.last_support_logvar_anchor_beta = float(anchor_beta.detach().item())
        else:
            residual = self.support_logvar_mlp(features).squeeze(1)
            if self.support_logvar_missing_only:
                residual = residual * features[:, 0, :, :]

        residual_det = residual.detach()
        missing = features[:, 0, :, :].detach() > 0.5
        self.last_support_logvar_residual_mean = float(residual_det.mean().item())
        self.last_support_logvar_residual_missing_mean = (
            float(residual_det[missing].mean().item()) if missing.any() else None
        )
        if self.n_chem > 0 and self.n_psd > 0 and residual.shape[1] > self.n_chem:
            psd_res = residual_det[:, self.n_chem:, :]
            psd_support = same_feature_window[:, self.n_chem:, :].detach()
            psd_missing = missing[:, self.n_chem:, :]
            low = psd_missing & (psd_support < 0.10)
            high = psd_missing & (psd_support >= 0.50)
            self.last_support_logvar_residual_psd_low_support_mean = (
                float(psd_res[low].mean().item()) if low.any() else None
            )
            self.last_support_logvar_residual_psd_high_support_mean = (
                float(psd_res[high].mean().item()) if high.any() else None
            )

        logvar = logvar + residual
        self.last_support_logvar_residual_added_pre_clamp = logvar.detach()
        return torch.clamp(logvar, min=np.log(self.var_min), max=np.log(self.var_max))

    def _apply_feature_logvar_bias(self, logvar):
        """Add learnable feature/bin-wise base calibration bias to log variance."""
        self.last_feature_logvar_bias_mean = None
        self.last_feature_logvar_bias_chem_mean = None
        self.last_feature_logvar_bias_psd_mean = None
        self.last_feature_logvar_bias_psd_min = None
        self.last_feature_logvar_bias_psd_max = None
        self.last_feature_logvar_bias_added_pre_clamp = None

        if (not self.use_feature_logvar_bias) or self.feature_logvar_bias is None or logvar is None:
            return logvar

        raw_bias = self.feature_logvar_bias.to(device=logvar.device, dtype=logvar.dtype)
        if self.feature_logvar_bias_constraint == 'nonnegative':
            # Interpret feature_logvar_bias_init as the raw parameter init in
            # this mode. A negative init (e.g. -6) starts nearly inert while
            # keeping usable positive gradient and forbidding interval shrinkage.
            bias = F.softplus(raw_bias)
        else:
            bias = raw_bias
        if self.feature_logvar_bias_scope == 'psd':
            mask = torch.zeros_like(bias)
            if self.n_chem > 0 and self.n_psd > 0:
                mask[self.n_chem:] = 1.0
            else:
                mask[:] = 1.0
            bias = bias * mask
        elif self.feature_logvar_bias_scope == 'chem':
            mask = torch.zeros_like(bias)
            if self.n_chem > 0:
                mask[:self.n_chem] = 1.0
            else:
                mask[:] = 1.0
            bias = bias * mask

        bias_det = bias.detach()
        self.last_feature_logvar_bias_mean = float(bias_det.mean().item())
        if self.n_chem > 0:
            self.last_feature_logvar_bias_chem_mean = float(bias_det[:self.n_chem].mean().item())
        if self.n_psd > 0 and bias_det.numel() > self.n_chem:
            psd_bias = bias_det[self.n_chem:]
            self.last_feature_logvar_bias_psd_mean = float(psd_bias.mean().item())
            self.last_feature_logvar_bias_psd_min = float(psd_bias.min().item())
            self.last_feature_logvar_bias_psd_max = float(psd_bias.max().item())

        logvar = logvar + bias.view(1, -1, 1)
        self.last_feature_logvar_bias_added_pre_clamp = logvar.detach()
        return torch.clamp(logvar, min=np.log(self.var_min), max=np.log(self.var_max))

    def forward(self, z, cond, enc_h_seq=None, obs_mask=None, local_context=None,
                attn_weighted_support_t=None):
        """
        Args:
            z: [Batch, Latent_dim]
            cond: [Batch, Window, Aux_dim]
            enc_h_seq: [Batch, Hidden, Window] from Encoder
            obs_mask: [Batch, Window, Target_dim]
            attn_weighted_support_t: [Batch, Window, Target_dim] attention-weighted
                cross-feature support map from encoder (detached). Each value is
                sum_{f'} attn_weight(f'->f) * obs_mask[f',t], giving per-feature
                per-timestep importance-weighted support. Used in variance pathway only.

        Returns:
            mean: [Batch, Output_dim, Window]
            logvar: [Batch, Output_dim, Window] or None
        """
        B = z.shape[0]
        if cond is not None and cond.shape[-1] > 0 and self.cond_proj is not None:
            cond_t = cond.permute(0, 2, 1)  # [B, aux, W]
            cond_h = self.cond_proj(cond_t)  # [B, hidden, W]
        else:
            cond_h = z.new_zeros(B, self.hidden_dim, self.window_size)
        local_gate_vals = []
        self.last_local_context_gate = None
        self.last_logvar_diagnostics = None
        logvar_diag = {}
        
        if self.use_progressive_decoder:
            # Step 1: Compact expansion → initial_steps
            h = self.latent_proj(z)  # [B, hidden * 12]
            h = h.view(B, -1, self.initial_steps)  # [B, hidden, initial_steps]
            upsample_stage_idx = 0

            # Optional coarse local context map: low-dim, low-resolution encoder
            # summary injected only at seed stage so local detail remains
            # bottlenecked and cannot bypass the full decoder stack too easily.
            if self.use_local_context_map and local_context is not None:
                if self.local_context_fusion_mode == 'add' and self.local_context_seed_proj is not None:
                    local_seed = local_context
                    if self.local_context_seed_resize is not None:
                        local_seed = self.local_context_seed_resize(local_seed)
                    local_seed = self.local_context_seed_proj(local_seed)
                    local_gate = torch.sigmoid(self.local_context_gate)
                    local_gate_vals.append(local_gate.detach())
                    h = h + local_gate * local_seed
                elif (
                    self.local_context_fusion_mode == 'attn'
                    and self.local_context_attn_location in {'upsample', 'both'}
                ):
                    h = self._apply_local_context_upsample_attn_fusion(
                        h,
                        stage_idx=upsample_stage_idx,
                        local_context=local_context,
                        obs_mask=obs_mask,
                    )
                    upsample_stage_idx += 1
            
            # Step 2: Progressive upsample
            for block in self.upsample_blocks:
                h = block(h)
                if (
                    self.local_context_fusion_mode == 'attn'
                    and self.local_context_attn_location in {'upsample', 'both'}
                    and h.shape[-1] <= self.window_size
                ):
                    h = self._apply_local_context_upsample_attn_fusion(
                        h,
                        stage_idx=upsample_stage_idx,
                        local_context=local_context,
                        obs_mask=obs_mask,
                    )
                    upsample_stage_idx += 1
                
            # Step 2.5: Ensure exact length match
            if h.shape[-1] != self.window_size:
                h = self.final_upsample(h)
                if (
                    self.local_context_fusion_mode == 'attn'
                    and self.local_context_attn_location in {'upsample', 'both'}
                ):
                    h = self._apply_local_context_upsample_attn_fusion(
                        h,
                        stage_idx=upsample_stage_idx,
                        local_context=local_context,
                        obs_mask=obs_mask,
                    )
                    upsample_stage_idx += 1

            # Higher-resolution local context injection keeps coarse encoder
            # summaries closer to the final time axis so gap boundary signals
            # are less diluted by repeated upsampling.
            if (
                self.use_local_context_map
                and self.local_context_fusion_mode == 'add'
                and local_context is not None
                and self.local_context_post_proj is not None
            ):
                local_high = local_context
                if local_high.shape[-1] != h.shape[-1]:
                    local_high = F.interpolate(
                        local_high,
                        size=h.shape[-1],
                        mode='linear',
                        align_corners=False,
                    )
                local_high = self.local_context_post_proj(local_high)
                local_gate = torch.sigmoid(self.local_context_post_gate)
                local_gate_vals.append(local_gate.detach())
                h = h + local_gate * local_high

            # Step 3: Retrieve encoder context before temporal refinement.
            h = self._apply_decoder_cross_attn_fusion(h, enc_h_seq, obs_mask)

            # Step 4: FiLM-conditioned TCN (Pre-LayerNorm + Residual Scaling + Sequential FiLM)
            if self.use_tcn:
                if (
                    self.local_context_fusion_mode == 'attn'
                    and self.local_context_attn_fusion is not None
                    and self.local_context_attn_location in {'mid_tcn', 'both'}
                ):
                    split_idx = self.local_context_attn_after_tcn_layers
                    if split_idx > 0:
                        h = self._apply_decoder_tcn_layers(h, cond_h, z, 0, split_idx)
                    h = self._apply_local_context_attn_fusion(h, local_context, obs_mask)
                    if split_idx < len(self.tcn_layers):
                        h = self._apply_decoder_tcn_layers(h, cond_h, z, split_idx, None)
                else:
                    h = self._apply_decoder_tcn_layers(h, cond_h, z, 0, None)
        else:
            # Legacy path
            h = self.latent_proj(z)  # [B, hidden * window]
            h = h.view(B, -1, self.window_size)  # [B, hidden, W]
            h = h + cond_h

            h = self._apply_decoder_cross_attn_fusion(h, enc_h_seq, obs_mask)

            if self.use_tcn:
                for layer in self.tcn_layers:
                    h = layer(h)

        if local_gate_vals:
            self.last_local_context_gate = float(torch.stack(local_gate_vals).mean().item())
        elif self.local_context_fusion_mode != 'attn':
            self.last_local_context_gate = None

        # Step 5: Latent skip connection before output heads.
        if self.use_z_skip:
            z_skip = self.z_skip_proj(z).unsqueeze(-1)                # [B, hidden, 1]
            z_gate = torch.sigmoid(self.z_skip_gate(z)).unsqueeze(-1) # [B, hidden, 1]
            h = h + z_gate * z_skip

        # Final decoder normalization stabilizes the accumulated residual stream
        # before mean/logvar specialization and output projection.
        h = self.final_output_norm(h.transpose(1, 2)).transpose(1, 2)

        # Shared decoder trunk with two small residual heads. This lets mean and
        # log-variance specialize near the output without duplicating the full decoder.
        if self.use_dual_output_heads:
            h_mean = h + self.mean_head(h)
        else:
            h_mean = h

        use_detached_variance_now = self._use_detached_variance_now()

        if use_detached_variance_now:
            # Detach the shared decoder state first so variance learning does not
            # backprop into the mean-optimized trunk, while still letting the
            # variance-side normalization affine parameters learn useful channel
            # rescaling for uncertainty estimation.
            h_var_base = self.variance_input_norm(h.detach().transpose(1, 2)).transpose(1, 2)
            h_var_inputs = [h_var_base]
            if self.variance_path_use_latent:
                z_for_variance = z.detach() if self.variance_path_detach_latent else z
                z_broadcast = z_for_variance.unsqueeze(-1).expand(-1, -1, h.shape[-1])
                h_var_inputs.append(z_broadcast)
            if self.variance_path_use_mask:
                if obs_mask is None:
                    mask_feat = torch.zeros(
                        h.shape[0], self.variance_mask_dim, h.shape[-1],
                        device=h.device, dtype=h.dtype
                    )
                else:
                    mask_feat = self.variance_mask_proj(obs_mask.permute(0, 2, 1).to(h.dtype))
                h_var_inputs.append(mask_feat)
            if self.use_variance_attn_support:
                if attn_weighted_support_t is not None and self.variance_attn_support_proj is not None:
                    # attn_weighted_support_t: [B, W, C] → permute → [B, C, W]
                    support_feat = self.variance_attn_support_proj(
                        attn_weighted_support_t.permute(0, 2, 1).to(h.dtype)
                    )  # [B, variance_attn_support_dim, W]
                else:
                    support_feat = torch.zeros(
                        h.shape[0], self.variance_attn_support_dim, h.shape[-1],
                        device=h.device, dtype=h.dtype
                    )
                h_var_inputs.append(support_feat)
            h_var_input = torch.cat(h_var_inputs, dim=1)
            h_var = self.variance_stem(h_var_input)
        elif self.use_dual_output_heads and self.logvar_head is not None:
            h_logvar = h + self.logvar_head(h)
        else:
            h_logvar = h

        # Output projections
        mean = self.output_proj(h_mean)  # [B, output, W]
        
        logvar = None
        if self.heteroscedastic:
            if self.n_chem > 0 and self.n_psd > 0:
                if use_detached_variance_now:
                    h_logvar_chem = h_var + self.variance_refine_chem(h_var)
                    h_logvar_psd = h_var + self.variance_refine_psd(h_var)
                    logvar_chem = self.logvar_proj_chem(h_logvar_chem)
                    logvar_psd = self.logvar_proj_psd(h_logvar_psd)
                else:
                    logvar_chem = self.logvar_proj_chem(h_logvar)
                    logvar_psd = self.logvar_proj_psd(h_logvar)
                logvar_diag['base_pre_clamp'] = torch.cat([logvar_chem, logvar_psd], dim=1).detach()
                logvar_chem = torch.clamp(logvar_chem, min=np.log(self.var_min), max=np.log(self.var_max))
                logvar_psd = torch.clamp(logvar_psd, min=np.log(self.var_min), max=np.log(self.var_max))
                logvar = torch.cat([logvar_chem, logvar_psd], dim=1)
                logvar_diag['base_post_clamp'] = logvar.detach()
            else:
                if use_detached_variance_now:
                    h_logvar = h_var + self.variance_refine(h_var)
                logvar = self.logvar_proj(h_logvar)
                logvar_diag['base_pre_clamp'] = logvar.detach()
                logvar = torch.clamp(logvar, min=np.log(self.var_min), max=np.log(self.var_max))
                logvar_diag['base_post_clamp'] = logvar.detach()

            if self.local_context_attn_logvar_support_boost > 0.0 and obs_mask is not None:
                support = obs_mask.to(logvar.dtype).mean(dim=-1).unsqueeze(1)
                low_support = (1.0 - support).clamp(0.0, 1.0)
                logvar_diag['support_boost_pre'] = logvar.detach()
                boosted_logvar = logvar + self.local_context_attn_logvar_support_boost * low_support
                logvar_diag['support_boost_added_pre_clamp'] = boosted_logvar.detach()
                logvar = torch.clamp(
                    boosted_logvar,
                    min=np.log(self.var_min),
                    max=np.log(self.var_max),
                )
                logvar_diag['support_boost_post_clamp'] = logvar.detach()

            logvar_diag['feature_bias_pre'] = logvar.detach()
            logvar = self._apply_feature_logvar_bias(logvar)
            feature_bias_added = getattr(self, 'last_feature_logvar_bias_added_pre_clamp', None)
            if feature_bias_added is not None:
                logvar_diag['feature_bias_added_pre_clamp'] = feature_bias_added.detach()
            logvar_diag['feature_bias_post_clamp'] = logvar.detach()
            logvar_diag['support_residual_pre'] = logvar.detach()
            logvar = self._apply_support_logvar_residual(logvar, obs_mask)
            support_res_added = getattr(self, 'last_support_logvar_residual_added_pre_clamp', None)
            if support_res_added is not None:
                logvar_diag['support_residual_added_pre_clamp'] = support_res_added.detach()
            logvar_diag['support_residual_post_clamp'] = logvar.detach()
            logvar_diag['final'] = logvar.detach()
            self.last_logvar_diagnostics = logvar_diag
        
        return mean, logvar


# ==============================================================================
# Main Model
# ==============================================================================
class ImputationVAE_Graph(nn.Module):
    """
    Graph-Enhanced UQ-VAE for Imputation with Dynamic Graph Learning.
    
    Learns physical/chemical relationships between features using self-attention
    at the input level, enabling interpretable feature relationship analysis.
    """
    
    def __init__(self, target_dim, aux_dim, window_size, latent_dim=256,
                 hidden_dims=[512, 512, 512], encoder_layers=5, decoder_layers=5,
                 kernel_size=3, dropout=0.1, heteroscedastic=True, n_graph_heads=4,
                 var_min=1e-3, var_max=10.0,
                 n_chem=0, use_input_graph_layer=True, use_cross_modal_graph=True,
                 use_tcn=True, n_input_graph_layers=1, use_progressive_decoder=False,
                 decoder_initial_steps=12,
                 cond_film_last_n=None,
                 cond_film_gamma_scale=0.5,
                 use_decoder_cross_attn=False, n_cross_attn_heads=4,
                 decoder_cross_attn_missing_only=False,
                 use_parallel_graph=False, use_realnvp=False, realnvp_layers=4,
                 use_temporal_cnn=True,
                 use_hybrid_time_encoding=False,
                 time_numeric_dim=4,
                 time_cyc_dim=6,
                 time_hybrid_dim=6,
                 hour_embed_dim=8,
                 dow_embed_dim=4,
                 month_embed_dim=4,
                 film_kernel_size=1,
                 film_gamma_kernel_size=None,
                 film_beta_kernel_size=None,
                 film_temporal_last_n=0,
                 film_temporal_last_kernel_size=3,
                 z_film_alpha_init=-2.0,
                 z_skip_gate_init=-2.0,
                 use_z_skip=True,
                 decoder_cross_attn_gate_init=-1.5,
                 use_dual_output_heads=False,
                 output_head_hidden_dim=None,
                 use_detached_variance_pathway=False,
                 variance_detach_start_epoch=None,
                 variance_path_use_latent=True,
                 variance_path_detach_latent=False,
                 variance_head_hidden_dim=None,
                 variance_path_use_mask=False,
                 variance_mask_dim=32,
                 variance_use_grouped_conv=False,
                 use_local_context_map=False,
                 local_context_dim=32,
                 local_context_steps=None,
                 local_context_gate_init=-2.0,
                 local_context_observe_aware=False,
                 local_context_observe_aware_blend_gate_init=None,
                 local_context_injection_mode='seed',
                 local_context_fusion_mode='add',
                 local_context_attn_heads=4,
                 local_context_attn_window_tokens=1,
                 local_context_attn_gate_init=-2.0,
                 local_context_attn_after_tcn_layers=None,
                 local_context_attn_location='mid_tcn',
                 local_context_attn_support_bias_scale=2.0,
                 local_context_attn_gate_support_power=0.0,
                 local_context_attn_gate_support_floor=0.0,
                 local_context_attn_logvar_support_boost=0.0,
                 use_pregraph_feature_temporal_attn=False,
                 use_pregraph_depthwise_tcn=True,
                 use_axial_observed_attn=False,
                 axial_attn_dim=64,
                 axial_attn_heads=4,
                 axial_time_gate_init=0.0,
                 axial_cross_gate_init=0.0,
                 axial_cross_time_chunk=4,
                 axial_null_output=False,
                 pregraph_feature_temporal_attn_dim=64,
                 pregraph_feature_temporal_attn_heads=4,
                 pregraph_feature_temporal_attn_gate_init=-1.0,
                 pregraph_feature_temporal_attn_chunk_size=256,
                 pregraph_feature_temporal_attn_record_weights=False,
                 pregraph_feature_temporal_attn_mode='dense',
                 use_temporal_refiner=False,
                 temporal_refiner_dim=128,
                 temporal_refiner_heads=4,
                 temporal_refiner_gate_init=-2.0,
                 temporal_refiner_fixed_gate=None,
                 use_decoder_final_norm=False,
                 use_latent_pooled_norm=False,
                 latent_logvar_min=None,
                 latent_logvar_max=None,
                 enable_cross_modal_floor=False,
                 disable_rel_scale=False, disable_prior_bias=False, disable_aux_bias=False,
                 use_homogeneous=False, ignore_obs_mask=False,
                 use_graph_ffn=False, graph_ffn_mult=4,
                 use_token_graph_trunk=False, token_graph_dim=None,
                 token_graph_out_gate_init=-1.0,
                 use_local_chunk_graph=False,
                 local_chunk_graph_mode='parallel',
                 local_chunk_graph_chunk_size=6,
                 local_chunk_graph_dim=128,
                 local_chunk_graph_heads=4,
                 local_chunk_graph_gate_init=-2.0,
                 local_chunk_graph_ffn_mult=4,
                 local_chunk_graph_use_mask_embed=False,
                 local_chunk_graph_out_proj_init_std=0.0,
                 use_external_history_context=False,
                 external_history_dim=128,
                 external_history_heads=4,
                 external_history_gate_init=-2.0,
                 external_history_use_retrieval_bias=False,
                 external_history_time_decay=0.0,
                 external_history_support_bias=0.0,
                 external_history_null_penalty=0.0,
                 history_num_chunks=28,
                 history_chunk_size=24,
                 history_support_dim=6,
                 use_variance_attn_support=False,
                 variance_attn_support_dim=32,
                 use_support_logvar_residual=False,
                 support_logvar_hidden_dim=32,
                 support_logvar_missing_only=True,
                 support_logvar_monotone=False,
                 support_logvar_monotone_init=-2.0,
                 support_logvar_use_anchor=False,
                 support_logvar_anchor_init=-2.0,
                 use_feature_logvar_bias=False,
                 feature_logvar_bias_scope='psd',
                 feature_logvar_bias_init=0.0,
                 feature_logvar_bias_constraint='none',
                 use_learnable_likelihood_df=False,
                 likelihood_df_scope='family',
                 likelihood_df_init=3.0,
                 likelihood_df_min=2.1,
                 likelihood_df_max=30.0,
                 use_learnable_prior_df=False,
                 prior_df_init=3.0,
                 prior_df_min=2.1,
                 prior_df_max=30.0):
        super().__init__()
        self.ignore_obs_mask = ignore_obs_mask

        self.use_realnvp = use_realnvp
        if use_realnvp:
            self.flow = RealNVP(latent_dim, n_layers=realnvp_layers, hidden_dim=max(64, latent_dim))
        else:
            self.flow = None
        
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.heteroscedastic = heteroscedastic
        self.n_chem = n_chem
        self.use_hybrid_time_encoding = use_hybrid_time_encoding
        self.time_numeric_dim = time_numeric_dim
        self.time_cyc_dim = time_cyc_dim
        self.film_kernel_size = film_kernel_size
        self.film_gamma_kernel_size = film_gamma_kernel_size
        self.film_beta_kernel_size = film_beta_kernel_size
        self.film_temporal_last_n = max(0, int(film_temporal_last_n))
        self.film_temporal_last_kernel_size = int(film_temporal_last_kernel_size)
        self.use_latent_pooled_norm = bool(use_latent_pooled_norm)
        self.latent_logvar_min = latent_logvar_min
        self.latent_logvar_max = latent_logvar_max
        self.use_decoder_final_norm = bool(use_decoder_final_norm)
        self.use_z_skip = bool(use_z_skip)
        self.use_graph_ffn = bool(use_graph_ffn)
        self.graph_ffn_mult = int(graph_ffn_mult)
        self.use_token_graph_trunk = bool(use_token_graph_trunk)
        self.token_graph_dim = (
            None if token_graph_dim is None else int(token_graph_dim)
        )
        self.token_graph_out_gate_init = float(token_graph_out_gate_init)
        self.use_local_chunk_graph = bool(use_local_chunk_graph)
        self.local_chunk_graph_mode = str(local_chunk_graph_mode)
        self.local_chunk_graph_chunk_size = int(local_chunk_graph_chunk_size)
        self.local_chunk_graph_dim = int(local_chunk_graph_dim)
        self.local_chunk_graph_heads = int(local_chunk_graph_heads)
        self.local_chunk_graph_gate_init = float(local_chunk_graph_gate_init)
        self.local_chunk_graph_ffn_mult = int(local_chunk_graph_ffn_mult)
        self.local_chunk_graph_use_mask_embed = bool(local_chunk_graph_use_mask_embed)
        self.local_chunk_graph_out_proj_init_std = float(local_chunk_graph_out_proj_init_std)
        self.use_variance_attn_support = bool(use_variance_attn_support)
        self.variance_attn_support_dim = int(variance_attn_support_dim)
        self.use_support_logvar_residual = bool(use_support_logvar_residual)
        self.support_logvar_hidden_dim = int(support_logvar_hidden_dim)
        self.support_logvar_missing_only = bool(support_logvar_missing_only)
        self.support_logvar_monotone = bool(support_logvar_monotone)
        self.support_logvar_monotone_init = float(support_logvar_monotone_init)
        self.support_logvar_use_anchor = bool(support_logvar_use_anchor)
        self.support_logvar_anchor_init = float(support_logvar_anchor_init)
        self.use_feature_logvar_bias = bool(use_feature_logvar_bias)
        self.feature_logvar_bias_scope = str(feature_logvar_bias_scope)
        self.feature_logvar_bias_init = float(feature_logvar_bias_init)
        self.feature_logvar_bias_constraint = str(feature_logvar_bias_constraint)
        self.use_learnable_likelihood_df = bool(use_learnable_likelihood_df)
        self.likelihood_df_scope = str(likelihood_df_scope)
        if self.likelihood_df_scope not in {'family', 'feature'}:
            raise ValueError("likelihood_df_scope must be one of {'family', 'feature'}")
        self.likelihood_df_init = float(likelihood_df_init)
        self.likelihood_df_min = float(likelihood_df_min)
        self.likelihood_df_max = float(likelihood_df_max)
        if self.likelihood_df_min <= 2.0:
            raise ValueError("likelihood_df_min must be > 2.0 for finite Student-t variance")
        if self.likelihood_df_max <= self.likelihood_df_min:
            raise ValueError("likelihood_df_max must be greater than likelihood_df_min")
        self.use_learnable_prior_df = bool(use_learnable_prior_df)
        self.prior_df_init = float(prior_df_init)
        self.prior_df_min = float(prior_df_min)
        self.prior_df_max = float(prior_df_max)
        if self.prior_df_min <= 2.0:
            raise ValueError("prior_df_min must be > 2.0 for finite Student-t variance")
        if self.prior_df_max <= self.prior_df_min:
            raise ValueError("prior_df_max must be greater than prior_df_min")
        self.variance_path_detach_latent = bool(variance_path_detach_latent)
        self.use_local_context_map = bool(use_local_context_map)
        self.local_context_dim = int(local_context_dim)
        self.local_context_steps = (
            decoder_initial_steps if local_context_steps is None else int(local_context_steps)
        )
        self.local_context_gate_init = float(local_context_gate_init)
        self.local_context_observe_aware = bool(local_context_observe_aware)
        self.local_context_observe_aware_blend_gate_init = (
            None
            if local_context_observe_aware_blend_gate_init is None
            else float(local_context_observe_aware_blend_gate_init)
        )
        self.local_context_injection_mode = str(local_context_injection_mode)
        self.local_context_fusion_mode = str(local_context_fusion_mode)
        self.local_context_attn_heads = int(local_context_attn_heads)
        self.local_context_attn_window_tokens = int(local_context_attn_window_tokens)
        self.local_context_attn_gate_init = float(local_context_attn_gate_init)
        self.local_context_attn_after_tcn_layers = (
            None if local_context_attn_after_tcn_layers is None else int(local_context_attn_after_tcn_layers)
        )
        self.local_context_attn_location = str(local_context_attn_location)
        self.local_context_attn_support_bias_scale = float(local_context_attn_support_bias_scale)
        self.local_context_attn_gate_support_power = max(
            0.0, float(local_context_attn_gate_support_power)
        )
        self.local_context_attn_gate_support_floor = min(
            1.0, max(0.0, float(local_context_attn_gate_support_floor))
        )
        self.local_context_attn_logvar_support_boost = max(
            0.0, float(local_context_attn_logvar_support_boost)
        )
        self.use_external_history_context = bool(use_external_history_context)
        self.external_history_dim = int(external_history_dim)
        self.external_history_heads = int(external_history_heads)
        self.external_history_gate_init = float(external_history_gate_init)
        self.external_history_use_retrieval_bias = bool(external_history_use_retrieval_bias)
        self.external_history_time_decay = float(external_history_time_decay)
        self.external_history_support_bias = float(external_history_support_bias)
        self.external_history_null_penalty = float(external_history_null_penalty)
        self.history_num_chunks = int(history_num_chunks)
        self.history_chunk_size = int(history_chunk_size)
        self.history_support_dim = int(history_support_dim)
        self.use_pregraph_feature_temporal_attn = bool(use_pregraph_feature_temporal_attn)
        self.use_pregraph_depthwise_tcn = bool(use_pregraph_depthwise_tcn)
        self.use_axial_observed_attn = bool(use_axial_observed_attn)
        self.axial_attn_dim = int(axial_attn_dim)
        self.axial_attn_heads = int(axial_attn_heads)
        self.axial_time_gate_init = float(axial_time_gate_init)
        self.axial_cross_gate_init = float(axial_cross_gate_init)
        self.axial_cross_time_chunk = int(axial_cross_time_chunk)
        self.axial_null_output = bool(axial_null_output)
        self.pregraph_feature_temporal_attn_dim = int(pregraph_feature_temporal_attn_dim)
        self.pregraph_feature_temporal_attn_heads = int(pregraph_feature_temporal_attn_heads)
        self.pregraph_feature_temporal_attn_gate_init = float(pregraph_feature_temporal_attn_gate_init)
        self.pregraph_feature_temporal_attn_chunk_size = int(pregraph_feature_temporal_attn_chunk_size)
        self.pregraph_feature_temporal_attn_record_weights = bool(pregraph_feature_temporal_attn_record_weights)
        self.pregraph_feature_temporal_attn_mode = str(pregraph_feature_temporal_attn_mode)
        self.use_temporal_refiner = bool(use_temporal_refiner)
        self.temporal_refiner_dim = int(temporal_refiner_dim)
        self.temporal_refiner_heads = int(temporal_refiner_heads)
        self.temporal_refiner_gate_init = float(temporal_refiner_gate_init)
        self.temporal_refiner_fixed_gate = (
            None if temporal_refiner_fixed_gate is None else float(temporal_refiner_fixed_gate)
        )
        self.current_epoch = -1

        self.likelihood_df_raw = None
        if self.use_learnable_likelihood_df:
            if self.likelihood_df_scope == 'family':
                df_shape = (2,)
            else:
                df_shape = (target_dim,)
            init = min(max(self.likelihood_df_init, self.likelihood_df_min + 1e-4), self.likelihood_df_max - 1e-4)
            ratio = (init - self.likelihood_df_min) / (self.likelihood_df_max - self.likelihood_df_min)
            raw_init = float(np.log(ratio / (1.0 - ratio)))
            self.likelihood_df_raw = nn.Parameter(torch.full(df_shape, raw_init, dtype=torch.float32))

        self.prior_df_raw = None
        if self.use_learnable_prior_df:
            init = min(max(self.prior_df_init, self.prior_df_min + 1e-4), self.prior_df_max - 1e-4)
            ratio = (init - self.prior_df_min) / (self.prior_df_max - self.prior_df_min)
            raw_init = float(np.log(ratio / (1.0 - ratio)))
            self.prior_df_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # Keep cond dimensionality stable by default (time_hybrid_dim=6).
        self.time_hybrid_encoder = None
        if use_hybrid_time_encoding and aux_dim >= (time_numeric_dim + time_cyc_dim):
            self.time_hybrid_encoder = TimeHybridEncoder(
                out_dim=time_hybrid_dim,
                hour_embed_dim=hour_embed_dim,
                dow_embed_dim=dow_embed_dim,
                month_embed_dim=month_embed_dim,
                dropout=dropout,
            )
        
        # Input: target + aux (no raw mask channels — use learned embedding instead)
        input_dim = target_dim + aux_dim
        
        # Vectorized mask embedding: single Embedding(2, target_dim)
        # Row 0 = offset for missing positions, Row 1 = offset for observed positions
        # Replaces raw binary mask channels with gradient-controllable signals
        self.mask_embed = nn.Embedding(2, target_dim)
        nn.init.normal_(self.mask_embed.weight, mean=0.0, std=0.01)
        
        # Encoder with Graph Layer (Input-Level)
        self.encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            window_size=window_size,
            num_layers=encoder_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            n_graph_heads=n_graph_heads,
            target_dim=target_dim,
            aux_dim=aux_dim,
            use_input_graph_layer=use_input_graph_layer,
            use_cross_modal_graph=use_cross_modal_graph,
            use_tcn=use_tcn,
            n_input_graph_layers=n_input_graph_layers,
            use_parallel_graph=use_parallel_graph,
            use_temporal_cnn=use_temporal_cnn,
            n_chem=n_chem,
            enable_cross_modal_floor=enable_cross_modal_floor,
            disable_rel_scale=disable_rel_scale,
            disable_prior_bias=disable_prior_bias,
            disable_aux_bias=disable_aux_bias,
            use_homogeneous=use_homogeneous,
            ignore_obs_mask=self.ignore_obs_mask,
            use_graph_ffn=self.use_graph_ffn,
            graph_ffn_mult=self.graph_ffn_mult,
            use_token_graph_trunk=self.use_token_graph_trunk,
            token_graph_dim=self.token_graph_dim,
            token_graph_out_gate_init=self.token_graph_out_gate_init,
            use_local_chunk_graph=self.use_local_chunk_graph,
            local_chunk_graph_mode=self.local_chunk_graph_mode,
            local_chunk_graph_chunk_size=self.local_chunk_graph_chunk_size,
            local_chunk_graph_dim=self.local_chunk_graph_dim,
            local_chunk_graph_heads=self.local_chunk_graph_heads,
            local_chunk_graph_gate_init=self.local_chunk_graph_gate_init,
            local_chunk_graph_ffn_mult=self.local_chunk_graph_ffn_mult,
            local_chunk_graph_use_mask_embed=self.local_chunk_graph_use_mask_embed,
            local_chunk_graph_out_proj_init_std=self.local_chunk_graph_out_proj_init_std,
            use_latent_pooled_norm=self.use_latent_pooled_norm,
            latent_logvar_min=self.latent_logvar_min,
            latent_logvar_max=self.latent_logvar_max,
            use_local_context_map=self.use_local_context_map,
            local_context_dim=self.local_context_dim,
            local_context_steps=self.local_context_steps,
            local_context_observe_aware=self.local_context_observe_aware,
            local_context_observe_aware_blend_gate_init=(
                self.local_context_observe_aware_blend_gate_init
            ),
            use_pregraph_feature_temporal_attn=self.use_pregraph_feature_temporal_attn,
            use_pregraph_depthwise_tcn=self.use_pregraph_depthwise_tcn,
            use_axial_observed_attn=self.use_axial_observed_attn,
            axial_attn_dim=self.axial_attn_dim,
            axial_attn_heads=self.axial_attn_heads,
            axial_time_gate_init=self.axial_time_gate_init,
            axial_cross_gate_init=self.axial_cross_gate_init,
            axial_cross_time_chunk=self.axial_cross_time_chunk,
            axial_null_output=self.axial_null_output,
            pregraph_feature_temporal_attn_dim=self.pregraph_feature_temporal_attn_dim,
            pregraph_feature_temporal_attn_heads=self.pregraph_feature_temporal_attn_heads,
            pregraph_feature_temporal_attn_gate_init=self.pregraph_feature_temporal_attn_gate_init,
            pregraph_feature_temporal_attn_chunk_size=self.pregraph_feature_temporal_attn_chunk_size,
            pregraph_feature_temporal_attn_record_weights=self.pregraph_feature_temporal_attn_record_weights,
            pregraph_feature_temporal_attn_mode=self.pregraph_feature_temporal_attn_mode,
            use_temporal_refiner=self.use_temporal_refiner,
            temporal_refiner_dim=self.temporal_refiner_dim,
            temporal_refiner_heads=self.temporal_refiner_heads,
            temporal_refiner_gate_init=self.temporal_refiner_gate_init,
            temporal_refiner_fixed_gate=self.temporal_refiner_fixed_gate,
        )

        # Decoder with separate variance for Chem/PSD
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            aux_dim=aux_dim,
            hidden_dims=hidden_dims,
            output_dim=target_dim,
            window_size=window_size,
            num_layers=decoder_layers,
            var_min=var_min,
            var_max=var_max,
            kernel_size=kernel_size,
            dropout=dropout,
            heteroscedastic=heteroscedastic,
            n_chem=n_chem,
            use_tcn=use_tcn,
            use_progressive_decoder=use_progressive_decoder,
            decoder_initial_steps=decoder_initial_steps,
            cond_film_last_n=cond_film_last_n,
            cond_film_gamma_scale=cond_film_gamma_scale,
            use_decoder_cross_attn=use_decoder_cross_attn,
            n_cross_attn_heads=n_cross_attn_heads,
            decoder_cross_attn_missing_only=decoder_cross_attn_missing_only,
            film_kernel_size=film_kernel_size,
            film_gamma_kernel_size=film_gamma_kernel_size,
            film_beta_kernel_size=film_beta_kernel_size,
            film_temporal_last_n=film_temporal_last_n,
            film_temporal_last_kernel_size=film_temporal_last_kernel_size,
            z_film_alpha_init=z_film_alpha_init,
            z_skip_gate_init=z_skip_gate_init,
            use_z_skip=self.use_z_skip,
            decoder_cross_attn_gate_init=decoder_cross_attn_gate_init,
            use_dual_output_heads=use_dual_output_heads,
            output_head_hidden_dim=output_head_hidden_dim,
            use_detached_variance_pathway=use_detached_variance_pathway,
            variance_detach_start_epoch=variance_detach_start_epoch,
            variance_path_use_latent=variance_path_use_latent,
            variance_path_detach_latent=self.variance_path_detach_latent,
            variance_head_hidden_dim=variance_head_hidden_dim,
            variance_path_use_mask=variance_path_use_mask,
            variance_mask_dim=variance_mask_dim,
            variance_use_grouped_conv=variance_use_grouped_conv,
            use_local_context_map=self.use_local_context_map,
            local_context_dim=self.local_context_dim,
            local_context_steps=self.local_context_steps,
            local_context_gate_init=self.local_context_gate_init,
            local_context_injection_mode=self.local_context_injection_mode,
            local_context_fusion_mode=self.local_context_fusion_mode,
            local_context_attn_heads=self.local_context_attn_heads,
            local_context_attn_window_tokens=self.local_context_attn_window_tokens,
            local_context_attn_gate_init=self.local_context_attn_gate_init,
            local_context_attn_after_tcn_layers=self.local_context_attn_after_tcn_layers,
            local_context_attn_location=self.local_context_attn_location,
            local_context_attn_support_bias_scale=self.local_context_attn_support_bias_scale,
            local_context_attn_gate_support_power=self.local_context_attn_gate_support_power,
            local_context_attn_gate_support_floor=self.local_context_attn_gate_support_floor,
            local_context_attn_logvar_support_boost=self.local_context_attn_logvar_support_boost,
            use_variance_attn_support=self.use_variance_attn_support,
            variance_attn_support_n_features=target_dim,
            variance_attn_support_dim=self.variance_attn_support_dim,
            use_support_logvar_residual=self.use_support_logvar_residual,
            support_logvar_hidden_dim=self.support_logvar_hidden_dim,
            support_logvar_missing_only=self.support_logvar_missing_only,
            support_logvar_monotone=self.support_logvar_monotone,
            support_logvar_monotone_init=self.support_logvar_monotone_init,
            support_logvar_use_anchor=self.support_logvar_use_anchor,
            support_logvar_anchor_init=self.support_logvar_anchor_init,
            use_feature_logvar_bias=self.use_feature_logvar_bias,
            feature_logvar_bias_scope=self.feature_logvar_bias_scope,
            feature_logvar_bias_init=self.feature_logvar_bias_init,
            feature_logvar_bias_constraint=self.feature_logvar_bias_constraint,
            use_decoder_final_norm=use_decoder_final_norm,
            ignore_obs_mask=self.ignore_obs_mask
        )

        self.external_history_context = None
        if self.use_external_history_context:
            if not self.use_local_context_map:
                raise ValueError("use_external_history_context=True requires use_local_context_map=True")
            self.external_history_context = ExternalHistoryContext(
                target_dim=target_dim,
                cond_dim=aux_dim,
                context_dim=self.local_context_dim,
                context_steps=self.local_context_steps,
                window_size=window_size,
                history_chunk_size=self.history_chunk_size,
                history_num_chunks=self.history_num_chunks,
                history_support_dim=self.history_support_dim,
                hidden_dim=self.external_history_dim,
                n_heads=self.external_history_heads,
                dropout=dropout,
                gate_init=self.external_history_gate_init,
                use_retrieval_bias=self.external_history_use_retrieval_bias,
                time_decay=self.external_history_time_decay,
                support_bias=self.external_history_support_bias,
                null_penalty=self.external_history_null_penalty,
            )
        
        # Store last attention weights for analysis
        self.last_graph_attention = None
        self.last_graph_attention_heads = None
        self.last_cross_modal_attention = None
        self.last_cross_modal_attention_heads = None
        self.last_pregraph_feature_temporal_attn_gate = None
        self.last_pregraph_feature_temporal_attn_entropy_missing = None
        self.last_pregraph_feature_temporal_attn_entropy_observed = None
        self.last_axial_time_gate_mean = None
        self.last_axial_time_gate_chem_mean = None
        self.last_axial_time_gate_psd_mean = None
        self.last_axial_cross_gate_mean = None
        self.last_axial_cross_gate_chem_mean = None
        self.last_axial_cross_gate_psd_mean = None
        self.last_axial_time_no_key_fraction = None
        self.last_axial_cross_no_key_fraction = None
        self.last_axial_cross_valid_query_fraction = None
        self.last_axial_cross_entropy_missing = None
        self.last_axial_cross_top1_mass = None
        self.last_axial_cross_top3_mass = None
        self.last_axial_psd_to_chem_mass = None
        self.last_axial_psd_to_psd_mass = None
        self.last_local_chunk_graph_gate = None
        self.last_local_chunk_graph_out_proj_norm = None
        self.last_local_chunk_graph_obs_ratio_mean = None
        self.last_temporal_refiner_gate = None
        self.last_temporal_refiner_attn_entropy_missing = None
        self.last_temporal_refiner_attn_entropy_observed = None
        self.last_external_history_gate = None
        self.last_external_history_attn_entropy = None
        self.last_external_history_valid_fraction = None
        self.last_external_history_null_fraction = None
        self.last_external_history_top1_mass = None
        self.last_external_history_top3_mass = None
        self.last_external_history_attended_time_dist = None
        self.last_external_history_attended_support = None
        self.last_external_history_attended_null_fraction = None
        self.last_support_logvar_residual_mean = None
        self.last_support_logvar_residual_missing_mean = None
        self.last_support_logvar_residual_psd_low_support_mean = None
        self.last_support_logvar_residual_psd_high_support_mean = None
        self.last_support_logvar_monotone_beta = None

    def get_likelihood_df(self, num_features=None, device=None, dtype=None):
        """Return likelihood Student-t df as scalar/family/feature tensor.

        This controls only p(y | z, x_obs, cond). The latent Student-t prior df
        remains fixed in the trainer unless explicitly changed there.
        """
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else torch.float32

        if self.likelihood_df_raw is None:
            return torch.tensor(3.0, device=device, dtype=dtype)

        df_values = (
            self.likelihood_df_min
            + (self.likelihood_df_max - self.likelihood_df_min)
            * torch.sigmoid(self.likelihood_df_raw.to(device=device, dtype=dtype))
        )
        if num_features is None:
            return df_values

        if self.likelihood_df_scope == 'feature':
            if df_values.numel() != int(num_features):
                raise ValueError(
                    f"feature-scope likelihood df has {df_values.numel()} values, "
                    f"but num_features={num_features}"
                )
            return df_values

        n_chem = int(self.n_chem or 0)
        if n_chem <= 0:
            # PSD-only experiments still use family-scope df semantics: index 1
            # is the PSD family when the two-family parameterization is active.
            df_idx = 1 if df_values.numel() > 1 else 0
            return df_values[df_idx].expand(int(num_features))
        if n_chem >= int(num_features):
            return df_values[0].expand(int(num_features))
        out = torch.empty(int(num_features), device=device, dtype=dtype)
        out[:n_chem] = df_values[0]
        out[n_chem:] = df_values[1]
        return out

    def get_likelihood_df_metadata(self):
        """JSON-friendly current likelihood df values for training/inference logs."""
        with torch.no_grad():
            df = self.get_likelihood_df(device=torch.device('cpu'), dtype=torch.float32).detach().cpu()
            meta = {
                'learnable': bool(self.likelihood_df_raw is not None),
                'scope': self.likelihood_df_scope,
                'min': float(self.likelihood_df_min),
                'max': float(self.likelihood_df_max),
            }
            if self.likelihood_df_raw is None:
                meta['value'] = 3.0
            elif self.likelihood_df_scope == 'family':
                meta['chem'] = float(df[0].item())
                meta['psd'] = float(df[1].item())
            else:
                meta['mean'] = float(df.mean().item())
                meta['min_value'] = float(df.min().item())
                meta['max_value'] = float(df.max().item())
                meta['values'] = [float(x) for x in df.tolist()]
            return meta

    def get_prior_df(self, device=None, dtype=None):
        """Return latent Student-t prior df as a scalar tensor."""
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else torch.float32
        if self.prior_df_raw is None:
            return torch.tensor(3.0, device=device, dtype=dtype)
        return (
            self.prior_df_min
            + (self.prior_df_max - self.prior_df_min)
            * torch.sigmoid(self.prior_df_raw.to(device=device, dtype=dtype))
        )

    def get_prior_df_metadata(self):
        """JSON-friendly current latent prior df value for training/inference logs."""
        with torch.no_grad():
            df = self.get_prior_df(device=torch.device('cpu'), dtype=torch.float32).detach().cpu()
            return {
                'learnable': bool(self.prior_df_raw is not None),
                'min': float(self.prior_df_min),
                'max': float(self.prior_df_max),
                'value': float(df.item()),
            }
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _encode_cond_features(self, cond):
        """Optionally replace cyclical time channels with hybrid-encoded features."""
        if self.time_hybrid_encoder is None:
            return cond

        if cond.shape[-1] < (self.time_numeric_dim + self.time_cyc_dim):
            return cond

        left = cond[:, :, :self.time_numeric_dim]
        time_cyc = cond[:, :, self.time_numeric_dim:self.time_numeric_dim + self.time_cyc_dim]
        right = cond[:, :, self.time_numeric_dim + self.time_cyc_dim:]

        time_hybrid = self.time_hybrid_encoder(time_cyc)
        return torch.cat([left, time_hybrid, right], dim=-1)
    
    def enable_cross_modal_imputation(self, enable=True):
        """Toggle BERT-style cross-modal imputation mode.

        When enabled, the CrossModalGraphLayer uses a soft sigmoid gate instead
        of the hard 10% obs_rate threshold, allowing gradient flow even when a
        modality is fully masked (required for curriculum masking).
        The InputGraphLayer cross-modal floor is always active if the parameter
        was created (via enable_cross_modal_floor=True at construction).
        """
        if (hasattr(self.encoder, 'cross_modal_graph_layer')
                and self.encoder.cross_modal_graph_layer is not None):
            self.encoder.cross_modal_graph_layer.use_soft_gate = enable

    def forward(self, x, cond, mask, history=None, sample_latent=True):
        """
        Args:
            x: [Batch, Window, Target_dim] - Input features (masked)
            cond: [Batch, Window, Aux_dim] - Conditioning features
            mask: [Batch, Window, Target_dim] - Observation mask (1=observed)

        Returns:
            recon_mean: [Batch, Window, Target_dim]
            recon_logvar: [Batch, Window, Target_dim] or None
            mu: [Batch, Latent_dim]
            logvar: [Batch, Latent_dim]
            graph_attention: [Batch, C, C] - Learned feature relationships
        """
        # Apply vectorized mask embedding (additive)
        # mask: [B, W, D] with values 0/1 → embedding adds learned offset
        embed_0 = self.mask_embed(torch.zeros(1, dtype=torch.long, device=x.device))  # [1, D]
        embed_1 = self.mask_embed(torch.ones(1, dtype=torch.long, device=x.device))   # [1, D]
        embed_offset = embed_0 + mask.float() * (embed_1 - embed_0)  # [B, W, D]
        # Transpose to [B, D, W] so it matches the inner operations
        embed_offset_t = embed_offset.permute(0, 2, 1)
        
        cond = self._encode_cond_features(cond)
        history_context = None
        if self.external_history_context is not None and history is not None:
            h_aux = history.get('history_aux')
            h_hour = history.get('history_hour')
            if h_aux is not None and h_hour is not None:
                h_cond = torch.cat([h_aux, h_hour], dim=-1)
                B_h, K_h, L_h, C_h = h_cond.shape
                h_cond = self._encode_cond_features(
                    h_cond.reshape(B_h * K_h, L_h, C_h)
                ).reshape(B_h, K_h, L_h, -1)
                history = dict(history)
                history['history_cond'] = h_cond
                history_context = self.external_history_context(history, cond, mask)

        # Prepare input (target without embedding + cond)
        # Embedding will be injected safetly AFTER instance normalization inside encoder
        inputs = torch.cat([x, cond], dim=-1)  # [B, W, target_dim + aux_dim]
        inputs = inputs.permute(0, 2, 1)  # [B, input_dim, W]
        
        # Create obs_mask for encoder (target features + cond)
        # Graph attention layers extract target_mask from this internally
        B, W, _ = x.shape
        cond_dim = cond.shape[-1]
        full_obs_mask = torch.cat([
            mask,  # Target features: use actual observation mask
            torch.ones(B, W, cond_dim, device=x.device),  # Cond: always observed
        ], dim=-1)  # [B, W, target_dim + aux_dim]
        
        # Encode with observation mask (mask info reaches graph attention
        # via full_obs_mask parameter, NOT as input channels)
        mu, logvar, graph_attention, h_seq, local_context, attn_weighted_support_t = self.encoder(inputs, full_obs_mask, embed_offset=embed_offset_t)
        if history_context is not None:
            if local_context is None:
                local_context = history_context
            else:
                if history_context.shape[-1] != local_context.shape[-1]:
                    history_context = F.interpolate(
                        history_context,
                        size=local_context.shape[-1],
                        mode='linear',
                        align_corners=False,
                    )
                local_context = local_context + history_context
            self.last_external_history_gate = self.external_history_context.last_gate
            self.last_external_history_attn_entropy = self.external_history_context.last_attn_entropy
            self.last_external_history_valid_fraction = self.external_history_context.last_valid_fraction
            self.last_external_history_null_fraction = self.external_history_context.last_null_fraction
            self.last_external_history_top1_mass = self.external_history_context.last_top1_mass
            self.last_external_history_top3_mass = self.external_history_context.last_top3_mass
            self.last_external_history_attended_time_dist = self.external_history_context.last_attended_time_dist
            self.last_external_history_attended_support = self.external_history_context.last_attended_support
            self.last_external_history_attended_null_fraction = self.external_history_context.last_attended_null_fraction
        else:
            self.last_external_history_gate = None
            self.last_external_history_attn_entropy = None
            self.last_external_history_valid_fraction = None
            self.last_external_history_null_fraction = None
            self.last_external_history_top1_mass = None
            self.last_external_history_top3_mass = None
            self.last_external_history_attended_time_dist = None
            self.last_external_history_attended_support = None
            self.last_external_history_attended_null_fraction = None
        self.last_attn_weighted_support_t = attn_weighted_support_t  # [B, W, C] or None; used by inference for SRC diagnostic
        self.last_graph_attention = graph_attention.detach().mean(dim=0) if graph_attention is not None else None
        self.last_graph_attention_heads = self.encoder.last_input_graph_attention_heads
        self.last_cross_modal_attention = self.encoder.last_cross_modal_attention
        self.last_cross_modal_attention_heads = self.encoder.last_cross_modal_attention_heads
        self.last_pregraph_feature_temporal_attn_gate = getattr(
            self.encoder, 'last_pregraph_feature_temporal_attn_gate', None
        )
        self.last_pregraph_feature_temporal_attn_entropy_missing = getattr(
            self.encoder, 'last_pregraph_feature_temporal_attn_entropy_missing', None
        )
        self.last_pregraph_feature_temporal_attn_entropy_observed = getattr(
            self.encoder, 'last_pregraph_feature_temporal_attn_entropy_observed', None
        )
        for _name in (
            'last_axial_time_gate_mean',
            'last_axial_time_gate_chem_mean',
            'last_axial_time_gate_psd_mean',
            'last_axial_cross_gate_mean',
            'last_axial_cross_gate_chem_mean',
            'last_axial_cross_gate_psd_mean',
            'last_axial_time_no_key_fraction',
            'last_axial_cross_no_key_fraction',
            'last_axial_cross_valid_query_fraction',
            'last_axial_cross_entropy_missing',
            'last_axial_cross_top1_mass',
            'last_axial_cross_top3_mass',
            'last_axial_psd_to_chem_mass',
            'last_axial_psd_to_psd_mass',
        ):
            setattr(self, _name, getattr(self.encoder, _name, None))
        self.last_local_chunk_graph_gate = getattr(
            self.encoder, 'last_local_chunk_graph_gate', None
        )
        self.last_local_chunk_graph_out_proj_norm = getattr(
            self.encoder, 'last_local_chunk_graph_out_proj_norm', None
        )
        self.last_local_chunk_graph_obs_ratio_mean = getattr(
            self.encoder, 'last_local_chunk_graph_obs_ratio_mean', None
        )
        
        # Sample latent during normal VAE inference/training.  Inference
        # ablations can set sample_latent=False to isolate likelihood-only
        # predictive spread without changing the training path.
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        
        # Flow
        self.last_log_det_J = None
        self.last_z0 = z
        self.last_zK = z
        if self.use_realnvp:
            z, self.last_log_det_J = self.flow(z)
            self.last_zK = z
        
        # Decode
        self.decoder.current_epoch = self.current_epoch
        recon_mean, recon_logvar = self.decoder(
            z, cond, enc_h_seq=h_seq, obs_mask=mask, local_context=local_context,
            attn_weighted_support_t=attn_weighted_support_t,
        )
        self.last_local_context_gate = getattr(self.decoder, 'last_local_context_gate', None)
        lca = getattr(self.decoder, 'local_context_attn_fusion', None)
        self.last_local_context_attn_entropy = getattr(lca, 'last_attn_entropy', None)
        self.last_local_context_attn_center_distance = getattr(lca, 'last_attn_center_distance', None)
        self.last_local_context_attn_support_mean = getattr(lca, 'last_attn_support_mean', None)
        self.last_local_context_attn_high_support_mass = getattr(lca, 'last_attn_high_support_mass', None)
        self.last_local_context_generation_support_mean = getattr(
            self.encoder, 'last_local_context_generation_support_mean', None
        )
        self.last_local_context_observe_aware_blend_gate = getattr(
            self.encoder, 'last_local_context_observe_aware_blend_gate', None
        )
        self.last_local_context_gate_low_support_mean = getattr(lca, 'last_gate_low_support_mean', None)
        self.last_local_context_gate_high_support_mean = getattr(lca, 'last_gate_high_support_mean', None)
        self.last_support_logvar_residual_mean = getattr(
            self.decoder, 'last_support_logvar_residual_mean', None
        )
        self.last_support_logvar_residual_missing_mean = getattr(
            self.decoder, 'last_support_logvar_residual_missing_mean', None
        )
        self.last_support_logvar_residual_psd_low_support_mean = getattr(
            self.decoder, 'last_support_logvar_residual_psd_low_support_mean', None
        )
        self.last_support_logvar_residual_psd_high_support_mean = getattr(
            self.decoder, 'last_support_logvar_residual_psd_high_support_mean', None
        )
        self.last_support_logvar_monotone_beta = getattr(
            self.decoder, 'last_support_logvar_monotone_beta', None
        )
        self.last_support_logvar_anchor_beta = getattr(
            self.decoder, 'last_support_logvar_anchor_beta', None
        )
        self.last_feature_logvar_bias_mean = getattr(
            self.decoder, 'last_feature_logvar_bias_mean', None
        )
        self.last_feature_logvar_bias_chem_mean = getattr(
            self.decoder, 'last_feature_logvar_bias_chem_mean', None
        )
        self.last_feature_logvar_bias_psd_mean = getattr(
            self.decoder, 'last_feature_logvar_bias_psd_mean', None
        )
        self.last_feature_logvar_bias_psd_min = getattr(
            self.decoder, 'last_feature_logvar_bias_psd_min', None
        )
        self.last_feature_logvar_bias_psd_max = getattr(
            self.decoder, 'last_feature_logvar_bias_psd_max', None
        )
        self.last_logvar_diagnostics = getattr(
            self.decoder, 'last_logvar_diagnostics', None
        )
        self.last_temporal_refiner_gate = getattr(self.encoder, 'last_temporal_refiner_gate', None)
        self.last_temporal_refiner_attn_entropy_missing = getattr(
            self.encoder, 'last_temporal_refiner_attn_entropy_missing', None
        )
        self.last_temporal_refiner_attn_entropy_observed = getattr(
            self.encoder, 'last_temporal_refiner_attn_entropy_observed', None
        )
        
        # Reshape outputs: [B, D, W] -> [B, W, D]
        recon_mean = recon_mean.permute(0, 2, 1)
        if recon_logvar is not None:
            recon_logvar = recon_logvar.permute(0, 2, 1)
        
        return recon_mean, recon_logvar, mu, logvar, graph_attention
    
    def _enable_mc_dropout(self):
        """
        Enable MC Dropout: set only nn.Dropout modules to train mode.
        LayerNorm and other modules remain in eval mode for stable inference.
        This provides an additional source of stochasticity beyond z-sampling,
        yielding better calibrated epistemic uncertainty estimates.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def compute_uncertainty(self, x, cond, mask, n_samples=50, dist_type='gaussian', history=None,
                            return_extra_quantiles=False, return_samples=False,
                            enable_mc_dropout=True, sample_latent=True,
                            sample_likelihood=True,
                            mc_batch_size=1,
                            amp_dtype=None):
        """
        Compute epistemic and aleatoric uncertainty via MC Dropout + z-sampling.

        When `return_samples=True`, the raw generative samples tensor (shape
        [N, B, W, D]) is appended to the result tuple. This is required by the
        sample-level overlap-add aggregator (`v2_sample_pos`), which needs the
        per-MC-sample predictive draws to compute mixture-correct quantiles
        and law-of-total-variance across overlapping windows.

        mc_batch_size: int, default 1.
            Number of MC samples drawn per forward via batch-dim replication.
            Set higher (e.g. 8-32) to amortize Python / kernel-launch overhead
            on GPU. Memory scales linearly: each chunked forward uses
            mc_batch_size x the activation memory of one serial forward.
            Semantically equivalent to mc_batch_size=1 because PyTorch dropout
            produces independent masks per replicated batch element.

        amp_dtype: torch.dtype | None, default None.
            If not None and CUDA is in use, the model forward runs under
            `torch.autocast(dtype=amp_dtype)` (typically `torch.bfloat16`).
            CRITICAL: the autocast scope is intentionally NARROW — it covers
            only the forward, then `recon_mean` / `recon_logvar` are immediately
            cast back to float32 for the likelihood sampling, variance
            accumulation, and quantile reductions. This protects calibration
            (CRPS / PICP / interval endpoints) from bf16 reduction error while
            still letting the matmul/conv-heavy forward use Tensor Cores.
        """
        # Set model to eval, then selectively enable dropout for MC Dropout.
        # The extra flags are inference-only ablation controls:
        #   enable_mc_dropout=False: deterministic weights, no dropout masks
        #   sample_latent=False: use z=mu instead of VAE reparameterization
        #   sample_likelihood=False: use decoder mean as the predictive sample
        self.eval()
        if enable_mc_dropout:
            self._enable_mc_dropout()

        if int(mc_batch_size) < 1:
            raise ValueError(f"mc_batch_size must be >= 1, got {mc_batch_size}")
        mc_batch_size = min(int(mc_batch_size), int(n_samples))

        B = x.shape[0]

        all_means = []
        all_logvars = []
        all_samples = []  # Full generative samples
        # Prepare running stats for graph attention to save memory
        fn_graph_attn_sum = None

        def _replicate_history(h, k):
            """Block-replicate batch dim of every tensor in a history dict by k."""
            if h is None or k == 1:
                return h
            out = {}
            for key, v in h.items():
                if isinstance(v, torch.Tensor):
                    out[key] = v.repeat(k, *([1] * (v.dim() - 1)))
                else:
                    out[key] = v
            return out

        n_emitted = 0
        while n_emitted < n_samples:
            cur_mc = min(mc_batch_size, n_samples - n_emitted)

            if cur_mc == 1:
                x_in, cond_in, mask_in = x, cond, mask
                history_in = history
            else:
                # Block-replicate: [x[0], x[1], ..., x[B-1], x[0], x[1], ...]
                # so a subsequent view(cur_mc, B, ...) recovers [MC, B, ...] order.
                x_in    = x.repeat(cur_mc, *([1] * (x.dim() - 1)))
                cond_in = cond.repeat(cur_mc, *([1] * (cond.dim() - 1)))
                mask_in = mask.repeat(cur_mc, *([1] * (mask.dim() - 1)))
                history_in = _replicate_history(history, cur_mc)

            # Narrow autocast: covers forward only. Subsequent likelihood
            # sampling, variance accumulation, and quantile reductions stay
            # in fp32 to protect calibration (CRPS / PICP / interval endpoints)
            # from bf16/fp16 reduction error.
            use_amp = (
                amp_dtype is not None
                and torch.cuda.is_available()
                and x.is_cuda
            )
            fwd_ctx = (
                torch.autocast(device_type='cuda', dtype=amp_dtype)
                if use_amp else nullcontext()
            )

            with torch.no_grad():
                with fwd_ctx:
                    val = self.forward(
                        x_in, cond_in, mask_in, history=history_in,
                        sample_latent=sample_latent,
                    )
                recon_mean, recon_logvar, _, _, graph_attn = val[0], val[1], val[2], val[3], val[4]

                # Force fp32 for stable post-forward math (sampling / variance /
                # quantile reductions). graph_attn is only used for a running
                # sum that becomes a mean, so we let it stay in its native
                # dtype to save one cast.
                if recon_mean.dtype != torch.float32:
                    recon_mean = recon_mean.float()
                if recon_logvar is not None and recon_logvar.dtype != torch.float32:
                    recon_logvar = recon_logvar.float()

                # --- Generative Sampling Step ---
                # Draw one sample from the predicted distribution p(y|x, z_i).
                # All ops are element-wise / broadcast over leading batch dims,
                # so the same logic works for both [B, W, D] and [cur_mc*B, W, D].
                if recon_logvar is None:
                    y_sample = recon_mean  # Fallback if non-heteroscedastic
                elif not sample_likelihood:
                    y_sample = recon_mean
                elif dist_type == 'student_t':
                    df = self.get_likelihood_df(
                        recon_mean.shape[-1],
                        device=recon_mean.device,
                        dtype=recon_mean.dtype,
                    ).view(1, 1, -1)
                    df_full = df.expand_as(recon_mean)
                    variance = torch.exp(recon_logvar)
                    sigma = torch.sqrt((variance * (df - 2.0) / df).clamp(min=1e-10))
                    # Sample from Standard Student-T(df) via Gamma/Normal mixture
                    chi2 = 2.0 * torch._standard_gamma((df_full / 2.0).clamp(min=1e-6))
                    t_eps = torch.randn_like(recon_mean) * torch.sqrt(df_full / chi2.clamp(min=1e-10))
                    y_sample = recon_mean + sigma * t_eps
                else:
                    # Gaussian sampling: y = mu + sigma * eps
                    std = torch.exp(0.5 * recon_logvar)
                    y_sample = recon_mean + std * torch.randn_like(recon_mean)

            # Reshape chunked outputs back to per-MC-sample order and append.
            if cur_mc == 1:
                all_means.append(recon_mean)
                if recon_logvar is not None:
                    all_logvars.append(recon_logvar)
                all_samples.append(y_sample)
                if graph_attn is not None:
                    if fn_graph_attn_sum is None:
                        fn_graph_attn_sum = graph_attn.clone()
                    else:
                        fn_graph_attn_sum += graph_attn
            else:
                # [cur_mc*B, ...] -> [cur_mc, B, ...] via the block-replicated layout.
                rm_split = recon_mean.view(cur_mc, B, *recon_mean.shape[1:])
                ys_split = y_sample.view(cur_mc, B, *y_sample.shape[1:])
                rl_split = (
                    recon_logvar.view(cur_mc, B, *recon_logvar.shape[1:])
                    if recon_logvar is not None else None
                )
                for j in range(cur_mc):
                    all_means.append(rm_split[j])
                    all_samples.append(ys_split[j])
                    if rl_split is not None:
                        all_logvars.append(rl_split[j])
                if graph_attn is not None:
                    ga_split = graph_attn.view(cur_mc, B, *graph_attn.shape[1:])
                    ga_chunk_sum = ga_split.sum(dim=0)
                    if fn_graph_attn_sum is None:
                        fn_graph_attn_sum = ga_chunk_sum
                    else:
                        fn_graph_attn_sum += ga_chunk_sum

            n_emitted += cur_mc
        
        # Restore full eval mode after sampling
        self.eval()
        
        # Stack samples for prediction stats
        means = torch.stack(all_means, dim=0)    # [N, B, W, D]
        samples = torch.stack(all_samples, dim=0) # [N, B, W, D]
        
        # Epistemic: variance and quantiles across samples (predicted means)
        epistemic_var = means.var(dim=0)  # [B, W, D]
        pred_mean = means.mean(dim=0)    # [B, W, D]
        epi_q05 = torch.quantile(means, 0.05, dim=0)  # [B, W, D]
        epi_q95 = torch.quantile(means, 0.95, dim=0)  # [B, W, D]
        epi_q025 = torch.quantile(means, 0.025, dim=0)  # [B, W, D]
        epi_q975 = torch.quantile(means, 0.975, dim=0)  # [B, W, D]
        
        # Quantile-based prediction intervals (5th and 95th percentiles)
        # IMPROVEMENT: Calculate from SAMPLES (generative) instead of MEANS (epistemic only)
        pred_q05 = torch.quantile(samples, 0.05, dim=0)  # [B, W, D]
        pred_q95 = torch.quantile(samples, 0.95, dim=0)  # [B, W, D]
        pred_q025 = torch.quantile(samples, 0.025, dim=0)  # [B, W, D]
        pred_q975 = torch.quantile(samples, 0.975, dim=0)  # [B, W, D]
        
        # Aleatoric: average predicted variance
        aleatoric_var = None
        if all_logvars:
            logvars = torch.stack(all_logvars, dim=0)
            # logvar = log(variance) for both Gaussian and Student-t
            aleatoric_var = torch.exp(logvars).mean(dim=0)
        
        # Total uncertainty
        total_var = epistemic_var + (aleatoric_var if aleatoric_var is not None else 0)
        
        # Average graph attention (handle None for no-graph models)
        if fn_graph_attn_sum is not None:
            avg_graph_attn = fn_graph_attn_sum / n_samples  # [B, C, C]
            # KEY FIX: Update the stored attention for visualization to be the AVERAGE, not the last sample
            self.last_graph_attention = avg_graph_attn.detach().mean(dim=0) # [C, C]
        else:
            avg_graph_attn = None
            self.last_graph_attention = None
        
        result = (pred_mean, epistemic_var, aleatoric_var, total_var, avg_graph_attn, pred_q05, pred_q95, epi_q05, epi_q95)
        if return_extra_quantiles:
            result = result + (pred_q025, pred_q975, epi_q025, epi_q975)
        if return_samples:
            # samples: [N, B, W, D] generative draws; means: [N, B, W, D] per-MC predicted means
            result = result + (samples, means)
        return result

    def get_learned_graph(self):
        """Return the learned feature relationship matrix."""
        if self.last_graph_attention is not None:
            return self.last_graph_attention.cpu().numpy()
        return None

    def get_learned_graph_heads(self):
        """Return per-head learned feature relationship matrices."""
        if self.last_graph_attention_heads is not None:
            return self.last_graph_attention_heads.cpu().numpy()
        return None
    
    def get_cross_modal_graph(self):
        """Return the learned target-aux cross-modal attention matrix."""
        if self.last_cross_modal_attention is not None:
            return self.last_cross_modal_attention.cpu().numpy()
        return None
    
    def get_cross_modal_graph_heads(self):
        """Return per-head target-aux cross-modal attention matrices."""
        if self.last_cross_modal_attention_heads is not None:
            return self.last_cross_modal_attention_heads.cpu().numpy()
        return None

    def get_learned_graph_heads_batch(self):
        """Return per-batch per-head learned feature relationship tensor [B, h, C, C]."""
        if hasattr(self.encoder, 'last_input_graph_attention_heads_batch') and self.encoder.last_input_graph_attention_heads_batch is not None:
            return self.encoder.last_input_graph_attention_heads_batch.cpu().numpy()
        return None

    def get_cross_modal_graph_heads_batch(self):
        """Return per-batch per-head cross-modal attention tensor [B, h, C_t, C_a]."""
        if hasattr(self.encoder, 'last_cross_modal_attention_heads_batch') and self.encoder.last_cross_modal_attention_heads_batch is not None:
            return self.encoder.last_cross_modal_attention_heads_batch.cpu().numpy()
        return None
        
    def get_learned_graph_batch(self):
        """Return the per-batch learned feature relationship tensor [B, C, C]."""
        if hasattr(self.encoder, 'last_input_graph_attention_batch') and self.encoder.last_input_graph_attention_batch is not None:
            return self.encoder.last_input_graph_attention_batch.cpu().numpy()
        return None

    def get_cross_modal_graph_batch(self):
        """Return the per-batch target-aux cross-modal attention tensor [B, C_t, C_a]."""
        if hasattr(self.encoder, 'last_cross_modal_attention_batch') and self.encoder.last_cross_modal_attention_batch is not None:
            return self.encoder.last_cross_modal_attention_batch.cpu().numpy()
        return None

    def get_learned_graph_per_layer(self):
        """Return attention matrices from each stacked InputGraphLayer.
        
        Returns:
            list of dicts, each containing:
                'avg': [C, C] mean attention
                'heads': [h, C, C] per-head attention
            Returns None if no stacked layers.
        """
        if not hasattr(self.encoder, 'last_input_graph_attention_per_layer'):
            return None
        per_layer = self.encoder.last_input_graph_attention_per_layer
        if not per_layer:
            return None
        result = []
        for layer_data in per_layer:
            entry = {
                'avg': layer_data['avg'].cpu().numpy() if layer_data['avg'] is not None else None,
                'heads': layer_data['heads'].cpu().numpy() if layer_data['heads'] is not None else None,
            }
            result.append(entry)
        return result


def load_from_uq_model(graph_model, uq_model_path, device):
    """
    Initialize graph model from pretrained UQ model weights (where possible).
    """
    uq_state = torch.load(uq_model_path, map_location=device)
    graph_state = graph_model.state_dict()
    
    # Copy matching weights
    copied = 0
    for name, param in uq_state.items():
        if name in graph_state and graph_state[name].shape == param.shape:
            graph_state[name] = param
            copied += 1
    
    graph_model.load_state_dict(graph_state)
    print(f"📦 Loaded {copied} layers from UQ model")
    
    return graph_model
