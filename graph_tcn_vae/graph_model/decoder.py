"""Graph-temporal decoder used by :class:`ImputationVAE_Graph`.

The implementation is kept behavior- and checkpoint-compatible with the
legacy public ``model_graph_uq.GraphDecoder`` while the monolith is reduced to
a compatibility facade.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations as param

from ..graph_blocks.decoder_attention import (
    LocalContextMemoryAttention as _SplitLocalContextMemoryAttention,
    MaskedTemporalCrossAttention as _SplitMaskedTemporalCrossAttention,
)


class GraphDecoder(nn.Module):
    """TCN decoder with progressive upsampling, FiLM, and uncertainty heads."""

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
        self.film_gamma_kernel_size = (
            film_gamma_kernel_size if film_gamma_kernel_size is not None else film_kernel_size
        )
        self.film_beta_kernel_size = (
            film_beta_kernel_size if film_beta_kernel_size is not None else film_kernel_size
        )
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
        self.local_context_steps = None if local_context_steps is None else int(local_context_steps)
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
                "local_context_injection_mode must be one of {'seed', 'post_upsample', 'both'}"
            )
        if self.local_context_fusion_mode not in {'add', 'attn'}:
            raise ValueError("local_context_fusion_mode must be one of {'add', 'attn'}")
        if self.local_context_fusion_mode == 'attn' and not use_progressive_decoder:
            raise ValueError("local_context_fusion_mode='attn' requires use_progressive_decoder=True")
        if self.local_context_attn_location not in {'mid_tcn', 'upsample', 'both'}:
            raise ValueError(
                "local_context_attn_location must be one of {'mid_tcn', 'upsample', 'both'}"
            )

        hidden = hidden_dims[-1]
        self.hidden_dim = hidden
        self.aux_dim = int(aux_dim)

        if use_progressive_decoder:
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

            self.upsample_blocks = nn.ModuleList()
            current_steps = self.initial_steps
            while current_steps < window_size:
                self.upsample_blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    param.weight_norm(nn.Conv1d(hidden, hidden, 3, padding=1)),
                    nn.GELU(),
                ))
                current_steps *= 2
            self.final_upsample = nn.Upsample(
                size=window_size, mode='linear', align_corners=False
            )
            if self.aux_dim > 0:
                self.cond_proj = nn.Sequential(
                    nn.Conv1d(self.aux_dim, 64, 1),
                    nn.GELU(),
                    nn.Conv1d(64, hidden, 1),
                )
            else:
                self.cond_proj = None
            self.film_gammas = nn.ModuleList()
            self.film_betas = nn.ModuleList()
            self.z_film_gammas = nn.ModuleList()
            self.z_film_betas = nn.ModuleList()
            self.dec_norms = nn.ModuleList([
                nn.LayerNorm(hidden, elementwise_affine=False) for _ in range(num_layers)
            ])
            self.z_film_alpha = nn.Parameter(
                torch.full((num_layers,), self.z_film_alpha_init)
            )
            self.residual_alpha = nn.Parameter(torch.full((num_layers,), -1.38))
            if local_context_attn_after_tcn_layers is None:
                self.local_context_attn_after_tcn_layers = num_layers // 2
            else:
                self.local_context_attn_after_tcn_layers = int(
                    local_context_attn_after_tcn_layers
                )
            self.local_context_attn_after_tcn_layers = max(
                0, min(num_layers, self.local_context_attn_after_tcn_layers)
            )
            for i in range(num_layers):
                gamma_k = self.film_gamma_kernel_size
                beta_k = self.film_beta_kernel_size
                if self.film_temporal_last_n > 0 and i >= (
                    num_layers - self.film_temporal_last_n
                ):
                    gamma_k = self.film_temporal_last_kernel_size
                    beta_k = self.film_temporal_last_kernel_size
                gamma = nn.Conv1d(
                    hidden, hidden, gamma_k,
                    padding=gamma_k // 2, padding_mode='reflect',
                )
                beta = nn.Conv1d(
                    hidden, hidden, beta_k,
                    padding=beta_k // 2, padding_mode='reflect',
                )
                z_gamma = nn.Linear(latent_dim, hidden)
                z_beta = nn.Linear(latent_dim, hidden)
                nn.init.zeros_(gamma.bias)
                nn.init.zeros_(gamma.weight)
                nn.init.zeros_(beta.bias)
                nn.init.zeros_(beta.weight)
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
                        self.local_context_attn_fusion = _SplitLocalContextMemoryAttention(
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
                            _SplitLocalContextMemoryAttention(
                                dec_dim=hidden,
                                ctx_dim=self.local_context_dim,
                                n_heads=self.local_context_attn_heads,
                                window_tokens=self.local_context_attn_window_tokens,
                                gate_init=self.local_context_attn_gate_init,
                                support_bias_scale=self.local_context_attn_support_bias_scale,
                                gate_support_power=self.local_context_attn_gate_support_power,
                                gate_support_floor=self.local_context_attn_gate_support_floor,
                                dropout=dropout,
                            ) for _ in self.local_context_upsample_stage_lengths
                        ])
                else:
                    if self.local_context_injection_mode in {'seed', 'both'}:
                        self.local_context_seed_proj = nn.Conv1d(
                            self.local_context_dim, hidden, 1
                        )
                        nn.init.zeros_(self.local_context_seed_proj.weight)
                        nn.init.zeros_(self.local_context_seed_proj.bias)
                        if self.local_context_steps != self.initial_steps:
                            self.local_context_seed_resize = nn.AdaptiveAvgPool1d(
                                self.initial_steps
                            )
                        self.local_context_gate = nn.Parameter(
                            torch.tensor(self.local_context_gate_init)
                        )
                    if self.local_context_injection_mode in {'post_upsample', 'both'}:
                        self.local_context_post_proj = nn.Conv1d(
                            self.local_context_dim, hidden, 1
                        )
                        nn.init.zeros_(self.local_context_post_proj.weight)
                        nn.init.zeros_(self.local_context_post_proj.bias)
                        self.local_context_post_gate = nn.Parameter(
                            torch.tensor(self.local_context_gate_init)
                        )
        else:
            self.latent_proj = nn.Linear(latent_dim, hidden * window_size)
            self.cond_proj = (
                nn.Conv1d(self.aux_dim, hidden, 1) if self.aux_dim > 0 else None
            )
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
            self.dec_cross_attn = _SplitMaskedTemporalCrossAttention(
                dec_dim=hidden,
                enc_dim=hidden_dims[-1],
                n_heads=n_cross_attn_heads,
                dropout=dropout,
            )
            self.cross_attn_gate = nn.Parameter(
                torch.full((hidden,), self.decoder_cross_attn_gate_init)
            )
            self.cross_attn_fuse_gate = nn.Conv1d(hidden * 2, hidden, 1)
            self.cross_attn_fuse_proj = nn.Conv1d(hidden * 2, hidden, 1)
            nn.init.zeros_(self.cross_attn_fuse_gate.weight)
            nn.init.constant_(self.cross_attn_fuse_gate.bias, -2.0)
            nn.init.zeros_(self.cross_attn_fuse_proj.weight)
            nn.init.zeros_(self.cross_attn_fuse_proj.bias)

        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (num_layers - 1 - i)
            self.tcn_layers.append(nn.Sequential(
                param.weight_norm(nn.Conv1d(
                    hidden, hidden, kernel_size,
                    padding=(kernel_size - 1) * dilation // 2,
                    dilation=dilation,
                )),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

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
            nn.init.zeros_(self.mean_head[-1].weight)
            nn.init.zeros_(self.mean_head[-1].bias)
            use_shared_logvar_head = (
                not self.use_detached_variance_pathway
                or self.variance_detach_start_epoch is not None
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
            attn_support_dim = (
                self.variance_attn_support_dim if self.use_variance_attn_support else 0
            )
            var_in_dim = (
                hidden
                + (latent_dim if self.variance_path_use_latent else 0)
                + mask_dim
                + attn_support_dim
            )
            self.variance_input_norm = nn.LayerNorm(hidden)
            if self.variance_path_use_mask:
                self.variance_mask_proj = nn.Sequential(
                    nn.Conv1d(output_dim, self.variance_mask_dim, 1), nn.GELU()
                )
            if self.use_variance_attn_support and variance_attn_support_n_features > 0:
                self.variance_attn_support_proj = nn.Conv1d(
                    variance_attn_support_n_features,
                    self.variance_attn_support_dim,
                    1,
                )
                nn.init.zeros_(self.variance_attn_support_proj.weight)
                nn.init.zeros_(self.variance_attn_support_proj.bias)
            grouped_conv_groups = 1
            if self.variance_use_grouped_conv:
                grouped_conv_groups = max(1, hidden // 4)
                grouped_conv_groups = math.gcd(var_hidden, grouped_conv_groups)
                grouped_conv_groups = max(1, grouped_conv_groups)
            self.variance_stem = nn.Sequential(
                nn.Conv1d(var_in_dim, var_hidden, 1),
                nn.GELU(),
                nn.Conv1d(
                    var_hidden, var_hidden, 3,
                    padding=1, groups=grouped_conv_groups,
                ),
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
            self.feature_logvar_bias = nn.Parameter(torch.full(
                (output_dim,), self.feature_logvar_bias_init, dtype=torch.float32
            ))

        if heteroscedastic:
            if n_chem > 0 and self.n_psd > 0:
                self.logvar_proj_chem = nn.Conv1d(hidden, n_chem, 1)
                self.logvar_proj_psd = nn.Conv1d(hidden, self.n_psd, 1)
            else:
                self.logvar_proj = nn.Conv1d(hidden, output_dim, 1)

    def _apply_decoder_cross_attn_fusion(self, h, enc_h_seq=None, obs_mask=None):
        if not (self.use_decoder_cross_attn and enc_h_seq is not None):
            return h
        ca_out = self.dec_cross_attn(
            h, enc_h_seq, obs_mask if not self.ignore_obs_mask else None
        )
        fuse_in = torch.cat([h, ca_out], dim=1)
        local_gate = torch.sigmoid(self.cross_attn_fuse_gate(fuse_in))
        fused_delta = self.cross_attn_fuse_proj(fuse_in)
        global_gate = torch.sigmoid(self.cross_attn_gate).unsqueeze(0).unsqueeze(-1)
        if self.decoder_cross_attn_missing_only and obs_mask is not None:
            query_missing = (
                (obs_mask == 0).any(dim=-1, keepdim=True)
                .permute(0, 2, 1).to(h.dtype)
            )
            local_gate = local_gate * query_missing
            fused_delta = fused_delta * query_missing
        return h + global_gate * local_gate * fused_delta

    def _apply_decoder_tcn_layers(self, h, cond_h, z, start_idx=0, end_idx=None):
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
                gamma_c = 1.0 + self.cond_film_gamma_scale * torch.tanh(
                    gamma_proj(cond_h)
                )
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

    def _apply_local_context_attn_fusion(
        self, h, local_context=None, obs_mask=None, attn_module=None
    ):
        if attn_module is None:
            attn_module = self.local_context_attn_fusion
        if attn_module is None or local_context is None:
            return h
        if obs_mask is None:
            support_full = torch.ones(
                h.shape[0], 1, self.window_size,
                device=h.device, dtype=h.dtype,
            )
        else:
            support_full = obs_mask.to(h.dtype).mean(dim=-1).unsqueeze(1)
        support_tokens = F.adaptive_avg_pool1d(
            support_full, local_context.shape[-1]
        )
        support_high = F.adaptive_avg_pool1d(support_full, h.shape[-1])
        attn_delta, _ = attn_module(
            h,
            local_context,
            support_tokens=support_tokens,
            support_high=support_high,
        )
        self.last_local_context_gate = attn_module.last_gate_mean
        return h + attn_delta

    def _apply_local_context_upsample_attn_fusion(
        self, h, stage_idx, local_context=None, obs_mask=None
    ):
        if self.local_context_upsample_attn_fusions is None:
            return h
        if stage_idx < 0 or stage_idx >= len(self.local_context_upsample_attn_fusions):
            return h
        return self._apply_local_context_attn_fusion(
            h,
            local_context=local_context,
            obs_mask=obs_mask,
            attn_module=self.local_context_upsample_attn_fusions[stage_idx],
        )

    def _use_detached_variance_now(self):
        if not self.use_detached_variance_pathway:
            return False
        if self.variance_detach_start_epoch is None:
            return True
        if self.current_epoch < 0:
            return True
        return self.current_epoch >= self.variance_detach_start_epoch

    def _support_logvar_features(self, obs_mask):
        obs = obs_mask.permute(0, 2, 1).to(dtype=torch.float32)
        _, channels, window = obs.shape
        missing = 1.0 - obs
        same_feature_window = obs.mean(dim=-1, keepdim=True).expand(
            -1, -1, window
        )
        pos = torch.arange(
            window, device=obs.device, dtype=torch.long
        ).view(1, 1, window)
        obs_bool = obs > 0.5
        neg_large = torch.full(
            (1, 1, window), -window - 1,
            device=obs.device, dtype=torch.long,
        )
        pos_large = torch.full(
            (1, 1, window), window + 1,
            device=obs.device, dtype=torch.long,
        )
        left_candidates = torch.where(obs_bool, pos, neg_large)
        left_idx = torch.cummax(left_candidates, dim=-1).values
        left_dist = (pos - left_idx).clamp(min=0, max=window + 1)
        left_dist = torch.where(
            left_idx < 0, torch.full_like(left_dist, window + 1), left_dist
        )
        right_candidates = torch.where(obs_bool, pos, pos_large)
        right_idx = torch.flip(
            torch.cummin(
                torch.flip(right_candidates, dims=[-1]), dim=-1
            ).values,
            dims=[-1],
        )
        right_dist = (right_idx - pos).clamp(min=0, max=window + 1)
        right_dist = torch.where(
            right_idx > window,
            torch.full_like(right_dist, window + 1),
            right_dist,
        )
        nearest_dist = (
            torch.minimum(left_dist, right_dist)
            .clamp(max=window).to(obs.dtype) / max(1, window)
        )
        if self.n_chem > 0 and self.n_psd > 0 and channels == self.output_dim:
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
                chem_w.expand(-1, self.n_chem, window),
                psd_w.expand(-1, self.n_psd, window),
            ], dim=1)
            other_family_w = torch.cat([
                psd_w.expand(-1, self.n_chem, window),
                chem_w.expand(-1, self.n_psd, window),
            ], dim=1)
        else:
            family_t = obs.mean(dim=1, keepdim=True).expand(-1, channels, -1)
            family_w = obs.mean(dim=(1, 2), keepdim=True).expand(
                -1, channels, window
            )
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
        low_support_score = missing * (
            1.0 - same_feature_window
        ).clamp(0.0, 1.0)
        return features, low_support_score, same_feature_window

    def _apply_support_logvar_residual(self, logvar, obs_mask):
        self.last_support_logvar_residual_mean = None
        self.last_support_logvar_residual_missing_mean = None
        self.last_support_logvar_residual_psd_low_support_mean = None
        self.last_support_logvar_residual_psd_high_support_mean = None
        self.last_support_logvar_monotone_beta = None
        self.last_support_logvar_anchor_beta = None
        self.last_support_logvar_residual_added_pre_clamp = None
        if not self.use_support_logvar_residual or obs_mask is None or logvar is None:
            return logvar
        features, low_support_score, same_feature_window = (
            self._support_logvar_features(obs_mask)
        )
        features = features.to(dtype=logvar.dtype)
        low_support_score = low_support_score.to(dtype=logvar.dtype)
        if self.support_logvar_monotone:
            beta = F.softplus(self.support_logvar_monotone_beta).to(
                dtype=logvar.dtype
            )
            residual = beta * low_support_score
            self.last_support_logvar_monotone_beta = float(beta.detach().item())
            if self.support_logvar_use_anchor and self.support_logvar_anchor_beta is not None:
                anchor_beta = F.softplus(self.support_logvar_anchor_beta).to(
                    dtype=logvar.dtype
                )
                nearest_anchor = features[:, 2].to(dtype=logvar.dtype)
                missing_gate = features[:, 0].to(dtype=logvar.dtype)
                residual = residual + anchor_beta * nearest_anchor * missing_gate
                self.last_support_logvar_anchor_beta = float(
                    anchor_beta.detach().item()
                )
        else:
            residual = self.support_logvar_mlp(features).squeeze(1)
            if self.support_logvar_missing_only:
                residual = residual * features[:, 0]
        residual_det = residual.detach()
        missing = features[:, 0].detach() > 0.5
        self.last_support_logvar_residual_mean = float(
            residual_det.mean().item()
        )
        self.last_support_logvar_residual_missing_mean = (
            float(residual_det[missing].mean().item()) if missing.any() else None
        )
        if self.n_chem > 0 and self.n_psd > 0 and residual.shape[1] > self.n_chem:
            psd_res = residual_det[:, self.n_chem:]
            psd_support = same_feature_window[:, self.n_chem:].detach()
            psd_missing = missing[:, self.n_chem:]
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
        return torch.clamp(
            logvar, min=np.log(self.var_min), max=np.log(self.var_max)
        )

    def _apply_feature_logvar_bias(self, logvar):
        self.last_feature_logvar_bias_mean = None
        self.last_feature_logvar_bias_chem_mean = None
        self.last_feature_logvar_bias_psd_mean = None
        self.last_feature_logvar_bias_psd_min = None
        self.last_feature_logvar_bias_psd_max = None
        self.last_feature_logvar_bias_added_pre_clamp = None
        if not self.use_feature_logvar_bias or self.feature_logvar_bias is None or logvar is None:
            return logvar
        raw_bias = self.feature_logvar_bias.to(
            device=logvar.device, dtype=logvar.dtype
        )
        bias = (
            F.softplus(raw_bias)
            if self.feature_logvar_bias_constraint == 'nonnegative'
            else raw_bias
        )
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
            self.last_feature_logvar_bias_chem_mean = float(
                bias_det[:self.n_chem].mean().item()
            )
        if self.n_psd > 0 and bias_det.numel() > self.n_chem:
            psd_bias = bias_det[self.n_chem:]
            self.last_feature_logvar_bias_psd_mean = float(psd_bias.mean().item())
            self.last_feature_logvar_bias_psd_min = float(psd_bias.min().item())
            self.last_feature_logvar_bias_psd_max = float(psd_bias.max().item())
        logvar = logvar + bias.view(1, -1, 1)
        self.last_feature_logvar_bias_added_pre_clamp = logvar.detach()
        return torch.clamp(
            logvar, min=np.log(self.var_min), max=np.log(self.var_max)
        )

    def forward(self, z, cond, enc_h_seq=None, obs_mask=None, local_context=None,
                attn_weighted_support_t=None):
        batch_size = z.shape[0]
        if cond is not None and cond.shape[-1] > 0 and self.cond_proj is not None:
            cond_h = self.cond_proj(cond.permute(0, 2, 1))
        else:
            cond_h = z.new_zeros(
                batch_size, self.hidden_dim, self.window_size
            )
        local_gate_vals = []
        self.last_local_context_gate = None
        self.last_logvar_diagnostics = None
        logvar_diag = {}

        if self.use_progressive_decoder:
            h = self.latent_proj(z).view(batch_size, -1, self.initial_steps)
            upsample_stage_idx = 0
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
                        h, upsample_stage_idx, local_context, obs_mask
                    )
                    upsample_stage_idx += 1
            for block in self.upsample_blocks:
                h = block(h)
                if (
                    self.local_context_fusion_mode == 'attn'
                    and self.local_context_attn_location in {'upsample', 'both'}
                    and h.shape[-1] <= self.window_size
                ):
                    h = self._apply_local_context_upsample_attn_fusion(
                        h, upsample_stage_idx, local_context, obs_mask
                    )
                    upsample_stage_idx += 1
            if h.shape[-1] != self.window_size:
                h = self.final_upsample(h)
                if (
                    self.local_context_fusion_mode == 'attn'
                    and self.local_context_attn_location in {'upsample', 'both'}
                ):
                    h = self._apply_local_context_upsample_attn_fusion(
                        h, upsample_stage_idx, local_context, obs_mask
                    )
                    upsample_stage_idx += 1
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
            h = self._apply_decoder_cross_attn_fusion(h, enc_h_seq, obs_mask)
            if self.use_tcn:
                if (
                    self.local_context_fusion_mode == 'attn'
                    and self.local_context_attn_fusion is not None
                    and self.local_context_attn_location in {'mid_tcn', 'both'}
                ):
                    split_idx = self.local_context_attn_after_tcn_layers
                    if split_idx > 0:
                        h = self._apply_decoder_tcn_layers(
                            h, cond_h, z, 0, split_idx
                        )
                    h = self._apply_local_context_attn_fusion(
                        h, local_context, obs_mask
                    )
                    if split_idx < len(self.tcn_layers):
                        h = self._apply_decoder_tcn_layers(
                            h, cond_h, z, split_idx, None
                        )
                else:
                    h = self._apply_decoder_tcn_layers(h, cond_h, z, 0, None)
        else:
            h = self.latent_proj(z).view(batch_size, -1, self.window_size)
            h = h + cond_h
            h = self._apply_decoder_cross_attn_fusion(h, enc_h_seq, obs_mask)
            if self.use_tcn:
                for layer in self.tcn_layers:
                    h = layer(h)

        if local_gate_vals:
            self.last_local_context_gate = float(
                torch.stack(local_gate_vals).mean().item()
            )
        elif self.local_context_fusion_mode != 'attn':
            self.last_local_context_gate = None
        if self.use_z_skip:
            z_skip = self.z_skip_proj(z).unsqueeze(-1)
            z_gate = torch.sigmoid(self.z_skip_gate(z)).unsqueeze(-1)
            h = h + z_gate * z_skip
        h = self.final_output_norm(h.transpose(1, 2)).transpose(1, 2)
        h_mean = h + self.mean_head(h) if self.use_dual_output_heads else h
        use_detached_variance_now = self._use_detached_variance_now()
        if use_detached_variance_now:
            h_var_base = self.variance_input_norm(
                h.detach().transpose(1, 2)
            ).transpose(1, 2)
            h_var_inputs = [h_var_base]
            if self.variance_path_use_latent:
                z_for_variance = (
                    z.detach() if self.variance_path_detach_latent else z
                )
                h_var_inputs.append(
                    z_for_variance.unsqueeze(-1).expand(-1, -1, h.shape[-1])
                )
            if self.variance_path_use_mask:
                if obs_mask is None:
                    mask_feat = torch.zeros(
                        h.shape[0], self.variance_mask_dim, h.shape[-1],
                        device=h.device, dtype=h.dtype,
                    )
                else:
                    mask_feat = self.variance_mask_proj(
                        obs_mask.permute(0, 2, 1).to(h.dtype)
                    )
                h_var_inputs.append(mask_feat)
            if self.use_variance_attn_support:
                if attn_weighted_support_t is not None and self.variance_attn_support_proj is not None:
                    support_feat = self.variance_attn_support_proj(
                        attn_weighted_support_t.permute(0, 2, 1).to(h.dtype)
                    )
                else:
                    support_feat = torch.zeros(
                        h.shape[0], self.variance_attn_support_dim, h.shape[-1],
                        device=h.device, dtype=h.dtype,
                    )
                h_var_inputs.append(support_feat)
            h_var = self.variance_stem(torch.cat(h_var_inputs, dim=1))
        elif self.use_dual_output_heads and self.logvar_head is not None:
            h_logvar = h + self.logvar_head(h)
        else:
            h_logvar = h

        mean = self.output_proj(h_mean)
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
                logvar_diag['base_pre_clamp'] = torch.cat(
                    [logvar_chem, logvar_psd], dim=1
                ).detach()
                logvar_chem = torch.clamp(
                    logvar_chem,
                    min=np.log(self.var_min), max=np.log(self.var_max),
                )
                logvar_psd = torch.clamp(
                    logvar_psd,
                    min=np.log(self.var_min), max=np.log(self.var_max),
                )
                logvar = torch.cat([logvar_chem, logvar_psd], dim=1)
                logvar_diag['base_post_clamp'] = logvar.detach()
            else:
                if use_detached_variance_now:
                    h_logvar = h_var + self.variance_refine(h_var)
                logvar = self.logvar_proj(h_logvar)
                logvar_diag['base_pre_clamp'] = logvar.detach()
                logvar = torch.clamp(
                    logvar,
                    min=np.log(self.var_min), max=np.log(self.var_max),
                )
                logvar_diag['base_post_clamp'] = logvar.detach()
            if self.local_context_attn_logvar_support_boost > 0.0 and obs_mask is not None:
                support = obs_mask.to(logvar.dtype).mean(dim=-1).unsqueeze(1)
                low_support = (1.0 - support).clamp(0.0, 1.0)
                logvar_diag['support_boost_pre'] = logvar.detach()
                boosted_logvar = (
                    logvar
                    + self.local_context_attn_logvar_support_boost * low_support
                )
                logvar_diag['support_boost_added_pre_clamp'] = boosted_logvar.detach()
                logvar = torch.clamp(
                    boosted_logvar,
                    min=np.log(self.var_min), max=np.log(self.var_max),
                )
                logvar_diag['support_boost_post_clamp'] = logvar.detach()
            logvar_diag['feature_bias_pre'] = logvar.detach()
            logvar = self._apply_feature_logvar_bias(logvar)
            if self.last_feature_logvar_bias_added_pre_clamp is not None:
                logvar_diag['feature_bias_added_pre_clamp'] = (
                    self.last_feature_logvar_bias_added_pre_clamp.detach()
                )
            logvar_diag['feature_bias_post_clamp'] = logvar.detach()
            logvar_diag['support_residual_pre'] = logvar.detach()
            logvar = self._apply_support_logvar_residual(logvar, obs_mask)
            if self.last_support_logvar_residual_added_pre_clamp is not None:
                logvar_diag['support_residual_added_pre_clamp'] = (
                    self.last_support_logvar_residual_added_pre_clamp.detach()
                )
            logvar_diag['support_residual_post_clamp'] = logvar.detach()
            logvar_diag['final'] = logvar.detach()
            self.last_logvar_diagnostics = logvar_diag
        return mean, logvar
