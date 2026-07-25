"""Graph-temporal encoder used by :class:`ImputationVAE_Graph`.

This module is intentionally behavior-compatible with the legacy public
``model_graph_uq.GraphEncoder`` implementation.  The legacy module re-exports
this class so existing import paths and checkpoint keys remain stable.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrizations as param

from ..graph_blocks.attention import (
    AxialObservedAttentionBlock as _SplitAxialObservedAttentionBlock,
    PreGraphPerFeatureTemporalAttention as _SplitPreGraphPerFeatureTemporalAttention,
    TemporalAttentionPool as _SplitTemporalAttentionPool,
)
from ..graph_blocks.tcn import DepthwiseTCN as _SplitDepthwiseTCN
from ..graph_blocks.temporal_refiner import (
    TemporalObservationRefiner as _SplitTemporalObservationRefiner,
)
from ..graph_layers.cross_modal_graph import (
    CrossModalGraphLayer as _SplitCrossModalGraphLayer,
)
from ..graph_layers.input_graph import InputGraphLayer as _SplitInputGraphLayer
from ..graph_layers.local_chunk_graph import (
    LocalChunkGraphBranch as _SplitLocalChunkGraphBranch,
)
from ..graph_layers.token_graph import (
    TokenGraphCrossBlock as _SplitTokenGraphCrossBlock,
    TokenGraphSelfBlock as _SplitTokenGraphSelfBlock,
)


class GraphEncoder(nn.Module):
    """TCN encoder with graph layers for feature relationships."""

    def __init__(self, input_dim, hidden_dims, latent_dim, window_size, num_layers=5,
                 kernel_size=3, dropout=0.1, n_graph_heads=4, target_dim=None,
                 aux_dim=None, use_input_graph_layer=True, use_cross_modal_graph=True,
                 use_tcn=True, n_input_graph_layers=1, use_parallel_graph=False,
                 use_temporal_cnn=True, n_chem=0, enable_cross_modal_floor=False,
                 disable_rel_scale=False, disable_prior_bias=False, disable_aux_bias=False,
                 cross_modal_query_gate_mode='legacy_hard',
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
        self.token_graph_dim = None if token_graph_dim is None else int(token_graph_dim)
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
        self.local_context_steps = None if local_context_steps is None else int(local_context_steps)
        self.local_context_observe_aware = bool(local_context_observe_aware)
        self.local_context_observe_aware_blend_gate_init = (
            None if local_context_observe_aware_blend_gate_init is None
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
            self.token_shared_self_block = _SplitTokenGraphSelfBlock(
                d_model=token_dim, n_heads=n_graph_heads, dropout=dropout,
                ffn_mult=self.graph_ffn_mult,
            )
            self.token_branch_self_block = _SplitTokenGraphSelfBlock(
                d_model=token_dim, n_heads=n_graph_heads, dropout=dropout,
                ffn_mult=self.graph_ffn_mult,
            )
            self.token_aux_embed = None
            self.token_aux_embed_norm = None
            self.token_branch_cross_block = None
            if use_cross_modal_graph and target_dim is not None and aux_dim is not None and aux_dim > 0:
                self.token_aux_embed = nn.Linear(window_size, token_dim)
                self.token_aux_embed_norm = nn.LayerNorm(token_dim)
                self.token_branch_cross_block = _SplitTokenGraphCrossBlock(
                    d_model=token_dim, n_heads=n_graph_heads, dropout=dropout,
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
            self.n_input_graph_layers = n_input_graph_layers if use_input_graph_layer else 0
            if use_input_graph_layer and n_input_graph_layers > 0:
                n_graph_features = target_dim if target_dim is not None else input_dim
                self.input_graph_layers = nn.ModuleList([
                    _SplitInputGraphLayer(
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
                    ) for _ in range(n_input_graph_layers)
                ])
            else:
                self.input_graph_layers = None
            if use_cross_modal_graph and target_dim is not None and aux_dim is not None and aux_dim > 0:
                self.cross_modal_graph_layer = _SplitCrossModalGraphLayer(
                    target_dim=target_dim,
                    aux_dim=aux_dim,
                    window_size=window_size,
                    n_heads=n_graph_heads,
                    head_dim=64,
                    dropout=dropout,
                    use_temporal_cnn=use_temporal_cnn,
                    disable_aux_bias=disable_aux_bias,
                    query_gate_mode=cross_modal_query_gate_mode,
                    use_ffn=self.use_graph_ffn,
                    ffn_mult=self.graph_ffn_mult,
                )
            else:
                self.cross_modal_graph_layer = None
            self._has_parallel_gate = (
                use_parallel_graph and use_input_graph_layer and n_input_graph_layers > 0
                and use_cross_modal_graph and target_dim is not None
                and aux_dim is not None and aux_dim > 0
            )
            if self._has_parallel_gate:
                self.parallel_gate = nn.Sigmoid()
                self.gate_norm = nn.LayerNorm(window_size)
                if aux_dim is not None and aux_dim > 0:
                    self.aux_gate_proj = nn.Conv1d(aux_dim, target_dim, 1)
                    nn.init.zeros_(self.aux_gate_proj.weight)
                    nn.init.zeros_(self.aux_gate_proj.bias)
                    self.target_gate_proj = nn.Conv1d(
                        target_dim, target_dim, 1, groups=target_dim
                    )
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

        required_layers = int(np.ceil(np.log2(window_size)))
        if self.use_temporal_cnn and self.use_pregraph_depthwise_tcn and target_dim is not None:
            self.target_tcn = _SplitDepthwiseTCN(
                target_dim, num_layers=required_layers, kernel_size=3
            )
            self.aux_tcn = (
                _SplitDepthwiseTCN(aux_dim, num_layers=required_layers, kernel_size=3)
                if aux_dim is not None and aux_dim > 0 else nn.Identity()
            )
        else:
            self.target_tcn = nn.Identity()
            self.aux_tcn = nn.Identity()

        self.axial_observed_attn = None
        if self.use_axial_observed_attn and target_dim is not None:
            self.axial_observed_attn = _SplitAxialObservedAttentionBlock(
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
            self.pregraph_feature_temporal_attn = _SplitPreGraphPerFeatureTemporalAttention(
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
            self.local_chunk_graph = _SplitLocalChunkGraphBranch(
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

        self.input_proj = nn.Conv1d(input_dim, hidden_dims[0], 1)
        self.tcn_layers = nn.ModuleList()
        self.tcn_downsample = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dims[min(i, len(hidden_dims) - 1)]
            out_ch = (
                hidden_dims[min(i + 1, len(hidden_dims) - 1)]
                if i < num_layers - 1 else hidden_dims[-1]
            )
            dilation = 2 ** i
            self.tcn_layers.append(nn.Sequential(
                param.weight_norm(nn.Conv1d(
                    in_ch, out_ch, kernel_size,
                    padding=(kernel_size - 1) * dilation // 2,
                    dilation=dilation,
                )),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            self.tcn_downsample.append(
                nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
            )
        self.attn_pool = _SplitTemporalAttentionPool(hidden_dims[-1])
        self.latent_pooled_norm = (
            nn.LayerNorm(hidden_dims[-1]) if self.use_latent_pooled_norm else nn.Identity()
        )
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.temporal_refiner = None
        if self.use_temporal_refiner:
            self.temporal_refiner = _SplitTemporalObservationRefiner(
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
                nn.Conv1d(hidden_dims[-1], self.local_context_dim, 1), nn.GELU()
            )
            self.local_context_pool = nn.AdaptiveAvgPool1d(self.local_context_steps)

        self.last_input_graph_attention = None
        self.last_input_graph_attention_heads = None
        self.last_input_graph_attention_batch = None
        self.last_input_graph_attention_heads_batch = None
        self.last_input_graph_attention_per_layer = []
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
        attn_weights = None
        cross_attn = None
        if self.graph_target_dim is None or self.graph_target_dim >= x.shape[1]:
            x_target = x
            x_aux = None
            target_mask = obs_mask
        else:
            x_target = x[:, :self.graph_target_dim, :]
            target_mask = (
                obs_mask[:, :, :self.graph_target_dim] if obs_mask is not None else None
            )
            if self.graph_aux_dim is not None and self.graph_aux_dim > 0:
                aux_start = self.graph_target_dim
                aux_end = self.graph_target_dim + self.graph_aux_dim
                x_aux = x[:, aux_start:aux_end, :]
            else:
                x_aux = None

        x_processed_list = []
        if self.graph_target_dim is not None:
            target_mask_t = (
                target_mask.permute(0, 2, 1).float()
                if target_mask is not None else torch.ones_like(x_target)
            )
            x_target_masked = x_target * target_mask_t
            obs_count = target_mask_t.sum(dim=2, keepdim=True).clamp(min=1)
            x_target_mean = x_target_masked.sum(dim=2, keepdim=True) / obs_count
            x_target_centered = (x_target - x_target_mean) * target_mask_t
            x_target_var = (
                (x_target_centered ** 2).sum(dim=2, keepdim=True)
                / obs_count.clamp(min=2)
            )
            x_target_std = torch.sqrt(x_target_var + 1e-8)
            x_target_normed = (x_target_centered / x_target_std) * target_mask_t
            if embed_offset is not None:
                x_target_normed = x_target_normed + embed_offset
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
                self.last_pregraph_feature_temporal_attn_gate = self.pregraph_feature_temporal_attn.last_gate
                self.last_pregraph_feature_temporal_attn_entropy_missing = self.pregraph_feature_temporal_attn.last_missing_query_attn_entropy
                self.last_pregraph_feature_temporal_attn_entropy_observed = self.pregraph_feature_temporal_attn.last_observed_query_attn_entropy
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
            if x_aux is not None:
                x_aux_mean = x_aux.mean(dim=2, keepdim=True)
                x_aux_std = x_aux.std(dim=2, keepdim=True) + 1e-8
                x_aux_normed = (x_aux - x_aux_mean) / x_aux_std
                feat_aux = self.aux_tcn(x_aux_normed)
                x_aux_processed = feat_aux
                x_processed_list.append(feat_aux)
            else:
                x_aux_processed = None
                x_aux_normed = None
        else:
            x_target_processed = x_target
            x_aux_processed = x_aux
            x_aux_normed = None
            target_mask_t = None
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

        x_flat_processed = (
            torch.cat(x_processed_list, dim=1) if x_processed_list else x
        )
        x_graph_base = x_target_processed
        x_local_chunk = None
        self.last_local_chunk_graph_gate = None
        self.last_local_chunk_graph_out_proj_norm = None
        self.last_local_chunk_graph_obs_ratio_mean = None
        if self.local_chunk_graph is not None:
            x_local_chunk = self.local_chunk_graph(
                x_target_processed, chunk_obs_mask=target_mask_t
            )
            self.last_local_chunk_graph_gate = self.local_chunk_graph.last_gate
            self.last_local_chunk_graph_out_proj_norm = self.local_chunk_graph.last_out_proj_weight_norm
            self.last_local_chunk_graph_obs_ratio_mean = self.local_chunk_graph.last_obs_ratio_mean
            if self.local_chunk_graph_mode == 'sequential_pre':
                x_graph_base = x_local_chunk

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
            self.last_input_graph_attention_heads_batch = self.token_branch_self_block.last_attention_weights_heads_batch
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
                self.last_cross_modal_attention_heads = self.token_branch_cross_block.last_attention_weights_heads
                self.last_cross_modal_attention_heads_batch = self.token_branch_cross_block.last_attention_weights_heads_batch
                gate_input = torch.cat([shared_tokens, self_tokens, x_cross_tokens], dim=-1)
                gate = torch.sigmoid(self.token_branch_gate_proj(gate_input))
                fused_tokens = self.token_fuse_norm(
                    shared_tokens
                    + gate * (self_tokens - shared_tokens)
                    + (1.0 - gate) * (x_cross_tokens - shared_tokens)
                )
                self.last_parallel_gate = gate.detach().mean(dim=(0, 2))
            else:
                fused_tokens = self_tokens
            delta_w = self.token_out_proj(fused_tokens)
            out_gate = torch.sigmoid(self.token_out_gate)
            x_target_enhanced = self.token_out_norm(x_graph_base + out_gate * delta_w)
        else:
            has_self = self.input_graph_layers is not None
            has_cross = self.cross_modal_graph_layer is not None and x_aux is not None
            x_self = None
            if has_self:
                x_self_input = x_graph_base
                self.last_input_graph_attention_per_layer = []
                for graph_layer in self.input_graph_layers:
                    x_self_input, attn_weights = graph_layer(
                        x_self_input, obs_mask=target_mask, x_aux=x_aux_processed
                    )
                    self.last_input_graph_attention_per_layer.append({
                        'avg': attn_weights.detach().mean(dim=0),
                        'batch': attn_weights.detach(),
                        'heads': getattr(graph_layer, 'last_attention_weights_heads', None),
                        'heads_batch': getattr(graph_layer, 'last_attention_weights_heads_batch', None),
                    })
                last_layer = self.input_graph_layers[-1]
                self.last_input_graph_attention = attn_weights.detach().mean(dim=0)
                self.last_input_graph_attention_batch = attn_weights.detach()
                self.last_input_graph_attention_heads = getattr(
                    last_layer, 'last_attention_weights_heads', None
                )
                self.last_input_graph_attention_heads_batch = getattr(
                    last_layer, 'last_attention_weights_heads_batch', None
                )
                x_self = x_self_input
            x_cross = None
            if has_cross:
                cross_query = (
                    x_graph_base if self._has_parallel_gate and has_self
                    else x_self if x_self is not None else x_graph_base
                )
                x_cross_output, cross_attn = self.cross_modal_graph_layer(
                    cross_query, x_aux_processed, target_mask
                )
                self.last_cross_modal_attention = cross_attn.detach().mean(dim=0)
                self.last_cross_modal_attention_batch = cross_attn.detach()
                self.last_cross_modal_attention_heads = getattr(
                    self.cross_modal_graph_layer, 'last_attention_weights_heads', None
                )
                self.last_cross_modal_attention_heads_batch = getattr(
                    self.cross_modal_graph_layer,
                    'last_attention_weights_heads_batch', None,
                )
                x_cross = x_cross_output
            if self._has_parallel_gate and has_self and has_cross:
                delta_self = x_self - x_graph_base
                delta_cross = x_cross - x_graph_base
                if self.aux_gate_proj is not None and x_aux_normed is not None:
                    gate_input = (
                        self.aux_gate_proj(x_aux_normed)
                        + self.target_gate_proj(x_target_normed)
                    )
                else:
                    gate_input = torch.zeros_like(x_graph_base)
                gate = self.parallel_gate(gate_input)
                x_target_enhanced = self.gate_norm(
                    x_graph_base + gate * delta_self + (1 - gate) * delta_cross
                )
                self.last_parallel_gate = gate.detach().mean(dim=(0, 2))
            elif has_self and has_cross:
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
        final_features = []
        if self.graph_target_dim is not None and x_target_enhanced is not None:
            final_features.append(x_target_enhanced)
            if x_aux_processed is not None:
                final_features.append(x_aux_processed)
            x_fused = torch.cat(final_features, dim=1)
        else:
            x_fused = x_flat_processed
        h = self.input_proj(x_fused)
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
                target_obs_mask=target_mask if not self.ignore_obs_mask else None,
            )
            self.last_temporal_refiner_gate = self.temporal_refiner.last_gate
            self.last_temporal_refiner_attn_entropy_missing = self.temporal_refiner.last_missing_query_attn_entropy
            self.last_temporal_refiner_attn_entropy_observed = self.temporal_refiner.last_observed_query_attn_entropy
        else:
            self.last_temporal_refiner_gate = None
            self.last_temporal_refiner_attn_entropy_missing = None
            self.last_temporal_refiner_attn_entropy_observed = None
        h_pooled = self.attn_pool(
            h_base, obs_mask=target_mask if not self.ignore_obs_mask else None
        )
        h_pooled = self.latent_pooled_norm(h_pooled)
        mu = self.fc_mu(h_pooled)
        logvar = self.fc_logvar(h_pooled)
        if self.latent_logvar_min is not None or self.latent_logvar_max is not None:
            logvar = torch.clamp(
                logvar, min=self.latent_logvar_min, max=self.latent_logvar_max
            )
        local_context = None
        if self.use_local_context_map:
            local_features = self.local_context_proj(h_refined)
            local_context_raw = self.local_context_pool(local_features)
            self.last_local_context_generation_support_mean = None
            self.last_local_context_observe_aware_blend_gate = None
            if self.local_context_observe_aware and target_mask is not None and not self.ignore_obs_mask:
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
        attn_weighted_support_t = None
        if attn_weights is not None and target_mask is not None:
            aw = attn_weights.detach()
            attn_w_avg = aw.mean(dim=1) if aw.dim() == 4 else aw
            attn_weighted_support_t = torch.einsum(
                'bfc,btc->btf', attn_w_avg, target_mask.float()
            )
        return (
            mu,
            logvar,
            attn_weights,
            h_base,
            local_context,
            attn_weighted_support_t,
        )
