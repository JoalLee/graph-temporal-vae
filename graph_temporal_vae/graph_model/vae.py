"""Top-level Graph-enhanced Temporal-VAE model and compatibility helpers."""

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..flows import RealNVP
from ..graph_blocks.time_encoding import TimeHybridEncoder
from ..graph_layers.external_history import ExternalHistoryContext
from ..model_config import ModelConfig
from .decoder import GraphDecoder
from .encoder import GraphEncoder


class ImputationVAE_Graph(nn.Module):
    """Graph-enhanced probabilistic VAE for multivariate time-series imputation."""

    def __init__(self, target_dim, aux_dim, window_size, latent_dim=256,
                 latent_mode='variational',
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
                 cross_modal_query_gate_mode='legacy_hard',
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
        self.latent_mode = str(latent_mode)
        if self.latent_mode not in {'variational', 'deterministic'}:
            raise ValueError(
                "latent_mode must be one of {'variational', 'deterministic'}"
            )
        self.ignore_obs_mask = ignore_obs_mask
        self.cross_modal_query_gate_mode = str(cross_modal_query_gate_mode)
        self.use_realnvp = use_realnvp
        self.flow = (
            RealNVP(latent_dim, n_layers=realnvp_layers, hidden_dim=max(64, latent_dim))
            if use_realnvp else None
        )
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
        self.token_graph_dim = None if token_graph_dim is None else int(token_graph_dim)
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
            None if local_context_observe_aware_blend_gate_init is None
            else float(local_context_observe_aware_blend_gate_init)
        )
        self.local_context_injection_mode = str(local_context_injection_mode)
        self.local_context_fusion_mode = str(local_context_fusion_mode)
        self.local_context_attn_heads = int(local_context_attn_heads)
        self.local_context_attn_window_tokens = int(local_context_attn_window_tokens)
        self.local_context_attn_gate_init = float(local_context_attn_gate_init)
        self.local_context_attn_after_tcn_layers = (
            None if local_context_attn_after_tcn_layers is None
            else int(local_context_attn_after_tcn_layers)
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
            df_shape = (2,) if self.likelihood_df_scope == 'family' else (target_dim,)
            init = min(
                max(self.likelihood_df_init, self.likelihood_df_min + 1e-4),
                self.likelihood_df_max - 1e-4,
            )
            ratio = (
                (init - self.likelihood_df_min)
                / (self.likelihood_df_max - self.likelihood_df_min)
            )
            raw_init = float(np.log(ratio / (1.0 - ratio)))
            self.likelihood_df_raw = nn.Parameter(torch.full(
                df_shape, raw_init, dtype=torch.float32
            ))
        self.prior_df_raw = None
        if self.use_learnable_prior_df:
            init = min(
                max(self.prior_df_init, self.prior_df_min + 1e-4),
                self.prior_df_max - 1e-4,
            )
            ratio = (
                (init - self.prior_df_min)
                / (self.prior_df_max - self.prior_df_min)
            )
            raw_init = float(np.log(ratio / (1.0 - ratio)))
            self.prior_df_raw = nn.Parameter(
                torch.tensor(raw_init, dtype=torch.float32)
            )

        self.time_hybrid_encoder = None
        if use_hybrid_time_encoding and aux_dim >= (time_numeric_dim + time_cyc_dim):
            self.time_hybrid_encoder = TimeHybridEncoder(
                out_dim=time_hybrid_dim,
                hour_embed_dim=hour_embed_dim,
                dow_embed_dim=dow_embed_dim,
                month_embed_dim=month_embed_dim,
                dropout=dropout,
            )
        input_dim = target_dim + aux_dim
        self.mask_embed = nn.Embedding(2, target_dim)
        nn.init.normal_(self.mask_embed.weight, mean=0.0, std=0.01)

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
            cross_modal_query_gate_mode=cross_modal_query_gate_mode,
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
            ignore_obs_mask=self.ignore_obs_mask,
        )
        self.external_history_context = None
        if self.use_external_history_context:
            if not self.use_local_context_map:
                raise ValueError(
                    "use_external_history_context=True requires use_local_context_map=True"
                )
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

        for name in (
            'last_graph_attention', 'last_graph_attention_heads',
            'last_cross_modal_attention', 'last_cross_modal_attention_heads',
            'last_pregraph_feature_temporal_attn_gate',
            'last_pregraph_feature_temporal_attn_entropy_missing',
            'last_pregraph_feature_temporal_attn_entropy_observed',
            'last_axial_time_gate_mean', 'last_axial_time_gate_chem_mean',
            'last_axial_time_gate_psd_mean', 'last_axial_cross_gate_mean',
            'last_axial_cross_gate_chem_mean', 'last_axial_cross_gate_psd_mean',
            'last_axial_time_no_key_fraction', 'last_axial_cross_no_key_fraction',
            'last_axial_cross_valid_query_fraction',
            'last_axial_cross_entropy_missing', 'last_axial_cross_top1_mass',
            'last_axial_cross_top3_mass', 'last_axial_psd_to_chem_mass',
            'last_axial_psd_to_psd_mass', 'last_local_chunk_graph_gate',
            'last_local_chunk_graph_out_proj_norm',
            'last_local_chunk_graph_obs_ratio_mean', 'last_temporal_refiner_gate',
            'last_temporal_refiner_attn_entropy_missing',
            'last_temporal_refiner_attn_entropy_observed',
            'last_external_history_gate', 'last_external_history_attn_entropy',
            'last_external_history_valid_fraction',
            'last_external_history_null_fraction',
            'last_external_history_top1_mass', 'last_external_history_top3_mass',
            'last_external_history_attended_time_dist',
            'last_external_history_attended_support',
            'last_external_history_attended_null_fraction',
            'last_support_logvar_residual_mean',
            'last_support_logvar_residual_missing_mean',
            'last_support_logvar_residual_psd_low_support_mean',
            'last_support_logvar_residual_psd_high_support_mean',
            'last_support_logvar_monotone_beta',
        ):
            setattr(self, name, None)

    @classmethod
    def from_config(
        cls,
        target_dim,
        aux_dim,
        window_size,
        config=None,
        **overrides,
    ):
        """Construct the model from a validated :class:`ModelConfig`.

        The legacy constructor remains unchanged for backward compatibility.
        Explicit overrides are validated and applied to a copied config.
        """
        config = ModelConfig() if config is None else config
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig instance")
        if overrides:
            config = config.with_overrides(**overrides)
        return cls(
            target_dim=target_dim,
            aux_dim=aux_dim,
            window_size=window_size,
            **config.to_kwargs(),
        )

    def get_likelihood_df(self, num_features=None, device=None, dtype=None):
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
            df_idx = 1 if df_values.numel() > 1 else 0
            return df_values[df_idx].expand(int(num_features))
        if n_chem >= int(num_features):
            return df_values[0].expand(int(num_features))
        out = torch.empty(int(num_features), device=device, dtype=dtype)
        out[:n_chem] = df_values[0]
        out[n_chem:] = df_values[1]
        return out

    def get_likelihood_df_metadata(self):
        with torch.no_grad():
            df = self.get_likelihood_df(
                device=torch.device('cpu'), dtype=torch.float32
            ).detach().cpu()
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
        with torch.no_grad():
            df = self.get_prior_df(
                device=torch.device('cpu'), dtype=torch.float32
            ).detach().cpu()
            return {
                'learnable': bool(self.prior_df_raw is not None),
                'min': float(self.prior_df_min),
                'max': float(self.prior_df_max),
                'value': float(df.item()),
            }

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _encode_cond_features(self, cond):
        if self.time_hybrid_encoder is None:
            return cond
        if cond.shape[-1] < self.time_numeric_dim + self.time_cyc_dim:
            return cond
        left = cond[:, :, :self.time_numeric_dim]
        time_cyc = cond[
            :, :, self.time_numeric_dim:self.time_numeric_dim + self.time_cyc_dim
        ]
        right = cond[:, :, self.time_numeric_dim + self.time_cyc_dim:]
        return torch.cat([left, self.time_hybrid_encoder(time_cyc), right], dim=-1)

    def enable_cross_modal_imputation(self, enable=True):
        if (
            hasattr(self.encoder, 'cross_modal_graph_layer')
            and self.encoder.cross_modal_graph_layer is not None
        ):
            self.encoder.cross_modal_graph_layer.query_gate_mode = (
                'soft' if enable else self.cross_modal_query_gate_mode
            )

    def forward(self, x, cond, mask, history=None, sample_latent=True):
        embed_0 = self.mask_embed(torch.zeros(
            1, dtype=torch.long, device=x.device
        ))
        embed_1 = self.mask_embed(torch.ones(
            1, dtype=torch.long, device=x.device
        ))
        embed_offset_t = (
            embed_0 + mask.float() * (embed_1 - embed_0)
        ).permute(0, 2, 1)
        cond = self._encode_cond_features(cond)
        history_context = None
        if self.external_history_context is not None and history is not None:
            h_aux = history.get('history_aux')
            h_hour = history.get('history_hour')
            if h_aux is not None and h_hour is not None:
                h_cond = torch.cat([h_aux, h_hour], dim=-1)
                batch, chunks, length, channels = h_cond.shape
                h_cond = self._encode_cond_features(
                    h_cond.reshape(batch * chunks, length, channels)
                ).reshape(batch, chunks, length, -1)
                history = dict(history)
                history['history_cond'] = h_cond
                history_context = self.external_history_context(history, cond, mask)
        inputs = torch.cat([x, cond], dim=-1).permute(0, 2, 1)
        batch, window, _ = x.shape
        full_obs_mask = torch.cat([
            mask,
            torch.ones(
                batch, window, cond.shape[-1], device=x.device
            ),
        ], dim=-1)
        (
            mu,
            logvar,
            graph_attention,
            h_seq,
            local_context,
            attn_weighted_support_t,
        ) = self.encoder(inputs, full_obs_mask, embed_offset=embed_offset_t)
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
            history_names = (
                'last_gate', 'last_attn_entropy', 'last_valid_fraction',
                'last_null_fraction', 'last_top1_mass', 'last_top3_mass',
                'last_attended_time_dist', 'last_attended_support',
                'last_attended_null_fraction',
            )
            model_names = (
                'last_external_history_gate',
                'last_external_history_attn_entropy',
                'last_external_history_valid_fraction',
                'last_external_history_null_fraction',
                'last_external_history_top1_mass',
                'last_external_history_top3_mass',
                'last_external_history_attended_time_dist',
                'last_external_history_attended_support',
                'last_external_history_attended_null_fraction',
            )
            for model_name, history_name in zip(model_names, history_names):
                setattr(
                    self, model_name,
                    getattr(self.external_history_context, history_name),
                )
        else:
            for name in (
                'last_external_history_gate',
                'last_external_history_attn_entropy',
                'last_external_history_valid_fraction',
                'last_external_history_null_fraction',
                'last_external_history_top1_mass',
                'last_external_history_top3_mass',
                'last_external_history_attended_time_dist',
                'last_external_history_attended_support',
                'last_external_history_attended_null_fraction',
            ):
                setattr(self, name, None)
        self.last_attn_weighted_support_t = attn_weighted_support_t
        self.last_graph_attention = (
            graph_attention.detach().mean(dim=0)
            if graph_attention is not None else None
        )
        self.last_graph_attention_heads = self.encoder.last_input_graph_attention_heads
        self.last_cross_modal_attention = self.encoder.last_cross_modal_attention
        self.last_cross_modal_attention_heads = self.encoder.last_cross_modal_attention_heads
        for name in (
            'last_pregraph_feature_temporal_attn_gate',
            'last_pregraph_feature_temporal_attn_entropy_missing',
            'last_pregraph_feature_temporal_attn_entropy_observed',
            'last_axial_time_gate_mean', 'last_axial_time_gate_chem_mean',
            'last_axial_time_gate_psd_mean', 'last_axial_cross_gate_mean',
            'last_axial_cross_gate_chem_mean', 'last_axial_cross_gate_psd_mean',
            'last_axial_time_no_key_fraction', 'last_axial_cross_no_key_fraction',
            'last_axial_cross_valid_query_fraction',
            'last_axial_cross_entropy_missing', 'last_axial_cross_top1_mass',
            'last_axial_cross_top3_mass', 'last_axial_psd_to_chem_mass',
            'last_axial_psd_to_psd_mass', 'last_local_chunk_graph_gate',
            'last_local_chunk_graph_out_proj_norm',
            'last_local_chunk_graph_obs_ratio_mean',
        ):
            setattr(self, name, getattr(self.encoder, name, None))
        should_sample_latent = (
            self.latent_mode == 'variational' and bool(sample_latent)
        )
        z = self.reparameterize(mu, logvar) if should_sample_latent else mu
        self.last_log_det_J = None
        self.last_z0 = z
        self.last_zK = z
        if self.use_realnvp:
            z, self.last_log_det_J = self.flow(z)
            self.last_zK = z
        self.decoder.current_epoch = self.current_epoch
        recon_mean, recon_logvar = self.decoder(
            z,
            cond,
            enc_h_seq=h_seq,
            obs_mask=mask,
            local_context=local_context,
            attn_weighted_support_t=attn_weighted_support_t,
        )
        self.last_local_context_gate = getattr(
            self.decoder, 'last_local_context_gate', None
        )
        lca = getattr(self.decoder, 'local_context_attn_fusion', None)
        for name, source in (
            ('last_local_context_attn_entropy', 'last_attn_entropy'),
            ('last_local_context_attn_center_distance', 'last_attn_center_distance'),
            ('last_local_context_attn_support_mean', 'last_attn_support_mean'),
            ('last_local_context_attn_high_support_mass', 'last_attn_high_support_mass'),
            ('last_local_context_gate_low_support_mean', 'last_gate_low_support_mean'),
            ('last_local_context_gate_high_support_mean', 'last_gate_high_support_mean'),
        ):
            setattr(self, name, getattr(lca, source, None))
        self.last_local_context_generation_support_mean = getattr(
            self.encoder, 'last_local_context_generation_support_mean', None
        )
        self.last_local_context_observe_aware_blend_gate = getattr(
            self.encoder, 'last_local_context_observe_aware_blend_gate', None
        )
        for name in (
            'last_support_logvar_residual_mean',
            'last_support_logvar_residual_missing_mean',
            'last_support_logvar_residual_psd_low_support_mean',
            'last_support_logvar_residual_psd_high_support_mean',
            'last_support_logvar_monotone_beta',
            'last_support_logvar_anchor_beta',
            'last_feature_logvar_bias_mean',
            'last_feature_logvar_bias_chem_mean',
            'last_feature_logvar_bias_psd_mean',
            'last_feature_logvar_bias_psd_min',
            'last_feature_logvar_bias_psd_max',
            'last_logvar_diagnostics',
        ):
            setattr(self, name, getattr(self.decoder, name, None))
        for name in (
            'last_temporal_refiner_gate',
            'last_temporal_refiner_attn_entropy_missing',
            'last_temporal_refiner_attn_entropy_observed',
        ):
            setattr(self, name, getattr(self.encoder, name, None))
        recon_mean = recon_mean.permute(0, 2, 1)
        if recon_logvar is not None:
            recon_logvar = recon_logvar.permute(0, 2, 1)
        return recon_mean, recon_logvar, mu, logvar, graph_attention

    def _enable_mc_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def compute_uncertainty(self, x, cond, mask, n_samples=50,
                            dist_type='gaussian', history=None,
                            return_extra_quantiles=False, return_samples=False,
                            enable_mc_dropout=True, sample_latent=True,
                            sample_likelihood=True, mc_batch_size=1,
                            amp_dtype=None):
        self.eval()
        if enable_mc_dropout:
            self._enable_mc_dropout()
        if int(mc_batch_size) < 1:
            raise ValueError(f"mc_batch_size must be >= 1, got {mc_batch_size}")
        mc_batch_size = min(int(mc_batch_size), int(n_samples))
        batch = x.shape[0]
        all_means = []
        all_logvars = []
        all_samples = []
        graph_attn_sum = None

        def replicate_history(value, count):
            if value is None or count == 1:
                return value
            out = {}
            for key, item in value.items():
                if isinstance(item, torch.Tensor):
                    out[key] = item.repeat(
                        count, *([1] * (item.dim() - 1))
                    )
                else:
                    out[key] = item
            return out

        emitted = 0
        while emitted < n_samples:
            current = min(mc_batch_size, n_samples - emitted)
            if current == 1:
                x_in, cond_in, mask_in = x, cond, mask
                history_in = history
            else:
                x_in = x.repeat(current, *([1] * (x.dim() - 1)))
                cond_in = cond.repeat(current, *([1] * (cond.dim() - 1)))
                mask_in = mask.repeat(current, *([1] * (mask.dim() - 1)))
                history_in = replicate_history(history, current)
            use_amp = (
                amp_dtype is not None and torch.cuda.is_available() and x.is_cuda
            )
            context = (
                torch.autocast(device_type='cuda', dtype=amp_dtype)
                if use_amp else nullcontext()
            )
            with torch.no_grad():
                with context:
                    value = self.forward(
                        x_in,
                        cond_in,
                        mask_in,
                        history=history_in,
                        sample_latent=sample_latent,
                    )
                recon_mean, recon_logvar, graph_attn = value[0], value[1], value[4]
                if recon_mean.dtype != torch.float32:
                    recon_mean = recon_mean.float()
                if recon_logvar is not None and recon_logvar.dtype != torch.float32:
                    recon_logvar = recon_logvar.float()
                if recon_logvar is None or not sample_likelihood:
                    y_sample = recon_mean
                elif dist_type == 'student_t':
                    df = self.get_likelihood_df(
                        recon_mean.shape[-1],
                        device=recon_mean.device,
                        dtype=recon_mean.dtype,
                    ).view(1, 1, -1)
                    df_full = df.expand_as(recon_mean)
                    variance = torch.exp(recon_logvar)
                    sigma = torch.sqrt(
                        (variance * (df - 2.0) / df).clamp(min=1e-10)
                    )
                    chi2 = 2.0 * torch._standard_gamma(
                        (df_full / 2.0).clamp(min=1e-6)
                    )
                    t_eps = (
                        torch.randn_like(recon_mean)
                        * torch.sqrt(df_full / chi2.clamp(min=1e-10))
                    )
                    y_sample = recon_mean + sigma * t_eps
                else:
                    y_sample = (
                        recon_mean
                        + torch.exp(0.5 * recon_logvar)
                        * torch.randn_like(recon_mean)
                    )
            if current == 1:
                all_means.append(recon_mean)
                if recon_logvar is not None:
                    all_logvars.append(recon_logvar)
                all_samples.append(y_sample)
                if graph_attn is not None:
                    graph_attn_sum = (
                        graph_attn.clone()
                        if graph_attn_sum is None
                        else graph_attn_sum + graph_attn
                    )
            else:
                mean_split = recon_mean.view(
                    current, batch, *recon_mean.shape[1:]
                )
                sample_split = y_sample.view(
                    current, batch, *y_sample.shape[1:]
                )
                logvar_split = (
                    recon_logvar.view(current, batch, *recon_logvar.shape[1:])
                    if recon_logvar is not None else None
                )
                for index in range(current):
                    all_means.append(mean_split[index])
                    all_samples.append(sample_split[index])
                    if logvar_split is not None:
                        all_logvars.append(logvar_split[index])
                if graph_attn is not None:
                    chunk_sum = graph_attn.view(
                        current, batch, *graph_attn.shape[1:]
                    ).sum(dim=0)
                    graph_attn_sum = (
                        chunk_sum if graph_attn_sum is None
                        else graph_attn_sum + chunk_sum
                    )
            emitted += current
        self.eval()
        means = torch.stack(all_means, dim=0)
        samples = torch.stack(all_samples, dim=0)
        epistemic_var = means.var(dim=0)
        pred_mean = means.mean(dim=0)
        epi_q05 = torch.quantile(means, 0.05, dim=0)
        epi_q95 = torch.quantile(means, 0.95, dim=0)
        epi_q025 = torch.quantile(means, 0.025, dim=0)
        epi_q975 = torch.quantile(means, 0.975, dim=0)
        pred_q05 = torch.quantile(samples, 0.05, dim=0)
        pred_q95 = torch.quantile(samples, 0.95, dim=0)
        pred_q025 = torch.quantile(samples, 0.025, dim=0)
        pred_q975 = torch.quantile(samples, 0.975, dim=0)
        aleatoric_var = None
        if all_logvars:
            aleatoric_var = torch.exp(
                torch.stack(all_logvars, dim=0)
            ).mean(dim=0)
        total_var = epistemic_var + (
            aleatoric_var if aleatoric_var is not None else 0
        )
        if graph_attn_sum is not None:
            avg_graph_attn = graph_attn_sum / n_samples
            self.last_graph_attention = avg_graph_attn.detach().mean(dim=0)
        else:
            avg_graph_attn = None
            self.last_graph_attention = None
        result = (
            pred_mean, epistemic_var, aleatoric_var, total_var,
            avg_graph_attn, pred_q05, pred_q95, epi_q05, epi_q95,
        )
        if return_extra_quantiles:
            result = result + (
                pred_q025, pred_q975, epi_q025, epi_q975,
            )
        if return_samples:
            result = result + (samples, means)
        return result

    def get_learned_graph(self):
        return (
            self.last_graph_attention.cpu().numpy()
            if self.last_graph_attention is not None else None
        )

    def get_learned_graph_heads(self):
        return (
            self.last_graph_attention_heads.cpu().numpy()
            if self.last_graph_attention_heads is not None else None
        )

    def get_cross_modal_graph(self):
        return (
            self.last_cross_modal_attention.cpu().numpy()
            if self.last_cross_modal_attention is not None else None
        )

    def get_cross_modal_graph_heads(self):
        return (
            self.last_cross_modal_attention_heads.cpu().numpy()
            if self.last_cross_modal_attention_heads is not None else None
        )

    def get_learned_graph_heads_batch(self):
        value = getattr(
            self.encoder, 'last_input_graph_attention_heads_batch', None
        )
        return value.cpu().numpy() if value is not None else None

    def get_cross_modal_graph_heads_batch(self):
        value = getattr(
            self.encoder, 'last_cross_modal_attention_heads_batch', None
        )
        return value.cpu().numpy() if value is not None else None

    def get_learned_graph_batch(self):
        value = getattr(self.encoder, 'last_input_graph_attention_batch', None)
        return value.cpu().numpy() if value is not None else None

    def get_cross_modal_graph_batch(self):
        value = getattr(self.encoder, 'last_cross_modal_attention_batch', None)
        return value.cpu().numpy() if value is not None else None

    def get_learned_graph_per_layer(self):
        per_layer = getattr(
            self.encoder, 'last_input_graph_attention_per_layer', None
        )
        if not per_layer:
            return None
        return [{
            'avg': (
                item['avg'].cpu().numpy() if item['avg'] is not None else None
            ),
            'heads': (
                item['heads'].cpu().numpy()
                if item['heads'] is not None else None
            ),
        } for item in per_layer]


def load_from_uq_model(graph_model, uq_model_path, device):
    """Initialize matching model layers from a legacy UQ checkpoint."""
    uq_state = torch.load(uq_model_path, map_location=device, weights_only=True)
    graph_state = graph_model.state_dict()
    copied = 0
    for name, parameter in uq_state.items():
        if name in graph_state and graph_state[name].shape == parameter.shape:
            graph_state[name] = parameter
            copied += 1
    graph_model.load_state_dict(graph_state)
    print(f"Loaded {copied} layers from UQ model")
    return graph_model
