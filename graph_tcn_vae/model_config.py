"""Typed architecture configuration for :class:`ImputationVAE_Graph`.

Dataset-derived dimensions (``target_dim``, ``aux_dim``, and ``window_size``)
remain explicit model arguments. All optional architecture and ablation
settings are represented here without changing the legacy constructor API.
"""

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Optional Graph-TCN-VAE architecture and ablation settings."""

    latent_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    encoder_layers: int = 5
    decoder_layers: int = 5
    kernel_size: int = 3
    dropout: float = 0.1
    heteroscedastic: bool = True
    n_graph_heads: int = 4
    var_min: float = 0.001
    var_max: float = 10.0
    n_chem: int = 0
    use_input_graph_layer: bool = True
    use_cross_modal_graph: bool = True
    use_tcn: bool = True
    n_input_graph_layers: int = 1
    use_progressive_decoder: bool = False
    decoder_initial_steps: int = 12
    cond_film_last_n: Optional[int] = None
    cond_film_gamma_scale: float = 0.5
    use_decoder_cross_attn: bool = False
    n_cross_attn_heads: int = 4
    decoder_cross_attn_missing_only: bool = False
    use_parallel_graph: bool = False
    use_realnvp: bool = False
    realnvp_layers: int = 4
    use_temporal_cnn: bool = True
    use_hybrid_time_encoding: bool = False
    time_numeric_dim: int = 4
    time_cyc_dim: int = 6
    time_hybrid_dim: int = 6
    hour_embed_dim: int = 8
    dow_embed_dim: int = 4
    month_embed_dim: int = 4
    film_kernel_size: int = 1
    film_gamma_kernel_size: Optional[int] = None
    film_beta_kernel_size: Optional[int] = None
    film_temporal_last_n: int = 0
    film_temporal_last_kernel_size: int = 3
    z_film_alpha_init: float = -2.0
    z_skip_gate_init: float = -2.0
    use_z_skip: bool = True
    decoder_cross_attn_gate_init: float = -1.5
    use_dual_output_heads: bool = False
    output_head_hidden_dim: Optional[int] = None
    use_detached_variance_pathway: bool = False
    variance_detach_start_epoch: Optional[int] = None
    variance_path_use_latent: bool = True
    variance_path_detach_latent: bool = False
    variance_head_hidden_dim: Optional[int] = None
    variance_path_use_mask: bool = False
    variance_mask_dim: int = 32
    variance_use_grouped_conv: bool = False
    use_local_context_map: bool = False
    local_context_dim: int = 32
    local_context_steps: Optional[int] = None
    local_context_gate_init: float = -2.0
    local_context_observe_aware: bool = False
    local_context_observe_aware_blend_gate_init: Optional[float] = None
    local_context_injection_mode: str = "seed"
    local_context_fusion_mode: str = "add"
    local_context_attn_heads: int = 4
    local_context_attn_window_tokens: int = 1
    local_context_attn_gate_init: float = -2.0
    local_context_attn_after_tcn_layers: Optional[int] = None
    local_context_attn_location: str = "mid_tcn"
    local_context_attn_support_bias_scale: float = 2.0
    local_context_attn_gate_support_power: float = 0.0
    local_context_attn_gate_support_floor: float = 0.0
    local_context_attn_logvar_support_boost: float = 0.0
    use_pregraph_feature_temporal_attn: bool = False
    use_pregraph_depthwise_tcn: bool = True
    use_axial_observed_attn: bool = False
    axial_attn_dim: int = 64
    axial_attn_heads: int = 4
    axial_time_gate_init: float = 0.0
    axial_cross_gate_init: float = 0.0
    axial_cross_time_chunk: int = 4
    axial_null_output: bool = False
    pregraph_feature_temporal_attn_dim: int = 64
    pregraph_feature_temporal_attn_heads: int = 4
    pregraph_feature_temporal_attn_gate_init: float = -1.0
    pregraph_feature_temporal_attn_chunk_size: int = 256
    pregraph_feature_temporal_attn_record_weights: bool = False
    pregraph_feature_temporal_attn_mode: str = "dense"
    use_temporal_refiner: bool = False
    temporal_refiner_dim: int = 128
    temporal_refiner_heads: int = 4
    temporal_refiner_gate_init: float = -2.0
    temporal_refiner_fixed_gate: Optional[float] = None
    use_decoder_final_norm: bool = False
    use_latent_pooled_norm: bool = False
    latent_logvar_min: Optional[float] = None
    latent_logvar_max: Optional[float] = None
    enable_cross_modal_floor: bool = False
    disable_rel_scale: bool = False
    disable_prior_bias: bool = False
    disable_aux_bias: bool = False
    cross_modal_query_gate_mode: str = "legacy_hard"
    use_homogeneous: bool = False
    ignore_obs_mask: bool = False
    use_graph_ffn: bool = False
    graph_ffn_mult: int = 4
    use_token_graph_trunk: bool = False
    token_graph_dim: Optional[int] = None
    token_graph_out_gate_init: float = -1.0
    use_local_chunk_graph: bool = False
    local_chunk_graph_mode: str = "parallel"
    local_chunk_graph_chunk_size: int = 6
    local_chunk_graph_dim: int = 128
    local_chunk_graph_heads: int = 4
    local_chunk_graph_gate_init: float = -2.0
    local_chunk_graph_ffn_mult: int = 4
    local_chunk_graph_use_mask_embed: bool = False
    local_chunk_graph_out_proj_init_std: float = 0.0
    use_external_history_context: bool = False
    external_history_dim: int = 128
    external_history_heads: int = 4
    external_history_gate_init: float = -2.0
    external_history_use_retrieval_bias: bool = False
    external_history_time_decay: float = 0.0
    external_history_support_bias: float = 0.0
    external_history_null_penalty: float = 0.0
    history_num_chunks: int = 28
    history_chunk_size: int = 24
    history_support_dim: int = 6
    use_variance_attn_support: bool = False
    variance_attn_support_dim: int = 32
    use_support_logvar_residual: bool = False
    support_logvar_hidden_dim: int = 32
    support_logvar_missing_only: bool = True
    support_logvar_monotone: bool = False
    support_logvar_monotone_init: float = -2.0
    support_logvar_use_anchor: bool = False
    support_logvar_anchor_init: float = -2.0
    use_feature_logvar_bias: bool = False
    feature_logvar_bias_scope: str = "psd"
    feature_logvar_bias_init: float = 0.0
    feature_logvar_bias_constraint: str = "none"
    use_learnable_likelihood_df: bool = False
    likelihood_df_scope: str = "family"
    likelihood_df_init: float = 3.0
    likelihood_df_min: float = 2.1
    likelihood_df_max: float = 30.0
    use_learnable_prior_df: bool = False
    prior_df_init: float = 3.0
    prior_df_min: float = 2.1
    prior_df_max: float = 30.0

    @classmethod
    def field_names(cls) -> set[str]:
        """Return the accepted architecture option names."""
        return {item.name for item in fields(cls)}

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
        """Build a config while rejecting misspelled or stale fields."""
        unknown = set(values) - cls.field_names()
        if unknown:
            raise TypeError(f"ModelConfig got unexpected field(s): {sorted(unknown)}")
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> "ModelConfig":
        """Return a copied config with validated overrides."""
        unknown = set(overrides) - self.field_names()
        if unknown:
            raise TypeError(f"ModelConfig got unexpected field(s): {sorted(unknown)}")
        return replace(self, **overrides)

    def to_kwargs(self) -> Dict[str, Any]:
        """Return independent constructor kwargs, including copied lists."""
        return asdict(self)
