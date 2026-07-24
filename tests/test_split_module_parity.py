"""Compatibility contracts for the split Graph-TCN-VAE implementation."""

import inspect

import torch

from graph_tcn_vae import model_graph_uq as facade
from graph_tcn_vae.flows import AffineCouplingLayer, RealNVP, ReverseLayer
from graph_tcn_vae.graph_blocks import (
    AxialObservedAttentionBlock,
    DepthwiseTCN,
    LocalContextMemoryAttention,
    MaskedTemporalCrossAttention,
    PreGraphPerFeatureTemporalAttention,
    RotarySelfAttention,
    TemporalAttentionPool,
    TemporalObservationRefiner,
    TimeHybridEncoder,
    WindowTokenFFN,
)
from graph_tcn_vae.graph_layers import (
    CrossModalGraphLayer,
    ExternalHistoryContext,
    InputGraphLayer,
    LocalChunkGraphBranch,
    TokenGraphCrossBlock,
    TokenGraphFFN,
    TokenGraphSelfBlock,
)
from graph_tcn_vae.graph_model import (
    GraphDecoder,
    GraphEncoder,
    ImputationVAE_Graph,
)
from graph_tcn_vae.model_config import ModelConfig
from graph_tcn_vae.vanilla_vae import VanillaVAE


def test_model_graph_uq_facade_reexports_split_symbols():
    expected = {
        "AffineCouplingLayer": AffineCouplingLayer,
        "AxialObservedAttentionBlock": AxialObservedAttentionBlock,
        "CrossModalGraphLayer": CrossModalGraphLayer,
        "DepthwiseTCN": DepthwiseTCN,
        "ExternalHistoryContext": ExternalHistoryContext,
        "GraphDecoder": GraphDecoder,
        "GraphEncoder": GraphEncoder,
        "ImputationVAE_Graph": ImputationVAE_Graph,
        "InputGraphLayer": InputGraphLayer,
        "LocalChunkGraphBranch": LocalChunkGraphBranch,
        "LocalContextMemoryAttention": LocalContextMemoryAttention,
        "MaskedTemporalCrossAttention": MaskedTemporalCrossAttention,
        "ModelConfig": ModelConfig,
        "PreGraphPerFeatureTemporalAttention": PreGraphPerFeatureTemporalAttention,
        "RealNVP": RealNVP,
        "ReverseLayer": ReverseLayer,
        "RotarySelfAttention": RotarySelfAttention,
        "TemporalAttentionPool": TemporalAttentionPool,
        "TemporalObservationRefiner": TemporalObservationRefiner,
        "TimeHybridEncoder": TimeHybridEncoder,
        "TokenGraphCrossBlock": TokenGraphCrossBlock,
        "TokenGraphFFN": TokenGraphFFN,
        "TokenGraphSelfBlock": TokenGraphSelfBlock,
        "VanillaVAE": VanillaVAE,
        "WindowTokenFFN": WindowTokenFFN,
    }
    for name, implementation in expected.items():
        assert getattr(facade, name) is implementation
        assert name in facade.__all__


def _small_26e_style_kwargs():
    return {
        "target_dim": 7,
        "aux_dim": 11,
        "window_size": 12,
        "latent_dim": 8,
        "hidden_dims": [12, 12],
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.0,
        "heteroscedastic": True,
        "n_graph_heads": 1,
        "n_chem": 3,
        "use_input_graph_layer": True,
        "use_cross_modal_graph": True,
        "use_tcn": True,
        "n_input_graph_layers": 2,
        "use_progressive_decoder": True,
        "decoder_initial_steps": 3,
        "cond_film_last_n": 2,
        "cond_film_gamma_scale": 0.3,
        "use_parallel_graph": True,
        "use_realnvp": True,
        "realnvp_layers": 2,
        "use_temporal_cnn": True,
        "use_hybrid_time_encoding": True,
        "time_numeric_dim": 5,
        "time_cyc_dim": 6,
        "time_hybrid_dim": 6,
        "use_dual_output_heads": True,
        "use_detached_variance_pathway": True,
        "variance_path_use_latent": True,
        "variance_head_hidden_dim": 16,
        "use_local_context_map": True,
        "local_context_dim": 4,
        "local_context_steps": 3,
        "local_context_observe_aware": True,
        "local_context_injection_mode": "post_upsample",
        "use_pregraph_feature_temporal_attn": True,
        "pregraph_feature_temporal_attn_dim": 8,
        "pregraph_feature_temporal_attn_heads": 1,
        "pregraph_feature_temporal_attn_chunk_size": 0,
        "use_decoder_final_norm": True,
        "use_latent_pooled_norm": True,
        "use_graph_ffn": True,
        "use_homogeneous": True,
        "use_feature_logvar_bias": True,
        "feature_logvar_bias_scope": "psd",
    }


def test_split_model_strict_state_round_trip_and_deterministic_forward():
    kwargs = _small_26e_style_kwargs()
    torch.manual_seed(123)
    source = facade.ImputationVAE_Graph(**kwargs).eval()
    torch.manual_seed(999)
    restored = ImputationVAE_Graph(**kwargs).eval()
    restored.load_state_dict(source.state_dict(), strict=True)

    x = torch.randn(2, 12, 7)
    cond = torch.randn(2, 12, 11)
    mask = (torch.rand(2, 12, 7) > 0.2).float()
    with torch.no_grad():
        source_output = source(x, cond, mask, sample_latent=False)
        restored_output = restored(x, cond, mask, sample_latent=False)

    for source_value, restored_value in zip(source_output, restored_output):
        if source_value is None:
            assert restored_value is None
        else:
            torch.testing.assert_close(source_value, restored_value)


def test_runtime_model_uses_split_module_paths():
    model = facade.ImputationVAE_Graph(**_small_26e_style_kwargs())
    assert type(model).__module__ == "graph_tcn_vae.graph_model.vae"
    assert type(model.encoder).__module__ == "graph_tcn_vae.graph_model.encoder"
    assert type(model.decoder).__module__ == "graph_tcn_vae.graph_model.decoder"
    assert type(model.flow).__module__ == "graph_tcn_vae.flows"


def test_model_config_matches_legacy_constructor_defaults():
    signature = inspect.signature(ImputationVAE_Graph.__init__)
    optional = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name not in {"self", "target_dim", "aux_dim", "window_size"}
    }
    config = ModelConfig()
    assert set(optional) == ModelConfig.field_names()
    for name, parameter in optional.items():
        assert getattr(config, name) == parameter.default


def test_from_config_matches_direct_constructor_state():
    model_options = _small_26e_style_kwargs()
    dimensions = {
        key: model_options.pop(key)
        for key in ("target_dim", "aux_dim", "window_size")
    }
    config = ModelConfig.from_dict(model_options)

    torch.manual_seed(123)
    direct = ImputationVAE_Graph(**dimensions, **model_options)
    torch.manual_seed(123)
    configured = ImputationVAE_Graph.from_config(**dimensions, config=config)

    assert direct.state_dict().keys() == configured.state_dict().keys()
    for name, value in direct.state_dict().items():
        torch.testing.assert_close(value, configured.state_dict()[name])


def test_vanilla_vae_strict_state_round_trip():
    kwargs = {
        "input_dim": 5,
        "window_size": 8,
        "latent_dim": 4,
        "hidden_dims": [12, 8],
        "target_dim": 3,
        "chem_dim": 1,
        "psd_dim": 2,
        "use_realnvp": True,
        "realnvp_layers": 2,
    }
    torch.manual_seed(123)
    source = facade.VanillaVAE(**kwargs).eval()
    restored = VanillaVAE(**kwargs).eval()
    restored.load_state_dict(source.state_dict(), strict=True)

    x = torch.randn(2, 8, 3)
    cond = torch.randn(2, 8, 2)
    mask = (torch.rand(2, 8, 3) > 0.2).float()
    with torch.no_grad():
        source_output = source(x, cond, mask, sample_latent=False)
        restored_output = restored(x, cond, mask, sample_latent=False)
    for source_value, restored_value in zip(source_output, restored_output):
        if source_value is None:
            assert restored_value is None
        else:
            torch.testing.assert_close(source_value, restored_value)
