"""Backward-compatible imports for the Graph-TCN-VAE architecture.

The implementation is split across ``flows``, ``graph_blocks``,
``graph_layers``, ``graph_model``, and ``vanilla_vae``. Existing code may
continue importing the historical symbols from this module.
"""

from .flows import AffineCouplingLayer, RealNVP, ReverseLayer
from .graph_blocks import (
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
from .graph_layers import (
    CrossModalGraphLayer,
    ExternalHistoryContext,
    InputGraphLayer,
    LocalChunkGraphBranch,
    TokenGraphCrossBlock,
    TokenGraphFFN,
    TokenGraphSelfBlock,
)
from .graph_model import (
    GraphDecoder,
    GraphEncoder,
    ImputationVAE_Graph,
    load_from_uq_model,
)
from .model_config import ModelConfig
from .vanilla_vae import VanillaVAE

__all__ = [
    "AffineCouplingLayer",
    "AxialObservedAttentionBlock",
    "CrossModalGraphLayer",
    "DepthwiseTCN",
    "ExternalHistoryContext",
    "GraphDecoder",
    "GraphEncoder",
    "ImputationVAE_Graph",
    "InputGraphLayer",
    "LocalChunkGraphBranch",
    "LocalContextMemoryAttention",
    "MaskedTemporalCrossAttention",
    "ModelConfig",
    "PreGraphPerFeatureTemporalAttention",
    "RealNVP",
    "ReverseLayer",
    "RotarySelfAttention",
    "TemporalAttentionPool",
    "TemporalObservationRefiner",
    "TimeHybridEncoder",
    "TokenGraphCrossBlock",
    "TokenGraphFFN",
    "TokenGraphSelfBlock",
    "VanillaVAE",
    "WindowTokenFFN",
    "load_from_uq_model",
]
