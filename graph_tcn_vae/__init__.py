"""Graph-enhanced TCN-VAE model architectures."""

from .api import fit_multimodal, impute_multimodal, validate_multimodal_data
from .bundle import inspect_bundle
from .config import TrainConfig
from .contracts import (
    DataSchema,
    InferenceConfig,
    ModalityFiles,
    ModalityInputs,
    ModalityPreprocessing,
    PreprocessingConfig,
)
from .infer import impute, load_bundle
from .latent_diffusion import (
    ConditionalLatentResidualDiffusion,
    HeldoutLikeMaskConfig,
    LatentResidualDiffusionConfig,
    LatentTeacherResult,
    build_latent_condition,
    optimize_latent_target,
    sample_heldout_like_mask,
)
from .model import ImputationVAE
from .model_config import ModelConfig
from .model_graph_pred import PredictionVAE_Graph
from .model_graph_uq import ImputationVAE_Graph
from .model_uq import ImputationVAE_UQ
from .train import train_from_config

__all__ = [
    "DataSchema",
    "InferenceConfig",
    "ModalityFiles",
    "ModalityInputs",
    "ModalityPreprocessing",
    "PreprocessingConfig",
    "ConditionalLatentResidualDiffusion",
    "HeldoutLikeMaskConfig",
    "LatentResidualDiffusionConfig",
    "LatentTeacherResult",
    "ImputationVAE",
    "ImputationVAE_UQ",
    "ImputationVAE_Graph",
    "ModelConfig",
    "PredictionVAE_Graph",
    "TrainConfig",
    "build_latent_condition",
    "fit_multimodal",
    "impute_multimodal",
    "validate_multimodal_data",
    "inspect_bundle",
    "optimize_latent_target",
    "sample_heldout_like_mask",
    "train_from_config",
    "load_bundle",
    "impute",
]
