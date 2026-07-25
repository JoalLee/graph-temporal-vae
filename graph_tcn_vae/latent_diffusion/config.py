"""Configuration for conditional latent residual diffusion."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LatentResidualDiffusionConfig:
    """Hyperparameters for a DDPM over context-compatible latent corrections."""

    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden_dim: int = 512
    time_embedding_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.0
    clip_denoised: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timesteps < 2:
            raise ValueError("timesteps must be >= 2")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("beta_start and beta_end must satisfy 0 < start < end < 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.time_embedding_dim < 2:
            raise ValueError("time_embedding_dim must be >= 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.clip_denoised is not None and self.clip_denoised <= 0.0:
            raise ValueError("clip_denoised must be positive when provided")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
