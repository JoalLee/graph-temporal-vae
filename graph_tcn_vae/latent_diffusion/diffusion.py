"""Conditional DDPM for context-compatible latent residuals."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .config import LatentResidualDiffusionConfig
from .denoiser import ConditionalResidualDenoiser


class ConditionalLatentResidualDiffusion(nn.Module):
    """Model ``p(delta_z | observed context)`` with a compact latent-space DDPM.

    The caller supplies a fixed-size condition vector.  This keeps the diffusion
    module independent of the particular VAE encoder and lets legacy checkpoints
    use an adapter without changing baseline behaviour.
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        config: Optional[LatentResidualDiffusionConfig] = None,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or condition_dim < 1:
            raise ValueError("latent_dim and condition_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.config = config or LatentResidualDiffusionConfig()
        self.denoiser = ConditionalResidualDenoiser(
            latent_dim=self.latent_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.config.hidden_dim,
            time_embedding_dim=self.config.time_embedding_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.timesteps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        ).clamp_min(1e-20)
        posterior_mean_coef1 = (
            betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        )
        posterior_mean_coef2 = (
            (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _validate_inputs(
        self,
        delta_z: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        if delta_z.ndim != 2 or delta_z.shape[1] != self.latent_dim:
            raise ValueError(
                f"delta_z must have shape [batch, {self.latent_dim}], "
                f"got {tuple(delta_z.shape)}"
            )
        if condition.ndim != 2 or condition.shape != (
            delta_z.shape[0],
            self.condition_dim,
        ):
            raise ValueError(
                "condition must have shape "
                f"[batch, {self.condition_dim}], got {tuple(condition.shape)}"
            )

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
        selected = values.gather(0, timesteps)
        return selected.view(selected.shape[0], *([1] * (ndim - 1)))

    def q_sample(
        self,
        delta_z: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the forward diffusion process to a clean latent correction."""

        return (
            self._extract(self.sqrt_alpha_bars, timesteps, delta_z.ndim) * delta_z
            + self._extract(
                self.sqrt_one_minus_alpha_bars, timesteps, delta_z.ndim
            )
            * noise
        )

    def loss(
        self,
        delta_z: torch.Tensor,
        condition: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the standard epsilon-prediction denoising objective."""

        self._validate_inputs(delta_z, condition)
        batch = delta_z.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.config.timesteps,
                (batch,),
                device=delta_z.device,
            )
        if timesteps.shape != (batch,) or timesteps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("timesteps must be an integer tensor with shape [batch]")
        if noise is None:
            noise = torch.randn_like(delta_z)
        if noise.shape != delta_z.shape:
            raise ValueError("noise must have the same shape as delta_z")

        noisy_delta = self.q_sample(delta_z, timesteps.long(), noise)
        predicted_noise = self.denoiser(noisy_delta, timesteps.long(), condition)
        return torch.mean((predicted_noise - noise) ** 2)

    def _predict_x0(
        self,
        noisy_delta: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bars, timesteps, noisy_delta.ndim)
        clean = (
            noisy_delta
            - torch.sqrt((1.0 - alpha_bar).clamp_min(1e-20)) * predicted_noise
        ) / torch.sqrt(alpha_bar.clamp_min(1e-20))
        if self.config.clip_denoised is not None:
            clean = clean.clamp(
                min=-self.config.clip_denoised,
                max=self.config.clip_denoised,
            )
        return clean

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        num_samples: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw latent corrections with shape ``[samples, batch, latent_dim]``."""

        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                "condition must have shape "
                f"[batch, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        batch = condition.shape[0]
        repeated_condition = condition.unsqueeze(0).expand(
            num_samples, -1, -1
        ).reshape(num_samples * batch, self.condition_dim)
        current = torch.randn(
            num_samples * batch,
            self.latent_dim,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )

        for step in range(self.config.timesteps - 1, -1, -1):
            timesteps = torch.full(
                (current.shape[0],),
                step,
                device=current.device,
                dtype=torch.long,
            )
            predicted_noise = self.denoiser(
                current,
                timesteps,
                repeated_condition,
            )
            clean = self._predict_x0(current, timesteps, predicted_noise)
            mean = (
                self._extract(self.posterior_mean_coef1, timesteps, current.ndim)
                * clean
                + self._extract(
                    self.posterior_mean_coef2, timesteps, current.ndim
                )
                * current
            )
            if step == 0:
                current = mean
            else:
                noise = torch.randn(
                    current.shape,
                    device=current.device,
                    dtype=current.dtype,
                    generator=generator,
                )
                variance = self._extract(
                    self.posterior_variance, timesteps, current.ndim
                )
                current = mean + torch.sqrt(variance) * noise

        return current.view(num_samples, batch, self.latent_dim)

    def checkpoint_payload(self) -> dict:
        """Return a self-describing state payload for experiment checkpoints."""

        return {
            "artifact_type": "conditional_latent_residual_diffusion",
            "latent_dim": self.latent_dim,
            "condition_dim": self.condition_dim,
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: dict,
        *,
        map_location: Optional[torch.device | str] = None,
    ) -> "ConditionalLatentResidualDiffusion":
        if payload.get("artifact_type") != "conditional_latent_residual_diffusion":
            raise ValueError("payload is not a conditional latent residual diffusion checkpoint")
        model = cls(
            latent_dim=int(payload["latent_dim"]),
            condition_dim=int(payload["condition_dim"]),
            config=LatentResidualDiffusionConfig(**payload["config"]),
        )
        state_dict = payload["state_dict"]
        if map_location is not None:
            state_dict = {
                key: value.to(map_location) if isinstance(value, torch.Tensor) else value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=True)
        return model
