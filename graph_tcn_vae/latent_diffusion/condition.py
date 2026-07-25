"""Condition-vector construction for latent residual diffusion."""

from __future__ import annotations

from typing import Optional

import torch


def _sequence_moments(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, channels, time]")
    mean = value.mean(dim=-1)
    std = value.std(dim=-1, unbiased=False)
    return torch.cat([mean, std], dim=1)


def _mask_family_summary(mask: torch.Tensor) -> torch.Tensor:
    """Return five geometry features for one target family."""

    batch = mask.shape[0]
    if mask.shape[-1] == 0:
        return torch.zeros(batch, 5, device=mask.device, dtype=mask.dtype)
    mask = mask.float()
    time_fraction = mask.mean(dim=2)
    feature_fraction = mask.mean(dim=1)
    return torch.stack(
        [
            mask.mean(dim=(1, 2)),
            time_fraction.std(dim=1, unbiased=False),
            time_fraction.amin(dim=1),
            time_fraction.amax(dim=1),
            feature_fraction.std(dim=1, unbiased=False),
        ],
        dim=1,
    )


def build_latent_condition(
    *,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    encoder_sequence: Optional[torch.Tensor],
    local_context: Optional[torch.Tensor],
    observation_mask: torch.Tensor,
    n_chem: int,
    extra_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a fixed vector from the masked posterior and encoder context.

    ``encoder_sequence`` and ``local_context`` use channel-first shapes
    ``[batch, channels, time]``.  ``observation_mask`` uses ``1=observed`` and
    shape ``[batch, time, targets]``.
    """

    if mu.ndim != 2 or logvar.shape != mu.shape:
        raise ValueError("mu and logvar must share shape [batch, latent_dim]")
    if observation_mask.ndim != 3 or observation_mask.shape[0] != mu.shape[0]:
        raise ValueError(
            "observation_mask must have shape [batch, time, target_dim]"
        )
    target_dim = observation_mask.shape[-1]
    if not 0 <= n_chem <= target_dim:
        raise ValueError("n_chem must be between zero and target_dim")

    vectors = [mu, logvar]
    if encoder_sequence is not None:
        if encoder_sequence.shape[0] != mu.shape[0]:
            raise ValueError("encoder_sequence batch dimension does not match mu")
        vectors.append(_sequence_moments(encoder_sequence, "encoder_sequence"))
    if local_context is not None:
        if local_context.shape[0] != mu.shape[0]:
            raise ValueError("local_context batch dimension does not match mu")
        vectors.append(_sequence_moments(local_context, "local_context"))

    mask = observation_mask.float()
    vectors.extend(
        [
            _mask_family_summary(mask),
            _mask_family_summary(mask[:, :, :n_chem]),
            _mask_family_summary(mask[:, :, n_chem:]),
        ]
    )
    if extra_features is not None:
        if extra_features.ndim != 2 or extra_features.shape[0] != mu.shape[0]:
            raise ValueError("extra_features must have shape [batch, features]")
        vectors.append(extra_features)

    condition = torch.cat(vectors, dim=1)
    if not bool(torch.isfinite(condition).all()):
        raise ValueError("condition contains non-finite values")
    return condition
