"""Held-out-like dynamic masking for latent-diffusion teacher generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class HeldoutLikeMaskConfig:
    target_ratio: float = 0.10
    mean_duration: float = 48.0
    std_duration: float = 24.0
    min_duration: int = 3
    max_duration: int = 168
    psd_blocks_per_sample: int = 1
    chem_blocks_per_feature: int = 1
    psd_block_prob: Optional[float] = None
    chem_block_prob: Optional[float] = None
    point_dropout: float = 0.0
    min_psd_observed_fraction: float = 0.20
    min_chem_observed_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_ratio <= 1.0:
            raise ValueError("target_ratio must be in [0, 1]")
        if self.mean_duration <= 0.0 or self.std_duration < 0.0:
            raise ValueError("duration parameters are invalid")
        if self.min_duration < 1 or self.max_duration < self.min_duration:
            raise ValueError("duration bounds are invalid")
        if self.psd_blocks_per_sample < 1 or self.chem_blocks_per_feature < 1:
            raise ValueError("block counts must be >= 1")
        for name, value in (
            ("psd_block_prob", self.psd_block_prob),
            ("chem_block_prob", self.chem_block_prob),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.point_dropout <= 1.0:
            raise ValueError("point_dropout must be in [0, 1]")


def _sample_contiguous_blocks(
    batch_shape: tuple[int, ...],
    seq_len: int,
    *,
    mean_duration: float,
    std_duration: float,
    min_duration: int,
    max_duration: int,
    block_prob: float,
    n_blocks: int,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    max_duration = min(max_duration, seq_len)
    shape = (*batch_shape, n_blocks)
    if std_duration > 0.0:
        durations = (
            torch.randn(shape, device=device, generator=generator) * std_duration
            + mean_duration
        )
    else:
        durations = torch.full(shape, mean_duration, device=device)
    durations = durations.round().clamp(min=min_duration, max=max_duration).long()
    max_starts = (seq_len - durations + 1).clamp_min(1)
    starts = (
        torch.rand(shape, device=device, generator=generator) * max_starts.float()
    ).floor().long()
    ends = starts + durations
    active = torch.rand(shape, device=device, generator=generator) < block_prob
    time_index = torch.arange(seq_len, device=device)
    return (
        (time_index >= starts.unsqueeze(-1))
        & (time_index < ends.unsqueeze(-1))
        & active.unsqueeze(-1)
    ).any(dim=-2)


def sample_heldout_like_mask(
    observed_mask: torch.Tensor,
    *,
    n_chem: int,
    config: HeldoutLikeMaskConfig = HeldoutLikeMaskConfig(),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample artificial held-out points from currently observed values.

    PSD gaps are synchronized across all PSD bins. Chem gaps are sampled per
    feature. The returned boolean mask uses ``True=artificially held out``.
    """

    if observed_mask.ndim != 3:
        raise ValueError("observed_mask must have shape [batch, time, targets]")
    batch, seq_len, target_dim = observed_mask.shape
    if not 0 <= n_chem <= target_dim:
        raise ValueError("n_chem must be between zero and target_dim")
    n_psd = target_dim - n_chem
    observed = observed_mask.bool()
    heldout = torch.zeros_like(observed)

    expected_fraction = max(config.mean_duration / max(seq_len, 1), 1e-6)
    default_probability = min(1.0, config.target_ratio / expected_fraction)
    psd_probability = (
        config.psd_block_prob
        if config.psd_block_prob is not None
        else min(1.0, default_probability / config.psd_blocks_per_sample)
    )
    chem_probability = (
        config.chem_block_prob
        if config.chem_block_prob is not None
        else min(1.0, default_probability / config.chem_blocks_per_feature)
    )

    if n_psd > 0:
        psd_time = _sample_contiguous_blocks(
            (batch,),
            seq_len,
            mean_duration=config.mean_duration,
            std_duration=config.std_duration,
            min_duration=config.min_duration,
            max_duration=config.max_duration,
            block_prob=float(psd_probability),
            n_blocks=config.psd_blocks_per_sample,
            device=observed.device,
            generator=generator,
        )
        psd_drop = psd_time.unsqueeze(-1).expand(-1, -1, n_psd)
        heldout[:, :, n_chem:] = psd_drop & observed[:, :, n_chem:]

    if n_chem > 0:
        chem_time = _sample_contiguous_blocks(
            (batch, n_chem),
            seq_len,
            mean_duration=config.mean_duration,
            std_duration=config.std_duration,
            min_duration=config.min_duration,
            max_duration=config.max_duration,
            block_prob=float(chem_probability),
            n_blocks=config.chem_blocks_per_feature,
            device=observed.device,
            generator=generator,
        )
        chem_drop = chem_time.permute(0, 2, 1)
        heldout[:, :, :n_chem] = chem_drop & observed[:, :, :n_chem]

    if config.point_dropout > 0.0:
        random_drop = (
            torch.rand(
                observed.shape,
                device=observed.device,
                generator=generator,
            )
            < config.point_dropout
        )
        heldout |= random_drop & observed & ~heldout

    remaining = observed & ~heldout
    if n_psd > 0:
        psd_fraction = remaining[:, :, n_chem:].float().mean(dim=(1, 2))
        restore = psd_fraction < config.min_psd_observed_fraction
        if bool(restore.any()):
            heldout[restore, :, n_chem:] = False
    if n_chem > 0:
        chem_fraction = remaining[:, :, :n_chem].float().mean(dim=(1, 2))
        restore = chem_fraction < config.min_chem_observed_fraction
        if bool(restore.any()):
            heldout[restore, :, :n_chem] = False

    return heldout
