"""Teacher optimization for context-compatible latent correction targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class LatentTeacherResult:
    """Result of optimizing only the latent while keeping the decoder fixed."""

    optimized_latent: torch.Tensor
    delta_z: torch.Tensor
    initial_data_loss: float
    final_data_loss: float
    final_total_loss: float
    rms_displacement: float
    steps: int


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or target_mask.shape != target.shape:
        raise ValueError(
            "decode output, target, and target_mask must have identical shapes"
        )
    mask = target_mask.bool()
    if not bool(mask.any()):
        raise ValueError("target_mask must select at least one teacher target")
    if not bool(torch.isfinite(target[mask]).all()):
        raise ValueError("target contains non-finite values inside target_mask")
    residual = torch.where(mask, prediction - target, torch.zeros_like(prediction))
    return residual.square().sum() / mask.sum().to(dtype=prediction.dtype)


def optimize_latent_target(
    *,
    base_latent: torch.Tensor,
    decode: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    target_mask: torch.Tensor,
    steps: int = 20,
    learning_rate: float = 0.03,
    regularization: float = 1.0,
) -> LatentTeacherResult:
    """Optimize a latent pseudo-target under a fixed masked decoder context.

    ``decode`` must close over a frozen decoder and the masked encoder context.
    The function only requests gradients for the candidate latent, preventing
    baseline parameter gradients from accumulating even when the closure refers
    to modules whose parameters still have ``requires_grad=True``.
    """

    if base_latent.ndim != 2:
        raise ValueError("base_latent must have shape [batch, latent_dim]")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")

    base = base_latent.detach()
    candidate = base.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)

    with torch.no_grad():
        initial_prediction = decode(base)
        initial_data_loss = _masked_mse(
            initial_prediction,
            target,
            target_mask,
        )

    final_data_loss = initial_data_loss
    final_total_loss = initial_data_loss
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = decode(candidate)
        data_loss = _masked_mse(prediction, target, target_mask)
        displacement_loss = (candidate - base).square().mean()
        total_loss = data_loss + regularization * displacement_loss
        gradient = torch.autograd.grad(total_loss, candidate, only_inputs=True)[0]
        candidate.grad = gradient
        optimizer.step()
        final_data_loss = data_loss.detach()
        final_total_loss = total_loss.detach()

    with torch.no_grad():
        optimized = candidate.detach()
        final_prediction = decode(optimized)
        final_data_loss = _masked_mse(final_prediction, target, target_mask)
        delta = optimized - base
        final_total_loss = final_data_loss + regularization * delta.square().mean()
        rms_displacement = torch.sqrt(delta.square().mean())

    return LatentTeacherResult(
        optimized_latent=optimized,
        delta_z=delta,
        initial_data_loss=float(initial_data_loss.item()),
        final_data_loss=float(final_data_loss.item()),
        final_total_loss=float(final_total_loss.item()),
        rms_displacement=float(rms_displacement.item()),
        steps=int(steps),
    )
