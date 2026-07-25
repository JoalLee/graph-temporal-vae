#!/usr/bin/env python3
"""Prepare and train conditional latent residual diffusion for a legacy 26e model.

The baseline Graph-TCN-VAE remains frozen.  Artificial held-out-like masks are
sampled only from naturally observed training values.  For each masked window,
a context-compatible teacher latent is optimized against the artificial PSD
held-out truth.  The diffusion model then learns the resulting correction
``delta_z = z_teacher - mu_masked`` from masked encoder context alone.

This script is an experiment adapter.  The reusable diffusion implementation is
under ``graph_tcn_vae.latent_diffusion``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_tcn_vae.latent_diffusion import (  # noqa: E402
    ConditionalLatentResidualDiffusion,
    HeldoutLikeMaskConfig,
    LatentResidualDiffusionConfig,
    build_latent_condition,
    optimize_latent_target,
    sample_heldout_like_mask,
)
from experiments.diagnose_latent_bottleneck import (  # noqa: E402
    _build_model,
    _decode_state,
    _encode_state,
    _import_legacy,
    _resolve_device,
)


@dataclass(frozen=True)
class TeacherDatasetSummary:
    split: str
    requested_windows: int
    accepted_windows: int
    skipped_without_psd_mask: int
    condition_dim: int
    latent_dim: int
    mean_initial_psd_mse: float
    mean_final_psd_mse: float
    mean_teacher_gain: float
    mean_rms_displacement: float


def _load_frozen_legacy_model(
    legacy_repo: Path,
    experiment_dir: Path,
    checkpoint_name: str,
    config_name: str,
    device: torch.device,
) -> dict[str, Any]:
    legacy = _import_legacy(legacy_repo)
    legacy_inference = legacy["legacy_inference"]
    config_path = experiment_dir / config_name
    checkpoint_path = experiment_dir / checkpoint_name
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    config = json.loads(config_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = legacy_inference._extract_state_dict(checkpoint)
    expected_cond_dim = legacy_inference._infer_checkpoint_cond_dim(state_dict, config)
    use_traffic = expected_cond_dim == 12
    train_df, source_col_info, _, _ = legacy_inference.load_train_data(
        use_traffic=use_traffic
    )
    train_df, col_info = legacy["select_cross_modal_dataframe"](
        train_df,
        source_col_info,
        config,
    )
    window_size = int(config.get("window_size", 168))
    processed = legacy["data_preprocessing"](
        train_df,
        scaler_type="standard",
        window_size=window_size,
        auxiliary_columns=col_info["met_cols"],
    )
    full = processed["full"]
    target = full["target_features"].detach().cpu()
    natural_missing = full["target_mask"].detach().cpu().bool()
    aux = full["auxiliary_features"].detach().cpu()
    hour = full["hour_features"].detach().cpu()
    cond_dim = int(aux.shape[-1] + hour.shape[-1])
    if cond_dim != expected_cond_dim:
        raise RuntimeError(
            f"Condition dimension mismatch: data={cond_dim}, checkpoint={expected_cond_dim}"
        )

    target_dim = int(target.shape[-1])
    n_chem = len(col_info["chem_cols"])
    state_dict = legacy_inference._upgrade_legacy_mask_embed(state_dict, target_dim)
    model = _build_model(
        legacy["ImputationVAE_Graph"],
        legacy["ModelConfig"],
        config,
        state_dict,
        target_dim=target_dim,
        cond_dim=cond_dim,
        window_size=window_size,
        n_chem=n_chem,
    )
    model.current_epoch = int(config.get("epochs", 0))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    stride = int(config.get("stride", 24))
    starts = np.asarray(
        legacy["compute_sliding_window_starts"](
            len(target),
            window_size,
            stride,
            include_tail=True,
        ),
        dtype=np.int64,
    )
    return {
        "legacy": legacy,
        "config": config,
        "model": model,
        "target": target,
        "natural_observed": ~natural_missing,
        "aux": aux,
        "hour": hour,
        "starts": starts,
        "window_size": window_size,
        "stride": stride,
        "n_chem": n_chem,
        "target_dim": target_dim,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
    }


def _split_window_indices(
    starts: np.ndarray,
    *,
    total_length: int,
    window_size: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if validation_fraction <= 0.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    train_end = int(total_length * train_fraction)
    validation_end = int(total_length * (train_fraction + validation_fraction))
    train = np.asarray(
        [i for i, start in enumerate(starts) if int(start) + window_size <= train_end],
        dtype=np.int64,
    )
    validation = np.asarray(
        [
            i
            for i, start in enumerate(starts)
            if int(start) >= train_end and int(start) + window_size <= validation_end
        ],
        dtype=np.int64,
    )
    if train.size == 0 or validation.size == 0:
        raise RuntimeError(
            f"Empty chronological split: train={train.size}, validation={validation.size}"
        )
    return train, validation


def _subsample_indices(
    indices: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    if maximum is None or maximum <= 0 or indices.size <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=maximum, replace=False)
    return np.asarray(sorted(selected.tolist()), dtype=np.int64)


def _longest_true_run(mask: torch.Tensor) -> torch.Tensor:
    """Return longest contiguous true run for each row of ``[batch, time]``."""

    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, time]")
    current = torch.zeros(mask.shape[0], device=mask.device, dtype=torch.float32)
    longest = torch.zeros_like(current)
    for step in range(mask.shape[1]):
        current = torch.where(mask[:, step], current + 1.0, torch.zeros_like(current))
        longest = torch.maximum(longest, current)
    return longest


def _mask_extra_features(
    artificial_mask: torch.Tensor,
    observation_mask: torch.Tensor,
    n_chem: int,
) -> torch.Tensor:
    batch, window, target_dim = artificial_mask.shape
    psd_mask = artificial_mask[:, :, n_chem:]
    psd_time = psd_mask.any(dim=2) if psd_mask.shape[-1] else torch.zeros(
        batch, window, dtype=torch.bool, device=artificial_mask.device
    )
    chem_mask = artificial_mask[:, :, :n_chem]
    return torch.stack(
        [
            artificial_mask.float().mean(dim=(1, 2)),
            chem_mask.float().mean(dim=(1, 2)) if n_chem else torch.zeros(batch, device=artificial_mask.device),
            psd_mask.float().mean(dim=(1, 2)) if target_dim > n_chem else torch.zeros(batch, device=artificial_mask.device),
            _longest_true_run(psd_time) / float(window),
            observation_mask.float().mean(dim=(1, 2)),
        ],
        dim=1,
    )


def _sample_mask_with_psd_target(
    natural_observed: torch.Tensor,
    *,
    n_chem: int,
    config: HeldoutLikeMaskConfig,
    generator: torch.Generator,
    attempts: int,
) -> torch.Tensor | None:
    for _ in range(attempts):
        artificial = sample_heldout_like_mask(
            natural_observed,
            n_chem=n_chem,
            config=config,
            generator=generator,
        )
        if artificial[:, :, n_chem:].any():
            return artificial
    return None


def _prepare_teacher_split(
    *,
    split: str,
    window_indices: np.ndarray,
    source: dict[str, Any],
    device: torch.device,
    batch_size: int,
    mask_config: HeldoutLikeMaskConfig,
    mask_attempts: int,
    teacher_steps: int,
    teacher_lr: float,
    teacher_regularization: float,
    seed: int,
) -> tuple[dict[str, Any], TeacherDatasetSummary]:
    model = source["model"]
    target = source["target"]
    natural_observed = source["natural_observed"]
    aux = source["aux"]
    hour = source["hour"]
    starts = source["starts"]
    window_size = source["window_size"]
    n_chem = source["n_chem"]

    condition_rows: list[torch.Tensor] = []
    delta_rows: list[torch.Tensor] = []
    window_rows: list[torch.Tensor] = []
    initial_losses: list[float] = []
    final_losses: list[float] = []
    displacements: list[float] = []
    skipped = 0
    generator = torch.Generator(device=device).manual_seed(seed)

    for left in range(0, len(window_indices), batch_size):
        ids = window_indices[left : left + batch_size]
        batch_starts = starts[ids]
        truth = torch.stack(
            [target[int(start) : int(start) + window_size] for start in batch_starts]
        ).to(device)
        natural_obs = torch.stack(
            [
                natural_observed[int(start) : int(start) + window_size]
                for start in batch_starts
            ]
        ).to(device)
        cond = torch.stack(
            [
                torch.cat(
                    [
                        aux[int(start) : int(start) + window_size],
                        hour[int(start) : int(start) + window_size],
                    ],
                    dim=-1,
                )
                for start in batch_starts
            ]
        ).to(device)
        artificial = _sample_mask_with_psd_target(
            natural_obs,
            n_chem=n_chem,
            config=mask_config,
            generator=generator,
            attempts=mask_attempts,
        )
        if artificial is None:
            skipped += len(ids)
            continue
        has_psd_target = artificial[:, :, n_chem:].any(dim=(1, 2))
        skipped += int((~has_psd_target).sum().item())
        if not bool(has_psd_target.any()):
            continue
        truth = truth[has_psd_target]
        natural_obs = natural_obs[has_psd_target]
        cond = cond[has_psd_target]
        artificial = artificial[has_psd_target]
        ids = np.asarray(ids)[has_psd_target.detach().cpu().numpy()]
        masked_obs = natural_obs & ~artificial
        masked_x = truth * masked_obs.float()

        with torch.no_grad():
            masked_state = _encode_state(model, masked_x, cond, masked_obs.float())
        psd_teacher_mask = torch.zeros_like(artificial)
        psd_teacher_mask[:, :, n_chem:] = artificial[:, :, n_chem:]

        def decode(candidate_z0: torch.Tensor) -> torch.Tensor:
            z_k = (
                model.flow(candidate_z0)[0]
                if bool(getattr(model, "use_realnvp", False))
                else candidate_z0
            )
            return _decode_state(
                model,
                z_k,
                masked_state,
                variance_latent=candidate_z0,
            )

        teacher = optimize_latent_target(
            base_latent=masked_state.mu,
            decode=decode,
            target=truth,
            target_mask=psd_teacher_mask,
            steps=teacher_steps,
            learning_rate=teacher_lr,
            regularization=teacher_regularization,
        )
        extra = _mask_extra_features(artificial, masked_obs, n_chem)
        condition = build_latent_condition(
            mu=masked_state.mu,
            logvar=masked_state.logvar,
            encoder_sequence=masked_state.h_seq,
            local_context=masked_state.local_context,
            observation_mask=masked_obs.float(),
            n_chem=n_chem,
            extra_features=extra,
        )
        condition_rows.append(condition.detach().cpu())
        delta_rows.append(teacher.delta_z.detach().cpu())
        window_rows.append(torch.as_tensor(ids, dtype=torch.long))
        initial_losses.append(teacher.initial_data_loss)
        final_losses.append(teacher.final_data_loss)
        displacements.append(teacher.rms_displacement)

    if not condition_rows:
        raise RuntimeError(f"No teacher examples were produced for split={split}")
    conditions = torch.cat(condition_rows, dim=0)
    delta_z = torch.cat(delta_rows, dim=0)
    window_ids = torch.cat(window_rows, dim=0)
    payload = {
        "schema_version": 1,
        "artifact_type": "latent_residual_teacher_dataset",
        "split": split,
        "conditions": conditions,
        "delta_z": delta_z,
        "window_ids": window_ids,
        "teacher_initial_psd_mse": torch.tensor(initial_losses),
        "teacher_final_psd_mse": torch.tensor(final_losses),
        "teacher_rms_displacement": torch.tensor(displacements),
        "mask_config": asdict(mask_config),
        "teacher_config": {
            "steps": teacher_steps,
            "learning_rate": teacher_lr,
            "regularization": teacher_regularization,
            "seed": seed,
        },
    }
    summary = TeacherDatasetSummary(
        split=split,
        requested_windows=int(len(window_indices)),
        accepted_windows=int(len(conditions)),
        skipped_without_psd_mask=int(skipped),
        condition_dim=int(conditions.shape[1]),
        latent_dim=int(delta_z.shape[1]),
        mean_initial_psd_mse=float(np.mean(initial_losses)),
        mean_final_psd_mse=float(np.mean(final_losses)),
        mean_teacher_gain=float(np.mean(np.asarray(initial_losses) - np.asarray(final_losses))),
        mean_rms_displacement=float(np.mean(displacements)),
    )
    return payload, summary


def _normalization(train: dict[str, Any]) -> dict[str, torch.Tensor]:
    condition = train["conditions"].float()
    delta = train["delta_z"].float()
    return {
        "condition_mean": condition.mean(dim=0),
        "condition_std": condition.std(dim=0, unbiased=False).clamp_min(1e-5),
        "delta_mean": delta.mean(dim=0),
        "delta_std": delta.std(dim=0, unbiased=False).clamp_min(1e-5),
    }


def _normalize_dataset(
    payload: dict[str, Any],
    stats: dict[str, torch.Tensor],
) -> TensorDataset:
    condition = (
        payload["conditions"].float() - stats["condition_mean"]
    ) / stats["condition_std"]
    delta = (payload["delta_z"].float() - stats["delta_mean"]) / stats["delta_std"]
    return TensorDataset(condition, delta)


def _evaluate_diffusion_loss(
    model: ConditionalLatentResidualDiffusion,
    loader: DataLoader,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for condition, delta in loader:
            condition = condition.to(device)
            delta = delta.to(device)
            timesteps = torch.randint(
                0,
                model.config.timesteps,
                (len(condition),),
                device=device,
                generator=generator,
            )
            noise = torch.randn(
                delta.shape,
                device=device,
                dtype=delta.dtype,
                generator=generator,
            )
            loss = model.loss(
                delta,
                condition,
                timesteps=timesteps,
                noise=noise,
            )
            total += float(loss) * len(condition)
            count += len(condition)
    return total / max(count, 1)


def _train_diffusion(
    *,
    train_payload: dict[str, Any],
    validation_payload: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    config: LatentResidualDiffusionConfig,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    stats = _normalization(train_payload)
    train_dataset = _normalize_dataset(train_payload, stats)
    validation_dataset = _normalize_dataset(validation_payload, stats)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    model = ConditionalLatentResidualDiffusion(
        latent_dim=int(train_payload["delta_z"].shape[1]),
        condition_dim=int(train_payload["conditions"].shape[1]),
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float]] = []
    best_validation = math.inf
    best_payload: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for condition, delta in train_loader:
            condition = condition.to(device)
            delta = delta.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(delta, condition)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(condition)
            train_count += len(condition)
        train_loss = train_total / max(train_count, 1)
        validation_loss = _evaluate_diffusion_loss(
            model,
            validation_loader,
            device,
            seed=seed + 10_000,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_payload = model.checkpoint_payload()

    if best_payload is None:
        raise RuntimeError("Diffusion training produced no checkpoint")
    best_payload["normalization"] = stats
    best_payload["training"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "best_validation_loss": best_validation,
        "history": history,
    }
    checkpoint_path = output_dir / "best_latent_residual_diffusion.pt"
    torch.save(best_payload, checkpoint_path)

    reloaded = ConditionalLatentResidualDiffusion.from_checkpoint_payload(
        best_payload,
        map_location=device,
    ).to(device).eval()
    val_conditions = (
        validation_payload["conditions"].float() - stats["condition_mean"]
    ) / stats["condition_std"]
    with torch.no_grad():
        sampled = reloaded.sample(
            val_conditions.to(device),
            num_samples=8,
            generator=torch.Generator(device=device).manual_seed(seed + 20_000),
        ).mean(dim=0).cpu()
    sampled_delta = sampled * stats["delta_std"] + stats["delta_mean"]
    teacher_delta = validation_payload["delta_z"].float()
    zero_mse = float(teacher_delta.square().mean())
    sample_mean_mse = float((sampled_delta - teacher_delta).square().mean())
    return {
        "checkpoint": str(checkpoint_path),
        "best_validation_loss": best_validation,
        "zero_correction_mse_to_teacher": zero_mse,
        "sample_mean_mse_to_teacher": sample_mean_mse,
        "history": history,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _load_frozen_legacy_model(
        args.legacy_repo.resolve(),
        args.experiment_dir.resolve(),
        args.checkpoint_name,
        args.config_name,
        device,
    )
    train_indices, validation_indices = _split_window_indices(
        source["starts"],
        total_length=len(source["target"]),
        window_size=source["window_size"],
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    train_indices = _subsample_indices(
        train_indices,
        args.max_train_windows,
        args.seed,
    )
    validation_indices = _subsample_indices(
        validation_indices,
        args.max_validation_windows,
        args.seed + 1,
    )
    legacy_mask_config = source["config"].get("dynamic_masking_config", {}) or {}
    mask_config = HeldoutLikeMaskConfig(
        target_ratio=float(legacy_mask_config.get("target_ratio", 0.10)),
        mean_duration=float(legacy_mask_config.get("mean_duration", 48.0)),
        std_duration=float(legacy_mask_config.get("std_duration", 24.0)),
        min_duration=int(legacy_mask_config.get("min_duration", 3)),
        max_duration=int(legacy_mask_config.get("max_duration", 168)),
        psd_blocks_per_sample=int(legacy_mask_config.get("psd_blocks_per_sample", 1)),
        chem_blocks_per_feature=int(legacy_mask_config.get("chem_blocks_per_feature", 1)),
        point_dropout=float(legacy_mask_config.get("point_dropout", 0.0)),
    )

    train_payload, train_summary = _prepare_teacher_split(
        split="train",
        window_indices=train_indices,
        source=source,
        device=device,
        batch_size=args.teacher_batch_size,
        mask_config=mask_config,
        mask_attempts=args.mask_attempts,
        teacher_steps=args.teacher_steps,
        teacher_lr=args.teacher_lr,
        teacher_regularization=args.teacher_regularization,
        seed=args.seed,
    )
    validation_payload, validation_summary = _prepare_teacher_split(
        split="validation",
        window_indices=validation_indices,
        source=source,
        device=device,
        batch_size=args.teacher_batch_size,
        mask_config=mask_config,
        mask_attempts=args.mask_attempts,
        teacher_steps=args.teacher_steps,
        teacher_lr=args.teacher_lr,
        teacher_regularization=args.teacher_regularization,
        seed=args.seed + 1000,
    )
    torch.save(train_payload, output_dir / "teacher_train.pt")
    torch.save(validation_payload, output_dir / "teacher_validation.pt")

    diffusion_config = LatentResidualDiffusionConfig(
        timesteps=args.diffusion_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        hidden_dim=args.hidden_dim,
        time_embedding_dim=args.time_embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        clip_denoised=args.clip_denoised,
    )
    training = _train_diffusion(
        train_payload=train_payload,
        validation_payload=validation_payload,
        output_dir=output_dir,
        device=device,
        config=diffusion_config,
        batch_size=args.diffusion_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "conditional_latent_residual_diffusion_run",
        "baseline_checkpoint": str(source["checkpoint_path"]),
        "baseline_config": str(source["config_path"]),
        "device": str(device),
        "chronological_split": {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
        },
        "train_teacher": asdict(train_summary),
        "validation_teacher": asdict(validation_summary),
        "mask_config": asdict(mask_config),
        "diffusion_config": diffusion_config.to_dict(),
        "training": training,
        "interpretation_boundary": (
            "Teacher targets use artificial held-out truth and only establish a "
            "trainable latent-correction target. Scientific validity requires a "
            "separate untouched held-out evaluation with empirical CRPS, PICP, "
            "PSD structure, and physical closure."
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-repo", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--config-name", default="config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-validation-windows", type=int, default=None)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--mask-attempts", type=int, default=8)
    parser.add_argument("--teacher-steps", type=int, default=20)
    parser.add_argument("--teacher-lr", type=float, default=0.03)
    parser.add_argument("--teacher-regularization", type=float, default=1.0)
    parser.add_argument("--diffusion-timesteps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--clip-denoised", type=float, default=None)
    parser.add_argument("--diffusion-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
