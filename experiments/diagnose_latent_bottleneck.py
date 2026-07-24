#!/usr/bin/env python3
"""Diagnose whether a legacy 26e checkpoint is bottlenecked by its latent posterior.

This script intentionally runs against the original Imputation_VAE repository so
that checkpoint reconstruction and preprocessing remain identical to the legacy
experiment.  It evaluates a 2 x 2 intervention on each inference window:

    latent source  : full-input posterior vs held-out-input posterior
    context source : full-input encoder context vs held-out-input encoder context

The central comparison is ``latent_oracle``: decode the full-input latent while
keeping every other decoder input from the held-out run.  A material improvement
there is direct evidence that p(z | x_obs) is part of the bottleneck.  Conversely,
``context_oracle`` isolates encoder skip/local-context information.

No training is performed and no checkpoint is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


VARIANT_LABELS = {
    "masked_baseline": "masked latent + masked context",
    "latent_oracle": "full latent + masked context",
    "context_oracle": "masked latent + full context",
    "full_oracle": "full latent + full context",
    "latent_zero": "zero post-flow latent + masked context",
    "latent_shuffled": "batch-shuffled masked latent + masked context",
    "latent_midpoint": "midpoint(full, masked) latent + masked context",
}

OPTIMIZED_VARIANT = "latent_optimized"

GAP_BINS = (
    (0.0, 12.0, "0-12h"),
    (12.0, 24.0, "12-24h"),
    (24.0, 48.0, "24-48h"),
    (48.0, 96.0, "48-96h"),
    (96.0, 168.000001, "96-168h"),
)


@dataclass
class EncodedState:
    """All decoder-relevant values produced by one encoder pass."""

    mu: torch.Tensor
    logvar: torch.Tensor
    z0: torch.Tensor
    z_k: torch.Tensor
    cond: torch.Tensor
    obs_mask: torch.Tensor
    h_seq: torch.Tensor | None
    local_context: torch.Tensor | None
    support: torch.Tensor | None
    graph_attention: torch.Tensor | None
    graph_source_memory: torch.Tensor | None
    obs_value_context: torch.Tensor


class OverlapAccumulator:
    """Position-weighted overlap accumulator for deterministic window outputs."""

    def __init__(self, length: int, features: int, weights: np.ndarray) -> None:
        self.sum = np.zeros((length, features), dtype=np.float64)
        self.weight = np.zeros((length, 1), dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)

    def add(self, start: int, values: np.ndarray) -> None:
        length = min(values.shape[0], self.sum.shape[0] - start)
        if length <= 0:
            return
        w = self.weights[:length, None]
        self.sum[start : start + length] += values[:length] * w
        self.weight[start : start + length] += w

    def finalize(self) -> np.ndarray:
        out = np.full_like(self.sum, np.nan, dtype=np.float64)
        valid = self.weight[:, 0] > 0
        out[valid] = self.sum[valid] / self.weight[valid]
        return out


def _position_weights(window_size: int, edge_fraction: float = 0.2) -> np.ndarray:
    weights = np.ones(window_size, dtype=np.float64)
    edge = int(round(window_size * edge_fraction))
    if edge <= 0:
        return weights
    ramp = np.linspace(1.0 / (edge + 1), 1.0, edge, dtype=np.float64)
    weights[:edge] = ramp
    weights[-edge:] = ramp[::-1]
    return weights


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true[valid], dtype=np.float64)
    y_pred = np.asarray(y_pred[valid], dtype=np.float64)
    if y_true.size == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "r2": float("nan"),
            "slope": float("nan"),
            "top10_bias": float("nan"),
            "top10_underprediction_rate": float("nan"),
        }
    residual = y_pred - y_true
    centered = y_true - np.mean(y_true)
    variance = float(np.sum(centered**2))
    slope = (
        float(np.sum(centered * (y_pred - np.mean(y_pred))) / variance)
        if variance > 0
        else float("nan")
    )
    threshold = float(np.quantile(y_true, 0.9))
    top = y_true >= threshold
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "r2": _safe_r2(y_true, y_pred),
        "slope": slope,
        "top10_bias": float(np.mean(residual[top])) if np.any(top) else float("nan"),
        "top10_underprediction_rate": (
            float(np.mean(y_pred[top] < y_true[top])) if np.any(top) else float("nan")
        ),
    }


def _symmetric_kl_diag_gaussian(
    mu_a: torch.Tensor,
    logvar_a: torch.Tensor,
    mu_b: torch.Tensor,
    logvar_b: torch.Tensor,
) -> torch.Tensor:
    """Per-example symmetric KL for two diagonal Gaussian posteriors."""

    logvar_a = logvar_a.clamp(-20.0, 20.0)
    logvar_b = logvar_b.clamp(-20.0, 20.0)
    var_a = torch.exp(logvar_a)
    var_b = torch.exp(logvar_b)
    delta2 = (mu_a - mu_b).pow(2)
    kl_ab = 0.5 * torch.sum(logvar_b - logvar_a + (var_a + delta2) / var_b - 1.0, dim=-1)
    kl_ba = 0.5 * torch.sum(logvar_a - logvar_b + (var_b + delta2) / var_a - 1.0, dim=-1)
    return 0.5 * (kl_ab + kl_ba)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    return torch.sum(a * b, dim=-1) / denom.clamp_min(1e-12)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _import_legacy(legacy_repo: Path) -> dict[str, Any]:
    legacy_repo = legacy_repo.resolve()
    if not legacy_repo.exists():
        raise FileNotFoundError(f"Legacy repository not found: {legacy_repo}")
    sys.path.insert(0, str(legacy_repo))
    sys.path.insert(0, str(legacy_repo.parent))

    from experiments.ablation import ablation_inference as legacy_inference
    from experiments.ablation.cross_modal_utils import select_cross_modal_dataframe
    from imputation_vae.model_graph_uq import ImputationVAE_Graph, ModelConfig
    from Shared.data_processing import compute_sliding_window_starts, data_preprocessing

    return {
        "legacy_inference": legacy_inference,
        "select_cross_modal_dataframe": select_cross_modal_dataframe,
        "ImputationVAE_Graph": ImputationVAE_Graph,
        "ModelConfig": ModelConfig,
        "compute_sliding_window_starts": compute_sliding_window_starts,
        "data_preprocessing": data_preprocessing,
    }


def _build_model(
    model_class: type[torch.nn.Module],
    model_config_class: type,
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    *,
    target_dim: int,
    cond_dim: int,
    window_size: int,
    n_chem: int,
) -> torch.nn.Module:
    """Reconstruct a legacy model from its saved typed architecture config."""

    if hasattr(model_config_class, "field_names"):
        field_names = model_config_class.field_names()
    else:
        field_names = set(getattr(model_config_class, "__dataclass_fields__", {}))
    config_values = {key: value for key, value in config.items() if key in field_names}
    config_values["n_chem"] = n_chem
    if hasattr(model_config_class, "from_dict"):
        typed_config = model_config_class.from_dict(config_values)
    else:
        typed_config = model_config_class(**config_values)

    model = model_class(
        target_dim=target_dim,
        aux_dim=cond_dim,
        window_size=window_size,
        config=typed_config,
    )
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    return model


def _encode_state(
    model: torch.nn.Module,
    x: torch.Tensor,
    cond_raw: torch.Tensor,
    obs_mask: torch.Tensor,
) -> EncodedState:
    """Run exactly the encoder-side portion of legacy ``forward``."""

    cond = model._encode_cond_features(cond_raw)
    embed_0 = model.mask_embed(torch.zeros(1, dtype=torch.long, device=x.device))
    embed_1 = model.mask_embed(torch.ones(1, dtype=torch.long, device=x.device))
    embed_offset = embed_0 + obs_mask.float() * (embed_1 - embed_0)
    inputs = torch.cat([x, cond], dim=-1).permute(0, 2, 1)
    batch, window, _ = x.shape
    full_obs_mask = torch.cat(
        [
            obs_mask,
            torch.ones(batch, window, cond.shape[-1], device=x.device, dtype=obs_mask.dtype),
        ],
        dim=-1,
    )
    (
        mu,
        logvar,
        graph_attention,
        h_seq,
        local_context,
        support,
    ) = model.encoder(inputs, full_obs_mask, embed_offset=embed_offset.permute(0, 2, 1))
    z0 = mu
    z_k = z0
    if bool(getattr(model, "use_realnvp", False)):
        z_k, _ = model.flow(z0)
    return EncodedState(
        mu=mu,
        logvar=logvar,
        z0=z0,
        z_k=z_k,
        cond=cond,
        obs_mask=obs_mask,
        h_seq=h_seq,
        local_context=local_context,
        support=support,
        graph_attention=graph_attention,
        graph_source_memory=getattr(model.encoder, "graph_source_memory", None),
        obs_value_context=torch.cat([x * obs_mask.float(), obs_mask.float()], dim=-1),
    )


def _decode_state(
    model: torch.nn.Module,
    z: torch.Tensor,
    context: EncodedState,
    *,
    variance_latent: torch.Tensor,
) -> torch.Tensor:
    """Decode one latent/context combination and return [B, W, D] mean."""

    model.decoder.current_epoch = int(getattr(model, "current_epoch", 0))
    mean, _ = model.decoder(
        z,
        context.cond,
        enc_h_seq=context.h_seq,
        obs_mask=context.obs_mask,
        obs_value_context=context.obs_value_context,
        local_context=context.local_context,
        attn_weighted_support_t=context.support,
        variance_latent=variance_latent,
        graph_source_memory=context.graph_source_memory,
        encoder_graph_relation=context.graph_attention,
    )
    return mean.permute(0, 2, 1)


def _optimize_latent_for_psd(
    model: torch.nn.Module,
    masked_state: EncodedState,
    truth: torch.Tensor,
    heldout_mask: torch.Tensor,
    *,
    n_chem: int,
    steps: int,
    learning_rate: float,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Optimize pre-flow latent while keeping masked encoder context fixed.

    This is a diagnostic upper bound, not an inference algorithm: held-out truth
    is used only to ask whether the existing decoder contains a substantially
    better latent solution than q(z | x_obs) supplies.
    """

    initial_z0 = masked_state.mu.detach()
    psd_mask = heldout_mask[:, :, n_chem:].bool()
    if steps <= 0 or not torch.any(psd_mask):
        prediction = _decode_state(
            model,
            masked_state.z_k,
            masked_state,
            variance_latent=masked_state.mu,
        )
        return prediction.detach(), initial_z0, {
            "initial_psd_mse": float("nan"),
            "final_psd_mse": float("nan"),
            "z0_rms_displacement": 0.0,
        }

    z0 = initial_z0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z0], lr=learning_rate)
    initial_loss = None
    final_loss = None
    truth_psd = truth[:, :, n_chem:]

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        z_k = model.flow(z0)[0] if bool(getattr(model, "use_realnvp", False)) else z0
        prediction = _decode_state(
            model,
            z_k,
            masked_state,
            variance_latent=z0,
        )
        squared_error = (prediction[:, :, n_chem:] - truth_psd).pow(2)
        data_loss = squared_error[psd_mask].mean()
        if initial_loss is None:
            initial_loss = float(data_loss.detach())
        penalty = (z0 - initial_z0).pow(2).mean()
        loss = data_loss + regularization * penalty
        loss.backward()
        optimizer.step()
        final_loss = float(data_loss.detach())

    with torch.no_grad():
        z_k = model.flow(z0)[0] if bool(getattr(model, "use_realnvp", False)) else z0
        prediction = _decode_state(
            model,
            z_k,
            masked_state,
            variance_latent=z0,
        )
        final_loss = float(
            (prediction[:, :, n_chem:] - truth_psd).pow(2)[psd_mask].mean()
        )
        displacement = float(torch.sqrt((z0 - initial_z0).pow(2).mean()))
    return prediction.detach(), z0.detach(), {
        "initial_psd_mse": float(initial_loss),
        "final_psd_mse": float(final_loss),
        "z0_rms_displacement": displacement,
    }


def _window_indices(
    starts: np.ndarray,
    heldout_mask: np.ndarray,
    window_size: int,
    max_windows: int | None,
    seed: int,
) -> np.ndarray:
    candidates = np.asarray(
        [i for i, start in enumerate(starts) if heldout_mask[int(start) : int(start) + window_size].any()],
        dtype=np.int64,
    )
    if max_windows is None or max_windows <= 0 or candidates.size <= max_windows:
        return candidates
    rng = np.random.default_rng(seed)
    # Preserve coverage across the time axis rather than taking only early windows.
    edges = np.linspace(0, candidates.size, max_windows + 1, dtype=int)
    selected = []
    for left, right in zip(edges[:-1], edges[1:]):
        block = candidates[left:right]
        if block.size:
            selected.append(int(rng.choice(block)))
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _family_window_mae(
    truth: np.ndarray,
    pred: np.ndarray,
    heldout: np.ndarray,
    columns: slice,
) -> float:
    mask = heldout[:, columns]
    if not np.any(mask):
        return float("nan")
    yt = truth[:, columns][mask]
    yp = pred[:, columns][mask]
    valid = np.isfinite(yt) & np.isfinite(yp)
    return float(np.mean(np.abs(yp[valid] - yt[valid]))) if np.any(valid) else float("nan")


def _inverse_target_scaling(values: np.ndarray, scaler: Any) -> np.ndarray:
    log_values = values * np.asarray(scaler.scale_)[None, :] + np.asarray(scaler.mean_)[None, :]
    return np.clip(np.expm1(log_values), 0.0, None)


def _psd_spectrum_metrics(
    truth_scaled: np.ndarray,
    prediction_scaled: np.ndarray,
    evaluation_mask: np.ndarray,
    scaler: Any,
    n_chem: int,
    diameters: np.ndarray,
) -> dict[str, float]:
    psd_mask = evaluation_mask[:, n_chem:]
    usable_rows = np.mean(psd_mask, axis=1) >= 0.9
    if not np.any(usable_rows):
        return {"n_spectra": 0}

    truth_raw = _inverse_target_scaling(truth_scaled, scaler)[:, n_chem:]
    pred_raw = _inverse_target_scaling(prediction_scaled, scaler)[:, n_chem:]
    correlations: list[float] = []
    peak_log_errors: list[float] = []
    peak_amplitude_ratios: list[float] = []
    for row in np.flatnonzero(usable_rows):
        yt = truth_raw[row]
        yp = pred_raw[row]
        valid = np.isfinite(yt) & np.isfinite(yp)
        if np.sum(valid) < 3:
            continue
        yt = yt[valid]
        yp = yp[valid]
        ds = diameters[valid]
        if np.std(yt) > 0 and np.std(yp) > 0:
            correlations.append(float(np.corrcoef(yt, yp)[0, 1]))
        true_peak = int(np.argmax(yt))
        pred_peak = int(np.argmax(yp))
        peak_log_errors.append(float(abs(np.log10(ds[pred_peak]) - np.log10(ds[true_peak]))))
        peak_amplitude_ratios.append(float(yp[pred_peak] / max(yt[true_peak], 1e-12)))

    if not peak_log_errors:
        return {"n_spectra": 0}
    median_log_error = float(np.median(peak_log_errors))
    return {
        "n_spectra": int(len(peak_log_errors)),
        "median_spectrum_correlation": (
            float(np.nanmedian(correlations)) if correlations else float("nan")
        ),
        "median_peak_log10_diameter_error": median_log_error,
        "median_peak_diameter_factor_error": float(10.0**median_log_error),
        "median_peak_amplitude_ratio": float(np.median(peak_amplitude_ratios)),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    legacy = _import_legacy(args.legacy_repo)
    legacy_inference = legacy["legacy_inference"]
    experiment_dir = args.experiment_dir.resolve()
    config_path = experiment_dir / args.config_name
    checkpoint_path = experiment_dir / args.checkpoint_name
    heldout_path = experiment_dir / args.heldout_name
    gap_path = experiment_dir / args.gap_name
    for path in (config_path, checkpoint_path, heldout_path):
        if not path.exists():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text())
    device = _resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = legacy_inference._extract_state_dict(checkpoint)

    expected_cond_dim = legacy_inference._infer_checkpoint_cond_dim(state_dict, config)
    use_traffic = expected_cond_dim == 12
    train_df, source_col_info, _, _ = legacy_inference.load_train_data(use_traffic=use_traffic)
    train_df, col_info = legacy["select_cross_modal_dataframe"](
        train_df, source_col_info, config
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
    natural_missing = full["target_mask"].detach().cpu().numpy().astype(bool)
    aux = full["auxiliary_features"].detach().cpu()
    hour = full["hour_features"].detach().cpu()
    cond_dim = int(aux.shape[-1] + hour.shape[-1])
    if cond_dim != expected_cond_dim:
        raise RuntimeError(
            f"Condition dimension mismatch: data={cond_dim}, checkpoint={expected_cond_dim}"
        )

    heldout = np.load(heldout_path).astype(bool)
    if heldout.shape != tuple(target.shape):
        raise RuntimeError(
            f"Held-out mask shape {heldout.shape} != target shape {tuple(target.shape)}"
        )
    gap_map = np.load(gap_path) if gap_path.exists() else np.zeros_like(heldout, dtype=float)
    naturally_observed = ~natural_missing
    invalid_heldout = heldout & ~naturally_observed
    if invalid_heldout.any():
        # These points cannot be evaluated and should not be treated as oracle data.
        heldout = heldout & naturally_observed

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
            len(target), window_size, stride, include_tail=True
        ),
        dtype=np.int64,
    )
    if args.selection_family == "psd":
        selection_mask = heldout[:, n_chem:]
    elif args.selection_family == "chem":
        selection_mask = heldout[:, :n_chem]
    else:
        selection_mask = heldout
    selected_indices = _window_indices(
        starts, selection_mask, window_size, args.max_windows, args.seed
    )
    if selected_indices.size == 0:
        raise RuntimeError("No selected window contains evaluable held-out observations")

    position_weights = _position_weights(window_size, args.edge_fraction)
    variant_labels = dict(VARIANT_LABELS)
    if args.optimize_latent_steps > 0:
        variant_labels[OPTIMIZED_VARIANT] = (
            "truth-optimized pre-flow latent + masked context (diagnostic upper bound)"
        )
    accumulators = {
        name: OverlapAccumulator(len(target), target_dim, position_weights)
        for name in variant_labels
    }
    window_records: list[dict[str, Any]] = []
    mu_full_all: list[np.ndarray] = []
    mu_masked_all: list[np.ndarray] = []
    logvar_full_all: list[np.ndarray] = []
    logvar_masked_all: list[np.ndarray] = []
    forward_parity_max_abs: float | None = None
    optimized_batch_records: list[dict[str, float]] = []

    target_np = target.numpy().astype(np.float64)
    full_obs_np = naturally_observed.astype(np.float32)
    masked_obs_np = (naturally_observed & ~heldout).astype(np.float32)

    for batch_left in range(0, selected_indices.size, args.batch_size):
            batch_ids = selected_indices[batch_left : batch_left + args.batch_size]
            batch_starts = starts[batch_ids]
            x_true = torch.stack(
                [target[int(start) : int(start) + window_size] for start in batch_starts]
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
            full_obs = torch.from_numpy(
                np.stack(
                    [full_obs_np[int(start) : int(start) + window_size] for start in batch_starts]
                )
            ).to(device)
            masked_obs = torch.from_numpy(
                np.stack(
                    [masked_obs_np[int(start) : int(start) + window_size] for start in batch_starts]
                )
            ).to(device)
            x_full = x_true * full_obs
            x_masked = x_true * masked_obs

            with torch.no_grad():
                full_state = _encode_state(model, x_full, cond, full_obs)
                masked_state = _encode_state(model, x_masked, cond, masked_obs)
            if masked_state.z_k.shape[0] > 1:
                permutation = torch.roll(
                    torch.arange(masked_state.z_k.shape[0], device=device), shifts=1
                )
                shuffled_z = masked_state.z_k[permutation]
            else:
                shuffled_z = masked_state.z_k

            with torch.no_grad():
                predictions = {
                    "masked_baseline": _decode_state(
                        model,
                        masked_state.z_k,
                        masked_state,
                        variance_latent=masked_state.mu,
                    ),
                "latent_oracle": _decode_state(
                    model,
                    full_state.z_k,
                    masked_state,
                    variance_latent=full_state.mu,
                ),
                "context_oracle": _decode_state(
                    model,
                    masked_state.z_k,
                    full_state,
                    variance_latent=masked_state.mu,
                ),
                "full_oracle": _decode_state(
                    model,
                    full_state.z_k,
                    full_state,
                    variance_latent=full_state.mu,
                ),
                "latent_zero": _decode_state(
                    model,
                    torch.zeros_like(masked_state.z_k),
                    masked_state,
                    variance_latent=torch.zeros_like(masked_state.mu),
                ),
                "latent_shuffled": _decode_state(
                    model,
                    shuffled_z,
                    masked_state,
                    variance_latent=masked_state.mu,
                ),
                    "latent_midpoint": _decode_state(
                        model,
                        0.5 * (full_state.z_k + masked_state.z_k),
                        masked_state,
                        variance_latent=0.5 * (full_state.mu + masked_state.mu),
                    ),
                }
            if args.optimize_latent_steps > 0:
                batch_heldout = torch.from_numpy(
                    np.stack(
                        [heldout[int(start) : int(start) + window_size] for start in batch_starts]
                    )
                ).to(device)
                optimized_prediction, _, optimization_stats = _optimize_latent_for_psd(
                    model,
                    masked_state,
                    x_true,
                    batch_heldout,
                    n_chem=n_chem,
                    steps=args.optimize_latent_steps,
                    learning_rate=args.optimize_latent_lr,
                    regularization=args.optimize_latent_regularization,
                )
                predictions[OPTIMIZED_VARIANT] = optimized_prediction
                optimized_batch_records.append(
                    {
                        "batch_left": int(batch_left),
                        "batch_size": int(len(batch_ids)),
                        **optimization_stats,
                    }
                )
            if forward_parity_max_abs is None:
                forward_reference = model(
                    x_masked,
                    cond,
                    masked_obs,
                    sample_latent=False,
                )[0]
                forward_parity_max_abs = float(
                    torch.max(
                        torch.abs(forward_reference - predictions["masked_baseline"])
                    ).item()
                )
                if forward_parity_max_abs > args.parity_tolerance:
                    raise RuntimeError(
                        "Direct encoder/decoder diagnostic path does not match legacy "
                        f"model.forward: max_abs={forward_parity_max_abs:.6g}, "
                        f"tolerance={args.parity_tolerance:.6g}"
                    )
            prediction_np = {
                name: value.float().cpu().numpy().astype(np.float64)
                for name, value in predictions.items()
            }

            mu_full = full_state.mu.float().cpu()
            mu_masked = masked_state.mu.float().cpu()
            logvar_full = full_state.logvar.float().cpu()
            logvar_masked = masked_state.logvar.float().cpu()
            z_full = full_state.z_k.float().cpu()
            z_masked = masked_state.z_k.float().cpu()
            symmetric_kl = _symmetric_kl_diag_gaussian(
                mu_full, logvar_full, mu_masked, logvar_masked
            )
            mu_distance = torch.linalg.vector_norm(mu_full - mu_masked, dim=-1) / math.sqrt(
                mu_full.shape[-1]
            )
            zk_distance = torch.linalg.vector_norm(z_full - z_masked, dim=-1) / math.sqrt(
                z_full.shape[-1]
            )
            flow_full = torch.linalg.vector_norm(z_full - mu_full, dim=-1) / math.sqrt(
                z_full.shape[-1]
            )
            flow_masked = torch.linalg.vector_norm(z_masked - mu_masked, dim=-1) / math.sqrt(
                z_masked.shape[-1]
            )
            cosine = _cosine_similarity(mu_full, mu_masked)

            mu_full_all.append(mu_full.numpy())
            mu_masked_all.append(mu_masked.numpy())
            logvar_full_all.append(logvar_full.numpy())
            logvar_masked_all.append(logvar_masked.numpy())

            for local_i, start in enumerate(batch_starts):
                start = int(start)
                truth_window = target_np[start : start + window_size]
                heldout_window = heldout[start : start + window_size]
                gap_window = gap_map[start : start + window_size]
                for name, values in prediction_np.items():
                    accumulators[name].add(start, values[local_i])

                record: dict[str, Any] = {
                    "window_index": int(batch_ids[local_i]),
                    "start": start,
                    "heldout_fraction": float(np.mean(heldout_window)),
                    "chem_heldout_fraction": float(np.mean(heldout_window[:, :n_chem])),
                    "psd_heldout_fraction": float(np.mean(heldout_window[:, n_chem:])),
                    "chem_max_gap_h": float(np.max(gap_window[:, :n_chem], initial=0.0)),
                    "psd_max_gap_h": float(np.max(gap_window[:, n_chem:], initial=0.0)),
                    "mu_distance_per_dim": float(mu_distance[local_i]),
                    "zk_distance_per_dim": float(zk_distance[local_i]),
                    "mu_cosine": float(cosine[local_i]),
                    "symmetric_kl_per_dim": float(symmetric_kl[local_i] / mu_full.shape[-1]),
                    "posterior_std_full_mean": float(
                        torch.exp(0.5 * logvar_full[local_i]).mean()
                    ),
                    "posterior_std_masked_mean": float(
                        torch.exp(0.5 * logvar_masked[local_i]).mean()
                    ),
                    "flow_displacement_full_per_dim": float(flow_full[local_i]),
                    "flow_displacement_masked_per_dim": float(flow_masked[local_i]),
                }
                for name, values in prediction_np.items():
                    record[f"{name}_chem_mae"] = _family_window_mae(
                        truth_window, values[local_i], heldout_window, slice(0, n_chem)
                    )
                    record[f"{name}_psd_mae"] = _family_window_mae(
                        truth_window, values[local_i], heldout_window, slice(n_chem, target_dim)
                    )
                window_records.append(record)

    aggregated = {name: acc.finalize() for name, acc in accumulators.items()}
    covered = np.isfinite(aggregated["masked_baseline"])
    evaluation_mask = heldout & covered
    target_scaler = processed["scalers"]["target"]

    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for name, prediction in aggregated.items():
        for family, feature_slice in (
            ("chem", slice(0, n_chem)),
            ("psd", slice(n_chem, target_dim)),
            ("all", slice(0, target_dim)),
        ):
            family_mask = evaluation_mask[:, feature_slice]
            metrics = _regression_metrics(
                target_np[:, feature_slice][family_mask],
                prediction[:, feature_slice][family_mask],
            )
            metric_rows.append(
                {
                    "variant": name,
                    "description": variant_labels[name],
                    "family": family,
                    **metrics,
                }
            )
        for low, high, label in GAP_BINS:
            bin_mask = evaluation_mask & (gap_map > low) & (gap_map <= high)
            for family, feature_slice in (
                ("chem", slice(0, n_chem)),
                ("psd", slice(n_chem, target_dim)),
            ):
                family_mask = bin_mask[:, feature_slice]
                metrics = _regression_metrics(
                    target_np[:, feature_slice][family_mask],
                    prediction[:, feature_slice][family_mask],
                )
                gap_rows.append(
                    {
                        "variant": name,
                        "family": family,
                        "gap_bin": label,
                        **metrics,
                    }
                )

    metrics_df = pd.DataFrame(metric_rows)
    gap_df = pd.DataFrame(gap_rows)
    window_df = pd.DataFrame(window_records)

    spectrum = {
        name: _psd_spectrum_metrics(
            target_np,
            prediction,
            evaluation_mask,
            target_scaler,
            n_chem,
            np.asarray([float(value) for value in col_info["psd_cols"]], dtype=np.float64),
        )
        for name, prediction in aggregated.items()
    }

    mu_full_np = np.concatenate(mu_full_all, axis=0)
    mu_masked_np = np.concatenate(mu_masked_all, axis=0)
    logvar_full_np = np.concatenate(logvar_full_all, axis=0)
    logvar_masked_np = np.concatenate(logvar_masked_all, axis=0)
    active_threshold = float(args.active_unit_threshold)
    latent_summary = {
        "n_windows": int(mu_full_np.shape[0]),
        "latent_dim": int(mu_full_np.shape[1]),
        "active_units_full": int(np.sum(np.var(mu_full_np, axis=0) > active_threshold)),
        "active_units_masked": int(np.sum(np.var(mu_masked_np, axis=0) > active_threshold)),
        "median_posterior_std_full": float(np.median(np.exp(0.5 * logvar_full_np))),
        "median_posterior_std_masked": float(np.median(np.exp(0.5 * logvar_masked_np))),
        "mean_mu_distance_per_dim": float(window_df["mu_distance_per_dim"].mean()),
        "mean_zk_distance_per_dim": float(window_df["zk_distance_per_dim"].mean()),
        "mean_symmetric_kl_per_dim": float(window_df["symmetric_kl_per_dim"].mean()),
        "mean_mu_cosine": float(window_df["mu_cosine"].mean()),
        "corr_psd_gap_vs_mu_distance": float(
            window_df[["psd_max_gap_h", "mu_distance_per_dim"]].corr(method="spearman").iloc[0, 1]
        ),
        "corr_psd_gap_vs_symmetric_kl": float(
            window_df[["psd_max_gap_h", "symmetric_kl_per_dim"]].corr(method="spearman").iloc[0, 1]
        ),
    }

    def metric(variant: str, family: str, field: str) -> float:
        row = metrics_df[(metrics_df.variant == variant) & (metrics_df.family == family)]
        return float(row.iloc[0][field])

    baseline_mae = metric("masked_baseline", "psd", "mae")
    latent_mae = metric("latent_oracle", "psd", "mae")
    context_mae = metric("context_oracle", "psd", "mae")
    full_mae = metric("full_oracle", "psd", "mae")
    zero_mae = metric("latent_zero", "psd", "mae")
    shuffled_mae = metric("latent_shuffled", "psd", "mae")
    latent_gain = baseline_mae - latent_mae
    context_gain = baseline_mae - context_mae
    total_gain = baseline_mae - full_mae
    gain_denominator = max(abs(total_gain), 1e-12)
    latent_share = latent_gain / gain_denominator
    context_share = context_gain / gain_denominator

    baseline_pred = aggregated["masked_baseline"]
    zero_pred = aggregated["latent_zero"]
    shuffled_pred = aggregated["latent_shuffled"]
    psd_eval = evaluation_mask[:, n_chem:]
    true_psd_std = float(np.nanstd(target_np[:, n_chem:][psd_eval]))
    zero_delta = float(
        np.nanmean(np.abs(baseline_pred[:, n_chem:][psd_eval] - zero_pred[:, n_chem:][psd_eval]))
    )
    shuffled_delta = float(
        np.nanmean(
            np.abs(
                baseline_pred[:, n_chem:][psd_eval]
                - shuffled_pred[:, n_chem:][psd_eval]
            )
        )
    )
    sensitivity_scale = max(true_psd_std, 1e-12)

    if latent_gain > 0.01 and latent_share >= 0.25:
        latent_verdict = "material_latent_posterior_bottleneck"
    elif latent_gain > 0 and latent_share >= 0.1:
        latent_verdict = "secondary_latent_posterior_bottleneck"
    else:
        latent_verdict = "little_evidence_latent_posterior_is_primary"

    if context_gain > max(2.0 * latent_gain, 0.01):
        context_verdict = "encoder_context_path_dominates"
    elif latent_gain > max(2.0 * context_gain, 0.01):
        context_verdict = "latent_path_dominates"
    else:
        context_verdict = "latent_and_context_are_coupled_or_comparable"

    if max(zero_delta, shuffled_delta) / sensitivity_scale < 0.02:
        usage_verdict = "decoder_weakly_uses_latent"
    else:
        usage_verdict = "decoder_output_is_sensitive_to_latent"

    optimized_interpretation = None
    if OPTIMIZED_VARIANT in set(metrics_df["variant"]):
        optimized_mae = metric(OPTIMIZED_VARIANT, "psd", "mae")
        optimized_gain = baseline_mae - optimized_mae
        finite_batches = [
            item for item in optimized_batch_records
            if np.isfinite(item.get("final_psd_mse", float("nan")))
        ]
        mean_displacement = (
            float(np.mean([item["z0_rms_displacement"] for item in finite_batches]))
            if finite_batches else float("nan")
        )
        if optimized_gain >= 0.02:
            capacity_verdict = "material_decoder_latent_headroom"
        elif optimized_gain > 0:
            capacity_verdict = "limited_decoder_latent_headroom"
        else:
            capacity_verdict = "no_decoder_latent_headroom_detected"
        optimized_interpretation = {
            "psd_mae": optimized_mae,
            "psd_mae_gain_vs_masked": optimized_gain,
            "mean_z0_rms_displacement": mean_displacement,
            "capacity_verdict": capacity_verdict,
            "caveat": (
                "This truth-optimized latent is an upper-bound intervention and can "
                "overfit held-out values; it does not establish that a diffusion model "
                "can infer the same correction from observed context."
            ),
        }

    interpretation = {
        "latent_verdict": latent_verdict,
        "context_verdict": context_verdict,
        "latent_usage_verdict": usage_verdict,
        "optimized_latent": optimized_interpretation,
        "psd_mae": {
            "masked_baseline": baseline_mae,
            "latent_oracle": latent_mae,
            "context_oracle": context_mae,
            "full_oracle": full_mae,
            "latent_zero": zero_mae,
            "latent_shuffled": shuffled_mae,
        },
        "psd_oracle_gains": {
            "latent_gain": latent_gain,
            "context_gain": context_gain,
            "total_full_oracle_gain": total_gain,
            "latent_share_of_total_gain": latent_share,
            "context_share_of_total_gain": context_share,
        },
        "latent_sensitivity": {
            "true_psd_standard_deviation": true_psd_std,
            "mean_abs_delta_zero_latent": zero_delta,
            "mean_abs_delta_shuffled_latent": shuffled_delta,
            "zero_delta_over_true_std": zero_delta / sensitivity_scale,
            "shuffled_delta_over_true_std": shuffled_delta / sensitivity_scale,
        },
        "decision_rule": (
            "Conditional latent diffusion is justified only when the full-input latent "
            "materially rescues masked-context predictions and the decoder is demonstrably "
            "sensitive to in-distribution latent changes."
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    gap_df.to_csv(output_dir / "gap_metrics.csv", index=False)
    window_df.to_csv(output_dir / "window_latent_diagnostics.csv", index=False)
    if optimized_batch_records:
        pd.DataFrame(optimized_batch_records).to_csv(
            output_dir / "latent_optimization_batches.csv", index=False
        )
    summary = {
        "schema_version": 1,
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(checkpoint_path),
        "legacy_repo": str(args.legacy_repo.resolve()),
        "device": str(device),
        "window_size": window_size,
        "stride": stride,
        "selected_windows": int(selected_indices.size),
        "selection_family": args.selection_family,
        "covered_heldout_points": int(np.sum(evaluation_mask)),
        "n_chem": n_chem,
        "n_psd": target_dim - n_chem,
        "used_realnvp": bool(getattr(model, "use_realnvp", False)),
        "forward_parity_max_abs": forward_parity_max_abs,
        "forward_parity_tolerance": float(args.parity_tolerance),
        "invalid_naturally_missing_heldout_points_removed": int(np.sum(invalid_heldout)),
        "latent_summary": latent_summary,
        "latent_optimization": {
            "steps": int(args.optimize_latent_steps),
            "learning_rate": float(args.optimize_latent_lr),
            "regularization": float(args.optimize_latent_regularization),
            "batches": optimized_batch_records,
        },
        "psd_spectrum_metrics": spectrum,
        "interpretation": interpretation,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(_json_ready(summary["interpretation"]), indent=2, sort_keys=True))
    print(f"Wrote diagnostics to {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-repo", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--config-name", default="config.json")
    parser.add_argument("--heldout-name", default="heldout_mask.npy")
    parser.add_argument("--gap-name", default="gap_duration_map.npy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional deterministic time-stratified subset; omit for all held-out windows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-family",
        choices=("all", "chem", "psd"),
        default="all",
        help="Select windows containing held-out points from this modality.",
    )
    parser.add_argument("--edge-fraction", type=float, default=0.2)
    parser.add_argument("--active-unit-threshold", type=float, default=1e-2)
    parser.add_argument("--parity-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--optimize-latent-steps",
        type=int,
        default=0,
        help="Diagnostic PSD-only latent optimization steps; zero disables it.",
    )
    parser.add_argument("--optimize-latent-lr", type=float, default=0.03)
    parser.add_argument("--optimize-latent-regularization", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not 0.0 <= args.edge_fraction < 0.5:
        raise ValueError("--edge-fraction must be in [0, 0.5)")
    run(args)


if __name__ == "__main__":
    main()
