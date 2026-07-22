"""Load a checkpoint bundle produced by `train.py` and impute new CSVs."""
from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
import torch

from .data import NaNAwareStandardScaler, compute_window_starts, load_frame, make_condition
from .model_graph_uq import ImputationVAE_Graph
from .utils import setup_device


def _validate_bundle(bundle):
    required = {
        "state_dict", "model_kwargs", "target_cols", "aux_cols", "window_size",
        "scaler_target", "scaler_aux", "config",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Invalid checkpoint bundle; missing keys: {missing}")
    if not bundle["target_cols"]:
        raise ValueError("Checkpoint bundle must contain at least one target column")
    if bundle["aux_missing_mode"] not in {"legacy_zero_fill", "mask_channel"}:
        raise ValueError(f"Unsupported aux_missing_mode: {bundle['aux_missing_mode']!r}")
    if len(bundle["scaler_target"]["mean"]) != len(bundle["target_cols"]):
        raise ValueError("Target scaler schema does not match target_cols")
    if len(bundle["scaler_aux"]["mean"]) != len(bundle["aux_cols"]):
        raise ValueError("Aux scaler schema does not match aux_cols")
    schema = bundle.get("schema")
    if schema is not None:
        expected_cond_dim = len(bundle["aux_cols"]) * (2 if bundle["aux_mask_channel"] else 1)
        if schema.get("target_dim") != len(bundle["target_cols"]):
            raise ValueError("Checkpoint target schema dimension does not match target_cols")
        if schema.get("cond_dim") != expected_cond_dim:
            raise ValueError("Checkpoint condition schema dimension does not match aux mask mode")


def load_bundle(path, device=None):
    device = device or setup_device()
    bundle = torch.load(path, map_location=device, weights_only=False)
    bundle.setdefault("bundle_version", 1)
    bundle.setdefault("aux_missing_mode", "legacy_zero_fill")
    bundle.setdefault("aux_mask_channel", bundle["aux_missing_mode"] == "mask_channel")
    _validate_bundle(bundle)

    aux_dim = len(bundle["aux_cols"]) * (2 if bundle["aux_mask_channel"] else 1)

    model = ImputationVAE_Graph(
        target_dim=len(bundle["target_cols"]),
        aux_dim=aux_dim,
        window_size=bundle["window_size"],
        **bundle["model_kwargs"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()

    return {
        "model": model,
        "scaler_target": NaNAwareStandardScaler.from_dict(bundle["scaler_target"]),
        "scaler_aux": NaNAwareStandardScaler.from_dict(bundle["scaler_aux"]),
        "target_cols": bundle["target_cols"],
        "aux_cols": bundle["aux_cols"],
        "window_size": bundle["window_size"],
        "aux_missing_mode": bundle["aux_missing_mode"],
        "aux_mask_channel": bundle["aux_mask_channel"],
        "bundle_version": bundle["bundle_version"],
        "schema": bundle.get("schema"),
        "timestamp_col": bundle["config"].get("timestamp_col"),
        "device": device,
    }


def trapezoid_position_weights(window_size):
    """Position envelope used to downweight window edges during overlap-add."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if window_size == 1:
        return np.ones(1, dtype=np.float64)
    weights = np.ones(window_size, dtype=np.float64)
    weights[[0, -1]] = 0.5
    return weights


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    centers = np.cumsum(weights) - 0.5 * weights
    return float(np.interp(quantile * weights.sum(), centers, values))


def aggregate_window_samples(
    window_samples,
    window_starts=None,
    total_length=None,
    position_weights=None,
    quantiles=(0.05, 0.95),
):
    """Aggregate overlapping predictive samples at sample level.

    ``window_samples`` is either ``[n_windows, n_mc, window, features]`` or
    an iterable of ``(start, [n_mc, window, features])`` pairs.  The latter is
    used by inference so windows can be consumed one at a time.  Quantiles are
    weighted percentiles of the overlap mixture, rather than averages of
    per-window quantiles.
    """
    if window_starts is not None:
        arrays = np.asarray(window_samples, dtype=np.float64)
        if arrays.ndim != 4:
            raise ValueError("window_samples must have shape [windows, mc, window, features]")
        pairs = zip(window_starts, arrays)
        n_windows, _n_mc, window_size, n_features = arrays.shape
    else:
        pairs = iter(window_samples)
        first = next(pairs, None)
        if first is None:
            raise ValueError("window_samples cannot be empty")
        first_start, first_array = first
        first_array = np.asarray(first_array, dtype=np.float64)
        if first_array.ndim != 3:
            raise ValueError("per-window samples must have shape [mc, window, features]")
        window_size, n_features = first_array.shape[1:]
        pairs = chain([(first_start, first_array)], pairs)

    if total_length is None:
        if window_starts is None:
            raise ValueError("total_length is required for streaming aggregation")
        total_length = max(int(start) for start in window_starts) + window_size
    if position_weights is None:
        position_weights = trapezoid_position_weights(window_size)
    position_weights = np.asarray(position_weights, dtype=np.float64)
    if position_weights.shape != (window_size,) or np.any(position_weights <= 0):
        raise ValueError("position_weights must be positive and have length window_size")

    values_by_position = [[] for _ in range(total_length)]
    weights_by_position = [[] for _ in range(total_length)]
    for start, samples in pairs:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 3 or samples.shape[1:] != (window_size, n_features):
            raise ValueError("all windows must have shape [mc, window, features]")
        start = int(start)
        end = start + window_size
        if start < 0 or end > total_length:
            raise ValueError(f"window [{start}, {end}) exceeds total_length={total_length}")
        for local_pos, global_pos in enumerate(range(start, end)):
            values_by_position[global_pos].append(samples[:, local_pos, :])
            weights_by_position[global_pos].append(
                np.full(samples.shape[0], position_weights[local_pos], dtype=np.float64)
            )

    mean = np.full((total_length, n_features), np.nan, dtype=np.float64)
    variance = np.full_like(mean, np.nan)
    quantile_values = {float(q): np.full_like(mean, np.nan) for q in quantiles}
    for pos in range(total_length):
        if not values_by_position[pos]:
            continue
        values = np.concatenate(values_by_position[pos], axis=0)
        weights = np.concatenate(weights_by_position[pos], axis=0)
        normalizer = weights.sum()
        mean[pos] = (values * weights[:, None]).sum(axis=0) / normalizer
        variance[pos] = np.maximum(
            (values * values * weights[:, None]).sum(axis=0) / normalizer - mean[pos] ** 2,
            0.0,
        )
        for q, output in quantile_values.items():
            for feature in range(n_features):
                output[pos, feature] = _weighted_quantile(values[:, feature], weights, q)

    return {"mean": mean, "variance": variance, "quantiles": quantile_values}


def impute(csv_paths, bundle, output_csv, stride=None, n_mc_samples=50, timestamp_col=None):
    """Impute/predict on new CSVs using a trained checkpoint bundle.

    `bundle` may be a path to a checkpoint file or an already-loaded bundle
    dict from `load_bundle`. Writes a tidy long-format CSV: one row per
    (timestamp, feature) with the raw observed value (if any), the imputed
    mean/std, and a 5-95% predictive interval.
    """
    if isinstance(bundle, (str, Path)):
        bundle = load_bundle(bundle)

    model = bundle["model"]
    device = bundle["device"]
    target_cols = bundle["target_cols"]
    aux_cols = bundle["aux_cols"]
    window_size = bundle["window_size"]
    scaler_target = bundle["scaler_target"]
    scaler_aux = bundle["scaler_aux"]
    aux_mask_channel = bool(bundle.get("aux_mask_channel", False))

    ts_col = timestamp_col or bundle["timestamp_col"]
    if not ts_col:
        raise ValueError("timestamp_col not found in bundle config; pass timestamp_col= explicitly")

    stride = stride or max(1, window_size // 2)
    stride = min(stride, window_size)

    frame = load_frame(csv_paths, ts_col, target_cols, aux_cols)
    n = len(frame)
    starts = compute_window_starts(n, window_size, stride)
    if not starts:
        raise ValueError(f"Series length {n} is shorter than window_size={window_size}")

    target_raw = frame[target_cols].to_numpy(dtype=np.float64)
    aux_raw = frame[aux_cols].to_numpy(dtype=np.float64) if aux_cols else np.zeros((n, 0))
    obs_mask_full = (~np.isnan(target_raw)).astype(np.float32)

    target_scaled = np.nan_to_num(scaler_target.transform(target_raw), nan=0.0)
    aux_scaled = scaler_aux.transform(aux_raw) if aux_cols else aux_raw
    aux_observed = ~np.isnan(aux_raw)

    def iter_window_samples():
        model.eval()
        with torch.no_grad():
            for start in starts:
                end = start + window_size
                mask_win = obs_mask_full[start:end]
                x_win = target_scaled[start:end] * mask_win
                cond_win = make_condition(aux_scaled[start:end], aux_observed[start:end], aux_mask_channel)

                x_t = torch.from_numpy(x_win).float().unsqueeze(0).to(device)
                cond_t = torch.from_numpy(cond_win).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask_win).float().unsqueeze(0).to(device)

                result = model.compute_uncertainty(
                    x_t, cond_t, mask_t, n_samples=max(2, n_mc_samples), return_samples=True,
                )
                samples = result[-2].squeeze(1).cpu().numpy()
                yield start, samples

    aggregated = aggregate_window_samples(
        iter_window_samples(), total_length=n,
        position_weights=trapezoid_position_weights(window_size),
        quantiles=(0.05, 0.95),
    )
    mean_scaled = aggregated["mean"]
    var_scaled = aggregated["variance"]
    q05_scaled = aggregated["quantiles"][0.05]
    q95_scaled = aggregated["quantiles"][0.95]

    mean_out = scaler_target.inverse_transform(mean_scaled)
    std_out = np.sqrt(var_scaled) * scaler_target.std_
    q05_out = scaler_target.inverse_transform(q05_scaled)
    q95_out = scaler_target.inverse_transform(q95_scaled)

    # Restore observed values verbatim; there's no imputation uncertainty there.
    mean_out = np.where(obs_mask_full == 1, target_raw, mean_out)
    std_out = np.where(obs_mask_full == 1, 0.0, std_out)
    q05_out = np.where(obs_mask_full == 1, target_raw, q05_out)
    q95_out = np.where(obs_mask_full == 1, target_raw, q95_out)

    frames = []
    for j, col in enumerate(target_cols):
        frames.append(pd.DataFrame({
            "timestamp": frame.index,
            "feature": col,
            "observed": target_raw[:, j],
            "imputed_mean": mean_out[:, j],
            "imputed_std": std_out[:, j],
            "q05": q05_out[:, j],
            "q95": q95_out[:, j],
        }))
    result_df = pd.concat(frames, ignore_index=True)
    result_df.to_csv(output_csv, index=False)
    return result_df
