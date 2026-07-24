"""Load a checkpoint bundle produced by `train.py` and impute new CSVs."""
import math
from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import (
    NaNAwareStandardScaler,
    compute_window_starts,
    inverse_target_values,
    load_frame,
    make_condition,
    transform_target_values,
)
from .model_graph_uq import ImputationVAE_Graph
from .utils import is_interactive, setup_device


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
    if bundle.get("target_transform", "none") not in {"none", "log1p"}:
        raise ValueError(f"Unsupported target_transform: {bundle.get('target_transform')!r}")
    if bundle.get("target_output_transform", "none") not in {"none", "log1p"}:
        raise ValueError(
            "Unsupported target_output_transform: "
            f"{bundle.get('target_output_transform')!r}"
        )
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
    # Bundle files are data, not executable Python objects.  weights_only=True
    # blocks arbitrary pickle object construction when loading an untrusted file.
    bundle = torch.load(path, map_location=device, weights_only=True)
    bundle.setdefault("bundle_version", 1)
    bundle.setdefault("aux_missing_mode", "legacy_zero_fill")
    bundle.setdefault("aux_mask_channel", bundle["aux_missing_mode"] == "mask_channel")
    bundle.setdefault("target_transform", bundle.get("config", {}).get("target_transform", "none"))
    bundle.setdefault(
        "time_grid",
        {
            "frequency": bundle.get("config", {}).get("expected_frequency"),
            "timezone": None,
            "policy": bundle.get("config", {}).get("time_grid_policy", "row_order"),
            "duplicate_timestamp_policy": bundle.get("config", {}).get(
                "duplicate_timestamp_policy", "first"
            ),
        },
    )
    bundle.setdefault(
        "target_output_transform",
        bundle.get("config", {}).get("target_output_transform", bundle["target_transform"]),
    )
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
        "stride": bundle.get("stride", bundle.get("config", {}).get("stride")),
        "aux_missing_mode": bundle["aux_missing_mode"],
        "aux_mask_channel": bundle["aux_mask_channel"],
        "target_transform": bundle.get("target_transform", "none"),
        "target_output_transform": bundle.get("target_output_transform", "none"),
        "bundle_version": bundle["bundle_version"],
        "schema": bundle.get("schema"),
        "timestamp_col": bundle["config"].get("timestamp_col"),
        "time_grid": bundle.get("time_grid", {}),
        "device": device,
    }


def trapezoid_position_weights(window_size, edge_frac=0.2):
    """Position envelope used to downweight window edges during overlap-add.

    1 in the middle, linear ramp over ``edge_frac`` of the window at each
    edge, matching the research trapezoidal_window_weights envelope: an
    edge timestep has less surrounding context within that window's
    representation, so it should contribute less to the overlap-add mix.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if window_size == 1:
        return np.ones(1, dtype=np.float64)
    edge_len = max(1, int(round(window_size * edge_frac)))
    if 2 * edge_len >= window_size:
        edge_len = max(1, (window_size - 1) // 2)
    weights = np.ones(window_size, dtype=np.float64)
    ramp = (np.arange(edge_len) + 1.0) / (edge_len + 1.0)
    weights[:edge_len] = ramp
    weights[-edge_len:] = ramp[::-1]
    return weights


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    centers = np.cumsum(weights) - 0.5 * weights
    return float(np.interp(quantile * weights.sum(), centers, values))


def summary_to_output_scale(mean_model, variance_model, quantile_values_model, transform):
    """Map aggregated (model-space) summary statistics to the output scale.

    This must run ONCE on the aggregated mean/variance/quantiles, not once
    per MC sample before aggregation. ``expm1`` (the inverse of the log1p
    target transform) is convex, so averaging already-exponentiated
    heavy-tailed Student-t draws is biased high (Jensen's inequality) and
    numerically unstable -- a single extreme draw dominates the arithmetic
    mean once exponentiated, which is what produced R^2 << 0 on log1p-scale
    targets before this was fixed. Matches the research pipeline: aggregate
    the mean/quantiles in the linear model space, then apply the inverse
    transform once to those summary statistics. Quantiles come out exact
    under this ordering because the transform is monotonic; std uses a
    first-order (delta-method) approximation, ``exp(mean) * std_model``,
    matching the reference's own uncertainty propagation through expm1.
    """
    mean_out = inverse_target_values(mean_model, transform)
    std_model = np.sqrt(np.maximum(variance_model, 0.0))
    if transform == "log1p":
        std_out = np.exp(mean_model) * std_model
    else:
        std_out = std_model
    quantiles_out = {q: inverse_target_values(v, transform) for q, v in quantile_values_model.items()}
    return mean_out, std_out, quantiles_out


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
    # Pure-CPU weighted-quantile pass over every timestep x feature; on a
    # long series with many features this can take a while after all GPU
    # work is already done, so it gets its own bar rather than looking like
    # a silent hang.
    for pos in tqdm(range(total_length), desc="aggregating", disable=not is_interactive()):
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


def _compute_support_diagnostics(obs_mask, context_window=72):
    """Describe each missing run without claiming model calibration.

    Risk tiers are operational heuristics based on gap length and bilateral
    observed context. They are not learned probabilities and should not replace
    held-out evaluation on the user's dataset.
    """
    observed = np.asarray(obs_mask, dtype=bool)
    n_rows, n_features = observed.shape
    gap_length = np.zeros((n_rows, n_features), dtype=np.int32)
    left_context = np.full((n_rows, n_features), np.nan, dtype=np.float64)
    right_context = np.full((n_rows, n_features), np.nan, dtype=np.float64)
    risk = np.full((n_rows, n_features), "observed", dtype=object)

    for feature in range(n_features):
        missing = ~observed[:, feature]
        starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
        ends = np.flatnonzero(missing & ~np.r_[missing[1:], False]) + 1
        for start, end in zip(starts, ends):
            length = int(end - start)
            left = observed[max(0, start - context_window):start, feature]
            right = observed[end:min(n_rows, end + context_window), feature]
            left_fraction = float(left.mean()) if left.size else 0.0
            right_fraction = float(right.mean()) if right.size else 0.0
            gap_length[start:end, feature] = length
            left_context[start:end, feature] = left_fraction
            right_context[start:end, feature] = right_fraction
            if length <= 12 and left_fraction >= 0.75 and right_fraction >= 0.75:
                tier = "low"
            elif length >= 96 or min(left_fraction, right_fraction) < 0.25:
                tier = "high"
            else:
                tier = "moderate"
            risk[start:end, feature] = tier

    return {
        "gap_length": gap_length,
        "left_context_fraction": left_context,
        "right_context_fraction": right_context,
        "heuristic_risk_tier": risk,
    }


def impute(
    csv_paths,
    bundle,
    output_csv,
    stride=None,
    n_mc_samples=50,
    timestamp_col=None,
    inference_batch_size=4,
    mc_batch_size=1,
    support_context_window=72,
):
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
    target_transform = bundle.get("target_transform", "none")
    target_output_transform = bundle.get("target_output_transform", target_transform)

    ts_col = timestamp_col or bundle["timestamp_col"]
    if not ts_col:
        raise ValueError("timestamp_col not found in bundle config; pass timestamp_col= explicitly")

    # Reuse the training window stride unless the caller explicitly overrides
    # it.  The 26e reference uses stride=24; falling back to window//2 would
    # silently change the overlap geometry during inference.
    if stride is None:
        stride = bundle.get("stride") or max(1, window_size // 2)
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if stride > window_size:
        raise ValueError("stride cannot exceed window_size")
    if not isinstance(n_mc_samples, int) or n_mc_samples < 2:
        raise ValueError("n_mc_samples must be an integer >= 2")
    if inference_batch_size < 1 or mc_batch_size < 1:
        raise ValueError("inference_batch_size and mc_batch_size must be positive")
    if support_context_window < 1:
        raise ValueError("support_context_window must be positive")

    time_grid = bundle.get("time_grid", {})
    frame = load_frame(
        csv_paths,
        ts_col,
        target_cols,
        aux_cols,
        expected_frequency=time_grid.get("frequency"),
        time_grid_policy=time_grid.get("policy", "row_order"),
        duplicate_timestamp_policy=time_grid.get(
            "duplicate_timestamp_policy", "error"
        ),
    )
    n = len(frame)
    starts = compute_window_starts(n, window_size, stride)
    if not starts:
        raise ValueError(f"Series length {n} is shorter than window_size={window_size}")

    target_raw = frame[target_cols].to_numpy(dtype=np.float64)
    target_model_space = transform_target_values(target_raw, target_transform)
    aux_raw = frame[aux_cols].to_numpy(dtype=np.float64) if aux_cols else np.zeros((n, 0))
    obs_mask_full = (~np.isnan(target_raw)).astype(np.float32)

    target_scaled = np.nan_to_num(scaler_target.transform(target_model_space), nan=0.0)
    aux_scaled = scaler_aux.transform(aux_raw) if aux_cols else aux_raw
    aux_observed = ~np.isnan(aux_raw)

    n_batches = math.ceil(len(starts) / inference_batch_size)
    print(
        f"[graph-tcn-vae] {n} rows -> {len(starts)} windows ({n_batches} batches), "
        f"{len(target_cols)} targets, {n_mc_samples} MC samples, stride={stride}, device={device}"
    )

    def compute_window_predictions():
        """Run the model once per window batch and split its output into two
        separate per-window streams for the cross-window aggregator:

        - mean_chunks: the decoder's own point-estimate mean (`compute_uncertainty`
          result[0]), averaged only over MC-*dropout* draws -- no injected
          Student-t/Gaussian noise. This is what the point-accuracy estimate
          (`imputed_mean`) should be built from: it is exactly as stable as
          the decoder's clamped-variance output, with no risk of a single
          heavy-tailed noise draw dragging the aggregate.
        - sample_chunks: the full generative draws (result[-2], mean + noise),
          used ONLY for quantiles/std, where reflecting the full predictive
          distribution (not just the epistemic spread of the mean) is
          actually what's wanted.

        Averaging the noisy draws for the point estimate too (what this
        package used to do) is a real, not just theoretical, failure mode:
        on a real held-out eval it produced psd_heldout_r2 in the thousands-
        negative range, because a Student-t/Gaussian draw at even a
        moderate multiple of predicted std, once passed through the log1p
        output transform, can be enormous.
        """
        model.eval()
        mean_chunks = []
        sample_chunks = []
        with torch.no_grad():
            batch_starts_list = range(0, len(starts), inference_batch_size)
            for batch_start in tqdm(batch_starts_list, desc="impute windows", total=n_batches, disable=not is_interactive()):
                batch_starts = starts[batch_start:batch_start + inference_batch_size]
                masks = np.stack([obs_mask_full[s:s + window_size] for s in batch_starts])
                xs = np.stack([target_scaled[s:s + window_size] * masks[i] for i, s in enumerate(batch_starts)])
                conds = np.stack([
                    make_condition(aux_scaled[s:s + window_size], aux_observed[s:s + window_size], aux_mask_channel)
                    for s in batch_starts
                ])

                x_t = torch.from_numpy(xs).float().to(device)
                cond_t = torch.from_numpy(conds).float().to(device)
                mask_t = torch.from_numpy(masks).float().to(device)
                result = model.compute_uncertainty(
                    x_t,
                    cond_t,
                    mask_t,
                    n_samples=n_mc_samples,
                    return_samples=True,
                    mc_batch_size=mc_batch_size,
                )
                pred_mean_scaled = result[0].cpu().numpy()  # [batch, window, features], noise-free
                samples_scaled = result[-2].cpu().numpy()  # [MC, batch, window, features], noisy
                pred_mean_model = (
                    pred_mean_scaled * scaler_target.std_[None, None, :]
                    + scaler_target.mean_[None, None, :]
                )
                samples_model = (
                    samples_scaled * scaler_target.std_[None, None, None, :]
                    + scaler_target.mean_[None, None, None, :]
                )
                for i, start in enumerate(batch_starts):
                    mean_chunks.append((start, pred_mean_model[i][None, :, :]))  # [1, window, features]
                    sample_chunks.append((start, samples_model[:, i]))  # [MC, window, features]
        return mean_chunks, sample_chunks

    mean_chunks, sample_chunks = compute_window_predictions()
    position_weights = trapezoid_position_weights(window_size)
    mean_agg = aggregate_window_samples(
        mean_chunks, total_length=n, position_weights=position_weights, quantiles=()
    )
    dist_agg = aggregate_window_samples(
        sample_chunks, total_length=n, position_weights=position_weights, quantiles=(0.05, 0.95)
    )
    mean_out, std_out, quantiles_out = summary_to_output_scale(
        mean_agg["mean"], dist_agg["variance"], dist_agg["quantiles"], target_output_transform
    )
    q05_out = quantiles_out[0.05]
    q95_out = quantiles_out[0.95]

    # Restore observed values in the public/raw output space; there's no
    # imputation uncertainty there.  The input CSV may already be transformed
    # (e.g. the 26e artifact stores log1p targets), so target_raw is not
    # necessarily the physical output value.
    observed_output = inverse_target_values(target_raw, target_output_transform)
    mean_out = np.where(obs_mask_full == 1, observed_output, mean_out)
    std_out = np.where(obs_mask_full == 1, 0.0, std_out)
    q05_out = np.where(obs_mask_full == 1, observed_output, q05_out)
    q95_out = np.where(obs_mask_full == 1, observed_output, q95_out)

    support = _compute_support_diagnostics(
        obs_mask_full, context_window=support_context_window
    )
    frames = []
    for j, col in enumerate(target_cols):
        frames.append(pd.DataFrame({
            "timestamp": frame.index,
            "feature": col,
            "observed": observed_output[:, j],
            "is_imputed": obs_mask_full[:, j] == 0,
            "imputed_mean": mean_out[:, j],
            "imputed_std": std_out[:, j],
            "q05": q05_out[:, j],
            "q95": q95_out[:, j],
            "gap_length": support["gap_length"][:, j],
            "left_context_fraction": support["left_context_fraction"][:, j],
            "right_context_fraction": support["right_context_fraction"][:, j],
            "heuristic_risk_tier": support["heuristic_risk_tier"][:, j],
        }))
    result_df = pd.concat(frames, ignore_index=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    return result_df
