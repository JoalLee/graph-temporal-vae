"""Held-out evaluation for a graph-tcn-vae checkpoint bundle.

Reconstructs the SAME fixed anchor-constrained held-out mask used during
training (identical seed/ratio/n_chem -> identical mask, see
train.py:sample_anchor_constrained_heldout_mask under shared_full_heldout_mask),
forces those points to look unobserved to the model at inference time (as
training did), and scores predictions against true values only at those
points. `impute()` alone cannot produce this number because it always
treats real observations as observed -- this is the missing piece for
answering "how close does this checkpoint get to a reference run's
reported held-out R²/MAE?".

Usage:
    python examples/heldout_eval.py \\
        --bundle checkpoints/run1.pt --csv data.csv \\
        --n-chem 32 -o heldout_metrics.json
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_tcn_vae.data import (
    compute_window_starts,
    inverse_target_values,
    load_frame,
    make_condition,
    sample_anchor_constrained_heldout_mask,
    transform_target_values,
)
from graph_tcn_vae.infer import (
    aggregate_window_samples,
    load_bundle,
    summary_to_output_scale,
    trapezoid_position_weights,
)
from graph_tcn_vae.utils import is_interactive


def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--csv", required=True, help="Comma-separated CSV path(s).")
    ap.add_argument("--n-chem", type=int, required=True, help="First N target columns treated as the Chem modality.")
    ap.add_argument("--selection-mask-ratio", type=float, default=0.1)
    ap.add_argument("--selection-val-seed", type=int, default=42)
    ap.add_argument("--n-mc-samples", type=int, default=50)
    ap.add_argument("--stride", type=int, default=None, help="Defaults to window_size // 2.")
    ap.add_argument("--inference-batch-size", type=int, default=4)
    ap.add_argument("-o", "--output", default=None, help="Path to write the aggregate metrics JSON.")
    ap.add_argument(
        "--predictions-csv", default=None,
        help="Optional path to write one row per held-out (timestamp, feature): both the scaled "
             "(standardized, pre-output-transform model space) and physical (post-transform) "
             "observed value and prediction, so the reported metrics can be checked by hand -- "
             "e.g. to rule out a double-applied output transform.",
    )
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
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
    ts_col = bundle["timestamp_col"]

    csv_paths = [v.strip() for v in args.csv.split(",") if v.strip()]
    frame = load_frame(csv_paths, ts_col, target_cols, aux_cols)
    n = len(frame)
    target_raw = frame[target_cols].to_numpy(dtype=np.float64)
    target_model_space = transform_target_values(target_raw, target_transform)
    aux_raw = frame[aux_cols].to_numpy(dtype=np.float64) if aux_cols else np.zeros((n, 0))
    obs_mask_full = ~np.isnan(target_raw)

    heldout_mask = sample_anchor_constrained_heldout_mask(
        obs_mask_full, ratio=args.selection_mask_ratio, seed=args.selection_val_seed, n_chem=args.n_chem,
    ).astype(bool)

    # Force held-out points to look unobserved to the model, exactly as
    # training did (fixed_mask carved out of the training input mask).
    input_obs_mask = (obs_mask_full & ~heldout_mask).astype(np.float32)

    target_scaled = np.nan_to_num(scaler_target.transform(target_model_space), nan=0.0)
    aux_scaled = scaler_aux.transform(aux_raw) if aux_cols else aux_raw
    aux_observed = ~np.isnan(aux_raw)

    # Prefer the training stride stored in the bundle.  The reference 26e
    # protocol uses stride=24; window_size//2 would produce a different
    # overlap geometry when --stride is omitted.
    stride = args.stride or bundle.get("stride") or max(1, window_size // 2)
    stride = min(stride, window_size)
    starts = compute_window_starts(n, window_size, stride)
    if not starts:
        raise ValueError(f"Series length {n} is shorter than window_size={window_size}")

    n_batches = math.ceil(len(starts) / args.inference_batch_size)
    print(
        f"[heldout_eval] {n} rows -> {len(starts)} windows ({n_batches} batches), "
        f"{heldout_mask.sum()}/{obs_mask_full.sum()} observed points held out "
        f"({100 * heldout_mask.sum() / obs_mask_full.sum():.2f}%), "
        f"{args.n_mc_samples} MC samples, stride={stride}, device={device}"
    )

    def compute_window_predictions():
        # See infer.impute's compute_window_predictions for why this is two
        # streams: mean_chunks (the decoder's own noise-free point estimate,
        # MC-dropout-averaged only) drives the R^2/MAE point estimate;
        # sample_chunks (full generative mean+noise draws) drives quantiles
        # only. Averaging noisy draws for the point estimate is what turned
        # a real held-out eval's psd_heldout_r2 into the thousands-negative.
        model.eval()
        mean_chunks = []
        sample_chunks = []
        with torch.no_grad():
            batch_starts_list = range(0, len(starts), args.inference_batch_size)
            for batch_start in tqdm(
                batch_starts_list, desc="heldout-eval windows", total=n_batches, disable=not is_interactive()
            ):
                batch_starts = starts[batch_start:batch_start + args.inference_batch_size]
                masks = np.stack([input_obs_mask[s:s + window_size] for s in batch_starts])
                xs = np.stack([
                    target_scaled[s:s + window_size] * masks[i] for i, s in enumerate(batch_starts)
                ])
                conds = np.stack([
                    make_condition(
                        aux_scaled[s:s + window_size], aux_observed[s:s + window_size], aux_mask_channel
                    )
                    for s in batch_starts
                ])
                x_t = torch.from_numpy(xs).float().to(device)
                cond_t = torch.from_numpy(conds).float().to(device)
                mask_t = torch.from_numpy(masks).float().to(device)
                result = model.compute_uncertainty(
                    x_t, cond_t, mask_t, n_samples=max(2, args.n_mc_samples), return_samples=True,
                )
                pred_mean_scaled = result[0].cpu().numpy()
                samples_scaled = result[-2].cpu().numpy()
                pred_mean_model = (
                    pred_mean_scaled * scaler_target.std_[None, None, :]
                    + scaler_target.mean_[None, None, :]
                )
                samples_model = (
                    samples_scaled * scaler_target.std_[None, None, None, :]
                    + scaler_target.mean_[None, None, None, :]
                )
                for i, start in enumerate(batch_starts):
                    mean_chunks.append((start, pred_mean_model[i][None, :, :]))
                    sample_chunks.append((start, samples_model[:, i]))
        return mean_chunks, sample_chunks

    mean_chunks, sample_chunks = compute_window_predictions()
    position_weights = trapezoid_position_weights(window_size)
    mean_agg = aggregate_window_samples(
        mean_chunks, total_length=n, position_weights=position_weights, quantiles=()
    )
    dist_agg = aggregate_window_samples(
        sample_chunks, total_length=n, position_weights=position_weights, quantiles=(0.025, 0.975)
    )
    mean_out, std_out, quantiles_out = summary_to_output_scale(
        mean_agg["mean"], dist_agg["variance"], dist_agg["quantiles"], target_output_transform
    )
    q025 = quantiles_out[0.025]
    q975 = quantiles_out[0.975]

    observed_output = inverse_target_values(target_raw, target_output_transform)

    n_chem = args.n_chem
    results = {}
    for group_name, cols_slice in (("chem", slice(0, n_chem)), ("psd", slice(n_chem, None))):
        mask_g = heldout_mask[:, cols_slice]
        y_true = observed_output[:, cols_slice][mask_g]
        y_pred = mean_out[:, cols_slice][mask_g]
        q025_g = q025[:, cols_slice][mask_g]
        q975_g = q975[:, cols_slice][mask_g]
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred, q025_g, q975_g = y_true[valid], y_pred[valid], q025_g[valid], q975_g[valid]
        picp = float(np.mean((y_true >= q025_g) & (y_true <= q975_g)))
        results[f"{group_name}_heldout_r2"] = _r2_score(y_true, y_pred)
        results[f"{group_name}_heldout_mae"] = _mae(y_true, y_pred)
        results[f"{group_name}_heldout_picp95"] = picp
        results[f"{group_name}_heldout_n"] = int(len(y_true))

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    if args.predictions_csv:
        # One row per held-out (timestamp, feature), at every level a
        # transform-order bug could hide: the standardized (z-score) model
        # input/output the network actually sees, and the fully
        # de-standardized + inverse-transformed physical value the reported
        # metrics are computed from. Comparing these by hand for a few rows
        # is the fastest way to confirm the output transform was applied
        # exactly once, in the right place.
        rows, cols = np.nonzero(heldout_mask)
        families = np.where(cols < n_chem, "chem", "psd")
        frames_out = pd.DataFrame({
            "timestamp": frame.index[rows],
            "feature": [target_cols[c] for c in cols],
            "family": families,
            "scaled_observed": target_scaled[rows, cols],
            "scaled_pred_mean": mean_agg["mean"][rows, cols],
            "physical_observed": observed_output[rows, cols],
            "physical_pred_mean": mean_out[rows, cols],
            "physical_pred_std": std_out[rows, cols],
            "physical_q025": q025[rows, cols],
            "physical_q975": q975[rows, cols],
        })
        frames_out.to_csv(args.predictions_csv, index=False)
        print(f"wrote {len(frames_out)} held-out predictions to {args.predictions_csv}")


if __name__ == "__main__":
    main()
