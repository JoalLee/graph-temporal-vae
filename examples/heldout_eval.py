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
from graph_tcn_vae.infer import aggregate_window_samples, load_bundle, trapezoid_position_weights
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
    ap.add_argument("-o", "--output", default=None)
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

    stride = args.stride or max(1, window_size // 2)
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

    def iter_window_samples():
        model.eval()
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
                samples_scaled = result[-2].cpu().numpy()
                samples_model = (
                    samples_scaled * scaler_target.std_[None, None, None, :]
                    + scaler_target.mean_[None, None, None, :]
                )
                samples = inverse_target_values(samples_model, target_output_transform)
                for i, start in enumerate(batch_starts):
                    yield start, samples[:, i]

    aggregated = aggregate_window_samples(
        iter_window_samples(), total_length=n,
        position_weights=trapezoid_position_weights(window_size),
        quantiles=(0.025, 0.975),
    )
    mean_out = aggregated["mean"]
    q025 = aggregated["quantiles"][0.025]
    q975 = aggregated["quantiles"][0.975]

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


if __name__ == "__main__":
    main()
