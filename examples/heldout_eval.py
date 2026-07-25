"""Held-out evaluation for a graph-temporal-vae checkpoint bundle.

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
from scipy.stats import norm
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_temporal_vae.contracts import ModalityFiles
from graph_temporal_vae.data import (
    compute_window_starts,
    load_frame,
    load_modality_frame,
    make_condition,
    sample_anchor_constrained_heldout_mask,
)
from graph_temporal_vae.infer import (
    load_bundle,
    summary_to_output_scale,
    trapezoid_position_weights,
)
from graph_temporal_vae.preprocessing import (
    observed_targets_to_output,
    transform_auxiliary,
    transform_targets,
)
from graph_temporal_vae.utils import is_interactive
from graph_temporal_vae.window_aggregation import StreamingWindowAggregator

# Same category definitions as the research repo's ablation_heldout_eval.py
# (CHEM_GROUPS / PSD_GROUPS), so per-category macro-averages are directly
# comparable to a reference run's reported numbers, not just the overall
# chem/psd split.
CHEM_GROUPS = {
    "gases": ["SO2", "NO", "NO2", "CO", "O3"],
    "ions": ["Na+", "NH4+", "Cl-", "NO2-", "NO3-", "SO42-"],
    "carbon": ["OC", "EC"],
    "metal": ["K", "Ca", "Ti", "V", "Cr", "Al", "Si", "Mn", "Fe", "Ni", "Cu", "Zn", "As", "Se", "Br", "Ba", "Pb"],
    "pm": ["PM2.5", "PM10"],
}
PSD_GROUPS = {
    "nucleation": lambda d: d < 30,
    "aitken": lambda d: 30 <= d < 100,
    "accumulation": lambda d: 100 <= d < 1000,
    "coarse_sub2.5": lambda d: 1000 <= d <= 2500,
    "coarse_super2.5": lambda d: d > 2500,
}


def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def category_indices(target_cols, n_chem):
    """(name, column-index-list) pairs matching the research repo's
    CHEM_GROUPS/PSD_GROUPS categorization: overall, chem, psd, then Chem
    sub-groups by species and PSD sub-groups by particle diameter bin.
    """
    chem_indices = list(range(n_chem))
    psd_indices = list(range(n_chem, len(target_cols)))
    cats = [("overall", list(range(len(target_cols)))), ("chem", chem_indices), ("psd", psd_indices)]
    for name, cols in CHEM_GROUPS.items():
        idx = [i for i in chem_indices if target_cols[i] in cols]
        if idx:
            cats.append((name, idx))
    # PSD sub-groups need the column name to parse as a particle diameter
    # (the 26e convention). A general dataset's non-chem columns may not be
    # named that way at all -- they still count in the "psd" category above,
    # just not in any diameter-based sub-group.
    psd_diameters = {}
    for i in psd_indices:
        try:
            psd_diameters[i] = float(target_cols[i])
        except ValueError:
            continue
    for name, in_range in PSD_GROUPS.items():
        idx = [i for i, d in psd_diameters.items() if in_range(d)]
        if idx:
            cats.append((name, idx))
    return cats


def _macro_average_metrics(
    y_true_cols,
    y_pred_cols,
    mask_cols,
    sigma_cols=None,
    q_lo_cols=None,
    q_hi_cols=None,
    empirical_crps_model_space_cols=None,
    min_points=10,
    clip_r2=True,
):
    """Per-feature R^2/MAE/RMSE/SMAPE(/CRPS/PICP), then average across
    features -- matches the research repo's ablation_heldout_eval.py
    compute_heldout_metrics exactly (see its 'Macro-average: compute
    per-feature then average' comment). Held-out R^2 is clipped to >= 0 per
    feature before averaging there (clip_r2=True); the reference does NOT
    clip the observed-point counterpart (clip_r2=False), which this
    function also supports so both comparisons match exactly.

    A single statistic pooled across every feature's points at once (what
    this script did originally) is DIFFERENT: it's dominated by whichever
    features have the largest magnitude/variance, so a model that fits a
    handful of high-variance features very well can look far better pooled
    than macro-averaged, even with mediocre or negative per-feature fit on
    most other features. sigma_cols/q_lo_cols/q_hi_cols are optional so this
    also serves the plain R^2/MAE-only use (e.g. the "observed" comparison).
    ``empirical_crps_model_space_cols`` is kept separate from the Gaussian
    physical-output CRPS so the two score definitions cannot be confused.
    """
    r2_list, mae_list, rmse_list, smape_list = [], [], [], []
    crps_list, empirical_crps_list, picp_list = [], [], []
    for j in range(y_true_cols.shape[1]):
        col_mask = mask_cols[:, j]
        if col_mask.sum() < min_points:
            continue
        y_true = y_true_cols[col_mask, j]
        y_pred = y_pred_cols[col_mask, j]
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if sigma_cols is not None:
            valid &= np.isfinite(sigma_cols[col_mask, j])
        if q_lo_cols is not None:
            valid &= np.isfinite(q_lo_cols[col_mask, j]) & np.isfinite(q_hi_cols[col_mask, j])
        if empirical_crps_model_space_cols is not None:
            valid &= np.isfinite(empirical_crps_model_space_cols[col_mask, j])
        if valid.sum() < min_points:
            continue
        y_true_v, y_pred_v = y_true[valid], y_pred[valid]

        r2 = _r2_score(y_true_v, y_pred_v)
        r2_list.append(max(0.0, r2) if clip_r2 else r2)
        mae_list.append(_mae(y_true_v, y_pred_v))
        rmse_list.append(float(np.sqrt(np.mean((y_true_v - y_pred_v) ** 2))))

        denom = np.abs(y_true_v) + np.abs(y_pred_v)
        smape = np.zeros_like(y_true_v)
        smape_valid = denom > 1e-8
        smape[smape_valid] = 2 * np.abs(y_true_v[smape_valid] - y_pred_v[smape_valid]) / denom[smape_valid]
        smape_list.append(float(np.mean(smape)) * 100.0)

        if sigma_cols is not None:
            sigma_v = np.maximum(sigma_cols[col_mask, j][valid], 1e-6)
            z = (y_true_v - y_pred_v) / sigma_v
            crps = sigma_v * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
            crps_list.append(float(np.mean(crps)))
        if empirical_crps_model_space_cols is not None:
            empirical_crps_list.append(
                float(np.mean(empirical_crps_model_space_cols[col_mask, j][valid]))
            )
        if q_lo_cols is not None:
            q_lo_v = q_lo_cols[col_mask, j][valid]
            q_hi_v = q_hi_cols[col_mask, j][valid]
            picp_list.append(float(np.mean((y_true_v >= q_lo_v) & (y_true_v <= q_hi_v))))

    out = {"n_features": len(r2_list)}
    if r2_list:
        out["r2"] = float(np.mean(r2_list))
        out["mae"] = float(np.mean(mae_list))
        out["rmse"] = float(np.mean(rmse_list))
        out["smape"] = float(np.mean(smape_list))
    if crps_list:
        out["crps"] = float(np.mean(crps_list))
    if empirical_crps_list:
        out["empirical_crps_model_space"] = float(np.mean(empirical_crps_list))
    if picp_list:
        out["picp"] = float(np.mean(picp_list)) * 100.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--csv", default=None, help="Legacy comma-separated CSV path(s).")
    ap.add_argument("--chem-csv", default=None)
    ap.add_argument("--psd-csv", default=None)
    ap.add_argument("--met-csv", default=None)
    ap.add_argument("--n-chem", type=int, default=None, help="Defaults to the bundle data schema.")
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
    data_schema = bundle["data_schema"]
    preprocessing = bundle["preprocessing"]
    target_cols = data_schema.target_cols
    aux_cols = data_schema.auxiliary_cols
    window_size = bundle["window_size"]
    scaler_target = bundle["scaler_target"]
    scaler_aux = bundle["scaler_aux"]
    aux_mask_channel = preprocessing.aux_mask_channel
    output_transforms = bundle["target_output_transforms"]
    ts_col = data_schema.timestamp_col

    modality_files = None
    if args.chem_csv or args.psd_csv or args.met_csv:
        if args.csv:
            raise ValueError("Use modality CSV flags or --csv, not both")
        modality_files = ModalityFiles(
            chemistry=[v.strip() for v in (args.chem_csv or "").split(",") if v.strip()],
            psd=[v.strip() for v in (args.psd_csv or "").split(",") if v.strip()],
            meteorology=[v.strip() for v in (args.met_csv or "").split(",") if v.strip()],
        )
        frame, _ = load_modality_frame(
            modality_files, ts_col, expected_schema=data_schema
        )
    else:
        if not args.csv:
            raise ValueError("Provide --csv or modality-specific CSV flags")
        csv_paths = [v.strip() for v in args.csv.split(",") if v.strip()]
        frame = load_frame(csv_paths, ts_col, target_cols, aux_cols)
    n = len(frame)
    target_raw = frame[target_cols].to_numpy(dtype=np.float64)
    target_model_space = transform_targets(target_raw, data_schema, preprocessing)
    aux_raw = frame[aux_cols].to_numpy(dtype=np.float64) if aux_cols else np.zeros((n, 0))
    aux_model_space = transform_auxiliary(aux_raw, preprocessing)
    obs_mask_full = ~np.isnan(target_raw)

    heldout_mask = sample_anchor_constrained_heldout_mask(
        obs_mask_full,
        ratio=args.selection_mask_ratio,
        seed=args.selection_val_seed,
        n_chem=data_schema.n_chem if args.n_chem is None else args.n_chem,
    ).astype(bool)

    # Force held-out points to look unobserved to the model, exactly as
    # training did (fixed_mask carved out of the training input mask).
    input_obs_mask = (obs_mask_full & ~heldout_mask).astype(np.float32)

    target_scaled = np.nan_to_num(scaler_target.transform(target_model_space), nan=0.0)
    aux_scaled = scaler_aux.transform(aux_model_space) if aux_cols else aux_model_space
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

    # Two bounded-memory aggregators preserve the established two-stream
    # semantics: clean decoder means drive point metrics, while generative
    # draws drive uncertainty summaries.  The distribution aggregator also
    # computes exact weighted empirical CRPS in de-standardized model space
    # for every naturally observed target, including the fixed held-out set.
    position_weights = trapezoid_position_weights(window_size)
    mean_aggregator = StreamingWindowAggregator(
        total_length=n,
        window_size=window_size,
        n_features=len(target_cols),
        position_weights=position_weights,
        quantiles=(),
    )
    distribution_aggregator = StreamingWindowAggregator(
        total_length=n,
        window_size=window_size,
        n_features=len(target_cols),
        position_weights=position_weights,
        quantiles=(0.025, 0.05, 0.95, 0.975),
        crps_targets=target_model_space,
        crps_mask=obs_mask_full,
    )

    model.eval()
    with torch.no_grad():
        batch_starts_list = range(0, len(starts), args.inference_batch_size)
        for batch_start in tqdm(
            batch_starts_list,
            desc="heldout-eval windows",
            total=n_batches,
            disable=not is_interactive(),
        ):
            batch_starts = starts[
                batch_start:batch_start + args.inference_batch_size
            ]
            masks = np.stack([
                input_obs_mask[start:start + window_size]
                for start in batch_starts
            ])
            xs = np.stack([
                target_scaled[start:start + window_size] * masks[index]
                for index, start in enumerate(batch_starts)
            ])
            conds = np.stack([
                make_condition(
                    aux_scaled[start:start + window_size],
                    aux_observed[start:start + window_size],
                    aux_mask_channel,
                )
                for start in batch_starts
            ])
            x_t = torch.from_numpy(xs).float().to(device)
            cond_t = torch.from_numpy(conds).float().to(device)
            mask_t = torch.from_numpy(masks).float().to(device)
            result = model.compute_uncertainty(
                x_t,
                cond_t,
                mask_t,
                n_samples=max(2, args.n_mc_samples),
                return_samples=True,
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
            for index, start in enumerate(batch_starts):
                mean_aggregator.add(
                    start, pred_mean_model[index][None, :, :]
                )
                distribution_aggregator.add(start, samples_model[:, index])

    mean_agg = mean_aggregator.finish()
    # 0.05/0.95 (90% interval) matches the research repo's PICP definition;
    # 0.025/0.975 is retained for the wider predictions CSV diagnostic.
    dist_agg = distribution_aggregator.finish()
    mean_out, std_out, quantiles_out = summary_to_output_scale(
        mean_agg["mean"], dist_agg["variance"], dist_agg["quantiles"], output_transforms
    )
    q025, q05, q95, q975 = quantiles_out[0.025], quantiles_out[0.05], quantiles_out[0.95], quantiles_out[0.975]

    observed_output = observed_targets_to_output(target_raw, data_schema, preprocessing)

    n_chem = data_schema.n_chem if args.n_chem is None else args.n_chem
    if bundle.get("data_interface") == "modality_files" and n_chem != data_schema.n_chem:
        raise ValueError(
            f"--n-chem={n_chem} conflicts with bundle schema n_chem={data_schema.n_chem}"
        )
    results = {}
    for cat_name, cols in category_indices(target_cols, n_chem):
        y_true_g = observed_output[:, cols]
        y_pred_g = mean_out[:, cols]
        sigma_g = std_out[:, cols]
        q05_g, q95_g = q05[:, cols], q95[:, cols]
        empirical_crps_g = dist_agg["crps"][:, cols]

        # Primary metric: per-feature R^2/MAE/RMSE/SMAPE/CRPS/PICP, negative
        # R^2 clipped to 0, then averaged across the features in this
        # category -- matches the research repo's ablation_heldout_eval.py
        # compute_heldout_metrics exactly. This is what a reference run's
        # own reported "chem_r2"/"psd_r2" means.
        held_mask_g = heldout_mask[:, cols]
        held = _macro_average_metrics(
            y_true_g,
            y_pred_g,
            held_mask_g,
            sigma_cols=sigma_g,
            q_lo_cols=q05_g,
            q_hi_cols=q95_g,
            empirical_crps_model_space_cols=empirical_crps_g,
        )
        for key, value in held.items():
            results[f"{cat_name}_heldout_{key}"] = value

        # Observed-point counterpart (points the model saw as input): a
        # reconstruction-fidelity sanity check, expected close to 1.0.
        # The reference does NOT clip this R^2 to 0 (only the held-out one).
        obs_mask_g = (obs_mask_full[:, cols] & ~heldout_mask[:, cols])
        obs = _macro_average_metrics(
            y_true_g,
            y_pred_g,
            obs_mask_g,
            sigma_cols=sigma_g,
            q_lo_cols=q05_g,
            q_hi_cols=q95_g,
            empirical_crps_model_space_cols=empirical_crps_g,
            clip_r2=False,
        )
        for key, value in obs.items():
            results[f"{cat_name}_observed_{key}"] = value

        # Secondary/diagnostic: pool every feature's held-out points into one
        # R^2/MAE instead of averaging per-feature. This is a DIFFERENT
        # statistic (dominated by whichever features have the largest
        # magnitude/variance) -- kept only so a large gap against the
        # macro-averaged number above is visible, not silently lost.
        y_true_p = y_true_g[held_mask_g]
        y_pred_p = y_pred_g[held_mask_g]
        valid_p = np.isfinite(y_true_p) & np.isfinite(y_pred_p)
        results[f"{cat_name}_heldout_r2_pooled"] = _r2_score(y_true_p[valid_p], y_pred_p[valid_p])
        results[f"{cat_name}_heldout_mae_pooled"] = _mae(y_true_p[valid_p], y_pred_p[valid_p])
        results[f"{cat_name}_heldout_n"] = int(valid_p.sum())

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    if args.predictions_csv:
        # One row per held-out (timestamp, feature), at every level a
        # transform-order bug could hide:
        #   scaled_*    -- z-score standardized space (what the network
        #                  literally inputs/outputs before de-standardizing).
        #   model_*     -- de-standardized, but BEFORE target_output_transform
        #                  (e.g. still log1p space if target_transform=none
        #                  and target_output_transform=log1p).
        #   physical_*  -- fully inverse-transformed; what the reported
        #                  metrics are computed from.
        # mean_agg["mean"] is already de-standardized (mean_chunks holds
        # pred_mean_model = pred_mean_scaled * std + mean), so the z-scored
        # prediction is recovered by inverting that same affine map --
        # aggregation is a weighted average, which commutes with any affine
        # transform, so this is exactly the same value a separate aggregation
        # pass over pred_mean_scaled would give.
        model_pred_mean = mean_agg["mean"]
        scaled_pred_mean = (model_pred_mean - scaler_target.mean_[None, :]) / scaler_target.std_[None, :]
        rows, cols = np.nonzero(heldout_mask)
        families = np.where(cols < n_chem, "chem", "psd")
        frames_out = pd.DataFrame({
            "timestamp": frame.index[rows],
            "feature": [target_cols[c] for c in cols],
            "family": families,
            "scaled_observed": target_scaled[rows, cols],
            "scaled_pred_mean": scaled_pred_mean[rows, cols],
            "model_observed": target_model_space[rows, cols],
            "model_pred_mean": model_pred_mean[rows, cols],
            "model_empirical_crps": dist_agg["crps"][rows, cols],
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
