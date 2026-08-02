"""Held-out evaluation for a graph-temporal-vae checkpoint bundle.

Holds out a fixed fraction of the genuinely observed points, forces them to
look unobserved to the model, and scores the predictions against the values
that were hidden. `impute()` alone cannot produce this number: it always
writes real observations straight through to the output, so the model's own
prediction for an observed cell is discarded and never compared to anything.
This script is what answers "how accurate is this checkpoint, per feature?".

Two things the reported numbers depend on, both worth stating in any writeup:

* The held-out set is generated *for this evaluation*, not replayed from the
  training loop. The default anchor-constrained protocol holds out a share of
  every feature's observed points so each feature has enough scored points for
  a per-feature R^2; the training loop's own `block` protocol draws one block
  per modality per window, which is right for augmentation but too sparse to
  score features individually. Pass `--selection-mask-mode block` to use it.
* Below-detection-limit cells may be selected for the held-out set, but they
  are never mixed into exact-value metrics. They are scored separately by an
  interval probability/NLL and the reported prediction is conditioned on
  ``y <= MDL``.

Usage:
    python examples/heldout_eval.py \\
        --bundle outputs/iop_.../model.pt \\
        --chem-csv data/iop_clean/chem.csv \\
        --psd-csv data/iop_clean/psd.csv \\
        --met-csv data/iop_clean/met.csv \\
        -o heldout_metrics.json --predictions-csv heldout_predictions.csv
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
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_temporal_vae.censoring import (
    STATE_CENSORED,
    STATE_MISSING,
    STATE_OBSERVED,
    CensoringConfig,
    apply_input_fill,
    build_state_matrix,
)
from graph_temporal_vae.contracts import ModalityFiles
from graph_temporal_vae.data import (
    compute_window_starts,
    extract_censor_marker_mask,
    load_frame,
    load_modality_frame,
    make_condition,
    sample_anchor_constrained_heldout_mask,
    sample_block_heldout_mask_to_ratio,
)
from graph_temporal_vae.infer import (
    _bundle_censor_thresholds,
    load_bundle,
    summary_to_output_scale,
    trapezoid_position_weights,
    truncate_below_limit,
)
from graph_temporal_vae.train import load_external_heldout_mask
from graph_temporal_vae.preprocessing import (
    inverse_targets,
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
    # Undefined, not zero, in two cases sklearn treats differently: an empty
    # selection (it raises) and a constant truth vector (it returns 0.0, which
    # would read as "explains nothing" for a genuinely flat feature).
    if len(y_true) == 0 or np.var(y_true) <= 0:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _mae(y_true, y_pred):
    if len(y_true) == 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred))


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
    ap.add_argument(
        "--selection-mask-mode", choices=["anchor_constrained", "block"],
        default="anchor_constrained",
        help="Held-out protocol for THIS evaluation. 'anchor_constrained' (default) holds "
             "out a fixed fraction of every feature's observed points, bounded by observed "
             "anchors, which is what gives each feature enough scored points for a per-feature "
             "R2. 'block' replays the training-time augmentation protocol instead: it draws "
             "one block per modality and yields far fewer scored points.",
    )
    ap.add_argument("--selection-mask-ratio", type=float, default=None,
                    help="Defaults to the ratio stored in the checkpoint.")
    ap.add_argument("--selection-val-seed", type=int, default=None,
                    help="Defaults to the seed stored in the checkpoint.")
    ap.add_argument(
        "--selection-mask-path", default=None,
        help="Optional external full-timeline .npy held-out mask. When supplied, "
             "do not resample a mask; validate and score this exact matrix.",
    )
    ap.add_argument(
        "--selection-mask-columns-path", default=None,
        help="Optional one-column target_col CSV used to validate external mask order.",
    )
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
        # New bundles store the research-compatible wind vector schema. Keep
        # older bundles with raw WS/WD readable, while allowing a canonical
        # bundle to evaluate directly from a raw legacy CSV.
        canonicalize_wind = (
            {"wind_u", "wind_v"}.issubset(aux_cols)
            and not {"WS", "WD"}.issubset(aux_cols)
        )
        frame = load_frame(
            csv_paths,
            ts_col,
            target_cols,
            aux_cols,
            canonicalize_wind=canonicalize_wind,
        )
    n = len(frame)
    target_raw = frame[target_cols].to_numpy(dtype=np.float64)
    censoring = bundle.get("censoring") or CensoringConfig()
    if isinstance(censoring, dict):
        censoring = CensoringConfig.from_dict(censoring)
    marker_mask = extract_censor_marker_mask(frame, target_cols)
    if data_schema.n_chem < marker_mask.shape[1]:
        marker_mask[:, data_schema.n_chem:] = False
    state_full = build_state_matrix(
        target_raw, data_schema, censoring, marker_mask=marker_mask
    )
    censor_mask_full = state_full == STATE_CENSORED
    target_input_raw = apply_input_fill(target_raw, state_full, data_schema, censoring)
    target_model_space = transform_targets(target_input_raw, data_schema, preprocessing)
    aux_raw = frame[aux_cols].to_numpy(dtype=np.float64) if aux_cols else np.zeros((n, 0))
    aux_model_space = transform_auxiliary(aux_raw, preprocessing)
    obs_mask_full = state_full == STATE_OBSERVED
    known_mask_full = obs_mask_full | censor_mask_full
    eligible_mask_full = state_full != STATE_MISSING
    censor_threshold_scaled = _bundle_censor_thresholds(bundle, data_schema)
    censor_threshold_model = (
        censor_threshold_scaled * scaler_target.std_ + scaler_target.mean_
    )

    training_config = bundle.get("training_config", {})
    mask_mode = args.selection_mask_mode
    ratio = args.selection_mask_ratio
    if ratio is None:
        ratio = training_config.get("selection_mask_ratio", 0.1)
    seed = args.selection_val_seed
    if seed is None:
        seed = training_config.get("selection_val_seed", 42)
    n_chem = data_schema.n_chem if args.n_chem is None else args.n_chem

    external_mask_diagnostics = None
    if args.selection_mask_columns_path and not args.selection_mask_path:
        raise ValueError(
            "--selection-mask-columns-path requires --selection-mask-path"
        )

    # An external mask is a benchmark artifact: validate its shape and target
    # order, but do not resample or intersect it here. Natural missingness is
    # excluded from exact/interval metric validity below; censored overlap is
    # retained for interval scoring and reported separately.
    if args.selection_mask_path:
        heldout_mask, external_mask_diagnostics = load_external_heldout_mask(
            args.selection_mask_path,
            args.selection_mask_columns_path,
            expected_rows=n,
            target_cols=target_cols,
            observed_mask=obs_mask_full,
            censored_mask=censor_mask_full,
        )
        mask_mode = "external"
    # Reproduce the protocol the checkpoint was trained under when no external
    # matrix is supplied; a mismatch here silently scores a different set of
    # cells than the run's own validation metric.
    elif mask_mode == "block":
        heldout_mask = sample_block_heldout_mask_to_ratio(
            eligible_mask_full,
            {
                "mode": "block",
                "target_ratio": ratio,
                "mean_duration": training_config.get("dynamic_mask_mean_duration", 48.0),
                "std_duration": training_config.get("dynamic_mask_std_duration", 24.0),
                "min_duration": training_config.get("dynamic_mask_min_duration", 3),
                "max_duration": training_config.get("dynamic_mask_max_duration", 168),
                "chem_blocks": training_config.get("dynamic_mask_chem_blocks", 1),
                "psd_blocks": training_config.get("dynamic_mask_psd_blocks", 1),
                "duration_source": training_config.get("dynamic_mask_duration_source", "parametric"),
                "n_chem": n_chem,
                "ensure_nonempty": True,
            },
            seed=seed,
        ).astype(bool)
    else:
        heldout_mask = sample_anchor_constrained_heldout_mask(
            eligible_mask_full, ratio=ratio, seed=seed, n_chem=n_chem,
            mean_duration=training_config.get("dynamic_mask_mean_duration", 48.0),
            std_duration=training_config.get("dynamic_mask_std_duration", 24.0),
            min_duration=training_config.get("dynamic_mask_min_duration", 3),
            max_duration=training_config.get("dynamic_mask_max_duration", 168),
            duration_source=training_config.get("dynamic_mask_duration_source", "parametric"),
        ).astype(bool)
    protocol_text = (
        f"[heldout_eval] selection protocol: mode={mask_mode} "
        f"ratio={ratio} seed={seed}"
    )
    if external_mask_diagnostics is not None:
        protocol_text += (
            f", source={external_mask_diagnostics['source']}"
            f", requested={external_mask_diagnostics['requested_cells']}"
            f", observed={external_mask_diagnostics['observed_cells']}"
            f", natural_missing_overlap={external_mask_diagnostics['natural_missing_overlap_cells']}"
            f", censored_overlap={external_mask_diagnostics['censored_overlap_cells']}"
        )
    if censoring.active:
        protocol_text += (
            f", {int(censor_mask_full.sum())} censored cells; selected censored cells "
            "use interval scoring"
        )
    print(protocol_text + f", dist={bundle.get('uncertainty_dist_type', 'gaussian')}" )

    # Force held-out points to look unobserved to the model, exactly as
    # training did (fixed_mask carved out of the training input mask).
    # Non-detects stay visible: training keeps them in the encoder input.
    input_obs_mask = (known_mask_full & ~heldout_mask).astype(np.float32)

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
        f"{heldout_mask.sum()}/{eligible_mask_full.sum()} eligible cells held out "
        f"({100 * heldout_mask.sum() / eligible_mask_full.sum():.2f}%), "
        f"{args.n_mc_samples} MC samples, stride={stride}, "
        f"dist={bundle.get('uncertainty_dist_type', 'gaussian')}, device={device}"
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
                dist_type=bundle.get("uncertainty_dist_type", "gaussian"),
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
    heldout_observed_mask = heldout_mask & obs_mask_full
    heldout_censored_mask = heldout_mask & censor_mask_full
    # Keep the unconstrained prediction for the diagnostic below, but use the
    # same conditional/truncated distribution as ``impute()`` for censored HO
    # output. This enforces the known interval without treating the marker's
    # numeric payload as an exact target or hard-clamping every prediction.
    constrained_mean_model, constrained_variance_model, p_below_limit = truncate_below_limit(
        mean_agg["mean"],
        dist_agg["variance"],
        censor_threshold_model,
        heldout_censored_mask,
    )
    mean_eval_model = np.where(
        heldout_censored_mask, constrained_mean_model, mean_agg["mean"]
    )
    variance_eval_model = np.where(
        heldout_censored_mask, constrained_variance_model, dist_agg["variance"]
    )
    mean_out_unconstrained, _, _ = summary_to_output_scale(
        mean_agg["mean"], dist_agg["variance"], dist_agg["quantiles"], output_transforms
    )
    mean_out, std_out, quantiles_out = summary_to_output_scale(
        mean_eval_model, variance_eval_model, dist_agg["quantiles"], output_transforms
    )
    q025, q05, q95, q975 = quantiles_out[0.025], quantiles_out[0.05], quantiles_out[0.95], quantiles_out[0.975]
    limit_output = inverse_targets(
        censor_threshold_model[None, :], data_schema, preprocessing
    )[0]
    finite_limit = np.isfinite(limit_output)[None, :]
    for array in (q025, q05, q95, q975):
        np.copyto(
            array,
            np.minimum(array, np.broadcast_to(limit_output[None, :], array.shape)),
            where=heldout_censored_mask & finite_limit,
        )

    observed_output = observed_targets_to_output(target_raw, data_schema, preprocessing)

    n_chem = data_schema.n_chem if args.n_chem is None else args.n_chem
    if bundle.get("data_interface") == "modality_files" and n_chem != data_schema.n_chem:
        raise ValueError(
            f"--n-chem={n_chem} conflicts with bundle schema n_chem={data_schema.n_chem}"
        )
    results = {}
    if external_mask_diagnostics is not None:
        results["selection_mask_protocol"] = {
            "mode": "external",
            "ratio_argument": ratio,
            "seed_argument": seed,
            **external_mask_diagnostics,
        }
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
        held_mask_g = heldout_observed_mask[:, cols]
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

        # Censored HO cells have no exact scalar target. Report the model's
        # unconstrained Gaussian-approximate interval probability/NLL and a
        # raw-limit violation diagnostic, while the public prediction above is
        # the truncated conditional mean used for a final imputation result.
        censored_g = heldout_censored_mask[:, cols]
        censored_valid = (
            censored_g
            & np.isfinite(p_below_limit[:, cols])
            & np.isfinite(limit_output[None, :][:, cols])
        )
        censored_n = int(censored_valid.sum())
        results[f"{cat_name}_censored_ho_n"] = censored_n
        if censored_n:
            p_below = p_below_limit[:, cols][censored_valid]
            raw_mean = mean_out_unconstrained[:, cols][censored_valid]
            constrained_mean = mean_out[:, cols][censored_valid]
            limits = np.broadcast_to(
                limit_output[cols][None, :], censored_valid.shape
            )[censored_valid]
            results[f"{cat_name}_censored_ho_p_below_mdl"] = float(np.mean(p_below))
            results[f"{cat_name}_censored_ho_gaussian_nll"] = float(
                np.mean(-np.log(np.clip(p_below, 1e-12, 1.0)))
            )
            results[f"{cat_name}_censored_ho_raw_mean_above_mdl_rate"] = float(
                np.mean(raw_mean > limits)
            )
            results[f"{cat_name}_censored_ho_constrained_mean_above_mdl_rate"] = float(
                np.mean(constrained_mean > limits)
            )

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
        model_pred_mean = mean_eval_model
        scaled_pred_mean = (model_pred_mean - scaler_target.mean_[None, :]) / scaler_target.std_[None, :]
        rows, cols = np.nonzero(heldout_mask)
        families = np.where(cols < n_chem, "chem", "psd")
        frames_out = pd.DataFrame({
            "timestamp": frame.index[rows],
            "feature": [target_cols[c] for c in cols],
            "family": families,
            "observation_state": np.where(
                heldout_censored_mask[rows, cols], "censored", "observed"
            ),
            "detection_limit": limit_output[cols],
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
            "p_below_mdl": p_below_limit[rows, cols],
        })
        frames_out.to_csv(args.predictions_csv, index=False)
        print(f"wrote {len(frames_out)} held-out predictions to {args.predictions_csv}")


if __name__ == "__main__":
    main()
