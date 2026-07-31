#!/usr/bin/env python3
"""Compare stage-1 (pre-refit) vs final (post-refit) predictions on cells
neither stage ever trained on.

Requires a bundle trained with ``full_data_refit_epochs > 0`` on data
produced by ``prepare_frozen_eval.py`` -- that gives you both
``<model>_stage1.pt`` (saved right after stage-1 selection, before refit
touches the weights) and the final ``model.pt``, plus a truth table of cells
excluded from the CSVs the whole run trained on.

This is the one comparison that isn't confounded: Global HO cells get
directly optimized during refit (so of course a post-refit checkpoint scores
"better" on them -- they're training targets by then), and heldout_eval.py's
own masks only hide cells at inference time, not from the loss. Cells frozen
out via prepare_frozen_eval.py are NaN in the CSVs both stages ever saw, so
neither one had a gradient from them.

Usage:
    python scripts/score_stage_drift.py \\
        --stage1-bundle outputs/<run>/model_stage1.pt \\
        --final-bundle outputs/<run>/model.pt \\
        --chem-csv data/iop_clean/frozen_eval_chem.csv \\
        --psd-csv data/iop_clean/frozen_eval_psd.csv \\
        --met-csv data/iop_clean/met.csv \\
        --truth-csv data/iop_clean/frozen_eval_truth.csv \\
        -o frozen_eval_drift.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from graph_temporal_vae.api import impute_multimodal
from graph_temporal_vae.contracts import InferenceConfig
from graph_temporal_vae.infer import load_bundle


def score_checkpoint(bundle_path, chem_csv, psd_csv, met_csv, truth, n_mc_samples, stride):
    bundle = load_bundle(Path(bundle_path))
    result = impute_multimodal(
        chemistry=chem_csv, psd=psd_csv, meteorology=met_csv,
        bundle=bundle,
        config=InferenceConfig(n_mc_samples=n_mc_samples, stride=stride),
    )
    result = result.rename(columns={"timestamp": "time"}) if "timestamp" in result.columns else result
    time_col = "timestamp" if "timestamp" in result.columns else "time"
    merged = truth.merge(
        result[[time_col, "feature", "imputed_mean"]],
        left_on=["timestamp", "feature"], right_on=[time_col, "feature"], how="left",
    )
    missing = merged["imputed_mean"].isna().sum()
    if missing:
        print(f"    warning: {missing} frozen cells had no prediction (outside inference window?)")
    return merged.dropna(subset=["imputed_mean"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-bundle", required=True)
    parser.add_argument("--final-bundle", required=True)
    parser.add_argument("--chem-csv", required=True)
    parser.add_argument("--psd-csv", required=True)
    parser.add_argument("--met-csv", required=True)
    parser.add_argument("--truth-csv", required=True)
    parser.add_argument("--n-mc-samples", type=int, default=50)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)

    truth = pd.read_csv(args.truth_csv, parse_dates=["timestamp"])

    print("scoring stage-1 (pre-refit) checkpoint...")
    stage1_scored = score_checkpoint(
        args.stage1_bundle, args.chem_csv, args.psd_csv, args.met_csv,
        truth, args.n_mc_samples, args.stride,
    )
    print("scoring final (post-refit) checkpoint...")
    final_scored = score_checkpoint(
        args.final_bundle, args.chem_csv, args.psd_csv, args.met_csv,
        truth, args.n_mc_samples, args.stride,
    )

    merged = stage1_scored.merge(
        final_scored[["timestamp", "feature", "imputed_mean"]],
        on=["timestamp", "feature"], suffixes=("_stage1", "_final"),
    )

    report = {}
    for family in ["chem", "psd", "overall"]:
        sub = merged if family == "overall" else merged[merged["family"] == family]
        if len(sub) == 0:
            continue
        true_vals = sub["true_value"].to_numpy()
        stage1_pred = sub["imputed_mean_stage1"].to_numpy()
        final_pred = sub["imputed_mean_final"].to_numpy()
        drift = final_pred - stage1_pred
        report[family] = {
            "n": int(len(sub)),
            "stage1_r2": float(r2_score(true_vals, stage1_pred)),
            "final_r2": float(r2_score(true_vals, final_pred)),
            "stage1_mae": float(mean_absolute_error(true_vals, stage1_pred)),
            "final_mae": float(mean_absolute_error(true_vals, final_pred)),
            "mean_abs_drift": float(np.abs(drift).mean()),
            "median_abs_drift": float(np.median(np.abs(drift))),
            "max_abs_drift": float(np.abs(drift).max()),
            # Drift relative to the actual value scale, so it's comparable
            # across chem (small numbers) and psd (large numbers).
            "mean_abs_drift_pct_of_true_std": float(
                np.abs(drift).mean() / max(true_vals.std(), 1e-9) * 100
            ),
        }
        print(
            f"{family:8s} n={report[family]['n']:6d}  "
            f"R2 stage1={report[family]['stage1_r2']:.4f} -> final={report[family]['final_r2']:.4f}  "
            f"MAE stage1={report[family]['stage1_mae']:.4f} -> final={report[family]['final_mae']:.4f}  "
            f"mean|drift|={report[family]['mean_abs_drift']:.4f} "
            f"({report[family]['mean_abs_drift_pct_of_true_std']:.1f}% of true std)"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
