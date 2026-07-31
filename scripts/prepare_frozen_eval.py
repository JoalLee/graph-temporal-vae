#!/usr/bin/env python3
"""Carve out a "frozen eval" set: cells excluded from BOTH training stages.

Why this needs to exist: neither of the library's built-in held-out
mechanisms is actually blind to both stage-1 training AND the stage-2
full-data refit.

* Global HO (``shared_full_heldout_mask``) is excluded from stage-1's
  input/loss -- but stage 2 explicitly *restores* it before continuing
  training. Scoring a post-refit checkpoint against Global HO cells measures
  something close to "training accuracy after refit", not generalization.
* ``heldout_eval.py``'s masks (``anchor_constrained`` or ``block``) are
  test-time-only: they hide cells from the encoder for one inference pass,
  but the model's weights were shaped by seeing those same values during
  training (nothing excluded them from the loss to begin with, unless the
  mask happens to reconstruct Global HO exactly).

The only way to get a cell set genuinely untouched by both stages is to
remove it from the CSV before the pipeline ever sees it: a NaN cell is
ineligible for every masking mechanism in the codebase, in both stages, by
construction. This script does exactly that -- it writes modified copies of
chem.csv/psd.csv with a small, real-gap-shaped sample of cells blanked out,
plus a truth table recording what was actually there.

Usage:
    python scripts/prepare_frozen_eval.py \\
        --chem-csv data/iop_clean/chem.csv \\
        --psd-csv data/iop_clean/psd.csv \\
        --train-config examples/iop_train_config_realnvp_fb03.json \\
        --ratio 0.05 \\
        --seed 999 \\
        -o data/iop_clean/frozen_eval
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from graph_temporal_vae.data import sample_anchor_constrained_heldout_mask


def eligible_chem_mask(chem_df, thresholds):
    """Observed and not a non-detect (detect='zero' convention)."""
    values = chem_df.to_numpy(dtype=np.float64)
    observed = ~np.isnan(values)
    has_threshold = np.array([col in thresholds for col in chem_df.columns])
    censored = observed & has_threshold[None, :] & (values == 0.0)
    return observed & ~censored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chem-csv", required=True)
    parser.add_argument("--psd-csv", required=True)
    parser.add_argument("--train-config", required=True,
                        help="Read censoring thresholds and duration stats from here.")
    parser.add_argument("--ratio", type=float, default=0.05,
                        help="Fraction of eligible cells per family to freeze out.")
    parser.add_argument("--seed", type=int, default=999,
                        help="Distinct from selection_val_seed so this doesn't "
                             "collide with Global HO's own cell selection.")
    parser.add_argument("-o", "--output-prefix", required=True,
                        help="Writes <prefix>_chem.csv, <prefix>_psd.csv, <prefix>_truth.csv")
    args = parser.parse_args(argv)

    train_config = json.loads(Path(args.train_config).read_text())
    thresholds = train_config.get("censoring", {}).get("thresholds", {})
    duration_kwargs = dict(
        mean_duration=train_config.get("dynamic_mask_mean_duration", 48.0),
        std_duration=train_config.get("dynamic_mask_std_duration", 24.0),
        min_duration=train_config.get("dynamic_mask_min_duration", 3),
        max_duration=train_config.get("dynamic_mask_max_duration", 168),
        duration_source=train_config.get("dynamic_mask_duration_source", "parametric"),
    )

    chem_df = pd.read_csv(args.chem_csv, parse_dates=["time"]).set_index("time")
    psd_df = pd.read_csv(args.psd_csv, parse_dates=["time"]).set_index("time")

    chem_eligible = eligible_chem_mask(chem_df, thresholds)
    psd_eligible = (~psd_df.isna()).to_numpy()

    chem_frozen_mask = sample_anchor_constrained_heldout_mask(
        chem_eligible, ratio=args.ratio, seed=args.seed,
        n_chem=chem_eligible.shape[1], **duration_kwargs,
    )
    psd_frozen_mask = sample_anchor_constrained_heldout_mask(
        psd_eligible, ratio=args.ratio, seed=args.seed + 1,
        n_chem=0, **duration_kwargs,
    )

    truth_rows = []
    chem_out = chem_df.copy()
    for col_idx, col in enumerate(chem_df.columns):
        rows = np.flatnonzero(chem_frozen_mask[:, col_idx])
        for row_idx in rows:
            truth_rows.append({
                "family": "chem",
                "timestamp": chem_df.index[row_idx],
                "feature": col,
                "true_value": chem_df.iloc[row_idx, col_idx],
            })
        chem_out.iloc[rows, col_idx] = np.nan

    psd_out = psd_df.copy()
    for col_idx, col in enumerate(psd_df.columns):
        rows = np.flatnonzero(psd_frozen_mask[:, col_idx])
        for row_idx in rows:
            truth_rows.append({
                "family": "psd",
                "timestamp": psd_df.index[row_idx],
                "feature": col,
                "true_value": psd_df.iloc[row_idx, col_idx],
            })
        psd_out.iloc[rows, col_idx] = np.nan

    truth = pd.DataFrame(truth_rows)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    chem_out.reset_index().to_csv(f"{out_prefix}_chem.csv", index=False)
    psd_out.reset_index().to_csv(f"{out_prefix}_psd.csv", index=False)
    truth.to_csv(f"{out_prefix}_truth.csv", index=False)

    print(f"chem: {int(chem_frozen_mask.sum())} cells frozen out of "
          f"{int(chem_eligible.sum())} eligible ({args.ratio:.1%} target)")
    print(f"psd:  {int(psd_frozen_mask.sum())} cells frozen out of "
          f"{int(psd_eligible.sum())} eligible ({args.ratio:.1%} target)")
    print(f"wrote {out_prefix}_chem.csv, {out_prefix}_psd.csv, {out_prefix}_truth.csv")
    print(
        "\nTrain on the *_chem.csv/*_psd.csv outputs (with full_data_refit_epochs > 0 "
        "so a _stage1.pt checkpoint is saved), then compare stage1.pt vs the final "
        "model.pt against *_truth.csv with scripts/score_stage_drift.py."
    )


if __name__ == "__main__":
    sys.exit(main())
