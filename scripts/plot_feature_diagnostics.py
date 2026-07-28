#!/usr/bin/env python3
"""Per-feature breakdown of observation state and predictive uncertainty.

With 74 chem species and 230 PSD bins at very different missingness and
censoring rates, an aggregate metric hides which features are actually hard.
This reads the bundle's ``censoring_report`` (observed/censored/missing
fractions per column, computed once at training time) together with
``imputed_long.csv`` (per-cell predictive std from the imputation run) and
plots them side by side, sorted by how little real signal a feature has.

Usage:
    python scripts/plot_feature_diagnostics.py outputs/iop_.../model.pt \
        outputs/iop_.../imputed_long.csv -o diagnostics.png
    python scripts/plot_feature_diagnostics.py outputs/iop_.../model.pt \
        outputs/iop_.../imputed_long.csv --top 30
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


def load_state_fractions(bundle_path):
    """Observed/censored/missing fraction per target column, from training time."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    schema = bundle["data_schema"]
    per_column = bundle.get("censoring_report", {}).get("per_column", {})
    rows = []
    for column in schema["chemistry_cols"] + schema["psd_cols"]:
        censored = per_column.get(column, {}).get("censored_fraction", 0.0)
        missing = per_column.get(column, {}).get("missing_fraction", 0.0)
        rows.append({
            "feature": column,
            "censored_fraction": censored,
            "missing_fraction": missing,
            "observed_fraction": max(0.0, 1.0 - censored - missing),
            "modality": "chem" if column in schema["chemistry_cols"] else "psd",
        })
    return pd.DataFrame(rows).set_index("feature")


def load_predictive_std(imputed_csv):
    """Mean predictive std per feature, over cells that were actually imputed."""
    frame = pd.read_csv(imputed_csv, usecols=["feature", "is_imputed", "imputed_std"],
                        low_memory=False)
    imputed = frame[frame["is_imputed"].astype(bool)]
    if imputed.empty:
        return pd.Series(dtype=float, name="mean_predictive_std")
    return imputed.groupby("feature")["imputed_std"].mean().rename("mean_predictive_std")


def plot(bundle_path, imputed_csv, output_path, top_n):
    stats = load_state_fractions(bundle_path)
    std = load_predictive_std(imputed_csv)
    stats = stats.join(std, how="left")

    # Least-observed-signal first: this is "hardest to reconstruct", the
    # question a single aggregate metric can't answer.
    stats["signal_fraction"] = stats["observed_fraction"]
    ranked = stats.sort_values("signal_fraction").head(top_n)

    fig, (ax_frac, ax_std) = plt.subplots(
        1, 2, figsize=(12, max(4, 0.28 * len(ranked))), sharey=True
    )
    y = np.arange(len(ranked))
    colors = {"chem": "tab:blue", "psd": "tab:orange"}

    left = np.zeros(len(ranked))
    for state, color in (("observed_fraction", "#c7e9c0"),
                        ("censored_fraction", "#fdae6b"),
                        ("missing_fraction", "#bdbdbd")):
        ax_frac.barh(y, ranked[state], left=left, color=color, label=state.replace("_fraction", ""))
        left += ranked[state].to_numpy()
    ax_frac.set_yticks(y)
    ax_frac.set_yticklabels(ranked.index)
    ax_frac.set_xlabel("fraction of cells")
    ax_frac.set_title("Observation state")
    ax_frac.legend(fontsize=8, loc="lower right")
    ax_frac.invert_yaxis()

    bar_colors = [colors[m] for m in ranked["modality"]]
    ax_std.barh(y, ranked["mean_predictive_std"].fillna(0.0), color=bar_colors)
    ax_std.set_xlabel("mean predictive std (imputed cells, output units)")
    ax_std.set_title("Uncertainty when imputed")

    fig.suptitle(
        f"Per-feature diagnostics — {top_n} features with the least observed signal"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")

    saturated = ranked[ranked["signal_fraction"] < 0.1]
    if len(saturated):
        print(
            f"{len(saturated)} feature(s) have <10% real observations: "
            f"{list(saturated.index)}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle")
    parser.add_argument("imputed_csv")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--top", type=int, default=40,
                        help="Number of least-observed features to show.")
    args = parser.parse_args(argv)
    output = args.output or Path(args.imputed_csv).with_name("feature_diagnostics.png")
    plot(args.bundle, args.imputed_csv, output, args.top)


if __name__ == "__main__":
    sys.exit(main())
