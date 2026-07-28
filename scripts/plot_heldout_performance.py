#!/usr/bin/env python3
"""Per-feature held-out accuracy, ranked and set against data availability.

Reads the per-cell predictions written by ``examples/heldout_eval.py
--predictions-csv`` (one row per held-out timestamp/feature, with both the
hidden truth and the model's prediction) and answers the question a single
aggregate ``val_ho_mse`` cannot: which species does this checkpoint actually
reconstruct well, and does poor accuracy track how little real data a feature
has?

R^2 is computed in *model space* (post-transform, pre-scaling), because the
physical space for chem spans several orders of magnitude across species and a
physical-space R^2 would be dominated by the largest few.

Usage:
    python scripts/plot_heldout_performance.py outputs/iop_.../heldout_predictions.csv
    python scripts/plot_heldout_performance.py outputs/iop_.../heldout_predictions.csv \\
        --bundle outputs/iop_.../model.pt --family chem
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402


def per_feature_metrics(predictions, min_points):
    """R^2 / MAE / PICP per feature, in model space."""
    rows = []
    for feature, group in predictions.groupby("feature"):
        truth = group["model_observed"].to_numpy()
        pred = group["model_pred_mean"].to_numpy()
        finite = np.isfinite(truth) & np.isfinite(pred)
        truth, pred = truth[finite], pred[finite]
        if len(truth) < min_points or np.var(truth) <= 0:
            continue
        covered = (
            (group["physical_observed"] >= group["physical_q025"])
            & (group["physical_observed"] <= group["physical_q975"])
        )
        rows.append({
            "feature": feature,
            "family": group["family"].iloc[0],
            "n": len(truth),
            "r2": r2_score(truth, pred),
            "mae": mean_absolute_error(truth, pred),
            "picp": float(covered.mean()) * 100.0,
        })
    return pd.DataFrame(rows).set_index("feature")


def load_availability(bundle_path):
    """Censored/missing fractions per column, recorded at training time."""
    import torch

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    per_column = bundle.get("censoring_report", {}).get("per_column", {})
    if not per_column:
        return None
    return pd.DataFrame([
        {
            "feature": column,
            "unavailable_fraction": stats["censored_fraction"] + stats["missing_fraction"],
        }
        for column, stats in per_column.items()
    ]).set_index("feature")["unavailable_fraction"]


def plot(predictions_csv, bundle_path, family, top_n, min_points, output_path):
    predictions = pd.read_csv(predictions_csv, low_memory=False)
    if family != "all":
        predictions = predictions[predictions["family"] == family]
        if predictions.empty:
            raise SystemExit(f"No rows with family={family!r} in {predictions_csv}")

    metrics = per_feature_metrics(predictions, min_points)
    if metrics.empty:
        raise SystemExit(
            f"No feature had >= {min_points} held-out points with non-constant truth"
        )
    availability = load_availability(bundle_path) if bundle_path else None

    ranked = metrics.sort_values("r2", ascending=False)
    shown = ranked.head(top_n) if len(ranked) > top_n else ranked

    has_scatter = availability is not None
    fig, axes = plt.subplots(
        1, 2 if has_scatter else 1,
        figsize=(14 if has_scatter else 8, max(4, 0.26 * len(shown))),
        squeeze=False,
    )

    ax = axes[0][0]
    colors = ["tab:blue" if f == "chem" else "tab:orange" for f in shown["family"]]
    y = np.arange(len(shown))
    ax.barh(y, shown["r2"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(shown.index, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("held-out $R^2$ (model space)")
    ax.set_title(f"Per-feature held-out accuracy ({family})")
    ax.grid(axis="x", alpha=0.3)

    if has_scatter:
        ax = axes[0][1]
        joined = metrics.join(availability, how="inner")
        for fam, color in (("chem", "tab:blue"), ("psd", "tab:orange")):
            subset = joined[joined["family"] == fam]
            if len(subset):
                ax.scatter(subset["unavailable_fraction"] * 100, subset["r2"],
                           s=18, alpha=0.7, color=color, label=fam)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("censored + missing (% of cells)")
        ax.set_ylabel("held-out $R^2$")
        ax.set_title("Accuracy vs. data availability")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")

    negative = ranked[ranked["r2"] < 0]
    print(
        f"scored {len(ranked)} features | median R2 {ranked['r2'].median():.3f} | "
        f"median PICP {ranked['picp'].median():.1f}% (target 95%)"
    )
    if len(negative):
        # Negative R^2 means the prediction is worse than that feature's own
        # held-out mean, i.e. the model adds nothing for it.
        print(f"{len(negative)} feature(s) with R2 < 0: {list(negative.index[:15])}")

    metrics_path = Path(output_path).with_suffix(".csv")
    ranked.to_csv(metrics_path)
    print(f"wrote {metrics_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions_csv")
    parser.add_argument("--bundle", default=None,
                        help="Checkpoint, to overlay each feature's censored+missing fraction.")
    parser.add_argument("--family", choices=["all", "chem", "psd"], default="chem")
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--min-points", type=int, default=10,
                        help="Skip features with fewer held-out points than this.")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)
    output = args.output or Path(args.predictions_csv).with_name(
        f"heldout_performance_{args.family}.png"
    )
    plot(args.predictions_csv, args.bundle, args.family, args.top, args.min_points, output)


if __name__ == "__main__":
    sys.exit(main())
