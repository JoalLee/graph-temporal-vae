#!/usr/bin/env python3
"""Time x feature heatmap of observed / censored / missing cells.

Direct visualization of the three-state observation model: green means a real
measurement, orange means a non-detect (informative, not absent), gray means
truly missing. Useful for spotting whether non-detects cluster in time
(instrument-level dropout, which the current per-value censoring model does
not represent) versus scatter across timestamps (per-value censoring, what
this data actually looks like).

Chem and PSD are plotted separately: chem features are named columns; PSD
uses particle diameter (nm, log scale) so the size-dependent instrument
changeover (SMPS vs. APS) is visible on the axis.

Usage:
    python scripts/plot_observation_heatmap.py outputs/iop_.../imputed_long.csv
    python scripts/plot_observation_heatmap.py outputs/iop_.../imputed_long.csv \
        --modality chem -o chem_states.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

STATE_CODE = {"missing": 0, "observed": 1, "censored": 2}
STATE_COLORS = ["#bdbdbd", "#74c476", "#fd8d3c"]  # missing, observed, censored


def _is_psd_column(name):
    try:
        float(name)
        return True
    except ValueError:
        return False


def load_state_grid(imputed_csv, modality):
    frame = pd.read_csv(
        imputed_csv, usecols=["timestamp", "feature", "observation_state"], low_memory=False
    )
    is_psd = frame["feature"].map(_is_psd_column)
    frame = frame[is_psd if modality == "psd" else ~is_psd]
    if frame.empty:
        raise SystemExit(f"No {modality} columns found in {imputed_csv}")

    pivot = frame.pivot(index="timestamp", columns="feature", values="observation_state")
    if modality == "psd":
        pivot = pivot[sorted(pivot.columns, key=float)]
    codes = pivot.map(STATE_CODE.get).to_numpy(dtype=float)
    return pivot.index, pivot.columns, codes


def plot(imputed_csv, modality, output_path):
    timestamps, features, codes = load_state_grid(imputed_csv, modality)
    cmap = ListedColormap(STATE_COLORS)

    height = max(5, 0.16 * len(features)) if modality == "chem" else 6
    fig, ax = plt.subplots(figsize=(12, height))
    ax.imshow(codes.T, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="none")

    n_ticks = min(8, len(timestamps))
    tick_idx = np.linspace(0, len(timestamps) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(timestamps[i])[:10] for i in tick_idx], rotation=30, ha="right")

    if modality == "psd":
        ax.set_ylabel("particle diameter (nm)")
        n_y = min(12, len(features))
        y_idx = np.linspace(0, len(features) - 1, n_y, dtype=int)
        ax.set_yticks(y_idx)
        ax.set_yticklabels([f"{float(features[i]):.0f}" for i in y_idx])
    else:
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=6)
        ax.set_ylabel("chemistry species")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLORS]
    ax.legend(handles, ["missing", "observed", "censored"], loc="upper right",
             bbox_to_anchor=(1.15, 1.0), fontsize=8)
    ax.set_title(f"Observation state — {modality}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("imputed_csv")
    parser.add_argument("--modality", choices=["chem", "psd"], default="chem")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)
    output = args.output or Path(args.imputed_csv).with_name(f"observation_heatmap_{args.modality}.png")
    plot(args.imputed_csv, args.modality, output)


if __name__ == "__main__":
    sys.exit(main())
