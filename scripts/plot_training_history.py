#!/usr/bin/env python3
"""Plot the per-epoch training history written next to a checkpoint bundle.

Every ``graph-temporal-vae train`` run writes ``<bundle>_history.csv`` as it
goes (one row per epoch: loss, held-out metrics, LR, KL beta). This script
turns that CSV into the four curves you actually need to read a run: is it
overfitting, did the schedules behave, is the censored term improving or
being traded off against the observed one.

Usage:
    python scripts/plot_training_history.py outputs/iop_.../model_history.csv
    python scripts/plot_training_history.py outputs/iop_.../model_history.csv -o run.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot(history_csv, output_path=None):
    df = pd.read_csv(history_csv)
    best_epoch = df.loc[df["is_best"] == 1, "epoch"].max()

    has_calibration = "val_ho_z2" in df.columns and df["val_ho_z2"].notna().any()
    n_rows = 3 if has_calibration else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.5 * n_rows), sharex=True)

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="train_loss", color="tab:blue")
    ax.plot(df["epoch"], df["val_ho_nll"], label="val_ho_nll", color="tab:orange")
    if df["val_censored_nll"].notna().any():
        ax.plot(df["epoch"], df["val_censored_nll"], label="val_censored_nll",
                 color="tab:red", linestyle="--")
    ax.set_ylabel("loss / NLL")
    ax.set_title("Training vs. held-out loss")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["val_ho_mse"], color="tab:green")
    ax.set_ylabel("val_ho_mse")
    ax.set_title("Held-out MSE (scaled space)")

    ax = axes[1, 0]
    ax.plot(df["epoch"], df["lr"], color="tab:purple")
    ax.set_yscale("log")
    ax.set_ylabel("learning rate")
    ax.set_xlabel("epoch")
    ax.set_title("LR schedule")

    ax = axes[1, 1]
    ax.plot(df["epoch"], df["kl_beta"], color="tab:brown")
    ax.set_ylabel("KL beta")
    ax.set_title("KL annealing schedule")

    if has_calibration:
        # Splits the NLL gap into "predictions are wrong" vs "the model is
        # sharper than it should be", which the NLL alone cannot distinguish.
        ax = axes[2, 0]
        ax.plot(df["epoch"], df["train_z2"], label="train", color="tab:blue")
        ax.plot(df["epoch"], df["val_ho_z2"], label="held-out", color="tab:orange")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_ylabel(r"mean $z^2$")
        ax.set_xlabel("epoch")
        ax.set_title(r"Calibration: $E[z^2]$ (1.0 = calibrated, >1 overconfident)")
        ax.legend(fontsize=8)

        ax = axes[2, 1]
        ax.plot(df["epoch"], df["train_log_sigma2"], label="train", color="tab:blue")
        ax.plot(df["epoch"], df["val_ho_log_sigma2"], label="held-out", color="tab:orange")
        ax.set_ylabel(r"mean $\log \sigma^2$")
        ax.set_xlabel("epoch")
        ax.set_title("Sharpness: predicted log-variance")
        ax.legend(fontsize=8)
    else:
        axes[1, 1].set_xlabel("epoch")

    for ax in axes.flat:
        if pd.notna(best_epoch):
            ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{Path(history_csv).stem}  (best checkpoint: epoch {int(best_epoch)})"
        if pd.notna(best_epoch) else Path(history_csv).stem
    )
    fig.tight_layout()

    output_path = output_path or Path(history_csv).with_suffix(".png")
    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")

    if has_calibration:
        best = df[df["is_best"] == 1].tail(1)
        row = best.iloc[0] if len(best) else df.iloc[-1]
        print(
            f"at epoch {int(row['epoch'])}:  "
            f"E[z^2] train={row['train_z2']:.2f} held-out={row['val_ho_z2']:.2f}  |  "
            f"log(sigma^2) train={row['train_log_sigma2']:.2f} "
            f"held-out={row['val_ho_log_sigma2']:.2f}"
        )
        # Attribute the train/held-out NLL gap. Both terms enter the NLL with a
        # 0.5 coefficient, so their differences are directly comparable.
        d_accuracy = 0.5 * (row["val_ho_z2"] - row["train_z2"])
        d_sharpness = 0.5 * (row["val_ho_log_sigma2"] - row["train_log_sigma2"])
        total = abs(d_accuracy) + abs(d_sharpness)
        if total > 0:
            print(
                f"NLL gap attribution: accuracy (z^2) {d_accuracy:+.3f}, "
                f"sharpness (log sigma^2) {d_sharpness:+.3f} "
                f"-> {'accuracy' if abs(d_accuracy) > abs(d_sharpness) else 'sharpness'} dominates "
                f"({max(abs(d_accuracy), abs(d_sharpness)) / total * 100:.0f}%)"
            )
        if row["val_ho_z2"] > 1.5:
            print(
                f"note: held-out E[z^2]={row['val_ho_z2']:.2f} >> 1 — the model is "
                "overconfident on unseen cells; its intervals are too narrow."
            )

    # A widening gap between train_loss and val_ho_nll while val_ho_nll itself
    # is flat or rising is the standard "capacity outran the data" signature.
    tail = df.tail(max(1, len(df) // 5))
    if tail["val_ho_nll"].iloc[-1] > tail["val_ho_nll"].iloc[0] and len(tail) > 3:
        print(
            "note: val_ho_nll rose over the final "
            f"{len(tail)} epochs ({tail['val_ho_nll'].iloc[0]:.4f} -> "
            f"{tail['val_ho_nll'].iloc[-1]:.4f}) while training continued — "
            "check for overfitting."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("history_csv")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)
    plot(args.history_csv, args.output)


if __name__ == "__main__":
    sys.exit(main())
