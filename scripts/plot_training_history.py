#!/usr/bin/env python3
"""Plot the per-epoch training history written next to a checkpoint bundle.

Every ``graph-temporal-vae train`` run writes ``<bundle>_history.csv`` as it
goes (one row per epoch: loss, held-out metrics, LR, KL beta). This turns that
CSV into the curves that answer whether a run is overfitting, whether its
schedules behaved, and whether the censored term is improving or being traded
off against the observed one.

Panels are grouped by whether the quantities they show are actually on the
same scale. ``train_loss`` includes the KL term, per-modality feature
weighting, and the censored Tobit term all folded together; it is not on the
same footing as ``val_ho_nll`` (a plain reconstruction NLL on held-out cells)
and is plotted alone so it is never misread as directly comparable to
anything next to it. Genuinely comparable pairs -- the same formula evaluated
on two different cell sets -- share a panel:

* ``train_ho_nll``/``train_ho_mse`` vs. ``val_ho_nll``/``val_ho_mse``:
  identical formulas, held-out cells from the training-period fixed mask vs.
  the validation split. Only present when the config sets
  ``train_ho_enabled: true``.
* ``val_ho_nll`` vs. ``val_censored_nll``: identical NLL formula (Gaussian/
  Student-t reconstruction likelihood), held-out observed cells vs. held-out
  censored cells within the same validation split.
* ``train_z2``/``train_log_sigma2`` vs. their held-out counterparts: the
  accuracy/sharpness decomposition of the NLL, same units on both sides.

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


def _has(df, *columns):
    return all(c in df.columns and df[c].notna().any() for c in columns)


def _mark_best(ax, best_epoch):
    if pd.notna(best_epoch):
        ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)


def plot(history_csv, output_path=None):
    df = pd.read_csv(history_csv)
    best_epoch = df.loc[df["is_best"] == 1, "epoch"].max()
    has_train_ho = _has(df, "train_ho_nll")
    has_calibration = _has(df, "val_ho_z2")

    # Build the panel list dynamically: every panel is either a lone metric
    # with its own scale, or an explicitly comparable pair sharing one axis.
    panels = [("train_loss_only", None)]
    panels.append(("held_out_nll", None))
    panels.append(("val_ho_mse", None))
    if has_train_ho:
        panels.append(("train_vs_val_ho_nll", None))
    if has_calibration:
        panels.append(("z2", None))
        panels.append(("log_sigma2", None))
    panels.append(("lr", None))
    panels.append(("kl_beta", None))

    n_cols = 2
    n_rows = (len(panels) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.3 * n_rows), sharex=True)
    flat_axes = list(axes.reshape(n_rows, n_cols).flat)

    for (kind, _), ax in zip(panels, flat_axes):
        if kind == "train_loss_only":
            ax.plot(df["epoch"], df["train_loss"], label="total", color="tab:blue")
            title = "Training objective (KL + feature weights + Tobit — not\n" \
                    "comparable in scale to the held-out NLL panels)"
            if _has(df, "train_recon", "train_weighted_kl"):
                ax.plot(df["epoch"], df["train_recon"], label="recon", color="tab:green")
                ax.plot(df["epoch"], df["train_weighted_kl"], label="beta*KL", color="tab:red")
                ax.legend(fontsize=8)
                title += "\n(total = recon + beta*KL, exactly)"
            ax.set_ylabel("loss")
            ax.set_title(title, fontsize=9)

        elif kind == "held_out_nll":
            ax.plot(df["epoch"], df["val_ho_nll"], label="observed", color="tab:orange")
            if _has(df, "val_censored_nll"):
                ax.plot(df["epoch"], df["val_censored_nll"], label="censored",
                       color="tab:red", linestyle="--")
            ax.set_ylabel("held-out NLL")
            ax.set_title("Held-out NLL: observed vs. censored cells\n(directly comparable — same formula)",
                        fontsize=9)
            ax.legend(fontsize=8)

        elif kind == "val_ho_mse":
            ax.plot(df["epoch"], df["val_ho_mse"], label="val-period HO", color="tab:green")
            title = "Held-out MSE (scaled space)"
            if has_train_ho and _has(df, "train_ho_mse"):
                ax.plot(df["epoch"], df["train_ho_mse"], label="train-period HO", color="tab:blue")
                ax.legend(fontsize=8)
                title += "\n(directly comparable — same formula, same as the NLL panel)"
            ax.set_ylabel("held-out MSE")
            ax.set_title(title, fontsize=9)

        elif kind == "train_vs_val_ho_nll":
            ax.plot(df["epoch"], df["train_ho_nll"], label="train-period HO", color="tab:blue")
            ax.plot(df["epoch"], df["val_ho_nll"], label="val-period HO", color="tab:orange")
            ax.set_ylabel("held-out NLL")
            ax.set_title("Train vs. held-out NLL, same cell type\n(directly comparable — this is the real generalization gap)",
                        fontsize=9)
            ax.legend(fontsize=8)

        elif kind == "z2":
            ax.plot(df["epoch"], df["train_z2"], label="train", color="tab:blue")
            ax.plot(df["epoch"], df["val_ho_z2"], label="held-out", color="tab:orange")
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
            ax.set_ylabel(r"mean $z^2$")
            ax.set_title(r"Calibration: $E[z^2]$ (1.0 = calibrated, >1 overconfident)", fontsize=9)
            ax.legend(fontsize=8)

        elif kind == "log_sigma2":
            ax.plot(df["epoch"], df["train_log_sigma2"], label="train", color="tab:blue")
            ax.plot(df["epoch"], df["val_ho_log_sigma2"], label="held-out", color="tab:orange")
            ax.set_ylabel(r"mean $\log \sigma^2$")
            ax.set_title("Sharpness: predicted log-variance", fontsize=9)
            ax.legend(fontsize=8)

        elif kind == "lr":
            ax.plot(df["epoch"], df["lr"], color="tab:purple")
            ax.set_yscale("log")
            ax.set_ylabel("learning rate")
            ax.set_title("LR schedule")

        elif kind == "kl_beta":
            ax.plot(df["epoch"], df["kl_beta"], color="tab:brown")
            ax.set_ylabel("KL beta")
            ax.set_title("KL annealing schedule")

        ax.set_xlabel("epoch")
        _mark_best(ax, best_epoch)

    for ax in flat_axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(
        f"{Path(history_csv).stem}  (best checkpoint: epoch {int(best_epoch)})"
        if pd.notna(best_epoch) else Path(history_csv).stem
    )
    fig.tight_layout()

    output_path = output_path or Path(history_csv).with_suffix(".png")
    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")

    if has_train_ho:
        best = df[df["is_best"] == 1].tail(1)
        row = best.iloc[0] if len(best) else df.iloc[-1]
        gap = row["val_ho_nll"] - row["train_ho_nll"]
        print(
            f"at epoch {int(row['epoch'])}: train-period HO NLL={row['train_ho_nll']:.4f}, "
            f"val-period HO NLL={row['val_ho_nll']:.4f}  ->  gap={gap:+.4f} "
            "(the real generalization gap, both sides same formula)"
        )
        if _has(df, "train_ho_mse"):
            mse_gap = row["val_ho_mse"] - row["train_ho_mse"]
            print(
                f"                train-period HO MSE={row['train_ho_mse']:.4f}, "
                f"val-period HO MSE={row['val_ho_mse']:.4f}  ->  gap={mse_gap:+.4f}"
            )

    if has_calibration:
        best = df[df["is_best"] == 1].tail(1)
        row = best.iloc[0] if len(best) else df.iloc[-1]
        print(
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


def plot_refit(history_csv, output_path=None):
    """Plot a full-data-refit history (``<bundle>_refit_history.csv``).

    Stage two restores the global-HO cells into training and continues at a
    small fixed LR with beta pinned at ``kl_max_beta``, monitored by a
    deterministic dynamic-pattern mask (``dynamic_ho_*``). That monitor is
    NOT independent validation -- its target values are inside full-data
    training -- so this is a diagnostic of the refit's own behavior (is it
    still improving, did calibration or the KL divergence drift), not a
    generalization report. See ``Trainer.refit_full_data`` for why.

    x-axis is ``model_epoch`` (continuing stage one's numbering) so this plot
    lines up visually with ``model_history.png`` from the same run.

    Usage:
        python scripts/plot_training_history.py outputs/iop_.../model_refit_history.csv
    """
    df = pd.read_csv(history_csv)
    best_epoch = df.loc[df["is_best"] == 1, "model_epoch"].max()
    has_kl = _has(df, "train_kl", "dynamic_ho_kl")

    panels = ["train_loss_only", "dynamic_ho_nll", "dynamic_ho_mse", "z2", "log_sigma2"]
    if has_kl:
        panels.append("kl")
    panels.append("lr")

    n_cols = 2
    n_rows = (len(panels) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.3 * n_rows), sharex=True)
    flat_axes = list(axes.reshape(n_rows, n_cols).flat)
    x = df["model_epoch"]

    for kind, ax in zip(panels, flat_axes):
        if kind == "train_loss_only":
            ax.plot(x, df["train_loss"], label="total", color="tab:blue")
            title = "Training objective (full-data, KL pinned at max —\n" \
                    "not comparable to the dynamic_ho panels)"
            if _has(df, "train_recon", "train_weighted_kl"):
                ax.plot(x, df["train_recon"], label="recon", color="tab:green")
                ax.plot(x, df["train_weighted_kl"], label="beta*KL", color="tab:red")
                ax.legend(fontsize=8)
                title += "\n(total = recon + beta*KL, exactly)"
            ax.set_ylabel("loss")
            ax.set_title(title, fontsize=9)

        elif kind == "dynamic_ho_nll":
            ax.plot(x, df["dynamic_ho_nll"], color="tab:orange")
            ax.set_ylabel("dynamic_ho_nll")
            ax.set_title("Dynamic-HO NLL\n(monitor only — targets are inside full-data training)",
                        fontsize=9)

        elif kind == "dynamic_ho_mse":
            ax.plot(x, df["dynamic_ho_mse"], color="tab:green")
            ax.set_ylabel("dynamic_ho_mse")
            ax.set_title("Dynamic-HO MSE (the refit early-stopping metric)", fontsize=9)

        elif kind == "z2":
            ax.plot(x, df["train_z2"], label="train", color="tab:blue")
            ax.plot(x, df["dynamic_ho_z2"], label="dynamic-HO", color="tab:orange")
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
            ax.set_ylabel(r"mean $z^2$")
            ax.set_title(r"Calibration: $E[z^2]$ (1.0 = calibrated, >1 overconfident)", fontsize=9)
            ax.legend(fontsize=8)

        elif kind == "log_sigma2":
            ax.plot(x, df["train_log_sigma2"], label="train", color="tab:blue")
            ax.plot(x, df["dynamic_ho_log_sigma2"], label="dynamic-HO", color="tab:orange")
            ax.set_ylabel(r"mean $\log \sigma^2$")
            ax.set_title("Sharpness: predicted log-variance", fontsize=9)
            ax.legend(fontsize=8)

        elif kind == "kl":
            ax.plot(x, df["train_kl"], label="train", color="tab:blue")
            ax.plot(x, df["dynamic_ho_kl"], label="dynamic-HO", color="tab:orange")
            ax.set_ylabel("KL divergence")
            ax.set_title("Raw KL divergence under fixed beta=kl_max_beta\n"
                        "(does the posterior keep moving at the low refit LR?)", fontsize=9)
            ax.legend(fontsize=8)

        elif kind == "lr":
            ax.plot(x, df["lr"], color="tab:purple")
            ax.set_ylabel("learning rate")
            ax.set_title(f"Fixed refit LR = {df['lr'].iloc[0]:.2e}", fontsize=9)

        ax.set_xlabel("model epoch (continues stage 1)")
        _mark_best(ax, best_epoch)

    for ax in flat_axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(
        f"{Path(history_csv).stem}  (kl_beta fixed at {df['kl_beta'].iloc[0]:.3g}, "
        f"best refit epoch: model_epoch {int(best_epoch)})"
        if pd.notna(best_epoch) else Path(history_csv).stem
    )
    fig.tight_layout()

    output_path = output_path or Path(history_csv).with_suffix(".png")
    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")

    best = df[df["is_best"] == 1].tail(1)
    row = best.iloc[0] if len(best) else df.iloc[-1]
    print(
        f"at model_epoch {int(row['model_epoch'])}: dynamic_ho_mse={row['dynamic_ho_mse']:.4f}, "
        f"dynamic_ho_nll={row['dynamic_ho_nll']:.4f}"
    )
    if has_kl:
        print(f"KL divergence: train={row['train_kl']:.4f}, dynamic-HO={row['dynamic_ho_kl']:.4f}")
        drift = df["dynamic_ho_kl"].iloc[-1] - df["dynamic_ho_kl"].iloc[0]
        if abs(drift) > 0.1 * max(df["dynamic_ho_kl"].iloc[0], 1e-6):
            direction = "collapsed further" if drift < 0 else "grew"
            print(
                f"note: dynamic-HO KL {direction} by {drift:+.4f} over the refit "
                "phase despite a fixed beta — the posterior is still moving, not settled."
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("history_csv")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)
    df_columns = pd.read_csv(args.history_csv, nrows=0).columns
    if "dynamic_ho_mse" in df_columns:
        plot_refit(args.history_csv, args.output)
    else:
        plot(args.history_csv, args.output)


if __name__ == "__main__":
    sys.exit(main())
