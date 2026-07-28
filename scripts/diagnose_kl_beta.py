#!/usr/bin/env python3
"""Sweep kl_max_beta and report whether the latent posterior collapses.

Run this BEFORE a real training run, not after: it exists to answer "what is
the largest kl_max_beta I can use without the posterior collapsing on this
architecture and dataset?" so you don't have to find out 190 epochs into a
full run.

Background: a VAE's KL term regularizes the encoder's posterior q(z|x) toward
the prior. Too little regularization risks the classic "posterior does
whatever it wants" failure; too much can drive q(z|x) all the way to the
prior, making z carry no information about x at all -- "posterior collapse".
Once that happens the whole generative pathway degenerates toward a
deterministic mapping from conditioning alone, which tends to report
overconfident, badly calibrated uncertainty (this was diagnosed empirically:
in one run, the settled KL divergence hit exactly 0.000000 and stayed there
for 100+ epochs, coinciding with the held-out E[z^2] calibration ratio
accelerating 27x in the same window).

The diagnostic signal is the raw KL divergence itself (``val_kl``, added to
``model_history.csv`` alongside the calibration panels), not accuracy metrics
like MSE or z^2 -- those are downstream symptoms and can look fine for a
while even as KL is already at zero.

This trains one SHORT run per candidate beta (same architecture and data as
the base config; only kl_max_beta/epochs/patience are overridden) and reports,
per beta: whether the settled KL/latent-dim fell below the collapse
threshold, when, and what held-out z^2/MSE looked like at the end.

Usage:
    python scripts/diagnose_kl_beta.py \\
        --train-config examples/iop_train_config.json \\
        --beta-values 0.1,0.3,0.5,1.0 \\
        --epochs 150 \\
        -o outputs/kl_diagnostic

A coarse grid only brackets the safe range; treat the recommended value as a
starting point; a value one grid step below it is a safer bet if the two
neighboring points disagree sharply -- rerun the sweep with a finer grid near
the boundary if it matters for a production run. Every candidate uses the
SAME model_kwargs as the base config (capacity changes the collapse
threshold) -- only epochs/patience are shortened for speed.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from graph_temporal_vae.config import TrainConfig  # noqa: E402
from graph_temporal_vae.train import train_from_config  # noqa: E402


def log(message):
    print(f"[kl_diagnostic] {message}", flush=True)


def run_one(base_config_dict, beta, args, out_dir):
    config_dict = dict(base_config_dict)
    config_dict["kl_max_beta"] = beta
    config_dict["epochs"] = args.epochs
    # No early stopping during the sweep by default: cutting a run short
    # before the posterior has had time to settle would hide a slow collapse
    # and understate the risk of the tested beta.
    config_dict["patience"] = args.patience if args.patience is not None else args.epochs
    if args.kl_warmup_epochs is not None:
        config_dict["kl_warmup_epochs"] = args.kl_warmup_epochs
    # The refit phase is a separate concern (fixed beta, restored targets)
    # and would roughly double every candidate's cost for no diagnostic value.
    config_dict["full_data_refit_epochs"] = 0
    config_dict["full_data_refit_patience"] = None
    config_dict["full_data_refit_lr"] = None
    config_dict["seed"] = args.seed

    bundle_path = out_dir / f"beta_{beta:g}" / "model.pt"
    log(f"training kl_max_beta={beta:g}  (epochs={config_dict['epochs']}, "
        f"patience={config_dict['patience']}, kl_warmup_epochs={config_dict.get('kl_warmup_epochs')})")
    config = TrainConfig(**config_dict)
    train_from_config(config, str(bundle_path))
    history_path = bundle_path.with_name(bundle_path.stem + "_history.csv")
    history = pd.read_csv(history_path)
    if "val_kl" not in history.columns:
        raise SystemExit(
            "model_history.csv has no val_kl column -- this diagnostic needs the "
            "raw-KL-divergence logging added to Trainer; check graph_temporal_vae version."
        )
    return history


def analyze(history, latent_dim, threshold_nats_per_dim, window_fraction, kl_warmup_epochs):
    """Classify whether the posterior settled into a collapsed state."""
    window = max(1, int(len(history) * window_fraction))
    tail = history.tail(window)
    settled_kl = float(tail["val_kl"].mean())
    settled_per_dim = settled_kl / max(latent_dim, 1)
    collapsed = settled_per_dim < threshold_nats_per_dim

    onset_epoch = None
    if collapsed:
        below = history["val_kl"] / max(latent_dim, 1) < threshold_nats_per_dim
        # First epoch after which it never recovers above threshold again --
        # a permanent collapse, not a transient dip early in KL warmup.
        for idx in range(len(history)):
            if below.iloc[idx:].all():
                onset_epoch = int(history["epoch"].iloc[idx])
                break

    collapsed_before_warmup_ends = (
        onset_epoch is not None
        and kl_warmup_epochs is not None
        and onset_epoch < kl_warmup_epochs
    )
    return {
        "settled_val_kl": settled_kl,
        "settled_val_kl_per_dim": settled_per_dim,
        "collapsed": collapsed,
        "collapse_onset_epoch": onset_epoch,
        "collapsed_before_warmup_ends": collapsed_before_warmup_ends,
        "final_z2": float(history["val_ho_z2"].iloc[-1]) if "val_ho_z2" in history else None,
        "final_mse": float(history["val_ho_mse"].iloc[-1]),
    }


def plot(results, threshold_nats_per_dim, latent_dim, output_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("viridis")
    betas = sorted(results)
    for index, beta in enumerate(betas):
        history = results[beta]["history"]
        per_dim = (history["val_kl"] / max(latent_dim, 1)).clip(lower=1e-6)
        status = "collapsed" if results[beta]["analysis"]["collapsed"] else "stable"
        color = cmap(index / max(1, len(betas) - 1))
        ax.plot(history["epoch"], per_dim, label=f"beta={beta:g} ({status})", color=color)
    ax.axhline(threshold_nats_per_dim, color="black", linestyle=":", linewidth=1,
              label="collapse threshold")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out KL divergence (nats / latent dim)")
    ax.set_title("Posterior collapse sweep")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")


def recommend(results):
    safe = [beta for beta, r in results.items() if not r["analysis"]["collapsed"]]
    return max(safe) if safe else None


def write_report(results, recommended, output_path, threshold_nats_per_dim, kl_warmup_epochs):
    lines = [
        "# KL beta diagnostic",
        "",
        f"Collapse threshold: {threshold_nats_per_dim:g} nats/latent-dim, sustained through the "
        "end of the run. `val_kl` is the raw, un-weighted posterior KL divergence -- the "
        "diagnostic signal itself, not a downstream symptom.",
        "",
        "| beta | settled KL/dim | collapsed | onset epoch | before warmup ends? | final held-out z2 | final held-out MSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for beta in sorted(results):
        a = results[beta]["analysis"]
        lines.append(
            f"| {beta:g} | {a['settled_val_kl_per_dim']:.4f} | "
            f"{'yes' if a['collapsed'] else 'no'} | "
            f"{a['collapse_onset_epoch'] if a['collapse_onset_epoch'] is not None else '—'} | "
            f"{'**yes**' if a['collapsed_before_warmup_ends'] else 'no'} | "
            f"{a['final_z2']:.1f} | {a['final_mse']:.4f} |"
        )
    lines.append("")
    if recommended is not None:
        lines.append(
            f"**Recommendation: `kl_max_beta = {recommended:g}`** -- the largest tested value "
            "whose posterior did not collapse. This is a coarse-grid bracket, not a precise "
            "boundary: if a beta one step higher collapsed sharply, treat this value as "
            "optimistic and consider a finer sweep, or go one step lower for margin."
        )
    else:
        lines.append(
            "**No tested beta avoided collapse.** Try lower values, or use free-bits KL "
            "regularization (a per-dimension floor below which KL stops contributing "
            "gradient) instead of lowering beta further."
        )
    any_early = any(r["analysis"]["collapsed_before_warmup_ends"] for r in results.values())
    if any_early:
        lines += [
            "",
            "**Warning**: at least one candidate collapsed before its own KL warmup even "
            "finished ramping (`kl_warmup_epochs`" + (f"={kl_warmup_epochs}" if kl_warmup_epochs else "") +
            "). That candidate's beta is too aggressive for this architecture/data regardless "
            "of how long you train it.",
        ]
    Path(output_path).write_text("\n".join(lines) + "\n")
    print(f"wrote {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-config", required=True,
                        help="Base TrainConfig JSON; only kl_max_beta/epochs/patience are overridden.")
    parser.add_argument("--beta-values", default="0.1,0.3,0.5,1.0")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Short-run budget per candidate. Should comfortably exceed "
                             "kl_warmup_epochs so a slow collapse has time to show up.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Defaults to --epochs (no early stopping during the sweep).")
    parser.add_argument("--kl-warmup-epochs", type=int, default=None,
                        help="Override; defaults to the base config's value.")
    parser.add_argument("--collapse-threshold-nats-per-dim", type=float, default=0.01)
    parser.add_argument("--window-fraction", type=float, default=0.2,
                        help="Fraction of final epochs averaged to decide the 'settled' KL value.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args(argv)

    base_config = json.loads(Path(args.train_config).read_text())
    latent_dim = base_config.get("model_kwargs", {}).get("latent_dim", 1)
    kl_warmup_epochs = args.kl_warmup_epochs or base_config.get("kl_warmup_epochs")
    betas = [float(value) for value in args.beta_values.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for beta in betas:
        history = run_one(base_config, beta, args, out_dir)
        analysis = analyze(
            history, latent_dim, args.collapse_threshold_nats_per_dim,
            args.window_fraction, kl_warmup_epochs,
        )
        results[beta] = {"history": history, "analysis": analysis}
        log(
            f"beta={beta:g}: settled KL/dim={analysis['settled_val_kl_per_dim']:.4f}  "
            f"collapsed={analysis['collapsed']}"
            + (f" (onset epoch {analysis['collapse_onset_epoch']})" if analysis["collapsed"] else "")
            + f"  final_z2={analysis['final_z2']:.1f}"
        )

    plot(results, args.collapse_threshold_nats_per_dim, latent_dim, out_dir / "kl_sweep.png")
    recommended = recommend(results)
    write_report(
        results, recommended, out_dir / "kl_sweep_report.md",
        args.collapse_threshold_nats_per_dim, kl_warmup_epochs,
    )
    if recommended is not None:
        log(f"recommended kl_max_beta = {recommended:g}")
    else:
        log("no tested beta avoided collapse -- try lower values or free-bits")


if __name__ == "__main__":
    sys.exit(main())
