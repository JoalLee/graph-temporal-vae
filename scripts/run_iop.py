#!/usr/bin/env python3
"""Train the Graph-enhanced Temporal-VAE on the IOP data and impute the gaps.

Runs the full pipeline in one process: validate the time grid and schema, train
with the below-detection-limit (Tobit) likelihood enabled, then impute every
gap with uncertainty and summarize what came out.

Detection limits are read from the train config's ``censoring.thresholds``
block. Build or refresh that table with ``scripts/build_mdl_table.py``, which
needs a local AeroViz install.

Examples
--------
Full run, timestamped output directory under ``outputs/``::

    python scripts/run_iop.py

Quick end-to-end check before committing to a long run::

    python scripts/run_iop.py --smoke

Skip training and re-impute from a checkpoint you already have::

    python scripts/run_iop.py --skip-train --bundle outputs/iop_.../model.pt

Compare against the old behavior, where a non-detect is an ordinary zero::

    python scripts/run_iop.py --no-censoring --run-name ablation_no_censoring
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from graph_temporal_vae.api import validate_multimodal_data  # noqa: E402
from graph_temporal_vae.bundle import inspect_bundle  # noqa: E402
from graph_temporal_vae.config import TrainConfig  # noqa: E402
from graph_temporal_vae.contracts import InferenceConfig  # noqa: E402
from graph_temporal_vae.infer import (  # noqa: E402
    impute,
    load_bundle,
    write_imputed_wide_outputs,
)
from graph_temporal_vae.train import train_from_config  # noqa: E402

DEFAULT_MODALITIES = {
    "chemistry": ["data/iop_clean/chem.csv"],
    "psd": ["data/iop_clean/psd.csv"],
    "meteorology": ["data/iop_clean/met.csv"],
}


def log(message):
    print(f"[run_iop] {message}", flush=True)


def modality_paths(modality_files):
    """Plain {modality: [paths]} from either a dict or a ModalityFiles."""
    if isinstance(modality_files, dict):
        return modality_files
    return {
        "chemistry": list(modality_files.chemistry),
        "psd": list(modality_files.psd),
        "meteorology": list(modality_files.meteorology),
    }


def describe_device():
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_smoke_overrides(config_dict):
    """Shrink the run to a few epochs and a tiny model, keeping every code path."""
    config_dict.update(
        epochs=4, patience=4, lr_warmup_epochs=1, kl_warmup_epochs=2,
        full_data_refit_epochs=2, full_data_refit_patience=2,
        model_kwargs={
            "latent_dim": 16, "hidden_dims": [32, 32], "encoder_layers": 2,
            "decoder_layers": 2, "n_graph_heads": 2, "dropout": 0.0,
            "heteroscedastic": True,
        },
    )
    return config_dict


def step_validate(modality_files, timestamp_col, out_dir):
    log("1/3 validating data")
    report = validate_multimodal_data(
        data=modality_files,
        timestamp_col=timestamp_col,
        expected_frequency="1h",
        time_grid_policy="strict",
        duplicate_timestamp_policy="error",
    )
    (out_dir / "validate.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit("Data validation failed; see validate.json")
    log(f"    valid, {report['rows']} rows from {report['start']}")
    return report


def step_train(config, out_dir):
    log("2/3 training")
    bundle_path = out_dir / "model.pt"
    best_val = train_from_config(config, str(bundle_path))
    selection_label = (
        "global-HO" if config.shared_full_heldout_mask else "validation"
    )
    log(
        f"    best {selection_label} metric "
        f"({config.validation_metric}) = {best_val:.4f}"
    )
    training_summary = load_bundle(bundle_path).get("training_summary", {})
    refit = training_summary.get("refit")
    if refit is not None:
        log(
            "    full-data refit: "
            f"{refit['epochs_completed']} epoch(s), "
            f"best dynamic_ho_mse={refit['best_monitor_mse']:.4f}, "
            f"early_stopping={refit['early_stopping_enabled']}"
        )
    (out_dir / "bundle.json").write_text(
        json.dumps(inspect_bundle(str(bundle_path)), indent=2, sort_keys=True, default=str)
    )
    return bundle_path


def step_impute(bundle_path, modality_files, out_dir, args):
    log("3/3 imputing")
    bundle = load_bundle(bundle_path)
    result = impute(
        None,
        bundle,
        str(out_dir / "imputed_long.csv"),
        modality_files=modality_files,
        inference_config=InferenceConfig(
            stride=args.impute_stride,
            n_mc_samples=args.mc_samples,
            inference_batch_size=args.inference_batch_size,
            interval_lower=0.05,
            interval_upper=0.95,
        ),
    )
    wide_outputs = write_imputed_wide_outputs(
        result,
        out_dir,
        bundle["data_schema"],
    )
    for modality, path in wide_outputs.items():
        log(f"    {modality} wide output: {path}")
    return result


def step_heldout_eval(bundle_path, modality_files, out_dir, args):
    """Score the checkpoint on cells whose truth was hidden from the model.

    Separate from imputation on purpose: `impute` writes real observations
    straight through, so it can never tell you how accurate the model is.
    """
    log("held-out evaluation")
    paths = modality_paths(modality_files)
    call = [
        sys.executable, str(REPO_ROOT / "examples" / "heldout_eval.py"),
        "--bundle", str(bundle_path),
        "--chem-csv", ",".join(paths["chemistry"]),
        "--psd-csv", ",".join(paths["psd"]),
        "--met-csv", ",".join(paths["meteorology"]),
        "--n-mc-samples", str(args.mc_samples),
        "--stride", str(args.heldout_stride),
        "-o", str(out_dir / "heldout_metrics.json"),
        "--predictions-csv", str(out_dir / "heldout_predictions.csv"),
    ]
    result = subprocess.run(call, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"    held-out eval failed: {result.stderr.strip().splitlines()[-3:]}")
        return False
    for line in result.stdout.splitlines():
        if line.startswith("[heldout_eval]"):
            log(f"    {line}")
    metrics = json.loads((out_dir / "heldout_metrics.json").read_text())
    for key in ("overall_heldout_r2", "chem_heldout_r2", "psd_heldout_r2",
                "overall_heldout_picp"):
        if key in metrics:
            log(f"    {key} = {metrics[key]:.4f}")
    return True


def step_plots(bundle_path, out_dir):
    """Best-effort: a missing matplotlib should never fail the run itself."""
    log("plotting diagnostics")
    history_csv = bundle_path.with_name(bundle_path.stem + "_history.csv")
    imputed_csv = out_dir / "imputed_long.csv"
    calls = [
        [sys.executable, str(REPO_ROOT / "scripts" / "plot_training_history.py"), str(history_csv)],
        [sys.executable, str(REPO_ROOT / "scripts" / "plot_feature_diagnostics.py"),
         str(bundle_path), str(imputed_csv)],
    ]
    # A run with full_data_refit_epochs > 0 writes a second, differently
    # shaped history file for the refit stage; plot_training_history.py
    # auto-detects it by its dynamic_ho_* columns.
    refit_history_csv = bundle_path.with_name(bundle_path.stem + "_refit_history.csv")
    if refit_history_csv.exists():
        calls.append(
            [sys.executable, str(REPO_ROOT / "scripts" / "plot_training_history.py"),
             str(refit_history_csv)]
        )
    calls += [
        [sys.executable, str(REPO_ROOT / "scripts" / "plot_observation_heatmap.py"),
         str(imputed_csv), "--modality", "chem"],
        [sys.executable, str(REPO_ROOT / "scripts" / "plot_observation_heatmap.py"),
         str(imputed_csv), "--modality", "psd"],
    ]
    predictions_csv = out_dir / "heldout_predictions.csv"
    if predictions_csv.exists():
        for fam in ("chem", "psd"):
            calls.append([
                sys.executable, str(REPO_ROOT / "scripts" / "plot_heldout_performance.py"),
                str(predictions_csv), "--bundle", str(bundle_path), "--family", fam,
            ])
    for call in calls:
        result = subprocess.run(call, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"    plot failed ({Path(call[1]).name}): {result.stderr.strip().splitlines()[-1:]}")
        else:
            log(f"    {result.stdout.strip().splitlines()[-1]}")


def summarize(result, out_dir):
    """Report what was imputed, and check the censored cells against their limits."""
    states = result["observation_state"].value_counts().to_dict()
    total = len(result)
    lines = ["# IOP imputation summary", "", "## Cell states", ""]
    for state in ("observed", "censored", "missing"):
        count = states.get(state, 0)
        lines.append(f"- {state}: {count:,} ({count / total * 100:.1f}%)")

    violations = 0
    censored = result[result["observation_state"] == "censored"]
    if len(censored):
        # A non-detect means the true value is below the limit. If an imputed
        # value exceeded it, the reported result would contradict the
        # measurement that produced it, so this must be exactly zero.
        over_mean = int((censored["imputed_mean"] > censored["detection_limit"] + 1e-9).sum())
        over_upper = int((censored["q_upper"] > censored["detection_limit"] + 1e-9).sum())
        violations = over_mean + over_upper
        lines += [
            "", "## Detection-limit consistency", "",
            f"- censored cells: {len(censored):,}",
            f"- imputed_mean above its limit: **{over_mean}** (must be 0)",
            f"- q_upper above its limit: **{over_upper}** (must be 0)",
        ]
        if violations:
            lines.append("")
            lines.append("> **A censored prediction broke its own detection limit.**")

    missing = result[result["observation_state"] == "missing"]
    if len(missing):
        lines += [
            "", "## Imputed gaps", "",
            f"- imputed cells: {len(missing):,}",
            f"- mean predictive std: {missing['imputed_std'].mean():.4g}",
        ]
        widest = (
            missing.assign(width=missing["q_upper"] - missing["q_lower"])
            .groupby("feature")["width"].mean().sort_values(ascending=False).head(10)
        )
        lines += ["", "Widest mean 90% intervals (least certain features):", ""]
        lines += [f"- `{feature}`: {width:.4g}" for feature, width in widest.items()]

    report = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(report)
    print()
    print(report)
    return violations


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--train-config", default="examples/iop_train_config.json",
        help="Training config JSON, including the censoring/threshold block.",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Output directory name under --output-root. Defaults to a timestamp.",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny model and 4 epochs: exercises every step in about a minute.",
    )
    parser.add_argument(
        "--no-censoring", action="store_true",
        help="Disable below-detection-limit handling, restoring the old "
             "behavior where a reported zero is an ordinary observation. "
             "Use this for an ablation, not for production runs.",
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse an existing checkpoint; requires --bundle.")
    parser.add_argument("--bundle", default=None, help="Checkpoint to impute with.")
    parser.add_argument("--skip-impute", action="store_true")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating the training-curve and diagnostic plots.")
    parser.add_argument("--skip-heldout-eval", action="store_true",
                        help="Skip per-feature held-out scoring (an extra inference pass).")
    parser.add_argument("--heldout-stride", type=int, default=8,
                        help="Window stride for the held-out scoring pass.")
    parser.add_argument("--impute-stride", type=int, default=1,
                        help="1 gives maximum window overlap and the smoothest output.")
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.skip_train and not args.bundle:
        raise SystemExit("--skip-train requires --bundle")

    run_name = args.run_name or f"iop_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"run: {run_name}")
    log(f"output: {out_dir}")
    log(f"torch {torch.__version__} on {describe_device()}")

    config_dict = json.loads(Path(args.train_config).read_text())
    if args.smoke:
        log("smoke mode: 4 epochs, tiny model")
        config_dict = apply_smoke_overrides(config_dict)
    if args.no_censoring:
        log("censoring DISABLED: non-detects will be treated as ordinary zeros")
        config_dict.setdefault("censoring", {})["enabled"] = False

    config = TrainConfig(**config_dict)
    # Persist the validated/resolved config, not only the raw JSON input. In
    # particular, implicit Student-t uncertainty resolution must be visible in
    # the run artifact.
    config_dict = config.to_dict()
    modality_files = config.modality_files or DEFAULT_MODALITIES
    if config.censoring.active:
        log(f"censoring on: {len(config.censoring.thresholds)} detection limits, "
            f"loss={config.censoring.loss}")

    (out_dir / "config.used.json").write_text(json.dumps(config_dict, indent=2, sort_keys=True))

    step_validate(modality_files, config.timestamp_col, out_dir)

    bundle_path = Path(args.bundle) if args.skip_train else step_train(config, out_dir)
    if args.skip_impute:
        log(f"done (imputation skipped) -> {out_dir}")
        return 0

    result = step_impute(bundle_path, modality_files, out_dir, args)
    violations = summarize(result, out_dir)

    if not args.skip_heldout_eval:
        bundle_training = load_bundle(bundle_path).get("training_config", {})
        if int(bundle_training.get("full_data_refit_epochs", 0)) > 0:
            log(
                "held-out evaluation skipped: the final refit restored the "
                "global-HO targets, so they are no longer independent; use "
                "the stage-one training history for selection evidence"
            )
        else:
            step_heldout_eval(bundle_path, modality_files, out_dir, args)

    if not args.no_plots:
        step_plots(bundle_path, out_dir)

    log(f"done -> {out_dir}")
    for path in sorted(out_dir.iterdir()):
        log(f"    {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
