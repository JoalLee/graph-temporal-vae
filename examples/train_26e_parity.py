"""Run the public trainer with the audited 26e reference configuration.

The input CSV is produced by ``prepare_26e_input.py``.  Keeping this runner
explicit prevents the general-purpose CLI defaults (short windows, split
validation, AdamW disabled weight decay, mask-channel aux, and legacy-vs-block
masking choices) from silently changing a reproduction run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Allow direct execution from the repository root, e.g.
# ``python examples/train_26e_parity.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_tcn_vae.config import TrainConfig
from graph_tcn_vae.train import train_from_config
from prepare_26e_input import AUX_COLS, CHEM_COLS, TIME_COLS


def build_26e_config(
    input_csv: str | Path,
    model_config: str | Path,
    epochs: int = 2000,
    target_transform: str = "log1p",
) -> TrainConfig:
    columns = pd.read_csv(input_csv, nrows=0).columns.tolist()
    reserved = {"time", *CHEM_COLS, *AUX_COLS, *TIME_COLS}
    psd_cols = [col for col in columns if col not in reserved]
    if len(CHEM_COLS) != 32 or len(psd_cols) != 230:
        raise ValueError(
            f"Expected 32 Chem + 230 PSD columns, got {len(CHEM_COLS)} + {len(psd_cols)}"
        )

    with open(model_config) as handle:
        model_kwargs = json.load(handle)

    return TrainConfig(
        csv=[str(input_csv)],
        timestamp_col="time",
        target_cols=CHEM_COLS + psd_cols,
        aux_cols=AUX_COLS + TIME_COLS,
        target_transform=target_transform,
        target_output_transform="log1p",
        scaler_fit_scope="full",
        window_size=168,
        stride=24,
        val_fraction=0.0,
        batch_size=64,
        epochs=epochs,
        lr=2.5e-4,
        lr_min=2e-6,
        weight_decay=1e-2,
        patience=100,
        train_loader_num_workers=4,
        val_loader_num_workers=2,
        denoise_prob=0.0,
        # These are the reference train_uq defaults: legacy masking uses six
        # PSD blocks, eight Chem blocks, and 4% point dropout.
        dynamic_mask_target_ratio=0.10,
        dynamic_mask_mean_duration=48.0,
        dynamic_mask_std_duration=24.0,
        dynamic_mask_min_duration=3,
        dynamic_mask_max_duration=168,
        dynamic_mask_chem_blocks=1,
        dynamic_mask_psd_blocks=1,
        dynamic_masking_mode="legacy",
        dynamic_random_point_drop_prob=0.04,
        selection_val_seed=42,
        selection_mask_mode="anchor_constrained",
        selection_mask_ratio=0.10,
        shared_full_heldout_mask=True,
        validation_metric="ho_mse",
        val_crps_mc_samples=20,
        val_crps_every_n_epochs=1,
        val_crps_dist_type="gaussian",
        val_mc_batch_size=1,
        use_adaptive_lr=True,
        lr_reduce_factor=0.5,
        lr_reduce_patience=60,
        lr_reduce_threshold=1e-4,
        lr_reduce_cooldown=10,
        lr_warmup_ratio=0.05,
        kl_warmup_ratio=0.10,
        kl_strategy="cosine",
        use_amp=True,
        amp_dtype="auto",
        prior_type="student_t",
        use_gnll=True,
        use_student_t_nll=True,
        loss_normalization="window_feature_sum",
        chem_feature_weight=12.0,
        psd_feature_weight=1.0,
        # The reference condition is 5 aux + 6 time = 11 channels, without
        # an additional aux observedness channel.
        aux_mask_channel=False,
        seed=42,
        model_kwargs=model_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw CSV from prepare_26e_input.py")
    parser.add_argument("--model-config", default="examples/26e_parity_model_config.json")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--target-transform", choices=["none", "log1p"], default="log1p",
        help="Transform present in the input contract. Use 'none' for the private pre-log1p experiment_input_raw_26e.csv; the preparation helper writes raw values, so its default is 'log1p'.",
    )
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    config = build_26e_config(
        args.input, args.model_config, epochs=args.epochs, target_transform=args.target_transform
    )
    best_val = train_from_config(config, args.output)
    print(f"saved checkpoint bundle to {args.output} (best ho_mse={best_val:.6f})")


if __name__ == "__main__":
    main()
