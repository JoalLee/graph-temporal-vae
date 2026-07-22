"""`graph-tcn-vae train` / `graph-tcn-vae impute` command-line entry point."""
import argparse
import json

from .config import TrainConfig
from .infer import impute as run_impute
from .train import train_from_config


def _csv_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser():
    parser = argparse.ArgumentParser(prog="graph-tcn-vae", description="Train / impute with Graph-TCN-VAE models.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model on CSV data and save a checkpoint bundle.")
    train_p.add_argument("--csv", required=True, help="Comma-separated CSV path(s), joined on the timestamp column.")
    train_p.add_argument("--timestamp-col", required=True)
    train_p.add_argument("--target-cols", required=True, help="Comma-separated columns to impute/predict.")
    train_p.add_argument("--aux-cols", default="", help="Comma-separated conditioning columns (met, time, etc.).")
    train_p.add_argument("--target-transform", choices=["none", "log1p"], default="none",
                         help="Transform targets before scaling/training; log1p matches the research preprocessing.")
    train_p.add_argument("--window-size", type=int, default=48)
    train_p.add_argument("--stride", type=int, default=24)
    train_p.add_argument("--val-fraction", type=float, default=0.15)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--patience", type=int, default=15)
    train_p.add_argument("--denoise-prob", type=float, default=0.0,
                         help="Optional extra random point-drop probability; dynamic HO masking is always enabled.")
    train_p.add_argument("--dynamic-mask-target-ratio", type=float, default=0.10)
    train_p.add_argument("--dynamic-mask-mean-duration", type=float, default=48.0)
    train_p.add_argument("--dynamic-mask-std-duration", type=float, default=24.0)
    train_p.add_argument("--dynamic-mask-min-duration", type=int, default=3)
    train_p.add_argument("--dynamic-mask-max-duration", type=int, default=168)
    train_p.add_argument("--dynamic-mask-chem-blocks", type=int, default=1)
    train_p.add_argument("--dynamic-mask-psd-blocks", type=int, default=1)
    train_p.add_argument("--selection-val-seed", type=int, default=100003)
    train_p.add_argument(
        "--validation-metric", choices=["ho_nll", "ho_mse", "ho_crps"], default="ho_nll",
        help="Held-out selection metric for early stopping.",
    )
    train_p.add_argument("--val-crps-mc-samples", type=int, default=20)
    train_p.add_argument("--val-crps-every-n-epochs", type=int, default=1)
    train_p.add_argument("--seed", type=int, default=0)

    train_p.add_argument("--n-chem", type=int, default=0, help="First N target columns treated as the Chem modality.")
    train_p.add_argument("--latent-dim", type=int, default=256)
    train_p.add_argument("--hidden-dims", default="512,512,512")
    train_p.add_argument("--encoder-layers", type=int, default=5)
    train_p.add_argument("--decoder-layers", type=int, default=5)
    train_p.add_argument("--n-graph-heads", type=int, default=4)
    train_p.add_argument("--dropout", type=float, default=0.1)
    train_p.add_argument("--heteroscedastic", dest="heteroscedastic", action="store_true", default=True)
    train_p.add_argument("--no-heteroscedastic", dest="heteroscedastic", action="store_false")
    train_p.add_argument(
        "--model-config", default=None,
        help="Optional JSON file of extra ImputationVAE_Graph kwargs; overrides the flags above on conflict.",
    )
    train_p.add_argument("-o", "--output", required=True, help="Path to write the checkpoint bundle (.pt).")

    infer_p = sub.add_parser("impute", help="Impute/predict on new CSVs using a trained checkpoint bundle.")
    infer_p.add_argument("--bundle", required=True)
    infer_p.add_argument("--csv", required=True, help="Comma-separated CSV path(s).")
    infer_p.add_argument("--timestamp-col", default=None, help="Defaults to the column used at training time.")
    infer_p.add_argument("--stride", type=int, default=None, help="Defaults to window_size // 2.")
    infer_p.add_argument("--n-mc-samples", type=int, default=50)
    infer_p.add_argument("--inference-batch-size", type=int, default=4,
                         help="Number of sliding windows per model call.")
    infer_p.add_argument("--mc-batch-size", type=int, default=1,
                         help="Number of MC draws replicated per model call.")
    infer_p.add_argument("-o", "--output", required=True)

    return parser


def _model_kwargs_from_args(args):
    model_kwargs = {}
    if args.model_config:
        with open(args.model_config) as f:
            model_kwargs.update(json.load(f))

    model_kwargs.setdefault("n_chem", args.n_chem)
    model_kwargs.setdefault("latent_dim", args.latent_dim)
    model_kwargs.setdefault("hidden_dims", [int(v) for v in args.hidden_dims.split(",")])
    model_kwargs.setdefault("encoder_layers", args.encoder_layers)
    model_kwargs.setdefault("decoder_layers", args.decoder_layers)
    model_kwargs.setdefault("n_graph_heads", args.n_graph_heads)
    model_kwargs.setdefault("dropout", args.dropout)
    model_kwargs.setdefault("heteroscedastic", args.heteroscedastic)
    return model_kwargs


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = TrainConfig(
            csv=_csv_list(args.csv),
            timestamp_col=args.timestamp_col,
            target_cols=_csv_list(args.target_cols),
            aux_cols=_csv_list(args.aux_cols),
            target_transform=args.target_transform,
            window_size=args.window_size,
            stride=args.stride,
            val_fraction=args.val_fraction,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            denoise_prob=args.denoise_prob,
            dynamic_mask_target_ratio=args.dynamic_mask_target_ratio,
            dynamic_mask_mean_duration=args.dynamic_mask_mean_duration,
            dynamic_mask_std_duration=args.dynamic_mask_std_duration,
            dynamic_mask_min_duration=args.dynamic_mask_min_duration,
            dynamic_mask_max_duration=args.dynamic_mask_max_duration,
            dynamic_mask_chem_blocks=args.dynamic_mask_chem_blocks,
            dynamic_mask_psd_blocks=args.dynamic_mask_psd_blocks,
            selection_val_seed=args.selection_val_seed,
            validation_metric=args.validation_metric,
            val_crps_mc_samples=args.val_crps_mc_samples,
            val_crps_every_n_epochs=args.val_crps_every_n_epochs,
            seed=args.seed,
            model_kwargs=_model_kwargs_from_args(args),
        )
        best_val = train_from_config(config, args.output)
        print(f"saved checkpoint bundle to {args.output} (best val loss={best_val:.4f})")

    elif args.command == "impute":
        run_impute(
            _csv_list(args.csv),
            args.bundle,
            args.output,
            stride=args.stride,
            n_mc_samples=args.n_mc_samples,
            timestamp_col=args.timestamp_col,
            inference_batch_size=args.inference_batch_size,
            mc_batch_size=args.mc_batch_size,
        )
        print(f"wrote imputed output to {args.output}")


if __name__ == "__main__":
    main()
