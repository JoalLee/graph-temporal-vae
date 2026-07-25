"""`graph-temporal-vae train` / `graph-temporal-vae impute` command-line entry point."""
import argparse
import json

from .api import validate_multimodal_data
from .bundle import inspect_bundle
from .config import TrainConfig
from .contracts import (
    InferenceConfig,
    ModalityFiles,
    ModalityPreprocessing,
    PreprocessingConfig,
)
from .data import load_frame, load_modality_frame
from .infer import impute as run_impute, load_bundle
from .model_config import ModelConfig
from .train import train_from_config


def _csv_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser():
    parser = argparse.ArgumentParser(prog="graph-temporal-vae", description="Train / impute with Graph-enhanced Temporal-VAE models.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model and save a checkpoint bundle.")
    train_p.add_argument(
        "--train-config", default=None,
        help="JSON file accepted by TrainConfig; when provided, other train settings are ignored.",
    )
    train_p.add_argument(
        "--csv", default=None,
        help="Legacy comma-separated CSV path(s), joined on the timestamp column.",
    )
    train_p.add_argument("--target-cols", default=None, help="Legacy comma-separated target columns.")
    train_p.add_argument("--aux-cols", default="", help="Legacy comma-separated conditioning columns.")
    train_p.add_argument("--chem-csv", default=None, help="Chemistry CSV path(s); all non-time columns are targets.")
    train_p.add_argument("--psd-csv", default=None, help="PSD CSV path(s); column names must be diameters in nm.")
    train_p.add_argument("--met-csv", default=None, help="Meteorology/auxiliary CSV path(s); all non-time columns are conditions.")
    train_p.add_argument("--timestamp-col", default="time")
    train_p.add_argument("--target-transform", choices=["none", "log1p"], default="none",
                         help="Transform targets before scaling/training; log1p matches the research preprocessing.")
    train_p.add_argument(
        "--target-output-transform", choices=["none", "log1p"], default=None,
        help="Inverse transform for output values; defaults to --target-transform. "
             "Use log1p when the input CSV is already log1p-transformed.",
    )
    train_p.add_argument("--scaler-fit-scope", choices=["train", "full"], default="train")
    train_p.add_argument("--chem-transform", choices=["none", "log1p"], default=None)
    train_p.add_argument("--psd-transform", choices=["none", "log1p"], default=None)
    train_p.add_argument("--met-transform", choices=["none", "log1p"], default="none")
    train_p.add_argument("--chem-output-transform", choices=["none", "log1p"], default=None)
    train_p.add_argument("--psd-output-transform", choices=["none", "log1p"], default=None)
    train_p.add_argument("--chem-scaler", choices=["standard", "robust", "minmax", "none"], default="standard")
    train_p.add_argument("--psd-scaler", choices=["standard", "robust", "minmax", "none"], default="standard")
    train_p.add_argument("--met-scaler", choices=["standard", "robust", "minmax", "none"], default="standard")
    train_p.add_argument(
        "--expected-frequency",
        default=None,
        help="Expected pandas frequency such as 1h or 30min. If omitted, the modal positive interval is used.",
    )
    train_p.add_argument(
        "--time-grid-policy",
        choices=["strict", "reindex", "row_order"],
        default="strict",
        help="Reject irregular timestamps, insert missing grid rows, or preserve legacy row-order semantics.",
    )
    train_p.add_argument(
        "--duplicate-timestamp-policy", choices=["error", "first"], default="error"
    )
    train_p.add_argument("--window-size", type=int, default=48)
    train_p.add_argument("--stride", type=int, default=24)
    train_p.add_argument("--val-fraction", type=float, default=0.15)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--lr-min", type=float, default=1e-6)
    train_p.add_argument("--weight-decay", type=float, default=0.0)
    train_p.add_argument("--patience", type=int, default=15)
    train_p.add_argument("--train-loader-num-workers", type=int, default=0)
    train_p.add_argument("--val-loader-num-workers", type=int, default=0)
    train_p.add_argument("--denoise-prob", type=float, default=0.0,
                         help="Optional extra random point-drop probability; dynamic HO masking is always enabled.")
    train_p.add_argument("--dynamic-mask-target-ratio", type=float, default=0.10)
    train_p.add_argument("--dynamic-mask-mean-duration", type=float, default=48.0)
    train_p.add_argument("--dynamic-mask-std-duration", type=float, default=24.0)
    train_p.add_argument("--dynamic-mask-min-duration", type=int, default=3)
    train_p.add_argument("--dynamic-mask-max-duration", type=int, default=168)
    train_p.add_argument("--dynamic-mask-chem-blocks", type=int, default=1)
    train_p.add_argument("--dynamic-mask-psd-blocks", type=int, default=1)
    train_p.add_argument("--dynamic-masking-mode", choices=["block", "legacy"], default="block")
    train_p.add_argument("--dynamic-random-point-drop-prob", type=float, default=0.0)
    train_p.add_argument("--selection-val-seed", type=int, default=100003)
    train_p.add_argument("--selection-mask-mode", choices=["block", "anchor_constrained"], default="block")
    train_p.add_argument("--selection-mask-ratio", type=float, default=0.10)
    train_p.add_argument("--shared-full-heldout-mask", action="store_true")
    train_p.add_argument(
        "--validation-metric", choices=["ho_nll", "ho_mse", "ho_crps"], default="ho_nll",
        help="Held-out selection metric for early stopping.",
    )
    train_p.add_argument("--val-crps-mc-samples", type=int, default=20)
    train_p.add_argument("--val-crps-every-n-epochs", type=int, default=1)
    train_p.add_argument("--val-crps-dist-type", choices=["gaussian", "student_t"], default="gaussian")
    train_p.add_argument("--val-mc-batch-size", type=int, default=1)
    train_p.add_argument("--use-adaptive-lr", action="store_true",
                         help="Switch to ReduceLROnPlateau (monitoring held-out MSE) after warmup, as in the 26e reference.")
    train_p.add_argument("--lr-reduce-factor", type=float, default=0.5)
    train_p.add_argument("--lr-reduce-patience", type=int, default=10)
    train_p.add_argument("--lr-reduce-threshold", type=float, default=1e-4)
    train_p.add_argument("--lr-reduce-cooldown", type=int, default=2)
    train_p.add_argument(
        "--lr-warmup-epochs", type=int, default=None,
        help="Absolute LR warmup length. Overrides --lr-warmup-ratio. Use this (not the ratio) "
             "when reproducing a reference run that preserves an absolute warmup length under a "
             "reduced epoch budget, e.g. 100 epochs of warmup even when --epochs is 700, not 2000.",
    )
    train_p.add_argument("--lr-warmup-ratio", type=float, default=None,
                         help="LR warmup as a fraction of --epochs. Defaults to 0.05 if neither this nor --lr-warmup-epochs is set.")
    train_p.add_argument("--kl-warmup-epochs", type=int, default=None,
                         help="Absolute KL warmup length. Overrides --kl-warmup-ratio.")
    train_p.add_argument("--kl-warmup-ratio", type=float, default=None)
    train_p.add_argument("--kl-strategy", choices=["linear", "cosine", "cyclical"], default="cosine")
    train_p.add_argument("--use-amp", action="store_true")
    train_p.add_argument("--amp-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    train_p.add_argument("--prior-type", choices=["gaussian", "laplace", "student_t"], default="gaussian")
    train_p.add_argument("--use-student-t-nll", action="store_true")
    train_p.add_argument("--loss-normalization", choices=["observed_mean", "window_feature_sum"], default="observed_mean")
    train_p.add_argument("--chem-feature-weight", type=float, default=1.0)
    train_p.add_argument("--psd-feature-weight", type=float, default=1.0)
    train_p.add_argument("--aux-mask-channel", dest="aux_mask_channel", action="store_true", default=True)
    train_p.add_argument("--no-aux-mask-channel", dest="aux_mask_channel", action="store_false")
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

    infer_p = sub.add_parser("impute", help="Impute/predict using a trained checkpoint bundle.")
    infer_p.add_argument("--bundle", required=True)
    infer_p.add_argument(
        "--inference-config", default=None,
        help="Optional JSON file accepted by InferenceConfig.",
    )
    infer_p.add_argument("--csv", default=None, help="Legacy comma-separated CSV path(s).")
    infer_p.add_argument("--chem-csv", default=None)
    infer_p.add_argument("--psd-csv", default=None)
    infer_p.add_argument("--met-csv", default=None)
    infer_p.add_argument("--timestamp-col", default=None, help="Defaults to the column used at training time.")
    infer_p.add_argument("--stride", type=int, default=None, help="Defaults to the stride saved in the bundle.")
    infer_p.add_argument("--n-mc-samples", type=int, default=50)
    infer_p.add_argument("--interval-lower", type=float, default=0.05)
    infer_p.add_argument("--interval-upper", type=float, default=0.95)
    infer_p.add_argument("--inference-batch-size", type=int, default=4,
                         help="Number of sliding windows per model call.")
    infer_p.add_argument("--mc-batch-size", type=int, default=1,
                         help="Number of MC draws replicated per model call.")
    infer_p.add_argument(
        "--support-context-window",
        type=int,
        default=72,
        help="Rows on each side used for operational gap-support diagnostics.",
    )
    infer_p.add_argument("-o", "--output", required=True)

    inspect_p = sub.add_parser(
        "inspect-bundle", help="Validate and summarize a checkpoint bundle."
    )
    inspect_p.add_argument("--bundle", required=True)

    validate_p = sub.add_parser(
        "validate-data", help="Validate schema, time grid, and missingness without training."
    )
    validate_p.add_argument("--csv", default=None, help="Legacy comma-separated CSV path(s).")
    validate_p.add_argument("--target-cols", default=None)
    validate_p.add_argument("--aux-cols", default="")
    validate_p.add_argument("--chem-csv", default=None)
    validate_p.add_argument("--psd-csv", default=None)
    validate_p.add_argument("--met-csv", default=None)
    validate_p.add_argument("--timestamp-col", default=None)
    validate_p.add_argument(
        "--bundle",
        default=None,
        help="Optional checkpoint whose training schema must match the supplied data.",
    )
    validate_p.add_argument("--expected-frequency", default=None)
    validate_p.add_argument(
        "--time-grid-policy", choices=["strict", "reindex", "row_order"], default="strict"
    )
    validate_p.add_argument(
        "--duplicate-timestamp-policy", choices=["error", "first"], default="error"
    )

    return parser


def _modality_files_from_args(args):
    chemistry = _csv_list(args.chem_csv) if getattr(args, "chem_csv", None) else []
    psd = _csv_list(args.psd_csv) if getattr(args, "psd_csv", None) else []
    meteorology = _csv_list(args.met_csv) if getattr(args, "met_csv", None) else []
    if not chemistry and not psd and not meteorology:
        return None
    return ModalityFiles(chemistry=chemistry, psd=psd, meteorology=meteorology)


def _preprocessing_from_args(args):
    chem_transform = args.chem_transform or args.target_transform
    psd_transform = args.psd_transform or args.target_transform
    shared_output = args.target_output_transform
    return PreprocessingConfig(
        chemistry=ModalityPreprocessing(
            transform=chem_transform,
            scaler=args.chem_scaler,
            output_transform=args.chem_output_transform or shared_output or chem_transform,
        ),
        psd=ModalityPreprocessing(
            transform=psd_transform,
            scaler=args.psd_scaler,
            output_transform=args.psd_output_transform or shared_output or psd_transform,
        ),
        meteorology=ModalityPreprocessing(
            transform=args.met_transform,
            scaler=args.met_scaler,
            output_transform=args.met_transform,
        ),
        fit_scope=args.scaler_fit_scope,
        aux_mask_channel=args.aux_mask_channel,
    )


def _model_kwargs_from_args(args, *, include_n_chem=True):
    model_kwargs = {}
    if args.model_config:
        with open(args.model_config) as f:
            configured = json.load(f)
        # Validate unknown/stale fields without expanding unspecified options
        # into a full 147-field default dictionary.
        ModelConfig.from_dict(configured)
        model_kwargs.update(configured)

    if include_n_chem:
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
        if args.train_config:
            config = TrainConfig.from_json(args.train_config)
            best_val = train_from_config(config, args.output)
            print(f"saved checkpoint bundle to {args.output} (best val loss={best_val:.4f})")
            return
        modality_files = _modality_files_from_args(args)
        if modality_files is not None and (args.csv or args.target_cols or args.aux_cols):
            parser.error("Use modality CSV flags or legacy --csv/--target-cols/--aux-cols, not both")
        if modality_files is None and (not args.csv or not args.target_cols):
            parser.error("Legacy mode requires --csv and --target-cols; otherwise provide --chem-csv and/or --psd-csv")
        config = TrainConfig(
            csv=_csv_list(args.csv) if args.csv else [],
            timestamp_col=args.timestamp_col,
            target_cols=_csv_list(args.target_cols) if args.target_cols else [],
            aux_cols=_csv_list(args.aux_cols),
            modality_files=modality_files,
            preprocessing=_preprocessing_from_args(args),
            target_transform=args.target_transform,
            target_output_transform=args.target_output_transform,
            scaler_fit_scope=args.scaler_fit_scope,
            expected_frequency=args.expected_frequency,
            time_grid_policy=args.time_grid_policy,
            duplicate_timestamp_policy=args.duplicate_timestamp_policy,
            window_size=args.window_size,
            stride=args.stride,
            val_fraction=args.val_fraction,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            lr_min=args.lr_min,
            weight_decay=args.weight_decay,
            patience=args.patience,
            train_loader_num_workers=args.train_loader_num_workers,
            val_loader_num_workers=args.val_loader_num_workers,
            denoise_prob=args.denoise_prob,
            dynamic_mask_target_ratio=args.dynamic_mask_target_ratio,
            dynamic_mask_mean_duration=args.dynamic_mask_mean_duration,
            dynamic_mask_std_duration=args.dynamic_mask_std_duration,
            dynamic_mask_min_duration=args.dynamic_mask_min_duration,
            dynamic_mask_max_duration=args.dynamic_mask_max_duration,
            dynamic_mask_chem_blocks=args.dynamic_mask_chem_blocks,
            dynamic_mask_psd_blocks=args.dynamic_mask_psd_blocks,
            dynamic_masking_mode=args.dynamic_masking_mode,
            dynamic_random_point_drop_prob=args.dynamic_random_point_drop_prob,
            selection_val_seed=args.selection_val_seed,
            selection_mask_mode=args.selection_mask_mode,
            selection_mask_ratio=args.selection_mask_ratio,
            shared_full_heldout_mask=args.shared_full_heldout_mask,
            validation_metric=args.validation_metric,
            val_crps_mc_samples=args.val_crps_mc_samples,
            val_crps_every_n_epochs=args.val_crps_every_n_epochs,
            val_crps_dist_type=args.val_crps_dist_type,
            val_mc_batch_size=args.val_mc_batch_size,
            use_adaptive_lr=args.use_adaptive_lr,
            lr_reduce_factor=args.lr_reduce_factor,
            lr_reduce_patience=args.lr_reduce_patience,
            lr_reduce_threshold=args.lr_reduce_threshold,
            lr_reduce_cooldown=args.lr_reduce_cooldown,
            lr_warmup_epochs=args.lr_warmup_epochs,
            lr_warmup_ratio=args.lr_warmup_ratio,
            kl_warmup_epochs=args.kl_warmup_epochs,
            kl_warmup_ratio=args.kl_warmup_ratio,
            kl_strategy=args.kl_strategy,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
            prior_type=args.prior_type,
            use_student_t_nll=args.use_student_t_nll,
            loss_normalization=args.loss_normalization,
            chem_feature_weight=args.chem_feature_weight,
            psd_feature_weight=args.psd_feature_weight,
            aux_mask_channel=args.aux_mask_channel,
            seed=args.seed,
            model_kwargs=_model_kwargs_from_args(
                args, include_n_chem=modality_files is None
            ),
        )
        best_val = train_from_config(config, args.output)
        print(f"saved checkpoint bundle to {args.output} (best val loss={best_val:.4f})")

    elif args.command == "impute":
        modality_files = _modality_files_from_args(args)
        if modality_files is not None and args.csv:
            parser.error("Use modality CSV flags or legacy --csv, not both")
        if modality_files is None and not args.csv:
            parser.error("Provide --csv or modality-specific CSV flags")
        if args.inference_config:
            with open(args.inference_config) as handle:
                inference_config = InferenceConfig.from_dict(json.load(handle))
        else:
            inference_config = InferenceConfig(
                stride=args.stride,
                n_mc_samples=args.n_mc_samples,
                inference_batch_size=args.inference_batch_size,
                mc_batch_size=args.mc_batch_size,
                interval_lower=args.interval_lower,
                interval_upper=args.interval_upper,
                support_context_window=args.support_context_window,
            )
        run_impute(
            _csv_list(args.csv) if args.csv else None,
            args.bundle,
            args.output,
            timestamp_col=args.timestamp_col,
            modality_files=modality_files,
            inference_config=inference_config,
        )
        print(f"wrote imputed output to {args.output}")

    elif args.command == "inspect-bundle":
        print(json.dumps(inspect_bundle(args.bundle), indent=2, sort_keys=True))

    elif args.command == "validate-data":
        modality_files = _modality_files_from_args(args)
        if modality_files is not None and (args.csv or args.target_cols or args.aux_cols):
            parser.error("Use modality CSV flags or legacy CSV flags, not both")

        loaded = load_bundle(args.bundle) if args.bundle else None
        expected_schema = loaded["data_schema"] if loaded else None
        timestamp_col = (
            args.timestamp_col
            or (expected_schema.timestamp_col if expected_schema is not None else "time")
        )
        if modality_files is not None:
            report = validate_multimodal_data(
                data=modality_files,
                timestamp_col=timestamp_col,
                expected_frequency=args.expected_frequency,
                time_grid_policy=args.time_grid_policy,
                duplicate_timestamp_policy=args.duplicate_timestamp_policy,
                expected_schema=expected_schema,
            )
        else:
            if not args.csv:
                parser.error("Provide modality CSV flags or legacy --csv")
            if loaded is not None:
                target_cols = loaded["target_cols"]
                aux_cols = loaded["aux_cols"]
            else:
                if not args.target_cols:
                    parser.error("Legacy validation without --bundle requires --target-cols")
                target_cols = _csv_list(args.target_cols)
                aux_cols = _csv_list(args.aux_cols)
            frame = load_frame(
                _csv_list(args.csv),
                timestamp_col,
                target_cols,
                aux_cols,
                expected_frequency=(
                    expected_schema.frequency if expected_schema is not None
                    else args.expected_frequency
                ),
                time_grid_policy=(
                    expected_schema.time_grid_policy if expected_schema is not None
                    else args.time_grid_policy
                ),
                duplicate_timestamp_policy=(
                    expected_schema.duplicate_timestamp_policy if expected_schema is not None
                    else args.duplicate_timestamp_policy
                ),
            )
            report = {
                "valid": True,
                "rows": len(frame),
                "start": str(frame.index.min()),
                "end": str(frame.index.max()),
                "frequency": frame.attrs.get("frequency"),
                "timezone": frame.attrs.get("timezone"),
                "data_schema": expected_schema.to_dict() if expected_schema else None,
                "target_missing_fraction": frame[target_cols].isna().mean().to_dict(),
                "auxiliary_missing_fraction": frame[aux_cols].isna().mean().to_dict()
                if aux_cols else {},
            }
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
