"""Load a checkpoint bundle produced by `train.py` and impute new CSVs."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .censoring import (
    STATE_CENSORED,
    STATE_MISSING,
    STATE_OBSERVED,
    CensoringConfig,
    apply_input_fill,
    build_state_matrix,
)
from .contracts import (
    DataSchema,
    InferenceConfig,
    ModalityFiles,
    ModalityInputs,
    PreprocessingConfig,
)
from .data import compute_window_starts, load_frame, load_modality_frame, make_condition
from .model_config import ModelConfig
from .model_graph_uq import ImputationVAE_Graph
from .preprocessing import (
    NaNAwareAffineScaler,
    inverse_targets,
    observed_targets_to_output,
    preprocessing_from_legacy,
    target_output_transforms,
    transform_auxiliary,
    transform_targets,
)
from .utils import is_interactive, setup_device
from .window_aggregation import StreamingWindowAggregator, aggregate_ordered_windows


def _validate_scaler_state(state, expected_features, label):
    center = state.get("center", state.get("mean"))
    scale = state.get("scale", state.get("std"))
    if center is None or scale is None:
        raise ValueError(f"{label} scaler must contain center/scale statistics")
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if center.ndim != 1 or scale.ndim != 1:
        raise ValueError(f"{label} scaler statistics must be one-dimensional")
    if len(center) != expected_features or len(scale) != expected_features:
        raise ValueError(
            f"{label} scaler dimension does not match schema: "
            f"expected={expected_features}, center={len(center)}, scale={len(scale)}"
        )
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError(f"{label} scaler statistics must be finite")
    if np.any(scale <= 0):
        raise ValueError(f"{label} scaler scale values must be positive")
    feature_kinds = list(state.get("feature_kinds", []))
    if feature_kinds and len(feature_kinds) != expected_features:
        raise ValueError(f"{label} scaler feature_kinds length does not match schema")
    return feature_kinds


def _validate_bundle(bundle):
    required = {
        "state_dict", "model_kwargs", "target_cols", "aux_cols", "window_size",
        "scaler_target", "scaler_aux", "config",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Invalid checkpoint bundle; missing keys: {missing}")
    bundle_version = int(bundle.get("bundle_version", 1))
    if bundle_version not in {1, 2, 3}:
        raise ValueError(f"Unsupported bundle_version={bundle_version}")
    if not isinstance(bundle["state_dict"], dict) or not bundle["state_dict"]:
        raise ValueError("Checkpoint state_dict must be a non-empty mapping")
    if not isinstance(bundle["window_size"], int) or bundle["window_size"] < 1:
        raise ValueError("Checkpoint window_size must be a positive integer")
    stride = bundle.get("stride", bundle.get("config", {}).get("stride"))
    if stride is not None and (not isinstance(stride, int) or not 1 <= stride <= bundle["window_size"]):
        raise ValueError("Checkpoint stride must be between 1 and window_size")
    ModelConfig.from_dict(bundle["model_kwargs"])
    if not bundle["target_cols"]:
        raise ValueError("Checkpoint bundle must contain at least one target column")
    if bundle["aux_missing_mode"] not in {"legacy_zero_fill", "mask_channel"}:
        raise ValueError(f"Unsupported aux_missing_mode: {bundle['aux_missing_mode']!r}")
    if bundle.get("target_transform", "none") not in {"none", "log1p", "mixed"}:
        raise ValueError(f"Unsupported target_transform: {bundle.get('target_transform')!r}")
    if bundle.get("target_output_transform", "none") not in {"none", "log1p", "mixed"}:
        raise ValueError(
            "Unsupported target_output_transform: "
            f"{bundle.get('target_output_transform')!r}"
        )
    target_cols = list(bundle["target_cols"])
    aux_cols = list(bundle["aux_cols"])
    if len(set(target_cols)) != len(target_cols):
        raise ValueError("Checkpoint target_cols contain duplicates")
    if len(set(aux_cols)) != len(aux_cols):
        raise ValueError("Checkpoint aux_cols contain duplicates")
    overlap = sorted(set(target_cols) & set(aux_cols))
    if overlap:
        raise ValueError(f"Checkpoint columns appear as both target and auxiliary: {overlap}")
    target_feature_kinds = _validate_scaler_state(
        bundle["scaler_target"], len(target_cols), "Target"
    )
    aux_feature_kinds = _validate_scaler_state(
        bundle["scaler_aux"], len(aux_cols), "Auxiliary"
    )
    if bundle.get("architecture_version", 1) != 1:
        raise ValueError(
            f"Unsupported architecture_version={bundle.get('architecture_version')}"
        )
    if bundle.get("state_dict_format_version", 1) != 1:
        raise ValueError(
            f"Unsupported state_dict_format_version={bundle.get('state_dict_format_version')}"
        )
    if "data_schema" in bundle:
        resolved = DataSchema.from_dict(bundle["data_schema"])
        if resolved.schema_version != 1:
            raise ValueError(f"Unsupported data schema_version={resolved.schema_version}")
        if resolved.target_cols != list(bundle["target_cols"]):
            raise ValueError("data_schema target order does not match target_cols")
        if resolved.auxiliary_cols != list(bundle["aux_cols"]):
            raise ValueError("data_schema meteorology order does not match aux_cols")
        if int(bundle["model_kwargs"].get("n_chem", 0)) != resolved.n_chem:
            raise ValueError("model n_chem does not match data_schema chemistry columns")
        if len(bundle.get("target_output_transforms", [])) != resolved.target_dim:
            raise ValueError("target_output_transforms must match data_schema target dimension")
        preprocessing = PreprocessingConfig.from_dict(bundle["preprocessing"])
        expected_output_transforms = target_output_transforms(resolved, preprocessing)
        if list(bundle["target_output_transforms"]) != expected_output_transforms:
            raise ValueError(
                "target_output_transforms do not match the stored preprocessing contract"
            )
        expected_target_kinds = (
            [preprocessing.chemistry.scaler] * len(resolved.chemistry_cols)
            + [preprocessing.psd.scaler] * len(resolved.psd_cols)
        )
        if target_feature_kinds and target_feature_kinds != expected_target_kinds:
            raise ValueError("Target scaler feature kinds do not match preprocessing")
        expected_aux_kinds = [preprocessing.meteorology.scaler] * resolved.aux_dim
        if aux_feature_kinds and aux_feature_kinds != expected_aux_kinds:
            raise ValueError("Auxiliary scaler feature kinds do not match preprocessing")
        if bool(bundle["aux_mask_channel"]) != preprocessing.aux_mask_channel:
            raise ValueError("aux_mask_channel does not match preprocessing contract")
        expected_aux_mode = (
            "mask_channel" if preprocessing.aux_mask_channel else "legacy_zero_fill"
        )
        if bundle["aux_missing_mode"] != expected_aux_mode:
            raise ValueError("aux_missing_mode does not match preprocessing contract")
    schema = bundle.get("schema")
    if schema is not None:
        expected_cond_dim = len(bundle["aux_cols"]) * (2 if bundle["aux_mask_channel"] else 1)
        if schema.get("target_dim") != len(bundle["target_cols"]):
            raise ValueError("Checkpoint target schema dimension does not match target_cols")
        if schema.get("cond_dim") != expected_cond_dim:
            raise ValueError("Checkpoint condition schema dimension does not match aux mask mode")


def load_bundle(path, device=None):
    device = device or setup_device()
    # Bundle files are data, not executable Python objects.  weights_only=True
    # blocks arbitrary pickle object construction when loading an untrusted file.
    bundle = torch.load(path, map_location=device, weights_only=True)
    bundle.setdefault("bundle_version", 1)
    bundle.setdefault("architecture_version", 1)
    bundle.setdefault("state_dict_format_version", 1)
    bundle.setdefault("aux_missing_mode", "legacy_zero_fill")
    bundle.setdefault("aux_mask_channel", bundle["aux_missing_mode"] == "mask_channel")
    bundle.setdefault("target_transform", bundle.get("config", {}).get("target_transform", "none"))
    bundle.setdefault(
        "time_grid",
        {
            "frequency": bundle.get("config", {}).get("expected_frequency"),
            "timezone": None,
            "policy": bundle.get("config", {}).get("time_grid_policy", "row_order"),
            "duplicate_timestamp_policy": bundle.get("config", {}).get(
                "duplicate_timestamp_policy", "first"
            ),
        },
    )
    bundle.setdefault(
        "target_output_transform",
        bundle.get("config", {}).get("target_output_transform", bundle["target_transform"]),
    )

    if "data_schema" in bundle:
        data_schema = DataSchema.from_dict(bundle["data_schema"])
    else:
        n_chem = int(bundle.get("model_kwargs", {}).get("n_chem", 0))
        data_schema = DataSchema(
            timestamp_col=bundle.get("config", {}).get("timestamp_col", "time"),
            chemistry_cols=list(bundle["target_cols"][:n_chem]),
            psd_cols=list(bundle["target_cols"][n_chem:]),
            meteorology_cols=list(bundle["aux_cols"]),
            frequency=bundle["time_grid"].get("frequency"),
            timezone=bundle["time_grid"].get("timezone"),
            time_grid_policy=bundle["time_grid"].get("policy", "row_order"),
            duplicate_timestamp_policy=bundle["time_grid"].get(
                "duplicate_timestamp_policy", "first"
            ),
        )
    if "preprocessing" in bundle:
        preprocessing = PreprocessingConfig.from_dict(bundle["preprocessing"])
    else:
        preprocessing = preprocessing_from_legacy(
            target_transform=bundle["target_transform"],
            target_output_transform=bundle["target_output_transform"],
            scaler_fit_scope=bundle.get("config", {}).get("scaler_fit_scope", "train"),
            aux_mask_channel=bundle["aux_mask_channel"],
        )
    bundle.setdefault("data_schema", data_schema.to_dict())
    bundle.setdefault("preprocessing", preprocessing.to_dict())
    bundle.setdefault("data_interface", "legacy_columns")
    bundle.setdefault(
        "target_output_transforms",
        target_output_transforms(data_schema, preprocessing),
    )
    _validate_bundle(bundle)

    aux_dim = data_schema.aux_dim * (2 if preprocessing.aux_mask_channel else 1)
    model = ImputationVAE_Graph(
        target_dim=data_schema.target_dim,
        aux_dim=aux_dim,
        window_size=bundle["window_size"],
        **bundle["model_kwargs"],
    )
    try:
        model.load_state_dict(bundle["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint state_dict is incompatible with the stored model/schema configuration"
        ) from exc
    model.to(device)
    model.eval()

    training_config = dict(bundle.get("config", {}))
    uncertainty_dist_type = training_config.get("val_crps_dist_type")
    if uncertainty_dist_type is None:
        uncertainty_dist_type = (
            "student_t" if training_config.get("use_student_t_nll", False) else "gaussian"
        )

    return {
        "model": model,
        "scaler_target": NaNAwareAffineScaler.from_dict(bundle["scaler_target"]),
        "scaler_aux": NaNAwareAffineScaler.from_dict(bundle["scaler_aux"]),
        "target_cols": data_schema.target_cols,
        "aux_cols": data_schema.auxiliary_cols,
        "window_size": bundle["window_size"],
        "stride": bundle.get("stride", bundle.get("config", {}).get("stride")),
        "aux_missing_mode": bundle["aux_missing_mode"],
        "aux_mask_channel": preprocessing.aux_mask_channel,
        "target_transform": bundle.get("target_transform", "none"),
        "target_output_transform": bundle.get("target_output_transform", "none"),
        "target_output_transforms": list(bundle["target_output_transforms"]),
        "preprocessing": preprocessing,
        # Non-detect handling must match training exactly: the same cells have
        # to be recognized, filled, and bounded at inference time.
        "censoring": CensoringConfig.from_dict(bundle["censoring"])
        if bundle.get("censoring")
        else CensoringConfig(),
        "censor_threshold_scaled": bundle.get("censor_threshold_scaled"),
        "data_schema": data_schema,
        "data_interface": bundle.get("data_interface", "legacy_columns"),
        "bundle_version": bundle["bundle_version"],
        "architecture_version": bundle["architecture_version"],
        "state_dict_format_version": bundle["state_dict_format_version"],
        "schema": bundle.get("schema"),
        "model_kwargs": dict(bundle["model_kwargs"]),
        "training_config": training_config,
        "uncertainty_dist_type": uncertainty_dist_type,
        "timestamp_col": data_schema.timestamp_col,
        "time_grid": bundle.get("time_grid", {}),
        "device": device,
    }


def trapezoid_position_weights(window_size, edge_frac=0.2):
    """Position envelope used to downweight window edges during overlap-add.

    1 in the middle, linear ramp over ``edge_frac`` of the window at each
    edge, matching the research trapezoidal_window_weights envelope: an
    edge timestep has less surrounding context within that window's
    representation, so it should contribute less to the overlap-add mix.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if window_size == 1:
        return np.ones(1, dtype=np.float64)
    edge_len = max(1, int(round(window_size * edge_frac)))
    if 2 * edge_len >= window_size:
        edge_len = max(1, (window_size - 1) // 2)
    weights = np.ones(window_size, dtype=np.float64)
    ramp = (np.arange(edge_len) + 1.0) / (edge_len + 1.0)
    weights[:edge_len] = ramp
    weights[-edge_len:] = ramp[::-1]
    return weights


def _bundle_censor_thresholds(bundle, data_schema):
    """Scaled-space detection limits saved at training time, or all-NaN."""
    stored = bundle.get("censor_threshold_scaled")
    thresholds = np.full(data_schema.target_dim, np.nan, dtype=np.float64)
    if not stored:
        return thresholds
    if len(stored) != data_schema.target_dim:
        raise ValueError(
            f"Bundle censor_threshold_scaled has {len(stored)} entries but the schema "
            f"has {data_schema.target_dim} targets"
        )
    for index, value in enumerate(stored):
        if value is not None:
            thresholds[index] = float(value)
    return thresholds


_HAZARD_TAIL_CUTOFF = -5.0


def _standard_normal_pdf(values):
    return np.exp(-0.5 * values ** 2) / math.sqrt(2.0 * math.pi)


def _inverse_mills_ratio(alpha):
    """``phi(a) / Phi(a)``, stable into the far-left tail.

    A direct ratio collapses to 0/0 once ``Phi`` underflows (around a = -10 in
    float64), which would silently skip the truncation and let a censored cell
    report a value above its own detection limit. The asymptotic Mills-ratio
    expansion takes over well before that, where it agrees to ~1e-4 and tends
    to ``-a``, which drives the truncated mean to the limit as it should.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    near = np.clip(alpha, _HAZARD_TAIL_CUTOFF, None)
    near_hazard = _standard_normal_pdf(near) / np.clip(_standard_normal_cdf(near), 1e-300, None)

    tail = np.clip(alpha, None, _HAZARD_TAIL_CUTOFF)
    tail_sq = tail * tail
    series = 1.0 - 1.0 / tail_sq + 3.0 / tail_sq ** 2 - 15.0 / tail_sq ** 3
    tail_hazard = -tail / np.clip(series, 1e-12, None)
    return np.where(alpha >= _HAZARD_TAIL_CUTOFF, near_hazard, tail_hazard)


def _standard_normal_cdf(values):
    # numpy has no erf; torch is already a hard dependency and ndtr is exact.
    return torch.special.ndtr(torch.as_tensor(values, dtype=torch.float64)).numpy()


def truncate_below_limit(mean_model, variance_model, limit_model, censored):
    """Condition the predictive distribution on ``y <= limit`` where censored.

    A non-detect is only known to lie below the detection limit, so reporting
    an unconstrained mean would contradict the measurement. Returns the
    truncated mean and variance plus ``P(y <= limit)`` for every cell that has
    a limit, which is also informative for genuinely missing cells.
    """
    mean_model = np.asarray(mean_model, dtype=np.float64)
    variance_model = np.asarray(variance_model, dtype=np.float64)
    limit = np.asarray(limit_model, dtype=np.float64)
    has_limit = np.isfinite(limit)
    p_below = np.full(mean_model.shape, np.nan, dtype=np.float64)
    if not has_limit.any():
        return mean_model, variance_model, p_below

    sigma = np.sqrt(np.maximum(variance_model, 1e-12))
    safe_limit = np.where(has_limit[None, :], limit[None, :], 0.0)
    alpha = (safe_limit - mean_model) / sigma
    cdf = _standard_normal_cdf(alpha)
    p_below = np.where(has_limit[None, :], cdf, np.nan)

    hazard = _inverse_mills_ratio(alpha)
    truncated_mean = mean_model - sigma * hazard
    # The leading terms of the variance factor cancel as the constraint becomes
    # binding; the floor keeps it positive, which is the right limit anyway
    # (all the mass is pinned just below the detection limit).
    truncated_var = variance_model * np.maximum(
        1.0 - alpha * hazard - hazard ** 2, 1e-12
    )

    apply = np.asarray(censored, dtype=bool) & has_limit[None, :]
    return (
        np.where(apply, truncated_mean, mean_model),
        np.where(apply, truncated_var, variance_model),
        p_below,
    )


def summary_to_output_scale(mean_model, variance_model, quantile_values_model, transform):
    """Map model-space summaries to per-feature public output scales.

    ``transform`` may be one transform string or a sequence matching the
    feature dimension. This allows Chem and PSD to use different preprocessing
    while preserving the same aggregate-then-invert ordering.
    """
    mean_model = np.asarray(mean_model, dtype=np.float64)
    variance_model = np.asarray(variance_model, dtype=np.float64)
    n_features = mean_model.shape[-1]
    transforms = [transform] * n_features if isinstance(transform, str) else list(transform)
    if len(transforms) != n_features:
        raise ValueError(
            f"Expected {n_features} output transforms, received {len(transforms)}"
        )
    if any(value not in {"none", "log1p"} for value in transforms):
        raise ValueError("Output transforms must be 'none' or 'log1p'")

    mean_out = mean_model.copy()
    std_model = np.sqrt(np.maximum(variance_model, 0.0))
    std_out = std_model.copy()
    quantiles_out = {
        q: np.asarray(values, dtype=np.float64).copy()
        for q, values in quantile_values_model.items()
    }
    for feature, feature_transform in enumerate(transforms):
        if feature_transform != "log1p":
            continue
        mean_out[..., feature] = np.expm1(mean_model[..., feature]).clip(min=0.0)
        std_out[..., feature] = np.exp(mean_model[..., feature]) * std_model[..., feature]
        for values in quantiles_out.values():
            values[..., feature] = np.expm1(values[..., feature]).clip(min=0.0)
    return mean_out, std_out, quantiles_out


def aggregate_window_samples(
    window_samples,
    window_starts=None,
    total_length=None,
    position_weights=None,
    quantiles=(0.05, 0.95),
):
    """Aggregate overlapping samples without retaining completed timestamps.

    ``window_samples`` is either ``[windows, MC, window, features]`` or an
    ordered iterable of ``(start, [MC, window, features])`` pairs.  Results
    retain the previous exact weighted-mixture definition; only the lifetime
    of intermediate samples changes.
    """
    if window_starts is not None:
        arrays = np.asarray(window_samples, dtype=np.float64)
        if arrays.ndim != 4:
            raise ValueError(
                "window_samples must have shape [windows, mc, window, features]"
            )
        starts = [int(start) for start in window_starts]
        if len(starts) != arrays.shape[0]:
            raise ValueError("window_starts must match the number of windows")
        window_size, n_features = arrays.shape[2:]
        pairs = zip(starts, arrays)
        if total_length is None:
            if not starts:
                raise ValueError("window samples cannot be empty")
            total_length = max(starts) + window_size
    else:
        pairs = iter(window_samples)
        first = next(pairs, None)
        if first is None:
            raise ValueError("window samples cannot be empty")
        first_start, first_array = first
        first_array = np.asarray(first_array, dtype=np.float64)
        if first_array.ndim != 3:
            raise ValueError(
                "per-window samples must have shape [mc, window, features]"
            )
        window_size, n_features = first_array.shape[1:]
        if total_length is None:
            raise ValueError("total_length is required for an iterable input")

        remaining_pairs = pairs

        def ordered_pairs():
            yield int(first_start), first_array
            yield from remaining_pairs

        pairs = ordered_pairs()

    if position_weights is None:
        position_weights = trapezoid_position_weights(window_size)
    return aggregate_ordered_windows(
        pairs,
        total_length=int(total_length),
        window_size=window_size,
        n_features=n_features,
        position_weights=position_weights,
        quantiles=quantiles,
    )


def _compute_support_diagnostics(obs_mask, context_window=72):
    """Describe each missing run without claiming model calibration.

    Risk tiers are operational heuristics based on gap length and bilateral
    observed context. They are not learned probabilities and should not replace
    held-out evaluation on the user's dataset.
    """
    observed = np.asarray(obs_mask, dtype=bool)
    n_rows, n_features = observed.shape
    gap_length = np.zeros((n_rows, n_features), dtype=np.int32)
    left_context = np.full((n_rows, n_features), np.nan, dtype=np.float64)
    right_context = np.full((n_rows, n_features), np.nan, dtype=np.float64)
    risk = np.full((n_rows, n_features), "observed", dtype=object)

    for feature in range(n_features):
        missing = ~observed[:, feature]
        starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
        ends = np.flatnonzero(missing & ~np.r_[missing[1:], False]) + 1
        for start, end in zip(starts, ends):
            length = int(end - start)
            left = observed[max(0, start - context_window):start, feature]
            right = observed[end:min(n_rows, end + context_window), feature]
            left_fraction = float(left.mean()) if left.size else 0.0
            right_fraction = float(right.mean()) if right.size else 0.0
            gap_length[start:end, feature] = length
            left_context[start:end, feature] = left_fraction
            right_context[start:end, feature] = right_fraction
            if length <= 12 and left_fraction >= 0.75 and right_fraction >= 0.75:
                tier = "low"
            elif length >= 96 or min(left_fraction, right_fraction) < 0.25:
                tier = "high"
            else:
                tier = "moderate"
            risk[start:end, feature] = tier

    return {
        "gap_length": gap_length,
        "left_context_fraction": left_context,
        "right_context_fraction": right_context,
        "heuristic_risk_tier": risk,
    }


def impute(
    csv_paths,
    bundle,
    output_csv=None,
    stride=None,
    n_mc_samples=50,
    timestamp_col=None,
    inference_batch_size=4,
    mc_batch_size=1,
    support_context_window=72,
    *,
    modality_files=None,
    inference_config=None,
):
    """Impute/predict on new CSVs using a trained checkpoint bundle.

    `bundle` may be a path to a checkpoint file or an already-loaded bundle
    dict from `load_bundle`. Writes a tidy long-format CSV: one row per
    (timestamp, feature) with the raw observed value (if any), the imputed
    mean/std, and a 5-95% predictive interval.
    """
    if isinstance(bundle, (str, Path)):
        bundle = load_bundle(bundle)

    model = bundle["model"]
    device = bundle["device"]
    data_schema = bundle["data_schema"]
    preprocessing = bundle["preprocessing"]
    target_cols = data_schema.target_cols
    aux_cols = data_schema.auxiliary_cols
    window_size = bundle["window_size"]
    scaler_target = bundle["scaler_target"]
    scaler_aux = bundle["scaler_aux"]
    aux_mask_channel = preprocessing.aux_mask_channel
    output_transforms = bundle["target_output_transforms"]

    if isinstance(inference_config, dict):
        inference_config = InferenceConfig.from_dict(inference_config)
    if inference_config is None:
        inference_config = InferenceConfig(
            stride=stride,
            n_mc_samples=n_mc_samples,
            inference_batch_size=inference_batch_size,
            mc_batch_size=mc_batch_size,
            support_context_window=support_context_window,
        )
    stride = inference_config.stride
    n_mc_samples = inference_config.n_mc_samples
    inference_batch_size = inference_config.inference_batch_size
    mc_batch_size = inference_config.mc_batch_size
    support_context_window = inference_config.support_context_window
    lower_q = inference_config.interval_lower
    upper_q = inference_config.interval_upper

    ts_col = timestamp_col or data_schema.timestamp_col
    if not ts_col:
        raise ValueError("timestamp_col not found in bundle config; pass timestamp_col= explicitly")

    # Reuse the training window stride unless the caller explicitly overrides
    # it.  The 26e reference uses stride=24; falling back to window//2 would
    # silently change the overlap geometry during inference.
    if stride is None:
        stride = bundle.get("stride") or max(1, window_size // 2)
    if stride > window_size:
        raise ValueError("stride cannot exceed window_size")

    if isinstance(modality_files, dict):
        modality_files = ModalityInputs.from_dict(modality_files)
    elif isinstance(modality_files, ModalityFiles):
        modality_files = ModalityInputs.from_files(modality_files)
    if modality_files is not None:
        frame, _ = load_modality_frame(
            modality_files,
            ts_col,
            expected_schema=data_schema,
        )
    else:
        if not csv_paths:
            raise ValueError("Provide csv_paths or modality_files for inference")
        time_grid = bundle.get("time_grid", {})
        frame = load_frame(
            csv_paths,
            ts_col,
            target_cols,
            aux_cols,
            expected_frequency=time_grid.get("frequency"),
            time_grid_policy=time_grid.get("policy", "row_order"),
            duplicate_timestamp_policy=time_grid.get(
                "duplicate_timestamp_policy", "error"
            ),
        )
    n = len(frame)
    starts = compute_window_starts(n, window_size, stride)
    if not starts:
        raise ValueError(f"Series length {n} is shorter than window_size={window_size}")

    target_raw = frame[target_cols].to_numpy(dtype=np.float64)
    # Reproduce the training-time observation states so non-detects are fed to
    # the encoder the same way they were during fitting, and are reported as
    # interval-constrained rather than as ordinary observations.
    censoring = bundle.get("censoring") or CensoringConfig()
    if isinstance(censoring, dict):
        censoring = CensoringConfig.from_dict(censoring)
    state_full = build_state_matrix(target_raw, data_schema, censoring)
    censor_mask_full = (state_full == STATE_CENSORED).astype(np.float32)
    target_input_raw = apply_input_fill(target_raw, state_full, data_schema, censoring)
    target_model_space = transform_targets(target_input_raw, data_schema, preprocessing)
    aux_raw = frame[aux_cols].to_numpy(dtype=np.float64) if aux_cols else np.zeros((n, 0))
    aux_model_space = transform_auxiliary(aux_raw, preprocessing)
    obs_mask_full = (state_full == STATE_OBSERVED).astype(np.float32)
    # A non-detect is known to the encoder even though it is not a point value.
    known_mask_full = obs_mask_full + censor_mask_full
    # The bundle stores limits in scaled space because that is where the
    # decoder is supervised, but the window aggregators emit model space.
    censor_threshold_scaled = _bundle_censor_thresholds(bundle, data_schema)
    censor_threshold_model = (
        censor_threshold_scaled * scaler_target.std_ + scaler_target.mean_
    )

    target_scaled = np.nan_to_num(scaler_target.transform(target_model_space), nan=0.0)
    aux_scaled = scaler_aux.transform(aux_model_space) if aux_cols else aux_model_space
    aux_observed = ~np.isnan(aux_raw)

    n_batches = math.ceil(len(starts) / inference_batch_size)
    print(
        f"[graph-temporal-vae] {n} rows -> {len(starts)} windows ({n_batches} batches), "
        f"{len(target_cols)} targets, {n_mc_samples} MC samples, stride={stride}, device={device}"
    )

    # Keep two statistical streams, but aggregate each window immediately.
    # The noise-free decoder means drive the point estimate; full generative
    # draws drive variance and quantiles.  A timestamp is finalized once the
    # next window start passes it, so raw MC history is bounded by the active
    # overlap rather than the number of windows in the full series.
    position_weights = trapezoid_position_weights(window_size)
    mean_aggregator = StreamingWindowAggregator(
        total_length=n,
        window_size=window_size,
        n_features=len(target_cols),
        position_weights=position_weights,
        quantiles=(),
    )
    distribution_aggregator = StreamingWindowAggregator(
        total_length=n,
        window_size=window_size,
        n_features=len(target_cols),
        position_weights=position_weights,
        quantiles=(lower_q, upper_q),
    )

    model.eval()
    with torch.no_grad():
        batch_starts_list = range(0, len(starts), inference_batch_size)
        for batch_start in tqdm(
            batch_starts_list,
            desc="impute windows",
            total=n_batches,
            disable=not is_interactive(),
        ):
            batch_starts = starts[batch_start:batch_start + inference_batch_size]
            masks = np.stack([
                known_mask_full[start:start + window_size]
                for start in batch_starts
            ])
            xs = np.stack([
                target_scaled[start:start + window_size] * masks[index]
                for index, start in enumerate(batch_starts)
            ])
            conds = np.stack([
                make_condition(
                    aux_scaled[start:start + window_size],
                    aux_observed[start:start + window_size],
                    aux_mask_channel,
                )
                for start in batch_starts
            ])

            x_t = torch.from_numpy(xs).float().to(device)
            cond_t = torch.from_numpy(conds).float().to(device)
            mask_t = torch.from_numpy(masks).float().to(device)
            result = model.compute_uncertainty(
                x_t,
                cond_t,
                mask_t,
                n_samples=n_mc_samples,
                dist_type=bundle.get("uncertainty_dist_type", "gaussian"),
                return_samples=True,
                mc_batch_size=mc_batch_size,
            )
            pred_mean_scaled = result[0].cpu().numpy()
            samples_scaled = result[-2].cpu().numpy()
            pred_mean_model = (
                pred_mean_scaled * scaler_target.std_[None, None, :]
                + scaler_target.mean_[None, None, :]
            )
            samples_model = (
                samples_scaled * scaler_target.std_[None, None, None, :]
                + scaler_target.mean_[None, None, None, :]
            )
            for index, start in enumerate(batch_starts):
                mean_aggregator.add(
                    start, pred_mean_model[index][None, :, :]
                )
                distribution_aggregator.add(start, samples_model[:, index])

    mean_agg = mean_aggregator.finish()
    dist_agg = distribution_aggregator.finish()
    # Non-detects are reported under the constraint that produced them, so the
    # mean and interval respect the detection limit instead of contradicting it.
    mean_model, variance_model, p_below_limit = truncate_below_limit(
        mean_agg["mean"], dist_agg["variance"], censor_threshold_model, censor_mask_full == 1
    )
    mean_out, std_out, quantiles_out = summary_to_output_scale(
        mean_model, variance_model, dist_agg["quantiles"], output_transforms
    )
    q_lower_out = quantiles_out[lower_q]
    q_upper_out = quantiles_out[upper_q]

    # Restore observed values in the public/raw output space; there's no
    # imputation uncertainty there.  The input CSV may already be transformed
    # (e.g. the 26e artifact stores log1p targets), so target_raw is not
    # necessarily the physical output value.
    observed_output = observed_targets_to_output(target_raw, data_schema, preprocessing)
    mean_out = np.where(obs_mask_full == 1, observed_output, mean_out)
    std_out = np.where(obs_mask_full == 1, 0.0, std_out)
    q_lower_out = np.where(obs_mask_full == 1, observed_output, q_lower_out)
    q_upper_out = np.where(obs_mask_full == 1, observed_output, q_upper_out)

    # A censored cell is not usable context for gap-support diagnostics: it
    # constrains the value without pinning it down.
    support = _compute_support_diagnostics(
        obs_mask_full, context_window=support_context_window
    )
    # Quantiles for a non-detect cannot exceed the limit it was censored at.
    limit_output = inverse_targets(
        censor_threshold_model[None, :], data_schema, preprocessing
    )[0]
    limit_output = np.where(np.isfinite(censor_threshold_model), limit_output, np.nan)
    censored_cells = censor_mask_full == 1
    for array in (q_upper_out, q_lower_out):
        np.copyto(
            array,
            np.minimum(array, np.broadcast_to(limit_output[None, :], array.shape)),
            where=censored_cells & np.isfinite(limit_output)[None, :],
        )

    state_labels = np.select(
        [state_full == STATE_OBSERVED, state_full == STATE_CENSORED],
        ["observed", "censored"],
        default="missing",
    )
    frames = []
    for j, col in enumerate(target_cols):
        frames.append(pd.DataFrame({
            "timestamp": frame.index,
            "feature": col,
            "observed": observed_output[:, j],
            "observation_state": state_labels[:, j],
            "detection_limit": (
                limit_output[j] if np.isfinite(limit_output[j]) else np.nan
            ),
            "p_below_limit": p_below_limit[:, j],
            "is_imputed": obs_mask_full[:, j] == 0,
            "imputed_mean": mean_out[:, j],
            "imputed_std": std_out[:, j],
            "q_lower": q_lower_out[:, j],
            "q_upper": q_upper_out[:, j],
            "interval_lower": lower_q,
            "interval_upper": upper_q,
            **({"q05": q_lower_out[:, j], "q95": q_upper_out[:, j]}
               if lower_q == 0.05 and upper_q == 0.95 else {}),
            "gap_length": support["gap_length"][:, j],
            "left_context_fraction": support["left_context_fraction"][:, j],
            "right_context_fraction": support["right_context_fraction"][:, j],
            "heuristic_risk_tier": support["heuristic_risk_tier"][:, j],
        }))
    result_df = pd.concat(frames, ignore_index=True)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
    return result_df
