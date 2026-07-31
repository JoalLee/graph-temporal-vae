"""High-level multimodal training and inference interface.

These functions are the preferred Python seam for package users. They accept
CSV paths, pandas DataFrames, or mixtures of both and hide schema discovery,
column ordering, preprocessing persistence, and tensor construction.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import TrainConfig
from .contracts import (
    InferenceConfig,
    ModalityFiles,
    ModalityInputs,
    PreprocessingConfig,
)
from .data import load_modality_frame
from .infer import impute, load_bundle
from .train import train_from_config


def _coerce_modality_inputs(
    chemistry=None,
    psd=None,
    meteorology=None,
    *,
    data=None,
) -> ModalityInputs:
    if data is not None:
        if any(value is not None for value in (chemistry, psd, meteorology)):
            raise ValueError(
                "Pass either data=ModalityInputs/ModalityFiles or individual modality arguments, not both"
            )
        if isinstance(data, ModalityInputs):
            return data
        if isinstance(data, ModalityFiles):
            return ModalityInputs.from_files(data)
        if isinstance(data, Mapping):
            return ModalityInputs.from_dict(dict(data))
        raise TypeError("data must be ModalityInputs, ModalityFiles, or a compatible mapping")
    return ModalityInputs(
        chemistry=chemistry,
        psd=psd,
        meteorology=meteorology,
    )


def _coerce_train_config(config, *, timestamp_col=None, preprocessing=None) -> TrainConfig:
    if config is None:
        resolved = TrainConfig(
            timestamp_col=timestamp_col or "time",
            preprocessing=preprocessing,
        )
    elif isinstance(config, TrainConfig):
        resolved = deepcopy(config)
    elif isinstance(config, Mapping):
        resolved = TrainConfig(**dict(config))
    elif isinstance(config, (str, Path)):
        resolved = TrainConfig.from_json(config)
    else:
        raise TypeError("config must be TrainConfig, a mapping, a JSON path, or None")

    # Explicit high-level data arguments are authoritative. Runtime DataFrames
    # must never be serialized into the checkpoint's training config.
    resolved.csv = []
    resolved.target_cols = []
    resolved.aux_cols = []
    resolved.modality_files = None
    if timestamp_col is not None:
        resolved.timestamp_col = timestamp_col
    if preprocessing is not None:
        if isinstance(preprocessing, Mapping):
            preprocessing = PreprocessingConfig.from_dict(dict(preprocessing))
        if not isinstance(preprocessing, PreprocessingConfig):
            raise TypeError("preprocessing must be PreprocessingConfig or a compatible mapping")
        resolved.preprocessing = preprocessing
        resolved.scaler_fit_scope = preprocessing.fit_scope
        resolved.aux_mask_channel = preprocessing.aux_mask_channel
    return resolved


def fit_multimodal(
    chemistry=None,
    psd=None,
    meteorology=None,
    *,
    data=None,
    output,
    config=None,
    timestamp_col=None,
    preprocessing=None,
):
    """Train from modality paths/DataFrames and return the loaded bundle.

    Parameters are deliberately small: data sources, an output checkpoint, and
    an optional :class:`TrainConfig` (or JSON/mapping equivalent). Chem columns
    are placed before numerically sorted PSD bins; the resolved schema and all
    fitted preprocessing statistics are stored in the checkpoint.
    """
    inputs = _coerce_modality_inputs(
        chemistry,
        psd,
        meteorology,
        data=data,
    )
    train_config = _coerce_train_config(
        config,
        timestamp_col=timestamp_col,
        preprocessing=preprocessing,
    )
    frame, schema = load_modality_frame(
        inputs,
        train_config.timestamp_col,
        expected_frequency=train_config.expected_frequency,
        time_grid_policy=train_config.time_grid_policy,
        duplicate_timestamp_policy=train_config.duplicate_timestamp_policy,
        add_time_cyclical_features=train_config.add_time_cyclical_features,
    )
    best_validation = train_from_config(
        train_config,
        str(output),
        prepared_data=(frame, schema),
    )
    loaded = load_bundle(output)
    loaded["bundle_path"] = str(output)
    loaded["best_validation"] = float(best_validation)
    return loaded


def impute_multimodal(
    chemistry=None,
    psd=None,
    meteorology=None,
    *,
    data=None,
    bundle,
    output=None,
    config: Optional[InferenceConfig] = None,
    timestamp_col=None,
):
    """Impute modality paths/DataFrames and return a tidy pandas DataFrame."""
    inputs = _coerce_modality_inputs(
        chemistry,
        psd,
        meteorology,
        data=data,
    )
    if isinstance(config, Mapping):
        config = InferenceConfig.from_dict(dict(config))
    if config is not None and not isinstance(config, InferenceConfig):
        raise TypeError("config must be InferenceConfig, a compatible mapping, or None")
    return impute(
        None,
        bundle,
        output,
        timestamp_col=timestamp_col,
        modality_files=inputs,
        inference_config=config,
    )


def validate_multimodal_data(
    chemistry=None,
    psd=None,
    meteorology=None,
    *,
    data=None,
    timestamp_col="time",
    expected_frequency=None,
    time_grid_policy="strict",
    duplicate_timestamp_policy="error",
    expected_schema=None,
):
    """Validate modality inputs and return a JSON-serializable report."""
    inputs = _coerce_modality_inputs(
        chemistry,
        psd,
        meteorology,
        data=data,
    )
    frame, schema = load_modality_frame(
        inputs,
        timestamp_col,
        expected_schema=expected_schema,
        expected_frequency=expected_frequency,
        time_grid_policy=time_grid_policy,
        duplicate_timestamp_policy=duplicate_timestamp_policy,
    )
    modality_missing = {
        "chemistry": frame[schema.chemistry_cols].isna().mean().to_dict()
        if schema.chemistry_cols
        else {},
        "psd": frame[schema.psd_cols].isna().mean().to_dict()
        if schema.psd_cols
        else {},
        "meteorology": frame[schema.meteorology_cols].isna().mean().to_dict()
        if schema.meteorology_cols
        else {},
    }
    return {
        "valid": True,
        "rows": len(frame),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "frequency": schema.frequency,
        "timezone": schema.timezone,
        "data_schema": schema.to_dict(),
        "modality_missing_fraction": modality_missing,
    }
