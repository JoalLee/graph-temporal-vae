"""Checkpoint bundle inspection and integrity reporting."""

from pathlib import Path

import torch

from .infer import load_bundle


def inspect_bundle(bundle):
    """Validate a checkpoint and return a JSON-serializable summary.

    Passing a path performs the complete load path, including schema checks,
    strict state-dict loading, and model construction. A previously loaded
    bundle returned by :func:`graph_tcn_vae.load_bundle` is also accepted.
    """
    if isinstance(bundle, (str, Path)):
        loaded = load_bundle(bundle, device=torch.device("cpu"))
        bundle_path = str(bundle)
    elif isinstance(bundle, dict) and "model" in bundle:
        loaded = bundle
        bundle_path = loaded.get("bundle_path")
    else:
        raise TypeError("bundle must be a checkpoint path or a loaded bundle dictionary")

    schema = loaded["data_schema"]
    preprocessing = loaded["preprocessing"]
    model = loaded["model"]
    psd_range = None
    if schema.psd_diameters_nm:
        psd_range = {
            "minimum_nm": min(schema.psd_diameters_nm),
            "maximum_nm": max(schema.psd_diameters_nm),
        }

    return {
        "valid": True,
        "bundle_path": bundle_path,
        "versions": {
            "bundle": loaded["bundle_version"],
            "architecture": loaded["architecture_version"],
            "state_dict_format": loaded["state_dict_format_version"],
            "data_schema": schema.schema_version,
        },
        "data_interface": loaded["data_interface"],
        "window": {
            "size": loaded["window_size"],
            "stride": loaded["stride"],
        },
        "dimensions": {
            "targets": schema.target_dim,
            "chemistry": schema.n_chem,
            "psd": len(schema.psd_cols),
            "meteorology": schema.aux_dim,
            "condition": (
                schema.aux_dim * (2 if preprocessing.aux_mask_channel else 1)
            ),
        },
        "columns": {
            "chemistry": list(schema.chemistry_cols),
            "psd": list(schema.psd_cols),
            "meteorology": list(schema.meteorology_cols),
        },
        "psd": {
            "diameters_nm": list(schema.psd_diameters_nm),
            "range": psd_range,
        },
        "time_grid": {
            "timestamp_col": schema.timestamp_col,
            "frequency": schema.frequency,
            "timezone": schema.timezone,
            "policy": schema.time_grid_policy,
            "duplicate_timestamp_policy": schema.duplicate_timestamp_policy,
        },
        "preprocessing": preprocessing.to_dict(),
        "model": {
            "class": type(model).__name__,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "kwargs": dict(loaded.get("model_kwargs", {})),
        },
        "training": {
            "validation_metric": loaded.get("training_config", {}).get(
                "validation_metric"
            ),
            "selection_mask_mode": loaded.get("training_config", {}).get(
                "selection_mask_mode"
            ),
            "seed": loaded.get("training_config", {}).get("seed"),
        },
    }
