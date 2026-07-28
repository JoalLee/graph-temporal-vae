"""Left-censored (below detection limit) observation handling.

Trace-level measurements are frequently reported as a non-detect rather than a
concentration.  A non-detect is *not* a missing value: it carries the
information ``0 <= y <= MDL``.  Encoding it as a literal zero and feeding it to
a reconstruction likelihood tells the model the concentration is exactly zero,
which biases the predictive mean downward and shrinks the predictive variance.

This module keeps the three observation states explicit:

``OBSERVED``
    A real measurement; supervised with the ordinary reconstruction likelihood.
``CENSORED``
    A non-detect; supervised with a Tobit term that places predictive mass
    below the detection limit instead of fitting a point value.
``MISSING``
    Genuinely absent; excluded from the likelihood entirely.

Thresholds are supplied per target column in the *physical* units of the input
CSV.  See ``scripts/build_mdl_table.py`` for building one from instrument
specifications.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .contracts import DataSchema, PreprocessingConfig
from .preprocessing import transform_targets

# Observation-state codes. Kept as small ints so they survive a round trip
# through numpy arrays and checkpoint metadata without a custom dtype.
STATE_MISSING = 0
STATE_OBSERVED = 1
STATE_CENSORED = 2

SUPPORTED_DETECT = {"zero", "at_or_below_threshold"}
SUPPORTED_INPUT_FILL = {"half_threshold", "threshold", "zero"}
SUPPORTED_LOSS = {"tobit", "ignore"}


@dataclass
class CensoringConfig:
    """Contract for detecting and supervising below-detection-limit values.

    ``thresholds`` maps a target column name to its detection limit in the
    same physical units as the CSV.  Columns absent from the mapping are never
    treated as censored, so a partial table is valid and safe.

    ``detect='zero'`` treats an exact zero as a non-detect, which matches
    instruments that report a zero when a peak is not resolved.
    ``detect='at_or_below_threshold'`` additionally censors any positive value
    at or below the threshold, for data where sub-MDL values are reported
    rather than zeroed.

    ``input_fill`` chooses the point value shown to the encoder for a censored
    cell.  ``half_threshold`` (MDL/2) is the conventional substitution; the
    likelihood still treats the cell as an interval, so this only affects what
    the encoder reads, never what the decoder is scored against.

    ``loss='ignore'`` downgrades censored cells to missing.  It exists as an
    ablation and as an escape hatch when a threshold table is not trusted.
    """

    enabled: bool = False
    thresholds: Dict[str, float] = field(default_factory=dict)
    detect: str = "zero"
    input_fill: str = "half_threshold"
    loss: str = "tobit"

    def __post_init__(self):
        if self.detect not in SUPPORTED_DETECT:
            raise ValueError(
                f"detect must be one of {sorted(SUPPORTED_DETECT)}, got {self.detect!r}"
            )
        if self.input_fill not in SUPPORTED_INPUT_FILL:
            raise ValueError(
                f"input_fill must be one of {sorted(SUPPORTED_INPUT_FILL)}, got {self.input_fill!r}"
            )
        if self.loss not in SUPPORTED_LOSS:
            raise ValueError(
                f"loss must be one of {sorted(SUPPORTED_LOSS)}, got {self.loss!r}"
            )
        cleaned: Dict[str, float] = {}
        for column, value in dict(self.thresholds).items():
            if value is None:
                continue
            threshold = float(value)
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError(
                    f"Censoring threshold for {column!r} must be a positive finite "
                    f"number, got {value!r}"
                )
            cleaned[str(column)] = threshold
        self.thresholds = cleaned

    @property
    def active(self) -> bool:
        """True when the config will actually censor anything."""
        return bool(self.enabled and self.thresholds)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "CensoringConfig":
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_thresholds(schema: DataSchema, config: Optional[CensoringConfig]) -> np.ndarray:
    """Per-target-column thresholds in physical units; NaN where uncensored."""
    thresholds = np.full(schema.target_dim, np.nan, dtype=np.float64)
    if config is None or not config.active:
        return thresholds
    known = set(schema.target_cols)
    unknown = sorted(set(config.thresholds) - known)
    if unknown:
        raise ValueError(
            f"Censoring thresholds reference columns outside the target schema: {unknown}"
        )
    for index, column in enumerate(schema.target_cols):
        if column in config.thresholds:
            thresholds[index] = config.thresholds[column]
    return thresholds


def build_state_matrix(
    raw_targets,
    schema: DataSchema,
    config: Optional[CensoringConfig],
) -> np.ndarray:
    """Classify every raw target cell as MISSING, OBSERVED, or CENSORED."""
    values = np.asarray(raw_targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != schema.target_dim:
        raise ValueError(
            f"raw_targets must be (rows, {schema.target_dim}), got {values.shape}"
        )
    state = np.where(np.isnan(values), STATE_MISSING, STATE_OBSERVED).astype(np.int8)
    if config is None or not config.active or config.loss == "ignore":
        if config is not None and config.active and config.loss == "ignore":
            censored = _detect_censored(values, schema, config)
            state[censored] = STATE_MISSING
        return state
    state[_detect_censored(values, schema, config)] = STATE_CENSORED
    return state


def _detect_censored(values, schema: DataSchema, config: CensoringConfig) -> np.ndarray:
    thresholds = resolve_thresholds(schema, config)
    has_threshold = np.isfinite(thresholds)
    observed = ~np.isnan(values)
    censored = observed & has_threshold[None, :] & (values == 0.0)
    if config.detect == "at_or_below_threshold":
        # NaN thresholds compare False, so uncensored columns stay untouched.
        with np.errstate(invalid="ignore"):
            censored |= observed & (values <= thresholds[None, :])
    return censored


def apply_input_fill(
    raw_targets,
    state: np.ndarray,
    schema: DataSchema,
    config: Optional[CensoringConfig],
) -> np.ndarray:
    """Replace censored cells with the point value shown to the encoder.

    The returned array is only ever used to build the encoder input.  The
    decoder is scored against the censoring interval, not against this value.
    """
    values = np.asarray(raw_targets, dtype=np.float64).copy()
    if config is None or not config.active or config.input_fill == "zero":
        return values
    thresholds = resolve_thresholds(schema, config)
    fill = thresholds * (0.5 if config.input_fill == "half_threshold" else 1.0)
    censored = state == STATE_CENSORED
    if not censored.any():
        return values
    filled = np.broadcast_to(fill[None, :], values.shape)
    return np.where(censored, filled, values)


def model_space_thresholds(
    schema: DataSchema,
    preprocessing: PreprocessingConfig,
    scaler,
    config: Optional[CensoringConfig],
) -> np.ndarray:
    """Map physical thresholds into the scaled space the decoder predicts in.

    A censored cell is supervised as ``P(y <= c)``, so ``c`` must live in the
    same space as ``recon_mean``: physical -> modality transform -> affine
    scaling.  Columns without a threshold return NaN and are never used,
    because the censor mask is empty for them.
    """
    thresholds = resolve_thresholds(schema, config)
    known = np.isfinite(thresholds)
    if not known.any():
        return thresholds
    # transform_targets rejects negatives and needs a full-width row; fill the
    # unknown columns with a harmless zero and discard them afterwards.
    row = np.where(known, thresholds, 0.0)[None, :]
    model_space = transform_targets(row, schema, preprocessing)
    scaled = (model_space - scaler.center_[None, :]) / scaler.scale_[None, :]
    return np.where(known, scaled[0], np.nan)


def censoring_report(
    state: np.ndarray,
    schema: DataSchema,
    config: Optional[CensoringConfig],
) -> Dict[str, Any]:
    """Summarize the observation-state split for logging and bundle metadata."""
    state = np.asarray(state)
    total = int(state.size)
    counts = {
        "observed": int((state == STATE_OBSERVED).sum()),
        "censored": int((state == STATE_CENSORED).sum()),
        "missing": int((state == STATE_MISSING).sum()),
    }
    # Every target column, not just censored ones: a caller building a
    # per-feature missingness table (e.g. to plot against held-out accuracy)
    # needs an entry for columns like PSD bins that are never censored, or
    # they silently read back as "0% missing" instead of missing entirely.
    per_column: Dict[str, Dict[str, float]] = {}
    thresholds = resolve_thresholds(schema, config)
    for index, column in enumerate(schema.target_cols):
        column_state = state[:, index]
        per_column[column] = {
            "censored_fraction": float((column_state == STATE_CENSORED).mean()),
            "missing_fraction": float((column_state == STATE_MISSING).mean()),
            "threshold": float(thresholds[index]) if np.isfinite(thresholds[index]) else None,
        }
    n_censored_columns = sum(1 for stats in per_column.values() if stats["censored_fraction"] > 0)
    return {
        "cells": total,
        "counts": counts,
        "fractions": {key: (value / total if total else 0.0) for key, value in counts.items()},
        "n_censored_columns": n_censored_columns,
        "per_column": per_column,
    }


def high_censoring_columns(report: Dict[str, Any], threshold_fraction: float = 0.9) -> List[str]:
    """Columns censored so often that they carry almost no concentration signal."""
    return sorted(
        column
        for column, stats in report.get("per_column", {}).items()
        if stats["censored_fraction"] >= threshold_fraction
    )
