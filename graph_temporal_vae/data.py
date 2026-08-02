"""CSV loading, normalization, and windowing for the train/impute pipeline.

This module turns an arbitrary tabular time-series CSV (a timestamp column
plus named target/auxiliary columns, NaN = missing) into the
``(target, cond, mask)`` windows the ``ImputationVAE_Graph`` forward pass
expects. Column selection is config-driven (explicit name lists), not
positional, so it works for any dataset shape.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .contracts import TIME_CYCLICAL_COLS, DataSchema, ModalityFiles, ModalityInputs


SUPPORTED_TARGET_TRANSFORMS = {"none", "log1p"}
WIND_RAW_COLUMNS = ("WS", "WD")
WIND_COMPONENT_COLUMNS = ("wind_u", "wind_v")
WIND_ENCODING = "ws_wd_to_uv_v1"

# A trailing underscore is the QC convention used by the Minion source files
# for a numeric result that is below the detection limit.  Keep the numeric
# payload and carry the marker separately; replacing it with NaN or zero would
# destroy the distinction between a non-detect and a genuinely absent/zero
# observation.
_CENSORED_NUMERIC_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)_\s*$"
)


def _parse_censored_numeric_series(series):
    """Parse numeric values and return a boolean mask for trailing ``_`` markers."""
    text = series.astype("string")
    marker = text.str.fullmatch(_CENSORED_NUMERIC_RE).fillna(False).to_numpy(dtype=bool)
    if marker.any():
        text = text.str.replace(_CENSORED_NUMERIC_RE, r"\1", regex=True)
    numeric = pd.to_numeric(text, errors="raise")
    return numeric, marker


def extract_censor_marker_mask(frame, target_cols):
    """Return the loader-produced marker mask in ``target_cols`` order.

    The mask is stored in ``DataFrame.attrs`` so the numeric frame remains
    compatible with the existing public data contract.  Frames created by
    callers that have no marker metadata receive an all-False mask.
    """
    target_cols = list(target_cols)
    expected = (len(frame), len(target_cols))
    raw_mask = frame.attrs.get("censor_marker_mask")
    if raw_mask is None:
        return np.zeros(expected, dtype=bool)
    raw_mask = np.asarray(raw_mask, dtype=bool)
    raw_cols = list(frame.attrs.get("censor_marker_columns", target_cols))
    if raw_mask.ndim != 2 or raw_mask.shape[0] != len(frame):
        raise ValueError(
            "censor marker mask must have one row per loaded timestamp: "
            f"expected_rows={len(frame)}, got={raw_mask.shape}"
        )
    if len(raw_cols) != raw_mask.shape[1] or len(set(raw_cols)) != len(raw_cols):
        raise ValueError("censor marker metadata has invalid column ordering")
    missing = sorted(set(target_cols) - set(raw_cols))
    extra = sorted(set(raw_cols) - set(target_cols))
    if missing or extra:
        raise ValueError(
            "censor marker columns do not match loaded targets: "
            f"missing={missing}, extra={extra}"
        )
    order = [raw_cols.index(column) for column in target_cols]
    return raw_mask[:, order]


def compute_time_cyclical_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour/day-of-week/month sin-cos features derived from a timestamp index.

    Column order matches ``TIME_CYCLICAL_COLS``. These are computed from the
    timestamp only, so they are always fully observed and reproduce
    identically at training and inference time -- unlike meteorology, there
    is no gap to fill or mismatch to guard against.
    """
    hour_frac = index.hour + index.minute / 60.0
    dow = index.dayofweek
    month = index.month - 1
    hour_angle = 2 * np.pi * hour_frac / 24.0
    dow_angle = 2 * np.pi * dow / 7.0
    month_angle = 2 * np.pi * month / 12.0
    return pd.DataFrame(
        {
            "time_hour_sin": np.sin(hour_angle),
            "time_hour_cos": np.cos(hour_angle),
            "time_dow_sin": np.sin(dow_angle),
            "time_dow_cos": np.cos(dow_angle),
            "time_month_sin": np.sin(month_angle),
            "time_month_cos": np.cos(month_angle),
        },
        index=index,
    )[TIME_CYCLICAL_COLS]


def transform_target_values(array, transform="none"):
    """Transform target values before fitting/scaling the model input."""
    if transform not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(
            f"Unsupported target transform {transform!r}; "
            f"choose from {sorted(SUPPORTED_TARGET_TRANSFORMS)}"
        )
    values = np.asarray(array, dtype=np.float64)
    if transform == "none":
        return values.copy()
    finite = np.isfinite(values)
    if np.any(values[finite] < 0):
        raise ValueError("target_transform='log1p' requires non-negative finite target values")
    out = values.copy()
    out[finite] = np.log1p(out[finite])
    return out


def inverse_target_values(array, transform="none"):
    """Map model-space target values back to the physical output scale."""
    if transform not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(
            f"Unsupported target transform {transform!r}; "
            f"choose from {sorted(SUPPORTED_TARGET_TRANSFORMS)}"
        )
    values = np.asarray(array, dtype=np.float64)
    if transform == "none":
        return values.copy()
    return np.expm1(values).clip(min=0.0)


class NaNAwareStandardScaler:
    """Per-feature z-score scaler that ignores NaNs when fitting.

    Persisted via ``to_dict``/``from_dict`` so training and inference always
    share the exact same statistics instead of each re-fitting from scratch.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, array):
        values = np.asarray(array, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Scaler input must be 2-D, got shape {values.shape}")
        if values.shape[1] == 0:
            self.mean_ = np.zeros(0, dtype=np.float64)
            self.std_ = np.ones(0, dtype=np.float64)
            return self
        all_nan = np.isnan(values).all(axis=0)
        if all_nan.any():
            indices = np.flatnonzero(all_nan).tolist()
            raise ValueError(
                f"Cannot fit scaler: all values are missing in column indices {indices}"
            )
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        self.mean_ = mean
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, array):
        scaled = (array - self.mean_) / self.std_
        return np.nan_to_num(scaled, nan=0.0)

    def fit_transform(self, array):
        return self.fit(array).transform(array)

    def inverse_transform(self, array):
        return array * self.std_ + self.mean_

    def inverse_transform_std(self, std_array):
        return std_array * self.std_

    def to_dict(self):
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, state):
        scaler = cls()
        scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
        scaler.std_ = np.asarray(state["std"], dtype=np.float64)
        return scaler


def _resolve_time_grid(index, expected_frequency=None):
    """Return a concrete pandas offset for an observed timestamp index."""
    if expected_frequency:
        try:
            return pd.tseries.frequencies.to_offset(expected_frequency)
        except ValueError as exc:
            raise ValueError(
                f"Invalid expected_frequency={expected_frequency!r}"
            ) from exc
    if len(index) < 2:
        return None
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return None
    mode = diffs.mode()
    if mode.empty or mode.iloc[0] <= pd.Timedelta(0):
        return None
    return pd.tseries.frequencies.to_offset(mode.iloc[0])


def _source_label(source, index=None):
    if isinstance(source, pd.DataFrame):
        suffix = f":{index}" if index is not None else ""
        return f"<dataframe{suffix}>"
    return str(source)


def _read_tabular_source(source, *, header_only=False):
    if isinstance(source, pd.DataFrame):
        return source.iloc[:0].copy() if header_only else source.copy()
    return pd.read_csv(source, nrows=0 if header_only else None)


def _matching_column(columns, wanted):
    """Find one column by case-insensitive name, rejecting ambiguous headers."""
    matches = [
        column
        for column in columns
        if isinstance(column, str) and column.casefold() == wanted.casefold()
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple columns match {wanted!r}: {matches}")
    return matches[0] if matches else None


def canonicalize_wind_column_names(columns):
    """Replace a raw ``WS``/``WD`` pair with the canonical ``wind_u``/``wind_v`` pair."""
    columns = list(columns)
    raw_speed = _matching_column(columns, WIND_RAW_COLUMNS[0])
    raw_direction = _matching_column(columns, WIND_RAW_COLUMNS[1])
    if raw_speed is None and raw_direction is None:
        return columns
    if raw_speed is None or raw_direction is None:
        # Leave a one-sided pair untouched so generic datasets can still use a
        # standalone WS or WD column. A schema that requires wind_u/wind_v will
        # fail later with the normal missing-column error.
        return columns

    vector_u = _matching_column(columns, WIND_COMPONENT_COLUMNS[0])
    vector_v = _matching_column(columns, WIND_COMPONENT_COLUMNS[1])
    if vector_u is not None or vector_v is not None:
        raise ValueError(
            "Meteorology contains both raw WS/WD and derived wind_u/wind_v; "
            "provide only one wind representation"
        )

    output = []
    inserted = False
    raw_columns = {raw_speed, raw_direction}
    for column in columns:
        if column in raw_columns:
            if not inserted:
                output.extend(WIND_COMPONENT_COLUMNS)
                inserted = True
        else:
            output.append(column)
    return output


def canonicalize_wind_columns(frame, auxiliary_columns=None):
    """Convert raw meteorological WS/WD values into circular-safe vector components.

    The conversion follows the research pipeline's meteorological convention:
    ``theta = (270 - WD)`` in mathematical-angle degrees, followed by
    ``wind_u = WS*cos(theta)`` and ``wind_v = WS*sin(theta)``.  Only columns
    named in ``auxiliary_columns`` are eligible for conversion, so a target
    column with a coincidental name is never rewritten.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    auxiliary_columns = list(frame.columns if auxiliary_columns is None else auxiliary_columns)
    output_auxiliary_columns = canonicalize_wind_column_names(auxiliary_columns)
    if output_auxiliary_columns == auxiliary_columns:
        return frame

    raw_speed = _matching_column(auxiliary_columns, WIND_RAW_COLUMNS[0])
    raw_direction = _matching_column(auxiliary_columns, WIND_RAW_COLUMNS[1])
    if raw_speed not in frame.columns or raw_direction not in frame.columns:
        raise ValueError(
            "WS/WD were requested as auxiliary columns but could not be found "
            "in the loaded frame"
        )

    original_columns = list(frame.columns)
    output = frame.copy()
    speed = pd.to_numeric(output[raw_speed], errors="coerce").to_numpy(dtype=float)
    direction = pd.to_numeric(output[raw_direction], errors="coerce").to_numpy(dtype=float)
    direction_rad = np.deg2rad((270.0 - direction) % 360.0)
    output[WIND_COMPONENT_COLUMNS[0]] = speed * np.cos(direction_rad)
    output[WIND_COMPONENT_COLUMNS[1]] = speed * np.sin(direction_rad)

    keep_columns = []
    inserted = False
    raw_columns = {raw_speed, raw_direction}
    for column in original_columns:
        if column in raw_columns:
            if not inserted:
                keep_columns.extend(WIND_COMPONENT_COLUMNS)
                inserted = True
        else:
            keep_columns.append(column)
    output = output.loc[:, keep_columns]
    output.attrs = frame.attrs.copy()
    output.attrs["wind_encoding"] = WIND_ENCODING
    return output


def _available_source_columns(csv_paths):
    columns = []
    for source in csv_paths:
        table = _read_tabular_source(source, header_only=True)
        for column in table.columns:
            if column not in columns:
                columns.append(column)
    return columns


def _resolve_wind_source_columns(csv_paths, target_cols, aux_cols):
    """Resolve canonical wind requests to raw source aliases when necessary."""
    aux_cols = list(aux_cols)
    requested_u = _matching_column(aux_cols, WIND_COMPONENT_COLUMNS[0])
    requested_v = _matching_column(aux_cols, WIND_COMPONENT_COLUMNS[1])
    if requested_u is None or requested_v is None:
        return aux_cols

    # Do not reinterpret a raw pair that is explicitly part of the target.
    if (
        _matching_column(target_cols, WIND_RAW_COLUMNS[0]) is not None
        or _matching_column(target_cols, WIND_RAW_COLUMNS[1]) is not None
    ):
        return aux_cols

    available = _available_source_columns(csv_paths)
    source_u = _matching_column(available, WIND_COMPONENT_COLUMNS[0])
    source_v = _matching_column(available, WIND_COMPONENT_COLUMNS[1])
    source_speed = _matching_column(available, WIND_RAW_COLUMNS[0])
    source_direction = _matching_column(available, WIND_RAW_COLUMNS[1])
    if source_u is not None or source_v is not None:
        if source_speed is not None or source_direction is not None:
            raise ValueError(
                "Input sources contain both raw WS/WD and derived wind_u/wind_v"
            )
        return aux_cols
    if source_speed is None or source_direction is None:
        return aux_cols

    return [
        (
            source_speed
            if column == requested_u
            else source_direction
            if column == requested_v
            else column
        )
        for column in aux_cols
    ]


def _prepare_timestamp_index(df, timestamp_col, source_label):
    if timestamp_col in df.columns:
        raw_timestamps = df[timestamp_col]
        try:
            timestamps = pd.to_datetime(raw_timestamps, errors="raise")
        except Exception as exc:
            raise ValueError(f"Invalid timestamps in {source_label}: {exc}") from exc
        if timestamps.isna().any():
            raise ValueError(f"Missing timestamps found in {source_label}")
        return df.assign(**{timestamp_col: timestamps}).set_index(timestamp_col)
    if isinstance(df.index, pd.DatetimeIndex):
        indexed = df.copy()
        indexed.index = pd.to_datetime(indexed.index, errors="raise")
        indexed.index.name = timestamp_col
        if indexed.index.isna().any():
            raise ValueError(f"Missing timestamps found in {source_label}")
        return indexed
    raise ValueError(f"'{timestamp_col}' not found in columns of {source_label}")


def load_frame(
    csv_paths,
    timestamp_col,
    target_cols,
    aux_cols,
    *,
    expected_frequency=None,
    time_grid_policy="strict",
    duplicate_timestamp_policy="error",
    canonicalize_wind=False,
):
    """Load named time-series columns and enforce an explicit time-grid contract.

    ``time_grid_policy`` is ``strict`` (reject missing/irregular timestamps),
    ``reindex`` (insert missing grid rows as NaN), or ``row_order`` (legacy
    behavior; timestamps are sorted but row spacing is not validated).
    """
    if isinstance(csv_paths, (str, Path, pd.DataFrame)):
        csv_paths = [csv_paths]
    csv_paths = list(csv_paths)
    if not csv_paths:
        raise ValueError("At least one CSV path is required")
    target_cols = list(target_cols)
    aux_cols = list(aux_cols)
    overlap = sorted(set(target_cols) & set(aux_cols))
    if overlap:
        raise ValueError(f"Columns cannot be both target and auxiliary: {overlap}")
    if not target_cols:
        raise ValueError("At least one target column is required")
    if time_grid_policy not in {"strict", "reindex", "row_order"}:
        raise ValueError(
            "time_grid_policy must be 'strict', 'reindex', or 'row_order'"
        )
    if duplicate_timestamp_policy not in {"error", "first"}:
        raise ValueError(
            "duplicate_timestamp_policy must be 'error' or 'first'"
        )

    source_aux_cols = (
        _resolve_wind_source_columns(csv_paths, target_cols, aux_cols)
        if canonicalize_wind
        else aux_cols
    )
    required = target_cols + source_aux_cols
    selected_frames = []
    marker_frames = []
    sources = {}
    for source_index, source in enumerate(csv_paths):
        label = _source_label(source, source_index)
        df = _prepare_timestamp_index(
            _read_tabular_source(source), timestamp_col, label
        )
        duplicate_count = int(df.index.duplicated(keep=False).sum())
        if duplicate_count:
            if duplicate_timestamp_policy == "error":
                raise ValueError(
                    f"{label} contains {duplicate_count} rows with duplicate timestamps"
                )
            df = df[~df.index.duplicated(keep="first")]

        present = [column for column in required if column in df.columns]
        for column in present:
            if column in sources:
                raise ValueError(
                    f"Column {column!r} appears in multiple CSVs/sources: "
                    f"{sources[column]} and {label}"
                )
            sources[column] = label
        if present:
            selected = df[present].copy()
            marker_frame = pd.DataFrame(False, index=df.index, columns=present)
            for column in present:
                if column in target_cols:
                    try:
                        selected[column], marker = _parse_censored_numeric_series(
                            selected[column]
                        )
                    except Exception as exc:
                        raise ValueError(f"Column {column!r} must be numeric") from exc
                    marker_frame[column] = marker
            selected_frames.append(selected)
            marker_frames.append(marker_frame)

    missing = [column for column in required if column not in sources]
    if missing:
        raise ValueError(f"Columns not found in the loaded data: {missing}")
    if not selected_frames:
        raise ValueError("No requested columns were loaded")

    merged = pd.concat(selected_frames, axis=1, join="outer").sort_index()
    marker_merged = pd.concat(marker_frames, axis=1, join="outer").sort_index()
    if not merged.index.is_monotonic_increasing:
        raise ValueError("Timestamp index could not be sorted monotonically")
    if merged.index.has_duplicates:
        raise ValueError("Duplicate timestamps remain after CSV merge")

    for column in required:
        try:
            merged[column] = pd.to_numeric(merged[column], errors="raise")
        except Exception as exc:
            raise ValueError(f"Column {column!r} must be numeric") from exc
    all_missing_targets = [
        column for column in target_cols if merged[column].isna().all()
    ]
    if all_missing_targets:
        raise ValueError(
            f"Target columns contain no observed values: {all_missing_targets}"
        )

    frequency = None
    if time_grid_policy != "row_order" and len(merged.index) > 1:
        frequency = _resolve_time_grid(merged.index, expected_frequency)
        if frequency is None:
            raise ValueError(
                "Could not determine a positive time frequency; pass "
                "expected_frequency explicitly or use time_grid_policy='row_order'"
            )
        full_index = pd.date_range(
            start=merged.index.min(),
            end=merged.index.max(),
            freq=frequency,
            tz=merged.index.tz,
        )
        if not merged.index.equals(full_index):
            missing_rows = int(len(full_index.difference(merged.index)))
            off_grid_rows = int(len(merged.index.difference(full_index)))
            if time_grid_policy == "strict":
                raise ValueError(
                    "Timestamp grid is irregular: "
                    f"frequency={frequency.freqstr}, missing_grid_rows={missing_rows}, "
                    f"off_grid_rows={off_grid_rows}. Use time_grid_policy='reindex' "
                    "to insert missing rows as NaN."
                )
            if off_grid_rows:
                raise ValueError(
                    f"Cannot reindex: {off_grid_rows} timestamps are off the "
                    f"{frequency.freqstr} grid"
                )
            merged = merged.reindex(full_index)
            merged.index.name = timestamp_col
            marker_merged = marker_merged.reindex(full_index, fill_value=False)
            marker_merged.index.name = timestamp_col

    resolved_aux_cols = (
        canonicalize_wind_column_names(source_aux_cols)
        if canonicalize_wind
        else aux_cols
    )
    if canonicalize_wind:
        merged = canonicalize_wind_columns(merged, source_aux_cols)
    merged = merged[[*target_cols, *resolved_aux_cols]]
    marker_merged = marker_merged.reindex(columns=target_cols, fill_value=False).fillna(False)

    merged.attrs["frequency"] = frequency.freqstr if frequency is not None else None
    merged.attrs["timezone"] = str(merged.index.tz) if merged.index.tz is not None else None
    merged.attrs["time_grid_policy"] = time_grid_policy
    if canonicalize_wind and set(WIND_COMPONENT_COLUMNS).issubset(resolved_aux_cols):
        merged.attrs["wind_encoding"] = WIND_ENCODING
    merged.attrs["censor_marker_mask"] = marker_merged.to_numpy(dtype=bool)
    merged.attrs["censor_marker_columns"] = list(target_cols)
    return merged


def _discover_modality_columns(sources_input, timestamp_col, modality):
    """Return non-timestamp columns in deterministic source/column order."""
    columns = []
    sources = {}
    for source_index, source in enumerate(sources_input):
        label = _source_label(source, source_index)
        table = _read_tabular_source(source, header_only=True)
        has_timestamp_column = timestamp_col in table.columns
        has_datetime_index = isinstance(table.index, pd.DatetimeIndex)
        if not has_timestamp_column and not has_datetime_index:
            raise ValueError(f"'{timestamp_col}' not found in columns of {label}")
        discovered = [
            column for column in table.columns if column != timestamp_col
        ]
        if not discovered:
            raise ValueError(f"{modality} source {label} contains no value columns")
        for column in discovered:
            if column in sources:
                raise ValueError(
                    f"Column {column!r} appears in multiple {modality} sources: "
                    f"{sources[column]} and {label}"
                )
            sources[column] = label
            columns.append(column)
    return columns


def _resolve_psd_columns(columns):
    parsed = []
    for column in columns:
        try:
            diameter = float(column)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"PSD column {column!r} is not a numeric particle diameter in nm"
            ) from exc
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError(f"PSD diameter must be positive and finite, got {column!r}")
        parsed.append((diameter, column))
    parsed.sort(key=lambda item: item[0])
    diameters = [item[0] for item in parsed]
    if any(b <= a for a, b in zip(diameters, diameters[1:])):
        raise ValueError("PSD CSV contains duplicate particle diameters")
    return [item[1] for item in parsed], diameters


def load_modality_frame(
    files,
    timestamp_col="time",
    *,
    expected_schema=None,
    expected_frequency=None,
    time_grid_policy="strict",
    duplicate_timestamp_policy="error",
    add_time_cyclical_features=False,
):
    """Load Chem, PSD, and meteorology CSVs without manual column lists.

    Chemistry and meteorology preserve file/column order. PSD columns must be
    numeric diameters in nm and are sorted into strictly increasing diameter
    order. When ``expected_schema`` is supplied (inference), exact column sets
    are enforced and the training-time order is reused.
    """
    if isinstance(files, dict):
        files = ModalityInputs.from_dict(files)
    elif isinstance(files, ModalityFiles):
        files = ModalityInputs.from_files(files)
    if not isinstance(files, ModalityInputs):
        raise TypeError(
            "files must be ModalityFiles, ModalityInputs, or a compatible dict"
        )
    if isinstance(expected_schema, dict):
        expected_schema = DataSchema.from_dict(expected_schema)

    chemistry_cols = _discover_modality_columns(
        files.chemistry, timestamp_col, "chemistry"
    ) if files.chemistry else []
    psd_discovered = _discover_modality_columns(
        files.psd, timestamp_col, "PSD"
    ) if files.psd else []
    psd_cols, psd_diameters = _resolve_psd_columns(psd_discovered) if psd_discovered else ([], [])
    meteorology_input_cols = _discover_modality_columns(
        files.meteorology, timestamp_col, "meteorology"
    ) if files.meteorology else []
    canonicalize_wind = True

    if expected_schema is not None:
        if timestamp_col != expected_schema.timestamp_col:
            raise ValueError(
                f"timestamp_col={timestamp_col!r} does not match bundle schema "
                f"{expected_schema.timestamp_col!r}"
            )
        # Keep older bundles whose schema explicitly contains raw WS/WD
        # readable. New schemas use wind_u/wind_v and accept either raw or
        # already-derived input at inference time.
        expected_uses_raw_wind = (
            _matching_column(expected_schema.meteorology_cols, WIND_RAW_COLUMNS[0])
            is not None
            and _matching_column(expected_schema.meteorology_cols, WIND_RAW_COLUMNS[1])
            is not None
            and _matching_column(expected_schema.meteorology_cols, WIND_COMPONENT_COLUMNS[0])
            is None
            and _matching_column(expected_schema.meteorology_cols, WIND_COMPONENT_COLUMNS[1])
            is None
        )
        canonicalize_wind = not expected_uses_raw_wind
        meteorology_cols = (
            canonicalize_wind_column_names(meteorology_input_cols)
            if canonicalize_wind
            else list(meteorology_input_cols)
        )
        discovered_by_modality = {
            "chemistry": chemistry_cols,
            "psd": psd_cols,
            "meteorology": meteorology_cols,
        }
        expected_by_modality = {
            "chemistry": expected_schema.chemistry_cols,
            "psd": expected_schema.psd_cols,
            "meteorology": expected_schema.meteorology_cols,
        }
        for modality, expected in expected_by_modality.items():
            discovered = discovered_by_modality[modality]
            missing = sorted(set(expected) - set(discovered))
            extra = sorted(set(discovered) - set(expected))
            if missing or extra:
                raise ValueError(
                    f"{modality} columns do not match training schema: "
                    f"missing={missing}, extra={extra}"
                )
        chemistry_cols = list(expected_schema.chemistry_cols)
        psd_cols = list(expected_schema.psd_cols)
        psd_diameters = list(expected_schema.psd_diameters_nm)
        meteorology_cols = list(expected_schema.meteorology_cols)
        expected_frequency = expected_schema.frequency or expected_frequency
        time_grid_policy = expected_schema.time_grid_policy
        duplicate_timestamp_policy = expected_schema.duplicate_timestamp_policy
        add_time_cyclical_features = expected_schema.add_time_cyclical_features
    else:
        meteorology_cols = canonicalize_wind_column_names(meteorology_input_cols)

    all_modality_columns = [*chemistry_cols, *psd_cols, *meteorology_cols]
    cross_modality_duplicates = sorted(
        {column for column in all_modality_columns if all_modality_columns.count(column) > 1}
    )
    if cross_modality_duplicates:
        raise ValueError(
            "Columns cannot appear in multiple modalities: "
            f"{cross_modality_duplicates}"
        )

    target_cols = [*chemistry_cols, *psd_cols]
    frame = load_frame(
        files.all_sources,
        timestamp_col,
        target_cols,
        meteorology_input_cols,
        expected_frequency=expected_frequency,
        time_grid_policy=time_grid_policy,
        duplicate_timestamp_policy=duplicate_timestamp_policy,
        canonicalize_wind=canonicalize_wind,
    )
    actual_timezone = frame.attrs.get("timezone")
    if expected_schema is not None and actual_timezone != expected_schema.timezone:
        raise ValueError(
            "timezone does not match training schema: "
            f"expected={expected_schema.timezone!r}, actual={actual_timezone!r}"
        )
    marker_mask = frame.attrs.get("censor_marker_mask")
    marker_columns = frame.attrs.get("censor_marker_columns")
    if add_time_cyclical_features:
        frame = pd.concat([frame, compute_time_cyclical_features(frame.index)], axis=1)
    schema = DataSchema(
        timestamp_col=timestamp_col,
        chemistry_cols=chemistry_cols,
        psd_cols=psd_cols,
        meteorology_cols=meteorology_cols,
        frequency=frame.attrs.get("frequency"),
        timezone=actual_timezone,
        time_grid_policy=time_grid_policy,
        duplicate_timestamp_policy=duplicate_timestamp_policy,
        psd_diameters_nm=psd_diameters,
        add_time_cyclical_features=add_time_cyclical_features,
    )
    frame = frame[[*schema.target_cols, *schema.auxiliary_cols]]
    frame.attrs.update(
        frequency=schema.frequency,
        timezone=schema.timezone,
        time_grid_policy=schema.time_grid_policy,
    )
    if marker_mask is not None:
        frame.attrs["censor_marker_mask"] = marker_mask
        frame.attrs["censor_marker_columns"] = (
            list(marker_columns) if marker_columns is not None else list(schema.target_cols)
        )
    return frame, schema


def compute_window_starts(n, window_size, stride):
    """Sliding-window start indices; always includes a final tail window."""
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")
    if stride > window_size:
        raise ValueError("stride cannot exceed window_size because it leaves uncovered rows")
    if n < window_size:
        return []
    starts = list(range(0, n - window_size + 1, stride))
    last_start = n - window_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def chronological_split_index(n, val_fraction):
    val_fraction = min(max(val_fraction, 0.0), 0.9)
    return int(round(n * (1 - val_fraction)))


def make_condition(aux, aux_observed=None, use_mask_channel=True):
    """Build the conditioning channels consumed by the public model adapter.

    Auxiliary values are zero-filled after scaling, but their observedness is
    retained in a separate channel.  This prevents a missing auxiliary value
    from being confused with a genuine scaled zero and keeps the target mask
    independent from auxiliary availability.
    """
    aux = np.asarray(aux, dtype=np.float32)
    if aux.ndim != 2:
        raise ValueError(f"aux must be a 2-D array, got shape {aux.shape}")
    values = np.nan_to_num(aux, nan=0.0).astype(np.float32, copy=False)
    if not use_mask_channel:
        return values
    if aux_observed is None:
        aux_observed = ~np.isnan(aux)
    aux_observed = np.asarray(aux_observed, dtype=bool)
    if aux_observed.shape != values.shape:
        raise ValueError(
            f"aux_observed shape {aux_observed.shape} does not match aux shape {values.shape}"
        )
    return np.concatenate([values, aux_observed.astype(np.float32)], axis=-1)


def _empirical_gap_durations(observed_mask):
    """Real contiguous-missing run lengths (in rows), pooled across columns.

    Sensor dropouts are heavy-tailed and right-skewed (many short blips, a
    few long outages) -- no Normal(mean, std) draw can match that shape.
    Bootstrapping straight from the real run lengths sidesteps picking a
    single summary statistic entirely. Returns an empty array when the
    column set has no missing runs (e.g. synthetic all-observed data), so
    callers can fall back to the parametric draw.
    """
    observed_mask = np.asarray(observed_mask, dtype=bool)
    durations = []
    for col in range(observed_mask.shape[1]):
        missing = (~observed_mask[:, col]).astype(np.int8)
        if not missing.any():
            continue
        edges = np.diff(np.concatenate(([0], missing, [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        durations.extend((ends - starts).tolist())
    return np.asarray(durations, dtype=np.int64)


def _sample_block_mask(
    length, rng, mean_duration, std_duration, min_duration, max_duration,
    duration_pool=None,
):
    if length <= 0:
        return np.zeros(0, dtype=bool)
    if duration_pool is not None and len(duration_pool):
        duration = int(rng.choice(duration_pool))
    elif std_duration:
        duration = int(round(rng.normal(mean_duration, std_duration)))
    else:
        duration = int(mean_duration)
    duration = max(int(min_duration), min(int(max_duration), duration, length))
    start = int(rng.integers(0, length - duration + 1))
    mask = np.zeros(length, dtype=bool)
    mask[start:start + duration] = True
    return mask


def sample_dynamic_heldout_mask(observed_mask, config=None, seed=0, rng=None):
    """Sample a contiguous, observation-intersected held-out mask.

    The public adapter uses the same shape of masking protocol as the
    research trainer: PSD-like targets share time blocks, while chemistry-like
    targets receive feature-specific blocks.  The returned mask never marks a
    genuinely missing target as held out.
    """
    observed_mask = np.asarray(observed_mask, dtype=bool)
    if observed_mask.ndim != 2:
        raise ValueError(f"observed_mask must be 2-D, got shape {observed_mask.shape}")
    config = dict(config or {})
    rng = rng or np.random.default_rng(seed)
    target_ratio = float(config.get("target_ratio", 0.10))
    if target_ratio <= 0 or not observed_mask.any():
        return np.zeros_like(observed_mask, dtype=bool)

    window_size, n_features = observed_mask.shape
    mean_duration = float(config.get("mean_duration", 48))
    std_duration = float(config.get("std_duration", 24))
    min_duration = max(1, int(config.get("min_duration", 3)))
    max_duration = max(min_duration, int(config.get("max_duration", 168)))
    n_chem = min(max(0, int(config.get("n_chem", 0))), n_features)
    if config.get("mode", "block") == "legacy":
        heldout = np.zeros_like(observed_mask, dtype=bool)
        n_psd = n_features - n_chem
        max_psd_blocks = max(0, int(config.get("legacy_psd_blocks", 6)))
        max_chem_blocks = max(0, int(config.get("legacy_chem_blocks", 8)))
        psd_max_len = max(1, int(window_size * 0.15))
        chem_max_len = max(1, int(window_size * 0.10))

        if n_psd > 0:
            for _ in range(max_psd_blocks):
                start = int(rng.integers(0, max(1, window_size)))
                length = int(rng.integers(1, psd_max_len + 1))
                end = min(window_size, start + length)
                heldout[start:end, n_chem:] |= observed_mask[start:end, n_chem:]

        if n_chem > 0:
            for _ in range(max_chem_blocks):
                start = int(rng.integers(0, max(1, window_size)))
                length = int(rng.integers(1, chem_max_len + 1))
                end = min(window_size, start + length)
                feature = int(rng.integers(0, n_chem))
                heldout[start:end, feature] |= observed_mask[start:end, feature]

        random_prob = float(config.get("random_point_drop_prob", 0.04))
        if random_prob > 0:
            heldout |= (rng.random(observed_mask.shape) < random_prob) & observed_mask
        return heldout

    chem_blocks = max(1, int(config.get("chem_blocks", 1)))
    psd_blocks = max(1, int(config.get("psd_blocks", 1)))
    expected_block_fraction = max(mean_duration / max(window_size, 1), 1e-6)
    default_block_prob = min(1.0, target_ratio / expected_block_fraction)
    chem_block_prob = float(config.get("chem_block_prob", default_block_prob / chem_blocks))
    psd_block_prob = float(config.get("psd_block_prob", default_block_prob / psd_blocks))

    heldout = np.zeros_like(observed_mask, dtype=bool)

    use_empirical_duration = config.get("duration_source") == "empirical"

    def add_block(columns, count, per_feature, block_prob):
        nonlocal heldout
        if len(columns) == 0:
            return
        duration_pool = (
            _empirical_gap_durations(observed_mask[:, columns])
            if use_empirical_duration else None
        )
        for _ in range(count):
            if rng.random() >= min(max(block_prob, 0.0), 1.0):
                continue
            time_mask = _sample_block_mask(
                window_size, rng, mean_duration, std_duration,
                min_duration, max_duration, duration_pool=duration_pool,
            )
            if per_feature:
                for col in columns:
                    heldout[:, col] |= time_mask & observed_mask[:, col]
            else:
                heldout[:, columns] |= time_mask[:, None] & observed_mask[:, columns]

    # This modality split mirrors the research protocol.  If n_chem=0, all
    # targets use the full-spectrum path, which is the safe generic default.
    add_block(np.arange(n_chem), chem_blocks, per_feature=True, block_prob=chem_block_prob)
    add_block(
        np.arange(n_chem, n_features), psd_blocks,
        per_feature=False, block_prob=psd_block_prob,
    )

    # On very short or sparse windows a sampled block can miss all observed
    # points. Validation uses ensure_nonempty so its selection metric is
    # defined; training may legitimately have no block on a given window.
    if config.get("ensure_nonempty", False) and heldout.sum() == 0:
        observed_positions = np.argwhere(observed_mask)
        if len(observed_positions):
            row, col = observed_positions[int(rng.integers(len(observed_positions)))]
            heldout[row, col] = True
    return heldout


def sample_block_heldout_mask_to_ratio(
    observed_mask, config=None, seed=0, rng=None, row_weights=None
):
    """Fill block masks to a cell quota independently for Chem and PSD.

    This sampler operates on a complete timeline. It keeps drawing contiguous
    time blocks until each target family reaches the requested fraction of
    eligible observed cells. The last block may be shortened to limit quota
    overshoot. Optional row weights let dynamic training target the effective
    number of overlapping-window occurrences rather than only unique cells.
    """
    observed_mask = np.asarray(observed_mask, dtype=bool)
    if observed_mask.ndim != 2:
        raise ValueError(f"observed_mask must be 2-D, got shape {observed_mask.shape}")
    config = dict(config or {})
    rng = rng or np.random.default_rng(seed)
    target_ratio = float(config.get("target_ratio", 0.10))
    if not 0 <= target_ratio <= 1:
        raise ValueError("target_ratio must be in [0, 1]")
    if target_ratio <= 0 or not observed_mask.any():
        return np.zeros_like(observed_mask, dtype=bool)
    if target_ratio >= 1:
        return observed_mask.copy()

    length, n_features = observed_mask.shape
    if row_weights is None:
        row_weights = np.ones(length, dtype=np.float64)
    else:
        row_weights = np.asarray(row_weights, dtype=np.float64)
        if row_weights.shape != (length,):
            raise ValueError(f"row_weights must have shape ({length},)")
        if not np.isfinite(row_weights).all() or (row_weights < 0).any():
            raise ValueError("row_weights must be finite and non-negative")
    cell_weights = row_weights[:, None]
    mean_duration = float(config.get("mean_duration", 48))
    std_duration = float(config.get("std_duration", 24))
    min_duration = max(1, int(config.get("min_duration", 3)))
    max_duration = max(min_duration, int(config.get("max_duration", 168)))
    n_chem = min(max(0, int(config.get("n_chem", 0))), n_features)
    heldout = np.zeros_like(observed_mask, dtype=bool)

    def trim_to_remaining(candidate, remaining):
        if float((candidate * cell_weights).sum()) <= remaining:
            return candidate
        active_rows = np.flatnonzero(candidate.any(axis=1))
        if active_rows.size == 0:
            return candidate
        first, last = int(active_rows[0]), int(active_rows[-1])
        row_counts = (
            candidate[first:last + 1]
            * cell_weights[first:last + 1]
        ).sum(axis=1)
        prefix_counts = np.cumsum(row_counts)
        suffix_counts = np.cumsum(row_counts[::-1])
        prefix_len = int(np.argmin(np.abs(prefix_counts - remaining))) + 1
        suffix_len = int(np.argmin(np.abs(suffix_counts - remaining))) + 1
        prefix_error = abs(float(prefix_counts[prefix_len - 1]) - remaining)
        suffix_error = abs(float(suffix_counts[suffix_len - 1]) - remaining)
        use_prefix = prefix_error < suffix_error or (
            prefix_error == suffix_error and bool(rng.integers(0, 2))
        )
        trimmed = np.zeros_like(candidate)
        if use_prefix:
            trimmed[first:first + prefix_len] = candidate[first:first + prefix_len]
        else:
            start = last - suffix_len + 1
            trimmed[start:last + 1] = candidate[start:last + 1]
        return trimmed

    def fill_family(columns, blocks_per_round):
        if len(columns) == 0:
            return
        eligible = observed_mask[:, columns]
        family_weights = cell_weights * eligible
        eligible_weight = float(family_weights.sum())
        if eligible_weight <= 0:
            return
        target_count = max(1.0, eligible_weight * target_ratio)
        attempts = 0
        max_attempts = max(1000, 20 * length)
        duration_pool = (
            _empirical_gap_durations(eligible)
            if config.get("duration_source") == "empirical" else None
        )

        def current_count():
            return float((heldout[:, columns] * cell_weights).sum())

        while current_count() < target_count and attempts < max_attempts:
            for _ in range(max(1, blocks_per_round)):
                attempts += 1
                time_mask = _sample_block_mask(
                    length, rng, mean_duration, std_duration,
                    min_duration, max_duration, duration_pool=duration_pool,
                )
                candidate = time_mask[:, None] & eligible & ~heldout[:, columns]
                if not candidate.any():
                    if attempts >= max_attempts:
                        break
                    continue
                current = current_count()
                candidate = trim_to_remaining(candidate, target_count - current)
                heldout[:, columns] |= candidate
                if current_count() >= target_count:
                    break

    fill_family(np.arange(n_chem), int(config.get("chem_blocks", 1)))
    fill_family(np.arange(n_chem, n_features), int(config.get("psd_blocks", 1)))

    if config.get("ensure_nonempty", False) and heldout.sum() == 0:
        observed_positions = np.argwhere(observed_mask)
        if len(observed_positions):
            row, col = observed_positions[int(rng.integers(len(observed_positions)))]
            heldout[row, col] = True
    return heldout


def _observed_runs(mask_1d):
    indices = np.flatnonzero(np.asarray(mask_1d, dtype=bool))
    if indices.size == 0:
        return
    cuts = np.where(np.diff(indices) > 1)[0] + 1
    for group in np.split(indices, cuts):
        yield int(group[0]), int(group[-1])


def _sample_anchor_constrained_series(observed_1d, target_count, rng,
                                      mean_duration=48.0, std_duration=24.0,
                                      min_duration=3, max_duration=168,
                                      duration_pool=None):
    available = np.asarray(observed_1d, dtype=bool).copy()
    heldout = np.zeros_like(available, dtype=bool)
    target_count = int(target_count)
    current = 0
    attempts = 0
    while current < target_count and attempts < 200_000:
        attempts += 1
        remaining = target_count - current
        if duration_pool is not None and len(duration_pool):
            duration = int(np.clip(rng.choice(duration_pool), min_duration, max_duration))
        else:
            duration = int(np.clip(
                rng.normal(mean_duration, std_duration), min_duration, max_duration
            ))
        if remaining < min_duration or current + duration > target_count:
            duration = remaining
        candidates = []
        weights = []
        for run_start, run_end in _observed_runs(available):
            n_starts = run_end - run_start + 1 - duration - 1
            if n_starts > 0:
                candidates.append((run_start, run_end))
                weights.append(n_starts)
        if not candidates:
            if duration <= min_duration:
                break
            continue
        weights = np.asarray(weights, dtype=np.float64)
        run_start, run_end = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
        start = int(rng.integers(run_start + 1, run_end - duration + 1))
        end = start + duration
        heldout[start:end] = True
        available[start:end] = False
        current += duration
    return heldout


def sample_anchor_constrained_heldout_mask(
    observed_mask, ratio=0.10, seed=42, n_chem=None,
    mean_duration=48.0, std_duration=24.0, min_duration=3, max_duration=168,
    duration_source="parametric",
):
    """Generate 26e-style held-out gaps bounded by observed anchors.

    Chemistry is sampled per feature. PSD-like features share time gaps and
    require every PSD feature to be observed at the candidate timestamps.

    ``duration_source="empirical"`` bootstraps gap duration from each
    family's own real contiguous-missing run lengths instead of drawing from
    Normal(mean_duration, std_duration) -- see ``_empirical_gap_durations``
    for why a single mean/std can't represent real, heavy-tailed gap
    statistics. Falls back to the parametric draw when a family has no real
    gaps to bootstrap from.
    """
    observed_mask = np.asarray(observed_mask, dtype=bool)
    if observed_mask.ndim != 2:
        raise ValueError("observed_mask must be 2-D")
    n_rows, n_features = observed_mask.shape
    n_chem = n_features if n_chem is None else min(max(int(n_chem), 0), n_features)
    rng = np.random.default_rng(seed)
    heldout = np.zeros_like(observed_mask, dtype=bool)
    use_empirical = duration_source == "empirical"
    duration_kwargs = dict(
        mean_duration=mean_duration, std_duration=std_duration,
        min_duration=min_duration, max_duration=max_duration,
    )

    chem_pool = (
        _empirical_gap_durations(observed_mask[:, :n_chem])
        if use_empirical and n_chem > 0 else None
    )
    for feature in range(n_chem):
        observed_count = int(observed_mask[:, feature].sum())
        heldout[:, feature] = _sample_anchor_constrained_series(
            observed_mask[:, feature], int(observed_count * ratio), rng,
            duration_pool=chem_pool, **duration_kwargs,
        )

    if n_chem < n_features:
        psd_observed = observed_mask[:, n_chem:].all(axis=1)
        psd_pool = (
            _empirical_gap_durations(observed_mask[:, n_chem:])
            if use_empirical else None
        )
        heldout_time = _sample_anchor_constrained_series(
            psd_observed, int(psd_observed.sum() * ratio), rng,
            duration_pool=psd_pool, **duration_kwargs,
        )
        heldout[np.ix_(heldout_time, np.arange(n_chem, n_features))] = True
    return heldout


class WindowedTimeSeriesDataset(Dataset):
    """Windows a (already NaN-filled, already scaled) target/aux array pair.

    In 'train' mode, dynamic contiguous blocks of already-observed target
    points are hidden from the input while remaining in the loss target. In
    'val' mode, an externally generated selection mask is held fixed.
    """

    def __init__(
        self,
        target,
        aux,
        window_size,
        stride,
        mode="train",
        denoise_prob=0.0,
        seed=0,
        aux_mask=None,
        aux_mask_channel=True,
        dynamic_mask_config=None,
        selection_mask=None,
        fixed_mask=None,
        censor_mask=None,
    ):
        self.target = np.asarray(target, dtype=np.float32)
        self.aux = np.asarray(aux, dtype=np.float32)
        if self.target.ndim != 2 or self.aux.ndim != 2:
            raise ValueError("target and aux must both be 2-D arrays")
        if len(self.target) != len(self.aux):
            raise ValueError("target and aux must have the same number of rows")
        self.aux_mask = (~np.isnan(self.aux)) if aux_mask is None else np.asarray(aux_mask, dtype=bool)
        if self.aux_mask.shape != self.aux.shape:
            raise ValueError("aux_mask must have the same shape as aux")
        self.window_size = window_size
        self.mode = mode
        self.denoise_prob = denoise_prob if mode == "train" else 0.0
        self.aux_mask_channel = bool(aux_mask_channel)
        self.dynamic_mask_config = dict(dynamic_mask_config or {}) if mode == "train" else None
        self.dynamic_mask_scope = (
            self.dynamic_mask_config.get("scope", "window")
            if self.dynamic_mask_config else "window"
        )
        if self.dynamic_mask_scope not in {"window", "timeline_epoch"}:
            raise ValueError("dynamic mask scope must be 'window' or 'timeline_epoch'")
        self.selection_mask = None if selection_mask is None else np.asarray(selection_mask, dtype=bool)
        if self.selection_mask is not None and self.selection_mask.shape != self.target.shape:
            raise ValueError("selection_mask must have the same shape as target")
        self.fixed_mask = None if fixed_mask is None else np.asarray(fixed_mask, dtype=bool)
        if self.fixed_mask is not None and self.fixed_mask.shape != self.target.shape:
            raise ValueError("fixed_mask must have the same shape as target")
        # Below-detection-limit cells. They are excluded from obs_mask (they
        # are not point observations) but stay in the encoder input, where
        # they carry their substituted value.
        self.censor_mask = None if censor_mask is None else np.asarray(censor_mask, dtype=bool)
        if self.censor_mask is not None and self.censor_mask.shape != self.target.shape:
            raise ValueError("censor_mask must have the same shape as target")
        self.starts = compute_window_starts(len(self.target), window_size, stride)
        self.window_coverage = np.zeros(len(self.target), dtype=np.float64)
        for start in self.starts:
            self.window_coverage[start:start + self.window_size] += 1.0
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._timeline_dynamic_mask = None
        if self.dynamic_mask_scope == "timeline_epoch":
            self.set_epoch(0)

    def __len__(self):
        return len(self.starts)

    def set_epoch(self, epoch):
        """Create one absolute-time dynamic mask shared by every window."""
        if not self.dynamic_mask_config or self.dynamic_mask_scope != "timeline_epoch":
            return
        eligible = ~np.isnan(self.target)
        if self.fixed_mask is not None:
            eligible &= ~self.fixed_mask
        epoch_seed = int(
            np.random.SeedSequence([self.seed, int(epoch)]).generate_state(1)[0]
        )
        self._timeline_dynamic_mask = sample_block_heldout_mask_to_ratio(
            eligible,
            self.dynamic_mask_config,
            seed=epoch_seed,
            row_weights=self.window_coverage,
        )

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.window_size

        target_win = self.target[start:end]
        aux_win = self.aux[start:end]
        aux_mask_win = self.aux_mask[start:end]

        present = ~np.isnan(target_win)
        if self.censor_mask is None:
            censor_mask = np.zeros_like(present, dtype=np.float32)
            obs_mask = present.astype(np.float32)
        else:
            censored = present & self.censor_mask[start:end]
            censor_mask = censored.astype(np.float32)
            obs_mask = (present & ~censored).astype(np.float32)
        target_clean = np.nan_to_num(target_win, nan=0.0)
        cond = make_condition(aux_win, aux_mask_win, self.aux_mask_channel)

        # Everything the encoder is allowed to read: real observations plus
        # non-detects, which are informative even though they are not points.
        known_mask = obs_mask + censor_mask
        input_mask = known_mask.copy()
        heldout_mask = np.zeros_like(obs_mask)
        if self.fixed_mask is not None:
            heldout_mask = (self.fixed_mask[start:end] & obs_mask.astype(bool)).astype(np.float32)
            # A fixed selection mask is a permanent blind held-out set, not a
            # per-epoch denoising drop: exclude it from obs_mask (not just
            # input_mask) so the reconstruction loss -- computed over
            # obs_mask in the training loop -- never supervises on these
            # positions either. Otherwise the model gets direct gradient
            # signal to reconstruct exactly the points later reported as
            # held-out accuracy, inflating held-out metrics.
            obs_mask = obs_mask * (1.0 - heldout_mask)
            known_mask = obs_mask + censor_mask
            input_mask = known_mask.copy()
        if self.dynamic_mask_config:
            # Non-detects are droppable from the input too, so the model has
            # to predict "below the limit" from context rather than by
            # reading the substituted value back out.
            if self.dynamic_mask_scope == "timeline_epoch":
                dynamic_mask = self._timeline_dynamic_mask[start:end].astype(np.float32)
            else:
                dynamic_mask = sample_dynamic_heldout_mask(
                    known_mask.astype(bool), self.dynamic_mask_config, rng=self.rng
                ).astype(np.float32)
            # Held-out metrics need a ground-truth scalar, which a censored
            # cell does not have: score only the observed positions.
            heldout_mask = np.maximum(heldout_mask, dynamic_mask * obs_mask)
            input_mask = known_mask * (1.0 - dynamic_mask)
        elif self.selection_mask is not None:
            selection_mask = (self.selection_mask[start:end] & obs_mask.astype(bool)).astype(np.float32)
            heldout_mask = np.maximum(heldout_mask, selection_mask)
            input_mask = known_mask * (1.0 - selection_mask)
        if self.denoise_prob > 0:
            drop = (known_mask == 1.0) & (self.rng.random(known_mask.shape) < self.denoise_prob)
            input_mask[drop] = 0.0
        input_x = target_clean * input_mask

        return {
            "target": torch.from_numpy(target_clean),
            "cond": torch.from_numpy(cond),
            "obs_mask": torch.from_numpy(obs_mask),
            "censor_mask": torch.from_numpy(censor_mask),
            "heldout_mask": torch.from_numpy(heldout_mask),
            "input_x": torch.from_numpy(input_x),
            "input_mask": torch.from_numpy(input_mask),
        }
