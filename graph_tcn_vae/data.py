"""CSV loading, normalization, and windowing for the train/impute pipeline.

This module turns an arbitrary tabular time-series CSV (a timestamp column
plus named target/auxiliary columns, NaN = missing) into the
``(target, cond, mask)`` windows the ``ImputationVAE_Graph`` forward pass
expects. Column selection is config-driven (explicit name lists), not
positional, so it works for any dataset shape.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SUPPORTED_TARGET_TRANSFORMS = {"none", "log1p"}


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


def load_frame(
    csv_paths,
    timestamp_col,
    target_cols,
    aux_cols,
    *,
    expected_frequency=None,
    time_grid_policy="strict",
    duplicate_timestamp_policy="error",
):
    """Load named time-series columns and enforce an explicit time-grid contract.

    ``time_grid_policy`` is ``strict`` (reject missing/irregular timestamps),
    ``reindex`` (insert missing grid rows as NaN), or ``row_order`` (legacy
    behavior; timestamps are sorted but row spacing is not validated).
    """
    if isinstance(csv_paths, (str, Path)):
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

    required = target_cols + aux_cols
    selected_frames = []
    sources = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        if timestamp_col not in df.columns:
            raise ValueError(f"'{timestamp_col}' not found in columns of {path}")
        try:
            timestamps = pd.to_datetime(df[timestamp_col], errors="raise")
        except Exception as exc:
            raise ValueError(f"Invalid timestamps in {path}: {exc}") from exc
        if timestamps.isna().any():
            raise ValueError(f"Missing timestamps found in {path}")
        df = df.assign(**{timestamp_col: timestamps}).set_index(timestamp_col)
        duplicate_count = int(df.index.duplicated(keep=False).sum())
        if duplicate_count:
            if duplicate_timestamp_policy == "error":
                raise ValueError(
                    f"{path} contains {duplicate_count} rows with duplicate timestamps"
                )
            df = df[~df.index.duplicated(keep="first")]

        present = [column for column in required if column in df.columns]
        for column in present:
            if column in sources:
                raise ValueError(
                    f"Column {column!r} appears in multiple CSVs: "
                    f"{sources[column]} and {path}"
                )
            sources[column] = str(path)
        if present:
            selected_frames.append(df[present])

    missing = [column for column in required if column not in sources]
    if missing:
        raise ValueError(f"Columns not found in the loaded data: {missing}")
    if not selected_frames:
        raise ValueError("No requested columns were loaded")

    merged = pd.concat(selected_frames, axis=1, join="outer").sort_index()
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

    merged.attrs["frequency"] = frequency.freqstr if frequency is not None else None
    merged.attrs["timezone"] = str(merged.index.tz) if merged.index.tz is not None else None
    merged.attrs["time_grid_policy"] = time_grid_policy
    return merged


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


def _sample_block_mask(length, rng, mean_duration, std_duration, min_duration, max_duration):
    if length <= 0:
        return np.zeros(0, dtype=bool)
    duration = int(round(rng.normal(mean_duration, std_duration))) if std_duration else int(mean_duration)
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

    def add_block(columns, count, per_feature, block_prob):
        nonlocal heldout
        if len(columns) == 0:
            return
        for _ in range(count):
            if rng.random() >= min(max(block_prob, 0.0), 1.0):
                continue
            time_mask = _sample_block_mask(
                window_size, rng, mean_duration, std_duration,
                min_duration, max_duration,
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


def _observed_runs(mask_1d):
    indices = np.flatnonzero(np.asarray(mask_1d, dtype=bool))
    if indices.size == 0:
        return
    cuts = np.where(np.diff(indices) > 1)[0] + 1
    for group in np.split(indices, cuts):
        yield int(group[0]), int(group[-1])


def _sample_anchor_constrained_series(observed_1d, target_count, rng,
                                      mean_duration=48.0, std_duration=24.0,
                                      min_duration=3, max_duration=168):
    available = np.asarray(observed_1d, dtype=bool).copy()
    heldout = np.zeros_like(available, dtype=bool)
    target_count = int(target_count)
    current = 0
    attempts = 0
    while current < target_count and attempts < 200_000:
        attempts += 1
        remaining = target_count - current
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


def sample_anchor_constrained_heldout_mask(observed_mask, ratio=0.10, seed=42, n_chem=None):
    """Generate 26e-style held-out gaps bounded by observed anchors.

    Chemistry is sampled per feature. PSD-like features share time gaps and
    require every PSD feature to be observed at the candidate timestamps.
    """
    observed_mask = np.asarray(observed_mask, dtype=bool)
    if observed_mask.ndim != 2:
        raise ValueError("observed_mask must be 2-D")
    n_rows, n_features = observed_mask.shape
    n_chem = n_features if n_chem is None else min(max(int(n_chem), 0), n_features)
    rng = np.random.default_rng(seed)
    heldout = np.zeros_like(observed_mask, dtype=bool)

    for feature in range(n_chem):
        observed_count = int(observed_mask[:, feature].sum())
        heldout[:, feature] = _sample_anchor_constrained_series(
            observed_mask[:, feature], int(observed_count * ratio), rng
        )

    if n_chem < n_features:
        psd_observed = observed_mask[:, n_chem:].all(axis=1)
        heldout_time = _sample_anchor_constrained_series(
            psd_observed, int(psd_observed.sum() * ratio), rng
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
        self.selection_mask = None if selection_mask is None else np.asarray(selection_mask, dtype=bool)
        if self.selection_mask is not None and self.selection_mask.shape != self.target.shape:
            raise ValueError("selection_mask must have the same shape as target")
        self.fixed_mask = None if fixed_mask is None else np.asarray(fixed_mask, dtype=bool)
        if self.fixed_mask is not None and self.fixed_mask.shape != self.target.shape:
            raise ValueError("fixed_mask must have the same shape as target")
        self.starts = compute_window_starts(len(self.target), window_size, stride)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.window_size

        target_win = self.target[start:end]
        aux_win = self.aux[start:end]
        aux_mask_win = self.aux_mask[start:end]

        obs_mask = (~np.isnan(target_win)).astype(np.float32)
        target_clean = np.nan_to_num(target_win, nan=0.0)
        cond = make_condition(aux_win, aux_mask_win, self.aux_mask_channel)

        input_mask = obs_mask.copy()
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
            input_mask = obs_mask.copy()
        if self.dynamic_mask_config:
            dynamic_mask = sample_dynamic_heldout_mask(
                obs_mask.astype(bool), self.dynamic_mask_config, rng=self.rng
            ).astype(np.float32)
            heldout_mask = np.maximum(heldout_mask, dynamic_mask)
            input_mask = obs_mask * (1.0 - dynamic_mask)
        elif self.selection_mask is not None:
            selection_mask = (self.selection_mask[start:end] & obs_mask.astype(bool)).astype(np.float32)
            heldout_mask = np.maximum(heldout_mask, selection_mask)
            input_mask = obs_mask * (1.0 - selection_mask)
        if self.denoise_prob > 0:
            drop = (obs_mask == 1.0) & (self.rng.random(obs_mask.shape) < self.denoise_prob)
            input_mask[drop] = 0.0
        input_x = target_clean * input_mask

        return {
            "target": torch.from_numpy(target_clean),
            "cond": torch.from_numpy(cond),
            "obs_mask": torch.from_numpy(obs_mask),
            "heldout_mask": torch.from_numpy(heldout_mask),
            "input_x": torch.from_numpy(input_x),
            "input_mask": torch.from_numpy(input_mask),
        }
