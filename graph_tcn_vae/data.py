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
        mean = np.nanmean(array, axis=0)
        std = np.nanstd(array, axis=0)
        # A column that is entirely NaN, or constant, gets a neutral mean/std
        # so transform() doesn't produce NaN/inf for it.
        self.mean_ = np.nan_to_num(mean, nan=0.0)
        self.std_ = np.where(np.isnan(std) | (std < 1e-8), 1.0, std)
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


def load_frame(csv_paths, timestamp_col, target_cols, aux_cols):
    """Load one or more CSVs, join on timestamp, sort chronologically.

    Multiple paths are outer-joined on the timestamp index, which lets
    co-located instruments (e.g. chemistry, PSD, meteorology) live in
    separate files sharing a common time axis.
    """
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if timestamp_col not in df.columns:
            raise ValueError(f"'{timestamp_col}' not found in columns of {path}")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)
        df = df[~df.index.duplicated(keep="first")]
        frames.append(df)

    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.join(extra, how="outer")
    merged = merged.sort_index()

    required = list(target_cols) + list(aux_cols)
    missing = [col for col in required if col not in merged.columns]
    if missing:
        raise ValueError(f"Columns not found in the loaded data: {missing}")

    return merged


def compute_window_starts(n, window_size, stride):
    """Sliding-window start indices; always includes a final tail window."""
    if n < window_size:
        return []
    starts = list(range(0, n - window_size + 1, max(1, stride)))
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
        if self.dynamic_mask_config:
            heldout_mask = sample_dynamic_heldout_mask(
                obs_mask.astype(bool), self.dynamic_mask_config, rng=self.rng
            ).astype(np.float32)
            input_mask = obs_mask * (1.0 - heldout_mask)
        elif self.selection_mask is not None:
            heldout_mask = (self.selection_mask[start:end] & obs_mask.astype(bool)).astype(np.float32)
            input_mask = obs_mask * (1.0 - heldout_mask)
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
