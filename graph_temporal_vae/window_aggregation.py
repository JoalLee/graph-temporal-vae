"""Bounded-memory aggregation for overlapping predictive windows.

The external seam is intentionally small: add windows in nondecreasing start
order, then call :meth:`StreamingWindowAggregator.finish`.  Exact weighted
means, variances, quantiles, and optional empirical CRPS are finalized as soon
as no future window can cover a timestamp, so memory depends on the active
overlap rather than the full time-series length.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


WindowSamples = Tuple[int, np.ndarray]


def _validate_sample_matrix(values: np.ndarray, weights: np.ndarray) -> None:
    if values.ndim != 2:
        raise ValueError("values must have shape [samples, features]")
    if weights.ndim != 1 or weights.shape[0] != values.shape[0]:
        raise ValueError("weights must have one positive value per sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("sample values must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("sample weights must be finite and positive")


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> Mapping[float, np.ndarray]:
    """Exact weighted quantiles for ``[samples, features]`` values.

    This matches ``numpy.interp`` over weighted sample centers, which is the
    definition used by the previous full-history implementation, but sorts
    all features in one vectorized pass.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    _validate_sample_matrix(values, weights)

    requested = tuple(float(q) for q in quantiles)
    if any(q < 0.0 or q > 1.0 for q in requested):
        raise ValueError("quantiles must be between 0 and 1")
    if not requested:
        return {}

    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    weight_matrix = np.broadcast_to(weights[:, None], values.shape)
    sorted_weights = np.take_along_axis(weight_matrix, order, axis=0)
    centers = np.cumsum(sorted_weights, axis=0) - 0.5 * sorted_weights
    total_weight = float(weights.sum())
    n_samples, n_features = values.shape
    columns = np.arange(n_features)

    outputs = {}
    for quantile in requested:
        target = quantile * total_weight
        upper = np.sum(centers < target, axis=0)
        result = np.empty(n_features, dtype=np.float64)

        left_edge = upper == 0
        right_edge = upper >= n_samples
        middle = ~(left_edge | right_edge)
        result[left_edge] = sorted_values[0, columns[left_edge]]
        result[right_edge] = sorted_values[-1, columns[right_edge]]

        if np.any(middle):
            middle_columns = columns[middle]
            upper_index = upper[middle]
            lower_index = upper_index - 1
            lower_center = centers[lower_index, middle_columns]
            upper_center = centers[upper_index, middle_columns]
            lower_value = sorted_values[lower_index, middle_columns]
            upper_value = sorted_values[upper_index, middle_columns]
            fraction = (target - lower_center) / (upper_center - lower_center)
            result[middle] = lower_value + fraction * (upper_value - lower_value)

        outputs[quantile] = result
    return outputs


def weighted_empirical_crps(
    values: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Exact weighted empirical CRPS in ``O(M log M)`` per feature.

    ``values`` has shape ``[M, features]`` and ``target`` has shape
    ``[features]``.  Sorting replaces the quadratic pairwise-distance matrix
    while preserving the exact empirical-distribution score.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    _validate_sample_matrix(values, weights)
    if target.shape != (values.shape[1],):
        raise ValueError("target must have one value per feature")
    if not np.all(np.isfinite(target)):
        raise ValueError("CRPS targets must be finite")

    probabilities = weights / weights.sum()
    first_term = np.sum(
        probabilities[:, None] * np.abs(values - target[None, :]), axis=0
    )

    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    probability_matrix = np.broadcast_to(probabilities[:, None], values.shape)
    sorted_probabilities = np.take_along_axis(probability_matrix, order, axis=0)
    cumulative_probability = np.cumsum(sorted_probabilities, axis=0)
    half_pairwise_term = np.sum(
        sorted_probabilities
        * sorted_values
        * (2.0 * cumulative_probability - sorted_probabilities - 1.0),
        axis=0,
    )
    return np.maximum(first_term - half_pairwise_term, 0.0)


class StreamingWindowAggregator:
    """Aggregate ordered overlapping windows with bounded active memory."""

    def __init__(
        self,
        *,
        total_length: int,
        window_size: int,
        n_features: int,
        position_weights: Optional[np.ndarray] = None,
        quantiles: Sequence[float] = (0.05, 0.95),
        crps_targets: Optional[np.ndarray] = None,
        crps_mask: Optional[np.ndarray] = None,
    ) -> None:
        if total_length < 1 or window_size < 1 or n_features < 1:
            raise ValueError("total_length, window_size, and n_features must be positive")
        self.total_length = int(total_length)
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.quantiles = tuple(float(q) for q in quantiles)
        if any(q < 0.0 or q > 1.0 for q in self.quantiles):
            raise ValueError("quantiles must be between 0 and 1")

        if position_weights is None:
            position_weights = np.ones(self.window_size, dtype=np.float64)
        self.position_weights = np.asarray(position_weights, dtype=np.float64)
        if (
            self.position_weights.shape != (self.window_size,)
            or not np.all(np.isfinite(self.position_weights))
            or np.any(self.position_weights <= 0)
        ):
            raise ValueError(
                "position_weights must be finite, positive, and have length window_size"
            )

        if (crps_targets is None) != (crps_mask is None):
            raise ValueError("crps_targets and crps_mask must be provided together")
        self.crps_targets = None
        self.crps_mask = None
        self.crps = None
        if crps_targets is not None:
            targets = np.asarray(crps_targets, dtype=np.float64)
            mask = np.asarray(crps_mask, dtype=bool)
            expected_shape = (self.total_length, self.n_features)
            if targets.shape != expected_shape or mask.shape != expected_shape:
                raise ValueError(
                    "crps_targets and crps_mask must have shape "
                    "[total_length, n_features]"
                )
            self.crps_targets = targets
            self.crps_mask = mask
            self.crps = np.full(expected_shape, np.nan, dtype=np.float64)

        output_shape = (self.total_length, self.n_features)
        self.mean = np.full(output_shape, np.nan, dtype=np.float64)
        self.variance = np.full(output_shape, np.nan, dtype=np.float64)
        self.overlap_count = np.zeros(self.total_length, dtype=np.int64)
        self.sample_count = np.zeros(self.total_length, dtype=np.int64)
        self.effective_sample_size = np.full(self.total_length, np.nan, dtype=np.float64)
        self.quantile_values = {
            quantile: np.full(output_shape, np.nan, dtype=np.float64)
            for quantile in self.quantiles
        }

        self._active_values: dict[int, list[np.ndarray]] = {}
        self._active_weights: dict[int, list[np.ndarray]] = {}
        self._last_start: Optional[int] = None
        self._finished = False
        self.peak_active_positions = 0
        self.peak_active_values = 0

    def add(self, start: int, samples: np.ndarray) -> None:
        """Add one ``[MC, window, features]`` window in start-time order."""
        if self._finished:
            raise RuntimeError("cannot add windows after finish()")
        start = int(start)
        if self._last_start is not None and start < self._last_start:
            raise ValueError("windows must be added in nondecreasing start order")
        end = start + self.window_size
        if start < 0 or end > self.total_length:
            raise ValueError(f"window [{start}, {end}) exceeds total_length={self.total_length}")

        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 3 or samples.shape[1:] != (
            self.window_size,
            self.n_features,
        ):
            raise ValueError(
                "samples must have shape [mc, window_size, n_features]"
            )
        if samples.shape[0] < 1 or not np.all(np.isfinite(samples)):
            raise ValueError("samples must contain at least one finite MC draw")

        # Any active timestamp before this start cannot be covered by this or
        # any later window, so it is complete and can be released now.
        self._finalize_before(start)
        for local_position, global_position in enumerate(range(start, end)):
            self._active_values.setdefault(global_position, []).append(
                samples[:, local_position, :]
            )
            self._active_weights.setdefault(global_position, []).append(
                np.full(
                    samples.shape[0],
                    self.position_weights[local_position],
                    dtype=np.float64,
                )
            )

        self._last_start = start
        self.peak_active_positions = max(
            self.peak_active_positions, len(self._active_values)
        )
        active_value_count = sum(
            chunk.shape[0]
            for chunks in self._active_values.values()
            for chunk in chunks
        )
        self.peak_active_values = max(self.peak_active_values, active_value_count)

    def _finalize_before(self, cutoff: int) -> None:
        positions = [position for position in self._active_values if position < cutoff]
        for position in sorted(positions):
            self._finalize_position(position)

    def _finalize_position(self, position: int) -> None:
        value_chunks = self._active_values.pop(position)
        weight_chunks = self._active_weights.pop(position)
        values = np.concatenate(value_chunks, axis=0)
        weights = np.concatenate(weight_chunks, axis=0)
        normalizer = weights.sum()
        self.overlap_count[position] = len(value_chunks)
        self.sample_count[position] = values.shape[0]
        self.effective_sample_size[position] = normalizer ** 2 / np.sum(weights ** 2)
        self.mean[position] = np.sum(values * weights[:, None], axis=0) / normalizer
        self.variance[position] = np.maximum(
            np.sum(values * values * weights[:, None], axis=0) / normalizer
            - self.mean[position] ** 2,
            0.0,
        )
        for quantile, result in weighted_quantiles(
            values, weights, self.quantiles
        ).items():
            self.quantile_values[quantile][position] = result

        if self.crps is not None:
            score_mask = self.crps_mask[position] & np.isfinite(
                self.crps_targets[position]
            )
            if np.any(score_mask):
                self.crps[position, score_mask] = weighted_empirical_crps(
                    values[:, score_mask],
                    weights,
                    self.crps_targets[position, score_mask],
                )

    def finish(self) -> Mapping[str, object]:
        """Finalize remaining timestamps and return full-length summaries."""
        if not self._finished:
            self._finalize_before(self.total_length)
            self._finished = True
        result = {
            "mean": self.mean,
            "variance": self.variance,
            "overlap_count": self.overlap_count,
            "sample_count": self.sample_count,
            "effective_sample_size": self.effective_sample_size,
            "quantiles": self.quantile_values,
            "peak_active_positions": self.peak_active_positions,
            "peak_active_values": self.peak_active_values,
        }
        if self.crps is not None:
            result["crps"] = self.crps
        return result


def aggregate_ordered_windows(
    windows: Iterable[WindowSamples],
    *,
    total_length: int,
    window_size: int,
    n_features: int,
    position_weights: Optional[np.ndarray] = None,
    quantiles: Sequence[float] = (0.05, 0.95),
    crps_targets: Optional[np.ndarray] = None,
    crps_mask: Optional[np.ndarray] = None,
) -> Mapping[str, object]:
    """Convenience adapter over :class:`StreamingWindowAggregator`."""
    aggregator = StreamingWindowAggregator(
        total_length=total_length,
        window_size=window_size,
        n_features=n_features,
        position_weights=position_weights,
        quantiles=quantiles,
        crps_targets=crps_targets,
        crps_mask=crps_mask,
    )
    seen = False
    for start, samples in windows:
        aggregator.add(start, samples)
        seen = True
    if not seen:
        raise ValueError("window samples cannot be empty")
    return aggregator.finish()
