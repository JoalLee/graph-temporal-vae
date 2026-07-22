import numpy as np
import pandas as pd
import pytest

from graph_tcn_vae.data import (
    NaNAwareStandardScaler,
    WindowedTimeSeriesDataset,
    chronological_split_index,
    compute_window_starts,
    inverse_target_values,
    load_frame,
    transform_target_values,
)


def test_scaler_round_trip_ignores_nan():
    array = np.array([[1.0, np.nan], [2.0, 10.0], [3.0, 12.0], [np.nan, 14.0]])
    scaler = NaNAwareStandardScaler().fit(array)

    scaled = scaler.transform(array)
    assert np.isfinite(scaled).all()

    restored = scaler.inverse_transform(scaled)
    # NaN positions were zero-filled by transform(), so they won't round-trip;
    # observed positions must.
    assert restored[1, 0] == pytest.approx(2.0)
    assert restored[1, 1] == pytest.approx(10.0)
    assert restored[2, 1] == pytest.approx(12.0)


def test_scaler_handles_constant_and_allnan_columns():
    array = np.array([[5.0, np.nan], [5.0, np.nan], [5.0, np.nan]])
    scaler = NaNAwareStandardScaler().fit(array)
    scaled = scaler.transform(array)
    assert np.isfinite(scaled).all()


def test_log1p_target_transform_round_trip_and_rejects_negative_values():
    values = np.array([[0.0, 9.0], [np.nan, 3.0]])
    transformed = transform_target_values(values, "log1p")
    assert np.isnan(transformed[1, 0])
    assert np.allclose(inverse_target_values(transformed, "log1p"), values, equal_nan=True)

    with pytest.raises(ValueError, match="non-negative"):
        transform_target_values(np.array([[1.0, -0.1]]), "log1p")


def test_compute_window_starts_covers_full_series():
    starts = compute_window_starts(n=50, window_size=10, stride=7)
    assert starts[0] == 0
    assert starts[-1] == 40  # tail window always included
    # every position 0..49 is covered by at least one window
    covered = set()
    for s in starts:
        covered.update(range(s, s + 10))
    assert covered == set(range(50))


def test_compute_window_starts_too_short_returns_empty():
    assert compute_window_starts(n=5, window_size=10, stride=1) == []


def test_chronological_split_index_respects_fraction():
    idx = chronological_split_index(100, val_fraction=0.2)
    assert idx == 80


def test_load_frame_joins_multiple_csvs_on_timestamp(tmp_path):
    ts = pd.date_range("2024-01-01", periods=6, freq="h")
    df_a = pd.DataFrame({"time": ts, "a": range(6)})
    df_b = pd.DataFrame({"time": ts, "b": range(6, 12)})
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)

    frame = load_frame([path_a, path_b], "time", target_cols=["a"], aux_cols=["b"])
    assert list(frame["a"]) == list(range(6))
    assert list(frame["b"]) == list(range(6, 12))
    assert frame.index.is_monotonic_increasing


def test_load_frame_missing_column_raises(tmp_path):
    ts = pd.date_range("2024-01-01", periods=3, freq="h")
    df = pd.DataFrame({"time": ts, "a": [1, 2, 3]})
    path = tmp_path / "a.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_frame([path], "time", target_cols=["a", "missing"], aux_cols=[])


def test_windowed_dataset_shapes_and_mask():
    n, window_size, stride = 40, 8, 4
    target = np.random.randn(n, 3).astype(np.float32)
    target[5:9, 0] = np.nan  # a genuine gap
    aux = np.random.randn(n, 2).astype(np.float32)

    dataset = WindowedTimeSeriesDataset(target, aux, window_size, stride, mode="train", denoise_prob=0.2, seed=0)
    assert len(dataset) == len(compute_window_starts(n, window_size, stride))

    sample = dataset[0]
    assert sample["target"].shape == (window_size, 3)
    assert sample["cond"].shape == (window_size, 4)
    assert sample["obs_mask"].shape == (window_size, 3)
    assert sample["input_mask"].shape == (window_size, 3)
    # denoising can only turn observed points off, never turn missing points on
    assert bool(((sample["input_mask"] == 1) & (sample["obs_mask"] == 0)).any()) is False


def test_windowed_dataset_val_mode_has_no_denoising():
    n, window_size, stride = 20, 5, 5
    target = np.random.randn(n, 2).astype(np.float32)
    aux = np.random.randn(n, 1).astype(np.float32)

    dataset = WindowedTimeSeriesDataset(target, aux, window_size, stride, mode="val")
    sample = dataset[0]
    assert bool((sample["input_mask"] == sample["obs_mask"]).all())


def test_windowed_dataset_exposes_aux_missingness_and_dynamic_heldout_mask():
    target = np.ones((16, 2), dtype=np.float32)
    aux = np.ones((16, 2), dtype=np.float32)
    aux[3, 0] = np.nan
    aux[7, 1] = np.nan

    dataset = WindowedTimeSeriesDataset(
        target,
        aux,
        window_size=8,
        stride=8,
        mode="train",
        denoise_prob=0.0,
        aux_mask_channel=True,
        dynamic_mask_config={
            "target_ratio": 0.5,
            "mean_duration": 2,
            "std_duration": 0,
            "min_duration": 2,
            "max_duration": 2,
            "n_chem": 1,
        },
        seed=0,
    )

    sample = dataset[0]
    assert sample["cond"].shape == (8, 4)
    # First half is the scaled/value channel; second half is an independent
    # observedness channel for the auxiliary inputs.
    assert sample["cond"][3, 2].item() == 0.0
    assert sample["cond"][3, 3].item() == 1.0
    assert sample["cond"][7, 2].item() == 1.0
    assert sample["cond"][7, 3].item() == 0.0

    heldout = sample["heldout_mask"].numpy()
    assert heldout.shape == (8, 2)
    assert bool((heldout <= sample["obs_mask"].numpy()).all())
    assert bool((sample["input_mask"].numpy() == sample["obs_mask"].numpy() - heldout).all())


def test_windowed_dataset_val_uses_fixed_selection_mask():
    target = np.ones((8, 1), dtype=np.float32)
    aux = np.zeros((8, 0), dtype=np.float32)
    selection = np.zeros((8, 1), dtype=bool)
    selection[[1, 6], 0] = True

    dataset = WindowedTimeSeriesDataset(
        target,
        aux,
        window_size=4,
        stride=4,
        mode="val",
        selection_mask=selection,
    )
    first = dataset[0]
    second = dataset[1]
    assert first["heldout_mask"].numpy().ravel().tolist() == [False, True, False, False]
    assert second["heldout_mask"].numpy().ravel().tolist() == [False, False, True, False]
    assert bool((first["input_mask"] == first["obs_mask"] - first["heldout_mask"]).all())
