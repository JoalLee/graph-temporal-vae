import numpy as np
import pandas as pd
import pytest

from graph_temporal_vae.contracts import (
    DataSchema,
    ModalityFiles,
    ModalityInputs,
    ModalityPreprocessing,
    PreprocessingConfig,
)
from graph_temporal_vae.data import (
    NaNAwareStandardScaler,
    WindowedTimeSeriesDataset,
    chronological_split_index,
    compute_window_starts,
    inverse_target_values,
    load_frame,
    load_modality_frame,
    transform_target_values,
)
from graph_temporal_vae.preprocessing import (
    NaNAwareAffineScaler,
    fit_target_scaler,
    observed_targets_to_output,
    transform_targets,
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


def test_scaler_handles_constant_columns_but_rejects_allnan_columns():
    constant = np.array([[5.0], [5.0], [5.0]])
    scaler = NaNAwareStandardScaler().fit(constant)
    assert np.isfinite(scaler.transform(constant)).all()

    all_nan = np.array([[5.0, np.nan], [5.0, np.nan], [5.0, np.nan]])
    with pytest.raises(ValueError, match="all values are missing"):
        NaNAwareStandardScaler().fit(all_nan)


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


def test_load_frame_rejects_irregular_grid_by_default(tmp_path):
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 07:00"]
            ),
            "a": [1.0, 2.0, 3.0],
        }
    )
    path = tmp_path / "irregular.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Timestamp grid is irregular"):
        load_frame([path], "time", target_cols=["a"], aux_cols=[])


def test_load_frame_can_reindex_missing_grid_rows(tmp_path):
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"]
            ),
            "a": [1.0, 2.0, 4.0],
        }
    )
    path = tmp_path / "missing_hour.csv"
    df.to_csv(path, index=False)
    frame = load_frame(
        [path],
        "time",
        target_cols=["a"],
        aux_cols=[],
        expected_frequency="1h",
        time_grid_policy="reindex",
    )
    assert len(frame) == 4
    assert pd.isna(frame.loc[pd.Timestamp("2024-01-01 02:00"), "a"])
    assert frame.attrs["frequency"].lower() == "h"


def test_load_frame_rejects_duplicate_timestamps(tmp_path):
    df = pd.DataFrame(
        {
            "time": ["2024-01-01 00:00", "2024-01-01 00:00"],
            "a": [1.0, 2.0],
        }
    )
    path = tmp_path / "duplicate.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_frame([path], "time", target_cols=["a"], aux_cols=[])


def test_load_frame_rejects_all_missing_target(tmp_path):
    ts = pd.date_range("2024-01-01", periods=3, freq="h")
    path = tmp_path / "all_missing.csv"
    pd.DataFrame({"time": ts, "a": [np.nan, np.nan, np.nan]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="no observed values"):
        load_frame([path], "time", target_cols=["a"], aux_cols=[])


def test_load_frame_rejects_same_column_from_multiple_csvs(tmp_path):
    ts = pd.date_range("2024-01-01", periods=3, freq="h")
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    pd.DataFrame({"time": ts, "a": [1, 2, 3]}).to_csv(path_a, index=False)
    pd.DataFrame({"time": ts, "a": [4, 5, 6]}).to_csv(path_b, index=False)
    with pytest.raises(ValueError, match="appears in multiple CSVs"):
        load_frame([path_a, path_b], "time", target_cols=["a"], aux_cols=[])


def test_load_frame_missing_column_raises(tmp_path):
    ts = pd.date_range("2024-01-01", periods=3, freq="h")
    df = pd.DataFrame({"time": ts, "a": [1, 2, 3]})
    path = tmp_path / "a.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_frame([path], "time", target_cols=["a", "missing"], aux_cols=[])


def test_load_modality_frame_discovers_roles_and_sorts_psd_bins(tmp_path):
    ts = pd.date_range("2024-01-01", periods=6, freq="h")
    chem_path = tmp_path / "chem.csv"
    psd_path = tmp_path / "psd.csv"
    met_path = tmp_path / "met.csv"
    pd.DataFrame({"time": ts, "SO2": np.arange(6), "NO3-": np.arange(6) + 1}).to_csv(
        chem_path, index=False
    )
    pd.DataFrame({"time": ts, "100.0": np.arange(6) + 2, "12.0": np.arange(6) + 3}).to_csv(
        psd_path, index=False
    )
    pd.DataFrame({"time": ts, "AT": np.arange(6) + 20, "RH": np.arange(6) + 50}).to_csv(
        met_path, index=False
    )

    frame, schema = load_modality_frame(
        ModalityFiles(chemistry=[chem_path], psd=[psd_path], meteorology=[met_path]),
        timestamp_col="time",
    )

    assert schema.chemistry_cols == ["SO2", "NO3-"]
    assert schema.psd_cols == ["12.0", "100.0"]
    assert schema.psd_diameters_nm == [12.0, 100.0]
    assert schema.meteorology_cols == ["AT", "RH"]
    assert schema.target_cols == ["SO2", "NO3-", "12.0", "100.0"]
    assert list(frame.columns) == schema.target_cols + schema.meteorology_cols
    assert schema.n_chem == 2


def test_load_modality_frame_enforces_training_schema_at_inference(tmp_path):
    ts = pd.date_range("2024-01-01", periods=4, freq="h")
    chem_path = tmp_path / "chem.csv"
    pd.DataFrame({"time": ts, "SO2": [1, 2, 3, 4]}).to_csv(chem_path, index=False)
    files = ModalityFiles(chemistry=[chem_path])
    _frame, schema = load_modality_frame(files, "time")

    changed_path = tmp_path / "changed.csv"
    pd.DataFrame({"time": ts, "SO2": [1, 2, 3, 4], "NO": [0, 0, 0, 0]}).to_csv(
        changed_path, index=False
    )
    with pytest.raises(ValueError, match="extra=.*NO"):
        load_modality_frame(
            ModalityFiles(chemistry=[changed_path]),
            "time",
            expected_schema=schema,
        )


def test_load_modality_frame_rejects_non_numeric_psd_column_names(tmp_path):
    ts = pd.date_range("2024-01-01", periods=3, freq="h")
    psd_path = tmp_path / "psd.csv"
    pd.DataFrame({"time": ts, "small_bin": [1, 2, 3]}).to_csv(psd_path, index=False)
    with pytest.raises(ValueError, match="not a numeric particle diameter"):
        load_modality_frame(ModalityFiles(psd=[psd_path]), "time")


def test_load_modality_frame_accepts_dataframes_and_datetime_index():
    ts = pd.date_range("2024-01-01", periods=6, freq="h")
    chemistry = pd.DataFrame({"SO2": np.arange(6, dtype=float)}, index=ts)
    psd = pd.DataFrame({"100.0": np.arange(6) + 2, "12.0": np.arange(6) + 3}, index=ts)
    meteorology = pd.DataFrame({"time": ts, "AT": np.arange(6) + 20})

    frame, schema = load_modality_frame(
        ModalityInputs(
            chemistry=chemistry,
            psd=psd,
            meteorology=meteorology,
        ),
        "time",
    )

    assert schema.target_cols == ["SO2", "12.0", "100.0"]
    assert schema.meteorology_cols == ["AT"]
    assert frame.index.name == "time"
    assert len(frame) == 6


def test_load_modality_frame_supports_chem_only_and_psd_only():
    ts = pd.date_range("2024-01-01", periods=4, freq="h")
    chem_frame, chem_schema = load_modality_frame(
        ModalityInputs(chemistry=pd.DataFrame({"time": ts, "SO2": [1, 2, 3, 4]})),
        "time",
    )
    assert chem_schema.chemistry_cols == ["SO2"]
    assert chem_schema.psd_cols == []
    assert list(chem_frame.columns) == ["SO2"]

    psd_frame, psd_schema = load_modality_frame(
        ModalityInputs(psd=pd.DataFrame({"time": ts, "100": [1, 2, 3, 4], "10": [2, 3, 4, 5]})),
        "time",
    )
    assert psd_schema.chemistry_cols == []
    assert psd_schema.psd_cols == ["10", "100"]
    assert list(psd_frame.columns) == ["10", "100"]


def test_load_modality_frame_rejects_duplicate_numeric_psd_diameters():
    ts = pd.date_range("2024-01-01", periods=3, freq="h")
    psd = pd.DataFrame({"time": ts, "12": [1, 2, 3], "12.0": [2, 3, 4]})
    with pytest.raises(ValueError, match="duplicate particle diameters"):
        load_modality_frame(ModalityInputs(psd=psd), "time")


def test_load_modality_frame_rejects_cross_modality_column_collision():
    ts = pd.date_range("2024-01-01", periods=3, freq="h")
    chemistry = pd.DataFrame({"time": ts, "SO2": [1, 2, 3]})
    meteorology = pd.DataFrame({"time": ts, "SO2": [4, 5, 6]})
    with pytest.raises(ValueError, match="multiple modalities"):
        load_modality_frame(
            ModalityInputs(chemistry=chemistry, meteorology=meteorology),
            "time",
        )


def test_load_modality_frame_rejects_inference_timezone_mismatch():
    utc = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    _frame, schema = load_modality_frame(
        ModalityInputs(chemistry=pd.DataFrame({"time": utc, "SO2": [1, 2, 3, 4]})),
        "time",
    )
    naive = pd.date_range("2024-01-01", periods=4, freq="h")
    with pytest.raises(ValueError, match="timezone does not match"):
        load_modality_frame(
            ModalityInputs(chemistry=pd.DataFrame({"time": naive, "SO2": [1, 2, 3, 4]})),
            "time",
            expected_schema=schema,
        )


def test_modality_preprocessing_supports_independent_transform_and_scaler_modes():
    schema = DataSchema(
        chemistry_cols=["SO2"],
        psd_cols=["12.0", "100.0"],
        psd_diameters_nm=[12.0, 100.0],
    )
    config = PreprocessingConfig(
        chemistry=ModalityPreprocessing(transform="log1p", scaler="standard"),
        psd=ModalityPreprocessing(transform="none", scaler="robust"),
        meteorology=ModalityPreprocessing(transform="none", scaler="minmax"),
    )
    raw = np.array([[0.0, 1.0, 10.0], [3.0, 3.0, 20.0], [8.0, 100.0, 30.0]])
    model_space = transform_targets(raw, schema, config)
    assert np.allclose(model_space[:, 0], np.log1p(raw[:, 0]))
    assert np.array_equal(model_space[:, 1:], raw[:, 1:])

    scaler = fit_target_scaler(model_space, schema, config)
    assert scaler.feature_kinds_ == ["standard", "robust", "robust"]
    restored = scaler.inverse_transform(scaler.transform(model_space))
    assert np.allclose(restored, model_space)


def test_observed_output_contract_avoids_double_log1p_inverse():
    schema = DataSchema(chemistry_cols=["SO2"])
    physical_config = PreprocessingConfig(
        chemistry=ModalityPreprocessing(
            transform="log1p", scaler="standard", output_transform="log1p"
        )
    )
    physical = np.array([[0.0], [9.0]])
    assert np.allclose(
        observed_targets_to_output(physical, schema, physical_config), physical
    )

    pretransformed_config = PreprocessingConfig(
        chemistry=ModalityPreprocessing(
            transform="none", scaler="standard", output_transform="log1p"
        )
    )
    logged = np.log1p(physical)
    assert np.allclose(
        observed_targets_to_output(logged, schema, pretransformed_config), physical
    )


def test_affine_scaler_state_round_trip_preserves_scaler_kind():
    values = np.array([[1.0, 10.0], [2.0, 20.0], [100.0, 30.0]])
    scaler = NaNAwareAffineScaler("robust").fit(values)
    restored = NaNAwareAffineScaler.from_dict(scaler.to_dict())
    assert restored.kind == "robust"
    assert np.allclose(restored.transform(values), scaler.transform(values))


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
