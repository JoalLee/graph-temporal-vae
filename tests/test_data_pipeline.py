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
    compute_time_cyclical_features,
    compute_window_starts,
    extract_censor_marker_mask,
    inverse_target_values,
    load_frame,
    load_modality_frame,
    sample_block_heldout_mask_to_ratio,
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


def test_load_frame_parses_numeric_qc_marker_and_retains_mask(tmp_path):
    path = tmp_path / "marked.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "chem": ["0.15_", "0.2", np.nan],
            "psd": ["0", "1.5_", "2.0"],
        }
    ).to_csv(path, index=False)

    frame = load_frame(
        [path], "time", target_cols=["chem", "psd"], aux_cols=[]
    )

    np.testing.assert_allclose(frame["chem"].to_numpy(), [0.15, 0.2, np.nan], equal_nan=True)
    np.testing.assert_allclose(frame["psd"].to_numpy(), [0.0, 1.5, 2.0])
    marker_mask = extract_censor_marker_mask(frame, ["chem", "psd"])
    np.testing.assert_array_equal(
        marker_mask,
        np.array([[True, False], [False, True], [False, False]]),
    )


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


def test_load_frame_converts_raw_wind_pair_to_vector_components(tmp_path):
    ts = pd.date_range("2024-01-01", periods=4, freq="h")
    path = tmp_path / "raw_wind.csv"
    raw = pd.DataFrame(
        {
            "time": ts,
            "target": [1.0, 2.0, 3.0, 4.0],
            "AT": [20.0, 21.0, 22.0, 23.0],
            "WS": [1.0, 2.0, 3.0, 4.0],
            "WD": [0.0, 90.0, 180.0, 270.0],
        }
    )
    raw.to_csv(path, index=False)

    frame = load_frame(
        [path],
        "time",
        target_cols=["target"],
        aux_cols=["AT", "WS", "WD"],
        canonicalize_wind=True,
    )

    direction_rad = np.deg2rad((270.0 - raw["WD"].to_numpy()) % 360.0)
    expected_u = raw["WS"].to_numpy() * np.cos(direction_rad)
    expected_v = raw["WS"].to_numpy() * np.sin(direction_rad)
    assert list(frame.columns) == ["target", "AT", "wind_u", "wind_v"]
    np.testing.assert_allclose(frame["wind_u"], expected_u, atol=1e-12)
    np.testing.assert_allclose(frame["wind_v"], expected_v, atol=1e-12)
    assert frame.attrs["wind_encoding"] == "ws_wd_to_uv_v1"


def test_load_modality_frame_converts_wind_and_reuses_schema_at_inference():
    ts = pd.date_range("2024-01-01", periods=4, freq="h")
    chemistry = pd.DataFrame({"time": ts, "SO2": [1.0, 2.0, 3.0, 4.0]})
    meteorology = pd.DataFrame(
        {
            "time": ts,
            "AT": [20.0, 21.0, 22.0, 23.0],
            "RH": [50.0, 51.0, 52.0, 53.0],
            "WS": [1.0, 2.0, 3.0, 4.0],
            "WD": [0.0, 90.0, 180.0, 270.0],
        }
    )
    files = ModalityInputs(chemistry=chemistry, meteorology=meteorology)

    train_frame, schema = load_modality_frame(files, "time")
    infer_frame, infer_schema = load_modality_frame(
        files, "time", expected_schema=schema
    )

    assert schema.meteorology_cols == ["AT", "RH", "wind_u", "wind_v"]
    assert infer_schema.meteorology_cols == schema.meteorology_cols
    assert list(train_frame.columns) == schema.target_cols + schema.auxiliary_cols
    assert list(infer_frame.columns) == list(train_frame.columns)
    pd.testing.assert_frame_equal(
        train_frame.reset_index(drop=True), infer_frame.reset_index(drop=True)
    )
    assert "WS" not in train_frame.columns
    assert "WD" not in train_frame.columns


def test_load_modality_frame_rejects_mixed_raw_and_derived_wind_columns():
    ts = pd.date_range("2024-01-01", periods=3, freq="h")
    chemistry = pd.DataFrame({"time": ts, "SO2": [1.0, 2.0, 3.0]})
    meteorology = pd.DataFrame(
        {
            "time": ts,
            "AT": [20.0, 21.0, 22.0],
            "WS": [1.0, 2.0, 3.0],
            "WD": [0.0, 90.0, 180.0],
            "wind_u": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="both raw WS/WD and derived wind_u/wind_v"):
        load_modality_frame(
            ModalityInputs(chemistry=chemistry, meteorology=meteorology),
            "time",
        )


def test_compute_time_cyclical_features_matches_known_timestamps():
    # 2024-01-01 00:00 is a Monday in January: hour=0, dow=0, month-index=0.
    index = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 06:00", "2024-07-04 18:00"])
    features = compute_time_cyclical_features(index)

    assert list(features.columns) == [
        "time_hour_sin", "time_hour_cos",
        "time_dow_sin", "time_dow_cos",
        "time_month_sin", "time_month_cos",
    ]
    row0 = features.iloc[0]
    assert row0["time_hour_sin"] == pytest.approx(0.0, abs=1e-9)
    assert row0["time_hour_cos"] == pytest.approx(1.0, abs=1e-9)
    assert row0["time_dow_sin"] == pytest.approx(0.0, abs=1e-9)
    assert row0["time_dow_cos"] == pytest.approx(1.0, abs=1e-9)
    assert row0["time_month_sin"] == pytest.approx(0.0, abs=1e-9)
    assert row0["time_month_cos"] == pytest.approx(1.0, abs=1e-9)

    # 06:00 is a quarter turn around the 24h clock.
    row1 = features.iloc[1]
    assert row1["time_hour_sin"] == pytest.approx(1.0, abs=1e-9)
    assert row1["time_hour_cos"] == pytest.approx(0.0, abs=1e-9)

    # Every value stays on the unit circle regardless of the calendar date.
    for _, row in features.iterrows():
        assert row["time_hour_sin"] ** 2 + row["time_hour_cos"] ** 2 == pytest.approx(1.0)
        assert row["time_dow_sin"] ** 2 + row["time_dow_cos"] ** 2 == pytest.approx(1.0)
        assert row["time_month_sin"] ** 2 + row["time_month_cos"] ** 2 == pytest.approx(1.0)


def test_load_modality_frame_can_append_time_cyclical_features(tmp_path):
    ts = pd.date_range("2024-01-01", periods=6, freq="h")
    chem_path = tmp_path / "chem.csv"
    met_path = tmp_path / "met.csv"
    pd.DataFrame({"time": ts, "SO2": np.arange(6, dtype=float)}).to_csv(chem_path, index=False)
    pd.DataFrame({"time": ts, "AT": np.arange(6, dtype=float) + 20}).to_csv(met_path, index=False)

    frame, schema = load_modality_frame(
        ModalityFiles(chemistry=[chem_path], meteorology=[met_path]),
        timestamp_col="time",
        add_time_cyclical_features=True,
    )

    assert schema.add_time_cyclical_features is True
    assert schema.meteorology_cols == ["AT"]
    assert schema.auxiliary_cols == [
        "AT", "time_hour_sin", "time_hour_cos",
        "time_dow_sin", "time_dow_cos", "time_month_sin", "time_month_cos",
    ]
    assert schema.aux_dim == 7
    assert list(frame.columns) == schema.target_cols + schema.auxiliary_cols
    expected = compute_time_cyclical_features(ts)
    pd.testing.assert_series_equal(
        frame["time_hour_sin"].reset_index(drop=True),
        expected["time_hour_sin"].reset_index(drop=True),
        check_names=False,
    )

    # Round-tripping the schema through to_dict/from_dict preserves the flag.
    restored = DataSchema.from_dict(schema.to_dict())
    assert restored.add_time_cyclical_features is True
    assert restored.aux_dim == 7


def test_load_modality_frame_reproduces_time_features_at_inference_from_schema(tmp_path):
    ts = pd.date_range("2024-01-01", periods=6, freq="h")
    chem_path = tmp_path / "chem.csv"
    met_path = tmp_path / "met.csv"
    pd.DataFrame({"time": ts, "SO2": np.arange(6, dtype=float)}).to_csv(chem_path, index=False)
    pd.DataFrame({"time": ts, "AT": np.arange(6, dtype=float) + 20}).to_csv(met_path, index=False)
    files = ModalityFiles(chemistry=[chem_path], meteorology=[met_path])

    _train_frame, schema = load_modality_frame(
        files, "time", add_time_cyclical_features=True
    )

    # Inference call omits add_time_cyclical_features entirely -- it must be
    # picked up from expected_schema, not silently dropped.
    infer_frame, infer_schema = load_modality_frame(files, "time", expected_schema=schema)
    assert infer_schema.add_time_cyclical_features is True
    assert list(infer_frame.columns) == schema.target_cols + schema.auxiliary_cols


def test_load_modality_frame_time_cyclical_features_default_off_unchanged(tmp_path):
    ts = pd.date_range("2024-01-01", periods=4, freq="h")
    chem_path = tmp_path / "chem.csv"
    met_path = tmp_path / "met.csv"
    pd.DataFrame({"time": ts, "SO2": [1, 2, 3, 4]}).to_csv(chem_path, index=False)
    pd.DataFrame({"time": ts, "AT": [20, 21, 22, 23]}).to_csv(met_path, index=False)

    _frame, schema = load_modality_frame(
        ModalityFiles(chemistry=[chem_path], meteorology=[met_path]), "time"
    )

    assert schema.add_time_cyclical_features is False
    assert schema.auxiliary_cols == schema.meteorology_cols == ["AT"]
    assert schema.aux_dim == 1


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


def test_block_heldout_mask_reaches_ratio_for_each_target_family():
    observed = np.ones((672, 10), dtype=bool)
    config = {
        "target_ratio": 0.15,
        "mean_duration": 24,
        "std_duration": 0,
        "min_duration": 24,
        "max_duration": 24,
        "n_chem": 4,
        "chem_blocks": 1,
        "psd_blocks": 1,
    }

    heldout = sample_block_heldout_mask_to_ratio(observed, config, seed=100003)
    repeated = sample_block_heldout_mask_to_ratio(observed, config, seed=100003)

    assert np.array_equal(heldout, repeated)
    assert bool((heldout <= observed).all())
    assert heldout[:, :4].mean() == pytest.approx(0.15, abs=0.002)
    assert heldout[:, 4:].mean() == pytest.approx(0.15, abs=0.002)
    # A 15% quota on a 672-row timeline needs several 24-row blocks, not the
    # single block that previously produced only a few percent held out.
    assert np.flatnonzero(heldout[:, :4].any(axis=1)).size > 24
    assert np.flatnonzero(heldout[:, 4:].any(axis=1)).size > 24


def test_timeline_epoch_dynamic_mask_is_shared_by_overlapping_windows():
    target = np.ones((32, 4), dtype=np.float32)
    aux = np.zeros((32, 0), dtype=np.float32)
    fixed = np.zeros_like(target, dtype=bool)
    fixed[10:12, 0] = True
    dataset = WindowedTimeSeriesDataset(
        target,
        aux,
        window_size=8,
        stride=4,
        mode="train",
        denoise_prob=0.0,
        dynamic_mask_config={
            "scope": "timeline_epoch",
            "target_ratio": 0.25,
            "mean_duration": 4,
            "std_duration": 0,
            "min_duration": 4,
            "max_duration": 4,
            "n_chem": 2,
        },
        fixed_mask=fixed,
        seed=7,
    )

    def collect_absolute_masks():
        seen = {}
        for idx, start in enumerate(dataset.starts):
            local = dataset[idx]["heldout_mask"].numpy().astype(bool)
            for offset in range(dataset.window_size):
                absolute_row = start + offset
                if absolute_row in seen:
                    assert np.array_equal(seen[absolute_row], local[offset])
                else:
                    seen[absolute_row] = local[offset].copy()
        return np.stack([seen[row] for row in range(len(target))])

    dataset.set_epoch(3)
    epoch_three = collect_absolute_masks()
    dynamic_three = dataset._timeline_dynamic_mask.copy()
    dataset.set_epoch(3)
    assert np.array_equal(epoch_three, collect_absolute_masks())
    dataset.set_epoch(4)
    epoch_four = collect_absolute_masks()

    assert not np.array_equal(epoch_three, epoch_four)
    assert not bool((dataset._timeline_dynamic_mask & fixed).any())
    eligible = ~fixed
    coverage = dataset.window_coverage[:, None]
    chem_ratio = (
        (dynamic_three[:, :2] * coverage).sum()
        / (eligible[:, :2] * coverage).sum()
    )
    psd_ratio = (
        (dynamic_three[:, 2:] * coverage).sum()
        / (eligible[:, 2:] * coverage).sum()
    )
    assert chem_ratio == pytest.approx(0.25, abs=0.03)
    assert psd_ratio == pytest.approx(0.25, abs=0.03)


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
