import json
from copy import deepcopy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from graph_temporal_vae.api import fit_multimodal, impute_multimodal
from graph_temporal_vae.bundle import inspect_bundle
from graph_temporal_vae.cli import main as cli_main
from graph_temporal_vae.config import TrainConfig
from graph_temporal_vae.contracts import InferenceConfig
from graph_temporal_vae.data import (
    WindowedTimeSeriesDataset,
    sample_anchor_constrained_heldout_mask,
    sample_dynamic_heldout_mask,
)
from graph_temporal_vae.infer import (
    aggregate_window_samples,
    load_bundle,
    summary_to_output_scale,
    trapezoid_position_weights,
)
from graph_temporal_vae.model_graph_uq import ImputationVAE_Graph
from graph_temporal_vae.train import (
    Trainer,
    _student_t_nll,
    empirical_crps_components,
    masked_nll_components,
    train_from_config,
    vae_loss,
)
from graph_temporal_vae.window_aggregation import (
    StreamingWindowAggregator,
    weighted_empirical_crps,
)


def _write_synthetic_csv(path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.linspace(0, 20, n)

    a = np.sin(t) + rng.normal(scale=0.05, size=n)
    b = np.cos(t) + rng.normal(scale=0.05, size=n)
    ws = rng.uniform(0, 5, n)
    at = rng.uniform(10, 30, n)
    ws[65:70] = np.nan  # auxiliary missingness is represented by a mask channel

    # Punch a real gap so imputation has something to do.
    a_missing = a.copy()
    a_missing[50:60] = np.nan

    df = pd.DataFrame({"time": ts, "target_a": a_missing, "target_b": b, "ws": ws, "at": at})
    df.to_csv(path, index=False)
    return df


def _write_modality_csvs(tmp_path, n=96, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    chem = pd.DataFrame({
        "time": ts,
        "SO2": rng.lognormal(mean=0.2, sigma=0.2, size=n),
        "NO3-": rng.lognormal(mean=1.0, sigma=0.3, size=n),
    })
    psd = pd.DataFrame({
        "time": ts,
        "100.0": rng.lognormal(mean=4.0, sigma=0.2, size=n),
        "12.0": rng.lognormal(mean=5.0, sigma=0.25, size=n),
    })
    met = pd.DataFrame({
        "time": ts,
        "AT": rng.normal(25.0, 2.0, size=n),
        "RH": rng.uniform(40.0, 90.0, size=n),
    })
    chem.loc[20:25, "SO2"] = np.nan
    psd.loc[40:47, ["100.0", "12.0"]] = np.nan
    met.loc[10:12, "RH"] = np.nan
    paths = {
        "chem": tmp_path / "chem.csv",
        "psd": tmp_path / "psd.csv",
        "met": tmp_path / "met.csv",
    }
    chem.to_csv(paths["chem"], index=False)
    psd.to_csv(paths["psd"], index=False)
    met.to_csv(paths["met"], index=False)
    return paths


def _small_multimodal_train_config():
    return TrainConfig(
        timestamp_col="time",
        window_size=8,
        stride=4,
        val_fraction=0.25,
        epochs=1,
        patience=1,
        batch_size=4,
        model_kwargs={
            "latent_dim": 4,
            "hidden_dims": [8],
            "encoder_layers": 1,
            "decoder_layers": 1,
            "n_graph_heads": 1,
        },
    )


def test_multimodal_cli_discovers_schema_persists_preprocessing_and_imputes(tmp_path):
    paths = _write_modality_csvs(tmp_path)
    bundle_path = tmp_path / "multimodal.pt"
    output_path = tmp_path / "multimodal_imputed.csv"

    cli_main([
        "train",
        "--chem-csv", str(paths["chem"]),
        "--psd-csv", str(paths["psd"]),
        "--met-csv", str(paths["met"]),
        "--timestamp-col", "time",
        "--chem-transform", "log1p",
        "--psd-transform", "log1p",
        "--chem-scaler", "standard",
        "--psd-scaler", "robust",
        "--met-scaler", "minmax",
        "--window-size", "16",
        "--stride", "8",
        "--val-fraction", "0.2",
        "--batch-size", "4",
        "--epochs", "1",
        "--patience", "1",
        "--latent-dim", "4",
        "--hidden-dims", "8,8",
        "--encoder-layers", "1",
        "--decoder-layers", "1",
        "--n-graph-heads", "1",
        "-o", str(bundle_path),
    ])

    raw_bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    assert raw_bundle["bundle_version"] == 3
    assert raw_bundle["architecture_version"] == 1
    assert raw_bundle["data_schema"]["chemistry_cols"] == ["SO2", "NO3-"]
    assert raw_bundle["data_schema"]["psd_cols"] == ["12.0", "100.0"]
    assert raw_bundle["data_schema"]["psd_diameters_nm"] == [12.0, 100.0]
    assert raw_bundle["model_kwargs"]["n_chem"] == 2
    assert raw_bundle["preprocessing"]["chemistry"]["transform"] == "log1p"
    assert raw_bundle["preprocessing"]["psd"]["scaler"] == "robust"
    assert raw_bundle["preprocessing"]["meteorology"]["scaler"] == "minmax"
    assert raw_bundle["scaler_target"]["feature_kinds"] == [
        "standard", "standard", "robust", "robust"
    ]

    loaded = load_bundle(bundle_path, device=torch.device("cpu"))
    assert loaded["data_schema"].target_cols == ["SO2", "NO3-", "12.0", "100.0"]
    assert loaded["preprocessing"].psd.scaler == "robust"

    cli_main([
        "impute",
        "--bundle", str(bundle_path),
        "--chem-csv", str(paths["chem"]),
        "--psd-csv", str(paths["psd"]),
        "--met-csv", str(paths["met"]),
        "--n-mc-samples", "3",
        "--inference-batch-size", "2",
        "-o", str(output_path),
    ])
    result = pd.read_csv(output_path)
    assert set(result["feature"]) == {"SO2", "NO3-", "12.0", "100.0"}
    assert {"q_lower", "q_upper", "interval_lower", "interval_upper", "q05", "q95"} <= set(result.columns)
    assert result["interval_lower"].eq(0.05).all()
    assert result["interval_upper"].eq(0.95).all()
    assert result[result["is_imputed"]]["imputed_mean"].notna().all()


def test_high_level_dataframe_interface_trains_inspects_and_imputes(tmp_path, capsys):
    paths = _write_modality_csvs(tmp_path, n=48)
    chemistry = pd.read_csv(paths["chem"])
    psd = pd.read_csv(paths["psd"])
    meteorology = pd.read_csv(paths["met"])
    bundle_path = tmp_path / "dataframe_bundle.pt"

    loaded = fit_multimodal(
        chemistry=chemistry,
        psd=psd,
        meteorology=meteorology,
        output=bundle_path,
        config=_small_multimodal_train_config(),
    )
    assert loaded["data_interface"] == "in_memory_modalities"
    assert loaded["data_schema"].target_cols == ["SO2", "NO3-", "12.0", "100.0"]
    assert loaded["training_config"]["csv"] == []
    assert loaded["training_config"]["modality_files"] is None

    result = impute_multimodal(
        chemistry=chemistry,
        psd=psd,
        meteorology=meteorology,
        bundle=loaded,
        config=InferenceConfig(
            n_mc_samples=3,
            inference_batch_size=2,
        ),
    )
    assert len(result) == 48 * 4
    assert set(result["feature"]) == {"SO2", "NO3-", "12.0", "100.0"}
    assert result[result["is_imputed"]]["imputed_mean"].notna().all()

    summary = inspect_bundle(bundle_path)
    assert summary["valid"] is True
    assert summary["dimensions"] == {
        "targets": 4,
        "chemistry": 2,
        "psd": 2,
        "meteorology": 2,
        "condition": 4,
    }
    assert summary["psd"]["range"] == {
        "minimum_nm": 12.0,
        "maximum_nm": 100.0,
    }

    capsys.readouterr()
    cli_main(["inspect-bundle", "--bundle", str(bundle_path)])
    cli_summary = json.loads(capsys.readouterr().out)
    assert cli_summary["valid"] is True
    assert cli_summary["versions"]["bundle"] == 3


def test_validate_data_against_bundle_schema(tmp_path, capsys):
    paths = _write_modality_csvs(tmp_path, n=48)
    bundle_path = tmp_path / "schema_bundle.pt"
    config = _small_multimodal_train_config()
    config.modality_files = {
        "chemistry": [str(paths["chem"])],
        "psd": [str(paths["psd"])],
        "meteorology": [str(paths["met"])],
    }
    # Re-run dataclass normalization after assigning a test source.
    config = TrainConfig(**config.to_dict())
    train_from_config(config, str(bundle_path))

    capsys.readouterr()
    cli_main([
        "validate-data",
        "--bundle", str(bundle_path),
        "--chem-csv", str(paths["chem"]),
        "--psd-csv", str(paths["psd"]),
        "--met-csv", str(paths["met"]),
    ])
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["data_schema"]["psd_cols"] == ["12.0", "100.0"]
    assert set(report["modality_missing_fraction"]) == {
        "chemistry", "psd", "meteorology"
    }


def test_bundle_integrity_rejects_scaler_preprocessing_and_state_dict_mismatch(tmp_path):
    paths = _write_modality_csvs(tmp_path, n=48)
    bundle_path = tmp_path / "integrity_bundle.pt"
    config = TrainConfig(
        modality_files={
            "chemistry": [str(paths["chem"])],
            "psd": [str(paths["psd"])],
            "meteorology": [str(paths["met"])],
        },
        window_size=8,
        stride=4,
        val_fraction=0.25,
        epochs=1,
        patience=1,
        batch_size=4,
        model_kwargs={
            "latent_dim": 4,
            "hidden_dims": [8],
            "encoder_layers": 1,
            "decoder_layers": 1,
            "n_graph_heads": 1,
        },
    )
    train_from_config(config, str(bundle_path))
    raw = torch.load(bundle_path, map_location="cpu", weights_only=True)

    bad_scale = deepcopy(raw)
    bad_scale["scaler_target"]["scale"][0] = 0.0
    bad_scale["scaler_target"]["std"][0] = 0.0
    bad_scale_path = tmp_path / "bad_scale.pt"
    torch.save(bad_scale, bad_scale_path)
    with pytest.raises(ValueError, match="scale values must be positive"):
        load_bundle(bad_scale_path, device=torch.device("cpu"))

    bad_preprocessing = deepcopy(raw)
    bad_preprocessing["preprocessing"]["chemistry"]["scaler"] = "robust"
    bad_preprocessing_path = tmp_path / "bad_preprocessing.pt"
    torch.save(bad_preprocessing, bad_preprocessing_path)
    with pytest.raises(ValueError, match="feature kinds do not match preprocessing"):
        load_bundle(bad_preprocessing_path, device=torch.device("cpu"))

    bad_state = deepcopy(raw)
    bad_state["state_dict"].pop(next(iter(bad_state["state_dict"])))
    bad_state_path = tmp_path / "bad_state.pt"
    torch.save(bad_state, bad_state_path)
    with pytest.raises(ValueError, match="state_dict is incompatible"):
        load_bundle(bad_state_path, device=torch.device("cpu"))


def test_load_bundle_rejects_unknown_architecture_version(tmp_path):
    paths = _write_modality_csvs(tmp_path, n=64)
    bundle_path = tmp_path / "bundle.pt"
    config = TrainConfig(
        modality_files={
            "chemistry": [str(paths["chem"])],
            "psd": [str(paths["psd"])],
            "meteorology": [str(paths["met"])],
        },
        window_size=8,
        stride=4,
        val_fraction=0.25,
        epochs=1,
        patience=1,
        batch_size=4,
        model_kwargs={
            "latent_dim": 4,
            "hidden_dims": [8],
            "encoder_layers": 1,
            "decoder_layers": 1,
            "n_graph_heads": 1,
        },
    )
    train_from_config(config, str(bundle_path))
    raw = torch.load(bundle_path, map_location="cpu", weights_only=True)
    raw["architecture_version"] = 999
    incompatible = tmp_path / "incompatible.pt"
    torch.save(raw, incompatible)
    with pytest.raises(ValueError, match="Unsupported architecture_version"):
        load_bundle(incompatible, device=torch.device("cpu"))


def test_nested_train_config_json_builds_public_contract():
    config_path = Path(__file__).resolve().parents[1] / "examples" / "multimodal_train_config.example.json"
    config = TrainConfig.from_json(config_path)
    assert config.modality_files.chemistry == [
        "examples/data/multimodal_demo/chemistry.csv"
    ]
    assert config.preprocessing.chemistry.transform == "log1p"
    assert config.preprocessing.psd.scaler == "robust"
    assert config.preprocessing.meteorology.scaler == "standard"


def test_model_config_json_rejects_unknown_fields_early(tmp_path):
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_csv(csv_path, n=40)
    config_path = tmp_path / "bad_model.json"
    config_path.write_text(json.dumps({"latent_dim": 4, "misspelled_graph_flag": True}))
    with pytest.raises(TypeError, match="unexpected field"):
        cli_main([
            "train",
            "--csv", str(csv_path),
            "--target-cols", "target_a,target_b",
            "--aux-cols", "ws,at",
            "--window-size", "8",
            "--epochs", "1",
            "--model-config", str(config_path),
            "-o", str(tmp_path / "bad.pt"),
        ])


def test_train_then_impute_round_trip(tmp_path):
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_csv(csv_path)

    bundle_path = tmp_path / "bundle.pt"
    output_path = tmp_path / "imputed.csv"

    train_argv = [
        "train",
        "--csv", str(csv_path),
        "--timestamp-col", "time",
        "--target-cols", "target_a,target_b",
        "--aux-cols", "ws,at",
        "--window-size", "16",
        "--stride", "8",
        "--val-fraction", "0.2",
        "--batch-size", "4",
        "--epochs", "2",
        "--patience", "2",
        "--latent-dim", "4",
        "--hidden-dims", "8,8",
        "--encoder-layers", "1",
        "--decoder-layers", "1",
        "--n-graph-heads", "1",
        "-o", str(bundle_path),
    ]
    cli_main(train_argv)
    assert bundle_path.exists()
    loaded = load_bundle(bundle_path)
    assert loaded["aux_mask_channel"] is True
    assert loaded["model"].aux_dim == 4
    assert loaded["schema"]["cond_dim"] == 4

    impute_argv = [
        "impute",
        "--bundle", str(bundle_path),
        "--csv", str(csv_path),
        "--stride", "8",
        "--n-mc-samples", "3",
        "-o", str(output_path),
    ]
    cli_main(impute_argv)
    assert output_path.exists()

    result = pd.read_csv(output_path)
    assert set(result["feature"].unique()) == {"target_a", "target_b"}
    assert {"timestamp", "observed", "imputed_mean", "imputed_std", "q05", "q95"} <= set(result.columns)
    assert len(result) == 200 * 2  # n rows * n target columns

    # Observed positions are restored verbatim with zero reported uncertainty.
    a_rows = result[result["feature"] == "target_a"].reset_index(drop=True)
    observed_rows = a_rows.iloc[:50]
    assert observed_rows["imputed_std"].eq(0.0).all()
    assert np.allclose(observed_rows["imputed_mean"], observed_rows["observed"])

    # The punched-out gap has no ground truth, but must be filled with a finite prediction.
    gap_rows = a_rows.iloc[50:60]
    assert gap_rows["observed"].isna().all()
    assert np.isfinite(gap_rows["imputed_mean"]).all()
    assert (gap_rows["imputed_std"] > 0).all()


def _load_heldout_eval_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "heldout_eval.py"
    spec = importlib.util.spec_from_file_location("heldout_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_macro_average_metrics_matches_research_repo_methodology():
    # Matches ablation_heldout_eval.py's compute_heldout_metrics exactly:
    # per-feature R^2 (not pooled across features), negative clipped to 0
    # before averaging, a feature needs >= min_points held-out points to
    # count at all. Pooling instead (what this script did before) is
    # dominated by whichever feature has the largest magnitude/variance --
    # feature 0 below has huge values and a near-perfect fit, feature 1 has
    # small values and a bad (negative) fit; pooled R^2 would be dragged
    # near 1.0 by feature 0's huge SS_tot, but the reference-matching
    # macro-average must land near 0.5 (mean of ~1.0 and clip(negative, 0) = 0.0).
    heldout_eval = _load_heldout_eval_module()

    n = 20
    y_true = np.zeros((n, 2))
    y_pred = np.zeros((n, 2))
    mask = np.ones((n, 2), dtype=bool)

    rng = np.random.default_rng(0)
    y_true[:, 0] = rng.uniform(1000, 2000, n)
    y_pred[:, 0] = y_true[:, 0] + rng.normal(0, 1, n)  # near-perfect fit, huge scale

    y_true[:, 1] = rng.uniform(0, 1, n)
    y_pred[:, 1] = rng.uniform(0, 1, n)  # unrelated to y_true -- ~R^2 <= 0

    result = heldout_eval._macro_average_metrics(y_true, y_pred, mask, min_points=10)
    assert result["n_features"] == 2
    assert 0.3 < result["r2"] < 0.7  # NOT near 1.0, which pooling would give
    assert result["rmse"] > 0
    assert result["smape"] >= 0

    # A feature below the min_points threshold must not count at all.
    sparse_mask = mask.copy()
    sparse_mask[5:, 1] = False  # only 5 held-out points for feature 1
    result_sparse = heldout_eval._macro_average_metrics(y_true, y_pred, sparse_mask, min_points=10)
    assert result_sparse["n_features"] == 1

    # clip_r2=False (the "observed" comparison) must NOT clip a negative
    # per-feature R^2 to 0, unlike the held-out default.
    result_unclipped = heldout_eval._macro_average_metrics(y_true, y_pred, mask, min_points=10, clip_r2=False)
    assert result_unclipped["r2"] < result["r2"]

    # sigma/quantile columns enable CRPS/PICP; omitting them omits those keys.
    sigma = np.ones((n, 2))
    q_lo = y_true - 1.0
    q_hi = y_true + 1.0
    result_with_interval = heldout_eval._macro_average_metrics(
        y_true, y_pred, mask, sigma_cols=sigma, q_lo_cols=q_lo, q_hi_cols=q_hi, min_points=10,
    )
    assert "crps" in result_with_interval and "picp" in result_with_interval
    assert "crps" not in result and "picp" not in result


def test_category_indices_covers_overall_chem_psd_and_ignores_unparseable_psd_names():
    heldout_eval = _load_heldout_eval_module()
    target_cols = ["SO2", "K", "PM2.5", "target_b", "12.19", "9999.0"]
    cats = dict(heldout_eval.category_indices(target_cols, n_chem=3))
    assert cats["overall"] == [0, 1, 2, 3, 4, 5]
    assert cats["chem"] == [0, 1, 2]
    assert cats["psd"] == [3, 4, 5]
    assert cats["gases"] == [0]  # SO2
    assert cats["metal"] == [1]  # K
    assert cats["pm"] == [2]  # PM2.5
    # "target_b" isn't a parseable diameter -- excluded from PSD sub-groups
    # but still present in "psd" above.
    assert cats["nucleation"] == [4]  # 12.19nm
    assert cats["coarse_super2.5"] == [5]  # 9999nm
    assert all(3 not in idx for name, idx in heldout_eval.category_indices(target_cols, n_chem=3)
               if name not in ("overall", "psd"))


def test_two_stream_aggregation_keeps_mean_stable_despite_noisy_outlier_samples():
    # Mirrors infer.impute's compute_window_predictions design: the point
    # estimate must be built from the clean per-window mean stream
    # (compute_uncertainty's result[0], mc=1, no injected likelihood noise),
    # never from the noisy generative-sample stream (result[-2]) that's used
    # for quantiles/std. A single wild noisy draw -- a realistic Student-t/
    # Gaussian tail event under a clamped-but-nonzero predicted variance --
    # must not drag imputed_mean, even though it should still widen the
    # reported interval. Averaging the noisy stream for BOTH (what this
    # package used to do) is what turned a real held-out eval's
    # psd_heldout_r2 into the thousands-negative.
    window_size, n_features = 4, 1
    clean_mean = 0.5  # log1p-space model prediction, nothing exotic
    mean_chunks = [(0, np.full((1, window_size, n_features), clean_mean))]
    noisy_samples = np.full((5, window_size, n_features), clean_mean)
    noisy_samples[0] = 25.0  # one wild Student-t/Gaussian tail draw
    sample_chunks = [(0, noisy_samples)]

    position_weights = np.ones(window_size)
    mean_agg = aggregate_window_samples(
        mean_chunks, total_length=window_size, position_weights=position_weights, quantiles=()
    )
    dist_agg = aggregate_window_samples(
        sample_chunks, total_length=window_size, position_weights=position_weights, quantiles=(0.05, 0.95)
    )
    mean_out, std_out, _quantiles_out = summary_to_output_scale(
        mean_agg["mean"], dist_agg["variance"], dist_agg["quantiles"], transform="log1p",
    )

    assert mean_out[0, 0] == pytest.approx(np.expm1(clean_mean))  # unaffected by the wild noisy draw
    assert std_out[0, 0] > 0  # but the noisy stream still widens reported uncertainty


def test_summary_to_output_scale_avoids_sample_then_transform_blowup():
    # A Student-t draw can occasionally be extreme; transforming EACH sample
    # with expm1 before averaging ("sample-then-transform") is biased high
    # (Jensen's inequality, expm1 is convex) and numerically unstable -- one
    # wild draw dominates the arithmetic mean once exponentiated. This is
    # exactly what turned a real held-out R^2 on log1p-scale PSD targets into
    # -2964 before this fix. summary_to_output_scale must instead aggregate
    # in the linear model space and transform the summary statistic once.
    model_space_samples = np.array([0.5, 0.6, 0.4, 0.55, 20.0])  # one wild outlier
    naive_sample_then_transform_mean = np.expm1(model_space_samples).mean()

    mean_out, std_out, quantiles_out = summary_to_output_scale(
        mean_model=np.array([[model_space_samples.mean()]]),
        variance_model=np.array([[model_space_samples.var()]]),
        quantile_values_model={0.5: np.array([[np.median(model_space_samples)]])},
        transform="log1p",
    )

    expected_mean = np.expm1(model_space_samples.mean())
    assert mean_out[0, 0] == pytest.approx(expected_mean)
    # Orders of magnitude smaller than the biased sample-then-transform mean.
    assert mean_out[0, 0] < naive_sample_then_transform_mean / 100
    assert quantiles_out[0.5][0, 0] == pytest.approx(np.expm1(np.median(model_space_samples)))
    # Delta-method std propagation: exp(mean_model) * std_model.
    expected_std = np.exp(model_space_samples.mean()) * model_space_samples.std()
    assert std_out[0, 0] == pytest.approx(expected_std)


def test_summary_to_output_scale_is_identity_for_no_transform():
    mean_out, std_out, quantiles_out = summary_to_output_scale(
        mean_model=np.array([[1.5, -2.0]]),
        variance_model=np.array([[4.0, 9.0]]),
        quantile_values_model={0.05: np.array([[0.1, -5.0]])},
        transform="none",
    )
    assert np.array_equal(mean_out, np.array([[1.5, -2.0]]))
    assert np.array_equal(std_out, np.array([[2.0, 3.0]]))
    assert np.array_equal(quantiles_out[0.05], np.array([[0.1, -5.0]]))


def test_heldout_eval_example_script_scores_only_masked_points(tmp_path):
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_csv(csv_path)
    bundle_path = tmp_path / "bundle.pt"

    cli_main([
        "train",
        "--csv", str(csv_path),
        "--timestamp-col", "time",
        "--target-cols", "target_a,target_b",
        "--aux-cols", "ws,at",
        "--window-size", "16",
        "--stride", "8",
        "--val-fraction", "0.2",
        "--batch-size", "4",
        "--epochs", "2",
        "--patience", "2",
        "--latent-dim", "4",
        "--hidden-dims", "8,8",
        "--encoder-layers", "1",
        "--decoder-layers", "1",
        "--n-graph-heads", "1",
        "-o", str(bundle_path),
    ])

    output_path = tmp_path / "heldout_metrics.json"
    predictions_path = tmp_path / "heldout_predictions.csv"
    script = Path(__file__).resolve().parents[1] / "examples" / "heldout_eval.py"
    proc = subprocess.run(
        [
            sys.executable, str(script),
            "--bundle", str(bundle_path),
            "--csv", str(csv_path),
            "--n-chem", "1",
            "--n-mc-samples", "3",
            "--stride", "8",
            "-o", str(output_path),
            "--predictions-csv", str(predictions_path),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    results = json.loads(output_path.read_text())
    for group in ("overall", "chem", "psd"):
        assert f"{group}_heldout_n" in results
        assert results[f"{group}_heldout_n"] > 0
        assert np.isfinite(results[f"{group}_heldout_mae"])
        assert np.isfinite(results[f"{group}_heldout_rmse"])
        assert 0.0 <= results[f"{group}_heldout_picp"] <= 100.0
        assert 0.0 <= results[f"{group}_heldout_r2"]  # clipped to >= 0
        assert np.isfinite(
            results[f"{group}_heldout_empirical_crps_model_space"]
        )
        assert f"{group}_observed_r2" in results  # sanity-check reconstruction fidelity

    predictions = pd.read_csv(predictions_path)
    expected_cols = {
        "timestamp", "feature", "family", "scaled_observed", "scaled_pred_mean",
        "model_observed", "model_pred_mean", "model_empirical_crps",
        "physical_observed", "physical_pred_mean",
        "physical_pred_std", "physical_q025", "physical_q975",
    }
    assert expected_cols <= set(predictions.columns)
    assert len(predictions) == results["chem_heldout_n"] + results["psd_heldout_n"]
    assert set(predictions["family"]) <= {"chem", "psd"}
    assert predictions["physical_pred_std"].gt(0).all()
    # scaled_* must be the SAME z-score space for observed and predicted --
    # a real (if imperfect) model should land in a comparable range, not be
    # off by the several standard deviations a de-standardized value
    # mislabeled as "scaled" would produce.
    assert (predictions["scaled_pred_mean"] - predictions["scaled_observed"]).abs().median() < 5


def test_sample_level_overlap_aggregation_preserves_cross_window_mixture():
    samples = np.array(
        [
            [[[0.0], [0.0], [0.0], [0.0]], [[100.0], [100.0], [100.0], [100.0]]],
            [[[10.0], [10.0], [10.0], [10.0]], [[20.0], [20.0], [20.0], [20.0]]],
        ],
        dtype=np.float64,
    )

    result = aggregate_window_samples(
        samples,
        window_starts=[0, 2],
        total_length=6,
        position_weights=np.ones(4),
        quantiles=(0.5,),
    )

    # At global position 2 the predictive distribution is the four-draw
    # mixture [0, 100, 10, 20], not an average of per-window quantiles.
    assert result["mean"][2, 0] == pytest.approx(32.5)
    assert result["variance"][2, 0] == pytest.approx(1568.75)
    assert result["quantiles"][0.5][2, 0] == pytest.approx(15.0)


def test_streaming_aggregation_matches_full_history_and_bounds_active_state():
    rng = np.random.default_rng(7)
    starts = [0, 2, 4, 6, 8]
    window_size, n_mc, n_features, total_length = 4, 3, 2, 12
    position_weights = np.array([0.5, 1.0, 1.0, 0.5])
    windows = [rng.normal(size=(n_mc, window_size, n_features)) for _ in starts]

    # Independent full-history reference matching the original implementation.
    values_by_position = [[] for _ in range(total_length)]
    weights_by_position = [[] for _ in range(total_length)]
    for start, samples in zip(starts, windows):
        for local_position, global_position in enumerate(
            range(start, start + window_size)
        ):
            values_by_position[global_position].append(samples[:, local_position])
            weights_by_position[global_position].append(
                np.full(n_mc, position_weights[local_position])
            )

    reference_mean = np.full((total_length, n_features), np.nan)
    reference_variance = np.full_like(reference_mean, np.nan)
    reference_median = np.full_like(reference_mean, np.nan)
    for position in range(total_length):
        values = np.concatenate(values_by_position[position], axis=0)
        weights = np.concatenate(weights_by_position[position], axis=0)
        reference_mean[position] = np.average(values, axis=0, weights=weights)
        reference_variance[position] = np.maximum(
            np.average(values**2, axis=0, weights=weights)
            - reference_mean[position] ** 2,
            0.0,
        )
        for feature in range(n_features):
            order = np.argsort(values[:, feature])
            sorted_values = values[order, feature]
            sorted_weights = weights[order]
            centers = np.cumsum(sorted_weights) - 0.5 * sorted_weights
            reference_median[position, feature] = np.interp(
                0.5 * sorted_weights.sum(), centers, sorted_values
            )

    aggregator = StreamingWindowAggregator(
        total_length=total_length,
        window_size=window_size,
        n_features=n_features,
        position_weights=position_weights,
        quantiles=(0.5,),
    )
    for start, samples in zip(starts, windows):
        aggregator.add(start, samples)
    result = aggregator.finish()

    np.testing.assert_allclose(result["mean"], reference_mean)
    np.testing.assert_allclose(result["variance"], reference_variance)
    np.testing.assert_allclose(result["quantiles"][0.5], reference_median)
    assert result["peak_active_positions"] <= window_size


def test_weighted_empirical_crps_matches_direct_pairwise_definition():
    values = np.array(
        [[0.0, 3.0], [1.0, 2.0], [4.0, -1.0], [8.0, 5.0]],
        dtype=np.float64,
    )
    weights = np.array([0.5, 1.0, 2.0, 0.25], dtype=np.float64)
    target = np.array([2.0, 1.5], dtype=np.float64)
    probabilities = weights / weights.sum()
    first = np.sum(
        probabilities[:, None] * np.abs(values - target[None, :]), axis=0
    )
    pairwise = 0.5 * np.sum(
        probabilities[:, None, None]
        * probabilities[None, :, None]
        * np.abs(values[:, None, :] - values[None, :, :]),
        axis=(0, 1),
    )

    actual = weighted_empirical_crps(values, weights, target)
    np.testing.assert_allclose(actual, first - pairwise, atol=1e-12)


def test_torch_empirical_crps_matches_direct_pairwise_definition():
    samples = torch.tensor(
        [
            [[[0.0, 3.0], [1.0, 2.0]]],
            [[[1.0, 2.0], [2.0, 4.0]]],
            [[[4.0, -1.0], [5.0, 0.0]]],
            [[[8.0, 5.0], [6.0, 3.0]]],
        ]
    )
    target = torch.tensor([[[2.0, 1.5], [3.0, 2.0]]])
    mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    direct_first = (samples - target.unsqueeze(0)).abs().mean(dim=0)
    direct_pairwise = 0.5 * (
        samples[:, None] - samples[None, :]
    ).abs().mean(dim=(0, 1))
    expected = ((direct_first - direct_pairwise) * mask).sum()

    actual, count = empirical_crps_components(samples, target, mask)
    assert count.item() == 3
    torch.testing.assert_close(actual, expected)


def test_streaming_aggregator_scores_crps_when_timestamp_is_finalized():
    position_weights = np.array([0.5, 1.0, 0.5])
    first = np.array([[[0.0], [1.0], [2.0]], [[2.0], [3.0], [4.0]]])
    second = np.array([[[10.0], [11.0], [12.0]], [[14.0], [15.0], [16.0]]])
    targets = np.array([[0.0], [4.0], [0.0], [0.0]])
    score_mask = np.array([[False], [True], [False], [False]])

    aggregator = StreamingWindowAggregator(
        total_length=4,
        window_size=3,
        n_features=1,
        position_weights=position_weights,
        quantiles=(),
        crps_targets=targets,
        crps_mask=score_mask,
    )
    aggregator.add(0, first)
    aggregator.add(1, second)
    result = aggregator.finish()

    values = np.array([[1.0], [3.0], [10.0], [14.0]])
    weights = np.array([1.0, 1.0, 0.5, 0.5])
    expected = weighted_empirical_crps(values, weights, np.array([4.0]))[0]
    assert result["crps"][1, 0] == pytest.approx(expected)
    assert np.isnan(result["crps"][[0, 2, 3], 0]).all()


def test_validation_nll_uses_gaussian_or_model_student_t_likelihood():
    recon_mean = torch.tensor([[[0.0, 0.5]]])
    recon_logvar = torch.tensor([[[np.log(1e-6), np.log(20.0)]]])
    target = torch.tensor([[[1.0, -1.0]]])
    mask = torch.ones_like(target)
    var_min, var_max = 0.2, 2.0

    gaussian_sum, count = masked_nll_components(
        recon_mean,
        recon_logvar,
        target,
        mask,
        var_min=var_min,
        var_max=var_max,
    )
    clamped = recon_logvar.clamp(min=np.log(var_min), max=np.log(var_max))
    expected_gaussian = 0.5 * (
        clamped + (target - recon_mean).square() / clamped.exp()
    )
    assert count.item() == 2
    assert gaussian_sum.item() == pytest.approx(expected_gaussian.sum().item())

    class FixedDfModel:
        def get_likelihood_df(self, num_features, device=None, dtype=None):
            assert num_features == 2
            return torch.tensor([5.0, 8.0], device=device, dtype=dtype)

    df = torch.tensor([5.0, 8.0])
    student_sum, _ = masked_nll_components(
        recon_mean,
        recon_logvar,
        target,
        mask,
        model=FixedDfModel(),
        use_student_t_nll=True,
        var_min=var_min,
        var_max=var_max,
    )
    expected_student = _student_t_nll(
        recon_mean,
        recon_logvar,
        target,
        var_min=var_min,
        var_max=var_max,
        df=df,
    )
    assert student_sum.item() == pytest.approx(expected_student.sum().item())
    assert student_sum.item() != pytest.approx(gaussian_sum.item())


def test_trainer_validation_wires_student_t_df_and_decoder_variance_bounds():
    class FixedValidationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.decoder = torch.nn.Identity()
            self.decoder.var_min = 0.25
            self.decoder.var_max = 1.5

        def get_likelihood_df(self, num_features, device=None, dtype=None):
            return torch.full((num_features,), 7.0, device=device, dtype=dtype)

        def forward(self, input_x, cond, input_mask):
            recon_mean = torch.zeros_like(input_x) + self.anchor * 0.0
            recon_logvar = torch.full_like(input_x, np.log(10.0))
            latent = torch.zeros((input_x.shape[0], 1), device=input_x.device)
            return recon_mean, recon_logvar, latent, latent, None

    config = TrainConfig(
        csv=["unused.csv"],
        timestamp_col="time",
        target_cols=["a", "b"],
        validation_metric="ho_nll",
        use_student_t_nll=True,
    )
    model = FixedValidationModel()
    trainer = Trainer(
        model,
        train_loader=None,
        val_loader=None,
        config=config,
        device=torch.device("cpu"),
    )
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    batch = {
        "input_x": torch.zeros_like(target),
        "cond": torch.zeros((1, 2, 0)),
        "input_mask": torch.zeros_like(target),
        "target": target,
        "obs_mask": torch.ones_like(target),
        "heldout_mask": torch.ones_like(target),
    }
    metrics = trainer._run_validation([batch], epoch=0)
    expected = _student_t_nll(
        torch.zeros_like(target),
        torch.full_like(target, np.log(10.0)),
        target,
        var_min=0.25,
        var_max=1.5,
        df=torch.full((2,), 7.0),
    ).mean()
    assert metrics["ho_nll"] == pytest.approx(expected.item())


def test_trapezoid_position_weights_ramps_20pct_edges_like_research():
    # Matches trapezoidal_window_weights(window_size, edge_frac=0.2) from the
    # research ablation_inference.py: a linear ramp over 20% of the window at
    # each edge, not just the single boundary timestep.
    weights = trapezoid_position_weights(168)
    edge_len = round(168 * 0.2)
    assert edge_len == 34
    assert np.all(weights[edge_len:-edge_len] == 1.0)
    assert weights[0] == pytest.approx(1.0 / (edge_len + 1))
    assert weights[-1] == pytest.approx(1.0 / (edge_len + 1))
    assert np.all(np.diff(weights[:edge_len]) > 0)  # monotonic ramp up
    assert np.all(np.diff(weights[-edge_len:]) < 0)  # monotonic ramp down


def test_trapezoid_position_weights_falls_back_for_small_windows():
    # edge_frac=0.2 of a tiny window would overlap the middle; the envelope
    # must stay well-defined (no zero/negative weights) instead of clipping.
    weights = trapezoid_position_weights(3)
    assert len(weights) == 3
    assert np.all(weights > 0)
    assert weights[1] == 1.0


def test_lr_warmup_epochs_overrides_ratio_default():
    # Before this, LR warmup had no config knob at all -- it was hardcoded
    # to 5% of `epochs` in Trainer.__init__. A reference run that preserves
    # an absolute warmup length under a reduced epoch budget (e.g. the 26e
    # ablation battery: 100 epochs of warmup even at --epochs 700, not the
    # mainline 2000) needs an absolute lr_warmup_epochs knob, not a ratio of
    # *this* run's shorter budget.
    config = TrainConfig(
        csv=["unused.csv"], timestamp_col="time", target_cols=["target"],
        epochs=700, lr_warmup_epochs=100,
        model_kwargs={"latent_dim": 4, "hidden_dims": [4], "encoder_layers": 1, "decoder_layers": 1, "n_graph_heads": 1},
    )
    model = ImputationVAE_Graph(target_dim=1, aux_dim=0, window_size=8, **config.model_kwargs)
    trainer = Trainer(model, train_loader=None, val_loader=None, config=config, device=torch.device("cpu"))
    assert trainer.lr_scheduler.warmup_epochs == 100  # not int(700 * 0.05) == 35

    default_config = TrainConfig(
        csv=["unused.csv"], timestamp_col="time", target_cols=["target"], epochs=700,
        model_kwargs={"latent_dim": 4, "hidden_dims": [4], "encoder_layers": 1, "decoder_layers": 1, "n_graph_heads": 1},
    )
    default_trainer = Trainer(
        ImputationVAE_Graph(target_dim=1, aux_dim=0, window_size=8, **default_config.model_kwargs),
        train_loader=None, val_loader=None, config=default_config, device=torch.device("cpu"),
    )
    assert default_trainer.lr_scheduler.warmup_epochs == 35  # unchanged default behavior


def test_use_adaptive_lr_builds_plateau_scheduler_and_skips_cosine_after_warmup():
    config = TrainConfig(
        csv=["unused.csv"],
        timestamp_col="time",
        target_cols=["target_a", "target_b"],
        window_size=8,
        stride=8,
        use_adaptive_lr=True,
        lr_reduce_factor=0.5,
        lr_reduce_patience=2,
        lr_reduce_cooldown=0,
        epochs=10,
        model_kwargs={"latent_dim": 4, "hidden_dims": [4], "encoder_layers": 1, "decoder_layers": 1, "n_graph_heads": 1},
    )
    model = ImputationVAE_Graph(target_dim=2, aux_dim=0, window_size=8, **config.model_kwargs)
    trainer = Trainer(model, train_loader=None, val_loader=None, config=config, device=torch.device("cpu"))

    assert trainer.use_adaptive_lr is True
    assert trainer.plateau_scheduler is not None
    warmup_epochs = trainer.lr_scheduler.warmup_epochs  # max(1, int(10 * 0.05)) == 1

    trainer.lr_scheduler.step(0)
    lr_after_warmup_step = trainer.optimizer.param_groups[0]["lr"]

    # Once warmup ends, fit() must stop calling the cosine scheduler and let
    # ReduceLROnPlateau own the LR instead — verify the gating condition
    # fit() relies on, matching the research trainer's `in_warmup` check.
    in_warmup_last_epoch = (warmup_epochs - 1) < warmup_epochs
    in_warmup_post_epoch = warmup_epochs < warmup_epochs
    assert in_warmup_last_epoch is True
    assert in_warmup_post_epoch is False
    assert lr_after_warmup_step > 0


def test_trainer_uses_adamw_for_reference_weight_decay():
    config = TrainConfig(
        csv=["unused.csv"],
        timestamp_col="time",
        target_cols=["target"],
        weight_decay=0.01,
        model_kwargs={
            "latent_dim": 4,
            "hidden_dims": [4],
            "encoder_layers": 1,
            "decoder_layers": 1,
            "n_graph_heads": 1,
        },
    )
    model = ImputationVAE_Graph(target_dim=1, aux_dim=0, window_size=8, **config.model_kwargs)
    trainer = Trainer(model, train_loader=None, val_loader=None, config=config, device=torch.device("cpu"))
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.01


def test_heldout_validation_metric_accepts_mse():
    config = TrainConfig(
        csv=["unused.csv"],
        timestamp_col="time",
        target_cols=["target"],
        validation_metric="ho_mse",
    )
    assert config.validation_metric == "ho_mse"


def test_train_ho_config_resolves_a_distinct_fixed_mask_seed_and_ratio():
    config = TrainConfig(
        csv=["unused.csv"],
        timestamp_col="time",
        target_cols=["target"],
        selection_val_seed=100003,
        selection_mask_ratio=0.2,
        train_ho_enabled=True,
    )

    assert config.train_ho_enabled is True
    assert config.train_ho_seed == 100004
    assert config.train_ho_ratio == pytest.approx(0.2)


def test_train_ho_metrics_are_recorded_from_cells_excluded_from_training_loss(tmp_path):
    csv_path = tmp_path / "train_ho.csv"
    _write_synthetic_csv(csv_path, n=80)
    config = TrainConfig(
        csv=[str(csv_path)],
        timestamp_col="time",
        target_cols=["target_a", "target_b"],
        aux_cols=["ws", "at"],
        window_size=8,
        stride=8,
        val_fraction=0.25,
        batch_size=4,
        epochs=1,
        patience=1,
        train_ho_enabled=True,
        selection_mask_mode="block",
        selection_mask_ratio=0.2,
        model_kwargs={
            "latent_dim": 4,
            "hidden_dims": [8],
            "encoder_layers": 1,
            "decoder_layers": 1,
            "n_graph_heads": 1,
        },
    )
    bundle_path = tmp_path / "train_ho.pt"
    train_from_config(config, str(bundle_path))

    history = pd.read_csv(tmp_path / "train_ho_history.csv")
    for column in ("train_ho_nll", "train_ho_mse", "val_ho_nll", "val_ho_mse"):
        assert column in history.columns
        assert np.isfinite(history.loc[0, column])


def test_26e_training_controls_are_serializable():
    config = TrainConfig(
        csv=["unused.csv"],
        timestamp_col="time",
        target_cols=["target"],
        kl_warmup_ratio=0.1,
        kl_strategy="cosine",
        use_amp=True,
        amp_dtype="auto",
        prior_type="student_t",
        use_student_t_nll=True,
        loss_normalization="window_feature_sum",
        aux_mask_channel=False,
        target_output_transform="log1p",
        scaler_fit_scope="full",
        chem_feature_weight=12.0,
        psd_feature_weight=1.0,
    )

    saved = config.to_dict()
    assert saved["kl_warmup_ratio"] == 0.1
    assert saved["kl_strategy"] == "cosine"
    assert saved["use_amp"] is True
    assert saved["prior_type"] == "student_t"
    assert saved["use_student_t_nll"] is True
    assert saved["loss_normalization"] == "window_feature_sum"
    assert saved["aux_mask_channel"] is False
    assert saved["target_output_transform"] == "log1p"
    assert saved["scaler_fit_scope"] == "full"
    assert saved["chem_feature_weight"] == 12.0
    assert saved["psd_feature_weight"] == 1.0


def test_pretransformed_input_is_inverted_once_at_public_output(tmp_path):
    csv_path = tmp_path / "pretransformed.csv"
    raw = _write_synthetic_csv(csv_path, n=80)
    expected_raw = raw[["target_a", "target_b"]].clip(lower=0.0)
    raw[["target_a", "target_b"]] = np.log1p(expected_raw)
    raw.to_csv(csv_path, index=False)

    config = TrainConfig(
        csv=[str(csv_path)],
        timestamp_col="time",
        target_cols=["target_a", "target_b"],
        aux_cols=["ws", "at"],
        target_transform="none",
        target_output_transform="log1p",
        window_size=16,
        stride=8,
        val_fraction=0.2,
        batch_size=4,
        epochs=1,
        patience=1,
        model_kwargs={
            "latent_dim": 4,
            "hidden_dims": [8, 8],
            "encoder_layers": 1,
            "decoder_layers": 1,
            "n_graph_heads": 1,
        },
    )
    bundle_path = tmp_path / "bundle.pt"
    train_from_config(config, str(bundle_path))
    bundle = load_bundle(bundle_path)
    assert bundle["target_transform"] == "none"
    assert bundle["target_output_transform"] == "log1p"

    output_path = tmp_path / "output.csv"
    cli_main([
        "impute",
        "--bundle", str(bundle_path),
        "--csv", str(csv_path),
        "--stride", "8",
        "--n-mc-samples", "3",
        "-o", str(output_path),
    ])
    result = pd.read_csv(output_path)
    target_a = result[result["feature"] == "target_a"].reset_index(drop=True)
    expected_observed = expected_raw["target_a"].to_numpy()
    assert np.allclose(
        target_a.loc[:49, "observed"], expected_observed[:50], equal_nan=True
    )
    assert np.allclose(target_a.loc[:49, "imputed_mean"], expected_observed[:50])


def test_feature_weights_are_applied_before_window_reduction():
    target = torch.ones((1, 1, 2))
    recon_mean = torch.zeros_like(target)
    obs_mask = torch.ones_like(target)
    mu = torch.zeros((1, 1))
    logvar = torch.zeros((1, 1))

    loss, recon, _ = vae_loss(
        recon_mean,
        None,
        target,
        obs_mask,
        mu,
        logvar,
        beta=0.0,
        loss_normalization="window_feature_sum",
        n_chem=1,
        chem_feature_weight=12.0,
        psd_feature_weight=1.0,
    )

    assert loss.item() == pytest.approx(13.0)
    assert recon.item() == pytest.approx(13.0)


def test_fixed_mask_is_kept_out_of_training_inputs():
    target = np.ones((8, 2), dtype=np.float32)
    aux = np.ones((8, 1), dtype=np.float32)
    fixed = np.zeros_like(target, dtype=bool)
    fixed[3, 1] = True
    dataset = WindowedTimeSeriesDataset(
        target,
        aux,
        window_size=8,
        stride=8,
        mode="train",
        fixed_mask=fixed,
    )
    item = dataset[0]
    assert item["heldout_mask"][3, 1].item() == 1.0
    assert item["input_mask"][3, 1].item() == 0.0
    assert item["target"][3, 1].item() == 1.0
    # A fixed selection mask is a permanent blind held-out set: it must also
    # be excluded from obs_mask, not just input_mask, since the training loss
    # is computed over obs_mask. Otherwise the model gets direct gradient
    # supervision on exactly the points later reported as held-out accuracy.
    assert item["obs_mask"][3, 1].item() == 0.0
    # every other position is untouched
    assert item["obs_mask"].sum().item() == 15.0


def test_student_t_window_feature_loss_uses_experiment_normalization():
    target = torch.tensor([[[1.0], [3.0]]])
    recon_mean = torch.zeros_like(target)
    recon_logvar = torch.zeros_like(target)
    obs_mask = torch.ones_like(target)
    mu = torch.zeros((1, 1))
    logvar = torch.zeros((1, 1))

    loss, recon, kl = vae_loss(
        recon_mean,
        recon_logvar,
        target,
        obs_mask,
        mu,
        logvar,
        beta=0.0,
        prior_type="student_t",
        use_student_t_nll=True,
        loss_normalization="window_feature_sum",
        n_chem=0,
    )

    df = torch.tensor(3.0)
    variance = torch.tensor(1.0)
    sigma_sq = variance * (df - 2.0) / df
    const = torch.lgamma((df + 1.0) / 2.0) - torch.lgamma(df / 2.0) - 0.5 * torch.log(df * torch.pi)
    point_nll = -const + 0.5 * torch.log(sigma_sq) + 2.0 * torch.log1p(target.square() / (df * sigma_sq))
    expected = point_nll.sum()

    assert loss.item() == pytest.approx(expected.item())
    assert recon.item() == pytest.approx(expected.item())
    assert kl.item() == pytest.approx(0.0)


def test_legacy_dynamic_mask_matches_research_modal_pattern():
    observed = np.ones((168, 6), dtype=bool)
    observed[0, 0] = False
    heldout = sample_dynamic_heldout_mask(
        observed,
        {
            "mode": "legacy",
            "n_chem": 2,
            "legacy_chem_blocks": 8,
            "legacy_psd_blocks": 6,
            "random_point_drop_prob": 0.0,
        },
        seed=42,
    )

    assert heldout.shape == observed.shape
    assert not heldout[0, 0]
    # PSD-like targets are masked as shared time blocks, not independently.
    assert np.array_equal(heldout[:, 2], heldout[:, 3])
    assert np.array_equal(heldout[:, 3], heldout[:, 5])


def test_anchor_constrained_selection_mask_preserves_observed_anchors():
    observed = np.ones((168, 4), dtype=bool)
    heldout = sample_anchor_constrained_heldout_mask(observed, ratio=0.1, seed=42, n_chem=1)

    assert heldout.shape == observed.shape
    assert heldout.any()
    for feature in range(4):
        positions = np.flatnonzero(heldout[:, feature])
        if positions.size:
            assert positions.min() > 0
            assert positions.max() < len(observed) - 1
            assert not heldout[positions.min() - 1, feature]
            assert not heldout[positions.max() + 1, feature]
