import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import t as scipy_student_t

from graph_temporal_vae.censoring import (
    STATE_CENSORED,
    STATE_MISSING,
    STATE_OBSERVED,
    CensoringConfig,
    apply_input_fill,
    build_state_matrix,
    censoring_report,
    high_censoring_columns,
    model_space_thresholds,
)
from graph_temporal_vae.config import TrainConfig
from graph_temporal_vae.contracts import DataSchema, ModalityPreprocessing, PreprocessingConfig
from graph_temporal_vae.data import WindowedTimeSeriesDataset
from graph_temporal_vae.infer import impute, load_bundle, truncate_below_limit
from graph_temporal_vae.preprocessing import fit_target_scaler, transform_targets
from graph_temporal_vae.train import (
    log_ndtr,
    student_t_log_cdf,
    train_from_config,
    vae_loss,
)


def _schema():
    return DataSchema(
        timestamp_col="time",
        chemistry_cols=["a", "b"],
        psd_cols=["10.0", "20.0"],
        psd_diameters_nm=[10.0, 20.0],
        meteorology_cols=["m"],
    )


def _preprocessing():
    return PreprocessingConfig(
        chemistry=ModalityPreprocessing(transform="log1p", scaler="standard", output_transform="log1p"),
        psd=ModalityPreprocessing(transform="log1p", scaler="standard", output_transform="log1p"),
        meteorology=ModalityPreprocessing(transform="none", scaler="standard"),
    )


# --- config contract -------------------------------------------------------

def test_config_rejects_nonpositive_threshold():
    with pytest.raises(ValueError, match="positive finite"):
        CensoringConfig(enabled=True, thresholds={"a": 0.0})
    with pytest.raises(ValueError, match="positive finite"):
        CensoringConfig(enabled=True, thresholds={"a": -1.0})


def test_config_drops_none_thresholds_and_reports_active():
    config = CensoringConfig(enabled=True, thresholds={"a": 0.5, "b": None})
    assert config.thresholds == {"a": 0.5}
    assert config.active
    # Enabled but with nothing to act on is inert, not an error.
    assert not CensoringConfig(enabled=True, thresholds={}).active
    assert not CensoringConfig(enabled=False, thresholds={"a": 0.5}).active


def test_unknown_threshold_column_is_rejected():
    config = CensoringConfig(enabled=True, thresholds={"nope": 0.5})
    with pytest.raises(ValueError, match="outside the target schema"):
        build_state_matrix(np.zeros((3, 4)), _schema(), config)


# --- state classification --------------------------------------------------

def test_three_states_are_distinguished():
    schema = _schema()
    config = CensoringConfig(enabled=True, thresholds={"a": 0.2})
    values = np.array([
        [0.0, 0.0, 1.0, 1.0],      # a censored; b zero but has no threshold
        [np.nan, 1.0, 1.0, 1.0],   # a missing
        [5.0, 1.0, 1.0, 1.0],      # a observed
    ])
    state = build_state_matrix(values, schema, config)
    assert state[0, 0] == STATE_CENSORED
    assert state[0, 1] == STATE_OBSERVED  # no threshold -> a zero stays a value
    assert state[1, 0] == STATE_MISSING
    assert state[2, 0] == STATE_OBSERVED


def test_at_or_below_threshold_detects_reported_subliminal_values():
    schema = _schema()
    values = np.array([[0.1, 1.0, 1.0, 1.0]])
    zero_only = build_state_matrix(values, schema, CensoringConfig(
        enabled=True, thresholds={"a": 0.2}, detect="zero"))
    assert zero_only[0, 0] == STATE_OBSERVED
    inclusive = build_state_matrix(values, schema, CensoringConfig(
        enabled=True, thresholds={"a": 0.2}, detect="at_or_below_threshold"))
    assert inclusive[0, 0] == STATE_CENSORED


def test_loss_ignore_demotes_censored_to_missing():
    schema = _schema()
    config = CensoringConfig(enabled=True, thresholds={"a": 0.2}, loss="ignore")
    state = build_state_matrix(np.array([[0.0, 1.0, 1.0, 1.0]]), schema, config)
    assert state[0, 0] == STATE_MISSING


def test_disabled_censoring_leaves_zeros_as_observations():
    schema = _schema()
    state = build_state_matrix(np.array([[0.0, 0.0, 0.0, 0.0]]), schema, CensoringConfig())
    assert (state == STATE_OBSERVED).all()


# --- input fill and threshold mapping --------------------------------------

def test_input_fill_substitutes_half_threshold():
    schema = _schema()
    config = CensoringConfig(enabled=True, thresholds={"a": 0.4})
    values = np.array([[0.0, 1.0, 1.0, 1.0]])
    state = build_state_matrix(values, schema, config)
    filled = apply_input_fill(values, state, schema, config)
    assert filled[0, 0] == pytest.approx(0.2)
    assert filled[0, 1] == 1.0  # untouched
    # The original array must not be mutated: it is still the raw record.
    assert values[0, 0] == 0.0


def test_model_space_threshold_matches_manual_transform_chain():
    schema, preprocessing = _schema(), _preprocessing()
    config = CensoringConfig(enabled=True, thresholds={"a": 0.4})
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.5, 5.0, size=(50, 4))
    scaler = fit_target_scaler(transform_targets(raw, schema, preprocessing), schema, preprocessing)

    scaled = model_space_thresholds(schema, preprocessing, scaler, config)
    expected = (np.log1p(0.4) - scaler.center_[0]) / scaler.scale_[0]
    assert scaled[0] == pytest.approx(expected)
    # Columns without a limit are never used and stay explicitly undefined.
    assert np.isnan(scaled[1:]).all()


# --- log_ndtr --------------------------------------------------------------

def test_log_ndtr_matches_torch_reference_including_far_tail():
    values = torch.tensor(
        [-60.0, -20.0, -10.0, -5.001, -5.0, -4.999, -1.0, 0.0, 3.0], dtype=torch.float64
    )
    reference = torch.special.log_ndtr(values)
    assert torch.allclose(log_ndtr(values), reference, rtol=1e-4, atol=1e-6)


def test_log_ndtr_gradient_is_finite_in_the_deep_tail():
    values = torch.tensor([-80.0, -6.0, -5.0, -4.0, 0.0, 10.0], requires_grad=True)
    log_ndtr(values).sum().backward()
    assert torch.isfinite(values.grad).all()


def test_student_t_log_cdf_matches_scipy_and_has_finite_gradient():
    values = torch.tensor(
        [-20.0, -8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0, 20.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    df = torch.tensor(3.0, dtype=torch.float64)
    actual = student_t_log_cdf(values, df)
    expected = torch.as_tensor(
        scipy_student_t.logcdf(values.detach().numpy(), df.item()), dtype=torch.float64
    )
    assert torch.allclose(actual.detach(), expected, rtol=1e-5, atol=5e-6)
    actual.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_student_t_config_resolves_uncertainty_distribution():
    assert TrainConfig(use_student_t_nll=True).val_crps_dist_type == "student_t"
    assert TrainConfig(use_student_t_nll=False).val_crps_dist_type == "gaussian"
    assert TrainConfig(
        use_student_t_nll=True, val_crps_dist_type="gaussian"
    ).val_crps_dist_type == "gaussian"


# --- Tobit loss ------------------------------------------------------------

def _loss_inputs(mean_value):
    recon_mean = torch.full((1, 2, 2), float(mean_value), requires_grad=True)
    recon_logvar = torch.zeros((1, 2, 2))
    target = torch.zeros((1, 2, 2))
    mu = torch.zeros((1, 4))
    logvar = torch.zeros((1, 4))
    return recon_mean, recon_logvar, target, mu, logvar


def test_censored_term_pushes_prediction_below_the_limit():
    """The gradient must reduce a prediction that sits above the limit."""
    recon_mean, recon_logvar, target, mu, logvar = _loss_inputs(2.0)
    censor_mask = torch.ones((1, 2, 2))
    threshold = torch.zeros(2)
    loss, _, _ = vae_loss(
        recon_mean, recon_logvar, target, torch.zeros((1, 2, 2)), mu, logvar, beta=0.0,
        censor_mask=censor_mask, censor_threshold=threshold,
    )
    loss.backward()
    assert (recon_mean.grad > 0).all()  # descending the gradient lowers the mean


def test_student_t_censored_term_uses_student_t_cdf():
    recon_mean, recon_logvar, target, mu, logvar = _loss_inputs(1.0)
    censor_mask = torch.ones((1, 2, 2))
    threshold = torch.zeros(2)
    loss, _, _ = vae_loss(
        recon_mean,
        recon_logvar,
        target,
        torch.zeros((1, 2, 2)),
        mu,
        logvar,
        beta=0.0,
        use_student_t_nll=True,
        censor_mask=censor_mask,
        censor_threshold=threshold,
    )
    expected = -student_t_log_cdf(
        torch.tensor(-3.0**0.5), torch.tensor(3.0)
    )
    assert float(loss.detach()) == pytest.approx(float(expected), rel=1e-5, abs=1e-5)


def test_censored_term_is_cheaper_when_already_below_the_limit():
    threshold = torch.zeros(2)
    censor_mask = torch.ones((1, 2, 2))
    losses = []
    for mean_value in (-3.0, 3.0):
        recon_mean, recon_logvar, target, mu, logvar = _loss_inputs(mean_value)
        loss, _, _ = vae_loss(
            recon_mean, recon_logvar, target, torch.zeros((1, 2, 2)), mu, logvar, beta=0.0,
            censor_mask=censor_mask, censor_threshold=threshold,
        )
        losses.append(float(loss.detach()))
    assert losses[0] < losses[1]


def test_censored_cells_do_not_pull_the_mean_toward_zero():
    """A non-detect must not be supervised as if it were an exact zero."""
    recon_mean, recon_logvar, target, mu, logvar = _loss_inputs(-4.0)
    threshold = torch.zeros(2)
    censored_loss, _, _ = vae_loss(
        recon_mean, recon_logvar, target, torch.zeros((1, 2, 2)), mu, logvar, beta=0.0,
        censor_mask=torch.ones((1, 2, 2)), censor_threshold=threshold,
    )
    recon_mean2, recon_logvar2, target2, mu2, logvar2 = _loss_inputs(-4.0)
    as_observed_loss, _, _ = vae_loss(
        recon_mean2, recon_logvar2, target2, torch.ones((1, 2, 2)), mu2, logvar2, beta=0.0,
    )
    # Treating it as an observed zero penalizes a far-below-limit prediction
    # heavily; the censored treatment barely penalizes it at all.
    assert float(censored_loss.detach()) < float(as_observed_loss.detach())


def test_nan_threshold_for_uncensored_features_does_not_poison_gradients():
    recon_mean, recon_logvar, target, mu, logvar = _loss_inputs(1.0)
    censor_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    threshold = torch.tensor([0.0, float("nan")])
    loss, _, _ = vae_loss(
        recon_mean, recon_logvar, target, torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        mu, logvar, beta=0.0, censor_mask=censor_mask, censor_threshold=threshold,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(recon_mean.grad).all()


def test_loss_is_unchanged_when_no_cell_is_censored():
    recon_mean, recon_logvar, target, mu, logvar = _loss_inputs(1.0)
    obs = torch.ones((1, 2, 2))
    baseline, _, _ = vae_loss(recon_mean, recon_logvar, target, obs, mu, logvar, beta=0.5)
    with_empty_censor, _, _ = vae_loss(
        recon_mean, recon_logvar, target, obs, mu, logvar, beta=0.5,
        censor_mask=torch.zeros((1, 2, 2)), censor_threshold=torch.zeros(2),
    )
    assert float(baseline.detach()) == pytest.approx(float(with_empty_censor.detach()))


# --- dataset plumbing ------------------------------------------------------

def _dataset(censor_mask, **kwargs):
    target = np.tile(np.arange(4, dtype=np.float64), (12, 1))
    aux = np.zeros((12, 1))
    return WindowedTimeSeriesDataset(
        target, aux, window_size=6, stride=6, censor_mask=censor_mask, **kwargs
    )


def test_dataset_splits_observed_and_censored_disjointly():
    censor = np.zeros((12, 4), dtype=bool)
    censor[:, 0] = True
    item = _dataset(censor, mode="val")[0]
    obs, cens = item["obs_mask"].numpy(), item["censor_mask"].numpy()
    assert (obs * cens).sum() == 0
    assert cens[:, 0].all() and not obs[:, 0].any()
    # Censored cells are still visible to the encoder.
    assert item["input_mask"].numpy()[:, 0].all()


def test_heldout_selection_never_scores_a_censored_cell():
    censor = np.zeros((12, 4), dtype=bool)
    censor[:, 0] = True
    dataset = _dataset(
        censor, mode="train",
        dynamic_mask_config={"target_ratio": 0.9, "mean_duration": 4, "std_duration": 0,
                             "min_duration": 2, "max_duration": 6, "n_chem": 4},
    )
    for index in range(len(dataset)):
        item = dataset[index]
        heldout = item["heldout_mask"].numpy()
        assert heldout[:, 0].sum() == 0, "a non-detect has no ground-truth scalar to score"
        assert (heldout * item["censor_mask"].numpy()).sum() == 0


def test_dataset_without_censor_mask_matches_previous_behaviour():
    target = np.tile(np.arange(4, dtype=np.float64), (12, 1))
    aux = np.zeros((12, 1))
    plain = WindowedTimeSeriesDataset(target, aux, 6, 6, mode="val")[0]
    empty = WindowedTimeSeriesDataset(
        target, aux, 6, 6, mode="val", censor_mask=np.zeros((12, 4), dtype=bool)
    )[0]
    for key in ("obs_mask", "input_mask", "input_x", "target"):
        assert torch.equal(plain[key], empty[key])
    assert plain["censor_mask"].sum() == 0


# --- truncation ------------------------------------------------------------

def test_truncated_mean_matches_the_analytic_value():
    # For mu=0, sigma=1, limit=0: P=0.5 and E[y|y<=0] = -phi(0)/Phi(0).
    mean, variance, p_below = truncate_below_limit(
        np.array([[0.0]]), np.array([[1.0]]), np.array([0.0]), np.array([[True]])
    )
    assert p_below[0, 0] == pytest.approx(0.5)
    assert mean[0, 0] == pytest.approx(-np.sqrt(2.0 / np.pi))
    assert variance[0, 0] == pytest.approx(1.0 - 2.0 / np.pi)


@pytest.mark.parametrize(
    "mean,std",
    [(0.0, 1.0), (5.0, 1.0), (20.0, 1.0), (5.0, 0.01), (100.0, 0.001), (1e4, 1e-6)],
)
def test_truncated_mean_never_exceeds_the_limit(mean, std):
    """A confident, far-above-limit prediction must still be pulled under it.

    Regression guard: a direct phi/Phi hazard ratio silently collapses to zero
    once Phi underflows, which skipped the truncation entirely and let a
    non-detect report a value above its own detection limit.
    """
    truncated, _, _ = truncate_below_limit(
        np.array([[mean]]), np.array([[std ** 2]]), np.array([0.0]), np.array([[True]])
    )
    assert truncated[0, 0] <= 0.0


def test_inverse_mills_ratio_is_continuous_across_the_tail_cutoff():
    from graph_temporal_vae.infer import _inverse_mills_ratio

    left = _inverse_mills_ratio(np.array([-5.001]))[0]
    right = _inverse_mills_ratio(np.array([-4.999]))[0]
    assert left == pytest.approx(right, rel=1e-3)
    # Deep in the tail the hazard must approach -alpha, not zero.
    assert _inverse_mills_ratio(np.array([-50.0]))[0] == pytest.approx(50.0, rel=1e-3)


def test_truncation_only_touches_censored_cells():
    mean, variance, _ = truncate_below_limit(
        np.array([[5.0, 5.0]]), np.array([[1.0, 1.0]]), np.array([0.0, 0.0]),
        np.array([[True, False]]),
    )
    assert mean[0, 0] < 0.0
    assert mean[0, 1] == 5.0 and variance[0, 1] == 1.0


# --- reporting -------------------------------------------------------------

def test_report_counts_and_flags_saturated_columns():
    schema = _schema()
    config = CensoringConfig(enabled=True, thresholds={"a": 1.0, "b": 1.0})
    values = np.ones((10, 4))
    values[:, 0] = 0.0      # a fully censored
    values[:3, 1] = 0.0     # b 30% censored
    state = build_state_matrix(values, schema, config)
    report = censoring_report(state, schema, config)
    assert report["counts"]["censored"] == 13
    assert report["per_column"]["a"]["censored_fraction"] == 1.0
    assert report["per_column"]["b"]["censored_fraction"] == pytest.approx(0.3)
    assert high_censoring_columns(report, 0.9) == ["a"]


# --- end to end ------------------------------------------------------------

def _write_modality_csvs(tmp_path, n=180, seed=0):
    """Chemistry with a heavily non-detected trace species reported as zero."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    chem = pd.DataFrame({"time": ts})
    chem["major"] = rng.uniform(2.0, 8.0, n)
    trace = rng.uniform(0.0, 0.4, n)
    trace[trace < 0.3] = 0.0  # non-detects arrive as literal zeros
    chem["trace"] = trace
    psd = pd.DataFrame({"time": ts})
    for diameter in (10.0, 20.0, 40.0):
        psd[str(diameter)] = rng.uniform(1.0, 100.0, n)
    met = pd.DataFrame({"time": ts, "temp": rng.normal(20.0, 3.0, n)})
    paths = {}
    for name, frame in (("chem", chem), ("psd", psd), ("met", met)):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def _train_config(paths, tmp_path, censoring):
    return TrainConfig(
        timestamp_col="time",
        modality_files={"chemistry": [paths["chem"]], "psd": [paths["psd"]],
                        "meteorology": [paths["met"]]},
        preprocessing=PreprocessingConfig(
            chemistry=ModalityPreprocessing("log1p", "standard", "log1p"),
            psd=ModalityPreprocessing("log1p", "standard", "log1p"),
            meteorology=ModalityPreprocessing("none", "standard"),
        ),
        censoring=censoring,
        window_size=12, stride=6, val_fraction=0.3, batch_size=4, epochs=2, patience=2,
        model_kwargs={"latent_dim": 4, "hidden_dims": [8, 8], "encoder_layers": 1,
                      "decoder_layers": 1, "n_graph_heads": 1, "dropout": 0.0,
                      "heteroscedastic": True},
    )


def test_end_to_end_censored_run_respects_the_detection_limit(tmp_path):
    paths = _write_modality_csvs(tmp_path)
    censoring = CensoringConfig(enabled=True, thresholds={"trace": 0.3})
    bundle_path = tmp_path / "model.pt"
    train_from_config(_train_config(paths, tmp_path, censoring), str(bundle_path))

    bundle = load_bundle(str(bundle_path))
    assert bundle["censoring"].active
    assert bundle["censor_threshold_scaled"] is not None

    result = impute(
        None, str(bundle_path), None,
        modality_files={"chemistry": [paths["chem"]], "psd": [paths["psd"]],
                        "meteorology": [paths["met"]]},
    )
    censored = result[result["observation_state"] == "censored"]
    assert len(censored) > 0
    assert (censored["feature"] == "trace").all()
    # The reported value must not contradict the measurement that produced it.
    assert (censored["imputed_mean"] <= censored["detection_limit"] + 1e-9).all()
    assert (censored["q_upper"] <= censored["detection_limit"] + 1e-9).all()
    assert censored["p_below_limit"].between(0.0, 1.0).all()


def test_scaler_fit_excludes_censored_cells(tmp_path):
    """Substituted non-detects must not drag the normalization center down."""
    paths = _write_modality_csvs(tmp_path)
    plain = tmp_path / "plain.pt"
    censored = tmp_path / "censored.pt"
    train_from_config(_train_config(paths, tmp_path, None), str(plain))
    train_from_config(
        _train_config(paths, tmp_path, CensoringConfig(enabled=True, thresholds={"trace": 0.3})),
        str(censored),
    )
    trace_index = load_bundle(str(plain))["target_cols"].index("trace")
    plain_center = load_bundle(str(plain))["scaler_target"].center_[trace_index]
    censored_center = load_bundle(str(censored))["scaler_target"].center_[trace_index]
    assert censored_center > plain_center


def test_censoring_disabled_is_bit_identical_to_the_previous_pipeline(tmp_path):
    paths = _write_modality_csvs(tmp_path)
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    train_from_config(_train_config(paths, tmp_path, None), str(first))
    train_from_config(
        _train_config(paths, tmp_path, CensoringConfig(enabled=False, thresholds={"trace": 0.3})),
        str(second),
    )
    a, b = load_bundle(str(first)), load_bundle(str(second))
    assert np.allclose(a["scaler_target"].center_, b["scaler_target"].center_)
    for key, tensor in a["model"].state_dict().items():
        assert torch.allclose(tensor, b["model"].state_dict()[key]), key


def test_cli_accepts_a_threshold_table(tmp_path):
    from graph_temporal_vae.cli import main as cli_main

    paths = _write_modality_csvs(tmp_path)
    table = tmp_path / "mdl.json"
    table.write_text(json.dumps({"trace": 0.3}))
    bundle_path = tmp_path / "cli.pt"
    cli_main([
        "train",
        "--chem-csv", paths["chem"], "--psd-csv", paths["psd"], "--met-csv", paths["met"],
        "--timestamp-col", "time", "--window-size", "12", "--stride", "6",
        "--epochs", "1", "--batch-size", "4", "--val-fraction", "0.3",
        "--latent-dim", "4", "--hidden-dims", "8,8", "--encoder-layers", "1",
        "--decoder-layers", "1", "--n-graph-heads", "1",
        "--chem-transform", "log1p", "--psd-transform", "log1p",
        "--censoring-thresholds", str(table),
        "-o", str(bundle_path),
    ])
    assert load_bundle(str(bundle_path))["censoring"].active
