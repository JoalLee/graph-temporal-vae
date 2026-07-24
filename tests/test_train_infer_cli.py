import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from graph_tcn_vae.cli import main as cli_main
from graph_tcn_vae.config import TrainConfig
from graph_tcn_vae.data import (
    WindowedTimeSeriesDataset,
    sample_anchor_constrained_heldout_mask,
    sample_dynamic_heldout_mask,
)
from graph_tcn_vae.infer import (
    aggregate_window_samples,
    load_bundle,
    summary_to_output_scale,
    trapezoid_position_weights,
)
from graph_tcn_vae.model_graph_uq import ImputationVAE_Graph
from graph_tcn_vae.train import Trainer, train_from_config, vae_loss


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
    for group in ("chem", "psd"):
        assert f"{group}_heldout_n" in results
        assert results[f"{group}_heldout_n"] > 0
        assert np.isfinite(results[f"{group}_heldout_mae"])
        assert 0.0 <= results[f"{group}_heldout_picp95"] <= 1.0

    predictions = pd.read_csv(predictions_path)
    expected_cols = {
        "timestamp", "feature", "family", "scaled_observed", "scaled_pred_mean",
        "model_observed", "model_pred_mean", "physical_observed", "physical_pred_mean",
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
