import numpy as np
import pandas as pd
import pytest
import torch

from graph_tcn_vae.cli import main as cli_main
from graph_tcn_vae.config import TrainConfig
from graph_tcn_vae.data import sample_dynamic_heldout_mask
from graph_tcn_vae.infer import aggregate_window_samples, load_bundle
from graph_tcn_vae.train import Trainer, vae_loss


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
    )

    saved = config.to_dict()
    assert saved["kl_warmup_ratio"] == 0.1
    assert saved["kl_strategy"] == "cosine"
    assert saved["use_amp"] is True
    assert saved["prior_type"] == "student_t"
    assert saved["use_student_t_nll"] is True
    assert saved["loss_normalization"] == "window_feature_sum"
    assert saved["aux_mask_channel"] is False


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
