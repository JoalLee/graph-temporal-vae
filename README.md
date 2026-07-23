# Graph-Temporal VAE

Graph-Temporal VAE provides model architecture implementations for graph-enhanced temporal variational autoencoders designed for uncertainty-aware time-series imputation, plus a reference CLI for training on your own CSV data and running imputation/uncertainty inference against it.

It is a lightweight public package. Full research experiments, private datasets, trained checkpoints, analysis notebooks, and thesis materials are intentionally not included — see [What Is Not Included](#what-is-not-included).

## Why This Matters

High-resolution environmental monitoring systems often produce rich but incomplete time series. In aerosol supersite measurements, instrument downtime, calibration, flow instability, liquid handling issues, and severe-weather protection shutdowns can fragment co-located particle size distribution (PSD), chemical speciation, and meteorological records. These gaps are not simple isolated missing values; they often occur as structured, modality-dependent outages in a coupled physicochemical system.

Incomplete aerosol records limit downstream analyses such as mass closure, optical closure, source apportionment, exposure assessment, and climate-relevant aerosol process studies. Common gap-handling approaches such as deletion, zero-fill, mean substitution, or short-segment interpolation can discard expensive observations or introduce systematic bias, especially when full PSD spectra or groups of chemical measurements are unavailable.

This project is motivated by the need for imputation models that can recover coupled chemical and microphysical aerosol states while also reporting uncertainty. For scientific use, an imputed value should not be treated as equivalent to a direct measurement; its predictive uncertainty and operating context should travel with it.

## Purpose

The main purpose of this package is to make the Graph-TCN-VAE model architecture reusable outside the original research workspace. The architecture combines:

- feature-space graph learning, where chemical species and PSD size bins are represented as feature nodes rather than monitoring stations;
- temporal encoding with dilated temporal convolutional blocks, which captures local, diurnal, and multi-day structure inside each moving window;
- probabilistic latent-variable modeling, which represents unresolved ambiguity in partially observed aerosol states;
- heteroscedastic decoding, which returns predictive means together with feature- and time-dependent uncertainty estimates;
- optional auxiliary conditioning, allowing meteorological and temporal variables to guide reconstruction without being reconstruction targets.

The implementation is site-agnostic at the architecture level. It can be adapted to other multivariate time-series imputation problems where missingness is structured, variables are interdependent, and uncertainty estimates are required.

## Main Use Cases

- Reconstructing missing aerosol chemical speciation and PSD time series from co-located monitoring instruments.
- Building uncertainty-aware "virtual sensor" workflows for environmental monitoring data.
- Studying feature-to-feature dependencies in high-dimensional time-series systems through learned graph attention or sensitivity analysis.
- Prototyping graph-temporal VAE models for other scientific sensor networks with structured missingness.
- Providing a clean architecture reference for papers, thesis work, or downstream model extensions without exposing private datasets or experiment artifacts.

## Included Models

- `ImputationVAE`: baseline conditional TCN-VAE.
- `ImputationVAE_UQ`: uncertainty-aware VAE with heteroscedastic output support.
- `ImputationVAE_Graph`: graph-enhanced UQ-VAE with feature-level graph attention and optional cross-modal conditioning.
- `PredictionVAE_Graph`: graph-enhanced VAE variant for forecasting.

## Installation

```bash
git clone https://github.com/JoalLee/graph-temporal-vae.git
cd graph-temporal-vae
pip install -e ".[dev]"
```

## Minimal Usage

```python
import torch
from graph_tcn_vae import ImputationVAE_Graph

batch_size = 2
window_size = 48
target_dim = 8
aux_dim = 4

model = ImputationVAE_Graph(
    target_dim=target_dim,
    aux_dim=aux_dim,
    window_size=window_size,
    latent_dim=16,
    hidden_dims=[32, 32],
    encoder_layers=2,
    decoder_layers=2,
    n_graph_heads=2,
    n_chem=4,
)

x = torch.randn(batch_size, window_size, target_dim)
cond = torch.randn(batch_size, window_size, aux_dim)
mask = torch.randint(0, 2, (batch_size, window_size, target_dim)).float()

model.eval()
with torch.no_grad():
    recon_mean, recon_logvar, mu, logvar, graph_attention = model(x, cond, mask)

print(recon_mean.shape)
```

See `examples/minimal_forward.py` for a runnable example.

## Input Shapes

For imputation models:

- `x`: `[batch, window, target_dim]`, masked target time series.
- `cond`: `[batch, window, aux_dim]`, auxiliary or conditioning features.
- `mask`: `[batch, window, target_dim]`, observation mask where `1` means observed and `0` means missing.

`ImputationVAE_Graph` returns:

- `recon_mean`: `[batch, window, target_dim]`
- `recon_logvar`: `[batch, window, target_dim]` or `None`
- `mu`: `[batch, latent_dim]`
- `logvar`: `[batch, latent_dim]`
- `graph_attention`: learned feature relationship tensor when available.

## Training and Inference

The `graph-tcn-vae` command trains `ImputationVAE_Graph` on your own CSV data and saves a single self-contained checkpoint **bundle** (weights + model kwargs + column names + the training-fit normalization stats), so later `impute` runs need nothing else to reproduce the exact model.

Column selection is name-based: point `--target-cols` at the variables you want imputed/predicted and `--aux-cols` at conditioning variables (meteorology, engineered time features, etc.). Multiple `--csv` paths are outer-joined on `--timestamp-col`, so co-located instruments can live in separate files.

```bash
graph-tcn-vae train \
  --csv path/to/data.csv \
  --timestamp-col time \
  --target-cols species_a,species_b,species_c \
  --aux-cols wind_speed,wind_dir,temperature \
  --window-size 48 --stride 24 \
  --epochs 200 \
  -o checkpoints/run1.pt

graph-tcn-vae impute \
  --bundle checkpoints/run1.pt \
  --csv path/to/new_data.csv \
  -o imputed.csv
```

`--target-transform` describes the values in the input CSV: `log1p` applies one log1p transform before scaling/training, while `none` leaves the CSV values unchanged. `--target-output-transform` describes the public output scale and defaults to `--target-transform`. This separation is important when the input CSV is already log1p-preprocessed, as in the 26e experiment artifact:

```bash
graph-tcn-vae train \
  --csv experiment_input_log1p.csv \
  --timestamp-col time \
  --target-cols species_a,species_b \
  --target-transform none \
  --target-output-transform log1p \
  -o checkpoints/pretransformed_run.pt
```

`impute` writes a tidy, long-format CSV: one row per `(timestamp, feature)`, with the observed value in the configured output scale, plus `imputed_mean`, `imputed_std`, and a 5–95% predictive interval (`q05`/`q95`) from `compute_uncertainty`. Observed points are restored in that output scale with zero reported uncertainty; only genuinely missing points get a model-derived estimate. Overlapping inference windows use sample-level overlap-add with a trapezoidal position envelope, so quantiles and cross-window disagreement are retained instead of averaging per-window quantiles.

For the 26e Chem/PSD weighting protocol, set `--n-chem 32 --chem-feature-weight 12 --psd-feature-weight 1 --loss-normalization window_feature_sum`. The defaults are 1/1 general-dataset weighting with `observed_mean` normalization.

By default the learning rate follows linear warmup + cosine annealing for the whole run. The 26e reference instead switches to `ReduceLROnPlateau` (monitoring held-out MSE) once warmup ends; pass `--use-adaptive-lr` (plus `--lr-reduce-factor/--lr-reduce-patience/--lr-reduce-threshold/--lr-reduce-cooldown` to match a specific reference run) to reproduce that behavior.

When auxiliary columns are configured, the CLI automatically appends one observedness channel per auxiliary column to `cond`. A missing auxiliary value is therefore represented as `(zero-filled value, mask=0)` and is not silently treated as a real standardized zero. The target `mask` remains target-only.

Any `ImputationVAE_Graph` constructor flag (see `graph_tcn_vae/model_graph_uq.py`) can be set via `--model-config path/to/config.json`, which takes priority over the convenience flags (`--latent-dim`, `--hidden-dims`, `--encoder-layers`, `--decoder-layers`, `--n-graph-heads`, `--n-chem`, `--heteroscedastic`).

The same functionality is available programmatically:

```python
from graph_tcn_vae import TrainConfig, train_from_config, load_bundle, impute

config = TrainConfig(
    csv=["path/to/data.csv"],
    timestamp_col="time",
    target_cols=["species_a", "species_b"],
    aux_cols=["wind_speed", "temperature"],
)
train_from_config(config, "checkpoints/run1.pt")

impute(["path/to/new_data.csv"], "checkpoints/run1.pt", "imputed.csv")
```

Training uses dynamic contiguous held-out masking on observed target values. The validation mask is generated once from the same protocol and then held fixed for early stopping, preventing epoch-to-epoch mask noise from changing the selection target. `--validation-metric ho_nll` is the calibration-aware default; `ho_mse` selects directly on point-estimation error, while `ho_crps` is available with configurable MC sample count and evaluation interval. This is still a general-purpose reference implementation, not a reproduction of any specific thesis training run; W&B logging and extensive gate/attention diagnostics remain research-specific.

`impute` always reports zero uncertainty at genuinely observed points, so it cannot answer "how accurate is this model on data it never saw?" `examples/heldout_eval.py` fills that gap: it regenerates the exact fixed held-out mask a bundle was trained with (same `--n-chem`/ratio/seed), forces those points to look unobserved at inference time, and reports R²/MAE/PICP against their true values, split by Chem/PSD:

```bash
python examples/heldout_eval.py \
  --bundle checkpoints/run1.pt --csv path/to/data.csv \
  --n-chem 32 -o heldout_metrics.json
```

### Progress output

Training and inference print one summary line up front (row/window counts, device, epoch or MC-sample budget) and one line per epoch during training. On an interactive terminal this is backed by `tqdm` progress bars (per-epoch, per-batch, and per-window during inference); when stdout/stderr isn't a TTY (`nohup`, a background job, CI), the bars disable themselves automatically and only the plain summary lines are printed, so log files stay readable instead of filling up with `\r`-fragmented bar redraws.

## What Is Not Included

This public package does not include:

- the private research workspace's experiment scripts, ablation harnesses, or W&B/diagnostic instrumentation,
- private or licensed datasets,
- trained model checkpoints or generated results,
- thesis drafts or analysis notebooks.

The training/inference CLI above is a general-purpose reference pipeline shipped with this package; it is separate from the above.

## Development Check

```bash
pytest
python examples/minimal_forward.py
```

## Citation

Citation information will be added after publication.
