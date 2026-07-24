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

See `examples/minimal_forward.py` for a runnable architecture example.

## Guided Nanzi Demo

A compact 96-hour excerpt from the Nanzi aerosol supersite record is included at `examples/data/nanzi_demo_96h.csv`. The dataset contains four composition targets, four representative PSD bins, meteorological/time conditioning variables, and natural missing values.

Open `examples/nanzi_demo_workflow.ipynb` for a complete walkthrough:

1. inspect missingness;
2. validate the hourly timestamp grid and column contract;
3. train a deliberately small demonstration model;
4. save and reload a self-contained checkpoint bundle;
5. impute natural gaps with predictive intervals;
6. inspect gap length, bilateral context support, and heuristic risk tiers.

The excerpt is intended only to demonstrate the software interface. It is too short for scientifically valid aerosol reconstruction, calibrated uncertainty, or comparison with the reported 26e research results.

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

The `graph-tcn-vae` command trains `ImputationVAE_Graph` and saves a self-contained checkpoint **bundle** containing weights, the resolved data schema, preprocessing choices, fitted affine statistics, architecture versions, and model settings. Inference reuses this contract and never re-fits preprocessing.

### Multimodal data contract

The preferred interface supplies each measurement modality separately:

- `--chem-csv`: every non-timestamp column is a chemistry target; file/column order is preserved;
- `--psd-csv`: every non-timestamp column is a PSD target; column names must be positive particle diameters in nm and are sorted numerically;
- `--met-csv`: every non-timestamp column is an auxiliary conditioning variable;
- `--timestamp-col`: shared timestamp column, default `time`;
- missing measurements are represented by empty CSV cells/`NaN`.

At least one of Chem or PSD is required. Files are outer-joined on timestamp, Chem targets are placed before PSD targets, and `n_chem` is inferred automatically. The exact discovered columns and ordering are stored in `DataSchema` and enforced on future inference data.

```bash
graph-tcn-vae validate-data \
  --chem-csv data/chemistry.csv \
  --psd-csv data/psd.csv \
  --met-csv data/meteorology.csv \
  --timestamp-col time \
  --expected-frequency 1h

graph-tcn-vae train \
  --chem-csv data/chemistry.csv \
  --psd-csv data/psd.csv \
  --met-csv data/meteorology.csv \
  --chem-transform log1p --chem-scaler standard \
  --psd-transform log1p --psd-scaler robust \
  --met-transform none --met-scaler standard \
  --window-size 48 --stride 24 \
  --epochs 200 \
  -o checkpoints/run1.pt

graph-tcn-vae impute \
  --bundle checkpoints/run1.pt \
  --chem-csv data/new_chemistry.csv \
  --psd-csv data/new_psd.csv \
  --met-csv data/new_meteorology.csv \
  --inference-config examples/inference_config.example.json \
  -o imputed.csv
```

The legacy `--csv --target-cols --aux-cols` interface remains supported for general tabular datasets and old scripts.

Each modality independently selects an input transform (`none` or `log1p`) and affine scaler (`standard`, `robust`, `minmax`, or `none`). `standard` uses mean/standard deviation, `robust` uses median/IQR, and `minmax` uses minimum/range. `--scaler-fit-scope train` is the leakage-safe default; `full` exists only for protocols that intentionally fit preprocessing on the complete timeline.

`--chem-output-transform` and `--psd-output-transform` define how de-standardized model values are mapped to the public output scale. They normally match their input transform. The separate setting is needed when a CSV already contains log1p values:

```bash
graph-tcn-vae train \
  --csv experiment_input_log1p.csv \
  --timestamp-col time \
  --target-cols species_a,species_b \
  --target-transform none \
  --target-output-transform log1p \
  -o checkpoints/pretransformed_run.pt
```

`impute` writes a tidy, long-format CSV: one row per `(timestamp, feature)`, with `observed`, `is_imputed`, `imputed_mean`, `imputed_std`, `q_lower`, `q_upper`, and the interval probabilities. The default 5–95% interval also provides compatibility aliases `q05`/`q95`. Observed points are restored in the configured output scale with zero reported uncertainty; only genuinely missing points get a model-derived estimate. Overlapping inference windows use exact sample-level overlap-add with a trapezoidal position envelope. The bounded-memory aggregator keeps samples only for timestamps that can still be covered by a later window, then finalizes and releases them. Aggregation occurs before each feature's output transform, avoiding biased sample-wise exponentiation.

For the 26e Chem/PSD weighting protocol, set `--n-chem 32 --chem-feature-weight 12 --psd-feature-weight 1 --loss-normalization window_feature_sum`. The defaults are 1/1 general-dataset weighting with `observed_mean` normalization.

By default the learning rate follows linear warmup + cosine annealing for the whole run. The 26e reference instead switches to `ReduceLROnPlateau` (monitoring held-out MSE) once warmup ends; pass `--use-adaptive-lr` (plus `--lr-reduce-factor/--lr-reduce-patience/--lr-reduce-threshold/--lr-reduce-cooldown` to match a specific reference run) to reproduce that behavior.

When auxiliary columns are configured, the CLI automatically appends one observedness channel per auxiliary column to `cond`. A missing auxiliary value is therefore represented as `(zero-filled value, mask=0)` and is not silently treated as a real standardized zero. The target `mask` remains target-only.

To reproduce the private 26e Chem/PSD reference input contract, first derive the
five meteorological/BLH features and six cyclic time features from the raw
source files, then use the explicit parity runner:

```bash
python examples/prepare_26e_input.py \
  --chem data/chem_2024_2025_clean.csv \
  --psd data/psd_2024_2025.csv \
  --blh data/blh.csv \
  -o data/experiment_input_raw_26e.csv

python examples/train_26e_parity.py \
  --input data/experiment_input_raw_26e.csv \
  -o checkpoints/26e_parity.pt
```

This reference runner intentionally uses the full 15,336-row timeline (633
windows at `window=168, stride=24`), `seed=42` for both training and the
anchor-constrained held-out mask, `ho_mse` selection, AdamW with
`weight_decay=0.01`, legacy dynamic masking, and the 26e Student-t/loss
weighting schedule. It is separate from the general CLI defaults. If the
input is an already log1p-transformed artifact, pass
`--target-transform none`; the preparation helper above writes raw targets and
therefore uses the runner default `--target-transform log1p`.

Any validated `ModelConfig` field (see `graph_tcn_vae/model_config.py`) can be set via `--model-config path/to/config.json`. Unknown or stale fields fail before training starts. A complete nested train configuration can instead be supplied with `--train-config examples/multimodal_train_config.example.json`.

The same functionality is available programmatically:

```python
from graph_tcn_vae import (
    InferenceConfig,
    ModalityFiles,
    ModalityPreprocessing,
    PreprocessingConfig,
    TrainConfig,
    impute,
    train_from_config,
)

files = ModalityFiles(
    chemistry=["data/chemistry.csv"],
    psd=["data/psd.csv"],
    meteorology=["data/meteorology.csv"],
)
preprocessing = PreprocessingConfig(
    chemistry=ModalityPreprocessing(transform="log1p", scaler="standard"),
    psd=ModalityPreprocessing(transform="log1p", scaler="robust"),
    meteorology=ModalityPreprocessing(transform="none", scaler="standard"),
)
config = TrainConfig(
    modality_files=files,
    timestamp_col="time",
    preprocessing=preprocessing,
    window_size=48,
    stride=24,
)
train_from_config(config, "checkpoints/run1.pt")

impute(
    None,
    "checkpoints/run1.pt",
    "imputed.csv",
    modality_files=files,
    inference_config=InferenceConfig(n_mc_samples=50),
)
```

Training uses dynamic contiguous held-out masking on observed target values. The validation mask is generated once from the same protocol and then held fixed for early stopping, preventing epoch-to-epoch mask noise from changing the selection target. `--validation-metric ho_nll` is the calibration-aware default and now follows the configured training likelihood: Gaussian training uses Gaussian held-out NLL, while `--use-student-t-nll` uses the same Student-t formulation, model likelihood degrees of freedom, and decoder variance bounds during validation. `ho_mse` selects directly on point-estimation error, while `ho_crps` is available with configurable MC sample count and evaluation interval; empirical CRPS is evaluated with an exact sorted-sample formula rather than an `MC × MC` pairwise tensor. This is still a general-purpose reference implementation, not a reproduction of any specific thesis training run; W&B logging and extensive gate/attention diagnostics remain research-specific.

`impute` always reports zero uncertainty at genuinely observed points, so it cannot answer "how accurate is this model on data it never saw?" `examples/heldout_eval.py` fills that gap: it regenerates the exact fixed held-out mask a bundle was trained with (same `--n-chem`/ratio/seed), forces those points to look unobserved at inference time, and reports R²/MAE/PICP against their true values, split by Chem/PSD. It preserves the existing physical-output Gaussian CRPS for reference comparability and additionally reports `empirical_crps_model_space`, computed exactly from the bounded-memory overlap mixture using a sorted weighted-sample formula:

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
