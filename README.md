# Graph-Temporal VAE

Graph-Temporal VAE provides model architecture implementations for graph-enhanced temporal variational autoencoders designed for uncertainty-aware time-series imputation.

It is a lightweight public package focused on reusable model code. Full research experiments, datasets, checkpoints, analysis notebooks, and thesis materials are intentionally not included.

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

## What Is Not Included

This public package does not include:

- training pipelines from the research workspace,
- experiment scripts,
- private or licensed datasets,
- generated results,
- model checkpoints,
- thesis drafts or analysis notebooks.

## Development Check

```bash
pytest
python examples/minimal_forward.py
```

## Citation

Citation information will be added after publication.
