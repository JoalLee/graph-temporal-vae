# Graph TCN-VAE

This repository provides the model architecture implementation for a graph-enhanced TCN-VAE for time-series imputation with uncertainty estimation.

It is a lightweight public package focused on reusable model code. Full research experiments, datasets, checkpoints, analysis notebooks, and thesis materials are intentionally not included.

## Included Models

- `ImputationVAE`: baseline conditional TCN-VAE.
- `ImputationVAE_UQ`: uncertainty-aware VAE with heteroscedastic output support.
- `ImputationVAE_Graph`: graph-enhanced UQ-VAE with feature-level graph attention and optional cross-modal conditioning.
- `PredictionVAE_Graph`: graph-enhanced VAE variant for forecasting.

## Installation

```bash
git clone https://github.com/<your-user>/graph-tcn-vae.git
cd graph-tcn-vae
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
