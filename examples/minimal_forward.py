from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_tcn_vae import ImputationVAE_Graph


def main():
    torch.manual_seed(7)

    batch_size = 2
    window_size = 24
    target_dim = 6
    aux_dim = 3

    model = ImputationVAE_Graph(
        target_dim=target_dim,
        aux_dim=aux_dim,
        window_size=window_size,
        latent_dim=12,
        hidden_dims=[24, 24],
        encoder_layers=2,
        decoder_layers=2,
        n_graph_heads=2,
        n_chem=3,
        dropout=0.0,
    )

    x = torch.randn(batch_size, window_size, target_dim)
    cond = torch.randn(batch_size, window_size, aux_dim)
    mask = torch.randint(0, 2, (batch_size, window_size, target_dim)).float()
    x_masked = x * mask

    model.eval()
    with torch.no_grad():
        recon_mean, recon_logvar, mu, logvar, graph_attention = model(
            x_masked,
            cond,
            mask,
        )

    print("recon_mean:", tuple(recon_mean.shape))
    print("recon_logvar:", None if recon_logvar is None else tuple(recon_logvar.shape))
    print("mu:", tuple(mu.shape))
    print("logvar:", tuple(logvar.shape))
    print(
        "graph_attention:",
        None if graph_attention is None else tuple(graph_attention.shape),
    )


if __name__ == "__main__":
    main()
