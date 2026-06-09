import torch

from graph_tcn_vae import ImputationVAE, ImputationVAE_Graph, ImputationVAE_UQ


def _inputs(batch_size=2, window_size=24, target_dim=6, aux_dim=3):
    torch.manual_seed(11)
    x = torch.randn(batch_size, window_size, target_dim)
    cond = torch.randn(batch_size, window_size, aux_dim)
    mask = torch.randint(0, 2, (batch_size, window_size, target_dim)).float()
    return x * mask, cond, mask


def test_base_vae_forward_shapes():
    x, cond, mask = _inputs()
    model = ImputationVAE(
        target_dim=6,
        aux_dim=3,
        window_size=24,
        latent_dim=12,
        hidden_dims=[24],
        encoder_layers=2,
        decoder_layers=2,
    )
    model.eval()

    with torch.no_grad():
        recon, mu, logvar = model(x, cond, mask)

    assert recon.shape == x.shape
    assert mu.shape == (x.shape[0], 12)
    assert logvar.shape == (x.shape[0], 12)


def test_uq_vae_forward_shapes():
    x, cond, mask = _inputs()
    model = ImputationVAE_UQ(
        target_dim=6,
        aux_dim=3,
        window_size=24,
        latent_dim=12,
        hidden_dims=[24],
        encoder_layers=2,
        decoder_layers=2,
        heteroscedastic=True,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        recon_mean, recon_logvar, mu, logvar = model(x, cond, mask)

    assert recon_mean.shape == x.shape
    assert recon_logvar.shape == x.shape
    assert mu.shape == (x.shape[0], 12)
    assert logvar.shape == (x.shape[0], 12)


def test_graph_vae_forward_shapes():
    x, cond, mask = _inputs()
    model = ImputationVAE_Graph(
        target_dim=6,
        aux_dim=3,
        window_size=24,
        latent_dim=12,
        hidden_dims=[24, 24],
        encoder_layers=2,
        decoder_layers=2,
        n_graph_heads=2,
        n_chem=3,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        recon_mean, recon_logvar, mu, logvar, _graph_attention = model(x, cond, mask)

    assert recon_mean.shape == x.shape
    assert recon_logvar.shape == x.shape
    assert mu.shape == (x.shape[0], 12)
    assert logvar.shape == (x.shape[0], 12)
