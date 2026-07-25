import pytest
import torch

from graph_temporal_vae import ImputationVAE, ImputationVAE_Graph, ImputationVAE_UQ
from graph_temporal_vae.model_graph_uq import CrossModalGraphLayer


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


def test_cross_modal_query_gate_mode_legacy_hard_blocks_unobserved_target_queries():
    # A target feature observed 0% of the window has obs_rate=0 < the legacy
    # 10% threshold: 'legacy_hard' must hard-mask that query row to zero, so
    # it gets NO aux-conditioning signal at all.
    torch.manual_seed(0)
    layer = CrossModalGraphLayer(
        target_dim=2, aux_dim=3, window_size=8, n_heads=1, head_dim=8, query_gate_mode='legacy_hard'
    )
    layer.eval()
    x_target = torch.randn(1, 2, 8)
    x_aux = torch.randn(1, 3, 8)
    target_mask = torch.ones(1, 8, 2)
    target_mask[:, :, 0] = 0.0  # feature 0 fully unobserved

    with torch.no_grad():
        layer(x_target, x_aux, target_mask)

    assert torch.allclose(layer.last_attention_weights[0], torch.zeros(3))


def test_cross_modal_query_gate_mode_none_still_attends_unobserved_target_queries():
    # 'none' mode -- what 26e_maskdist_heldout_like_gate_none actually uses --
    # must NOT suppress the same fully-unobserved query row: a missing target
    # feature should still be able to retrieve aux evidence.
    torch.manual_seed(0)
    layer = CrossModalGraphLayer(
        target_dim=2, aux_dim=3, window_size=8, n_heads=1, head_dim=8, query_gate_mode='none'
    )
    layer.eval()
    x_target = torch.randn(1, 2, 8)
    x_aux = torch.randn(1, 3, 8)
    target_mask = torch.ones(1, 8, 2)
    target_mask[:, :, 0] = 0.0  # feature 0 fully unobserved

    with torch.no_grad():
        layer(x_target, x_aux, target_mask)

    row = layer.last_attention_weights[0]
    assert row.sum().item() == pytest.approx(1.0, abs=1e-5)  # a valid softmax, not zeroed
    assert (row > 0).all()


def test_cross_modal_query_gate_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        CrossModalGraphLayer(target_dim=2, aux_dim=3, window_size=8, query_gate_mode='bogus')


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
