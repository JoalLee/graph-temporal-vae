"""Behavioral parity contracts for the staged model_graph_uq split.

The public monolith remains the compatibility reference until GraphEncoder,
GraphDecoder, and ImputationVAE_Graph are moved.  These tests prevent the
new graph_blocks/graph_layers modules from drifting while the migration is in
progress.
"""

import torch

from graph_tcn_vae import model_graph_uq as legacy
from graph_tcn_vae.graph_blocks.tcn import DepthwiseTCN
from graph_tcn_vae.graph_blocks.time_encoding import TimeHybridEncoder
from graph_tcn_vae.graph_layers.cross_modal_graph import CrossModalGraphLayer
from graph_tcn_vae.graph_layers.input_graph import InputGraphLayer


def _assert_same_state_and_output(old_cls, new_cls, kwargs, inputs):
    torch.manual_seed(123)
    old = old_cls(**kwargs).eval()
    torch.manual_seed(123)
    new = new_cls(**kwargs).eval()

    assert old.state_dict().keys() == new.state_dict().keys()
    for name, old_value in old.state_dict().items():
        torch.testing.assert_close(old_value, new.state_dict()[name])

    with torch.no_grad():
        old_output = old(*inputs)
        new_output = new(*inputs)

    if isinstance(old_output, tuple):
        assert isinstance(new_output, tuple)
        assert len(old_output) == len(new_output)
        for old_value, new_value in zip(old_output, new_output):
            if old_value is None:
                assert new_value is None
            else:
                torch.testing.assert_close(old_value, new_value)
    else:
        torch.testing.assert_close(old_output, new_output)


def test_time_hybrid_encoder_matches_monolith():
    cyc = torch.tensor(
        [[[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]]
    )
    _assert_same_state_and_output(
        legacy.TimeHybridEncoder,
        TimeHybridEncoder,
        {"out_dim": 6, "dropout": 0.0},
        (cyc,),
    )


def test_depthwise_tcn_matches_monolith():
    x = torch.randn(2, 4, 16)
    _assert_same_state_and_output(
        legacy.DepthwiseTCN,
        DepthwiseTCN,
        {"channels": 4, "num_layers": 3, "kernel_size": 3},
        (x,),
    )


def test_input_graph_layer_matches_monolith_default_semantics():
    x = torch.randn(2, 5, 12)
    mask = (torch.rand(2, 12, 5) > 0.2).float()
    kwargs = {
        "n_features": 5,
        "window_size": 12,
        "n_heads": 1,
        "head_dim": 8,
        "dropout": 0.0,
        "aux_dim": 2,
        "n_chem": 2,
        "use_homogeneous": True,
        "use_ffn": True,
    }
    _assert_same_state_and_output(
        legacy.InputGraphLayer,
        InputGraphLayer,
        kwargs,
        (x, mask),
    )


def test_cross_modal_graph_layer_matches_monolith_default_semantics():
    target = torch.randn(2, 5, 12)
    aux = torch.randn(2, 3, 12)
    target_mask = (torch.rand(2, 12, 5) > 0.2).float()
    kwargs = {
        "target_dim": 5,
        "aux_dim": 3,
        "window_size": 12,
        "n_heads": 1,
        "head_dim": 8,
        "dropout": 0.0,
        "use_ffn": True,
        "query_gate_mode": "legacy_hard",
    }
    _assert_same_state_and_output(
        legacy.CrossModalGraphLayer,
        CrossModalGraphLayer,
        kwargs,
        (target, aux, target_mask),
    )
