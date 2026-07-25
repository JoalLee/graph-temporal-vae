import pytest
import torch

from graph_tcn_vae.latent_diffusion import (
    ConditionalLatentResidualDiffusion,
    HeldoutLikeMaskConfig,
    LatentResidualDiffusionConfig,
    build_latent_condition,
    optimize_latent_target,
    sample_heldout_like_mask,
)


def test_diffusion_loss_is_finite_and_backpropagates():
    torch.manual_seed(7)
    config = LatentResidualDiffusionConfig(
        timesteps=12,
        hidden_dim=32,
        time_embedding_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    model = ConditionalLatentResidualDiffusion(
        latent_dim=8,
        condition_dim=20,
        config=config,
    )
    delta_z = torch.randn(4, 8)
    condition = torch.randn(4, 20)

    loss = model.loss(delta_z, condition)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_diffusion_can_fit_a_fixed_denoising_batch():
    torch.manual_seed(13)
    model = ConditionalLatentResidualDiffusion(
        latent_dim=4,
        condition_dim=6,
        config=LatentResidualDiffusionConfig(
            timesteps=6,
            hidden_dim=32,
            time_embedding_dim=12,
            num_layers=2,
            dropout=0.0,
        ),
    )
    delta = torch.randn(12, 4)
    condition = torch.randn(12, 6)
    timesteps = torch.arange(12) % 6
    noise = torch.randn_like(delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    initial = float(
        model.loss(delta, condition, timesteps=timesteps, noise=noise).detach()
    )
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(delta, condition, timesteps=timesteps, noise=noise)
        loss.backward()
        optimizer.step()
    final = float(
        model.loss(delta, condition, timesteps=timesteps, noise=noise).detach()
    )

    assert final < initial * 0.1


def test_diffusion_sampling_is_reproducible_with_generator():
    config = LatentResidualDiffusionConfig(
        timesteps=8,
        hidden_dim=24,
        time_embedding_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    model = ConditionalLatentResidualDiffusion(
        latent_dim=6,
        condition_dim=10,
        config=config,
    )
    model.eval()
    condition = torch.randn(3, 10)

    first = model.sample(
        condition,
        num_samples=4,
        generator=torch.Generator().manual_seed(19),
    )
    second = model.sample(
        condition,
        num_samples=4,
        generator=torch.Generator().manual_seed(19),
    )

    assert first.shape == (4, 3, 6)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_diffusion_checkpoint_payload_roundtrip_preserves_samples():
    config = LatentResidualDiffusionConfig(
        timesteps=6,
        hidden_dim=20,
        time_embedding_dim=8,
        num_layers=2,
        dropout=0.0,
    )
    original = ConditionalLatentResidualDiffusion(5, 9, config)
    restored = ConditionalLatentResidualDiffusion.from_checkpoint_payload(
        original.checkpoint_payload()
    )
    condition = torch.randn(2, 9)

    original_sample = original.sample(
        condition,
        num_samples=2,
        generator=torch.Generator().manual_seed(31),
    )
    restored_sample = restored.sample(
        condition,
        num_samples=2,
        generator=torch.Generator().manual_seed(31),
    )

    assert torch.equal(original_sample, restored_sample)


def test_diffusion_rejects_incompatible_condition_shape():
    model = ConditionalLatentResidualDiffusion(
        latent_dim=4,
        condition_dim=7,
        config=LatentResidualDiffusionConfig(timesteps=4, hidden_dim=16, time_embedding_dim=8),
    )

    with pytest.raises(ValueError, match="condition"):
        model.loss(torch.zeros(2, 4), torch.zeros(2, 6))


def test_teacher_latent_optimization_improves_masked_target_loss():
    torch.manual_seed(3)
    base_latent = torch.zeros(2, 3)
    target = torch.tensor(
        [
            [[1.0, -0.5, 0.25], [1.0, -0.5, 0.25]],
            [[-0.25, 0.75, 0.5], [-0.25, 0.75, 0.5]],
        ]
    )
    target_mask = torch.ones_like(target)

    def decode(latent: torch.Tensor) -> torch.Tensor:
        return latent.unsqueeze(1).expand(-1, 2, -1)

    result = optimize_latent_target(
        base_latent=base_latent,
        decode=decode,
        target=target,
        target_mask=target_mask,
        steps=40,
        learning_rate=0.08,
        regularization=0.05,
    )

    assert result.final_data_loss < result.initial_data_loss * 0.1
    assert result.optimized_latent.shape == base_latent.shape
    assert torch.allclose(
        result.delta_z,
        result.optimized_latent - base_latent,
    )
    assert result.rms_displacement > 0.0


def test_condition_builder_combines_latent_context_and_mask_geometry():
    torch.manual_seed(5)
    condition = build_latent_condition(
        mu=torch.randn(2, 4),
        logvar=torch.randn(2, 4),
        encoder_sequence=torch.randn(2, 6, 8),
        local_context=torch.randn(2, 3, 4),
        observation_mask=torch.ones(2, 8, 5),
        n_chem=2,
    )

    assert condition.shape == (2, 8 + 12 + 6 + 15)
    assert torch.isfinite(condition).all()


def test_heldout_like_mask_uses_synchronized_psd_blocks():
    observed = torch.ones(3, 12, 7)
    heldout = sample_heldout_like_mask(
        observed,
        n_chem=2,
        config=HeldoutLikeMaskConfig(
            target_ratio=1.0,
            mean_duration=4,
            std_duration=0,
            min_duration=4,
            max_duration=4,
            psd_block_prob=1.0,
            chem_block_prob=0.0,
        ),
    )

    psd = heldout[:, :, 2:]
    assert heldout.shape == observed.shape
    assert psd.any()
    assert torch.equal(psd, psd[:, :, :1].expand_as(psd))
    assert not heldout[:, :, :2].any()


def test_teacher_latent_optimization_requires_observed_teacher_targets():
    with pytest.raises(ValueError, match="target_mask"):
        optimize_latent_target(
            base_latent=torch.zeros(1, 2),
            decode=lambda latent: latent.unsqueeze(1),
            target=torch.zeros(1, 1, 2),
            target_mask=torch.zeros(1, 1, 2),
            steps=2,
        )
