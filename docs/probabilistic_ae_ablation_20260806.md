# Probabilistic AE ablation (2026-08-06)

## Question

Does a deterministic encoder latent improve Minion imputation accuracy and
physical consistency when the decoder remains a heteroscedastic probabilistic
likelihood?

This is an architecture ablation, not a post-hoc correction. The
probabilistic AE (PAE) predicts a likelihood distribution exactly as the VAE
does, but it uses the encoder mean deterministically and removes the KL term.

## Matched comparison

- VAE control:
  `examples/minion_26e_aeroviz_qc_own_source_config.json`
- PAE treatment:
  `examples/minion_26e_aeroviz_qc_probabilistic_ae_config.json`
- Deliberate treatment difference:
  `model_kwargs.latent_mode = "deterministic"`
- Held fixed: input files, preprocessing, seed 42, fixed
  `anchor_constrained` selection mask, dynamic training-mask protocol,
  architecture width/depth, RealNVP transform, likelihood, optimizer, and
  training schedule.

RealNVP is retained in the first matched PAE arm as a deterministic latent
transform. Removing it would change two architectural factors at once and is
reserved for a later ablation if needed.

## Expected invariants

- Repeated forward passes have identical predictive means when dropout is off.
- Likelihood draws remain different, so predictive intervals still exist.
- `train_kl`, `train_weighted_kl`, and recorded `kl_beta` are all zero.
- Bundles persist and restore `latent_mode = "deterministic"`.
- Older bundles without `latent_mode` load as the existing variational mode.

## Local verification record

Core implementation commit: `7712e3f`.

```text
.venv/bin/python -m pytest -q \
  tests/test_model_forward.py \
  tests/test_split_module_parity.py \
  tests/test_train_infer_cli.py

69 passed in 7.85s
```

The integration test trains a one-epoch PAE, reloads its bundle, and verifies
the zero-KL history fields. A separate uncertainty test verifies identical
decoder means but distinct likelihood samples.

## Full-run commands

```bash
.venv/bin/python -m graph_temporal_vae.cli train \
  --train-config examples/minion_26e_aeroviz_qc_probabilistic_ae_config.json \
  --output runs/minion_26e_aeroviz_qc_probabilistic_ae_seed42/model.pt
```

The VAE control must be trained or selected from the exact control config and
same fixed mask. Compare overall, chemistry, and PSD held-out metrics
separately; report MSE/RMSE/R2 together with NLL, CRPS, PICP, interval width,
and the physical-QC violation counts. A single seed or one aggregate metric is
not enough to claim that PAE is better.
