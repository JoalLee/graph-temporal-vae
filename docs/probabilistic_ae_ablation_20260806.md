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
- The existing VAE control's persisted `config.used.json` has
  `censoring.enabled = false`; the PAE treatment is explicitly pinned to the
  same value. The source config currently says `true`, so copying it without
  this correction would not be a matched experiment.

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

## GB10 smoke record

The committed implementation was synced to
`/home/jhenyulee/project/graph-temporal-vae` without replacing the remote
`train.py`: its pre-existing changes were preserved and only commit
`7712e3f`'s PAE patch was applied. A one-epoch small-model smoke then exercised
the real Aeroviz-QC Minion inputs and fixed seed-42 held-out mask.

This first smoke used the source config's `censoring.enabled = true` before the
persisted VAE control was compared. It remains valid as an implementation-path
check, but it is not the matched treatment and must not be used in the VAE/PAE
quality comparison. The final treatment config is pinned to `false` above.

```text
output: runs/minion_pae_smoke_20260806/model.pt
rows: 15,336
targets: 262 (32 chemistry + 230 PSD)
fixed global-HO cells: 219,090
natural-missing overlap: 0
censored overlap: 8
latent_mode: deterministic
kl_beta: 0.0
train_kl: 0.0
train_weighted_kl: 0.0
train_loss: 150972.234375
train_recon: 150972.234375
validation HO MSE: 0.980582594871521
```

Artifact hashes:

```text
3307638ea10415374422ebca4dc8e4c55ce3cb066b7bc4ffebe3d14673478322  model.pt
32b51d7d23e1472021dc21099062b0bd624cb2b0d10c84f110179102e04232d0  model_history.csv
```

The smoke MSE is not a model-quality result: the network was reduced to
342,826 parameters and trained for one epoch. It only validates the remote
data/mask/training/bundle path and the deterministic-latent loss invariants.

## Matched full run

Before launch, the resolved PAE config was compared against
`runs/aeroviz_qc_own_source_20260805/config.used.json`. After removing only
the PAE treatment field, the comparison returned `top_diff=[]` and
`model_diff=[]`.

The matched full run was launched on GB10 at:

```text
runs/aeroviz_qc_probabilistic_ae_20260806/
initial main PID: 1281952
parameters: 17,131,120
epochs configured: 2,000 (early stopping patience: 100)
GPU memory at launch: 13,123 MiB
```

Initial log sanity check:

```text
epoch 1: val_ho_mse=1.2704, KL=0.000/0.000
epoch 2: val_ho_mse=1.2045, KL=0.000/0.000
epoch 3: val_ho_mse=1.1429, KL=0.000/0.000
```

These early values only confirm that optimization is progressing and the
zero-KL treatment is active. Final model-quality conclusions require the
selected checkpoint and held-out/physical-QC evaluation.

## Completed-run inspection

The formal run stopped normally at epoch 785 by early stopping and saved the
best MSE checkpoint from epoch 685. The matched VAE control stopped at epoch
942 and selected epoch 842. Both bundles contain 262 targets, 219,090 held-out
cells, the same Aeroviz-QC data package, and the same seed-42 external mask.

The seed-42 mask is reused for checkpoint selection, so the following table is
diagnostic rather than an independent generalization estimate:

| held-out metric | PAE | VAE control | PAE − VAE |
|---|---:|---:|---:|
| Overall R² | 0.602893 | 0.603731 | −0.000838 |
| Overall MAE | 968.385 | 960.103 | +8.282 |
| Overall RMSE | 1793.447 | 1780.417 | +13.030 |
| Overall CRPS | 800.158 | 793.987 | +6.171 |
| Overall PICP | 56.142% | 53.450% | +2.692 pp |
| Chemistry R² | 0.595998 | 0.580401 | +0.015597 |
| Chemistry model-space CRPS | 0.218973 | 0.222480 | −0.003506 |
| PSD R² | 0.603852 | 0.606977 | −0.003125 |
| PSD CRPS | 884.084 | 876.899 | +7.184 |

The PAE therefore does not improve the overall accuracy objective. Its small
Chemistry gain is offset by slightly worse PSD and overall physical-scale
errors. PICP improves, but 56.1% remains far below the nominal 90% interval.
At the selected checkpoint, `val_ho_z2` is 13.55 for PAE and 16.35 for VAE;
both are far above the calibrated value 1. The deterministic latent reduces
the severity but does not solve the under-dispersion.

To reduce the risk of reading the seed-42 result as a selection artifact, the
same two bundles were evaluated with a newly generated
`anchor_constrained` mask (`seed=43`, ratio 0.1, 219,090 cells, 10 MC
samples):

| seed-43 held-out metric | PAE | VAE control | PAE − VAE |
|---|---:|---:|---:|
| Overall R² | 0.251358 | 0.312228 | −0.060869 |
| Overall MAE | 1351.649 | 1208.661 | +142.988 |
| Overall RMSE | 2476.124 | 2294.564 | +181.560 |
| Overall CRPS | 1030.273 | 939.809 | +90.463 |
| Overall PICP | 66.268% | 66.257% | +0.012 pp |
| Chemistry R² | 0.341243 | 0.367213 | −0.025970 |
| PSD R² | 0.238853 | 0.304578 | −0.065724 |

This independent-style check points in the opposite direction: PAE is worse
overall, in Chemistry, and especially in PSD. Therefore this single matched
PAE arm should remain an ablation/baseline; it is not a justified replacement
for the VAE.

Basic physical-output checks found zero negative `physical_pred_mean`,
`physical_q025`, or `physical_q975` values for both runs. This is only a
non-negativity check. Because the matched configs intentionally have
`censoring.enabled=false`, the held-out evaluation has zero censored cells and
does not establish MDL, mass-closure, or ion-charge-balance compliance. The
intervals also have extreme right tails: the PAE's maximum physical `q975` is
about `4.47e8`, so improved PICP must not be interpreted as a physically
credible uncertainty solution.

Conclusion: the implementation is functioning, but deterministic latent alone
does not resolve the original model problem. The remaining evidence points
more strongly to decoder variance/calibration, observed-versus-held-out shift,
and objective/data-mask formulation than to latent sampling as the dominant
cause.

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
