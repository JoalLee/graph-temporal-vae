# Latent Bottleneck Diagnosis: `26e_maskdist_heldout_like_gate_none`

Date: 2026-07-25

## Question

Does the current VAE lose imputation quality because the masked-input posterior
`q(z | x_obs, mask, context)` cannot locate a sufficiently informative latent,
and would conditional latent diffusion therefore be a justified next step?

## Checkpoint and environment

- Legacy repository: `/home/jhenyulee/project/Imputation_VAE`
- Checkpoint: `result/26e_component_ablation/26e_maskdist_heldout_like_gate_none/best_model.pth`
- Remote host: `gb10-jylee` (`NVIDIA GB10`)
- Targets: 32 Chem + 230 PSD
- Window / stride: 168 h / 24 h
- Latent dimension: 256
- Latent flow: RealNVP enabled
- Evaluation: official `heldout_mask.npy`

The diagnostic implementation is `experiments/diagnose_latent_bottleneck.py`.
It does not retrain or modify the checkpoint.

## Validation of the diagnostic harness

The manually separated encoder/decoder path reproduced
`model.forward(sample_latent=False)` with maximum absolute difference `0.0`.

The deterministic overlap-aggregated baseline also closely reproduced the
official held-out point metrics:

| Metric | Diagnostic | Official artifact |
| --- | ---: | ---: |
| Chem MAE | 0.2827 | 0.2862 |
| PSD MAE | 0.3302 | 0.3309 |

The full diagnostic covered 620 windows and 212,577 held-out points.

## Intervention 1: latent/context swaps

Each window was encoded twice:

1. full-observation input;
2. official held-out input.

The decoder was then evaluated with swapped latent and encoder-context sources.
All results below are standardized-scale PSD metrics.

| Variant | PSD MAE | PSD R² | Interpretation |
| --- | ---: | ---: | --- |
| masked latent + masked context | 0.3302 | 0.751 | deterministic baseline |
| full latent + masked context | 0.3571 | 0.691 | direct full-posterior swap degrades |
| masked latent + full context | 0.3446 | 0.732 | direct context swap degrades |
| full latent + full context | 0.3822 | 0.651 | observed-domain reconstruction is not an oracle |
| midpoint latent + masked context | 0.3342 | 0.738 | interpolation does not improve aggregate MAE |
| zero latent + masked context | 0.6725 | 0.075 | latent path is essential |
| batch-shuffled latent + masked context | 0.3434 | 0.722 | correct sample-specific latent matters |

### Interpretation boundary

The full-input posterior is jointly produced with an observed-domain encoder
context. It is therefore not a compatible oracle for the masked-domain decoder
context. Its failure to improve does **not** prove that latent modeling is
unimportant. It shows that latent and encoder context are coupled and that a
naive `q(z | x_full)` transplant is invalid as a performance upper bound.

The zero-latent control does establish that the decoder strongly uses the latent
path. The decoder is not ignoring `z`.

## Posterior displacement

Across all 620 windows:

- all 256 latent units were active under the chosen variance threshold;
- median posterior standard deviation was approximately 0.135–0.136;
- mean pre-flow posterior-mean distance per dimension was 0.187;
- mean post-flow latent distance per dimension was 0.390;
- PSD maximum gap length had Spearman correlation 0.493 with posterior-mean
  displacement and 0.486 with symmetric posterior KL.

Longer PSD gaps therefore move the inferred latent farther from the
full-observation representation. This is consistent with increasing latent
ambiguity, although it does not by itself prove that the posterior family is the
main performance bottleneck.

The posterior median standard deviation is close to `exp(-2) = 0.1353`, the
standard deviation implied by the configured `logvar=-4` lower bound. This
suggests a narrow posterior, but variance saturation must not be equated with a
causal bottleneck without the latent-capacity intervention below.

## Intervention 2: truth-optimized latent upper bound

To test decoder latent capacity directly, the masked encoder context was fixed
and the **pre-flow latent only** was optimized against held-out PSD truth. Model
parameters remained frozen. This is a diagnostic upper bound, not a deployable
imputation procedure.

The main comparison used 64 time-stratified windows containing PSD held-out
blocks, 20 Adam steps, learning rate 0.03, and several penalties on displacement
from the masked posterior mean.

| Latent penalty | RMS latent displacement | PSD MAE | PSD RMSE | PSD R² | Top-decile bias |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0 | 0.3309 | 0.4780 | 0.7500 | -0.6545 |
| 1.0 | 0.105 | 0.3150 | 0.4506 | 0.7778 | -0.5866 |
| 0.1 | 0.322 | 0.2980 | 0.4252 | 0.8022 | -0.5427 |
| 0.01 | 0.459 | 0.2956 | 0.4218 | 0.8054 | -0.5412 |

For penalty 0.01, gap-specific PSD MAE changed as follows:

| Gap duration | Baseline | Optimized latent |
| --- | ---: | ---: |
| 24–48 h | 0.3235 | 0.2885 |
| 48–96 h | 0.3187 | 0.2913 |
| 96–168 h | 0.3976 | 0.3286 |

Even the strongly regularized latent displacement of 0.105—smaller than the
median posterior standard deviation—provided a measurable PSD gain. Therefore,
the current decoder contains useful latent directions that the masked posterior
does not consistently select.

## What the latent correction changes

The optimized latent mainly improved broad regime and pointwise intensity:

- aggregate MAE, RMSE, R², top-decile bias, and long-gap error improved;
- median PSD spectrum correlation was essentially unchanged;
- median peak-diameter error was essentially unchanged;
- the dedicated per-spectrum peak-amplitude ratio did **not** improve: its
  median changed from 0.911 to 0.893, despite the improvement in pointwise
  top-decile bias.

This distinction indicates that a global 256-dimensional latent can correct
broad concentration regime and event intensity without materially repairing
particle-size modal geometry. Fine PSD shape and modal peak amplitude are still
likely controlled by encoder sequence/local context and the output decoder
structure.

## Current diagnosis

1. **Posterior collapse / latent non-use is not the main issue.** Zeroing the
   latent severely degrades the model, and all latent units vary across windows.
2. **Naively replacing the Gaussian posterior with samples from a more flexible
   distribution is not sufficient.** Arbitrary or full-domain latent swaps are
   incompatible with the masked encoder context and degrade reconstruction.
3. **There is real latent headroom.** A small, context-specific correction of the
   masked posterior mean materially improves PSD reconstruction, especially for
   long gaps and high values.
4. **The remaining PSD shape limitation is not primarily global-latent.** The
   latent correction improves amplitude more than modal geometry.

## Implication for diffusion design

The evidence supports testing **conditional latent residual diffusion**, not an
unconditional diffusion prior and not an immediate full decoder replacement.
The target should be a context-compatible correction:

`delta_z = z_better - mu_masked`

conditioned on the masked encoder representation, mask geometry, gap length,
Chem, meteorology, and local temporal context.

The diffusion branch should initially retain the existing decoder and compare:

1. deterministic masked posterior mean;
2. a simple Gaussian/MLP latent-residual predictor;
3. conditional latent residual diffusion;
4. PSD output-residual diffusion as a separate decoder-structure control.

Conditional latent diffusion is justified only if it can infer useful residuals
from observed context without access to held-out truth and improves empirical
CRPS / coverage or PSD joint structure across repeated masks. The truth-optimized
result proves decoder capacity, not deployable predictability.
