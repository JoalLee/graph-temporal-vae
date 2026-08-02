# Minion 26e anchor mask with censored held-out cells

Date: 2026-08-02  
Code commit: `8014c47`  
Protocol: non-diffusion 26e configuration with a regenerated seed-42 mask

## Decision

Chemistry values carrying a numeric trailing `_` are present observations
with an upper bound at the MDL. They are therefore eligible for held-out
selection, but they are not exact scalar targets. The exact-value metrics
(R², MAE, RMSE, CRPS, and PICP) use only `OBSERVED` held-out cells. Censored
held-out cells use the Tobit interval likelihood and report
`P(Y <= MDL)`/interval NLL separately.

The final imputation path uses the existing censor-aware truncated predictive
mean for a known censored cell. This is not a generic clamp of all outputs:
ordinary observed and missing predictions are not changed by this constraint.

## Package

The active package is:

`data/research_data/minion_2024_2025_26e_split_censored_anchor_seed42_with_censored_ho/`

The previous `..._anchor_seed42/` package is retained as the earlier
exclusion-based benchmark and is no longer referenced by the active config.
The three CSV contents are unchanged from the previous package; the mask and
processing summary are new.

## Mask protocol

- mode: `anchor_constrained`;
- seed: `42`;
- held-out ratio: `0.10`;
- gap duration: Normal(48, 24), clipped to 3–168 hours;
- chemistry: sampled independently by feature;
- PSD: sampled by common time gaps, masking all 230 bins together;
- natural missing cells: ineligible;
- chemistry censored cells: eligible, but scored as intervals;
- PSD zeros: ordinary observed values, not censored.

## Generated result

| quantity | value |
|---|---:|
| rows | 15,336 |
| target shape | `(15336, 262)` |
| chemistry mask cells | 42,766 |
| PSD mask cells | 175,260 |
| PSD masked timesteps | 762 |
| total mask cells | 218,026 |
| natural-missing overlap | 0 |
| censored overlap | 4,116 |
| PSD zero cells | 80,797 |
| PSD zero cells classified censored | 0 |

Full target state counts remain:

- missing: 1,825,760;
- observed: 2,149,746;
- censored: 42,526.

The output summary contains all source/mask hashes and generation parameters:

`data/research_data/minion_2024_2025_26e_split_censored_anchor_seed42_with_censored_ho/processing_summary_26e_with_blh.json`

## Validation record

- Focused censoring/data-pipeline/train-inference tests: `128 passed`.
- The new fixed-mask test verifies that a selected censored cell is removed
  from the input and ordinary censor loss, exposed through
  `heldout_censor_mask`, and not counted in exact-value `heldout_mask`.
- The generated mask has zero natural-missing overlap and zero PSD-zero
  censoring overlap.
