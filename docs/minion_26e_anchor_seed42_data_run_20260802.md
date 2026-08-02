# Minion 26e regenerated anchor mask package

Date: 2026-08-02  
Branch: `main`  
Protocol: non-diffusion 26e configuration with a newly generated mask

## Package

The new package is:

`data/research_data/minion_2024_2025_26e_split_censored_anchor_seed42/`

It contains the 32 chemistry columns, 230 PSD bins, `AT/RH/WS/WD/BLH`, the
new mask, and the mask-column order file. The raw merged QC source is not
rewritten. The chemistry `_` payloads are retained and represented separately
in the model's censoring state.

The active run config
`examples/minion_26e_main_external_mask_config.json` now points to this new
package and its newly generated mask. The old external mask is no longer used
by that config.

## Mask rules

The rules match the prior 26e anchor-constrained protocol:

- mode: `anchor_constrained`;
- mask seed: `42`;
- held-out ratio: `0.10`;
- gap duration: Normal(48, 24), clipped to 3–168 hours;
- chemistry: sampled independently by feature;
- PSD: sampled by common time gaps, masking all 230 bins together;
- every selected gap is bounded by observed anchors.

The eligible pool is the current post-censoring `OBSERVED` state. Thus natural
missing chemistry and chemistry censored cells are excluded from mask
selection. PSD zeros remain ordinary observed values because PSD has no MDL
threshold in this protocol.

## Generated result

| quantity | value |
|---|---:|
| rows | 15,336 |
| target shape | `(15336, 262)` |
| chemistry mask cells | 38,516 |
| PSD mask cells | 175,260 |
| PSD masked timesteps | 762 |
| total mask cells | 213,776 |
| natural-missing overlap | 0 |
| censored overlap | 0 |
| PSD zero cells | 80,797 |
| PSD zero cells classified censored | 0 |

Current full target state counts are:

- missing: 1,825,760;
- observed: 2,149,746;
- censored: 42,526.

The output processing summary records the source hashes and the complete mask
generation parameters:

`data/research_data/minion_2024_2025_26e_split_censored_anchor_seed42/processing_summary_26e_with_blh.json`

## Validation

- The new mask has zero overlap with natural missing or censored cells.
- The mask has shape `(15336, 262)` and 213,776 selected cells.
- The PSD mask is shared across all PSD bins at each selected timestep.
- PSD zero values remain non-censored.
- Focused tests: 127 passed.

This package is a new mask benchmark on the corrected data; its mask cells
should not be compared as if they were byte-identical to the old external
seed-42 artifact.
