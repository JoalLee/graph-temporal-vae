# Minion 26e external-mask provenance audit

Date: 2026-08-02  
Branch: `main`  
Scope: non-diffusion Minion 26e preparation and seed-42 held-out mask

## Finding

The current `anchor_constrained` sampler does not create the observed
natural-missing overlap reported for the supplied seed-42 mask. When the
sampler is run on the corrected current data's observed-state matrix, the
generated mask contains 213,776 cells and has zero overlap with either
natural-missing or censored cells.

The supplied
`data/research_data/minion_2024_2025_26e_split_censored/heldout_mask_full_seed42.npy`
is an external fixed benchmark artifact. Because
`selection_mask_path` is configured, training loads that matrix verbatim;
`selection_mask_mode: "anchor_constrained"` is not used to regenerate or
repair it. The seed-42 matrix is preserved byte-for-byte as requested.

## Cell-level comparison

The old prepared split converted numeric QC markers such as `52.4_` to NaN.
The corrected split preserves the numeric payload and passes a chemistry-only
marker mask into censoring. The source rows and target-column order are equal
between the two splits, and the external mask has 212,577 requested cells.

| External-mask cell state | Old split | Corrected split |
|---|---:|---:|
| observed | 211,882 | 211,286 |
| natural missing | 695 | 668 |
| censored | not represented | 623 |
| total | 212,577 | 212,577 |

The transition table explains the change:

- 668 old-missing cells remain natural missing;
- 27 old-missing cells are now censored because they carry a numeric `_`
  marker; and
- 596 old-observed cells are exact-zero chemistry values classified as
  censored under the configured IOP MDL handling.

All 668 natural-missing overlaps and all 623 censored overlaps are chemistry
cells. PSD zeros do not enter the censoring state and have zero overlap in
this audit.

## Interpretation and protocol

The 668 cells demonstrate that the external mask is not a subset of the
current data's observed cells. The mask file alone does not establish whether
it was generated from an older data/QC version or by a different procedure;
the confirmed fact is the provenance mismatch. Calling this artifact a
current-data anchored mask would therefore be inaccurate.

For exact seed-42 benchmark reproducibility, keep the external matrix and
report its natural-missing/censored overlap. For a strict anchored-mask
experiment, regenerate a new mask from the corrected observed-state matrix;
that is a different benchmark and must not overwrite the supplied seed-42
artifact.

## Checks

- `pytest -q tests/test_censoring.py tests/test_data_pipeline.py tests/test_train_infer_cli.py`: 126 passed.
- Current-data sampler control: generated 213,776 cells, natural-missing overlap 0, censored overlap 0.
- External mask remains shape `(15336, 262)` and contains 212,577 requested cells.
