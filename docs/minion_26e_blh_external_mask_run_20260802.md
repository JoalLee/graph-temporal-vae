# Minion 26e BLH + external held-out-mask run

Date: 2026-08-02  
Branch: `main` (non-diffusion)  
Training source: `bdd29f6`  
Evaluator source: `9c81d15`

## Protocol

- Time range: `2024-02-01 00:00` through `2025-10-31 23:00`, 15,336 hourly rows.
- Targets: 32 chemistry variables plus 230 PSD bins.
- Meteorology: `AT`, `RH`, `WS`, `WD`, and BLH from `Imputation_VAE/data/train/blh.csv`, station `NZ`; the package input contract is `AT`, `RH`, `wind_u`, `wind_v`, `BLH`.
- Auxiliary input: 5 meteorology channels plus 6 cyclical time channels, 11 channels total; `cross_modal_query_gate_mode="none"` as requested, so auxiliary data is not disabled at low observation rate.
- Training: window 168, stride 24, batch 64, maximum 2,000 epochs, seed 42, `anchor_constrained`, Student-t NLL, 12:1 chemistry:PSD feature weighting, `shared_full_heldout_mask=true`.
- External mask: `heldout_mask_full_seed42.npy`, shape `(15336, 262)`, 212,577 requested cells. Of these, 211,882 overlap observed QC values and 695 overlap pre-existing natural missing values; the matrix was not rewritten or intersected before loading.
- Inference: 50 Monte Carlo samples, stride 24, same external mask and column-order file as training.

The held-out R²/MAE/CRPS/PICP values below are macro-averages over features. Held-out R² is clipped at zero per feature, matching the reference evaluator. `*_pooled` is shown separately because it is magnitude-dominated. PICP uses the 5th–95th percentile interval (nominal 90%). “Observed” is the model's raw prediction on non-held-out observed cells, not `impute()`'s observation restoration.

## Execution result

- Training early-stopped at epoch 371.
- Validation-best checkpoint: epoch 271, `val_ho_mse=0.20974764227867126`.
- Checkpoint: `/home/jhenyulee/project/graph-tcn-vae_runs/minion_26e_main_bdd29f6/outputs/minion_26e_main_external_mask/best_model.pt`
- Inference wrote 212,577 prediction rows; 211,882 had finite ground truth for scoring.
- Metrics: `/home/jhenyulee/project/graph-tcn-vae_runs/minion_26e_main_bdd29f6/outputs/minion_26e_main_external_mask/external_mask_eval/metrics.json`
- Per-feature diagnostics: `/home/jhenyulee/project/graph-tcn-vae_runs/minion_26e_main_bdd29f6/outputs/minion_26e_main_external_mask/external_mask_eval/per_feature_metrics.csv`

| family | held-out macro R² | held-out pooled R² | held-out PICP | observed macro R² | observed PICP |
|---|---:|---:|---:|---:|---:|
| Overall | 0.5814 | 0.6956 | 71.13% | 0.7136 | 98.24% |
| Chemistry | 0.5782 | 0.7700 | 78.51% | 0.5686 | 96.32% |
| PSD | 0.5819 | 0.6815 | 70.10% | 0.7338 | 98.51% |

Category held-out macro R²: chemistry gases 0.7375, ions 0.5617, carbon 0.7337, metals 0.4825, PM 0.8868; PSD nucleation 0.4167, Aitken 0.6305, accumulation 0.6976, coarse <2.5 μm 0.5346, coarse >2.5 μm 0.5217.

## Interpretation

1. The BLH and external-mask integration is working and is provenance-auditable. The 695 natural-missing overlaps are excluded from numeric scoring; they are not evidence that the QC file should be changed.
2. The main model issue is uncertainty transfer: held-out PICP is only 71.13% overall against a nominal 90%, while observed-point PICP is 98.24%. The model is over-covered on cells it sees but under-covered when the held-out mask removes information. This is not fixed by post-hoc output adjustment in this run; post-hoc remains a baseline only.
3. Macro versus pooled R² is materially different (chemistry 0.578 vs 0.770; PSD 0.582 vs 0.681). A pooled headline would hide weak features and should not be used as the sole claim.
4. The weakest chemistry features are Al (raw per-feature R² -0.259, clipped to 0 in the macro metric), Ba (0.100), and Na+ (0.157). The weakest PSD bins include 19,177.9 nm (0.188), 19,810 nm (0.190), and several 12.2–13.9 nm bins around 0.30–0.32. A mid-range PSD bin around 72.5 nm reaches 0.764.
5. The large-diameter zeros are present in the QC data as expected low counts (for example, zero fractions are about 0.69–0.71 at 19,178–19,810 nm). They were treated as valid target values; the low tail R² therefore points to a zero-inflated/low-signal modeling and metric difficulty, not a data-cleaning defect. Small-diameter bins also have large absolute errors, so the issue is not only large-particle scarcity.

The run does not establish a causal BLH improvement because there is no paired no-BLH control with the same data, mask, seed, and checkpoint protocol. It establishes that BLH is included correctly and gives a baseline for the next non-post-hoc model experiment: mask/gap-stratified evaluation plus an architecture/objective change aimed at missing-case uncertainty and feature-balanced learning.

## Tests and operational notes

- Full local suite after the external-mask evaluator change: `135 passed`.
- The first GB10 inference attempt exposed missing `scipy` and `scikit-learn` in the core-only environment; both were already declared under the repository's `dev` optional dependencies and were installed into the isolated run environment with uv.
- A retry initially used the pre-feature evaluator source and rejected the external-mask flags; the final inference used evaluator commit `9c81d15` and completed successfully.
