# IOP PSD/QC model-diagnosis progress

Date: 2026-08-01  
Status: paused by user before the next experiment was evaluated

## Scope and provenance

- The target artifact is the remote run at `/home/jhenyulee/project/graph-temporal-vae/outputs/iop_psdqc_20260729`.
- The current local worktree is `main` at commit `c2ac9ce` and was clean when this note was written.
- The diffusion worktree is separate: `exp/psd-residual-diffusion` at `36ab2c3`. Its earlier diagnosis and ablation record are in `experiments/iop_psdqc_20260729_model_diagnosis.md` in that worktree; no diffusion-branch changes were made during this resumed step.
- Post-hoc correction has not been selected as a solution. It remains a possible baseline only.

## Confirmed evidence

1. The aggregate result hides a chemistry-specific failure. In the seed-7 final-block evaluation (`full_beta_feature_seed7`), physical-space chemistry held-out performance was R² `0.5108`, raw mean R² `0.4886`, and raw minimum R² `-0.5140`, while PSD raw mean R² was `0.7349`. Examples of poor chemistry features included Pd (`-0.514`), Ca2+ (`-0.298`), and Cl- (`-0.172`).
2. This is not explained solely by the number of positive observations. Several features with many positive observations still showed weak point performance or low predictions; for example Fe and Al had approximately `0.015` and `0.017` held-out R² in the inspected feature table. The current evidence therefore points to an objective/data-imbalance and representation problem, not a proven implementation bug.
3. The primary PICP is computed from q05–q95, so its nominal coverage is 90%, not 95%. The observed aggregate values around 91–94% are not evidence of undercoverage against a 95% target. Interval width and per-feature calibration still require stratified analysis.
4. The sparse robust-scale fallback plus target-feature KL/anchor-constrained candidate removed the extreme raw minimum seen in the original candidate (approximately `-20.3` to `-0.514` in the inspected seed-7 comparison), but chemistry remained materially weaker than PSD. This is useful evidence, not yet a sufficient multi-seed solution.

## Rejected or unresolved interventions

- `family_loss_scale="target_dim"` with family-balanced reconstruction was rejected as a solution. Although its seed-7 aggregate R² was `0.7216` and PSD raw mean R² was `0.7824`, chemistry raw mean R² fell to `0.3878`; the training objective also became strongly negative and validation latent variance grew to an implausible scale, indicating a variance-collapse/overconfidence pathology.
- A follow-up run was launched with the same family-balanced setup but `family_loss_scale="mean"` and the old-runtime equivalent of target-feature KL (`kl_max_beta=1/304`). It was intended to test whether objective-scale alignment removes the preceding pathology. The checkpoint/evaluation result was deliberately not collected before pausing.

## Next resumption point

1. Confirm or terminate the outstanding remote run before starting another experiment; do not treat it as evidence until its checkpoint, history, fixed-protocol held-out metrics, and per-feature table are archived.
2. If the stable family-mean result is promising, repeat it on at least two additional seeds under the same evaluator and compare chemistry, PSD, raw-minimum, CRPS, PICP90, interval width, and per-feature bias.
3. Port only evidence-supported changes into the active implementation with test-first coverage: explicit KL normalization semantics, stable family-loss scaling, and the sparse robust-scale fallback. Keep defaults unchanged until paired multi-seed evidence supports changing them.
4. Commit each logical implementation/evidence boundary and update this record with the exact command, checkpoint, evaluator JSON, and acceptance/rejection decision.

