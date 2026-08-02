# Minion 26e GB10 training run

Date: 2026-08-02  
Protocol: non-diffusion mainline VAE, regenerated seed-42 anchor mask with
censored chemistry cells eligible for interval held-out evaluation

## Pre-launch provenance

- local source commit: `c75729ac21a792a422b9138b5fd3bd6268788016`
- remote run root:
  `/home/jhenyulee/project/graph-temporal-vae/runs/minion_26e_censored_ho_20260802/`
- remote source directory: `.../repo/`
- remote Python: parent `graph-temporal-vae/.venv/bin/python`
- remote environment: PyTorch `2.13.0+cu130`, CUDA available
- source files were synced from tracked files only; existing remote outputs were
  not overwritten

The six data/config artifacts were hash-checked locally and remotely. The
active data package is:

`data/research_data/minion_2024_2025_26e_split_censored_anchor_seed42_with_censored_ho/`

Important hashes:

| artifact | SHA-256 |
|---|---|
| chemistry CSV | `fd9c516e592cb51b34c99550ffe3f07b379979610d3b10abd6f50fcf153774bf` |
| PSD CSV | `c7e26704e993a08ac52b98043f9a9701d433717a9264c3f099aeaed50f0d2dcd` |
| meteorology CSV | `8310dbba20aff8cd4930d052e9b691bc7e4fa888dff32781423b000fa2b5103b` |
| seed-42 held-out mask | `7691d438444582dc71ce0fe8e3361dc7ae4945b244ade8c3f62eac3f87ee5810` |
| mask column order | `97c66c2acd929511c6f81d6c9e2c6950944f7e5f55c82d112d2df7e4e0ddba9c` |
| processing summary | `7b3bddd3893f804b984c71b1906d15b27ddc24398f278790da9c25241693edaa` |

Remote schema validation passed before launch. The resolved config is
`examples/minion_26e_main_external_mask_config.json`; it uses 2,000 epochs,
batch size 64, seed 42, `anchor_constrained`, external full-HO mask,
`censoring.loss=tobit`, chemistry feature weight 12, PSD feature weight 1,
and `aux_mask_channel=false`.

## Execution record

- launch PID: `3269013` (the process has exited normally)
- remote training log: `/home/jhenyulee/project/graph-temporal-vae/runs/minion_26e_censored_ho_20260802/train.log`
- checkpoint: `/home/jhenyulee/project/graph-temporal-vae/runs/minion_26e_censored_ho_20260802/minion_26e_censored_ho_seed42.pt`
- checkpoint SHA-256: `a715461dcd6315b7696a44e33194cf7b245290ffca5837b60736e48f4d7e99da`
- history CSV SHA-256: `b2f395dc240e8f97727ace99ac643dad4aa14d2d2441e97f057a3db496374771`
- completed by early stopping at epoch `645/2000`
- selected epoch: `545`, metric `val_ho_mse=0.2467405647`

The remote bundle inspection passed. It records 218,026 global held-out
cells, 213,910 exact observed cells, 4,116 censored chemistry cells, and zero
natural-missing overlap.

## Same-mask held-out evaluation

Evaluation output:

`/home/jhenyulee/project/graph-temporal-vae/runs/minion_26e_censored_ho_20260802/heldout_metrics.json`

The evaluation replayed the exact external mask with 20 MC samples. The
held-out set is the same global mask used for checkpoint selection, so this is
an auditable validation/diagnostic result, not an independent test estimate.
The exact-value metrics exclude the 4,116 censored cells.

| scope | held-out n | pooled R² | pooled MAE | PICP (nominal 90%) |
|---|---:|---:|---:|---:|
| overall exact observed | 213,910 | 0.6904 | 894.98 | 53.29% |
| chemistry exact observed | 38,650 | 0.5850 | 12.43 | 65.34% |
| PSD exact observed | 175,260 | 0.6771 | 1,089.61 | 51.62% |

Censored chemistry held-out cells: `n=4,116`, mean `P(Y <= MDL)=0.5127`,
raw predictive mean above MDL rate `0.5015`, and constrained predictive mean
above MDL rate `0.0`. The last number only confirms the censor-aware output
constraint; it is not evidence that the interval probabilities are calibrated.

## Interpretation

The run is operationally successful and the new censored-HO protocol is
working. It does not yet solve the modeling problem: exact-value held-out
accuracy is moderate, while the 90% predictive intervals cover only about
half of the held-out exact values. At the selected MSE epoch the training
history also reports `val_ho_z2=10.823`, consistent with severe
under-dispersion. The next architecture/objective experiment should therefore
target uncertainty calibration and feature-family imbalance; post-hoc output
conditioning remains a reporting constraint/baseline, not the proposed fix.
