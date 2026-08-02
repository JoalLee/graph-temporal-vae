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

Training has not been launched at the time of this commit. After launch, add
the exact PID, start timestamp, command, log path, and final checkpoint status
here.
