# IOP `log1p` ablation (2026-08-06)

## Question

Does removing the Chemistry/PSD `log1p` input and output transforms reduce the
physical-space high-value underestimation observed in the IOP model?

This is a preprocessing/objective ablation. It is not a post-hoc correction.

## Matched runs

| arm | config | GB10 output |
|---|---|---|
| `log1p` control | `examples/iop_train_config_realnvp_fb03_psdqc.json` | `/home/jhenyulee/project/graph-temporal-vae/outputs/iop_psdqc_log1p_matched_20260806` |
| `none` treatment | `examples/iop_train_config_realnvp_fb03_psdqc_nolog.json` | `/home/jhenyulee/project/graph-temporal-vae/outputs/iop_psdqc_nolog_20260806` |

The treatment changes only:

```text
preprocessing.chemistry.transform        log1p -> none
preprocessing.chemistry.output_transform log1p -> none
preprocessing.psd.transform              log1p -> none
preprocessing.psd.output_transform      log1p -> none
```

Held fixed: the three IOP input files, full-data scaler fit, censoring/Tobit
thresholds, seed 0, model architecture, RealNVP, optimizer, loss weights,
dynamic masking, and training schedule. Both runs used the current
`run_iop.py` held-out protocol: `anchor_constrained`, ratio `0.15`, seed
`100003`, 50 MC samples, stride 8.

The no-log1p config was added in commit `4b944f7`:
`exp: add IOP no-log1p ablation config`.

Both runs validated the same 672-row IOP period and produced the same held-out
key set: 28,448 rows, consisting of 26,274 exact observed cells and 2,174
censored cells. The old `outputs/iop_psdqc_20260729` artifact was not used as
the control for the final comparison because it was generated before the
current held-out protocol and had a different held-out row count.

## Training selection diagnostics

| metric | `log1p` | `none` |
|---|---:|---:|
| best global-HO `val_ho_mse` | 1.5066 | 1.6495 |
| best epoch | 138 | 164 |

These values are not comparable as physical accuracy: the two arms optimize
different transformed target geometries. They only show that `log1p` is an
easier optimization scale for this configuration.

## Official held-out metrics

The package metrics include interval scoring for censored cells. Lower is
better for MAE, RMSE, and CRPS; higher is better for R² and PICP.

| metric | `log1p` | `none` | `none - log1p` |
|---|---:|---:|---:|
| Overall R² | 0.6642 | 0.6833 | +0.0190 |
| Overall MAE | 539.50 | 501.31 | −38.18 |
| Overall RMSE | 987.33 | 1008.46 | +21.14 |
| Overall CRPS | 451.00 | 416.18 | −34.82 |
| Overall PICP | 94.15% | 93.99% | −0.16 pp |
| Chemistry R² | 0.3968 | 0.4331 | +0.0363 |
| Chemistry MAE | 0.3391 | 0.3025 | −0.0366 |
| Chemistry CRPS | 0.2640 | 0.2399 | −0.0241 |
| PSD R² | 0.7386 | 0.7529 | +0.0142 |
| PSD MAE | 689.52 | 640.73 | −48.79 |
| PSD CRPS | 576.43 | 531.92 | −44.50 |

No-log1p improves the main R²/MAE/CRPS metrics in this IOP seed, but its RMSE
is worse. This means it reduces the typical error while leaving larger
occasional errors. PICP is nearly unchanged and remains only a calibration
diagnostic, not evidence of physical validity.

## Physical-space bias and high-value tail

These statistics use exact observed held-out cells only; censored cells are not
treated as point truths. The high-value group is the top 10% of observed
values within the reported family.

| diagnostic | `log1p` | `none` |
|---|---:|---:|
| Overall mean(pred)/mean(obs) | 0.9165 | 0.9733 |
| Overall top-10% ratio | 0.8763 | 0.9371 |
| Overall top-10% under-rate | 79.07% | 45.93% |
| Chemistry mean ratio | 0.9399 | 0.9565 |
| Chemistry top-10% ratio | 0.9101 | 0.9381 |
| PSD mean ratio | 0.9165 | 0.9733 |
| PSD top-10% ratio | 0.8681 | 0.9263 |
| PSD top-10% under-rate | 81.54% | 48.80% |

The log1p arm has an almost matched mean in its transformed model space
(overall model-space mean ratio about `0.998`), but its physical-space ratio is
only `0.916`. This is consistent with the existing direct inverse-transform
contract: a central prediction in log space is not the arithmetic mean after
`expm1`. Removing log1p reduces this compression, especially for PSD's upper
tail.

## Physical-domain failure of raw-space output

The `none` arm uses an unconstrained heteroscedastic likelihood in raw units.
It therefore does not preserve non-negativity:

| output/state | negative mean | negative lower interval | negative upper interval |
|---|---:|---:|---:|
| censored Chemistry | 5,666 / 14,895 (38.04%) | 12,501 (83.93%) | 722 (4.85%) |
| missing Chemistry | 92 / 3,907 (2.35%) | 1,072 (27.44%) | 17 (0.44%) |
| missing PSD | 496 / 9,002 (5.51%) | 2,399 (26.65%) | 0 (0.00%) |

For held-out exact observed predictions, the `none` arm also had negative means
in 1.48% of Chemistry rows and 5.65% of PSD rows; the log1p arm had zero
negative means or intervals. The run summary still reports zero predictions
*above* the MDL because negative values are below the limit; that check does
not establish non-negativity.

## Conclusion

`log1p` has a material effect. It improves positivity and interval validity but
compresses high physical values; this is a shared physical-output/objective
issue, not evidence that the data's PSD zeros are invalid.

Simply removing `log1p` is not the final solution: it improves typical accuracy
and high-tail bias in this IOP seed, but violates the physical non-negativity
rule and increases RMSE. A simple output clip would be a post-hoc baseline,
not the architectural solution. The next principled comparison should use a
positivity-compatible raw-scale likelihood/output parameterization, or a
transform-aware objective that evaluates the physical arithmetic expectation
without treating clipping as the fix. This single IOP seed is evidence for the
direction, not sufficient multi-seed final-model evidence.
