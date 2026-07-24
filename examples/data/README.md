# Nanzi demo data

`nanzi_demo_96h.csv` is a compact 96-hour excerpt from the aligned Nanzi aerosol supersite record used to demonstrate the public package workflow.

It contains:

- four composition targets: `SO2`, `NO2`, `PM2.5`, `PM10`;
- four representative PSD diameter bins: `11.8`, `100.313...`, `491.388...`, `19810.0`;
- five conditioning variables: `AT`, `RH`, `BLH`, `hour_sin`, `hour_cos`;
- natural missing values, including synchronized PSD gaps.

The excerpt is intentionally small so a user can validate the schema, run a short training job, save a checkpoint bundle, and execute imputation without downloading the full research dataset.

This file is a workflow demonstration, not a benchmark dataset. A model trained on 96 hours is not scientifically adequate for aerosol reconstruction, uncertainty calibration, or comparison with the reported 26e research results.

## Multimodal interface example

`multimodal_demo/` contains the same public input contract in three deliberately tiny files:

- `chemistry.csv`: named chemical targets;
- `psd.csv`: PSD targets whose column names are particle diameters in nm;
- `meteorology.csv`: auxiliary conditioning variables.

The files are used by `examples/multimodal_train_config.example.json` to demonstrate automatic column discovery, PSD diameter sorting, per-modality preprocessing, and versioned schema persistence. They contain only 24 rows and are strictly an interface smoke test.
