"""Prepare the raw Chem/PSD/BLH files with the 26e reference aux schema.

The private 26e experiment does not train directly from the three source
files.  It derives wind components, selects the NZ boundary-layer-height
series, and appends six cyclic time features before scaling.  This helper
creates one raw CSV with that same column contract; the public trainer should
then be called with ``--target-transform log1p``.

Example:
    python examples/prepare_26e_input.py \
        --chem data/chem_2024_2025_clean.csv \
        --psd data/psd_2024_2025.csv \
        --blh data/blh.csv \
        -o data/experiment_input_raw_26e.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CHEM_COLS = [
    "SO2", "NO", "NO2", "CO", "O3", "K", "Ca", "Ti", "V", "Cr",
    "Al", "Si", "Mn", "Fe", "Ni", "Cu", "Zn", "As", "Se", "Br",
    "Ba", "Pb", "OC", "EC", "Na+", "NH4+", "Cl-", "NO2-", "NO3-",
    "SO42-", "PM2.5", "PM10",
]
AUX_COLS = ["AT", "RH", "wind_u", "wind_v", "BLH"]
TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]


def _read_indexed(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    frame = frame[~frame.index.duplicated(keep="first")]
    return frame.sort_index()


def prepare_26e_input(chem_path: str | Path, psd_path: str | Path, blh_path: str | Path) -> pd.DataFrame:
    """Return raw target values plus the exact 26e 11-column aux schema."""
    chem = _read_indexed(chem_path)
    psd = _read_indexed(psd_path)
    blh = _read_indexed(blh_path)

    missing_chem = [col for col in CHEM_COLS + ["AT", "RH", "WS", "WD"] if col not in chem]
    if missing_chem:
        raise ValueError(f"Chem/meteorology file is missing columns: {missing_chem}")
    if not psd.columns.tolist():
        raise ValueError("PSD file contains no target columns")

    # Match train_ablation.py: negative Chem readings are treated as missing
    # before the log1p transform is applied in the reference pipeline.
    chem = chem.copy()
    chem[CHEM_COLS] = chem[CHEM_COLS].where(chem[CHEM_COLS] >= 0)

    met = chem[["AT", "RH", "WS", "WD"]].copy()
    wd_rad = np.radians((270.0 - met["WD"]) % 360.0)
    met["wind_u"] = met["WS"] * np.cos(wd_rad)
    met["wind_v"] = met["WS"] * np.sin(wd_rad)
    met = met[["AT", "RH", "wind_u", "wind_v"]]

    station = next((col for col in blh.columns if col.lower() == "nz"), None)
    if station is None:
        raise ValueError(f"BLH file must contain the NZ station column; found {list(blh.columns)}")
    met["BLH"] = pd.to_numeric(blh[station], errors="coerce")
    met = met[AUX_COLS]

    index = chem.index.union(psd.index).union(met.index).sort_values()
    chem = chem.reindex(index)
    psd = psd.reindex(index)
    met = met.reindex(index)

    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    month_phase = (index.month.to_numpy() - 1) / 12.0
    time = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month_phase),
            "month_cos": np.cos(2 * np.pi * month_phase),
        },
        index=index,
    )
    return pd.concat([chem[CHEM_COLS], psd, met, time], axis=1).reset_index(names="time")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chem", required=True)
    parser.add_argument("--psd", required=True)
    parser.add_argument("--blh", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = prepare_26e_input(args.chem, args.psd, args.blh)
    frame.to_csv(output, index=False)
    print(f"wrote {output}: {len(frame)} rows, {len(frame.columns) - 1} data columns")
    print(f"targets={len(CHEM_COLS) + len(pd.read_csv(args.psd, nrows=0).columns) - 1}, aux={AUX_COLS + TIME_COLS}")


if __name__ == "__main__":
    main()
