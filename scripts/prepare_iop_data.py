"""Normalize the raw IOP CSVs into the schema the multimodal loader expects.

Fixes applied:
  * strip the UTF-8 BOM and whitespace from column names
  * rename every timestamp column to ``time`` with an ISO-8601 value
  * de-duplicate repeated column names (chemistry carries ``SO2`` twice:
    the gas analyzer channel and the MARGA channel)
  * verify the 1 h grid is strict, contiguous, and shared by all modalities

Usage:
    python scripts/prepare_iop_data.py data/ data/iop_clean/
"""
import sys
from pathlib import Path

import pandas as pd

SOURCES = {
    "chem": "iop_chem.csv",
    "psd": "iop_psd.csv",
    "met": "iop_met.csv",
}
# Second occurrence of a repeated name -> explicit name.
RENAMES = {"chem": {"SO2": "SO2_marga"}}
# Names that clash across modalities; every column must be globally unique.
# ``P`` is phosphorus in chemistry and barometric pressure in meteorology.
CROSS_MODALITY_RENAMES = {"met": {"P": "Pressure"}}


def dedupe(columns, overrides):
    seen, out = {}, []
    for col in columns:
        n = seen.get(col, 0)
        seen[col] = n + 1
        out.append(col if n == 0 else overrides.get(col, f"{col}__{n}"))
    return out


def load(path, overrides, cross_renames):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    frame.columns = dedupe(frame.columns, overrides)
    frame = frame.rename(columns={frame.columns[0]: "time"})
    frame = frame.rename(columns=cross_renames)
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.sort_values("time").reset_index(drop=True)


def main(src_dir, out_dir):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        key: load(src_dir / name, RENAMES.get(key, {}), CROSS_MODALITY_RENAMES.get(key, {}))
        for key, name in SOURCES.items()
    }

    grids = {k: f["time"] for k, f in frames.items()}
    reference = grids["psd"]
    steps = reference.diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta("1h"):
        raise SystemExit(f"psd time grid is not a strict 1h grid: {steps}")
    for key, grid in grids.items():
        if not grid.equals(reference):
            raise SystemExit(f"{key} time grid does not match psd")

    for key, frame in frames.items():
        target = out_dir / f"{key}.csv"
        frame.to_csv(target, index=False, date_format="%Y-%m-%d %H:%M:%S")
        gap = frame.drop(columns="time").isna().mean().mean() * 100
        print(f"{target}: {frame.shape[0]} rows x {frame.shape[1] - 1} cols, {gap:.2f}% missing")

    print(f"time range: {reference.iloc[0]} .. {reference.iloc[-1]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data",
         sys.argv[2] if len(sys.argv) > 2 else "data/iop_clean")
