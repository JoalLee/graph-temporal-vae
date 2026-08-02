#!/usr/bin/env python3
"""Prepare the three multimodal 26e input packages and its fixed HO mask.

This keeps the QC chemistry/meteorology source untouched. BLH is taken from
the research repository's ``NZ`` station column and joined on the hourly
timestamp. The output meteorology order is intentional: AT, RH, WS, WD, BLH;
the graph package then canonicalizes WS/WD to wind_u/wind_v in that order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_START = "2024-02-01 00:00:00"
DEFAULT_END = "2025-10-31 23:00:00"
CHEM_NAME = "chemistry_26e_core_20240201_20251031.csv"
PSD_NAME = "psd_26e_full_20240201_20251031.csv"
MET_NAME = "meteorology_26e_at_rh_wind_blh_20240201_20251031.csv"
MASK_NAME = "heldout_mask_full_seed42.npy"
MASK_COLUMNS_NAME = "heldout_mask_full_columns_seed42.csv"
MINION_26E_CHEMISTRY_COLUMNS = [
    "SO2", "NO", "NO2", "CO", "O3", "K", "Ca", "Ti", "V", "Cr",
    "Al", "Si", "Mn", "Fe", "Ni", "Cu", "Zn", "As", "Se", "Br",
    "Ba", "Pb", "OC", "EC", "Na+", "NH4+", "Cl-", "NO2-", "NO3-",
    "SO42-", "PM2.5", "PM10",
]
_NUMERIC_MARKER_RE = re.compile(
    r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?_\s*$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chemistry",
        type=Path,
        help="Already split chemistry CSV with a lowercase 'time' column.",
    )
    parser.add_argument(
        "--meteorology",
        type=Path,
        help="Already split meteorology CSV with a lowercase 'time' column.",
    )
    parser.add_argument(
        "--merged-qc",
        type=Path,
        help=(
            "Raw merged Minion QC CSV. Its Time column and metadata row are "
            "normalized, and the 26e chemistry/meteorology columns are extracted "
            "without stripping numeric '_' markers."
        ),
    )
    parser.add_argument("--psd", type=Path, required=True)
    parser.add_argument("--blh", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--mask-columns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blh-station", default="NZ")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    return parser.parse_args()


def _read_time_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    time_columns = [column for column in frame.columns if str(column).casefold() == "time"]
    if len(time_columns) != 1:
        raise ValueError(f"{path} must contain a 'time' column")
    frame = frame.rename(columns={time_columns[0]: "time"})
    parsed_time = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.loc[parsed_time.notna()].copy()
    frame["time"] = parsed_time.loc[frame.index]
    if frame["time"].duplicated().any():
        raise ValueError(f"{path} contains duplicate timestamps")
    return frame.sort_values("time").reset_index(drop=True)


def _extract_merged_qc(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the 26e Chem and meteorology roles from the raw merged file."""
    frame = _read_time_csv(path)
    required = [*MINION_26E_CHEMISTRY_COLUMNS, "AT", "RH", "WS", "WD"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required merged-QC columns: {missing}")
    chemistry = frame[["time", *MINION_26E_CHEMISTRY_COLUMNS]].copy()
    meteorology = frame[["time", "AT", "RH", "WS", "WD"]].copy()
    return chemistry, meteorology


def _marker_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for column in frame.columns:
        if column == "time":
            continue
        values = frame[column].astype("string")
        marker = values.str.fullmatch(_NUMERIC_MARKER_RE).fillna(False)
        if int(marker.sum()):
            counts[column] = int(marker.sum())
    return counts


def _restrict(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, label: str):
    result = frame.loc[frame["time"].between(start, end)].copy()
    expected = pd.date_range(start, end, freq="h")
    actual = pd.DatetimeIndex(result["time"])
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        raise ValueError(
            f"{label} is not the exact hourly requested range: "
            f"rows={len(result)}, missing={len(missing)}, off_range={len(extra)}"
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare(args: argparse.Namespace) -> dict:
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        raise ValueError("--end must not precede --start")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merged_qc is not None:
        if args.chemistry is not None or args.meteorology is not None:
            raise ValueError("Use --merged-qc or --chemistry/--meteorology, not both")
        raw_chemistry, raw_meteorology = _extract_merged_qc(args.merged_qc)
        chemistry = _restrict(raw_chemistry, start, end, "chemistry")
        meteorology = _restrict(raw_meteorology, start, end, "meteorology")
    else:
        if args.chemistry is None or args.meteorology is None:
            raise ValueError(
                "Provide --merged-qc, or provide both --chemistry and --meteorology"
            )
        chemistry = _restrict(_read_time_csv(args.chemistry), start, end, "chemistry")
        meteorology = _restrict(
            _read_time_csv(args.meteorology), start, end, "meteorology"
        )
    psd = _restrict(_read_time_csv(args.psd), start, end, "PSD")
    blh = _restrict(_read_time_csv(args.blh), start, end, "BLH")

    station_lookup = {str(column).casefold(): column for column in blh.columns}
    station_column = station_lookup.get(str(args.blh_station).casefold())
    if station_column is None:
        raise ValueError(
            f"BLH station {args.blh_station!r} not found; "
            f"available={list(blh.columns)}"
        )
    blh_values = pd.to_numeric(blh[station_column], errors="coerce")
    if blh_values.isna().any():
        raise ValueError(
            f"BLH station {station_column!r} contains {int(blh_values.isna().sum())} missing values"
        )
    blh_series = pd.DataFrame({"time": blh["time"], "BLH": blh_values})
    meteorology = meteorology.merge(blh_series, on="time", how="left", validate="one_to_one")
    if meteorology["BLH"].isna().any():
        raise ValueError("BLH join left missing values in the requested meteorology range")

    required_met = ["AT", "RH", "WS", "WD", "BLH"]
    missing_met = [column for column in required_met if column not in meteorology]
    if missing_met:
        raise ValueError(f"meteorology is missing required columns: {missing_met}")
    meteorology = meteorology[["time", *required_met]]

    mask = np.load(args.mask, allow_pickle=False)
    target_columns = chemistry.columns[1:].tolist() + psd.columns[1:].tolist()
    expected_mask_shape = (len(chemistry), len(target_columns))
    if mask.ndim != 2 or tuple(mask.shape) != expected_mask_shape:
        raise ValueError(
            "heldout mask shape does not match the prepared target grid: "
            f"expected={expected_mask_shape}, got={mask.shape}"
        )
    mask_columns = pd.read_csv(args.mask_columns)
    if list(mask_columns.columns) != ["target_col"]:
        raise ValueError("mask columns file must contain exactly 'target_col'")
    if mask_columns["target_col"].astype(str).tolist() != [str(c) for c in target_columns]:
        raise ValueError("mask columns do not match chemistry + PSD output order")

    outputs = {
        "chemistry": args.output_dir / CHEM_NAME,
        "psd": args.output_dir / PSD_NAME,
        "meteorology": args.output_dir / MET_NAME,
        "mask": args.output_dir / MASK_NAME,
        "mask_columns": args.output_dir / MASK_COLUMNS_NAME,
    }
    chemistry.to_csv(outputs["chemistry"], index=False)
    psd.to_csv(outputs["psd"], index=False)
    meteorology.to_csv(outputs["meteorology"], index=False)
    np.save(outputs["mask"], mask)
    mask_columns.to_csv(outputs["mask_columns"], index=False)

    summary = {
        "time_start": start.isoformat(sep=" "),
        "time_end": end.isoformat(sep=" "),
        "rows": len(chemistry),
        "chemistry_columns": chemistry.columns[1:].tolist(),
        "psd_columns": psd.columns[1:].tolist(),
        "meteorology_columns": meteorology.columns[1:].tolist(),
        "blh_source": str(args.blh),
        "blh_station_source_column": str(station_column),
        "chemistry_marker_policy": {
            "suffix": "_",
            "scope": "chemistry_only",
            "preserve_numeric_payload": True,
            "marker_cells": int(sum(_marker_counts(chemistry).values())),
            "marker_cells_by_column": _marker_counts(chemistry),
        },
        "mask_shape": list(mask.shape),
        "mask_cells": int(mask.sum()),
        "files": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in outputs.items()
        },
    }
    summary_path = args.output_dir / "processing_summary_26e_with_blh.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = _parse_args()
    summary = _prepare(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
