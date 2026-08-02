#!/usr/bin/env python3
"""Prepare the three multimodal 26e input packages and its fixed HO mask.

This keeps the QC chemistry/meteorology source untouched. BLH is taken from
the research repository's ``NZ`` station column and joined on the hourly
timestamp. The output meteorology order is intentional: AT, RH, WS, WD, BLH;
the graph package then canonicalizes WS/WD to wind_u/wind_v in that order.

The mask may either be copied from an existing artifact or generated from the
prepared data with the historical 26e anchor-constrained protocol. Generated
masks use every present target cell, including chemistry non-detect markers;
natural missing cells cannot become held-out targets. Censored cells are
scored with their interval constraint rather than as exact scalar values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_temporal_vae.censoring import (  # noqa: E402
    CensoringConfig,
    STATE_CENSORED,
    STATE_MISSING,
    STATE_OBSERVED,
    build_state_matrix,
)
from graph_temporal_vae.contracts import DataSchema  # noqa: E402
from graph_temporal_vae.data import sample_anchor_constrained_heldout_mask  # noqa: E402


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
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--mask-columns", type=Path)
    parser.add_argument(
        "--generate-anchor-mask",
        action="store_true",
        help="Generate a new seedable anchor-constrained mask from this data.",
    )
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--mask-ratio", type=float, default=0.10)
    parser.add_argument("--mask-mean-duration", type=float, default=48.0)
    parser.add_argument("--mask-std-duration", type=float, default=24.0)
    parser.add_argument("--mask-min-duration", type=int, default=3)
    parser.add_argument("--mask-max-duration", type=int, default=168)
    parser.add_argument(
        "--mask-duration-source",
        choices=["parametric", "empirical"],
        default="parametric",
    )
    parser.add_argument(
        "--censoring-config",
        type=Path,
        help=(
            "JSON containing either a top-level 'censoring' object or a "
            "CensoringConfig mapping; required for generated masks."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blh-station", default="NZ")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    args = parser.parse_args()
    has_external_mask = args.mask is not None or args.mask_columns is not None
    if args.generate_anchor_mask and has_external_mask:
        parser.error("Use --generate-anchor-mask or --mask/--mask-columns, not both")
    if args.generate_anchor_mask and args.censoring_config is None:
        parser.error("--generate-anchor-mask requires --censoring-config")
    if not args.generate_anchor_mask and (
        args.mask is None or args.mask_columns is None
    ):
        parser.error(
            "Provide both --mask and --mask-columns, or use --generate-anchor-mask"
        )
    return args


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


def _numeric_chemistry_and_markers(
    chemistry: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Parse chemistry payloads while retaining the source marker mask."""
    chemistry_columns = chemistry.columns[1:].tolist()
    numeric = chemistry[["time", *chemistry_columns]].copy()
    markers = np.zeros((len(chemistry), len(chemistry_columns)), dtype=bool)
    for index, column in enumerate(chemistry_columns):
        values = chemistry[column].astype("string")
        marker = values.str.fullmatch(_NUMERIC_MARKER_RE).fillna(False)
        markers[:, index] = marker.to_numpy()
        payload = values.where(
            ~marker,
            values.str.replace(r"_\s*$", "", regex=True),
        )
        numeric[column] = pd.to_numeric(payload, errors="coerce")
    return numeric, markers


def _load_censoring_config(path: Path) -> CensoringConfig:
    payload = json.loads(path.read_text())
    if "censoring" in payload:
        payload = payload["censoring"]
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object for censoring")
    config = CensoringConfig.from_dict(payload)
    if not config.active:
        raise ValueError(
            f"{path} must enable censoring with at least one threshold to generate a mask"
        )
    return config


def _generate_anchor_mask(
    chemistry: pd.DataFrame,
    psd: pd.DataFrame,
    *,
    censoring: CensoringConfig,
    seed: int,
    ratio: float,
    mean_duration: float,
    std_duration: float,
    min_duration: int,
    max_duration: int,
    duration_source: str,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Generate the historical anchor mask from the current source state."""
    if not 0 < ratio <= 1:
        raise ValueError("mask ratio must be in (0, 1]")
    if min_duration < 1 or max_duration < min_duration:
        raise ValueError("mask duration bounds are invalid")
    numeric_chemistry, chemistry_markers = _numeric_chemistry_and_markers(chemistry)
    numeric_psd = psd.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    target_columns = chemistry.columns[1:].tolist() + psd.columns[1:].tolist()
    raw_targets = np.column_stack(
        [
            numeric_chemistry.iloc[:, 1:].to_numpy(dtype=float),
            numeric_psd.to_numpy(dtype=float),
        ]
    )
    marker_mask = np.zeros_like(raw_targets, dtype=bool)
    marker_mask[:, : len(chemistry.columns) - 1] = chemistry_markers
    schema = DataSchema(
        chemistry_cols=chemistry.columns[1:].tolist(),
        psd_cols=psd.columns[1:].tolist(),
        frequency="h",
        time_grid_policy="strict",
    )
    state = build_state_matrix(raw_targets, schema, censoring, marker_mask)
    # A numeric ``_`` marker is a present chemistry observation with an upper
    # bound, not natural missingness. It is therefore eligible for the fixed
    # HO mask, while the later evaluator must keep it out of exact-value
    # metrics and score it with the MDL interval instead.
    eligible_mask = state != STATE_MISSING
    heldout = sample_anchor_constrained_heldout_mask(
        eligible_mask,
        ratio=ratio,
        seed=seed,
        n_chem=len(schema.chemistry_cols),
        mean_duration=mean_duration,
        std_duration=std_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        duration_source=duration_source,
    )
    mask_columns = pd.DataFrame({"target_col": target_columns})
    psd_values = numeric_psd.to_numpy(dtype=float)
    psd_zero = np.isfinite(psd_values) & (psd_values == 0.0)
    diagnostics = {
        "source": "current_prepared_data",
        "mode": "anchor_constrained",
        "seed": int(seed),
        "ratio": float(ratio),
        "mean_duration": float(mean_duration),
        "std_duration": float(std_duration),
        "min_duration": int(min_duration),
        "max_duration": int(max_duration),
        "duration_source": duration_source,
        "n_chem": len(schema.chemistry_cols),
        "n_psd": len(schema.psd_cols),
        "state_counts": {
            "missing": int((state == STATE_MISSING).sum()),
            "observed": int((state == STATE_OBSERVED).sum()),
            "censored": int((state == STATE_CENSORED).sum()),
        },
        "marker_cells": int(chemistry_markers.sum()),
        "psd_zero_cells": int(psd_zero.sum()),
        "psd_zero_censored_cells": int(
            ((state[:, len(schema.chemistry_cols):] == STATE_CENSORED) & psd_zero).sum()
        ),
        "mask_cells": int(heldout.sum()),
        "natural_missing_overlap_cells": int(
            (heldout & (state == STATE_MISSING)).sum()
        ),
        "censored_overlap_cells": int(
            (heldout & (state == STATE_CENSORED)).sum()
        ),
        "chem_mask_cells": int(heldout[:, : schema.n_chem].sum()),
        "psd_mask_cells": int(heldout[:, schema.n_chem :].sum()),
        "psd_mask_timesteps": int(heldout[:, schema.n_chem :].any(axis=1).sum()),
    }
    return heldout, mask_columns, diagnostics


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

    target_columns = chemistry.columns[1:].tolist() + psd.columns[1:].tolist()
    if args.generate_anchor_mask:
        censoring = _load_censoring_config(args.censoring_config)
        mask, mask_columns, mask_generation = _generate_anchor_mask(
            chemistry,
            psd,
            censoring=censoring,
            seed=args.mask_seed,
            ratio=args.mask_ratio,
            mean_duration=args.mask_mean_duration,
            std_duration=args.mask_std_duration,
            min_duration=args.mask_min_duration,
            max_duration=args.mask_max_duration,
            duration_source=args.mask_duration_source,
        )
        mask_generation["censoring_config"] = str(args.censoring_config)
    else:
        mask = np.load(args.mask, allow_pickle=False)
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
        mask_generation = {
            "source": str(args.mask),
            "mode": "external",
            "seed": None,
            "ratio": None,
        }

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
        "mask_generation": mask_generation,
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
