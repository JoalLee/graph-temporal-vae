"""Build a detection-limit table for censored-likelihood training, and audit it.

Reads instrument specifications from an installed AeroViz, converts them into
the physical units of the target CSV, matches them against the actual column
names, and reports how well the declared limits explain the observed
non-detects.  The audit matters more than the extraction: a threshold that
disagrees with the data will make the Tobit term confidently wrong.

The emitted JSON is consumed by ``CensoringConfig.thresholds``; the package
itself never imports AeroViz.

Usage:
    python scripts/build_mdl_table.py --chem-csv data/iop_clean/chem.csv \
        -o examples/iop_mdl.json --report outputs/mdl_audit.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Xact 625i limits are published in ng/m3; the target CSVs are in ug/m3.
XRF_NG_PER_UG = 1000.0

# AeroViz applies these in OCEC._QC by setting at-or-below-limit values to NaN,
# so OC/EC non-detects arrive as missing rather than as zeros. They are listed
# for completeness and flagged in the audit rather than silently trusted.
OCEC_MDL_UG = {
    "Thermal_OC": 0.3,
    "Optical_OC": 0.3,
    "Thermal_EC": 0.015,
    "Optical_EC": 0.015,
}

# AeroViz names on the left, dataset column names on the right.
COLUMN_ALIASES = {
    "G-SO2": "SO2_marga",
}

# W is listed at 0.0001 ng/m3, three orders of magnitude below the next
# smallest element and far below anything the instrument resolves. Treated as
# a bad entry rather than a real limit; the fallback policy handles it.
SUSPECT_MDL = {"W": "AeroViz lists 0.0001 ng/m3, ~3 orders below any other element"}


def load_aeroviz_mdl(aeroviz_root, igac_source):
    """Extract IGAC and XRF limits without importing AeroViz's runtime deps."""
    config_path = Path(aeroviz_root) / "rawDataReader" / "config" / "supported_instruments.py"
    if not config_path.exists():
        raise SystemExit(f"AeroViz instrument config not found: {config_path}")
    namespace = {}
    exec(compile(config_path.read_text(), str(config_path), "exec"), namespace)
    meta = namespace["meta"]

    igac = {k: v for k, v in (meta["IGAC"].get("MDL") or {}).items() if v is not None}
    xrf_ng = {k: v for k, v in (meta["XRF"].get("MDL") or {}).items() if v is not None}
    xrf = {k: v / XRF_NG_PER_UG for k, v in xrf_ng.items()}

    if igac_source == "script":
        script_path = Path(aeroviz_root) / "rawDataReader" / "script" / "IGAC.py"
        igac = _parse_igac_script_mdl(script_path)
    return {"IGAC": igac, "XRF": xrf, "OCEC": dict(OCEC_MDL_UG)}


def _parse_igac_script_mdl(path):
    """Read the `_mdl` literal out of IGAC._QC without executing the module."""
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_mdl" for t in node.targets
        ):
            return {k.value: float(v.value) for k, v in zip(node.value.keys, node.value.values)}
    raise SystemExit(f"Could not find an `_mdl` mapping in {path}")


def build_table(frame, instrument_mdl, fallback):
    """Match limits to columns and classify how each one was resolved."""
    columns = [c for c in frame.columns if c != "time"]
    thresholds, entries = {}, []

    declared = {}
    for instrument, table in instrument_mdl.items():
        for name, value in table.items():
            column = COLUMN_ALIASES.get(name, name)
            if column in columns:
                declared.setdefault(column, (instrument, float(value)))

    for column in columns:
        values = frame[column]
        n_zero = int((values == 0).sum())
        positive = values[values > 0].dropna()
        source, threshold, note = None, None, ""

        if column in declared and column not in SUSPECT_MDL:
            source, threshold = declared[column]
        elif column in SUSPECT_MDL:
            note = SUSPECT_MDL[column]

        if threshold is None and n_zero > 0 and fallback == "data-min" and len(positive):
            # With no trustworthy limit, the smallest detected value is the
            # tightest defensible upper bound for the censoring point.
            threshold = float(positive.min())
            source = "provisional:data-min"

        if threshold is not None:
            thresholds[column] = threshold

        entries.append({
            "column": column,
            "threshold": threshold,
            "source": source or "none",
            "note": note,
            "n_zero": n_zero,
            "zero_fraction": float((values == 0).mean()),
            "nan_fraction": float(values.isna().mean()),
            "positive_below_threshold_fraction": (
                float((positive < threshold).mean()) if threshold is not None and len(positive) else 0.0
            ),
            "min_positive": float(positive.min()) if len(positive) else None,
            "median_positive": float(positive.median()) if len(positive) else None,
        })
    return thresholds, entries


def write_report(entries, path, igac_source):
    censored = [e for e in entries if e["n_zero"] > 0]
    censored.sort(key=lambda e: -e["zero_fraction"])
    covered = [e for e in censored if e["threshold"] is not None]
    uncovered = [e for e in censored if e["threshold"] is None]
    # A limit that also sits above most reported positives is not the rule that
    # produced the zeros, so the Tobit constraint would be too loose.
    inconsistent = [e for e in covered if e["positive_below_threshold_fraction"] > 0.5]

    lines = [
        "# Detection-limit audit",
        "",
        f"- IGAC source: `{igac_source}`",
        f"- Columns with at least one non-detect: **{len(censored)}**",
        f"- Covered by a threshold: **{len(covered)}**; uncovered: **{len(uncovered)}**",
        f"- Thresholds inconsistent with the reported positives: **{len(inconsistent)}**",
        "",
        "## Columns with non-detects",
        "",
        "| column | zero% | NaN% | threshold | source | pos<thr% | min pos |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in censored:
        thr = f"{e['threshold']:.6g}" if e["threshold"] is not None else "—"
        mp = f"{e['min_positive']:.4g}" if e["min_positive"] is not None else "—"
        lines.append(
            f"| {e['column']} | {e['zero_fraction'] * 100:.1f}% | {e['nan_fraction'] * 100:.1f}% "
            f"| {thr} | {e['source']} | {e['positive_below_threshold_fraction'] * 100:.0f}% | {mp} |"
        )

    if inconsistent:
        lines += [
            "",
            "## Thresholds that do not explain the zeros",
            "",
            "More than half of the *reported positive* values sit below the declared limit, so",
            "the limit is a nominal instrument specification rather than the rule that produced",
            "the zeros. The censored term is still valid, but its bound is loose: reported",
            "sub-limit positives are treated as ordinary observations.",
            "",
        ]
        lines += [
            f"- `{e['column']}`: {e['positive_below_threshold_fraction'] * 100:.0f}% of positives "
            f"below {e['threshold']:.6g}"
            for e in inconsistent
        ]

    notes = [e for e in entries if e["note"]]
    if notes:
        lines += ["", "## Rejected declared limits", ""]
        lines += [f"- `{e['column']}`: {e['note']}" for e in notes]

    lines += [
        "",
        "## Caveats",
        "",
        "- OC/EC: AeroViz's `OCEC._QC` sets at-or-below-limit values to NaN, so those",
        "  non-detects arrive as *missing* and cannot be recovered from the CSV. Any OC/EC",
        "  threshold here therefore has nothing to act on.",
        "- Ions: AeroViz's `IGAC._QC` substitutes sub-limit ions with MDL/2. No such spike",
        "  appears in this data, so the ion columns did not pass through that QC step.",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
    return {"censored": len(censored), "covered": len(covered),
            "uncovered": len(uncovered), "inconsistent": len(inconsistent)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chem-csv", required=True)
    parser.add_argument(
        "--aeroviz-root",
        default=str(Path(sys.prefix) / "lib" / "python3.12" / "site-packages" / "AeroViz"),
        help="Installed AeroViz package directory.",
    )
    parser.add_argument(
        "--igac-source", choices=["central", "script"], default="central",
        help="'central' reads config/supported_instruments.py; 'script' reads the "
             "different limits hardcoded in rawDataReader/script/IGAC.py.",
    )
    parser.add_argument(
        "--fallback", choices=["data-min", "none"], default="data-min",
        help="How to threshold a column that has non-detects but no trustworthy limit.",
    )
    parser.add_argument("-o", "--output", required=True, help="Threshold JSON path.")
    parser.add_argument("--report", default=None, help="Markdown audit report path.")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.chem_csv)
    instrument_mdl = load_aeroviz_mdl(args.aeroviz_root, args.igac_source)
    thresholds, entries = build_table(frame, instrument_mdl, args.fallback)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(thresholds)} thresholds to {args.output}")

    if args.report:
        stats = write_report(entries, args.report, args.igac_source)
        print(f"wrote audit to {args.report}: {stats}")


if __name__ == "__main__":
    main()
