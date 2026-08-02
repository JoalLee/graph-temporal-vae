import numpy as np
import pandas as pd

from graph_temporal_vae.censoring import CensoringConfig
from scripts.prepare_minion_26e_inputs import _generate_anchor_mask


def test_generated_anchor_mask_includes_present_censored_cells_but_not_missing():
    n_rows = 240
    time = pd.date_range("2024-02-01", periods=n_rows, freq="h")
    chemistry = pd.DataFrame({
        "time": time,
        "Al": np.ones(n_rows),
        "K": np.ones(n_rows, dtype=object),
    })
    chemistry.loc[20:24, "Al"] = 0.0
    # Put the marker in the deterministic seed-42 selected eligible run so
    # this test checks that `_` cells are genuinely allowed into HO.
    chemistry.loc[114:118, "K"] = "0.05_"
    chemistry.loc[80:84, "Al"] = np.nan

    psd = pd.DataFrame({
        "time": time,
        "11.8": np.ones(n_rows),
        "19.8": np.ones(n_rows),
    })
    psd.loc[30:34, "11.8"] = 0.0
    psd.loc[30:34, "19.8"] = 0.0

    mask, mask_columns, diagnostics = _generate_anchor_mask(
        chemistry,
        psd,
        censoring=CensoringConfig(
            enabled=True,
            thresholds={"Al": 0.1, "K": 0.1},
            detect="zero",
            input_fill="half_threshold",
            loss="tobit",
        ),
        seed=42,
        ratio=0.1,
        mean_duration=12.0,
        std_duration=0.0,
        min_duration=3,
        max_duration=24,
        duration_source="parametric",
    )

    assert mask.shape == (n_rows, 4)
    assert mask_columns["target_col"].tolist() == ["Al", "K", "11.8", "19.8"]
    assert mask.any()
    assert diagnostics["natural_missing_overlap_cells"] == 0
    assert diagnostics["censored_overlap_cells"] > 0
    assert diagnostics["censored_overlap_cells"] <= diagnostics["marker_cells"] + 10
    assert diagnostics["psd_zero_cells"] == 10
    assert diagnostics["psd_zero_censored_cells"] == 0
