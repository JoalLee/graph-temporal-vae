#!/usr/bin/env python3
"""Convenience entry point for the latent/context bottleneck diagnostic."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "diagnose_latent_bottleneck.py"
    )
    runpy.run_path(str(script), run_name="__main__")
