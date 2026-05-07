"""Kaggle-facing overview script for EPISIM."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_attached_dataset() -> Path:
    input_root = Path("/kaggle/input")
    matches = sorted(input_root.rglob("design_catalog.csv"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(
        "Could not locate design_catalog.csv under /kaggle/input. "
        "Attach the EPISIM Simulation Lab Demos dataset to this kernel."
    )


def main() -> None:
    try:
        from episim.lab import list_designs, run_design
    except ModuleNotFoundError:
        dataset_dir = _find_attached_dataset()
        catalog = pd.read_csv(dataset_dir / "design_catalog.csv")
        summary = pd.read_csv(dataset_dir / "summary_metrics.csv")
        print("EPISIM design catalog from attached Kaggle dataset")
        print(catalog[["key", "title", "family"]].to_string(index=False))
        print("\nAvailable summary metrics")
        print(summary.head(25).to_string(index=False))
        return

    print("EPISIM design catalog")
    for design in list_designs():
        print(f"- {design.key}: {design.title}")
    study = run_design("cross_sectional", seed_value=20260508, n=800)
    print("\nCross-sectional summary")
    for key, value in study.results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
