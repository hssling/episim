"""Regression-discontinuity simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.core import seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n: int = 2500,
    cutoff: float = 0.0,
    treatment_effect: float = -0.35,
    bandwidth: float = 0.45,
) -> Study:
    """Simulate a sharp regression-discontinuity design with local linear fit."""
    with seed(seed_value) as rng:
        running = rng.uniform(-1, 1, n)
        treatment = (running >= cutoff).astype(int)
        centered = running - cutoff
        outcome = (
            0.55
            + 0.45 * centered
            - 0.20 * centered**2
            + treatment_effect * treatment
            + rng.normal(0, 0.18, n)
        )
        df = pd.DataFrame(
            {
                "id": np.arange(n),
                "running": running,
                "centered_running": centered,
                "treatment": treatment,
                "outcome": outcome,
            }
        )
        local = df.loc[df["centered_running"].abs() <= bandwidth].copy()
        x = np.column_stack(
            [
                np.ones(len(local)),
                local["treatment"].to_numpy(),
                local["centered_running"].to_numpy(),
                (local["treatment"] * local["centered_running"]).to_numpy(),
            ]
        )
        beta = np.linalg.lstsq(x, local["outcome"].to_numpy(), rcond=None)[0]
        side_summary = (
            local.assign(side=np.where(local["treatment"] == 1, "right", "left"))
            .groupby("side", as_index=False)
            .agg(n=("outcome", "size"), mean_outcome=("outcome", "mean"))
        )
        left_mean = float(
            side_summary.loc[side_summary["side"] == "left", "mean_outcome"].iloc[0]
        )
        right_mean = float(
            side_summary.loc[side_summary["side"] == "right", "mean_outcome"].iloc[0]
        )
        return build_study(
            design="regression_discontinuity",
            seed_value=seed_value,
            data=df,
            params={
                "n": n,
                "cutoff": cutoff,
                "bandwidth": bandwidth,
                "treatment_effect": treatment_effect,
            },
            results={
                "local_n": int(len(local)),
                "estimated_jump": round(float(beta[1]), 3),
                "left_mean_outcome": round(left_mean, 3),
                "right_mean_outcome": round(right_mean, 3),
            },
            artifacts={"local_bandwidth_sample": local, "side_summary": side_summary},
        )
