"""Survival-analysis simulation with Cox-like hazard-ratio summary."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.core import seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n: int = 3000,
    hazard_ratio: float = 0.70,
    censoring_time: float = 5.0,
) -> Study:
    """Simulate time-to-event data and estimate a person-time hazard ratio."""
    with seed(seed_value) as rng:
        age = rng.normal(64, 9, n)
        treatment = rng.binomial(1, 0.5, n)
        base_rate = 0.12 * np.exp(0.025 * (age - 64))
        rate = base_rate * np.where(treatment == 1, hazard_ratio, 1.0)
        event_time = rng.exponential(1 / rate)
        time = np.minimum(event_time, censoring_time)
        event = (event_time <= censoring_time).astype(int)
        df = pd.DataFrame(
            {
                "id": np.arange(n),
                "age": age,
                "treatment": treatment,
                "time": time,
                "event": event,
            }
        )
        treated = df["treatment"] == 1
        h_t = df.loc[treated, "event"].sum() / df.loc[treated, "time"].sum()
        h_c = df.loc[~treated, "event"].sum() / df.loc[~treated, "time"].sum()
        survival_summary = (
            df.groupby("treatment", as_index=False)
            .agg(
                events=("event", "sum"),
                person_time=("time", "sum"),
                median_time=("time", "median"),
            )
        )
        return build_study(
            design="survival_cox",
            seed_value=seed_value,
            data=df,
            params={
                "n": n,
                "hazard_ratio": hazard_ratio,
                "censoring_time": censoring_time,
            },
            results={
                "estimated_hazard_ratio": round(float(h_t / h_c), 3),
                "event_fraction": round(float(df["event"].mean()), 3),
                "treated_event_rate": round(float(df.loc[treated, "event"].mean()), 3),
                "control_event_rate": round(
                    float(df.loc[~treated, "event"].mean()), 3
                ),
            },
            artifacts={"survival_summary": survival_summary},
        )
