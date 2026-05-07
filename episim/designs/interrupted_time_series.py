"""Interrupted time-series simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.core import seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n_periods: int = 60,
    intervention_period: int = 36,
    baseline_rate: float = 0.30,
    pre_slope: float = 0.002,
    level_change: float = -0.08,
    slope_change: float = -0.004,
    observations_per_period: int = 250,
) -> Study:
    """Simulate and fit a segmented interrupted time-series model."""
    with seed(seed_value) as rng:
        period = np.arange(n_periods)
        post = (period >= intervention_period).astype(int)
        time_after = np.maximum(0, period - intervention_period + 1)
        true_rate = (
            baseline_rate
            + pre_slope * period
            + level_change * post
            + slope_change * time_after
        )
        true_rate = np.clip(true_rate, 0.01, 0.95)
        events = rng.binomial(observations_per_period, true_rate)
        df = pd.DataFrame(
            {
                "period": period,
                "post": post,
                "time_after_intervention": time_after,
                "n": observations_per_period,
                "events": events,
                "rate": events / observations_per_period,
                "true_rate": true_rate,
            }
        )
        design = np.column_stack([np.ones(n_periods), period, post, time_after])
        beta = np.linalg.lstsq(design, df["rate"].to_numpy(), rcond=None)[0]
        return build_study(
            design="interrupted_time_series",
            seed_value=seed_value,
            data=df,
            params={
                "n_periods": n_periods,
                "intervention_period": intervention_period,
                "observations_per_period": observations_per_period,
            },
            results={
                "estimated_baseline_rate": round(float(beta[0]), 3),
                "estimated_pre_slope": round(float(beta[1]), 4),
                "estimated_level_change": round(float(beta[2]), 3),
                "estimated_slope_change": round(float(beta[3]), 4),
                "pre_mean_rate": round(float(df.loc[df["post"] == 0, "rate"].mean()), 3),
                "post_mean_rate": round(float(df.loc[df["post"] == 1, "rate"].mean()), 3),
            },
            artifacts={"time_series": df},
        )
