"""Stepped-wedge cluster trial simulation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from episim.core import logistic, seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n_clusters: int = 12,
    n_periods: int = 6,
    persons_per_cluster_period: int = 50,
    treatment_effect: float = -0.60,
    secular_trend: float = -0.06,
) -> Study:
    """Simulate a stepped-wedge rollout with repeated cross-sections."""
    with seed(seed_value) as rng:
        rollout = {
            int(cluster): int(1 + cluster % (n_periods - 1))
            for cluster in rng.permutation(np.arange(n_clusters))
        }
        cluster_effects = rng.normal(0, 0.35, n_clusters)
        rows: list[dict[str, Any]] = []
        for cluster in range(n_clusters):
            for period in range(n_periods):
                treatment = int(period >= rollout[cluster])
                for person in range(persons_per_cluster_period):
                    rows.append(
                        {
                            "id": int(
                                (cluster * n_periods + period)
                                * persons_per_cluster_period
                                + person
                            ),
                            "cluster": cluster,
                            "period": period,
                            "treatment": treatment,
                            "cluster_effect": float(cluster_effects[cluster]),
                            "baseline_risk": float(rng.normal(0, 1)),
                        }
                    )
        df = pd.DataFrame(rows)
        df = logistic(
            df,
            "event ~ treatment + period + cluster_effect + baseline_risk",
            betas={
                "treatment": treatment_effect,
                "period": secular_trend,
                "cluster_effect": 1.0,
                "baseline_risk": 0.35,
            },
            intercept=-1.7,
            rng=rng,
        )
        cluster_period = (
            df.groupby(["cluster", "period", "treatment"], as_index=False)
            .agg(n=("event", "size"), event_rate=("event", "mean"))
            .sort_values(["cluster", "period"])
        )
        treated_rate = float(df.loc[df["treatment"] == 1, "event"].mean())
        control_rate = float(df.loc[df["treatment"] == 0, "event"].mean())
        return build_study(
            design="stepped_wedge",
            seed_value=seed_value,
            data=df,
            params={
                "n_clusters": n_clusters,
                "n_periods": n_periods,
                "persons_per_cluster_period": persons_per_cluster_period,
                "treatment_effect": treatment_effect,
                "secular_trend": secular_trend,
            },
            results={
                "n_total": int(len(df)),
                "treated_event_rate": round(treated_rate, 3),
                "control_event_rate": round(control_rate, 3),
                "marginal_risk_difference": round(treated_rate - control_rate, 3),
                "rollout_period_min": min(rollout.values()),
                "rollout_period_max": max(rollout.values()),
            },
            artifacts={"cluster_period_summary": cluster_period},
        )
