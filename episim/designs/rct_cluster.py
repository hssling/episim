"""Cluster-randomized trial simulation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from episim.analytics import risk_difference, risk_ratio
from episim.core import logistic, seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n_clusters: int = 30,
    cluster_size: int = 80,
    treatment_effect: float = -0.55,
    cluster_sd: float = 0.45,
) -> Study:
    """Simulate a parallel cluster-randomized trial with individual outcomes."""
    with seed(seed_value) as rng:
        clusters = np.arange(n_clusters)
        treated_clusters = set(rng.permutation(clusters)[: n_clusters // 2].tolist())
        rows: list[dict[str, Any]] = []
        cluster_effects = rng.normal(0, cluster_sd, n_clusters)
        for cluster in clusters:
            treatment = int(cluster in treated_clusters)
            for person in range(cluster_size):
                rows.append(
                    {
                        "id": int(cluster * cluster_size + person),
                        "cluster": int(cluster),
                        "treatment": treatment,
                        "cluster_effect": float(cluster_effects[cluster]),
                        "age": float(rng.normal(60, 9)),
                        "baseline_risk": float(rng.normal(0, 1)),
                    }
                )
        df = pd.DataFrame(rows)
        df = logistic(
            df,
            "event_12m ~ treatment + age + baseline_risk + cluster_effect",
            betas={
                "treatment": treatment_effect,
                "age": 0.025,
                "baseline_risk": 0.40,
                "cluster_effect": 1.0,
            },
            intercept=-2.2,
            rng=rng,
        )
        rr, rr_lo, rr_hi = risk_ratio(
            df, exposure="treatment", outcome="event_12m", rng=rng
        )
        rd, rd_lo, rd_hi = risk_difference(
            df, exposure="treatment", outcome="event_12m", rng=rng
        )
        cluster_summary = (
            df.groupby(["cluster", "treatment"], as_index=False)
            .agg(n=("event_12m", "size"), event_rate=("event_12m", "mean"))
            .sort_values("cluster")
        )
        treated_cluster_rate = float(
            cluster_summary.loc[
                cluster_summary["treatment"] == 1, "event_rate"
            ].mean()
        )
        control_cluster_rate = float(
            cluster_summary.loc[
                cluster_summary["treatment"] == 0, "event_rate"
            ].mean()
        )
        return build_study(
            design="rct_cluster",
            seed_value=seed_value,
            data=df,
            params={
                "n_clusters": n_clusters,
                "cluster_size": cluster_size,
                "treatment_effect": treatment_effect,
                "cluster_sd": cluster_sd,
            },
            results={
                "n_total": int(len(df)),
                "n_clusters": n_clusters,
                "risk_ratio": round(rr, 3),
                "risk_ratio_ci": [round(rr_lo, 3), round(rr_hi, 3)],
                "risk_difference": round(rd, 3),
                "risk_difference_ci": [round(rd_lo, 3), round(rd_hi, 3)],
                "treated_cluster_event_rate": round(treated_cluster_rate, 3),
                "control_cluster_event_rate": round(control_cluster_rate, 3),
            },
            artifacts={"cluster_summary": cluster_summary},
        )
