"""Network contagion simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.core import seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n_nodes: int = 600,
    n_days: int = 80,
    mean_degree: int = 8,
    transmission_probability: float = 0.035,
    recovery_probability: float = 0.12,
    initial_infected: int = 8,
) -> Study:
    """Simulate SIR-like contagion on a random contact network."""
    with seed(seed_value) as rng:
        edge_p = mean_degree / max(n_nodes - 1, 1)
        upper = rng.random((n_nodes, n_nodes)) < edge_p
        adjacency = np.triu(upper, 1)
        adjacency = adjacency | adjacency.T
        state = np.zeros(n_nodes, dtype=int)
        state[rng.choice(n_nodes, initial_infected, replace=False)] = 1
        rows = []
        for day in range(n_days + 1):
            rows.append(
                {
                    "day": day,
                    "susceptible": int((state == 0).sum()),
                    "infectious": int((state == 1).sum()),
                    "recovered": int((state == 2).sum()),
                }
            )
            if day == n_days or (state == 1).sum() == 0:
                continue
            infectious_neighbors = adjacency[:, state == 1].sum(axis=1)
            infection_p = 1 - (1 - transmission_probability) ** infectious_neighbors
            new_infections = (rng.random(n_nodes) < infection_p) & (state == 0)
            recoveries = (rng.random(n_nodes) < recovery_probability) & (state == 1)
            state[new_infections] = 1
            state[recoveries] = 2
        df = pd.DataFrame(rows)
        degree = adjacency.sum(axis=1)
        nodes = pd.DataFrame({"node": np.arange(n_nodes), "degree": degree, "final_state": state})
        return build_study(
            design="network_contagion",
            seed_value=seed_value,
            data=df,
            params={
                "n_nodes": n_nodes,
                "n_days": n_days,
                "mean_degree": mean_degree,
                "transmission_probability": transmission_probability,
                "recovery_probability": recovery_probability,
                "initial_infected": initial_infected,
            },
            results={
                "peak_infectious": int(df["infectious"].max()),
                "peak_day": int(df.loc[df["infectious"].idxmax(), "day"]),
                "final_attack_rate": round(float((state != 0).mean()), 3),
                "mean_degree_observed": round(float(degree.mean()), 2),
            },
            artifacts={"node_summary": nodes},
        )
