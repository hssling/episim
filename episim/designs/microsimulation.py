"""Individual life-table microsimulation."""
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
    cycles: int = 20,
    intervention: bool = True,
) -> Study:
    """Simulate individual disease and mortality transitions over annual cycles."""
    with seed(seed_value) as rng:
        age = rng.normal(55, 8, n)
        diseased = np.zeros(n, dtype=int)
        dead = np.zeros(n, dtype=int)
        qaly = np.zeros(n)
        cost = np.zeros(n)
        rows = []
        incidence_multiplier = 0.75 if intervention else 1.0
        for cycle in range(cycles):
            alive = dead == 0
            disease_p = np.clip((0.025 + 0.002 * (age - 55)) * incidence_multiplier, 0.005, 0.30)
            new_disease = (rng.random(n) < disease_p) & alive & (diseased == 0)
            diseased[new_disease] = 1
            death_p = np.clip(0.01 + 0.003 * (age - 55) + 0.035 * diseased, 0.002, 0.50)
            new_death = (rng.random(n) < death_p) & alive
            dead[new_death] = 1
            utility = np.where(diseased == 1, 0.72, 0.90)
            qaly += np.where(alive, utility, 0.0)
            cost += np.where(alive, 500 + 1800 * diseased + (250 if intervention else 0), 0.0)
            rows.append(
                {
                    "cycle": cycle,
                    "alive": int((dead == 0).sum()),
                    "diseased": int(((dead == 0) & (diseased == 1)).sum()),
                    "dead": int(dead.sum()),
                }
            )
            age += 1
        df = pd.DataFrame(
            {
                "id": np.arange(n),
                "qaly": qaly,
                "cost": cost,
                "dead": dead,
                "diseased": diseased,
            }
        )
        trajectory = pd.DataFrame(rows)
        return build_study(
            design="microsimulation_lifetable",
            seed_value=seed_value,
            data=df,
            params={"n": n, "cycles": cycles, "intervention": intervention},
            results={
                "mean_qaly": round(float(df["qaly"].mean()), 3),
                "mean_cost": round(float(df["cost"].mean()), 2),
                "disease_prevalence": round(float(df["diseased"].mean()), 3),
                "mortality_fraction": round(float(df["dead"].mean()), 3),
            },
            artifacts={"cycle_trajectory": trajectory},
        )
