"""Agent-style stochastic SEIR simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.core import seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n_population: int = 5000,
    n_days: int = 120,
    beta: float = 0.32,
    sigma: float = 0.20,
    gamma: float = 0.12,
    initial_infected: int = 15,
) -> Study:
    """Simulate daily SEIR transitions with binomial stochasticity."""
    with seed(seed_value) as rng:
        susceptible = n_population - initial_infected
        exposed = 0
        infectious = initial_infected
        recovered = 0
        rows = []
        for day in range(n_days + 1):
            rows.append(
                {
                    "day": day,
                    "susceptible": susceptible,
                    "exposed": exposed,
                    "infectious": infectious,
                    "recovered": recovered,
                }
            )
            if day == n_days:
                break
            force = 1 - np.exp(-beta * infectious / n_population)
            new_exposed = rng.binomial(susceptible, np.clip(force, 0, 1))
            new_infectious = rng.binomial(exposed, sigma)
            new_recovered = rng.binomial(infectious, gamma)
            susceptible -= new_exposed
            exposed += new_exposed - new_infectious
            infectious += new_infectious - new_recovered
            recovered += new_recovered
        df = pd.DataFrame(rows)
        return build_study(
            design="agent_based_seir",
            seed_value=seed_value,
            data=df,
            params={
                "n_population": n_population,
                "n_days": n_days,
                "beta": beta,
                "sigma": sigma,
                "gamma": gamma,
                "initial_infected": initial_infected,
            },
            results={
                "peak_infectious": int(df["infectious"].max()),
                "peak_day": int(df.loc[df["infectious"].idxmax(), "day"]),
                "final_recovered": int(df.iloc[-1]["recovered"]),
                "attack_rate": round(float(df.iloc[-1]["recovered"] / n_population), 3),
            },
        )
