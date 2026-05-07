"""Instrumental-variable simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.core import seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n: int = 4000,
    instrument_strength: float = 0.28,
    treatment_effect: float = -0.18,
) -> Study:
    """Simulate a binary-instrument encouragement design and Wald estimate."""
    with seed(seed_value) as rng:
        z = rng.binomial(1, 0.5, n)
        u = rng.normal(0, 1, n)
        age = rng.normal(60, 10, n)
        exposure_p = np.clip(0.25 + instrument_strength * z + 0.12 * u, 0.02, 0.98)
        exposure = rng.binomial(1, exposure_p)
        outcome = (
            0.55
            + treatment_effect * exposure
            + 0.10 * u
            + 0.003 * (age - 60)
            + rng.normal(0, 0.20, n)
        )
        df = pd.DataFrame(
            {
                "id": np.arange(n),
                "instrument": z,
                "exposure": exposure,
                "age": age,
                "outcome": outcome,
            }
        )
        first_stage = float(
            df.loc[z == 1, "exposure"].mean() - df.loc[z == 0, "exposure"].mean()
        )
        reduced_form = float(
            df.loc[z == 1, "outcome"].mean() - df.loc[z == 0, "outcome"].mean()
        )
        wald = reduced_form / first_stage
        naive = float(
            df.loc[exposure == 1, "outcome"].mean()
            - df.loc[exposure == 0, "outcome"].mean()
        )
        return build_study(
            design="instrumental_variables",
            seed_value=seed_value,
            data=df,
            params={
                "n": n,
                "instrument_strength": instrument_strength,
                "treatment_effect": treatment_effect,
            },
            results={
                "first_stage": round(first_stage, 3),
                "reduced_form": round(reduced_form, 3),
                "wald_late": round(float(wald), 3),
                "naive_difference": round(naive, 3),
            },
        )
