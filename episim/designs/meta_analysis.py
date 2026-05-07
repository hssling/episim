"""Random-effects meta-analysis simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.core import seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n_studies: int = 12,
    true_effect: float = -0.20,
    tau: float = 0.08,
) -> Study:
    """Simulate study estimates and fit DerSimonian-Laird random effects."""
    with seed(seed_value) as rng:
        study_n = rng.integers(120, 900, n_studies)
        se = 1 / np.sqrt(study_n / 4)
        theta = rng.normal(true_effect, tau, n_studies)
        estimate = rng.normal(theta, se)
        fixed_w = 1 / se**2
        fixed = float(np.sum(fixed_w * estimate) / np.sum(fixed_w))
        q = float(np.sum(fixed_w * (estimate - fixed) ** 2))
        c = float(np.sum(fixed_w) - np.sum(fixed_w**2) / np.sum(fixed_w))
        tau2 = max(0.0, (q - (n_studies - 1)) / c)
        random_w = 1 / (se**2 + tau2)
        pooled = float(np.sum(random_w * estimate) / np.sum(random_w))
        pooled_se = float(np.sqrt(1 / np.sum(random_w)))
        i2 = max(0.0, (q - (n_studies - 1)) / q) if q > 0 else 0.0
        df = pd.DataFrame(
            {
                "study": [f"study_{i + 1}" for i in range(n_studies)],
                "n": study_n,
                "estimate": estimate,
                "se": se,
                "weight_random": random_w,
            }
        )
        return build_study(
            design="meta_analysis",
            seed_value=seed_value,
            data=df,
            params={"n_studies": n_studies, "true_effect": true_effect, "tau": tau},
            results={
                "pooled_effect": round(pooled, 3),
                "pooled_ci": [
                    round(pooled - 1.96 * pooled_se, 3),
                    round(pooled + 1.96 * pooled_se, 3),
                ],
                "tau2": round(tau2, 4),
                "i2": round(float(i2), 3),
                "q": round(q, 3),
            },
        )
