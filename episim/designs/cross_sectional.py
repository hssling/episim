"""Cross-sectional study design."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.analytics import odds_ratio, prevalence
from episim.core import cohort as make_cohort
from episim.core import logistic, seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    df: pd.DataFrame,
    *,
    exposure: str,
    outcome: str,
    seed_value: int,
) -> Study:
    """Wrap a prepared DataFrame as a cross-sectional Study object."""
    return build_study(
        design="cross_sectional",
        seed_value=seed_value,
        data=df,
        params={"n": len(df), "exposure": exposure, "outcome": outcome},
    )


def simulate(
    *,
    seed_value: int,
    n: int = 1_200,
    exposure_prevalence: float = 0.30,
    exposure_log_odds: float = 0.60,
    outcome_intercept: float = -3.0,
) -> Study:
    """Generate and analyse a complete cross-sectional study."""
    with seed(seed_value) as rng:
        df = make_cohort(
            n=n,
            age=("normal", 50, 12),
            sex=("bernoulli", 0.50),
            risk=("normal", 0, 1),
            rng=rng,
        )
        exposure_lp = (
            np.log(exposure_prevalence / (1 - exposure_prevalence))
            + 0.02 * (df["age"] - 50)
            + 0.18 * df["sex"]
            + 0.30 * df["risk"]
        )
        df["exposure"] = rng.binomial(1, 1 / (1 + np.exp(-exposure_lp)))
        df = logistic(
            df,
            "y ~ exposure + age + sex + risk",
            betas={
                "exposure": exposure_log_odds,
                "age": 0.03,
                "sex": 0.12,
                "risk": 0.35,
            },
            intercept=outcome_intercept,
            rng=rng,
        )
        prev, prev_lo, prev_hi = prevalence(df, outcome="y", rng=rng)
        or_, or_lo, or_hi = odds_ratio(df, exposure="exposure", outcome="y", rng=rng)
        return build_study(
            design="cross_sectional",
            seed_value=seed_value,
            data=df,
            params={
                "n": n,
                "exposure_prevalence": exposure_prevalence,
                "exposure_log_odds": exposure_log_odds,
                "outcome_intercept": outcome_intercept,
            },
            results={
                "prevalence": round(prev, 3),
                "prevalence_ci": [round(prev_lo, 3), round(prev_hi, 3)],
                "odds_ratio": round(or_, 3),
                "odds_ratio_ci": [round(or_lo, 3), round(or_hi, 3)],
                "exposure_rate": round(float(df["exposure"].mean()), 3),
            },
        )
