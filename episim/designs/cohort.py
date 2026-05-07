"""Prospective cohort study simulation."""
from __future__ import annotations

import numpy as np

from episim.analytics import risk_difference, risk_ratio
from episim.core import apply_mcar, logistic, seed
from episim.core import cohort as make_cohort
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n: int = 4_000,
    p_attrition: float = 0.08,
    exposure_effect: float = 0.75,
) -> Study:
    """Simulate a prospective cohort with follow-up attrition."""
    with seed(seed_value) as rng:
        df = make_cohort(
            n=n,
            age=("normal", 56, 10),
            sex=("bernoulli", 0.51),
            vulnerability=("normal", 0, 1),
            rng=rng,
        )
        age_z = (df["age"] - df["age"].mean()) / df["age"].std()
        exposure_lp = -0.6 + 0.5 * age_z + 0.25 * df["sex"] + 0.45 * df["vulnerability"]
        df["exposure"] = rng.binomial(1, 1 / (1 + np.exp(-exposure_lp)))
        df = logistic(
            df,
            "event_24m ~ exposure + age + sex + vulnerability",
            betas={
                "exposure": exposure_effect,
                "age": 0.03,
                "sex": 0.10,
                "vulnerability": 0.35,
            },
            intercept=-2.6,
            rng=rng,
        )
        followed = apply_mcar(df, p_drop=p_attrition, rng=rng)
        rr, rr_lo, rr_hi = risk_ratio(
            followed, exposure="exposure", outcome="event_24m", rng=rng
        )
        rd, rd_lo, rd_hi = risk_difference(
            followed, exposure="exposure", outcome="event_24m", rng=rng
        )
        results = {
            "n_baseline": n,
            "n_followed": int(len(followed)),
            "attrition_fraction": round(1 - len(followed) / n, 3),
            "risk_ratio": round(rr, 3),
            "risk_ratio_ci": [round(rr_lo, 3), round(rr_hi, 3)],
            "risk_difference": round(rd, 3),
            "risk_difference_ci": [round(rd_lo, 3), round(rd_hi, 3)],
            "event_rate_exposed": round(
                float(followed.loc[followed["exposure"] == 1, "event_24m"].mean()), 3
            ),
            "event_rate_unexposed": round(
                float(followed.loc[followed["exposure"] == 0, "event_24m"].mean()), 3
            ),
        }
        return build_study(
            design="cohort",
            seed_value=seed_value,
            data=followed,
            params={
                "n": n,
                "p_attrition": p_attrition,
                "exposure_effect": exposure_effect,
            },
            results=results,
        )
