"""Parallel-group randomized trial simulation."""
from __future__ import annotations

from episim.analytics import risk_difference, risk_ratio
from episim.core import cohort as make_cohort
from episim.core import logistic, seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n: int = 2_400,
    allocation_ratio: float = 0.5,
    treatment_effect: float = -0.7,
) -> Study:
    """Simulate a two-arm randomized controlled trial."""
    with seed(seed_value) as rng:
        df = make_cohort(
            n=n,
            age=("normal", 61, 9),
            sex=("bernoulli", 0.50),
            baseline_risk=("normal", 0, 1),
            rng=rng,
        )
        df["treatment"] = rng.binomial(1, allocation_ratio, n)
        df = logistic(
            df,
            "event_12m ~ treatment + age + sex + baseline_risk",
            betas={
                "treatment": treatment_effect,
                "age": 0.028,
                "sex": 0.06,
                "baseline_risk": 0.42,
            },
            intercept=-2.4,
            rng=rng,
        )
        rr, rr_lo, rr_hi = risk_ratio(
            df, exposure="treatment", outcome="event_12m", rng=rng
        )
        rd, rd_lo, rd_hi = risk_difference(
            df, exposure="treatment", outcome="event_12m", rng=rng
        )
        results = {
            "n_total": n,
            "n_treatment": int(df["treatment"].sum()),
            "n_control": int(n - df["treatment"].sum()),
            "event_rate_treatment": round(
                float(df.loc[df["treatment"] == 1, "event_12m"].mean()), 3
            ),
            "event_rate_control": round(
                float(df.loc[df["treatment"] == 0, "event_12m"].mean()), 3
            ),
            "risk_ratio": round(rr, 3),
            "risk_ratio_ci": [round(rr_lo, 3), round(rr_hi, 3)],
            "risk_difference": round(rd, 3),
            "risk_difference_ci": [round(rd_lo, 3), round(rd_hi, 3)],
        }
        return build_study(
            design="rct_parallel",
            seed_value=seed_value,
            data=df,
            params={
                "n": n,
                "allocation_ratio": allocation_ratio,
                "treatment_effect": treatment_effect,
            },
            results=results,
        )
