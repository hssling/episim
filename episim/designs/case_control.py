"""Case-control study simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.analytics import odds_ratio
from episim.core import cohort as make_cohort
from episim.core import logistic, seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def _sample_rows(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n > len(df):
        msg = f"Requested {n} rows from a frame of size {len(df)}."
        raise ValueError(msg)
    order = rng.permutation(len(df))[:n]
    return df.iloc[order].copy()


def run(
    *,
    seed_value: int,
    n_source: int = 8_000,
    n_cases: int = 400,
    controls_per_case: int = 2,
    exposure_log_odds: float = 0.9,
    outcome_intercept: float = -2.2,
) -> Study:
    """Simulate a case-control sample from a source population."""
    with seed(seed_value) as rng:
        source = make_cohort(
            n=n_source,
            age=("normal", 58, 11),
            sex=("bernoulli", 0.52),
            ses=("normal", 0, 1),
            rng=rng,
        )
        age_z = (source["age"] - source["age"].mean()) / source["age"].std()
        exposure_lp = -0.4 + 0.45 * age_z + 0.30 * source["sex"] - 0.35 * source["ses"]
        source["exposure"] = rng.binomial(1, 1 / (1 + np.exp(-exposure_lp)))
        source = logistic(
            source,
            "case ~ exposure + age + sex + ses",
            betas={
                "exposure": exposure_log_odds,
                "age": 0.025,
                "sex": 0.18,
                "ses": -0.22,
            },
            intercept=outcome_intercept,
            rng=rng,
        )

        cases = source.loc[source["case"] == 1]
        controls = source.loc[source["case"] == 0]
        sampled_cases = _sample_rows(cases, n_cases, rng)
        sampled_controls = _sample_rows(controls, n_cases * controls_per_case, rng)
        sampled_cases = sampled_cases.assign(sampled_case_control_status="case")
        sampled_controls = sampled_controls.assign(sampled_case_control_status="control")
        study_df = (
            pd.concat([sampled_cases, sampled_controls], ignore_index=True)
            .sample(frac=1.0, random_state=seed_value)
            .reset_index(drop=True)
        )
        or_, or_lo, or_hi = odds_ratio(
            study_df, exposure="exposure", outcome="case", rng=rng
        )
        results = {
            "source_cases": int(len(cases)),
            "source_controls": int(len(controls)),
            "sampled_cases": int(len(sampled_cases)),
            "sampled_controls": int(len(sampled_controls)),
            "odds_ratio": round(or_, 3),
            "odds_ratio_ci": [round(or_lo, 3), round(or_hi, 3)],
            "exposure_prevalence_cases": round(
                float(sampled_cases["exposure"].mean()), 3
            ),
            "exposure_prevalence_controls": round(
                float(sampled_controls["exposure"].mean()), 3
            ),
        }
        params = {
            "n_source": n_source,
            "n_cases": n_cases,
            "controls_per_case": controls_per_case,
            "exposure_log_odds": exposure_log_odds,
        }
        return build_study(
            design="case_control",
            seed_value=seed_value,
            data=study_df,
            params=params,
            results=results,
        )
