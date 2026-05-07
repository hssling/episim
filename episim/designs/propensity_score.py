"""Propensity-score weighting simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from episim.core import seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def _smd(x_treated: pd.Series, x_control: pd.Series) -> float:
    pooled = np.sqrt((x_treated.var() + x_control.var()) / 2)
    return float((x_treated.mean() - x_control.mean()) / pooled)


def run(
    *,
    seed_value: int,
    n: int = 3500,
    treatment_effect: float = -0.12,
) -> Study:
    """Simulate confounded treatment assignment and inverse-probability weighting."""
    with seed(seed_value) as rng:
        age = rng.normal(62, 9, n)
        frailty = rng.normal(0, 1, n)
        access = rng.normal(0, 1, n)
        lp = -0.3 + 0.025 * (age - 62) + 0.55 * frailty + 0.30 * access
        p_treat = 1 / (1 + np.exp(-lp))
        treatment = rng.binomial(1, p_treat)
        outcome = (
            0.40
            + treatment_effect * treatment
            + 0.006 * (age - 62)
            + 0.18 * frailty
            + rng.normal(0, 0.20, n)
        )
        df = pd.DataFrame(
            {
                "id": np.arange(n),
                "age": age,
                "frailty": frailty,
                "access": access,
                "treatment": treatment,
                "outcome": outcome,
            }
        )
        x = StandardScaler().fit_transform(df[["age", "frailty", "access"]].to_numpy())
        model = LogisticRegression(max_iter=1000).fit(x, treatment)
        ps = np.clip(model.predict_proba(x)[:, 1], 0.02, 0.98)
        weights = np.where(treatment == 1, 1 / ps, 1 / (1 - ps))
        df["propensity_score"] = ps
        df["iptw"] = weights

        treated = df["treatment"] == 1
        naive = float(df.loc[treated, "outcome"].mean() - df.loc[~treated, "outcome"].mean())
        wt_t = np.average(df.loc[treated, "outcome"], weights=df.loc[treated, "iptw"])
        wt_c = np.average(df.loc[~treated, "outcome"], weights=df.loc[~treated, "iptw"])
        balance_rows = []
        for covariate in ("age", "frailty", "access"):
            balance_rows.append(
                {
                    "covariate": covariate,
                    "smd_unweighted": round(
                        abs(_smd(df.loc[treated, covariate], df.loc[~treated, covariate])),
                        3,
                    ),
                }
            )
        balance = pd.DataFrame(balance_rows)
        return build_study(
            design="propensity_score",
            seed_value=seed_value,
            data=df,
            params={"n": n, "treatment_effect": treatment_effect},
            results={
                "naive_difference": round(naive, 3),
                "iptw_difference": round(float(wt_t - wt_c), 3),
                "mean_propensity": round(float(ps.mean()), 3),
                "max_unweighted_smd": round(float(balance["smd_unweighted"].max()), 3),
            },
            artifacts={"covariate_balance": balance},
        )
