"""Outcome models on top of cohort DataFrames."""
# ruff: noqa: N806  -- statistical convention: design-matrix `X` uppercase
from __future__ import annotations

import numpy as np
import pandas as pd
from formulaic import Formula


def _design_matrix(df: pd.DataFrame, formula: str) -> tuple[str, pd.DataFrame]:
    """Parse 'y ~ x1 + x2'; return outcome name and design matrix without intercept."""
    f = Formula(formula)
    lhs, rhs = f.lhs, f.rhs
    outcome = list(lhs)[0]
    X = rhs.get_model_matrix(df).drop(columns=["Intercept"], errors="ignore")
    return outcome, X


def logistic(
    df: pd.DataFrame,
    formula: str,
    betas: dict[str, float],
    rng: np.random.Generator,
    *,
    intercept: float = 0.0,
) -> pd.DataFrame:
    """Append a binary outcome from a logistic model y ~ X."""
    outcome, X = _design_matrix(df, formula)
    eta = np.full(len(df), float(intercept))
    for col, beta in betas.items():
        if col not in X.columns:
            raise KeyError(
                f"Predictor {col!r} not produced by formula columns {list(X.columns)}"
            )
        eta += beta * X[col].to_numpy()
    p = 1.0 / (1.0 + np.exp(-eta))
    out = df.copy()
    out[outcome] = rng.binomial(1, p)
    return out
