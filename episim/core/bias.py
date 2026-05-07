"""Bias primitives: measurement error, ascertainment."""
from __future__ import annotations

import numpy as np
import pandas as pd


def measurement_error(
    df: pd.DataFrame, on: str, sd: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Add Gaussian measurement error of standard deviation sd to column on."""
    out = df.copy()
    out[on] = out[on] + rng.normal(0.0, sd, len(out))
    return out


def ascertainment(
    df: pd.DataFrame,
    true_outcome: str,
    access_col: str,
    observed_outcome: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate observed_outcome = true_outcome × Bernoulli(access).

    Models detection probability as a 0..1 access score."""
    out = df.copy()
    detect = rng.binomial(1, np.clip(out[access_col].to_numpy(), 0, 1))
    out[observed_outcome] = (out[true_outcome].to_numpy() * detect).astype(int)
    return out
