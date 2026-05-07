"""Point estimators with bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BOOT = 1_000


def _bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.nanpercentile(values, 100 * alpha / 2))
    hi = float(np.nanpercentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


def prevalence(
    df: pd.DataFrame,
    outcome: str,
    rng: np.random.Generator,
    *,
    n_boot: int = DEFAULT_BOOT,
) -> tuple[float, float, float]:
    """Sample prevalence with percentile bootstrap 95 % CI."""
    y = df[outcome].to_numpy()
    p = float(y.mean())
    n = len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = y[idx].mean()
    lo, hi = _bootstrap_ci(boots)
    return p, lo, hi


def odds_ratio(
    df: pd.DataFrame,
    exposure: str,
    outcome: str,
    rng: np.random.Generator,
    *,
    n_boot: int = DEFAULT_BOOT,
) -> tuple[float, float, float]:
    """2×2 odds ratio with percentile bootstrap 95 % CI (continuity-corrected)."""

    def _or(sample: pd.DataFrame) -> float:
        a = ((sample[exposure] == 1) & (sample[outcome] == 1)).sum() + 0.5
        b = ((sample[exposure] == 1) & (sample[outcome] == 0)).sum() + 0.5
        c = ((sample[exposure] == 0) & (sample[outcome] == 1)).sum() + 0.5
        d = ((sample[exposure] == 0) & (sample[outcome] == 0)).sum() + 0.5
        return float((a * d) / (b * c))

    or_ = _or(df)
    n = len(df)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = _or(df.iloc[idx])
    lo, hi = _bootstrap_ci(boots)
    return or_, lo, hi
