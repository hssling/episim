"""Point estimators with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

DEFAULT_BOOT = 1_000


def _bootstrap_ci(
    values: npt.NDArray[np.float64], alpha: float = 0.05
) -> tuple[float, float]:
    lo = float(np.nanpercentile(values, 100 * alpha / 2))
    hi = float(np.nanpercentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


def _binary_table(
    df: pd.DataFrame, exposure: str, outcome: str
) -> tuple[float, float, float, float]:
    a = float(((df[exposure] == 1) & (df[outcome] == 1)).sum())
    b = float(((df[exposure] == 1) & (df[outcome] == 0)).sum())
    c = float(((df[exposure] == 0) & (df[outcome] == 1)).sum())
    d = float(((df[exposure] == 0) & (df[outcome] == 0)).sum())
    return a, b, c, d


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
    """2x2 odds ratio with percentile bootstrap 95 % CI."""

    def _or(sample: pd.DataFrame) -> float:
        a, b, c, d = _binary_table(sample, exposure, outcome)
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
        return float((a * d) / (b * c))

    or_ = _or(df)
    n = len(df)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = _or(df.iloc[idx])
    lo, hi = _bootstrap_ci(boots)
    return or_, lo, hi


def risk_ratio(
    df: pd.DataFrame,
    exposure: str,
    outcome: str,
    rng: np.random.Generator,
    *,
    n_boot: int = DEFAULT_BOOT,
) -> tuple[float, float, float]:
    """Risk ratio with percentile bootstrap CI and continuity correction."""

    def _rr(sample: pd.DataFrame) -> float:
        a, b, c, d = _binary_table(sample, exposure, outcome)
        risk_exp = (a + 0.5) / (a + b + 1.0)
        risk_unexp = (c + 0.5) / (c + d + 1.0)
        return float(risk_exp / risk_unexp)

    rr = _rr(df)
    n = len(df)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = _rr(df.iloc[idx])
    lo, hi = _bootstrap_ci(boots)
    return rr, lo, hi


def risk_difference(
    df: pd.DataFrame,
    exposure: str,
    outcome: str,
    rng: np.random.Generator,
    *,
    n_boot: int = DEFAULT_BOOT,
) -> tuple[float, float, float]:
    """Risk difference with percentile bootstrap CI."""

    def _rd(sample: pd.DataFrame) -> float:
        exposed = sample.loc[sample[exposure] == 1, outcome]
        unexposed = sample.loc[sample[exposure] == 0, outcome]
        return float(exposed.mean() - unexposed.mean())

    rd = _rd(df)
    n = len(df)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = _rd(df.iloc[idx])
    lo, hi = _bootstrap_ci(boots)
    return rd, lo, hi
