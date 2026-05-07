"""Population/cohort generators."""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

DistributionSpec: TypeAlias = tuple[str | float | int, ...]


def _sample(
    spec: DistributionSpec, n: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Sample n values from a distribution spec like ('normal', mu, sigma)."""
    name = spec[0]
    if name == "normal":
        _, mu, sigma = spec
        return np.asarray(rng.normal(float(mu), float(sigma), n))
    if name == "bernoulli":
        _, p = spec
        return np.asarray(rng.binomial(1, float(p), n))
    if name == "uniform":
        _, lo, hi = spec
        return np.asarray(rng.uniform(float(lo), float(hi), n))
    raise ValueError(f"Unknown distribution: {name!r}")


def cohort(
    n: int,
    rng: np.random.Generator,
    *,
    age: DistributionSpec | None = None,
    sex: DistributionSpec | None = None,
    **other: DistributionSpec,
) -> pd.DataFrame:
    """Generate a cohort of size n with optional covariates.

    Distribution specs supported:
      ('normal', mu, sigma), ('bernoulli', p), ('uniform', lo, hi).
    """
    df = pd.DataFrame({"id": np.arange(n)})
    if age is not None:
        df["age"] = _sample(age, n, rng)
    if sex is not None:
        df["sex"] = _sample(sex, n, rng).astype(int)
    for name, spec in other.items():
        df[name] = _sample(spec, n, rng)
    return df
