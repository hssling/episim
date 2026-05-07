"""Tests for episim.analytics.estimators."""
import numpy as np
import pandas as pd

from episim.analytics.estimators import odds_ratio, prevalence
from episim.core.reproducibility import seed


def test_prevalence_basic() -> None:
    df = pd.DataFrame({"y": [1, 1, 0, 0, 0]})
    p, lo, hi = prevalence(df, outcome="y", rng=np.random.default_rng(42))
    assert 0.39 < p < 0.41
    assert lo < p < hi


def test_odds_ratio_directionality() -> None:
    df = pd.DataFrame({
        "exposure": [1] * 100 + [0] * 100,
        "y":        [1] * 80 + [0] * 20 + [1] * 20 + [0] * 80,
    })
    with seed(42) as rng:
        or_, lo, hi = odds_ratio(df, exposure="exposure", outcome="y", rng=rng)
    assert or_ > 5.0
    assert lo < or_ < hi


def test_odds_ratio_null_includes_one() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "exposure": rng.binomial(1, 0.5, 5_000),
        "y": rng.binomial(1, 0.3, 5_000),
    })
    or_, lo, hi = odds_ratio(df, exposure="exposure", outcome="y", rng=rng)
    assert lo < 1.0 < hi
