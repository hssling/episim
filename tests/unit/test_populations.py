"""Tests for episim.core.populations."""
import pandas as pd
import pytest

from episim.core.populations import cohort
from episim.core.reproducibility import seed


def test_cohort_returns_dataframe_with_expected_n() -> None:
    with seed(42) as rng:
        df = cohort(n=100, rng=rng)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "id" in df.columns


def test_cohort_age_normal() -> None:
    with seed(42) as rng:
        df = cohort(n=10_000, age=("normal", 50, 10), rng=rng)
    assert "age" in df.columns
    assert 47 < df["age"].mean() < 53
    assert 9 < df["age"].std() < 11


def test_cohort_sex_bernoulli() -> None:
    with seed(42) as rng:
        df = cohort(n=10_000, sex=("bernoulli", 0.5), rng=rng)
    assert set(df["sex"].unique()) == {0, 1}
    assert 0.45 < df["sex"].mean() < 0.55


def test_cohort_uniform() -> None:
    with seed(42) as rng:
        df = cohort(n=10_000, rng=rng, score=("uniform", 0, 1))
    assert "score" in df.columns
    assert 0.0 <= df["score"].min()
    assert df["score"].max() <= 1.0


def test_cohort_is_deterministic_under_seed() -> None:
    with seed(42) as rng1:
        df1 = cohort(n=200, age=("normal", 50, 10), rng=rng1)
    with seed(42) as rng2:
        df2 = cohort(n=200, age=("normal", 50, 10), rng=rng2)
    pd.testing.assert_frame_equal(df1, df2)


def test_cohort_unknown_distribution_raises() -> None:
    with pytest.raises(ValueError, match="Unknown distribution"):
        with seed(42) as rng:
            cohort(n=10, age=("alien", 0, 1), rng=rng)
