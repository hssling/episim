"""Tests for episim.core.outcomes."""
import pandas as pd
import pytest

from episim.core.outcomes import logistic
from episim.core.populations import cohort
from episim.core.reproducibility import seed


def test_logistic_adds_outcome_column() -> None:
    with seed(42) as rng:
        df = cohort(n=1_000, age=("normal", 50, 10), rng=rng)
        df["exposure"] = rng.binomial(1, 0.3, 1_000)
        df = logistic(df, "y ~ exposure + age",
                      betas={"exposure": 0.6, "age": 0.04}, rng=rng)
    assert "y" in df.columns
    assert set(df["y"].unique()) <= {0, 1}


def test_logistic_higher_exposure_increases_outcome_rate() -> None:
    with seed(42) as rng:
        df = cohort(n=10_000, rng=rng)
        df["exposure"] = rng.binomial(1, 0.5, 10_000)
        df = logistic(df, "y ~ exposure",
                      betas={"exposure": 1.5}, intercept=-1.0, rng=rng)
    rate_exp = df.loc[df["exposure"] == 1, "y"].mean()
    rate_unexp = df.loc[df["exposure"] == 0, "y"].mean()
    assert rate_exp > rate_unexp


def test_logistic_unknown_predictor_raises() -> None:
    with seed(42) as rng:
        df = cohort(n=200, rng=rng)
        df["x"] = rng.normal(0, 1, 200)
        with pytest.raises(KeyError, match="Predictor"):
            logistic(df, "y ~ x", betas={"unknown": 0.5}, rng=rng)


def test_logistic_is_deterministic_under_seed() -> None:
    def make() -> pd.DataFrame:
        with seed(42) as rng:
            df = cohort(n=200, rng=rng)
            df["x"] = rng.normal(0, 1, 200)
            return logistic(df, "y ~ x", betas={"x": 0.5}, rng=rng)
    pd.testing.assert_frame_equal(make(), make())
