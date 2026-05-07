"""Tests for episim.core.bias."""
import numpy as np
import pandas as pd

from episim.core.bias import ascertainment, measurement_error
from episim.core.reproducibility import seed


def test_measurement_error_adds_noise_column() -> None:
    df = pd.DataFrame({"x": np.zeros(1_000)})
    with seed(42) as rng:
        out = measurement_error(df, on="x", sd=1.0, rng=rng)
    assert "x" in out.columns
    assert 0.9 < out["x"].std() < 1.1
    assert -0.1 < out["x"].mean() < 0.1


def test_ascertainment_attenuates_outcome_with_low_access() -> None:
    df = pd.DataFrame({
        "y_true": [1] * 1_000 + [0] * 1_000,
        "access": [0.1] * 1_000 + [0.1] * 1_000,
    })
    with seed(42) as rng:
        out = ascertainment(df, true_outcome="y_true", access_col="access",
                            observed_outcome="y_obs", rng=rng)
    assert out["y_obs"].sum() < df["y_true"].sum()


def test_ascertainment_full_access_recovers_truth() -> None:
    df = pd.DataFrame({
        "y_true": [1, 0, 1, 0, 1],
        "access": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    with seed(7) as rng:
        out = ascertainment(df, true_outcome="y_true", access_col="access",
                            observed_outcome="y_obs", rng=rng)
    assert (out["y_obs"].to_numpy() == df["y_true"].to_numpy()).all()
