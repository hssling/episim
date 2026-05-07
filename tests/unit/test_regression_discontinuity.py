"""Tests for episim.designs.regression_discontinuity."""
from episim.designs import regression_discontinuity


def test_regression_discontinuity_estimates_negative_jump() -> None:
    study = regression_discontinuity.run(seed_value=20260508, n=1800)
    assert study.design == "regression_discontinuity"
    assert study.results["estimated_jump"] < 0.0
    assert study.results["right_mean_outcome"] < study.results["left_mean_outcome"]
    assert "local_bandwidth_sample" in study.artifacts


def test_regression_discontinuity_is_deterministic() -> None:
    a = regression_discontinuity.run(seed_value=66, n=1500)
    b = regression_discontinuity.run(seed_value=66, n=1500)
    assert a.results == b.results
