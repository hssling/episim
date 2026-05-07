"""Tests for episim.designs.interrupted_time_series."""
from episim.designs import interrupted_time_series


def test_interrupted_time_series_estimates_negative_change() -> None:
    study = interrupted_time_series.run(seed_value=20260508)
    assert study.design == "interrupted_time_series"
    assert study.results["estimated_level_change"] < 0.0
    assert study.results["post_mean_rate"] < study.results["pre_mean_rate"]
    assert "time_series" in study.artifacts


def test_interrupted_time_series_is_deterministic() -> None:
    a = interrupted_time_series.run(seed_value=55, n_periods=48)
    b = interrupted_time_series.run(seed_value=55, n_periods=48)
    assert a.results == b.results
