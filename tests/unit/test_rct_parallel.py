"""Tests for episim.designs.rct_parallel."""
from episim.core.reproducibility import Study
from episim.designs import rct_parallel


def test_rct_parallel_has_protective_treatment_effect() -> None:
    study = rct_parallel.run(seed_value=20260508, n=2_000)
    assert isinstance(study, Study)
    assert study.design == "rct_parallel"
    assert study.results["risk_ratio"] < 1.0
    assert study.results["event_rate_treatment"] < study.results["event_rate_control"]


def test_rct_parallel_randomization_is_balanced() -> None:
    study = rct_parallel.run(seed_value=7, n=2_400)
    n_treat = study.results["n_treatment"]
    assert 1_050 < n_treat < 1_350
