"""Tests for episim.designs.case_control."""
from episim.core.reproducibility import Study
from episim.designs import case_control


def test_case_control_returns_real_study() -> None:
    study = case_control.run(seed_value=20260508, n_source=5_000, n_cases=250)
    assert isinstance(study, Study)
    assert study.design == "case_control"
    assert len(study.data) == 750
    assert study.results["odds_ratio"] > 1.0
    assert (
        study.results["exposure_prevalence_cases"]
        > study.results["exposure_prevalence_controls"]
    )


def test_case_control_is_deterministic() -> None:
    a = case_control.run(seed_value=12345, n_source=4_500, n_cases=200)
    b = case_control.run(seed_value=12345, n_source=4_500, n_cases=200)
    assert a.results == b.results
