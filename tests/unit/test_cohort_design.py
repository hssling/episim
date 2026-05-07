"""Tests for episim.designs.cohort."""
from episim.core.reproducibility import Study
from episim.designs import cohort


def test_cohort_study_has_expected_directionality() -> None:
    study = cohort.run(seed_value=20260508, n=2_500)
    assert isinstance(study, Study)
    assert study.design == "cohort"
    assert study.results["risk_ratio"] > 1.0
    assert study.results["event_rate_exposed"] > study.results["event_rate_unexposed"]
    assert 0.0 < study.results["attrition_fraction"] < 0.2


def test_cohort_is_deterministic() -> None:
    a = cohort.run(seed_value=8080, n=2_000)
    b = cohort.run(seed_value=8080, n=2_000)
    assert a.results == b.results
