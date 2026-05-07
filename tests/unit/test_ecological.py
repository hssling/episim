"""Tests for episim.designs.ecological."""
from episim.core.reproducibility import Study
from episim.designs import ecological


def test_peai_pipeline_returns_non_placeholder_results() -> None:
    study = ecological.run_peai(
        seed_value=20260508,
        n_experts=20,
        n_dev=500,
        n_external=300,
        n_prospective=700,
    )
    assert isinstance(study, Study)
    assert study.design == "ecological_peai"
    assert "phase3" in study.results
    assert "phase4" in study.results
    assert "weighting_convergence" in study.artifacts
    assert "prospective_discrimination" in study.artifacts
    assert study.results["phase4"]["max_ascertainment_attenuation"] > 0.0
    assert study.results["phase4"]["max_ses_equalised_odds_gap"] >= 0.0


def test_peai_pipeline_is_deterministic() -> None:
    a = ecological.run_peai(seed_value=11, n_dev=350, n_external=250, n_prospective=500)
    b = ecological.run_peai(seed_value=11, n_dev=350, n_external=250, n_prospective=500)
    assert a.results == b.results
