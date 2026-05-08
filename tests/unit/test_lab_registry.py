"""Tests for EPISIM lab registry and runner surfaces."""
from episim.core.reproducibility import Study
from episim.lab import get_design, list_designs, run_design, study_preview


def test_registry_exposes_multiple_designs() -> None:
    designs = list_designs()
    keys = {design.key for design in designs}
    assert "cross_sectional" in keys
    assert "ecological_peai" in keys
    assert len(designs) >= 5


def test_runner_executes_registered_design() -> None:
    study = run_design("cross_sectional", seed_value=77, n=600)
    assert isinstance(study, Study)
    assert study.design == "cross_sectional"
    assert study.results["prevalence"] > 0.0


def test_runner_ignores_stale_overrides_for_selected_design() -> None:
    study = run_design(
        "case_control",
        seed_value=77,
        n=600,
        n_interviews=12,
        n_source=3000,
        n_cases=120,
    )
    assert study.design == "case_control"
    assert study.results["sampled_cases"] == 120


def test_study_preview_returns_frames() -> None:
    study = run_design("case_control", seed_value=88, n_source=4000, n_cases=200)
    data_preview, result_preview = study_preview(study)
    assert len(data_preview) <= 8
    assert not result_preview.empty
    assert "odds_ratio" in result_preview.columns


def test_get_design_unknown_key_raises() -> None:
    try:
        get_design("unknown")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for unknown design key.")
