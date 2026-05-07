"""Tests for week-5-to-week-8 roadmap designs."""
from episim.designs import (
    instrumental_variables,
    markov_decision,
    meta_analysis,
    microsimulation,
    network_contagion,
    propensity_score,
    qualitative_mixed_methods,
    seir,
    survival_cox,
)


def test_instrumental_variables_direction_and_determinism() -> None:
    study = instrumental_variables.run(seed_value=101)
    repeat = instrumental_variables.run(seed_value=101)
    assert study.results == repeat.results
    assert study.results["first_stage"] > 0.1
    assert study.results["wald_late"] < 0.0


def test_propensity_score_weighting_moves_toward_effect() -> None:
    study = propensity_score.run(seed_value=102)
    assert study.results["iptw_difference"] < 0.0
    assert "covariate_balance" in study.artifacts


def test_survival_cox_recovers_protective_hazard_ratio() -> None:
    study = survival_cox.run(seed_value=103)
    assert study.results["estimated_hazard_ratio"] < 1.0
    assert "survival_summary" in study.artifacts


def test_meta_analysis_pools_negative_effect() -> None:
    study = meta_analysis.run(seed_value=104)
    assert study.results["pooled_effect"] < 0.0
    assert study.results["i2"] >= 0.0


def test_seir_has_epidemic_curve() -> None:
    study = seir.run(seed_value=105, n_population=1500, n_days=80)
    assert study.results["peak_infectious"] > 15
    assert study.results["attack_rate"] > 0.0


def test_microsimulation_outputs_qalys_and_costs() -> None:
    study = microsimulation.run(seed_value=106, n=1000, cycles=10)
    assert study.results["mean_qaly"] > 0.0
    assert study.results["mean_cost"] > 0.0
    assert "cycle_trajectory" in study.artifacts


def test_markov_decision_outputs_icer() -> None:
    study = markov_decision.run(seed_value=107)
    assert study.results["incremental_qaly"] > 0.0
    assert study.results["icer"] > 0.0


def test_network_contagion_spreads_on_network() -> None:
    study = network_contagion.run(seed_value=108, n_nodes=300, n_days=60)
    assert study.results["final_attack_rate"] > 0.0
    assert "node_summary" in study.artifacts


def test_qualitative_mixed_methods_reaches_saturation() -> None:
    study = qualitative_mixed_methods.run(seed_value=109)
    assert study.results["themes_identified"] > 0
    assert study.results["saturation_interview"] <= 40
    assert "survey_strand" in study.artifacts
