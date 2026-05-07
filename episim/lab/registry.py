"""Registry of implemented EPISIM lab designs."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from episim.core.reproducibility import Study
from episim.designs import (
    case_control,
    cohort,
    cross_sectional,
    ecological,
    instrumental_variables,
    interrupted_time_series,
    markov_decision,
    meta_analysis,
    microsimulation,
    network_contagion,
    propensity_score,
    qualitative_mixed_methods,
    rct_cluster,
    rct_parallel,
    regression_discontinuity,
    seir,
    stepped_wedge,
    survival_cox,
)


@dataclass(frozen=True)
class DesignSpec:
    """Metadata describing a runnable design in the lab."""

    key: str
    title: str
    family: str
    disciplines: tuple[str, ...]
    description: str
    parameters: dict[str, Any]
    runner: Callable[..., Study]


_DESIGNS: tuple[DesignSpec, ...] = (
    DesignSpec(
        key="cross_sectional",
        title="Cross-sectional prevalence study",
        family="observational",
        disciplines=("epidemiology", "public health", "social medicine"),
        description=(
            "Single-wave observational design with exposure-outcome association "
            "and prevalence estimation."
        ),
        parameters={
            "seed_value": 20260507,
            "n": 1200,
            "exposure_prevalence": 0.30,
            "exposure_log_odds": 0.60,
            "outcome_intercept": -3.0,
        },
        runner=cross_sectional.simulate,
    ),
    DesignSpec(
        key="case_control",
        title="Case-control study",
        family="observational",
        disciplines=("epidemiology", "clinical research", "health services"),
        description=(
            "Source-population sampling of incident cases and matched controls "
            "with odds-ratio recovery."
        ),
        parameters={
            "seed_value": 20260507,
            "n_source": 8000,
            "n_cases": 400,
            "controls_per_case": 2,
            "exposure_log_odds": 0.90,
            "outcome_intercept": -2.2,
        },
        runner=case_control.run,
    ),
    DesignSpec(
        key="cohort",
        title="Prospective cohort",
        family="observational",
        disciplines=("epidemiology", "population health", "aging research"),
        description=(
            "Longitudinal follow-up design with attrition, risk ratios, and "
            "risk differences."
        ),
        parameters={
            "seed_value": 20260507,
            "n": 4000,
            "p_attrition": 0.08,
            "exposure_effect": 0.75,
        },
        runner=cohort.run,
    ),
    DesignSpec(
        key="rct_parallel",
        title="Parallel-group randomized trial",
        family="experimental",
        disciplines=("clinical trials", "rehabilitation", "implementation science"),
        description=(
            "Two-arm randomized experiment estimating absolute and relative "
            "treatment effects."
        ),
        parameters={
            "seed_value": 20260507,
            "n": 2400,
            "allocation_ratio": 0.50,
            "treatment_effect": -0.70,
        },
        runner=rct_parallel.run,
    ),
    DesignSpec(
        key="rct_cluster",
        title="Cluster-randomized trial",
        family="experimental",
        disciplines=("cluster trials", "community medicine", "implementation science"),
        description=(
            "Cluster-level randomization with individual outcomes and cluster summaries."
        ),
        parameters={
            "seed_value": 20260507,
            "n_clusters": 30,
            "cluster_size": 80,
            "treatment_effect": -0.55,
            "cluster_sd": 0.45,
        },
        runner=rct_cluster.run,
    ),
    DesignSpec(
        key="stepped_wedge",
        title="Stepped-wedge cluster trial",
        family="experimental",
        disciplines=("implementation science", "public health", "health systems"),
        description=(
            "Sequential cluster rollout with repeated cross-sections and secular trend."
        ),
        parameters={
            "seed_value": 20260507,
            "n_clusters": 12,
            "n_periods": 6,
            "persons_per_cluster_period": 50,
            "treatment_effect": -0.60,
            "secular_trend": -0.06,
        },
        runner=stepped_wedge.run,
    ),
    DesignSpec(
        key="interrupted_time_series",
        title="Interrupted time series",
        family="quasi-experimental",
        disciplines=("policy evaluation", "health services", "public health"),
        description=(
            "Segmented time-series experiment estimating level and slope changes."
        ),
        parameters={
            "seed_value": 20260507,
            "n_periods": 60,
            "intervention_period": 36,
            "baseline_rate": 0.30,
            "pre_slope": 0.002,
            "level_change": -0.08,
            "slope_change": -0.004,
            "observations_per_period": 250,
        },
        runner=interrupted_time_series.run,
    ),
    DesignSpec(
        key="regression_discontinuity",
        title="Regression discontinuity",
        family="quasi-experimental",
        disciplines=("policy evaluation", "education", "economics", "clinical triage"),
        description=(
            "Sharp threshold assignment with local linear effect estimation."
        ),
        parameters={
            "seed_value": 20260507,
            "n": 2500,
            "cutoff": 0.0,
            "treatment_effect": -0.35,
            "bandwidth": 0.45,
        },
        runner=regression_discontinuity.run,
    ),
    DesignSpec(
        key="instrumental_variables",
        title="Instrumental variables",
        family="causal inference",
        disciplines=("epidemiology", "economics", "policy evaluation"),
        description="Encouragement design with first stage, reduced form, and Wald LATE.",
        parameters={
            "seed_value": 20260507,
            "n": 4000,
            "instrument_strength": 0.28,
            "treatment_effect": -0.18,
        },
        runner=instrumental_variables.run,
    ),
    DesignSpec(
        key="propensity_score",
        title="Propensity-score weighting",
        family="causal inference",
        disciplines=("epidemiology", "clinical research", "health services"),
        description="Confounded treatment assignment with inverse-probability weighting.",
        parameters={"seed_value": 20260507, "n": 3500, "treatment_effect": -0.12},
        runner=propensity_score.run,
    ),
    DesignSpec(
        key="survival_cox",
        title="Survival / Cox-style analysis",
        family="time-to-event",
        disciplines=("clinical epidemiology", "oncology", "aging research"),
        description="Time-to-event simulation with censoring and person-time hazard ratio.",
        parameters={
            "seed_value": 20260507,
            "n": 3000,
            "hazard_ratio": 0.70,
            "censoring_time": 5.0,
        },
        runner=survival_cox.run,
    ),
    DesignSpec(
        key="meta_analysis",
        title="Random-effects meta-analysis",
        family="evidence synthesis",
        disciplines=("evidence synthesis", "clinical guidelines", "public health"),
        description="Study-level effects with DerSimonian-Laird random-effects pooling.",
        parameters={"seed_value": 20260507, "n_studies": 12, "true_effect": -0.20, "tau": 0.08},
        runner=meta_analysis.run,
    ),
    DesignSpec(
        key="agent_based_seir",
        title="Agent-style SEIR",
        family="dynamic transmission",
        disciplines=("infectious disease", "public health", "systems science"),
        description="Stochastic susceptible-exposed-infectious-recovered epidemic simulation.",
        parameters={
            "seed_value": 20260507,
            "n_population": 5000,
            "n_days": 120,
            "beta": 0.32,
            "sigma": 0.20,
            "gamma": 0.12,
            "initial_infected": 15,
        },
        runner=seir.run,
    ),
    DesignSpec(
        key="microsimulation_lifetable",
        title="Life-table microsimulation",
        family="decision science",
        disciplines=("health economics", "aging", "prevention science"),
        description="Individual annual transitions for disease, mortality, costs, and QALYs.",
        parameters={"seed_value": 20260507, "n": 3000, "cycles": 20, "intervention": True},
        runner=microsimulation.run,
    ),
    DesignSpec(
        key="markov_decision",
        title="Markov decision model",
        family="decision science",
        disciplines=("health economics", "policy", "clinical decision analysis"),
        description="Three-state decision model comparing standard care and prevention.",
        parameters={"seed_value": 20260507, "cycles": 30, "discount": 0.03},
        runner=markov_decision.run,
    ),
    DesignSpec(
        key="network_contagion",
        title="Network contagion",
        family="dynamic transmission",
        disciplines=("network science", "public health", "sociology"),
        description="SIR-style contagion over a random contact network.",
        parameters={
            "seed_value": 20260507,
            "n_nodes": 600,
            "n_days": 80,
            "mean_degree": 8,
            "transmission_probability": 0.035,
            "recovery_probability": 0.12,
            "initial_infected": 8,
        },
        runner=network_contagion.run,
    ),
    DesignSpec(
        key="qualitative_mixed_methods",
        title="Qualitative / mixed-methods",
        family="mixed methods",
        disciplines=("qualitative research", "implementation science", "humanities"),
        description="Interview saturation plus quantitative survey strand integration.",
        parameters={
            "seed_value": 20260507,
            "n_interviews": 40,
            "n_survey": 500,
            "n_latent_themes": 10,
            "minimum_saturation_run": 5,
        },
        runner=qualitative_mixed_methods.run,
    ),
    DesignSpec(
        key="ecological_peai",
        title="Ecological/index-development PEAI lab",
        family="ecological",
        disciplines=("epidemiology", "aging", "health equity", "social science"),
        description=(
            "Four-phase PEAI simulation covering weighting convergence, external "
            "transportability, ascertainment bias, and fairness auditing."
        ),
        parameters={
            "seed_value": 20260507,
            "n_experts": 30,
            "n_dev": 1500,
            "n_external": 800,
            "n_prospective": 2800,
        },
        runner=ecological.run_peai,
    ),
)


def list_designs() -> tuple[DesignSpec, ...]:
    """Return the registry of implemented designs."""
    return _DESIGNS


def get_design(key: str) -> DesignSpec:
    """Resolve a design key from the registry."""
    for design in _DESIGNS:
        if design.key == key:
            return design
    raise KeyError(f"Unknown design: {key!r}")
