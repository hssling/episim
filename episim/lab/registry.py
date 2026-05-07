"""Registry of implemented EPISIM lab designs."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from episim.core.reproducibility import Study
from episim.designs import case_control, cohort, cross_sectional, ecological, rct_parallel


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
