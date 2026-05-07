"""Markov decision-analytic model simulation."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def _run_strategy(
    transition: npt.NDArray[np.float64],
    *,
    cycles: int,
    discount: float,
    costs: npt.NDArray[np.float64],
    utilities: npt.NDArray[np.float64],
) -> tuple[pd.DataFrame, float, float]:
    state = np.array([1.0, 0.0, 0.0])
    rows = []
    total_cost = 0.0
    total_qaly = 0.0
    for cycle in range(cycles):
        weight = 1 / ((1 + discount) ** cycle)
        total_cost += float(state @ costs) * weight
        total_qaly += float(state @ utilities) * weight
        rows.append(
            {
                "cycle": cycle,
                "healthy": state[0],
                "diseased": state[1],
                "dead": state[2],
            }
        )
        state = state @ transition
    return pd.DataFrame(rows), total_cost, total_qaly


def run(
    *,
    seed_value: int,
    cycles: int = 30,
    discount: float = 0.03,
) -> Study:
    """Compare standard care with prevention using a three-state Markov model."""
    standard = np.array(
        [[0.90, 0.08, 0.02], [0.00, 0.88, 0.12], [0.00, 0.00, 1.00]]
    )
    prevention = np.array(
        [[0.93, 0.05, 0.02], [0.00, 0.90, 0.10], [0.00, 0.00, 1.00]]
    )
    utilities = np.array([0.90, 0.68, 0.0])
    standard_costs = np.array([600.0, 3500.0, 0.0])
    prevention_costs = np.array([950.0, 3500.0, 0.0])
    std_path, std_cost, std_qaly = _run_strategy(
        standard,
        cycles=cycles,
        discount=discount,
        costs=standard_costs,
        utilities=utilities,
    )
    int_path, int_cost, int_qaly = _run_strategy(
        prevention,
        cycles=cycles,
        discount=discount,
        costs=prevention_costs,
        utilities=utilities,
    )
    inc_cost = int_cost - std_cost
    inc_qaly = int_qaly - std_qaly
    trajectory = pd.concat(
        [std_path.assign(strategy="standard"), int_path.assign(strategy="prevention")],
        ignore_index=True,
    )
    return build_study(
        design="markov_decision",
        seed_value=seed_value,
        data=trajectory,
        params={"cycles": cycles, "discount": discount},
        results={
            "standard_cost": round(std_cost, 2),
            "standard_qaly": round(std_qaly, 3),
            "prevention_cost": round(int_cost, 2),
            "prevention_qaly": round(int_qaly, 3),
            "incremental_cost": round(inc_cost, 2),
            "incremental_qaly": round(inc_qaly, 3),
            "icer": round(inc_cost / inc_qaly, 2),
        },
        artifacts={"state_trace": trajectory},
    )
