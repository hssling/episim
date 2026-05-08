"""Helpers for running EPISIM lab experiments."""
from __future__ import annotations

from inspect import Parameter, signature
from typing import Any

import pandas as pd

from episim.core.reproducibility import Study
from episim.lab.registry import DesignSpec, get_design


def resolve_design_parameters(design: DesignSpec, overrides: dict[str, Any]) -> dict[str, Any]:
    """Return defaults plus runner-compatible overrides for a registered design."""
    params = dict(design.parameters)
    accepted = _accepted_parameter_names(design)
    params.update({key: value for key, value in overrides.items() if key in accepted})
    return params


def run_design(key: str, **overrides: Any) -> Study:
    """Run a registered design with parameter overrides."""
    design = get_design(key)
    params = resolve_design_parameters(design, overrides)
    return design.runner(**params)


def study_preview(study: Study, rows: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact data and result previews for UI/reporting surfaces."""
    data_preview = study.data.head(rows).copy()
    if study.results:
        result_preview = pd.json_normalize(study.results, sep=".")
    else:
        result_preview = pd.DataFrame({"message": ["No summary metrics recorded."]})
    return data_preview, result_preview


def _accepted_parameter_names(design: DesignSpec) -> set[str]:
    sig = signature(design.runner)
    if any(param.kind == Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return set(design.parameters)
    return {
        name
        for name, param in sig.parameters.items()
        if param.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
    }
