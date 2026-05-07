"""Helpers for running EPISIM lab experiments."""
from __future__ import annotations

from typing import Any

import pandas as pd

from episim.core.reproducibility import Study
from episim.lab.registry import get_design


def run_design(key: str, **overrides: Any) -> Study:
    """Run a registered design with parameter overrides."""
    design = get_design(key)
    params = dict(design.parameters)
    params.update(overrides)
    return design.runner(**params)


def study_preview(study: Study, rows: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact data and result previews for UI/reporting surfaces."""
    data_preview = study.data.head(rows).copy()
    if study.results:
        result_preview = pd.json_normalize(study.results, sep=".")
    else:
        result_preview = pd.DataFrame({"message": ["No summary metrics recorded."]})
    return data_preview, result_preview
