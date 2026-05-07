"""Shared helpers for study-design modules."""
from __future__ import annotations

from typing import Any

import pandas as pd

from episim import __version__
from episim.core.reproducibility import Study


def build_study(
    *,
    design: str,
    seed_value: int,
    data: pd.DataFrame,
    params: dict[str, Any],
    results: dict[str, Any] | None = None,
    artifacts: dict[str, pd.DataFrame] | None = None,
) -> Study:
    """Build a Study object with consistent metadata across designs."""
    return Study(
        seed=seed_value,
        design=design,
        params=params,
        data=data,
        library_version=__version__,
        results=results or {},
        artifacts=artifacts or {},
    )
