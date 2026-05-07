"""Cross-sectional study design."""
from __future__ import annotations

import pandas as pd

from episim import __version__
from episim.core.reproducibility import Study


def run(
    df: pd.DataFrame,
    *,
    exposure: str,
    outcome: str,
    seed_value: int,
) -> Study:
    """Wrap a prepared DataFrame as a cross-sectional Study object."""
    return Study(
        seed=seed_value,
        design="cross_sectional",
        params={"n": len(df), "exposure": exposure, "outcome": outcome},
        data=df,
        library_version=__version__,
    )
