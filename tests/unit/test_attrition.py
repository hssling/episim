"""Tests for episim.core.attrition."""
import numpy as np
import pandas as pd

from episim.core.attrition import apply_mar, apply_mcar, apply_mnar
from episim.core.reproducibility import seed


def test_mcar_drops_approximately_p() -> None:
    df = pd.DataFrame({"id": np.arange(10_000), "y": np.zeros(10_000)})
    with seed(42) as rng:
        kept = apply_mcar(df, p_drop=0.10, rng=rng)
    assert 8_700 < len(kept) < 9_300


def test_mar_drops_more_for_higher_x() -> None:
    df = pd.DataFrame({"x": np.concatenate([np.zeros(5_000), np.ones(5_000) * 3])})
    with seed(42) as rng:
        kept = apply_mar(df, formula="1 / (1 + exp(-x))", rng=rng)
    low = (kept["x"] == 0).sum()
    high = (kept["x"] == 3).sum()
    assert low > high


def test_mnar_drops_more_for_higher_y() -> None:
    df = pd.DataFrame({"y": np.concatenate([np.zeros(5_000), np.ones(5_000)])})
    with seed(42) as rng:
        kept = apply_mnar(df, outcome="y", p_drop_if_y=0.7, p_drop_if_not_y=0.05, rng=rng)
    assert (kept["y"] == 0).sum() > (kept["y"] == 1).sum()
