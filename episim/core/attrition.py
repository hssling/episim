"""Attrition mechanisms: MCAR, MAR, MNAR."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_mcar(
    df: pd.DataFrame, p_drop: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Drop rows uniformly at random with probability p_drop."""
    keep = rng.random(len(df)) >= p_drop
    return df.loc[keep].reset_index(drop=True)


def apply_mar(
    df: pd.DataFrame, formula: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Drop rows with probability p computed from a Python expression of column names.

    Example: ``1 / (1 + exp(-(0.5 * age - 25)))``. The expression is evaluated
    in a namespace exposing every column plus ``np`` and ``exp/log`` as their
    numpy counterparts.
    """
    ns: dict[str, object] = {c: df[c].to_numpy() for c in df.columns}
    ns.update({"np": np, "exp": np.exp, "log": np.log})
    p = eval(formula, {"__builtins__": {}}, ns)  # noqa: S307
    keep = rng.random(len(df)) >= np.asarray(p, dtype=float)
    return df.loc[keep].reset_index(drop=True)


def apply_mnar(
    df: pd.DataFrame,
    outcome: str,
    p_drop_if_y: float,
    p_drop_if_not_y: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """MNAR: drop probability depends on the outcome itself."""
    p = np.where(df[outcome].to_numpy() == 1, p_drop_if_y, p_drop_if_not_y)
    keep = rng.random(len(df)) >= p
    return df.loc[keep].reset_index(drop=True)
