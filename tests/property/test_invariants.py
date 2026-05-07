"""Hypothesis-based invariants on EPISIM core primitives."""
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from episim.core import cohort, seed
from episim.core.attrition import apply_mcar


@given(n=st.integers(min_value=10, max_value=2_000))
@settings(max_examples=20, deadline=None)
def test_cohort_length_equals_n(n: int) -> None:
    with seed(7) as rng:
        df = cohort(n=n, rng=rng)
    assert len(df) == n
    assert "id" in df.columns


@given(
    n=st.integers(min_value=200, max_value=2_000),
    p=st.floats(min_value=0.0, max_value=0.9, exclude_min=False),
)
@settings(max_examples=20, deadline=None)
def test_mcar_kept_fraction_close_to_one_minus_p(n: int, p: float) -> None:
    df = pd.DataFrame({"id": range(n)})
    with seed(42) as rng:
        kept = apply_mcar(df, p_drop=p, rng=rng)
    expected = (1 - p) * n
    tol = max(50, int(0.10 * n))
    assert abs(len(kept) - expected) <= tol
