"""Tests for episim.designs.cross_sectional."""
from episim.core import cohort, logistic, seed
from episim.core.reproducibility import Study
from episim.designs import cross_sectional


def test_cross_sectional_returns_study_object() -> None:
    with seed(42) as rng:
        df = cohort(n=1_000, age=("normal", 50, 10), rng=rng)
        df["exposure"] = rng.binomial(1, 0.3, 1_000)
        df = logistic(df, "y ~ exposure + age",
                      betas={"exposure": 0.6, "age": 0.04}, rng=rng)
        study = cross_sectional.run(df, exposure="exposure", outcome="y",
                                    seed_value=42)
    assert isinstance(study, Study)
    assert study.design == "cross_sectional"
    assert study.params["n"] == 1_000
    assert study.params["exposure"] == "exposure"
    assert study.params["outcome"] == "y"
