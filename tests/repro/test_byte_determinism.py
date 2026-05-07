"""Byte-level determinism: same seed -> same SHA-256 of output."""
import json
from pathlib import Path

import pandas as pd

from episim import __version__
from episim.core import cohort, logistic, seed
from episim.core.reproducibility import Study, sha256_dataframe
from episim.designs import cross_sectional


def _run(seed_value: int) -> pd.DataFrame:
    with seed(seed_value) as rng:
        df = cohort(n=500, age=("normal", 50, 10), rng=rng)
        df["exposure"] = rng.binomial(1, 0.3, 500)
        df = logistic(df, "y ~ exposure + age",
                      betas={"exposure": 0.6, "age": 0.04}, rng=rng)
        return cross_sectional.run(df, exposure="exposure", outcome="y",
                                   seed_value=seed_value).data


def test_same_seed_gives_identical_sha256() -> None:
    h1 = sha256_dataframe(_run(20260507))
    h2 = sha256_dataframe(_run(20260507))
    assert h1 == h2


def test_different_seeds_give_different_sha256() -> None:
    h1 = sha256_dataframe(_run(20260507))
    h2 = sha256_dataframe(_run(99999))
    assert h1 != h2


def test_archive_writes_consistent_manifest(tmp_path: Path) -> None:
    df = _run(20260507)
    study = Study(seed=20260507, design="cross_sectional", params={"n": 500},
                  data=df, library_version=__version__)
    out = study.archive(tmp_path)
    m = json.loads((out / "manifest.json").read_text())
    assert m["data_sha256"] == sha256_dataframe(df)
