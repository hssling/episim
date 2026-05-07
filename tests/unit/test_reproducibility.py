"""Tests for episim.core.reproducibility."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from episim.core.reproducibility import Study, seed, sha256_dataframe


def test_seed_returns_generator() -> None:
    with seed(42) as rng:
        assert isinstance(rng, np.random.Generator)


def test_seed_is_deterministic() -> None:
    with seed(42) as rng1:
        a = rng1.integers(0, 100, 10)
    with seed(42) as rng2:
        b = rng2.integers(0, 100, 10)
    assert (a == b).all()


def test_sha256_dataframe_is_stable() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    h1 = sha256_dataframe(df)
    h2 = sha256_dataframe(df.copy())
    assert h1 == h2
    assert len(h1) == 64


def test_study_archive_writes_manifest(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2]})
    study = Study(
        seed=42, design="cross_sectional",
        params={"n": 2},
        data=df,
        library_version="0.1.0a1",
        results={"summary": {"metric": 1.23}},
        artifacts={"table 1": pd.DataFrame({"metric": ["x"], "value": [1.23]})},
    )
    study.archive(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 42
    assert manifest["design"] == "cross_sectional"
    assert manifest["library_version"] == "0.1.0a1"
    assert "data_sha256" in manifest
    assert manifest["results_file"] == "results.json"
    assert len(manifest["artifacts"]) == 1
    assert (tmp_path / "data.csv").exists()
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "artifacts" / "table_1.csv").exists()
