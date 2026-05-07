"""Tests for episim.reporting.manifest."""
import json
from pathlib import Path

import pandas as pd

from episim.core.reproducibility import Study
from episim.reporting.manifest import write_manifest


def test_write_manifest_writes_to_path(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "y": [0, 1, 0]})
    study = Study(seed=1, design="cross_sectional", params={"n": 3},
                  data=df, library_version="0.1.0a1",
                  results={"summary": {"n": 3}})
    out = write_manifest(study, tmp_path / "run")
    assert out.is_dir()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == 1
    assert manifest["n_rows"] == 3
    assert (out / "results.json").exists()
