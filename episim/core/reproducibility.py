"""Reproducibility primitives: seeded RNG, Study metadata, SHA-256 manifest."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@contextmanager
def seed(value: int) -> Generator[np.random.Generator, None, None]:
    """Yield a seeded numpy Generator deterministic for the given seed."""
    rng = np.random.default_rng(value)
    yield rng


def sha256_dataframe(df: pd.DataFrame) -> str:
    """Stable SHA-256 of a DataFrame's bytes (sorted, str-coerced columns)."""
    cols = sorted(df.columns, key=str)
    out = df[cols].copy()
    out.columns = pd.Index([str(c) for c in cols])
    return hashlib.sha256(out.to_csv(index=False).encode("utf-8")).hexdigest()


@dataclass
class Study:
    """A simulation result: the data plus full reproducibility metadata."""

    seed: int
    design: str
    params: dict[str, Any]
    data: pd.DataFrame
    library_version: str
    results: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, pd.DataFrame] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.UTC).isoformat()
    )

    def archive(self, path: str | Path) -> Path:
        """Write data + manifest to a directory; return the directory."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(out / "data.csv", index=False)
        if self.results:
            (out / "results.json").write_text(
                json.dumps(self.results, indent=2, default=str)
            )
        artifact_entries: list[dict[str, Any]] = []
        if self.artifacts:
            artifact_dir = out / "artifacts"
            artifact_dir.mkdir(exist_ok=True)
            for name, frame in self.artifacts.items():
                slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "artifact"
                filename = f"{slug}.csv"
                frame.to_csv(artifact_dir / filename, index=False)
                artifact_entries.append(
                    {
                        "name": name,
                        "file": f"artifacts/{filename}",
                        "rows": len(frame),
                        "columns": list(frame.columns),
                        "sha256": sha256_dataframe(frame),
                    }
                )
        manifest = {
            "seed": self.seed,
            "design": self.design,
            "params": self.params,
            "library_version": self.library_version,
            "timestamp": self.timestamp,
            "data_sha256": sha256_dataframe(self.data),
            "n_rows": len(self.data),
            "columns": list(self.data.columns),
            "results_file": "results.json" if self.results else None,
            "artifacts": artifact_entries,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        return out
