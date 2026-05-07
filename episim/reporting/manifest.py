"""Reproducibility manifest writer."""
from __future__ import annotations

from pathlib import Path

from episim.core.reproducibility import Study


def write_manifest(study: Study, path: str | Path) -> Path:
    """Write Study + manifest to a directory; return the directory."""
    return study.archive(path)
