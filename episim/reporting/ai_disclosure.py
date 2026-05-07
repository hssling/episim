"""AI-disclosure block templates."""
from __future__ import annotations


def block(mode: str, version: str, doi: str, llm: str | None = None) -> str:
    """Return a templated AI-disclosure paragraph for a manuscript.

    mode = 'library_only' (no LLM in the analysis) or 'generative'
    (LLM-mediated parameter extraction).
    """
    if mode == "library_only":
        return (
            f"Simulation conducted using EPISIM v{version} ({doi}). "
            "No generative-AI was used in design, analysis, or interpretation. "
            "All decisions and final wording were made and verified by the author."
        )
    if mode == "generative":
        llm_name = llm or "(unspecified LLM)"
        return (
            "Simulation parameters were specified in natural language and parsed by "
            f"EPISIM v{version}'s generative layer ({doi}; LLM: {llm_name}). "
            "All design, parameter, and interpretation decisions were verified by the "
            "author against the deterministic re-run from the saved seed."
        )
    raise ValueError(f"Unknown disclosure mode: {mode!r}")
