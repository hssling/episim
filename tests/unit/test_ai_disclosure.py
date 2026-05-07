"""Tests for AI disclosure block."""
import pytest

from episim.reporting.ai_disclosure import block


def test_library_only_mode() -> None:
    text = block(mode="library_only", version="0.1.0a1",
                 doi="10.5281/zenodo.placeholder")
    assert "EPISIM v0.1.0a1" in text
    assert "10.5281/zenodo.placeholder" in text
    assert "No generative-AI" in text


def test_generative_mode() -> None:
    text = block(mode="generative", version="0.1.0a1",
                 doi="10.5281/zenodo.placeholder", llm="Claude Opus 4.7")
    assert "Claude Opus 4.7" in text
    assert "EPISIM" in text


def test_generative_mode_default_llm() -> None:
    text = block(mode="generative", version="0.1.0a1",
                 doi="10.5281/zenodo.placeholder")
    assert "(unspecified LLM)" in text


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        block(mode="alien", version="0.1.0a1", doi="x")
