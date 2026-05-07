"""Per-design Gradio wrapper for EPISIM Spaces."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from episim.lab import get_design, run_design, study_preview  # noqa: E402

DESIGN_KEY = os.environ.get("EPISIM_DESIGN", "cross_sectional")


def _defaults() -> str:
    return json.dumps(get_design(DESIGN_KEY).parameters, indent=2)


def _run(overrides_json: str) -> tuple[object, object]:
    params = json.loads(overrides_json) if overrides_json.strip() else {}
    study = run_design(DESIGN_KEY, **params)
    data, results = study_preview(study)
    return results, data


with gr.Blocks(title=f"EPISIM {DESIGN_KEY}") as demo:
    gr.Markdown(f"# EPISIM: `{DESIGN_KEY}`")
    params = gr.Code(value=_defaults(), language="json", label="Parameters")
    run = gr.Button("Run", variant="primary")
    results = gr.Dataframe(label="Results")
    data = gr.Dataframe(label="Data preview")
    run.click(_run, inputs=params, outputs=[results, data])


if __name__ == "__main__":
    demo.launch()
