"""Combined Gradio lab for EPISIM experiments."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

APP_PATH = Path(__file__).resolve()
ROOT = next(
    (
        candidate
        for candidate in (APP_PATH.parent, *APP_PATH.parents)
        if (candidate / "episim").is_dir()
    ),
    APP_PATH.parent,
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from episim.lab import get_design, list_designs, run_design, study_preview  # noqa: E402


def _catalog_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for design in list_designs():
        rows.append(
            [
                design.key,
                design.title,
                design.family,
                ", ".join(design.disciplines),
            ]
        )
    return rows


def _design_defaults(key: str) -> tuple[str, str]:
    design = get_design(key)
    description = (
        f"**{design.title}**\n\n"
        f"{design.description}\n\n"
        f"- Family: `{design.family}`\n"
        f"- Disciplines: `{', '.join(design.disciplines)}`"
    )
    return description, json.dumps(design.parameters, indent=2)


def _run_experiment(
    key: str, overrides_json: str
) -> tuple[Any, Any, Any, str, str]:
    overrides = json.loads(overrides_json) if overrides_json.strip() else {}
    study = run_design(key, **overrides)
    data_preview, result_preview = study_preview(study)

    out_dir = Path(tempfile.mkdtemp(prefix=f"episim_{key}_"))
    study.archive(out_dir)
    archive_base = out_dir / "bundle"
    bundle = shutil.make_archive(str(archive_base), "zip", root_dir=out_dir)

    artifact_rows: list[dict[str, Any]] = []
    for name, frame in study.artifacts.items():
        artifact_rows.append(
            {
                "artifact": name,
                "rows": len(frame),
                "columns": ", ".join(str(col) for col in frame.columns),
            }
        )
    artifact_preview = pd.DataFrame(
        artifact_rows or [{"artifact": "none", "rows": 0, "columns": ""}]
    )
    summary = (
        f"Completed `{study.design}` with seed `{study.seed}`.\n\n"
        f"- Records: `{len(study.data)}`\n"
        f"- Summary metrics stored: `{bool(study.results)}`\n"
        f"- Artifact tables: `{len(study.artifacts)}`"
    )
    return (
        result_preview,
        data_preview,
        artifact_preview,
        summary + f"\n\nArchive ready: `{Path(bundle).name}`",
        bundle,
    )


with gr.Blocks(title="EPISIM Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # EPISIM Lab

        Reproducible simulation console for observational, experimental, and ecological
        study designs in medicine and allied sciences.
        """
    )
    with gr.Row():
        design = gr.Dropdown(
            choices=[spec.key for spec in list_designs()],
            value="cross_sectional",
            label="Design",
        )
        run_button = gr.Button("Run experiment", variant="primary")
    description = gr.Markdown()
    overrides = gr.Code(label="Parameters (JSON)", language="json")

    with gr.Tab("Catalog"):
        gr.Dataframe(
            headers=["key", "title", "family", "disciplines"],
            value=_catalog_rows(),
            interactive=False,
            wrap=True,
        )
    with gr.Tab("Results"):
        result_table = gr.Dataframe(label="Summary metrics", interactive=False, wrap=True)
        data_preview = gr.Dataframe(label="Data preview", interactive=False, wrap=True)
        artifact_preview = gr.Dataframe(label="Artifacts", interactive=False, wrap=True)
        status = gr.Markdown()
        bundle_file = gr.File(label="Archive bundle")

    design.change(fn=_design_defaults, inputs=design, outputs=[description, overrides])
    run_button.click(
        fn=_run_experiment,
        inputs=[design, overrides],
        outputs=[result_table, data_preview, artifact_preview, status, bundle_file],
    )
    demo.load(fn=_design_defaults, inputs=design, outputs=[description, overrides])


if __name__ == "__main__":
    demo.launch()
