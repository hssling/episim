# Deployment Surfaces

EPISIM is intended to publish on three complementary surfaces.

## GitHub

- canonical source repo
- CI, docs, releases, tags
- notebook and manuscript provenance

Use `PUSH_TO_GITHUB.md` for the initial public push.

## Hugging Face Spaces

- public interactive experiment console
- suitable for teaching demos and methods workshops
- implemented surface: `apps/hf_space/`
- per-design wrapper: `apps/per_design/`

Minimal path:

```bash
cd apps/hf_space
python app.py
```

## Kaggle

- public benchmark bundles
- shareable notebooks and example outputs
- packaging assets live in `platforms/kaggle/`

## Docker

```bash
docker build -t episim:0.8-alpha .
docker run -p 7860:7860 episim:0.8-alpha
```

## Zenodo and JOSS

- Zenodo metadata: `.zenodo.json`
- JOSS skeleton: `paper.md`, `paper.bib`
