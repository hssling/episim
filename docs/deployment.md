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

Minimal path:

```bash
cd apps/hf_space
python app.py
```

## Kaggle

- public benchmark bundles
- shareable notebooks and example outputs
- packaging assets live in `platforms/kaggle/`
