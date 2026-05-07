---
title: EPISIM Lab
sdk: gradio
app_file: app.py
pinned: false
---

# EPISIM Lab Space

This folder is a Hugging Face Space-ready surface for EPISIM.

## What it does

- runs implemented EPISIM designs from a shared catalog
- previews summary metrics and simulated data
- writes a deterministic archive bundle for each run

## How to deploy

1. Create a new Gradio Space on Hugging Face.
2. Upload the contents of this folder together with the `episim/` package directory, or
   sync the whole repository and point the Space to `apps/hf_space/app.py`.
3. Ensure the dependencies in `requirements.txt` are installed.
