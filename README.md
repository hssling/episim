# EPISIM - Epidemiology Platform for In Silico Methods

[![CI](https://github.com/hssling/episim/actions/workflows/ci.yml/badge.svg)](https://github.com/hssling/episim/actions/workflows/ci.yml)
[![Docs](https://github.com/hssling/episim/actions/workflows/docs.yml/badge.svg)](https://hssling.github.io/episim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Open-source, reproducible simulation lab for epidemiological, clinical, public-health,
and allied-science study designs.

```bash
pip install episim
```

```python
from episim.lab import run_design

study = run_design("cohort", seed_value=20260507, n=4_000)
study.archive("results/cohort_demo")
```

## Status

Alpha lab build with deterministic study archives and a growing design registry.
Implemented surfaces currently include:

- `cross_sectional`
- `case_control`
- `cohort`
- `rct_parallel`
- `ecological_peai`

## Product surface

- Package API for deterministic simulation and artifact archiving
- Lab registry via `episim.lab`
- Notebooks for implemented designs
- Hugging Face Space app in `apps/hf_space/`
- Kaggle publishing metadata in `platforms/kaggle/`

## Roadmap

| Phase | Timing | Contents |
|---|---|---|
| **1** | Weeks 1-8 | Expand the registry toward the full 18-design teaching and methods lab |
| **2** | Weeks 9-14 | Pre-registration / methods-research lab with power, bias, design-comparison, and fairness audit packages |
| **3** | Months 4-6 | Generative AI simulation engine: natural-language to EPISIM `Study` to manuscript draft |
| **4** | Months 7-9 | Beyond medicine: educational, economic, psychological, sociological, and qualitative designs |

## Licence

MIT (code), CC-BY 4.0 (reporting checklists and figure templates).

## Citation

```text
Siddalingaiah H. S. EPISIM: Epidemiology Platform for In Silico Methods.
Alpha lab build (2026). https://github.com/hssling/episim
```
