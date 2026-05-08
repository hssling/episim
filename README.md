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

EPISIM can also run a complete simulated research workflow from a plain-language
research question:

```python
from episim.research import conduct_research

bundle = conduct_research(
    "Does a randomized lifestyle intervention reduce frailty in community elders?",
    seed_value=20260508,
    n=2_000,
)
bundle.archive("results/frailty_simulated_research")
print(bundle.plan.hypotheses)
print(bundle.report.markdown)
```

The archived research bundle includes protocol files, collected synthetic data,
cleaned analysis data, data-cleaning log, analysis-step log, a realistic
synthetic data-collection toolset, database dictionary, collection-event audit
trail, SQLite research database, analysis tables, PNG figures, guideline
checklist, references, declarations, and a structured manuscript draft.

For higher-fidelity calibration, EPISIM outputs standard CSV and SQLite assets
that can be compared with external synthetic-data or cloud data services such
as Google Cloud healthcare/FHIR tooling, Sensitive Data Protection workflows,
or Vertex AI tabular workflows. EPISIM does not send data to external services
unless a user builds and authenticates such an integration.

## Status

Alpha lab build with deterministic study archives and the full Phase-1 design registry.
Implemented surfaces currently include:

- `cross_sectional`
- `case_control`
- `cohort`
- `rct_parallel`
- `rct_cluster`
- `stepped_wedge`
- `interrupted_time_series`
- `regression_discontinuity`
- `instrumental_variables`
- `propensity_score`
- `survival_cox`
- `meta_analysis`
- `agent_based_seir`
- `microsimulation_lifetable`
- `markov_decision`
- `network_contagion`
- `qualitative_mixed_methods`
- `ecological_peai`

## Product surface

- Package API for deterministic simulation and artifact archiving
- Lab registry via `episim.lab`
- Research-question-to-report workflow via `episim.research`
- Manuscript-grade research assets: tables, figures, checklist, declarations,
  references, and synthetic research database
- 18 notebooks for the full Phase-1 design catalog
- Hugging Face Space app in `apps/hf_space/`
- Kaggle publishing metadata in `platforms/kaggle/`
- Dockerfile, Zenodo metadata, and JOSS paper skeleton

## Roadmap

| Phase | Timing | Contents |
|---|---|---|
| **1** | Weeks 1-8 | Full 18-design teaching and methods lab |
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
