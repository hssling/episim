# EPISIM — Epidemiology Platform for In Silico Methods

[![CI](https://github.com/hssling/episim/actions/workflows/ci.yml/badge.svg)](https://github.com/hssling/episim/actions/workflows/ci.yml)
[![Docs](https://github.com/hssling/episim/actions/workflows/docs.yml/badge.svg)](https://hssling.github.io/episim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Open-source, reproducible simulation of epidemiological and allied-science research designs.

```bash
pip install episim
```

```python
from episim.core import cohort, logistic, seed
from episim.designs import cross_sectional

with seed(20260507) as rng:
    pop = cohort(n=1_000, age=("normal", 50, 12), sex=("bernoulli", 0.5), rng=rng)
    pop["exposure"] = rng.binomial(1, 0.3, 1_000)
    pop = logistic(pop, "y ~ exposure + age",
                   betas={"exposure": 0.6, "age": 0.04}, intercept=-3.0, rng=rng)
    study = cross_sectional.run(pop, exposure="exposure", outcome="y", seed_value=20260507)

study.archive("results/")
```

## Status

**v0.1.0-alpha** — Foundation release. Week 1 of an 8-week roadmap covering 18 study-design notebooks (Phase 1 — teaching/methods-demonstration lab). See [the spec](https://github.com/hssling/episim/blob/main/docs/superpowers/specs) and [implementation plan](https://github.com/hssling/episim/blob/main/docs/superpowers/plans).

## Roadmap

| Phase | Timing | Contents |
|---|---|---|
| **1** | Weeks 1–8 | Teaching/methods-demonstration lab — 18 notebooks (cross-sectional, case-control, cohort, RCT parallel, RCT cluster, stepped-wedge, ecological, ITS, regression-discontinuity, IV, propensity-score, Cox, meta-analysis, agent-based SEIR, microsimulation, Markov decision-analytic, network, qualitative/mixed-methods) |
| **2** | Weeks 9–14 | Pre-registration / methods-research lab — power + bias + design-comparison + fairness audit emitted as an OSF-depositable package |
| **3** | Months 4–6 | Generative-AI simulation engine — natural-language → EPISIM `Study` → manuscript draft, with audit-trail and deterministic re-run |
| **4** | Months 7–9 | Beyond medicine — educational, economic, psychological, sociological/qualitative designs |

## Licence

MIT (code), CC-BY 4.0 (reporting checklists and figure templates).

## Citation

```
Siddalingaiah H. S. EPISIM: Epidemiology Platform for In Silico Methods.
Version 0.1.0-alpha (2026). https://github.com/hssling/episim
```
