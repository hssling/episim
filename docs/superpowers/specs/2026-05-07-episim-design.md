# EPISIM — Epidemiology Platform for In Silico Methods

**Design specification.**
**Version:** 1.0 — 7 May 2026.
**Author:** Dr Siddalingaiah H. S. (with frontier-AI assistance, fully disclosed).
**Status:** Approved through Section 6 of brainstorming; pending user review of this spec; then implementation-plan stage.

---

## 1. One-line summary

EPISIM is an open-source Python platform for **in silico simulation of epidemiological and allied-science research designs**, deployed across **seven surfaces** from one source-of-truth: (i) a GitHub repository, (ii) a pip-installable library (`pip install episim`), (iii) a curated library of 18 study-design notebooks with deterministic outputs and auto-generated reporting-checklist artefacts, (iv) a Hugging Face Spaces web app, (v) Kaggle-mirrored notebooks, (vi) a Zenodo-DOI'd citable release on every tag, and (vii) a GHCR Docker image and MkDocs documentation site. It grows in three phases: (1) teaching/methods-demonstration lab, (2) pre-registration / methods-research lab, (3) natural-language generative simulation engine.

## 2. Goals and non-goals

### Goals

* Enable any researcher to specify and run a deterministic simulation of any of 18 (initially) study designs in < 30 lines of code or in a web UI.
* Make every simulation output a **citable, reviewable, publication-grade artefact** — deterministic seed, SHA-256-manifested outputs, auto-filled reporting checklist (STROBE / CONSORT / TRIPOD-AI / STRESS / PRISMA / GRIPP2 / SAGER), and a methods-narrative auto-draft.
* Build the platform across five surfaces (GitHub, PyPI, HF Spaces, Kaggle, Zenodo, MkDocs, GHCR Docker) from one source-of-truth without duplicated maintenance.
* Be **demonstrably useful from week 2** (combined HF Space live with 1–2 notebooks) and **fully shipped as v0.8 with JOSS submission by week 8**.
* Lay clean architectural seams for phase 2 (pre-registration mode) and phase 3 (LLM-mediated natural-language interface) without rework.
* Extend, by month 9, beyond medicine into educational, economic, psychological, sociological/qualitative research designs — fulfilling the "allied sciences and humanities" remit.

### Non-goals (v0.x)

* Not a frontier-AI agent that reasons about epidemiology autonomously (that arrives in phase 3).
* Not a wet-lab data-analysis platform (no real-data ingest from EHRs / surveys; users bring their own).
* Not a competitor to broad ML platforms (PyMC, Stan); EPISIM specifically targets *study-design simulation* with reporting-checklist outputs.
* Not a meta-analysis or systematic-review automation tool (although a meta-analysis simulation notebook is included).

## 3. Architecture

### 3.1 Repository layout

```
EPISIM/                                 ← GitHub source-of-truth
├── episim/                             ← pip-installable Python package
│   ├── __init__.py
│   ├── core/                           ← shared primitives
│   │   ├── populations.py              ← cohort/population generators
│   │   ├── exposures.py                ← exposure assignment
│   │   ├── outcomes.py                 ← outcome models (logistic / Cox / count / continuous)
│   │   ├── confounding.py              ← confounder + DAG utilities
│   │   ├── bias.py                     ← measurement error, ascertainment, selection
│   │   ├── attrition.py                ← MCAR / MAR / MNAR drop-out
│   │   └── reproducibility.py          ← seeded RNG, SHA-256 manifest, Study object
│   ├── designs/                        ← 18 study-design modules
│   │   ├── cross_sectional.py
│   │   ├── case_control.py
│   │   ├── cohort.py
│   │   ├── rct_parallel.py
│   │   ├── rct_cluster.py
│   │   ├── stepped_wedge.py
│   │   ├── ecological.py               ← PEAI lives here
│   │   ├── interrupted_ts.py
│   │   ├── regression_discontinuity.py
│   │   ├── instrumental_variables.py
│   │   ├── propensity_score.py
│   │   ├── survival_cox.py
│   │   ├── meta_analysis.py
│   │   ├── agent_based_seir.py
│   │   ├── microsimulation_lifetable.py
│   │   ├── markov_decision.py
│   │   ├── network_contagion.py
│   │   └── qualitative_mixed_methods.py
│   ├── analytics/
│   │   ├── estimators.py               ← point + bootstrap CI (percentile + BCa)
│   │   ├── multiple_testing.py         ← BH-FDR, Bonferroni
│   │   ├── calibration.py              ← decile plot + slope + in-the-large
│   │   ├── discrimination.py           ← AUROC, NRI, decision-curve
│   │   ├── fairness.py                 ← equalised odds, predictive parity, recalibration
│   │   ├── transportability.py         ← Pearl-Bareinboim style
│   │   └── sensitivity.py              ← E-value, tipping-point
│   ├── reporting/
│   │   ├── strobe.py
│   │   ├── consort.py
│   │   ├── tripod_ai.py
│   │   ├── stress.py
│   │   ├── prisma.py
│   │   ├── gripp2.py
│   │   ├── sager.py
│   │   ├── narrative.py                ← deterministic methods-narrative builder
│   │   ├── manifest.py                 ← reproducibility manifest writer
│   │   ├── ai_disclosure.py            ← two-flavour AI-disclosure block
│   │   └── checklists/                 ← YAML files (community-contributable)
│   │       ├── strobe.yaml
│   │       ├── consort.yaml
│   │       ├── tripod_ai.yaml
│   │       ├── stress.yaml
│   │       ├── prisma.yaml
│   │       ├── gripp2.yaml
│   │       └── sager.yaml
│   └── viz/                            ← shared figure templates
│       ├── consort_flow.py
│       ├── forest.py
│       ├── calibration.py
│       ├── fairness.py
│       └── dag.py
├── notebooks/                          ← 18 thin notebooks
│   ├── 01_cross_sectional.ipynb
│   ├── 02_case_control.ipynb
│   ├── 03_cohort.ipynb
│   ├── 04_rct_parallel.ipynb
│   ├── 05_rct_cluster.ipynb
│   ├── 06_stepped_wedge.ipynb
│   ├── 07_ecological.ipynb             ← PEAI notebook
│   ├── 08_interrupted_ts.ipynb
│   ├── 09_regression_discontinuity.ipynb
│   ├── 10_instrumental_variables.ipynb
│   ├── 11_propensity_score.ipynb
│   ├── 12_survival_cox.ipynb
│   ├── 13_meta_analysis.ipynb
│   ├── 14_agent_based_seir.ipynb
│   ├── 15_microsimulation_lifetable.ipynb
│   ├── 16_markov_decision.ipynb
│   ├── 17_network_contagion.ipynb
│   └── 18_qualitative_mixed_methods.ipynb
├── apps/                               ← combined HF Space (v0.1 → v0.4) + per-design (v0.5+)
│   ├── episim_lab/                     ← combined Gradio app with 18-design dropdown
│   └── per_design/                     ← per-design Gradio apps from v0.5
├── tests/                              ← pytest suite (unit + property + reproducibility + nbmake)
├── docs/                               ← MkDocs source → GitHub Pages
├── .github/workflows/
│   ├── ci.yml                          ← lint + tests + nbmake on PR
│   ├── release.yml                     ← PyPI + Docker + Zenodo on tag
│   ├── docs.yml                        ← MkDocs → GH Pages on main
│   ├── hf-sync.yml                     ← HF Spaces app sync on main
│   └── kaggle-sync.yml                 ← Kaggle mirror on tag
├── pyproject.toml
├── requirements.lock
├── Dockerfile
├── README.md
├── CITATION.cff
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
└── LICENSE                             ← MIT (code) + CC-BY 4.0 (checklists/figures)
```

### 3.2 The five-surface deployment

| Surface | Audience | Artefact | Trigger |
|---|---|---|---|
| GitHub `hssling/episim` | Methodologists, contributors | Source + notebooks + docs | `git push` (canonical) |
| PyPI `episim` | Library users | Python package | CI on `v*` tag |
| HF Spaces `episim-lab` | Non-technical users | Gradio web app | CI on merge to main |
| Kaggle `episim/<design>` | Learners, MD/MPH students | Notebook mirrors | CI on `v*` tag |
| Zenodo | Citers | DOI'd source archive | GH-Zenodo on `v*` tag |
| MkDocs (GH Pages) | Doc readers | API + tutorial site | CI on merge to main |
| GHCR Docker `episim:v*` | Reproducibility-critical users | Pinned environment image | CI on `v*` tag |

## 4. Primitive API contract

### 4.1 Style

* **R-style formulas** parsed via `formulaic`. Users write `"y ~ exposure + age + sex + age:sex"`.
* **Imperative pipeline.** Each function returns a new `pandas.DataFrame` plus a `Study` metadata object.
* **Thin OO wrapper** in `episim.designs.Builder` so each method becomes a Gradio widget for app generation.
* **Pure functions, no global state.** Every randomness driven by an explicit `rng` argument or via the `seed()` context manager.
* **Universal data type:** `pandas.DataFrame` with conventional column names: `id`, `exposure`, `outcome`, plus user-named covariates.
* **Failures explicit.** Underpowered designs return `power_warning=True`; never silent.

### 4.2 Canonical user-facing example

```python
from episim.core import populations, outcomes, bias, reproducibility
from episim.designs import case_control
from episim.analytics import estimators, fairness
from episim.reporting import strobe, narrative, manifest, ai_disclosure

with reproducibility.seed(20260507) as rng:
    pop = populations.cohort(n=10_000, age=("normal", 50, 12), sex=("bernoulli", 0.5), rng=rng)
    pop = outcomes.logistic(pop, "y ~ exposure + age + sex", betas={"exposure": 0.6, "age": 0.04, "sex": 0.2})
    pop = bias.measurement_error(pop, on="exposure", sd=0.2, rng=rng)

    study = case_control.run(pop, n_cases=500, n_controls=1000, matched_on=["age", "sex"], rng=rng)
    or_est = estimators.odds_ratio(study, with_ci="bca")
    fair = fairness.equalised_odds(study, by="sex", threshold=0.10)

    strobe.checklist(study).save("strobe.docx")
    narrative.draft(study).save("methods.md")
    ai_disclosure.block(mode="library_only").save("ai_disclosure.md")
    manifest.write(study, path="run/")
```

### 4.3 The `Study` metadata object

Every design module returns `(data: DataFrame, study: Study)`. `Study` carries: seed, EPISIM version, design name, parameter dict, indicator-formula spec, SHA-256 of `data`, timestamp, citation block. `Study.archive(path)` writes a complete reproducible bundle.

## 5. Reproducibility and testing contract

### 5.1 Reproducibility

* Every public function takes an explicit `rng: np.random.Generator` or uses the active `seed()` context.
* Every output bundle includes `manifest.json` with seed, library version, parameter dict, SHA-256 of every file.
* Pinned dependencies in both `pyproject.toml` (declarative) and `requirements.lock` (deterministic).
* Docker image per release on GHCR for full environment reproducibility.

### 5.2 Testing

* **Unit tests** in `tests/unit/`. Target ≥ 90 % line coverage of `episim/core/`. Coverage gate enforced in CI.
* **Property tests** via `hypothesis`. Examples: a parallel-arm RCT with no treatment effect should produce 95 % CIs containing 0 in ≥ 94 % of seeds across 1000 runs.
* **Reproducibility tests** in `tests/repro/`: same seed → same SHA-256 of every output.
* **Notebook smoke tests** via `nbmake`: every notebook in `notebooks/` runs end-to-end in CI; failures block merge.

### 5.3 CI/CD (GitHub Actions)

* **`ci.yml` (on PR):** `ruff` + `mypy --strict` + `pytest --cov` (≥ 90 % gate) + `nbmake notebooks/`.
* **`docs.yml` (on merge to main):** rebuild MkDocs, deploy to GH Pages.
* **`hf-sync.yml` (on merge to main):** push apps/ to corresponding HF Spaces.
* **`release.yml` (on `v*` tag):** build + publish to PyPI; build + push Docker image to GHCR; create GitHub release; trigger Zenodo DOI.
* **`kaggle-sync.yml` (on `v*` tag):** push notebooks via Kaggle API.

### 5.4 Versioning

Semantic versioning. v0.x = pre-stable (API may break in minor releases). v1.0 when API frozen, all 18 notebooks shipped, ≥ 1 external feedback round complete.

## 6. Reporting + AI-disclosure layer

### 6.1 Five artefacts per Study

1. **Reporting-guideline checklist** (STROBE / CONSORT / TRIPOD-AI / STRESS / PRISMA / GRIPP2 / SAGER as applicable to the design). Source: YAML files in `episim/reporting/checklists/` (CC-BY 4.0; community-contributable). Output: filled-in `.md` and `.docx`.
2. **Methods narrative auto-draft.** Deterministic prose generated from `Study` metadata; written into the conventional Methods-section structure for the design. No LLM in v0.1; LLM-augmented narrative in phase 3.
3. **Figure pack.** Standardised matplotlib figures (CONSORT flow, forest plot, calibration plot, fairness audit, DAG) auto-rendered from `Study` metadata. Recognisable EPISIM look across all 18 notebooks.
4. **Reproducibility manifest** — `manifest.json` with seed, version, parameter dict, SHA-256 of every output, EPISIM citation, Zenodo DOI of the package release used.
5. **AI-disclosure block** — two flavours, dropping straight into a manuscript:
   * `mode="library_only"`: "Simulation conducted using EPISIM v0.x [DOI]. No generative-AI was used in design, analysis, or interpretation."
   * `mode="generative"`: "Simulation parameters were specified in natural language and parsed by EPISIM's generative layer (LLM: Claude Opus 4.7 via LiteLLM). All design, parameter, and interpretation decisions were verified by the author against the deterministic re-run from the saved seed."

### 6.2 Licensing within the reporting layer

* Code (Python in `episim/`): **MIT**.
* YAML checklists and figure templates: **CC-BY 4.0** so community PRs of guideline updates retain author attribution.

## 7. Phase roadmap

### 7.1 Phase 1 — Teaching/methods-demonstration lab (Weeks 1–8)

| Release | Week | Contents | Visible on |
|---|---|---|---|
| v0.1.0-alpha | 1 | Repo skeleton; `episim.core` primitives; CI; docs site live; notebook 1 (cross-sectional) | GitHub, MkDocs |
| v0.2 | 2 | + 2 notebooks (cohort, RCT parallel); combined HF Space live | + HF Spaces |
| v0.3 | 3 | + 3 notebooks (case-control, RCT cluster, ecological — *PEAI port*) | |
| v0.4 | 4 | + 3 notebooks (stepped-wedge, ITS, regression discontinuity) | |
| v0.5 | 5 | + 3 notebooks (IV, propensity-score, survival/Cox); per-design HF Spaces start | per-design HF Spaces |
| v0.6 | 6 | + 3 notebooks (meta-analysis, agent-based SEIR, microsimulation life-table) | |
| v0.7 | 7 | + 2 notebooks (Markov decision-analytic, network/contagion) | |
| v0.8 | 8 | + 1 notebook (qualitative/mixed-methods power); polish; PyPI; Kaggle; Zenodo DOI; Docker; **JOSS submission** | All seven surfaces |

### 7.2 Phase 2 — Methods-research / pre-registration lab (Weeks 9–14)

| Release | Approx. timing | Contents |
|---|---|---|
| v0.9 | Weeks 9–12 | "Pre-registration mode": investigator describes a planned real-world study; EPISIM runs power + bias-sensitivity + design-comparison + fairness audit; emits a *pre-registration package* (analysis plan + simulation evidence) suitable for OSF deposition. |
| v1.0 | Week 14 | API freeze; v1.0 release; SMMR/Stat Med methods paper. |

### 7.3 Phase 3 — Generative-AI simulation engine (Months 4–6)

| Release | Timing | Contents |
|---|---|---|
| v1.x-genai | Months 4–6 | Natural-language interface: prose → LLM-extracted parameters → EPISIM `Study` spec → deterministic simulation → manuscript draft. Audit trail: every LLM call logged; deterministic re-run from parsed spec is byte-identical. LLM provider abstracted via **LiteLLM**. |
| v2.0 | Month 6 | Multi-agent peer-review-personae (extending the technique used for our own NMJI manuscript): the same LLM, prompted as different reviewer personae, produces independent critique that the user can act on before submission. |

### 7.4 Phase 4 — Beyond medicine (Months 7–9)

| Release | Timing | Contents |
|---|---|---|
| v2.x | Months 7–9 | Notebook packs for educational research designs (RCT in classrooms, cluster-stepped-wedge in school districts), economics/policy (synthetic control, difference-in-differences), psychology (within-subjects, IRT simulation), sociology/qualitative (Q-methodology, content-analysis power). |

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 18-notebook scope underestimates author time | High | High | Each notebook independently shippable; HF Space live from week 2; scope can compress to 12 if week 4 is behind without breaking the platform identity |
| Maintenance burden of 18+ notebooks + 18+ apps + 5 surfaces | Medium | High | One source of truth; CI does the multi-surface fan-out; tests catch regressions |
| Reporting-checklist drift (e.g., STROBE 2.0 release) | Medium | Medium | YAML + community PRs; versioned checklists |
| LLM API cost in phase 3 | Medium | Medium | LiteLLM lets institutional users plug their own key; default uses small-model routing |
| Reviewer hostility to AI-assisted methodology papers | Medium | Medium | Two-flavour AI-disclosure; library-only mode is fully classical |
| Notebook smoke-test runtime (CI minutes) | Low | Low | Cap each notebook at < 30 s runtime; tag long-running examples as `@slow` and skip in CI |
| External contributor onboarding | Medium | Low | CONTRIBUTING.md + checklist YAMLs as easy first issues; "good first issue" label |

## 9. Success criteria

### v0.8 (week 8)
* 18 notebooks shipped, all passing CI smoke tests.
* `pip install episim` works, ≥ 90 % core test coverage, Docker image on GHCR, Zenodo DOI assigned.
* Combined HF Space live; ≥ 5 design notebooks have working web UIs.
* JOSS submission accepted into review queue.

### v1.0 (week 14)
* API frozen; SMMR or Stat Med submission of the methods paper.
* ≥ 3 external users (i.e., not the author) have run an EPISIM study and given feedback.
* Phase 2 pre-registration mode shipped.

### v2.0 (month 6)
* Phase 3 generative interface live behind an institutional auth wall.
* ≥ 1 published manuscript in any journal that cites EPISIM's `Study.archive()` artefact for reproducibility.

## 10. Open questions deferred to implementation phase

* Exact YAML schema for reporting checklists (will design alongside notebook 1).
* Exact `Study` JSON-schema for cross-language portability (deferred to phase 2).
* Whether to support R via `reticulate` or rpy2 — likely yes from v1.0 onward; not v0.1.
* Whether to support DAGitty integration for confounder visualisation — likely yes by v0.5.

These are deliberately deferred so we don't burn brainstorming cycles on them now; they'll be resolved in concrete PRs at the relevant week.

---

**This design is approved through Section 6. The implementation plan will be produced via the writing-plans skill once you sign off on this written spec.**
