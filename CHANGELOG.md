# Changelog

All notable changes to EPISIM are documented here. Format: Keep-a-Changelog. Versioning: SemVer.

## [v0.11.0-alpha] - 2026-05-08

### Added

- Added explicit collected-data, cleaned-analysis-dataset, data-cleaning-log,
  and analysis-step artifacts to the end-to-end research workflow.
- Expanded the Hugging Face research workflow display to show collected data,
  cleaned data, cleaning steps, and analysis steps before final manuscript output.
- Extended the SQLite research database with collected, cleaned, cleaning-log,
  and analysis-step tables.

## [v0.10.0-alpha] - 2026-05-08

### Added

- Strengthened `episim.research` with manuscript-grade asset production:
  analysis tables, variable summaries, data-quality tables, interpretation
  tables, figure plan, PNG figure generation, guideline checklist, references,
  and structured declarations.
- Added realistic synthetic data-capture/database production assets:
  database dictionary, collection-event audit log, and SQLite research database
  containing observations, tools, follow-up, outcomes, checklist, declarations,
  references, and analysis tables.
- Expanded Hugging Face Space research workflow outputs for database dictionary,
  collection-event audit previews, variable summaries, and guideline checklist.

## [v0.9.0-alpha] - 2026-05-08

### Added

- `episim.research` end-to-end simulated research workflow that turns a
  plain-language research question into a protocol, design selection, aims,
  objectives, hypotheses, methodology, data-collection tools, synthetic
  observations, follow-up schedule, outcome record, results, discussion,
  conclusions, and archiveable manuscript-style report.
- Hugging Face Space UI support for the research-question workflow.
- Research workflow API documentation and unit tests.

## [v0.8.0-alpha] - 2026-05-08

### Added

- Full Phase-1 design catalog: RCT cluster, stepped-wedge, interrupted time series, regression discontinuity, instrumental variables, propensity score, survival/Cox-style time-to-event, meta-analysis, agent-style SEIR, microsimulation, Markov decision analysis, network contagion, and qualitative/mixed-methods.
- Notebooks `06` through `18` covering the full roadmap design catalog.
- Docker, Zenodo, and JOSS submission skeleton assets.
- Per-design Hugging Face Space wrapper.

### Changed

- Promoted the alpha package version to `0.8.0a1`.
- Expanded the lab registry to cover all 18 Phase-1 notebooks/designs.

## [v0.2.0-alpha] - 2026-05-08

### Added

- `episim.lab` registry and runner for catalogued study designs.
- Richer `Study` archives with `results.json` and artifact CSV tables.
- App-ready `cross_sectional.simulate()` workflow.
- Real design modules for case-control, prospective cohort, and parallel-group RCT simulations.
- `ecological.run_peai()` four-phase PEAI lab with weighting convergence, transportability, ascertainment-bias diagnostics, and fairness auditing.
- Five smoke-tested notebooks: cross-sectional, case-control, cohort, RCT parallel, and ecological PEAI.
- Hugging Face Space-ready Gradio app in `apps/hf_space/`.
- Kaggle publishing metadata and overview script in `platforms/kaggle/`.
- Lab and deployment documentation pages.

### Changed

- Reworked the public README around the lab product surface.
- Expanded analytics with risk ratio and risk difference estimators.
- Switched docs to the built-in Read the Docs MkDocs theme for strict local builds without Material warning noise.

## [v0.1.0-alpha] - 2026-05-08

### Added

- `episim.core.reproducibility` with `seed()` context, `Study` dataclass, and SHA-256 manifest.
- `episim.core.populations.cohort` parameterised cohort generator.
- `episim.core.outcomes.logistic` formulaic-based logistic outcome model.
- `episim.core.bias` measurement error and ascertainment-bias primitives.
- `episim.core.attrition` MCAR / MAR / MNAR mechanisms.
- `episim.analytics.estimators` prevalence and odds-ratio with bootstrap CIs.
- `episim.designs.cross_sectional` first design module.
- `episim.reporting.ai_disclosure.block` disclosure helper.
- `episim.reporting.manifest.write_manifest` study archiver.
- First notebook: `01_cross_sectional.ipynb`.
- CI, docs, and coverage gates for the foundation release.
