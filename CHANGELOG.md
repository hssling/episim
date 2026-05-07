# Changelog

All notable changes to EPISIM are documented here. Format: Keep-a-Changelog. Versioning: SemVer.

## [v0.1.0-alpha] — 2026-05-08

### Added

- `episim.core.reproducibility` — `seed()` context, `Study` dataclass, SHA-256 manifest.
- `episim.core.populations.cohort` — parameterised cohort generator (normal / bernoulli / uniform).
- `episim.core.outcomes.logistic` — formulaic-based logistic outcome model.
- `episim.core.bias` — measurement error and ascertainment-bias primitives.
- `episim.core.attrition` — MCAR / MAR / MNAR mechanisms.
- `episim.analytics.estimators` — prevalence and odds-ratio with bootstrap CIs.
- `episim.designs.cross_sectional` — first design module.
- `episim.reporting.ai_disclosure.block` — two-flavour AI disclosure (`library_only`, `generative`).
- `episim.reporting.manifest.write_manifest` — `Study` archiver.
- First notebook: `01_cross_sectional.ipynb`.
- CI: ruff + mypy + pytest 90 % coverage + nbmake.
- Docs: MkDocs Material + mkdocstrings deployed to GitHub Pages.
- 100 % line + branch coverage of `episim/` modules at v0.1.0-alpha.
