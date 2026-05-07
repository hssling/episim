# Contributing to EPISIM

Welcome — and thank you for considering a contribution.

## Quick start

```bash
git clone https://github.com/hssling/episim
cd episim
pip install -e ".[dev,docs]"
pytest
```

## Standards

- **Tests first.** Every new public function ships with at least one unit test (≥ 90 % line coverage gate).
- **Type-checked.** `mypy --strict` must pass on every PR.
- **Lint-clean.** `ruff check episim tests` must pass.
- **Notebooks.** Any new notebook must pass `pytest --nbmake notebooks/<file>.ipynb` in under 30 seconds; long-running examples should be marked `# %% [markdown]` with reproduction parameters scaled down for CI.

## Reporting-checklist contributions

The YAML files in `episim/reporting/checklists/` are licensed CC-BY 4.0 and explicitly designed for community PR contributions when reporting guidelines update (e.g. STROBE 2.0).

## License

By contributing, you agree your code is MIT-licensed and your checklist YAML is CC-BY 4.0.
