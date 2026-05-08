# Lab Surface

EPISIM now exposes a lab-facing registry over implemented designs.

## Implemented designs

- `cross_sectional`: prevalence study with exposure-outcome association recovery.
- `case_control`: source-population case-control sampling with odds-ratio estimation.
- `cohort`: longitudinal follow-up with attrition, risk ratio, and risk difference.
- `rct_parallel`: two-arm randomized experiment with treatment-effect summaries.
- `rct_cluster`: cluster-level randomization with individual outcomes.
- `stepped_wedge`: sequential cluster rollout with repeated cross-sections.
- `interrupted_time_series`: segmented level and slope change simulation.
- `regression_discontinuity`: threshold assignment with local linear estimation.
- `instrumental_variables`: encouragement design with Wald LATE.
- `propensity_score`: inverse-probability weighting under confounding.
- `survival_cox`: censored time-to-event simulation with hazard-ratio summary.
- `meta_analysis`: DerSimonian-Laird random-effects pooling.
- `agent_based_seir`: stochastic SEIR epidemic simulation.
- `microsimulation_lifetable`: individual health-state life-table simulation.
- `markov_decision`: cost-QALY Markov decision model.
- `network_contagion`: SIR-style spread over a random contact network.
- `qualitative_mixed_methods`: interview saturation plus survey strand.
- `ecological_peai`: four-phase PEAI simulation with transportability, ascertainment-bias, and fairness auditing.

## Python API

```python
from episim.lab import list_designs, run_design

for design in list_designs():
    print(design.key, design.title)

study = run_design("cohort", seed_value=20260508, n=3000)
study.archive("results/cohort_demo")
```

## Research question workflow

`episim.research` provides a higher-level workflow for simulated research
planning and execution. It turns a user question into aims, objectives,
hypotheses, design selection, methodology, data-collection tools, simulated
observations, follow-up schedules, outcome recording, results, discussion, and
conclusions.

```python
from episim.research import conduct_research

bundle = conduct_research(
    "Does a randomized lifestyle intervention reduce frailty in community elders?",
    seed_value=20260508,
    n=2000,
)

bundle.plan.design_key
bundle.instruments.head()
bundle.outcome_record
bundle.report.markdown
bundle.archive("results/frailty_research_bundle")
```

The archive contains:

- `protocol.json` and `protocol.md`
- `report.md` with structured abstract, methods, results, discussion,
  conclusions, references, and declarations
- `collected_synthetic_data.csv`
- `cleaned_analysis_dataset.csv`
- `data_cleaning_log.csv`
- `analysis_steps.csv`
- `data_collection_tools.csv`
- `database_dictionary.csv`
- `collection_events.csv`
- `synthetic_research_database.sqlite`
- analysis tables under `tables/`
- PNG figures and `figure_plan.csv` under `figures/`
- `guideline_checklist.csv`
- `realism_audit.csv`
- `sensitivity_analysis.csv`
- `readiness_checklist.csv`

The workflow generates synthetic data only. It is suitable for methods
development, teaching, protocol refinement, and feasibility simulation; it is
not evidence that a real intervention or exposure works in the real world.

External realism calibration can be layered around EPISIM by using the exported
CSV or SQLite research database with authenticated tools such as Google Cloud
healthcare/FHIR pipelines, Sensitive Data Protection de-identification
workflows, or Vertex AI tabular workflows. EPISIM itself keeps the default
workflow local, deterministic, and credential-free.

## Interactive app

The combined Gradio app lives at `apps/hf_space/app.py` and is suitable for
Hugging Face Spaces deployment.
