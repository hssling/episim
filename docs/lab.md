# Lab Surface

EPISIM now exposes a lab-facing registry over implemented designs.

## Implemented designs

- `cross_sectional`: prevalence study with exposure-outcome association recovery.
- `case_control`: source-population case-control sampling with odds-ratio estimation.
- `cohort`: longitudinal follow-up with attrition, risk ratio, and risk difference.
- `rct_parallel`: two-arm randomized experiment with treatment-effect summaries.
- `ecological_peai`: four-phase PEAI simulation with transportability, ascertainment-bias, and fairness auditing.

## Python API

```python
from episim.lab import list_designs, run_design

for design in list_designs():
    print(design.key, design.title)

study = run_design("cohort", seed_value=20260508, n=3000)
study.archive("results/cohort_demo")
```

## Interactive app

The combined Gradio app lives at `apps/hf_space/app.py` and is suitable for
Hugging Face Spaces deployment.
