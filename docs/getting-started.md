# Getting started

```bash
pip install episim
```

```python
from episim.lab import run_design

study = run_design("cohort", seed_value=42, n=3_000)
study.archive("results/")
```
