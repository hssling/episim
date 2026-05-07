# Getting started

```bash
pip install episim
```

```python
from episim.core import cohort, logistic, seed
from episim.designs import cross_sectional

with seed(42) as rng:
    pop = cohort(n=1_000, age=("normal", 50, 12), rng=rng)
    pop["exposure"] = rng.binomial(1, 0.3, 1_000)
    pop = logistic(pop, "y ~ exposure + age",
                    betas={"exposure": 0.6, "age": 0.04}, rng=rng)
    study = cross_sectional.run(pop, exposure="exposure", outcome="y",
                                 seed_value=42)
study.archive("results/")
```
