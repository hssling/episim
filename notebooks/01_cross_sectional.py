# %% [markdown]
# # EPISIM 01 — Cross-sectional study
#
# A minimal end-to-end demonstration. Generates a 1,000-person cohort with
# age, sex, and a binary exposure, simulates a binary outcome from a logistic
# model, computes the prevalence and odds ratio with bootstrap 95 % CIs,
# and writes a deterministic Study archive.

# %%
from episim.core import cohort, logistic, seed
from episim.core.bias import measurement_error
from episim.analytics import odds_ratio, prevalence
from episim.designs import cross_sectional
from episim.reporting import block, write_manifest

# %% [markdown]
# ## 1. Simulate the cohort

# %%
with seed(20260507) as rng:
    pop = cohort(
        n=1_000,
        age=("normal", 50, 12),
        sex=("bernoulli", 0.5),
        rng=rng,
    )
    pop["exposure"] = rng.binomial(1, 0.3, 1_000)
    pop = logistic(
        pop, "y ~ exposure + age",
        betas={"exposure": 0.6, "age": 0.04},
        intercept=-3.0, rng=rng,
    )
    pop = measurement_error(pop, on="age", sd=1.0, rng=rng)
    p, p_lo, p_hi = prevalence(pop, outcome="y", rng=rng)
    or_, or_lo, or_hi = odds_ratio(pop, exposure="exposure", outcome="y", rng=rng)
    study = cross_sectional.run(pop, exposure="exposure", outcome="y", seed_value=20260507)

print(f"Prevalence: {p:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
print(f"Odds ratio: {or_:.3f} (95% CI {or_lo:.3f}-{or_hi:.3f})")

# %% [markdown]
# ## 2. Archive the Study and emit AI disclosure

# %%
write_manifest(study, "01_cross_sectional_run/")
disclosure = block(mode="library_only", version="0.1.0a1",
                   doi="10.5281/zenodo.placeholder")
print(disclosure)
