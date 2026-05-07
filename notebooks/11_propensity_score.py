# %% [markdown]
# # EPISIM 11 - Propensity-score weighting

# %%
from episim.designs import propensity_score

# %%
study = propensity_score.run(seed_value=20260508)
study.results

# %%
study.artifacts["covariate_balance"]
