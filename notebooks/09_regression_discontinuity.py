# %% [markdown]
# # EPISIM 09 - Regression discontinuity

# %%
from episim.designs import regression_discontinuity

# %%
study = regression_discontinuity.run(seed_value=20260508, n=1800)
study.results

# %%
study.artifacts["side_summary"]
