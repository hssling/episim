# %% [markdown]
# # EPISIM 12 - Survival / Cox-style analysis

# %%
from episim.designs import survival_cox

# %%
study = survival_cox.run(seed_value=20260508)
study.results

# %%
study.artifacts["survival_summary"]
