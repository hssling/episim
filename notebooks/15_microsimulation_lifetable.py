# %% [markdown]
# # EPISIM 15 - Life-table microsimulation

# %%
from episim.designs import microsimulation

# %%
study = microsimulation.run(seed_value=20260508, n=1000, cycles=10)
study.results

# %%
study.artifacts["cycle_trajectory"].head()
