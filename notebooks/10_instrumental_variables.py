# %% [markdown]
# # EPISIM 10 - Instrumental variables

# %%
from episim.designs import instrumental_variables

# %%
study = instrumental_variables.run(seed_value=20260508)
study.results

# %%
study.data.head()
