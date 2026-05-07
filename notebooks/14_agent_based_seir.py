# %% [markdown]
# # EPISIM 14 - Agent-style SEIR

# %%
from episim.designs import seir

# %%
study = seir.run(seed_value=20260508, n_population=1500, n_days=80)
study.results

# %%
study.data.head()
