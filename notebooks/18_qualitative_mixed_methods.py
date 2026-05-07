# %% [markdown]
# # EPISIM 18 - Qualitative / mixed-methods

# %%
from episim.designs import qualitative_mixed_methods

# %%
study = qualitative_mixed_methods.run(seed_value=20260508)
study.results

# %%
study.data.head()
