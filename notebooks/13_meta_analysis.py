# %% [markdown]
# # EPISIM 13 - Random-effects meta-analysis

# %%
from episim.designs import meta_analysis

# %%
study = meta_analysis.run(seed_value=20260508)
study.results

# %%
study.data.head()
