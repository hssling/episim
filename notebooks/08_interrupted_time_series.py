# %% [markdown]
# # EPISIM 08 - Interrupted time series

# %%
from episim.designs import interrupted_time_series

# %%
study = interrupted_time_series.run(seed_value=20260508)
study.results

# %%
study.data.head()
