# %% [markdown]
# # EPISIM 04 — Parallel-group RCT
#
# Simulate a randomized trial and inspect treatment effect summaries.

# %%
from episim.designs import rct_parallel

# %%
study = rct_parallel.run(seed_value=20260508, n=1800)
study.results

# %%
study.data.head()
