# %% [markdown]
# # EPISIM 07 - Stepped-wedge cluster trial

# %%
from episim.designs import stepped_wedge

# %%
study = stepped_wedge.run(
    seed_value=20260508,
    n_clusters=10,
    n_periods=5,
    persons_per_cluster_period=40,
)
study.results

# %%
study.artifacts["cluster_period_summary"].head()
