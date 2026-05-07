# %% [markdown]
# # EPISIM 06 - Cluster-randomized trial

# %%
from episim.designs import rct_cluster

# %%
study = rct_cluster.run(seed_value=20260508, n_clusters=24, cluster_size=60)
study.results

# %%
study.artifacts["cluster_summary"].head()
