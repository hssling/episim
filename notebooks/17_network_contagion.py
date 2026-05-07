# %% [markdown]
# # EPISIM 17 - Network contagion

# %%
from episim.designs import network_contagion

# %%
study = network_contagion.run(seed_value=20260508, n_nodes=300, n_days=60)
study.results

# %%
study.data.head()
