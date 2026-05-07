# %% [markdown]
# # EPISIM 16 - Markov decision model

# %%
from episim.designs import markov_decision

# %%
study = markov_decision.run(seed_value=20260508)
study.results

# %%
study.artifacts["state_trace"].head()
