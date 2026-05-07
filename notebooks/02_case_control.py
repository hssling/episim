# %% [markdown]
# # EPISIM 02 — Case-control study
#
# Run a full case-control simulation and inspect the recovered odds ratio.

# %%
from episim.designs import case_control

# %%
study = case_control.run(seed_value=20260508, n_source=6000, n_cases=300)
study.results

# %%
study.data.head()
