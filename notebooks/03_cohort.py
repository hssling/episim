# %% [markdown]
# # EPISIM 03 — Prospective cohort
#
# Simulate longitudinal follow-up with attrition and estimate risk contrasts.

# %%
from episim.designs import cohort

# %%
study = cohort.run(seed_value=20260508, n=3000, p_attrition=0.10)
study.results

# %%
study.data.head()
