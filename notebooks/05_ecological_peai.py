# %% [markdown]
# # EPISIM 05 — Ecological PEAI lab
#
# Run a reduced PEAI simulation to inspect weighting convergence, transportability,
# ascertainment bias, and fairness outputs.

# %%
from episim.designs import ecological

# %%
study = ecological.run_peai(
    seed_value=20260508,
    n_experts=20,
    n_dev=500,
    n_external=300,
    n_prospective=700,
)
study.results["phase4"]

# %%
study.artifacts["prospective_discrimination"].head()
