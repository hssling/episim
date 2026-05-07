"""Qualitative and mixed-methods simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from episim.core import seed
from episim.core.reproducibility import Study
from episim.designs._helpers import build_study


def run(
    *,
    seed_value: int,
    n_interviews: int = 40,
    n_survey: int = 500,
    n_latent_themes: int = 10,
    minimum_saturation_run: int = 5,
) -> Study:
    """Simulate qualitative saturation plus a quantitative survey strand."""
    with seed(seed_value) as rng:
        theme_prob = np.linspace(1.0, 0.20, n_latent_themes)
        theme_prob = theme_prob / theme_prob.sum()
        seen: set[int] = set()
        no_new_run = 0
        rows = []
        saturation_interview = n_interviews
        for interview in range(1, n_interviews + 1):
            n_mentions = int(rng.integers(1, 5))
            themes = set(
                rng.choice(
                    n_latent_themes, n_mentions, replace=False, p=theme_prob
                ).tolist()
            )
            new_themes = themes - seen
            seen |= themes
            no_new_run = no_new_run + 1 if not new_themes else 0
            if no_new_run >= minimum_saturation_run and saturation_interview == n_interviews:
                saturation_interview = interview
            rows.append(
                {
                    "interview": interview,
                    "themes_mentioned": len(themes),
                    "new_themes": len(new_themes),
                    "cumulative_themes": len(seen),
                }
            )
        qual = pd.DataFrame(rows)
        exposure = rng.binomial(1, 0.45, n_survey)
        acceptability = 3.2 + 0.45 * exposure + rng.normal(0, 0.8, n_survey)
        survey = pd.DataFrame(
            {
                "respondent": np.arange(n_survey),
                "exposure": exposure,
                "acceptability": np.clip(acceptability, 1, 5),
            }
        )
        diff = float(
            survey.loc[survey["exposure"] == 1, "acceptability"].mean()
            - survey.loc[survey["exposure"] == 0, "acceptability"].mean()
        )
        return build_study(
            design="qualitative_mixed_methods",
            seed_value=seed_value,
            data=qual,
            params={
                "n_interviews": n_interviews,
                "n_survey": n_survey,
                "n_latent_themes": n_latent_themes,
                "minimum_saturation_run": minimum_saturation_run,
            },
            results={
                "saturation_interview": int(saturation_interview),
                "themes_identified": int(qual["cumulative_themes"].max()),
                "survey_acceptability_difference": round(diff, 3),
                "mixed_methods_inference": "convergent" if diff > 0.25 else "discordant",
            },
            artifacts={"survey_strand": survey},
        )
