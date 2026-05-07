"""Ecological and index-development study simulations."""
# ruff: noqa: N806
from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from episim.core.reproducibility import Study
from episim.designs._helpers import build_study

LOADINGS = {
    "E": [0.85, 0.80, 0.75],
    "S": [0.80, 0.78, 0.70],
    "B": [0.75, 0.72, 0.65],
    "P": [0.78, 0.74, 0.68],
    "H": [0.82, 0.76, 0.70],
}
TRUE_BETAS = {"E": 0.35, "S": 0.25, "B": 0.30, "P": 0.20, "H": -0.40}
INDICATORS = [f"{k}{j}" for k in LOADINGS for j in (1, 2, 3)]
VARIANTS = [
    "peai_equal",
    "peai_theoretical",
    "peai_pca",
    "peai_entropy",
    "peai_elasticnet",
    "peai_gbt",
]


def _simulate_peai_cohort(
    n: int, geography: str, rng: np.random.Generator
) -> pd.DataFrame:
    age = rng.normal(63, 8, n).clip(45, 95)
    sex = rng.binomial(1, 0.52, n)
    taluk = rng.integers(1, 11, n)
    ses_quintile = rng.integers(1, 6, n)
    dev = rng.normal(0, 1, n)
    domains: dict[str, npt.NDArray[np.float64]] = {}
    for key in ("E", "S", "B", "P", "H"):
        load_dev = -0.5 if key != "H" else 0.6
        domains[key] = load_dev * dev + rng.normal(0, math.sqrt(1 - 0.25), n)

    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "geography": geography,
            "age": age,
            "sex": sex,
            "taluk": taluk,
            "ses_quintile": ses_quintile,
        }
    )
    for key, loadings in LOADINGS.items():
        for idx, loading in enumerate(loadings, 1):
            err_sd = math.sqrt(1 - loading**2)
            df[f"{key}{idx}"] = loading * domains[key] + rng.normal(0, err_sd, n)
        df[f"{key}_true"] = domains[key]

    interaction = 0.10 * domains["E"] * domains["B"]
    age_z = (age - age.mean()) / age.std()
    ses_z = (ses_quintile - 3) / 1.4
    hab = (
        TRUE_BETAS["E"] * domains["E"]
        + TRUE_BETAS["S"] * domains["S"]
        + TRUE_BETAS["B"] * domains["B"]
        + TRUE_BETAS["P"] * domains["P"]
        + TRUE_BETAS["H"] * domains["H"]
        + interaction
        + 0.45 * age_z
        + 0.10 * sex
        + 0.20 * ses_z
        + rng.normal(0, 0.4, n)
    )
    df["hab_true"] = hab

    p_frail = 1 / (1 + np.exp(-(0.7 * hab - 0.3)))
    p_adl = 1 / (1 + np.exp(-(0.6 * hab + 0.3 * age_z - 1.0)))
    p_mm_true = 1 / (1 + np.exp(-(0.9 * hab + 0.4 * age_z - 0.4)))
    detect = 1 / (1 + np.exp(-(1.0 * domains["H"] + 0.5)))
    p_mm_observed = p_mm_true * detect
    log_h = -3.5 + 0.6 * hab + 0.7 * age_z + 0.15 * sex
    h = np.exp(log_h)
    p_mort = 1 - np.exp(-h * 5)
    t = rng.exponential(1.0 / h)

    df["frailty_24m"] = rng.binomial(1, p_frail, n)
    df["adl_decline_36m"] = rng.binomial(1, p_adl, n)
    df["multimorbidity_60m_true"] = rng.binomial(1, p_mm_true, n)
    df["multimorbidity_60m_observed"] = rng.binomial(1, p_mm_observed, n)
    df["mortality_60m"] = rng.binomial(1, p_mort, n)
    df["time_years"] = np.minimum(t, 5.0)
    df["event"] = (t <= 5.0).astype(int)
    return df


def _phase1_delphi(rng: np.random.Generator, n_experts: int) -> dict[str, Any]:
    expert_estimates = []
    for _ in range(n_experts):
        expert_estimates.append(
            {k: TRUE_BETAS[k] + rng.normal(0, 0.10) for k in TRUE_BETAS}
        )
    df = pd.DataFrame(expert_estimates)
    consensus = df.median()
    iqr = df.quantile(0.75) - df.quantile(0.25)
    abs_med = consensus.abs()
    weights = abs_med / abs_med.sum()
    return {
        "n_experts": n_experts,
        "consensus_medians": consensus.round(3).to_dict(),
        "consensus_iqr": iqr.round(3).to_dict(),
        "fraction_indicators_with_consensus": round(
            float(((iqr / df.median().abs()) < 0.30).mean()), 3
        ),
        "delphi_weights": weights.round(3).to_dict(),
    }


def _build_peai_variants(
    df: pd.DataFrame, delphi_weights: dict[str, float]
) -> pd.DataFrame:
    X = df[INDICATORS].to_numpy()
    Xz = StandardScaler().fit_transform(X)
    out = df.copy()
    out["peai_equal"] = Xz.mean(axis=1)

    sign = {"E": 1, "S": 1, "B": 1, "P": 1, "H": -1}
    theoretical = np.asarray(
        [sign[col[0]] * delphi_weights.get(col[0], 0.0) for col in INDICATORS]
    )
    out["peai_theoretical"] = Xz @ theoretical

    pca = PCA(n_components=1).fit(Xz)
    out["peai_pca"] = pca.transform(Xz)[:, 0]
    burden = Xz[:, :12].mean(axis=1)
    if np.corrcoef(out["peai_pca"], burden)[0, 1] < 0:
        out["peai_pca"] *= -1

    Xpos = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)
    p = Xpos / (Xpos.sum(axis=0) + 1e-9)
    entropy = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0), axis=0) / np.log(
        len(X)
    )
    w_ent = (1 - entropy) / (1 - entropy).sum()
    signed_entropy = w_ent * np.asarray([sign[col[0]] for col in INDICATORS])
    out["peai_entropy"] = Xz @ signed_entropy
    return out


def _project_supervised_variants(
    df_dev: pd.DataFrame,
    df_other: pd.DataFrame,
    seed_value: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xd = df_dev[INDICATORS].to_numpy()
    scaler = StandardScaler().fit(Xd)
    Xd_z = scaler.transform(Xd)
    y = df_dev["frailty_24m"].to_numpy()
    en = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=seed_value
    ).fit(Xd_z, y)
    gbt = GradientBoostingClassifier(
        random_state=seed_value, n_estimators=200, max_depth=3
    ).fit(Xd_z, y)
    dev = df_dev.copy()
    other = df_other.copy()
    dev["peai_elasticnet"] = Xd_z @ en.coef_
    dev["peai_gbt"] = gbt.predict_proba(Xd_z)[:, 1]
    Xo_z = scaler.transform(df_other[INDICATORS].to_numpy())
    other["peai_elasticnet"] = Xo_z @ en.coef_
    other["peai_gbt"] = gbt.predict_proba(Xo_z)[:, 1]
    return dev, other


def _cronbach_alpha(x: npt.NDArray[np.float64]) -> float:
    k = x.shape[1]
    var_i = x.var(axis=0, ddof=1).sum()
    var_t = x.sum(axis=1).var(ddof=1)
    return float((k / (k - 1)) * (1 - var_i / var_t))


def _phase3_validation(
    df_dev: pd.DataFrame, df_ext: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    convergence = df_dev[VARIANTS].rank().corr(method="spearman").round(3)
    criterion_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    criterion_summary: dict[str, Any] = {}
    transport_summary: dict[str, Any] = {}
    for key in LOADINGS:
        cols = [f"{key}{j}" for j in (1, 2, 3)]
        alpha_rows.append(
            {
                "domain": key,
                "cronbach_alpha": round(
                    _cronbach_alpha(df_dev[cols].to_numpy()), 3
                ),
            }
        )
    for variant in VARIANTS:
        frail = stats.spearmanr(df_dev[variant], df_dev["frailty_24m"])
        hab = stats.pearsonr(df_dev[variant], df_dev["hab_true"])
        criterion_summary[variant] = {
            "spearman_vs_frailty": round(float(frail.statistic), 3),
            "pearson_vs_hab_true": round(float(hab.statistic), 3),
        }
        criterion_rows.append(
            {
                "variant": variant,
                "spearman_vs_frailty": round(float(frail.statistic), 3),
                "spearman_pvalue": round(float(frail.pvalue), 4),
                "pearson_vs_hab_true": round(float(hab.statistic), 3),
                "pearson_pvalue": round(float(hab.pvalue), 4),
            }
        )
        auc_dev = roc_auc_score(df_dev["frailty_24m"], df_dev[variant])
        auc_ext = roc_auc_score(df_ext["frailty_24m"], df_ext[variant])
        transport_summary[variant] = {
            "auroc_dev": round(float(auc_dev), 3),
            "auroc_external": round(float(auc_ext), 3),
            "drop": round(float(auc_dev - auc_ext), 3),
        }
        transport_rows.append({"variant": variant, **transport_summary[variant]})
    summary = {
        "weighting_convergence_min_offdiag": round(
            float((convergence.to_numpy() - np.eye(len(VARIANTS))).max()), 3
        ),
        "criterion_validity": criterion_summary,
        "external_transportability": transport_summary,
    }
    artifacts = {
        "weighting_convergence": convergence.reset_index(names="variant"),
        "criterion_validity": pd.DataFrame(criterion_rows),
        "external_transportability": pd.DataFrame(transport_rows),
        "sub_index_alpha": pd.DataFrame(alpha_rows),
    }
    return summary, artifacts


def _prospective_evaluation(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame]:
    keep = rng.random(len(df)) > 0.10
    df_p = df.loc[keep].reset_index(drop=True)
    base_cov = ["age", "sex", "ses_quintile"]
    Xb = StandardScaler().fit_transform(df_p[base_cov].to_numpy())
    endpoints = [
        "frailty_24m",
        "adl_decline_36m",
        "multimorbidity_60m_observed",
        "multimorbidity_60m_true",
        "mortality_60m",
    ]
    discrim_rows: list[dict[str, Any]] = []
    fair_rows: list[dict[str, Any]] = []
    ascertainment_rows: list[dict[str, Any]] = []
    endpoint_summary: dict[str, Any] = {}

    for endpoint in endpoints:
        y = df_p[endpoint].to_numpy()
        if y.sum() < 30 or (1 - y).sum() < 30:
            continue
        base_mod = LogisticRegression(max_iter=1000).fit(Xb, y)
        p_base = base_mod.predict_proba(Xb)[:, 1]
        auc_base = roc_auc_score(y, p_base)
        endpoint_summary[endpoint] = {}
        for variant in VARIANTS:
            X = np.column_stack([Xb, df_p[variant].to_numpy()])
            mod = LogisticRegression(max_iter=1000).fit(X, y)
            p = mod.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, p)
            up_e = np.mean(p[y == 1] > p_base[y == 1])
            down_e = np.mean(p[y == 1] < p_base[y == 1])
            up_n = np.mean(p[y == 0] > p_base[y == 0])
            down_n = np.mean(p[y == 0] < p_base[y == 0])
            nri = (up_e - down_e) - (up_n - down_n)
            row = {
                "endpoint": endpoint,
                "variant": variant,
                "auroc": round(float(auc), 3),
                "auroc_baseline_age_sex_ses": round(float(auc_base), 3),
                "delta_auroc": round(float(auc - auc_base), 3),
                "brier": round(float(brier_score_loss(y, p)), 3),
                "average_precision": round(float(average_precision_score(y, p)), 3),
                "nri_vs_baseline": round(float(nri), 3),
            }
            endpoint_summary[endpoint][variant] = row
            discrim_rows.append(row)

        if endpoint == "frailty_24m":
            for variant in VARIANTS:
                X = np.column_stack([Xb, df_p[variant].to_numpy()])
                mod = LogisticRegression(max_iter=1000).fit(X, y)
                p = mod.predict_proba(X)[:, 1]
                thr = np.quantile(p, 0.80)
                yhat = (p >= thr).astype(int)
                ses_low = df_p["ses_quintile"] <= 2
                ses_high = df_p["ses_quintile"] >= 4
                sex_female = df_p["sex"] == 1
                sex_male = df_p["sex"] == 0

                def _gap(
                    mask_a: pd.Series,
                    mask_b: pd.Series,
                    predicted: npt.NDArray[np.int64],
                ) -> float:
                    tpr_a = float(
                        predicted[mask_a & (df_p["frailty_24m"] == 1)].mean()
                    )
                    fpr_a = float(
                        predicted[mask_a & (df_p["frailty_24m"] == 0)].mean()
                    )
                    tpr_b = float(
                        predicted[mask_b & (df_p["frailty_24m"] == 1)].mean()
                    )
                    fpr_b = float(
                        predicted[mask_b & (df_p["frailty_24m"] == 0)].mean()
                    )
                    return max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))

                fair_rows.append(
                    {
                        "variant": variant,
                        "ses_equalised_odds_gap": round(
                            _gap(ses_low, ses_high, yhat), 3
                        ),
                        "sex_equalised_odds_gap": round(
                            _gap(sex_female, sex_male, yhat), 3
                        ),
                    }
                )

    true_df = pd.DataFrame(discrim_rows).query("endpoint == 'multimorbidity_60m_true'")
    obs_df = pd.DataFrame(discrim_rows).query("endpoint == 'multimorbidity_60m_observed'")
    if not true_df.empty and not obs_df.empty:
        merged = true_df.merge(obs_df, on="variant", suffixes=("_true", "_obs"))
        for _, row in merged.iterrows():
            ascertainment_rows.append(
                {
                    "variant": row["variant"],
                    "auroc_true": row["auroc_true"],
                    "auroc_observed": row["auroc_obs"],
                    "ascertainment_attenuation": round(
                        float(row["auroc_true"] - row["auroc_obs"]), 3
                    ),
                }
            )

    fairness = pd.DataFrame(fair_rows)
    ascertainment = pd.DataFrame(ascertainment_rows)
    summary = {
        "n_prospective_after_attrition": int(len(df_p)),
        "max_frailty_auroc": round(
            float(
                pd.DataFrame(discrim_rows)
                .query("endpoint == 'frailty_24m'")["auroc"]
                .max()
            ),
            3,
        ),
        "max_ascertainment_attenuation": round(
            float(ascertainment["ascertainment_attenuation"].max()), 3
        )
        if not ascertainment.empty
        else 0.0,
        "max_ses_equalised_odds_gap": round(
            float(fairness["ses_equalised_odds_gap"].max()), 3
        )
        if not fairness.empty
        else 0.0,
    }
    artifacts = {
        "prospective_discrimination": pd.DataFrame(discrim_rows),
        "fairness_audit": fairness,
        "ascertainment_bias": ascertainment,
    }
    return summary, artifacts, df_p


def run_peai(
    *,
    seed_value: int = 20260507,
    n_experts: int = 30,
    n_dev: int = 1_500,
    n_external: int = 800,
    n_prospective: int = 2_800,
) -> Study:
    """Run the PEAI ecological/index-development simulation end to end."""
    rng = np.random.default_rng(seed_value)
    phase1 = _phase1_delphi(rng, n_experts=n_experts)
    dev = _build_peai_variants(
        _simulate_peai_cohort(n_dev, "development", rng),
        phase1["delphi_weights"],
    )
    external = _build_peai_variants(
        _simulate_peai_cohort(n_external, "external", rng),
        phase1["delphi_weights"],
    )
    dev, external = _project_supervised_variants(dev, external, seed_value)
    phase3, phase3_artifacts = _phase3_validation(dev, external)

    prospective = _build_peai_variants(
        _simulate_peai_cohort(n_prospective, "prospective", rng),
        phase1["delphi_weights"],
    )
    dev, prospective = _project_supervised_variants(dev, prospective, seed_value)
    phase4, phase4_artifacts, prospective_kept = _prospective_evaluation(
        prospective, rng
    )

    summary = {
        "phase1": phase1,
        "phase3": phase3,
        "phase4": phase4,
    }
    artifacts = {
        "development_variants": dev[
            ["id", "geography", "age", "sex", "ses_quintile"] + VARIANTS
        ].copy(),
        "external_variants": external[
            ["id", "geography", "age", "sex", "ses_quintile"] + VARIANTS
        ].copy(),
        **phase3_artifacts,
        **phase4_artifacts,
    }
    params = {
        "n_experts": n_experts,
        "n_dev": n_dev,
        "n_external": n_external,
        "n_prospective": n_prospective,
    }
    return build_study(
        design="ecological_peai",
        seed_value=seed_value,
        data=prospective_kept,
        params=params,
        results=summary,
        artifacts=artifacts,
    )
