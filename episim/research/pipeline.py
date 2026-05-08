"""Question-to-report orchestration for complete simulated studies."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from episim import __version__
from episim.core.reproducibility import Study
from episim.lab import get_design, list_designs, run_design, study_preview
from episim.reporting.ai_disclosure import block as ai_disclosure


@dataclass(frozen=True)
class ResearchPlan:
    """Protocol-level plan derived from a user research question."""

    question: str
    design_key: str
    title: str
    aim: str
    objectives: tuple[str, ...]
    hypotheses: tuple[str, ...]
    population: str
    exposure_or_intervention: str
    comparator: str
    outcomes: tuple[str, ...]
    methodology: str
    analysis_plan: tuple[str, ...]
    ethics_statement: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ResearchReport:
    """Manuscript-style report text and structured reporting artifacts."""

    title: str
    keywords: tuple[str, ...]
    abstract: str
    introduction: str
    methods: str
    results: str
    discussion: str
    conclusion: str
    limitations: tuple[str, ...]
    next_steps: tuple[str, ...]
    markdown: str


@dataclass(frozen=True)
class ResearchBundle:
    """Complete deterministic simulated research output."""

    plan: ResearchPlan
    study: Study
    collected_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    cleaning_log: pd.DataFrame
    analysis_steps_record: pd.DataFrame
    instruments: pd.DataFrame
    database_dictionary: pd.DataFrame
    collection_events: pd.DataFrame
    follow_up_schedule: pd.DataFrame
    outcome_record: pd.DataFrame
    observations: pd.DataFrame
    analysis_tables: dict[str, pd.DataFrame]
    figure_plan: pd.DataFrame
    realism_audit: pd.DataFrame
    sensitivity_analysis: pd.DataFrame
    readiness_checklist: pd.DataFrame
    guideline_checklist: pd.DataFrame
    references: pd.DataFrame
    declarations: pd.DataFrame
    report: ResearchReport

    def archive(self, path: str | Path) -> Path:
        """Write protocol, data tools, observations, report, and study archive."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "protocol.json").write_text(
            json.dumps(asdict(self.plan), indent=2, default=str),
            encoding="utf-8",
        )
        (out / "protocol.md").write_text(_protocol_markdown(self.plan), encoding="utf-8")
        (out / "report.md").write_text(self.report.markdown, encoding="utf-8")
        self.collected_data.to_csv(out / "collected_synthetic_data.csv", index=False)
        self.cleaned_data.to_csv(out / "cleaned_analysis_dataset.csv", index=False)
        self.cleaning_log.to_csv(out / "data_cleaning_log.csv", index=False)
        self.analysis_steps_record.to_csv(out / "analysis_steps.csv", index=False)
        self.instruments.to_csv(out / "data_collection_tools.csv", index=False)
        self.database_dictionary.to_csv(out / "database_dictionary.csv", index=False)
        self.collection_events.to_csv(out / "collection_events.csv", index=False)
        self.follow_up_schedule.to_csv(out / "follow_up_schedule.csv", index=False)
        self.outcome_record.to_csv(out / "outcome_record.csv", index=False)
        self.observations.to_csv(out / "observations.csv", index=False)
        self.realism_audit.to_csv(out / "realism_audit.csv", index=False)
        self.sensitivity_analysis.to_csv(out / "sensitivity_analysis.csv", index=False)
        self.readiness_checklist.to_csv(out / "readiness_checklist.csv", index=False)
        self.guideline_checklist.to_csv(out / "guideline_checklist.csv", index=False)
        self.references.to_csv(out / "references.csv", index=False)
        self.declarations.to_csv(out / "declarations.csv", index=False)
        table_dir = out / "tables"
        table_dir.mkdir(exist_ok=True)
        for name, frame in self.analysis_tables.items():
            frame.to_csv(table_dir / f"{name}.csv", index=False)
        figure_dir = out / "figures"
        figure_dir.mkdir(exist_ok=True)
        self.figure_plan.to_csv(figure_dir / "figure_plan.csv", index=False)
        _write_figures(self.study, self.analysis_tables, figure_dir)
        _write_sqlite_database(self, out / "synthetic_research_database.sqlite")
        self.study.archive(out / "study")
        return out


def plan_research(
    question: str,
    *,
    design_key: str | None = None,
    seed_value: int = 20260508,
    **parameter_overrides: Any,
) -> ResearchPlan:
    """Create a deterministic protocol plan from a research question."""
    clean_question = _clean_question(question)
    selected_design = design_key or _infer_design_key(clean_question)
    design = get_design(selected_design)
    params = dict(design.parameters)
    params["seed_value"] = seed_value
    params.update(parameter_overrides)
    title = _title_from_question(clean_question, design.title)
    target = _target_phrase(clean_question)
    exposure = _exposure_phrase(clean_question, design.family)
    comparator = _comparator_phrase(design.family)
    outcomes = _outcome_phrases(clean_question, selected_design)
    return ResearchPlan(
        question=clean_question,
        design_key=selected_design,
        title=title,
        aim=f"To simulate and evaluate {target} using a {design.title.lower()}.",
        objectives=(
            f"Define a reproducible synthetic population relevant to: {clean_question}",
            f"Collect simulated measurements using the {design.title.lower()} workflow.",
            "Estimate the primary effect or association with design-appropriate methods.",
            "Generate a complete transparent report with limitations and next steps.",
        ),
        hypotheses=_hypotheses(exposure, outcomes, design.family),
        population=_population_phrase(clean_question),
        exposure_or_intervention=exposure,
        comparator=comparator,
        outcomes=outcomes,
        methodology=_methodology_text(design.key, design.title, design.description),
        analysis_plan=_analysis_steps(design.key),
        ethics_statement=(
            "This workflow uses fully synthetic, seed-reproducible records. It does not "
            "contain real patient, participant, or community-identifiable data."
        ),
        parameters=params,
    )


def conduct_research(
    question: str,
    *,
    design_key: str | None = None,
    seed_value: int = 20260508,
    target_profile: dict[str, dict[str, float]] | None = None,
    sensitivity_runs: int = 3,
    **parameter_overrides: Any,
) -> ResearchBundle:
    """Run a complete simulated research study from question to report."""
    plan = plan_research(
        question,
        design_key=design_key,
        seed_value=seed_value,
        **parameter_overrides,
    )
    study = run_design(plan.design_key, **plan.parameters)
    collected_data = _make_collected_data(study)
    cleaned_data, cleaning_log = _make_cleaned_dataset(study)
    analysis_steps_record = _make_analysis_steps_record(plan, study)
    instruments = _make_instruments(plan, study)
    database_dictionary = _make_database_dictionary(plan, study, instruments)
    collection_events = _make_collection_events(plan, study)
    follow_up = _make_follow_up_schedule(plan.design_key)
    outcome_record = _make_outcome_record(plan, study)
    observations = _make_observations(study)
    analysis_tables = _make_analysis_tables(plan, study, outcome_record)
    realism_audit = _make_realism_audit(cleaned_data, target_profile)
    sensitivity_analysis = _make_sensitivity_analysis(plan, sensitivity_runs)
    readiness_checklist = _make_readiness_checklist(
        plan=plan,
        target_profile=target_profile,
        realism_audit=realism_audit,
        sensitivity_analysis=sensitivity_analysis,
    )
    analysis_tables["table_6_realism_audit"] = realism_audit
    analysis_tables["table_7_sensitivity_analysis"] = sensitivity_analysis
    analysis_tables["table_8_readiness_checklist"] = readiness_checklist
    figure_plan = _make_figure_plan(plan, study)
    guideline_checklist = _make_guideline_checklist(plan, study, analysis_tables)
    references = _make_references(plan.design_key)
    declarations = _make_declarations(plan)
    report = _make_report(
        plan,
        study,
        cleaning_log,
        analysis_steps_record,
        instruments,
        follow_up,
        outcome_record,
        analysis_tables,
        realism_audit,
        sensitivity_analysis,
        readiness_checklist,
        guideline_checklist,
        references,
        declarations,
    )
    return ResearchBundle(
        plan=plan,
        study=study,
        collected_data=collected_data,
        cleaned_data=cleaned_data,
        cleaning_log=cleaning_log,
        analysis_steps_record=analysis_steps_record,
        instruments=instruments,
        database_dictionary=database_dictionary,
        collection_events=collection_events,
        follow_up_schedule=follow_up,
        outcome_record=outcome_record,
        observations=observations,
        analysis_tables=analysis_tables,
        figure_plan=figure_plan,
        realism_audit=realism_audit,
        sensitivity_analysis=sensitivity_analysis,
        readiness_checklist=readiness_checklist,
        guideline_checklist=guideline_checklist,
        references=references,
        declarations=declarations,
        report=report,
    )


def _clean_question(question: str) -> str:
    cleaned = " ".join(question.strip().split())
    if not cleaned:
        raise ValueError("A non-empty research question is required.")
    return cleaned


def _infer_design_key(question: str) -> str:
    q = question.lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("qualitative_mixed_methods", ("interview", "theme", "lived experience", "humanities")),
        ("meta_analysis", ("meta-analysis", "systematic review", "pooled", "evidence synthesis")),
        ("agent_based_seir", ("seir", "epidemic", "outbreak", "infection transmission")),
        ("network_contagion", ("network", "contagion", "peer effect", "contact network")),
        ("markov_decision", ("markov", "cost-effectiveness", "qaly", "decision model")),
        ("microsimulation_lifetable", ("lifetable", "life-table", "microsimulation")),
        ("survival_cox", ("survival", "time to event", "mortality", "hazard")),
        (
            "instrumental_variables",
            ("instrumental", "instrument", "encouragement", "natural experiment"),
        ),
        (
            "propensity_score",
            ("propensity", "confounding", "inverse probability", "observational treatment"),
        ),
        ("regression_discontinuity", ("threshold", "cutoff", "eligibility score", "discontinuity")),
        ("interrupted_time_series", ("interrupted", "time series", "before and after", "policy")),
        ("stepped_wedge", ("stepped wedge", "rollout", "phased implementation")),
        ("rct_cluster", ("cluster", "school", "village", "hospital-level", "community trial")),
        ("rct_parallel", ("randomized", "randomised", "trial", "placebo")),
        ("case_control", ("case-control", "rare disease", "cases and controls")),
        ("cohort", ("cohort", "follow-up", "incidence", "risk of", "prospective")),
        ("ecological_peai", ("ecological", "index", "area-level", "fairness")),
    )
    for key, keywords in rules:
        if any(keyword in q for keyword in keywords):
            return key
    return "cross_sectional"


def _title_from_question(question: str, design_title: str) -> str:
    short = question.rstrip("?.")
    short = short[0].upper() + short[1:] if short else question
    return f"{short}: a simulated {design_title.lower()}"


def _target_phrase(question: str) -> str:
    words = question.rstrip("?.")
    return words[:140]


def _population_phrase(question: str) -> str:
    q = question.lower()
    if "older" in q or "ageing" in q or "aging" in q:
        return "Synthetic older-adult population with age, sex, exposure, and outcome structure."
    if "student" in q or "school" in q:
        return "Synthetic education or school-linked participant population."
    if "patient" in q or "clinical" in q:
        return "Synthetic clinical population generated without real patient records."
    if "community" in q or "village" in q:
        return "Synthetic community-based participant population."
    return "Synthetic study population matched to the broad domain of the question."


def _exposure_phrase(question: str, family: str) -> str:
    q = question.lower()
    if family == "experimental":
        return "Simulated intervention assignment"
    if "diet" in q:
        return "Dietary or nutrition-related exposure"
    if "exercise" in q or "physical activity" in q:
        return "Physical activity exposure"
    if "ai" in q or "algorithm" in q:
        return "AI-enabled exposure or decision-support process"
    if "policy" in q:
        return "Policy or implementation exposure"
    return "Primary exposure or intervention specified by the research question"


def _comparator_phrase(family: str) -> str:
    if family == "experimental":
        return "Usual care, control, delayed intervention, or non-treated arm."
    if family in {"causal inference", "quasi-experimental"}:
        return "Counterfactual comparison group approximated by the selected design."
    if family == "evidence synthesis":
        return "Between-study distribution of comparator-adjusted effects."
    return "Participants or units without the primary exposure."


def _outcome_phrases(question: str, design_key: str) -> tuple[str, ...]:
    q = question.lower()
    if design_key == "qualitative_mixed_methods":
        return ("theme saturation", "acceptability score", "integrated interpretation")
    if "mortality" in q or design_key == "survival_cox":
        return ("time-to-event outcome", "censoring status", "hazard ratio")
    if "infection" in q or design_key in {"agent_based_seir", "network_contagion"}:
        return ("incident infections", "peak prevalence", "final attack rate")
    if "cost" in q or design_key in {"markov_decision", "microsimulation_lifetable"}:
        return ("costs", "QALYs", "incremental net benefit")
    return ("primary health or social outcome", "effect estimate", "uncertainty interval")


def _hypotheses(
    exposure: str,
    outcomes: tuple[str, ...],
    family: str,
) -> tuple[str, ...]:
    verb = "changes" if family in {"experimental", "quasi-experimental"} else "is associated with"
    return (
        f"Null hypothesis: {exposure} has no effect on {outcomes[0]}.",
        f"Alternative hypothesis: {exposure} {verb} {outcomes[0]}.",
    )


def _methodology_text(design_key: str, title: str, description: str) -> str:
    checklist = _reporting_guideline(design_key)
    return (
        f"The study uses a {title.lower()} implemented in EPISIM. {description} "
        f"The simulation follows {checklist}-aligned reporting elements where applicable: "
        "eligibility, assignment or sampling, measurement, bias controls, analysis, and "
        "reproducibility metadata."
    )


def _analysis_steps(design_key: str) -> tuple[str, ...]:
    default = (
        "Generate deterministic synthetic records from the selected design.",
        "Summarize sample structure, measurements, and primary outcomes.",
        "Estimate design-specific association, causal, or decision metrics.",
        "Interpret findings as simulated evidence, not real-world efficacy proof.",
    )
    custom: dict[str, tuple[str, ...]] = {
        "cohort": (
            "Assemble baseline exposed and unexposed groups.",
            "Simulate longitudinal follow-up and attrition.",
            "Estimate cumulative incidence, risk difference, and risk ratio.",
            "Assess whether attrition could alter interpretation.",
        ),
        "rct_parallel": (
            "Randomize participants to intervention or control arms.",
            "Record outcomes after simulated follow-up.",
            "Estimate absolute and relative treatment effects.",
            "Summarize CONSORT-style flow and outcome completeness.",
        ),
        "survival_cox": (
            "Generate event times and administrative censoring.",
            "Compute person-time and event indicators.",
            "Estimate a hazard-ratio-style effect summary.",
            "Report censoring and follow-up limitations.",
        ),
        "qualitative_mixed_methods": (
            "Simulate interview-level theme discovery.",
            "Track saturation across sequential interviews.",
            "Simulate a quantitative survey strand.",
            "Integrate qualitative themes and quantitative scores.",
        ),
    }
    return custom.get(design_key, default)


def _reporting_guideline(design_key: str) -> str:
    if design_key in {"rct_parallel", "rct_cluster", "stepped_wedge"}:
        return "CONSORT"
    if design_key == "meta_analysis":
        return "PRISMA"
    if design_key == "qualitative_mixed_methods":
        return "COREQ/GRAMMS"
    if design_key in {"markov_decision", "microsimulation_lifetable"}:
        return "CHEERS"
    if design_key in {"agent_based_seir", "network_contagion"}:
        return "STRESS/ODD"
    return "STROBE"


def _make_instruments(plan: ResearchPlan, study: Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in study.data.columns:
        rows.append(
            {
                "instrument_section": _instrument_section(str(col)),
                "variable": str(col),
                "collection_mode": _collection_mode(str(col), plan.design_key),
                "timing": _variable_timing(str(col), plan.design_key),
                "definition": _variable_definition(str(col), plan),
                "quality_control": "Range checks, missingness checks, and seed audit.",
            }
        )
    if not rows:
        rows.append(
            {
                "instrument_section": "study_log",
                "variable": "study_summary",
                "collection_mode": "simulation log",
                "timing": "analysis",
                "definition": "Study-level synthetic result summary.",
                "quality_control": "Manifested deterministic output.",
            }
        )
    return pd.DataFrame(rows)


def _make_collected_data(study: Study) -> pd.DataFrame:
    collected = study.data.copy()
    collected.insert(0, "episim_record_id", range(1, len(collected) + 1))
    collected.insert(1, "synthetic_source_status", "collected")
    return collected


def _make_cleaned_dataset(study: Study) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = study.data.copy()
    cleaned.insert(0, "episim_record_id", range(1, len(cleaned) + 1))
    rows: list[dict[str, Any]] = []
    for column in cleaned.columns:
        series = cleaned[column]
        before_missing = int(series.isna().sum())
        action = "retained"
        detail = "No cleaning action required."
        if before_missing > 0:
            if pd.api.types.is_numeric_dtype(series):
                cleaned[column] = series.fillna(series.median())
                action = "median_imputation"
                detail = "Numeric missing values filled with synthetic sample median."
            else:
                cleaned[column] = series.fillna("missing")
                action = "missing_category"
                detail = "Non-numeric missing values assigned explicit missing category."
        if column == "episim_record_id":
            action = "primary_key_created"
            detail = "Sequential non-identifying synthetic research database key created."
        rows.append(
            {
                "variable": str(column),
                "before_missing": before_missing,
                "after_missing": int(cleaned[column].isna().sum()),
                "action": action,
                "detail": detail,
            }
        )
    return cleaned, pd.DataFrame(rows)


def _make_analysis_steps_record(plan: ResearchPlan, study: Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step_number, step in enumerate(plan.analysis_plan, start=1):
        rows.append(
            {
                "step_number": step_number,
                "analysis_step": step,
                "input_table": "cleaned_analysis_dataset",
                "output": _step_output(step_number, study),
                "reproducibility_note": f"Seed {study.seed}; design {plan.design_key}.",
            }
        )
    rows.append(
        {
            "step_number": len(rows) + 1,
            "analysis_step": "Generate manuscript assets and reproducibility archive.",
            "input_table": "all generated research tables",
            "output": "report.md, figures, SQLite database, and study manifest",
            "reproducibility_note": "Archive is deterministic for the saved seed and parameters.",
        }
    )
    return pd.DataFrame(rows)


def _step_output(step_number: int, study: Study) -> str:
    if step_number == 1:
        return f"Collected and cleaned {len(study.data)} synthetic records."
    if step_number == 2:
        return "Variable summary, data-quality checks, and database dictionary."
    if step_number == 3:
        return f"{len(study.results)} primary/secondary metrics in outcome_record."
    return "Interpretation, limitations, and next-step recommendations."


def _make_database_dictionary(
    plan: ResearchPlan,
    study: Study,
    instruments: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position, column in enumerate(study.data.columns, start=1):
        series = study.data[column]
        instrument_row = instruments.loc[instruments["variable"] == str(column)].head(1)
        rows.append(
            {
                "table_name": "synthetic_observations",
                "field_order": position,
                "field_name": str(column),
                "storage_type": _storage_type(series),
                "nullable": bool(series.isna().any()),
                "primary_key": False,
                "foreign_key": None,
                "source_instrument_section": (
                    str(instrument_row["instrument_section"].iloc[0])
                    if not instrument_row.empty
                    else "unknown"
                ),
                "collection_timing": _variable_timing(str(column), plan.design_key),
                "validation_rule": _validation_rule(series),
                "realism_note": _realism_note(str(column), plan),
            }
        )
    rows.insert(
        0,
        {
            "table_name": "synthetic_observations",
            "field_order": 0,
            "field_name": "episim_record_id",
            "storage_type": "integer",
            "nullable": False,
            "primary_key": True,
            "foreign_key": None,
            "source_instrument_section": "database_key",
            "collection_timing": "database_creation",
            "validation_rule": "unique sequential synthetic identifier",
            "realism_note": "Mimics a research database record key without identifying a person.",
        },
    )
    return pd.DataFrame(rows)


def _storage_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "real"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    return "text"


def _validation_rule(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (
            f"range {round(float(numeric.min()), 4)} to "
            f"{round(float(numeric.max()), 4)}; missing allowed: {bool(series.isna().any())}"
        )
    values = sorted(str(value) for value in series.dropna().unique()[:8])
    return "allowed values: " + ", ".join(values) if values else "free text synthetic field"


def _realism_note(column: str, plan: ResearchPlan) -> str:
    lowered = column.lower()
    if any(token in lowered for token in ("age", "sex", "cluster", "site")):
        return "Baseline demographic/context field commonly captured at enrollment."
    if any(token in lowered for token in ("treatment", "intervention", "exposure")):
        return f"Operationalises {plan.exposure_or_intervention}."
    if any(token in lowered for token in ("outcome", "event", "score", "death")):
        return f"Endpoint field aligned with {plan.outcomes[0]}."
    return "Synthetic covariate or process field retained for analysis reproducibility."


def _make_collection_events(plan: ResearchPlan, study: Study) -> pd.DataFrame:
    n_events = min(len(study.data), 250)
    if n_events == 0:
        return pd.DataFrame(
            columns=[
                "event_id",
                "episim_record_id",
                "timepoint",
                "mode",
                "collector_role",
                "completion_status",
                "query_flag",
                "audit_note",
            ]
        )
    schedule = _make_follow_up_schedule(plan.design_key)
    timepoints = schedule["timepoint"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    event_id = 1
    for record_id in range(1, n_events + 1):
        for timepoint in timepoints:
            query_flag = (record_id + len(timepoint)) % 17 == 0
            rows.append(
                {
                    "event_id": event_id,
                    "episim_record_id": record_id,
                    "timepoint": timepoint,
                    "mode": _event_mode(plan.design_key, timepoint),
                    "collector_role": _collector_role(plan.design_key),
                    "completion_status": "complete" if not query_flag else "queried_resolved",
                    "query_flag": query_flag,
                    "audit_note": (
                        "synthetic range/query check resolved"
                        if query_flag
                        else "synthetic source entry accepted"
                    ),
                }
            )
            event_id += 1
    return pd.DataFrame(rows)


def _event_mode(design_key: str, timepoint: str) -> str:
    if design_key == "qualitative_mixed_methods":
        return "interview_or_survey"
    if design_key in {"agent_based_seir", "network_contagion"}:
        return "daily_simulation_tick"
    if "baseline" in timepoint:
        return "baseline_crf"
    if "analysis" in timepoint:
        return "analysis_dataset_lock"
    return "follow_up_contact"


def _collector_role(design_key: str) -> str:
    if design_key == "qualitative_mixed_methods":
        return "qualitative_interviewer_and_survey_coordinator"
    if design_key in {"rct_parallel", "rct_cluster", "stepped_wedge"}:
        return "trial_research_coordinator"
    if design_key in {"agent_based_seir", "network_contagion"}:
        return "simulation_engine"
    return "field_investigator"


def _instrument_section(column: str) -> str:
    lowered = column.lower()
    if any(token in lowered for token in ("age", "sex", "cluster", "site", "period")):
        return "eligibility_and_context"
    if any(token in lowered for token in ("exposure", "treatment", "intervention", "arm")):
        return "exposure_or_assignment"
    if any(
        token in lowered
        for token in ("outcome", "event", "score", "infect", "death", "qaly", "cost")
    ):
        return "outcome_measurement"
    return "covariates_and_process"


def _collection_mode(column: str, design_key: str) -> str:
    lowered = column.lower()
    if design_key == "qualitative_mixed_methods":
        return "interview guide and survey form"
    if "cluster" in lowered or "period" in lowered:
        return "site or time-period case report form"
    if "event" in lowered or "outcome" in lowered:
        return "follow-up outcome assessment"
    return "baseline structured case record form"


def _variable_timing(column: str, design_key: str) -> str:
    lowered = column.lower()
    if design_key in {"cohort", "survival_cox"} and any(
        token in lowered for token in ("event", "time", "censor")
    ):
        return "follow-up"
    if "period" in lowered or "day" in lowered or "cycle" in lowered:
        return "repeated measures"
    if "outcome" in lowered or "event" in lowered:
        return "endpoint"
    return "baseline"


def _variable_definition(column: str, plan: ResearchPlan) -> str:
    lowered = column.lower().replace("_", " ")
    if "exposure" in lowered or "treatment" in lowered:
        return plan.exposure_or_intervention
    if "outcome" in lowered or "event" in lowered:
        return plan.outcomes[0]
    return f"Synthetic {lowered} field generated for the {plan.design_key} design."


def _make_follow_up_schedule(design_key: str) -> pd.DataFrame:
    schedules: dict[str, tuple[tuple[str, str, str], ...]] = {
        "cohort": (
            ("baseline", "eligibility, exposure, covariates", "structured form"),
            ("6 months", "interim status and retention", "follow-up contact"),
            ("12 months", "interim outcome screen", "follow-up contact"),
            ("24 months", "primary endpoint", "outcome assessment"),
        ),
        "rct_parallel": (
            ("baseline", "eligibility and randomization", "trial CRF"),
            ("post-intervention", "primary outcome", "blinded assessment"),
        ),
        "rct_cluster": (
            ("baseline", "cluster enrollment", "site log"),
            ("follow-up", "individual outcomes", "cluster CRF"),
        ),
        "stepped_wedge": (
            ("each period", "cluster-period exposure status", "rollout log"),
            ("each period", "individual outcome", "repeated cross-section CRF"),
        ),
        "survival_cox": (
            ("baseline", "risk factors and treatment", "clinical form"),
            ("continuous", "event time and censoring", "survival follow-up log"),
        ),
    }
    default = (
        ("baseline", "synthetic eligibility, exposure, and covariates", "structured form"),
        ("analysis", "outcome and summary metrics", "analysis log"),
    )
    rows = [
        {"timepoint": item[0], "activity": item[1], "tool": item[2]}
        for item in schedules.get(design_key, default)
    ]
    return pd.DataFrame(rows)


def _make_outcome_record(plan: ResearchPlan, study: Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, value in study.results.items():
        rows.append(
            {
                "outcome_domain": _metric_domain(metric),
                "metric": metric,
                "value": value,
                "interpretation": _metric_interpretation(metric, value, plan),
            }
        )
    if not rows:
        rows.append(
            {
                "outcome_domain": "process",
                "metric": "records",
                "value": len(study.data),
                "interpretation": "Synthetic records generated successfully.",
            }
        )
    return pd.DataFrame(rows)


def _metric_domain(metric: str) -> str:
    lowered = metric.lower()
    if any(token in lowered for token in ("risk", "odds", "hazard", "effect", "ratio", "late")):
        return "effect_estimation"
    if any(token in lowered for token in ("cost", "qaly", "benefit")):
        return "decision_outcome"
    if any(token in lowered for token in ("attrition", "censor", "follow")):
        return "follow_up_quality"
    return "descriptive_or_process"


def _metric_interpretation(metric: str, value: Any, plan: ResearchPlan) -> str:
    if isinstance(value, int | float):
        return f"Simulated {metric} was {value}; interpret within the {plan.design_key} design."
    return f"Simulated {metric} summary generated for the {plan.design_key} design."


def _make_observations(study: Study, rows: int = 25) -> pd.DataFrame:
    data_preview, _ = study_preview(study, rows=rows)
    observation = data_preview.copy()
    observation.insert(0, "observation_id", range(1, len(observation) + 1))
    observation.insert(1, "source", "synthetic_episim_record")
    return observation


def _make_analysis_tables(
    plan: ResearchPlan,
    study: Study,
    outcome_record: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {
        "table_1_study_profile": _study_profile_table(plan, study),
        "table_2_variable_summary": _variable_summary_table(study.data),
        "table_3_primary_results": outcome_record.copy(),
        "table_4_data_quality": _data_quality_table(study.data),
        "table_5_analysis_interpretation": _analysis_interpretation_table(plan, study),
    }
    if study.artifacts:
        tables["table_6_artifact_index"] = _artifact_index_table(study)
    return tables


def _study_profile_table(plan: ResearchPlan, study: Study) -> pd.DataFrame:
    design = get_design(plan.design_key)
    rows = [
        ("research_question", plan.question),
        ("design", design.title),
        ("design_family", design.family),
        ("reporting_guideline", _reporting_guideline(plan.design_key)),
        ("synthetic_records", len(study.data)),
        ("seed", study.seed),
        ("population", plan.population),
        ("exposure_or_intervention", plan.exposure_or_intervention),
        ("comparator", plan.comparator),
        ("primary_outcome", plan.outcomes[0]),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _variable_summary_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in data.columns:
        series = data[column]
        non_missing = int(series.notna().sum())
        row: dict[str, Any] = {
            "variable": str(column),
            "dtype": str(series.dtype),
            "non_missing": non_missing,
            "missing": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
        }
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            row.update(
                {
                    "mean": round(float(numeric.mean()), 4),
                    "sd": round(float(numeric.std(ddof=1)), 4)
                    if numeric.notna().sum() > 1
                    else 0.0,
                    "min": round(float(numeric.min()), 4),
                    "median": round(float(numeric.median()), 4),
                    "max": round(float(numeric.max()), 4),
                }
            )
        else:
            mode = series.mode(dropna=True)
            row.update(
                {
                    "mean": None,
                    "sd": None,
                    "min": None,
                    "median": None,
                    "max": None,
                    "most_common": str(mode.iloc[0]) if not mode.empty else None,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _data_quality_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in data.columns:
        series = data[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "check": "missingness",
                "variable": str(column),
                "value": missing,
                "denominator": len(series),
                "rate": round(missing / max(len(series), 1), 4),
                "status": "pass" if missing == 0 else "review",
            }
        )
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            outlier_count = int(
                (
                    (numeric < numeric.quantile(0.01))
                    | (numeric > numeric.quantile(0.99))
                ).sum()
            )
            rows.append(
                {
                    "check": "distribution_tail",
                    "variable": str(column),
                    "value": outlier_count,
                    "denominator": int(numeric.notna().sum()),
                    "rate": round(outlier_count / max(int(numeric.notna().sum()), 1), 4),
                    "status": "informational",
                }
            )
    return pd.DataFrame(rows)


def _analysis_interpretation_table(plan: ResearchPlan, study: Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, value in study.results.items():
        rows.append(
            {
                "metric": metric,
                "value": value,
                "analysis_role": _metric_domain(metric),
                "direction": _direction_label(value),
                "manuscript_sentence": _metric_sentence(metric, value, plan),
            }
        )
    if not rows:
        rows.append(
            {
                "metric": "records",
                "value": len(study.data),
                "analysis_role": "process",
                "direction": "not_applicable",
                "manuscript_sentence": (
                    f"The simulation generated {len(study.data)} synthetic records."
                ),
            }
        )
    return pd.DataFrame(rows)


def _direction_label(value: Any) -> str:
    if isinstance(value, int | float):
        if value > 1:
            return "above_reference"
        if value < 0:
            return "negative"
        if 0 < value < 1:
            return "below_one_or_fractional"
    return "descriptive"


def _metric_sentence(metric: str, value: Any, plan: ResearchPlan) -> str:
    if isinstance(value, int | float):
        return (
            f"For {plan.outcomes[0]}, the simulated {metric.replace('_', ' ')} "
            f"was {value} under the {plan.design_key} design."
        )
    return (
        f"The simulated {metric.replace('_', ' ')} was recorded as {value}; "
        "interpretation depends on the design-specific scale."
    )


def _artifact_index_table(study: Study) -> pd.DataFrame:
    rows = []
    for name, frame in study.artifacts.items():
        rows.append(
            {
                "artifact": name,
                "rows": len(frame),
                "columns": ", ".join(str(col) for col in frame.columns),
            }
        )
    return pd.DataFrame(rows)


def _make_realism_audit(
    cleaned_data: pd.DataFrame,
    target_profile: dict[str, dict[str, float]] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = target_profile or {}
    for column in cleaned_data.columns:
        if column == "episim_record_id":
            continue
        series = cleaned_data[column]
        numeric = pd.to_numeric(series, errors="coerce")
        target = targets.get(str(column), {})
        if numeric.notna().any():
            mean_value = float(numeric.mean())
            sd_value = float(numeric.std(ddof=1)) if numeric.notna().sum() > 1 else 0.0
            min_value = float(numeric.min())
            max_value = float(numeric.max())
            target_mean = target.get("mean")
            target_min = target.get("min")
            target_max = target.get("max")
            deviation = (
                abs(mean_value - target_mean) / max(abs(target_mean), 1.0)
                if target_mean is not None
                else None
            )
            status = _realism_status(deviation, min_value, max_value, target_min, target_max)
            rows.append(
                {
                    "variable": str(column),
                    "observed_mean": round(mean_value, 4),
                    "observed_sd": round(sd_value, 4),
                    "observed_min": round(min_value, 4),
                    "observed_max": round(max_value, 4),
                    "target_mean": target_mean,
                    "target_min": target_min,
                    "target_max": target_max,
                    "relative_mean_deviation": round(deviation, 4)
                    if deviation is not None
                    else None,
                    "status": status,
                    "calibration_note": _calibration_note(target),
                }
            )
        else:
            rows.append(
                {
                    "variable": str(column),
                    "observed_mean": None,
                    "observed_sd": None,
                    "observed_min": None,
                    "observed_max": None,
                    "target_mean": None,
                    "target_min": None,
                    "target_max": None,
                    "relative_mean_deviation": None,
                    "status": "descriptive_review",
                    "calibration_note": "Non-numeric field; inspect category frequencies.",
                }
            )
    if not rows:
        rows.append(
            {
                "variable": "dataset",
                "observed_mean": None,
                "observed_sd": None,
                "observed_min": None,
                "observed_max": None,
                "target_mean": None,
                "target_min": None,
                "target_max": None,
                "relative_mean_deviation": None,
                "status": "no_fields",
                "calibration_note": "No auditable fields were available.",
            }
        )
    return pd.DataFrame(rows)


def _realism_status(
    deviation: float | None,
    observed_min: float,
    observed_max: float,
    target_min: float | None,
    target_max: float | None,
) -> str:
    if target_min is not None and observed_min < target_min:
        return "outside_target_range"
    if target_max is not None and observed_max > target_max:
        return "outside_target_range"
    if deviation is None:
        return "needs_external_target"
    if deviation <= 0.10:
        return "calibrated_close"
    if deviation <= 0.25:
        return "calibrated_moderate"
    return "calibration_review"


def _calibration_note(target: dict[str, float]) -> str:
    if target:
        return "Compared with user-supplied target profile."
    return (
        "No external target supplied; calibrate with registry, literature, "
        "public synthetic FHIR/Synthea data, or authenticated cloud outputs."
    )


def _make_sensitivity_analysis(plan: ResearchPlan, sensitivity_runs: int) -> pd.DataFrame:
    runs = max(1, min(int(sensitivity_runs), 12))
    metric_rows: list[dict[str, Any]] = []
    base_seed = int(plan.parameters.get("seed_value", 20260508))
    for index in range(runs):
        params = dict(plan.parameters)
        params["seed_value"] = base_seed + index
        study = run_design(plan.design_key, **params)
        numeric_results = {
            key: value for key, value in study.results.items() if isinstance(value, int | float)
        }
        if not numeric_results:
            metric_rows.append(
                {
                    "metric": "records",
                    "seed": params["seed_value"],
                    "value": len(study.data),
                    "summary_type": "run",
                }
            )
        for metric, value in numeric_results.items():
            metric_rows.append(
                {
                    "metric": metric,
                    "seed": params["seed_value"],
                    "value": float(value),
                    "summary_type": "run",
                }
            )
    raw = pd.DataFrame(metric_rows)
    summaries: list[dict[str, Any]] = []
    for metric, group in raw.groupby("metric", sort=True):
        values = pd.to_numeric(group["value"], errors="coerce")
        summaries.append(
            {
                "metric": metric,
                "seed": "all",
                "value": round(float(values.mean()), 6),
                "summary_type": "mean",
            }
        )
        summaries.append(
            {
                "metric": metric,
                "seed": "all",
                "value": round(float(values.std(ddof=1)), 6)
                if len(values.dropna()) > 1
                else 0.0,
                "summary_type": "sd",
            }
        )
        summaries.append(
            {
                "metric": metric,
                "seed": "all",
                "value": round(float(values.min()), 6),
                "summary_type": "min",
            }
        )
        summaries.append(
            {
                "metric": metric,
                "seed": "all",
                "value": round(float(values.max()), 6),
                "summary_type": "max",
            }
        )
    return pd.concat([raw, pd.DataFrame(summaries)], ignore_index=True)


def _make_readiness_checklist(
    *,
    plan: ResearchPlan,
    target_profile: dict[str, dict[str, float]] | None,
    realism_audit: pd.DataFrame,
    sensitivity_analysis: pd.DataFrame,
) -> pd.DataFrame:
    has_target = bool(target_profile)
    realism_flags = int(
        realism_audit["status"].isin(["outside_target_range", "calibration_review"]).sum()
    )
    rows = [
        (
            "research_question",
            "complete",
            f"Question mapped to {plan.design_key}.",
        ),
        (
            "design_execution",
            "complete",
            "Registered EPISIM design executed with deterministic seed.",
        ),
        (
            "realism_calibration",
            "complete" if has_target and realism_flags == 0 else "needs_review",
            "Target profile supplied and checked."
            if has_target
            else "No external target profile supplied.",
        ),
        (
            "sensitivity_analysis",
            "complete" if not sensitivity_analysis.empty else "needs_review",
            "Multi-seed metric stability table generated.",
        ),
        (
            "manuscript_assets",
            "complete",
            "Report, tables, figures, references, and declarations generated.",
        ),
        (
            "real_world_inference",
            "not_applicable",
            "Synthetic study only; real-world inference requires empirical calibration.",
        ),
    ]
    return pd.DataFrame(rows, columns=["domain", "status", "note"])


def _make_figure_plan(plan: ResearchPlan, study: Study) -> pd.DataFrame:
    rows = [
        {
            "figure": "figure_1_metric_summary.png",
            "title": "Primary simulated result metrics",
            "purpose": "Visual summary of numeric design-specific estimates.",
        },
        {
            "figure": "figure_2_data_structure.png",
            "title": "Synthetic data structure",
            "purpose": "Shows available numeric fields and their relative distributions.",
        },
    ]
    if len(_candidate_numeric_columns(study.data)) >= 1:
        rows.append(
            {
                "figure": "figure_3_outcome_distribution.png",
                "title": f"Distribution relevant to {plan.outcomes[0]}",
                "purpose": "Inspects the main synthetic outcome or first numeric endpoint.",
            }
        )
    return pd.DataFrame(rows)


def _candidate_numeric_columns(data: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in data.columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.notna().any():
            columns.append(str(column))
    return columns


def _write_figures(
    study: Study,
    analysis_tables: dict[str, pd.DataFrame],
    figure_dir: Path,
) -> None:
    _write_metric_figure(study, figure_dir / "figure_1_metric_summary.png")
    _write_data_structure_figure(
        analysis_tables["table_2_variable_summary"],
        figure_dir / "figure_2_data_structure.png",
    )
    _write_distribution_figure(study.data, figure_dir / "figure_3_outcome_distribution.png")


def _write_metric_figure(study: Study, path: Path) -> None:
    numeric_results = {
        key: value for key, value in study.results.items() if isinstance(value, int | float)
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if numeric_results:
        labels = list(numeric_results)[:8]
        values = [float(numeric_results[label]) for label in labels]
        ax.barh(range(len(labels)), values, color="#2f6f73")
        ax.set_yticks(range(len(labels)), labels=[label.replace("_", " ") for label in labels])
        ax.set_xlabel("Simulated estimate")
    else:
        ax.text(0.5, 0.5, "No numeric summary metrics", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("EPISIM simulated result metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_data_structure_figure(summary: pd.DataFrame, path: Path) -> None:
    plot_data = summary.head(12).copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = pd.to_numeric(plot_data["unique_values"], errors="coerce").fillna(0)
    ax.bar(range(len(plot_data)), values, color="#bf6f3f")
    ax.set_xticks(
        range(len(plot_data)),
        labels=[str(value) for value in plot_data["variable"]],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Unique values")
    ax.set_title("Synthetic data structure")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_distribution_figure(data: pd.DataFrame, path: Path) -> None:
    candidates = _candidate_numeric_columns(data)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if candidates:
        column = candidates[-1]
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        ax.hist(values, bins=min(30, max(5, int(np.sqrt(len(values))))), color="#495f8c")
        ax.set_xlabel(column.replace("_", " "))
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {column.replace('_', ' ')}")
    else:
        ax.text(0.5, 0.5, "No numeric columns available", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_sqlite_database(bundle: ResearchBundle, path: Path) -> None:
    with sqlite3.connect(path) as con:
        _to_sqlite(bundle.collected_data, "collected_synthetic_data", con)
        _to_sqlite(bundle.cleaned_data, "cleaned_analysis_dataset", con)
        _to_sqlite(bundle.cleaning_log, "data_cleaning_log", con)
        _to_sqlite(bundle.analysis_steps_record, "analysis_steps", con)
        _to_sqlite(bundle.cleaned_data, "synthetic_observations", con)
        _to_sqlite(bundle.instruments, "data_collection_tools", con)
        _to_sqlite(bundle.database_dictionary, "database_dictionary", con)
        _to_sqlite(bundle.collection_events, "collection_events", con)
        _to_sqlite(bundle.follow_up_schedule, "follow_up_schedule", con)
        _to_sqlite(bundle.outcome_record, "outcome_record", con)
        _to_sqlite(bundle.realism_audit, "realism_audit", con)
        _to_sqlite(bundle.sensitivity_analysis, "sensitivity_analysis", con)
        _to_sqlite(bundle.readiness_checklist, "readiness_checklist", con)
        _to_sqlite(bundle.guideline_checklist, "guideline_checklist", con)
        _to_sqlite(bundle.references, "references", con)
        _to_sqlite(bundle.declarations, "declarations", con)
        for name, frame in bundle.analysis_tables.items():
            _to_sqlite(frame, name, con)


def _to_sqlite(frame: pd.DataFrame, table_name: str, con: sqlite3.Connection) -> None:
    safe = frame.map(_sqlite_value)
    safe.to_sql(table_name, con, index=False, if_exists="replace")


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, default=str)
    if pd.isna(value):
        return None
    return value


def _make_guideline_checklist(
    plan: ResearchPlan,
    study: Study,
    analysis_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    guideline = _reporting_guideline(plan.design_key)
    items = _guideline_items(guideline)
    rows: list[dict[str, Any]] = []
    for item, expectation in items:
        rows.append(
            {
                "guideline": guideline,
                "item": item,
                "expectation": expectation,
                "status": "addressed",
                "location": _guideline_location(item),
                "evidence": _guideline_evidence(item, plan, study, analysis_tables),
            }
        )
    return pd.DataFrame(rows)


def _guideline_items(guideline: str) -> tuple[tuple[str, str], ...]:
    core = (
        ("title_abstract", "Identify design and summarize objective, methods, results."),
        ("background_rationale", "Explain rationale and scientific context."),
        ("objectives", "State specific objectives or hypotheses."),
        ("study_design", "Present key study-design elements early."),
        ("setting_participants", "Describe population, eligibility, and setting."),
        ("variables_data_sources", "Define variables, tools, and measurement sources."),
        ("bias_reproducibility", "Describe bias controls and reproducibility safeguards."),
        ("statistical_methods", "Describe analytical methods and uncertainty handling."),
        ("results_participants", "Report generated sample and outcome completeness."),
        ("main_results", "Report primary estimates and interpretation."),
        ("limitations", "Discuss limitations and generalisability."),
        ("funding_declarations", "Provide declarations, funding, and conflicts."),
    )
    if guideline == "CONSORT":
        return core + (
            ("allocation", "Describe allocation, comparator, and intervention delivery."),
            ("participant_flow", "Summarize simulated recruitment and follow-up flow."),
        )
    if guideline == "PRISMA":
        return core + (
            ("eligibility_information_sources", "Describe study-level inclusion logic."),
            ("synthesis_methods", "Describe pooling and heterogeneity methods."),
        )
    if guideline == "COREQ/GRAMMS":
        return core + (
            ("research_team_reflexivity", "Describe analyst role and reflexivity."),
            ("integration", "Explain qualitative-quantitative integration."),
        )
    if guideline == "CHEERS":
        return core + (
            ("perspective_time_horizon", "Describe economic perspective and time horizon."),
            ("resource_valuation", "Report cost and outcome valuation assumptions."),
        )
    if guideline == "STRESS/ODD":
        return core + (
            ("model_entities_processes", "Describe simulated agents/entities and processes."),
            ("initialisation_sensitivity", "Report initialisation and sensitivity needs."),
        )
    return core


def _guideline_location(item: str) -> str:
    if item in {"title_abstract"}:
        return "Title; Abstract"
    if item in {"background_rationale", "objectives"}:
        return "Introduction"
    if item in {
        "study_design",
        "setting_participants",
        "variables_data_sources",
        "bias_reproducibility",
        "statistical_methods",
        "allocation",
        "eligibility_information_sources",
        "synthesis_methods",
        "research_team_reflexivity",
        "integration",
        "perspective_time_horizon",
        "resource_valuation",
        "model_entities_processes",
        "initialisation_sensitivity",
    }:
        return "Methods"
    if item in {"results_participants", "main_results", "participant_flow"}:
        return "Results"
    if item == "limitations":
        return "Discussion"
    return "Declarations"


def _guideline_evidence(
    item: str,
    plan: ResearchPlan,
    study: Study,
    analysis_tables: dict[str, pd.DataFrame],
) -> str:
    if item == "main_results":
        return f"{len(study.results)} design-specific metrics in table_3_primary_results."
    if item == "variables_data_sources":
        return f"{len(study.data.columns)} synthetic variables documented."
    if item == "results_participants":
        return f"{len(study.data)} synthetic records generated with seed {study.seed}."
    if item == "study_design":
        return f"Design selected as {plan.design_key}."
    if item == "statistical_methods":
        return "; ".join(plan.analysis_plan)
    if item == "bias_reproducibility":
        return "Seeded run, manifest, synthetic-data disclosure, and data-quality table."
    if item == "funding_declarations":
        return "Structured declarations table produced."
    return f"Addressed in {len(analysis_tables)} analysis tables and manuscript sections."


def _make_references(design_key: str) -> pd.DataFrame:
    guideline = _reporting_guideline(design_key)
    references = [
        {
            "key": "equator",
            "citation": (
                "EQUATOR Network. Enhancing the QUAlity and Transparency Of health "
                "Research reporting guideline library."
            ),
            "url": "https://www.equator-network.org/",
        },
        {
            "key": "strobe",
            "citation": (
                "von Elm E, Altman DG, Egger M, Pocock SJ, Gotzsche PC, "
                "Vandenbroucke JP. The STROBE statement."
            ),
            "url": "https://www.strobe-statement.org/",
        },
        {
            "key": "episim",
            "citation": (
                f"Siddalingaiah H. S. EPISIM: Epidemiology Platform for In Silico "
                f"Methods. Version {__version__}."
            ),
            "url": "https://github.com/hssling/episim",
        },
    ]
    additions: dict[str, dict[str, str]] = {
        "CONSORT": {
            "key": "consort",
            "citation": "Schulz KF, Altman DG, Moher D. CONSORT statement for trials.",
            "url": "https://www.consort-spirit.org/",
        },
        "PRISMA": {
            "key": "prisma",
            "citation": "Page MJ et al. PRISMA 2020 statement for systematic reviews.",
            "url": "https://www.prisma-statement.org/prisma-2020",
        },
        "COREQ/GRAMMS": {
            "key": "coreq_gramms",
            "citation": "COREQ and GRAMMS reporting guidance for qualitative and mixed methods.",
            "url": "https://www.equator-network.org/",
        },
        "CHEERS": {
            "key": "cheers",
            "citation": "CHEERS reporting guidance for health economic evaluations.",
            "url": "https://www.equator-network.org/reporting-guidelines/cheers/",
        },
        "STRESS/ODD": {
            "key": "stress_odd",
            "citation": "STRESS and ODD guidance for simulation and agent-based models.",
            "url": "https://www.equator-network.org/",
        },
    }
    if guideline in additions:
        references.insert(1, additions[guideline])
    return pd.DataFrame(references)


def _make_declarations(plan: ResearchPlan) -> pd.DataFrame:
    rows = [
        ("ethics_approval", plan.ethics_statement),
        ("consent", "Not applicable: no real participants or identifiable records were used."),
        (
            "data_availability",
            "All synthetic outputs are generated in the research bundle archive.",
        ),
        (
            "code_availability",
            "EPISIM source code and deterministic seed are reported in the archive.",
        ),
        ("funding", "No specific funding was simulated or declared by the software."),
        ("conflicts_of_interest", "No conflicts of interest were simulated or declared."),
        (
            "author_contributions",
            "User supplied the research question; EPISIM generated synthetic assets.",
        ),
        (
            "clinical_policy_disclaimer",
            "Synthetic findings are not clinical, policy, or public-health evidence.",
        ),
    ]
    return pd.DataFrame(rows, columns=["declaration", "statement"])


def _make_report(
    plan: ResearchPlan,
    study: Study,
    cleaning_log: pd.DataFrame,
    analysis_steps_record: pd.DataFrame,
    instruments: pd.DataFrame,
    follow_up: pd.DataFrame,
    outcome_record: pd.DataFrame,
    analysis_tables: dict[str, pd.DataFrame],
    realism_audit: pd.DataFrame,
    sensitivity_analysis: pd.DataFrame,
    readiness_checklist: pd.DataFrame,
    guideline_checklist: pd.DataFrame,
    references: pd.DataFrame,
    declarations: pd.DataFrame,
) -> ResearchReport:
    result_lines = _result_lines(study)
    guideline = _reporting_guideline(plan.design_key)
    table_inventory = ", ".join(name for name in analysis_tables)
    limitations = (
        (
            "All records are synthetic and should be used for method development, "
            "training, and planning."
        ),
        "Parameter values are generic defaults unless explicitly overridden by the user.",
        "External validity requires comparison with real-world domain data before policy use.",
    )
    next_steps = (
        "Refine simulation parameters using literature, registry, or pilot data.",
        "Run sensitivity analyses across plausible effect sizes and bias mechanisms.",
        (
            "Convert the protocol into a real ethics-approved study only if real data "
            "collection is intended."
        ),
    )
    keywords = (
        "simulation",
        "epidemiology",
        plan.design_key.replace("_", " "),
        "synthetic data",
        guideline,
    )
    abstract = (
        f"Background: {plan.question}\n"
        f"Objective: {plan.aim}\n"
        f"Methods: EPISIM generated a {guideline}-aligned synthetic {plan.design_key} "
        f"study using seed {study.seed}. Structured data-collection tools, follow-up "
        "logs, data-quality checks, analysis tables, and figures were produced.\n"
        f"Results: {len(study.data)} records and {len(study.results)} summary metrics "
        f"were generated. {result_lines[0] if result_lines else ''}\n"
        "Conclusions: The workflow provides simulated evidence for design development "
        "and protocol testing, not real-world effectiveness evidence."
    )
    introduction = (
        f"The research question was: {plan.question} The simulation was designed to "
        "test whether the proposed design can operationalise the target population, "
        "measurement strategy, follow-up logic, outcome capture, and analysis workflow "
        "before any real participant data are collected."
    )
    methods = (
        f"Design and guideline basis: {plan.methodology}\n\n"
        f"Population and setting: {plan.population}\n\n"
        f"Exposure/intervention and comparator: {plan.exposure_or_intervention}; "
        f"comparator was {plan.comparator}\n\n"
        f"Data collection: {len(instruments)} variable-level fields were mapped to "
        "instrument sections, timing, collection mode, definitions, and quality-control "
        f"checks. The follow-up schedule contained {len(follow_up)} planned timepoints.\n\n"
        f"Data management: {len(study.data)} collected synthetic records were converted "
        "into a cleaned analysis dataset with a non-identifying record key. The cleaning "
        f"log contains {len(cleaning_log)} variable-level checks.\n\n"
        f"Analysis: {'; '.join(plan.analysis_plan)} Data-quality checks, variable "
        f"summaries, primary estimates, and interpretation tables were generated. "
        f"Analysis table inventory: {table_inventory}. A step-by-step analysis log "
        f"contains {len(analysis_steps_record)} recorded steps. Sensitivity analysis "
        f"contains {len(sensitivity_analysis)} rows across seed perturbations."
    )
    results = (
        f"The simulation generated {len(study.data)} records with seed {study.seed}. "
        f"Outcome recording produced {len(outcome_record)} metrics. "
        f"Realism audit produced {len(realism_audit)} field checks and readiness "
        f"assessment produced {len(readiness_checklist)} quality gates. "
        f"The guideline checklist marked {len(guideline_checklist)} reporting items as "
        "addressed in the generated manuscript package. "
        + " ".join(result_lines)
    )
    discussion = (
        "The simulated study demonstrates whether the proposed design can collect the "
        "required measurements, maintain a transparent follow-up structure, and return "
        "design-appropriate estimates. The additional tables and figures allow reviewers "
        "to inspect data structure, missingness, model outputs, and interpretation logic. "
        "Findings should be interpreted as scenario-based methodological evidence rather "
        "than empirical clinical or social proof."
    )
    conclusion = (
        f"EPISIM completed an end-to-end synthetic {plan.design_key} study for the "
        "research question, including protocol, tools, observations, outcome recording, "
        "analysis, and manuscript-style interpretation."
    )
    markdown = _report_markdown(
        plan=plan,
        keywords=keywords,
        abstract=abstract,
        introduction=introduction,
        methods=methods,
        results=results,
        discussion=discussion,
        conclusion=conclusion,
        limitations=limitations,
        next_steps=next_steps,
        guideline_checklist=guideline_checklist,
        cleaning_log=cleaning_log,
        analysis_steps_record=analysis_steps_record,
        realism_audit=realism_audit,
        sensitivity_analysis=sensitivity_analysis,
        readiness_checklist=readiness_checklist,
        references=references,
        declarations=declarations,
    )
    return ResearchReport(
        title=plan.title,
        keywords=keywords,
        abstract=abstract,
        introduction=introduction,
        methods=methods,
        results=results,
        discussion=discussion,
        conclusion=conclusion,
        limitations=limitations,
        next_steps=next_steps,
        markdown=markdown,
    )


def _result_lines(study: Study) -> list[str]:
    lines: list[str] = []
    for metric, value in list(study.results.items())[:8]:
        lines.append(f"{metric}={value}.")
    return lines


def _protocol_markdown(plan: ResearchPlan) -> str:
    return "\n".join(
        [
            f"# {plan.title}",
            "",
            "## Research Question",
            plan.question,
            "",
            "## Aim",
            plan.aim,
            "",
            "## Objectives",
            *[f"- {item}" for item in plan.objectives],
            "",
            "## Hypotheses",
            *[f"- {item}" for item in plan.hypotheses],
            "",
            "## Methodology",
            plan.methodology,
            "",
            "## Analysis Plan",
            *[f"- {item}" for item in plan.analysis_plan],
            "",
            "## Ethics",
            plan.ethics_statement,
            "",
        ]
    )


def _report_markdown(
    *,
    plan: ResearchPlan,
    keywords: tuple[str, ...],
    abstract: str,
    introduction: str,
    methods: str,
    results: str,
    discussion: str,
    conclusion: str,
    limitations: tuple[str, ...],
    next_steps: tuple[str, ...],
    guideline_checklist: pd.DataFrame,
    cleaning_log: pd.DataFrame,
    analysis_steps_record: pd.DataFrame,
    realism_audit: pd.DataFrame,
    sensitivity_analysis: pd.DataFrame,
    readiness_checklist: pd.DataFrame,
    references: pd.DataFrame,
    declarations: pd.DataFrame,
) -> str:
    guideline = _reporting_guideline(plan.design_key)
    return "\n".join(
        [
            f"# {plan.title}",
            "",
            "## Title Page",
            f"**Design:** {plan.design_key}",
            f"**Reporting framework:** {guideline}",
            "**Article type:** Synthetic simulation study",
            "",
            "## Abstract",
            abstract,
            "",
            "## Keywords",
            ", ".join(keywords),
            "",
            "## Introduction",
            introduction,
            "",
            "### Aim And Objectives",
            plan.aim,
            "",
            *[f"- {item}" for item in plan.objectives],
            "",
            "### Hypotheses",
            *[f"- {item}" for item in plan.hypotheses],
            "",
            "## Methods",
            methods,
            "",
            "### Simulated Data Collection Tools",
            (
                "The data collection tool package maps every generated field to an "
                "instrument section, timing, collection mode, operational definition, "
                "and quality-control check."
            ),
            "",
            "### Data Cleaning And Database Production",
            (
                "The archive includes the collected synthetic dataset, cleaned analysis "
                "dataset, data dictionary, cleaning log, collection-event audit trail, "
                "and a SQLite research database."
            ),
            "",
            _markdown_table(cleaning_log[["variable", "action", "after_missing"]].head(20)),
            "",
            "### Analysis Steps",
            _markdown_table(analysis_steps_record),
            "",
            "### Realism And Calibration Audit",
            _markdown_table(
                realism_audit[
                    ["variable", "status", "relative_mean_deviation", "calibration_note"]
                ].head(20)
            ),
            "",
            "### Sensitivity Analysis",
            _markdown_table(sensitivity_analysis.head(24)),
            "",
            "### Reporting Guideline Checklist",
            _markdown_table(guideline_checklist[["item", "status", "location"]].head(20)),
            "",
            "## Results",
            results,
            "",
            "### Tables And Figures",
            (
                "The archive contains table_1_study_profile, table_2_variable_summary, "
                "table_3_primary_results, table_4_data_quality, "
                "table_5_analysis_interpretation, and PNG figures for metric summaries, "
                "data structure, and outcome distribution."
            ),
            "",
            "## Discussion",
            discussion,
            "",
            "## Limitations",
            *[f"- {item}" for item in limitations],
            "",
            "## Readiness Checklist",
            _markdown_table(readiness_checklist),
            "",
            "## Conclusions",
            conclusion,
            "",
            "## Next Steps",
            *[f"- {item}" for item in next_steps],
            "",
            "## Declarations",
            _markdown_table(declarations),
            "",
            "## References",
            *[
                f"- {row.citation} {row.url}"
                for row in references.itertuples(index=False)
            ],
            "",
            "## AI And Simulation Disclosure",
            ai_disclosure("library_only", version=__version__, doi="not-yet-assigned"),
            "",
            f"Generated by EPISIM {__version__}.",
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows generated._"
    display = frame.head(max_rows).copy()
    columns = [str(column) for column in display.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in display.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in display.columns]
        rows.append("| " + " | ".join(values) + " |")
    suffix = "" if len(frame) <= max_rows else f"\n\n_Showing {max_rows} of {len(frame)} rows._"
    return "\n".join([header, separator, *rows]) + suffix


def supported_research_designs() -> tuple[str, ...]:
    """Return design keys available to the end-to-end research workflow."""
    return tuple(spec.key for spec in list_designs())
