"""Question-to-report orchestration for complete simulated studies."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    instruments: pd.DataFrame
    follow_up_schedule: pd.DataFrame
    outcome_record: pd.DataFrame
    observations: pd.DataFrame
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
        self.instruments.to_csv(out / "data_collection_tools.csv", index=False)
        self.follow_up_schedule.to_csv(out / "follow_up_schedule.csv", index=False)
        self.outcome_record.to_csv(out / "outcome_record.csv", index=False)
        self.observations.to_csv(out / "observations.csv", index=False)
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
    instruments = _make_instruments(plan, study)
    follow_up = _make_follow_up_schedule(plan.design_key)
    outcome_record = _make_outcome_record(plan, study)
    observations = _make_observations(study)
    report = _make_report(plan, study, instruments, follow_up, outcome_record)
    return ResearchBundle(
        plan=plan,
        study=study,
        instruments=instruments,
        follow_up_schedule=follow_up,
        outcome_record=outcome_record,
        observations=observations,
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


def _make_report(
    plan: ResearchPlan,
    study: Study,
    instruments: pd.DataFrame,
    follow_up: pd.DataFrame,
    outcome_record: pd.DataFrame,
) -> ResearchReport:
    result_lines = _result_lines(study)
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
    abstract = (
        f"Background: {plan.question} Methods: EPISIM generated a synthetic "
        f"{plan.design_key} study using seed {study.seed}. Results: {len(study.data)} "
        f"records were produced and {len(study.results)} summary metrics were estimated. "
        "Conclusions: The workflow provides simulated evidence for design development, "
        "not a claim about real-world effectiveness."
    )
    methods = (
        f"{plan.methodology}\n\nPopulation: {plan.population}\n\n"
        f"Data collection tools included {len(instruments)} variable-level fields. "
        f"Follow-up schedule contained {len(follow_up)} planned timepoints. "
        f"Analysis used: {'; '.join(plan.analysis_plan)}"
    )
    results = (
        f"The simulation generated {len(study.data)} records. "
        f"Outcome recording produced {len(outcome_record)} metrics. "
        + " ".join(result_lines)
    )
    discussion = (
        "The simulated study demonstrates whether the proposed design can collect the "
        "required measurements, maintain a transparent follow-up structure, and return "
        "design-appropriate estimates. Findings should be interpreted as scenario-based "
        "methodological evidence rather than empirical clinical or social proof."
    )
    conclusion = (
        f"EPISIM completed an end-to-end synthetic {plan.design_key} study for the "
        "research question, including protocol, tools, observations, outcome recording, "
        "analysis, and manuscript-style interpretation."
    )
    markdown = _report_markdown(
        plan=plan,
        abstract=abstract,
        methods=methods,
        results=results,
        discussion=discussion,
        conclusion=conclusion,
        limitations=limitations,
        next_steps=next_steps,
    )
    return ResearchReport(
        title=plan.title,
        abstract=abstract,
        introduction=(
            f"The research question was: {plan.question} The simulation was designed "
            "to test study feasibility, measurement logic, and expected analytical outputs."
        ),
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
    abstract: str,
    methods: str,
    results: str,
    discussion: str,
    conclusion: str,
    limitations: tuple[str, ...],
    next_steps: tuple[str, ...],
) -> str:
    return "\n".join(
        [
            f"# {plan.title}",
            "",
            "## Abstract",
            abstract,
            "",
            "## Introduction",
            (
                f"This synthetic study addresses: {plan.question} It is intended for "
                "methods development, teaching, protocol refinement, and feasibility testing."
            ),
            "",
            "## Methods",
            methods,
            "",
            "## Results",
            results,
            "",
            "## Discussion",
            discussion,
            "",
            "## Limitations",
            *[f"- {item}" for item in limitations],
            "",
            "## Conclusions",
            conclusion,
            "",
            "## Next Steps",
            *[f"- {item}" for item in next_steps],
            "",
            "## AI And Simulation Disclosure",
            ai_disclosure("library_only", version=__version__, doi="not-yet-assigned"),
            "",
            f"Generated by EPISIM {__version__}.",
            "",
        ]
    )


def supported_research_designs() -> tuple[str, ...]:
    """Return design keys available to the end-to-end research workflow."""
    return tuple(spec.key for spec in list_designs())
