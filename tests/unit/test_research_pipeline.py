from __future__ import annotations

import sqlite3

from episim.research import conduct_research, plan_research, supported_research_designs


def test_plan_research_infers_cohort_from_follow_up_question() -> None:
    plan = plan_research(
        "Does physical activity reduce cognitive decline during follow-up in older adults?",
        seed_value=123,
        n=300,
    )
    assert plan.design_key == "cohort"
    assert plan.parameters["seed_value"] == 123
    assert plan.parameters["n"] == 300
    assert "follow" in " ".join(plan.analysis_plan).lower()
    assert plan.hypotheses


def test_conduct_research_returns_complete_research_bundle() -> None:
    bundle = conduct_research(
        "Does a randomized lifestyle intervention reduce frailty in community elders?",
        seed_value=321,
        n=500,
    )
    assert bundle.plan.design_key == "rct_parallel"
    assert bundle.study.design == "rct_parallel"
    assert len(bundle.study.data) == 500
    assert len(bundle.collected_data) == 500
    assert len(bundle.cleaned_data) == 500
    assert not bundle.cleaning_log.empty
    assert not bundle.analysis_steps_record.empty
    assert not bundle.instruments.empty
    assert not bundle.database_dictionary.empty
    assert not bundle.collection_events.empty
    assert not bundle.follow_up_schedule.empty
    assert not bundle.outcome_record.empty
    assert not bundle.observations.empty
    assert "table_1_study_profile" in bundle.analysis_tables
    assert "table_4_data_quality" in bundle.analysis_tables
    assert "table_6_realism_audit" in bundle.analysis_tables
    assert "table_7_sensitivity_analysis" in bundle.analysis_tables
    assert "table_8_readiness_checklist" in bundle.analysis_tables
    assert not bundle.figure_plan.empty
    assert not bundle.realism_audit.empty
    assert not bundle.sensitivity_analysis.empty
    assert not bundle.readiness_checklist.empty
    assert not bundle.guideline_checklist.empty
    assert not bundle.references.empty
    assert not bundle.declarations.empty
    assert "## Methods" in bundle.report.markdown
    assert "## Results" in bundle.report.markdown
    assert "## References" in bundle.report.markdown
    assert "## Declarations" in bundle.report.markdown
    assert "synthetic" in bundle.report.abstract.lower()


def test_research_bundle_archive_writes_complete_outputs(tmp_path) -> None:
    bundle = conduct_research(
        "What themes explain patient trust in AI triage tools?",
        seed_value=456,
        n_interviews=15,
        n_survey=80,
    )
    out = bundle.archive(tmp_path / "research")
    expected = {
        "protocol.json",
        "protocol.md",
        "report.md",
        "collected_synthetic_data.csv",
        "cleaned_analysis_dataset.csv",
        "data_cleaning_log.csv",
        "analysis_steps.csv",
        "data_collection_tools.csv",
        "database_dictionary.csv",
        "collection_events.csv",
        "follow_up_schedule.csv",
        "outcome_record.csv",
        "observations.csv",
        "realism_audit.csv",
        "sensitivity_analysis.csv",
        "readiness_checklist.csv",
        "guideline_checklist.csv",
        "references.csv",
        "declarations.csv",
        "synthetic_research_database.sqlite",
    }
    assert expected.issubset({path.name for path in out.iterdir()})
    assert (out / "study" / "manifest.json").exists()
    assert (out / "tables" / "table_2_variable_summary.csv").exists()
    assert (out / "figures" / "figure_1_metric_summary.png").exists()
    with sqlite3.connect(out / "synthetic_research_database.sqlite") as con:
        tables = {
            row[0]
            for row in con.execute(
                "select name from sqlite_master where type='table'"
            ).fetchall()
        }
    assert "synthetic_observations" in tables
    assert "collected_synthetic_data" in tables
    assert "cleaned_analysis_dataset" in tables
    assert "data_cleaning_log" in tables
    assert "analysis_steps" in tables
    assert "realism_audit" in tables
    assert "sensitivity_analysis" in tables
    assert "readiness_checklist" in tables
    assert "collection_events" in tables
    assert bundle.plan.design_key == "qualitative_mixed_methods"


def test_supported_research_designs_matches_registry() -> None:
    designs = supported_research_designs()
    assert "cross_sectional" in designs
    assert "qualitative_mixed_methods" in designs
    assert len(designs) == 18
