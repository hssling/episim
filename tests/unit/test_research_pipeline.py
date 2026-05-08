from __future__ import annotations

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
    assert not bundle.instruments.empty
    assert not bundle.follow_up_schedule.empty
    assert not bundle.outcome_record.empty
    assert not bundle.observations.empty
    assert "## Methods" in bundle.report.markdown
    assert "## Results" in bundle.report.markdown
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
        "data_collection_tools.csv",
        "follow_up_schedule.csv",
        "outcome_record.csv",
        "observations.csv",
    }
    assert expected.issubset({path.name for path in out.iterdir()})
    assert (out / "study" / "manifest.json").exists()
    assert bundle.plan.design_key == "qualitative_mixed_methods"


def test_supported_research_designs_matches_registry() -> None:
    designs = supported_research_designs()
    assert "cross_sectional" in designs
    assert "qualitative_mixed_methods" in designs
    assert len(designs) == 18
