"""Tests for episim.designs.stepped_wedge."""
from episim.designs import stepped_wedge


def test_stepped_wedge_recovers_lower_treated_rate() -> None:
    study = stepped_wedge.run(
        seed_value=20260508,
        n_clusters=10,
        n_periods=5,
        persons_per_cluster_period=40,
    )
    assert study.design == "stepped_wedge"
    assert study.results["treated_event_rate"] < study.results["control_event_rate"]
    assert study.results["marginal_risk_difference"] < 0.0
    assert "cluster_period_summary" in study.artifacts


def test_stepped_wedge_is_deterministic() -> None:
    a = stepped_wedge.run(seed_value=44, n_clusters=8, n_periods=5)
    b = stepped_wedge.run(seed_value=44, n_clusters=8, n_periods=5)
    assert a.results == b.results
