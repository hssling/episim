"""Tests for episim.designs.rct_cluster."""
from episim.designs import rct_cluster


def test_rct_cluster_recovers_protective_effect() -> None:
    study = rct_cluster.run(seed_value=20260508, n_clusters=24, cluster_size=60)
    assert study.design == "rct_cluster"
    assert study.results["risk_ratio"] < 1.0
    assert study.results["treated_cluster_event_rate"] < study.results["control_cluster_event_rate"]
    assert "cluster_summary" in study.artifacts


def test_rct_cluster_is_deterministic() -> None:
    a = rct_cluster.run(seed_value=33, n_clusters=20, cluster_size=50)
    b = rct_cluster.run(seed_value=33, n_clusters=20, cluster_size=50)
    assert a.results == b.results
