"""Focused analysis-family normalization contracts."""

def test_normalise_contract_family_bridges_registry_keys() -> None:
    """The alias layer must map registry analysis_type keys onto contract
    buckets so the stamped plan.analysis_type drives figure enforcement."""
    from easyicu.research_agent.contracts import step_families

    assert step_families._normalise_contract_family("survival") == "survival"
    assert (
        step_families._normalise_contract_family("trajectory_clustering") == "clustering"
    )
    # Result-bearing families that own their bucket pass through identically.
    for key in (
        "dynamic_prediction",
        "causal_inference",
        "treatment_response",
        "validation",
    ):
        assert step_families._normalise_contract_family(key) == key
    # Families without a figure/metric contract fall back to the heuristic.
    assert step_families._normalise_contract_family("association_study") == ""
    assert step_families._normalise_contract_family("descriptive_epidemiology") == ""
    assert step_families._normalise_contract_family("multimodal") == ""
    assert step_families._normalise_contract_family(None) == ""
    # Legacy contract-bucket words still pass through unchanged.
    assert step_families._normalise_contract_family("clustering") == "clustering"
