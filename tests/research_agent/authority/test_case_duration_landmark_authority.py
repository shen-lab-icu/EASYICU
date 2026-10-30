"""Clinical proxy and OLS must remain bound to the selected source."""
import pytest
import pandas as pd

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import CaseDurationSelection, TimeOriginSelection
from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN, SourceTimeAxisBinding
from easyicu.research_agent.authority.case_duration_landmark import (
    CaseDurationLandmarkAuthority, apply_case_duration_landmark, propose_case_duration_landmark_authority,
)
from easyicu.research_agent.schema import AnalysisStep
from easyicu.webserver.time_origin_source_authority import CaseDurationSourceProvenance

_PROVENANCE = CaseDurationSourceProvenance("d" * 64, "e" * 64, "f" * 64)


def _inputs():
    profile = research_capability_profile("sic")
    origin = TimeOriginSelection(requested_origin="icu_admission", selected_origin=SIC_CARE_UNIT_ORIGIN,
                                 source_field=profile.care_unit_admission_proxy, source_identity_sha256="a" * 64)
    outcome = CaseDurationSelection(requested_outcome="los_icu", proposed_outcome="care_unit_to_case_closure_days",
                                    start_field=profile.care_unit_admission_proxy, end_field=profile.case_close_proxy,
                                    source_identity_sha256="a" * 64)
    axis = SourceTimeAxisBinding("a" * 64, "b" * 64, "c" * 64, SIC_CARE_UNIT_ORIGIN)
    step = AnalysisStep(step_id="primary", intent="Estimate the association", method="adjusted_association_models",
                        planned_analysis_role="primary", expected_outputs=["table:adjusted_association_estimates"],
                        model_requirements=[{"requirement_id": "primary", "outcome": "care_unit_to_case_closure_days",
                            "outcome_type": "continuous", "method_family": "statsmodels_ols", "exposure_source": "lact",
                            "analysis_role": "primary", "analysis_set": "complete_case",
                            "covariates": [], "model_terms": [{"name": "lact", "role": "exposure",
                                "coding": "continuous", "transform": "identity"}]}])
    return origin, outcome, axis, step


def test_candidate_binds_proxy_source_24h_rule_and_deterministic_ols():
    origin, outcome, axis, step = _inputs()
    candidate = propose_case_duration_landmark_authority(origin=origin, outcome=outcome, source_axis=axis, source_provenance=_PROVENANCE, primary_step=step)
    assert candidate.landmark_hours == 24
    assert candidate.risk_policy == "case_open_and_known_alive_strictly_after_landmark"
    assert candidate.requested_outcome == "los_icu"
    assert candidate.proposed_outcome == "care_unit_to_case_closure_days"
    assert CaseDurationLandmarkAuthority.model_validate(candidate.model_dump(mode="json")) == candidate


def test_candidate_refuses_different_source_or_non_ols_and_digest_tampering():
    origin, outcome, axis, step = _inputs()
    candidate = propose_case_duration_landmark_authority(origin=origin, outcome=outcome, source_axis=axis, source_provenance=_PROVENANCE, primary_step=step)
    with pytest.raises(ValueError):
        CaseDurationLandmarkAuthority.model_validate({**candidate.model_dump(mode="json"), "landmark_hours": 12})
    with pytest.raises(ValueError, match="source_mismatch"):
        propose_case_duration_landmark_authority(origin=origin, outcome=outcome.model_copy(update={"source_identity_sha256": "d" * 64}),
                                                 source_axis=axis, source_provenance=_PROVENANCE, primary_step=step)
    requirement = step.model_requirements[0].model_copy(update={"method_family": "quantile_regression"})
    with pytest.raises(ValueError, match="executor_unavailable"):
        propose_case_duration_landmark_authority(origin=origin, outcome=outcome, source_axis=axis, source_provenance=_PROVENANCE,
                                                 primary_step=step.model_copy(update={"model_requirements": [requirement]}))


def test_landmark_mechanics_exclude_unknown_and_preserve_source_bound_rows():
    origin, outcome, axis, step = _inputs()
    candidate = propose_case_duration_landmark_authority(origin=origin, outcome=outcome, source_axis=axis, source_provenance=_PROVENANCE, primary_step=step)
    frame = pd.DataFrame({"lact": [1.0, 2.0, 3.0, 4.0],
        "care_unit_to_case_closure_days": [2.0, 1.0, 3.0, 2.0],
        "death_during_case": [False, False, True, None], "death_time_during_case_hours": [None, None, 30.0, None]})
    filtered, counts = apply_case_duration_landmark(frame, authority=candidate, source_axis=axis, primary_step=step)
    assert filtered.lact.tolist() == [1.0, 3.0]
    assert counts == {"input_rows": 4, "eligible_rows": 2, "unidentifiable_rows": 1,
                      "closed_or_dead_at_landmark_rows": 1}
    other = SourceTimeAxisBinding("a" * 64, "b" * 64, "d" * 64, SIC_CARE_UNIT_ORIGIN)
    with pytest.raises(ValueError, match="source_mismatch"):
        apply_case_duration_landmark(frame, authority=candidate, source_axis=other, primary_step=step)
