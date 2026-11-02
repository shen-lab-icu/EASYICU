"""Synthetic source-bound case-duration landmark inputs."""
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import CaseDurationSelection, TimeOriginSelection
from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN, SourceTimeAxisBinding
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
