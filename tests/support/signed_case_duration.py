"""Synthetic contract fixture; never a clinical plan or real data grant."""
import json

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_substitutions import bind_source_substitutions
from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN, SourceTimeAxisBinding
from easyicu.research_agent.authority.case_duration_landmark import propose_case_duration_landmark_authority
from easyicu.research_agent.planning.cohort_contract import CohortDefinition
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ConceptDescriptor, TimeWindow, UserPreferences
from easyicu.webserver.time_origin_source_authority import CaseDurationSourceProvenance, RegisteredExportTimeOrigin
from tests.research_agent.authority.test_case_duration_landmark_authority import _inputs
from tests.support.source_substitutions import context


def signed_case_duration_case():
    origin, outcome, axis, primary = _inputs()
    source_proof = CaseDurationSourceProvenance("d" * 64, "e" * 64, "f" * 64)
    authority = propose_case_duration_landmark_authority(
        origin=origin, outcome=outcome, source_axis=axis, source_provenance=source_proof, primary_step=primary,
    )
    primary = primary.model_copy(update={"icu_rule_refs": [authority.plan_rule_ref],
                                         "inputs": ["artifact:analysis_cohort", "lact", outcome.proposed_outcome]})
    risk = AnalysisStep(step_id="landmark", intent="Apply the explicitly reviewed case landmark",
        method="signed_care_unit_landmark_cohort", inputs=[outcome.proposed_outcome, "death_during_case",
            "death_time_during_case_hours", "lact"], expected_outputs=["artifact:analysis_cohort"],
        icu_rule_refs=[authority.plan_rule_ref])
    plan = AnalysisPlan(research_question="Synthetic case-duration association", analysis_type="association_study",
                        cohort=CohortDefinition(name="synthetic", selection_mode="all_input_rows"), steps=[risk, primary])
    authority.validate_plan(plan)
    ctx = context("sic", outcome="icu_death")
    constraints = json.loads(ctx.user_preferences.data_constraints)
    identity = {"database": "sic", "registration_id": "src_example", "version": "1.0.6"}
    from easyicu.research_agent.canonical_json import canonical_sha256
    # Match the typed origin to this exact synthetic registry identity.
    identity_sha = canonical_sha256(identity)
    origin = origin.model_copy(update={"source_identity_sha256": identity_sha})
    outcome = outcome.model_copy(update={"source_identity_sha256": identity_sha})
    axis = SourceTimeAxisBinding(identity_sha, "b" * 64, "c" * 64, SIC_CARE_UNIT_ORIGIN)
    authority = propose_case_duration_landmark_authority(
        origin=origin, outcome=outcome, source_axis=axis, source_provenance=source_proof, primary_step=primary,
    )
    # The contract digest changes with the registration; bind both steps again.
    primary = primary.model_copy(update={"icu_rule_refs": [authority.plan_rule_ref]})
    risk = risk.model_copy(update={"icu_rule_refs": [authority.plan_rule_ref]})
    plan = plan.model_copy(update={"steps": [risk, primary]})
    authority.validate_plan(plan)
    constraints.update(source_identity=identity, database_time_origin_selection=origin.model_dump(mode="json"),
        database_case_duration_selection=outcome.model_dump(mode="json"))
    constraints["database_study_requirements"]["outcome"] = "los_icu"
    variables = [v.model_copy(update={"analysis_window": "first_day", "analysis_window_role": "exposure_definition"})
                 if v.name == "lact" else v for v in ctx.variables]
    variables.append(ConceptDescriptor(name=outcome.proposed_outcome, dtype="float64", role="outcome"))
    ctx = ctx.model_copy(update={"target_outcome": outcome.proposed_outcome, "variables": variables,
        "time_windows": [TimeWindow(name="first_day", anchor=SIC_CARE_UNIT_ORIGIN, start_hours=0, end_hours=24)],
        "user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    ctx = bind_source_substitutions(ctx)
    receipt = RegisteredExportTimeOrigin(identity_sha, "b" * 64, "c" * 64, SIC_CARE_UNIT_ORIGIN)

    def origin_attestor(_context, _selection):
        return origin, receipt

    def case_attestor(_context, _selection):
        return receipt, source_proof

    return ctx, plan, authority, receipt, source_proof, origin_attestor, case_attestor
