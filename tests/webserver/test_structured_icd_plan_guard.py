"""Synthetic host-bound ICD settings cannot become unverified plan approval."""
import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.orchestration.scientific_plan_review_gate import persist_or_validate_scientific_plan_review
from easyicu.research_agent.orchestration.workflow import HumanReviewPending, human_review_requests_for_plan
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ResearchContext, UserPreferences
from easyicu.webserver.agent_pipeline_runs import (
    _load_pending_scientific_review, _pending_plan_approval_allowed, _pending_review_reason_code,
    _pending_icd_source_required, _research_user_preferences,
)


def _study(grouped=True):
    context = ResearchContext.model_validate({
        "research_question": "Summarize observed signal measurements across ward categories.",
        "cohort": {"cohort_name": "synthetic", "database": "synthetic", "n_stays": 8,
                   "id_columns": ["stay_id"], "outcome_columns": ["signal_max"],
                   "requested_outcome_columns": ["signal_max"]},
        "primary_exposure": None,
        "target_outcome": "signal_max",
        "endpoint": {"name": "signal_max", "kind": "continuous", "absence_semantics": "absent_row_is_unmeasured"},
        "variables": [
            {"name": "stay_id", "role": "id", "dtype": "int64"},
            {"name": "admission_category", "role": "other", "dtype": "str",
             "observed_domain": {"levels": ["A", "B", "C"], "n_unique": 3}},
            {"name": "signal_max", "role": "outcome", "dtype": "float64",
             "description": "Maximum observed signal during the first day."},
        ] + [{"name": name, "role": "meta", "dtype": "int64"}
             for name in ("signal_measured", "signal_n")],
    })
    plan = AnalysisPlan(
        research_question=context.research_question, analysis_type="descriptive_epidemiology",
        endpoint=context.endpoint, steps=[
            AnalysisStep(
                step_id="define_cohort", planned_analysis_role="auxiliary",
                intent="Account for every synthetic stay.", method="cohort_definition_and_attrition",
                inputs=["stay_id", "signal_max"],
                expected_outputs=["artifact:analysis_cohort", "table:cohort_flow"],
                cohort_definition_spec={"identity_column": "stay_id", "analysis_set": "bound_universe"},
            ),
            AnalysisStep(
                step_id="grouped_primary", planned_analysis_role="primary",
                intent="Describe the measurement distribution by category.", method="empirical_distribution_summary",
                scientific_action_id="descriptive.descriptive_summary",
                inputs=["stay_id", "admission_category", "signal_max", "artifact:analysis_cohort"],
                expected_outputs=["table:cohort_summary", "table:distribution_summary"],
            ),
            AnalysisStep(
                step_id="continuous_secondary", planned_analysis_role="secondary",
                intent="Summarize the continuous measurement within categories.", method="continuous_outcome_summary",
                inputs=["artifact:analysis_cohort", "admission_category", "signal_max"],
                expected_outputs=["table:continuous_outcome_summary"],
            ),
            AnalysisStep(
                step_id="availability", planned_analysis_role="auxiliary",
                intent="Audit measurement completeness.", method="missing_data",
                inputs=["stay_id", "admission_category", "signal_max", "artifact:analysis_cohort"],
                expected_outputs=["table:measurement_audit", "table:measurement_process_audit"],
                measurement_audit_spec={"products": [
                    {"product_id": "measurement_audit", "audit": "measurement_missingness"},
                    {"product_id": "measurement_process_audit", "audit": "measurement_process"},
                ]},
            ),
        ],
    )
    if not grouped:
        plan = plan.model_copy(update={"steps": [
            step.model_copy(update={"inputs": [v for v in step.inputs if v != "admission_category"]})
            for step in plan.steps if step.step_id != "continuous_secondary"
        ]})
    return context, plan



def _register_review_context(directory, evidence, context):
    path = directory / "research_context.json"
    path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(kind="log", description="Synthetic reviewed context", source_path=path,
                           evidence_id="research_context", producer="pipeline", generation_mode="system")




@pytest.mark.parametrize("cohort", [
    {"preset": "icd", "icd_enabled": True, "icd_include": ["K92.2"]},
    {"preset": "adult_all", "icd_enabled": True, "icd_exclude": ["K92.2"]},
    {"preset": "adult_all", "icd_include": ["K92.2"]},
])
def test_active_icd_is_blocked_through_actual_host_producer_and_review(tmp_path, cohort):
    context, plan = _study()
    preferences = UserPreferences.model_validate(_research_user_preferences({"cohort": cohort}))
    context = context.model_copy(update={"user_preferences": preferences})
    review = build_plan_scientific_review(context=context, plan=plan)
    assert not review.approval_allowed
    assert any(f.code == "HOST_ICD_SOURCE_UNVERIFIED" and f.severity == "blocker" for f in review.findings)
    evidence = EvidenceStore(tmp_path)
    _register_review_context(tmp_path, evidence, context)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=review)
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="synthetic", thread_id="synthetic", run_dir=str(tmp_path), requests=requests)
    assert _load_pending_scientific_review(tmp_path, pending)["approval_allowed"] is False
    assert not _pending_plan_approval_allowed(run_dir=tmp_path, pending=pending, plan_recommendation_complete=True)


def test_explicitly_disabled_icd_does_not_block():
    context, plan = _study()
    preferences = UserPreferences.model_validate(_research_user_preferences({"cohort": {
        "preset": "adult_all", "icd_enabled": False, "icd_include": ["K92.2"]
    }}))
    assert "host_icd_source_proof_required" not in json.loads(preferences.data_constraints)
    context = context.model_copy(update={"user_preferences": preferences})
    assert build_plan_scientific_review(context=context, plan=plan).approval_allowed


def test_old_review_without_marker_cannot_approve_bound_icd(tmp_path):
    context, plan = _study()
    prefs = UserPreferences(data_constraints=json.dumps({"cohort": {
        "preset": "icd", "icd_enabled": True, "icd_include": ["K92.2"]
    }}))
    context = context.model_copy(update={"user_preferences": prefs})
    review = build_plan_scientific_review(context=context, plan=plan)
    assert review.approval_allowed
    evidence = EvidenceStore(tmp_path)
    _register_review_context(tmp_path, evidence, context)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=review)
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="synthetic", thread_id="synthetic", run_dir=str(tmp_path), requests=requests)
    before = (tmp_path / "scientific_plan_review.json").read_bytes()
    assert _load_pending_scientific_review(tmp_path, pending) == {}
    assert _pending_icd_source_required(tmp_path, pending)
    assert _pending_review_reason_code(
        request=requests[0], plan_recommendation_complete=True, scientific_plan_review={},
        host_icd_source_required=True,
    ) == "plan_icd_source_unverified"
    assert not _pending_plan_approval_allowed(run_dir=tmp_path, pending=pending, plan_recommendation_complete=True)
    assert (tmp_path / "scientific_plan_review.json").read_bytes() == before


@pytest.mark.parametrize("question", [
    "Describe outcomes after implantable cardioverter defibrillator (ICD) placement.",
    "Do not apply an ICD diagnosis filter.",
    "Describe patients with a T24.0 burn code.",
])
def test_free_text_is_not_claimed_as_a_host_structured_choice(question):
    context, plan = _study()
    context = context.model_copy(update={"research_question": question})
    plan = plan.model_copy(update={"research_question": question})
    review = build_plan_scientific_review(context=context, plan=plan)
    assert not any(f.code == "HOST_ICD_SOURCE_UNVERIFIED" for f in review.findings)


def test_old_host_policy_reason_is_visible_even_with_other_plan_stops():
    from types import SimpleNamespace

    request = SimpleNamespace(payload={"reason": "agent_plan_revision_nonconvergent"})
    assert _pending_review_reason_code(
        request=request, plan_recommendation_complete=False, scientific_plan_review={},
        host_icd_source_required=True,
    ) == "plan_icd_source_unverified"
    assert _pending_review_reason_code(
        request=request, plan_recommendation_complete=False, scientific_plan_review={},
    ) == "agent_plan_revision_nonconvergent"


@pytest.mark.parametrize("cohort", [
    {"preset": "icd"},
    {"preset": "legacy_unrecognized", "icd_enabled": True, "icd_include": "K92.2"},
    {"preset": "legacy_unrecognized", "icd_enabled": "invalid"},
])
def test_invalid_or_incomplete_legacy_cohort_cannot_erase_icd_proof_requirement(cohort):
    from easyicu.webserver import primary_cohort

    assert primary_cohort.icd_source_proof_required(cohort) is True
    with pytest.raises(primary_cohort.PrimaryCohortContractError):
        primary_cohort.normalize_execution_cohort(cohort)


def test_disabled_legacy_icd_axis_does_not_reactivate_residual_codes():
    from easyicu.webserver import primary_cohort

    assert not primary_cohort.icd_source_proof_required({
        "preset": "legacy_unrecognized", "icd_enabled": False, "icd_include": "K92.2",
    })
