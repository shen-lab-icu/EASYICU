"""The final plan boundary retains attached methods without changing its schema."""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.method_skills import METHOD_SKILLS
from easyicu.research_agent.orchestration.services import PipelineServices
from easyicu.extensions.method_attachments import builtin_attachment, method_constraints
from easyicu.webserver.skill_setup import apply_skill_setup, skill_planning_constraints
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, PlannedModelRequirement


REF = builtin_attachment("adjusted-exposure-outcome-study")


def _exact_plan(method_family):
    requirement = PlannedModelRequirement(
        requirement_id="synthetic-model", analysis_role="primary", analysis_set="complete_case",
        required_for_step_success=True, exposure_source="exposure", outcome="outcome",
        outcome_type="binary", method_family=method_family, covariates=["age"],
        model_terms=[
            {"name": "exposure", "role": "exposure", "coding": "continuous", "transform": "identity"},
            {"name": "age", "role": "covariate", "coding": "continuous", "transform": "identity"},
        ],
    )
    return AnalysisPlan(
        research_question="Is synthetic exposure associated with the outcome?", analysis_type="association_study",
        steps=[AnalysisStep(
            step_id="primary", intent="Estimate the synthetic association.", method="adjusted_association_models",
            inputs=["table:analysis_cohort"], expected_outputs=["table:adjusted_association_estimates"],
            planned_analysis_role="primary", model_requirements=[requirement],
        )],
    )


def _context():
    study = {"question": "Is synthetic exposure associated with the outcome?", "attached_skills": [REF]}
    study.update(apply_skill_setup(study))
    return SimpleNamespace(user_preferences=SimpleNamespace(data_constraints=json.dumps(skill_planning_constraints(study))))


def _plan():
    plan = _exact_plan("statsmodels_logit_mle")
    plan.steps[0].scientific_action_id = "association.adjusted_association"
    return plan


def test_the_public_pipeline_service_accepts_the_pinned_capability_and_action():
    PipelineServices().validate_plan_revision(plan=_plan(), context=_context())


def test_attachment_admission_runs_after_host_plan_compilation():
    services = PipelineServices()
    plan = _exact_plan("statsmodels_glm_binomial")
    services.validate_plan_revision(
        plan=plan, context=_context(),
        before_runtime_binding=SimpleNamespace(current_case=None, trajectory=None),
    )
    with pytest.raises(ValueError, match="skill_attachment_plan_capability_mismatch"):
        services.validate_plan_revision(plan=plan, context=_context())


@pytest.mark.parametrize("valid", [True, False])
def test_an_analysis_module_also_requires_a_coherent_primary_owner(valid):
    study = {"question": "Is synthetic exposure associated with the outcome?", "attached_skills": [builtin_attachment("adjusted-association-model")]}
    study.update(apply_skill_setup(study))
    context = SimpleNamespace(user_preferences=SimpleNamespace(data_constraints=json.dumps(skill_planning_constraints(study))))
    plan = _plan() if valid else _exact_plan("statsmodels_glm_binomial")
    plan.steps[0].scientific_action_id = "association.adjusted_association"
    plan.steps[0].scientific_capability = "association_adjusted_v1"
    if valid:
        PipelineServices().validate_plan_revision(plan=plan, context=context)
    else:
        with pytest.raises(ValueError, match="skill_attachment_plan_capability_mismatch"):
            PipelineServices().validate_plan_revision(plan=plan, context=context)


@pytest.mark.parametrize("change,code", [
    ("family", "skill_attachment_plan_family_mismatch"),
    ("capability", "skill_attachment_plan_capability_mismatch"),
    ("action", "skill_attachment_plan_action_mismatch"),
    ("constraints", "skill_attachment_constraint_mismatch"),
])
def test_the_pipeline_refuses_a_plan_that_drops_a_pin(change, code):
    plan, context = _plan(), _context()
    if change == "family":
        plan.analysis_type = "descriptive_epidemiology"
    elif change == "capability":
        plan = _exact_plan("statsmodels_glm_binomial")
        plan.steps[0].scientific_action_id = "association.adjusted_association"
        plan.steps[0].scientific_capability = "association_adjusted_v1"
    elif change == "action":
        plan.steps[0].scientific_action_id = None
    else:
        payload = json.loads(context.user_preferences.data_constraints)
        payload["method_skill_constraints"] = []
        context.user_preferences.data_constraints = json.dumps(payload)
    with pytest.raises(ValueError, match=code):
        PipelineServices().validate_plan_revision(plan=plan, context=context)


STUDY_METHODS = [spec for spec in METHOD_SKILLS if spec.layer in {"research_workflow", "analysis_module"}]
TOOL_SKILLS = [spec for spec in METHOD_SKILLS if spec not in STUDY_METHODS]


@pytest.mark.parametrize("spec", STUDY_METHODS, ids=lambda spec: spec.skill_id)
def test_every_catalog_method_has_a_canonical_design_coordinate(spec):
    from easyicu.research_agent.planning.analysis_types import canonical_analysis_family

    refs = [builtin_attachment(spec.skill_id)]
    constraints = method_constraints(refs)
    setup = apply_skill_setup({"attached_skills": refs, "question": "Synthetic research question."})
    family = constraints[0]["analysis_family"]
    assert canonical_analysis_family(family) == family
    assert setup["analysis_design"]["analysis_family"] == family
    assert constraints[0]["capability_id"] == spec.capability_id
    assert constraints[0]["action_ids"] == list(spec.action_ids)


@pytest.mark.parametrize("spec", TOOL_SKILLS, ids=lambda spec: spec.skill_id)
def test_a_data_or_planning_tool_is_not_a_study_method_to_attach(spec):
    from easyicu.extensions.contracts import ExtensionRegistryError

    with pytest.raises(ExtensionRegistryError, match="skill_attachment_disabled"):
        method_constraints([builtin_attachment(spec.skill_id)])


def test_a_researcher_choice_can_override_the_attachment_but_keeps_its_receipt():
    from easyicu.webserver.question_family_check import researcher_choice

    study = {"question": "Describe the prevalence of a synthetic outcome.", "attached_skills": [REF]}
    study.update(apply_skill_setup(study))
    study["question_family_check"] = researcher_choice(
        study["question_family_check"], study["question"],
        "Treat it as a descriptive study (proportions, rates or distributions)",
    )
    study.update(apply_skill_setup(study))
    payload = skill_planning_constraints(study)
    context = SimpleNamespace(user_preferences=SimpleNamespace(data_constraints=json.dumps(payload)))
    PipelineServices().validate_plan_revision(plan=SimpleNamespace(analysis_type="descriptive_epidemiology", steps=[]), context=context)
    assert payload["attached_skills"] == [REF]


@pytest.mark.parametrize("raw", [None, "legacy text", "[]", "{}"])
def test_legacy_runs_have_no_attachment_requirement(raw):
    PipelineServices().validate_plan_revision(plan=None, context=SimpleNamespace(user_preferences=SimpleNamespace(data_constraints=raw)))
