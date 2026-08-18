"""Planner reachability for the ICU scientific method and resource surface."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent import schema
from easyicu.research_agent.agents import core
from easyicu.research_agent.agents.core import PlannerAgent, ReplannerAgent
from easyicu.research_agent.authority.plan_scope import _step_scientific_signature
from easyicu.research_agent.planning.analysis_types import CATALOG_DETAIL_LADDER
from easyicu.research_agent.planning.scientific_action_catalog import (
    ScientificActionGapError,
    planner_scientific_action_guide,
    resolve_scientific_action_request,
    scientific_actions_for_analysis_type,
    validate_plan_scientific_action_selections,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.resources.coder import build_coder_resource_bundle


def _action(analysis_type: str, method_key: str):
    catalog = scientific_actions_for_analysis_type(analysis_type)
    return next(action for action in catalog.actions if action.method_key == method_key)


def test_planner_action_guide_closes_family_and_auxiliary_boundaries():
    guide = planner_scientific_action_guide("association_study")

    assert "scientific_action_id is either null or exactly one of" in guide
    assert "association.adjusted_association" in guide
    assert "Cohort-definition, Table 1, raw distribution" in guide
    assert "import another family prefix" in guide


def test_prediction_methods_bind_reviewed_resources_without_upgrading_owner():
    delong = _action("prediction_model", "delong_ci")
    dca = _action("prediction_model", "decision_curve")
    conformal = _action("prediction_model", "conformal_intervals")
    shap = _action("prediction_model", "feature_attribution")

    assert delong.execution_mode == "coder_generated"
    assert delong.kernel_imports == (
        "easyicu.research_agent.methods.delong_auc",
    )
    assert dca.execution_mode == "coder_generated"
    assert dca.kernel_imports == (
        "easyicu.research_agent.methods.decision_curve",
    )
    assert conformal.execution_mode == "coder_generated"
    assert conformal.kernel_imports == (
        "easyicu.research_agent.methods.conformal",
    )
    assert shap.execution_mode == "coder_generated"
    assert shap.software_packages == ("shap",)


def test_dynamic_prediction_is_a_typed_coder_action_with_reviewed_primitives():
    catalog = scientific_actions_for_analysis_type("dynamic_prediction")
    dynamic = _action("dynamic_prediction", "dynamic_prediction")

    assert catalog.primary_contract_id == "dynamic_prediction_landmark_v1"
    assert catalog.primary_contract_registered is True
    assert catalog.required_primary_action_ids == (
        "prediction.dynamic_prediction",
    )
    assert dynamic.execution_mode == "coder_generated"
    assert dynamic.kernel_imports == (
        "easyicu.research_agent.methods.dynamic_prediction",
        "easyicu.research_agent.methods.temporal_features",
    )
    assert dynamic.software_packages == ("sklearn",)
    assert dynamic.required_inputs
    assert dynamic.composition_action_ids
    assert "kernel:dynamic_prediction" in {
        primitive.resource_id for primitive in catalog.reviewed_primitives
    }


def test_action_gap_resolution_is_typed_and_never_auto_substitutes():
    composed = resolve_scientific_action_request(
        analysis_type="dynamic_prediction",
        action_id="prediction.dynamic_prediction",
    )
    assert composed.status == "composed"
    assert composed.executable is True
    assert composed.selected_action_ids == (
        "prediction.dynamic_prediction",
        "prediction.discrimination_calibration",
        "prediction.calibration_metrics",
        "prediction.internal_validation",
    )

    direct = resolve_scientific_action_request(
        analysis_type="prediction_model",
        action_id="prediction.delong_ci",
    )
    assert direct.status == "direct"
    assert direct.executable is True

    alternative = resolve_scientific_action_request(
        analysis_type="prediction_model",
        action_id="prediction.external_validation",
    )
    assert alternative.status == "alternative"
    assert alternative.selected_action_ids == ()
    assert alternative.alternative_action_ids == ("prediction.internal_validation",)
    assert alternative.requires_user_confirmation is True
    assert alternative.issue_code == "scientific_action_requires_user_choice"

    unknown = resolve_scientific_action_request(
        analysis_type="dynamic_prediction",
        action_id="prediction.dynamic_forecast",
    )
    assert unknown.status == "unavailable"
    assert unknown.issue_code == "scientific_action_unregistered"
    assert "prediction.dynamic_prediction" in unknown.alternative_action_ids
    assert unknown.to_dict()["schema_version"] == (
        "easyicu.scientific_action_resolution/1"
    )

    with pytest.raises(ScientificActionGapError) as captured:
        from easyicu.research_agent.planning.scientific_action_catalog import (
            scientific_action_for_id,
        )

        scientific_action_for_id(
            analysis_type="prediction_model",
            action_id="prediction.external_validation",
        )
    assert captured.value.issue_code == "scientific_action_requires_user_choice"


def test_dynamic_plan_requires_the_exact_primary_action_before_execution():
    missing = schema.AnalysisPlan(
        research_question="Update mortality risk during the ICU stay.",
        analysis_type="dynamic_prediction",
        steps=[
            schema.AnalysisStep(
                step_id="01_dynamic",
                planned_analysis_role="primary",
                intent="Fit a landmark risk model.",
                method="dynamic landmark prediction",
                expected_outputs=[
                    "table:dynamic_prediction_metrics",
                    "prediction:time_varying_discrimination",
                ],
            )
        ],
    )
    with pytest.raises(ScientificActionGapError) as captured:
        validate_plan_scientific_action_selections(
            plan=missing,
            inferred_analysis_type="dynamic_prediction",
        )
    assert captured.value.issue_code == "scientific_action_declaration_required"

    bound = missing.model_copy(deep=True)
    bound.steps[0].scientific_action_id = "prediction.dynamic_prediction"
    validate_plan_scientific_action_selections(
        plan=bound,
        inferred_analysis_type="dynamic_prediction",
    )


def test_survival_rmst_is_reachable_but_competing_risks_remains_unavailable():
    rmst = _action("survival", "rmst")
    cif = _action("survival", "competing_risks_cif")

    assert rmst.execution_mode == "coder_generated"
    assert rmst.kernel_imports == ("easyicu.research_agent.methods.rmst",)
    assert cif.execution_mode == "not_available"
    assert cif.runner is None
    assert cif.kernel_imports == ()

    guide = planner_scientific_action_guide("survival")
    assert "competing_risks_cif" in guide
    assert "not_available" in guide
    assert "never substitute" in guide


def test_family_primitives_reach_planner_as_support_not_estimand_authority():
    survival = scientific_actions_for_analysis_type("survival")
    primitives = {item.resource_id: item for item in survival.reviewed_primitives}

    assert "kernel:survival_inputs" in primitives
    assert "kernel:temporal_features" in primitives
    assert "kernel:ph_schoenfeld" in primitives
    assert "kernel:rmst" in primitives
    assert primitives["kernel:survival_inputs"].kind == "kernel"
    assert primitives["kernel:survival_inputs"].runtime_verification == (
        "source_digest_bound"
    )
    assert primitives["package:lifelines"].runtime_verification == (
        "runner_snapshot_required"
    )


def test_unregistered_primary_family_does_not_borrow_descriptive_authority():
    catalog = scientific_actions_for_analysis_type("multimodal")
    assert catalog.analysis_family == "descriptive"
    assert catalog.primary_contract_registered is False
    assert catalog.primary_contract_id is None

    guide = planner_scientific_action_guide("multimodal")
    assert "NO registered primary execution contract" in guide
    assert "do not claim a completed family workflow" in guide


@pytest.mark.parametrize("analysis_type", ["data_quality_audit", "measurement_bias_audit"])
def test_measurement_audit_types_reuse_the_existing_typed_primary_capability(
    analysis_type: str,
):
    catalog = scientific_actions_for_analysis_type(analysis_type)

    assert catalog.analysis_family == "descriptive"
    assert catalog.primary_contract_registered is True
    assert catalog.primary_contract_id == "descriptive_measurement_v1"
    assert _action(analysis_type, "missingness_audit").execution_mode == "host_owned"


def test_budget_ladder_retains_every_action_and_execution_boundary():
    expected = {
        action.action_id
        for action in scientific_actions_for_analysis_type("prediction_model").actions
    }
    sizes = []
    for detail in CATALOG_DETAIL_LADDER:
        guide = planner_scientific_action_guide(
            "prediction_model",
            detail=detail,
        )
        sizes.append(len(guide.encode("utf-8")))
        assert all(action_id in guide for action_id in expected)
        assert "host_owned" in guide
        assert "coder_generated" in guide
        assert "not_available" in guide
    assert sizes == sorted(sizes, reverse=True)


def test_real_planner_request_contains_the_compiled_scientific_actions():
    context = schema.ResearchContext(
        research_question=(
            "Develop and validate a calibrated ICU mortality prediction model."
        ),
        cohort=schema.CohortDescriptor(
            cohort_name="prediction cohort",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[
            schema.ConceptDescriptor(
                name="age",
                role=schema.VariableRole.DEMOGRAPHIC,
                dtype="float64",
            ),
            schema.ConceptDescriptor(
                name="death",
                role=schema.VariableRole.OUTCOME,
                dtype="int64",
            ),
        ],
        target_outcome="death",
        primary_exposure="age",
    )

    prompt = core._build_planner_user_prompt(context, catalog_detail="names_only")
    assert "ACTIONS exact; gaps fail" in prompt
    assert "prediction.delong_ci" in prompt
    assert "prediction.decision_curve" in prompt
    assert "prediction.conformal_intervals" in prompt


def test_replanner_request_keeps_the_exact_action_catalog():
    context = schema.ResearchContext(
        research_question="Compare survival time between two ICU groups.",
        cohort=schema.CohortDescriptor(
            cohort_name="survival cohort",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = schema.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="survival",
        steps=[
            schema.AnalysisStep(
                step_id="01_rmst",
                planned_analysis_role="secondary",
                intent="Estimate restricted mean survival time.",
                method="rmst",
                scientific_action_id="time_to_event.rmst",
                expected_outputs=["table:rmst"],
            )
        ],
    )
    llm = ScriptedMockLLMClient([plan.model_dump_json()])

    ReplannerAgent(llm).run(context=context, current_plan=plan)

    prompt = "\n".join(str(message.content or "") for message in llm.messages)
    assert "ACTIONS exact; gaps fail" in prompt
    assert "time_to_event.rmst" in prompt
    assert "time_to_event.competing_risks_cif" in prompt


def test_typed_plan_selection_refuses_cross_family_and_unavailable_actions():
    valid = schema.AnalysisPlan(
        research_question="Compare survival time between two groups.",
        analysis_type="survival",
        steps=[
            schema.AnalysisStep(
                step_id="rmst",
                planned_analysis_role="secondary",
                intent="Estimate restricted mean survival time.",
                method="rmst",
                scientific_action_id="time_to_event.rmst",
            )
        ],
    )
    validate_plan_scientific_action_selections(
        plan=valid,
        inferred_analysis_type="survival",
    )

    cross_family = valid.model_copy(deep=True)
    cross_family.steps[0].scientific_action_id = "prediction.delong_ci"
    with pytest.raises(ValueError, match="is not registered"):
        validate_plan_scientific_action_selections(
            plan=cross_family,
            inferred_analysis_type="survival",
        )

    unavailable = valid.model_copy(deep=True)
    unavailable.steps[0].scientific_action_id = (
        "time_to_event.competing_risks_cif"
    )
    with pytest.raises(ValueError, match="recognised but not available"):
        validate_plan_scientific_action_selections(
            plan=unavailable,
            inferred_analysis_type="survival",
        )

    inconsistent = valid.model_copy(deep=True)
    inconsistent.steps[0].method = "km_logrank"
    with pytest.raises(ValueError, match="scientific_action_method_mismatch"):
        validate_plan_scientific_action_selections(
            plan=inconsistent,
            inferred_analysis_type="survival",
        )


def test_real_planner_parser_fails_before_execution_on_unavailable_action():
    context = schema.ResearchContext(
        research_question="Develop an ICU mortality prediction model.",
        cohort=schema.CohortDescriptor(
            cohort_name="prediction cohort",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
    )
    raw = json.dumps(
        {
            "research_question": context.research_question,
            "analysis_type": "prediction_model",
            "steps": [
                {
                    "step_id": "01_external_validation",
                    "planned_analysis_role": "secondary",
                    "intent": "Externally validate the model.",
                    "method": "external_validation",
                    "scientific_action_id": "prediction.external_validation",
                    "inputs": [],
                    "expected_outputs": ["table:external_validation"],
                }
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="scientific_action_requires_user_choice.*require explicit user choice",
    ):
        PlannerAgent.__new__(PlannerAgent)._parse(raw, context)


def test_live_planner_requires_actions_but_historical_replay_remains_compatible():
    context = schema.ResearchContext(
        research_question="Describe ICU mortality by exposure group.",
        cohort=schema.CohortDescriptor(
            cohort_name="descriptive cohort",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
    )
    raw = json.dumps(
        {
            "research_question": context.research_question,
            "analysis_type": "descriptive",
            "steps": [
                {
                    "step_id": "01_summary",
                    "planned_analysis_role": "primary",
                    "intent": "Produce the descriptive result table.",
                    "method": "descriptive_summary",
                    "inputs": [],
                    "expected_outputs": ["table:descriptive_summary"],
                },
                {
                    "step_id": "02_figure",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Render the declared result.",
                    "method": "publication_figure",
                    "inputs": ["table:descriptive_summary"],
                    "expected_outputs": ["figure:descriptive_summary"],
                },
            ],
        }
    )

    # Compatibility mode is only for re-reading historical plans.
    PlannerAgent.__new__(PlannerAgent)._parse(raw, context)
    with pytest.raises(ValueError, match="scientific_action_required_for_result_step"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            raw,
            context,
            require_scientific_actions=True,
        )

    # A genuinely free-form method has no action coordinate to bind.  It stays
    # under the existing capability/review gates rather than being laundered
    # into the nearest catalog entry.
    freeform = json.loads(raw)
    freeform["steps"][0]["method"] = "bespoke_descriptive_protocol"
    PlannerAgent.__new__(PlannerAgent)._parse(
        json.dumps(freeform),
        context,
        require_scientific_actions=True,
    )


def test_scientific_action_is_part_of_resume_scientific_authority():
    approved = schema.AnalysisStep(
        step_id="01_discrimination",
        planned_analysis_role="secondary",
        intent="Estimate model discrimination uncertainty.",
        method="delong_ci",
        scientific_action_id="prediction.delong_ci",
    )
    drifted = approved.model_copy(
        update={"scientific_action_id": "prediction.decision_curve"}
    )

    assert _step_scientific_signature(approved) != _step_scientific_signature(
        drifted
    )


def test_typed_action_drives_exact_coder_action_and_kernel_selection():
    bundle = build_coder_resource_bundle(
        step_id="delong",
        profile_ref="profile:test",
        analysis_family="prediction_model",
        step_role="secondary",
        question="Estimate an AUROC confidence interval.",
        intent="Compute the DeLong confidence interval for model discrimination.",
        method="DeLong confidence interval",
        scientific_action_id="prediction.delong_ci",
        planner_inputs=("table:predictions",),
        expected_outputs=("table:auroc_delong",),
        resolved_input_bindings={},
        runtime_import_names=(
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "statsmodels",
            "sklearn",
            "pyarrow",
        ),
    )

    assert bundle.scientific_action_id == "prediction.delong_ci"
    action_ids = {
        selected.resource_id
        for selected in bundle.selections[0].selected
    }
    software_ids = {
        selected.resource_id
        for selected in bundle.selections[1].selected
    }
    assert "action:scientific:prediction.delong_ci" in action_ids
    assert (
        "software:easyicu.research_agent.methods.delong_auc" in software_ids
    )
    assert '"scientific_action_id":"prediction.delong_ci"' in (
        bundle.prompt_projection
    )


def test_dynamic_action_selects_sklearn_and_both_reviewed_timing_kernels():
    bundle = build_coder_resource_bundle(
        step_id="dynamic",
        profile_ref="profile:test",
        analysis_family="dynamic_prediction",
        step_role="primary",
        question="Update deterioration risk at 6 and 12 hours.",
        intent="Fit a patient-split landmark prediction model.",
        method="dynamic landmark prediction",
        scientific_action_id="prediction.dynamic_prediction",
        planner_inputs=("artifact:trajectory", "artifact:outcome_times"),
        expected_outputs=("prediction:time_varying_discrimination",),
        resolved_input_bindings={},
        runtime_import_names=(
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "statsmodels",
            "sklearn",
            "pyarrow",
        ),
    )

    software_ids = {
        selected.resource_id for selected in bundle.selections[1].selected
    }
    assert "software:sklearn" in software_ids
    assert (
        "software:easyicu.research_agent.methods.dynamic_prediction"
        in software_ids
    )
    assert (
        "software:easyicu.research_agent.methods.temporal_features"
        in software_ids
    )
    assert "patient-level" in bundle.prompt_projection

    with pytest.raises(ValueError, match="required software resources are unavailable"):
        build_coder_resource_bundle(
            step_id="dynamic-missing-runtime",
            profile_ref="profile:test",
            analysis_family="dynamic_prediction",
            step_role="primary",
            question="Update deterioration risk over time.",
            intent="Fit a patient-split landmark prediction model.",
            method="dynamic landmark prediction",
            scientific_action_id="prediction.dynamic_prediction",
            planner_inputs=("artifact:trajectory", "artifact:outcome_times"),
            expected_outputs=("prediction:time_varying_discrimination",),
            resolved_input_bindings={},
            runtime_import_names=("pandas", "numpy", "pyarrow"),
        )
