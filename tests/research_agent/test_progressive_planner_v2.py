"""Progressive Planner v2 contract/compiler regressions."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from easyicu.research_agent.agents.progressive_payload import (
    ProgressiveTransportSchemaError,
    progressive_foundation_structured_output_request,
    progressive_outline_structured_output_request,
    progressive_step_materialization_request,
    progressive_structured_output_request,
)
from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    _action_catalog,
    _bind_runtime_action_dependencies,
    _complete_case_variable_roster,
    _parse_foundation_materialization,
    _parse_step_materialization,
    candidate_analysis_types,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    exposure_outcome_distribution_executor_owns_step,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    robustness_replay_spec_is_emittable,
)
from easyicu.research_agent.execution.runners.scientific_reporting_executor import (
    scientific_reporting_executor_owns_step,
)
from easyicu.research_agent.planning.progressive_compiler import (
    assert_immutable_prefix,
    compile_progressive_plan,
)
from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.cohort_contract import concept_id_exists
from easyicu.research_agent.planning.progressive_artifacts import (
    ProgressiveCompileFailureReplay,
    ProgressivePlannerCheckpointRecorder,
    ProgressivePlanningArtifactError,
    load_progressive_compile_failure_replay,
    load_progressive_planner_checkpoint_chain,
    persist_progressive_planner_checkpoint,
    persist_progressive_planning_artifacts,
    persist_progressive_planning_authority,
)
from easyicu.research_agent.planning.progressive_contract import (
    PROGRESSIVE_HOST_COMPILED_OUTPUTS,
    ProgressiveFoundationMaterialization,
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
    ProgressivePlannerCheckpoint,
    ProgressivePredicateValue,
    ProgressiveStepMaterialization,
)
from easyicu.research_agent.planning.progressive_host_materialization import (
    host_materialize_progressive_step,
    normalize_progressive_cohort_identity,
)
from easyicu.research_agent.planning.progressive_resume import (
    ProgressivePrefixState,
    compile_progressive_prefix,
    validate_progressive_materialization_coordinate,
)
from easyicu.research_agent.orchestration.progressive_planning import (
    ProgressiveDesignCanaryDraft,
    run_progressive_planner,
)
from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
from easyicu.research_agent.planning.preplan_know_how import (
    verify_know_how_decisions,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.authority.plan_lifecycle import (
    build_normalized_plan_lineage,
)
from easyicu.research_agent.cohort.schema import (
    materialized_input_column_authority,
)
from easyicu.research_agent.providers.strict_json_schema import (
    closed_pydantic_json_schema,
)
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    MissingnessProfile,
    ObservationSemantics,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from easyicu.research_agent.reporting.article_contract import (
    build_article_analysis_contract,
    roles_covered_by_plan,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Estimate an exposure-outcome association with audit context.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_stays=120,
            id_columns=["stay_id"],
            outcome_columns=["outcome_flag"],
        ),
        variables=[
            ConceptDescriptor(
                name="exposure_flag",
                role=VariableRole.INTERVENTION,
                dtype="int64",
                observed_domain={
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            ),
            ConceptDescriptor(
                name="outcome_flag",
                role=VariableRole.OUTCOME,
                dtype="int64",
                observed_domain={
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            ),
            ConceptDescriptor(
                name="age_years",
                dtype="float64",
                observed_domain={"n_unique": 83, "min": 18.0, "max": 100.0},
            ),
            ConceptDescriptor(
                name="sex_code",
                dtype="object",
                observed_domain={
                    "n_unique": 2,
                    "is_binary": False,
                    "levels": ["A", "B"],
                },
            ),
        ],
        primary_exposure="exposure_flag",
        target_outcome="outcome_flag",
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"mode": "none"}, None),
        ({"mode": "string", "string_value": "adult"}, "adult"),
        ({"mode": "number", "number_value": 18.0}, 18.0),
        ({"mode": "boolean", "boolean_value": True}, True),
        ({"mode": "string_list", "string_list": ["A", "B"]}, ["A", "B"]),
        ({"mode": "number_list", "number_list": [0.0, 1.0]}, [0.0, 1.0]),
    ],
)
def test_progressive_predicate_value_materializes_its_declared_field(
    payload: dict[str, object],
    expected: object,
) -> None:
    assert ProgressivePredicateValue.model_validate(payload).materialize() == expected


def _payload() -> dict:
    return {
        "schema_version": "easyicu.progressive_plan_skeleton/1",
        "analysis_type": "association_study",
        "cohort": {
            "name": "primary",
            "selection_mode": "all_input_rows",
            "inclusion": [],
            "exclusion": [],
        },
        "display_labels": [
            {"key": "exposure_flag=0", "value": "Exposure absent"},
            {"key": "exposure_flag=1", "value": "Exposure present"},
        ],
        "robustness_intents": [
            {
                "spec_id": "complete_case",
                "axis": "missing",
                "description": "Refit the declared model on complete observations.",
                "missing_strategy": "complete_case",
                "complete_case_variables": [
                    "exposure_flag",
                    "outcome_flag",
                    "age_years",
                    "sex_code",
                ],
            }
        ],
        "steps": [
            {
                "step_id": "01_cohort",
                "planned_analysis_role": "auxiliary",
                "module_id": "cohort_definition",
                "objective": "Bind and account for the prespecified analysis universe.",
                "depends_on": [],
                "raw_inputs": [],
                "product_inputs": [],
                "outputs": [],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "02_table_one",
                "planned_analysis_role": "auxiliary",
                "module_id": "table_one",
                "objective": "Describe baseline variables by the declared exposure groups.",
                "depends_on": ["01_cohort"],
                "raw_inputs": ["exposure_flag", "age_years", "sex_code"],
                "product_inputs": [],
                "outputs": [],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": "exposure_flag",
                "table_one_mode": "descriptive_smd_only",
                "table_one_variables": [
                    {"name": "age_years", "summary": "median_iqr"},
                    {"name": "sex_code", "summary": "count_percent"},
                ],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "03_distribution",
                "planned_analysis_role": "secondary",
                "module_id": "exposure_outcome_distribution",
                "objective": "Estimate prevalence and absolute outcome risk by exposure.",
                "depends_on": ["01_cohort"],
                "raw_inputs": ["exposure_flag", "outcome_flag"],
                "product_inputs": [],
                "outputs": [],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": "exposure_flag",
                "outcome": "outcome_flag",
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": 1,
                "reference_exposure_level_index": 0,
                "comparison_exposure_level_index": 1,
                "primary_contrast_level_index": None,
                "denominator_policy": "all_declared_rows",
                "missing_exposure_policy": "fail_closed",
                "missing_outcome_policy": "fail_closed",
                "confidence_level": 0.95,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "04_measurement",
                "planned_analysis_role": "auxiliary",
                "module_id": "measurement_audit",
                "objective": "Audit missingness and observation-process coverage.",
                "depends_on": ["01_cohort"],
                "raw_inputs": [
                    "exposure_flag",
                    "outcome_flag",
                    "age_years",
                    "sex_code",
                ],
                "product_inputs": [],
                "outputs": [
                    {
                        "product_id": "table:measurement_missingness",
                        "semantic_role": "measurement_missingness",
                    },
                    {
                        "product_id": "table:measurement_process",
                        "semantic_role": "measurement_process",
                    },
                ],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "05_primary",
                "planned_analysis_role": "primary",
                "module_id": "adjusted_association",
                "objective": "Estimate the prespecified adjusted association.",
                "depends_on": ["01_cohort"],
                "raw_inputs": [
                    "exposure_flag",
                    "outcome_flag",
                    "age_years",
                    "sex_code",
                ],
                "product_inputs": [],
                "outputs": [],
                "scientific_action_id": "association.adjusted_association",
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": "exposure_flag",
                "outcome": "outcome_flag",
                "outcome_type": "binary",
                "model_terms": [
                    {
                        "name": "exposure_flag",
                        "role": "exposure",
                        "coding": "binary",
                        "reference_level_index": 0,
                    },
                    {
                        "name": "age_years",
                        "role": "covariate",
                        "coding": "continuous",
                        "reference_level_index": None,
                    },
                    {
                        "name": "sex_code",
                        "role": "covariate",
                        "coding": "binary",
                        "reference_level_index": 0,
                    },
                ],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "06_sensitivity",
                "planned_analysis_role": "sensitivity",
                "module_id": "custom_analysis",
                "objective": "Run the explicitly prespecified scientific sensitivity grid.",
                "depends_on": ["05_primary"],
                "raw_inputs": ["exposure_flag", "outcome_flag", "age_years"],
                "product_inputs": [
                    {
                        "producer_step_id": "05_primary",
                        "product_id": "table:adjusted_association_estimates",
                    }
                ],
                "outputs": [
                    {
                        "product_id": "table:scientific_sensitivity",
                        "semantic_role": "scientific_sensitivity",
                    }
                ],
                "scientific_action_id": None,
                "custom_method": "prespecified_scientific_sensitivity",
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": ["flexible_form"],
                "literature_bindings": [],
            },
            {
                "step_id": "07_figure",
                "planned_analysis_role": "auxiliary",
                "module_id": "visualization",
                "objective": "Render the exact descriptive and adjusted result products.",
                "depends_on": ["03_distribution", "05_primary"],
                "raw_inputs": [],
                "product_inputs": [
                    {
                        "producer_step_id": "03_distribution",
                        "product_id": "table:exposure_outcome_distribution",
                    },
                    {
                        "producer_step_id": "05_primary",
                        "product_id": "table:adjusted_association_estimates",
                    },
                ],
                "outputs": [
                    {
                        "product_id": "figure:primary_results",
                        "semantic_role": "figure",
                    }
                ],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
        ],
        "rationale": "Separate descriptive denominators from the adjusted association.",
    }


def _prediction_payload() -> dict:
    def step(**updates):
        payload = {
            "step_id": "step",
            "planned_analysis_role": "auxiliary",
            "module_id": "custom_analysis",
            "objective": "Execute one fully declared prediction workflow step.",
            "depends_on": [],
            "raw_inputs": [],
            "product_inputs": [],
            "outputs": [],
            "scientific_action_id": None,
            "custom_method": "prediction_runtime",
            "table_one_group_by": None,
            "table_one_mode": None,
            "table_one_variables": [],
            "primary_exposure": None,
            "outcome": None,
            "outcome_type": None,
            "model_terms": [],
            "event_level_index": None,
            "reference_exposure_level_index": None,
            "comparison_exposure_level_index": None,
            "primary_contrast_level_index": None,
            "denominator_policy": None,
            "missing_exposure_policy": None,
            "missing_outcome_policy": None,
            "confidence_level": None,
            "sensitivity_spec_ids": [],
            "literature_bindings": [],
        }
        payload.update(updates)
        return payload

    return {
        "schema_version": "easyicu.progressive_plan_skeleton/1",
        "analysis_type": "prediction_model",
        "cohort": {
            "name": "prediction_cohort",
            "selection_mode": "all_input_rows",
            "inclusion": [],
            "exclusion": [],
        },
        "display_labels": [],
        "robustness_intents": [],
        "know_how_decisions": [],
        "steps": [
            step(
                step_id="cohort",
                module_id="cohort_definition",
                objective="Bind and account for the prediction analysis cohort.",
                custom_method=None,
            ),
            step(
                step_id="primary_model",
                planned_analysis_role="primary",
                objective="Fit and evaluate the prespecified static prediction model.",
                depends_on=["cohort"],
                raw_inputs=["age_years", "sex_code", "outcome_flag"],
                outputs=[
                    {
                        "product_id": "table:prediction_scores",
                        "semantic_role": "custom",
                    },
                    {
                        "product_id": "table:model_performance",
                        "semantic_role": "custom",
                    },
                ],
                scientific_action_id="prediction.discrimination_calibration",
            ),
            step(
                step_id="validation",
                planned_analysis_role="secondary",
                objective="Summarize the fixed patient-separated validation split.",
                depends_on=["primary_model"],
                raw_inputs=["outcome_flag"],
                product_inputs=[
                    {
                        "producer_step_id": "primary_model",
                        "product_id": "table:prediction_scores",
                    }
                ],
                outputs=[
                    {
                        "product_id": "table:validation",
                        "semantic_role": "custom",
                    }
                ],
                scientific_action_id="prediction.internal_validation",
            ),
            step(
                step_id="calibration",
                planned_analysis_role="secondary",
                objective="Quantify calibration on the validation partition.",
                depends_on=["primary_model"],
                raw_inputs=["outcome_flag"],
                product_inputs=[
                    {
                        "producer_step_id": "primary_model",
                        "product_id": "table:prediction_scores",
                    }
                ],
                outputs=[
                    {
                        "product_id": "table:calibration",
                        "semantic_role": "custom",
                    }
                ],
                scientific_action_id="prediction.calibration_metrics",
            ),
            step(
                step_id="clinical_utility",
                planned_analysis_role="secondary",
                objective="Quantify net benefit across the fixed threshold grid.",
                depends_on=["primary_model"],
                raw_inputs=["outcome_flag"],
                product_inputs=[
                    {
                        "producer_step_id": "primary_model",
                        "product_id": "table:prediction_scores",
                    }
                ],
                outputs=[
                    {
                        "product_id": "table:clinical_utility",
                        "semantic_role": "custom",
                    }
                ],
                scientific_action_id="prediction.decision_curve",
            ),
            step(
                step_id="figure",
                module_id="visualization",
                objective="Render the four registered prediction result tables.",
                depends_on=["primary_model", "validation", "calibration"],
                product_inputs=[
                    {
                        "producer_step_id": "primary_model",
                        "product_id": "table:prediction_scores",
                    },
                    {
                        "producer_step_id": "primary_model",
                        "product_id": "table:model_performance",
                    },
                    {
                        "producer_step_id": "validation",
                        "product_id": "table:validation",
                    },
                    {
                        "producer_step_id": "calibration",
                        "product_id": "table:calibration",
                    },
                ],
                outputs=[
                    {
                        "product_id": "figure:prediction_results",
                        "semantic_role": "figure",
                    }
                ],
                custom_method=None,
            ),
        ],
        "rationale": "Separate model fitting, validation, calibration and rendering.",
    }


def test_prediction_action_contract_compiles_exact_products_and_dependencies():
    skeleton = ProgressivePlanSkeleton.model_validate(_prediction_payload())
    plan, _receipt = compile_progressive_plan(skeleton=skeleton, context=_context())

    primary = next(step for step in plan.steps if step.step_id == "primary_model")
    validation = next(step for step in plan.steps if step.step_id == "validation")
    figure = next(step for step in plan.steps if step.step_id == "figure")
    assert primary.expected_outputs == [
        "table:prediction_scores",
        "table:model_performance",
    ]
    assert "artifact:analysis_cohort" in primary.inputs
    assert validation.inputs == ["outcome_flag", "table:prediction_scores"]
    assert figure.inputs == [
        "table:prediction_scores",
        "table:model_performance",
        "table:validation",
        "table:calibration",
    ]
    contract = build_article_analysis_contract(
        _context(), analysis_type="prediction_model"
    )
    covered = roles_covered_by_plan(plan, contract)
    assert {
        "model_performance",
        "validation",
        "calibration",
        "clinical_utility",
    } <= covered


def test_prediction_action_contract_rejects_artifact_outputs_and_wrong_figure_owner():
    wrong_output = _prediction_payload()
    wrong_output["steps"][1]["outputs"][0]["product_id"] = "artifact:prediction_scores"
    with pytest.raises(
        ProgressivePlanCompileError,
        match="progressive_scientific_action_outputs_mismatch",
    ):
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(wrong_output),
            context=_context(),
        )

    wrong_owner = _prediction_payload()
    wrong_owner["steps"][5]["product_inputs"][0] = {
        "producer_step_id": "cohort",
        "product_id": "table:cohort_flow",
    }
    with pytest.raises(
        ProgressivePlanCompileError,
        match="progressive_product_dependency_mismatch",
    ):
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(wrong_owner),
            context=_context(),
        )


def _skeleton() -> ProgressivePlanSkeleton:
    return ProgressivePlanSkeleton.model_validate(_payload())


def _outline_payload(payload: dict | None = None) -> dict:
    source = payload or _payload()
    analysis_type = source["analysis_type"]
    return {
        "schema_version": "easyicu.progressive_plan_outline/1",
        "analysis_type": analysis_type,
        "cohort_objective": "Use the sealed cohort and preserve its denominator.",
        "design_selection": {
            "schema_version": "easyicu.research_design_selection/1",
            "claim_ceiling": "analysis_only",
            "candidates": [
                {
                    "design_id": "selected_primary_design",
                    "analysis_type": analysis_type,
                    "estimand": "Adjusted exposure contrast for outcome_flag.",
                    "time_zero": "Start of the sealed synthetic cohort episode.",
                    "observation_window": "The prespecified episode observation window.",
                    "primary_method": "Host-owned primary analysis method",
                    "required_variables": ["exposure_flag", "outcome_flag"],
                    "assumptions": ["The declared adjustment set is adequate."],
                    "literature_citation_keys": [],
                    "novelty_positioning": "Tests the question in the sealed cohort context.",
                    "figure_role": "Show the primary estimate with its uncertainty.",
                    "supports": "The prespecified primary association estimate.",
                    "cannot_prove": "A causal effect without stronger identification.",
                    "disposition": "selected",
                    "decision_reason": (
                        "Directly binds exposure_flag and outcome_flag to the "
                        "prespecified primary question."
                    ),
                },
                {
                    "design_id": "rejected_alternative_design",
                    "analysis_type": analysis_type,
                    "estimand": "Unadjusted exposure contrast for outcome_flag.",
                    "time_zero": "Start of the sealed synthetic cohort episode.",
                    "observation_window": "The prespecified episode observation window.",
                    "primary_method": "Unadjusted descriptive contrast",
                    "required_variables": ["exposure_flag", "outcome_flag"],
                    "assumptions": ["Crude group differences are interpretable."],
                    "literature_citation_keys": [],
                    "novelty_positioning": "Provides a less adjusted comparator design.",
                    "figure_role": "Show only the crude group contrast.",
                    "supports": "A descriptive difference between exposure groups.",
                    "cannot_prove": "An adjusted or causal exposure effect.",
                    "disposition": "rejected",
                    "decision_reason": (
                        "Reject because exposure_flag confounding is not addressed "
                        "for the outcome_flag question."
                    ),
                },
            ],
        },
        "steps": [
            {
                "step_id": step["step_id"],
                "planned_analysis_role": step["planned_analysis_role"],
                "module_id": step["module_id"],
                "objective": step["objective"],
                "depends_on": list(step["depends_on"]),
                "variable_names": [
                    "exposure_flag",
                    "outcome_flag",
                    "age_years",
                    "sex_code",
                ],
                "literature_citation_keys": list(
                    dict.fromkeys(
                        binding["citation_key"]
                        for binding in step["literature_bindings"]
                    )
                ),
                "scientific_action_id": step["scientific_action_id"],
            }
            for step in source["steps"]
        ],
        "rationale": source["rationale"],
    }


def _outline_with_repeated_robustness() -> dict:
    payload = _outline_payload()
    for sequence, suffix, objective in (
        (8, "a", "Replay all prespecified complete-case robustness intents."),
        (9, "b", "Replay another compatible robustness intent bundle."),
    ):
        payload["steps"].append(
            {
                "step_id": f"{sequence:02d}_robustness_{suffix}",
                "planned_analysis_role": "sensitivity",
                "module_id": "robustness_replay",
                "objective": objective,
                "depends_on": ["05_primary"],
                "variable_names": ["exposure_flag", "outcome_flag", "age_years"],
                "literature_citation_keys": [],
                "scientific_action_id": None,
            }
        )
    return payload


def _materialization_payloads(payload: dict | None = None) -> list[dict]:
    source = payload or _payload()
    outline = ProgressivePlanOutline.model_validate(_outline_payload(source))
    responses = []
    for outline_step, step in zip(outline.steps, source["steps"], strict=True):
        responses.append(
            {
                "schema_version": "easyicu.progressive_step_materialization/1",
                "outline_step_sha256": canonical_sha256(
                    outline_step.model_dump(mode="json")
                ),
                "foundation": None,
                "step": step,
            }
        )
    return responses


def _foundation_payload(payload: dict | None = None) -> dict:
    source = payload or _payload()
    outline = ProgressivePlanOutline.model_validate(_outline_payload(source))
    return {
        "schema_version": "easyicu.progressive_plan_foundation/1",
        "outline_sha256": canonical_sha256(outline.model_dump(mode="json")),
        "foundation": {
            "cohort": source["cohort"],
            "display_labels": source["display_labels"],
            "robustness_intents": source["robustness_intents"],
            "know_how_decisions": source.get("know_how_decisions", []),
        },
    }


def test_host_materializes_only_mechanical_outline_coordinates() -> None:
    outline = ProgressivePlanOutline.model_validate(_outline_payload())
    foundation = ProgressiveFoundationMaterialization.model_validate(
        _foundation_payload()
    ).foundation
    context = _context()

    cohort = host_materialize_progressive_step(
        context=context,
        outline=outline,
        outline_step=outline.steps[0],
        foundation=foundation,
        available_product_refs=(),
    )
    table_one = host_materialize_progressive_step(
        context=context,
        outline=outline,
        outline_step=outline.steps[1],
        foundation=foundation,
        available_product_refs=(("01_cohort", "artifact:analysis_cohort"),),
    )
    distribution = host_materialize_progressive_step(
        context=context,
        outline=outline,
        outline_step=outline.steps[2],
        foundation=foundation,
        available_product_refs=(("01_cohort", "artifact:analysis_cohort"),),
    )
    primary = host_materialize_progressive_step(
        context=context,
        outline=outline,
        outline_step=outline.steps[4],
        foundation=foundation,
        available_product_refs=(),
    )

    assert cohort is not None and cohort.step.raw_inputs
    assert table_one is not None
    assert table_one.step.table_one_group_by == context.primary_exposure
    assert distribution is not None
    assert distribution.step.primary_exposure == context.primary_exposure
    assert distribution.step.outcome == context.target_outcome
    # The model still owns the primary model-term and estimand decisions.
    assert primary is None


def _multi_identity_context(*, proven_stay: bool) -> ResearchContext:
    stay_id = "stay_id" if proven_stay else "episode_key"
    patient_id = "patient_id" if proven_stay else "person_key"
    return _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": [stay_id, patient_id],
                    "provenance": (
                        {
                            "analysis_unit": "icu_stay",
                            "stay_id_columns": [stay_id],
                            "patient_id_columns": [patient_id],
                        }
                        if proven_stay
                        else {"analysis_unit": "row"}
                    ),
                }
            ),
            "variables": [
                *_context().variables,
                ConceptDescriptor(name=stay_id, role=VariableRole.ID, dtype="int64"),
                ConceptDescriptor(name=patient_id, role=VariableRole.ID, dtype="int64"),
            ],
        }
    )


def test_host_cohort_uses_proven_stay_identity_not_patient_count_id() -> None:
    context = _multi_identity_context(proven_stay=True)
    outline_payload = _outline_payload()
    outline_payload["steps"][0]["variable_names"] = [
        "stay_id",
        "patient_id",
        "exposure_flag",
    ]
    outline = ProgressivePlanOutline.model_validate(outline_payload)
    foundation = ProgressiveFoundationMaterialization.model_validate(
        _foundation_payload()
    ).foundation

    materialization = host_materialize_progressive_step(
        context=context,
        outline=outline,
        outline_step=outline.steps[0],
        foundation=foundation,
        available_product_refs=(),
    )

    assert materialization is not None
    assert materialization.step.raw_inputs == ["stay_id", "exposure_flag"]


def test_host_cohort_does_not_guess_between_untyped_id_columns() -> None:
    context = _multi_identity_context(proven_stay=False)
    outline_payload = _outline_payload()
    outline_payload["steps"][0]["variable_names"] = [
        "episode_key",
        "person_key",
    ]
    outline = ProgressivePlanOutline.model_validate(outline_payload)
    foundation = ProgressiveFoundationMaterialization.model_validate(
        _foundation_payload()
    ).foundation

    assert (
        host_materialize_progressive_step(
            context=context,
            outline=outline,
            outline_step=outline.steps[0],
            foundation=foundation,
            available_product_refs=(),
        )
        is None
    )


def test_host_normalizes_proven_identity_without_dropping_planner_bindings() -> None:
    context = _multi_identity_context(proven_stay=True)
    payload = _materialization_payloads()[0]
    payload["step"]["raw_inputs"] = ["stay_id", "patient_id", "exposure_flag"]
    payload["step"]["literature_bindings"] = [
        {
            "citation_key": "strobe_2007",
            "design_elements": ["reporting"],
            "application": "Retain the Planner-authored reporting application.",
            "divergence": None,
        }
    ]
    materialization = ProgressiveStepMaterialization.model_validate(payload)

    normalized = normalize_progressive_cohort_identity(
        materialization,
        context=context,
    )

    assert normalized.step.raw_inputs == ["stay_id", "exposure_flag"]
    assert (
        normalized.step.literature_bindings
        == materialization.step.literature_bindings
    )


def test_host_does_not_fabricate_dynamic_literature_application() -> None:
    context = _multi_identity_context(proven_stay=True)
    outline_payload = _outline_payload()
    outline_payload["steps"][0]["variable_names"] = ["stay_id", "patient_id"]
    outline_payload["steps"][0]["literature_citation_keys"] = ["dynamic_card"]
    outline = ProgressivePlanOutline.model_validate(outline_payload)

    assert (
        host_materialize_progressive_step(
            context=context,
            outline=outline,
            outline_step=outline.steps[0],
            foundation=ProgressiveFoundationMaterialization.model_validate(
                _foundation_payload()
            ).foundation,
            available_product_refs=(),
        )
        is None
    )


def test_host_reuses_selected_design_decision_for_dynamic_figure_binding() -> None:
    outline_payload = _outline_payload()
    selected = outline_payload["design_selection"]["candidates"][0]
    selected["literature_citation_keys"] = ["dynamic_card"]
    selected["literature_design_decisions"] = [
        {
            "dimension": "table_and_figure_completeness",
            "citation_keys": ["dynamic_card"],
            "disposition": "adopt",
            "rationale": "Show cohort accounting, the primary estimate, and robustness.",
        }
    ]
    figure_payload = next(
        step
        for step in outline_payload["steps"]
        if step["module_id"] == "visualization"
    )
    figure_payload["literature_citation_keys"] = ["dynamic_card"]
    outline = ProgressivePlanOutline.model_validate(outline_payload)
    figure = next(step for step in outline.steps if step.module_id == "visualization")

    materialization = host_materialize_progressive_step(
        context=_context(),
        outline=outline,
        outline_step=figure,
        foundation=ProgressiveFoundationMaterialization.model_validate(
            _foundation_payload()
        ).foundation,
        available_product_refs=[
            (figure.depends_on[0], "table:primary_estimate"),
        ],
    )

    assert materialization is not None
    assert [
        binding.citation_key for binding in materialization.step.literature_bindings
    ] == ["dynamic_card"]
    assert "Show cohort accounting" in materialization.step.literature_bindings[
        0
    ].application


def test_host_materialization_keeps_one_schema_ledger_entry_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializations = _materialization_payloads()
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(materializations[index]) for index in (3, 4, 5)],
        ]
    )
    llm.supports_strict_json_schema = True
    monkeypatch.setattr(
        "easyicu.research_agent.agents.progressive_planner.llm_is_mockish",
        lambda _llm: False,
    )
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(_context())

    assert len(plan.steps) == 7
    assert len(llm.calls) == 5
    assert agent.last_prompt_metrics["host_step_materialization_count"] == 4
    assert agent.last_prompt_metrics["step_materialization_payload_bytes"].count(0) == 4
    assert len(agent.last_prompt_metrics["step_materialization_schema_sha256"]) == 7


def test_checkpoint_resume_reuses_host_materialized_null_schema_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializations = _materialization_payloads()
    dependency_context = {
        "cohort_file_sha256": "b" * 64,
        "llm_signature": "codex:gpt-test",
        "prompt_version": "test-v1",
    }
    monkeypatch.setattr(
        "easyicu.research_agent.agents.progressive_planner.llm_is_mockish",
        lambda _llm: False,
    )
    source_llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(materializations[index]) for index in (3, 4, 5)],
        ]
    )
    source_llm.supports_strict_json_schema = True
    source_checkpoints = []
    ProgressivePlannerAgent(source_llm).run(
        _context(),
        checkpoint_callback=source_checkpoints.append,
        resume_dependency_context=dependency_context,
    )
    resume_checkpoint = source_checkpoints[4]
    assert resume_checkpoint.prompt_metrics[
        "step_materialization_schema_sha256"
    ][:3] == [None, None, None]

    resumed_llm = ScriptedMockLLMClient(
        [json.dumps(materializations[index]) for index in (3, 4, 5)]
    )
    resumed_llm.supports_strict_json_schema = True
    plan = ProgressivePlannerAgent(resumed_llm).run(
        _context(),
        resume_checkpoint=resume_checkpoint,
        resume_dependency_context=dependency_context,
    )

    assert len(plan.steps) == 7
    assert len(resumed_llm.calls) == 3


def test_step_materialization_parser_collapses_normalized_raw_input_duplicates() -> None:
    payload = _materialization_payloads()[0]
    payload["step"]["raw_inputs"] = ["age_years", " age_years ", "sex_code"]

    parsed = _parse_step_materialization(json.dumps(payload))

    assert parsed.step.raw_inputs == ["age_years", "sex_code"]


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("raw_inputs", ["age_years", " "]),
        ("depends_on", ["01_cohort", "01_cohort"]),
    ],
)
def test_step_materialization_parser_keeps_other_invalid_rosters_fail_closed(
    field: str,
    values: list[str],
) -> None:
    payload = _materialization_payloads()[1]
    payload["step"][field] = values

    with pytest.raises(ValueError, match="unique non-empty values"):
        _parse_step_materialization(json.dumps(payload))


def test_foundation_parser_collapses_only_exact_duplicate_know_how_decisions() -> None:
    payload = _foundation_payload()
    decision = {
        "card_id": "card_a",
        "card_version": "1.0.0",
        "card_sha256": "a" * 64,
        "claim_id": "claim_a",
        "disposition": "adopted",
        "reason_code": "fits_estimand",
        "rationale": "The retrieved claim matches the prespecified estimand.",
        "citation_ids": ["citation_a"],
    }
    payload["foundation"]["know_how_decisions"] = [decision, dict(decision)]

    parsed = _parse_foundation_materialization(
        json.dumps(payload),
        host_cohort=None,
    )

    assert len(parsed.foundation.know_how_decisions) == 1


def test_foundation_parser_rejects_conflicting_duplicate_know_how_decisions() -> None:
    payload = _foundation_payload()
    adopted = {
        "card_id": "card_a",
        "card_version": "1.0.0",
        "card_sha256": "a" * 64,
        "claim_id": "claim_a",
        "disposition": "adopted",
        "reason_code": "fits_estimand",
        "rationale": "The retrieved claim matches the prespecified estimand.",
        "citation_ids": ["citation_a"],
    }
    rejected = {
        **adopted,
        "disposition": "rejected",
        "reason_code": "does_not_fit_estimand",
        "rationale": "The retrieved claim conflicts with the prespecified estimand.",
    }
    payload["foundation"]["know_how_decisions"] = [adopted, rejected]

    with pytest.raises(ValueError, match="must not repeat a card/claim pair"):
        _parse_foundation_materialization(
            json.dumps(payload),
            host_cohort=None,
        )


def test_foundation_parser_canonicalizes_only_exact_citation_permutations() -> None:
    payload = _foundation_payload()
    decision = {
        "card_id": "card_a",
        "card_version": "1.0.0",
        "card_sha256": "a" * 64,
        "claim_id": "claim_a",
        "disposition": "adopted",
        "reason_code": "fits_estimand",
        "rationale": "The retrieved claim matches the prespecified estimand.",
        "citation_ids": ["citation_b", "citation_a"],
    }
    payload["foundation"]["know_how_decisions"] = [decision]
    authority = {
        "card_a": {
            "version": "1.0.0",
            "file_sha256": "a" * 64,
            "claims": {"claim_a": ("citation_a", "citation_b")},
        }
    }

    parsed = _parse_foundation_materialization(
        json.dumps(payload),
        host_cohort=None,
        allowed_know_how_decisions=authority,
    )

    assert parsed.foundation.know_how_decisions[0].citation_ids == [
        "citation_a",
        "citation_b",
    ]
    verify_know_how_decisions(parsed.foundation.know_how_decisions, authority)


def test_foundation_parser_does_not_repair_changed_citation_membership() -> None:
    payload = _foundation_payload()
    payload["foundation"]["know_how_decisions"] = [
        {
            "card_id": "card_a",
            "card_version": "1.0.0",
            "card_sha256": "a" * 64,
            "claim_id": "claim_a",
            "disposition": "adopted",
            "reason_code": "fits_estimand",
            "rationale": "The retrieved claim matches the prespecified estimand.",
            "citation_ids": ["citation_a", "citation_c"],
        }
    ]
    authority = {
        "card_a": {
            "version": "1.0.0",
            "file_sha256": "a" * 64,
            "claims": {"claim_a": ("citation_a", "citation_b")},
        }
    }
    parsed = _parse_foundation_materialization(
        json.dumps(payload),
        host_cohort=None,
        allowed_know_how_decisions=authority,
    )

    with pytest.raises(ValueError, match="changed citation binding"):
        verify_know_how_decisions(parsed.foundation.know_how_decisions, authority)


def _walk_objects(node):
    if not isinstance(node, dict):
        return
    if isinstance(node.get("properties"), dict):
        yield node
        for child in node["properties"].values():
            yield from _walk_objects(child)
    for child in (node.get("$defs") or {}).values():
        yield from _walk_objects(child)
    for key in ("items", "not", "if", "then", "else"):
        yield from _walk_objects(node.get(key))
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        for child in node.get(key) or ():
            yield from _walk_objects(child)


def test_progressive_skeleton_schema_is_small_closed_and_case_neutral() -> None:
    schema = closed_pydantic_json_schema(ProgressivePlanSkeleton)
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"))

    assert len(encoded.encode("utf-8")) < 16_000
    assert "exposure_flag" not in encoded
    assert "outcome_flag" not in encoded
    for object_schema in _walk_objects(schema):
        assert set(object_schema["required"]) == set(object_schema["properties"])
        assert object_schema["additionalProperties"] is False


def test_progressive_outline_schema_is_tiny_closed_and_has_no_step_details() -> None:
    request = progressive_outline_structured_output_request(
        analysis_types=["association_study"],
        variable_names=["exposure_flag", "outcome_flag", "age_years"],
        scientific_action_ids=["association.adjusted_association"],
    )
    schema = json.loads(request.schema_json)
    encoded = request.canonical_payload_json

    assert len(encoded.encode("utf-8")) < 4_000
    assert request.name == "easyicu_progressive_plan_outline_v1"
    assert schema["properties"]["analysis_type"]["enum"] == [
        "association_study"
    ]
    step = schema["$defs"]["ProgressiveOutlineStep"]["properties"]
    assert set(step) == {
        "step_id",
        "planned_analysis_role",
        "module_id",
        "objective",
        "depends_on",
        "variable_names",
        "literature_citation_keys",
        "scientific_action_id",
    }
    for forbidden in (
        "raw_inputs",
        "product_inputs",
        "outputs",
        "model_terms",
        "literature_bindings",
        "denominator_policy",
    ):
        assert forbidden not in encoded
    for object_schema in _walk_objects(schema):
        assert set(object_schema["required"]) == set(object_schema["properties"])
        assert object_schema["additionalProperties"] is False


def test_outline_schema_separates_reviewed_design_cards_from_method_sources() -> None:
    request = progressive_outline_structured_output_request(
        analysis_types=["association_study"],
        variable_names=["exposure", "outcome", "age"],
        scientific_action_ids=["association.adjusted_association"],
        allowed_literature_citation_keys=["reviewed_card", "spline_method"],
        design_card_citation_keys=["reviewed_card"],
    )
    schema = json.loads(request.schema_json)

    candidate = schema["$defs"]["ResearchDesignCandidate"]["properties"]
    decision = schema["$defs"]["CandidateLiteratureDesignDecision"]["properties"]
    assert candidate["literature_citation_keys"]["items"]["enum"] == [
        "reviewed_card",
        "spline_method",
    ]
    assert decision["citation_keys"]["items"]["enum"] == ["reviewed_card"]


def test_outline_schema_rejects_design_card_key_outside_sealed_roster() -> None:
    with pytest.raises(
        ProgressiveTransportSchemaError,
        match="design-card citation keys must be a subset",
    ):
        progressive_outline_structured_output_request(
            analysis_types=["association_study"],
            variable_names=["exposure", "outcome"],
            scientific_action_ids=["association.adjusted_association"],
            allowed_literature_citation_keys=["method_source"],
            design_card_citation_keys=["unsealed_card"],
        )


def test_descriptive_outline_schema_advertises_only_scientific_step_owners() -> None:
    request = progressive_outline_structured_output_request(
        analysis_types=["descriptive_epidemiology"],
        variable_names=["exposure_flag", "outcome_flag"],
        scientific_action_ids=["descriptive.descriptive_summary"],
    )
    schema = json.loads(request.schema_json)
    modules = schema["$defs"]["ProgressiveOutlineStep"]["properties"][
        "module_id"
    ]["enum"]

    assert "adjusted_association" not in modules
    assert "robustness_replay" not in modules
    assert "custom_analysis" not in modules
    assert "visualization" not in modules
    assert "report" not in modules
    assert "measurement_audit" in modules
    assert "exposure_outcome_distribution" in modules


def test_progressive_foundation_schema_is_outline_bound_and_has_no_step_fields() -> None:
    outline = ProgressivePlanOutline.model_validate(_outline_payload())
    outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))

    request = progressive_foundation_structured_output_request(
        outline_sha256=outline_sha256,
        variable_names=["exposure_flag", "outcome_flag", "age_years"],
        cohort_concept_ids=["exposure_concept", "outcome_concept"],
    )
    schema = json.loads(request.schema_json)

    assert request.name == "easyicu_progressive_plan_foundation_v1"
    assert schema["properties"]["outline_sha256"]["const"] == outline_sha256
    assert "step" not in schema["properties"]
    assert "ProgressiveSkeletonStep" not in request.schema_json
    for object_schema in _walk_objects(schema):
        assert set(object_schema["required"]) == set(object_schema["properties"])
        assert object_schema["additionalProperties"] is False


def test_foundation_schema_compiles_robustness_shape_invariant() -> None:
    request = progressive_foundation_structured_output_request(
        outline_sha256="a" * 64,
        variable_names=["exposure_flag", "outcome_flag"],
    )
    schema = json.loads(request.schema_json)
    branches = schema["$defs"]["ProgressiveRobustnessIntent"]["anyOf"]

    assert len(branches) == 1
    by_strategy = {
        branch["properties"]["missing_strategy"]["const"]: branch
        for branch in branches
    }
    complete_case = by_strategy["complete_case"]["properties"]
    assert complete_case["axis"] == {"type": "string", "const": "missing"}
    assert complete_case["complete_case_variables"]["minItems"] == 1
    assert complete_case["complete_case_variables"]["items"]["enum"] == [
        "exposure_flag",
        "outcome_flag",
    ]


def test_descriptive_foundation_schema_forbids_effect_robustness_intents() -> None:
    request = progressive_foundation_structured_output_request(
        outline_sha256="a" * 64,
        variable_names=["exposure_flag", "outcome_flag"],
        analysis_type="descriptive_epidemiology",
    )
    schema = json.loads(request.schema_json)
    robustness = schema["$defs"]["ProgressivePlanFoundation"]["properties"][
        "robustness_intents"
    ]

    assert robustness["maxItems"] == 0


def test_foundation_schema_binds_host_owned_all_input_cohort() -> None:
    request = progressive_foundation_structured_output_request(
        outline_sha256="a" * 64,
        variable_names=["exposure_flag", "outcome_flag"],
        required_cohort_selection_mode="all_input_rows",
        required_cohort_name="synthetic",
    )
    schema = json.loads(request.schema_json)
    cohort = schema["$defs"]["ProgressiveCohortIntent"]["properties"]

    assert cohort["name"] == {"type": "string", "const": "synthetic"}
    assert cohort["selection_mode"] == {
        "type": "string",
        "const": "all_input_rows",
    }
    assert cohort["inclusion"]["maxItems"] == 0
    assert cohort["exclusion"]["maxItems"] == 0


def test_current_step_schema_locks_outline_coordinate_and_product_registry() -> None:
    outline_step = ProgressiveOutlineStep(
        step_id="05_primary",
        planned_analysis_role="primary",
        module_id="adjusted_association",
        objective="Estimate the prespecified adjusted association.",
        depends_on=["01_cohort"],
        variable_names=["exposure_flag", "outcome_flag", "age_years"],
        literature_citation_keys=["strobe_2007"],
        scientific_action_id="association.adjusted_association",
    )
    outline_sha256 = canonical_sha256(outline_step.model_dump(mode="json"))
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=outline_sha256,
        variable_names=["exposure_flag", "outcome_flag", "age_years"],
        scientific_action_ids=["association.adjusted_association"],
        allowed_literature_citation_keys=["strobe_2007"],
        available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
    )
    schema = json.loads(request.schema_json)

    assert request.name == "easyicu_progressive_step_materialization_v1"
    assert schema["properties"]["outline_step_sha256"]["const"] == outline_sha256
    assert schema["properties"]["foundation"] == {"type": "null"}
    step = schema["$defs"]["ProgressiveSkeletonStep"]["properties"]
    assert step["step_id"]["const"] == "05_primary"
    assert step["planned_analysis_role"]["const"] == "primary"
    assert step["module_id"]["const"] == "adjusted_association"
    assert step["objective"]["const"] == outline_step.objective
    assert step["depends_on"]["items"] == {
        "type": "string",
        "const": "01_cohort",
    }
    assert "prefixItems" not in request.schema_json
    assert step["scientific_action_id"]["const"] == (
        "association.adjusted_association"
    )
    assert step["raw_inputs"]["items"]["enum"] == [
        "exposure_flag",
        "outcome_flag",
        "age_years",
    ]
    product = schema["$defs"]["ProgressiveProductRef"]["anyOf"]
    assert product == [
        {
            "type": "object",
            "properties": {
                "producer_step_id": {"type": "string", "const": "01_cohort"},
                "product_id": {
                    "type": "string",
                    "const": "artifact:analysis_cohort",
                },
            },
            "required": ["producer_step_id", "product_id"],
            "additionalProperties": False,
        }
    ]


def test_custom_step_schema_separates_generic_and_scientific_sensitivity_shapes() -> None:
    outline_step = ProgressiveOutlineStep(
        step_id="06_sensitivity",
        planned_analysis_role="sensitivity",
        module_id="custom_analysis",
        objective="Run a prespecified scientific sensitivity analysis.",
        depends_on=["05_primary"],
        variable_names=["exposure_flag", "outcome_flag"],
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(
            outline_step.model_dump(mode="json")
        ),
        variable_names=outline_step.variable_names,
        scientific_action_ids=(),
        available_product_refs=[
            ("05_primary", "table:adjusted_association_estimates")
        ],
    )
    schema = json.loads(request.schema_json)
    branches = schema["$defs"]["ProgressiveSkeletonStep"]["anyOf"]
    generic, scientific = branches

    generic_properties = generic["properties"]
    assert generic_properties["outputs"]["items"]["properties"][
        "semantic_role"
    ]["const"] == "custom"
    generic_product_pattern = generic_properties["outputs"]["items"]["properties"][
        "product_id"
    ]["pattern"]
    assert "table" in generic_product_pattern
    assert "custom" not in generic_product_pattern
    assert generic_properties["sensitivity_spec_ids"]["maxItems"] == 0

    scientific_properties = scientific["properties"]
    assert scientific_properties["outputs"]["minItems"] == 1
    assert scientific_properties["outputs"]["maxItems"] == 1
    assert scientific_properties["outputs"]["items"]["properties"][
        "product_id"
    ]["pattern"].startswith("^table:")
    assert scientific_properties["outputs"]["items"]["properties"][
        "semantic_role"
    ]["const"] == "scientific_sensitivity"
    assert scientific_properties["sensitivity_spec_ids"]["minItems"] == 1


def test_current_step_schema_excludes_navigation_from_statistical_fields() -> None:
    outline_step = ProgressiveOutlineStep(
        step_id="05_primary",
        planned_analysis_role="primary",
        module_id="adjusted_association",
        objective="Estimate the prespecified adjusted association.",
        variable_names=[
            "stay_id",
            "exposure_flag",
            "outcome_flag",
            "age_years",
        ],
        scientific_action_id="association.adjusted_association",
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(
            outline_step.model_dump(mode="json")
        ),
        variable_names=outline_step.variable_names,
        executable_variable_names=[
            "exposure_flag",
            "outcome_flag",
            "age_years",
        ],
        scientific_action_ids=["association.adjusted_association"],
    )
    schema = json.loads(request.schema_json)
    step = schema["$defs"]["ProgressiveSkeletonStep"]["properties"]
    model_term = schema["$defs"]["ProgressiveModelTermIntent"]["properties"]

    assert "stay_id" in step["raw_inputs"]["items"]["enum"]
    assert "stay_id" not in step["primary_exposure"]["enum"]
    assert "stay_id" not in step["outcome"]["enum"]
    assert "stay_id" not in model_term["name"]["enum"]


def test_nonstatistical_step_accepts_navigation_only_raw_roster() -> None:
    outline_step = ProgressiveOutlineStep(
        step_id="01_cohort",
        planned_analysis_role="auxiliary",
        module_id="cohort_definition",
        objective="Account for the sealed cohort rows.",
        variable_names=["stay_id"],
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(
            outline_step.model_dump(mode="json")
        ),
        variable_names=["stay_id"],
        executable_variable_names=[],
        scientific_action_ids=[],
    )
    step = json.loads(request.schema_json)["$defs"][
        "ProgressiveSkeletonStep"
    ]["properties"]

    assert step["raw_inputs"]["items"] == {
        "type": "string",
        "enum": ["stay_id"],
    }
    assert step["module_id"] == {
        "type": "string",
        "const": "cohort_definition",
    }


def test_current_step_without_available_products_closes_product_inputs() -> None:
    outline_step = ProgressivePlanOutline.model_validate(
        {
            "analysis_type": "association_study",
            "cohort_objective": "Use the authorized input cohort without invention.",
            "steps": [
                {
                    "step_id": "01_cohort",
                    "planned_analysis_role": "auxiliary",
                    "module_id": "cohort_definition",
                    "objective": "Bind the authorized cohort and its denominator.",
                    "depends_on": [],
                    "variable_names": ["exposure_flag", "outcome_flag"],
                    "literature_citation_keys": [],
                    "scientific_action_id": None,
                }
            ],
            "rationale": "Start by binding the study population authority.",
        }
    ).steps[0]
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(
            outline_step.model_dump(mode="json")
        ),
        variable_names=["exposure_flag", "outcome_flag"],
        scientific_action_ids=[],
    )
    schema = json.loads(request.schema_json)

    assert schema["properties"]["foundation"] == {"type": "null"}
    step = schema["$defs"]["ProgressiveSkeletonStep"]["properties"]
    assert step["depends_on"] == {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 0,
        "maxItems": 0,
    }
    assert step["product_inputs"]["maxItems"] == 0
    assert step["table_one_group_by"] == {"type": "null"}
    assert step["table_one_mode"] == {"type": "null"}
    assert step["table_one_variables"]["maxItems"] == 0


def test_visualization_schema_hides_products_outside_direct_dependencies() -> None:
    outline_step = ProgressiveOutlineStep(
        step_id="08_figure",
        planned_analysis_role="auxiliary",
        module_id="visualization",
        objective="Render the directly declared result sources.",
        depends_on=["05_primary"],
        variable_names=["exposure_flag", "outcome_flag"],
        literature_citation_keys=[],
        scientific_action_id=None,
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
        variable_names=["exposure_flag", "outcome_flag"],
        scientific_action_ids=[],
        available_product_refs=[
            ("01_cohort", "table:cohort_flow"),
            ("05_primary", "table:adjusted_association_estimates"),
        ],
    )
    product_refs = json.loads(request.schema_json)["$defs"][
        "ProgressiveProductRef"
    ]["anyOf"]

    assert len(product_refs) == 1
    assert product_refs[0]["properties"]["producer_step_id"]["const"] == (
        "05_primary"
    )
    assert product_refs[0]["properties"]["product_id"]["const"] == (
        "table:adjusted_association_estimates"
    )


def test_current_table_one_step_requires_its_module_fields_in_schema() -> None:
    outline_step = ProgressiveOutlineStep(
        step_id="02_table_one",
        planned_analysis_role="auxiliary",
        module_id="table_one",
        objective="Describe baseline variables by the declared exposure groups.",
        depends_on=["01_cohort"],
        variable_names=["exposure_flag", "age_years", "sex_code"],
        literature_citation_keys=[],
        scientific_action_id=None,
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
        variable_names=["exposure_flag", "age_years", "sex_code"],
        scientific_action_ids=[],
        available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
    )
    step = json.loads(request.schema_json)["$defs"]["ProgressiveSkeletonStep"][
        "properties"
    ]

    assert step["table_one_group_by"] == {
        "type": "string",
        "enum": ["exposure_flag", "age_years", "sex_code"],
    }
    assert step["table_one_mode"]["type"] == "string"
    assert step["table_one_variables"]["minItems"] == 1


@pytest.mark.parametrize(
    "module_id",
    sorted(PROGRESSIVE_HOST_COMPILED_OUTPUTS),
)
def test_current_host_compiled_module_forbids_model_named_outputs(
    module_id: str,
) -> None:
    outline_step = ProgressiveOutlineStep(
        step_id=f"02_{module_id}",
        planned_analysis_role="auxiliary",
        module_id=module_id,
        objective="Materialize the registered module with host-owned product names.",
        depends_on=["01_cohort"],
        variable_names=["exposure_flag", "outcome_flag"],
        literature_citation_keys=[],
        scientific_action_id=None,
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
        variable_names=["exposure_flag", "outcome_flag"],
        scientific_action_ids=[],
        available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
    )
    outputs = json.loads(request.schema_json)["$defs"][
        "ProgressiveSkeletonStep"
    ]["properties"]["outputs"]

    assert outputs["minItems"] == 0
    assert outputs["maxItems"] == 0


def test_current_distribution_step_requires_non_null_contract_fields() -> None:
    outline_step = ProgressiveOutlineStep(
        step_id="03_distribution",
        planned_analysis_role="primary",
        module_id="exposure_outcome_distribution",
        objective="Estimate prevalence and absolute outcome risk by exposure.",
        depends_on=["01_cohort"],
        variable_names=["exposure_flag", "outcome_flag"],
        literature_citation_keys=[],
        scientific_action_id="descriptive.descriptive_summary",
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
        variable_names=["exposure_flag", "outcome_flag"],
        scientific_action_ids=["descriptive.descriptive_summary"],
        available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
    )
    step = json.loads(request.schema_json)["$defs"]["ProgressiveSkeletonStep"][
        "properties"
    ]

    required_non_null = {
        "primary_exposure",
        "outcome",
        "event_level_index",
        "reference_exposure_level_index",
        "comparison_exposure_level_index",
        "denominator_policy",
        "missing_exposure_policy",
        "missing_outcome_policy",
        "confidence_level",
    }
    assert all("anyOf" not in step[field] for field in required_non_null)


def test_current_adjusted_step_requires_model_contract_fields() -> None:
    outline_step = ProgressiveOutlineStep(
        step_id="05_primary",
        planned_analysis_role="primary",
        module_id="adjusted_association",
        objective="Estimate the prespecified adjusted association.",
        depends_on=["01_cohort"],
        variable_names=["exposure_flag", "outcome_flag", "age_years"],
        literature_citation_keys=[],
        scientific_action_id="association.adjusted_association",
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
        variable_names=["exposure_flag", "outcome_flag", "age_years"],
        scientific_action_ids=["association.adjusted_association"],
        available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
    )
    step = json.loads(request.schema_json)["$defs"]["ProgressiveSkeletonStep"][
        "properties"
    ]

    assert "anyOf" not in step["primary_exposure"]
    assert "anyOf" not in step["outcome"]
    assert "anyOf" not in step["outcome_type"]
    assert step["model_terms"]["minItems"] == 1


@pytest.mark.parametrize(
    "module_id",
    ["measurement_audit", "visualization", "report"],
)
def test_current_artifact_module_requires_an_explicit_output(module_id: str) -> None:
    outline_step = ProgressiveOutlineStep(
        step_id=f"02_{module_id}",
        planned_analysis_role="auxiliary",
        module_id=module_id,
        objective="Produce the declared governed artifact for downstream review.",
        depends_on=["01_cohort"],
        variable_names=["exposure_flag"],
        literature_citation_keys=[],
        scientific_action_id=None,
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
        variable_names=["exposure_flag"],
        scientific_action_ids=[],
        available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
    )
    step = json.loads(request.schema_json)["$defs"]["ProgressiveSkeletonStep"][
        "properties"
    ]

    assert step["outputs"]["minItems"] == 1


@pytest.mark.parametrize(
    ("module_id", "kind", "semantic_roles"),
    [
        ("visualization", "figure", {"figure"}),
        ("report", "report", {"report"}),
    ],
)
def test_current_artifact_schema_binds_product_kind_to_module(
    module_id: str,
    kind: str,
    semantic_roles: set[str],
) -> None:
    outline_step = ProgressiveOutlineStep(
        step_id=f"02_{module_id}",
        planned_analysis_role="auxiliary",
        module_id=module_id,
        objective="Produce the declared governed artifact for downstream review.",
        depends_on=["01_cohort"],
        variable_names=["exposure_flag"],
        literature_citation_keys=[],
        scientific_action_id=None,
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
        variable_names=["exposure_flag"],
        scientific_action_ids=[],
        available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
    )
    output = json.loads(request.schema_json)["$defs"]["ProgressiveOutputIntent"][
        "properties"
    ]

    assert output["product_id"]["pattern"] == (
        rf"^{kind}:[a-z][a-z0-9_]*$"
    )
    role = output["semantic_role"]
    assert set(role.get("enum") or [role.get("const")]) == semantic_roles


def test_compiler_materializes_host_owned_contracts_and_exact_wires() -> None:
    plan, receipt = compile_progressive_plan(
        skeleton=_skeleton(),
        context=_context(),
    )

    assert plan.analysis_type == "association_study"
    assert plan.cohort is not None
    assert plan.cohort.selection_mode == "all_input_rows"
    assert len(receipt.compiled_steps) == len(plan.steps) == 7
    assert len(receipt.analysis_plan_sha256) == 64

    by_id = {step.step_id: step for step in plan.steps}
    cohort = by_id["01_cohort"]
    assert cohort.expected_outputs == [
        "artifact:analysis_cohort",
        "table:cohort_flow",
    ]
    assert cohort.cohort_definition_spec.identity_column == "stay_id"

    table_one = by_id["02_table_one"].table_one_spec
    assert table_one is not None
    assert table_one.schema_version == "easyicu.table_one/2"
    assert table_one.p_values_required is False
    assert {item.test for item in table_one.variables} == {"none_descriptive_smd_only"}
    assert next(
        item for item in table_one.variables if item.name == "sex_code"
    ).levels == [
        "A",
        "B",
    ]

    distribution_step = by_id["03_distribution"]
    distribution = distribution_step.exposure_outcome_distribution_spec
    assert distribution is not None
    assert distribution_step.method == "descriptive"
    assert exposure_outcome_distribution_executor_owns_step(distribution_step)
    assert distribution.exposure_levels == [0, 1]
    assert distribution.outcome_positive_value == 1
    assert distribution.risk_difference_contrast.reference_exposure_level == 0
    assert distribution.risk_difference_contrast.comparison_exposure_level == 1

    primary = by_id["05_primary"]
    assert primary.scientific_action_id == "association.adjusted_association"
    assert primary.scientific_capability == "association_adjusted_v1"
    requirement = primary.model_requirements[0]
    assert requirement.method_family == "statsmodels_logit_mle"
    assert requirement.covariates == ["age_years", "sex_code"]
    assert requirement.exposure_levels == ["0", "1"]
    assert requirement.exposure_reference_level == "0"
    assert requirement.primary_contrast_level == "1"

    figure = by_id["07_figure"]
    assert figure.inputs == [
        "table:exposure_outcome_distribution",
        "table:adjusted_association_estimates",
    ]
    assert {item.input_key for item in figure.input_consumption_contracts} == {
        "table:exposure_outcome_distribution",
        "table:adjusted_association_estimates",
    }
    assert {item.mode for item in figure.input_consumption_contracts} == {"all_rows"}


def test_compiler_keeps_ordinal_linear_levels_out_of_treatment_contrasts() -> None:
    payload = json.loads(json.dumps(_payload()))
    primary = next(step for step in payload["steps"] if step["step_id"] == "05_primary")
    primary["raw_inputs"].append("severity_stage")
    primary["primary_exposure"] = "severity_stage"
    primary["model_terms"][0] = {
        "name": "severity_stage",
        "role": "exposure",
        "coding": "ordinal_linear",
        "reference_level_index": None,
    }
    context = _context().model_copy(
        update={
            "variables": [
                *_context().variables,
                ConceptDescriptor(
                    name="severity_stage",
                    role=VariableRole.INTERVENTION,
                    dtype="int64",
                    observed_domain={
                        "n_unique": 4,
                        "is_binary": False,
                        "levels": [0, 1, 2, 3],
                    },
                ),
            ]
        }
    )

    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=context,
    )

    requirement = next(
        step for step in plan.steps if step.step_id == "05_primary"
    ).model_requirements[0]
    exposure_term = next(
        term for term in requirement.model_terms if term.role == "exposure"
    )
    assert exposure_term.levels == ["0", "1", "2", "3"]
    assert exposure_term.transform == "declared_level_index"
    assert requirement.exposure_levels is None
    assert requirement.exposure_reference_level is None
    assert requirement.primary_contrast_level is None


def test_table_one_compiler_reports_known_missing_group_rows() -> None:
    context = _context()
    variables = [
        variable.model_copy(
            update={
                "missingness": MissingnessProfile(
                    fraction_missing=0.1,
                    n_missing=12,
                    n_total=120,
                    missingness_severity="medium",
                )
            }
        )
        if variable.name == "exposure_flag"
        else variable
        for variable in context.variables
    ]
    plan, _ = compile_progressive_plan(
        skeleton=_skeleton(),
        context=context.model_copy(update={"variables": variables}),
    )

    table_step = next(step for step in plan.steps if step.step_id == "02_table_one")
    assert table_step.table_one_spec is not None
    assert table_step.table_one_spec.missing_group_policy == "exclude_and_report"


def test_distribution_compiler_reports_known_missing_exposure_rows() -> None:
    context = _context()
    variables = [
        variable.model_copy(
            update={
                "missingness": MissingnessProfile(
                    fraction_missing=0.1,
                    n_missing=12,
                    n_total=120,
                    missingness_severity="medium",
                )
            }
        )
        if variable.name == "exposure_flag"
        else variable
        for variable in context.variables
    ]
    plan, _ = compile_progressive_plan(
        skeleton=_skeleton(),
        context=context.model_copy(update={"variables": variables}),
    )

    distribution_step = next(
        step for step in plan.steps if step.step_id == "03_distribution"
    )
    assert distribution_step.exposure_outcome_distribution_spec is not None
    assert (
        distribution_step.exposure_outcome_distribution_spec.missing_exposure_policy
        == "exclude_from_denominator"
    )


def test_report_orders_after_figure_without_reading_raster_as_data() -> None:
    payload = json.loads(json.dumps(_payload()))
    report = json.loads(json.dumps(payload["steps"][-1]))
    report.update(
        {
            "step_id": "08_report",
            "module_id": "report",
            "objective": "Assemble the source-bound scientific report.",
            "depends_on": ["07_figure"],
            "raw_inputs": ["age_years"],
            "product_inputs": [
                {
                    "producer_step_id": "05_primary",
                    "product_id": "table:adjusted_association_estimates",
                },
                {
                    "producer_step_id": "07_figure",
                    "product_id": "figure:primary_results",
                }
            ],
            "outputs": [
                {
                    "product_id": "report:analysis_results",
                    "semantic_role": "report",
                }
            ],
        }
    )
    payload["steps"].append(report)

    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=_context(),
    )
    compiled_report = plan.steps[-1]

    assert compiled_report.expected_outputs == ["report:analysis_results"]
    assert "figure:primary_results" not in compiled_report.inputs
    assert "age_years" not in compiled_report.inputs
    assert "table:adjusted_association_estimates" in compiled_report.inputs
    assert "artifact:analysis_cohort" in compiled_report.inputs
    assert compiled_report.method == "scientific_reporting"
    assert scientific_reporting_executor_owns_step(compiled_report)


def test_progressive_visualization_rejects_raw_cohort_inputs() -> None:
    payload = _payload()
    payload["steps"][-1]["raw_inputs"] = ["exposure_flag"]

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=_context(),
        )

    assert caught.value.reason_code == (
        "progressive_visualization_raw_input_forbidden"
    )
    assert caught.value.step_id == "07_figure"
    assert caught.value.path == "raw_inputs"


def test_progressive_visualization_rejects_untraceably_wide_source_bundle() -> None:
    payload = _payload()
    payload["steps"][-1]["product_inputs"] = [
        {
            "producer_step_id": "02_table_one",
            "product_id": "table:table_one",
        },
        {
            "producer_step_id": "03_distribution",
            "product_id": "table:exposure_outcome_distribution",
        },
        {
            "producer_step_id": "04_measurement",
            "product_id": "table:measurement_missingness",
        },
        {
            "producer_step_id": "04_measurement",
            "product_id": "table:measurement_process",
        },
        {
            "producer_step_id": "05_primary",
            "product_id": "table:adjusted_association_estimates",
        },
    ]

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=_context(),
        )

    assert caught.value.reason_code == (
        "progressive_visualization_source_budget_exceeded"
    )
    assert caught.value.step_id == "07_figure"
    assert caught.value.path == "product_inputs"
    assert caught.value.details["findings"][0]["source_product_count"] == 5


def test_progressive_visualization_rejects_analysis_cohort_artifact_source() -> None:
    payload = _payload()
    payload["steps"][-1]["product_inputs"] = [
        {
            "producer_step_id": "01_cohort",
            "product_id": "artifact:analysis_cohort",
        }
    ]

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=_context(),
        )

    assert caught.value.reason_code == (
        "progressive_visualization_source_kind_invalid"
    )
    assert caught.value.step_id == "07_figure"
    assert caught.value.path == "product_inputs"
    assert caught.value.details["findings"][0]["invalid_sources"] == [
        "artifact:analysis_cohort"
    ]


def test_measurement_compiler_closes_typed_observation_semantics_inputs() -> None:
    context = _context().model_copy(
        update={
            "variables": [
                *_context().variables,
                ConceptDescriptor(name="event_n", dtype="int64"),
                ConceptDescriptor(name="event_measured", dtype="int64"),
                ConceptDescriptor(
                    name="event_status",
                    dtype="int64",
                    observed_domain={
                        "n_unique": 2,
                        "is_binary": True,
                        "levels": [0, 1],
                    },
                    observation_semantics=ObservationSemantics(
                        kind="positive_only_event",
                        event_count_column="event_n",
                        measured_column="event_measured",
                        representative_column="event_status",
                    ),
                ),
            ]
        }
    )

    plan, _receipt = compile_progressive_plan(
        skeleton=_skeleton(),
        context=context,
    )

    measurement = next(step for step in plan.steps if step.step_id == "04_measurement")
    assert {"event_n", "event_measured", "event_status"} <= set(measurement.inputs)


def test_counts_only_compiler_emits_descriptive_table_and_distribution() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                inferred_analysis_family="descriptive_epidemiology",
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_family": "descriptive_epidemiology",
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                ),
            )
        }
    )
    payload = _payload()
    payload["analysis_type"] = "descriptive_epidemiology"
    payload["robustness_intents"] = []
    payload["steps"] = payload["steps"][:3]

    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=context,
    )
    bound = bind_context_dependence_authority(plan=plan, context=context)
    by_id = {step.step_id: step for step in bound.steps}

    table_one = by_id["02_table_one"].table_one_spec
    assert table_one is not None
    assert table_one.schema_version == "easyicu.table_one/2"
    assert table_one.p_values_required is False
    distribution = by_id["03_distribution"].exposure_outcome_distribution_spec
    assert distribution is not None
    assert distribution.schema_version == "easyicu.exposure_outcome_distribution/3"
    assert distribution.interval_method == "none_counts_only"
    assert distribution.confidence_level is None
    assert distribution.risk_difference_contrast is None


def test_descriptive_compiler_rejects_effect_robustness_before_plan_assembly() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                inferred_analysis_family="descriptive_epidemiology",
            )
        }
    )
    payload = _payload()
    payload["analysis_type"] = "descriptive_epidemiology"
    payload["steps"] = payload["steps"][:4]
    payload["steps"][2]["planned_analysis_role"] = "primary"

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=context,
        )

    assert (
        caught.value.reason_code
        == "progressive_descriptive_robustness_unavailable"
    )
    assert caught.value.path == "robustness_intents"


def test_progressive_foundation_rejects_non_replayable_robustness_intent() -> None:
    payload = _payload()
    payload["robustness_intents"] = [
        {
            "spec_id": "alternate_population",
            "axis": "cohort",
            "description": "Restrict the analysis to an alternate population.",
            "missing_strategy": "none",
            "complete_case_variables": [],
        }
    ]

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=_context(),
        )

    assert (
        caught.value.reason_code
        == "progressive_robustness_intent_not_replayable"
    )
    assert caught.value.path == "robustness_intents.alternate_population"


def test_complete_case_rejects_conditional_event_time_as_not_applicable() -> None:
    context = _context().model_copy(
        update={
            "variables": [
                *_context().variables,
                ConceptDescriptor(
                    name="event_time",
                    role=VariableRole.TIME,
                    dtype="float64",
                    observation_semantics=ObservationSemantics(
                        kind="conditional_event_time",
                        event_status_column="outcome_flag",
                        representative_column="event_time",
                        time_origin="cohort_entry",
                        time_unit="h",
                    ),
                ),
            ]
        }
    )
    payload = _payload()
    payload["robustness_intents"][0]["complete_case_variables"].append(
        "event_time"
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=context,
        )

    assert (
        caught.value.reason_code
        == "progressive_complete_case_includes_not_applicable_time"
    )
    assert caught.value.path.endswith("complete_case_variables")


def test_complete_case_roster_excludes_host_navigation_coordinates() -> None:
    from types import SimpleNamespace

    context = _context().model_copy(
        update={
            "variables": [
                ConceptDescriptor(
                    name="stay_id",
                    role=VariableRole.ID,
                    dtype="int64",
                ),
                *_context().variables,
            ],
            "materialized_inputs": SimpleNamespace(
                cohort=SimpleNamespace(
                    cohort_columns=[
                        "stay_id",
                        "exposure_flag",
                        "outcome_flag",
                        "age_years",
                        "sex_code",
                    ],
                    column_bindings={
                        "exposure_flag": object(),
                        "outcome_flag": object(),
                        "age_years": object(),
                        "sex_code": object(),
                    },
                )
            ),
        }
    )

    roster = _complete_case_variable_roster(
        context,
        ("stay_id", "exposure_flag", "outcome_flag"),
    )

    assert roster == ("exposure_flag", "outcome_flag")


def test_outline_requires_custom_owner_for_explicit_separate_product() -> None:
    payload = _payload()
    payload["steps"] = [
        step for step in payload["steps"] if step["module_id"] != "custom_analysis"
    ]
    outline = ProgressivePlanOutline.model_validate(_outline_payload(payload))

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=(
                "exposure_flag",
                "outcome_flag",
                "age_years",
                "sex_code",
            ),
            allowed_literature_citation_keys=(),
            required_custom_products=("table:prespecified_sensitivity",),
        )

    assert (
        caught.value.reason_code
        == "progressive_outline_separate_analysis_owner_missing"
    )
    assert caught.value.details["findings"][0]["required_products"] == [
        "table:prespecified_sensitivity"
    ]


def test_outline_rejects_categorical_distribution_for_continuous_exposure() -> None:
    payload = _outline_payload()
    distribution = next(
        step
        for step in payload["steps"]
        if step["module_id"] == "exposure_outcome_distribution"
    )
    distribution["variable_names"] = ["age_years", "outcome_flag"]
    outline = ProgressivePlanOutline.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=(
                "exposure_flag",
                "outcome_flag",
                "age_years",
                "sex_code",
            ),
            allowed_literature_citation_keys=(),
            closed_domain_variables=(
                "exposure_flag",
                "outcome_flag",
                "sex_code",
            ),
        )

    assert (
        caught.value.reason_code
        == "progressive_outline_distribution_domain_unavailable"
    )
    assert caught.value.details["step_id"] == distribution["step_id"]


def test_action_catalog_exposes_domain_semantics_to_planner() -> None:
    _action_ids, rows = _action_catalog(("association_study",))
    ordinal = next(
        row for row in rows if row["action_id"] == "association.ordinal_trend"
    )

    assert ">=3 ordered levels" in ordinal["purpose"]
    assert ordinal["name"]
    assert "notes" in ordinal


def test_outline_rejects_ordinal_action_for_binary_primary_exposure() -> None:
    payload = _outline_payload()
    custom = next(
        step for step in payload["steps"] if step["module_id"] == "custom_analysis"
    )
    custom["scientific_action_id"] = "association.ordinal_trend"
    outline = ProgressivePlanOutline.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=(
                "exposure_flag",
                "outcome_flag",
                "age_years",
                "sex_code",
            ),
            allowed_literature_citation_keys=(),
            ordered_domain_variables=(),
            primary_exposure="exposure_flag",
            target_outcome="outcome_flag",
        )

    assert caught.value.reason_code == (
        "progressive_outline_ordered_trend_domain_unsupported"
    )
    assert caught.value.details["step_id"] == custom["step_id"]


def test_outline_cannot_substitute_unrelated_closed_domains_for_primary_pair() -> None:
    payload = _outline_payload()
    distribution = next(
        step
        for step in payload["steps"]
        if step["module_id"] == "exposure_outcome_distribution"
    )
    distribution["variable_names"] = ["outcome_flag", "sex_code", "age_years"]
    outline = ProgressivePlanOutline.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=(
                "age_years",
                "exposure_flag",
                "outcome_flag",
                "sex_code",
            ),
            allowed_literature_citation_keys=(),
            closed_domain_variables=("outcome_flag", "sex_code"),
            primary_exposure="age_years",
            target_outcome="outcome_flag",
        )

    assert (
        caught.value.reason_code
        == "progressive_outline_distribution_domain_unavailable"
    )
    assert caught.value.details["step_id"] == distribution["step_id"]


def test_outline_rejects_secondary_custom_result_off_primary_lineage() -> None:
    payload = _outline_payload()
    custom = next(
        step for step in payload["steps"] if step["module_id"] == "custom_analysis"
    )
    custom["planned_analysis_role"] = "secondary"
    custom["depends_on"] = ["01_cohort"]
    outline = ProgressivePlanOutline.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=(
                "exposure_flag",
                "outcome_flag",
                "age_years",
                "sex_code",
            ),
            allowed_literature_citation_keys=(),
        )

    assert caught.value.reason_code == (
        "progressive_outline_secondary_custom_off_primary_lineage"
    )
    assert caught.value.details["findings"][0]["step_ids"] == [custom["step_id"]]


def test_outline_rejects_missing_method_layer_before_checkpoint() -> None:
    payload = _outline_payload()
    for step in payload["steps"]:
        step["literature_citation_keys"] = [
            key
            for key in step["literature_citation_keys"]
            if key != "strobe_2007"
        ]
    payload["steps"][0]["literature_citation_keys"] = ["record_2015"]
    outline = ProgressivePlanOutline.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=(
                "exposure_flag",
                "outcome_flag",
                "age_years",
                "sex_code",
            ),
            allowed_literature_citation_keys=(
                "strobe_2007",
                "record_2015",
                "durrleman_splines_1989",
            ),
            context_required_method_layers=("reporting_standard",),
        )

    assert caught.value.reason_code == "progressive_outline_method_layer_unbound"
    assert caught.value.details["findings"][0]["missing_method_layers"] == [
        "interpretation"
    ]


def test_outline_requires_functional_form_source_for_continuous_model_inputs() -> None:
    payload = _outline_payload()
    payload["steps"][0]["literature_citation_keys"] = ["strobe_2007"]
    outline = ProgressivePlanOutline.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=(
                "exposure_flag",
                "outcome_flag",
                "age_years",
                "sex_code",
            ),
            allowed_literature_citation_keys=(
                "strobe_2007",
                "durrleman_splines_1989",
            ),
            continuous_domain_variables=("age_years",),
            context_required_method_layers=("reporting_standard",),
        )

    assert caught.value.reason_code == "progressive_outline_method_layer_unbound"
    assert caught.value.details["findings"][0]["missing_method_layers"] == [
        "functional_form"
    ]


def test_outline_prompt_maps_method_sources_to_their_layers() -> None:
    prompt = ProgressivePlannerAgent.request_messages(
        _context(),
        allowed_literature_citation_keys=(
            "strobe_2007",
            "durrleman_splines_1989",
        ),
    )[-1].content

    assert "Host-known method layers for sealed citation keys" in prompt
    assert '"citation_key":"durrleman_splines_1989"' in prompt
    assert '"method_layers":["functional_form"]' in prompt


def test_retrieved_data_cards_expose_module_compatibility_without_level_values() -> None:
    cards = ProgressivePlannerAgent._retrieved_data_cards(
        _context(),
        ("exposure_flag", "age_years"),
    )
    by_name = {card["name"]: card for card in cards}

    assert by_name["exposure_flag"]["supports_closed_level_contrast"] is True
    assert by_name["exposure_flag"]["closed_domain_level_count"] == 2
    assert by_name["age_years"]["supports_closed_level_contrast"] is False
    assert by_name["age_years"]["closed_domain_level_count"] == 0
    assert "observed_domain" not in by_name["exposure_flag"]


def test_outline_rejects_repeated_host_compiled_singleton_module() -> None:
    outline = ProgressivePlanOutline.model_validate(
        _outline_with_repeated_robustness()
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=(
                "exposure_flag",
                "outcome_flag",
                "age_years",
                "sex_code",
            ),
            allowed_literature_citation_keys=(),
        )

    assert caught.value.reason_code == "progressive_outline_host_module_repeated"
    assert caught.value.details["path"] == "steps"
    assert caught.value.details["findings"] == [
        {
            "module_id": "robustness_replay",
            "step_ids": ["08_robustness_a", "09_robustness_b"],
            "host_products": [
                "table:robustness_matrix",
                "table:robustness_summary",
            ],
        }
    ]


def test_outline_requires_visualization_for_explicit_figure_output() -> None:
    payload = _outline_payload()
    payload["steps"] = [
        step for step in payload["steps"] if step["module_id"] != "visualization"
    ]
    outline = ProgressivePlanOutline.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=(
                "exposure_flag",
                "outcome_flag",
                "age_years",
                "sex_code",
            ),
            allowed_literature_citation_keys=(),
            required_visualization_step=True,
        )

    assert (
        caught.value.reason_code
        == "progressive_outline_visualization_owner_missing"
    )
    assert caught.value.details["findings"] == [
        {
            "required_module_id": "visualization",
            "source": "user_preferences.must_have_outputs",
        }
    ]


def test_outline_prompt_projects_explicit_separate_product_without_case_rules() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                must_have_outputs=(
                    "In a separate analysis step, emit "
                    "table:prespecified_sensitivity."
                )
            )
        }
    )

    prompt = ProgressivePlannerAgent.request_messages(context)[1].content

    assert "Host-resolved separate-analysis obligations" in prompt
    assert '"table:prespecified_sensitivity"' in prompt
    assert "custom_analysis outline step" in prompt


def test_outline_prompt_exposes_host_compiled_singleton_ownership() -> None:
    prompt = ProgressivePlannerAgent.request_messages(_context())[1].content

    assert "Host-compiled singleton module ownership" in prompt
    assert '"robustness_replay"' in prompt
    assert '"table:robustness_matrix"' in prompt
    assert "Each listed module may appear at most once" in prompt
    assert "all replayable robustness intents in one robustness_replay step" in prompt


def test_outline_binds_unique_runtime_product_owner_without_model_bookkeeping() -> None:
    outline = ProgressivePlanOutline(
        analysis_type="prediction_model",
        cohort_objective="Use the host-authorized analysis cohort.",
        steps=[
            ProgressiveOutlineStep(
                step_id="primary_model",
                planned_analysis_role="primary",
                module_id="custom_analysis",
                objective="Fit the prespecified static prediction model.",
                variable_names=["outcome_flag", "age_years"],
                scientific_action_id="prediction.discrimination_calibration",
            ),
            ProgressiveOutlineStep(
                step_id="calibration",
                planned_analysis_role="secondary",
                module_id="custom_analysis",
                objective="Assess validation-partition calibration.",
                variable_names=["outcome_flag"],
                scientific_action_id="prediction.calibration_metrics",
            ),
            ProgressiveOutlineStep(
                step_id="clinical_utility",
                planned_analysis_role="secondary",
                module_id="custom_analysis",
                objective="Assess validation-partition net benefit.",
                variable_names=["outcome_flag"],
                scientific_action_id="prediction.decision_curve",
            ),
        ],
        rationale="Separate scientific choices from host-owned product edges.",
    )

    bound = _bind_runtime_action_dependencies(outline)

    assert outline.steps[1].depends_on == []
    assert bound.steps[1].depends_on == ["primary_model"]
    assert bound.steps[2].depends_on == ["primary_model"]


def test_outline_prompt_projects_explicit_figure_obligation() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                must_have_outputs="Required outputs: one publication figure."
            )
        }
    )

    prompt = ProgressivePlannerAgent.request_messages(context)[1].content

    assert "Host-resolved presentation obligation" in prompt
    assert "Include at least one visualization outline step" in prompt
    assert "do not delegate the figure to a report step" in prompt


@pytest.mark.parametrize(
    ("module_id", "product_id", "semantic_role", "custom_method"),
    [
        (
            "custom_analysis",
            "table:duplicate_measurement_audit",
            "custom",
            "duplicate_measurement_audit",
        ),
        ("visualization", "figure:duplicate_presentation", "figure", None),
        ("report", "report:duplicate_presentation", "report", None),
    ],
)
def test_descriptive_compiler_rejects_nonstandard_or_duplicate_owners(
    module_id: str,
    product_id: str,
    semantic_role: str,
    custom_method: str | None,
) -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                inferred_analysis_family="descriptive_epidemiology",
            )
        }
    )
    payload = _payload()
    payload["analysis_type"] = "descriptive_epidemiology"
    payload["robustness_intents"] = []
    cohort_step = payload["steps"][0]
    step = payload["steps"][-1]
    step.update(
        {
            "step_id": "02_duplicate_presentation",
            "module_id": module_id,
            "objective": "Duplicate a presentation artifact already owned by the host.",
            "depends_on": ["01_cohort"],
            "raw_inputs": [],
            "product_inputs": [
                {
                    "producer_step_id": "01_cohort",
                    "product_id": "artifact:analysis_cohort",
                }
            ],
            "outputs": [
                {
                    "product_id": product_id,
                    "semantic_role": semantic_role,
                }
            ],
            "custom_method": custom_method,
        }
    )
    payload["steps"] = [cohort_step, step]

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=context,
        )

    assert caught.value.reason_code == "progressive_analysis_module_unavailable"
    assert caught.value.step_id == "02_duplicate_presentation"
    assert caught.value.path == "module_id"


def test_counts_only_primary_post_baseline_distribution_gets_typed_ceiling() -> None:
    base = _context()
    exposure = base.variable("exposure_flag")
    context = base.model_copy(
        update={
            "variables": [
                item.model_copy(
                    update={
                        "analysis_window": "icu_admission[0,24]h",
                        "analysis_window_role": "exposure_definition",
                    }
                )
                if item.name == exposure.name
                else item
                for item in base.variables
            ],
            "user_preferences": UserPreferences(
                inferred_analysis_family="descriptive_epidemiology",
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_family": "descriptive_epidemiology",
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                ),
            ),
        }
    )
    payload = _payload()
    payload["analysis_type"] = "descriptive_epidemiology"
    payload["robustness_intents"] = []
    payload["steps"] = payload["steps"][:3]
    payload["steps"][2]["planned_analysis_role"] = "primary"

    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=context,
    )
    bound = bind_context_dependence_authority(plan=plan, context=context)
    distribution_step = bound.steps[2]

    assert distribution_step.descriptive_claim is not None
    assert distribution_step.descriptive_claim.unresolved_limitations == (
        "post_baseline_exposure_opportunity_unresolved",
    )
    assert exposure_outcome_distribution_executor_owns_step(distribution_step)


def test_compiler_scopes_sealed_materialized_columns_for_cohort_validation() -> None:
    context = _context()
    context = context.model_copy(
        update={
            "variables": [
                ConceptDescriptor(
                    name="stay_id",
                    role=VariableRole.ID,
                    dtype="int64",
                    observed_domain={"n_unique": 120},
                ),
                *context.variables,
            ]
        }
    )
    payload = _payload()
    payload["cohort"] = {
        "name": "primary",
        "selection_mode": "predicate_filtered",
        "inclusion": [
            {
                "concept_id": "stay_id",
                "anchor": "icu_admission",
                "start_offset_hours": 0,
                "end_offset_hours": 24,
                "aggregation": "any",
                "op": "not_missing",
                "value": {"mode": "none"},
            }
        ],
        "exclusion": [],
    }
    prior_registry_answer = concept_id_exists("stay_id")

    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=context,
    )

    assert plan.cohort is not None
    assert plan.cohort.inclusion[0].concept_id == "stay_id"
    assert concept_id_exists("stay_id") is prior_registry_answer


def test_compiler_keeps_navigation_identity_out_of_executable_step_inputs() -> None:
    from types import SimpleNamespace

    context = _context().model_copy(
        update={
            "materialized_inputs": SimpleNamespace(
                cohort=SimpleNamespace(
                    cohort_columns=[
                        "stay_id",
                        "exposure_flag",
                        "outcome_flag",
                        "age_years",
                        "sex_code",
                    ],
                    column_bindings={
                        "exposure_flag": object(),
                        "outcome_flag": object(),
                        "age_years": object(),
                        "sex_code": object(),
                    },
                )
            )
        }
    )
    payload = _payload()
    for step in payload["steps"]:
        step["raw_inputs"] = ["stay_id", *step["raw_inputs"]]
    payload["robustness_intents"][0]["complete_case_variables"] = [
        "stay_id",
        *payload["robustness_intents"][0]["complete_case_variables"],
    ]

    authority = materialized_input_column_authority(context)
    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=context,
    )

    assert authority.sealed_columns[0] == "stay_id"
    assert authority.reserved_navigation_coordinates == ("stay_id",)
    assert "exposure_flag" in authority.executable_columns
    assert plan.steps[0].cohort_definition_spec is not None
    assert plan.steps[0].cohort_definition_spec.identity_column == "stay_id"
    assert all("stay_id" not in step.inputs for step in plan.steps)
    assert "exposure_flag" in plan.steps[1].inputs
    assert all(
        "stay_id" not in (spec.missing_override or {}).get("variables", ())
        for spec in plan.robustness_specs
    )


def test_custom_sensitivity_inherits_its_primary_model_inputs() -> None:
    payload = _payload()
    sensitivity = payload["steps"][5]
    sensitivity["raw_inputs"] = ["exposure_flag", "outcome_flag"]
    sensitivity["sensitivity_spec_ids"] = ["flexible_form"]

    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=_context(),
    )

    compiled = next(
        step for step in plan.steps if step.step_id == "06_sensitivity"
    )
    assert compiled.inputs[:4] == [
        "exposure_flag",
        "outcome_flag",
        "age_years",
        "sex_code",
    ]
    assert "artifact:analysis_cohort" in compiled.inputs
    assert "table:adjusted_association_estimates" in compiled.inputs
    assert compiled.scientific_capability == "association_freeform_v1"


def test_scientific_sensitivity_requires_closed_ids_and_a_binary_primary_parent() -> None:
    missing_ids = _payload()
    missing_ids["steps"][5]["sensitivity_spec_ids"] = []

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(missing_ids),
            context=_context(),
        )

    assert caught.value.reason_code == (
        "progressive_association_sensitivity_contract_invalid"
    )
    assert caught.value.step_id == "06_sensitivity"

    continuous_parent = _payload()
    continuous_parent["steps"][4]["outcome_type"] = "continuous"

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(continuous_parent),
            context=_context(),
        )

    assert caught.value.reason_code == (
        "progressive_association_sensitivity_parent_invalid"
    )
    assert caught.value.step_id == "06_sensitivity"


def test_compiler_reports_identical_distribution_contrast_at_its_owner() -> None:
    payload = _payload()
    payload["steps"][2]["comparison_exposure_level_index"] = 0
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == (
        "progressive_distribution_contrast_not_distinct"
    )
    assert caught.value.step_id == "03_distribution"
    assert caught.value.step_index == 2
    assert caught.value.path == "comparison_exposure_level_index"


def test_compiler_contains_distribution_contract_validation() -> None:
    payload = _payload()
    payload["steps"][2]["confidence_level"] = 0.5
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == "progressive_distribution_spec_invalid"
    assert caught.value.step_id == "03_distribution"
    assert caught.value.step_index == 2
    assert caught.value.path == "confidence_level"


def test_compiler_wires_product_reference_to_its_unique_host_owner() -> None:
    payload = _payload()
    payload["steps"][-1]["product_inputs"][1]["producer_step_id"] = "02_table_one"
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    plan, _receipt = compile_progressive_plan(
        skeleton=skeleton,
        context=_context(),
    )

    figure = next(step for step in plan.steps if step.step_id == "07_figure")
    assert "table:adjusted_association_estimates" in figure.inputs


def test_compiler_keeps_audit_dependencies_out_of_cohort_only_runtime_inputs() -> (
    None
):
    payload = _payload()
    payload["steps"][2]["product_inputs"] = [
        {
            "producer_step_id": "02_table_one",
            "product_id": "table:table_one",
        }
    ]
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    plan, _receipt = compile_progressive_plan(
        skeleton=skeleton,
        context=_context(),
    )

    distribution = next(step for step in plan.steps if step.step_id == "03_distribution")
    assert "artifact:analysis_cohort" in distribution.inputs
    assert "table:table_one" not in distribution.inputs
    assert exposure_outcome_distribution_executor_owns_step(distribution)


def test_cohort_only_module_still_rejects_an_unknown_product_reference() -> None:
    payload = _payload()
    payload["steps"][2]["product_inputs"] = [
        {
            "producer_step_id": "02_table_one",
            "product_id": "table:not_registered",
        }
    ]
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == "progressive_product_reference_mismatch"
    assert caught.value.step_id == "03_distribution"
    assert caught.value.path == "product_inputs"


def test_compiler_refuses_product_reference_without_a_host_owner() -> None:
    payload = _payload()
    payload["steps"][-1]["product_inputs"][1] = {
        "producer_step_id": "05_primary",
        "product_id": "table:unregistered_result",
    }
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == "progressive_product_reference_mismatch"
    assert caught.value.step_id == "07_figure"
    assert caught.value.path == "product_inputs"


def test_compiler_drops_group_by_from_table_one_rows() -> None:
    payload = _payload()
    payload["steps"][1]["table_one_variables"].insert(
        0,
        {"name": "exposure_flag", "summary": "count_percent"},
    )
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    plan, _receipt = compile_progressive_plan(
        skeleton=skeleton,
        context=_context(),
    )

    table_one = next(step for step in plan.steps if step.step_id == "02_table_one")
    assert table_one.table_one_spec is not None
    assert table_one.table_one_spec.group_by == "exposure_flag"
    assert [item.name for item in table_one.table_one_spec.variables] == [
        "age_years",
        "sex_code",
    ]


def test_compiler_contains_table_one_validation_errors() -> None:
    payload = _payload()
    payload["steps"][1]["table_one_variables"].append(
        {"name": "age_years", "summary": "mean_sd"}
    )
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == "progressive_table_one_contract_invalid"
    assert caught.value.step_id == "02_table_one"
    assert caught.value.path == "table_one_variables"


def test_compiler_materializes_the_locked_robustness_replay_bundle() -> None:
    payload = _payload()
    replay = payload["steps"][5]
    replay.update(
        {
            "step_id": "06_robustness",
            "module_id": "robustness_replay",
            "objective": (
                "Replay the already locked robustness grid without changing the estimand."
            ),
            "product_inputs": [
                {
                    "producer_step_id": "05_primary",
                    "product_id": "table:adjusted_association_estimates",
                }
            ],
            "outputs": [],
            "scientific_action_id": None,
            "custom_method": None,
            "sensitivity_spec_ids": ["complete_case"],
        }
    )
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    plan, _receipt = compile_progressive_plan(
        skeleton=skeleton,
        context=_context(),
    )

    step = next(item for item in plan.steps if item.step_id == "06_robustness")
    assert step.expected_outputs == [
        "table:robustness_matrix",
        "table:robustness_summary",
    ]
    assert step.method == "robustness_sensitivity"
    assert step.robustness_replay_spec is not None
    assert robustness_replay_spec_is_emittable(step)


def test_complete_case_replay_covers_every_primary_model_field() -> None:
    payload = _payload()
    payload["robustness_intents"][0]["complete_case_variables"] = [
        "exposure_flag",
        "outcome_flag",
    ]

    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=_context(),
    )

    assert plan.robustness_specs[0].missing_override == {
        "strategy": "complete_case",
        "variables": [
            "exposure_flag",
            "outcome_flag",
            "age_years",
            "sex_code",
        ],
        "audit_flags": None,
    }


def test_compiler_contains_duplicate_robustness_output_contract() -> None:
    payload = _payload()
    replay = payload["steps"][5]
    replay.update(
        {
            "step_id": "06_robustness",
            "module_id": "robustness_replay",
            "objective": "Replay the locked robustness grid without changing it.",
            "product_inputs": [
                {
                    "producer_step_id": "05_primary",
                    "product_id": "table:adjusted_association_estimates",
                }
            ],
            "outputs": [
                {
                    "product_id": "table:sensitivity_comparison",
                    "semantic_role": "robustness_matrix",
                }
            ],
            "scientific_action_id": None,
            "custom_method": None,
            "sensitivity_spec_ids": ["complete_case"],
        }
    )
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == "progressive_robustness_replay_spec_invalid"
    assert caught.value.step_id == "06_robustness"
    assert caught.value.path == "outputs"
    assert "one answer promised twice" in str(caught.value)


def test_compiler_contains_duplicate_measurement_output_contract() -> None:
    payload = _payload()
    payload["steps"][3]["outputs"].append(
        {
            "product_id": "table:measurement_missingness_alias",
            "semantic_role": "measurement_missingness",
        }
    )
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == "progressive_measurement_audit_spec_invalid"
    assert caught.value.step_id == "04_measurement"
    assert caught.value.path == "outputs"
    assert "one answer promised twice" in str(caught.value)


def test_compiler_reports_attributable_unknown_variable() -> None:
    payload = _payload()
    payload["steps"][4]["model_terms"][1]["name"] = "invented_covariate"
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.owner == "easyicu.planning.progressive_compiler_v1"
    assert caught.value.reason_code == "progressive_unknown_variable"
    assert caught.value.step_id == "05_primary"
    assert caught.value.step_index == 4
    assert caught.value.path == "model_terms"


def test_compiler_reports_outcome_covariate_at_the_model_owner() -> None:
    payload = _payload()
    payload["steps"][4]["model_terms"].append(
        {
            "name": "outcome_flag",
            "role": "covariate",
            "coding": "binary",
            "reference_level_index": 0,
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=_context(),
        )

    assert caught.value.reason_code == "progressive_model_outcome_is_covariate"
    assert caught.value.step_id == "05_primary"
    assert caught.value.step_index == 4
    assert caught.value.path == "model_terms"


def test_compiler_rejects_navigation_coordinate_as_a_model_term() -> None:
    from types import SimpleNamespace

    context = _context().model_copy(
        update={
            "materialized_inputs": SimpleNamespace(
                cohort=SimpleNamespace(
                    cohort_columns=[
                        "stay_id",
                        "exposure_flag",
                        "outcome_flag",
                        "age_years",
                        "sex_code",
                    ],
                    column_bindings={
                        "exposure_flag": object(),
                        "outcome_flag": object(),
                        "age_years": object(),
                        "sex_code": object(),
                    },
                )
            )
        }
    )
    payload = _payload()
    primary = payload["steps"][4]
    primary["raw_inputs"].append("stay_id")
    primary["model_terms"].append(
        {
            "name": "stay_id",
            "role": "covariate",
            "coding": "continuous",
            "reference_level_index": None,
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=context,
        )

    assert caught.value.reason_code == (
        "progressive_adjusted_model_uses_navigation_coordinate"
    )
    assert caught.value.step_id == "05_primary"
    assert caught.value.path == "model_terms[3]"


def test_compiler_coalesces_repeated_source_without_losing_design_intent() -> None:
    payload = _payload()
    payload["steps"][4]["literature_bindings"] = [
        {
            "citation_key": "topic_protocol",
            "design_elements": ["adjustment"],
            "application": "Use the declared adjustment set for the primary model.",
            "divergence": None,
        },
        {
            "citation_key": "topic_protocol",
            "design_elements": ["reporting"],
            "application": "Report the adjusted estimate with its uncertainty.",
            "divergence": "Do not adopt the source population restriction.",
        },
    ]
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    plan, _receipt = compile_progressive_plan(
        skeleton=skeleton,
        context=_context(),
        allowed_literature_citation_keys=["topic_protocol"],
    )

    primary = next(step for step in plan.steps if step.step_id == "05_primary")
    assert primary.literature_citation_keys == ["topic_protocol"]
    assert len(primary.literature_design_bindings) == 1
    binding = primary.literature_design_bindings[0]
    assert binding.design_elements == ["adjustment", "reporting"]
    assert binding.application == (
        "Use the declared adjustment set for the primary model.\n"
        "Report the adjusted estimate with its uncertainty."
    )
    assert binding.divergence == "Do not adopt the source population restriction."


def test_compiler_materializes_one_host_sealed_reporting_standard() -> None:
    plan, _receipt = compile_progressive_plan(
        skeleton=_skeleton(),
        context=_context(),
        allowed_literature_citation_keys=["strobe_2007", "record_2015"],
        host_reporting_method_source_keys=["strobe_2007"],
    )

    first_scientific = next(
        step
        for step in plan.steps
        if step.planned_analysis_role in {"primary", "secondary", "sensitivity"}
    )
    assert first_scientific.literature_citation_keys == ["strobe_2007"]
    assert [
        binding.model_dump(mode="json")
        for binding in first_scientific.literature_design_bindings
    ] == [
        {
            "citation_key": "strobe_2007",
            "design_elements": ["reporting"],
            "application": (
                "Apply the host-sealed article reporting standard to this "
                "study's methods and results."
            ),
            "divergence": None,
        }
    ]


def test_compiler_host_binds_interpretation_to_the_model_step() -> None:
    plan, _receipt = compile_progressive_plan(
        skeleton=_skeleton(),
        context=_context(),
        allowed_literature_citation_keys=["strobe_2007", "record_2015"],
        host_reporting_method_source_keys=["strobe_2007"],
    )

    primary = next(step for step in plan.steps if step.step_id == "05_primary")
    assert primary.literature_citation_keys == ["strobe_2007"]
    assert [
        binding.model_dump(mode="json")
        for binding in primary.literature_design_bindings
    ] == [
        {
            "citation_key": "strobe_2007",
            "design_elements": ["outcome"],
            "application": (
                "Report an absolute outcome measure alongside each model ratio "
                "estimate so interpretation is not ratio-only."
            ),
            "divergence": None,
        }
    ]


def test_compiler_host_binds_unique_missing_data_card_to_its_owner() -> None:
    plan, _receipt = compile_progressive_plan(
        skeleton=_skeleton(),
        context=_context(),
        allowed_literature_citation_keys=["sterne_missing_data_2009"],
    )

    measurement = next(step for step in plan.steps if step.step_id == "04_measurement")
    assert measurement.literature_citation_keys == [
        "sterne_missing_data_2009"
    ]
    assert [
        binding.model_dump(mode="json")
        for binding in measurement.literature_design_bindings
    ] == [
        {
            "citation_key": "sterne_missing_data_2009",
            "design_elements": ["missing_data", "robustness"],
            "application": (
                "Apply the host-compiled run-bound method obligation: Report "
                "the amount and pattern of missingness per variable, and state "
                "the assumption the chosen handling makes. Complete-case "
                "analysis is defensible when missingness is negligible or "
                "plausibly unrelated to the outcome given the covariates; say "
                "which applies. In routinely collected data, whether a "
                "measurement exists is itself informative and should be "
                "examined, not only imputed."
            ),
            "divergence": None,
        }
    ]


def test_compiler_closes_interpretation_from_one_model_selected_source() -> None:
    payload = _payload()
    payload["steps"][4]["literature_bindings"] = [
        {
            "citation_key": "strobe_2007",
            "design_elements": ["reporting"],
            "application": "Report the prespecified adjusted association.",
            "divergence": None,
        }
    ]
    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=_context(),
        allowed_literature_citation_keys=["strobe_2007"],
    )

    primary = next(step for step in plan.steps if step.step_id == "05_primary")
    assert primary.literature_citation_keys == ["strobe_2007"]
    assert primary.literature_design_bindings[0].design_elements == [
        "reporting",
        "outcome",
    ]
    assert primary.literature_design_bindings[0].application == (
        "Report the prespecified adjusted association.\n"
        "Report an absolute outcome measure alongside each model ratio estimate "
        "so interpretation is not ratio-only."
    )


def test_compiler_does_not_guess_between_multiple_reporting_standards() -> None:
    plan, _receipt = compile_progressive_plan(
        skeleton=_skeleton(),
        context=_context(),
        allowed_literature_citation_keys=["strobe_2007", "record_2015"],
        host_reporting_method_source_keys=["strobe_2007", "record_2015"],
    )

    assert all(not step.literature_citation_keys for step in plan.steps)


def test_compiler_refuses_host_reporting_source_outside_run_roster() -> None:
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=_skeleton(),
            context=_context(),
            allowed_literature_citation_keys=["record_2015"],
            host_reporting_method_source_keys=["strobe_2007"],
        )

    assert caught.value.reason_code == "progressive_host_reporting_source_unavailable"
    assert caught.value.step_id == "03_distribution"
    assert caught.value.path == "host_reporting_method_source_keys"


def test_compiler_refuses_lossy_repeated_source_coalescing() -> None:
    payload = _payload()
    payload["steps"][4]["literature_bindings"] = [
        {
            "citation_key": "topic_protocol",
            "design_elements": ["adjustment"],
            "application": "A" * 700,
            "divergence": None,
        },
        {
            "citation_key": "topic_protocol",
            "design_elements": ["reporting"],
            "application": "B" * 700,
            "divergence": None,
        },
    ]
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=skeleton,
            context=_context(),
            allowed_literature_citation_keys=["topic_protocol"],
        )

    assert caught.value.reason_code == "progressive_literature_merge_overflow"
    assert caught.value.step_id == "05_primary"
    assert caught.value.path == "literature_bindings.application"


def test_suffix_revision_cannot_change_compiled_prefix() -> None:
    skeleton = _skeleton()
    _plan, receipt = compile_progressive_plan(skeleton=skeleton, context=_context())
    revised = _payload()
    revised["steps"][1]["objective"] = "Rewrite an already compiled prefix step."
    revised_skeleton = ProgressivePlanSkeleton.model_validate(revised)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        assert_immutable_prefix(
            prior_receipt=receipt,
            revised_skeleton=revised_skeleton,
            locked_step_count=5,
        )

    assert caught.value.reason_code == "progressive_locked_prefix_changed"
    assert caught.value.step_id == "02_table_one"
    assert caught.value.step_index == 1


def test_cross_family_action_is_rejected_before_analysis_plan_acceptance() -> None:
    payload = _payload()
    payload["steps"][4]["scientific_action_id"] = "descriptive.descriptive_summary"
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == "progressive_scientific_action_invalid"
    assert caught.value.step_id == "05_primary"
    assert caught.value.path == "scientific_action_id"


def test_preflight_batches_independent_suffix_findings() -> None:
    payload = _payload()
    duplicate = json.loads(json.dumps(payload["steps"][3]))
    duplicate["step_id"] = "04b_measurement_detail"
    duplicate["depends_on"] = ["04_measurement"]
    payload["steps"].insert(4, duplicate)
    payload["steps"][5]["scientific_action_id"] = "descriptive.descriptive_summary"
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == "progressive_compile_batch_invalid"
    assert caught.value.step_id == "04b_measurement_detail"
    findings = caught.value.details["findings"]
    assert {item["reason_code"] for item in findings} == {
        "progressive_product_has_multiple_owners",
        "progressive_scientific_action_invalid",
    }


def test_preflight_preserves_specific_output_finding() -> None:
    payload = _payload()
    payload["steps"][3]["outputs"][0]["semantic_role"] = "figure"
    skeleton = ProgressivePlanSkeleton.model_validate(payload)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())

    assert caught.value.reason_code == "progressive_output_role_mismatch"
    assert caught.value.step_id == "04_measurement"
    assert caught.value.path == "outputs"


def test_run_bound_schema_closes_runtime_rosters_under_twelve_kib() -> None:
    request = progressive_structured_output_request(
        analysis_types=["association_study"],
        variable_names=["exposure_flag", "outcome_flag", "age_years"],
        cohort_concept_ids=["exposure_concept", "outcome_concept"],
        scientific_action_ids=["association.adjusted_association"],
        allowed_literature_citation_keys=["strobe_observational_reporting"],
    )
    schema = json.loads(request.schema_json)
    encoded = request.canonical_payload_json

    assert len(encoded.encode("utf-8")) < 12_000
    assert schema["properties"]["analysis_type"]["enum"] == ["association_study"]
    branches = schema["$defs"]["ProgressiveSkeletonStep"]["anyOf"]
    standard = next(
        branch
        for branch in branches
        if "enum" in branch["properties"]["module_id"]
    )
    custom = next(
        branch
        for branch in branches
        if branch["properties"]["module_id"].get("const") == "custom_analysis"
    )
    step = standard["properties"]
    assert "custom_analysis" not in step["module_id"]["enum"]
    assert step["custom_method"] == {"type": "null"}
    assert custom["properties"]["custom_method"]["type"] == "string"
    assert custom["properties"]["outputs"]["items"]["properties"][
        "semantic_role"
    ]["enum"] == ["scientific_sensitivity", "custom"]
    assert "table_one_variables" not in custom["properties"]
    assert step["raw_inputs"]["items"]["enum"] == [
        "exposure_flag",
        "outcome_flag",
        "age_years",
    ]
    assert step["scientific_action_id"]["anyOf"][0]["enum"] == [
        "association.adjusted_association"
    ]
    predicate = schema["$defs"]["ProgressiveCohortPredicate"]["properties"]
    assert predicate["concept_id"]["enum"] == [
        "exposure_concept",
        "outcome_concept",
    ]


def test_question_retrieval_keeps_association_when_notes_request_audits() -> None:
    context = _context().model_copy(
        update={
            "notes": (
                "Require missingness, observation-process, and component "
                "completeness audits before reporting."
            )
        }
    )

    candidates = candidate_analysis_types(context)

    assert candidates[0] == "association_study"


def test_agent_materializes_one_step_at_a_time_with_strict_transport() -> None:
    responses = [
        _outline_payload(),
        _foundation_payload(),
        *_materialization_payloads(),
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(_context())

    assert len(plan.steps) == 7
    assert plan.design_selection is not None
    assert plan.design_selection.selected.design_id == "selected_primary_design"
    assert plan.design_selection.claim_ceiling == "analysis_only"
    assert len(llm.calls) == 9
    requests = [call[1]["structured_output"] for call in llm.calls]
    assert requests[0].name == "easyicu_progressive_plan_outline_v1"
    assert requests[1].name == "easyicu_progressive_plan_foundation_v1"
    assert {request.name for request in requests[2:]} == {
        "easyicu_progressive_step_materialization_v1"
    }
    first_schema = requests[0].schema_json
    assert "raw_inputs" not in first_schema
    assert "product_inputs" not in first_schema
    assert "model_terms" not in first_schema
    foundation_prompt = llm.calls[1][0][-1].content
    assert "PROGRESSIVE PLAN-FOUNDATION AUTHORITY" in foundation_prompt
    assert "Do not return executable step fields" in foundation_prompt
    first_step_prompt = llm.calls[2][0][-1].content
    assert "Current outline step and host digest" in first_step_prompt
    assert "Do not return or rewrite any prefix or future step" in first_step_prompt
    assert agent.last_prompt_metrics["compile_revision_count"] == 0
    assert agent.last_prompt_metrics["step_materialization_count"] == 7
    assert agent.last_prompt_metrics["full_revision_count"] == 0
    assert agent.last_compile_receipt is not None
    assert agent.last_outline is not None
    assert agent.last_foundation is not None
    assert len(agent.last_materializations) == 7
    assert agent.last_skeleton is not None


def test_outline_authority_failure_is_retried_before_foundation() -> None:
    invalid_outline = _outline_payload()
    invalid_outline["steps"][0]["variable_names"].append("not_retrieved")
    responses = [
        invalid_outline,
        _outline_payload(),
        _foundation_payload(),
        *_materialization_payloads(),
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True

    plan = ProgressivePlannerAgent(llm).run(_context())

    assert len(plan.steps) == 7
    assert len(llm.calls) == 10
    assert llm.calls[0][1]["structured_output"].name == (
        "easyicu_progressive_plan_outline_v1"
    )
    assert llm.calls[1][1]["structured_output"].name == (
        "easyicu_progressive_plan_outline_v1"
    )
    assert "progressive_outline_variable_unavailable" in (
        llm.calls[1][0][-1].content
    )


def test_repeated_singleton_outline_is_retried_before_foundation() -> None:
    responses = [
        _outline_with_repeated_robustness(),
        _outline_payload(),
        _foundation_payload(),
        *_materialization_payloads(),
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True

    plan = ProgressivePlannerAgent(llm).run(_context())

    assert len(plan.steps) == 7
    assert len(llm.calls) == 10
    assert "progressive_outline_host_module_repeated" in (
        llm.calls[1][0][-1].content
    )
    assert "Host-compiled singleton module ownership" in (
        llm.calls[0][0][-1].content
    )


def test_missing_visualization_outline_is_retried_before_foundation() -> None:
    invalid_outline = _outline_payload()
    invalid_outline["steps"] = [
        step
        for step in invalid_outline["steps"]
        if step["module_id"] != "visualization"
    ]
    responses = [
        invalid_outline,
        _outline_payload(),
        _foundation_payload(),
        *_materialization_payloads(),
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                must_have_outputs="Required outputs: one publication figure."
            )
        }
    )

    plan = ProgressivePlannerAgent(llm).run(context)

    assert len(plan.steps) == 7
    assert len(llm.calls) == 10
    assert "progressive_outline_visualization_owner_missing" in (
        llm.calls[1][0][-1].content
    )
    assert "Host-resolved presentation obligation" in (
        llm.calls[0][0][-1].content
    )


def test_outline_uses_fourth_attempt_after_three_distinct_contract_repairs() -> None:
    missing_figure = _outline_payload()
    missing_figure["steps"] = [
        step
        for step in missing_figure["steps"]
        if step["module_id"] != "visualization"
    ]
    wrong_action = _outline_payload()
    primary = next(
        step for step in wrong_action["steps"] if step["step_id"] == "05_primary"
    )
    primary["scientific_action_id"] = "descriptive.missingness_audit"
    invalid_schema = _outline_payload()
    invalid_schema["steps"][0]["module_id"] = "not_a_registered_module"
    responses = [
        missing_figure,
        wrong_action,
        invalid_schema,
        _outline_payload(),
        _foundation_payload(),
        *_materialization_payloads(),
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                must_have_outputs="Required outputs: one publication figure."
            )
        }
    )

    plan = ProgressivePlannerAgent(llm).run(context)

    assert len(plan.steps) == 7
    assert len(llm.calls) == 12
    assert "progressive_outline_visualization_owner_missing" in (
        llm.calls[1][0][-1].content
    )
    assert "progressive_outline_action_unavailable" in (
        llm.calls[2][0][-1].content
    )


def test_agent_rejects_invalid_foundation_before_any_step_provider_call() -> None:
    invalid_foundation = _foundation_payload()
    invalid_foundation["foundation"]["cohort"] = {
        "name": "primary",
        "selection_mode": "predicate_filtered",
        "inclusion": [
            {
                "concept_id": "not_in_the_sealed_context",
                "anchor": "icu_admission",
                "start_offset_hours": 0,
                "end_offset_hours": 24,
                "aggregation": "any",
                "op": "not_missing",
                "value": {"mode": "none"},
            }
        ],
        "exclusion": [],
    }
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(invalid_foundation),
            json.dumps(invalid_foundation),
        ]
    )
    llm.supports_strict_json_schema = True
    checkpoints = []
    agent = ProgressivePlannerAgent(llm)

    with pytest.raises(StructuredResponseFailure) as caught:
        agent.run(_context(), checkpoint_callback=checkpoints.append)

    assert caught.value.__cause__ is not None
    assert getattr(caught.value.__cause__, "reason_code", None) == (
        "progressive_foundation_cohort_invalid"
    )
    assert getattr(caught.value.__cause__, "path", None) == "cohort"
    assert len(llm.calls) == 3
    assert [item.stage for item in checkpoints] == ["outline"]
    assert agent.last_foundation is None


def test_agent_repairs_missing_robustness_intent_before_step_materialization() -> None:
    source = _payload()
    source["steps"][5].update(
        {
            "step_id": "06_robustness",
            "module_id": "robustness_replay",
            "objective": "Replay the locked robustness grid without changing it.",
            "product_inputs": [
                {
                    "producer_step_id": "05_primary",
                    "product_id": "table:adjusted_association_estimates",
                }
            ],
            "outputs": [],
            "scientific_action_id": None,
            "custom_method": None,
            "sensitivity_spec_ids": ["complete_case"],
        }
    )
    invalid_foundation = _foundation_payload(source)
    invalid_foundation["foundation"]["robustness_intents"] = []
    responses = [
        _outline_payload(source),
        invalid_foundation,
        _foundation_payload(source),
        *_materialization_payloads(source),
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    checkpoints = []
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(_context(), checkpoint_callback=checkpoints.append)

    assert len(plan.steps) == 7
    assert len(llm.calls) == 10
    assert [item.stage for item in checkpoints[:2]] == ["outline", "foundation"]
    assert "progressive_foundation_robustness_intent_missing" in (
        llm.calls[2][0][-1].content
    )
    assert llm.calls[2][1]["structured_output"].name == (
        "easyicu_progressive_plan_foundation_v1"
    )


def test_agent_host_compiles_caller_bound_all_input_cohort() -> None:
    invalid_foundation = _foundation_payload()
    invalid_foundation["foundation"]["cohort"] = {
        "name": "model_reinterpreted_cohort",
        "selection_mode": "predicate_filtered",
        "inclusion": [
            {
                "concept_id": "age_years",
                "anchor": "ICU admission",
                "start_offset_hours": 0,
                "end_offset_hours": 24,
                "aggregation": "first",
                "op": ">=",
                "value": {"mode": "none"},
            }
        ],
        "exclusion": [],
    }
    responses = [
        _outline_payload(),
        invalid_foundation,
        *_materialization_payloads(),
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(
        _context(),
        required_primary_cohort_selection_mode="all_input_rows",
    )

    assert plan.cohort.name == "synthetic"
    assert plan.cohort.selection_mode == "all_input_rows"
    assert plan.cohort.inclusion == ()
    assert plan.cohort.exclusion == ()
    assert agent.last_prompt_metrics["foundation_cohort_owner"] == (
        "host_required_primary_cohort"
    )
    foundation_request = llm.calls[1][1]["structured_output"]
    foundation_schema = json.loads(foundation_request.schema_json)
    cohort = foundation_schema["$defs"]["ProgressiveCohortIntent"]["properties"]
    assert cohort["selection_mode"]["const"] == "all_input_rows"
    assert cohort["inclusion"]["maxItems"] == 0


def test_agent_emits_append_only_validated_prefix_checkpoints() -> None:
    responses = [
        _outline_payload(),
        _foundation_payload(),
        *_materialization_payloads(),
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []

    agent.run(_context(), checkpoint_callback=checkpoints.append)

    assert [item.stage for item in checkpoints] == [
        "outline",
        "foundation",
        *(["step"] * 7),
    ]
    assert [item.sequence for item in checkpoints] == list(range(9))
    assert [len(item.materializations) for item in checkpoints] == [
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert checkpoints[0].previous_checkpoint_sha256 is None
    assert all(
        current.previous_checkpoint_sha256 == previous.checkpoint_sha256
        for previous, current in zip(
            checkpoints[:-1], checkpoints[1:], strict=True
        )
    )


def test_agent_resumes_only_the_unmaterialized_suffix() -> None:
    materializations = _materialization_payloads()
    dependency_context = {
        "cohort_file_sha256": "b" * 64,
        "llm_signature": "codex:gpt-test",
        "prompt_version": "test-v1",
    }
    source_llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in materializations],
        ]
    )
    source_llm.supports_strict_json_schema = True
    source_agent = ProgressivePlannerAgent(source_llm)
    source_checkpoints = []
    source_agent.run(
        _context(),
        checkpoint_callback=source_checkpoints.append,
        resume_dependency_context=dependency_context,
    )
    resume_checkpoint = source_checkpoints[4]

    resumed_llm = ScriptedMockLLMClient(
        [json.dumps(item) for item in materializations[3:]]
    )
    resumed_llm.supports_strict_json_schema = True
    resumed_agent = ProgressivePlannerAgent(resumed_llm)
    resumed_checkpoints = []

    plan = resumed_agent.run(
        _context(),
        checkpoint_callback=resumed_checkpoints.append,
        resume_checkpoint=resume_checkpoint,
        resume_dependency_context=dependency_context,
    )

    assert len(plan.steps) == 7
    assert len(resumed_llm.calls) == 4
    assert resumed_agent.last_resume_validated is True
    assert [item.sequence for item in resumed_checkpoints] == [5, 6, 7, 8]
    assert (
        resumed_checkpoints[0].previous_checkpoint_sha256
        == resume_checkpoint.checkpoint_sha256
    )
    assert resumed_agent.last_prompt_metrics[
        "resume_reused_materialization_count"
    ] == 3
    assert resumed_agent.last_prompt_metrics[
        "current_run_step_materialization_count"
    ] == 4


def test_agent_rejects_resume_authority_drift_before_provider_call() -> None:
    dependency_context = {
        "cohort_file_sha256": "b" * 64,
        "llm_signature": "codex:gpt-test",
        "prompt_version": "test-v1",
    }
    source_llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    source_llm.supports_strict_json_schema = True
    source_agent = ProgressivePlannerAgent(source_llm)
    checkpoints = []
    source_agent.run(
        _context(),
        checkpoint_callback=checkpoints.append,
        resume_dependency_context=dependency_context,
    )
    changed_context = _context().model_copy(
        update={"research_question": "Estimate a different scientific target."}
    )
    resumed_llm = ScriptedMockLLMClient([])
    resumed_llm.supports_strict_json_schema = True

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(resumed_llm).run(
            changed_context,
            resume_checkpoint=checkpoints[4],
            resume_dependency_context=dependency_context,
        )

    assert caught.value.reason_code == (
        "progressive_resume_dependency_authority_mismatch"
    )
    assert resumed_llm.calls == []


def test_agent_rejects_resume_without_runtime_dependencies() -> None:
    source_llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    source_agent = ProgressivePlannerAgent(source_llm)
    checkpoints = []
    source_agent.run(_context(), checkpoint_callback=checkpoints.append)
    resumed_llm = ScriptedMockLLMClient([])

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(resumed_llm).run(
            _context(),
            resume_checkpoint=checkpoints[4],
        )

    assert caught.value.reason_code == (
        "progressive_resume_runtime_dependency_missing"
    )
    assert resumed_llm.calls == []


def test_agent_replays_a_complete_checkpoint_without_provider_calls() -> None:
    dependency_context = {
        "cohort_file_sha256": "b" * 64,
        "llm_signature": "codex:gpt-test",
        "prompt_version": "test-v1",
    }
    source_llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    source_llm.supports_strict_json_schema = True
    source_agent = ProgressivePlannerAgent(source_llm)
    checkpoints = []
    source_agent.run(
        _context(),
        checkpoint_callback=checkpoints.append,
        resume_dependency_context=dependency_context,
    )
    replay_llm = ScriptedMockLLMClient([])
    replay_llm.supports_strict_json_schema = True
    replay_agent = ProgressivePlannerAgent(replay_llm)

    plan = replay_agent.run(
        _context(),
        resume_checkpoint=checkpoints[-1],
        resume_dependency_context=dependency_context,
    )

    assert len(plan.steps) == 7
    assert replay_llm.calls == []
    assert replay_agent.last_resume_validated is True


def test_agent_repairs_only_the_current_materialization() -> None:
    materializations = _materialization_payloads()
    invalid_primary = json.loads(json.dumps(materializations[4]))
    invalid_primary["step"]["model_terms"][1]["name"] = "invented_covariate"
    responses = [
        _outline_payload(),
        _foundation_payload(),
        *materializations[:4],
        invalid_primary,
        materializations[4],
        *materializations[5:],
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(_context())

    assert len(plan.steps) == 7
    assert len(llm.calls) == 10
    repair_prompt = llm.calls[7][0][-1].content
    assert "HOST COMPILER OBSERVATION FOR THIS CURRENT STEP" in repair_prompt
    assert "progressive_unknown_variable" in repair_prompt
    assert '"step_id":"05_primary"' in repair_prompt
    assert "CURRENT UNLOCKED SUFFIX" not in repair_prompt
    assert "corrected complete skeleton" not in repair_prompt
    assert llm.calls[6][1]["structured_output"].authority_sha256 == (
        llm.calls[7][1]["structured_output"].authority_sha256
    )
    assert agent.last_prompt_metrics["compile_revision_count"] == 1
    assert agent.last_prompt_metrics["step_materialization_count"] == 7
    assert len(agent.last_prompt_metrics["step_materialization_schema_sha256"]) == 7
    assert len(
        agent.last_prompt_metrics["step_materialization_attempt_schema_sha256"]
    ) == 8
    assert agent.last_prompt_metrics["full_revision_count"] == 0


def test_agent_repairs_final_plan_dependent_method_layer_locally() -> None:
    payload = _payload()

    def binding(
        citation_key: str,
        design_elements: list[str],
        application: str,
    ) -> dict[str, object]:
        return {
            "citation_key": citation_key,
            "design_elements": design_elements,
            "application": application,
            "divergence": None,
        }

    payload["steps"][2]["literature_bindings"] = [
        binding(
            "strobe_2007",
            ["reporting"],
            "Report the prespecified exposure-outcome distribution.",
        )
    ]
    payload["steps"][3]["literature_bindings"] = [
        binding(
            "sterne_missing_data_2009",
            ["missing_data"],
            "Audit missingness before model fitting.",
        )
    ]
    payload["steps"][4]["literature_bindings"] = [
        binding(
            "record_2015",
            ["reporting"],
            "Report the prespecified adjusted association.",
        )
    ]
    payload["steps"][5]["literature_bindings"] = [
        binding(
            "durrleman_splines_1989",
            ["adjustment"],
            "Assess continuous-covariate functional form.",
        ),
        binding(
            "sterne_missing_data_2009",
            ["robustness"],
            "Assess the missing-data strategy.",
        ),
    ]
    payload["steps"][6]["literature_bindings"] = [
        binding(
            "strobe_2007",
            ["reporting"],
            "Report the completed observational analysis.",
        )
    ]
    materializations = _materialization_payloads(payload)
    repaired_final = json.loads(json.dumps(materializations[-1]))
    repaired_final["step"]["literature_bindings"][0]["design_elements"] = [
        "reporting",
        "outcome",
    ]
    repaired_final["step"]["literature_bindings"][0]["application"] = (
        "Report the completed analysis and interpret an absolute outcome "
        "measure alongside ratio estimates."
    )
    responses = [
        _outline_payload(payload),
        _foundation_payload(payload),
        *materializations[:-1],
        materializations[-1],
        repaired_final,
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(
        _context(),
        allowed_literature_citation_keys=[
            "strobe_2007",
            "record_2015",
            "sterne_missing_data_2009",
            "durrleman_splines_1989",
        ],
    )

    assert len(plan.steps) == 7
    assert len(llm.calls) == 10
    repair_prompt = llm.calls[-1][0][-1].content
    assert "progressive_final_method_layer_unbound" in repair_prompt
    assert '"missing_method_layers":["interpretation"]' in repair_prompt
    assert '"step_id":"07_figure"' in repair_prompt
    assert agent.last_prompt_metrics["compile_revision_count"] == 1
    assert plan.steps[-1].literature_design_bindings[0].design_elements == [
        "reporting",
        "outcome",
    ]


def test_agent_repairs_identical_distribution_contrast_locally() -> None:
    materializations = _materialization_payloads()
    invalid_distribution = json.loads(json.dumps(materializations[2]))
    invalid_distribution["step"]["comparison_exposure_level_index"] = 0
    responses = [
        _outline_payload(),
        _foundation_payload(),
        *materializations[:2],
        invalid_distribution,
        materializations[2],
        *materializations[3:],
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(_context())

    assert len(plan.steps) == 7
    assert len(llm.calls) == 10
    repair_prompt = llm.calls[5][0][-1].content
    assert "HOST COMPILER OBSERVATION FOR THIS CURRENT STEP" in repair_prompt
    assert "progressive_distribution_contrast_not_distinct" in repair_prompt
    assert '"step_id":"03_distribution"' in repair_prompt
    assert "CURRENT UNLOCKED SUFFIX" not in repair_prompt
    assert llm.calls[4][1]["structured_output"].authority_sha256 == (
        llm.calls[5][1]["structured_output"].authority_sha256
    )
    assert agent.last_prompt_metrics["compile_revision_count"] == 1
    assert agent.last_prompt_metrics["step_materialization_count"] == 7
    assert agent.last_prompt_metrics["full_revision_count"] == 0


def test_agent_repairs_outcome_covariate_at_the_current_model_step() -> None:
    materializations = _materialization_payloads()
    invalid_primary = json.loads(json.dumps(materializations[4]))
    invalid_primary["step"]["model_terms"].append(
        {
            "name": "outcome_flag",
            "role": "covariate",
            "coding": "binary",
            "reference_level_index": 0,
        }
    )
    responses = [
        _outline_payload(),
        _foundation_payload(),
        *materializations[:4],
        invalid_primary,
        materializations[4],
        *materializations[5:],
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(_context())

    primary = next(step for step in plan.steps if step.step_id == "05_primary")
    assert "outcome_flag" not in primary.model_requirements[0].covariates
    assert len(llm.calls) == 10
    assert "progressive_model_outcome_is_covariate" in (
        llm.calls[7][0][-1].content
    )
    assert "never include the outcome as a model term or covariate" in (
        llm.calls[6][0][-1].content
    )
    assert agent.last_prompt_metrics["compile_revision_count"] == 1


def test_prefix_pydantic_failure_becomes_attributable_compiler_finding() -> None:
    payload = _payload()
    outline = ProgressivePlanOutline.model_validate(_outline_payload(payload))
    foundation = ProgressiveFoundationMaterialization.model_validate(
        _foundation_payload(payload)
    ).foundation
    materializations = [
        ProgressiveStepMaterialization.model_validate(item)
        for item in _materialization_payloads(payload)
    ]
    earlier_primary = materializations[0].step.model_copy(
        update={"planned_analysis_role": "primary"}
    )
    current_primary = materializations[2].model_copy(
        update={
            "step": materializations[2].step.model_copy(
                update={"planned_analysis_role": "primary"}
            )
        }
    )
    state = ProgressivePrefixState(steps=(earlier_primary,))

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_prefix(
            state,
            current_primary,
            outline=outline,
            foundation=foundation,
            context=_context(),
            allowed_literature_citation_keys=(),
            allowed_know_how_decisions=None,
            reporting_method_source_keys=(),
        )

    assert caught.value.reason_code == "progressive_prefix_contract_invalid"
    assert caught.value.step_id == "03_distribution"
    assert caught.value.step_index == 1
    assert caught.value.easyicu_safe_diagnostic["owner"] == (
        "easyicu.planning.progressive_compiler_v1"
    )
    assert "at most one primary" in caught.value.details["findings"][0]["message"]


def test_agent_stops_after_one_host_compile_repair_and_keeps_attempts() -> None:
    materializations = _materialization_payloads()
    invalid_distribution = json.loads(json.dumps(materializations[2]))
    invalid_distribution["step"]["comparison_exposure_level_index"] = 0
    responses = [
        _outline_payload(),
        _foundation_payload(),
        *materializations[:2],
        invalid_distribution,
        invalid_distribution,
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        agent.run(_context())

    assert caught.value.reason_code == (
        "progressive_distribution_contrast_not_distinct"
    )
    assert len(llm.calls) == 6
    assert [item.revision for item in agent.last_compile_failure_attempts] == [
        0,
        1,
    ]
    assert {
        item.compiler_finding.reason_code
        for item in agent.last_compile_failure_attempts
    } == {"progressive_distribution_contrast_not_distinct"}
    assert agent.last_prompt_metrics["compile_revision_count"] == 1


class _RecordingEvidence:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def get(self, evidence_id_or_alias: str) -> object | None:
        return self.records.get(evidence_id_or_alias)

    def register_file(self, **kwargs: object) -> object:
        evidence_id = str(kwargs["evidence_id"])
        source_path = Path(str(kwargs["source_path"]))
        self.records[evidence_id] = {
            **dict(kwargs),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
        return self.records[evidence_id]


def test_progressive_compile_failure_persists_for_zero_provider_replay(
    tmp_path: Path,
) -> None:
    materializations = _materialization_payloads()
    invalid_distribution = json.loads(json.dumps(materializations[2]))
    invalid_distribution["step"]["comparison_exposure_level_index"] = 0
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in materializations[:2]],
            json.dumps(invalid_distribution),
            json.dumps(invalid_distribution),
        ]
    )
    llm.supports_strict_json_schema = True
    evidence = _RecordingEvidence()
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"synthetic replay cohort")

    with pytest.raises(ProgressivePlanCompileError):
        run_progressive_planner(
            planner=ProgressivePlannerAgent(llm),
            context=_context(),
            run_dir=tmp_path,
            evidence=evidence,
            prompt_pack_version="test-v1",
            resume_checkpoint_path=None,
            resume_checkpoint_sha256=None,
            cohort_path=cohort_path,
            llm_signature="mock:test",
            planner_kwargs={},
            know_how_binding=PlannerKnowHowBinding(),
            planning_contract_context="",
            finding_sink=lambda _finding: None,
        )

    replay_path = tmp_path / "progressive_compile_failure_replay.json"
    replay = load_progressive_compile_failure_replay(
        replay_path=replay_path,
        expected_artifact_sha256=str(
            evidence.records["progressive_compile_failure_replay"]["sha256"]
        ),
    )
    assert isinstance(replay, ProgressiveCompileFailureReplay)
    assert replay.prefix_checkpoint_sequence == 3
    assert len(replay.attempts) == 2
    assert evidence.records["progressive_compile_failure_replay"]["inputs"] == [
        "research_context",
        "progressive_planner_checkpoint_003",
    ]

    checkpoint = ProgressivePlannerCheckpoint.model_validate_json(
        (tmp_path / "progressive_planner_checkpoint_003.json").read_bytes()
    )
    assert checkpoint.foundation is not None
    state = ProgressivePrefixState()
    for materialization in checkpoint.materializations:
        state = compile_progressive_prefix(
            state,
            materialization,
            outline=checkpoint.outline,
            foundation=checkpoint.foundation.foundation,
            context=_context(),
            allowed_literature_citation_keys=(),
            allowed_know_how_decisions=None,
            reporting_method_source_keys=(),
        )
    with pytest.raises(ProgressivePlanCompileError) as replayed:
        compile_progressive_prefix(
            state,
            replay.attempts[0].materialization,
            outline=checkpoint.outline,
            foundation=checkpoint.foundation.foundation,
            context=_context(),
            allowed_literature_citation_keys=(),
            allowed_know_how_decisions=None,
            reporting_method_source_keys=(),
        )
    assert replayed.value.reason_code == (
        replay.attempts[0].compiler_finding.reason_code
    )

    replay_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProgressivePlanningArtifactError) as tampered:
        load_progressive_compile_failure_replay(
            replay_path=replay_path,
            expected_artifact_sha256=str(
                evidence.records["progressive_compile_failure_replay"]["sha256"]
            ),
        )
    assert tampered.value.reason_code == (
        "progressive_compile_replay_digest_mismatch"
    )


def test_progressive_checkpoints_persist_as_a_digest_verified_chain(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    evidence = _RecordingEvidence()
    paths = []

    def checkpoint_callback(checkpoint) -> None:
        paths.append(
            persist_progressive_planner_checkpoint(
                run_dir=tmp_path,
                evidence=evidence,
                checkpoint=checkpoint,
                prompt_pack_version="test",
            )
        )

    agent.run(_context(), checkpoint_callback=checkpoint_callback)

    assert [path.name for path in paths] == [
        f"progressive_planner_checkpoint_{index:03d}.json"
        for index in range(9)
    ]
    assert set(evidence.records) == {
        f"progressive_planner_checkpoint_{index:03d}" for index in range(9)
    }
    assert evidence.records["progressive_planner_checkpoint_008"]["inputs"] == [
        "research_context",
        "progressive_planner_checkpoint_007",
    ]

    loaded = load_progressive_planner_checkpoint_chain(
        last_checkpoint_path=paths[-1],
        expected_artifact_sha256=hashlib.sha256(paths[-1].read_bytes()).hexdigest(),
    )
    assert [item.sequence for item in loaded] == list(range(9))
    assert loaded[-1].checkpoint_sha256 == json.loads(
        paths[-1].read_text(encoding="utf-8")
    )["checkpoint_sha256"]


def test_progressive_design_canary_stops_after_one_validated_outline(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient([json.dumps(_outline_payload())])
    llm.supports_strict_json_schema = True
    evidence = _RecordingEvidence()
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"design canary cohort")

    result = run_progressive_planner(
        planner=ProgressivePlannerAgent(llm),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        prompt_pack_version="test-v1",
        resume_checkpoint_path=None,
        resume_checkpoint_sha256=None,
        cohort_path=cohort_path,
        llm_signature="mock:test",
        planner_kwargs={},
        know_how_binding=PlannerKnowHowBinding(),
        planning_contract_context="",
        finding_sink=lambda _finding: None,
        stop_after_outline=True,
    )

    assert isinstance(result, ProgressiveDesignCanaryDraft)
    assert result.checkpoint.stage == "outline"
    assert result.outline.design_selection is not None
    assert len(llm.calls) == 1
    assert not (tmp_path / "progressive_plan_foundation.json").exists()


def test_progressive_resume_loader_rejects_incomplete_source_chain(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []
    agent.run(_context(), checkpoint_callback=checkpoints.append)
    terminal = tmp_path / "progressive_planner_checkpoint_004.json"
    terminal.write_text(checkpoints[4].model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        load_progressive_planner_checkpoint_chain(
            last_checkpoint_path=terminal,
            expected_artifact_sha256=hashlib.sha256(
                terminal.read_bytes()
            ).hexdigest(),
        )

    assert caught.value.reason_code == "progressive_resume_checkpoint_missing"


def test_resume_checkpoint_recorder_imports_only_after_validation(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []
    agent.run(_context(), checkpoint_callback=checkpoints.append)
    evidence = _RecordingEvidence()
    recorder = ProgressivePlannerCheckpointRecorder(
        run_dir=tmp_path,
        evidence=evidence,
        prompt_pack_version="test",
        source_chain=tuple(checkpoints[:5]),
    )

    recorder.record(checkpoints[5])

    assert evidence.records == {}
    assert list(tmp_path.glob("progressive_planner_checkpoint_*.json")) == []

    receipt = recorder.persist_validated_resume()

    assert receipt.source_sequence == 4
    assert receipt.reused_materialization_count == 3
    assert receipt.new_checkpoint_count == 1
    assert set(evidence.records) == {
        f"progressive_planner_checkpoint_{index:03d}" for index in range(6)
    }
    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        recorder.record(checkpoints[6])
    assert caught.value.reason_code == (
        "progressive_resume_checkpoint_recorder_closed"
    )


def test_progressive_orchestrator_resumes_and_imports_validated_chain(
    tmp_path: Path,
) -> None:
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"development cohort authority")
    dependency_context = {
        "cohort_file_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "llm_signature": "mock:test",
        "prompt_version": "test-v1",
    }
    materializations = _materialization_payloads()
    source_agent = ProgressivePlannerAgent(
        ScriptedMockLLMClient(
            [
                json.dumps(_outline_payload()),
                json.dumps(_foundation_payload()),
                *[json.dumps(item) for item in materializations],
            ]
        )
    )
    source_agent.llm.supports_strict_json_schema = True
    source_checkpoints = []
    source_agent.run(
        _context(),
        checkpoint_callback=source_checkpoints.append,
        resume_dependency_context=dependency_context,
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_evidence = _RecordingEvidence()
    source_paths = [
        persist_progressive_planner_checkpoint(
            run_dir=source_dir,
            evidence=source_evidence,
            checkpoint=checkpoint,
            prompt_pack_version="test-v1",
        )
        for checkpoint in source_checkpoints[:5]
    ]
    resumed_llm = ScriptedMockLLMClient(
        [json.dumps(item) for item in materializations[3:]]
    )
    resumed_llm.supports_strict_json_schema = True
    findings = []
    current_evidence = _RecordingEvidence()
    current_dir = tmp_path / "current"
    current_dir.mkdir()

    result = run_progressive_planner(
        planner=ProgressivePlannerAgent(resumed_llm),
        context=_context(),
        run_dir=current_dir,
        evidence=current_evidence,
        prompt_pack_version="test-v1",
        resume_checkpoint_path=source_paths[-1],
        resume_checkpoint_sha256=hashlib.sha256(
            source_paths[-1].read_bytes()
        ).hexdigest(),
        cohort_path=cohort_path,
        llm_signature="mock:test",
        planner_kwargs={},
        know_how_binding=PlannerKnowHowBinding(),
        planning_contract_context="",
        finding_sink=findings.append,
    )

    assert result.generation_mode == "llm_progressive_v2_dev_resume"
    assert len(result.plan.steps) == 7
    assert len(resumed_llm.calls) == 4
    assert findings[0].detail["reason_code"] == (
        "progressive_development_checkpoint_resumed"
    )
    assert {
        key
        for key in current_evidence.records
        if key.startswith("progressive_planner_checkpoint_")
    } == {
        f"progressive_planner_checkpoint_{index:03d}" for index in range(9)
    }


def test_progressive_orchestrator_persists_validated_resume_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"development cohort authority")
    dependencies = {
        "cohort_file_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "llm_signature": "mock:test",
        "prompt_version": "test-v1",
    }
    source_agent = ProgressivePlannerAgent(
        ScriptedMockLLMClient(
            [
                json.dumps(_outline_payload()),
                json.dumps(_foundation_payload()),
                *[json.dumps(item) for item in _materialization_payloads()],
            ]
        )
    )
    checkpoints = []
    source_agent.run(
        _context(),
        checkpoint_callback=checkpoints.append,
        resume_dependency_context=dependencies,
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_evidence = _RecordingEvidence()
    paths = [
        persist_progressive_planner_checkpoint(
            run_dir=source_dir,
            evidence=source_evidence,
            checkpoint=checkpoint,
            prompt_pack_version="test-v1",
        )
        for checkpoint in checkpoints[:2]
    ]
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    evidence = _RecordingEvidence()
    planner = ProgressivePlannerAgent(ScriptedMockLLMClient([]))

    def interrupted(*_args, **_kwargs):
        planner.last_resume_validated = True
        raise KeyboardInterrupt("operator stop")

    monkeypatch.setattr(planner, "run", interrupted)
    with pytest.raises(KeyboardInterrupt, match="operator stop"):
        run_progressive_planner(
            planner=planner,
            context=_context(),
            run_dir=current_dir,
            evidence=evidence,
            prompt_pack_version="test-v1",
            resume_checkpoint_path=paths[-1],
            resume_checkpoint_sha256=hashlib.sha256(paths[-1].read_bytes()).hexdigest(),
            cohort_path=cohort_path,
            llm_signature="mock:test",
            planner_kwargs={},
            know_how_binding=PlannerKnowHowBinding(),
            planning_contract_context="",
            finding_sink=lambda _finding: None,
        )

    assert (current_dir / "progressive_planner_checkpoint_001.json").exists()


def test_progressive_checkpoint_rejects_mutated_predecessor(tmp_path: Path) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []
    agent.run(_context(), checkpoint_callback=checkpoints.append)
    evidence = _RecordingEvidence()
    first_path = persist_progressive_planner_checkpoint(
        run_dir=tmp_path,
        evidence=evidence,
        checkpoint=checkpoints[0],
        prompt_pack_version="test",
    )
    first_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        persist_progressive_planner_checkpoint(
            run_dir=tmp_path,
            evidence=evidence,
            checkpoint=checkpoints[1],
            prompt_pack_version="test",
        )

    assert caught.value.reason_code == (
        "progressive_source_artifact_digest_mismatch"
    )


def test_progressive_artifacts_bind_each_schema_authority(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    plan = agent.run(_context())
    assert agent.last_outline is not None
    assert agent.last_foundation is not None
    assert agent.last_skeleton is not None
    assert agent.last_compile_receipt is not None
    evidence = _RecordingEvidence()

    paths = persist_progressive_planning_artifacts(
        run_dir=tmp_path,
        evidence=evidence,
        outline=agent.last_outline,
        foundation=agent.last_foundation,
        materializations=agent.last_materializations,
        skeleton=agent.last_skeleton,
        compile_receipt=agent.last_compile_receipt,
        prompt_metrics=agent.last_prompt_metrics,
        prompt_pack_version="test",
    )

    ledger = json.loads(paths.materializations.read_text(encoding="utf-8"))
    requests = [call[1]["structured_output"] for call in llm.calls]
    assert ledger["outline_structured_output_authority_sha256"] == (
        requests[0].authority_sha256
    )
    assert ledger["foundation_structured_output_authority_sha256"] == (
        requests[1].authority_sha256
    )
    assert [
        item["structured_output_authority_sha256"]
        for item in ledger["materializations"]
    ] == [request.authority_sha256 for request in requests[2:]]
    assert [item["step_id"] for item in ledger["materializations"]] == [
        item.step.step_id for item in agent.last_materializations
    ]
    assert set(evidence.records) == {
        "progressive_plan_outline",
        "progressive_plan_foundation",
        "progressive_step_materializations",
        "progressive_plan_skeleton",
        "progressive_plan_compile_receipt",
    }
    assert evidence.records["progressive_plan_skeleton"]["inputs"] == [
        "progressive_plan_outline",
        "progressive_plan_foundation",
        "progressive_step_materializations",
        "research_context",
    ]

    metrics_path = tmp_path / "planner_prompt_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.planner_prompt_metrics/1",
                **agent.last_prompt_metrics,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence.register_file(
        evidence_id="planner_prompt_metrics",
        source_path=metrics_path,
    )
    plan_path = tmp_path / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(evidence_id="analysis_plan", source_path=plan_path)
    normalized = build_normalized_plan_lineage(
        proposed_plan=plan,
        proposed_source="llm_progressive_v2",
        pre_normalization_plan=plan,
        normalized_plan=plan,
        resume_scientific_semantics_changed=False,
        host_scientific_semantics_changed=False,
    )
    lifecycle_path = tmp_path / "plan_lifecycle_revision_0.json"
    lifecycle_path.write_text(
        normalized.model_dump_json(indent=2),
        encoding="utf-8",
    )
    evidence.register_file(
        evidence_id="plan_lifecycle_revision_0",
        source_path=lifecycle_path,
    )

    authority = persist_progressive_planning_authority(
        run_dir=tmp_path,
        evidence=evidence,
        proposed_plan_sha256=normalized.proposed.plan_sha256,
        normalized_plan_sha256=normalized.plan_sha256,
        normalized_plan_authority_sha256=normalized.authority_sha256,
        normalized_plan_evidence_id="plan_lifecycle_revision_0",
        normalized_plan_filename="plan_lifecycle_revision_0.json",
        prompt_pack_version="test",
    )

    assert authority.strict_transport_bound is True
    assert authority.compiled_analysis_plan_sha256 == normalized.proposed.plan_sha256
    assert authority.normalized_plan_authority_sha256 == normalized.authority_sha256
    assert [item.step_id for item in authority.ordered_steps] == [
        item.step_id for item in agent.last_outline.steps
    ]
    assert evidence.records["progressive_planning_authority"]["inputs"][-1] == (
        "plan_lifecycle_revision_0"
    )


def test_progressive_artifacts_fail_closed_on_schema_authority_drift(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    agent.run(_context())
    assert agent.last_outline is not None
    assert agent.last_foundation is not None
    assert agent.last_skeleton is not None
    assert agent.last_compile_receipt is not None
    drifted_metrics = dict(agent.last_prompt_metrics)
    drifted_metrics["step_materialization_schema_sha256"] = ["0" * 64]

    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        persist_progressive_planning_artifacts(
            run_dir=tmp_path,
            evidence=_RecordingEvidence(),
            outline=agent.last_outline,
            foundation=agent.last_foundation,
            materializations=agent.last_materializations,
            skeleton=agent.last_skeleton,
            compile_receipt=agent.last_compile_receipt,
            prompt_metrics=drifted_metrics,
            prompt_pack_version="test",
        )

    assert caught.value.reason_code == (
        "progressive_step_schema_authority_count_mismatch"
    )


def test_progressive_artifacts_do_not_overwrite_existing_evidence_identity(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    agent.run(_context())
    assert agent.last_outline is not None
    assert agent.last_foundation is not None
    assert agent.last_skeleton is not None
    assert agent.last_compile_receipt is not None
    evidence = _RecordingEvidence()
    paths = persist_progressive_planning_artifacts(
        run_dir=tmp_path,
        evidence=evidence,
        outline=agent.last_outline,
        foundation=agent.last_foundation,
        materializations=agent.last_materializations,
        skeleton=agent.last_skeleton,
        compile_receipt=agent.last_compile_receipt,
        prompt_metrics=agent.last_prompt_metrics,
        prompt_pack_version="test",
    )
    original_ledger = paths.materializations.read_bytes()
    changed_step = agent.last_materializations[0].step.model_copy(
        update={"objective": "A different unreviewed objective."}
    )
    changed_materializations = [
        agent.last_materializations[0].model_copy(update={"step": changed_step}),
        *agent.last_materializations[1:],
    ]

    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        persist_progressive_planning_artifacts(
            run_dir=tmp_path,
            evidence=evidence,
            outline=agent.last_outline,
            foundation=agent.last_foundation,
            materializations=changed_materializations,
            skeleton=agent.last_skeleton,
            compile_receipt=agent.last_compile_receipt,
            prompt_metrics=agent.last_prompt_metrics,
            prompt_pack_version="test",
        )

    assert caught.value.reason_code == (
        "progressive_existing_evidence_identity_mismatch"
    )
    assert paths.materializations.read_bytes() == original_ledger


def test_agent_rejects_materialization_coordinate_drift_without_full_rewrite() -> None:
    materializations = _materialization_payloads()
    materializations[2]["step"]["objective"] = "Rewrite the outline-owned objective."
    responses = [_outline_payload(), _foundation_payload(), *materializations]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    agent = ProgressivePlannerAgent(llm)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        agent.run(_context())

    assert caught.value.reason_code == "progressive_step_materialization_mismatch"
    assert caught.value.step_id == "03_distribution"
    assert caught.value.path == "objective"
    assert len(llm.calls) == 5
    assert agent.last_prompt_metrics["full_revision_count"] == 0


def test_step_materialization_must_preserve_outline_literature_roster() -> None:
    payload = _payload()
    payload["steps"][3]["literature_bindings"] = [
        {
            "citation_key": "sterne_missing_data_2009",
            "design_elements": ["missing_data"],
            "application": (
                "Apply the prespecified missing-data method to the measurement audit."
            ),
            "divergence": None,
        }
    ]
    outline = ProgressivePlanOutline.model_validate(_outline_payload(payload))
    materialization_payload = _materialization_payloads(payload)[3]
    materialization_payload["step"]["literature_bindings"] = []
    materialization = ProgressiveStepMaterialization.model_validate(
        materialization_payload
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        validate_progressive_materialization_coordinate(
            materialization,
            outline_step=outline.steps[3],
            outline_step_sha256=canonical_sha256(
                outline.steps[3].model_dump(mode="json")
            ),
            step_index=3,
        )

    assert caught.value.reason_code == (
        "progressive_step_literature_roster_mismatch"
    )
    assert caught.value.step_id == "04_measurement"
    assert caught.value.path == "literature_bindings"


def test_agent_repairs_duplicate_outline_literature_key_without_diagnostic_crash() -> None:
    payload = _payload()
    payload["steps"][2]["literature_bindings"] = [
        {
            "citation_key": "strobe_2007",
            "design_elements": ["reporting", "outcome"],
            "application": "Report the prespecified descriptive comparison.",
            "divergence": None,
        },
        {
            "citation_key": "record_2015",
            "design_elements": ["reporting"],
            "application": "Report the routinely collected data provenance.",
            "divergence": None,
        },
    ]
    for step_index in (4, 5):
        payload["steps"][step_index]["literature_bindings"] = [
            {
                "citation_key": "strobe_2007",
                "design_elements": ["reporting"],
                "application": "Report this prespecified scientific analysis.",
                "divergence": None,
            }
        ]
    materializations = _materialization_payloads(payload)
    duplicate = json.loads(json.dumps(materializations[2]))
    duplicate["step"]["literature_bindings"][1]["citation_key"] = "strobe_2007"
    responses = [
        _outline_payload(payload),
        _foundation_payload(payload),
        *materializations[:2],
        duplicate,
        materializations[2],
        *materializations[3:],
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(
        _context(),
        allowed_literature_citation_keys=["strobe_2007", "record_2015"],
    )

    assert len(plan.steps) == 7
    assert len(llm.calls) == 10
    assert agent.last_prompt_metrics["compile_revision_count"] == 1
    repair_prompt = llm.calls[5][0][-1].content
    assert "progressive_step_literature_roster_mismatch" in repair_prompt


def test_step_transport_requires_each_outline_literature_key_once() -> None:
    outline_step = ProgressiveOutlineStep(
        step_id="04_measurement",
        planned_analysis_role="auxiliary",
        module_id="measurement_audit",
        objective="Audit missingness and measurement-process coverage.",
        variable_names=["exposure_flag", "outcome_flag"],
        literature_citation_keys=["sterne_missing_data_2009"],
    )
    request = progressive_step_materialization_request(
        outline_step=outline_step,
        outline_step_sha256=canonical_sha256(
            outline_step.model_dump(mode="json")
        ),
        variable_names=outline_step.variable_names,
        scientific_action_ids=(),
        allowed_literature_citation_keys=outline_step.literature_citation_keys,
    )
    schema = json.loads(request.schema_json)
    bindings = schema["$defs"]["ProgressiveSkeletonStep"]["properties"][
        "literature_bindings"
    ]

    assert bindings["minItems"] == 1
    assert bindings["maxItems"] == 1
