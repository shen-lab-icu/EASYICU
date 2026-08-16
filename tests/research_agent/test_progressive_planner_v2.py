"""Progressive Planner v2 contract/compiler regressions."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.progressive_payload import (
    progressive_structured_output_request,
)
from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    candidate_analysis_types,
)
from easyicu.research_agent.planning.progressive_compiler import (
    assert_immutable_prefix,
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.providers.strict_json_schema import (
    closed_pydantic_json_schema,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
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
                "sensitivity_spec_ids": [],
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


def _skeleton() -> ProgressivePlanSkeleton:
    return ProgressivePlanSkeleton.model_validate(_payload())


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

    distribution = by_id["03_distribution"].exposure_outcome_distribution_spec
    assert distribution is not None
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
        "artifact:analysis_cohort",
        "table:exposure_outcome_distribution",
        "table:adjusted_association_estimates",
    ]
    assert {item.input_key for item in figure.input_consumption_contracts} == {
        "table:exposure_outcome_distribution",
        "table:adjusted_association_estimates",
    }
    assert {item.mode for item in figure.input_consumption_contracts} == {"all_rows"}


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
    step = schema["$defs"]["ProgressiveSkeletonStep"]["properties"]
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


def test_agent_repairs_only_rejected_suffix_with_strict_transport() -> None:
    first = _payload()
    first["steps"][4]["scientific_action_id"] = "descriptive.descriptive_summary"
    corrected = _payload()
    suffix = {
        "schema_version": "easyicu.progressive_suffix_revision/1",
        "replace_from_step_id": "05_primary",
        "replacement_steps": corrected["steps"][4:],
        "rationale": (
            "Correct the family-specific action while preserving the compiled prefix."
        ),
    }
    llm = ScriptedMockLLMClient([json.dumps(first), json.dumps(suffix)])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)

    plan = agent.run(_context())

    assert len(plan.steps) == 7
    assert len(llm.calls) == 2
    requests = [call[1]["structured_output"] for call in llm.calls]
    assert [request.name for request in requests] == [
        "easyicu_progressive_plan_skeleton_v1",
        "easyicu_progressive_plan_suffix_v1",
    ]
    second_prompt = llm.calls[1][0][-1].content
    assert "IMMUTABLE COMPILED PREFIX" in second_prompt
    assert '"step_id": "04_measurement"' in second_prompt
    assert agent.last_prompt_metrics["compile_revision_count"] == 1
    assert agent.last_prompt_metrics["suffix_revision_count"] == 1
    assert agent.last_compile_receipt is not None
    assert agent.last_skeleton is not None
