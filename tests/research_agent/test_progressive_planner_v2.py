"""Progressive Planner v2 contract/compiler regressions."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from easyicu.research_agent.agents.progressive_payload import (
    progressive_foundation_structured_output_request,
    progressive_outline_structured_output_request,
    progressive_step_materialization_request,
    progressive_structured_output_request,
)
from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    candidate_analysis_types,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    exposure_outcome_distribution_executor_owns_step,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    robustness_replay_spec_is_emittable,
)
from easyicu.research_agent.planning.progressive_compiler import (
    assert_immutable_prefix,
    compile_progressive_plan,
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
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
    ProgressivePlannerCheckpoint,
    ProgressivePredicateValue,
)
from easyicu.research_agent.planning.progressive_resume import (
    ProgressivePrefixState,
    compile_progressive_prefix,
)
from easyicu.research_agent.orchestration.progressive_planning import (
    run_progressive_planner,
)
from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.authority.plan_lifecycle import (
    build_normalized_plan_lineage,
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


def _outline_payload(payload: dict | None = None) -> dict:
    source = payload or _payload()
    return {
        "schema_version": "easyicu.progressive_plan_outline/1",
        "analysis_type": source["analysis_type"],
        "cohort_objective": "Use the sealed cohort and preserve its denominator.",
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

    assert len(branches) == 2
    by_strategy = {
        branch["properties"]["missing_strategy"]["const"]: branch
        for branch in branches
    }
    no_missing = by_strategy["none"]["properties"]
    assert no_missing["complete_case_variables"]["maxItems"] == 0
    complete_case = by_strategy["complete_case"]["properties"]
    assert complete_case["axis"] == {"type": "string", "const": "missing"}
    assert complete_case["complete_case_variables"]["minItems"] == 1
    assert complete_case["complete_case_variables"]["items"]["enum"] == [
        "exposure_flag",
        "outcome_flag",
    ]


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
        "artifact:analysis_cohort",
        "table:exposure_outcome_distribution",
        "table:adjusted_association_estimates",
    ]
    assert {item.input_key for item in figure.input_consumption_contracts} == {
        "table:exposure_outcome_distribution",
        "table:adjusted_association_estimates",
    }
    assert {item.mode for item in figure.input_consumption_contracts} == {"all_rows"}


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
        [json.dumps(_outline_payload()), json.dumps(invalid_foundation)]
    )
    llm.supports_strict_json_schema = True
    checkpoints = []
    agent = ProgressivePlannerAgent(llm)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        agent.run(_context(), checkpoint_callback=checkpoints.append)

    assert caught.value.reason_code == "progressive_foundation_cohort_invalid"
    assert caught.value.path == "cohort"
    assert len(llm.calls) == 2
    assert [item.stage for item in checkpoints] == ["outline"]
    assert agent.last_foundation is None


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
