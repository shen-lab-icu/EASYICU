"""Run-bound strict transport schema for Progressive Planner v2."""

from __future__ import annotations

import copy
import re
from functools import lru_cache
from typing import Any, Mapping, Sequence, get_args

from ..planning.method_literature import METHOD_CARDS
from ..planning.progressive_contract import (
    PROGRESSIVE_HOST_COMPILED_OUTPUTS,
    ProgressiveFoundationMaterialization,
    ProgressiveModuleId,
    ProgressiveOutlineStep,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
    ProgressiveStepMaterialization,
    ProgressiveSuffixRevision,
    progressive_module_ids_for_analysis_types,
)
from ..providers.protocol import StructuredOutputRequest
from ..providers.strict_json_schema import (
    StrictJsonSchemaError,
    assert_closed_json_schema,
    strictify_json_schema,
)


class ProgressiveTransportSchemaError(ValueError):
    """The run-bound progressive schema drifted from its contract model."""


def _closed_object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _string_enum(values: Sequence[str]) -> dict[str, Any]:
    normalized = list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )
    if not normalized:
        raise ProgressiveTransportSchemaError("run-bound enum cannot be empty")
    return {"type": "string", "enum": normalized}


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [copy.deepcopy(dict(schema)), {"type": "null"}]}


def _non_null(schema: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    """Return the single non-null branch of one optional field schema."""

    branches = schema.get("anyOf")
    if not isinstance(branches, list):
        raise ProgressiveTransportSchemaError(
            f"progressive optional field {field!r} has no anyOf branches"
        )
    non_null = [
        branch
        for branch in branches
        if isinstance(branch, dict) and branch.get("type") != "null"
    ]
    if len(non_null) != 1:
        raise ProgressiveTransportSchemaError(
            f"progressive optional field {field!r} has no unique non-null branch"
        )
    return copy.deepcopy(non_null[0])


def _bind_step_module_shape(
    definitions: dict[str, Any],
    *,
    locked_module_id: str | None = None,
    locked_has_dependencies: bool | None = None,
    locked_has_available_products: bool | None = None,
) -> None:
    """Compile Pydantic module-shape rules into the provider contract.

    A standard module's method is host-owned, so ``custom_method`` is not an
    alternate spelling of that method and must be null.  ``custom_analysis`` is
    the only branch that accepts free-form method text.  When the current
    outline coordinate locks a module, required non-null fields and output
    cardinality are also locked here so a response accepted by the transport
    cannot be rejected immediately by ``ProgressiveSkeletonStep`` for the same
    module-shape rule.
    """

    step = definitions.get("ProgressiveSkeletonStep")
    if not isinstance(step, dict) or not isinstance(step.get("properties"), dict):
        raise ProgressiveTransportSchemaError(
            "progressive skeleton step definition is unavailable"
        )
    properties = step["properties"]
    output_intent = definitions.get("ProgressiveOutputIntent")
    if not isinstance(output_intent, dict) or not isinstance(
        output_intent.get("properties"), dict
    ):
        raise ProgressiveTransportSchemaError(
            "progressive output intent definition is unavailable"
        )
    module_ids = list(get_args(ProgressiveModuleId))
    standard_ids = [value for value in module_ids if value != "custom_analysis"]

    if locked_module_id is not None:
        if locked_module_id not in module_ids:
            raise ProgressiveTransportSchemaError(
                f"unknown locked progressive module {locked_module_id!r}"
            )
        if locked_module_id == "table_one":
            properties["table_one_group_by"] = _non_null(
                properties["table_one_group_by"], field="table_one_group_by"
            )
            properties["table_one_mode"] = _non_null(
                properties["table_one_mode"], field="table_one_mode"
            )
            properties["table_one_variables"]["minItems"] = 1
        else:
            properties["table_one_group_by"] = {"type": "null"}
            properties["table_one_mode"] = {"type": "null"}
            properties["table_one_variables"]["maxItems"] = 0

        required_non_null: tuple[str, ...] = ()
        if locked_module_id == "adjusted_association":
            required_non_null = (
                "primary_exposure",
                "outcome",
                "outcome_type",
            )
            properties["model_terms"]["minItems"] = 1
        elif locked_module_id == "exposure_outcome_distribution":
            required_non_null = (
                "primary_exposure",
                "outcome",
                "event_level_index",
                "reference_exposure_level_index",
                "comparison_exposure_level_index",
                "denominator_policy",
                "missing_exposure_policy",
                "missing_outcome_policy",
                "confidence_level",
            )
        for field in required_non_null:
            properties[field] = _non_null(properties[field], field=field)

        if locked_module_id in {
            "measurement_audit",
            "custom_analysis",
            "visualization",
            "report",
        }:
            properties["outputs"]["minItems"] = 1
        if locked_module_id in PROGRESSIVE_HOST_COMPILED_OUTPUTS:
            properties["outputs"]["minItems"] = 0
            properties["outputs"]["maxItems"] = 0
        output_properties = output_intent["properties"]
        if locked_module_id == "visualization":
            output_properties["product_id"] = {
                "type": "string",
                "pattern": r"^figure:[a-z][a-z0-9_]*$",
            }
            output_properties["semantic_role"] = {
                "type": "string",
                "const": "figure",
            }
        elif locked_module_id == "report":
            output_properties["product_id"] = {
                "type": "string",
                "pattern": r"^report:[a-z][a-z0-9_]*$",
            }
            output_properties["semantic_role"] = {
                "type": "string",
                "const": "report",
            }
        if locked_module_id == "visualization" and not locked_has_dependencies:
            if not locked_has_available_products:
                raise ProgressiveTransportSchemaError(
                    "visualization outline requires an upstream dependency or "
                    "available product"
                )
            properties["product_inputs"]["minItems"] = 1

    standard = copy.deepcopy(step)
    standard["properties"]["module_id"] = (
        {"type": "string", "const": locked_module_id}
        if locked_module_id is not None and locked_module_id != "custom_analysis"
        else _string_enum(standard_ids)
    )
    standard["properties"]["custom_method"] = {"type": "null"}

    custom_fields = (
        "step_id",
        "planned_analysis_role",
        "objective",
        "depends_on",
        "raw_inputs",
        "product_inputs",
        "outputs",
        "scientific_action_id",
        "sensitivity_spec_ids",
        "literature_bindings",
    )
    custom_properties = {
        field: copy.deepcopy(properties[field]) for field in custom_fields
    }
    custom_properties["module_id"] = {
        "type": "string",
        "const": "custom_analysis",
    }
    custom_properties["custom_method"] = _non_null(
        properties["custom_method"], field="custom_method"
    )
    custom_output = copy.deepcopy(output_intent)
    custom_output["properties"]["semantic_role"] = _string_enum(
        ("scientific_sensitivity", "custom")
    )
    custom_properties["outputs"]["items"] = custom_output
    custom_properties["outputs"]["minItems"] = 1
    custom = _closed_object(custom_properties)
    if locked_module_id == "custom_analysis":
        generic_custom = copy.deepcopy(custom)
        generic_properties = generic_custom["properties"]
        generic_properties["outputs"]["items"]["properties"][
            "semantic_role"
        ] = {"type": "string", "const": "custom"}
        generic_properties["sensitivity_spec_ids"]["maxItems"] = 0

        scientific_sensitivity = copy.deepcopy(custom)
        sensitivity_properties = scientific_sensitivity["properties"]
        sensitivity_properties["outputs"]["minItems"] = 1
        sensitivity_properties["outputs"]["maxItems"] = 1
        sensitivity_output = sensitivity_properties["outputs"]["items"][
            "properties"
        ]
        sensitivity_output["product_id"] = {
            "type": "string",
            "pattern": r"^table:[a-z][a-z0-9_]*$",
        }
        sensitivity_output["semantic_role"] = {
            "type": "string",
            "const": "scientific_sensitivity",
        }
        sensitivity_properties["sensitivity_spec_ids"]["minItems"] = 1
        definitions["ProgressiveSkeletonStep"] = {
            "anyOf": [generic_custom, scientific_sensitivity]
        }
    elif locked_module_id is not None:
        definitions["ProgressiveSkeletonStep"] = standard
    else:
        definitions["ProgressiveSkeletonStep"] = {"anyOf": [standard, custom]}


def _exact_string_array(values: Sequence[str]) -> dict[str, Any]:
    normalized = [str(value).strip() for value in values]
    schema: dict[str, Any] = {
        "type": "array",
        "minItems": len(normalized),
        "maxItems": len(normalized),
    }
    if normalized:
        # OpenAI Structured Outputs supports array cardinality constraints but
        # not JSON Schema's tuple-only ``prefixItems`` keyword.  Restrict the
        # item vocabulary here; the progressive compiler's coordinate check
        # remains the final authority for order, duplicates, and exact values.
        schema["items"] = (
            {"type": "string", "const": normalized[0]}
            if len(normalized) == 1
            else _string_enum(normalized)
        )
    else:
        # ``maxItems: 0`` locks the value to the exact empty array; retain an
        # item schema so the array remains closed and provider-portable.
        schema["items"] = {"type": "string"}
    return schema


def _bind_outline_authorities(
    schema: dict[str, Any],
    definitions: dict[str, Any],
    *,
    analysis_types: tuple[str, ...],
    variable_names: tuple[str, ...],
    scientific_action_ids: tuple[str, ...],
    allowed_citation_keys: tuple[str, ...],
) -> None:
    properties = schema.get("properties")
    step = definitions.get("ProgressiveOutlineStep")
    if not isinstance(properties, dict) or not isinstance(step, dict):
        raise ProgressiveTransportSchemaError(
            "progressive outline properties are unavailable"
        )
    step_properties = step.get("properties")
    if not isinstance(step_properties, dict):
        raise ProgressiveTransportSchemaError(
            "progressive outline step properties are unavailable"
        )
    properties["analysis_type"] = _string_enum(analysis_types)
    step_properties["module_id"] = _string_enum(
        progressive_module_ids_for_analysis_types(analysis_types)
    )
    step_properties["variable_names"]["items"] = _string_enum(variable_names)
    citations = step_properties["literature_citation_keys"]
    if allowed_citation_keys:
        citations["items"] = _string_enum(allowed_citation_keys)
    else:
        citations["maxItems"] = 0
    step_properties["scientific_action_id"] = (
        _nullable(_string_enum(scientific_action_ids))
        if scientific_action_ids
        else {"type": "null"}
    )


def _bind_foundation_authorities(
    definitions: dict[str, Any],
    *,
    variable_names: tuple[str, ...],
    complete_case_variable_names: tuple[str, ...],
    cohort_concept_ids: tuple[str, ...],
    know_how_authority: tuple[tuple[str, str, str, str, tuple[str, ...]], ...],
    required_cohort_selection_mode: str | None,
    required_cohort_name: str | None,
    analysis_type: str | None,
) -> None:
    foundation = definitions.get("ProgressivePlanFoundation")
    cohort = definitions.get("ProgressiveCohortIntent")
    robustness = definitions.get("ProgressiveRobustnessIntent")
    predicate = definitions.get("ProgressiveCohortPredicate")
    if not all(
        isinstance(value, dict)
        for value in (foundation, cohort, robustness, predicate)
    ):
        raise ProgressiveTransportSchemaError(
            "progressive materialization foundation definitions are unavailable"
        )
    foundation_properties = foundation.get("properties")
    cohort_properties = cohort.get("properties")
    robustness_properties = robustness.get("properties")
    predicate_properties = predicate.get("properties")
    if not all(
        isinstance(value, dict)
        for value in (
            foundation_properties,
            cohort_properties,
            robustness_properties,
            predicate_properties,
        )
    ):
        raise ProgressiveTransportSchemaError(
            "progressive materialization foundation properties are unavailable"
        )
    complete_case_variables = copy.deepcopy(
        robustness_properties["complete_case_variables"]
    )
    complete_case_variables["items"] = _string_enum(
        complete_case_variable_names
    )
    required_complete_case_variables = copy.deepcopy(complete_case_variables)
    required_complete_case_variables["minItems"] = 1
    definitions["ProgressiveRobustnessIntent"] = {
        "anyOf": [
            _closed_object(
                {
                    "spec_id": copy.deepcopy(robustness_properties["spec_id"]),
                    "axis": {"type": "string", "const": "missing"},
                    "description": copy.deepcopy(
                        robustness_properties["description"]
                    ),
                    "missing_strategy": {
                        "type": "string",
                        "const": "complete_case",
                    },
                    "complete_case_variables": required_complete_case_variables,
                }
            ),
        ]
    }
    if str(analysis_type or "").strip().casefold() == "descriptive_epidemiology":
        foundation_properties["robustness_intents"]["maxItems"] = 0
    predicate_properties["concept_id"] = _string_enum(cohort_concept_ids)
    if required_cohort_selection_mode is not None:
        if required_cohort_selection_mode not in {
            "all_input_rows",
            "predicate_filtered",
        }:
            raise ProgressiveTransportSchemaError(
                "required cohort selection mode is unavailable"
            )
        cohort_properties["selection_mode"] = {
            "type": "string",
            "const": required_cohort_selection_mode,
        }
        if required_cohort_name:
            cohort_properties["name"] = {
                "type": "string",
                "const": required_cohort_name,
            }
        if required_cohort_selection_mode == "all_input_rows":
            cohort_properties["inclusion"]["maxItems"] = 0
            cohort_properties["exclusion"]["maxItems"] = 0
    decisions = foundation_properties["know_how_decisions"]
    if not know_how_authority:
        decisions["maxItems"] = 0
        return
    definition = definitions.get("ProgressiveKnowHowDecision")
    if not isinstance(definition, dict) or not isinstance(
        definition.get("properties"), dict
    ):
        raise ProgressiveTransportSchemaError(
            "progressive know-how decision definition is unavailable"
        )
    source = definition["properties"]
    branches = []
    for card_id, version, sha256, claim_id, citation_ids in know_how_authority:
        branches.append(
            _closed_object(
                {
                    "card_id": {"type": "string", "const": card_id},
                    "card_version": {"type": "string", "const": version},
                    "card_sha256": {"type": "string", "const": sha256},
                    "claim_id": {"type": "string", "const": claim_id},
                    "disposition": copy.deepcopy(source["disposition"]),
                    "reason_code": copy.deepcopy(source["reason_code"]),
                    "rationale": copy.deepcopy(source["rationale"]),
                    "citation_ids": _exact_string_array(citation_ids),
                }
            )
        )
    definitions["ProgressiveKnowHowDecision"] = {"anyOf": branches}


def _bind_materialization_coordinate(
    schema: dict[str, Any],
    definitions: dict[str, Any],
    *,
    outline_step: ProgressiveOutlineStep,
    outline_step_sha256: str,
    available_product_refs: tuple[tuple[str, str], ...],
) -> None:
    properties = schema.get("properties")
    step = definitions.get("ProgressiveSkeletonStep")
    if not isinstance(properties, dict) or not isinstance(step, dict):
        raise ProgressiveTransportSchemaError(
            "progressive materialization properties are unavailable"
        )
    step_properties = step.get("properties")
    if not isinstance(step_properties, dict):
        raise ProgressiveTransportSchemaError(
            "progressive materialization step properties are unavailable"
        )
    properties["outline_step_sha256"] = {
        "type": "string",
        "const": outline_step_sha256,
    }
    properties["foundation"] = {"type": "null"}
    step_properties["step_id"] = {
        "type": "string",
        "const": outline_step.step_id,
    }
    step_properties["planned_analysis_role"] = {
        "type": "string",
        "const": outline_step.planned_analysis_role,
    }
    step_properties["module_id"] = {
        "type": "string",
        "const": outline_step.module_id,
    }
    step_properties["objective"] = {
        "type": "string",
        "const": outline_step.objective,
    }
    step_properties["depends_on"] = _exact_string_array(outline_step.depends_on)
    step_properties["scientific_action_id"] = (
        {
            "type": "string",
            "const": outline_step.scientific_action_id,
        }
        if outline_step.scientific_action_id is not None
        else {"type": "null"}
    )
    product_inputs = step_properties["product_inputs"]
    if not available_product_refs:
        product_inputs["maxItems"] = 0
        return
    definitions["ProgressiveProductRef"] = {
        "anyOf": [
            _closed_object(
                {
                    "producer_step_id": {"type": "string", "const": producer},
                    "product_id": {"type": "string", "const": product},
                }
            )
            for producer, product in available_product_refs
        ]
    }


def _bind_step_rosters(
    definitions: dict[str, Any],
    *,
    variable_names: tuple[str, ...],
    executable_variable_names: tuple[str, ...] | None = None,
    scientific_action_ids: tuple[str, ...],
    allowed_citation_keys: tuple[str, ...],
) -> None:
    step = definitions.get("ProgressiveSkeletonStep")
    table_variable = definitions.get("ProgressiveTableOneVariable")
    model_term = definitions.get("ProgressiveModelTermIntent")
    literature = definitions.get("ProgressiveLiteratureBinding")
    if not all(
        isinstance(value, dict)
        for value in (step, table_variable, model_term, literature)
    ):
        raise ProgressiveTransportSchemaError(
            "progressive skeleton definitions are unavailable"
        )
    step_properties = step.get("properties")
    table_properties = table_variable.get("properties")
    model_properties = model_term.get("properties")
    literature_properties = literature.get("properties")
    if not all(
        isinstance(value, dict)
        for value in (
            step_properties,
            table_properties,
            model_properties,
            literature_properties,
        )
    ):
        raise ProgressiveTransportSchemaError(
            "progressive skeleton property maps are unavailable"
        )
    variable = _string_enum(variable_names)
    executable_variable = _string_enum(
        executable_variable_names
        if executable_variable_names is not None
        else variable_names
    )
    step_properties["raw_inputs"]["items"] = copy.deepcopy(variable)
    for field in ("table_one_group_by", "primary_exposure", "outcome"):
        step_properties[field] = _nullable(executable_variable)
    table_properties["name"] = copy.deepcopy(executable_variable)
    model_properties["name"] = copy.deepcopy(executable_variable)
    action_schema: dict[str, Any]
    if scientific_action_ids:
        action_schema = _nullable(_string_enum(scientific_action_ids))
    else:
        action_schema = {"type": "null"}
    step_properties["scientific_action_id"] = action_schema

    bindings = step_properties["literature_bindings"]
    if not allowed_citation_keys:
        bindings["maxItems"] = 0
        return
    global_elements = literature_properties["design_elements"]["items"].get("enum")
    if not isinstance(global_elements, list) or not global_elements:
        raise ProgressiveTransportSchemaError(
            "progressive literature design-element vocabulary is not closed"
        )
    method_elements: dict[str, set[str]] = {}
    for card in METHOD_CARDS:
        method_elements.setdefault(card.source_key, set()).update(card.design_elements)
    groups: dict[tuple[str, ...], list[str]] = {}
    for key in allowed_citation_keys:
        elements = tuple(sorted(method_elements.get(key, set(global_elements))))
        groups.setdefault(elements, []).append(key)
    branches: list[dict[str, Any]] = []
    for elements, keys in groups.items():
        element_schema = copy.deepcopy(literature_properties["design_elements"])
        element_schema["items"] = {"type": "string", "enum": list(elements)}
        branches.append(
            _closed_object(
                {
                    "citation_key": _string_enum(keys),
                    "design_elements": element_schema,
                    "application": copy.deepcopy(literature_properties["application"]),
                    "divergence": copy.deepcopy(literature_properties["divergence"]),
                }
            )
        )
    definitions["ProgressiveLiteratureBinding"] = {"anyOf": branches}


def _bind_initial_authorities(
    schema: dict[str, Any],
    definitions: dict[str, Any],
    *,
    analysis_types: tuple[str, ...],
    variable_names: tuple[str, ...],
    cohort_concept_ids: tuple[str, ...],
    know_how_authority: tuple[tuple[str, str, str, str, tuple[str, ...]], ...],
) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ProgressiveTransportSchemaError(
            "progressive plan root properties are unavailable"
        )
    properties["analysis_type"] = _string_enum(analysis_types)
    robustness = definitions.get("ProgressiveRobustnessIntent")
    predicate = definitions.get("ProgressiveCohortPredicate")
    if not isinstance(robustness, dict) or not isinstance(predicate, dict):
        raise ProgressiveTransportSchemaError(
            "progressive initial-only definitions are unavailable"
        )
    robustness["properties"]["complete_case_variables"]["items"] = _string_enum(
        variable_names
    )
    # A predicate concept may bind through a source concept instead of a raw
    # column.  The compiler/materializer remains the final authority; the run
    # roster only prevents free-form inventions at transport.
    predicate["properties"]["concept_id"] = _string_enum(cohort_concept_ids)

    decisions = properties["know_how_decisions"]
    if not know_how_authority:
        decisions["maxItems"] = 0
        return
    definition = definitions.get("ProgressiveKnowHowDecision")
    if not isinstance(definition, dict) or not isinstance(
        definition.get("properties"), dict
    ):
        raise ProgressiveTransportSchemaError(
            "progressive know-how decision definition is unavailable"
        )
    source = definition["properties"]
    branches = []
    for card_id, version, sha256, claim_id, citation_ids in know_how_authority:
        branches.append(
            _closed_object(
                {
                    "card_id": {"type": "string", "const": card_id},
                    "card_version": {"type": "string", "const": version},
                    "card_sha256": {"type": "string", "const": sha256},
                    "claim_id": {"type": "string", "const": claim_id},
                    "disposition": copy.deepcopy(source["disposition"]),
                    "reason_code": copy.deepcopy(source["reason_code"]),
                    "rationale": copy.deepcopy(source["rationale"]),
                    "citation_ids": _exact_string_array(citation_ids),
                }
            )
        )
    definitions["ProgressiveKnowHowDecision"] = {"anyOf": branches}


def _authority_rows(
    authority: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[tuple[str, str, str, str, tuple[str, ...]], ...]:
    rows: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    for card_id, card in sorted((authority or {}).items()):
        claims = card.get("claims") or {}
        if not isinstance(claims, Mapping):
            raise ProgressiveTransportSchemaError(
                f"know-how authority {card_id!r} has no claim mapping"
            )
        for claim_id, citation_ids in sorted(claims.items()):
            rows.append(
                (
                    str(card_id),
                    str(card.get("version") or ""),
                    str(card.get("file_sha256") or ""),
                    str(claim_id),
                    tuple(str(value) for value in citation_ids or ()),
                )
            )
    return tuple(rows)


def _closed_request(*, name: str, schema: dict[str, Any]) -> StructuredOutputRequest:
    strictify_json_schema(schema)
    try:
        assert_closed_json_schema(schema)
    except StrictJsonSchemaError as exc:
        raise ProgressiveTransportSchemaError(str(exc)) from exc
    return StructuredOutputRequest.from_schema(
        name=name,
        schema=schema,
        strict=True,
    )


def progressive_outline_structured_output_request(
    *,
    analysis_types: Sequence[str],
    variable_names: Sequence[str],
    scientific_action_ids: Sequence[str],
    allowed_literature_citation_keys: Sequence[str] = (),
) -> StructuredOutputRequest:
    """Return the tiny run-bound schema used for the first Planner response."""

    normalized_types = tuple(
        dict.fromkeys(
            str(value).strip() for value in analysis_types if str(value).strip()
        )
    )
    normalized_actions = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in scientific_action_ids
            if str(value).strip()
        )
    )
    normalized_variables = tuple(
        dict.fromkeys(
            str(value).strip() for value in variable_names if str(value).strip()
        )
    )
    normalized_citations = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in allowed_literature_citation_keys
            if str(value).strip()
        )
    )
    if not normalized_types or not normalized_variables:
        raise ProgressiveTransportSchemaError(
            "progressive outline transport requires analysis-type and variable rosters"
        )
    schema = copy.deepcopy(ProgressivePlanOutline.model_json_schema(mode="validation"))
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise ProgressiveTransportSchemaError("progressive outline schema has no $defs")
    _bind_outline_authorities(
        schema,
        definitions,
        analysis_types=normalized_types,
        variable_names=normalized_variables,
        scientific_action_ids=normalized_actions,
        allowed_citation_keys=normalized_citations,
    )
    return _closed_request(
        name="easyicu_progressive_plan_outline_v1",
        schema=schema,
    )


def progressive_foundation_structured_output_request(
    *,
    outline_sha256: str,
    variable_names: Sequence[str],
    complete_case_variable_names: Sequence[str] | None = None,
    cohort_concept_ids: Sequence[str] = (),
    allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None = None,
    required_cohort_selection_mode: str | None = None,
    required_cohort_name: str | None = None,
    analysis_type: str | None = None,
) -> StructuredOutputRequest:
    """Return the run-bound plan-wide contract without any step fields."""

    if not re.fullmatch(r"[0-9a-f]{64}", str(outline_sha256 or "")):
        raise ProgressiveTransportSchemaError(
            "outline_sha256 must be one canonical sha256"
        )
    normalized_variables = tuple(
        dict.fromkeys(
            str(value).strip() for value in variable_names if str(value).strip()
        )
    )
    if not normalized_variables:
        raise ProgressiveTransportSchemaError(
            "progressive foundation transport requires a variable roster"
        )
    normalized_concepts = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in (cohort_concept_ids or normalized_variables)
            if str(value).strip()
        )
    )
    normalized_complete_case_variables = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in (
                complete_case_variable_names
                if complete_case_variable_names is not None
                else normalized_variables
            )
            if str(value).strip()
        )
    )
    if not normalized_complete_case_variables:
        raise ProgressiveTransportSchemaError(
            "progressive foundation requires at least one eligible "
            "complete-case variable"
        )
    if not set(normalized_complete_case_variables).issubset(normalized_variables):
        raise ProgressiveTransportSchemaError(
            "complete-case variable roster must be a subset of the run roster"
        )
    schema = copy.deepcopy(
        ProgressiveFoundationMaterialization.model_json_schema(mode="validation")
    )
    properties = schema.get("properties")
    definitions = schema.get("$defs")
    if not isinstance(properties, dict) or not isinstance(definitions, dict):
        raise ProgressiveTransportSchemaError(
            "progressive foundation schema properties are unavailable"
        )
    properties["outline_sha256"] = {
        "type": "string",
        "const": str(outline_sha256),
    }
    _bind_foundation_authorities(
        definitions,
        variable_names=normalized_variables,
        complete_case_variable_names=normalized_complete_case_variables,
        cohort_concept_ids=normalized_concepts,
        know_how_authority=_authority_rows(allowed_know_how_decisions),
        required_cohort_selection_mode=required_cohort_selection_mode,
        required_cohort_name=required_cohort_name,
        analysis_type=analysis_type,
    )
    return _closed_request(
        name="easyicu_progressive_plan_foundation_v1",
        schema=schema,
    )


def progressive_step_materialization_request(
    *,
    outline_step: ProgressiveOutlineStep,
    outline_step_sha256: str,
    variable_names: Sequence[str],
    executable_variable_names: Sequence[str] | None = None,
    scientific_action_ids: Sequence[str],
    allowed_literature_citation_keys: Sequence[str] = (),
    available_product_refs: Sequence[tuple[str, str]] = (),
) -> StructuredOutputRequest:
    """Return one coordinate-bound schema for the current step only."""

    if not re.fullmatch(r"[0-9a-f]{64}", str(outline_step_sha256 or "")):
        raise ProgressiveTransportSchemaError(
            "outline_step_sha256 must be one canonical sha256"
        )
    normalized_variables = tuple(
        dict.fromkeys(
            str(value).strip() for value in variable_names if str(value).strip()
        )
    )
    normalized_actions = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in scientific_action_ids
            if str(value).strip()
        )
    )
    normalized_executable_variables = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in (
                executable_variable_names
                if executable_variable_names is not None
                else normalized_variables
            )
            if str(value).strip()
        )
    )
    normalized_citations = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in allowed_literature_citation_keys
            if str(value).strip()
        )
    )
    if not normalized_variables:
        raise ProgressiveTransportSchemaError(
            "progressive step transport requires a variable roster"
        )
    if not normalized_executable_variables:
        if outline_step.module_id in {
            "adjusted_association",
            "exposure_outcome_distribution",
            "table_one",
        }:
            raise ProgressiveTransportSchemaError(
                "statistical step transport requires an executable variable roster"
            )
        # Non-statistical module shapes replace these definitions with null
        # fields below. Keep their unused base definitions provider-valid even
        # when the outline contains only a host navigation coordinate.
        normalized_executable_variables = normalized_variables
    if not set(normalized_executable_variables).issubset(normalized_variables):
        raise ProgressiveTransportSchemaError(
            "executable variable roster must be a subset of the step roster"
        )
    if (
        outline_step.scientific_action_id is not None
        and outline_step.scientific_action_id not in normalized_actions
    ):
        raise ProgressiveTransportSchemaError(
            "outline scientific action is outside the run-bound action roster"
        )
    normalized_products: list[tuple[str, str]] = []
    for producer, product in available_product_refs:
        producer_id = str(producer or "").strip()
        product_id = str(product or "").strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,79}", producer_id):
            raise ProgressiveTransportSchemaError(
                f"invalid available product producer {producer_id!r}"
            )
        if not re.fullmatch(r"[a-z][a-z0-9_]*:[a-z][a-z0-9_]*", product_id):
            raise ProgressiveTransportSchemaError(
                f"invalid available product token {product_id!r}"
            )
        coordinate = (producer_id, product_id)
        if coordinate not in normalized_products:
            normalized_products.append(coordinate)
    schema = copy.deepcopy(
        ProgressiveStepMaterialization.model_json_schema(mode="validation")
    )
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise ProgressiveTransportSchemaError(
            "progressive materialization schema has no $defs"
        )
    _bind_step_rosters(
        definitions,
        variable_names=normalized_variables,
        executable_variable_names=normalized_executable_variables,
        scientific_action_ids=normalized_actions,
        allowed_citation_keys=normalized_citations,
    )
    step_definition = definitions.get("ProgressiveSkeletonStep")
    step_properties = (
        step_definition.get("properties")
        if isinstance(step_definition, dict)
        else None
    )
    if not isinstance(step_properties, dict) or not isinstance(
        step_properties.get("literature_bindings"), dict
    ):
        raise ProgressiveTransportSchemaError(
            "progressive step literature roster is unavailable"
        )
    step_properties["literature_bindings"]["minItems"] = len(
        normalized_citations
    )
    step_properties["literature_bindings"]["maxItems"] = len(
        normalized_citations
    )
    _bind_materialization_coordinate(
        schema,
        definitions,
        outline_step=outline_step,
        outline_step_sha256=outline_step_sha256,
        available_product_refs=tuple(normalized_products),
    )
    _bind_step_module_shape(
        definitions,
        locked_module_id=outline_step.module_id,
        locked_has_dependencies=bool(outline_step.depends_on),
        locked_has_available_products=bool(normalized_products),
    )
    return _closed_request(
        name="easyicu_progressive_step_materialization_v1",
        schema=schema,
    )


@lru_cache(maxsize=64)
def _request_cached(
    kind: str,
    analysis_types: tuple[str, ...],
    variable_names: tuple[str, ...],
    cohort_concept_ids: tuple[str, ...],
    scientific_action_ids: tuple[str, ...],
    allowed_citation_keys: tuple[str, ...],
    know_how_authority: tuple[tuple[str, str, str, str, tuple[str, ...]], ...],
) -> StructuredOutputRequest:
    model = ProgressivePlanSkeleton if kind == "initial" else ProgressiveSuffixRevision
    schema = copy.deepcopy(model.model_json_schema(mode="validation"))
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise ProgressiveTransportSchemaError("progressive schema has no $defs")
    _bind_step_rosters(
        definitions,
        variable_names=variable_names,
        scientific_action_ids=scientific_action_ids,
        allowed_citation_keys=allowed_citation_keys,
    )
    if kind == "initial":
        _bind_initial_authorities(
            schema,
            definitions,
            analysis_types=analysis_types,
            variable_names=variable_names,
            cohort_concept_ids=cohort_concept_ids,
            know_how_authority=know_how_authority,
        )
    _bind_step_module_shape(definitions)
    strictify_json_schema(schema)
    try:
        assert_closed_json_schema(schema)
    except StrictJsonSchemaError as exc:
        raise ProgressiveTransportSchemaError(str(exc)) from exc
    return StructuredOutputRequest.from_schema(
        name=(
            "easyicu_progressive_plan_skeleton_v1"
            if kind == "initial"
            else "easyicu_progressive_plan_suffix_v1"
        ),
        schema=schema,
        strict=True,
    )


def progressive_structured_output_request(
    *,
    analysis_types: Sequence[str],
    variable_names: Sequence[str],
    cohort_concept_ids: Sequence[str] = (),
    scientific_action_ids: Sequence[str],
    allowed_literature_citation_keys: Sequence[str] = (),
    allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None = None,
    suffix: bool = False,
) -> StructuredOutputRequest:
    """Return an immutable strict schema narrowed to this run's authorities."""

    normalized_types = tuple(
        dict.fromkeys(
            str(value).strip() for value in analysis_types if str(value).strip()
        )
    )
    normalized_variables = tuple(
        dict.fromkeys(
            str(value).strip() for value in variable_names if str(value).strip()
        )
    )
    normalized_actions = tuple(
        dict.fromkeys(
            str(value).strip() for value in scientific_action_ids if str(value).strip()
        )
    )
    normalized_citations = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in allowed_literature_citation_keys
            if str(value).strip()
        )
    )
    if not normalized_types or not normalized_variables:
        raise ProgressiveTransportSchemaError(
            "progressive transport requires analysis-type and variable rosters"
        )
    normalized_concepts = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in (cohort_concept_ids or normalized_variables)
            if str(value).strip()
        )
    )
    return _request_cached(
        "suffix" if suffix else "initial",
        normalized_types,
        normalized_variables,
        normalized_concepts,
        normalized_actions,
        normalized_citations,
        _authority_rows(allowed_know_how_decisions) if not suffix else (),
    )


__all__ = [
    "ProgressiveTransportSchemaError",
    "progressive_foundation_structured_output_request",
    "progressive_outline_structured_output_request",
    "progressive_step_materialization_request",
    "progressive_structured_output_request",
]
