"""Run-bound strict transport schema for Progressive Planner v2."""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any, Mapping, Sequence, get_args

from ..planning.method_literature import METHOD_CARDS
from ..planning.progressive_contract import (
    ProgressiveModuleId,
    ProgressivePlanSkeleton,
    ProgressiveSuffixRevision,
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


def _bind_step_module_shape(definitions: dict[str, Any]) -> None:
    """Compile the one cross-field rule the base schema cannot express cheaply.

    A standard module's method is host-owned, so ``custom_method`` is not an
    alternate spelling of that method and must be null.  ``custom_analysis`` is
    the only branch that accepts free-form method text.  Keeping its branch to
    the fields it can actually use avoids cloning the entire step schema and
    keeps the run-bound authority compact.
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

    standard = copy.deepcopy(step)
    standard["properties"]["module_id"] = _string_enum(standard_ids)
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
    definitions["ProgressiveSkeletonStep"] = {"anyOf": [standard, custom]}


def _bind_step_rosters(
    definitions: dict[str, Any],
    *,
    variable_names: tuple[str, ...],
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
    step_properties["raw_inputs"]["items"] = copy.deepcopy(variable)
    for field in ("table_one_group_by", "primary_exposure", "outcome"):
        step_properties[field] = _nullable(variable)
    table_properties["name"] = copy.deepcopy(variable)
    model_properties["name"] = copy.deepcopy(variable)
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
                    "citation_ids": {
                        "type": "array",
                        "prefixItems": [
                            {"type": "string", "const": value} for value in citation_ids
                        ],
                        "minItems": len(citation_ids),
                        "maxItems": len(citation_ids),
                    },
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
    "progressive_structured_output_request",
]
