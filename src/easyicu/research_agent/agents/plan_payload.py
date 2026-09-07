"""Schema-driven projection of raw Planner payloads.

This module owns the boundary between untrusted model JSON and the strict
``AnalysisPlan`` schema. It may discard presentation-only top/step chatter,
but an unknown key inside a scientific contract is a structured retry: silently
projecting it away could turn the Planner's intended design into a different,
valid-looking design. It normalizes only representation-level aliases.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..contracts.declared_product import (
    PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS,
    RUNTIME_BINDABLE_TYPED_INPUT_KINDS,
    typed_product as _canonical_typed_product,
)
from ..contracts.product_identity import CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN
from ..contracts.capability_ids import CAPABILITY_FAMILIES
from ..planning.method_literature import METHOD_CARDS
from ..planning.literature_bindings import (
    allowed_method_source_keys,
    normalize_literature_citation_keys,
    validate_literature_citation_bindings,
)
from ..planning.primary_result_contract import model_terms_retry_guide
from ..planning.runtime_suffix import RuntimePlanSuffixRevision
from ..planning.robustness_contract import (
    PLANNER_MISSING_OVERRIDE_FIELDS,
    PLANNER_OUTCOME_OVERRIDE_FIELDS,
    RobustnessSpec,
)
from ..providers.protocol import StructuredOutputRequest
from ..providers.strict_json_schema import (
    StrictJsonSchemaError,
    assert_closed_json_schema,
    strictify_json_schema,
)
from ..planning.scientific_review import (
    required_method_layers_for_context,
    required_method_layers_for_plan,
)
from ..schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    ExposureOutcomeDistributionSpec,
    ExposureOutcomeRiskDifferenceContrast,
    PlannedFigurePanelSpec,
    PlannedModelRequirement,
    TableOneSpec,
    TableOneVariableSpec,
)
from ..research_context.prompt_variables import opaque_level_tokens


class PlannerStructuredOutputSchemaError(ValueError):
    """The host could not derive a closed Planner transport schema."""


def _json_scalar_schema() -> Dict[str, Any]:
    """The exact finite scalar family accepted by closed-level validators."""

    return {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "boolean"},
        ]
    }


def _nullable_string_schema() -> Dict[str, Any]:
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _nullable_string_list_schema() -> Dict[str, Any]:
    return {
        "anyOf": [
            {"type": "array", "items": {"type": "string"}},
            {"type": "null"},
        ]
    }


def _closed_object_schema(properties: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _artifact_consumption_transport_schema(
    definition: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compile the host's mode-dependent consumption contract for transport."""

    properties = definition.get("properties")
    expected = {
        "schema_version",
        "input_key",
        "mode",
        "role_column",
        "expected_roles",
    }
    if not isinstance(properties, dict) or set(properties) != expected:
        raise PlannerStructuredOutputSchemaError(
            "ArtifactConsumptionContract schema drifted from its wire compiler"
        )

    def branch(
        modes: tuple[str, ...],
        *,
        role_column: Dict[str, Any],
        expected_roles: Dict[str, Any],
    ) -> Dict[str, Any]:
        input_key = copy.deepcopy(properties["input_key"])
        input_key["pattern"] = CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN
        return _closed_object_schema(
            {
                "schema_version": copy.deepcopy(properties["schema_version"]),
                "input_key": input_key,
                "mode": {
                    "type": "string",
                    **({"const": modes[0]} if len(modes) == 1 else {"enum": list(modes)}),
                },
                "role_column": role_column,
                "expected_roles": expected_roles,
            }
        )

    empty_roles = {"type": "array", "items": {"type": "string"}, "maxItems": 0}
    return {
        "anyOf": [
            branch(
                ("all_rows", "single_row"),
                role_column={"type": "null"},
                expected_roles=copy.deepcopy(empty_roles),
            ),
            branch(
                ("one_per_role",),
                role_column={"type": "string", "minLength": 1},
                expected_roles={
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
            ),
        ]
    }


def _literature_design_binding_transport_schema(
    definition: Mapping[str, Any],
    allowed_keys: Sequence[str],
) -> Dict[str, Any]:
    """Compile exact run-bound source/element pairs for strict transport.

    Method sources have a closed host-curated design vocabulary.  Leaving the
    two fields independent in JSON Schema lets a provider emit combinations
    that the unchanged literature authority must reject.  Topic and screened
    comparator sources remain eligible for the full typed design-element
    vocabulary because their relevance is judged from the sealed source
    excerpt after transport, not from method cards.
    """

    properties = definition.get("properties")
    expected = {
        "citation_key",
        "design_elements",
        "application",
        "divergence",
    }
    if not isinstance(properties, dict) or set(properties) != expected:
        raise PlannerStructuredOutputSchemaError(
            "LiteratureDesignBinding schema drifted from its wire compiler"
        )
    design_elements = properties["design_elements"]
    if not isinstance(design_elements, dict):
        raise PlannerStructuredOutputSchemaError(
            "LiteratureDesignBinding design_elements schema is not an array"
        )
    items = design_elements.get("items")
    global_elements = items.get("enum") if isinstance(items, dict) else None
    if not isinstance(global_elements, list) or not global_elements:
        raise PlannerStructuredOutputSchemaError(
            "LiteratureDesignBinding design-element vocabulary is not closed"
        )

    method_elements: dict[str, set[str]] = {}
    for card in METHOD_CARDS:
        method_elements.setdefault(card.source_key, set()).update(
            card.design_elements
        )

    # Several sources can have the exact same authority (for example the two
    # time-alignment cards, or all non-method sources whose excerpts are judged
    # later). Group only identical element sets so the source/element relation
    # is unchanged while the bounded schema does not repeat the application and
    # divergence fields once per citation.
    source_groups: dict[tuple[str, ...], list[str]] = {}
    for source_key in allowed_keys:
        allowed_elements = tuple(
            sorted(method_elements.get(source_key, set(global_elements)))
        )
        source_groups.setdefault(allowed_elements, []).append(source_key)

    branches: list[Dict[str, Any]] = []
    for allowed_elements, source_keys in source_groups.items():
        branch_elements = copy.deepcopy(design_elements)
        branch_elements["items"] = {
            "type": "string",
            "enum": list(allowed_elements),
        }
        branches.append(
            _closed_object_schema(
                {
                    "citation_key": {
                        "type": "string",
                        "enum": source_keys,
                    },
                    "design_elements": branch_elements,
                    "application": copy.deepcopy(properties["application"]),
                    "divergence": copy.deepcopy(properties["divergence"]),
                }
            )
        )
    if not branches:
        raise PlannerStructuredOutputSchemaError(
            "run-bound literature binding schema requires at least one source"
        )
    return {"anyOf": branches}


def _bind_literature_transport_authority(
    definitions: Mapping[str, Any],
    allowed_keys: Sequence[str],
) -> None:
    """Bind citation arrays and nested records to one sealed run roster."""

    analysis_step = definitions.get("AnalysisStep")
    if not isinstance(analysis_step, dict):
        raise PlannerStructuredOutputSchemaError(
            "AnalysisStep schema is unavailable for literature authority"
        )
    step_properties = analysis_step.get("properties")
    if not isinstance(step_properties, dict):
        raise PlannerStructuredOutputSchemaError(
            "AnalysisStep properties are unavailable for literature authority"
        )
    citation_roster = step_properties.get("literature_citation_keys")
    design_bindings = step_properties.get("literature_design_bindings")
    if not isinstance(citation_roster, dict) or not isinstance(
        design_bindings, dict
    ):
        raise PlannerStructuredOutputSchemaError(
            "AnalysisStep literature fields drifted from their wire compiler"
        )

    keys = tuple(allowed_keys)
    if not keys:
        citation_roster["maxItems"] = 0
        design_bindings["maxItems"] = 0
        return

    citation_roster["items"] = {"type": "string", "enum": list(keys)}
    binding_definition = definitions.get("LiteratureDesignBinding")
    if not isinstance(binding_definition, dict):
        raise PlannerStructuredOutputSchemaError(
            "LiteratureDesignBinding schema is unavailable"
        )
    definitions["LiteratureDesignBinding"] = (
        _literature_design_binding_transport_schema(binding_definition, keys)
    )


def _strictify_planner_transport_schema(node: Any) -> None:
    """Compatibility wrapper over the provider-neutral strict-schema owner."""

    strictify_json_schema(node)


def _assert_closed_planner_transport_schema(node: Any, *, path: str = "$") -> None:
    try:
        assert_closed_json_schema(node, path=path)
    except StrictJsonSchemaError as exc:
        raise PlannerStructuredOutputSchemaError(str(exc)) from exc


def _share_planner_transport_shapes(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Reference repeated scalar shapes without changing their accepted values."""

    shared = {
        "JsonScalar": _json_scalar_schema(),
        "NullableText": _nullable_string_schema(),
        "NullableTextList": _nullable_string_list_schema(),
    }
    used: set[str] = set()

    def compile_node(node: Any) -> Any:
        if isinstance(node, dict):
            for name, shape in shared.items():
                if node == shape:
                    used.add(name)
                    return {"$ref": f"#/$defs/{name}"}
            return {key: compile_node(value) for key, value in node.items()}
        if isinstance(node, list):
            return [compile_node(value) for value in node]
        return node

    compiled = compile_node(schema)
    definitions = compiled["$defs"]
    for name in sorted(used):
        if name in definitions:
            raise PlannerStructuredOutputSchemaError(
                f"Planner transport shape name collides with an authority model: {name}"
            )
        definitions[name] = shared[name]
    return compiled


def _planner_transport_schema(
    allowed_literature_citation_keys: tuple[str, ...] | None = None,
) -> Dict[str, Any]:
    """Return a strict transport schema derived from the authority model.

    Pydantic's validation schema contains open maps and several ``Any``
    level values. Those are not representable in strict JSON Schema. The
    replacements below do not choose science: they expose the exact scalar
    family and robustness keys already consumed by their owner validators.
    ``display_labels`` and model covariate decisions travel as key/value rows
    and are decoded back to public mappings before Pydantic validation.
    """

    schema = copy.deepcopy(AnalysisPlan.model_json_schema(mode="validation"))
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise PlannerStructuredOutputSchemaError("AnalysisPlan schema has no $defs")
    try:
        # Candidate-design comparison is owned by the Progressive Planner v2
        # outline. The classic one-shot Planner remains load-compatible with
        # the optional public field but must not repeat this high-entropy
        # authority inside its already budget-constrained transport schema.
        schema["properties"].pop("design_selection", None)
        definitions.pop("ResearchDesignSelection", None)
        definitions.pop("ResearchDesignCandidate", None)
        definitions.pop("CandidateLiteratureDesignDecision", None)
        robustness = definitions["RobustnessSpec"]["properties"]
        missing_override = _closed_object_schema(
            {
                "strategy": _nullable_string_schema(),
                "variables": _nullable_string_list_schema(),
                "audit_flags": _nullable_string_list_schema(),
            }
        )
        outcome_override = _closed_object_schema(
            {
                field: _nullable_string_schema()
                for field in PLANNER_OUTCOME_OVERRIDE_FIELDS
            }
        )
        if tuple(missing_override["properties"]) != tuple(
            PLANNER_MISSING_OVERRIDE_FIELDS
        ):
            raise PlannerStructuredOutputSchemaError(
                "missingness override schema drifted from its owner contract"
            )
        robustness["missing_override"] = {
            "anyOf": [missing_override, {"type": "null"}]
        }
        robustness["outcome_override"] = {
            "anyOf": [outcome_override, {"type": "null"}]
        }

        label_entry = _closed_object_schema(
            {
                "key": {"type": "string", "minLength": 1, "maxLength": 256},
                "value": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 256,
                },
            }
        )
        schema["properties"]["display_labels"] = {
            "type": "array",
            "items": label_entry,
        }
        model_requirement = definitions["PlannedModelRequirement"]["properties"]
        for field in ("covariate_rationales", "covariate_temporal_roles"):
            value_schema = copy.deepcopy(model_requirement[field]["additionalProperties"])
            model_requirement[field] = {
                "type": "array",
                "items": _closed_object_schema(
                    {
                        "key": {"type": "string", "minLength": 1},
                        "value": value_schema,
                    }
                ),
            }

        concept_value = definitions["ConceptPredicate"]["properties"]
        concept_value["value"] = {
            "anyOf": [
                _json_scalar_schema(),
                {"type": "array", "items": _json_scalar_schema()},
                {"type": "null"},
            ]
        }
        definitions["EndpointSpec"]["properties"]["levels"]["anyOf"][0][
            "items"
        ] = _json_scalar_schema()
        distribution = definitions["ExposureOutcomeDistributionSpec"]["properties"]
        distribution["outcome_positive_value"] = _json_scalar_schema()
        distribution["exposure_levels"]["items"] = _json_scalar_schema()
        distribution["outcome_levels"]["items"] = _json_scalar_schema()
        contrast = definitions["ExposureOutcomeRiskDifferenceContrast"]["properties"]
        contrast["reference_exposure_level"] = _json_scalar_schema()
        contrast["comparison_exposure_level"] = _json_scalar_schema()
        definitions["TableOneSpec"]["properties"]["group_levels"][
            "items"
        ] = _json_scalar_schema()
        definitions["TableOneVariableSpec"]["properties"]["levels"][
            "items"
        ] = _json_scalar_schema()
        # ``AnalysisStep`` validates this field against the stable capability
        # owner after transport.  Publishing only ``string | null`` here let a
        # strict-schema provider spend a complete response on prose labels that
        # the host was guaranteed to reject.  Derive the wire enum from the
        # same dependency-neutral vocabulary so transport and host validation
        # cannot drift into two authorities.
        definitions["AnalysisStep"]["properties"]["scientific_capability"] = {
            "anyOf": [
                {"type": "string", "enum": sorted(CAPABILITY_FAMILIES)},
                {"type": "null"},
            ]
        }
        definitions["ArtifactConsumptionContract"] = (
            _artifact_consumption_transport_schema(
                definitions["ArtifactConsumptionContract"]
            )
        )
        if allowed_literature_citation_keys is not None:
            _bind_literature_transport_authority(
                definitions,
                allowed_literature_citation_keys,
            )
    except (KeyError, TypeError) as exc:
        raise PlannerStructuredOutputSchemaError(
            "AnalysisPlan schema shape changed; review the transport projection"
        ) from exc

    _strictify_planner_transport_schema(schema)
    schema = _share_planner_transport_shapes(schema)
    _assert_closed_planner_transport_schema(schema)
    return schema


@lru_cache(maxsize=64)
def _planner_structured_output_request_cached(
    allowed_literature_citation_keys: tuple[str, ...] | None,
) -> StructuredOutputRequest:
    """Cache one immutable authority per normalized run-bound source roster."""

    return StructuredOutputRequest.from_schema(
        name="easyicu_analysis_plan_v1",
        schema=_planner_transport_schema(allowed_literature_citation_keys),
        strict=True,
    )


def planner_structured_output_request(
    allowed_literature_citation_keys: Sequence[str] | None = None,
) -> StructuredOutputRequest:
    """Return the strict schema bound to the run's sealed citation roster.

    ``None`` preserves the generic schema for offline inspection and budget
    baselines.  A Planner run passes an explicit (possibly empty) roster so the
    provider cannot invent a source or attach a curated method source to an
    unsupported design element.
    """

    normalized = (
        None
        if allowed_literature_citation_keys is None
        else normalize_literature_citation_keys(
            allowed_literature_citation_keys
        )
    )
    return _planner_structured_output_request_cached(normalized)


@lru_cache(maxsize=64)
def _runtime_suffix_structured_output_request_cached(
    allowed_literature_citation_keys: tuple[str, ...],
    replace_from_step_id: str,
    planned_analysis_role: str,
    allowed_inputs: tuple[str, ...],
    expected_outputs: tuple[str, ...],
    scientific_action_ids: tuple[str, ...],
) -> StructuredOutputRequest:
    """Compile one strict, coordinate-bound runtime step response schema."""

    plan_schema = _planner_transport_schema(allowed_literature_citation_keys)
    definitions = plan_schema.get("$defs")
    if not isinstance(definitions, dict):  # pragma: no cover - owner assertion
        raise PlannerStructuredOutputSchemaError(
            "runtime suffix plan schema has no definitions"
        )
    analysis_step = definitions.get("AnalysisStep")
    if not isinstance(analysis_step, dict) or not isinstance(
        analysis_step.get("properties"), dict
    ):
        raise PlannerStructuredOutputSchemaError(
            "runtime suffix AnalysisStep schema is unavailable"
        )
    step_properties = analysis_step["properties"]
    step_properties["step_id"] = {
        "type": "string",
        "const": replace_from_step_id,
    }
    step_properties["planned_analysis_role"] = {
        "type": "string",
        "const": planned_analysis_role,
    }
    if allowed_inputs:
        step_properties["inputs"]["items"] = {
            "type": "string",
            "enum": list(allowed_inputs),
        }
    else:
        step_properties["inputs"]["maxItems"] = 0
    output_items: dict[str, Any]
    if len(expected_outputs) == 1:
        output_items = {"type": "string", "const": expected_outputs[0]}
    elif expected_outputs:
        output_items = {
            "type": "string",
            "enum": list(dict.fromkeys(expected_outputs)),
        }
    else:
        output_items = {"type": "string"}
    step_properties["expected_outputs"] = {
        "type": "array",
        # OpenAI Structured Outputs rejects ``prefixItems`` and requires every
        # array schema to declare ``items``.  Lock the vocabulary and exact
        # cardinality here; the existing host-side coordinate validator remains
        # authoritative for order and duplicate rejection.
        "items": output_items,
        "minItems": len(expected_outputs),
        "maxItems": len(expected_outputs),
    }
    step_properties["scientific_action_id"] = (
        {
            "anyOf": [
                {
                    "type": "string",
                    "enum": list(scientific_action_ids),
                },
                {"type": "null"},
            ]
        }
        if scientific_action_ids
        else {"type": "null"}
    )

    schema = copy.deepcopy(
        RuntimePlanSuffixRevision.model_json_schema(mode="validation")
    )
    schema["$defs"] = definitions
    properties = schema.get("properties")
    if not isinstance(properties, dict):  # pragma: no cover - owner assertion
        raise PlannerStructuredOutputSchemaError(
            "runtime suffix root schema has no properties"
        )
    properties["replace_from_step_id"] = {
        "type": "string",
        "const": replace_from_step_id,
    }
    properties["replacement_step"] = {"$ref": "#/$defs/AnalysisStep"}
    _strictify_planner_transport_schema(schema)
    _assert_closed_planner_transport_schema(schema)
    return StructuredOutputRequest.from_schema(
        name="easyicu_runtime_plan_suffix_revision_v1",
        schema=schema,
        strict=True,
    )


def runtime_suffix_structured_output_request(
    *,
    allowed_literature_citation_keys: Sequence[str],
    replace_from_step_id: str,
    planned_analysis_role: str,
    allowed_inputs: Sequence[str],
    expected_outputs: Sequence[str],
    scientific_action_ids: Sequence[str],
) -> StructuredOutputRequest:
    """Return the run-bound strict schema for one observation-driven next step."""

    normalized_step_id = str(replace_from_step_id or "").strip()
    if not normalized_step_id:
        raise PlannerStructuredOutputSchemaError(
            "runtime suffix requires one replacement step id"
        )
    normalized_role = str(planned_analysis_role or "").strip()
    if normalized_role not in {"primary", "secondary", "sensitivity", "auxiliary"}:
        raise PlannerStructuredOutputSchemaError(
            "runtime suffix requires one valid planned analysis role"
        )
    return _runtime_suffix_structured_output_request_cached(
        normalize_literature_citation_keys(allowed_literature_citation_keys),
        normalized_step_id,
        normalized_role,
        tuple(
            dict.fromkeys(
                str(value).strip() for value in allowed_inputs if str(value).strip()
            )
        ),
        tuple(str(value).strip() for value in expected_outputs),
        tuple(
            dict.fromkeys(
                str(value).strip()
                for value in scientific_action_ids
                if str(value).strip()
            )
        ),
    )


def decode_planner_transport_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Compile the Planner wire representation before authority validation."""

    decoded: Dict[str, Any] = copy.deepcopy(dict(data))
    raw_labels = decoded.get("display_labels")
    if isinstance(raw_labels, list):
        labels: Dict[str, Any] = {}
        for index, item in enumerate(raw_labels):
            if not isinstance(item, dict) or set(item) != {"key", "value"}:
                raise PlannerStructuredOutputSchemaError(
                    f"display_labels[{index}] is not one exact key/value row"
                )
            key = item.get("key")
            if key in labels:
                raise PlannerStructuredOutputSchemaError(
                    f"display_labels repeats key {key!r}"
                )
            labels[key] = item.get("value")
        decoded["display_labels"] = labels

    raw_specs = decoded.get("robustness_specs")
    if isinstance(raw_specs, list):
        for raw_spec in raw_specs:
            if not isinstance(raw_spec, dict):
                continue
            for field in ("missing_override", "outcome_override"):
                override = raw_spec.get(field)
                if not isinstance(override, dict):
                    continue
                compact = {key: value for key, value in override.items() if value is not None}
                raw_spec[field] = compact or None

    # A typed design binding already declares which sealed source governs the
    # step.  ``literature_citation_keys`` is the flat roster consumed by later
    # validators and signatures, not a second scientific choice.  Compile the
    # binding coordinate into that roster while retaining every explicitly
    # cited key; an extra citation with no binding still fails downstream.
    raw_steps = decoded.get("steps")
    if isinstance(raw_steps, list):
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                continue
            requirements = raw_step.get("model_requirements")
            if isinstance(requirements, list):
                for requirement in requirements:
                    if isinstance(requirement, dict):
                        _decode_model_covariate_decisions(requirement)
            citations = raw_step.get("literature_citation_keys")
            bindings = raw_step.get("literature_design_bindings")
            if not isinstance(citations, list) or not isinstance(bindings, list):
                continue
            compiled = list(citations)
            for binding in bindings:
                if not isinstance(binding, dict) or "citation_key" not in binding:
                    continue
                citation_key = binding["citation_key"]
                if citation_key not in compiled:
                    compiled.append(citation_key)
            raw_step["literature_citation_keys"] = compiled
    return decoded


def parse_runtime_plan_suffix(raw: str) -> RuntimePlanSuffixRevision:
    """Parse a suffix through the same scientific transport boundary as a plan."""

    payload = json.loads(str(raw or "").strip())
    if isinstance(payload, dict) and isinstance(payload.get("replacement_step"), dict):
        payload["replacement_step"] = decode_planner_transport_payload(
            {"steps": [payload["replacement_step"]]}
        )["steps"][0]
    return RuntimePlanSuffixRevision.model_validate(payload)


def _decode_model_covariate_decisions(requirement: Dict[str, Any]) -> None:
    """Restore model maps without permitting duplicate scientific decisions."""

    for field in ("covariate_rationales", "covariate_temporal_roles"):
        rows = requirement.get(field)
        if not isinstance(rows, list):
            continue
        values: Dict[str, Any] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"key", "value"}:
                raise PlannerStructuredOutputSchemaError(
                    f"{field}[{index}] is not one exact key/value row"
                )
            raw_key = row["key"]
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise PlannerStructuredOutputSchemaError(
                    f"{field}[{index}] requires a non-empty covariate name"
                )
            key = raw_key.strip()
            if key in values:
                raise PlannerStructuredOutputSchemaError(f"{field} repeats key {key!r}")
            values[key] = row["value"]
        requirement[field] = values


def planner_descriptive_method_guidance(analysis_type: str) -> str:
    """Return the exact contracts for compact descriptive host owners."""

    if str(analysis_type).strip().casefold() != "descriptive_epidemiology":
        return ""
    return (
        "Two compact descriptive methods have exact host contracts. For "
        "`method='descriptive_distribution'`, declare exactly one typed cohort "
        "input followed by exactly one categorical grouping column and exactly "
        "one continuous value column, in that order; declare only "
        "`table:distribution_prevalence`. Do not add a third column or an "
        "association to that step. For a non-causal two-continuous-variable "
        "association, use a separate `method='descriptive_association'` step "
        "with exactly one typed cohort input followed by the predictor and "
        "outcome columns, in that order, and exactly one "
        "`statistic:<descriptive_name>` output. This contract computes a "
        "complete-case Spearman rho without adjustment or imputation. A figure "
        "of the grouped distribution consumes only the distribution table; a "
        "figure of the association scalar consumes only its statistic. Never "
        "bundle the grouped distribution and the association into one step.\n\n"
    )


def planner_descriptive_robustness_guidance(analysis_type: str) -> str:
    """Keep effect robustness out of descriptive-only analysis families."""

    if str(analysis_type).strip().casefold() != "descriptive_epidemiology":
        return ""
    return (
        " This replay contract applies only when a primary fitted "
        "effect and its uncertainty already exist. For "
        "`analysis_type='descriptive_epidemiology'`, do NOT declare "
        "`robustness_specs`, a `robustness_sensitivity` step, effect-style "
        "products such as `primary_or`, or a robustness forest plot. Use the "
        "typed measurement/missingness audits above for denominator and "
        "complete-case availability checks; any additional descriptive "
        "summary must remain a separately declared descriptive method.\n\n"
    )


def planner_adjusted_association_owner_guidance() -> str:
    """Name the one method family implemented by the sealed host owner."""

    return (
        "`table:adjusted_association_estimates` requires "
        "`method_family='statsmodels_logit_mle'`; no sealed executor owns "
        "`statsmodels_glm_binomial`, which needs an agent-coded step and a "
        "different output. "
    )


def planner_endpoint_and_optional_science_guidance() -> str:
    """Render host-owned endpoint and optional post-analysis boundaries."""

    return (
        "`ResearchContext.endpoint` is sealed host authority: copy exactly; never "
        "infer or repair. A required missing endpoint blocks execution.\n\n"
        "Leave `evalue_conversion_spec` null unless requested; it requires a "
        "baseline-risk evidence id, rate and population columns, and exact "
        "population. Leave `subgroup_analysis_spec` null unless requested; then "
        "bind a primary model requirement and declare predictor, outcome, subgroup "
        "columns, quantile buckets, minimum sizes, effect scale, adjustment roster, "
        "and multiplicity family. The Planner chooses these fields.\n\n"
    )


def render_methodological_principles(principles: Sequence[Any]) -> str:
    """Project case-neutral methodological principles into Planner guidance."""

    errors = [principle for principle in principles if principle.kind == "error"]
    cautions = [
        principle for principle in principles if principle.kind == "caution"
    ]
    lines = [
        "\n\nCROSS-CUTTING ICU METHODOLOGY (case-neutral; apply when planning):",
        "Objective errors to avoid — wrong under any study design:",
    ]
    lines.extend(
        f"- [{principle.phase}] {principle.principle}" for principle in errors
    )
    lines.append(
        "Defensible choices — state and justify in the plan; do not let them "
        "pass silently, but the analyst, not these rules, decides:"
    )
    lines.extend(
        f"- [{principle.phase}] {principle.principle}" for principle in cautions
    )
    return "\n".join(lines)


def _required_method_binding_options(
    method_cards: Sequence[Any],
    required_method_layers: Sequence[str],
) -> dict[str, dict[str, list[str]]]:
    """Project only run-available choices for context-required method layers."""

    required = {
        str(layer or "").strip()
        for layer in required_method_layers
        if str(layer or "").strip()
    }
    options: dict[str, dict[str, set[str]]] = {}
    for card in method_cards:
        if card.layer not in required:
            continue
        options.setdefault(card.layer, {}).setdefault(card.source_key, set()).update(
            card.design_elements
        )
    return {
        layer: {
            key: sorted(elements)
            for key, elements in sorted(sources.items())
        }
        for layer, sources in sorted(options.items())
    }


def _required_method_binding_examples(
    options: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, object]]:
    """Render one minimal schema-valid binding example per required layer."""

    examples: dict[str, dict[str, object]] = {}
    for layer, sources in sorted(options.items()):
        if not sources:
            continue
        source_key = sorted(sources)[0]
        elements = sources[source_key]
        if not elements:
            continue
        examples[layer] = {
            "literature_citation_keys": [source_key],
            "literature_design_bindings": [
                {
                    "citation_key": source_key,
                    "design_elements": [elements[0]],
                    "application": (
                        f"Apply the source's {layer} method card to this step."
                    ),
                    "divergence": None,
                }
            ],
        }
    return examples


def bind_literature_citation_authority(
    planning_contract_context: str,
    allowed_keys: Sequence[str],
    *,
    direct_comparator_keys: Sequence[str] = (),
    required_method_layers: Sequence[str] = (),
) -> str:
    """Append role-bound LiteratureBundle authority to the Planner profile.

    Citation keys alone are not a scientific design aid: the model also needs
    to know which source supports which methodological decision, and which
    retrieved records survived the direct-comparator screen.  This projection
    is deliberately assembled by the host from the sealed pre-plan bundle.
    """

    if not allowed_keys:
        return planning_contract_context
    method_source_keys = allowed_method_source_keys(allowed_keys)
    direct_keys = tuple(
        key
        for key in normalize_literature_citation_keys(direct_comparator_keys)
        if key in set(allowed_keys)
    )
    method_cards = [
        card for card in METHOD_CARDS if card.source_key in method_source_keys
    ]
    required_binding_options = _required_method_binding_options(
        method_cards,
        required_method_layers,
    )
    required_binding_examples = _required_method_binding_examples(
        required_binding_options
    )
    authority = (
        "PRE-PLAN LITERATURE CITATION AUTHORITY (exact, run-bound):\n"
        "- allowed_literature_citation_keys: "
        + json.dumps(list(allowed_keys), ensure_ascii=False)
        + (
            "\n- allowed_method_source_keys: "
            + json.dumps(list(method_source_keys), ensure_ascii=False)
            if method_source_keys
            else ""
        )
        + (
            "\n- screened_direct_comparator_keys: "
            + json.dumps(list(direct_keys), ensure_ascii=False)
            if direct_keys
            else "\n- screened_direct_comparator_keys: []"
        )
        + "\n- Every primary, secondary, and sensitivity step MUST choose one or "
        "more exact values from this list through literature_design_bindings. "
        "The host compiles each binding's citation_key into that step's "
        "literature_citation_keys roster. Do not cite an evidence artifact, "
        "analysis contract, study-design brief, or invented semantic label. "
        "A citation with no matching design binding remains invalid; auxiliary "
        "steps may leave both arrays empty. "
        + literature_design_binding_shape_guide()
        + " Do not invent or copy a source quotation; the host joins the sealed "
        "excerpt."
        + (
            " Every scientific step MUST include at least one exact "
            "allowed_method_source_key that supports its design or method; a "
            "disease-definition or database paper alone is insufficient. Add "
            "topic/direct-comparator keys when they support the step's population, "
            "exposure, outcome, or interpretation."
            if method_source_keys
            else ""
        )
        + (
            " When screened_direct_comparator_keys is non-empty, at least one "
            "primary analysis step MUST additionally bind one of those keys. "
            "Use its source excerpt only to compare population, time zero, "
            "exposure, outcome/estimand, and analysis choices. It is not "
            "automatic authority to copy eligibility criteria or change this "
            "study's sealed ResearchContext."
            if direct_keys
            else ""
        )
        + (
            "\n- method_decision_cards (host-curated; id|layer|supported_design_elements|question|requirement|source):\n"
            + "\n".join(
                "  - "
                + " | ".join(
                    (
                        card.id,
                        card.layer,
                        ",".join(card.design_elements),
                        card.question,
                        card.requirement,
                        card.source_key,
                    )
                )
                for card in method_cards
            )
            if method_cards
            else ""
        )
        + (
            "\n- case_applicable_required_method_layers: "
            + json.dumps(sorted(required_binding_options), ensure_ascii=False)
            + "\n- Cover every listed layer at least once through one of these "
            "exact source/design-element options: "
            + json.dumps(
                required_binding_options,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n- Minimal schema-valid examples by required layer (copy only "
            "the layers that truly govern a scientific estimator; support "
            "steps remain auxiliary): "
            + json.dumps(
                required_binding_examples,
                ensure_ascii=False,
                sort_keys=True,
            )
            if required_binding_options
            else ""
        )
    )
    return "\n\n".join(
        value for value in (planning_contract_context, authority) if value
    )


def literature_citation_retry_suffix(
    allowed_keys: Sequence[str],
    *,
    direct_comparator_keys: Sequence[str] = (),
    required_method_layers: Sequence[str] = (),
) -> str:
    """Render exact citation keys in the structured-retry reminder."""

    if not allowed_keys:
        return ""
    method_source_keys = allowed_method_source_keys(allowed_keys)
    direct_keys = tuple(
        key
        for key in normalize_literature_citation_keys(direct_comparator_keys)
        if key in set(allowed_keys)
    )
    method_element_map = {
        key: sorted(
            {
                element
                for card in METHOD_CARDS
                if card.source_key == key
                for element in card.design_elements
            }
        )
        for key in method_source_keys
    }
    required_binding_options = _required_method_binding_options(
        [card for card in METHOD_CARDS if card.source_key in method_source_keys],
        required_method_layers,
    )
    required_binding_examples = _required_method_binding_examples(
        required_binding_options
    )
    return (
        " Allowed literature_citation_keys for this run are exactly: "
        + json.dumps(list(allowed_keys), ensure_ascii=False)
        + "."
        + " Each scientific step must also include literature_design_bindings. "
        + literature_design_binding_shape_guide()
        + (
            " Every scientific step must include at least one method-source key "
            "from: "
            + json.dumps(list(method_source_keys), ensure_ascii=False)
            + ". Method-source bindings may use ONLY the design elements in "
            "this exact host-owned map (use another source or omit an "
            "unsupported element; never broaden a method card): "
            + json.dumps(method_element_map, ensure_ascii=False, sort_keys=True)
            + "."
            if method_source_keys
            else ""
        )
        + (
            " At least one primary step must also cite a screened direct "
            "comparator from: "
            + json.dumps(list(direct_keys), ensure_ascii=False)
            + "."
            if direct_keys
            else ""
        )
        + (
            " Case-applicable method layers that must be covered at least once "
            "are: "
            + json.dumps(sorted(required_binding_options), ensure_ascii=False)
            + ". Exact source/design-element options are: "
            + json.dumps(
                required_binding_options,
                ensure_ascii=False,
                sort_keys=True,
            )
            + ". Minimal schema-valid examples by required layer are: "
            + json.dumps(
                required_binding_examples,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "."
            if required_binding_options
            else ""
        )
    )


def literature_design_binding_shape_guide() -> str:
    """Publish the one exact nested JSON shape owned by the literature contract."""

    return (
        "Each record must use exactly this JSON shape: "
        '{"citation_key":"<exact allowed key>",'
        '"design_elements":["<one or more exact allowed elements>"],'
        '"application":"<how the source shapes this step>",'
        '"divergence":null}. '
        "`divergence` may instead be a concise string. The only permitted keys "
        "are `citation_key`, `design_elements`, `application`, and `divergence`; "
        "do not rename them."
    )


def descriptive_claim_shape_guide() -> str:
    """Publish the exact nested JSON shape for a descriptive claim ceiling."""

    return (
        "When this ceiling is required, emit exactly "
        '`"descriptive_claim":{"claim_ceiling":"descriptive_only",'
        '"unresolved_limitations":'
        '["post_baseline_exposure_opportunity_unresolved"]}`. '
        "`unresolved_limitations` is an array; do not rename it to a singular "
        "`limitation` field."
    )


def counts_only_distribution_guide() -> str:
    """Publish the no-uncertainty projection selected by typed study authority."""

    return (
        "For variance_estimator='none_counts_only', use one primary descriptive "
        "step outputting only table:exposure_outcome_distribution, "
        "and uses schema /3 with interval_method='none_counts_only'. Set repeated "
        "unit method, confidence, contrast, and dependence to null. No models, "
        "Table One, extra result tables, or mislabeled audits. Keep as siblings: "
        '{"descriptive_claim":{"claim_ceiling":"descriptive_only",'
        '"unresolved_limitations":'
        '["post_baseline_exposure_opportunity_unresolved"]},'
        '"exposure_outcome_distribution_spec":{'
        '"schema_version":"easyicu.exposure_outcome_distribution/3",'
        '"interval_method":"none_counts_only",'
        '"repeated_unit_interval_method":null,"risk_difference_contrast":null,'
        '"dependence":null,"confidence_level":null}}.'
    )


def interval_bearing_distribution_guide() -> str:
    """Publish the coupled /2 interval fields the transport enum cannot express."""

    return (
        "For interval-bearing schema /2, the closed design is exact: use "
        "schema_version='easyicu.exposure_outcome_distribution/2', "
        "interval_method='wilson', "
        "repeated_unit_interval_method='patient_cluster_robust_wald', and a "
        "non-null confidence_level. Keep the repeated-unit interval method "
        "declared even while dependence is null before host binding; it does "
        "not invent or authorize a grouping source."
    )


def descriptive_claim_example_fragment() -> str:
    """Render the worked-example fragment from the claim schema owner."""

    return (
        '      "descriptive_claim": {\n'
        '        "claim_ceiling": "descriptive_only",\n'
        '        "unresolved_limitations": '
        '["post_baseline_exposure_opportunity_unresolved"]\n'
        "      },\n"
    )


def figure_panel_shape_guide() -> str:
    """Publish the exact case-neutral visual-semantics declaration."""

    return (
        "For every visualization panel, emit one `figure_panels` record with "
        "exact keys `panel_id`, `figure_output`, `article_role`, `chart_type`, "
        "and `source_products`. Copy `article_role` and one accepted "
        "`chart_type` from the supplied ARTICLE FIGURE STRATEGY; "
        "`figure_output` must be this step's exact `figure:*` output and every "
        "`source_products` item must be this step's exact typed input. Input "
        "or output names and intent prose do not establish a visual role."
    )


def artifact_consumption_contract_shape_guide() -> str:
    """Publish the mode-dependent wire contract without choosing cardinality."""

    all_rows_example = json.dumps(
        {
            "schema_version": "easyicu.artifact_consumption/1",
            "input_key": "table:exact_product",
            "mode": "all_rows",
            "role_column": None,
            "expected_roles": [],
        },
        separators=(",", ":"),
    )
    return (
        "`input_consumption_contracts` item: "
        f"`{all_rows_example}`. "
        "`input_key` is lowercase `kind:product`. Modes are `all_rows`, "
        "`single_row`, and `one_per_role`: first two use `role_column:null`, "
        "`expected_roles:[]`; last uses non-empty role column and complete "
        "unique roles. "
        "Never rename `input_key` to `input` or `mode` to `cardinality`. "
    )


def planner_science_retry_guide() -> str:
    """Return schema-owned retry guidance outside the Planner god module."""

    def exact_keys(model: type) -> str:
        return ", ".join(f"`{name}`" for name in sorted(_declared_field_names(model)))

    opaque_binary = json.dumps(
        list(opaque_level_tokens(2)), ensure_ascii=True, separators=(",", ":")
    )

    optional_fields = (
        "Contract applicability is exact: `family_primary_result_requirement` is "
        "legal only on the primary step when `analysis_type` is `causal_inference` "
        "or `survival`. An `association_study` must omit that field and declare its "
        "supported adjusted model through `model_requirements`. Optional "
        "collections such as `know_how_decisions` use JSON arrays, never `null`. "
        + artifact_consumption_contract_shape_guide()
        + " "
        + figure_panel_shape_guide()
        + " "
        "Omit `AnalysisPlan.endpoint` or emit null."
    )
    binding_fields = (
        "A `literature_design_bindings` record is the source authority: the "
        "host compiles its `citation_key` into that same step's "
        "`literature_citation_keys` roster. A citation with no matching design "
        "binding remains invalid. `model_requirements` is legal "
        "only on a step whose method is exactly "
        "`adjusted_association_models` and whose expected outputs include "
        "`table:adjusted_association_estimates`; every other step emits "
        "`model_requirements: []` and uses its family-specific contract."
    )
    model_representation_fields = (
        "Within `model_requirements`, categorical `exposure_levels`, "
        "`exposure_reference_level`, and `primary_contrast_level` are symbolic "
        "model-term labels and therefore must be JSON strings. When the "
        "variable catalog publishes `opaque_levels`, copy those exact strings "
        "instead of guessing labels; for a two-level field the published form "
        f'is `{opaque_binary}`. Otherwise examples such as `["0", "1"]`, '
        '`"0"`, and `"1"` show the string representation only. This is deliberately '
        "different from `exposure_outcome_distribution_spec`, whose closed "
        "observed values preserve their source scalar types. Every "
        "`robustness_specs` item must include non-empty `spec_id`, `axis`, and "
        "`description`; the description states the prespecified scientific "
        "alternative rather than merely repeating the axis."
    )
    representation_fields = (
        "Use only schema product kinds: `protocol` is not a product kind. A "
        "non-executable future/feasibility step uses method "
        "`feasibility_protocol` and output `report:<name>`; never consume that "
        "terminal report downstream. "
        "`TableOneSpec` already requires "
        "`missingness_display='n_percent_by_group'`; do not add undeclared "
        "`report_missing_by_group`. An `exposure_outcome_distribution_spec` "
        "step must list its exact exposure and outcome in `inputs`. "
        "A `table_one_spec` step must list its `group_by` and every "
        "`variables[*].name` in that same step's `inputs`. "
        + interval_bearing_distribution_guide()
        + " "
        "Preserve observed scalar types (JSON numbers remain numbers). On retry retain "
        "every article role and required `robustness_specs`; do not fix one "
        "error by dropping an already-satisfied requirement. "
        + descriptive_claim_shape_guide()
    )
    exact_scientific_object_fields = (
        "Closed scientific objects accept only their schema-declared keys. "
        "`TableOneSpec` keys are: "
        + exact_keys(TableOneSpec)
        + ". `TableOneVariableSpec` keys are: "
        + exact_keys(TableOneVariableSpec)
        + ". `ExposureOutcomeDistributionSpec` keys are: "
        + exact_keys(ExposureOutcomeDistributionSpec)
        + ". Its `risk_difference_contrast` keys are: "
        + exact_keys(ExposureOutcomeRiskDifferenceContrast)
        + ". Standardized differences are selected by the declared "
        "`standardized_difference_mode`; do not add a second reporting switch "
        "or any explanatory key inside these closed objects. When a variable "
        "catalog publishes `opaque_levels`, every corresponding level array "
        "and scalar selector must copy those exact tokens. Do not translate "
        "them to yes/no labels or quoted numeric codes."
    )
    return (
        "\n\n"
        + model_terms_retry_guide()
        + "\n\n"
        + optional_fields
        + "\n\n"
        + binding_fields
        + "\n\n"
        + model_representation_fields
        + "\n\n"
        + representation_fields
        + "\n\n"
        + exact_scientific_object_fields
    )


def _canonicalise_figure_output_alias(token: object) -> object:
    """Canonicalize only a colon-typed alias for a declared figure product."""

    if not isinstance(token, str):
        return token
    parsed = _canonical_typed_product(token)
    if parsed is None or parsed[0] != "figure":
        return token
    _kind, _separator, name = token.partition(":")
    return f"figure:{name.strip()}"


def _is_untyped_figure_alias_output(token: object) -> bool:
    """Reject an underscore figure alias before it silently loses its role."""

    if not isinstance(token, str):
        return False
    text = token.strip()
    if not text or ":" in text:
        return False
    if text.lower().endswith((".png", ".svg", ".pdf", ".tif", ".tiff")):
        return False
    head, separator, _rest = text.partition("_")
    if not separator:
        return False
    probe = _canonical_typed_product(f"{head}:probe")
    return probe is not None and probe[0] == "figure"


def _canonicalise_planned_analysis_role(
    value: object,
    *,
    method: object,
) -> object:
    """Normalize a closed set of representation-only planner role variants."""

    if not isinstance(value, str):
        return value
    token = value.strip().casefold()
    if token in {"primary", "secondary", "sensitivity", "auxiliary"}:
        return token
    method_token = str(method or "").strip().casefold()
    if token == "robustness" and method_token == "robustness_sensitivity":
        return "sensitivity"
    return value


def _declared_field_names(model: type) -> set:
    """Read accepted fields from the declaring Pydantic model or dataclass."""

    fields = getattr(model, "model_fields", None)
    if fields is not None:
        return set(fields)
    if dataclasses.is_dataclass(model):
        return {field.name for field in dataclasses.fields(model)}
    raise TypeError(
        f"{model.__name__} declares neither pydantic model_fields nor "
        "dataclass fields, so its accepted Planner keys cannot be read "
        "from the schema; do not transcribe them by hand."
    )


class PlannerScientificProjectionError(ValueError):
    """The Planner emitted an unknown key inside a scientific value object."""

    issue_code = "planner_scientific_contract_unknown_key"
    owner = "easyicu.planning.plan_payload_projection_v1"

    def __init__(self, *, path: str, unknown_keys: List[str]) -> None:
        self.path = path
        self.unknown_keys = tuple(sorted(unknown_keys))
        super().__init__(
            f"{self.issue_code}: unknown key(s) at {path}: "
            + ", ".join(repr(key) for key in self.unknown_keys)
            + "; re-emit the scientific object using only its declared schema"
        )


def _require_exact_scientific_keys(
    raw: Dict[str, Any],
    *,
    allowed: set,
    path: str,
) -> None:
    unknown = [str(key) for key in raw if key not in allowed]
    if unknown:
        raise PlannerScientificProjectionError(path=path, unknown_keys=unknown)


def _require_runtime_supported_product_kinds(
    *,
    step_id: str,
    inputs: object,
    expected_outputs: object,
) -> None:
    """Reject typed product spellings the runtime cannot honour.

    Product *names* and the scientific dependency graph remain Planner-owned.
    The closed kind vocabulary is a representation/runtime contract: accepting
    ``text:x`` here only to reject it after the paid probe/replan cycle cannot
    make the plan more expressive.  A terminal ``report`` is intentionally a
    valid output but not a consumable input; the writer materialises it after
    the evidence-producing analysis steps have completed.
    """

    for field, values, supported in (
        ("inputs", inputs, RUNTIME_BINDABLE_TYPED_INPUT_KINDS),
        (
            "expected_outputs",
            expected_outputs,
            PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS,
        ),
    ):
        if not isinstance(values, list):
            continue
        for index, raw in enumerate(values):
            product = _canonical_typed_product(raw) if isinstance(raw, str) else None
            if product is None or product[0] in supported:
                continue
            terminal_report_note = (
                " A report product is terminal writer output and cannot be "
                "consumed by another analysis step."
                if field == "inputs" and product[0] == "report"
                else ""
            )
            raise ValueError(
                f"Planner step {step_id!r} declares unsupported typed product "
                f"kind {product[0]!r} at {field}[{index}]. Supported kinds are "
                f"{sorted(supported)!r}.{terminal_report_note} Re-emit the same "
                "scientific plan with a runtime-supported product kind."
            )


def _compile_mixed_figure_panel_steps(
    raw_steps: object,
) -> Tuple[object, List[str]]:
    """Move an exact mixed-step panel contract to a rendering-only child.

    This is structural compilation, not figure selection. It fires only when
    every declared figure is covered by a panel, every panel names an exact
    uniquely produced table/statistic source, and the analytic parent retains
    at least one non-figure output. Any ambiguous shape is left untouched for
    the schema to reject.
    """

    if not isinstance(raw_steps, list):
        return raw_steps, []
    output_owners: Dict[str, List[int]] = {}
    occupied_step_ids = {
        step.get("step_id")
        for step in raw_steps
        if isinstance(step, dict)
        and isinstance(step.get("step_id"), str)
        and step.get("step_id")
    }
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            continue
        raw_outputs = step.get("expected_outputs")
        if not isinstance(raw_outputs, list):
            continue
        for output in raw_outputs:
            if isinstance(output, str):
                output_owners.setdefault(output, []).append(index)

    compiled_steps: List[object] = []
    normalizations: List[str] = []
    for step_index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            compiled_steps.append(step)
            continue
        raw_method = step.get("method")
        method_head = (
            raw_method.strip().casefold().split(" with ", 1)[0]
            if isinstance(raw_method, str)
            else ""
        )
        panels = step.get("figure_panels")
        outputs = step.get("expected_outputs")
        if (
            not method_head
            or method_head == "visualization"
            or not isinstance(panels, list)
            or not panels
            or not isinstance(outputs, list)
            or not all(isinstance(output, str) for output in outputs)
        ):
            compiled_steps.append(step)
            continue
        figure_outputs = [
            output
            for output in outputs
            if isinstance(output, str)
            and (_canonical_typed_product(output) or (None, None))[0] == "figure"
        ]
        non_figure_outputs = [output for output in outputs if output not in figure_outputs]
        if not figure_outputs or not non_figure_outputs or not all(
            isinstance(panel, dict) for panel in panels
        ):
            compiled_steps.append(step)
            continue
        raw_panel_outputs = [panel.get("figure_output") for panel in panels]
        if not all(isinstance(output, str) for output in raw_panel_outputs):
            compiled_steps.append(step)
            continue
        panel_outputs = list(dict.fromkeys(raw_panel_outputs))
        if set(panel_outputs) != set(figure_outputs):
            compiled_steps.append(step)
            continue
        raw_source_lists = [panel.get("source_products") for panel in panels]
        if not all(
            isinstance(sources, list)
            and sources
            and all(isinstance(source, str) for source in sources)
            for sources in raw_source_lists
        ):
            compiled_steps.append(step)
            continue
        source_products = list(
            dict.fromkeys(source for sources in raw_source_lists for source in sources)
        )
        if not source_products or any(
            (_canonical_typed_product(source) or (None, None))[0]
            not in {"table", "statistic"}
            or len(output_owners.get(source, [])) != 1
            or output_owners[source][0] > step_index
            for source in source_products
        ):
            compiled_steps.append(step)
            continue
        raw_step_id = step.get("step_id")
        step_id = raw_step_id.strip() if isinstance(raw_step_id, str) else ""
        child_id = f"{step_id}_figure"
        if not step_id or child_id in occupied_step_ids:
            compiled_steps.append(step)
            continue
        occupied_step_ids.add(child_id)
        parent = copy.deepcopy(step)
        parent["expected_outputs"] = non_figure_outputs
        parent["figure_panels"] = []
        child = {
            "step_id": child_id,
            "planned_analysis_role": "auxiliary",
            "intent": (
                "Render the Planner-declared panels from their exact typed "
                f"source products for step {step_id}."
            ),
            "inputs": source_products,
            "expected_outputs": figure_outputs,
            "method": "visualization",
            "icu_rule_refs": ["visualization_rule"],
            "input_consumption_contracts": [
                {"input_key": source, "mode": "all_rows"}
                for source in source_products
                if (_canonical_typed_product(source) or (None, None))[0] == "table"
            ],
            "figure_panels": copy.deepcopy(panels),
        }
        compiled_steps.extend((parent, child))
        normalizations.append(
            f"{step_id}:mixed_figure_panels_compiled_to:{child_id}"
        )
    return compiled_steps, normalizations


def _normalise_plan_payload(
    data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Drop invented keys while preserving every field declared by the schema."""

    allowed_plan = _declared_field_names(AnalysisPlan)
    allowed_step = _declared_field_names(AnalysisStep)
    allowed_model_requirement = _declared_field_names(PlannedModelRequirement)
    allowed_consumption_contract = _declared_field_names(ArtifactConsumptionContract)
    allowed_figure_panel = _declared_field_names(PlannedFigurePanelSpec)
    allowed_table_one_spec = _declared_field_names(TableOneSpec)
    allowed_table_one_variable = _declared_field_names(TableOneVariableSpec)
    allowed_robustness_spec = _declared_field_names(RobustnessSpec)
    dropped: Dict[str, List[str]] = {
        "top_level": [],
        "steps": [],
        "model_requirements": [],
        "input_consumption_contracts": [],
        "figure_panels": [],
        "table_one_spec": [],
        "robustness_specs": [],
        "normalizations": [],
    }
    out = {key: value for key, value in data.items() if key in allowed_plan}
    dropped["top_level"] = [str(key) for key in data if key not in allowed_plan]
    out["steps"], dropped["normalizations"] = _compile_mixed_figure_panel_steps(
        out.get("steps")
    )
    steps = []
    for idx, raw_step in enumerate(out.get("steps", []) or []):
        if not isinstance(raw_step, dict):
            continue
        step_payload = {
            key: value for key, value in raw_step.items() if key in allowed_step
        }
        step_id = raw_step.get("step_id") or f"step[{idx}]"
        dropped["steps"].extend(
            f"{step_id}:{key}" for key in raw_step if key not in allowed_step
        )
        if "planned_analysis_role" in step_payload:
            step_payload["planned_analysis_role"] = _canonicalise_planned_analysis_role(
                step_payload["planned_analysis_role"],
                method=step_payload.get("method"),
            )
        requirements = []
        for req_idx, raw_requirement in enumerate(
            step_payload.get("model_requirements", []) or []
        ):
            if not isinstance(raw_requirement, dict):
                requirements.append(raw_requirement)
                continue
            requirement_payload = {
                key: value
                for key, value in raw_requirement.items()
                if key in allowed_model_requirement
            }
            requirement_id = (
                raw_requirement.get("requirement_id")
                or f"step[{idx}].model_requirements[{req_idx}]"
            )
            _require_exact_scientific_keys(
                raw_requirement,
                allowed=allowed_model_requirement,
                path=f"steps[{idx}].model_requirements[{req_idx}]({requirement_id})",
            )
            if "analysis_role" in requirement_payload:
                requirement_payload["analysis_role"] = (
                    _canonicalise_planned_analysis_role(
                        requirement_payload["analysis_role"],
                        method=step_payload.get("method"),
                    )
                )
            requirements.append(requirement_payload)
        if "model_requirements" in step_payload:
            step_payload["model_requirements"] = requirements
        consumption_contracts = []
        for contract_idx, raw_contract in enumerate(
            step_payload.get("input_consumption_contracts", []) or []
        ):
            if not isinstance(raw_contract, dict):
                consumption_contracts.append(raw_contract)
                continue
            contract_payload = {
                key: value
                for key, value in raw_contract.items()
                if key in allowed_consumption_contract
            }
            contract_id = (
                raw_contract.get("input_key")
                or f"step[{idx}].input_consumption_contracts[{contract_idx}]"
            )
            _require_exact_scientific_keys(
                raw_contract,
                allowed=allowed_consumption_contract,
                path=(
                    f"steps[{idx}].input_consumption_contracts[{contract_idx}]"
                    f"({contract_id})"
                ),
            )
            if contract_payload:
                consumption_contracts.append(contract_payload)
            else:
                dropped["input_consumption_contracts"].append(
                    f"{contract_id}:empty_after_normalization"
                )
        if "input_consumption_contracts" in step_payload:
            step_payload["input_consumption_contracts"] = consumption_contracts
        figure_panels = []
        for panel_idx, raw_panel in enumerate(
            step_payload.get("figure_panels", []) or []
        ):
            if not isinstance(raw_panel, dict):
                figure_panels.append(raw_panel)
                continue
            panel_payload = {
                key: value
                for key, value in raw_panel.items()
                if key in allowed_figure_panel
            }
            panel_id = (
                raw_panel.get("panel_id")
                or f"step[{idx}].figure_panels[{panel_idx}]"
            )
            _require_exact_scientific_keys(
                raw_panel,
                allowed=allowed_figure_panel,
                path=f"steps[{idx}].figure_panels[{panel_idx}]({panel_id})",
            )
            if panel_payload:
                figure_panels.append(panel_payload)
            else:
                dropped["figure_panels"].append(
                    f"{panel_id}:empty_after_normalization"
                )
        if "figure_panels" in step_payload:
            step_payload["figure_panels"] = figure_panels
        raw_table_one = step_payload.get("table_one_spec")
        if isinstance(raw_table_one, dict):
            _require_exact_scientific_keys(
                raw_table_one,
                allowed=allowed_table_one_spec,
                path=f"steps[{idx}].table_one_spec",
            )
            table_one_payload = {
                key: value
                for key, value in raw_table_one.items()
                if key in allowed_table_one_spec
            }
            variables = []
            for variable_index, raw_variable in enumerate(
                table_one_payload.get("variables", []) or []
            ):
                if not isinstance(raw_variable, dict):
                    variables.append(raw_variable)
                    continue
                variable_payload = {
                    key: value
                    for key, value in raw_variable.items()
                    if key in allowed_table_one_variable
                }
                _require_exact_scientific_keys(
                    raw_variable,
                    allowed=allowed_table_one_variable,
                    path=(f"steps[{idx}].table_one_spec.variables[{variable_index}]"),
                )
                variables.append(variable_payload)
            table_one_payload["variables"] = variables
            step_payload["table_one_spec"] = table_one_payload
        raw_outputs = step_payload.get("expected_outputs")
        if isinstance(raw_outputs, list):
            normalised_outputs: List[Any] = []
            for item in raw_outputs:
                if _is_untyped_figure_alias_output(item):
                    suggested = str(item).strip().partition("_")[2]
                    raise ValueError(
                        f"Planner step {step_id!r} declares figure output "
                        f"{item!r} with an underscore instead of the typed "
                        "'figure:' separator; re-emit it as "
                        f"'figure:{suggested}' so the declared figure binds "
                        "to an exact output file."
                    )
                normalised_outputs.append(_canonicalise_figure_output_alias(item))
            figure_identity_aliases: Dict[Tuple[str, str], List[str]] = {}
            for candidate in normalised_outputs:
                if not isinstance(candidate, str):
                    continue
                identity = _canonical_typed_product(candidate)
                if identity is None or identity[0] != "figure":
                    continue
                figure_identity_aliases.setdefault(identity, []).append(candidate)
            collisions = {
                identity: aliases
                for identity, aliases in figure_identity_aliases.items()
                if len(aliases) > 1
            }
            if collisions:
                detail = "; ".join(
                    f"figure:{product} declared as {sorted(set(aliases))}"
                    for (_kind, product), aliases in sorted(collisions.items())
                )
                raise ValueError(
                    f"Planner step {step_id!r} declares the same figure "
                    f"product under more than one output alias ({detail}); "
                    "declare each figure exactly once as 'figure:<name>'."
                )
            step_payload["expected_outputs"] = normalised_outputs
        method_head = (
            str(step_payload.get("method") or "")
            .strip()
            .casefold()
            .split(" with ", 1)[0]
        )
        if method_head == "visualization" and not (
            step_payload.get("expected_outputs") or []
        ):
            raise ValueError(
                f"Planner step {step_id!r} is a visualization but declares no "
                "typed figure output; either drop the redundant step or re-emit "
                "it with exactly the intended 'figure:<name>' product."
            )
        _require_runtime_supported_product_kinds(
            step_id=str(step_id),
            inputs=step_payload.get("inputs"),
            expected_outputs=step_payload.get("expected_outputs"),
        )
        steps.append(step_payload)
    out["steps"] = steps
    specs = []
    for idx, raw_spec in enumerate(out.get("robustness_specs", []) or []):
        if not isinstance(raw_spec, dict):
            specs.append(raw_spec)
            continue
        spec_payload = {
            key: value
            for key, value in raw_spec.items()
            if key in allowed_robustness_spec
        }
        spec_id = raw_spec.get("spec_id") or f"robustness_specs[{idx}]"
        _require_exact_scientific_keys(
            raw_spec,
            allowed=allowed_robustness_spec,
            path=f"robustness_specs[{idx}]({spec_id})",
        )
        specs.append(spec_payload)
    if "robustness_specs" in out:
        out["robustness_specs"] = specs
    analysis_type = str(out.get("analysis_type") or "").strip().casefold()
    descriptive_robustness_steps = [
        str(step.get("step_id") or "")
        for step in steps
        if str(step.get("method") or "").strip().casefold().split(" with ", 1)[0]
        == "robustness_sensitivity"
    ]
    if analysis_type == "descriptive_epidemiology" and descriptive_robustness_steps:
        raise ValueError(
            "A descriptive_epidemiology plan cannot route "
            f"{descriptive_robustness_steps!r} through method "
            "'robustness_sensitivity': that executor re-estimates an already "
            "fitted primary effect with an interval. Re-emit the descriptive "
            "plan without robustness_specs/robustness_sensitivity, and use "
            "typed missingness or denominator audits for descriptive "
            "sensitivity instead."
        )
    narrative_execution_steps = [
        str(step.get("step_id") or "")
        for step in steps
        if str(step.get("method") or "").strip().casefold().split(" with ", 1)[0]
        in {
            "descriptive_interpretation",
            "result_interpretation",
            "report_writing",
            "manuscript_writing",
        }
    ]
    if narrative_execution_steps:
        raise ValueError(
            "Analysis steps cannot execute narrative interpretation or writing "
            f"methods for {narrative_execution_steps!r}. Re-emit only the "
            "evidence-producing statistical, audit, and figure steps. The "
            "gate-bound result interpreter and manuscript writer consume the "
            "verified products after analysis execution; do not generate "
            "Python code to narrate or draft them."
        )
    return out, dropped


__all__ = [
    "_canonicalise_figure_output_alias",
    "_canonicalise_planned_analysis_role",
    "_declared_field_names",
    "_is_untyped_figure_alias_output",
    "_normalise_plan_payload",
    "PlannerScientificProjectionError",
    "PlannerStructuredOutputSchemaError",
    "decode_planner_transport_payload",
    "planner_structured_output_request",
    "runtime_suffix_structured_output_request",
    "parse_runtime_plan_suffix",
    "planner_descriptive_method_guidance",
    "planner_descriptive_robustness_guidance",
    "planner_adjusted_association_owner_guidance",
    "planner_endpoint_and_optional_science_guidance",
    "render_methodological_principles",
    "normalize_literature_citation_keys",
    "bind_literature_citation_authority",
    "descriptive_claim_example_fragment",
    "literature_citation_retry_suffix",
    "required_method_layers_for_context",
    "required_method_layers_for_plan",
    "validate_literature_citation_bindings",
]
