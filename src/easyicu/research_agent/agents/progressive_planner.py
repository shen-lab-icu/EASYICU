"""Progressive Planner v2: retrieve, skeletonize, compile, revise a suffix."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from ..canonical_json import canonical_sha256
from ..authority.declared_levels import observed_levels_for
from ..cohort.schema import (
    materialized_input_column_authority,
    validate_plan_typed_bindings_against_context,
)
from ..contracts.primary_cohort import primary_analysis_cohort_plan_findings
from ..planning.adjustment_authority import validate_plan_against_adjustment_authority
from ..planning.analysis_types import (
    infer_analysis_type,
    list_analysis_types,
    validate_host_authorized_analysis_family,
)
from ..planning.literature_bindings import (
    method_layers_for_source_keys,
    missing_required_method_layers,
    validate_literature_citation_bindings,
)
from ..planning.method_literature import (
    method_binding_support,
    reporting_method_source_keys_for_guidelines,
)
from ..planning.planner_output_contract import (
    validate_fresh_planner_typed_product_specs,
)
from ..planning.preplan_know_how import verify_know_how_decisions
from ..planning.primary_result_contract import validate_required_primary_result
from ..planning.progressive_compiler import (
    compile_progressive_plan,
    progressive_cohort_concept_ids,
    progressive_output_roles_for_module,
    validate_progressive_foundation,
)
from ..planning.progressive_contract import (
    PROGRESSIVE_HOST_COMPILED_OUTPUTS,
    ProgressiveCohortIntent,
    ProgressiveFoundationMaterialization,
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressivePlanCompileReceipt,
    ProgressivePlanOutline,
    ProgressivePlannerCheckpoint,
    ProgressivePlanSkeleton,
    ProgressiveStepMaterialization,
    progressive_module_ids_for_analysis_types,
)
from ..planning.progressive_artifacts import (
    ProgressiveCompileReplayAttempt,
    ProgressiveCompilerFinding,
    ProgressivePlannerCheckpointEmitter,
)
from ..planning.progressive_resume import (
    ProgressivePrefixState,
    assemble_progressive_skeleton,
    build_progressive_checkpoint_authorities,
    compile_progressive_prefix,
    restore_progressive_resume_foundation,
    restore_progressive_resume_prefix,
    restore_progressive_resume_prompt_metrics,
    validate_progressive_materialization_coordinate,
    validate_progressive_resume_runtime_dependencies,
)
from ..planning.robustness_contract import validate_planner_robustness_specs
from ..planning.scientific_action_catalog import scientific_actions_for_analysis_type
from ..planning.scientific_review import required_method_layers_for_context
from ..providers.capabilities import llm_supports_strict_json_schema
from ..providers.llm import llm_is_mockish
from ..providers.prompt_budget import DEFAULT_MAX_PROMPT_TOKENS
from ..providers.prompts import load_prompt_pack
from ..providers.protocol import LLMClient, LLMMessage, StructuredOutputRequest
from ..providers.structured_retry import call_llm_with_structured_retry
from ..reporting.article_contract import (
    build_article_analysis_contract,
    validate_plan_against_article_contract,
)
from ..research_context.outbound import format_outbound_safe_context
from ..schema import AnalysisPlan, ResearchContext
from .progressive_payload import (
    progressive_foundation_structured_output_request,
    progressive_outline_structured_output_request,
    progressive_step_materialization_request,
)
from .plan_payload import bind_literature_citation_authority


_GUIDE = load_prompt_pack()["progressive_planner"]
_MAX_INITIAL_PARSE_RETRIES = 2
# A compiler finding gets one targeted repair. Repeating the same expensive
# current-step generation is less useful than persisting it for local replay.
_MAX_COMPILE_REVISIONS = 1
_NON_REPAIRABLE_COORDINATE_FINDINGS = frozenset(
    {
        "progressive_step_foundation_coordinate_mismatch",
        "progressive_step_materialization_mismatch",
        "progressive_step_outline_digest_mismatch",
    }
)
_MAX_OUTLINE_OUTPUT_TOKENS = 4_000
_MAX_FOUNDATION_OUTPUT_TOKENS = 4_000
_MAX_STEP_OUTPUT_TOKENS = 8_000
_MAX_REQUEST_BYTES = DEFAULT_MAX_PROMPT_TOKENS * 4
_TYPED_PRODUCT_TOKEN = re.compile(
    r"\b(?:artifact|dataset|model|statistic|table):[a-z][a-z0-9_]*\b"
)
_SEPARATE_ANALYSIS_STEP = re.compile(r"\bseparate\s+analysis\s+step\b", re.I)
_EXPLICIT_FIGURE_OUTPUT = re.compile(r"\bfigures?\b", re.I)


def _sealed_cohort_predicate_binding_rows(
    context: ResearchContext,
    variable_names: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Project exact wide-column coordinates for cohort predicates."""

    selected = set(variable_names)
    rows: list[dict[str, Any]] = []
    for variable in context.variables:
        name = str(variable.name or "").strip()
        source_concept = str(variable.source_concept or "").strip()
        analysis_window = str(variable.analysis_window or "").strip()
        if not name or name not in selected or not source_concept or not analysis_window:
            continue
        aggregation = next(
            (
                candidate
                for candidate in (
                    "max",
                    "min",
                    "mean",
                    "median",
                    "last",
                    "first",
                    "sum",
                    "count",
                )
                if name.endswith("_" + ("n" if candidate == "count" else candidate))
            ),
            None,
        )
        match = re.fullmatch(
            r"([a-z][a-z0-9_]*)\[(-?[0-9]+(?:\.[0-9]+)?),"
            r"(-?[0-9]+(?:\.[0-9]+)?)\]h",
            analysis_window,
        )
        if aggregation is None or match is None:
            continue
        rows.append(
            {
                "concept_id": source_concept,
                "physical_column": name,
                "aggregation": aggregation,
                "anchor": match.group(1),
                "start_offset_hours": float(match.group(2)),
                "end_offset_hours": float(match.group(3)),
                "matches_primary_exposure": name == context.primary_exposure,
            }
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                not bool(row["matches_primary_exposure"]),
                str(row["concept_id"]),
                str(row["physical_column"]),
            ),
        )
    )


def _preserve_literature_roster_across_targeted_repair(
    *,
    current: ProgressiveStepMaterialization,
    previous: ProgressiveStepMaterialization | None,
    outline_step: ProgressiveOutlineStep,
) -> ProgressiveStepMaterialization:
    """Carry forward valid bindings omitted by a focused compiler repair.

    A compiler-directed retry may need to expand one binding (for example,
    adding a dependence design element) without revisiting the other sealed
    sources.  Schema-imperfect providers sometimes return only the bindings
    they changed.  When the previous attempt already covered the exact
    outline-owned roster, retain its untouched model-authored applications and
    overlay the current replacements.  Extras, duplicates, or an incomplete
    previous roster still fail closed in the compiler.
    """

    if previous is None or previous.step.step_id != current.step.step_id:
        return current
    expected = tuple(outline_step.literature_citation_keys)
    previous_bindings = list(previous.step.literature_bindings)
    current_bindings = list(current.step.literature_bindings)
    previous_keys = tuple(item.citation_key for item in previous_bindings)
    current_keys = tuple(item.citation_key for item in current_bindings)
    if (
        not expected
        or len(previous_keys) != len(set(previous_keys))
        or set(previous_keys) != set(expected)
        or len(current_keys) != len(set(current_keys))
        or not set(current_keys).issubset(set(expected))
        or set(current_keys) == set(expected)
    ):
        return current
    previous_by_key = {item.citation_key: item for item in previous_bindings}
    current_by_key = {item.citation_key: item for item in current_bindings}
    merged = [
        current_by_key.get(key, previous_by_key[key])
        for key in expected
    ]
    return current.model_copy(
        update={
            "step": current.step.model_copy(
                update={"literature_bindings": merged}
            )
        }
    )


def _outline_method_layer_deadlines(
    outline: ProgressivePlanOutline,
    required_layers: Sequence[str],
) -> dict[str, int]:
    """Return the last sealed step capable of binding each required layer.

    Outline validation proves source-key availability, while step
    materialization binds exact design elements and an application.  The last
    capable step is therefore the local repair boundary: once it is compiled,
    a missing layer could only be repaired by mutating an immutable prefix.
    """

    deadlines: dict[str, int] = {}
    for layer in required_layers:
        capable = [
            index
            for index, step in enumerate(outline.steps)
            if layer
            in set(method_layers_for_source_keys(step.literature_citation_keys))
        ]
        if capable:
            deadlines[str(layer)] = max(capable)
    return deadlines


def _bound_method_layers(
    materializations: Sequence[ProgressiveStepMaterialization],
) -> set[str]:
    """Project exact method-card layers bound by materialized design elements."""

    layers: set[str] = set()
    for materialization in materializations:
        for binding in materialization.step.literature_bindings:
            layers.update(
                method_binding_support(
                    binding.citation_key,
                    binding.design_elements,
                )["matched_layers"]
            )
    return layers


def _outline_shape_contract(
    *,
    analysis_types: Sequence[str],
    module_ids_by_analysis_type: Mapping[str, Sequence[str]],
) -> str:
    """Render the exact small outline shape for schema-imperfect transports.

    Strict JSON Schema remains the transport authority.  Some otherwise usable
    OpenAI-compatible endpoints accept ``response_format`` but do not reliably
    constrain every generated key.  Keep a case-neutral textual projection of
    the same public contract in the prompt and every retry so those endpoints
    can recover without weakening host validation.
    """

    template = {
        "schema_version": "easyicu.progressive_plan_outline/1",
        "analysis_type": "<copy one exact candidate analysis family>",
        "cohort_objective": "<8-600 characters>",
        "steps": [
            {
                "step_id": "<unique lowercase id>",
                "planned_analysis_role": "<primary|secondary|sensitivity|auxiliary>",
                "module_id": "<copy one exact allowed module id>",
                "objective": "<8-600 characters>",
                "depends_on": [],
                "variable_names": ["<copy a sealed variable name>"],
                "literature_citation_keys": [],
                "scientific_action_id": None,
            }
        ],
        "rationale": "<8-1200 characters>",
    }
    return (
        "Exact ProgressivePlanOutline JSON shape (replace every angle-bracket "
        "placeholder; preserve every key; add no other keys):\n"
        + json.dumps(template, ensure_ascii=False, separators=(",", ":"))
        + "\nCandidate analysis_type values: "
        + json.dumps(list(analysis_types), ensure_ascii=False, separators=(",", ":"))
        + "\nAllowed module_id values by analysis_type: "
        + json.dumps(
            {
                analysis_type: list(module_ids_by_analysis_type[analysis_type])
                for analysis_type in analysis_types
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nplanned_analysis_role must be exactly one of: primary, secondary, "
        "sensitivity, auxiliary. depends_on, variable_names, and "
        "literature_citation_keys must always be JSON arrays. "
        "scientific_action_id must be a retrieved action id or null."
    )


def _foundation_shape_contract(
    *,
    outline_sha256: str,
    host_cohort: ProgressiveCohortIntent | None,
    required_cohort_selection_mode: str | None = None,
    required_cohort_name: str | None = None,
) -> str:
    """Project the exact foundation envelope without adding case science."""

    cohort: Mapping[str, Any]
    if host_cohort is not None:
        cohort = host_cohort.model_dump(mode="json")
    elif required_cohort_selection_mode == "predicate_filtered":
        cohort = {
            "name": required_cohort_name or "<1-128 characters>",
            "selection_mode": "predicate_filtered",
            "inclusion": [
                {
                    "concept_id": "<copy an allowed cohort concept id>",
                    "anchor": "<copy an allowed anchor>",
                    "start_offset_hours": "<number>",
                    "end_offset_hours": "<greater number>",
                    "aggregation": "<max|min|mean|median|last|first|any|all|count|sum>",
                    "op": "<==|!=|<|<=|>|>=|in|not_in|missing|not_missing>",
                    "value": {
                        "mode": "<none|string|number|boolean|string_list|number_list>",
                        "string_value": None,
                        "number_value": None,
                        "boolean_value": None,
                        "string_list": [],
                        "number_list": [],
                    },
                }
            ],
            "exclusion": [],
        }
    else:
        cohort = {
            "name": required_cohort_name or "<1-128 characters>",
            "selection_mode": "<all_input_rows|predicate_filtered>",
            "inclusion": [],
            "exclusion": [],
        }
    template = {
        "schema_version": "easyicu.progressive_plan_foundation/1",
        "outline_sha256": outline_sha256,
        "foundation": {
            "cohort": cohort,
            "display_labels": [],
            "robustness_intents": [],
            "know_how_decisions": [],
        },
    }
    cohort_instruction = (
        "The cohort object shown above is caller-bound; copy it exactly."
        if host_cohort is not None
        else (
            "Replace the cohort placeholders and use predicate_filtered only "
            "with at least one valid inclusion or exclusion predicate."
        )
    )
    return (
        "Exact ProgressiveFoundationMaterialization JSON shape (preserve the "
        "root foundation wrapper and every displayed key; add no other keys):\n"
        + json.dumps(template, ensure_ascii=False, separators=(",", ":"))
        + "\nCopy schema_version and outline_sha256 exactly. "
        + cohort_instruction
        + " display_labels, robustness_intents, and know_how_decisions must "
        "always be JSON arrays, including when empty.\n"
        "If display_labels is nonempty, each item has exactly "
        '{"key":"<1-256 characters>","value":"<1-256 characters>"}. '
        "If robustness_intents is nonempty, each item has exactly "
        '{"spec_id":"<lowercase id>","axis":"<cohort|missing|outcome>",'
        '"description":"<8-600 characters>",'
        '"missing_strategy":"<none|complete_case>",'
        '"complete_case_variables":[]}. '
        "If know_how_decisions is nonempty, each item has exactly "
        '{"card_id":"<authorized id>","card_version":"<authorized version>",'
        '"card_sha256":"<authorized 64-char digest>",'
        '"claim_id":"<authorized id>",'
        '"disposition":"<adopted|rejected|unresolved|requires_confirmation>",'
        '"reason_code":"<lowercase id>","rationale":"<1-500 characters>",'
        '"citation_ids":["<authorized id>"]}. '
        "For every cohort predicate preserve exactly concept_id, anchor, "
        "start_offset_hours, end_offset_hours, aggregation, op, and value. "
        "Its value object must preserve all six displayed keys; populate only "
        "the field selected by mode and leave the others null or empty."
    )


def _step_materialization_shape_contract(
    *,
    outline_step: ProgressiveOutlineStep,
    outline_step_sha256: str,
) -> str:
    """Project the exact current-step envelope and closed step key roster."""

    step = {
        "step_id": outline_step.step_id,
        "planned_analysis_role": outline_step.planned_analysis_role,
        "module_id": outline_step.module_id,
        "objective": outline_step.objective,
        "depends_on": list(outline_step.depends_on),
        "raw_inputs": [],
        "product_inputs": [],
        "outputs": [],
        "scientific_action_id": outline_step.scientific_action_id,
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
    }
    template = {
        "schema_version": "easyicu.progressive_step_materialization/1",
        "outline_step_sha256": outline_step_sha256,
        "foundation": None,
        "step": step,
    }
    return (
        "Exact ProgressiveStepMaterialization JSON shape (preserve this root "
        "wrapper and every step key; add no other keys):\n"
        + json.dumps(template, ensure_ascii=False, separators=(",", ":"))
        + "\nCopy schema_version, outline_step_sha256, foundation=null, and "
        "the six outline-owned step coordinates exactly. Replace only the "
        "module-specific executable null/empty defaults required by the "
        "current method card. Never return variable_names, "
        "literature_citation_keys, literature_design_bindings, cohort, or "
        "expected_outputs inside step. raw_inputs may contain only sealed "
        "variable names, never kind:product tokens; governed products belong "
        "only in product_inputs.\n"
        "Nested item shapes, when used: product_inputs items are exactly "
        '{"producer_step_id":"<preceding step id>",'
        '"product_id":"<kind:product>"}; outputs items are exactly '
        '{"product_id":"<kind:product>","semantic_role":"<allowed role>"}; '
        "table_one_variables items are exactly "
        '{"name":"<sealed variable>",'
        '"summary":"<mean_sd|median_iqr|both|count_percent>"}; '
        "model_terms items are exactly "
        '{"name":"<sealed variable>","role":"<exposure|covariate>",'
        '"coding":"<continuous|binary|categorical|ordinal_linear>",'
        '"reference_level_index":null}; literature_bindings items are exactly '
        '{"citation_key":"<sealed key>","design_elements":["<allowed element>"],'
        '"application":"<8-1200 characters>","divergence":null}.'
    )


def _required_separate_analysis_products(
    context: ResearchContext,
) -> tuple[str, ...]:
    """Return nonstandard products explicitly assigned to a separate step.

    ``must_have_outputs`` is still a legacy prose field.  We do not infer new
    science from it; this helper recognizes only an explicit structural
    instruction and canonical typed product tokens already written by the
    caller.  Standard host products keep their registered owners.  Everything
    else needs a ``custom_analysis`` outline coordinate so it cannot disappear
    inside the generic robustness replay step.
    """

    preferences = context.user_preferences
    text = str(getattr(preferences, "must_have_outputs", None) or "")
    host_products = {
        product
        for products in PROGRESSIVE_HOST_COMPILED_OUTPUTS.values()
        for product, _role in products
    }
    required: list[str] = []
    for line in text.splitlines():
        if not _SEPARATE_ANALYSIS_STEP.search(line):
            continue
        for product in _TYPED_PRODUCT_TOKEN.findall(line):
            if product not in host_products and product not in required:
                required.append(product)
    return tuple(required)


def _requires_visualization_step(context: ResearchContext) -> bool:
    """Return whether the run-bound output request explicitly requires a figure."""

    preferences = context.user_preferences
    text = str(getattr(preferences, "must_have_outputs", None) or "")
    return bool(_EXPLICIT_FIGURE_OUTPUT.search(text))


def _bind_runtime_action_dependencies(
    outline: ProgressivePlanOutline,
) -> ProgressivePlanOutline:
    """Seal uniquely implied host-owned product edges in an outline.

    A runtime action contract already fixes which typed products an action
    consumes and emits. Requiring the Planner to repeat that same mechanical
    edge in ``depends_on`` adds no scientific choice and makes a valid design
    depend on redundant model bookkeeping. Add only edges with one preceding
    runtime-contract owner; ambiguous or unavailable owners remain untouched
    so the ordinary fail-closed validators can report them.
    """

    actions = {
        action.action_id: action
        for action in scientific_actions_for_analysis_type(outline.analysis_type).actions
    }
    prior_owners: dict[str, list[str]] = {}
    bound_steps: list[ProgressiveOutlineStep] = []
    for step in outline.steps:
        action = actions.get(str(step.scientific_action_id or ""))
        contract = action.runtime_contract if action is not None else None
        dependencies = list(step.depends_on)
        if contract is not None:
            for product_id in contract.required_product_inputs:
                owners = prior_owners.get(product_id, [])
                if len(owners) == 1 and owners[0] not in dependencies:
                    dependencies.append(owners[0])
        bound_step = (
            step
            if dependencies == step.depends_on
            else step.model_copy(update={"depends_on": dependencies})
        )
        bound_steps.append(bound_step)
        if contract is not None:
            for product_id, _semantic_role in contract.outputs:
                prior_owners.setdefault(product_id, []).append(step.step_id)
    if bound_steps == outline.steps:
        return outline
    return outline.model_copy(update={"steps": bound_steps})


def _complete_case_variable_roster(
    context: ResearchContext,
    variable_names: Sequence[str],
) -> tuple[str, ...]:
    """Return analysis fields eligible to determine complete-case membership."""

    executable_variables = _executable_analysis_variable_roster(
        context,
        variable_names,
    )
    return tuple(
        name
        for name in executable_variables
        if (
            (descriptor := context.variable(name)) is None
            or descriptor.observation_semantics is None
            or descriptor.observation_semantics.kind != "conditional_event_time"
        )
    )


def _executable_analysis_variable_roster(
    context: ResearchContext,
    variable_names: Sequence[str],
) -> tuple[str, ...]:
    """Exclude host navigation coordinates from statistical field rosters."""

    column_authority = materialized_input_column_authority(context)
    executable_columns = set(column_authority.executable_columns)
    return tuple(
        name
        for name in variable_names
        if (
            not column_authority.sealed_columns or name in executable_columns
        )
    )


def _tokens(value: object) -> set[str]:
    folded = str(value or "").casefold()
    compounds = re.findall(r"[a-z0-9_]+", folded)
    pieces = re.findall(r"[a-z0-9]+", folded.replace("_", " "))
    return {token for token in (*compounds, *pieces) if len(token) >= 2}


def candidate_analysis_types(
    context: ResearchContext,
    *,
    max_candidates: int = 4,
) -> tuple[str, ...]:
    """Retrieve a generous, question-relevant family subset.

    Scoring uses the research question only.  Case notes can contain required
    audits and sensitivities without changing the headline study family, which
    is exactly how a measurement-audit keyword displaced E1's association.
    """

    question = str(context.research_question or "").casefold()
    question_tokens = _tokens(question)
    scored: list[tuple[int, int, str]] = []
    for position, spec in enumerate(list_analysis_types()):
        score = 0
        for trigger in spec.trigger_terms:
            phrase = str(trigger).casefold().strip()
            if not phrase:
                continue
            if phrase in question:
                score += 5 + len(phrase.split())
            else:
                score += len(question_tokens & _tokens(phrase))
        if score:
            scored.append((score, -position, spec.key))
    scored.sort(reverse=True)
    candidates = [key for _score, _position, key in scored]
    inferred = infer_analysis_type(context).key
    if context.primary_exposure and context.target_outcome:
        candidates.insert(0, "association_study")
    candidates.extend([inferred, "descriptive_epidemiology"])
    authorized: list[str] = []
    for key in candidates:
        if key in authorized:
            continue
        try:
            validate_host_authorized_analysis_family(context, key)
        except ValueError:
            continue
        authorized.append(key)
        if len(authorized) >= max(1, int(max_candidates)):
            break
    if not authorized:
        raise ProgressivePlanCompileError(
            "progressive_no_authorized_analysis_type",
            "retrieval found no analysis family authorized by ResearchContext",
            path="analysis_type",
        )
    return tuple(authorized)


def select_progressive_variables(
    context: ResearchContext,
    *,
    max_variables: int = 48,
) -> tuple[str, ...]:
    """Retrieve a bounded, run-specific variable set without dropping anchors."""

    preferences = context.user_preferences
    search_text = "\n".join(
        value
        for value in (
            context.research_question,
            context.notes,
            getattr(preferences, "evaluation_focus", None),
            getattr(preferences, "data_constraints", None),
            getattr(preferences, "must_have_outputs", None),
            getattr(preferences, "timing_and_design", None),
            getattr(preferences, "subgroup_sensitivity", None),
        )
        if value
    ).casefold()
    text_tokens = _tokens(search_text)
    exact = {
        str(value).strip()
        for value in (
            context.primary_exposure,
            context.target_outcome,
            *context.cohort.outcome_columns,
            *context.cohort.time_columns,
            *(getattr(preferences, "covariates", ()) or ()),
        )
        if str(value or "").strip()
    }
    primary = context.variable(str(context.primary_exposure or ""))
    primary_concepts = {
        str(value).casefold()
        for value in (
            getattr(primary, "source_concept", None),
            *(getattr(primary, "derived_from_concepts", ()) or ()),
        )
        if str(value or "").strip()
    }
    scored: list[tuple[int, int, str]] = []
    for position, variable in enumerate(context.variables):
        name = variable.name
        folded_name = name.casefold()
        source = str(variable.source_concept or "").casefold()
        description_tokens = _tokens(variable.description)
        score = 0
        if name in exact:
            score += 10_000
        if folded_name and folded_name in search_text:
            score += 2_000
        if source and source in search_text:
            score += 1_000
        name_tokens = _tokens(name) - {
            "icu",
            "max",
            "min",
            "mean",
            "first",
            "last",
            "time",
            "measured",
            "value",
            "flag",
            "code",
        }
        score += 3_000 * len(name_tokens & text_tokens)
        score += 20 * len(description_tokens & text_tokens)
        if variable.role.value in {"outcome", "intervention", "treatment"}:
            score += 300
        if variable.role.value == "demographic":
            score += 2_500
        if (
            source in primary_concepts
            or {str(item).casefold() for item in variable.derived_from_concepts}
            & primary_concepts
        ):
            score += 500
        if any(
            folded_name.startswith(f"{concept}_")
            for concept in primary_concepts
            if concept
        ):
            score += 400
        # Prefer the compact audit-bearing representatives of a repeated
        # concept family before its alternate summaries. This keeps breadth
        # across components without encoding any ICU variable name.
        for suffix, bonus in (
            ("_max", 120),
            ("_measured", 110),
            ("_n", 100),
            ("_first_time", 90),
            ("_last_time", 80),
        ):
            if folded_name.endswith(suffix):
                score += bonus
                break
        scored.append((score, -position, name))
    scored.sort(reverse=True)
    limit = max(1, int(max_variables))
    variable_by_name = {variable.name: variable for variable in context.variables}
    source_counts: dict[str, int] = {}
    selected: list[str] = []
    for _score, _position, name in scored:
        variable = variable_by_name[name]
        source = str(variable.source_concept or name).casefold()
        if source in primary_concepts:
            source_limit = 5
        elif any(
            source.startswith(f"{concept}_") for concept in primary_concepts if concept
        ):
            source_limit = 3
        else:
            source_limit = 4
        if name not in exact and source_counts.get(source, 0) >= source_limit:
            continue
        selected.append(name)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= limit:
            break
    # Restore ResearchContext order so opaque level indices and prompt reviews
    # are deterministic and easy to compare across runs.
    selected_set = set(selected)
    ordered = tuple(
        variable.name for variable in context.variables if variable.name in selected_set
    )
    if not ordered:
        raise ProgressivePlanCompileError(
            "progressive_variable_retrieval_empty",
            "ResearchContext exposes no variables to the progressive planner",
            path="variables",
        )
    return ordered


def _action_catalog(
    analysis_types: Sequence[str],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    action_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    for analysis_type in analysis_types:
        catalog = scientific_actions_for_analysis_type(analysis_type)
        for action in catalog.actions:
            if action.execution_mode == "not_available":
                continue
            if action.action_id not in action_ids:
                action_ids.append(action.action_id)
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "action_id": action.action_id,
                    "execution_mode": action.execution_mode,
                    "produces": action.produces,
                    "required_inputs": list(action.required_inputs),
                    "runtime_contract": (
                        {
                            "outputs": [
                                {
                                    "product_id": product_id,
                                    "semantic_role": semantic_role,
                                }
                                for product_id, semantic_role in action.runtime_contract.outputs
                            ],
                            "required_product_inputs": list(
                                action.runtime_contract.required_product_inputs
                            ),
                            "article_roles": list(action.runtime_contract.article_roles),
                            "standard_executor": action.runtime_contract.standard_executor,
                        }
                        if action.runtime_contract is not None
                        else None
                    ),
                }
            )
    return tuple(action_ids), rows


def _parse_model(raw: str, model: type[Any]) -> Any:
    payload = json.loads(str(raw or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("progressive Planner response root must be an object")
    return model.model_validate(payload)


def _parse_step_materialization(
    raw: str,
    *,
    outline_step: ProgressiveOutlineStep | None = None,
    outline_step_sha256: str | None = None,
) -> ProgressiveStepMaterialization:
    payload = json.loads(str(raw or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("progressive Planner response root must be an object")
    if outline_step is not None and not outline_step_sha256:
        raise ValueError("host outline step digest is required")
    # Host coordinates remain fail-closed input. They are intentionally not
    # rewritten here: the coordinate validator below the parser must observe
    # and reject any model drift in the digest, step id, role, module,
    # objective, dependencies, or scientific action.
    step = payload.get("step")
    raw_inputs = step.get("raw_inputs") if isinstance(step, dict) else None
    if isinstance(raw_inputs, list) and all(
        isinstance(raw_input, str) for raw_input in raw_inputs
    ):
        normalized = [raw_input.strip() for raw_input in raw_inputs]
        if all(normalized):
            # Repeating an exact normalized source-column name adds no
            # scientific meaning and is a common structured-generation
            # artifact. Preserve first-seen order and leave every other roster
            # strict so dependency and sensitivity conflicts still fail closed.
            canonical_step = dict(step)
            product_inputs = canonical_step.get("product_inputs")
            bound_products = (
                {
                    str(item.get("product_id") or "").strip()
                    for item in product_inputs
                    if isinstance(item, Mapping)
                }
                if isinstance(product_inputs, list)
                else set()
            )
            # A typed product is not a cohort column.  When the same exact
            # product already has its governed producer/product edge, remove
            # only that redundant raw-input spelling.  An unbound product token
            # or any other unknown name still reaches the compiler and fails.
            canonical_step["raw_inputs"] = list(
                dict.fromkeys(
                    value
                    for value in normalized
                    if not (
                        _TYPED_PRODUCT_TOKEN.fullmatch(value)
                        and value in bound_products
                    )
                )
            )
            payload = dict(payload)
            payload["step"] = canonical_step
    return ProgressiveStepMaterialization.model_validate(payload)


def _parse_foundation_materialization(
    raw: str,
    *,
    host_cohort: ProgressiveCohortIntent | None,
    allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None = None,
) -> ProgressiveFoundationMaterialization:
    payload = json.loads(str(raw or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("progressive Planner response root must be an object")
    foundation = payload.get("foundation")
    if isinstance(foundation, dict):
        decisions = foundation.get("know_how_decisions")
        if isinstance(decisions, list):
            # An exact duplicate carries no additional scientific choice and is
            # a common structured-generation artifact. Collapse only bytewise-
            # equivalent JSON objects; two different decisions for the same
            # card/claim coordinate still reach the model validator and fail
            # closed as a real conflict.
            unique_decisions: list[Any] = []
            seen_decisions: set[str] = set()
            for decision in decisions:
                if isinstance(decision, dict) and allowed_know_how_decisions:
                    card = allowed_know_how_decisions.get(
                        str(decision.get("card_id") or "")
                    )
                    expected_citations = (
                        (card.get("claims") or {}).get(decision.get("claim_id"))
                        if isinstance(card, Mapping)
                        else None
                    )
                    observed_citations = decision.get("citation_ids")
                    # Citation order is presentation, not scientific authority.
                    # Canonicalize a permutation of the exact authority set;
                    # missing, added, or repeated citations remain untouched and
                    # fail in verify_know_how_decisions downstream.
                    if (
                        isinstance(observed_citations, list)
                        and expected_citations is not None
                        and len(observed_citations) == len(expected_citations)
                        and set(observed_citations) == set(expected_citations)
                    ):
                        decision = dict(decision)
                        decision["citation_ids"] = list(expected_citations)
                coordinate = json.dumps(
                    decision,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if coordinate in seen_decisions:
                    continue
                seen_decisions.add(coordinate)
                unique_decisions.append(decision)
            if unique_decisions != decisions:
                payload = dict(payload)
                foundation = dict(foundation)
                foundation["know_how_decisions"] = unique_decisions
                payload["foundation"] = foundation
    if host_cohort is not None:
        foundation = payload.get("foundation")
        if not isinstance(foundation, dict):
            raise ValueError("progressive Planner foundation must be an object")
        payload = dict(payload)
        payload["foundation"] = {
            **foundation,
            "cohort": host_cohort.model_dump(mode="json"),
        }
    return ProgressiveFoundationMaterialization.model_validate(payload)


def _primary_or_first_step_index(plan: AnalysisPlan) -> int:
    for index, step in enumerate(plan.steps):
        if step.planned_analysis_role == "primary":
            return index
    return 0


def _step_index_from_error(plan: AnalysisPlan, message: str) -> int:
    for index, step in enumerate(plan.steps):
        if step.step_id and step.step_id in message:
            return index
    return _primary_or_first_step_index(plan)


def _article_reporting_source_keys(
    *,
    article_context: ResearchContext,
    analysis_type: str,
    enforce_article_contract: bool,
) -> tuple[str, ...]:
    if not enforce_article_contract:
        return ()
    contract = build_article_analysis_contract(
        article_context,
        analysis_type=analysis_type,
    )
    return reporting_method_source_keys_for_guidelines(
        contract.reporting_guidelines
    )


def _accept_compiled_plan(
    *,
    plan: AnalysisPlan,
    agent_context: ResearchContext,
    article_context: ResearchContext,
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str],
    allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None,
    enforce_article_contract: bool,
    llm: LLMClient,
) -> None:
    """Apply the same fresh-plan authorities after host compilation."""

    try:
        if plan.robustness_specs:
            validate_planner_robustness_specs(plan.robustness_specs)
        validate_literature_citation_bindings(
            plan,
            allowed_literature_citation_keys,
            context=agent_context,
            direct_comparator_keys=direct_comparator_literature_keys,
        )
        if allowed_know_how_decisions is not None:
            verify_know_how_decisions(
                plan.know_how_decisions,
                allowed_know_how_decisions,
            )
        if enforce_article_contract:
            contract = build_article_analysis_contract(
                article_context,
                analysis_type=plan.analysis_type,
            )
            findings = validate_plan_against_article_contract(
                plan=plan,
                contract=contract,
            )
            missing_roles = sorted(
                {
                    str(role)
                    for finding in findings
                    for role in (finding.detail or {}).get("missing_roles", [])
                    if str(role).strip()
                }
            )
            if "robustness" in contract.required_roles and not plan.robustness_specs:
                missing_roles = sorted({*missing_roles, "robustness_specs"})
            if missing_roles:
                raise ValueError(
                    "progressive article contract is missing required role(s): "
                    + ", ".join(missing_roles)
                )
        if not llm_is_mockish(llm):
            validate_fresh_planner_typed_product_specs(
                plan,
                context=agent_context,
            )
        validate_plan_typed_bindings_against_context(
            plan=plan,
            context=agent_context,
        )
        validate_plan_against_adjustment_authority(
            plan=plan,
            context=agent_context,
        )
        cohort_findings = primary_analysis_cohort_plan_findings(plan=plan)
        if cohort_findings:
            raise ValueError(
                "primary cohort contract findings: "
                + json.dumps(
                    [item.detail for item in cohort_findings],
                    ensure_ascii=False,
                    default=str,
                )
            )
        validate_required_primary_result(plan=plan, context=agent_context)
    except ProgressivePlanCompileError:
        raise
    except ValueError as exc:
        message = str(exc)
        index = _step_index_from_error(plan, message)
        step = plan.steps[index] if plan.steps else None
        raise ProgressivePlanCompileError(
            "progressive_fresh_plan_gate_failed",
            message,
            step_id=step.step_id if step is not None else None,
            step_index=index if step is not None else None,
            path="fresh_plan_acceptance",
        ) from exc


class ProgressivePlannerAgent:
    """Emit a small plan skeleton, compile it, and revise only failed suffixes."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.last_prompt_metrics: dict[str, Any] = {}
        self.last_compile_receipt: Optional[ProgressivePlanCompileReceipt] = None
        self.last_outline: Optional[ProgressivePlanOutline] = None
        self.last_foundation: Optional[ProgressiveFoundationMaterialization] = None
        self.last_materializations: list[ProgressiveStepMaterialization] = []
        self.last_compile_failure_attempts: list[
            ProgressiveCompileReplayAttempt
        ] = []
        self.last_skeleton: Optional[ProgressivePlanSkeleton] = None
        self.last_resume_validated = False
        self.last_dropped_plan_keys: dict[str, list[str]] = {
            "top_level": [],
            "steps": [],
        }

    def capture_efficiency_metrics(self) -> None:
        """Copy the active Planner budget receipt into checkpoint metrics."""

        snapshot = getattr(self.llm, "efficiency_snapshot", None)
        if callable(snapshot) and self.last_prompt_metrics:
            self.last_prompt_metrics["efficiency_budget"] = snapshot()

    @staticmethod
    def _request_authorities(
        context: ResearchContext,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], list[dict[str, Any]]]:
        analysis_types = candidate_analysis_types(context)
        variables = select_progressive_variables(context)
        action_ids, action_rows = _action_catalog(analysis_types)
        return analysis_types, variables, action_ids, action_rows

    @staticmethod
    def _retrieved_data_cards(
        context: ResearchContext,
        variables: Sequence[str],
    ) -> list[dict[str, Any]]:
        selected = set(variables)
        variable_map = {variable.name: variable for variable in context.variables}
        cards = []
        for variable in context.variables:
            if variable.name not in selected:
                continue
            card = variable.model_dump(
                mode="json",
                include={
                    "name",
                    "role",
                    "dtype",
                    "source_concept",
                    "derived_from_concepts",
                },
            )
            level_count = len(
                observed_levels_for(name=variable.name, variables=variable_map)
            )
            card["closed_domain_level_count"] = level_count
            card["supports_closed_level_contrast"] = level_count >= 2
            cards.append(card)
        return cards

    @staticmethod
    def _user_prompt(
        context: ResearchContext,
        *,
        article_context: ResearchContext | None = None,
        analysis_types: Sequence[str],
        variables: Sequence[str],
        action_rows: Sequence[Mapping[str, Any]],
        allowed_literature_citation_keys: Sequence[str] = (),
        know_how_context: str = "",
        planning_contract_context: str = "",
    ) -> str:
        contract_context = article_context or context
        module_ids_by_analysis_type = {
            analysis_type: list(
                progressive_module_ids_for_analysis_types((analysis_type,))
            )
            for analysis_type in analysis_types
        }
        article_contracts = []
        for analysis_type in analysis_types:
            contract = build_article_analysis_contract(
                contract_context,
                analysis_type=analysis_type,
            )
            article_contracts.append(
                {
                    "analysis_type": analysis_type,
                    "analysis_family": contract.analysis_family,
                    "available_modules": list(
                        progressive_module_ids_for_analysis_types((analysis_type,))
                    ),
                    "required_roles": list(contract.required_roles),
                    "planner_owned_result_roles": list(
                        contract.planner_owned_result_roles
                    ),
                    "requirements": [
                        {
                            # Article-role requirements and executable outline
                            # modules are different vocabularies.  Calling both
                            # coordinates ``module_id`` led schema-imperfect
                            # OpenAI-compatible providers to copy requirement
                            # ids into ProgressiveOutlineStep.module_id, where
                            # they can never validate.
                            "article_requirement_id": item.module_id,
                            "role": item.role,
                            "required": item.required,
                        }
                        for item in contract.requirements
                    ],
                }
            )
        blocks = [
            "PROGRESSIVE PLANNER RUN AUTHORITY",
            "Candidate analysis families (choose exactly one):\n"
            + json.dumps(list(analysis_types), ensure_ascii=False),
            "Allowed ProgressiveOutlineStep.module_id values by candidate "
            "analysis family (use only these exact strings):\n"
            + json.dumps(
                module_ids_by_analysis_type,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            _outline_shape_contract(
                analysis_types=analysis_types,
                module_ids_by_analysis_type=module_ids_by_analysis_type,
            ),
            "Retrieved scientific actions (only these may be selected):\n"
            + json.dumps(list(action_rows), ensure_ascii=False, separators=(",", ":")),
            "Sealed literature citation keys:\n"
            + json.dumps(list(allowed_literature_citation_keys), ensure_ascii=False),
            "Candidate-specific host article role contracts:\n"
            + json.dumps(article_contracts, ensure_ascii=False, separators=(",", ":")),
            "Research question and sealed study anchors:\n"
            + json.dumps(
                {
                    "research_question": context.research_question,
                    "primary_exposure": context.primary_exposure,
                    "target_outcome": context.target_outcome,
                    "cohort": context.cohort.model_dump(
                        mode="json",
                        include={
                            "cohort_name",
                            "database",
                            "n_stays",
                            "id_columns",
                            "outcome_columns",
                        },
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Retrieved data cards (high-level fields only):\n"
            + json.dumps(
                ProgressivePlannerAgent._retrieved_data_cards(context, variables),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
        if planning_contract_context:
            blocks.append(
                "Additional run-specific article/task contract (binding; never "
                "global). Its pre-plan family classification is provisional; "
                "the candidate-specific contracts above govern the family you "
                "select, while every explicit task requirement remains binding:\n"
                + planning_contract_context
            )
        separate_products = _required_separate_analysis_products(context)
        if separate_products:
            blocks.append(
                "Host-resolved separate-analysis obligations:\n"
                + json.dumps(list(separate_products), ensure_ascii=False)
                + "\nEach listed product must have its own custom_analysis "
                "outline step. Do not fold that step into robustness_replay; "
                "the host will materialize its exact method, inputs, outputs, "
                "and product edges later."
            )
        if _requires_visualization_step(context):
            blocks.append(
                "Host-resolved presentation obligation:\nThe run-bound required "
                "outputs explicitly require a figure. Include at least one "
                "visualization outline step; do not delegate the figure to a "
                "report step. The host will bind its exact typed figure product "
                "when that step is materialized."
            )
        singleton_host_modules = {
            module_id: [product_id for product_id, _role in products]
            for module_id, products in PROGRESSIVE_HOST_COMPILED_OUTPUTS.items()
        }
        blocks.append(
            "Host-compiled singleton module ownership:\n"
            + json.dumps(
                singleton_host_modules,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\nEach listed module may appear at most once in the outline because "
            "the host owns its fixed typed products. Combine compatible intents "
            "for that module into its single step (including all replayable "
            "robustness intents in one robustness_replay step). A separately "
            "required nonstandard analysis needs a custom_analysis step with its "
            "own run-authorized product; never duplicate a singleton module."
        )
        blocks.append(
            "Primary-lineage rule: every secondary custom_analysis result must "
            "be downstream of the primary analysis through depends_on so its "
            "later typed input can consume the primary product. Put descriptive "
            "absolute-risk or other headline context after the primary model; "
            "an isolated raw-cohort custom result cannot satisfy a Planner-owned "
            "article result role. When an action runtime_contract lists "
            "required_product_inputs, depends_on must include the preceding "
            "action step that emits each required product. The host seals only "
            "a uniquely implied missing runtime-contract edge; ambiguous owners "
            "remain an error."
        )
        if know_how_context:
            blocks.append(
                "Retrieved protocol know-how (record every required claim disposition):\n"
                + know_how_context
            )
        blocks.append(
            "Return only a concise ProgressivePlanOutline. Do not return raw or "
            "product inputs, output tokens, Table 1 fields, model terms, level "
            "indices, denominator or missingness policies, or literature "
            "applications. For each step select only the retrieved variable "
            "names and sealed literature keys that its later materialization "
            "needs. The host requests and validates executable details only "
            "when it materializes the current step."
            " Every steps[].module_id must be copied exactly from the allowed "
            "ProgressiveOutlineStep.module_id list for the chosen family; "
            "article_requirement_id values are article obligations, not "
            "executable module ids."
        )
        return "\n\n".join(blocks)

    @classmethod
    def request_messages(
        cls,
        context: ResearchContext,
        *,
        know_how_context: str = "",
        planning_contract_context: str = "",
        strict_transport_schema: bool = False,
        structured_output: StructuredOutputRequest | None = None,
        allowed_literature_citation_keys: Sequence[str] = (),
    ) -> list[LLMMessage]:
        del strict_transport_schema, structured_output
        analysis_types, variables, _action_ids, action_rows = cls._request_authorities(
            context
        )
        return [
            LLMMessage(role="system", content=_GUIDE),
            LLMMessage(
                role="user",
                content=cls._user_prompt(
                    context,
                    article_context=context,
                    analysis_types=analysis_types,
                    variables=variables,
                    action_rows=action_rows,
                    allowed_literature_citation_keys=allowed_literature_citation_keys,
                    know_how_context=know_how_context,
                    planning_contract_context=planning_contract_context,
                ),
            ),
        ]

    @classmethod
    def request_metrics(
        cls,
        context: ResearchContext,
        *,
        know_how_context: str = "",
        planning_contract_context: str = "",
        strict_transport_schema: bool = False,
        structured_output: StructuredOutputRequest | None = None,
    ) -> dict[str, Any]:
        analysis_types, variables, action_ids, _rows = cls._request_authorities(context)
        request = structured_output
        if request is None and strict_transport_schema:
            request = progressive_outline_structured_output_request(
                analysis_types=analysis_types,
                variable_names=variables,
                scientific_action_ids=action_ids,
            )
        messages = cls.request_messages(
            context,
            know_how_context=know_how_context,
            planning_contract_context=planning_contract_context,
            strict_transport_schema=strict_transport_schema,
            structured_output=request,
        )
        message_bytes = sum(len(item.content.encode("utf-8")) for item in messages)
        schema_bytes = request.payload_bytes if request is not None else 0
        return {
            "message_payload_bytes": message_bytes,
            "structured_output_payload_bytes": schema_bytes,
            "structured_output_authority_sha256": (
                request.authority_sha256 if request is not None else None
            ),
            "total_bytes": message_bytes + schema_bytes,
            "selected_variable_count": len(variables),
            "selected_variable_roster_sha256": canonical_sha256(list(variables)),
            "candidate_analysis_types": list(analysis_types),
            "selected_scientific_action_ids": list(action_ids),
            "planner_strategy": "progressive_v2",
        }

    @staticmethod
    def _validate_outline_authority(
        outline: ProgressivePlanOutline,
        *,
        analysis_types: Sequence[str],
        variable_names: Sequence[str],
        allowed_literature_citation_keys: Sequence[str],
        required_custom_products: Sequence[str] = (),
        required_visualization_step: bool = False,
        closed_domain_variables: Sequence[str] | None = None,
        primary_exposure: str | None = None,
        target_outcome: str | None = None,
        context_required_method_layers: Sequence[str] | None = None,
    ) -> None:
        if outline.analysis_type not in set(analysis_types):
            raise ProgressivePlanCompileError(
                "progressive_outline_analysis_type_unavailable",
                f"outline selected unavailable analysis type {outline.analysis_type!r}",
                path="analysis_type",
            )
        allowed_actions, _rows = _action_catalog((outline.analysis_type,))
        allowed = set(allowed_actions)
        available_variables = set(variable_names)
        available_citations = set(allowed_literature_citation_keys)
        closed_domains = (
            set(closed_domain_variables)
            if closed_domain_variables is not None
            else None
        )
        allowed_modules = set(
            progressive_module_ids_for_analysis_types((outline.analysis_type,))
        )
        singleton_owners: dict[str, list[str]] = {}
        for step in outline.steps:
            if step.module_id in PROGRESSIVE_HOST_COMPILED_OUTPUTS:
                singleton_owners.setdefault(step.module_id, []).append(step.step_id)
        duplicated_singletons = {
            module_id: step_ids
            for module_id, step_ids in singleton_owners.items()
            if len(step_ids) > 1
        }
        if duplicated_singletons:
            findings = tuple(
                {
                    "module_id": module_id,
                    "step_ids": step_ids,
                    "host_products": [
                        product_id
                        for product_id, _role in PROGRESSIVE_HOST_COMPILED_OUTPUTS[
                            module_id
                        ]
                    ],
                }
                for module_id, step_ids in sorted(duplicated_singletons.items())
            )
            raise ProgressivePlanCompileError(
                "progressive_outline_host_module_repeated",
                "host-compiled singleton module(s) appear more than once: "
                + ", ".join(sorted(duplicated_singletons)),
                path="steps",
                findings=findings,
            )
        if required_visualization_step and not any(
            step.module_id == "visualization" for step in outline.steps
        ):
            raise ProgressivePlanCompileError(
                "progressive_outline_visualization_owner_missing",
                "the run-bound output contract requires a visualization outline step",
                path="steps",
                findings=(
                    {
                        "required_module_id": "visualization",
                        "source": "user_preferences.must_have_outputs",
                    },
                ),
            )
        if required_custom_products and not any(
            step.module_id == "custom_analysis" for step in outline.steps
        ):
            raise ProgressivePlanCompileError(
                "progressive_outline_separate_analysis_owner_missing",
                "run-specific separate-analysis product(s) require a "
                "custom_analysis outline step: "
                + ", ".join(required_custom_products),
                path="steps",
                findings=(
                    {"required_products": list(required_custom_products)},
                ),
            )
        primary_step_ids = {
            step.step_id
            for step in outline.steps
            if step.planned_analysis_role == "primary"
        }
        upstream_by_step = {
            step.step_id: set(step.depends_on) for step in outline.steps
        }

        def _has_primary_ancestor(step_id: str) -> bool:
            pending = list(upstream_by_step.get(step_id, ()))
            visited: set[str] = set()
            while pending:
                candidate = pending.pop()
                if candidate in primary_step_ids:
                    return True
                if candidate in visited:
                    continue
                visited.add(candidate)
                pending.extend(upstream_by_step.get(candidate, ()))
            return False

        detached_secondary_custom = [
            step.step_id
            for step in outline.steps
            if step.module_id == "custom_analysis"
            and step.planned_analysis_role == "secondary"
            and not _has_primary_ancestor(step.step_id)
        ]
        if detached_secondary_custom:
            raise ProgressivePlanCompileError(
                "progressive_outline_secondary_custom_off_primary_lineage",
                "secondary custom_analysis steps must descend from the primary "
                "analysis so their typed result remains on the primary lineage: "
                + ", ".join(detached_secondary_custom),
                path="steps.depends_on",
                findings=({"step_ids": detached_secondary_custom},),
            )
        for index, step in enumerate(outline.steps):
            if step.module_id not in allowed_modules:
                raise ProgressivePlanCompileError(
                    "progressive_outline_module_unavailable",
                    f"outline module {step.module_id!r} is unavailable for "
                    f"analysis type {outline.analysis_type!r}",
                    step_id=step.step_id,
                    step_index=index,
                    path="module_id",
                )
            unknown_variables = sorted(set(step.variable_names) - available_variables)
            if unknown_variables:
                raise ProgressivePlanCompileError(
                    "progressive_outline_variable_unavailable",
                    "outline selected variable(s) outside the retrieved roster: "
                    + ", ".join(unknown_variables),
                    step_id=step.step_id,
                    step_index=index,
                    path="variable_names",
                )
            if step.module_id == "exposure_outcome_distribution" and (
                closed_domains is not None
                and (
                    (
                        primary_exposure is not None
                        and target_outcome is not None
                        and not {primary_exposure, target_outcome}.issubset(
                            closed_domains
                        )
                    )
                    or (
                        (primary_exposure is None or target_outcome is None)
                        and len(set(step.variable_names) & closed_domains) < 2
                    )
                )
            ):
                raise ProgressivePlanCompileError(
                    "progressive_outline_distribution_domain_unavailable",
                    "exposure_outcome_distribution is categorical-only and "
                    "requires at least two selected variables with closed "
                    "domains of two or more levels; use adjusted_association, "
                    "custom_analysis, or visualization for a continuous exposure",
                    step_id=step.step_id,
                    step_index=index,
                    path="variable_names",
                )
            unknown_citations = sorted(
                set(step.literature_citation_keys) - available_citations
            )
            if unknown_citations:
                raise ProgressivePlanCompileError(
                    "progressive_outline_citation_unavailable",
                    "outline selected citation(s) outside the sealed roster: "
                    + ", ".join(unknown_citations),
                    step_id=step.step_id,
                    step_index=index,
                    path="literature_citation_keys",
                )
            action = step.scientific_action_id
            if action is not None and action not in allowed:
                raise ProgressivePlanCompileError(
                    "progressive_outline_action_unavailable",
                    f"outline action {action!r} is not available for "
                    f"analysis type {outline.analysis_type!r}",
                    step_id=step.step_id,
                    step_index=index,
                    path="scientific_action_id",
                )
        if context_required_method_layers is not None and available_citations:
            required_method_layers = set(context_required_method_layers)
            if any(
                step.module_id == "adjusted_association" for step in outline.steps
            ):
                required_method_layers.add("interpretation")
            selected_outline_citations = tuple(
                citation
                for step in outline.steps
                for citation in step.literature_citation_keys
            )
            missing_outline_layers = sorted(
                required_method_layers
                - set(method_layers_for_source_keys(selected_outline_citations))
            )
            if missing_outline_layers:
                raise ProgressivePlanCompileError(
                    "progressive_outline_method_layer_unbound",
                    "outline citation allocation cannot cover required method "
                    "layer(s): " + ", ".join(missing_outline_layers),
                    path="steps.literature_citation_keys",
                    findings=(
                        {
                            "missing_method_layers": missing_outline_layers,
                            "selected_citation_keys": list(
                                dict.fromkeys(selected_outline_citations)
                            ),
                        },
                    ),
                )

    @staticmethod
    def _foundation_prompt(
        *,
        context: ResearchContext,
        outline: ProgressivePlanOutline,
        outline_sha256: str,
        variables: Sequence[str],
        know_how_context: str,
        planning_contract_context: str,
        host_cohort: ProgressiveCohortIntent | None,
        required_cohort_selection_mode: str | None = None,
        required_cohort_name: str | None = None,
    ) -> str:
        blocks = [
            "PROGRESSIVE PLAN-FOUNDATION AUTHORITY",
            "Host-validated outline and digest:\n"
            + json.dumps(
                {
                    "outline": outline.model_dump(mode="json"),
                    "outline_sha256": outline_sha256,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Outbound-safe data authority for plan-wide choices:\n"
            + format_outbound_safe_context(context, variable_names=variables),
            _foundation_shape_contract(
                outline_sha256=outline_sha256,
                host_cohort=host_cohort,
                required_cohort_selection_mode=required_cohort_selection_mode,
                required_cohort_name=required_cohort_name,
            ),
        ]
        predicate_bindings = _sealed_cohort_predicate_binding_rows(
            context,
            variables,
        )
        if predicate_bindings:
            blocks.append(
                "Sealed executable cohort-predicate bindings. When a cohort "
                "predicate uses one of these source concepts and windows, copy "
                "the exact aggregation and coordinates from a matching row; "
                "do not substitute a descriptor aggregation_default. If the "
                "predicate operationalizes the configured primary exposure, "
                "use the row with matches_primary_exposure=true:\n"
                + json.dumps(
                    list(predicate_bindings),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        if planning_contract_context:
            blocks.append(
                "Run-specific article/task contract (binding):\n"
                + planning_contract_context
            )
        if know_how_context:
            blocks.append("Retrieved protocol know-how (binding):\n" + know_how_context)
        if host_cohort is not None:
            blocks.append(
                "Caller-bound cohort authority (host owned; copy exactly and do "
                "not reinterpret):\n"
                + host_cohort.model_dump_json()
            )
        blocks.append(
            "Return one ProgressiveFoundationMaterialization only. Bind cohort "
            "selection, display labels, robustness intents, and any authorized "
            "know-how decisions. Emit at most one decision for each exact "
            "card_id/claim_id pair. Every cohort predicate must use an interval "
            "with end_offset_hours greater than start_offset_hours. Do not return "
            "executable step fields."
        )
        blocks.append(
            "robustness_intents are only for the generic host replay owner. "
            "The progressive v1 foundation can compile only an explicit "
            "complete-case missing-data intent with its exact variable list. "
            "Timing, readmission/cohort restriction, outcome-definition, and "
            "functional-form analyses belong in custom_analysis outline steps "
            "and must not be duplicated into robustness_intents. A conditional "
            "event-time column whose event-absent rows are not applicable must "
            "not define complete-case membership."
        )
        if outline.analysis_type == "descriptive_epidemiology":
            blocks.append(
                "This descriptive family has no fitted primary effect or interval. "
                "Return robustness_intents=[]; denominator and complete-case "
                "availability belong to typed audit steps in the outline."
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _materialization_prompt(
        *,
        context: ResearchContext,
        outline: ProgressivePlanOutline,
        outline_step: ProgressiveOutlineStep,
        outline_step_sha256: str,
        variables: Sequence[str],
        action_rows: Sequence[Mapping[str, Any]],
        allowed_literature_citation_keys: Sequence[str],
        know_how_context: str,
        planning_contract_context: str,
        prefix_summary: Sequence[Mapping[str, Any]],
        available_product_refs: Sequence[tuple[str, str]],
        compiler_observation: Mapping[str, Any] | None = None,
    ) -> str:
        blocks = [
            "PROGRESSIVE CURRENT-STEP MATERIALIZATION AUTHORITY",
            "Selected analysis family:\n" + outline.analysis_type,
            "High-level cohort objective:\n" + outline.cohort_objective,
            "Current outline step and host digest:\n"
            + json.dumps(
                {
                    "outline_step": outline_step.model_dump(mode="json"),
                    "outline_step_sha256": outline_step_sha256,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Immutable materialized prefix summary (not editable):\n"
            + json.dumps(
                list(prefix_summary),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Host product registry visible to this materialization. These are "
            "planning edges only; runtime consumption still requires verified "
            "executed evidence/capsule authority:\n"
            + json.dumps(
                [
                    {"producer_step_id": producer, "product_id": product}
                    for producer, product in available_product_refs
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Retrieved scientific action/method cards for the selected family:\n"
            + json.dumps(
                list(action_rows),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Sealed literature citation keys:\n"
            + json.dumps(
                list(allowed_literature_citation_keys),
                ensure_ascii=False,
            ),
            "Outbound-safe data authority for this current step:\n"
            + format_outbound_safe_context(context, variable_names=variables),
            _step_materialization_shape_contract(
                outline_step=outline_step,
                outline_step_sha256=outline_step_sha256,
            ),
            "Compiler-owned semantic_role values for this module (use only "
            "these exact strings when outputs is nonempty):\n"
            + json.dumps(
                list(progressive_output_roles_for_module(outline_step.module_id)),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
        if planning_contract_context:
            blocks.append(
                "Run-specific article/task contract (binding):\n"
                + planning_contract_context
            )
        if know_how_context:
            blocks.append(
                "Retrieved protocol know-how (binding):\n" + know_how_context
            )
        if compiler_observation:
            blocks.append(
                "HOST COMPILER OBSERVATION FOR THIS CURRENT STEP:\n"
                + json.dumps(
                    dict(compiler_observation),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        if outline_step.module_id == "visualization":
            action_by_id = {
                str(row.get("action_id")): row
                for row in action_rows
                if row.get("action_id")
            }
            direct_contracts = []
            for prior in prefix_summary:
                if prior.get("step_id") not in outline_step.depends_on:
                    continue
                action_row = action_by_id.get(
                    str(prior.get("scientific_action_id") or "")
                )
                runtime_contract = (
                    action_row.get("runtime_contract")
                    if isinstance(action_row, Mapping)
                    else None
                )
                if isinstance(runtime_contract, Mapping) and runtime_contract.get(
                    "standard_executor"
                ):
                    direct_contracts.append(
                        {
                            "producer_step_id": prior.get("step_id"),
                            "runtime_contract": runtime_contract,
                        }
                    )
            executors = {
                str(item["runtime_contract"].get("standard_executor") or "")
                for item in direct_contracts
            }
            exact_profile = [
                {
                    "producer_step_id": item["producer_step_id"],
                    "product_id": output.get("product_id"),
                }
                for item in direct_contracts
                for output in item["runtime_contract"].get("outputs", [])
                if isinstance(output, Mapping) and output.get("product_id")
            ]
            blocks.append(
                "Rendering-only source contract: set raw_inputs=[] and do not "
                "bind artifact:analysis_cohort or any raw cohort column. Select "
                "at most four direct table:/statistic: product_inputs. Every "
                "selected parent must contribute reader-visible values to this "
                "figure and must retain its own independently traceable "
                "source-data projection; move other article roles to separate "
                "rendering steps instead of binding unused context."
            )
            if (
                len(executors) == 1
                and "" not in executors
                and 1 <= len(exact_profile) <= 4
            ):
                blocks.append(
                    "Host-renderable direct-result profile (binding): set "
                    "product_inputs to exactly these producer/product pairs in "
                    "this order so the deterministic composite renderer can own "
                    "the step:\n"
                    + json.dumps(
                        exact_profile,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        if outline_step.module_id == "adjusted_association":
            reserved_coordinates = tuple(
                name
                for name in materialized_input_column_authority(
                    context
                ).reserved_navigation_coordinates
                if name in set(variables)
            )
            blocks.append(
                "Adjusted-model term contract: declare the outcome only in the "
                "outcome field. model_terms must contain exactly one exposure "
                "matching primary_exposure plus prespecified covariates; never "
                "include the outcome as a model term or covariate."
            )
            if reserved_coordinates:
                blocks.append(
                    "Host-reserved navigation coordinates are not statistical "
                    "model fields and must not appear as primary_exposure, "
                    "outcome, or model_terms: "
                    + json.dumps(
                        list(reserved_coordinates),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        if outline_step.module_id == "cohort_definition":
            blocks.append(
                "Cohort-definition detail contract: keep raw_inputs=[], "
                "product_inputs=[], outputs=[], and every module-specific "
                "field null or empty. The host owns canonical cohort products."
            )
        if outline_step.module_id == "measurement_audit":
            blocks.append(
                "Measurement-audit detail contract: outputs must contain at "
                "least one table: product, and each semantic_role must be one "
                "exact compiler-owned value listed above. Use a unique role per "
                "output. Keep Table 1, association, contrast, denominator, and "
                "sensitivity-only fields null or empty."
            )
        if outline_step.module_id == "table_one":
            blocks.append(
                "Table 1 detail contract: set table_one_group_by to one selected "
                "closed-domain variable; set table_one_mode to exactly "
                "independent_inference or descriptive_smd_only; provide at "
                "least one table_one_variables item using the exact nested "
                "shape above; include the group and summarized variables in "
                "raw_inputs. Keep outputs=[] because the host owns the canonical "
                "Table 1 product."
            )
        if outline_step.module_id == "exposure_outcome_distribution":
            blocks.append(
                "Distribution detail contract: set primary_exposure, outcome, "
                "event_level_index, reference_exposure_level_index, "
                "comparison_exposure_level_index, denominator_policy, "
                "missing_exposure_policy, missing_outcome_policy, and a "
                "confidence_level between 0 and 1. Use only host-published "
                "zero-based opaque level indices. Include both variables in "
                "raw_inputs and keep outputs=[] because the host owns the "
                "canonical distribution product. Copy policy values as exact "
                "closed literals: denominator_policy is exactly "
                "all_declared_rows or observed_outcome_rows; "
                "missing_exposure_policy is exactly fail_closed or "
                "exclude_from_denominator; missing_outcome_policy is exactly "
                "fail_closed, exclude_from_denominator, or "
                "structural_absence_is_non_event. Set custom_method=null, "
                "Table 1 fields null/empty, model_terms=[], "
                "primary_contrast_level_index=null, and "
                "sensitivity_spec_ids=[]. Do not paraphrase a literal and do "
                "not use an empty string where null is required."
            )
        if outline_step.module_id == "custom_analysis":
            current_action = next(
                (
                    row
                    for row in action_rows
                    if row.get("action_id") == outline_step.scientific_action_id
                ),
                None,
            )
            if current_action and current_action.get("runtime_contract"):
                blocks.append(
                    "Host-owned scientific-action contract (binding): copy its "
                    "outputs exactly, bind exactly its required_product_inputs "
                    "from preceding depended-on producers, and do not substitute "
                    "artifact products or contextual tables:\n"
                    + json.dumps(
                        current_action["runtime_contract"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            blocks.append(
                "Custom-analysis output contract: a generic custom result uses "
                "a runtime-materializable typed product id, "
                "semantic_role='custom', and sensitivity_spec_ids=[]. Keep a "
                "secondary custom step in this generic shape. A "
                "scientific sensitivity uses exactly one table output with "
                "semantic_role='scientific_sensitivity' and a non-empty unique "
                "sensitivity_spec_ids roster, and is valid only when the outline "
                "role is sensitivity. Do not mix these two shapes."
            )
        blocks.append(
            "Return one ProgressiveStepMaterialization only. Copy every "
            "outline-owned coordinate exactly. Return foundation=null; the host "
            "already sealed it. Bind every sealed literature citation exactly "
            "once with a card-supported design application. Do not return or "
            "rewrite any prefix or future step."
        )
        return "\n\n".join(blocks)

    def _compile_and_accept(
        self,
        skeleton: ProgressivePlanSkeleton,
        *,
        agent_context: ResearchContext,
        article_context: ResearchContext,
        allowed_literature_citation_keys: Sequence[str],
        direct_comparator_literature_keys: Sequence[str],
        allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None,
        enforce_article_contract: bool,
    ) -> tuple[AnalysisPlan, ProgressivePlanCompileReceipt]:
        reporting_source_keys = _article_reporting_source_keys(
            article_context=article_context,
            analysis_type=skeleton.analysis_type,
            enforce_article_contract=enforce_article_contract,
        )
        plan, receipt = compile_progressive_plan(
            skeleton=skeleton,
            context=agent_context,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            allowed_know_how_decisions=allowed_know_how_decisions,
            host_reporting_method_source_keys=reporting_source_keys,
        )
        _accept_compiled_plan(
            plan=plan,
            agent_context=agent_context,
            article_context=article_context,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            allowed_know_how_decisions=allowed_know_how_decisions,
            enforce_article_contract=enforce_article_contract,
            llm=self.llm,
        )
        return plan, receipt

    def _materialize_remaining_steps(
        self,
        prefix_state: ProgressivePrefixState,
        *,
        context: ResearchContext,
        outline: ProgressivePlanOutline,
        foundation_materialization: ProgressiveFoundationMaterialization,
        scientific_action_ids: Sequence[str],
        action_rows: Sequence[Mapping[str, Any]],
        allowed_literature_citation_keys: Sequence[str],
        allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None,
        reporting_method_source_keys: Sequence[str],
        planning_contract_context: str,
        progress_callback: Optional[Callable[[Any], None]],
        checkpoint_emitter: ProgressivePlannerCheckpointEmitter,
        resumed: bool,
    ) -> ProgressivePrefixState:
        """Materialize and locally repair only the uncompiled suffix."""

        foundation = foundation_materialization.foundation
        required_outline_layers = set(required_method_layers_for_context(context))
        method_layer_deadlines = _outline_method_layer_deadlines(
            outline,
            sorted(required_outline_layers),
        )
        for step_index in range(len(prefix_state.steps), len(outline.steps)):
            outline_step = outline.steps[step_index]
            step_variables = tuple(outline_step.variable_names)
            step_citations = tuple(outline_step.literature_citation_keys)
            outline_step_sha256 = canonical_sha256(
                outline_step.model_dump(mode="json")
            )
            step_schema = None
            if llm_supports_strict_json_schema(self.llm):
                step_schema = progressive_step_materialization_request(
                    outline_step=outline_step,
                    outline_step_sha256=outline_step_sha256,
                    variable_names=step_variables,
                    executable_variable_names=(
                        _executable_analysis_variable_roster(
                            context,
                            step_variables,
                        )
                    ),
                    scientific_action_ids=scientific_action_ids,
                    allowed_literature_citation_keys=step_citations,
                    available_product_refs=prefix_state.available_product_refs,
                )
            compiler_observation: Mapping[str, Any] | None = None
            self.last_compile_failure_attempts = []
            for revision in range(_MAX_COMPILE_REVISIONS + 1):
                materialization_prompt = self._materialization_prompt(
                    context=context,
                    outline=outline,
                    outline_step=outline_step,
                    outline_step_sha256=outline_step_sha256,
                    variables=step_variables,
                    action_rows=action_rows,
                    allowed_literature_citation_keys=step_citations,
                    know_how_context="",
                    planning_contract_context=planning_contract_context,
                    prefix_summary=prefix_state.prompt_summary,
                    available_product_refs=prefix_state.available_product_refs,
                    compiler_observation=compiler_observation,
                )
                step_messages = [
                    LLMMessage(role="system", content=_GUIDE),
                    LLMMessage(role="user", content=materialization_prompt),
                ]
                step_payload_bytes = sum(
                    len(item.content.encode("utf-8")) for item in step_messages
                ) + (step_schema.payload_bytes if step_schema is not None else 0)
                if step_payload_bytes > _MAX_REQUEST_BYTES:
                    raise ProgressivePlanCompileError(
                        "progressive_step_prompt_budget_exceeded",
                        f"current-step request uses {step_payload_bytes} bytes; "
                        f"limit={_MAX_REQUEST_BYTES}",
                        step_id=outline_step.step_id,
                        step_index=step_index,
                        path="planner_request",
                    )
                self.last_prompt_metrics[
                    "step_materialization_attempt_payload_bytes"
                ].append(step_payload_bytes)
                self.last_prompt_metrics[
                    "step_materialization_attempt_schema_sha256"
                ].append(step_schema.authority_sha256 if step_schema else None)
                if resumed:
                    self.last_prompt_metrics[
                        "current_run_step_materialization_attempt_payload_bytes"
                    ].append(step_payload_bytes)
                    self.last_prompt_metrics[
                        "current_run_step_materialization_attempt_schema_sha256"
                    ].append(
                        step_schema.authority_sha256 if step_schema else None
                    )
                materialization = call_llm_with_structured_retry(
                    self.llm,
                    step_messages,
                    parser=lambda raw: _parse_step_materialization(
                        raw,
                        outline_step=outline_step,
                        outline_step_sha256=outline_step_sha256,
                    ),
                    role="progressive_planner_step_materialization",
                    max_retries=1,
                    max_tokens=_MAX_STEP_OUTPUT_TOKENS,
                    temperature=0.2,
                    include_failed_response_on_retry=False,
                    progress_callback=progress_callback,
                    structured_output=step_schema,
                    format_reminder=_step_materialization_shape_contract(
                        outline_step=outline_step,
                        outline_step_sha256=outline_step_sha256,
                    )
                    + "\nReturn exactly one current-step materialization; never "
                    "return other steps or flatten step fields into the root.",
                )
                self.capture_efficiency_metrics()
                prior_materialization = (
                    self.last_compile_failure_attempts[-1].materialization
                    if self.last_compile_failure_attempts
                    else None
                )
                materialization = (
                    _preserve_literature_roster_across_targeted_repair(
                        current=materialization,
                        previous=prior_materialization,
                        outline_step=outline_step,
                    )
                )
                try:
                    validate_progressive_materialization_coordinate(
                        materialization,
                        outline_step=outline_step,
                        outline_step_sha256=outline_step_sha256,
                        step_index=step_index,
                    )
                    candidate_state = compile_progressive_prefix(
                        prefix_state,
                        materialization,
                        outline=outline,
                        foundation=foundation,
                        context=context,
                        allowed_literature_citation_keys=(
                            allowed_literature_citation_keys
                        ),
                        allowed_know_how_decisions=allowed_know_how_decisions,
                        reporting_method_source_keys=reporting_method_source_keys,
                    )
                    deadline_layers = sorted(
                        layer
                        for layer, deadline in method_layer_deadlines.items()
                        if deadline == step_index
                    )
                    missing_deadline_layers = sorted(
                        set(deadline_layers)
                        - _bound_method_layers(candidate_state.materializations)
                    )
                    if missing_deadline_layers:
                        raise ProgressivePlanCompileError(
                            "progressive_step_required_method_layer_unbound",
                            "the current step is the last outline-sealed owner "
                            "capable of binding required method layer(s): "
                            + ", ".join(missing_deadline_layers),
                            step_id=outline_step.step_id,
                            step_index=step_index,
                            path="literature_bindings",
                            findings=[
                                {
                                    "missing_method_layers": missing_deadline_layers,
                                    "repair_scope": "current_step_only",
                                    "sealed_citation_keys": list(step_citations),
                                }
                            ],
                        )
                    if step_index == len(outline.steps) - 1:
                        assert candidate_state.plan is not None
                        missing_method_layers = missing_required_method_layers(
                            candidate_state.plan,
                            allowed_literature_citation_keys,
                            context=context,
                        )
                        if missing_method_layers:
                            raise ProgressivePlanCompileError(
                                "progressive_final_method_layer_unbound",
                                "the completed plan still lacks typed method-source "
                                "coverage for case-applicable layer(s): "
                                + ", ".join(missing_method_layers),
                                step_id=outline_step.step_id,
                                step_index=step_index,
                                path="literature_bindings",
                                findings=[
                                    {
                                        "missing_method_layers": list(
                                            missing_method_layers
                                        ),
                                        "repair_scope": "current_final_step_only",
                                    }
                                ],
                            )
                except ProgressivePlanCompileError as exc:
                    self.last_compile_failure_attempts.append(
                        ProgressiveCompileReplayAttempt(
                            revision=revision,
                            step_schema_authority_sha256=(
                                step_schema.authority_sha256
                                if step_schema is not None
                                else None
                            ),
                            materialization=materialization,
                            materialization_sha256=canonical_sha256(
                                materialization.model_dump(mode="json")
                            ),
                            compiler_finding=(
                                ProgressiveCompilerFinding.from_details(exc.details)
                            ),
                        )
                    )
                    if exc.reason_code in _NON_REPAIRABLE_COORDINATE_FINDINGS:
                        raise
                    if revision >= _MAX_COMPILE_REVISIONS:
                        raise
                    if exc.step_index is not None and int(exc.step_index) < step_index:
                        raise ProgressivePlanCompileError(
                            "progressive_materialization_invalidated_prefix",
                            "current-step materialization exposed a finding in the "
                            "already compiled prefix",
                            step_id=exc.step_id,
                            step_index=exc.step_index,
                            path=exc.path,
                        ) from exc
                    self.last_prompt_metrics["compile_revision_count"] += 1
                    if resumed:
                        self.last_prompt_metrics[
                            "current_run_compile_revision_count"
                        ] += 1
                    compiler_observation = exc.details
                    continue
                prefix_state = candidate_state
                self.last_compile_failure_attempts = []
                self.last_materializations = list(prefix_state.materializations)
                self.last_prompt_metrics[
                    "step_materialization_payload_bytes"
                ].append(step_payload_bytes)
                self.last_prompt_metrics[
                    "step_materialization_schema_sha256"
                ].append(step_schema.authority_sha256 if step_schema else None)
                self.last_prompt_metrics["step_materialization_count"] += 1
                if resumed:
                    self.last_prompt_metrics[
                        "current_run_step_materialization_count"
                    ] += 1
                checkpoint_emitter.emit(
                    stage="step",
                    outline=outline,
                    foundation=foundation_materialization,
                    materializations=self.last_materializations,
                    prompt_metrics=self.last_prompt_metrics,
                )
                break
        return prefix_state

    def run(
        self,
        context: ResearchContext,
        *,
        allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None = None,
        allowed_literature_citation_keys: Sequence[str] | None = None,
        direct_comparator_literature_keys: Sequence[str] | None = None,
        know_how_context: str = "",
        enforce_article_contract: bool = False,
        article_contract_context: Optional[ResearchContext] = None,
        planning_contract_context: str = "",
        progress_callback: Optional[Callable[[Any], None]] = None,
        checkpoint_callback: Optional[
            Callable[[ProgressivePlannerCheckpoint], None]
        ] = None,
        resume_checkpoint: ProgressivePlannerCheckpoint | None = None,
        resume_dependency_context: Mapping[str, Any] | None = None,
        required_primary_cohort_selection_mode: str | None = None,
    ) -> AnalysisPlan:
        self.last_resume_validated = False
        self.last_compile_failure_attempts = []
        if bool(allowed_know_how_decisions) != bool(know_how_context):
            raise ValueError(
                "Progressive Planner know-how authority and prompt must be supplied together"
            )
        article_context = article_contract_context or context
        if required_primary_cohort_selection_mode not in {
            None,
            "all_input_rows",
            "predicate_filtered",
        }:
            raise ValueError(
                "required_primary_cohort_selection_mode is unavailable"
            )
        host_cohort = (
            ProgressiveCohortIntent(
                name=context.cohort.cohort_name,
                selection_mode="all_input_rows",
                inclusion=[],
                exclusion=[],
            )
            if required_primary_cohort_selection_mode == "all_input_rows"
            else None
        )
        allowed_citations = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in (allowed_literature_citation_keys or ())
                if str(value).strip()
            )
        )
        direct_keys = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in (direct_comparator_literature_keys or ())
                if str(value).strip()
            )
        )
        analysis_types, variables, action_ids, action_rows = self._request_authorities(
            context
        )
        context_variable_map = {
            variable.name: variable for variable in context.variables
        }
        closed_domain_variables = tuple(
            variable.name
            for variable in context.variables
            if len(
                observed_levels_for(
                    name=variable.name,
                    variables=context_variable_map,
                )
            )
            >= 2
        )
        resolved_planning_contract_context = bind_literature_citation_authority(
            planning_contract_context,
            allowed_citations,
            direct_comparator_keys=direct_keys,
            required_method_layers=required_method_layers_for_context(context),
        )
        required_custom_products = _required_separate_analysis_products(context)
        required_visualization_step = _requires_visualization_step(context)
        if resume_checkpoint is not None:
            validate_progressive_resume_runtime_dependencies(
                resume_dependency_context
            )
        scientific_authority = {
            "analysis_types": list(analysis_types),
            "variables": list(variables),
            "scientific_action_ids": list(action_ids),
            "allowed_literature_citation_keys": list(allowed_citations),
            "direct_comparator_literature_keys": list(direct_keys),
            "allowed_know_how_decisions": dict(
                allowed_know_how_decisions or {}
            ),
            "know_how_context": know_how_context,
            "planning_contract_context": resolved_planning_contract_context,
            "required_primary_cohort_selection_mode": (
                required_primary_cohort_selection_mode
            ),
            "required_visualization_step": required_visualization_step,
            "host_cohort": (
                host_cohort.model_dump(mode="json")
                if host_cohort is not None
                else None
            ),
        }
        checkpoint_authorities = build_progressive_checkpoint_authorities(
            context=context,
            article_context=article_context,
            scientific_authority=scientific_authority,
            runtime_dependency_authority=resume_dependency_context,
        )
        checkpoint_emitter = ProgressivePlannerCheckpointEmitter(
            callback=checkpoint_callback,
            request_authority_sha256=(
                resume_checkpoint.request_authority_sha256
                if resume_checkpoint is not None
                else checkpoint_authorities.request_authority_sha256
            ),
            source_checkpoint=resume_checkpoint,
        )
        outline_schema = None
        if llm_supports_strict_json_schema(self.llm):
            outline_schema = progressive_outline_structured_output_request(
                analysis_types=analysis_types,
                variable_names=variables,
                scientific_action_ids=action_ids,
                allowed_literature_citation_keys=allowed_citations,
            )
        user_prompt = self._user_prompt(
            context,
            article_context=article_context,
            analysis_types=analysis_types,
            variables=variables,
            action_rows=action_rows,
            allowed_literature_citation_keys=allowed_citations,
            know_how_context=know_how_context,
            planning_contract_context=resolved_planning_contract_context,
        )
        user_prompt_without_know_how = self._user_prompt(
            context,
            article_context=article_context,
            analysis_types=analysis_types,
            variables=variables,
            action_rows=action_rows,
            allowed_literature_citation_keys=allowed_citations,
            planning_contract_context=resolved_planning_contract_context,
        )
        messages = [
            LLMMessage(role="system", content=_GUIDE),
            LLMMessage(role="user", content=user_prompt),
        ]
        message_bytes = sum(len(item.content.encode("utf-8")) for item in messages)
        schema_bytes = outline_schema.payload_bytes if outline_schema else 0
        total_bytes = message_bytes + schema_bytes
        if total_bytes > _MAX_REQUEST_BYTES:
            raise ProgressivePlanCompileError(
                "progressive_prompt_budget_exceeded",
                f"initial request uses {total_bytes} bytes; limit={_MAX_REQUEST_BYTES}",
                path="planner_request",
            )
        current_prompt_metrics = {
            "message_payload_bytes": message_bytes,
            "structured_output_payload_bytes": schema_bytes,
            "structured_output_authority_sha256": (
                outline_schema.authority_sha256 if outline_schema else None
            ),
            "total_bytes": total_bytes,
            "outline_request_payload_bytes": total_bytes,
            "outline_schema_bytes": schema_bytes,
            "foundation_request_payload_bytes": 0,
            "foundation_schema_bytes": 0,
            "foundation_structured_output_authority_sha256": None,
            "without_know_how_total_bytes": (
                len(_GUIDE.encode("utf-8"))
                + len(user_prompt_without_know_how.encode("utf-8"))
                + schema_bytes
            ),
            "selected_variable_count": len(variables),
            "selected_variable_roster": list(variables),
            "selected_variable_roster_sha256": canonical_sha256(list(variables)),
            "candidate_analysis_types": list(analysis_types),
            "selected_scientific_action_ids": list(action_ids),
            "planner_strategy": "progressive_v2",
            "foundation_cohort_owner": (
                "host_required_primary_cohort"
                if host_cohort is not None
                else "planner"
            ),
            "required_primary_cohort_selection_mode": (
                required_primary_cohort_selection_mode
            ),
            "required_visualization_step": required_visualization_step,
            "resume_dependency_authority_sha256": (
                checkpoint_authorities.resume_dependency_authority_sha256
            ),
            "compile_revision_count": 0,
            "step_materialization_count": 0,
            "step_materialization_payload_bytes": [],
            "step_materialization_schema_sha256": [],
            "step_materialization_attempt_payload_bytes": [],
            "step_materialization_attempt_schema_sha256": [],
            "suffix_revision_count": 0,
            "full_revision_count": 0,
            "suffix_request_payload_bytes": [],
        }
        if resume_checkpoint is not None:
            self.last_prompt_metrics = restore_progressive_resume_prompt_metrics(
                checkpoint=resume_checkpoint,
                current_prompt_metrics=current_prompt_metrics,
                expected_dependency_sha256=(
                    checkpoint_authorities.resume_dependency_authority_sha256
                ),
            )
            outline = resume_checkpoint.outline
        else:
            self.last_prompt_metrics = current_prompt_metrics

            def parse_outline(raw: str) -> ProgressivePlanOutline:
                parsed = _parse_model(raw, ProgressivePlanOutline)
                parsed = _bind_runtime_action_dependencies(parsed)
                self._validate_outline_authority(
                    parsed,
                    analysis_types=analysis_types,
                    variable_names=variables,
                    allowed_literature_citation_keys=allowed_citations,
                    required_custom_products=required_custom_products,
                    required_visualization_step=required_visualization_step,
                    closed_domain_variables=closed_domain_variables,
                    primary_exposure=context.primary_exposure,
                    target_outcome=context.target_outcome,
                    context_required_method_layers=(
                        required_method_layers_for_context(context)
                    ),
                )
                return parsed

            outline = call_llm_with_structured_retry(
                self.llm,
                messages,
                parser=parse_outline,
                role="progressive_planner_outline",
                max_retries=_MAX_INITIAL_PARSE_RETRIES,
                max_tokens=_MAX_OUTLINE_OUTPUT_TOKENS,
                temperature=0.2,
                include_failed_response_on_retry=False,
                progress_callback=progress_callback,
                structured_output=outline_schema,
                format_reminder=_outline_shape_contract(
                    analysis_types=analysis_types,
                    module_ids_by_analysis_type={
                        analysis_type: progressive_module_ids_for_analysis_types(
                            (analysis_type,)
                        )
                        for analysis_type in analysis_types
                    },
                )
                + "\nReturn one concise ProgressivePlanOutline only. Do not "
                "include executable step-detail fields. Never use "
                "article_requirement_id as module_id.",
            )
            self.capture_efficiency_metrics()
        self._validate_outline_authority(
            outline,
            analysis_types=analysis_types,
            variable_names=variables,
            allowed_literature_citation_keys=allowed_citations,
            required_custom_products=required_custom_products,
            required_visualization_step=required_visualization_step,
            closed_domain_variables=closed_domain_variables,
            primary_exposure=context.primary_exposure,
            target_outcome=context.target_outcome,
            context_required_method_layers=required_method_layers_for_context(
                context
            ),
        )
        self.last_outline = outline
        self.last_foundation = None
        self.last_materializations = []
        outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))
        if resume_checkpoint is not None:
            if self.last_prompt_metrics.get("outline_sha256") != outline_sha256:
                raise ProgressivePlanCompileError(
                    "progressive_resume_outline_digest_mismatch",
                    "development checkpoint metrics identify another outline",
                    path="resume_checkpoint.prompt_metrics.outline_sha256",
                )
        else:
            self.last_prompt_metrics["outline_sha256"] = outline_sha256
            checkpoint_emitter.emit(
                stage="outline",
                outline=outline,
                foundation=None,
                materializations=self.last_materializations,
                prompt_metrics=self.last_prompt_metrics,
            )

        foundation_schema = None
        if llm_supports_strict_json_schema(self.llm):
            complete_case_variables = _complete_case_variable_roster(
                context,
                variables,
            )
            foundation_schema = progressive_foundation_structured_output_request(
                outline_sha256=outline_sha256,
                variable_names=variables,
                complete_case_variable_names=complete_case_variables,
                cohort_concept_ids=progressive_cohort_concept_ids(context, variables),
                allowed_know_how_decisions=allowed_know_how_decisions,
                required_cohort_selection_mode=(
                    required_primary_cohort_selection_mode
                ),
                required_cohort_name=(
                    context.cohort.cohort_name
                    if required_primary_cohort_selection_mode is not None
                    else None
                ),
                analysis_type=outline.analysis_type,
            )
        foundation_prompt = self._foundation_prompt(
            context=context,
            outline=outline,
            outline_sha256=outline_sha256,
            variables=variables,
            know_how_context=know_how_context,
            planning_contract_context=resolved_planning_contract_context,
            host_cohort=host_cohort,
            required_cohort_selection_mode=required_primary_cohort_selection_mode,
            required_cohort_name=(
                context.cohort.cohort_name
                if required_primary_cohort_selection_mode is not None
                else None
            ),
        )
        foundation_messages = [
            LLMMessage(role="system", content=_GUIDE),
            LLMMessage(role="user", content=foundation_prompt),
        ]
        foundation_message_bytes = sum(
            len(item.content.encode("utf-8")) for item in foundation_messages
        )
        foundation_schema_bytes = (
            foundation_schema.payload_bytes if foundation_schema is not None else 0
        )
        foundation_total_bytes = foundation_message_bytes + foundation_schema_bytes
        if foundation_total_bytes > _MAX_REQUEST_BYTES:
            raise ProgressivePlanCompileError(
                "progressive_foundation_prompt_budget_exceeded",
                f"foundation request uses {foundation_total_bytes} bytes; "
                f"limit={_MAX_REQUEST_BYTES}",
                path="planner_request",
            )
        current_foundation_authority = (
            foundation_schema.authority_sha256 if foundation_schema else None
        )
        foundation_materialization = (
            restore_progressive_resume_foundation(
                checkpoint=resume_checkpoint,
                prompt_metrics=self.last_prompt_metrics,
                request_payload_bytes=foundation_total_bytes,
                schema_bytes=foundation_schema_bytes,
                schema_authority_sha256=current_foundation_authority,
            )
            if resume_checkpoint is not None
            else None
        )
        resume_foundation_reused = foundation_materialization is not None
        if foundation_materialization is None:
            self.last_prompt_metrics["foundation_request_payload_bytes"] = (
                foundation_total_bytes
            )
            self.last_prompt_metrics["foundation_schema_bytes"] = (
                foundation_schema_bytes
            )
            self.last_prompt_metrics[
                "foundation_structured_output_authority_sha256"
            ] = current_foundation_authority

            def parse_foundation(
                raw: str,
            ) -> ProgressiveFoundationMaterialization:
                parsed = _parse_foundation_materialization(
                    raw,
                    host_cohort=host_cohort,
                    allowed_know_how_decisions=allowed_know_how_decisions,
                )
                validate_progressive_foundation(
                    parsed.foundation,
                    context=context,
                    analysis_type=outline.analysis_type,
                )
                return parsed

            foundation_materialization = call_llm_with_structured_retry(
                self.llm,
                foundation_messages,
                parser=parse_foundation,
                role="progressive_planner_foundation",
                max_retries=1,
                max_tokens=_MAX_FOUNDATION_OUTPUT_TOKENS,
                temperature=0.2,
                include_failed_response_on_retry=False,
                progress_callback=progress_callback,
                structured_output=foundation_schema,
                format_reminder=_foundation_shape_contract(
                    outline_sha256=outline_sha256,
                    host_cohort=host_cohort,
                    required_cohort_selection_mode=(
                        required_primary_cohort_selection_mode
                    ),
                    required_cohort_name=(
                        context.cohort.cohort_name
                        if required_primary_cohort_selection_mode is not None
                        else None
                    ),
                )
                + "\nReturn exactly one ProgressiveFoundationMaterialization; "
                "never flatten foundation fields into the response root.",
            )
            self.capture_efficiency_metrics()
        assert foundation_materialization is not None
        if foundation_materialization.outline_sha256 != outline_sha256:
            raise ProgressivePlanCompileError(
                "progressive_foundation_outline_digest_mismatch",
                "plan foundation did not bind the host-validated outline digest",
                path="outline_sha256",
            )
        foundation = foundation_materialization.foundation
        if (
            required_primary_cohort_selection_mode is not None
            and foundation.cohort.selection_mode
            != required_primary_cohort_selection_mode
        ):
            raise ProgressivePlanCompileError(
                "progressive_foundation_cohort_mode_mismatch",
                "plan foundation did not preserve the caller-bound primary "
                "cohort selection mode",
                path="cohort.selection_mode",
            )
        validate_progressive_foundation(
            foundation,
            context=context,
            analysis_type=outline.analysis_type,
        )
        self.last_foundation = foundation_materialization
        if not resume_foundation_reused:
            checkpoint_emitter.emit(
                stage="foundation",
                outline=outline,
                foundation=foundation_materialization,
                materializations=self.last_materializations,
                prompt_metrics=self.last_prompt_metrics,
            )

        selected_action_ids, selected_action_rows = _action_catalog(
            (outline.analysis_type,)
        )
        reporting_source_keys = _article_reporting_source_keys(
            article_context=article_context,
            analysis_type=outline.analysis_type,
            enforce_article_contract=enforce_article_contract,
        )

        def current_step_schema_authority(
            outline_step: ProgressiveOutlineStep,
            outline_step_sha256: str,
            available_product_refs: Sequence[tuple[str, str]],
        ) -> str | None:
            if not llm_supports_strict_json_schema(self.llm):
                return None
            request = progressive_step_materialization_request(
                outline_step=outline_step,
                outline_step_sha256=outline_step_sha256,
                variable_names=tuple(outline_step.variable_names),
                executable_variable_names=(
                    _executable_analysis_variable_roster(
                        context,
                        tuple(outline_step.variable_names),
                    )
                ),
                scientific_action_ids=selected_action_ids,
                allowed_literature_citation_keys=tuple(
                    outline_step.literature_citation_keys
                ),
                available_product_refs=available_product_refs,
            )
            return request.authority_sha256

        prefix_state = (
            restore_progressive_resume_prefix(
                checkpoint=resume_checkpoint,
                outline=outline,
                foundation=foundation,
                context=context,
                step_schema_authority=current_step_schema_authority,
                allowed_literature_citation_keys=allowed_citations,
                allowed_know_how_decisions=allowed_know_how_decisions,
                reporting_method_source_keys=reporting_source_keys,
            )
            if resume_checkpoint is not None
            else ProgressivePrefixState()
        )
        if resume_checkpoint is not None:
            migrated_step_ids = [
                current.step.step_id
                for stored, current in zip(
                    resume_checkpoint.materializations,
                    prefix_state.materializations,
                    strict=True,
                )
                if stored != current
            ]
            if migrated_step_ids:
                self.last_prompt_metrics["runtime_contract_migrated_step_ids"] = (
                    migrated_step_ids
                )
        self.last_materializations = list(prefix_state.materializations)
        self.last_resume_validated = resume_checkpoint is not None

        prefix_state = self._materialize_remaining_steps(
            prefix_state,
            context=context,
            outline=outline,
            foundation_materialization=foundation_materialization,
            scientific_action_ids=selected_action_ids,
            action_rows=selected_action_rows,
            allowed_literature_citation_keys=allowed_citations,
            allowed_know_how_decisions=allowed_know_how_decisions,
            reporting_method_source_keys=reporting_source_keys,
            planning_contract_context=resolved_planning_contract_context,
            progress_callback=progress_callback,
            checkpoint_emitter=checkpoint_emitter,
            resumed=resume_checkpoint is not None,
        )

        if prefix_state.plan is None or prefix_state.receipt is None:
            raise RuntimeError("progressive outline produced no materialized steps")
        skeleton = assemble_progressive_skeleton(
            outline=outline,
            foundation=foundation,
            steps=prefix_state.steps,
        )
        plan, receipt = self._compile_and_accept(
            skeleton,
            agent_context=context,
            article_context=article_context,
            allowed_literature_citation_keys=allowed_citations,
            direct_comparator_literature_keys=direct_keys,
            allowed_know_how_decisions=allowed_know_how_decisions,
            enforce_article_contract=enforce_article_contract,
        )
        self.last_skeleton = skeleton
        self.last_compile_receipt = receipt
        self.last_prompt_metrics["final_skeleton_sha256"] = receipt.skeleton_sha256
        self.last_prompt_metrics["compiled_plan_sha256"] = receipt.analysis_plan_sha256
        return plan


__all__ = [
    "ProgressivePlannerAgent",
    "candidate_analysis_types",
    "progressive_cohort_concept_ids",
    "select_progressive_variables",
]
