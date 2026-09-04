"""Text projections of progressive-planning structured-output contracts."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from ..planning.progressive_contract import (
    ProgressiveCohortIntent,
    ProgressiveOutlineStep,
    ProgressivePlanOutline,
)


_COUNTS_ONLY_INFERENCE_TOKENS = (
    "confidence interval",
    "confidence intervals",
    "uncertainty",
    "standard error",
    "p-value",
    "p value",
    "置信区间",
    "不确定性",
    "标准误",
    "p值",
    "p 值",
)


def outline_shape_contract(
    *,
    analysis_types: Sequence[str],
    module_ids_by_analysis_type: Mapping[str, Sequence[str]],
) -> str:
    """Render the exact small outline shape for schema-imperfect transports."""

    template = {
        "schema_version": "easyicu.progressive_plan_outline/1",
        "analysis_type": "<copy one exact candidate analysis family>",
        "cohort_objective": "<8-600 characters>",
        "design_selection": {
            "schema_version": "easyicu.research_design_selection/1",
            "claim_ceiling": "analysis_only",
            "candidates": [
                {
                    "design_id": "<unique lowercase selected design id>",
                    "analysis_type": "<copy one exact candidate analysis family>",
                    "estimand": "<8-600 characters>",
                    "time_zero": "<8-400 characters>",
                    "observation_window": "<8-400 characters>",
                    "primary_method": "<3-300 characters>",
                    "required_variables": ["<copy a sealed variable name>"],
                    "assumptions": ["<one prespecified assumption>"],
                    "literature_citation_keys": ["<copy a sealed citation key>"],
                    "literature_design_decisions": [],
                    "novelty_positioning": "<8-600 characters>",
                    "figure_role": "<8-400 characters>",
                    "supports": "<8-500 characters>",
                    "cannot_prove": "<8-500 characters>",
                    "reviewable_plan": [
                        "<population and analysis unit>",
                        "<exposure definition, timing, and aggregation>",
                        "<outcome definition and follow-up>",
                        "<adjustment set and model>",
                        "<missing-data handling>",
                        "<sensitivity and pre-analysis feasibility checks>",
                    ],
                    "disposition": "selected",
                    "decision_reason": "<12-600 pre-result characters>",
                },
                {
                    "design_id": "<unique lowercase rejected design id>",
                    "analysis_type": "<copy one exact candidate analysis family>",
                    "estimand": "<8-600 characters distinct from selected>",
                    "time_zero": "<8-400 characters>",
                    "observation_window": "<8-400 characters>",
                    "primary_method": "<3-300 characters distinct from selected>",
                    "required_variables": ["<copy a sealed variable name>"],
                    "assumptions": ["<one prespecified assumption>"],
                    "literature_citation_keys": ["<copy a sealed citation key>"],
                    "literature_design_decisions": [],
                    "novelty_positioning": "<8-600 characters>",
                    "figure_role": "<8-400 characters>",
                    "supports": "<8-500 characters>",
                    "cannot_prove": "<8-500 characters>",
                    "reviewable_plan": None,
                    "disposition": "rejected",
                    "decision_reason": "<12-600 pre-result characters>",
                },
            ],
        },
        "steps": [
            {
                "step_id": "<unique lowercase id>",
                "planned_analysis_role": "<primary|secondary|sensitivity|auxiliary>",
                "module_id": "<copy one exact allowed module id>",
                "objective": "<8-600 characters>",
                "depends_on": [],
                "variable_names": ["<copy a sealed variable name>"],
                "literature_citation_keys": [
                    "<sealed method key for every primary/secondary/sensitivity step>"
                ],
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
        "literature_citation_keys must always be JSON arrays. Every primary, "
        "secondary, or sensitivity step must bind at least one sealed method "
        "key; only auxiliary steps may use an empty array. "
        "scientific_action_id must be a retrieved action id or null."
    )


def foundation_shape_contract(
    *,
    outline_sha256: str,
    host_cohort: ProgressiveCohortIntent | None,
    required_cohort_selection_mode: str | None = None,
    required_cohort_name: str | None = None,
    required_binary_display_label_scopes: Sequence[str] = (),
    required_reader_display_label_keys: Sequence[str] = (),
) -> str:
    """Project the exact foundation envelope without adding case science."""

    if host_cohort is not None:
        cohort = host_cohort.model_dump(mode="json")
    elif required_cohort_selection_mode == "predicate_filtered":
        cohort = {
            "name": required_cohort_name or "<1-128 characters>",
            "selection_mode": "predicate_filtered",
            "inclusion": [{
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
            }],
            "exclusion": [],
        }
    else:
        cohort = {
            "name": required_cohort_name or "<1-128 characters>",
            "selection_mode": "<all_input_rows|predicate_filtered>",
            "inclusion": [],
            "exclusion": [],
        }
    required_labels = [
        {"key": key, "value": "<reader-facing clinical variable label>"}
        for key in required_reader_display_label_keys
    ] + [
        {"key": f"{scope}=0", "value": "<reader-facing label for level 0>"}
        for scope in required_binary_display_label_scopes
    ] + [
        {"key": f"{scope}=1", "value": "<reader-facing label for level 1>"}
        for scope in required_binary_display_label_scopes
    ]
    template = {
        "schema_version": "easyicu.progressive_plan_foundation/1",
        "outline_sha256": outline_sha256,
        "foundation": {
            "cohort": cohort,
            "display_labels": required_labels,
            "robustness_intents": [],
            "know_how_decisions": [],
        },
    }
    cohort_instruction = (
        "The cohort object shown above is caller-bound; copy it exactly."
        if host_cohort is not None
        else "Replace the cohort placeholders and use predicate_filtered only with at least one valid inclusion or exclusion predicate."
    )
    return (
        "Exact ProgressiveFoundationMaterialization JSON shape (preserve the "
        "root foundation wrapper and every displayed key; add no other keys):\n"
        + json.dumps(template, ensure_ascii=False, separators=(",", ":"))
        + "\nCopy schema_version and outline_sha256 exactly. "
        + cohort_instruction
        + " display_labels, robustness_intents, and know_how_decisions must always be JSON arrays, including when empty.\n"
        "If display_labels is nonempty, each item has exactly "
        '{"key":"<1-256 characters>","value":"<1-256 characters>"}. '
        "If robustness_intents is nonempty, each item has exactly "
        '{"spec_id":"<lowercase id>","axis":"<cohort|missing|outcome>",'
        '"description":"<8-600 characters>","missing_strategy":"<none|complete_case>",'
        '"complete_case_variables":[]}. '
        "If know_how_decisions is nonempty, each item has exactly "
        '{"card_id":"<authorized id>","card_version":"<authorized version>",'
        '"card_sha256":"<authorized 64-char digest>","claim_id":"<authorized id>",'
        '"disposition":"<adopted|rejected|unresolved|requires_confirmation>",'
        '"reason_code":"<lowercase id>","rationale":"<1-500 characters>",'
        '"citation_ids":["<authorized id>"]}. '
        "For every cohort predicate preserve exactly concept_id, anchor, "
        "start_offset_hours, end_offset_hours, aggregation, op, and value. "
        "Its value object must preserve all six displayed keys; populate only "
        "the field selected by mode and leave the others null or empty."
    )


def selected_counts_only_inference_coordinate(
    outline: ProgressivePlanOutline,
) -> str | None:
    """Return the selected design field that exceeds a counts-only ceiling."""

    if outline.design_selection is None:
        return None
    selected = outline.design_selection.selected
    fields = (
        ("estimand", selected.estimand),
        ("primary_method", selected.primary_method),
        ("figure_role", selected.figure_role),
        ("supports", selected.supports),
        *tuple(
            (f"reviewable_plan[{index}]", value)
            for index, value in enumerate(selected.reviewable_plan or ())
        ),
    )
    for field, value in fields:
        normalized = " ".join(str(value or "").casefold().split())
        if any(token in normalized for token in _COUNTS_ONLY_INFERENCE_TOKENS):
            return field
    return None


def step_materialization_shape_contract(
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
        "raw_inputs": [], "product_inputs": [], "outputs": [],
        "scientific_action_id": outline_step.scientific_action_id,
        "custom_method": None, "table_one_group_by": None, "table_one_mode": None,
        "table_one_variables": [], "primary_exposure": None, "outcome": None,
        "outcome_type": None, "model_terms": [], "event_level_index": None,
        "reference_exposure_level_index": None,
        "comparison_exposure_level_index": None,
        "primary_contrast_level_index": None, "denominator_policy": None,
        "missing_exposure_policy": None, "missing_outcome_policy": None,
        "confidence_level": None, "sensitivity_spec_ids": [],
        "literature_bindings": [],
    }
    template = {
        "schema_version": "easyicu.progressive_step_materialization/1",
        "outline_step_sha256": outline_step_sha256,
        "foundation": None,
        "step": step,
    }
    return (
        "Exact ProgressiveStepMaterialization JSON shape (preserve this root wrapper and every step key; add no other keys):\n"
        + json.dumps(template, ensure_ascii=False, separators=(",", ":"))
        + "\nCopy schema_version, outline_step_sha256, foundation=null, and the six outline-owned step coordinates exactly. Replace only the module-specific executable null/empty defaults required by the current method card. Never return variable_names, literature_citation_keys, literature_design_bindings, cohort, or expected_outputs inside step. raw_inputs may contain only sealed variable names, never kind:product tokens; governed products belong only in product_inputs.\n"
        "Nested item shapes, when used: product_inputs items are exactly "
        '{"producer_step_id":"<preceding step id>","product_id":"<kind:product>"}; outputs items are exactly '
        '{"product_id":"<kind:product>","semantic_role":"<allowed role>"}; table_one_variables items are exactly '
        '{"name":"<sealed variable>","summary":"<mean_sd|median_iqr|both|count_percent>"}; model_terms items are exactly '
        '{"name":"<sealed variable>","role":"<exposure|covariate>","coding":"<continuous|binary|categorical|ordinal_linear>","reference_level_index":null}; literature_bindings items are exactly '
        '{"citation_key":"<sealed key>","design_elements":["<allowed element>"],"application":"<8-1200 characters>","divergence":null}.'
    )


__all__ = [
    "foundation_shape_contract",
    "outline_shape_contract",
    "selected_counts_only_inference_coordinate",
    "step_materialization_shape_contract",
]
