"""Provider-facing side of family-spec planning: prompt, strict schema, parser.

The Planner sees one compact request (the question, the sealed candidate
roster with the host's timing verdicts, the labels it must write, the screened
comparators it must apply) and returns one :class:`FamilyPlanSpec`.  The
strict JSON schema is run-bound: covariate names, label keys, and citation keys
are closed enums taken from the sealed request, so a provider that honors
strict schemas cannot spell a coordinate the host did not offer.  A route
without strict-schema support receives the same contract in words with its
first request; on every route the parser re-validates each coordinate.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional, Sequence

from ..canonical_json import canonical_sha256
from ..planning.family_spec import (
    FamilySpecError,
    build_family_spec_request,
    DESCRIPTIVE_FAMILY_ID,
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    LANDMARK_SURVIVAL_FAMILY_ID,
    PHENOTYPING_FAMILY_ID,
    PREDICTION_FAMILY_ID,
    SOURCE_FEASIBILITY_FAMILY_ID,
    build_descriptive_skeleton,
    build_landmark_association_skeleton,
    build_phenotyping_skeleton,
    build_fixed_window_trajectory_skeleton,
    build_landmark_survival_skeleton,
    build_prediction_skeleton,
    build_source_feasibility_skeleton,
)
from ..planning.family_spec.contract import (
    FAMILY_SPEC_SCHEMA_VERSION,
    LANDMARK_FAMILY_IDS,
    FamilyPlanSpec,
    FamilySpecRequest,
    validate_family_plan_spec,
)
from ..planning.literature_bindings import missing_required_method_layers
from ..planning.progressive_artifacts import ProgressivePlannerCheckpointEmitter
from ..planning.progressive_compiler import (
    required_binary_display_label_scopes,
    required_reader_display_label_keys,
    validate_progressive_foundation,
)
from ..planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanCompileReceipt,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
)
from ..planning.progressive_resume import (
    ProgressivePrefixState,
    assemble_progressive_skeleton,
    compile_progressive_prefix,
)
from ..providers.capabilities import llm_supports_strict_json_schema
from ..providers.protocol import LLMMessage, StructuredOutputRequest
from ..providers.strict_json_schema import (
    StrictJsonSchemaError,
    assert_closed_json_schema,
    strictify_json_schema,
)
from ..providers.structured_retry import call_llm_with_structured_retry
from ..reporting.article_contract import build_article_analysis_contract
from ..schema import AnalysisPlan, ResearchContext
from .progressive_attempt import ProgressivePlannerAttemptState

FAMILY_SPEC_STRATEGY = "family_spec_v1"

FAMILY_SPEC_ROLE = "family_spec_planner"
FAMILY_SPEC_MAX_OUTPUT_TOKENS = 6000

FAMILY_SPEC_GUIDE = """You are the EasyICU study statistician completing one typed planning spec.

The host has already fixed the study family, exposure, outcome, time zero, cohort eligibility, dependence handling, sensitivity axes, and every executable step. You decide only what a statistician decides at this point:

1. For a landmark association family: the adjustment set. Choose covariates ONLY from the candidates marked selectable. For each, give one concise clinical confounding rationale (why it can cause both the exposure and the outcome, and that it is fixed before time zero). Do not adjust for a consequence of the exposure or for the outcome. A candidate marked not selectable cannot be used, whatever the rationale; explain any omission in roster_decision_note. If the roster is an exact user-reviewed roster, return it unchanged.
   Code each covariate with one of that candidate's allowed_codings. A binary or categorical coding needs reference_level_index, the 0-based index of the reference level (below the candidate's closed_domain_size); a continuous coding takes reference_level_index null.
   For the descriptive family: no adjustment set (leave it empty). Instead choose baseline_variables for Table 1 ONLY from the candidates marked selectable; they describe the groups and imply no model.
   For the phenotyping family: no adjustment set. Choose feature_variables (at least two) ONLY from the feature candidates marked selectable — the numeric window-bound measurements that should define candidate phenotypes; choose baseline_variables for the cluster characterization from the baseline candidates; optionally choose one cohort_membership_column from the membership flags when the question restricts the population to rows with that flag. Label every selected feature and baseline variable.
   For the prediction family: no adjustment set. Choose feature_variables (at least two predictors) ONLY from the feature candidates marked selectable; do not include the outcome or anything measured after the observation window. Label every selected predictor.
   For the sealed landmark survival suite family: no adjustment set and no other roster. The host has sealed the exposure status and onset columns, the event and follow-up columns, the landmark, the horizon, the adjustment set and the PH policy; you only label the sealed columns and write the comparator applications.
   For the sealed fixed-window trajectory suite family: no adjustment set and no other roster. The host has sealed the coordinate concepts, the fixed grid, the candidate cluster grid, the selection rule and the stability design; you only label the sealed concepts and the outcome and write the comparator applications.
   For the sealed source-feasibility family: no adjustment set, no roster and no labels. The reviewed protocol found the requested treatment contrast not identifiable from the current source, so the host executes only the sealed fail-closed decision; you write the comparator applications (how each screened study's design differs from what this source can support) and nothing else.
2. Reader labels: a concise clinical label for every required variable key, derived from the sealed variable descriptions (never a restatement of the identifier). When level label keys such as `<exposure>=0` and `<exposure>=1` are required, give the two groups distinct clinical names.
3. Comparator applications: for each screened direct comparator, one sentence on how this study is compared with it (population, exposure, time zero, estimand) without copying its design and without claiming novelty.

Return exactly one JSON object and nothing else, in the response contract attached to the request: the provided schema, or the written response shape when no schema is attached. Copy request_sha256 exactly. Never invent variables, citations, results, or significance.
"""


def _candidate_rows(request: FamilySpecRequest) -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "semantic_role": item.semantic_role,
            "host_temporal_role": item.host_temporal_role,
            "allowed_codings": list(item.allowed_codings),
            "closed_domain_size": item.closed_domain_size,
            "selectable": item.selectable,
            "boundary": item.boundary,
        }
        for item in request.adjustment_candidates
    ]


def family_spec_user_prompt(
    request: FamilySpecRequest,
    *,
    variable_descriptions: dict[str, str],
    know_how_context: str = "",
    planning_contract_context: str = "",
    response_shape: str = "",
) -> str:
    """Build the single user prompt for one family-spec attempt.

    ``response_shape`` is the written response contract for a route that
    cannot enforce the strict schema; a strict route leaves it empty.
    """

    sections: list[str] = []
    sections.append("Research question:\n" + request.research_question)
    sections.append(
        "Host-fixed design (binding, not editable):\n"
        + json.dumps(
            {
                "family_id": request.family_id,
                "primary_exposure": request.primary_exposure,
                "exposure_kind": request.exposure_kind,
                "exposure_levels": request.exposure_levels,
                "reference_level": (
                    request.exposure_levels[request.reference_level_index]
                    if request.exposure_kind == "categorical"
                    else None
                ),
                "primary_contrast_level": (
                    request.exposure_levels[request.primary_contrast_level_index]
                    if request.exposure_kind == "categorical"
                    else None
                ),
                "exposure_companion_columns": request.exposure_companion_columns,
                "outcome": request.outcome,
                "landmark_hours": request.landmark_hours,
                "event_time_column": request.event_time_column,
                "observation_duration_column": request.observation_duration_column,
                "cluster_unit": request.cluster_unit,
                "secondary_continuous_outcome": request.secondary_continuous_outcome,
                "alternate_exposures": [item.execution_variables[0] for item in request.alternate_exposures],
                "first_stay_column": (
                    request.first_stay.execution_variables[0] if request.first_stay else None
                ),
                "adjustment_selection": request.adjustment_selection,
                "exact_roster": request.exact_roster,
                "sealed_suite": (
                    request.sealed_suite.model_dump(mode="json")
                    if request.sealed_suite is not None
                    else None
                ),
                "sealed_trajectory": (
                    request.sealed_trajectory.model_dump(mode="json")
                    if request.sealed_trajectory is not None
                    else None
                ),
                "sealed_feasibility": (
                    request.sealed_feasibility.model_dump(mode="json")
                    if request.sealed_feasibility is not None
                    else None
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    sections.append(
        (
            "Baseline-table variable candidates (host timing authority is binding; select only selectable=true):\n"
            if request.family_id in {DESCRIPTIVE_FAMILY_ID, PHENOTYPING_FAMILY_ID}
            else "Adjustment candidates (host timing authority is binding; select only selectable=true):\n"
        )
        + json.dumps(_candidate_rows(request), ensure_ascii=False)
    )
    if request.family_id in {PHENOTYPING_FAMILY_ID, PREDICTION_FAMILY_ID}:
        sections.append(
            (
                "Predictor candidates (select only selectable=true; at least two):\n"
                if request.family_id == PREDICTION_FAMILY_ID
                else "Fit-feature candidates (select only selectable=true; at least two):\n"
            )
            + json.dumps(
                [
                    {
                        "name": item.name,
                        "semantic_role": item.semantic_role,
                        "selectable": item.selectable,
                        "boundary": item.boundary,
                    }
                    for item in request.feature_candidates
                ],
                ensure_ascii=False,
            )
        )
        if request.accepted_feature_groups:
            sections.append(
                "Accepted primary inputs (the reviewed design keeps every one; choose at least "
                "one of each input's columns as a feature):\n"
                + json.dumps(
                    {group.concept: group.columns for group in request.accepted_feature_groups},
                    ensure_ascii=False,
                )
            )
        if request.family_id == PHENOTYPING_FAMILY_ID:
            sections.append(
                "Cohort membership flags (optional; choose at most one):\n"
                + json.dumps(request.membership_candidates)
            )
    described_keys = [
        *request.required_reader_label_keys,
        *(item.name for item in request.feature_candidates if item.selectable),
        *(item.name for item in request.adjustment_candidates if item.selectable),
    ]
    sections.append(
        "Sealed variable descriptions:\n"
        + json.dumps(
            {key: variable_descriptions.get(key, "") for key in dict.fromkeys(described_keys)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    sections.append(
        "Required reader label keys:\n"
        + json.dumps([*request.required_reader_label_keys, *request.level_label_keys])
    )
    sections.append(
        "Screened direct comparators (one application each):\n"
        + json.dumps(
            {
                key: request.comparator_titles.get(key, "")
                for key in request.direct_comparator_literature_keys
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if know_how_context:
        sections.append("Know-how context:\n" + know_how_context)
    if planning_contract_context:
        sections.append("Planning contract context:\n" + planning_contract_context)
    if response_shape:
        sections.append("Response shape (no schema is attached on this route):\n" + response_shape)
    sections.append("request_sha256: " + request.request_sha256)
    return "\n\n".join(sections)


def family_spec_messages(
    request: FamilySpecRequest,
    *,
    variable_descriptions: dict[str, str],
    know_how_context: str = "",
    planning_contract_context: str = "",
    response_shape: str = "",
) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=FAMILY_SPEC_GUIDE),
        LLMMessage(
            role="user",
            content=family_spec_user_prompt(
                request,
                variable_descriptions=variable_descriptions,
                know_how_context=know_how_context,
                planning_contract_context=planning_contract_context,
                response_shape=response_shape,
            ),
        ),
    ]


def _enum(values: Sequence[str]) -> dict[str, Any]:
    normalized = list(dict.fromkeys(str(value) for value in values if str(value)))
    return {"type": "string", "enum": normalized}


def family_spec_structured_output_request(
    request: FamilySpecRequest,
) -> StructuredOutputRequest:
    """Return the run-bound strict schema for one family-spec response."""

    selectable = [item.name for item in request.selectable_candidates]
    covariate_names = selectable or ["__no_selectable_covariate__"]
    feature_names = [item.name for item in request.feature_candidates if item.selectable]
    label_keys = list(
        dict.fromkeys(
            [
                *request.required_reader_label_keys,
                *request.level_label_keys,
                *feature_names,
                *selectable,
            ]
        )
    ) or ["__no_label_key__"]
    descriptive = request.family_id == DESCRIPTIVE_FAMILY_ID
    phenotyping = request.family_id == PHENOTYPING_FAMILY_ID
    prediction = request.family_id == PREDICTION_FAMILY_ID
    citation_keys = list(request.direct_comparator_literature_keys) or [
        "__no_direct_comparator__"
    ]
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "family_id",
            "request_sha256",
            "adjustment_set",
            *(["baseline_variables"] if descriptive or phenotyping else []),
            *(["feature_variables", "cohort_membership_column"] if phenotyping else []),
            *(["feature_variables"] if prediction else []),
            "reader_display_labels",
            "comparator_applications",
            "roster_decision_note",
        ],
        "properties": {
            "schema_version": {"type": "string", "enum": [FAMILY_SPEC_SCHEMA_VERSION]},
            "family_id": {"type": "string", "enum": [request.family_id]},
            "request_sha256": {"type": "string", "enum": [request.request_sha256]},
            "adjustment_set": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "coding", "reference_level_index", "clinical_rationale"],
                    "properties": {
                        "name": _enum(covariate_names),
                        "coding": _enum(["continuous", "binary", "categorical"]),
                        "reference_level_index": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "clinical_rationale": {"type": "string"},
                    },
                },
            },
            **(
                {"baseline_variables": {"type": "array", "items": _enum(covariate_names)}}
                if descriptive or phenotyping
                else {}
            ),
            **(
                {
                    "feature_variables": {
                        "type": "array",
                        "items": _enum(feature_names or ["__no_selectable_feature__"]),
                    },
                }
                if phenotyping or prediction
                else {}
            ),
            **(
                {
                    "cohort_membership_column": {
                        "anyOf": [
                            _enum(list(request.membership_candidates) or ["__no_membership_flag__"]),
                            {"type": "null"},
                        ]
                    },
                }
                if phenotyping
                else {}
            ),
            "reader_display_labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["key", "value"],
                    "properties": {"key": _enum(label_keys), "value": {"type": "string"}},
                },
            },
            "comparator_applications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["citation_key", "application"],
                    "properties": {
                        "citation_key": _enum(citation_keys),
                        "application": {"type": "string"},
                    },
                },
            },
            "roster_decision_note": {"type": "string"},
        },
    }
    strictify_json_schema(schema)
    try:
        assert_closed_json_schema(schema)
    except StrictJsonSchemaError as exc:
        raise ValueError(f"family spec schema is not closed: {exc}") from exc
    return StructuredOutputRequest.from_schema(name="family_plan_spec", schema=schema, strict=True)


def parse_family_plan_spec(raw: str, request: FamilySpecRequest) -> FamilyPlanSpec:
    """Parse one provider response and validate it against the sealed request.

    ``schema_version`` and ``family_id`` are fixed by the host and the sealed
    request, so the host writes them instead of rejecting an otherwise valid
    answer over an echo. ``request_sha256`` remains the response's binding to
    that request and must be copied exactly.
    """

    payload = json.loads(str(raw or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("family spec response root must be an object")
    spec = FamilyPlanSpec.model_validate(
        {
            **payload,
            "schema_version": FAMILY_SPEC_SCHEMA_VERSION,
            "family_id": request.family_id,
        }
    )
    validate_family_plan_spec(spec, request)
    return spec


def family_spec_response_shape(request: FamilySpecRequest) -> str:
    """The response contract in words, derived from the same sealed request.

    A route without strict-schema support receives it with the first request,
    so field names, closed values, and the coding rule are known before the
    first answer; every route receives it again as the retry reminder.
    """

    selectable = [item.name for item in request.selectable_candidates]
    features = [item.name for item in request.feature_candidates if item.selectable]
    lines = [
        "Exactly one JSON object with only these keys:",
        f'- "schema_version": "{FAMILY_SPEC_SCHEMA_VERSION}"',
        f'- "family_id": "{request.family_id}"',
        f'- "request_sha256": "{request.request_sha256}" (copy exactly)',
    ]
    if request.family_id not in LANDMARK_FAMILY_IDS:
        lines.append('- "adjustment_set": [] (this family fits no adjusted model)')
    else:
        if request.adjustment_selection == "exact":
            names = (
                f"exactly {json.dumps(list(request.exact_roster))} in this order, "
                "or [] to keep that sealed user-reviewed roster"
            )
        else:
            names = "one of " + json.dumps(selectable)
        lines.append(
            '- "adjustment_set": array of objects, each with exactly the keys '
            f'"name" ({names}), "coding" (one of that candidate\'s allowed_codings), '
            '"reference_level_index" (0-based index of the reference level, below the '
            "candidate's closed_domain_size, for binary or categorical coding; null for "
            'continuous coding), and "clinical_rationale" (one sentence, 16-500 characters)'
        )
    if request.family_id in {DESCRIPTIVE_FAMILY_ID, PHENOTYPING_FAMILY_ID}:
        lines.append('- "baseline_variables": array of distinct names from ' + json.dumps(selectable))
    if request.family_id in {PHENOTYPING_FAMILY_ID, PREDICTION_FAMILY_ID}:
        lines.append(
            '- "feature_variables": array of at least two distinct names from '
            + json.dumps(features)
            + (
                ", including at least one column of every accepted primary input"
                if request.accepted_feature_groups
                else ""
            )
        )
    if request.family_id == PHENOTYPING_FAMILY_ID:
        lines.append(
            '- "cohort_membership_column": one of '
            + json.dumps(list(request.membership_candidates))
            + " or null"
        )
    selects_variables = request.family_id in {
        DESCRIPTIVE_FAMILY_ID,
        PHENOTYPING_FAMILY_ID,
        PREDICTION_FAMILY_ID,
    } or (
        request.family_id in LANDMARK_FAMILY_IDS
        and request.adjustment_selection != "exact"
    )
    label_keys = [*request.required_reader_label_keys, *request.level_label_keys]
    example_key = next((key for key in label_keys if "_" in key), label_keys[0] if label_keys else "")
    lines.append(
        '- "reader_display_labels": array of {"key": ..., "value": ...} objects giving a '
        "concise clinical label for each of "
        + json.dumps(label_keys)
        + (" and for every variable you select" if selects_variables else "")
        + (
            "; write it from the sealed variable description, because a value that only "
            f'restates its key (such as "{example_key.replace("_", " ")}" for {example_key}) '
            "is rejected"
            if example_key
            else ""
        )
    )
    lines.append(
        '- "comparator_applications": array of {"citation_key": ..., "application": ...} '
        "objects, exactly one for each of "
        + json.dumps(list(request.direct_comparator_literature_keys))
    )
    lines.append('- "roster_decision_note": one or more sentences (8-1200 characters)')
    return "\n".join(lines)




def run_family_spec_attempt(
    *,
    llm: Any,
    attempt: ProgressivePlannerAttemptState,
    context: ResearchContext,
    article_context: ResearchContext,
    analysis_types: Sequence[str],
    variables: Sequence[str],
    action_ids: Sequence[str],
    allowed_citations: Sequence[str],
    direct_keys: Sequence[str],
    comparison_keys: Sequence[str],
    allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None,
    know_how_context: str,
    planning_contract_context: str,
    enforce_article_contract: bool,
    required_primary_cohort_selection_mode: str | None,
    required_visualization_step: bool,
    reporting_source_keys: Sequence[str],
    resume_dependency_authority_sha256: str | None,
    checkpoint_emitter: ProgressivePlannerCheckpointEmitter,
    progress_callback: Optional[Callable[[Any], None]],
    max_parse_retries: int,
    prompt_byte_limit: int,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline],
    validate_outline: Callable[[ProgressivePlanOutline], None],
    compile_and_accept: Callable[
        [ProgressivePlanSkeleton], tuple[AnalysisPlan, ProgressivePlanCompileReceipt]
    ],
    capture_efficiency_metrics: Callable[[], None],
) -> AnalysisPlan:
    """One spec call, then a host-projected skeleton through the same gates.

    The Planner fills the family spec; the host template projects outline,
    foundation, and every step; the fresh-plan outline authority, foundation
    validator, prefix compiler, and final acceptance run unchanged (they are
    injected by the Progressive agent).  A projection that any of them rejects
    raises the validator's own reason code: there is no Provider suffix repair
    on this path.
    """

    request = build_family_spec_request(
        context,
        analysis_types=analysis_types,
        variable_roster=variables,
        allowed_literature_citation_keys=allowed_citations,
        direct_comparator_literature_keys=direct_keys,
        comparison_literature_keys=comparison_keys,
        required_primary_cohort_selection_mode=required_primary_cohort_selection_mode,
        planning_contract_context=planning_contract_context,
    )
    descriptions = {
        name: " ".join(
            part
            for part in (
                str(getattr(descriptor, "description", "") or ""),
                *(
                    str(item)
                    for item in (getattr(descriptor, "clinical_caveats", None) or ())[:2]
                ),
            )
            if part
        )
        for name in variables
        if (descriptor := context.variable(name)) is not None
    }
    response_shape = family_spec_response_shape(request)
    schema = (
        family_spec_structured_output_request(request)
        if llm_supports_strict_json_schema(llm)
        else None
    )
    messages = family_spec_messages(
        request,
        variable_descriptions=descriptions,
        know_how_context=know_how_context,
        planning_contract_context=planning_contract_context,
        # An enforced schema already carries the shape; a route without one
        # would otherwise answer the first request blind.
        response_shape="" if schema is not None else response_shape,
    )
    message_bytes = sum(len(item.content.encode("utf-8")) for item in messages)
    schema_bytes = schema.payload_bytes if schema is not None else 0
    total_bytes = message_bytes + schema_bytes
    if total_bytes > prompt_byte_limit:
        raise ProgressivePlanCompileError(
            "progressive_prompt_budget_exceeded",
            f"family spec request uses {total_bytes} bytes; limit={prompt_byte_limit}",
            path="planner_request",
        )
    attempt.prompt_metrics = {
        "planner_strategy": FAMILY_SPEC_STRATEGY,
        "requested_planner_strategy": FAMILY_SPEC_STRATEGY,
        "family_spec_fallback_reason": None,
        "family_id": request.family_id,
        "family_spec_request_sha256": request.request_sha256,
        "family_spec_structured_output_authority_sha256": (
            schema.authority_sha256 if schema is not None else None
        ),
        "message_payload_bytes": message_bytes,
        "structured_output_payload_bytes": schema_bytes,
        # No outline or foundation request is made on this path; the artifact
        # chain records those transport authorities as absent.
        "structured_output_authority_sha256": None,
        "total_bytes": total_bytes,
        "outline_request_payload_bytes": 0,
        "outline_schema_bytes": 0,
        "foundation_request_payload_bytes": 0,
        "foundation_schema_bytes": 0,
        "foundation_structured_output_authority_sha256": None,
        "without_know_how_total_bytes": total_bytes,
        "selected_variable_count": len(variables),
        "selected_variable_roster": list(variables),
        "selected_variable_roster_sha256": canonical_sha256(list(variables)),
        "candidate_analysis_types": list(analysis_types),
        "selected_scientific_action_ids": list(action_ids),
        "foundation_cohort_owner": "family_template",
        "required_primary_cohort_selection_mode": required_primary_cohort_selection_mode,
        "required_visualization_step": required_visualization_step,
        "resume_dependency_authority_sha256": resume_dependency_authority_sha256,
        "compile_revision_count": 0,
        "step_materialization_count": 0,
        "host_step_materialization_count": 0,
        "step_materialization_payload_bytes": [],
        "step_materialization_schema_sha256": [],
        "step_materialization_attempt_payload_bytes": [],
        "step_materialization_attempt_schema_sha256": [],
        "suffix_revision_count": 0,
        "full_revision_count": 0,
        "suffix_request_payload_bytes": [],
        "plan_revision_projection": [],
        "foundation_plan_revision_projection": [],
    }

    def parse_spec(raw: str) -> FamilyPlanSpec:
        return parse_family_plan_spec(raw, request)

    spec = call_llm_with_structured_retry(
        llm,
        messages,
        parser=parse_spec,
        role=FAMILY_SPEC_ROLE,
        max_retries=max_parse_retries,
        max_tokens=FAMILY_SPEC_MAX_OUTPUT_TOKENS,
        temperature=0.2,
        # The spec is small: showing the rejected answer beside the
        # validator's reason lets the retry correct that answer instead of
        # regenerating it and failing on a different field.
        include_failed_response_on_retry=True,
        progress_callback=progress_callback,
        structured_output=schema,
        format_reminder=response_shape,
    )
    capture_efficiency_metrics()
    attempt.prompt_metrics["family_spec_sha256"] = canonical_sha256(
        spec.model_dump(mode="json")
    )
    attempt.prompt_metrics["family_spec_adjustment_set"] = [
        item.name for item in spec.adjustment_set
    ]

    builder = {
        DESCRIPTIVE_FAMILY_ID: build_descriptive_skeleton,
        PHENOTYPING_FAMILY_ID: build_phenotyping_skeleton,
        PREDICTION_FAMILY_ID: build_prediction_skeleton,
        LANDMARK_SURVIVAL_FAMILY_ID: build_landmark_survival_skeleton,
        FIXED_WINDOW_TRAJECTORY_FAMILY_ID: build_fixed_window_trajectory_skeleton,
        SOURCE_FEASIBILITY_FAMILY_ID: build_source_feasibility_skeleton,
    }.get(request.family_id, build_landmark_association_skeleton)
    try:
        draft = builder(request, spec, bind_outline=bind_outline)
    except FamilySpecError as exc:
        raise ProgressivePlanCompileError(
            f"progressive_{exc.reason_code}", str(exc), path=exc.path or "family_spec",
        ) from exc
    outline = draft.outline
    validate_outline(outline)
    attempt.outline = outline
    attempt.foundation = None
    attempt.materializations = []
    outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))
    attempt.prompt_metrics["outline_sha256"] = outline_sha256
    checkpoint_emitter.emit(
        stage="outline",
        outline=outline,
        foundation=None,
        materializations=[],
        prompt_metrics=attempt.prompt_metrics,
    )
    foundation_materialization = draft.foundation
    if foundation_materialization.outline_sha256 != outline_sha256:
        raise ProgressivePlanCompileError(
            "progressive_foundation_outline_digest_mismatch",
            "family template foundation did not bind the validated outline digest",
            path="outline_sha256",
        )
    foundation = foundation_materialization.foundation
    require_robustness_intent = bool(
        enforce_article_contract
        and "robustness"
        in build_article_analysis_contract(
            article_context, analysis_type=outline.analysis_type,
        ).required_roles
    )
    validate_progressive_foundation(
        foundation,
        context=context,
        analysis_type=outline.analysis_type,
        require_robustness_intent=require_robustness_intent,
        robustness_replay_required=any(
            step.module_id == "robustness_replay" for step in outline.steps
        ),
        required_binary_display_label_scopes=required_binary_display_label_scopes(
            context, outline.steps,
        ),
        required_reader_display_label_keys=required_reader_display_label_keys(
            context, outline.design_selection,
        ),
    )
    attempt.foundation = foundation_materialization
    checkpoint_emitter.emit(
        stage="foundation",
        outline=outline,
        foundation=foundation_materialization,
        materializations=[],
        prompt_metrics=attempt.prompt_metrics,
    )
    prefix_state = ProgressivePrefixState()
    for materialization in draft.materializations:
        prefix_state = compile_progressive_prefix(
            prefix_state,
            materialization,
            outline=outline,
            foundation=foundation,
            context=context,
            allowed_literature_citation_keys=allowed_citations,
            allowed_know_how_decisions=allowed_know_how_decisions,
            reporting_method_source_keys=reporting_source_keys,
        )
        attempt.materializations = list(prefix_state.materializations)
        attempt.prompt_metrics["host_step_materialization_count"] += 1
        attempt.prompt_metrics["step_materialization_payload_bytes"].append(0)
        attempt.prompt_metrics["step_materialization_schema_sha256"].append(None)
        checkpoint_emitter.emit(
            stage="step",
            outline=outline,
            foundation=foundation_materialization,
            materializations=attempt.materializations,
            prompt_metrics=attempt.prompt_metrics,
        )
    assert prefix_state.plan is not None
    missing_method_layers = missing_required_method_layers(
        prefix_state.plan, allowed_citations, context=context,
    )
    if missing_method_layers:
        raise ProgressivePlanCompileError(
            "progressive_final_method_layer_unbound",
            "family template left required method layers unbound",
            path="literature_bindings",
            findings=({"missing_method_layers": list(missing_method_layers)},),
        )
    skeleton = assemble_progressive_skeleton(
        outline=outline, foundation=foundation, steps=prefix_state.steps,
    )
    plan, receipt = compile_and_accept(skeleton)
    attempt.skeleton = skeleton
    attempt.compile_receipt = receipt
    attempt.prompt_metrics["final_skeleton_sha256"] = receipt.skeleton_sha256
    attempt.prompt_metrics["compiled_plan_sha256"] = receipt.analysis_plan_sha256
    capture_efficiency_metrics()
    return plan


__all__ = [
    "FAMILY_SPEC_GUIDE",
    "FAMILY_SPEC_MAX_OUTPUT_TOKENS",
    "FAMILY_SPEC_ROLE",
    "FAMILY_SPEC_STRATEGY",
    "family_spec_messages",
    "family_spec_response_shape",
    "family_spec_structured_output_request",
    "family_spec_user_prompt",
    "parse_family_plan_spec",
    "run_family_spec_attempt",
]
