"""Provider-facing side of family-spec planning: prompt, strict schema, parser.

The Planner sees one compact request (the question, the sealed candidate
roster with the host's timing verdicts, the labels it must write, the screened
comparators it must apply, and any reviewed comparator design cards it must
resolve) and returns one :class:`FamilyPlanSpec`.  The
strict JSON schema is run-bound: covariate names, label keys, and citation keys
are closed enums taken from the sealed request, so a provider that honors
strict schemas cannot spell a coordinate the host did not offer.  A route
without strict-schema support receives the same contract in words with its
first request; on every route the parser re-validates each coordinate.

When Progressive v2 stops for a family whose primary result contract only
a host template writes, :func:`family_result_stop` names this strategy.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional, Sequence, get_args

from .population_spec_transport import (
    bind_population_spec_transport, population_spec_contract, population_spec_shape,
)

from ..canonical_json import canonical_sha256
from ..planning.family_spec import (
    BASELINE_ASSOCIATION_FAMILY_ID,
    build_baseline_association_skeleton,
    FamilySpecError,
    FamilySkeletonDraft,
    build_family_spec_request,
    DESCRIPTIVE_FAMILY_ID,
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID,
    LANDMARK_SURVIVAL_FAMILY_ID,
    PHENOTYPING_FAMILY_ID,
    PREDICTION_FAMILY_ID,
    SOURCE_FEASIBILITY_FAMILY_ID,
    build_descriptive_skeleton,
    build_landmark_association_skeleton,
    build_phenotyping_skeleton,
    build_fixed_window_trajectory_skeleton,
    build_landmark_continuous_survival_skeleton,
    build_landmark_survival_skeleton,
    build_prediction_skeleton,
    build_source_feasibility_skeleton,
    family_template_id_for_context,
    keeps_unmeasured_covariate_rows,
    sealed_continuous_survival_suite_coordinates,
    sealed_feasibility_coordinates,
    sealed_survival_suite_coordinates,
    sealed_trajectory_suite_coordinates,
)
from ..contracts.model_retention import MISSING_CATEGORY_SHARE_THRESHOLD
from ..planning.family_spec.contract import (
    FAMILY_SPEC_SCHEMA_VERSION,
    LANDMARK_FAMILY_IDS,
    POPULATION_ANCHOR,
    DistributionMissingPolicies,
    FamilyPlanSpec,
    MAX_FIT_FEATURES,
    FamilySpecRequest,
    design_field_max_length,
    family_effect_measures,
    fit_roster_budget,
    literature_design_card_keys_by_dimension,
    planner_selects_adjustment,
    population_required,
    validate_family_plan_spec,
)
from ..planning.capability_gap import check_capability_gap
from ..planning.family_spec.landmark_categorical_template import reconcile_population_grouping
from ..planning.literature_bindings import missing_required_method_layers
from ..planning.question_requirements import (
    QUESTION_REQUIREMENTS_GUIDE,
    question_effect_measures,
    question_requirements_schema,
    question_requirements_shape,
)
from ..planning.literature_design_authority import (
    LITERATURE_DESIGN_DIMENSIONS,
    LiteratureDesignEvidenceCard,
    render_literature_design_card_facts,
)
from ..planning.primary_result_contract import family_result_stop_reason
from ..planning.progressive_artifacts import ProgressivePlannerCheckpointEmitter
from ..planning.progressive_compiler import (
    progressive_cohort_concept_ids,
    required_binary_display_label_scopes,
    required_reader_display_label_keys,
    validate_progressive_foundation,
)
from ..planning.progressive_contract import (
    ProgressiveCohortPredicate,
    ProgressivePlanCompileError,
    ProgressivePredicateValue,
    ProgressivePlanCompileReceipt,
    ProgressivePlanOutline,
    ProgressivePlannerCheckpoint,
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
from ..planning.publication_readiness import render_required_plan_shape
from ..schema import AnalysisPlan, ResearchContext
from ._support import _dump_raw
from .progressive_attempt import ProgressivePlannerAttemptState

FAMILY_SPEC_STRATEGY = "family_spec_v1"

FAMILY_SPEC_ROLE = "family_spec_planner"
FAMILY_SPEC_MAX_OUTPUT_TOKENS = 6000

FAMILY_SPEC_GUIDE = (
    """You are the EasyICU study statistician completing one typed planning spec.

The host has already fixed the study family, exposure, outcome, time zero, the typed cohort bounds, dependence handling, sensitivity axes, and every executable step. You decide only what a statistician decides at this point:

1. For baseline and landmark association families: the host binds the adjustment set before this call, using the stated/exact study roster or the versioned family minimal rule in exact_rationales. Preserve exact_roster, including an empty roster; you may leave adjustment_set empty to use it. Do not add, remove or reorder covariates. Propose any addition only with a clinical rationale in roster_decision_note for a separately reviewed study revision; the current model keeps the bound roster. For other requests marked planner_selectable, choose only selectable candidates, each with a concise confounding rationale and host-proven baseline timing. A rationale never grants timing authority.
   Code each covariate with one of that candidate's allowed_codings. A binary or categorical coding needs reference_level_index, the 0-based index of the reference level (below the candidate's closed_domain_size); a continuous coding takes reference_level_index null.
   For the descriptive family: no adjustment set (leave it empty). Choose baseline_variables for Table 1 ONLY when baseline summaries are requested; [] means no baseline table. A variable used only for eligibility does not require a baseline summary. Select only candidates marked selectable. Explicitly declare distribution_missing_policies before review: choose whether missing exposure/outcome stops execution (fail_closed) or leaves the descriptive denominator with counts reported (exclude_from_denominator). Missing outcomes stay in cohort/missingness audits, never become negative outcomes. The denominator is observed_outcome_rows exactly when missing outcomes are excluded; otherwise all_declared_rows. Metadata-only missingness is unknown, even when its placeholder counts are zero. Follow the user's declared policies; do not select or change them after reading prepared data.
   For the phenotyping family: no adjustment set. Choose feature_variables (at least two) ONLY from the feature candidates marked selectable — the numeric window-bound measurements that should define candidate phenotypes; choose baseline_variables for the cluster characterization from the baseline candidates; optionally choose one cohort_membership_column from the membership flags when the question restricts the population to rows with that flag. Label every selected feature and baseline variable.
   For the prediction family: no adjustment set. Choose feature_variables (at least two predictors) ONLY from the feature candidates marked selectable; do not include the outcome or anything measured after the observation window. Label every selected predictor.
   For the landmark survival suite family with a sealed_suite: no adjustment set and no other roster. The host has sealed the exposure status and onset columns, the event and follow-up columns, the landmark, the horizon, the adjustment set and the PH policy; you only label the sealed columns and write the comparator applications.
   For the landmark survival suite family with a proposed_suite instead: the host has proposed the exposure status, event and follow-up columns, the landmark and the horizon, but the adjustment set is yours: choose only selectable candidates, at least one whenever any is selectable, under the planner_selectable rules above (time zero is the landmark; the onset column is named but not yet materialized) and label every selected covariate.
   For the continuous landmark survival suite family: the exposure is one continuous value (its window summary from ICU admission to the landmark), modelled per unit. With a sealed_continuous_suite: no adjustment set and no other roster; you only label the sealed columns and write the comparator applications. With a proposed_continuous_suite instead: the host has proposed the exposure, event and follow-up columns, the landmark and the horizon, but the adjustment set is yours: choose it as for a landmark association family (time zero is the landmark) and label every selected covariate.
   For the sealed fixed-window trajectory suite family: no adjustment set and no other roster. The host has sealed the coordinate concepts, the fixed grid, the candidate cluster grid, the selection rule and the stability design; you only label the sealed concepts and the outcome and write the comparator applications.
   For the sealed source-feasibility family: no adjustment set, no roster and no labels. The reviewed protocol found the requested treatment contrast not identifiable from the current source, so the host executes only the sealed fail-closed decision; you write the comparator applications (how each screened study's design differs from what this source can support) and nothing else.
2. Reader labels: a concise clinical label for every required variable key, derived from the sealed variable descriptions (never a restatement of the identifier). When level label keys such as `<exposure>=0` and `<exposure>=1` are required, give the two groups distinct clinical names.
3. Comparator applications: for each screened direct comparator, one sentence on how this study is compared with it (population, exposure, time zero, estimand) without copying its design and without claiming novelty.
For a population restriction expressed by a typed kind, use population.population_spec as the complete population (including every pending_typed_population_bounds entry) and leave the legacy criteria/inclusion/exclusion arrays empty. The old already_applied view includes bounds that the legacy template adds; it does not authorize omitting those bounds from a complete population_spec. The typed owner compiles it and decides whether it is already applied, can be applied by the plan, or requires data extraction and review. A host-owned typed kind does not need a concept column. Do not replace first ICU within each hospitalization with each patient's first ICU or a nonmissing unit-type predicate. Only the legacy predicate representation follows the concept-only rules below.
4. Population, only when the request offers population concepts (otherwise omit it or return null): the host applies only the typed cohort bounds and the source export's own population. A population that the research question or the study's own cohort wording names beyond them (for example an age group, a diagnosis or syndrome, or a treatment received) is applied only by the population you state. List each restriction in population.criteria in the words that state it, with the offered concepts that express it, and apply every criterion that has concepts with at least one inclusion or exclusion predicate over one of them, anchored at icu_admission and decided by time zero (end_offset_hours at most the time zero shown). Every predicate applies a listed criterion. Give a criterion no concepts only when no offered concept expresses it over the window the criterion states. Return population null when the study includes every row the typed bounds keep. When population_required is true, the caller has bound a filtered cohort that no typed bound filters: state the population with at least one inclusion or exclusion predicate (a phenotyping plan may restrict by its membership flag instead). A restriction already_applied states (a typed bound, the source concept population, or a source inclusion or exclusion contract) is applied already: do not state it again. When source_selection_recorded is true, already_applied is everything that selected the input rows, and no other condition, treatment or age restriction has.
5. """
    + QUESTION_REQUIREMENTS_GUIDE
    + """

Return exactly one JSON object and nothing else, in the response contract attached to the request: the provided schema, or the written response shape when no schema is attached. Never invent variables, citations, results, or significance.
"""
)


def _requirement_effect_measures(request: FamilySpecRequest) -> tuple[str, ...]:
    """Measures an estimand requirement may name: both, wherever a binary primary exists.

    A family that reports only odds ratios still lets a requirement name the
    risk ratio it cannot report, so that gap is typed rather than prose.
    """

    return ("odds_ratio", "risk_ratio") if family_effect_measures(request) else ()


def _effect_measure_section(request: FamilySpecRequest) -> str:
    measures = family_effect_measures(request)
    if not measures:
        return ""
    choice = (
        "set effect_measure to risk_ratio (modified Poisson, robust SE) only when the "
        "question asks for a risk ratio or relative risk; otherwise null."
        if "risk_ratio" in measures
        else "its signed logistic primary reports odds ratios only, so a requested risk "
        "ratio is a capability gap."
    )
    # What the host read in the question, which the spec contract holds the
    # spec to (``question_requirements.question_effect_measures``).
    named = question_effect_measures(request.research_question)
    return (
        "Effect measure (binary primary; null means the odds ratio): " + choice
        + " An estimand requirement asking for a measure names it in effect_measure."
        + (
            " The question names: "
            + "; ".join(f"{item.measure} ({item.quote!r})" for item in named)
            + "."
            if named
            else ""
        )
    )


def _candidate_rows(request: FamilySpecRequest) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in request.adjustment_candidates:
        row: dict[str, Any] = {
            "name": item.name,
            "semantic_role": item.semantic_role,
            "host_temporal_role": item.host_temporal_role,
            "allowed_codings": list(item.allowed_codings),
            "closed_domain_size": item.closed_domain_size,
            "selectable": item.selectable,
            "boundary": item.boundary,
        }
        if item.missing_share is not None:
            row["missing_share"] = item.missing_share
        rows.append(row)
    return rows


def family_spec_user_prompt(
    request: FamilySpecRequest,
    *,
    context: ResearchContext | None = None,
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
    if context is not None:
        sections.append(render_required_plan_shape(context))
    host_design: dict[str, Any] = {
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
    }
    if request.proposed_suite is not None:
        # The proposal's roster is the Planner's to select, so it is not shown
        # as a binding empty roster beside the host-fixed coordinates.
        host_design["proposed_suite"] = request.proposed_suite.model_dump(
            mode="json", exclude={"adjustment_columns"}
        )
    if request.sealed_continuous_suite is not None:
        host_design["sealed_continuous_suite"] = request.sealed_continuous_suite.model_dump(
            mode="json"
        )
    if request.proposed_continuous_suite is not None:
        host_design["proposed_continuous_suite"] = (
            request.proposed_continuous_suite.model_dump(
                mode="json", exclude={"adjustment_columns"}
            )
        )
    sections.append(
        "Host-fixed design (binding, not editable):\n"
        + json.dumps(host_design, ensure_ascii=False, sort_keys=True)
    )
    if request.family_id == DESCRIPTIVE_FAMILY_ID:
        sections.append(
            "Descriptive missingness evidence (not a policy choice):\n"
            + json.dumps({
                "missingness_unobserved": request.distribution_missingness_unobserved,
                "legacy_host_default": (
                    request.distribution_missing_policies.model_dump(mode="json")
                    if request.distribution_missing_policies is not None else None
                ),
            }, sort_keys=True)
            + "\nDeclare distribution_missing_policies in the candidate for review. "
            "An unobserved missingness profile is not evidence that all values are present."
        )
    sections.append(
        (
            "Baseline-table variable candidates (host timing authority is binding; select only selectable=true):\n"
            if request.family_id in {DESCRIPTIVE_FAMILY_ID, PHENOTYPING_FAMILY_ID}
            else "Adjustment candidates (host timing authority is binding; select only selectable=true):\n"
        )
        + json.dumps(_candidate_rows(request), ensure_ascii=False)
    )
    if keeps_unmeasured_covariate_rows(request):
        sections.append(
            "Unmeasured covariates: the host keeps the unmeasured rows of a selected covariate "
            f"whose missing_share is at least {MISSING_CATEGORY_SHARE_THRESHOLD:g} as an explicit "
            "unmeasured state in the model, so do not drop a confounder because it is often "
            "unmeasured."
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
        if request.benchmark_omissions:
            sections.append(
                "Unavailable benchmark comparators (omit with these source reasons; do not substitute):\n"
                + json.dumps(request.benchmark_omissions, ensure_ascii=False)
            )
        if request.benchmark_candidates:
            sections.append(
                "Benchmark comparators (an existing score or probability the host compares the "
                "model with on the same validation stays; a benchmark requirement the plan "
                "answers names its column, and neither it nor a related column is a predictor):\n"
                + json.dumps(
                    [
                        {
                            "name": item.name,
                            "kind": item.kind,
                            "related_columns": item.related_columns,
                        }
                        for item in request.benchmark_candidates
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
    measure_section = _effect_measure_section(request)
    if measure_section:
        sections.append(measure_section)
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
    if request.literature_design_cards:
        sections.append(
            "\n".join(
                [
                    *render_literature_design_card_facts(request.literature_design_cards),
                    "Design decisions: one literature_design_decisions entry for each of "
                    + json.dumps(list(LITERATURE_DESIGN_DIMENSIONS))
                    + ". State whether this study adopts, adapts, diverges from, or finds "
                    "not applicable the reviewed choice, cite only cards that state that "
                    "dimension "
                    + json.dumps(literature_design_card_keys_by_dimension(request))
                    + ", and give a rationale specific to this question and the sealed "
                    "design without copying card text.",
                ]
            )
        )
    if request.population_concepts:
        sections.append(
            "Population authority (the host applies only what already_applied lists; a "
            "population stated beyond it is applied only by the population you write):\n"
            + json.dumps(_population_authority(request), ensure_ascii=False)
        )
        if request.prediction_time_hours is None:
            from ..acquisition.first_icu_stay import SELECTION_POLICY

            if not response_shape:
                sections.append(_typed_population_response_contract(request))
            sections.append(
                "Host-owned typed population policy for eicu/eicu_demo raw sources: "
                "first_icu_stay_per_hospitalization uses patienthealthsystemstayid, "
                "unitvisitnumber and unittype. The kind includes the ICU taxonomy and "
                "prior-unit completeness checks below; these are not separate concept "
                "predicates. This is the owner's rule, not proof that this input is "
                "already restricted. Only a verified receipt establishes that. If the "
                "study requests this kind, preserve these rules in its reviewable plan; "
                "other databases need their own verified owner. This is a population "
                "definition, not a separate longitudinal analysis: do not duplicate its "
                "unit taxonomy or prior-unit completeness as an analysis capability gap. "
                "If a named source concept also needs a question_requirements entry, "
                "use definition/definition_only and name the population element it defines; "
                "independently requested analyses still need their own coverage:\n"
                + json.dumps(SELECTION_POLICY, ensure_ascii=False, sort_keys=True)
            )
    if request.question_named_concepts:
        sections.append(
            "Concepts the research question names (question_named_concepts: each is "
            "accounted for by a question requirement that reads one of its columns, "
            "unless it is the sealed exposure or outcome):\n"
            + json.dumps(
                [
                    item.model_dump(mode="json")
                    for item in request.question_named_concepts
                ],
                ensure_ascii=False,
            )
        )
    if know_how_context:
        sections.append("Know-how context:\n" + know_how_context)
    if planning_contract_context:
        sections.append("Planning contract context:\n" + planning_contract_context)
    if response_shape:
        sections.append("Response shape (no schema is attached on this route):\n" + response_shape)
    # Retain the complete request identity in cassette lookup, without an echo.
    sections.append("Host request identity (bound by caller): " + request.request_sha256)
    return "\n\n".join(sections)


def family_spec_messages(
    request: FamilySpecRequest,
    *,
    context: ResearchContext | None = None,
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
                context=context, variable_descriptions=variable_descriptions,
                know_how_context=know_how_context,
                planning_contract_context=planning_contract_context,
                response_shape=response_shape,
            ),
        ),
    ]


def _enum(values: Sequence[str]) -> dict[str, Any]:
    normalized = list(dict.fromkeys(str(value) for value in values if str(value)))
    return {"type": "string", "enum": normalized}


_PREDICATE_AGGREGATIONS = list(get_args(ProgressiveCohortPredicate.model_fields["aggregation"].annotation))
_PREDICATE_OPS = list(get_args(ProgressiveCohortPredicate.model_fields["op"].annotation))
_PREDICATE_VALUE_MODES = list(get_args(ProgressivePredicateValue.model_fields["mode"].annotation))


def _population_authority(request: FamilySpecRequest) -> dict[str, Any]:
    """What the population the Planner states may read, and what is already applied.

    already_applied lists the typed bounds the template applies and what the
    source export is known to have applied: its concept population and its
    contracts (all of them when its selection is recorded, else the host's).
    """

    typed = request.cohort_selection_mode == "predicate_filtered"
    # Under a caller-bound population only the risk set is applied.
    bounds = typed and not request.caller_binds_all_input_rows
    applied = {
        key: value
        for key, value in (
            ("age_min", request.age_min if bounds else None),
            ("age_max", request.age_max if bounds else None),
            ("minimum_icu_hours", request.minimum_icu_hours if bounds else None),
            # A prediction model's stays: still in the ICU after it predicts,
            # and, where deaths are read by their time, none recorded before it.
            ("still_in_icu_after_prediction_time_hours", request.prediction_time_hours if typed else None),
            (
                "no_death_recorded_before_prediction_time_hours",
                request.prediction_time_hours
                if typed
                and request.prediction_death_time is not None
                and request.prediction_death_time.applied
                else None,
            ),
            ("source_concept_population", request.concept_cohort_definition),
            ("source_concept_population_window_end_hours", request.concept_cohort_window_end_hours),
            ("source_inclusion_contracts", list(request.source_applied_inclusion) or None),
            ("source_exclusion_contracts", list(request.source_applied_exclusion) or None),
        )
        if value is not None
    }
    return {
        "time_zero_hours_after_icu_admission": request.cohort_time_zero_hours,
        "already_applied": applied,
        # Stated only when recorded: an unrecorded selection may hold more
        # than already_applied lists, and nothing is claimed about it.
        **({"source_selection_recorded": True} if request.source_selection_recorded else {}),
        "study_cohort_wording": dict(request.study_cohort_wording),
        "population_concepts": list(request.population_concepts),
        "population_required": population_required(request),
    }


def _population_predicate_shape() -> dict[str, Any]:
    return {
        "concept_id": "<one of the population concepts>",
        "anchor": POPULATION_ANCHOR,
        "start_offset_hours": "<number>",
        "end_offset_hours": "<greater number, at most the time zero>",
        "aggregation": "<" + "|".join(_PREDICATE_AGGREGATIONS) + ">",
        "op": "<" + "|".join(_PREDICATE_OPS) + ">",
        "value": {
            "mode": "<" + "|".join(_PREDICATE_VALUE_MODES) + ">",
            "string_value": None,
            "number_value": None,
            "boolean_value": None,
            "string_list": [],
            "number_list": [],
        },
    }


def _typed_population_response_contract(request: FamilySpecRequest) -> str:
    """Show the nested transport and the host bounds a complete spec retains.

    This is a response template and a projection of existing request authority,
    never a population injected into the response or a researcher approval.
    """

    shape = population_spec_shape()
    shape["criteria"][0]["id"] = "c1"
    pending = []
    if request.cohort_selection_mode == "predicate_filtered" and not request.caller_binds_all_input_rows:
        if request.age_min is not None or request.age_max is not None:
            pending.append({"kind": "age_years", "min_years": request.age_min,
                            "max_years": request.age_max})
        if request.minimum_icu_hours is not None:
            pending.append({"kind": "icu_stay_hours", "min_hours": request.minimum_icu_hours,
                            "max_hours": None})
    return (
        "Typed population response template (angle brackets are instructions to replace, "
        "not literal output; add each selected kind's own fields below):\n"
        + json.dumps({"population": {"criteria": [], "inclusion": [], "exclusion": [],
                                     "population_spec": shape}}, ensure_ascii=False)
        + "\npopulation.population_spec must be one JSON object with exactly the key "
        "criteria; it is never an array, and is never a direct criterion object. Only "
        "population.population_spec.criteria is the array of typed criteria. Every "
        "criterion has id, quote, source, role, kind and that kind's own fields. "
        'Each id is a unique string matching ^c[1-9][0-9]?$: "c1", "c2", ..., "c99"; '
        'never a number, a kind name, "age", or "first_icu". Each quote copies its '
        "source words exactly, including the study's explicit cohort review/exclusion "
        "wording when source is study_wording; a display label alone is not source evidence. "
        "When source is question, quote the question or its canonical inclusion/exclusion text. A kind with no extra fields still keeps these "
        "five common fields. For age_years, include min_years and max_years as numbers or "
        "null (an open bound); for icu_stay_hours use min_hours and max_hours the same way. "
        "At least one bound must be a number. Do not write null as a string."
        "\npending_typed_population_bounds: "
        + json.dumps(pending, ensure_ascii=False)
        + "\nThese are already host-bound values, not new choices. If using population_spec, "
        "include criteria retaining every listed age/stay bound, alongside any other "
        "restriction the study requests. Do not omit them because the legacy "
        "already_applied view lists them: that template no longer supplies predicates "
        "when population_spec replaces the whole cohort. Choose the criterion ids and "
        "verbatim quotes yourself; this template does not select or approve other criteria."
    )


def _population_schema(concepts: Sequence[str]) -> dict[str, Any]:
    """The strict shape of a stated population over the offered concepts."""

    predicate = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "concept_id",
            "anchor",
            "start_offset_hours",
            "end_offset_hours",
            "aggregation",
            "op",
            "value",
        ],
        "properties": {
            "concept_id": _enum(concepts),
            "anchor": _enum([POPULATION_ANCHOR]),
            "start_offset_hours": {"type": "number"},
            "end_offset_hours": {"type": "number"},
            "aggregation": _enum(_PREDICATE_AGGREGATIONS),
            "op": _enum(_PREDICATE_OPS),
            "value": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "mode",
                    "string_value",
                    "number_value",
                    "boolean_value",
                    "string_list",
                    "number_list",
                ],
                "properties": {
                    "mode": _enum(_PREDICATE_VALUE_MODES),
                    "string_value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "number_value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "boolean_value": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                    "string_list": {"type": "array", "items": {"type": "string"}},
                    "number_list": {"type": "array", "items": {"type": "number"}},
                },
            },
        },
    }
    criterion = {
        "type": "object",
        "additionalProperties": False,
        "required": ["criterion", "concept_ids"],
        "properties": {
            "criterion": {"type": "string"},
            "concept_ids": {"type": "array", "items": _enum(concepts)},
        },
    }
    return {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["criteria", "inclusion", "exclusion", "population_spec"],
                "properties": {
                    "criteria": {"type": "array", "items": criterion},
                    "inclusion": {"type": "array", "items": predicate},
                    "exclusion": {"type": "array", "items": predicate},
                    "population_spec": {"type": "null"},
                },
            },
            {"type": "null"},
        ]
    }


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
    design_cards = bool(request.literature_design_cards)
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "family_id",
            "adjustment_set",
            *(["baseline_variables"] if descriptive or phenotyping else []),
            *(["distribution_missing_policies"] if descriptive else []),
            *(["feature_variables", "cohort_membership_column"] if phenotyping else []),
            *(["feature_variables"] if prediction else []),
            "reader_display_labels",
            "comparator_applications",
            *(["literature_design_decisions"] if design_cards else []),
            "roster_decision_note",
            *(["population"] if request.population_concepts else []),
            "question_requirements",
            *(["effect_measure"] if len(family_effect_measures(request)) > 1 else []),
        ],
        "properties": {
            "schema_version": {"type": "string", "enum": [FAMILY_SPEC_SCHEMA_VERSION]},
            "family_id": {"type": "string", "enum": [request.family_id]},
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
                {"distribution_missing_policies": DistributionMissingPolicies.model_json_schema()}
                if descriptive else {}
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
            **(
                {
                    "literature_design_decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "dimension",
                                "citation_keys",
                                "disposition",
                                "rationale",
                            ],
                            "properties": {
                                "dimension": _enum(LITERATURE_DESIGN_DIMENSIONS),
                                "citation_keys": {
                                    "type": "array",
                                    "items": _enum(
                                        [card.citation_key for card in request.literature_design_cards]
                                    ),
                                },
                                "disposition": _enum(
                                    ["adopt", "adapt", "diverge", "not_applicable"]
                                ),
                                "rationale": {"type": "string"},
                            },
                        },
                    },
                }
                if design_cards
                else {}
            ),
            "roster_decision_note": {"type": "string"},
            **(
                {"population": _population_schema(request.population_concepts)}
                if request.population_concepts
                else {}
            ),
            "question_requirements": question_requirements_schema(
                request.variable_roster,
                effect_measures=_requirement_effect_measures(request),
            ),
            **(
                {
                    "effect_measure": {
                        "anyOf": [_enum(list(family_effect_measures(request))), {"type": "null"}]
                    }
                }
                if len(family_effect_measures(request)) > 1
                else {}
            ),
        },
    }
    if request.population_concepts:
        population_properties = schema["properties"]["population"]["anyOf"][0]["properties"]
        bind_population_spec_transport(
            schema.setdefault("$defs", {}), population_properties,
            concept_ids=request.population_concepts,
            stated=request.prediction_time_hours is None,
        )
    strictify_json_schema(schema)
    try:
        assert_closed_json_schema(schema)
    except StrictJsonSchemaError as exc:
        raise ValueError(f"family spec schema is not closed: {exc}") from exc
    return StructuredOutputRequest.from_schema(name="family_plan_spec", schema=schema, strict=True)


def parse_family_plan_spec(raw: str, request: FamilySpecRequest) -> FamilyPlanSpec:
    """Parse one provider response and validate it against the sealed request.

    The host binds the envelope to the request sent in this call; copying an
    identity adds no authority. Saved specs use ``spec_from_mapping`` and
    ``validate_family_plan_spec`` instead, retaining their original binding.
    """

    payload = json.loads(str(raw or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("family spec response root must be an object")
    spec = FamilyPlanSpec.model_validate(
        {
            **payload,
            "schema_version": FAMILY_SPEC_SCHEMA_VERSION,
            "family_id": request.family_id,
            "request_sha256": request.request_sha256,
        }
    )
    validate_family_plan_spec(spec, request)
    return spec


def check_question_requirement_gaps(
    spec: FamilyPlanSpec,
    *,
    context: ResearchContext,
    planning_contract_context: str = "",
) -> None:
    """A gap the spec declares stops the plan, so the host checks it first.

    One the study's typed context contradicts goes back to the Planner with
    what the context shows (``planning.capability_gap``), under the outline
    route's code; a verified or unverifiable gap stands and is judged once the
    plan exists.
    """

    for index, item in enumerate(spec.question_requirements):
        if item.gap is None:
            continue
        check = check_capability_gap(
            item.gap,
            context=context,
            planning_contract_context=planning_contract_context,
        )
        if check.verification == "unverified":
            raise ProgressivePlanCompileError(
                "progressive_capability_gap_claim_unverified",
                f"question requirement {item.id} declares a capability gap the study "
                f"contradicts: {check.fact}; state what the plan answers, or a gap the "
                "study shows",
                path=f"question_requirements[{index}].gap",
                cause_code=item.gap.requirement,
            )


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
    ]
    planner_roster = planner_selects_adjustment(request)
    # A proposed survival suite adjusts like a landmark association family.
    if request.family_id not in LANDMARK_FAMILY_IDS and request.suite_proposal is None:
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
            + ("; at least one entry" if planner_roster and selectable else "")
        )
    if request.family_id in {DESCRIPTIVE_FAMILY_ID, PHENOTYPING_FAMILY_ID}:
        lines.append('- "baseline_variables": array of distinct names from ' + json.dumps(selectable))
    if request.family_id == DESCRIPTIVE_FAMILY_ID:
        lines.extend([
            '  Use [] when no baseline summaries are requested; eligibility-only variables do not require Table 1.',
            '- "distribution_missing_policies": object with exactly "denominator_policy" '
            '(all_declared_rows or observed_outcome_rows), "missing_exposure_policy" and '
            '"missing_outcome_policy" (each fail_closed or exclude_from_denominator). '
            'Use observed_outcome_rows exactly when missing_outcome_policy is exclude_from_denominator. '
            'Excluded missing values are counted and reported; missing outcomes remain in cohort audits. '
            'Choose explicitly before review; metadata-only zero counts do not prove completeness.',
        ])
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
        # State the roster bound the validator applies when the offered
        # candidates can exceed it, so a broad roster is trimmed on the first
        # draft instead of after a refusal.
        budget = fit_roster_budget(request)
        describes = request.family_id == PHENOTYPING_FAMILY_ID
        offered = len(features) + (
            len(selectable) + (1 if request.membership_candidates else 0) if describes else 0
        )
        if offered > budget or len(features) > MAX_FIT_FEATURES:
            fixed = "the row identity and the outcome" if request.outcome else "the row identity"
            lines.append(
                f"  The design names at most {design_field_max_length('required_variables')} "
                f"variables, {fixed} included: "
                + (
                    f"feature_variables (at most {MAX_FIT_FEATURES}), baseline_variables and any "
                    f"cohort_membership_column together name at most {budget}"
                    if describes
                    else f"at most {min(budget, MAX_FIT_FEATURES)} feature_variables"
                )
                + "; keep the most informative."
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
    } or planner_roster
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
    if request.literature_design_cards:
        lines.append(
            '- "literature_design_decisions": array of {"dimension": ..., "citation_keys": '
            '[...], "disposition": ..., "rationale": ...} objects, exactly one for each of '
            + json.dumps(list(LITERATURE_DESIGN_DIMENSIONS))
            + '; "disposition" is one of ["adopt", "adapt", "diverge", "not_applicable"], '
            '"citation_keys" names only cards that state that dimension '
            + json.dumps(literature_design_card_keys_by_dimension(request))
            + ', and "rationale" is 12-800 characters'
        )
    lines.append('- "roster_decision_note": one or more sentences (8-1200 characters)')
    if request.population_concepts:
        lines.append(
            '- "population": null, or an object with exactly the keys "criteria", "inclusion", '
            '"exclusion" and "population_spec". Use population_spec for the complete typed '
            'population, retaining pending host age/stay bounds, with the other three lists empty. '
            'Its kinds need no concept column unless their own fields name one. For the legacy '
            'predicate representation set population_spec null; "criteria" is an array of 1-6 objects, each exactly '
            '{"criterion": "<2-160 characters, in the words that state it>", "concept_ids": '
            "[0-6 distinct names from "
            + json.dumps(list(request.population_concepts), ensure_ascii=False)
            + ']}; "inclusion" and "exclusion" are arrays of predicates, each exactly '
            + json.dumps(_population_predicate_shape(), ensure_ascii=False, separators=(",", ":"))
            + ", with only the value field its mode selects filled. Every criterion with "
            "concepts is applied by at least one predicate over one of them, and every "
            "predicate reads a concept that a criterion names. "
            + (
                _typed_population_response_contract(request)
                + population_spec_contract(request.population_concepts)
                + " A phenotyping membership-column filter must use the legacy representation; "
                "do not combine it with population_spec."
                if request.prediction_time_hours is None else
                "This request has a template-owned prediction risk set: population_spec must "
                "be null and any additional population must use the legacy representation."
            )
        )
    lines.append(
        question_requirements_shape(
            request.variable_roster, effect_measures=_requirement_effect_measures(request)
        )
    )
    if len(family_effect_measures(request)) > 1:
        lines.append(
            '- "effect_measure": null (odds ratio) or one of '
            + json.dumps(list(family_effect_measures(request)))
        )
    return "\n".join(lines)




def _source_absolute_risk_scope(context: ResearchContext) -> str | None:
    """The absolute-risk population a revision's source plan bound, if any.

    The outline check reads the same binding (``validate_population_choice``).
    """

    from ..planning.population_requirements import context_population_requirements

    required = context_population_requirements(context)
    if required is None:
        return None
    return next(
        (
            population.population_scope
            for population in required.populations
            if population.output_product == "table:absolute_risk_context"
        ),
        None,
    )


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
    design_cards: Sequence[LiteratureDesignEvidenceCard] = (),
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
        ..., tuple[AnalysisPlan, ProgressivePlanCompileReceipt]
    ],
    capture_efficiency_metrics: Callable[[], None],
) -> AnalysisPlan:
    """One spec call, then a host-projected skeleton through the same gates.

    The Planner fills the family spec; the host template projects outline,
    foundation, and every step. Foundation validation runs inside parsing,
    so a population quote refusal reaches structured retry with its existing
    budget. The fresh-plan outline authority, prefix compiler, and final
    acceptance run unchanged (injected by the Progressive agent). A rejected
    projection keeps the validator's own reason code; there is no Provider
    suffix repair on this path. A request the host refuses stops the same way, so the run
    records the request's own reason code, and states that the Planner made no
    Provider call (``planner_provider_calls`` 0).
    """

    try:
        request = build_family_spec_request(
            context,
            analysis_types=analysis_types,
            variable_roster=variables,
            allowed_literature_citation_keys=allowed_citations,
            direct_comparator_literature_keys=direct_keys,
            comparison_literature_keys=comparison_keys,
            required_primary_cohort_selection_mode=required_primary_cohort_selection_mode,
            planning_contract_context=planning_contract_context,
            literature_design_cards=design_cards,
            cohort_concept_ids=progressive_cohort_concept_ids(context, variables),
        )
    except FamilySpecError as exc:
        error = ProgressivePlanCompileError(
            f"progressive_{exc.reason_code}", str(exc), path=exc.path or "family_spec",
            metrics={"planner_provider_calls": 0}, cause_code=exc.cause_code,
        )
        if exc.stated_variable is not None:
            error.easyicu_safe_diagnostic["stated_variable"] = exc.stated_variable
        raise error from exc
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
        request, context=article_context,
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

    def project_spec(spec: FamilyPlanSpec) -> FamilySkeletonDraft:
        builder = {
            BASELINE_ASSOCIATION_FAMILY_ID: build_baseline_association_skeleton,
            DESCRIPTIVE_FAMILY_ID: build_descriptive_skeleton,
            PHENOTYPING_FAMILY_ID: build_phenotyping_skeleton,
            PREDICTION_FAMILY_ID: build_prediction_skeleton,
            LANDMARK_SURVIVAL_FAMILY_ID: build_landmark_survival_skeleton,
            LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID: build_landmark_continuous_survival_skeleton,
            FIXED_WINDOW_TRAJECTORY_FAMILY_ID: build_fixed_window_trajectory_skeleton,
            SOURCE_FEASIBILITY_FAMILY_ID: build_source_feasibility_skeleton,
        }.get(request.family_id, build_landmark_association_skeleton)
        try:
            if builder is build_landmark_association_skeleton:
                draft = builder(
                    request, spec, bind_outline=bind_outline,
                    source_absolute_risk_scope=_source_absolute_risk_scope(article_context),
                )
            else:
                draft = builder(request, spec, bind_outline=bind_outline)
        except FamilySpecError as exc:
            raise ProgressivePlanCompileError(
                f"progressive_{exc.reason_code}", str(exc), path=exc.path or "family_spec",
            ) from exc
        draft = reconcile_population_grouping(draft, context=context, spec=spec)
        outline = draft.outline
        if draft.foundation.outline_sha256 != canonical_sha256(outline.model_dump(mode="json")):
            raise ProgressivePlanCompileError(
                "progressive_foundation_outline_digest_mismatch",
                "family template foundation did not bind the validated outline digest",
                path="outline_sha256",
            )
        require_robustness_intent = bool(
            enforce_article_contract
            and "robustness"
            in build_article_analysis_contract(
                article_context, analysis_type=outline.analysis_type,
            ).required_roles
        )
        # The same foundation owner checks the population's original words.
        # Keep it inside parsing so the model sees its located refusal within
        # the existing retry budget, before any checkpoint is published.
        validate_progressive_foundation(
            draft.foundation.foundation,
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
        return draft

    def retryable_parse_error(exc: Exception) -> bool:
        # JSON, Pydantic and family/question validation errors derive from
        # ValueError. Of the compiler's refusals, only the quote check moved
        # into retry; the capability-gap check was already retried here.
        return isinstance(exc, ValueError) and (
            not isinstance(exc, ProgressivePlanCompileError)
            or exc.reason_code in {
                "progressive_population_quote_not_verbatim",
                "progressive_capability_gap_claim_unverified",
            }
        )

    def parse_spec(raw: str) -> tuple[FamilyPlanSpec, FamilySkeletonDraft]:
        try:
            spec = parse_family_plan_spec(raw, request)
            check_question_requirement_gaps(
                spec, context=context, planning_contract_context=planning_contract_context
            )
            return spec, project_spec(spec)
        except ValueError as exc:
            if not retryable_parse_error(exc):
                raise
            # The located rejection comes first so even a quote beyond the
            # bounded response head is diagnosable. The shared capture owner
            # requires explicit opt-in and keeps redacted, private files only.
            # Reserve room for the response even if the allowed source is long.
            rejection = f"{getattr(exc, 'path', '')}: {exc}"[:1000]
            _dump_raw(
                f"{rejection}\n\nRejected response:\n{raw}",
                "family_spec_rejected",
            )
            raise

    spec, draft = call_llm_with_structured_retry(
        llm,
        messages,
        parser=parse_spec,
        retryable_parse_error=retryable_parse_error,
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
    # Retain proposals for review even when the executable roster is host-bound.
    attempt.prompt_metrics["family_spec_roster_decision_note"] = spec.roster_decision_note
    attempt.prompt_metrics["family_spec_bound_adjustment_set"] = (
        list(request.exact_roster) if request.adjustment_selection == "exact"
        else [item.name for item in spec.adjustment_set]
    )
    # What the question asks beyond the design, judged once the plan exists,
    # and that the plan's steps are the family template's.
    attempt.question_requirements = list(spec.question_requirements)
    attempt.family_template = True

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
    foundation = foundation_materialization.foundation
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
    # The template's reviewed roster is host-written, never Planner transport.
    plan, receipt = compile_and_accept(skeleton, adjustment_proposal=draft.adjustment_proposal)
    attempt.skeleton = skeleton
    attempt.compile_receipt = receipt
    attempt.prompt_metrics["final_skeleton_sha256"] = receipt.skeleton_sha256
    attempt.prompt_metrics["compiled_plan_sha256"] = receipt.analysis_plan_sha256
    capture_efficiency_metrics()
    return plan


def family_spec_fallback_reason(
    context: ResearchContext,
    *,
    analysis_types: Sequence[str],
    resume_checkpoint: ProgressivePlannerCheckpoint | None,
    stop_after_outline: bool,
    planning_contract_context: str = "",
) -> str | None:
    """Return why the family-spec strategy must yield to Progressive v2, or None."""

    if resume_checkpoint is not None:
        return "development_resume_checkpoint_uses_progressive_v2"
    if stop_after_outline:
        return "design_canary_uses_progressive_v2"
    family_id = family_template_id_for_context(
        context, analysis_types=analysis_types,
        planning_contract_context=planning_contract_context,
    )
    if family_id is None:
        return "no_family_template_for_context"
    if _stated_subgroups_need_outline(context, family_id, planning_contract_context):
        return "family_template_stated_subgroup_requires_outline"
    try:
        request = build_family_spec_request(
            context, analysis_types=analysis_types,
            variable_roster=[variable.name for variable in context.variables],
            allowed_literature_citation_keys=(),
            planning_contract_context=planning_contract_context,
        )
    except FamilySpecError:
        # Preserve the owner's typed refusal for an invalid request.
        return None
    if _template_sealed_for_run(family_id, planning_contract_context):
        # A host-sealed owner (a fail-closed feasibility decision, a signed
        # suite) defines this run's result; Progressive v2 must not replan it
        # because the question also named outcomes the sealed result omits.
        return None
    if set(context.cohort.requested_outcome_columns or ()) - set(request.covered_outcome_columns):
        return "family_template_requested_outcomes_not_supported"
    return None


def _stated_subgroups_need_outline(
    context: ResearchContext, family_id: str, planning_contract_context: str,
) -> bool:
    """Whether a stated subgroup needs a step this family template never drafts.

    The descriptive template reports its outcome in each level of the study's
    exposure, which answers a subgroup that is that exposure.  No template
    drafts a subgroup estimate.  Progressive planning can add the family's
    subgroup owner while the accepted-baseline gate keeps the original Table 1
    grouping and rows, so it plans instead, except where only the template
    plans the primary (a static prediction): scientific review then stops on
    the subgroup no route can answer.
    """

    from ..planning.adjustment_authority import context_stated_variables
    from ..planning.question_requirements import concept_relatives

    # A reading without an available concept is a review finding either way.
    subgroups = [
        item for item in context_stated_variables(context, "stated_subgroups")
        if item.concept and item.source_status == "available"
    ]
    if not subgroups or family_id in {PREDICTION_FAMILY_ID, BASELINE_ASSOCIATION_FAMILY_ID} or _template_sealed_for_run(
        family_id, planning_contract_context,
    ):
        return False
    exposure = str(context.primary_exposure or "").strip()
    relatives = concept_relatives(context)
    return not (family_id == DESCRIPTIVE_FAMILY_ID and exposure and all(
        exposure in relatives(item.concept) for item in subgroups
    ))


def _template_sealed_for_run(family_id: str, planning_contract_context: str) -> bool:
    """Whether the host sealed this template's owner for the current run."""

    sealed = {
        SOURCE_FEASIBILITY_FAMILY_ID: sealed_feasibility_coordinates,
        FIXED_WINDOW_TRAJECTORY_FAMILY_ID: sealed_trajectory_suite_coordinates,
        LANDMARK_SURVIVAL_FAMILY_ID: sealed_survival_suite_coordinates,
        LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID: sealed_continuous_survival_suite_coordinates,
    }.get(family_id)
    return sealed is not None and sealed(planning_contract_context) is not None


def family_result_stop(
    context: ResearchContext,
    *,
    families: Sequence[str],
    planner_strategy: str,
    planning_contract_context: str,
    outline_selected: bool = False,
) -> ProgressivePlanCompileError:
    """Progressive v2's stop for families whose result contract it cannot write.

    It names this strategy when the run uses another one and a host template
    plans these families.
    """

    templated = (
        planner_strategy != FAMILY_SPEC_STRATEGY
        and family_template_id_for_context(
            context,
            analysis_types=families,
            planning_contract_context=planning_contract_context,
        )
        is not None
    )
    reason, message = family_result_stop_reason(
        families,
        template_strategy=FAMILY_SPEC_STRATEGY if templated else None,
        outline_selected=outline_selected,
    )
    return ProgressivePlanCompileError(reason, message, path="analysis_type")


__all__ = [
    "FAMILY_SPEC_GUIDE",
    "FAMILY_SPEC_MAX_OUTPUT_TOKENS",
    "FAMILY_SPEC_ROLE",
    "FAMILY_SPEC_STRATEGY",
    "check_question_requirement_gaps",
    "family_result_stop",
    "family_spec_fallback_reason",
    "family_spec_messages",
    "family_spec_response_shape",
    "family_spec_structured_output_request",
    "family_spec_user_prompt",
    "parse_family_plan_spec",
    "run_family_spec_attempt",
]
