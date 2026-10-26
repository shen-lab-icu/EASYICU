"""Dry compile a complementary study through the existing family owners.

No model, extraction or execution runs here. An unsupported template, missing
coordinate or unclaimed capability omits the suggestion. A new study must still
go through ordinary data preparation, scientific review and approval.
"""
from __future__ import annotations

import json

from easyicu.research_agent.planning.capability_registry import resolve_primary_capability
from easyicu.research_agent.planning.family_spec import (
    build_baseline_association_skeleton, build_descriptive_skeleton,
    build_family_spec_request, build_landmark_association_skeleton,
    build_landmark_continuous_survival_skeleton, build_landmark_survival_skeleton,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.contract import DistributionMissingPolicies, FamilyPlanSpec
from easyicu.research_agent.planning.progressive_contract import ProgressiveCohortIntent, ProgressiveCohortPredicate, ProgressivePredicateValue
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_resume import assemble_progressive_skeleton
from easyicu.research_agent.schema import AnalysisPlan, ResearchContext, UserPreferences
from easyicu.webserver.study_family_scope import study_family_database_gap


def require_database_family(context: ResearchContext, family: str) -> None:
    if study_family_database_gap(
        {"analysis_design": {"analysis_family": family}},
        question=context.research_question, database=context.cohort.database,
    ):
        raise ValueError("followup_database_refused")
    if family == "causal_inference":
        # No causal follow-up owner is supplied by this feature.
        raise ValueError("followup_family_unavailable")


def _source_cohort(plan: AnalysisPlan) -> ProgressiveCohortIntent:
    cohort = plan.cohort
    if cohort is None or cohort.unapplied_population_criteria:
        raise ValueError("followup_population_unavailable")

    def predicate(item):
        value = item.value
        if value is None:
            wire = {"mode": "none"}
        elif isinstance(value, bool):
            wire = {"mode": "boolean", "boolean_value": value}
        elif isinstance(value, str):
            wire = {"mode": "string", "string_value": value}
        elif isinstance(value, (int, float)):
            wire = {"mode": "number", "number_value": value}
        else:
            mode = "string_list" if all(isinstance(v, str) for v in value) else "number_list"
            wire = {"mode": mode, mode: list(value)}
        return ProgressiveCohortPredicate(
            concept_id=item.concept_id, anchor=item.time_window.anchor,
            start_offset_hours=item.time_window.start_offset_hours, end_offset_hours=item.time_window.end_offset_hours,
            aggregation=item.aggregation, op=item.op, value=ProgressivePredicateValue.model_validate(wire),
        )
    return ProgressiveCohortIntent(name=cohort.name, selection_mode=cohort.selection_mode,
                                   inclusion=[predicate(p) for p in cohort.inclusion], exclusion=[predicate(p) for p in cohort.exclusion])


def compile_new_study(context: ResearchContext, *, family: str, source_plan: AnalysisPlan | None = None,
                      primary_exposure: str | None = None) -> tuple[ResearchContext, AnalysisPlan]:
    """Reconstruct the proposed reading from the sealed source, then compile it.

    Preserve population, endpoint, timing, dependence and the reviewed adjustment
    roster. The delta names the changed family and any descriptive grouping.
    The source reading is retained separately; source objects are never modified.
    """
    require_database_family(context, family)
    preferences = (context.user_preferences or UserPreferences()).model_copy(deep=True)
    if preferences.formal_result_scope:
        raise ValueError("followup_family_unavailable")
    constraints = json.loads(preferences.data_constraints or "{}")
    if constraints.get("analysis_design"):
        constraints["analysis_design"] = {**constraints["analysis_design"], "analysis_family": family}
    preferences.data_constraints = json.dumps(constraints)
    preferences.inferred_analysis_family = family
    exposure = primary_exposure or context.primary_exposure
    question = {
        "descriptive_epidemiology": "Describe {outcome} by {exposure} in the source study population.",
        "association_study": "Estimate the association of {exposure} with {outcome} in the source study population.",
        "survival": "Estimate time to {outcome} by {exposure} in the source study population.",
    }[family].format(exposure=exposure, outcome=context.target_outcome)
    proposed = context.model_copy(deep=True, update={"research_question": question, "user_preferences": preferences,
                                                   "primary_exposure": exposure})
    request = build_family_spec_request(
        proposed, analysis_types=(family,), variable_roster=[v.name for v in proposed.variables],
        allowed_literature_citation_keys=(),
    )
    builders = {
        "baseline_adjusted_association": build_baseline_association_skeleton,
        "landmark_categorical_association": build_landmark_association_skeleton,
        "landmark_spline_association": build_landmark_association_skeleton,
        "descriptive_exposure_outcome": build_descriptive_skeleton,
        "landmark_survival_suite": build_landmark_survival_skeleton,
        "landmark_continuous_survival_suite": build_landmark_continuous_survival_skeleton,
    }
    builder = builders.get(request.family_id)
    if builder is None or (family in {"association_study", "survival"} and request.adjustment_selection != "exact"):
        raise ValueError("followup_family_unavailable")
    missing = request.distribution_missing_policies
    if family == "descriptive_epidemiology" and missing is None:
        missing = DistributionMissingPolicies(denominator_policy="all_declared_rows",
                                             missing_exposure_policy="fail_closed", missing_outcome_policy="fail_closed")
    spec = FamilyPlanSpec(
        family_id=request.family_id, request_sha256=request.request_sha256,
        reader_display_labels=[{"key": key, "value": (source_plan.display_labels.get(key) if source_plan else None)
                                or "Source " + key.replace("_", " ")}
                               for key in dict.fromkeys([*request.required_reader_label_keys, *request.level_label_keys])],
        distribution_missing_policies=missing,
        roster_decision_note="Retain the source study's typed reading.",
    )
    validate_family_plan_spec(spec, request)
    draft = builder(request, spec)
    skeleton = assemble_progressive_skeleton(
        outline=draft.outline, foundation=draft.foundation.foundation,
        steps=[item.step for item in draft.materializations],
    )
    if source_plan is not None:
        skeleton = skeleton.model_copy(update={"cohort": _source_cohort(source_plan)})
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=proposed, adjustment_proposal=draft.adjustment_proposal)
    verdict = resolve_primary_capability(analysis_type=plan.analysis_type, plan=plan)
    if not verdict.coherent or verdict.execution_owner != "host_deterministic" or verdict.scientific_validation == "unsupported":
        raise ValueError("followup_capability_unavailable")
    return proposed, plan
