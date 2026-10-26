"""Read source requirements from existing typed context/plan coordinates."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from easyicu.databases.profiles import get_database_profile
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.research_agent.planning.analysis_types import infer_analysis_type
from .research_requirements import StudyRequirements, ThresholdRequirement, assess_study, family_requirements

if TYPE_CHECKING:
    from easyicu.research_agent.schema import AnalysisPlan, ResearchContext
    from .research_requirements import CapabilityDecision


def _constraints(context: ResearchContext) -> Mapping:
    raw = getattr(context.user_preferences, "data_constraints", None)
    try:
        constraints = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        constraints = {}
    return constraints if isinstance(constraints, Mapping) else {}


def _effective_unit(constraints: Mapping) -> str:
    design = constraints.get("analysis_design") or {}
    unit = design.get("analysis_unit", "icu_stay")
    return {"hospital_admission": "hospitalization", "row": "icu_stay"}.get(unit, unit)


def plan_capability_decision(context: ResearchContext, plan: AnalysisPlan) -> CapabilityDecision | None:
    """No question parsing and no inference of a field's clinical meaning.

    ``database_study_requirements`` can retain a host's original requirements
    while a reviewed plan declares substitute predicates. It is carried in the
    existing context constraints, not added to Planner's output schema.
    Unregistered cohorts keep their existing review and execution contracts.
    """
    database = context.cohort.database
    try:
        get_database_profile("miiv" if database == "miiv_demo" else database)
    except KeyError:
        return None
    # Only an unknown profile is unassessed. Errors in registered capability
    # metadata must still surface instead of silently bypassing approval.
    profile = research_capability_profile(context.cohort.database)
    constraints = _constraints(context)
    original = constraints.get("database_study_requirements")
    unit = _effective_unit(constraints)
    from easyicu.research_agent.methods.descriptive_inputs import plan_value_columns

    concepts = set()
    for name in (context.target_outcome, context.primary_exposure, *plan_value_columns(plan)):
        if name:
            variable = next((item for item in context.variables if item.name == name), None)
            concepts.add((variable.source_concept if variable else None) or name)
    thresholds = []
    if plan.cohort is not None:
        for predicate in (*plan.cohort.inclusion, *plan.cohort.exclusion):
            concepts.add(predicate.concept_id)
            if predicate.concept_id in {"age", "weight"} and predicate.op in {">", ">=", "<", "<=", "==", "!="}:
                thresholds.append(ThresholdRequirement(concept=predicate.concept_id, operator=predicate.op, value=predicate.value))
    requirements = StudyRequirements(
        family=plan.analysis_type or infer_analysis_type(context).key, analysis_unit=unit,
        outcome="hospital_death" if "death" in concepts else context.target_outcome,
        concepts=tuple(sorted(concepts)), thresholds=tuple(thresholds),
        censoring_unit="hospitalization" if "death" in concepts else None,
    )
    if original is not None:
        stated = StudyRequirements.model_validate(original)
        requirements = stated.model_copy(update={
            "family": requirements.family,
            "analysis_unit": unit if unit not in profile.analysis_units else stated.analysis_unit,
            "outcome": stated.outcome or requirements.outcome,
            "censoring_unit": stated.censoring_unit or requirements.censoring_unit,
            "concepts": tuple(sorted(set(stated.concepts) | concepts)),
            "thresholds": tuple(dict.fromkeys((*stated.thresholds, *thresholds))),
        })
    if "death" in concepts and not (
        requirements.outcome == "hospital_death" and "event_time" in family_requirements(requirements.family)
    ):
        # The generic death builder has this dictionary meaning even when the
        # source records another mortality status that needs a separate builder.
        profile = profile.model_copy(update={"mortality_outcomes": (profile.mortality,) if profile.mortality else ()})
    if "database_capability_binding" in constraints:
        from .research_substitutions import source_decision

        return source_decision(profile, requirements)
    return assess_study(profile, requirements)


def unapplied_substitution_capabilities(decision: CapabilityDecision, context: ResearchContext,
                                        plan: AnalysisPlan) -> tuple[str, ...]:
    """Approval cannot make unchanged predicates or grouping implement a proposal."""
    predicates = (*plan.cohort.inclusion, *plan.cohort.exclusion) if plan.cohort else ()
    coordinates = {(item.concept_id, item.op, item.value) for item in predicates
                   if item.concept_id in {"age", "weight"} and isinstance(item.value, (int, float))}
    missing = []
    for item in decision.substitutions:
        replacement = item.substitute
        if item.capability == "analysis_unit" and _effective_unit(_constraints(context)) != replacement:
            missing.append(item.capability)
        if (item.capability == "analysis_unit" and "database_capability_binding" in _constraints(context)
                and not ({item.source_field.column, "stay_id"} & set(context.cohort.id_columns))):
            missing.append(item.capability)
        if item.capability == "outcome_meaning" and (
            "database_capability_binding" in _constraints(context)
            or "event_time" in family_requirements(decision.requirements.family)
        ):
            targets = [context.target_outcome, *(requirement.outcome for step in plan.steps
                        for requirement in step.model_requirements)]
            targets.extend(step.exposure_outcome_distribution_spec.outcome for step in plan.steps
                           if step.exposure_outcome_distribution_spec is not None)
            expected = "death" if replacement == "hospital_death" else replacement
            if not targets[0] or any(target != expected for target in targets):
                missing.append(item.capability)
        if isinstance(replacement, ThresholdRequirement):
            applied = (replacement.concept, replacement.operator, replacement.value) in coordinates
            original = item.original
            retained = isinstance(original, ThresholdRequirement) and (original.concept, original.operator, original.value) in coordinates
            if not applied or retained:
                missing.append(item.capability)
    from .research_substitutions import source_model_term_errors

    return tuple(missing) + source_model_term_errors(context, plan)
