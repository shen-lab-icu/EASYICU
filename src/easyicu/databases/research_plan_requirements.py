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


def _time_origin_applied_for_supported_association(context, plan, replacement, runtime_authority=None) -> bool:
    """Recognize one explicit static association; all other designs stay blocked.

    The source export is reviewed separately and the resulting cohort receipt
    is checked again before executing. Never equate this origin to ICU-only.
    """
    from easyicu.research_agent.authority.baseline_association_runtime import BaselineAssociationRuntimeAuthority
    from easyicu.research_agent.authority.case_duration_landmark import CaseDurationLandmarkAuthority

    if isinstance(runtime_authority, CaseDurationLandmarkAuthority):
        from .research_substitutions import case_duration_selection, time_origin_selection

        try:
            runtime_authority.validate_plan(plan)
            origin = time_origin_selection(context)
            outcome = case_duration_selection(context)
            if origin is None or outcome is None:
                return False
            runtime_authority.validate_source_selections(origin, outcome)
            windows = tuple(context.time_windows or ())
            exposure = next((v for v in context.variables if v.name == context.primary_exposure), None)
            return bool(
                len(windows) == 1 and windows[0].anchor == replacement
                and windows[0].start_hours == 0 and windows[0].end_hours == 24
                and context.target_outcome == outcome.proposed_outcome
                and context.primary_exposure == runtime_authority.exposure_column
                and exposure is not None and exposure.analysis_window == windows[0].name
                and exposure.analysis_window_role == "exposure_definition"
            )
        except (ValueError, RuntimeError):
            return False

    if (plan.analysis_type not in {"association", "association_study"}
            or not isinstance(runtime_authority, BaselineAssociationRuntimeAuthority)):
        return False
    try:
        runtime_authority.validate_plan(plan)
        step = runtime_authority.governed_primary_step(plan)
    except (ValueError, RuntimeError):
        return False
    # A method name is also used by generic code generation. Only the signed
    # owner may be considered; a structural claim still needs the existing
    # runtime-authority verifier to accept the complete signed plan.
    from easyicu.research_agent.contracts.association_execution import baseline_association_execution_verdict

    if (step.method != "adjusted_association_models"
            or step.scientific_capability != "association_baseline_adjusted_v1"
            or not baseline_association_execution_verdict(step).claimed
            or not step.model_requirements):
        return False
    if context.target_outcome in {None, "los_icu"} or plan.cohort is None:
        return False
    windows = tuple(context.time_windows or ())
    if len(windows) != 1 or windows[0].anchor != replacement:
        return False
    exposure = next((v for v in context.variables if v.name == context.primary_exposure), None)
    if (exposure is None or exposure.analysis_window != windows[0].name
            or exposure.analysis_window_role != "exposure_definition"):
        return False
    if any(req.exposure_source != context.primary_exposure or req.outcome != context.target_outcome
           or req.outcome_type != "binary" for req in step.model_requirements):
        return False
    # The signed cohort step has not yet been taught to propagate this clock
    # into filtering. This first slice accepts the full universe only.
    if plan.cohort.inclusion or plan.cohort.exclusion or plan.cohort.unapplied_population_criteria:
        return False
    for endpoint in (context.endpoint, plan.endpoint):
        if endpoint is not None and (endpoint.kind != "binary" or endpoint.name != context.target_outcome
                                     or endpoint.time_origin is not None):
            return False
    return True


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
    if any(key in constraints for key in (
        "database_capability_binding", "database_time_origin_selection", "database_case_duration_selection",
    )):
        from .research_substitutions import case_duration_selection, source_decision, time_origin_selection

        return source_decision(profile, requirements, time_origin=time_origin_selection(context),
                               case_duration=case_duration_selection(context))
    return assess_study(profile, requirements)


def unapplied_substitution_capabilities(decision: CapabilityDecision, context: ResearchContext,
                                        plan: AnalysisPlan, *, runtime_authority=None) -> tuple[str, ...]:
    """Approval cannot make unchanged predicates or grouping implement a proposal."""
    predicates = (*plan.cohort.inclusion, *plan.cohort.exclusion) if plan.cohort else ()
    coordinates = {(item.concept_id, item.op, item.value) for item in predicates
                   if item.concept_id in {"age", "weight"} and isinstance(item.value, (int, float))}
    missing = []
    for item in decision.substitutions:
        replacement = item.substitute
        if item.capability == "time_zero":
            if not _time_origin_applied_for_supported_association(context, plan, replacement, runtime_authority):
                missing.append("time_zero")
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
