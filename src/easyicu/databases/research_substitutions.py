"""Compile source proposals for whole-plan review; grant no execution authority.

The original requirements and the exact proposed changes remain in context.
The existing scientific-review companion and approval gate bind the resulting
context and plan before Execute. Restoring a sealed context never rewrites it.
"""
from __future__ import annotations

from dataclasses import replace
import json

from .profiles import get_database_profile
from .research_capabilities import research_capability_profile
from .research_plan_requirements import _constraints, plan_capability_decision
from .research_requirements import CapabilityDecision, StudyRequirements, ThresholdRequirement, assess_study

BINDING_FIELD = "database_capability_binding"


def registered_source_profile(database):
    """Only an unregistered cohort is unassessed; metadata errors still surface."""
    try:
        get_database_profile("miiv" if database == "miiv_demo" else database)
    except KeyError:
        return None
    return research_capability_profile(database)


def source_decision(profile, requirements):
    """Select a named native endpoint for a new executable source proposal.

    The base assessment continues to describe generic dictionary ``death``.
    Refused studies cannot be repaired by choosing a different endpoint.
    """
    decision = assess_study(profile, requirements)
    if decision.status == "refusal":
        return decision
    if any(item.capability == "outcome_meaning" for item in decision.substitutions):
        native = next((item for item in profile.mortality_outcomes if item.meaning == "icu_death"), None)
        if native is not None:
            return assess_study(profile.model_copy(update={"mortality": native}), requirements)
    return decision


def _binding_payload(context, profile, decision):
    effective_unit = decision.requirements.analysis_unit
    target = decision.requirements.outcome
    target = "death" if target == "hospital_death" else target or context.target_outcome
    for item in decision.substitutions:
        if item.capability == "analysis_unit":
            effective_unit = item.substitute
        elif item.capability == "outcome_meaning":
            target = item.substitute
    identity = next((item.column for item in profile.units if item.unit == effective_unit), None)
    bands = {}
    for variable in context.variables:
        concept = variable.source_concept or variable.name
        numeric = getattr(profile, concept, None) if concept in {"age", "weight"} else None
        if numeric is not None and numeric.kind == "binned":
            bands[variable.name] = {"labels": list(numeric.bin_labels), "values": list(numeric.bin_values),
                                   "coding": "categorical", "ordered": True}
    mortality = next((item for item in profile.mortality_outcomes if item.meaning == target), None)
    return {"decision": decision.model_dump(mode="json"), "effective_analysis_unit": effective_unit,
            "identifier": identity, "outcome_meaning": target, "binned_covariates": bands,
            "mortality_timing": mortality.model_dump(mode="json") if mortality else None,
            "execution_authority": "scientific_plan_review_and_approved_executable_plan",
            "planning_rule": "source_coordinates_require_whole_plan_approval"}


def source_binding(context):
    """Revalidate the host proposal against the current source facts."""
    constraints = _constraints(context)
    binding = constraints.get(BINDING_FIELD)
    if binding is None:
        if BINDING_FIELD in context.cohort.provenance:
            raise ValueError("capability_substitution_binding_drift")
        return None
    requirements = StudyRequirements.model_validate(constraints["database_study_requirements"])
    profile = registered_source_profile(context.cohort.database)
    if profile is None:
        raise ValueError("capability_substitution_binding_drift")
    decision = source_decision(profile, requirements)
    if binding != _binding_payload(context, profile, decision):
        raise ValueError("capability_substitution_binding_drift")
    return binding


def bind_source_substitutions(context, *, restoring=False):
    """Prepare an inspectable candidate on existing, never invented columns."""
    from easyicu.research_agent.schema import AnalysisPlan

    source_binding(context)
    profile = registered_source_profile(context.cohort.database)
    if profile is None:
        return context
    constraints = dict(_constraints(context))
    original = constraints.get("database_study_requirements")
    if original is None:
        requirements = plan_capability_decision(context, AnalysisPlan(
            research_question=context.research_question, steps=[])).requirements
    else:
        requirements = StudyRequirements.model_validate(original)
    decision = source_decision(profile, requirements)
    # Refusals are still reported by the review/gate, never repaired here.
    if decision.status == "refusal":
        return context
    design = dict(constraints.get("analysis_design") or {})
    binding = _binding_payload(context, profile, decision)
    target, effective_unit, identity = (binding[key] for key in (
        "outcome_meaning", "effective_analysis_unit", "identifier"))
    if any(item.capability == "analysis_unit" for item in decision.substitutions):
        design["analysis_unit"] = effective_unit
        # A canonical stay_id is allowed only when an existing context already
        # declares it. The native source identifier remains explicit in binding.
        if identity not in context.cohort.id_columns and "stay_id" not in context.cohort.id_columns:
            raise ValueError("capability_substitution_identifier_unbound")
    if target != context.target_outcome and target not in {v.name for v in context.variables}:
        raise ValueError("capability_substitution_requires_extraction")
    variables = []
    bands = binding["binned_covariates"]
    for variable in context.variables:
        if variable.name in bands:
            # Existing numeric loader values denote bins, not measured years/kg.
            variable = variable.model_copy(update={"is_ordinal": True,
                "ordinal_levels": bands[variable.name]["values"]})
        variables.append(variable)
    if not decision.substitutions and not bands:
        return context
    constraints.update(database_study_requirements=requirements.model_dump(mode="json"))
    constraints[BINDING_FIELD] = binding
    if design:
        constraints["analysis_design"] = design
    notes = f"analysis_unit={effective_unit}; identifier={identity}; outcome_meaning={target}"
    prior_notes = context.cohort.notes or ""
    if notes in prior_notes.splitlines():
        notes = prior_notes
    elif prior_notes:
        notes = prior_notes + "\n" + notes
    provenance = {**context.cohort.provenance, BINDING_FIELD: binding}
    cohort = context.cohort.model_copy(update={"notes": notes, "provenance": provenance})
    updated = context.model_copy(update={"target_outcome": target, "variables": variables, "cohort": cohort,
        "user_preferences": context.user_preferences.model_copy(update={"data_constraints": json.dumps(constraints)})})
    if restoring and updated != context:
        raise ValueError("capability_substitution_binding_drift")
    return updated


def compile_source_predicate(context, predicate):
    """Apply only the exact threshold proposal carried to whole-plan review."""
    if predicate.concept_id not in {"age", "weight"} or predicate.op not in {">", ">=", "<", "<=", "==", "!="}:
        return predicate
    threshold = ThresholdRequirement(concept=predicate.concept_id, operator=predicate.op, value=predicate.value)
    binding = source_binding(context)
    if binding is not None:
        decision = CapabilityDecision.model_validate(binding["decision"])
        for item in decision.substitutions:
            if item.original == threshold and isinstance(item.substitute, ThresholdRequirement):
                return replace(predicate, op=item.substitute.operator, value=item.substitute.value)
    profile = registered_source_profile(context.cohort.database)
    if profile is None:
        return predicate
    decision = assess_study(profile, StudyRequirements(family="association_study", thresholds=(threshold,)))
    if decision.status != "exact":
        raise ValueError("numeric_threshold_not_resolvable")
    return predicate


def criterion_has_source_substitution(context, criterion):
    binding = source_binding(context)
    if binding is None:
        return False
    concept = "age" if criterion.kind == "age_years" else getattr(criterion, "concept", None)
    return any(item["capability"] == f"threshold:{concept}" for item in binding["decision"]["substitutions"])


def source_model_term_errors(context, plan):
    """A midpoint must not become a continuous adjustment value."""
    profile = registered_source_profile(context.cohort.database)
    if profile is None:
        return ()
    variables = {item.name: item for item in context.variables}
    invalid = []
    for step in plan.steps:
        for requirement in step.model_requirements:
            terms = {item.name: item for item in requirement.model_terms or ()}
            for name in requirement.covariates or ():
                variable = variables.get(name)
                concept = (variable.source_concept or variable.name) if variable else name
                numeric = getattr(profile, concept, None) if concept in {"age", "weight"} else None
                if numeric is None or numeric.kind != "binned":
                    continue
                term = terms.get(name)
                expected = [str(value) for value in numeric.bin_values]
                selected = set(term.levels or ()) if term else set()
                ordered = [value for value in expected if value in selected]
                if term is None or term.coding != "categorical" or len(ordered) < 2 or term.levels != ordered:
                    invalid.append(f"binned_covariate:{name}")
    return tuple(dict.fromkeys(invalid))


def source_plan_labels(context, labels):
    """Give the named endpoint a stable report label before approval hashes it."""
    return {**labels, "icu_death": "ICU death"} if context.target_outcome == "icu_death" else labels
