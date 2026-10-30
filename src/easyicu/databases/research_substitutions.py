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
from .research_requirements import CapabilityDecision, CapabilitySubstitution, StudyRequirements, ThresholdRequirement, assess_study, family_requirements

BINDING_FIELD = "database_capability_binding"
TIME_ORIGIN_FIELD = "database_time_origin_selection"
CASE_DURATION_FIELD = "database_case_duration_selection"


def time_origin_selection(context):
    """Validate a discussion candidate against the host's frozen source identity.

    This does not make the source, window, sidecar, or plan executable. The
    enclosing context and plan are separately digest-bound by scientific review.
    """
    from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from .research_requirements import TimeOriginSelection, review_only_time_origin

    constraints = _constraints(context)
    raw = constraints.get(TIME_ORIGIN_FIELD)
    if raw is None and TIME_ORIGIN_FIELD not in constraints:
        return None
    selection = TimeOriginSelection.model_validate(raw)
    identity = SourceIdentity.from_record(constraints.get("source_identity"))
    profile = registered_source_profile(context.cohort.database)
    proposal = review_only_time_origin(profile) if profile is not None else None
    if (identity is None or proposal is None
            or identity.database != context.cohort.database
            or selection.selected_origin != SIC_CARE_UNIT_ORIGIN
            or selection.source_field != proposal.source_field
            or selection.source_identity_sha256 != canonical_sha256(identity.to_dict())):
        raise ValueError("time_origin_selection_binding_drift")
    return selection


def case_duration_selection(context):
    """Validate a proposed endpoint against the same registered source axis."""
    from .research_requirements import CaseDurationSelection

    constraints = _constraints(context)
    if CASE_DURATION_FIELD not in constraints:
        return None
    candidate = CaseDurationSelection.model_validate(constraints[CASE_DURATION_FIELD])
    axis = time_origin_selection(context)
    profile = registered_source_profile(context.cohort.database)
    if (axis is None or profile is None or profile.case_close_proxy is None
            or candidate.source_identity_sha256 != axis.source_identity_sha256
            or candidate.start_field != axis.source_field
            or candidate.end_field != profile.case_close_proxy):
        raise ValueError("case_duration_selection_binding_drift")
    return candidate


def registered_source_profile(database):
    """Only an unregistered cohort is unassessed; metadata errors still surface."""
    try:
        get_database_profile("miiv" if database == "miiv_demo" else database)
    except KeyError:
        return None
    return research_capability_profile(database)


def source_decision(profile, requirements, *, time_origin=None, case_duration=None):
    """Select a named native endpoint for a new executable source proposal.

    Time-to-event alternatives belong to assess_study, including their order.
    Other families retain their existing native status proposal. Refusals
    cannot be repaired here by choosing a different endpoint.
    """
    effective_profile = profile
    if time_origin is not None:
        candidate = profile.care_unit_admission_proxy
        if candidate is None or candidate != time_origin.source_field:
            raise ValueError("time_origin_selection_binding_drift")
        effective_profile = profile.model_copy(update={"icu_admission": candidate})
    effective_requirements = requirements
    if case_duration is not None:
        if (profile.case_close_proxy is None or case_duration.end_field != profile.case_close_proxy
                or time_origin is None or case_duration.start_field != time_origin.source_field
                or requirements.outcome != case_duration.requested_outcome):
            raise ValueError("case_duration_selection_binding_drift")
        effective_requirements = requirements.model_copy(update={
            "outcome": case_duration.proposed_outcome,
            "concepts": tuple(case_duration.proposed_outcome if name == case_duration.requested_outcome else name
                              for name in requirements.concepts),
        })
    decision = assess_study(effective_profile, effective_requirements)
    if decision.status != "refusal" and "event_time" not in family_requirements(requirements.family):
        if any(item.capability == "outcome_meaning" for item in decision.substitutions):
            native = next((item for item in profile.mortality_outcomes if item.meaning == "icu_death"), None)
            if native is not None:
                decision = assess_study(effective_profile.model_copy(update={"mortality": native}), requirements)
    if time_origin is not None:
        change = CapabilitySubstitution(
            capability="time_zero", original=time_origin.requested_origin,
            substitute=time_origin.selected_origin, reason_code="care_unit_origin_not_icu_only",
            source_field=time_origin.source_field,
        )
        decision = decision.model_copy(update={
            "status": "refusal" if decision.missing else "substitution",
            "substitutions": (*decision.substitutions, change),
        })
    if case_duration is not None:
        change = CapabilitySubstitution(
            capability="outcome_meaning", original=case_duration.requested_outcome,
            substitute=case_duration.proposed_outcome, reason_code="case_close_is_not_icu_discharge",
            source_field=case_duration.end_field,
        )
        decision = decision.model_copy(update={
            "requirements": requirements, "status": "refusal" if decision.missing else "substitution",
            "substitutions": (*decision.substitutions, change),
        })
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
    approximations = {}
    for variable in context.variables:
        concept = variable.source_concept or variable.name
        numeric = getattr(profile, concept, None) if concept in {"age", "weight"} else None
        if numeric is not None and numeric.kind == "binned":
            bands[variable.name] = {"labels": list(numeric.bin_labels), "values": list(numeric.bin_values),
                                   "coding": "categorical", "ordered": True}
        elif numeric is not None and numeric.kind == "approximate":
            approximations[variable.name] = {
                "interpretation": numeric.field.interpretation if numeric.field else None,
                "coding": "unverified",
            }
    mortality = next((item for item in profile.mortality_outcomes if item.meaning == target), None)
    return {"decision": decision.model_dump(mode="json"), "effective_analysis_unit": effective_unit,
            "identifier": identity, "outcome_meaning": target, "binned_covariates": bands,
            **({"approximate_covariates": approximations} if approximations else {}),
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
    decision = source_decision(profile, requirements, time_origin=time_origin_selection(context),
                               case_duration=case_duration_selection(context))
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
    decision = source_decision(profile, requirements, time_origin=time_origin_selection(context),
                               case_duration=case_duration_selection(context))
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
    approximations = binding.get("approximate_covariates", {})
    for variable in context.variables:
        if variable.name in bands:
            # Existing numeric loader values denote bins, not measured years/kg.
            variable = variable.model_copy(update={"is_ordinal": True,
                "ordinal_levels": bands[variable.name]["values"]})
        elif variable.name in approximations:
            interpretation = approximations[variable.name].get("interpretation")
            if interpretation:
                previous = variable.description or ""
                description = interpretation if not previous else f"{previous}\n{interpretation}"
                if interpretation not in previous:
                    variable = variable.model_copy(update={"description": description})
        variables.append(variable)
    if not decision.substitutions and not bands and not binding.get("approximate_covariates"):
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


def approximate_variables_used(context, plan):
    """Find rounded variables consumed by the proposed plan, even if unbound."""
    profile = registered_source_profile(context.cohort.database)
    if profile is None:
        return ()
    from easyicu.research_agent.methods.descriptive_inputs import plan_value_columns

    names = {context.primary_exposure, context.target_outcome, *plan_value_columns(plan)}
    for step in plan.steps:
        names.update(step.inputs or ())
        for requirement in step.model_requirements:
            names.update((requirement.exposure_source, requirement.outcome))
            names.update(requirement.covariates or ())
            names.update(term.name for term in requirement.model_terms or ())
    variables = {item.name: item for item in context.variables}
    found = []
    for name in names:
        variable = variables.get(name)
        concept = (variable.source_concept or variable.name) if variable else name
        numeric = getattr(profile, concept, None) if concept in {"age", "weight"} else None
        if numeric is not None and numeric.kind == "approximate":
            found.append(name)
    return tuple(sorted(found))


def source_plan_labels(context, labels):
    """Give the named endpoint a stable report label before approval hashes it."""
    return {**labels, "icu_death": "ICU death"} if context.target_outcome == "icu_death" else labels
