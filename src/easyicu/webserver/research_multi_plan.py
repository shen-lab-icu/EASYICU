"""Recompile a reviewed design against target-owned metadata through db6a."""
from __future__ import annotations

import json

from easyicu.databases.research_plan_requirements import _constraints, plan_capability_decision, unapplied_substitution_capabilities
from easyicu.databases.research_substitutions import bind_source_substitutions, registered_source_profile, source_decision
from easyicu.research_agent.canonical_json import canonical_sha256
from .research_followup_compile import compile_new_study
from .research_multi_contracts import MultiDatabaseError
from .research_plan_revision import _scientific_binding, _step_coordinate


def scientific_design(plan):
    """Ignore display/transport identities, never methods or typed estimands."""
    wire = plan.model_dump(mode="json")
    steps = [_step_coordinate(step, wire) for step in wire.pop("steps")]
    for key in ("research_question", "display_labels", "revision"):
        wire.pop(key, None)
    return {"plan": _scientific_binding(wire), "steps": steps}


def require_same_design(actual, expected):
    if scientific_design(actual) != scientific_design(expected):
        raise MultiDatabaseError("multi_database_design_changed")


def source_requirements(source):
    decision = plan_capability_decision(source.context, source.plan)
    if decision is None:
        raise MultiDatabaseError("multi_database_design_unassessed")
    return decision.requirements


def target_decision(source, database):
    profile = registered_source_profile(database)
    if profile is None:
        raise MultiDatabaseError("source_profile_unavailable")
    return source_decision(profile, source_requirements(source))


def compile_same_study(source, target_context):
    """No source observations, cohort provenance or approval cross databases.

    Round-trip the source first: a family template that cannot express the
    reviewed design is a stop, not permission to drop a step or an estimand.
    Mortality event/censor definitions remain solely with db6a/db6i.
    """
    _, roundtrip = compile_new_study(source.context, family=source.plan.analysis_type, source_plan=source.plan)
    require_same_design(roundtrip, source.plan)
    decision = target_decision(source, target_context.cohort.database)
    if decision.status == "refusal":
        raise MultiDatabaseError("study_capability_refused")
    constraints = dict(_constraints(target_context))
    original = _constraints(source.context)
    # These are study declarations; intake/provenance stays target-owned.
    for key in ("analysis_design", "analysis_timing", "stated_subgroups", "cohort", "population_spec",
                "accepted_analysis_inputs", "exposure_groupings", "attached_skills"):
        if key in original:
            constraints[key] = original[key]
    constraints.pop("database_capability_binding", None)
    constraints["database_study_requirements"] = source_requirements(source).model_dump(mode="json")
    preferences = source.context.user_preferences.model_copy(deep=True, update={"data_constraints": json.dumps(constraints)})
    proposed = target_context.model_copy(deep=True, update={
        "research_question": source.context.research_question, "primary_exposure": source.context.primary_exposure,
        "target_outcome": source.context.target_outcome, "endpoint": source.context.endpoint,
        "user_preferences": preferences,
    })
    proposed = bind_source_substitutions(proposed)
    if proposed.target_outcome != source.context.target_outcome:
        if proposed.endpoint is None or proposed.endpoint.kind != "binary":
            raise MultiDatabaseError("multi_database_design_not_recompilable")
        proposed = proposed.model_copy(update={"endpoint": proposed.endpoint.model_copy(update={"name": proposed.target_outcome})})
    proposed, plan = compile_new_study(proposed, family=source.plan.analysis_type, source_plan=source.plan)
    if unapplied_substitution_capabilities(decision, proposed, plan):
        raise MultiDatabaseError("capability_substitution_not_applied")
    # A comparison identity deliberately retains adjusted definitions. Equal
    # labels alone never make different outcomes/populations comparable.
    return proposed, plan, decision


def estimand_identity(plan):
    return canonical_sha256(scientific_design(plan))
