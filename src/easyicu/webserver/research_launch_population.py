"""Bind a reviewed population mode to what a verified export already applied."""
from __future__ import annotations

import json
import math
from typing import Any, Mapping



def stated_population_mode(study: Mapping[str, Any]) -> str | None:
    """The population mode the saved study states for rows a Planner selects.

    Read only here; every stage then asks ``authorized_population_mode``.
    """
    from easyicu.webserver import primary_cohort

    return primary_cohort.planning_selection_mode(primary_cohort.planner_selectable_cohort(study.get("cohort")))

def question_population_mode(required_mode: str | None, *, study: Mapping[str, Any], question: str) -> str | None:
    """Leave the population open when the question restricts it to a recorded condition.

    A study bound to an export whose saved cohort states no criterion runs as
    every input row, and that mode forbids the Planner from stating any
    population criterion. A question such as 「在脓毒症 ICU 患者中」 names a
    condition EasyICU records, which the plan must select on
    (question_requirements, S1722), so every plan went back with no way to
    answer (UX leg 3 on rc_78a, 2026-11-20). Such a launch now leaves the mode
    to the Planner, as a metadata-only launch does. A researcher's recorded
    eligibility decision stands; nothing here selects or approves a criterion.
    """
    if required_mode != "all_input_rows":
        return required_mode
    authority = study.get("cohort_eligibility_authority")
    if isinstance(authority, Mapping) and authority.get("selection_state") in {"confirmed", "declined"}:
        return required_mode
    from .study_intent import stated_population_conditions

    return None if stated_population_conditions(question) else required_mode


def carried_population_mode(required_mode: str | None, *, candidate_definition: Mapping[str, Any] | None) -> str | None:
    """Leave the population open when the candidate carries a criterion it could not apply.

    A reviewed candidate whose stated criterion was not applied (and whose
    omission the researcher accepted) has every input row as its cohort.  The
    execution Planner then re-plans on the prepared package; binding it to
    every input row forbids it from stating any criterion, so the accepted
    omission vanished from the executed plan and its report (UX on rc_512,
    「成人 ICU 患者」, S1747).  The mode is left to the Planner, which restates the
    criterion: applied when the package can apply it, otherwise not applied
    again and shown.  Nothing here selects or approves a criterion.
    """
    if required_mode != "all_input_rows" or not isinstance(candidate_definition, Mapping):
        return required_mode
    unapplied = candidate_definition.get("unapplied_population_criteria")
    return None if isinstance(unapplied, (list, tuple)) and any(unapplied) else required_mode


def resolved_launch_population_mode(
    *, required_mode: str | None, question: str, database: str,
    acquisition: Any, declarations: Mapping[str, Any],
) -> str | None:
    """Resolve the saved mode against the same context the pipeline will build.

    The input is the acquired universe (a zero-row catalog during planning),
    so the context builder reads the host's verified grouping declarations.
    No study prose is reinterpreted here and no plan is approved.
    """
    if required_mode != "predicate_filtered":
        return required_mode
    from easyicu.research_agent.planning.cohort_predicate_domain import (
        authorized_population_mode, question_asks_population_share,
    )
    if not question_asks_population_share(question):
        return required_mode
    from easyicu.research_agent.research_context.builder import build_research_context

    context = build_research_context(
        research_question=question, database=database,
        cohort=acquisition.universe_path, **declarations,
    )
    return authorized_population_mode(required_mode, context)



def sealed_candidate_population_mode(
    *, inner_run: Any, capsule: Mapping[str, Any], stated_mode: str | None,
) -> str | None:
    """Resolve the stated mode with the candidate's own sealed planning context.

    The launch bound the candidate's mode through the same host decision
    (``resolved_launch_population_mode``); materialization must not re-derive
    it from the saved study alone. The context is read only when its bytes
    match the digest the verified input capsule sealed.
    """
    if stated_mode != "predicate_filtered":
        return stated_mode
    import hashlib
    from pathlib import Path

    from easyicu.research_agent.planning.cohort_predicate_domain import authorized_population_mode
    from easyicu.research_agent.research_context.typed import parse_research_context_json

    relative = str(capsule.get("context_relative_path") or "")
    path = Path(inner_run) / relative
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        return stated_mode
    try:
        raw = path.read_bytes()
    except OSError:
        return stated_mode
    if hashlib.sha256(raw).hexdigest() != str(capsule.get("context_sha256") or ""):
        return stated_mode
    # A candidate run may write its typed context (V2/V3); only the versioned owner reads every version (B34).
    context = parse_research_context_json(raw)
    return authorized_population_mode(stated_mode, context)

def prepare_source_population(study: Mapping[str, Any]) -> dict[str, Any]:
    """Project a source proposal for preparation; preserve its original request.

    This confers no execution authority. Scientific review, the human decision
    and the approved executable plan must bind the resulting context and plan.
    """
    from easyicu.databases.research_requirements import StudyRequirements
    from easyicu.databases.research_substitutions import registered_source_profile, source_decision
    from .research_capability_scope import study_requirements

    result = dict(study)
    design = dict(study.get("analysis_design") or {})
    database = (study.get("data_source") or {}).get("database")
    family = design.get("analysis_family")
    if not database or not family:
        return result
    profile = registered_source_profile(database)
    if profile is None:
        if "_database_study_requirements" in study:
            raise ValueError("capability_substitution_binding_drift")
        return result
    original = study.get("_database_study_requirements")
    requirements = (StudyRequirements.model_validate(original) if original is not None else
                    study_requirements(study, family=family, question=str(study.get("question") or "")))
    decision = source_decision(profile, requirements)
    if original is not None and study.get("_database_capability_decision") != decision.model_dump(mode="json"):
        raise ValueError("capability_substitution_binding_drift")
    if decision.status != "substitution":
        return result
    result["_database_study_requirements"] = requirements.model_dump(mode="json")
    result["_database_capability_decision"] = decision.model_dump(mode="json")
    for item in decision.substitutions:
        if item.capability == "analysis_unit":
            if original is not None and design.get("analysis_unit") != item.substitute:
                raise ValueError("capability_substitution_binding_drift")
            design["analysis_unit"] = item.substitute
        elif item.capability == "outcome_meaning":
            if original is not None and (study.get("execution_concepts") or {}).get("outcome") != item.substitute:
                raise ValueError("capability_substitution_binding_drift")
            result["execution_concepts"] = {**(study.get("execution_concepts") or {}), "outcome": item.substitute}
            result["outcome"] = item.substitute
    result["analysis_design"] = design
    return result


def concept_has_one_value_per_stay(concept_id: str) -> bool:
    """Whether the concept registry declares one value per stay (an ID table).

    Such a concept has no time column, so the cohort materializer reads its one
    value per stay and never summarizes it over a predicate's window
    (``_predicate_column``). An unreadable registry declares nothing.
    """
    try:
        from easyicu.concept.loader import load_dictionary

        definition = load_dictionary().get(str(concept_id))
    except Exception:  # noqa: BLE001 - an unreadable registry declares nothing
        return False
    return getattr(definition, "target", None) == "id_tbl"


def prepared_candidate_selection_mode(
    *, candidate_definition: Mapping[str, Any] | None, required_mode: str,
    source_basis: str | None, source_report: Mapping[str, Any] | None, database: str,
) -> str:
    """Keep the guard strict, projecting only source-proven inclusive age bounds.

    A zero-row candidate may need an age predicate that a later export has
    already executed. Reuse the population compiler's source proof, rather
    than require a second row filter solely to preserve the old mode label.
    Every other predicate, exclusion, window or unverified source retains its
    original constraint. This grants neither analysis nor plan approval.
    """
    if (required_mode != "predicate_filtered" or candidate_definition is None
            or source_basis != "export_contract" or not isinstance(source_report, Mapping)):
        return required_mode
    reported_rows = source_report.get("selected")
    if isinstance(reported_rows, bool) or not isinstance(reported_rows, int) or reported_rows <= 0:
        return required_mode
    from easyicu.research_agent.cohort.schema import coerce_cohort_definition
    from easyicu.research_agent.planning.population_compile import compile_population
    from easyicu.research_agent.planning.population_spec import PopulationSpec
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext, UserPreferences

    try:
        definition = coerce_cohort_definition(candidate_definition)
        if (definition is None or definition.selection_mode != required_mode
                or not definition.inclusion or definition.exclusion
                or definition.derived_from_named
                or definition.unapplied_population_criteria):
            return required_mode
        low = high = None
        for predicate in definition.inclusion:
            window = predicate.time_window
            if (predicate.concept_id != "age" or predicate.aggregation != "first"
                    or predicate.op not in {">=", "<="}
                    or window.anchor != "icu_admission" or window.start_offset_hours != 0
                    # A window that does not run to the end of the stay names
                    # the same value only for a concept the registry declares
                    # one value per stay (B22).
                    or (window.end_offset_hours != math.inf
                        and not (window.end_offset_hours > 0
                                 and concept_has_one_value_per_stay(predicate.concept_id)))
                    or isinstance(predicate.value, bool)
                    or not isinstance(predicate.value, (int, float))
                    or not math.isfinite(predicate.value)):
                return required_mode
            if predicate.op == ">=":
                low = predicate.value if low is None else max(low, predicate.value)
            else:
                high = predicate.value if high is None else min(high, predicate.value)
        # Implication alone is insufficient here: a stricter bound, another
        # filter or a cap would establish a different candidate denominator.
        steps = source_report.get("demographic_steps")
        if (source_report.get("count_unit") != "icu_stay"
                or source_report.get("mode") != "all_icu"
                or source_report.get("applied_filters") != ["demographics"]
                or not isinstance(steps, list) or len(steps) != 1
                or not isinstance(steps[0], Mapping) or steps[0].get("criterion") != "age"
                or source_report.get("concept_matches") is not None
                or source_report.get("max_patients_applied") is True):
            return required_mode
        parameters = steps[0].get("parameters")
        missing = steps[0].get("n_excluded_missing")
        icd, cap = source_report.get("icd"), source_report.get("cap")
        if (not isinstance(parameters, Mapping)
                or set(parameters) - {"age_min", "age_max"}
                or parameters.get("age_min") != low or parameters.get("age_max") != high
                or isinstance(missing, bool) or not isinstance(missing, int) or missing < 0
                or (isinstance(icd, Mapping) and icd.get("enabled") is True)
                or (isinstance(cap, Mapping) and cap.get("cut") is True)):
            return required_mode
        spec = PopulationSpec.model_validate({"criteria": [{
            "id": "c1", "kind": "age_years", "role": "include",
            "source": "preset", "quote": "Reviewed candidate age eligibility",
            "min_years": low, "max_years": high,
        }]})
        # This parent-export metadata projection reads no patient rows. Its
        # count and filters come from the report the launch owner verified.
        # The compiler also verifies that every stay count chains to selected;
        # excluded missing ages are a subset of the exclusions, not extra rows.
        context = ResearchContext(
            research_question="Reviewed candidate age eligibility",
            cohort=CohortDescriptor(cohort_name="export_metadata", database=database, n_stays=reported_rows),
            variables=[], user_preferences=UserPreferences(data_constraints=json.dumps({
                "source_selection": {"basis": source_basis, "export_report": dict(source_report)},
            })),
        )
        compiled = compile_population(spec, context)
        if (not compiled.blocking and compiled.criteria
                and all(item.disposition == "applied_by_source" for item in compiled.criteria)):
            return "all_input_rows"
    except (TypeError, ValueError):
        # An unreadable record cannot relax the existing population guard.
        pass
    return required_mode
