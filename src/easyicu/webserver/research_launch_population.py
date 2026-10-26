"""Bind a reviewed population mode to what a verified export already applied."""
from __future__ import annotations

import json
import math
from typing import Any, Mapping


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
        elif item.capability == "outcome_meaning" and item.substitute == "icu_death":
            if original is not None and (study.get("execution_concepts") or {}).get("outcome") != "icu_death":
                raise ValueError("capability_substitution_binding_drift")
            result["execution_concepts"] = {**(study.get("execution_concepts") or {}), "outcome": "icu_death"}
            result["outcome"] = "icu_death"
    result["analysis_design"] = design
    return result


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
                    or window.end_offset_hours != math.inf
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
