"""Codes-only study scope projection shared by launch, workflow and trial setup."""
from __future__ import annotations

from collections.abc import Mapping

from easyicu.databases.profiles import get_database_profile, public_database_keys
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import (
    StudyRequirements, ThresholdRequirement, assess_study,
)


def study_requirements(study: Mapping, *, family: str, question: str = "") -> StudyRequirements:
    from .study_intent import explicit_outcome_concepts
    from .mention_setup import mentioned_concepts
    from .study_scientific_configuration import ScientificConfiguration

    configuration = ScientificConfiguration.inspect(study)
    outcome = configuration.executable_target_outcome()
    named = explicit_outcome_concepts(question) if not outcome else ()
    concepts = tuple(dict.fromkeys(filter(None, (
        outcome, configuration.executable_primary_exposure(), *configuration.covariates(), *named, *mentioned_concepts(study),
    ))))
    design = study.get("analysis_design") or {}
    unit = design.get("analysis_unit", "icu_stay")
    unit = {"hospital_admission": "hospitalization", "row": "icu_stay"}.get(unit, unit)
    thresholds = []
    cohort = study.get("cohort") or {}
    from .primary_cohort import ADULT_AGE_FLOOR_YEARS, ADULT_COHORT_PRESETS

    for field, operator in (("age_min", ">="), ("age_max", "<=")):
        value = cohort.get(field)
        if field == "age_min" and cohort.get("preset") in ADULT_COHORT_PRESETS:
            value = max(ADULT_AGE_FLOOR_YEARS, float(value or 0))
        if value is not None and 0 < float(value):
            thresholds.append(ThresholdRequirement(concept="age", operator=operator, value=value))
    population = study.get("population_spec") or {}
    for criterion in population.get("criteria", ()):
        if criterion.get("kind") == "age_years":
            for field, operator in (("min_years", ">="), ("max_years", "<=")):
                if criterion.get(field) is not None:
                    thresholds.append(ThresholdRequirement(concept="age", operator=operator, value=criterion[field]))
        if criterion.get("kind") == "measurement" and criterion.get("concept") in {"age", "weight"}:
            thresholds.append(ThresholdRequirement(concept=criterion["concept"], operator=criterion["op"], value=criterion["value"]))
    return StudyRequirements(family=family, analysis_unit=unit,
                             outcome="hospital_death" if "death" in concepts else outcome or (named[0] if len(named) == 1 else None),
                             concepts=concepts, thresholds=tuple(thresholds),
                             censoring_unit="hospitalization" if "death" in concepts else None)


def eligible_family_databases(family: str, *, requirements: StudyRequirements | None = None) -> tuple[str, ...]:
    # A family-only suggestion probes default mortality and its disclosed
    # alternatives. A supplied study keeps its named endpoint and requirements.
    requirements = requirements or StudyRequirements(family=family, outcome="hospital_death")
    eligible = []
    for key in public_database_keys():
        try:
            profile = research_capability_profile(key)
        except KeyError:
            continue
        if assess_study(profile, requirements).status != "refusal":
            eligible.append(key)
    return tuple(eligible)


def capability_scope_gap(study: Mapping, *, family: str, database: str,
                         code: str, question: str = "") -> dict | None:
    try:
        get_database_profile("miiv" if database == "miiv_demo" else database)
    except KeyError:
        return None
    from .study_scientific_configuration import ScientificConfigurationError

    try:
        requirements = study_requirements(study, family=family, question=question)
    except ScientificConfigurationError:
        # The launch owner validates concept selection before the adjustment
        # roster. A scope probe must not reorder those existing refusals.
        return None
    try:
        profile = research_capability_profile(database)
    except KeyError:
        missing = [{"capability": "source_profile", "code": "source_profile_unavailable", "source": "data-sources.json"}]
    else:
        decision = assess_study(profile, requirements)
        if decision.status != "refusal":
            return None
        missing = [item.model_dump(mode="json") for item in decision.missing]
    return {"code": code, "database": database, "family": family,
            "missing_capabilities": missing,
            "supported_databases": list(eligible_family_databases(family, requirements=requirements))}
