"""Conservative completeness check for an explicitly adult study population.

This check never proposes an age cut-off or changes a population. It only
detects a narrow, unambiguous adult-only wording and asks the existing
population owners to account for it. Complex comparative and negative
wording stays with the planner and scientific review.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from ..research_context.concept_population import context_data_constraints
from .population_compile import _read_sources
from .population_spec import AgeYears, NotTyped, PopulationSpec

_AMBIGUOUS = re.compile(
    r"儿童|小儿|儿科|未成年|青少年|婴儿|新生儿|家长|父母|非成人|非成年|排除成人|排除成年|"
    r"不含成人|不包括成人|不限成人|不限于成人|并非仅|不局限于成人|比较|对比|其他患者|其他人群|成人与|成人和|成人及|成人对照|成人相比|"
    r"\b(?:child(?:ren)?|pediatric|paediatric|infants?|neonat\w*|"
    r"adolescen\w*|minors?|parents?|guardians?|non[ -]?adults?|not\s+adults?|"
    r"exclud(?:e|es|ed|ing)\s+adults?|without\s+adults?|not\s+only|"
    r"not\s+limited\s+to|does\s+not\s+include|other\s+(?:patients|participants|subjects)|versus|vs\.?|compar(?:e|ed|ing))\b",
    re.IGNORECASE,
)
_ADULT_POPULATION = re.compile(
    r"成人(?:患者|病人|病例|人群)|成年(?:患者|病人|病例|人群)|(?:(?<=研究)|(?<=纳入)|(?<=招募)|(?<=入组))\s*(?:成年人|成人)|"
    r"\badults?\s+(?:icu\s+)?(?:patients?|participants?|subjects?|population)\b|"
    r"\b(?:among|in|of|include|enrol(?:l)?|recruit)\s+adults\b",
    re.IGNORECASE,
)


def explicit_adult_population(question: str) -> str | None:
    """Return the original words only when the whole-population reading is clear."""
    text = str(question or "")
    if _AMBIGUOUS.search(text):
        return None
    match = _ADULT_POPULATION.search(text)
    return match.group(0).strip() if match else None


def _adult_lower_bound(value: Any) -> bool:
    # This is a validation threshold for an existing proposed bound, not a
    # source of a new eligibility decision. The plan still needs review.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 18


def _source_has_adult_bound(context: Any) -> bool:
    """Only source selections already accepted by the population owner count."""
    sources = _read_sources(context)
    if sources.executed is not None and _adult_lower_bound(sources.executed.get("age_min")):
        return True
    return any(
        item.stage == "export" and item.criterion == "age"
        and _adult_lower_bound(item.parameters.get("age_min"))
        for item in sources.steps
    )


def adult_population_accounted_for(context: Any, cohort: Any, spec: PopulationSpec | None) -> bool:
    """Check representation, leaving applicability to existing compile/review gates."""
    if _source_has_adult_bound(context):
        return True
    if spec is not None:
        return any(
            isinstance(item, AgeYears) and _adult_lower_bound(item.min_years)
            or isinstance(item, NotTyped) and re.search(r"成人|成年|\badults?\b", item.quote, re.IGNORECASE) is not None
            for item in spec.criteria
        )
    # Legacy typed predicates are validated by the compiler immediately before
    # this check; a cohort label or an age covariate alone never filters rows.
    return any(
        item.concept_id in {"age", "age_years"} and item.op in {">=", ">"}
        and _adult_lower_bound(item.value.materialize())
        for item in cohort.inclusion
    ) or any(
        item.concept_id in {"age", "age_years"} and item.op == "<"
        and _adult_lower_bound(item.value.materialize())
        for item in cohort.exclusion
    )


def adult_age_source_unverified(context: Any) -> bool:
    """A rounded source age is not proof of an exact adult boundary.

    Capability bindings are keyed by the physical variable name, not the
    source concept. Also check the source profile because an already-filtered
    source need not carry an age column into the planning context at all.
    """
    provenance = context.cohort.provenance
    constraints = context_data_constraints(context)
    bindings = (provenance.get("database_capability_binding"), constraints.get("database_capability_binding"))
    age_names = {"age", "age_years"}
    age_names.update(
        variable.name for variable in context.variables
        if str(variable.source_concept or "").lower() == "age"
    )
    for binding in bindings:
        approximate = binding.get("approximate_covariates") if isinstance(binding, Mapping) else None
        if isinstance(approximate, Mapping) and any(name in approximate for name in age_names):
            return True
    from easyicu.databases.research_substitutions import registered_source_profile

    profile = registered_source_profile(context.cohort.database)
    return profile is not None and profile.age.kind in {"approximate", "unavailable"}
