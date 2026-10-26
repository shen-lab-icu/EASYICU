"""Compare declared scientific requirements with evidenced source capabilities.

This is a proposal compiler, not a fallback executor. A substitution never
changes a population, endpoint or threshold here. The review/approval boundary
must bind the complete proposal before an execution owner may consume it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from easyicu.databases.research_capabilities import ConceptAvailability, Fact, ResearchCapabilityProfile, SourceField


class ThresholdRequirement(Fact):
    concept: Literal["age", "weight"]
    operator: Literal[">", ">=", "<", "<=", "==", "!="]
    value: float = Field(allow_inf_nan=False)


class StudyRequirements(Fact):
    family: str
    analysis_unit: str = "icu_stay"
    outcome: str | None = None
    concepts: tuple[str, ...] = ()
    thresholds: tuple[ThresholdRequirement, ...] = ()
    treatment_concepts: tuple[str, ...] = ()
    censoring_unit: Literal["hospitalization", "icu_stay"] | None = None


class CapabilitySubstitution(Fact):
    capability: str
    original: str | ThresholdRequirement
    substitute: str | ThresholdRequirement
    reason_code: str
    source_field: SourceField


class MissingCapability(Fact):
    capability: str
    code: str
    source: str


class CapabilityDecision(Fact):
    status: Literal["exact", "substitution", "refusal"]
    database: str
    requirements: StudyRequirements
    substitutions: tuple[CapabilitySubstitution, ...] = ()
    missing: tuple[MissingCapability, ...] = ()
    unassessed_concepts: tuple[str, ...] = ()
    unassessed_concept_details: tuple[ConceptAvailability, ...] = ()


def family_requirements(family: str) -> tuple[str, ...]:
    """Project existing family contracts; unknown families have no permission.

    EndpointSpec's time-to-event closure owns the event/censor requirement.
    TargetTrialSpec owns the ICU time zero and timed treatment strategies.
    The trajectory family's registered guardrail requires aligned time axes.
    These are necessary source facts, never a claim of end-to-end readiness.
    """
    from easyicu.research_agent.planning.analysis_types import canonical_analysis_family, get_analysis_type

    spec = get_analysis_type(canonical_analysis_family(family) or family)
    requirements = ["analysis_unit"]
    if spec.required_endpoint_kind == "time_to_event":
        requirements.extend(("time_zero", "event_time", "censor_time"))
    if spec.key == "causal_inference":
        requirements.extend(("time_zero", "treatment_timing"))
    if spec.key in {"trajectory_clustering", "dynamic_prediction"}:
        requirements.extend(("time_zero", "observation_timing"))
    return tuple(requirements)


def assess_study(profile: ResearchCapabilityProfile, requirements: StudyRequirements) -> CapabilityDecision:
    """Return exactly one disposition; missing facts dominate proposals.

    Prefer the named native mortality, including its stated ICU-discharge
    proxy, over the default mortality's different meaning or time. Only an
    absent native endpoint falls back to the default with an outcome-meaning
    substitution. An untimed native endpoint is refused, not silently changed
    to a different timed endpoint; censoring requirements remain independent.
    """
    missing: list[MissingCapability] = []
    substitutions: list[CapabilitySubstitution] = []
    unassessed_concepts: list[ConceptAvailability] = []

    def absent(capability: str, code: str, source: str) -> None:
        missing.append(MissingCapability(capability=capability, code=code, source=source))

    try:
        required = family_requirements(requirements.family)
    except KeyError:
        required = ()
        absent("family_requirements", "family_requirements_missing", "easyicu.research_agent.planning.analysis_types")
    unit = requirements.analysis_unit
    if unit not in profile.analysis_units:
        if unit == "hospitalization" and "icu_stay" in profile.analysis_units:
            substitute_unit = next(item for item in profile.units if item.unit == "icu_stay")
            substitutions.append(CapabilitySubstitution(
                capability="analysis_unit", original=unit, substitute="icu_stay",
                reason_code="hospitalization_identifier_unavailable", source_field=SourceField(
                    source=substitute_unit.source, table=substitute_unit.table, column=substitute_unit.column),
            ))
        else:
            absent("analysis_unit", "analysis_unit_unavailable", "data-sources.json#/id_cfg")
    mortality = next((item for item in profile.mortality_outcomes if item.meaning == requirements.outcome), profile.mortality)
    if requirements.outcome in {"hospital_death", "icu_death", "death_within_72h_after_icu_discharge"}:
        if mortality is None:
            absent("mortality_status", "mortality_status_unavailable", "concept-dict.json#/death")
        elif mortality.meaning != requirements.outcome:
            substitutions.append(CapabilitySubstitution(
                capability="outcome_meaning", original=requirements.outcome, substitute=mortality.meaning,
                reason_code="mortality_meaning_differs", source_field=mortality.status,
            ))
    concepts = list(requirements.concepts)
    if requirements.outcome and requirements.outcome not in {"hospital_death", "icu_death", "death_within_72h_after_icu_discharge"}:
        concepts.append(requirements.outcome)
    for concept in dict.fromkeys(concepts):
        item = profile.concept(concept)
        if concept == "los_hosp" and not profile.hospital_los:
            absent("hospital_los", "hospital_los_unavailable", item.source)
        elif concept == "los_icu" and not profile.icu_los:
            absent("icu_los", "icu_los_unavailable", item.source)
        elif not item.defined or item.derivation_status == "degraded":
            # Export columns and user-defined derivations belong to extraction;
            # absence from our vocabulary does not prove a source lacks data.
            # Partial recipes likewise need their existing runtime assessment;
            # neither declare them exact nor mistake them for absent mappings.
            unassessed_concepts.append(item)
        elif not item.mapped:
            absent(f"concept:{concept}", "required_concept_unmapped", item.source)
    from easyicu.outcome_availability import fixed_horizon_mortality_endpoint

    endpoint = fixed_horizon_mortality_endpoint(requirements.outcome or "")
    fixed_followup = endpoint is not None and profile.concept(endpoint.followup_concept).mapped
    if "time_zero" in required and profile.icu_admission is None:
        absent("time_zero", "time_zero_unavailable", profile.time_source)
    if "event_time" in required and not fixed_followup:
        if mortality is None or mortality.event_time is None:
            absent("event_time", "event_time_unavailable", "concept-dict.json#/death")
        elif mortality.time_kind != "recorded":
            substitutions.append(CapabilitySubstitution(
                capability="event_time", original="recorded_death_time", substitute=mortality.time_kind,
                reason_code="event_time_is_proxy_or_date", source_field=mortality.event_time,
            ))
    if "censor_time" in required and not fixed_followup:
        end = (profile.hospital_discharge if requirements.censoring_unit == "hospitalization"
               else profile.icu_discharge)
        if end is None:
            absent("censor_time", "censor_time_unavailable", "data-sources.json#/id_cfg")
    if "observation_timing" in required and not any(item.mapped and item.timed for item in profile.concepts):
        absent("observation_timing", "observation_timing_unavailable", profile.time_source)
    if "treatment_timing" in required:
        treatments = [profile.concept(name) for name in requirements.treatment_concepts]
        if not treatments:
            treatments = [item for item in profile.concepts if item.category == "medications" and item.mapped and item.timed]
        if not treatments or any(not item.mapped or not item.timed for item in treatments):
            absent("treatment_timing", "treatment_timing_unavailable", "concept-dict.json#/sources;target_trial_spec.TrialTreatment")
    for threshold in requirements.thresholds:
        _threshold_decision(profile, threshold, missing, substitutions)
    return CapabilityDecision(
        status="refusal" if missing else "substitution" if substitutions else "exact",
        database=profile.database, requirements=requirements,
        substitutions=tuple(substitutions), missing=tuple(missing),
        unassessed_concepts=tuple(item.concept for item in unassessed_concepts),
        unassessed_concept_details=tuple(unassessed_concepts),
    )


def _threshold_decision(profile: ResearchCapabilityProfile, threshold: ThresholdRequirement,
                        missing: list[MissingCapability], substitutions: list[CapabilitySubstitution]) -> None:
    representation = getattr(profile, threshold.concept)
    if representation.kind == "exact":
        return
    value, operator = threshold.value, threshold.operator
    edges = representation.bin_edges
    if representation.kind == "capped_or_shifted":
        if value <= representation.exact_through:
            return
        edges = (representation.exact_through + 1,)
        if value == edges[0] and operator in {">=", "<"}:
            return
    if representation.kind == "binned" and value in edges and operator in {">=", "<"}:
        return
    if not edges or operator in {"==", "!="} or representation.field is None:
        missing.append(MissingCapability(capability=f"threshold:{threshold.concept}",
                                         code="numeric_threshold_not_resolvable", source=representation.source))
        return
    nearest = min(edges, key=lambda edge: (abs(edge - value), edge))
    substitute = threshold.model_copy(update={"operator": ">=" if operator in {">", ">="} else "<", "value": nearest})
    substitutions.append(CapabilitySubstitution(
        capability=f"threshold:{threshold.concept}", original=threshold, substitute=substitute,
        reason_code="numeric_threshold_aligned_to_band", source_field=representation.field,
    ))
