"""Exact coverage decisions over closed coordinates and typed evidence."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from math import inf

from .contract import (
    Binding,
    CapabilityGap,
    CohortPredicate,
    Covered,
    Decision,
    DictionarySnapshot,
    FactAuthority,
    HostFact,
    Interval,
    Method,
    NotApplicable,
    NotCovered,
    Point,
    RequirementKey,
    Time,
    TimeZero,
    Unknown,
    VerifiedGap,
    Window,
)

_SECONDS = {"s": 1, "min": 60, "h": 3600, "d": 86400}


def _seconds(value: Decimal | str, unit: str) -> Fraction | float:
    if isinstance(value, Decimal):
        return Fraction(value) * _SECONDS[unit]
    return -inf if value == "-inf" else inf


def _time_matches(asked: Time, actual: Time, *, containment: bool) -> bool:
    if isinstance(asked, NotApplicable):
        return isinstance(actual, NotApplicable)
    if isinstance(actual, NotApplicable) or asked.anchor != actual.anchor:
        return False
    if isinstance(asked, Point) and isinstance(actual, Point):
        return _seconds(asked.offset, asked.unit) == _seconds(actual.offset, actual.unit)
    if not isinstance(asked, Interval) or not isinstance(actual, Interval):
        return False

    start = _seconds(asked.start, asked.unit)
    end = _seconds(asked.end, asked.unit)
    fact_start = _seconds(actual.start, actual.unit)
    fact_end = _seconds(actual.end, actual.unit)
    if not containment:
        return (
            start == fact_start and end == fact_end
            and asked.start_inclusive == actual.start_inclusive
            and asked.end_inclusive == actual.end_inclusive
        )
    return (
        (fact_start < start or (
            fact_start == start and (actual.start_inclusive or not asked.start_inclusive)
        ))
        and (end < fact_end or (
            end == fact_end and (actual.end_inclusive or not asked.end_inclusive)
        ))
    )


def _binding_pairs(bindings: tuple[Binding, ...]) -> set[tuple[str, str]]:
    return {(item.role, item.concept_id) for item in bindings}


def _bindings_match(asked: RequirementKey, fact: RequirementKey, dictionary: DictionarySnapshot) -> bool:
    wanted = _binding_pairs(asked.bindings)
    actual = _binding_pairs(fact.bindings)
    if wanted == actual:
        return True
    # The time-zero event may have one declared specialization. The outcome,
    # population and reference event are always exact, even for this facet.
    if not isinstance(asked.facet, TimeZero):
        return False
    missing = wanted - actual
    extra = actual - wanted
    if len(missing) != 1 or len(extra) != 1:
        return False
    [(wanted_role, wanted_id)] = missing
    [(actual_role, actual_id)] = extra
    return wanted_role == actual_role == "exposure" and any(
        edge.fact_concept_id == actual_id and edge.request_concept_id == wanted_id
        for edge in dictionary.event_specializations
    )


def _payload(key: RequirementKey) -> str:
    facet = key.facet
    if isinstance(facet, Method):
        facet = facet.model_copy(update={"parameters": tuple(sorted(
            facet.parameters, key=lambda parameter: parameter.name
        ))})
    # Preserve the types of scalar parameters (including bool versus int).
    return facet.model_dump_json()


def _key_matches(asked: RequirementKey, actual: RequirementKey, dictionary: DictionarySnapshot) -> bool:
    return (
        _payload(asked) == _payload(actual)
        and _bindings_match(asked, actual, dictionary)
        and _time_matches(
            asked.time, actual.time,
            containment=isinstance(asked.facet, Window) and asked.facet.purpose == "observation",
        )
    )


def _registered(key: RequirementKey, dictionary: DictionarySnapshot) -> bool:
    populations = dictionary.concept_ids if isinstance(key.facet, CohortPredicate) else dictionary.population_ids
    return all(
        item.concept_id in {
            "population": populations,
            "time_anchor": dictionary.anchor_ids,
        }.get(item.role, dictionary.concept_ids)
        for item in key.bindings
    )


def match(
    key: RequirementKey | None,
    *,
    authority: FactAuthority,
    dictionary: DictionarySnapshot,
    facts: tuple[HostFact, ...],
    gaps: tuple[VerifiedGap, ...] = (),
) -> Decision:
    if key is None:
        return Unknown()
    if authority.dictionary_version != dictionary.version:
        return Unknown(reason="dictionary_snapshot_mismatch")
    if not _registered(key, dictionary):
        return Unknown(reason="unregistered_coordinate")

    applicable = [
        fact.provenance for fact in facts
        if fact.provenance.authority == authority
        and _registered(fact.key, dictionary)
        and _key_matches(key, fact.key, dictionary)
    ]
    verified = [
        gap for gap in gaps
        if gap.provenance.authority == authority
        and _payload(key) == _payload(gap.key)
        and _binding_pairs(key.bindings) == _binding_pairs(gap.key.bindings)
        and _time_matches(key.time, gap.key.time, containment=False)
    ]
    if (applicable and verified) or len({gap.reason for gap in verified}) > 1:
        return Unknown(reason="conflicting_evidence")
    if applicable:
        return Covered(provenance=min(applicable, key=lambda item: item.source_field))
    if verified:
        gap = min(verified, key=lambda item: item.provenance.source_field)
        return CapabilityGap(gap=VerifiedGap(key=key, provenance=gap.provenance, reason=gap.reason))
    return NotCovered()
