"""Compile physical wide-column representations into analysis semantics.

Base-concept metadata describes the clinical value before materialisation.
Companion columns such as ``*_first_time`` and ``*_mean`` are different
representations and must not blindly inherit the base value's ordinal role.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..schema import (
    AggregationRule,
    ConceptDescriptor,
    VariableRole,
)

__all__ = ["compile_wide_representation_semantics"]


_RELATIVE_TIME_TRANSFORMS = frozenset(
    {
        "first_truthy_event_time",
        "last_truthy_event_time",
        "window_first_time",
        "window_last_time",
    }
)
_MEAN_TRANSFORMS = frozenset({"window_numeric_mean"})
_PRECOMPUTED_NUMERIC_TRANSFORMS = {
    "window_numeric_first": "first non-null value",
    "window_numeric_max": "maximum value",
    "window_numeric_mean": "arithmetic mean",
    "window_numeric_min": "minimum value",
}


def _without_base_score_caveats(values: Sequence[str]) -> list[str]:
    score_phrases = (
        "0–4 ordinal",
        "0-4 ordinal",
        "never average",
        "aggregate by max",
        "follows the same 0–4",
        "follows the same 0-4",
    )
    return [
        value
        for value in values
        if not any(phrase in value.lower() for phrase in score_phrases)
    ]


def _time_representation(
    descriptor: ConceptDescriptor, *, transform: str
) -> ConceptDescriptor:
    caveats = _without_base_score_caveats(descriptor.clinical_caveats)
    caveat = (
        "This column is a relative event/observation time, not the clinical "
        f"value of {descriptor.source_concept or descriptor.name}."
    )
    if caveat not in caveats:
        caveats.append(caveat)
    forbidden = list(descriptor.forbidden_transformations)
    if transform == "window_first_time":
        first_observation_caveat = (
            "This is the first non-null observation inside the materialization "
            "window, not a certified clinical onset or treatment initiation."
        )
        absence_caveat = (
            "A missing first-observation time means no qualifying observation "
            "was recorded inside that window; it does not prove that the event "
            "or treatment was absent."
        )
        restriction = (
            "Do not interpret this first-observation time as clinical onset or "
            "treatment initiation without typed state-transition authority."
        )
        for item in (first_observation_caveat, absence_caveat):
            if item not in caveats:
                caveats.append(item)
        if restriction not in forbidden:
            forbidden.append(restriction)
    return descriptor.model_copy(
        update={
            "role": VariableRole.TIME,
            "unit": "h",
            "valid_range": None,
            "allowed_aggregations": [AggregationRule.NONE],
            "aggregation_default": AggregationRule.NONE,
            "is_ordinal": False,
            "ordinal_levels": None,
            "pitfalls": [],
            "clinical_caveats": caveats,
            "forbidden_transformations": forbidden,
        }
    )


def _mean_representation(descriptor: ConceptDescriptor) -> ConceptDescriptor:
    if not descriptor.is_ordinal:
        return descriptor
    caveats = _without_base_score_caveats(descriptor.clinical_caveats)
    caveat = (
        "This is a precomputed arithmetic mean of an ordinal source concept. "
        "Treat it as a derived continuous summary, not as an ordinal score; do "
        "not use it as the primary clinical score without protocol authority."
    )
    caveats.append(caveat)
    forbidden = list(descriptor.forbidden_transformations)
    restriction = (
        "Do not use this mean-derived ordinal representation as the primary "
        "clinical score without an explicit study protocol."
    )
    if restriction not in forbidden:
        forbidden.append(restriction)
    return descriptor.model_copy(
        update={
            "role": VariableRole.OTHER,
            "allowed_aggregations": [AggregationRule.NONE],
            "aggregation_default": AggregationRule.NONE,
            "is_ordinal": False,
            "ordinal_levels": None,
            "pitfalls": [],
            "clinical_caveats": caveats,
            "forbidden_transformations": forbidden,
        }
    )


def _precomputed_numeric_representation(
    descriptor: ConceptDescriptor, *, transform: str
) -> ConceptDescriptor:
    label = _PRECOMPUTED_NUMERIC_TRANSFORMS[transform]
    caveats = list(descriptor.clinical_caveats)
    caveat = (
        f"This materialized column is the precomputed {label} within its "
        "declared analysis window. Do not reinterpret it using the source "
        "concept's default aggregation policy."
    )
    if caveat not in caveats:
        caveats.append(caveat)
    return descriptor.model_copy(
        update={
            # The temporal reduction has already happened upstream. Publishing
            # a second aggregation default here made *_max columns look like
            # median columns in reporting and could invite double aggregation.
            "allowed_aggregations": [AggregationRule.NONE],
            "aggregation_default": AggregationRule.NONE,
            "clinical_caveats": caveats,
        }
    )
def compile_wide_representation_semantics(
    descriptors: Sequence[ConceptDescriptor],
) -> list[ConceptDescriptor]:
    """Return descriptors whose analysis semantics match their representation."""

    compiled: list[ConceptDescriptor] = []
    for descriptor in descriptors:
        transform = str(descriptor.unit_normalization or "").strip().lower()
        if transform in _RELATIVE_TIME_TRANSFORMS:
            compiled.append(_time_representation(descriptor, transform=transform))
        elif transform in _MEAN_TRANSFORMS:
            mean_descriptor = _mean_representation(descriptor)
            compiled.append(
                _precomputed_numeric_representation(
                    mean_descriptor,
                    transform=transform,
                )
            )
        elif transform in _PRECOMPUTED_NUMERIC_TRANSFORMS:
            compiled.append(
                _precomputed_numeric_representation(
                    descriptor,
                    transform=transform,
                )
            )
        else:
            compiled.append(descriptor)
    return compiled
