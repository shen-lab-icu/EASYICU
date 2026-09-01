from __future__ import annotations

from easyicu.research_agent.research_context.representation_semantics import (
    compile_wide_representation_semantics,
)
from easyicu.research_agent.schema import (
    AggregationRule,
    ConceptDescriptor,
    VariableRole,
)


def _ordinal_descriptor(
    name: str,
    *,
    transform: str,
) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        dtype="float64",
        role=VariableRole.ORDINAL_SCORE,
        valid_range=[0.0, 4.0],
        allowed_aggregations=[
            AggregationRule.MAX_LAST,
            AggregationRule.FIRST_VALUE,
            AggregationRule.MEDIAN_ONLY,
        ],
        aggregation_default=AggregationRule.MAX_LAST,
        is_ordinal=True,
        ordinal_levels=[0, 1, 2, 3, 4],
        source_concept="organ_score",
        unit_normalization=transform,
        temporal_resolution="relative to icu_admission in h",
        clinical_caveats=[
            "SOFA components are 0–4 ordinal levels; never average. "
            "Aggregate by max within window."
        ],
    )


def test_first_time_companion_is_time_not_ordinal_score() -> None:
    descriptor = _ordinal_descriptor(
        "organ_score_first_time",
        transform="window_first_time",
    )

    compiled = compile_wide_representation_semantics([descriptor])[0]

    assert compiled.role == VariableRole.TIME
    assert compiled.unit == "h"
    assert compiled.valid_range is None
    assert compiled.is_ordinal is False
    assert compiled.ordinal_levels is None
    assert compiled.allowed_aggregations == [AggregationRule.NONE]
    assert all("0–4 ordinal" not in item for item in compiled.clinical_caveats)
    assert any(
        "first non-null observation" in item
        and "not a certified clinical onset or treatment initiation" in item
        for item in compiled.clinical_caveats
    )
    assert any(
        "does not prove that the event or treatment was absent"
        in item
        for item in compiled.clinical_caveats
    )


def test_mean_of_ordinal_source_is_explicit_derived_continuous_summary() -> None:
    descriptor = _ordinal_descriptor(
        "organ_score_mean",
        transform="window_numeric_mean",
    )

    compiled = compile_wide_representation_semantics([descriptor])[0]

    assert compiled.role == VariableRole.OTHER
    assert compiled.is_ordinal is False
    assert compiled.ordinal_levels is None
    assert compiled.allowed_aggregations == [AggregationRule.NONE]
    assert any(
        "precomputed arithmetic mean of an ordinal source" in item
        for item in compiled.clinical_caveats
    )
    assert any(
        "primary clinical score" in item
        for item in compiled.forbidden_transformations
    )


def test_nonordinal_mean_representation_keeps_role_but_not_source_default() -> None:
    descriptor = ConceptDescriptor(
        name="heart_rate_mean",
        dtype="float64",
        role=VariableRole.VITAL,
        unit_normalization="window_numeric_mean",
        is_ordinal=False,
    )

    compiled = compile_wide_representation_semantics([descriptor])[0]

    assert compiled.role == VariableRole.VITAL
    assert compiled.allowed_aggregations == [AggregationRule.NONE]
    assert compiled.aggregation_default == AggregationRule.NONE
    assert any("precomputed arithmetic mean" in item for item in compiled.clinical_caveats)


def test_precomputed_max_publishes_physical_representation_not_median_policy() -> None:
    descriptor = ConceptDescriptor(
        name="marker_max",
        dtype="float64",
        role=VariableRole.LAB,
        unit="mg/dL",
        unit_normalization="window_numeric_max",
        allowed_aggregations=[AggregationRule.MEDIAN_ONLY],
        aggregation_default=AggregationRule.MEDIAN_ONLY,
        is_ordinal=False,
    )

    compiled = compile_wide_representation_semantics([descriptor])[0]

    assert compiled.role == VariableRole.LAB
    assert compiled.allowed_aggregations == [AggregationRule.NONE]
    assert compiled.aggregation_default == AggregationRule.NONE
    assert any("precomputed maximum value" in item for item in compiled.clinical_caveats)
