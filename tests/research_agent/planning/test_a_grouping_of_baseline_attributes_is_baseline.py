"""A grouping formed only from baseline attributes is itself baseline.

Age groups are formed from ``age``, which the concept owner declares an
admission attribute.  The grouping column has no window of its own, so the
review read it through the host's outer feature window and held an
age-group association for a 24 h landmark it does not need (hosted pilot:
the landmark settings then ran the grouped exposure as a spline and the run
stopped).  A grouping of a measurement, or one whose source the context does
not carry, stays post-baseline.  Fixtures are synthetic.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID,
    EXPOSURE_GROUP_TRANSFORM_ID,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    post_baseline_exposure,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

from .scientific_review_fixtures import _context, _plan

_HELD = "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"


def _grouped(
    sources: list[str],
    *,
    transform: str = EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID,
    window: str | None = None,
    extra: tuple[ConceptDescriptor, ...] = (),
) -> ResearchContext:
    base = _context()
    exposure = ConceptDescriptor(
        name="exposure",
        role=VariableRole.OTHER,
        dtype="int64",
        source_concept="exposure",
        derived_from_concepts=sources,
        unit_normalization=transform,
        analysis_window=window,
        analysis_window_role="outer_observation_window" if window else None,
    )
    return base.model_copy(
        update={
            "variables": [
                exposure,
                *extra,
                *[v for v in base.variables if v.name != "exposure"],
            ],
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps({
                    "materialization_window": {
                        "role": "outer_observation_window",
                        "anchor": "icu_admission",
                        "hours": 24,
                    }
                }),
            ),
        }
    )


def _held(context: ResearchContext) -> bool:
    review = build_plan_scientific_review(context=context, plan=_plan())
    return _HELD in {item.code for item in review.findings}


@pytest.mark.parametrize(
    "transform", [EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID, EXPOSURE_GROUP_TRANSFORM_ID]
)
def test_age_groups_are_baseline_under_the_outer_window(transform: str) -> None:
    context = _grouped(["age"], transform=transform)

    assert post_baseline_exposure(context) == (False, None)
    assert not _held(context)


def test_a_grouping_of_a_measurement_stays_post_baseline() -> None:
    lactate = ConceptDescriptor(name="lact", role=VariableRole.LAB, dtype="float64")
    context = _grouped(["lact"], extra=(lactate,))

    assert post_baseline_exposure(context) == (
        True, "outer_materialization:icu_admission[0,24]h",
    )
    assert _held(context)


def test_one_measured_source_keeps_the_grouping_post_baseline() -> None:
    lactate = ConceptDescriptor(name="lact", role=VariableRole.LAB, dtype="float64")

    assert post_baseline_exposure(_grouped(["age", "lact"], extra=(lactate,)))[0]


@pytest.mark.parametrize("sources", [[], ["absent"]])
def test_a_source_the_context_does_not_carry_proves_nothing(sources: list[str]) -> None:
    assert post_baseline_exposure(_grouped(sources))[0]


def test_a_grouping_with_its_own_window_is_read_by_that_window() -> None:
    context = _grouped(["age"], window="icu_admission[0,24]h")

    assert post_baseline_exposure(context) == (True, "icu_admission[0,24]h")


def test_a_column_that_is_no_grouping_is_not_read_as_one() -> None:
    context = _grouped(["age"], transform="window_numeric_max")

    assert post_baseline_exposure(context)[0]
