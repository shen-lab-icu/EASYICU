"""A plan does not model a nominal exposure grouping's codes as a trend.

A nominal grouping's level codes name groups in no order
(``contracts.exposure_group_rules``), so the pre-approval scientific review
refuses a model term that reads them as a scale -- a continuous term or an
``ordinal_linear`` trend -- as it refuses any closed domain read as a number.
Categories pass, and an ordinal grouping's codes lie along its scale.
Synthetic context and plan only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID,
    EXPOSURE_GROUP_TRANSFORM_ID,
)
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)

from .scientific_review_fixtures import _context, _plan

_CONFLICT = "MODEL_TERM_CODING_CONFLICTS_WITH_DECLARED_DOMAIN"
_LEVELS = ["1", "2", "3"]


def _grouped_context(transform: str) -> ResearchContext:
    base = _context()
    return base.model_copy(
        update={
            "variables": [
                *base.variables,
                ConceptDescriptor(
                    name="lact_group_x1",
                    source_concept="lact",
                    role=VariableRole.OTHER,
                    dtype="int64",
                    unit_normalization=transform,
                    valid_range=[1, 3],
                    is_ordinal=transform == EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID,
                ),
            ]
        }
    )


def _plan_with(term: ModelTermSpec) -> AnalysisPlan:
    plan = _plan()
    primary = next(
        step for step in plan.steps if step.planned_analysis_role == "primary"
    )
    requirement = primary.model_requirements[0]
    requirement = requirement.model_copy(
        update={
            "covariates": ["age", term.name],
            "model_terms": [*requirement.model_terms, term],
        }
    )
    primary = primary.model_copy(update={"model_requirements": [requirement]})
    return plan.model_copy(
        update={
            "steps": [
                primary if step.step_id == primary.step_id else step
                for step in plan.steps
            ]
        }
    )


_TREND = ModelTermSpec(
    name="lact_group_x1",
    role="covariate",
    coding="ordinal_linear",
    levels=_LEVELS,
    transform="declared_level_index",
)
_NUMBER = ModelTermSpec(
    name="lact_group_x1",
    role="covariate",
    coding="continuous",
    transform="identity",
)
_CATEGORIES = ModelTermSpec(
    name="lact_group_x1",
    role="covariate",
    coding="categorical",
    levels=_LEVELS,
    reference_level="1",
    transform="treatment_contrast",
)


@pytest.mark.parametrize(
    ("transform", "term", "refused"),
    [
        pytest.param(EXPOSURE_GROUP_TRANSFORM_ID, _TREND, True, id="nominal-trend"),
        pytest.param(EXPOSURE_GROUP_TRANSFORM_ID, _NUMBER, True, id="nominal-number"),
        pytest.param(
            EXPOSURE_GROUP_TRANSFORM_ID, _CATEGORIES, False, id="nominal-categories"
        ),
        pytest.param(
            EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID, _TREND, False, id="ordinal-trend"
        ),
    ],
)
def test_a_nominal_group_enters_a_planned_model_only_as_categories(
    transform: str, term: ModelTermSpec, refused: bool
) -> None:
    review = build_plan_scientific_review(
        context=_grouped_context(transform), plan=_plan_with(term)
    )

    conflicts = [item for item in review.findings if item.code == _CONFLICT]
    assert bool(conflicts) is refused
    if refused:
        assert conflicts[0].severity == "blocker"
        assert "lact_group_x1" in conflicts[0].message
