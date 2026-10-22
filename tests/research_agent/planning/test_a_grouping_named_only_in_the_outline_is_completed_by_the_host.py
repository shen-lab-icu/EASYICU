"""A distribution step whose grouping only the outline names is completed by the host.

"Mortality in men and women": the study states no exposure, so the Planner
names the grouping in the outline step's roster beside the outcome.  The host
used to leave that step to the model, which on the hosted pilot left a level
index out three times and stopped planning.  An unambiguous roster -- one
closed-domain variable besides the outcome and the identifiers -- is now
completed by the host; an ambiguous or open one stays model judgment.

Synthetic contexts and generic wording only.
"""
from __future__ import annotations

import pytest

from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveFoundationMaterialization,
    ProgressivePlanOutline,
)
from easyicu.research_agent.planning.progressive_host_materialization import (
    host_materialize_progressive_step,
)

from .progressive_planner_fixtures import _context, _foundation_payload, _outline_payload

_DISTRIBUTION = 2


def _materialize(roster: list[str]):
    payload = _outline_payload()
    payload["steps"][_DISTRIBUTION]["variable_names"] = roster
    outline = ProgressivePlanOutline.model_validate(payload)
    assert outline.steps[_DISTRIBUTION].module_id == "exposure_outcome_distribution"
    return host_materialize_progressive_step(
        context=_context().model_copy(update={"primary_exposure": None}),
        outline=outline,
        outline_step=outline.steps[_DISTRIBUTION],
        foundation=ProgressiveFoundationMaterialization.model_validate(
            _foundation_payload()
        ).foundation,
        available_product_refs=(("01_cohort", "artifact:analysis_cohort"),),
    )


def test_the_one_grouping_beside_the_outcome_is_the_steps_exposure() -> None:
    materialization = _materialize(["sex_code", "outcome_flag", "stay_id"])

    assert materialization is not None
    step = materialization.step
    assert (step.primary_exposure, step.outcome) == ("sex_code", "outcome_flag")
    assert (step.reference_exposure_level_index, step.comparison_exposure_level_index) == (0, 1)


@pytest.mark.parametrize(
    "roster",
    [["sex_code", "exposure_flag", "outcome_flag"], ["age_years", "outcome_flag"], ["outcome_flag"]],
    ids=["two-groupings", "open-domain", "no-grouping"],
)
def test_an_ambiguous_or_open_roster_stays_model_judgment(roster: list[str]) -> None:
    assert _materialize(roster) is None
