"""A distribution step states its science; the host fills what has no choice.

Planner repair statistics (10-22) found one recurring rejection of the
exposure/outcome distribution step: its contract demanded variables, level
indices, denominator/missingness policies and a confidence level together,
and a step that left any of them null was refused without saying which.
The confidence level is a host convention (0.95) the interval decision
already governs, and a binary exposure's risk-difference contrast has no
choice -- its second level against its first, as a binary model term's
reference.  The contract now requires only the variables, the event level
and the denominator/missingness policies, and names what is missing.
Synthetic context and payload; no benchmark item.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.research_agent.planning.progressive_compiler import (
    ProgressivePlanCompileError,
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanSkeleton,
)

from .descriptive_interval_fixtures import descriptive_context
from .progressive_planner_fixtures import _payload


def _skeleton(**distribution) -> dict:
    payload = _payload()
    payload.update(analysis_type="descriptive_epidemiology", robustness_intents=[])
    payload["steps"] = [payload["steps"][0], payload["steps"][2]]
    payload["steps"][0]["raw_inputs"] = ["stay_id"]
    payload["steps"][1]["planned_analysis_role"] = "primary"
    payload["steps"][1].update(distribution)
    return payload


def _compile(payload: dict):
    context = descriptive_context(repeats=False)
    return compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context,
    )[0]


def test_the_host_fills_the_confidence_level_and_a_binary_contrast() -> None:
    stated = _compile(_skeleton()).steps[-1].exposure_outcome_distribution_spec
    filled = _compile(
        _skeleton(
            confidence_level=None,
            reference_exposure_level_index=None,
            comparison_exposure_level_index=None,
        )
    ).steps[-1].exposure_outcome_distribution_spec

    assert filled.confidence_level == 0.95
    assert filled.risk_difference_contrast is not None
    assert filled.risk_difference_contrast.reference_exposure_level == (
        filled.exposure_levels[0]
    )
    assert filled.risk_difference_contrast.comparison_exposure_level == (
        filled.exposure_levels[1]
    )
    assert filled.model_dump(exclude={"confidence_level", "risk_difference_contrast"}) == (
        stated.model_dump(exclude={"confidence_level", "risk_difference_contrast"})
    )


def test_a_stated_confidence_level_is_kept() -> None:
    spec = _compile(_skeleton(confidence_level=0.9)).steps[-1].exposure_outcome_distribution_spec

    assert spec.confidence_level == 0.9


def test_one_stated_contrast_level_without_the_other_is_still_refused() -> None:
    with pytest.raises(ProgressivePlanCompileError):
        _compile(_skeleton(comparison_exposure_level_index=None))


@pytest.mark.parametrize(
    "field", ["denominator_policy", "missing_exposure_policy", "missing_outcome_policy"],
)
def test_a_missing_policy_is_named(field: str) -> None:
    with pytest.raises(ValidationError) as refused:
        ProgressivePlanSkeleton.model_validate(_skeleton(**{field: None}))

    assert f"exposure_outcome_distribution requires {field}" in str(refused.value)
