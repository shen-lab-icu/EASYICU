"""A functional-form sensitivity is known by the form it states, not its name.

A step id is a name the Planner chose.  A step that carries a functional form
is a nonlinear-versus-linear check whatever it is called, and its refusal
names the exact method that check needs.  Contexts are synthetic.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.progressive_compiler import (
    ProgressivePlanCompileError,
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload


def test_a_step_with_a_form_and_an_invented_method_is_told_the_exact_method() -> None:
    """Hosted pilot: ``sensitivity_continuous_form`` carried a spline form with
    an invented method; the refusal it got never named the exact method, so
    three repairs repeated it."""

    invented = _payload()
    step = invented["steps"][5]
    step["step_id"] = "06_sensitivity_continuous_form"
    step["custom_method"] = "prespecified_rcs_logistic_sensitivity"
    step["functional_form_spec"] = {"target_column": "age_years", "knot_quantiles": [0.1, 0.5, 0.9]}
    step["sensitivity_spec_ids"] = ["age_rcs_vs_linear"]

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(invented),
            context=_context(),
        )

    assert caught.value.reason_code == "progressive_functional_form_method_unsupported"
    assert caught.value.path == "custom_method"
    assert "restricted_cubic_spline_sensitivity" in str(caught.value)


def test_the_same_step_with_the_exact_method_compiles() -> None:
    exact = _payload()
    step = exact["steps"][5]
    step["step_id"] = "06_sensitivity_continuous_form"
    step["custom_method"] = "restricted_cubic_spline_sensitivity"
    step["functional_form_spec"] = {"target_column": "age_years", "knot_quantiles": [0.1, 0.5, 0.9]}
    step["sensitivity_spec_ids"] = ["age_rcs_vs_linear"]

    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(exact),
        context=_context(),
    )

    compiled = next(item for item in plan.steps if item.step_id == "06_sensitivity_continuous_form")
    assert compiled.method == "restricted_cubic_spline_sensitivity"
