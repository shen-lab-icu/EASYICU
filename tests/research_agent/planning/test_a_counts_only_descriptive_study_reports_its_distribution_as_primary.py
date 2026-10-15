"""A counts-only descriptive study can report its distribution as its primary result.

A study whose design reports counts and proportions only
(``variance_estimator: none_counts_only``) lets every plan step through only
when the step is typed: cohort accounting, the typed exposure/outcome
distribution, a descriptive Table One, a measurement audit, rendering and
reports.  Its distribution's ownership verdict claims a primary step only
under a descriptive-only claim ceiling, and the compiler wrote that ceiling
only for an exposure measured after baseline.  So a real descriptive plan
whose primary result is its distribution -- the proportion in each group
and each group's outcome -- was refused (``counts_only_step_untyped``)
before any analysis.

The compiler now states the design's own limitation on that step,
``counts_only_design``; the verdict and the gate read it unchanged.  An
exposure measured after baseline keeps its own limitation, and a design
that is not counts-only compiles as before.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.contracts.claim_ceiling import DescriptiveClaimContract
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    exposure_outcome_distribution_executor_owns_step,
)
from easyicu.research_agent.planning.dependence_authority import (
    DependenceAuthorityError,
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.progressive_compiler import (
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.schema import ResearchContext, UserPreferences
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _payload,
)

_COUNTS_ONLY = {
    "analysis_family": "descriptive_epidemiology",
    "analysis_unit": "icu_stay",
    "variance_estimator": "none_counts_only",
}


def _study(design: dict | None, *, post_baseline: bool = False) -> ResearchContext:
    base = _context()
    variables = base.variables
    if post_baseline:
        exposure = base.variable("exposure_flag").name
        variables = [
            item.model_copy(
                update={
                    "analysis_window": "icu_admission[0,24]h",
                    "analysis_window_role": "exposure_definition",
                }
            )
            if item.name == exposure
            else item
            for item in variables
        ]
    preferences = UserPreferences(
        inferred_analysis_family="descriptive_epidemiology",
        data_constraints=(
            json.dumps({"analysis_design": design}) if design is not None else None
        ),
    )
    return base.model_copy(
        update={"variables": variables, "user_preferences": preferences}
    )


def _compile(context: ResearchContext, *, role: str = "primary"):
    payload = _payload()
    payload["analysis_type"] = "descriptive_epidemiology"
    payload["robustness_intents"] = []
    payload["steps"] = payload["steps"][:3]
    payload["steps"][2]["planned_analysis_role"] = role
    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=context,
    )
    return plan


def _distribution(plan):
    return next(step for step in plan.steps if step.step_id == "03_distribution")


def test_a_counts_only_primary_distribution_states_the_designs_limitation() -> None:
    context = _study(_COUNTS_ONLY)

    bound = bind_context_dependence_authority(plan=_compile(context), context=context)

    step = _distribution(bound)
    assert step.descriptive_claim == DescriptiveClaimContract(
        unresolved_limitations=("counts_only_design",)
    )
    assert step.exposure_outcome_distribution_spec.interval_method == "none_counts_only"
    assert exposure_outcome_distribution_executor_owns_step(step)


def test_without_the_limitation_the_gate_refuses_the_primary_distribution() -> None:
    """The refusal the real descriptive plan met, before the compiler stated it."""

    context = _study(_COUNTS_ONLY)
    plan = _compile(context)
    plan = plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"descriptive_claim": None})
                if step.step_id == "03_distribution"
                else step
                for step in plan.steps
            ]
        }
    )

    with pytest.raises(DependenceAuthorityError) as refused:
        bind_context_dependence_authority(plan=plan, context=context)

    assert refused.value.code == "counts_only_step_untyped"


@pytest.mark.parametrize(
    "design",
    [
        None,
        {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
    ],
    ids=["no_design", "clustered"],
)
def test_a_design_that_is_not_counts_only_compiles_as_before(design) -> None:
    assert _distribution(_compile(_study(design))).descriptive_claim is None


@pytest.mark.parametrize("role", ["secondary", "auxiliary"])
def test_only_the_primary_distribution_states_the_limitation(role: str) -> None:
    context = _study(_COUNTS_ONLY)

    bound = bind_context_dependence_authority(
        plan=_compile(context, role=role), context=context
    )

    assert _distribution(bound).descriptive_claim is None


def test_an_exposure_measured_after_baseline_keeps_its_own_limitation() -> None:
    context = _study(_COUNTS_ONLY, post_baseline=True)

    bound = bind_context_dependence_authority(plan=_compile(context), context=context)

    assert _distribution(bound).descriptive_claim.unresolved_limitations == (
        "post_baseline_exposure_opportunity_unresolved",
    )


def test_the_limitation_is_a_closed_code() -> None:
    assert DescriptiveClaimContract(
        unresolved_limitations=("counts_only_design",)
    ).claim_ceiling == "descriptive_only"
    with pytest.raises(ValidationError):
        DescriptiveClaimContract(unresolved_limitations=("counts_only",))
