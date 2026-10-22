"""A descriptive study with intervals reports its distribution as its primary result.

Since descriptive studies take their interval design from the source -- a
patient-grouped eICU study reports Wilson intervals clustered on the patient
-- the primary exposure/outcome distribution of a descriptive plan is no
longer counts-only.  The compiler wrote the descriptive-only claim ceiling
only for a counts-only design or an exposure measured after baseline, so that
primary step carried no ceiling: the distribution owner declined it ("neither
an auxiliary distribution nor a typed primary/secondary descriptive-only
result"), the primary capability was not reportable, and the plan's
automatic revision could not see why.

The compiler now states the study's own limitation on that step,
``descriptive_study_design``: unadjusted group risks with the intervals the
variance design grants, no adjusted or causal comparison.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.claim_ceiling import DescriptiveClaimContract
from easyicu.research_agent.contracts.descriptive_execution import (
    DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID,
    exposure_outcome_distribution_execution_verdict,
)
from easyicu.research_agent.planning.capability_registry import (
    resolve_primary_capability,
)
from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.progressive_compiler import (
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
)

from .descriptive_interval_fixtures import descriptive_context, descriptive_plan
from .progressive_planner_fixtures import _payload


def _grouped_study():
    context = descriptive_context("cluster_robust", grouped=True)
    return context, bind_context_dependence_authority(
        plan=descriptive_plan(context), context=context
    )


def test_a_grouped_descriptive_primary_states_the_studys_design() -> None:
    _context, plan = _grouped_study()

    step = plan.steps[-1]
    assert step.planned_analysis_role == "primary"
    assert step.exposure_outcome_distribution_spec.interval_method == "wilson"
    assert step.descriptive_claim == DescriptiveClaimContract(
        unresolved_limitations=("descriptive_study_design",)
    )


def test_the_distribution_owner_claims_the_primary_and_it_is_reportable() -> None:
    _context, plan = _grouped_study()

    assert exposure_outcome_distribution_execution_verdict(plan.steps[-1]).claimed
    verdict = resolve_primary_capability(
        analysis_type="descriptive_epidemiology", plan=plan
    )
    assert verdict.capability_id == DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID
    assert verdict.owner_claimed is True
    assert verdict.failure_reason is None


def test_without_the_ceiling_the_owner_declines_the_primary() -> None:
    """The refusal the real descriptive plan met, before the compiler stated it."""

    _context, plan = _grouped_study()
    step = plan.steps[-1].model_copy(update={"descriptive_claim": None})

    verdict = exposure_outcome_distribution_execution_verdict(step)

    assert not verdict.claimed
    assert "neither an auxiliary distribution" in verdict.reason


def test_the_plan_review_has_no_blocker() -> None:
    context, plan = _grouped_study()

    review = build_plan_scientific_review(context=context, plan=plan)

    assert not [f for f in review.findings if f.severity == "blocker"]


@pytest.mark.parametrize("role", ["secondary", "auxiliary"])
def test_only_the_primary_distribution_states_it(role: str) -> None:
    context = descriptive_context("cluster_robust", grouped=True)
    payload = _payload()
    payload.update(analysis_type="descriptive_epidemiology", robustness_intents=[])
    payload["steps"] = [payload["steps"][0], payload["steps"][2]]
    payload["steps"][0]["raw_inputs"] = ["stay_id"]
    payload["steps"][1]["planned_analysis_role"] = role

    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context,
    )

    assert plan.steps[-1].descriptive_claim is None
