"""A sealed binary model need not repeat a distribution-only event index."""

import pytest

from easyicu.research_agent.planning.progressive_compiler import _compile_ordered_stratified_contract
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCohortIntent, ProgressivePlanCompileError, ProgressivePlanSkeleton,
    ProgressiveSkeletonStep,
)
from .test_progressive_planner_v2 import (
    _metadata_only_ordinal_multi_outcome_context, _ordinal_primary_step,
)


@pytest.mark.parametrize("event_index", [None, 1])
def test_ordered_trend_reuses_closed_binary_parent_without_mutation(event_index):
    context, skeleton, trend = _case(event_index)
    before = skeleton.model_dump_json()
    assert _compile(context, skeleton, trend)
    assert skeleton.model_dump_json() == before


def _case(event_index=None):
    context = _metadata_only_ordinal_multi_outcome_context()
    parent = _ordinal_primary_step(coding="categorical").model_copy(update={"event_level_index": event_index})
    trend = ProgressiveSkeletonStep(
        step_id="gradient", module_id="custom_analysis", planned_analysis_role="secondary",
        objective="Estimate the declared ordered gradients.", depends_on=[parent.step_id],
        raw_inputs=["stage", "death", "los_days"], scientific_action_id="association.ordinal_trend",
        custom_method="registered_ordered_stratified_analysis",
        outputs=[{"product_id": "table:gradient", "semantic_role": "custom"}],
    )
    skeleton = ProgressivePlanSkeleton(
        analysis_type="association_study",
        cohort=ProgressiveCohortIntent(name="synthetic", selection_mode="all_input_rows"),
        steps=[parent, trend], rationale="Keep the registered ordered contrast separate.",
    )
    return context, skeleton, trend


def _compile(context, skeleton, trend):
    return _compile_ordered_stratified_contract(
        context=context, skeleton=skeleton, step=trend, step_index=1,
        variables={v.name: v for v in context.variables},
        output_pairs=[("table:gradient", "custom")],
    )


@pytest.mark.parametrize("invalid", ["reverse_event", "unknown_domain", "reverse_domain", "wrong_parent", "wrong_model"])
def test_ordered_trend_keeps_unsupported_parent_semantics_closed(invalid):
    context, skeleton, trend = _case()
    if invalid == "reverse_event":
        skeleton.steps[0] = skeleton.steps[0].model_copy(update={"event_level_index": 0})
    elif invalid == "unknown_domain":
        context = context.model_copy(update={"endpoint": None})
    elif invalid == "reverse_domain":
        context = context.model_copy(update={"endpoint": context.endpoint.model_copy(update={"levels": [1, 0]})})
    elif invalid == "wrong_parent":
        trend = trend.model_copy(update={"depends_on": ["not_the_primary"]})
    else:
        skeleton.steps[0] = skeleton.steps[0].model_copy(update={"outcome_type": "continuous"})
    with pytest.raises(ProgressivePlanCompileError):
        _compile(context, skeleton, trend)
