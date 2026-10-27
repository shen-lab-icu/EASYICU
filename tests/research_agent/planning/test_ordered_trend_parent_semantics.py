"""The ordered trend binds to its one typed binary primary model.

A sealed binary model need not repeat a distribution-only event index; the
trend inherits the primary's estimator (logistic or modified Poisson) and may
read a binary outcome alone or with one typed continuous outcome.
"""

import pytest

from easyicu.research_agent.schema import VariableRole

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


@pytest.mark.parametrize("measure", ["risk_ratio", "odds_ratio"])
def test_an_ordered_trend_inherits_the_primary_effect_measure(measure):
    """The owner refits the primary's own estimator, so either ratio keeps its trend.

    A logistic primary reports its trend per level as an odds ratio and a
    modified-Poisson primary as a risk ratio; neither stops the plan.
    """

    context, skeleton, trend = _case()
    # The question asks for the risk ratio, so the primary may declare it.
    asked = context.model_copy(update={
        "research_question": context.research_question.rstrip(".?") + ", reported as a risk ratio.",
    }) if measure == "risk_ratio" else context
    skeleton.steps[0] = ProgressiveSkeletonStep.model_validate(
        {**skeleton.steps[0].model_dump(mode="json"), "effect_measure": measure}
    )
    assert _compile(asked, skeleton, trend)


def _binary_only(context, skeleton, trend, raw_inputs=("stage", "death")):
    trend = trend.model_copy(update={"raw_inputs": list(raw_inputs)})
    skeleton.steps[1] = trend
    return context, skeleton, trend


def test_a_binary_only_trend_binds_the_ordered_endpoint_domain():
    """No continuous outcome is needed: the trend is the adjusted refit plus Cochran-Armitage."""

    context, skeleton, trend = _binary_only(*_case())
    without_continuous = context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"outcome_columns": ["death"]}),
        "variables": [item for item in context.variables if item.name != "los_days"],
    })
    assert _compile(context, skeleton, trend)
    assert _compile(without_continuous, skeleton, trend)


def test_a_trend_with_two_additional_inputs_is_ambiguous():
    context, skeleton, trend = _binary_only(
        *_case(), raw_inputs=("stage", "death", "los_days", "death_flag"),
    )
    with pytest.raises(ProgressivePlanCompileError) as refused:
        _compile(context, skeleton, trend)
    assert refused.value.reason_code == "progressive_ordered_trend_continuous_outcome_ambiguous"


@pytest.mark.parametrize("invalid", ["unordered", "reverse_domain", "logistic_free_parent"])
def test_a_binary_only_trend_keeps_its_domain_closed(invalid):
    context, skeleton, trend = _binary_only(*_case())
    if invalid == "unordered":
        context = context.model_copy(update={"variables": [
            item.model_copy(update={"is_ordinal": False, "role": VariableRole.OTHER})
            if item.name == "stage" else item
            for item in context.variables
        ]})
    elif invalid == "reverse_domain":
        context = context.model_copy(update={
            "endpoint": context.endpoint.model_copy(update={"levels": [1, 0]}),
        })
    else:
        skeleton.steps[0] = skeleton.steps[0].model_copy(update={"outcome_type": "continuous"})
    with pytest.raises(ProgressivePlanCompileError) as refused:
        _compile(context, skeleton, trend)
    assert refused.value.reason_code == {
        "unordered": "progressive_ordered_trend_domain_unsupported",
        "reverse_domain": "progressive_ordered_trend_domain_unsupported",
        "logistic_free_parent": "progressive_ordered_trend_primary_binding_invalid",
    }[invalid]


def test_a_primary_the_owner_cannot_refit_stops_the_plan(monkeypatch):
    """The compiler applies the owner's refit predicate, so the step is never left unowned."""

    from easyicu.research_agent.planning import progressive_compiler

    context, skeleton, trend = _case()
    compile_primary = progressive_compiler._compile_adjusted_association

    def without_roster(**kwargs):
        return [item.model_copy(update={"covariates": None}) for item in compile_primary(**kwargs)]

    monkeypatch.setattr(progressive_compiler, "_compile_adjusted_association", without_roster)
    with pytest.raises(ProgressivePlanCompileError) as refused:
        _compile(context, skeleton, trend)
    assert refused.value.reason_code == "progressive_ordered_trend_primary_binding_invalid"
    assert "no adjustment set" in str(refused.value)
