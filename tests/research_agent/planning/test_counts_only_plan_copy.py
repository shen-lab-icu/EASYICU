"""Counts-only review copy distinguishes a promise from an explicit disclaimer."""

from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents.progressive_prompt_contracts import (
    selected_counts_only_inference_coordinate,
)


def _outline_with_copy(field, value):
    selected = SimpleNamespace(
        estimand="Counts and proportions in the declared cohort.",
        primary_method="Counts and proportions only.",
        figure_role="Display observed counts and proportions.",
        supports="Describes the observed cohort distribution.",
        reviewable_plan=["Retain all declared rows."],
    )
    if field == "reviewable_plan[0]":
        selected.reviewable_plan = [value]
    else:
        setattr(selected, field, value)
    return SimpleNamespace(design_selection=SimpleNamespace(selected=selected))


@pytest.mark.parametrize(
    "field", ["estimand", "primary_method", "figure_role", "supports", "reviewable_plan[0]"],
)
@pytest.mark.parametrize(
    "copy",
    [
        "Describe counts and proportions without confidence intervals.",
        "Report counts only; no confidence intervals, standard errors or p-values.",
        "We will not report 95% confidence intervals or p values.",
        "No inferential uncertainty is reported for these counts.",
        "仅描述计数和比例，不报告置信区间。",
        "仅描述计数和比例，不提供置信区间、标准误或 p 值。",
        "不估计推断性不确定性，仅报告观察到的计数和比例。",
        "没有患者分组权威，无需计算置信区间和标准误。",
    ],
)
def test_counts_only_accepts_explicit_output_disclaimer(field, copy):
    assert selected_counts_only_inference_coordinate(_outline_with_copy(field, copy)) is None


@pytest.mark.parametrize(
    "copy",
    [
        "Report counts and confidence intervals.",
        "Report counts with uncertainty estimates.",
        "No patient identifiers; report confidence intervals.",
        "Without patient grouping, calculate standard errors.",
        "No confidence intervals, but report p-values.",
        "No confidence intervals and calculate p-values.",
        "Do not omit confidence intervals.",
        "Not without confidence intervals.",
        "Not only confidence intervals but also standard errors.",
        "No confidence intervals are omitted.",
        "按分母计算比例及其不确定性。",
        "没有患者标识，报告置信区间。",
        "不报告置信区间，但仍计算标准误。",
        "并非不报告置信区间。",
        "不是没有置信区间。",
        "不只是置信区间，还报告 p 值。",
    ],
)
def test_counts_only_rejects_promises_and_ambiguous_negation(copy):
    assert selected_counts_only_inference_coordinate(
        _outline_with_copy("estimand", copy)
    ) == "estimand"


def test_disclaimer_in_one_coordinate_does_not_mask_another():
    outline = _outline_with_copy("estimand", "Counts only, no confidence intervals.")
    outline.design_selection.selected.primary_method = "Report p-values."
    assert selected_counts_only_inference_coordinate(outline) == "primary_method"
