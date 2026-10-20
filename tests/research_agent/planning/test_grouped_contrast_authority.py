"""Synthetic grouped exposures keep one direction from planning to readers."""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_CONTRASTS_KEY,
    EXPOSURE_GROUP_LABELS_KEY,
    ExposureGroupRuleError,
    planned_group_column,
    read_group_contrast,
)
from easyicu.research_agent.agents.writer_display_labels import reader_contrast_instruction
from easyicu.research_agent.agents.reporting import WriterAgent
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.orchestration.exposure_group_labels import (
    exposure_group_label_rows,
)
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.research_context.builder import _exposure_group_projections
from tests.support.grouped_contrast_fixtures import _bmi, _record, grouped_context, skeleton


@pytest.mark.parametrize("model", [False, True], ids=["risk_difference", "adjusted_model"])
@pytest.mark.parametrize("reference,comparison", [(1, 3), (2, 1)])
def test_stated_group_contrast_refuses_a_different_v2_plan(model, reference, comparison):
    with pytest.raises(ProgressivePlanCompileError) as error:
        compile_progressive_plan(
            skeleton=skeleton(reference, comparison, model=model),
            context=grouped_context(stated=["reference", "contrast"]),
        )
    assert error.value.reason_code == "progressive_exposure_group_contrast_mismatch"
    assert error.value.step_id == ("05_primary" if model else "03_distribution")


@pytest.mark.parametrize("model", [False, True])
def test_stated_group_contrast_compiles_unchanged(model):
    plan, _ = compile_progressive_plan(skeleton=skeleton(model=model),
                                       context=grouped_context(stated=["reference", "contrast"]))
    assert plan.steps[-1].step_id == ("05_primary" if model else "03_distribution")


def test_reference_only_keeps_the_planners_unstated_comparison():
    plan, _ = compile_progressive_plan(skeleton=skeleton(2, 1),
                                       context=grouped_context(stated=["reference"]))
    assert plan.steps[-1].exposure_outcome_distribution_spec.risk_difference_contrast.comparison_exposure_level == 1


def test_legacy_context_cannot_silently_overrule_its_bound_contrast():
    with pytest.raises(ProgressivePlanCompileError):
        compile_progressive_plan(skeleton=skeleton(3, 1), context=grouped_context())


def test_unstated_group_contrast_is_the_compiled_plans_choice():
    plan, _ = compile_progressive_plan(skeleton=skeleton(3, 1), context=grouped_context(stated=[]))
    pair = plan.steps[-1].exposure_outcome_distribution_spec.risk_difference_contrast
    record = _record({**_bmi(), "reference": None, "contrast": None})
    record["compiled"]["groupings"][0]["variable"] = "exposure_flag"
    (row,) = exposure_group_label_rows(record, record_sha256="a" * 64, plan=plan)
    assert (row["reference"], row["contrast"]) == (
        pair.reference_exposure_level, pair.comparison_exposure_level,
    ) == (3, 1)


def test_an_unstated_preplan_default_is_not_shown_as_the_studys_choice():
    record = _record({**_bmi(), "reference": None, "contrast": None})
    (row,) = exposure_group_label_rows(record, record_sha256="a" * 64)
    assert (row["reference"], row["contrast"]) == (None, None)


@pytest.mark.parametrize("choices", [
    {"reference": None, "contrast": None},
    {"reference": "g2", "contrast": None},
    {"reference": "g2", "contrast": "g3"},
])
def test_choice_provenance_survives_the_metadata_context_builder(choices):
    record = _record({**_bmi(), **choices})
    item = record["compiled"]["groupings"][0]
    column = planned_group_column(
        item["derivation"], variable=item["variable"], labels=item["labels"],
        compared=item["compared"], groupings_record_sha256="a" * 64,
    ).record()
    projections = _exposure_group_projections(None, {"exposure_groups": [column]})
    assert projections[EXPOSURE_GROUP_CONTRASTS_KEY][0]["stated"] == [
        key for key, value in choices.items() if value is not None
    ]
    assert projections[EXPOSURE_GROUP_LABELS_KEY][0]["rule_wording_version"] == 2


@pytest.mark.parametrize("stated", [None, True, {}, ["contrast"], ["reference", "unknown"]])
def test_malformed_choice_provenance_fails_closed(stated):
    with pytest.raises(ExposureGroupRuleError):
        read_group_contrast(
            {"reference": 1, "contrast": 2, "stated": stated}, variable="bands", groups=3,
        )


def test_two_steps_cannot_reverse_one_grouped_contrast():
    original = skeleton(model=True).model_dump(mode="json")
    second = {**skeleton(1, 3).model_dump(mode="json")["steps"][-1],
              "step_id": "another", "planned_analysis_role": "secondary"}
    original["steps"].append(second)
    with pytest.raises(ProgressivePlanCompileError, match="conflicting compiled"):
        compile_progressive_plan(skeleton=ProgressivePlanSkeleton.model_validate(original),
                                 context=grouped_context(stated=[]))


def test_an_unmeasured_level_cannot_be_a_plan_selected_reference():
    context = grouped_context(stated=[])
    context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY][0]["levels"][-1]["group"] = "gU"
    with pytest.raises(ProgressivePlanCompileError):
        compile_progressive_plan(skeleton=skeleton(3, 1), context=context)


def test_the_writer_names_the_compiled_comparison_and_reference():
    context = grouped_context(stated=[])
    plan, _ = compile_progressive_plan(skeleton=skeleton(3, 1), context=context)
    instruction = reader_contrast_instruction(plan, plan.display_labels, context=context)
    assert "Band A vs Band C (reference)" in instruction
    assert "Band C vs Band A" not in instruction
    assert "subtract the reference risk from the comparison risk" in instruction
    client = ScriptedMockLLMClient(["The groups were compared."])
    WriterAgent(client, nature_writing_enabled=False)._call_section(
        section_name="Results", instruction="", context=context,
        evidence_ids=[], evidence_digest="", reader_display_labels=plan.display_labels,
        analysis_plan=plan,
    )
    assert any("Band A vs Band C (reference)" in message.content
               for message in client.calls[0][0])


def test_a_counts_only_plan_projects_no_implicit_contrast():
    plan, _ = compile_progressive_plan(skeleton=skeleton(3, 1), context=grouped_context(stated=[]))
    plan.steps[-1].exposure_outcome_distribution_spec.risk_difference_contrast = None
    record = _record({**_bmi(), "reference": None, "contrast": None})
    record["compiled"]["groupings"][0]["variable"] = "exposure_flag"
    (row,) = exposure_group_label_rows(record, record_sha256="a" * 64, plan=plan)
    assert (row["reference"], row["contrast"]) == (None, None)


def test_a_legacy_plan_disagreeing_with_a_stated_reference_is_unreadable():
    plan, _ = compile_progressive_plan(skeleton=skeleton(3, 1), context=grouped_context(stated=[]))
    record = _record(_bmi())
    record["compiled"]["groupings"][0]["variable"] = "exposure_flag"
    with pytest.raises(ValueError, match="changes a stated"):
        exposure_group_label_rows(record, record_sha256="a" * 64, plan=plan)


def test_a_historical_plan_cannot_project_an_unmeasured_reference():
    plan, _ = compile_progressive_plan(skeleton=skeleton(3, 1), context=grouped_context(stated=[]))
    raw = plan.model_dump(mode="json")
    raw["steps"][-1]["exposure_outcome_distribution_spec"]["risk_difference_contrast"]["reference_exposure_level"] = 4
    record = _record({
        **_bmi(), "scale": "nominal", "reference": None, "contrast": None,
        "unmeasured": {"handling": "own_group", "label": "Not recorded"},
    })
    record["compiled"]["groupings"][0]["variable"] = "exposure_flag"
    with pytest.raises(ValueError, match="two measured groups"):
        exposure_group_label_rows(record, record_sha256="a" * 64, plan=raw)


@pytest.mark.parametrize("choice", [[], {}, "", 0, False, "gU", "unknown"])
def test_malformed_explicit_record_choices_are_unreadable(choice):
    record = _record(_bmi())
    record["compiled"]["groupings"][0]["reference"] = choice
    with pytest.raises(ValueError, match="names a declared group"):
        exposure_group_label_rows(record, record_sha256="a" * 64)
