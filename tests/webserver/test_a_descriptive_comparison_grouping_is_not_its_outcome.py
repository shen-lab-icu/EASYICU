"""A descriptive question's comparison grouping is its grouping, not its outcome.

pilot1 10-26 (public eICU demo): 「入 ICU 第一天接受与未接受有创机械通气的患者，
基线特征有什么不同？」 had a descriptive design. The keyword outcome slot reads
the first clinical event a sentence names, so ventilation was sealed as the
target outcome and review stopped on REQUESTED_OUTCOME_COVERAGE_INCOMPLETE for
the very concept that defined the groups. Concept metadata only; no rows, no
provider. The questions are phrasings of a pattern, not benchmark answers.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import TimeWindow
from easyicu.webserver import research_launch_scientific as launch
from easyicu.webserver import study_intent

DESCRIPTIVE = "descriptive_epidemiology"
DATABASE = "eicu"


def _reads_as(words: str) -> str:
    """The one concept the reader reads ``words`` as, whatever its frame."""

    concepts = {concept for concept, _start, _end in study_intent._concept_spans(words)}
    assert len(concepts) == 1, (words, concepts)
    return next(iter(concepts))


VENT = "机械通气"


@pytest.mark.parametrize("question,words,evidence", [
    ("入 ICU 第一天接受与未接受有创机械通气的患者，基线特征有什么不同？", "有创机械通气", "接受与未接受有创机械通气"),
    ("使用和未使用机械通气的患者院内死亡比例分别是多少？", VENT, "使用和未使用机械通气"),
    ("未接受与接受机械通气患者的乳酸", VENT, "未接受与接受机械通气"),
    ("有无急性肾损伤的患者院内死亡比例", "急性肾损伤", "有无急性肾损伤"),
    ("有/无急性肾损伤患者的乳酸", "急性肾损伤", "有/无急性肾损伤"),
    ("机械通气组与非机械通气组的院内死亡比例", VENT, "机械通气组与非机械通气"),
    ("非机械通气组与机械通气组的乳酸", VENT, "非机械通气组与机械通气"),
    ("接受机械通气与未接受机械通气的患者的乳酸", VENT, "机械通气与未接受机械通气"),
    ("机械通气组与对照组的乳酸", VENT, "机械通气组与对照组"),
    ("按是否接受机械通气分组，描述第一天最高乳酸", VENT, "按是否接受机械通气分组"),
    ("按机械通气状态分层描述乳酸", VENT, "按机械通气状态分层"),
    ("机械通气与否的患者死亡比例", VENT, "机械通气与否"),
    ("不同性别患者的乳酸", "性别", "不同性别"),
    ("How does first-day maximum lactate differ between patients with and without mechanical ventilation?",
     "mechanical ventilation", "with and without mechanical ventilation"),
    ("In-hospital mortality in patients with vs without mechanical ventilation",
     "mechanical ventilation", "with vs without mechanical ventilation"),
    ("Compare lactate in ventilated vs non-ventilated patients.", "ventilated", "ventilated vs non-ventilated"),
    ("Lactate in patients who did and did not receive mechanical ventilation",
     "mechanical ventilation", "did and did not receive mechanical ventilation"),
    ("Lactate in patients receiving and not receiving mechanical ventilation",
     "mechanical ventilation", "receiving and not receiving mechanical ventilation"),
    ("Describe first-day maximum lactate by mechanical ventilation status.",
     "mechanical ventilation", "by mechanical ventilation"),
    ("Lactate stratified by sex", "sex", "stratified by sex"),
    ("Mortality across ventilation groups", "ventilation", "across ventilation groups"),
    ("Mortality by the presence or absence of AKI", "AKI", "by the presence or absence of AKI"),
    ("Lactate in patients with mechanical ventilation and those without",
     "mechanical ventilation", "mechanical ventilation and those without"),
])
def test_comparison_phrasings_state_their_grouping(question, words, evidence):
    groupings = study_intent.stated_comparison_groupings(question)
    assert [(item.concept_id, item.evidence) for item in groupings] == [(_reads_as(words), evidence)]


@pytest.mark.parametrize("question", [
    # One concept named without groups: its own frequency or a population.
    "入 ICU 第一天接受有创机械通气的患者比例是多少？",
    "机械通气患者的院内死亡率",
    "What proportion of patients received mechanical ventilation?",
    # A cohort that includes both, not a comparison.
    "Sepsis patients with or without shock",
    "接受或未接受机械通气的患者都纳入",
    # An agent, a definition or a time, not groups.
    "AKI defined by KDIGO creatinine criteria",
    "Lactate measured by blood gas in ventilated patients",
    "Mortality by day 28 in ventilated patients",
    "按 KDIGO 标准定义的 AKI 患者死亡率",
    "以乳酸为暴露分析死亡",
    "不同时间点的乳酸",
    # A refused grouping states none.
    "不按机械通气分组，描述乳酸",
    "Lactate, not stratified by mechanical ventilation",
    # 「无创」 is a kind of ventilation, not its absence.
    "无创机械通气与有创机械通气患者的乳酸",
    # A relationship names an exposure and an outcome, not groups.
    "机械通气与院内死亡有关吗",
    "Is mechanical ventilation associated with mortality?",
])
def test_words_without_a_comparison_state_no_grouping(question):
    assert study_intent.stated_comparison_groupings(question) == ()


def test_a_negative_group_does_not_negate_the_outcome_after_it():
    question = "机械通气组与非机械通气组的院内死亡比例分别是多少？"
    assert study_intent.deterministic_intent(question)["slots"]["outcome"]["value"] == "death"
    # A counted group is no negative group: 「而非两组」 still negates.
    assert study_intent.explicit_outcome_concepts("比较两组的乳酸，而非两组的院内死亡") == ()


def _coordinates(question, family=DESCRIPTIVE, **kwargs):
    return launch._metadata_only_planning_coordinates(
        question=question, database=DATABASE, analysis_family=family, **kwargs,
    )


TABLE_ONE = "入 ICU 第一天接受与未接受有创机械通气的患者，基线特征有什么不同？"
GROUPED_SUMMARY = "接受与未接受有创机械通气的患者，入 ICU 第一天最高乳酸的中位数有何不同？"


def test_the_pilot1_grouping_is_the_grouping_and_no_outcome():
    group = _reads_as("有创机械通气")
    # The keyword outcome slot reads the grouping's event, as it did on pilot1.
    assert study_intent.deterministic_intent(TABLE_ONE)["slots"]["outcome"]["value"] == group
    coordinates = _coordinates(TABLE_ONE, stated_exposure="入 ICU 第一天接受与未接受有创机械通气")
    assert coordinates["primary_exposure"] == group
    assert coordinates["target_outcome"] is None
    assert coordinates["endpoint"] is None
    assert [(row["role"], row["concept"], row["evidence"]) for row in coordinates["coordinate_readings"]] == [
        ("exposure", group, "接受与未接受有创机械通气"),
    ]


@pytest.mark.parametrize("question", [
    GROUPED_SUMMARY,
    "有无机械通气的患者入 ICU 第一天最高乳酸的中位数有何不同？",
    "按是否接受机械通气分组，入 ICU 第一天最高乳酸的中位数是多少？",
    "How does the first-day maximum lactate differ between patients with and without mechanical ventilation?",
    "Describe first-day maximum lactate by mechanical ventilation status.",
])
def test_a_grouped_summary_reads_its_measurement_as_the_outcome(question):
    coordinates = _coordinates(question)
    assert coordinates["primary_exposure"] == _reads_as(VENT)
    assert coordinates["target_outcome"] == _reads_as("乳酸")
    assert coordinates["endpoint"].kind == "continuous"
    assert coordinates["target_outcome_aggregation"] == "max"
    assert coordinates["target_outcome_window"] == {
        "anchor": "icu_admission", "start_hours": 0.0, "end_hours": 24.0,
    }
    assert {row["role"] for row in coordinates["coordinate_readings"]} == {"outcome", "exposure"}


@pytest.mark.parametrize("question", [
    "入 ICU 第一天接受与未接受有创机械通气的患者院内死亡率有何不同？",
    "机械通气组与非机械通气组的院内死亡比例分别是多少？",
    "In-hospital mortality in patients with vs without mechanical ventilation",
])
def test_a_named_outcome_keeps_its_slot_beside_the_grouping(question):
    coordinates = _coordinates(question)
    assert coordinates["target_outcome"] == "death"
    assert coordinates["endpoint"].kind == "binary"
    assert coordinates["primary_exposure"] == _reads_as(VENT)


def test_an_association_question_keeps_its_exposure_and_outcome_reading():
    descriptive = _coordinates(GROUPED_SUMMARY)
    association = _coordinates(GROUPED_SUMMARY, family="association_study")
    assert descriptive["target_outcome"] != association["target_outcome"]
    assert association["target_outcome"] == _reads_as(VENT)
    assert association["primary_exposure"] == _reads_as("乳酸")
    assert _coordinates(GROUPED_SUMMARY, family=None)["target_outcome"] == _reads_as(VENT)


def test_a_proportion_of_one_event_keeps_its_outcome():
    coordinates = _coordinates("入 ICU 第一天接受有创机械通气的患者比例是多少？")
    assert coordinates["target_outcome"] == _reads_as("有创机械通气")
    assert coordinates["primary_exposure"] is None


def test_a_configured_outcome_is_not_a_grouping():
    group = _reads_as("有创机械通气")
    coordinates = _coordinates(TABLE_ONE, configured_outcome=group)
    assert coordinates["target_outcome"] == group
    assert coordinates["primary_exposure"] is None


def test_the_study_field_naming_another_grouping_decides():
    # The field names sex, the question's words group by ventilation: the
    # readings disagree, so neither is sealed and the planner reads the question.
    coordinates = _coordinates(GROUPED_SUMMARY, stated_exposure="性别")
    assert coordinates["primary_exposure"] != _reads_as(VENT)


def test_two_stated_groupings_are_neither_the_outcome():
    question = "不同性别以及接受与未接受机械通气的患者，基线特征有什么不同？"
    assert {item.concept_id for item in study_intent.stated_comparison_groupings(question)} == {
        _reads_as("性别"), _reads_as(VENT),
    }
    assert _coordinates(question)["target_outcome"] is None


def test_a_stated_grouping_is_never_the_summarized_quantity():
    # Two groupings and a field naming a third: no exposure is sealed, and
    # the one measurement is still the outcome, not a grouping's event.
    question = "接受与未接受肾脏替代治疗以及有无机械通气的患者，入 ICU 第一天最高乳酸有何不同？"
    assert len(study_intent.stated_comparison_groupings(question)) == 2
    coordinates = _coordinates(question, stated_exposure="性别")
    assert coordinates["primary_exposure"] is None
    assert coordinates["target_outcome"] == _reads_as("乳酸")
    assert coordinates["endpoint"].kind == "continuous"


def test_nothing_a_groupings_words_denote_is_the_summarized_quantity(monkeypatch):
    # When the grouping's words are a named study concept, they also denote
    # its family (ventilation: PEEP, tidal volume); none of those is the
    # measurement the question summarizes.
    group, measurement = _reads_as(VENT), _reads_as("乳酸")
    family = (group, _reads_as("PEEP"), _reads_as("潮气量"))
    monkeypatch.setattr(study_intent, "named_study_concepts", lambda question: (
        (family, VENT), ((measurement,), "乳酸"),
    ))
    coordinates = _coordinates(GROUPED_SUMMARY)
    assert coordinates["primary_exposure"] == group
    assert coordinates["target_outcome"] == measurement


def _planning_context(question, coordinates, columns):
    endpoint = coordinates["endpoint"]
    target = endpoint.name if endpoint is not None else coordinates["target_outcome"]
    frame = pd.DataFrame({"stay_id": pd.Series(dtype="int64"), **{
        name: pd.Series(dtype=dtype) for name, dtype in columns.items()
    }})
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False,
    }
    built = build_research_context(
        research_question=question, cohort=frame, cohort_name="synthetic", database=DATABASE,
        id_columns=["stay_id"], target_outcome=target, endpoint=endpoint,
        primary_exposure=coordinates["primary_exposure"],
        time_windows=[TimeWindow(name="bound", anchor="icu_admission", start_hours=0, end_hours=24)],
        user_preferences={"inferred_analysis_family": DESCRIPTIVE, "data_constraints": json.dumps({
            "materialization_window": {"role": "outer_observation_window", "anchor": "icu_admission", "hours": 24},
        })},
    )
    return getattr(built, "context", built)


def _step(step_id, role, raw_inputs, **extra):
    return {
        "step_id": step_id, "module_id": "custom_analysis", "planned_analysis_role": role,
        "objective": "Describe the synthetic grouped rows.", "depends_on": ["cohort"],
        "raw_inputs": raw_inputs,
        "product_inputs": [{"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"}],
        **extra,
    }


def _review(context, steps, labels):
    from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review

    skeleton = ProgressivePlanSkeleton.model_validate({
        "analysis_type": DESCRIPTIVE, "rationale": "Describe the synthetic groups.",
        "cohort": {"name": "synthetic", "selection_mode": "all_input_rows"},
        "display_labels": [{"key": name, "value": name} for name in labels],
        "steps": [
            {"step_id": "cohort", "module_id": "cohort_definition", "planned_analysis_role": "auxiliary",
             "objective": "Account for the synthetic stay denominator.", "raw_inputs": ["stay_id"]},
            *steps,
        ],
    })
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=context)
    return plan, build_plan_scientific_review(context=context, plan=plan, require_reportable_capability=True)


def test_the_table_one_plan_no_longer_misses_its_grouping_as_an_outcome():
    group = _reads_as("有创机械通气")
    coordinates = _coordinates(TABLE_ONE)
    context = _planning_context(TABLE_ONE, coordinates, {group: "float64", "age": "float64", "sex": "string"})
    assert context.target_outcome is None
    assert context.primary_exposure == group
    # The pilot1 shape: a baseline summary of several measurements by the groups.
    _plan, review = _review(context, [_step(
        "baseline", "primary", ["age", "sex", group],
        scientific_action_id="descriptive.descriptive_summary", custom_method="grouped_summary",
        outputs=[{"product_id": product, "semantic_role": "custom"}
                 for product in ("table:cohort_summary", "table:distribution_summary")],
    )], ("age", "sex", group))
    codes = {finding.code for finding in review.findings}
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" not in codes
    assert "OUTCOME_DEFINITION_UNRESOLVED" not in codes
    assert review.facts["missing_model_outcomes"] == []


def test_the_reworded_grouped_summary_is_reportable():
    from easyicu.research_agent.planning.capability_registry import assess_scientific_capability

    group, measurement = _reads_as(VENT), _reads_as("乳酸")
    coordinates = _coordinates(GROUPED_SUMMARY)
    endpoint = coordinates["endpoint"].name
    context = _planning_context(GROUPED_SUMMARY, coordinates, {
        group: "float64", measurement: "float64", endpoint: "float64",
    })
    assert context.primary_exposure == group and context.target_outcome == endpoint
    # The grouped continuous-outcome contract: the overall empirical summary
    # as the primary and the grouped summary as its answering secondary.
    plan, review = _review(context, [
        _step("overall", "primary", [measurement],
              scientific_action_id="descriptive.descriptive_summary", custom_method="empirical_summary",
              outputs=[{"product_id": product, "semantic_role": "custom"}
                       for product in ("table:cohort_summary", "table:distribution_summary")]),
        {"step_id": "groups", "module_id": "continuous_outcome_summary", "planned_analysis_role": "secondary",
         "objective": "Describe the first-day maximum within each group.", "depends_on": ["cohort"],
         "primary_exposure": group, "raw_inputs": [group, endpoint]},
    ], (measurement, endpoint, group))
    assert plan.steps[-1].inputs == ["artifact:analysis_cohort", group, endpoint]
    assert assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=plan, context=context,
    ).claim_ceiling == "reportable"
    assert review.facts["missing_model_outcomes"] == []
    codes = {finding.code for finding in review.findings}
    assert not codes & {
        "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE", "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE",
        "OUTCOME_DEFINITION_UNRESOLVED",
    }
