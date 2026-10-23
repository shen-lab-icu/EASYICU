"""Typed population rules must leave room for every declared exposure level."""

from copy import deepcopy

import pytest

from easyicu.research_agent.contracts.exposure_group_rules import EXPOSURE_GROUP_LABELS_KEY
from easyicu.research_agent.planning.cohort_predicate_domain import population_emptied_exposure_levels
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
    cohort_concept_id_scope,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    render_agent_plan_revision_contract,
)
from easyicu.research_agent.schema import AnalysisStep, ConceptDescriptor, VariableRole

from .descriptive_interval_fixtures import counts_only_plan, descriptive_context


CODE = "POPULATION_EMPTIES_EXPOSURE_LEVEL"


def _study():
    context = descriptive_context("none_counts_only", repeats=False)
    parameters = {
        "concept": "entry_measure", "window": None, "scale": "ordinal",
        "groups": [
            {"id": "g1", "rule": {"summary": "value", "op": "<", "value": 47.0}},
            {"id": "g2", "rule": "otherwise"},
        ],
        "unmeasured": "exclude", "source_columns": {"value": "entry_measure"},
    }
    grouping = {
        "variable": "entry_band", "concept": "entry_measure", "scale": "ordinal",
        "parameters": parameters,
        "levels": [
            {"code": 1, "group": "g1", "label": "Lower"},
            {"code": 2, "group": "g2", "label": "Upper"},
        ],
    }
    context = context.model_copy(update={
        "research_question": "Summarize outcomes across the declared categories.",
        "primary_exposure": "entry_band",
        "variables": [
            *context.variables,
            ConceptDescriptor(name="entry_measure", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
            ConceptDescriptor(name="entry_band", role=VariableRole.DEMOGRAPHIC, dtype="int64",
                              unit_normalization="exposure_group_ordinal", valid_range=(1, 2)),
        ],
        "cohort": context.cohort.model_copy(update={"provenance": {
            **context.cohort.provenance, EXPOSURE_GROUP_LABELS_KEY: [grouping],
        }}),
    })
    plan = counts_only_plan(context)
    step = plan.steps[-1]
    spec = step.exposure_outcome_distribution_spec.model_copy(update={
        "exposure": "entry_band", "exposure_levels": [1, 2],
    })
    plan = plan.model_copy(update={"steps": [plan.steps[0], step.model_copy(update={
        "inputs": ["cohort:analysis_set", "entry_band", "outcome_flag"],
        "exposure_outcome_distribution_spec": spec,
    })]})
    return context, plan


def _restrict(plan, *, concept="entry_measure", op=">=", value=47.0, side="inclusion",
              aggregation="first", start=0, end=24, anchor="icu_admission"):
    with cohort_concept_id_scope([concept]):
        predicate = ConceptPredicate(concept, TimeWindow(anchor, float(start), float(end)), aggregation, op, value)
    return plan.model_copy(update={"cohort": CohortDefinition(
        name="synthetic", **{side: (predicate,)},
    )})


def test_a_previously_approvable_contradictory_plan_is_returned_for_repair():
    context, plan = _study()
    baseline = build_plan_scientific_review(context=context, plan=plan)
    assert baseline.approval_allowed, baseline.findings
    contradictory = _restrict(plan)
    review = build_plan_scientific_review(context=context, plan=contradictory)
    assert not review.approval_allowed
    finding, = [item for item in review.findings if item.code == CODE]
    assert finding.severity == "blocker"
    assert finding.remediation_route == "agent_plan_revision"
    repair = render_agent_plan_revision_contract(review)
    for fact in ("inclusion[0]", "entry_measure", "47", "entry_band", "g1"):
        assert fact in finding.message
        assert fact in repair
    assert "unrestricted" in repair
    assert "study-stated eligibility" in repair
    assert contradictory.cohort.inclusion  # The reviewer never silently changes the cohort.


@pytest.mark.parametrize("side,op,value,emptied", [
    ("inclusion", ">=", 47, ["g1"]),
    ("inclusion", ">", 47, ["g1"]),
    ("inclusion", "<", 47, ["g2"]),
    ("inclusion", "<=", 47, []),  # Equality leaves the upper group possible.
    ("inclusion", ">=", 19, []),
    ("inclusion", "==", 47, ["g1"]),
    ("inclusion", "!=", 47, []),
    ("inclusion", "in", [51, 62], ["g1"]),
    ("inclusion", "in", [31, 62], []),
    ("exclusion", ">=", 47, ["g2"]),
    ("exclusion", "<", 47, ["g1"]),
    ("exclusion", ">", 47, []),
    ("exclusion", "not_in", [51, 62], ["g1"]),
    ("exclusion", "in", [31, 62], []),
])
def test_threshold_boundaries_sets_and_exclusions(side, op, value, emptied):
    context, plan = _study()
    plan = _restrict(plan, side=side, op=op, value=value)
    assert [item.level for item in population_emptied_exposure_levels(context, plan)] == emptied


@pytest.mark.parametrize("restriction", [{"value": 19}, {"concept": "age_years", "value": 47}])
def test_eligibility_that_leaves_all_levels_possible_still_passes_review(restriction):
    context, plan = _study()
    plan = _restrict(plan, **restriction)
    review = build_plan_scientific_review(context=context, plan=plan)
    assert review.approval_allowed
    assert CODE not in {item.code for item in review.findings}


def _windowed(context):
    context = context.model_copy(deep=True)
    parameters = context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY][0]["parameters"]
    parameters["window"] = {"start_hours": 0, "end_hours": 6}
    parameters["source_columns"] = {"first": "entry_measure"}
    parameters["groups"][0]["rule"]["summary"] = "first"
    return context


@pytest.mark.parametrize("changes,empty", [
    ({"end": 6}, True),
    ({"end": 7}, False),
    ({"end": 6, "start": 1}, False),
    ({"end": 6, "anchor": "hospital_admission"}, False),
    ({"end": 6, "aggregation": "max"}, False),
    ({"end": 6, "concept": "age_years"}, False),
])
def test_only_the_same_summary_window_and_concept_prove_an_empty_level(changes, empty):
    context, plan = _study()
    found = population_emptied_exposure_levels(_windowed(context), _restrict(plan, **changes))
    assert bool(found) is empty


def test_first_match_priority_is_part_of_the_level_rule():
    context, plan = _study()
    context = deepcopy(context)
    parameters = context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY][0]["parameters"]
    parameters["groups"] = [
        {"id": "g1", "rule": {"summary": "value", "op": "<", "value": 23}},
        {"id": "g2", "rule": {"summary": "value", "op": "<", "value": 47}},
        {"id": "g3", "rule": "otherwise"},
    ]
    # The first rule alone allows g2 below 23; priority removes those values.
    found = population_emptied_exposure_levels(context, _restrict(plan, op="<", value=23))
    assert [item.level for item in found] == ["g2", "g3"]
    found = population_emptied_exposure_levels(context, _restrict(plan, op=">=", value=23))
    assert [item.level for item in found] == ["g1"]  # Two remaining levels do not excuse losing one.


def test_a_different_materialized_source_is_not_the_same_summary():
    context, plan = _study()
    parameters = context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY][0]["parameters"]
    parameters["source_columns"]["value"] = "another_measure"
    assert not population_emptied_exposure_levels(context, _restrict(plan))


def test_sharing_a_column_name_does_not_prove_the_same_declared_concept():
    context, plan = _study()
    parameters = context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY][0]["parameters"]
    parameters["concept"] = "another_measure"
    assert not population_emptied_exposure_levels(context, _restrict(plan))


def test_a_static_group_value_does_not_claim_equivalence_to_another_aggregation():
    context, plan = _study()
    assert not population_emptied_exposure_levels(context, _restrict(plan, aggregation="max"))


@pytest.mark.parametrize("side,op,value,empty", [
    ("inclusion", ">=", 19, ["gU"]),
    ("inclusion", "not_missing", None, ["gU"]),
    ("exclusion", "missing", None, ["gU"]),
    ("inclusion", "missing", None, ["g1", "g2"]),
    ("inclusion", "!=", 19, []),
])
def test_a_declared_unmeasured_level_must_also_remain_possible(side, op, value, empty):
    context, plan = _study()
    parameters = context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY][0]["parameters"]
    parameters.update(scale="nominal", unmeasured="own_group")
    found = population_emptied_exposure_levels(context, _restrict(plan, side=side, op=op, value=value))
    assert [item.level for item in found] == empty


def test_another_summary_can_be_unmeasured_without_the_filtered_summary_being_missing():
    context, plan = _study()
    context = _windowed(context)
    parameters = context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY][0]["parameters"]
    parameters.update(scale="nominal", unmeasured="own_group")
    parameters["groups"].insert(1, {
        "id": "g3", "rule": {"summary": "max", "op": ">", "value": 68},
    })
    parameters["source_columns"]["max"] = "entry_measure_max"
    assert not population_emptied_exposure_levels(context, _restrict(plan, end=6, value=19))


@pytest.mark.parametrize("side,op,value,empty", [
    ("inclusion", "==", "copper", ["violet"]),
    ("inclusion", "in", ["copper"], ["violet"]),
    ("exclusion", "==", "copper", ["copper"]),
    ("exclusion", "not_in", ["copper"], ["violet"]),
    ("inclusion", "in", ["copper", "violet"], []),
])
def test_categorical_levels_use_the_product_declaration(side, op, value, empty):
    context, plan = _study()
    context = context.model_copy(update={"variables": [*context.variables,
        ConceptDescriptor(name="category_token", role=VariableRole.DEMOGRAPHIC, dtype="object"),
    ]})
    step = plan.steps[-1]
    plan = plan.model_copy(update={"steps": [step.model_copy(update={
        "exposure_outcome_distribution_spec": step.exposure_outcome_distribution_spec.model_copy(update={
            "exposure": "category_token", "exposure_levels": ["copper", "violet"],
        }),
    })]})
    plan = _restrict(plan, concept="category_token", side=side, op=op, value=value)
    assert [item.level for item in population_emptied_exposure_levels(context, plan)] == empty


def test_a_second_refused_review_still_cannot_authorize_the_plan():
    context, plan = _study()
    contradictory = _restrict(plan)
    first = build_plan_scientific_review(context=context, plan=contradictory)
    second = build_plan_scientific_review(context=context, plan=contradictory)
    assert not first.approval_allowed and not second.approval_allowed
    assert [item.code for item in first.findings] == [item.code for item in second.findings]
    assert build_plan_scientific_review(context=context, plan=_restrict(plan, value=19)).approval_allowed


def test_table_one_alone_also_declares_levels_the_population_must_keep():
    context, plan = _study()
    table = AnalysisStep(
        step_id="baseline", intent="Describe the declared categories.", method="descriptive",
        inputs=["sex_code", "age_years"], expected_outputs=["table:table_one"],
        table_one_spec={
            "group_by": "sex_code", "group_levels": ["A", "B"],
            "variables": [{"name": "age_years", "variable_kind": "continuous",
                           "summary": "median_iqr", "test": "mann_whitney_or_kruskal"}],
        },
    )
    plan = _restrict(plan.model_copy(update={"steps": [table]}), concept="sex_code", op="==", value="A")
    finding, = population_emptied_exposure_levels(context, plan)
    assert (finding.grouping, finding.level) == ("sex_code", "B")
