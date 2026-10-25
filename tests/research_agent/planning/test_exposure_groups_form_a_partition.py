"""Synthetic partitions: no measured boundary belongs to two exposure groups."""

from __future__ import annotations

from copy import deepcopy
import json

import pyarrow as pa
import pytest

from easyicu.research_agent.contracts.exposure_group_rules import (
    ExposureGroupPartitionError,
    ExposureGroupRuleError,
    evaluate_grouping,
    read_grouping_rules,
)
from easyicu.research_agent.agents.exposure_grouping_planner import (
    EXPOSURE_GROUPING_GUIDE,
    ask_exposure_groupings,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.planning.exposure_group_compile import compile_exposure_groupings
from easyicu.research_agent.planning.exposure_group_spec import read_stated_exposure_groupings
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
)


_AGE = ConceptDescriptor(
    name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64", unit="years",
    source_concept="age", unit_normalization="stay_level_unique_value", valid_range=[0, 120],
)
_BMI = _AGE.model_copy(update={"name": "bmi", "source_concept": "bmi", "unit": "kg/m2", "valid_range": [8, 70]})


def _lab(name, summary, unit="mg/dL"):
    return ConceptDescriptor(
        name=name, role=VariableRole.LAB, dtype="float64", unit=unit, source_concept="glu",
        unit_normalization=f"window_numeric_{summary}", analysis_window="icu_admission[0,24]h",
        valid_range=[0, 1000],
    )


def _context(*variables):
    base = _planner_context()
    return base.model_copy(update={
        "variables": [*base.variables, *variables],
        "data_constraints": json.dumps({"materialization_window": {
            "role": "outer_observation_window", "anchor": "ICU admission", "hours": 24,
        }}),
    })


def _group(group_id, label, op, value):
    return {
        "id": group_id, "label": label,
        "rule": {"summary": "value", "op": op, "value": value, "unit": "years"},
    }


def _spec(*, lower="<=", upper=">=", quote="Compare younger and older adults"):
    return {
        "id": "x1", "concept": "age", "window": None, "scale": "ordinal",
        "groups": [_group("g1", "younger", lower, 57), _group("g2", "older", upper, 57)],
        "unmeasured": {"handling": "exclude"}, "quote": quote, "source": "question",
    }


def _compile(spec):
    return compile_exposure_groupings(
        read_stated_exposure_groupings({"groupings": [spec]}),
        _context(_AGE).model_copy(update={"research_question": spec["quote"]}),
    ).groupings[0]


def test_both_inclusive_bounds_are_refused_before_a_variable_is_declared():
    compiled = _compile(_spec())
    assert compiled.disposition == "not_applied"
    assert compiled.reason == "exposure_group_rules_overlap"
    assert compiled.variable is None
    assert all(text in compiled.detail for text in ("x1", "g1", "g2", "57"))


@pytest.mark.parametrize("ages", [[56, 57, 58], [56, 58]])
def test_assignment_cannot_hide_overlap_by_taking_the_first_match(ages):
    parameters = _compile(_spec()).derivation()
    parameters["source_columns"] = {"value": "age"}
    with pytest.raises(ExposureGroupRuleError, match="exposure_group_rules_overlap"):
        evaluate_grouping(read_grouping_rules(parameters), pa.table({"age": ages}))


@pytest.mark.parametrize("scale", ["nominal", "ordinal"])
@pytest.mark.parametrize(("lower", "upper", "reason"), [
    ("<", ">=", None), ("<=", ">", None),
    ("<=", ">=", "exposure_group_rules_overlap"),
    ("<", ">", "exposure_group_rules_not_exhaustive"),
])
def test_every_boundary_has_exactly_one_owner(scale, lower, upper, reason):
    compiled = _compile({**_spec(lower=lower, upper=upper), "scale": scale})
    assert compiled.reason == reason
    if reason is None:
        rules = read_grouping_rules(compiled.derivation())
        assert evaluate_grouping(rules, pa.table({"age": [56.9, 57, 57.1, None]})).to_pylist() == (
            [1, 2, 2, None] if lower == "<" else [1, 1, 2, None]
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_an_overlap_of_positive_width_is_refused_in_either_order(reverse):
    spec = _spec()
    spec["groups"][0]["rule"]["value"] = 60
    if reverse:
        spec["groups"].reverse()
    assert _compile(spec).reason == "exposure_group_rules_overlap"


@pytest.mark.parametrize(("concept", "summary", "unit", "low", "high", "descriptor", "values"), [
    ("age", "value", "years", 40, 70, _AGE, [39, 40, 69.9, 70]),
    ("bmi", "value", "kg/m2", 18.5, 30, _BMI, [18, 18.5, 29.9, 30]),
    ("glu", "max", "mg/dL", 80, 150, _lab("glu_max", "max"), [79, 80, 149.9, 150]),
    ("lact", "max", "mmol/L", 2, 4,
     _lab("lact_max", "max", unit="mmol/L").model_copy(update={"source_concept": "lact"}),
     [1.9, 2, 3.9, 4]),
])
def test_three_bands_include_a_between_interval(concept, summary, unit, low, high, descriptor, values):
    spec = _spec(lower="<")
    spec.update(concept=concept, window=None if summary == "value" else {"start_hours": 0, "end_hours": 24})
    spec["groups"] = [
        _group("g1", f"below {low}", "<", low),
        _group("g3", f"{high} or above", ">=", high),
        {"id": "g2", "label": f"{low}–<{high}", "rule": "otherwise"},
    ]
    for group in spec["groups"][:2]:
        group["rule"].update(summary=summary, unit=unit)
    compiled = compile_exposure_groupings(
        read_stated_exposure_groupings({"groupings": [spec]}), _context(descriptor),
    ).groupings[0]
    assert compiled.applied
    result = evaluate_grouping(read_grouping_rules(compiled.derivation()), pa.table({descriptor.name: values}))
    assert result.to_pylist() == [1, 2, 2, 3]


@pytest.mark.parametrize("wording", ["exclude [57, 60)", "排除 [57, 60)"])
@pytest.mark.parametrize("handling", ["exclude", "own_group"])
def test_a_stated_gap_is_recorded_and_excluded_without_hiding_unmeasured_values(wording, handling):
    spec = _spec(lower="<", quote=f"Compare age groups; {wording}")
    spec["groups"][1]["rule"]["value"] = 60
    spec["scale"] = "nominal"
    spec["unmeasured"] = {"handling": handling, **({"label": "missing"} if handling == "own_group" else {})}
    compiled = _compile(spec)
    assert compiled.applied
    expected = {"summary": "value", "unit": "years", "lower": 57, "upper": 60,
                "lower_inclusive": True, "upper_inclusive": False}
    assert compiled.record()["derivation"]["excluded_ranges"] == [expected]
    rules = read_grouping_rules(compiled.derivation())
    assert evaluate_grouping(rules, pa.table({"age": [56, 57, 59.9, 60, None]})).to_pylist() == [
        1, None, None, 2, 3 if handling == "own_group" else None,
    ]
    tampered = compiled.derivation()
    tampered["excluded_ranges"][0]["upper_inclusive"] = True
    with pytest.raises(ExposureGroupRuleError, match="excluded_ranges"):
        read_grouping_rules(tampered)
    del tampered["excluded_ranges"]
    with pytest.raises(ExposureGroupRuleError, match="measured stays meet no group"):
        evaluate_grouping(read_grouping_rules(tampered), pa.table({"age": [57]}))


@pytest.mark.parametrize("quote", [
    "Compare age groups", "Compare age groups; do not exclude [57, 60)",
    "Compare age groups; exclude [57, 59)", "Compare age groups; exclude [57, 60]",
])
def test_a_gap_requires_the_exact_positive_exclusion(quote):
    spec = _spec(lower="<", quote=quote)
    spec["groups"][1]["rule"]["value"] = 60
    assert _compile(spec).reason == "exposure_group_rules_not_exhaustive"


def test_a_single_excluded_boundary_is_recorded():
    compiled = _compile(_spec(lower="<", upper=">", quote="Compare ages; exclude [57, 57]"))
    assert compiled.applied
    assert evaluate_grouping(read_grouping_rules(compiled.derivation()), pa.table({"age": [56, 57, 58]})).to_pylist() == [1, None, 2]


def test_the_compiler_cannot_accept_an_exclusion_absent_from_the_study():
    spec = _spec(lower="<", upper=">", quote="Compare ages; exclude [57, 57]")
    compiled = compile_exposure_groupings(
        read_stated_exposure_groupings({"groupings": [spec]}), _context(_AGE),
    ).groupings[0]
    assert compiled.reason == "exposure_group_rules_not_exhaustive"
    assert not compiled.applied


@pytest.mark.parametrize(("labels", "lower", "upper", "expected"), [
    (("below 57", "57 or above"), "<", ">=", None),
    (("57 岁以下", "57 岁及以上"), "<", ">=", None),
    (("57 岁及以下", "above 57"), "<=", ">", None),
    (("below 57", "above 57"), "<=", ">", "exposure_group_label_rule_mismatch"),
    (("57 岁以下", "above 57"), "<=", ">", "exposure_group_label_rule_mismatch"),
    (("under 57", "57 or above"), "<", ">", "exposure_group_rules_not_exhaustive"),
])
def test_labels_and_rules_agree_in_english_and_chinese(labels, lower, upper, expected):
    spec = _spec(lower=lower, upper=upper)
    for group, label in zip(spec["groups"], labels):
        group["label"] = label
    assert _compile(spec).reason == expected


def test_wrong_between_label_is_refused():
    spec = _spec(lower="<")
    spec["groups"][1]["rule"]["value"] = 60
    spec["groups"].append({"id": "g3", "label": "57–<61", "rule": "otherwise"})
    assert _compile(spec).reason == "exposure_group_label_rule_mismatch"


def test_otherwise_cannot_add_an_empty_level_to_an_exhaustive_partition():
    spec = _spec(lower="<")
    spec["groups"].append({"id": "g3", "label": "remaining", "rule": "otherwise"})
    assert _compile(spec).reason == "exposure_group_rule_unreachable"


def test_distinct_groupings_are_checked_separately():
    first = _spec(lower="<")
    second = deepcopy(first)
    second["id"] = "x2"
    for group in second["groups"]:
        group["rule"]["value"] = 70
    result = compile_exposure_groupings(
        read_stated_exposure_groupings({"groupings": [first, second]}), _context(_AGE),
    )
    assert len(result.applied) == 2


def test_overlapping_cumulative_bands_are_not_silently_interpreted_as_between_rules():
    spec = _spec(lower="<")
    spec["groups"].insert(1, _group("g3", "middle", "<", 70))
    assert _compile(spec).reason == "exposure_group_rules_overlap"


@pytest.mark.parametrize("strict_schema", [False, True])
def test_structured_retry_reports_the_exact_conflict_then_accepts_the_correction(strict_schema):
    bad = _spec()
    good = deepcopy(bad)
    good["groups"][0]["rule"]["op"] = "<"
    client = ScriptedMockLLMClient([json.dumps({"groupings": [spec]}) for spec in (bad, good)])
    client.supports_strict_json_schema = strict_schema
    context = _context(_AGE).model_copy(update={"research_question": bad["quote"]})
    answer = ask_exposure_groupings(client, context=context, sources=[])
    retry = client.calls[1][0][-1].content
    assert all(text in retry for text in ("exposure_group_rules_overlap", "x1", "g1", "g2", "57", "younger", "older"))
    assert answer.groupings.groupings[0].groups[0].rule.op == "<"
    assert "below X" in EXPOSURE_GROUPING_GUIDE and "strict < X" in EXPOSURE_GROUPING_GUIDE


def test_assignment_stops_when_different_summaries_match_the_same_row():
    parameters = _compile(_spec(lower="<")).derivation()
    parameters["scale"] = "nominal"
    parameters["window"] = {"start_hours": 0, "end_hours": 24}
    parameters["groups"][0]["rule"].update(summary="min")
    parameters["groups"][1]["rule"].update(summary="max")
    parameters["source_columns"] = {"min": "min", "max": "max"}
    with pytest.raises(ExposureGroupPartitionError) as caught:
        evaluate_grouping(read_grouping_rules(parameters), pa.table({"min": [56], "max": [58]}))
    assert caught.value.reason == "exposure_group_rules_overlap"
    assert caught.value.groups == ("g1", "g2")
