"""An exposure grouping is stated as typed rules, and the host reads them.

A study that compares stays by where a measurement falls needs a closed-domain
exposure no export column holds.  The Planner states the groups as threshold
rules over one concept's summary, matched in the order listed, with
``otherwise`` for the measured stays no earlier group took and a choice for
the stays without a measurement.  The owner refuses a grouping that cannot
work as stated: a group no stay can reach, values no group takes, an ordinal
scale its ids do not follow.  The host then reads every rule as it reads a
population measurement, and a grouping it cannot read waits for an
extraction or is not applied, with a stable code.

Synthetic values throughout: a glucose grouping by the minimum and maximum
of the first day, and two groupings of a once-per-stay body mass index, one
coarse and one fine, each with a reference in the middle of its scale.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from easyicu.research_agent.planning.exposure_group_compile import (
    compile_exposure_groupings,
)
from easyicu.research_agent.planning.exposure_group_spec import (
    ExposureGroupingsRefused,
    grouping_labels,
    grouping_levels,
    read_stated_exposure_groupings,
    unquoted_groupings,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
)

_DAY = {"start_hours": 0, "end_hours": 24}


def _rule(summary: str, op: str, value: float, unit: str | None = None) -> dict:
    return {
        "summary": summary,
        "op": op,
        "value": value,
        **({"unit": unit} if unit else {}),
    }


def _group(group_id: str, label: str, rule: Any) -> dict:
    return {"id": group_id, "label": label, "rule": rule}


def _glucose(**changes: Any) -> dict:
    spec = {
        "id": "x1",
        "concept": "glu",
        "window": _DAY,
        "scale": "nominal",
        "groups": [
            _group("g1", "hypoglycaemia", _rule("min", "<", 70, "mg/dL")),
            _group("g2", "hyperglycaemia only", _rule("max", ">", 180, "mg/dL")),
            _group("g3", "normoglycaemia", "otherwise"),
        ],
        "unmeasured": {"handling": "own_group", "label": "no glucose on day one"},
        "quote": "glucose below 70 or above 180",
        "source": "question",
    }
    return {**spec, **changes}


def _coarse_bmi(**changes: Any) -> dict:
    spec = {
        "id": "x2",
        "concept": "bmi",
        "window": None,
        "scale": "ordinal",
        # Matched in this order; the ids follow the scale.
        "groups": [
            _group("g1", "under 18.5", _rule("value", "<", 18.5)),
            _group("g4", "30 or more", _rule("value", ">=", 30)),
            _group("g3", "25 to 29.9", _rule("value", ">=", 25)),
            _group("g2", "18.5 to 24.9", "otherwise"),
        ],
        "unmeasured": {"handling": "exclude"},
        "reference": "g2",
        "contrast": "g4",
        "quote": "body mass index category",
        "source": "outline",
    }
    return {**spec, **changes}


def _fine_bmi() -> dict:
    return _coarse_bmi(
        id="x3",
        groups=[
            _group("g1", "under 18.5", _rule("value", "<", 18.5)),
            _group("g6", "40 or more", _rule("value", ">=", 40)),
            _group("g5", "35 to 39.9", _rule("value", ">=", 35)),
            _group("g4", "30 to 34.9", _rule("value", ">=", 30)),
            _group("g3", "25 to 29.9", _rule("value", ">=", 25)),
            _group("g2", "18.5 to 24.9", "otherwise"),
        ],
    )


def _read(*specs: dict):
    return read_stated_exposure_groupings({"groupings": list(specs)})


def _refusal(*specs: dict) -> str:
    with pytest.raises(ExposureGroupingsRefused) as refused:
        _read(*specs)
    return " ".join(item["msg"] for item in refused.value.errors)


# -- what a grouping states ---------------------------------------------------


def test_a_nominal_grouping_by_two_summaries_states_its_levels() -> None:
    (glucose,) = _read(_glucose()).groupings

    assert grouping_levels(glucose) == ("g1", "g2", "g3", "gU")
    assert grouping_labels(glucose)["gU"] == "no glucose on day one"


def test_a_study_states_a_coarse_and_a_fine_ordinal_grouping_of_one_value() -> None:
    coarse, fine = _read(_coarse_bmi(), _fine_bmi()).groupings

    assert grouping_levels(coarse) == ("g1", "g2", "g3", "g4")
    assert grouping_levels(fine) == ("g1", "g2", "g3", "g4", "g5", "g6")
    # The reference sits in the middle of each scale.
    assert coarse.reference == fine.reference == "g2"
    assert coarse.contrast == "g4"


@pytest.mark.parametrize(
    ("groups", "found"),
    [
        pytest.param(
            [
                _group("g1", "low", _rule("min", "<", 70)),
                _group("g2", "lower", _rule("min", "<", 60)),
                _group("g3", "rest", "otherwise"),
            ],
            "exposure_group_rule_unreachable",
            id="taken-by-an-earlier-group",
        ),
        pytest.param(
            [
                _group("g1", "low", _rule("min", "<", 70)),
                # A maximum below 70 has a minimum below 70: g1 took it.
                _group("g2", "all low", _rule("max", "<", 70)),
                _group("g3", "rest", "otherwise"),
            ],
            "exposure_group_rule_unreachable",
            id="taken-through-min-below-max",
        ),
        pytest.param(
            [
                _group("g1", "low", _rule("min", "<", 70)),
                _group("g2", "not low", _rule("min", ">=", 70)),
                _group("g3", "rest", "otherwise"),
            ],
            "exposure_group_rule_unreachable",
            id="otherwise-takes-none",
        ),
        pytest.param(
            [
                _group("g1", "low", _rule("min", "<", 70)),
                _group("g2", "high", _rule("max", ">", 180)),
            ],
            "exposure_group_rules_not_exhaustive",
            id="values-no-group-takes",
        ),
        pytest.param(
            [
                _group("g1", "below 70", _rule("min", "<", 70)),
                _group("g2", "above 70", _rule("min", ">", 70)),
            ],
            "exposure_group_rules_not_exhaustive",
            id="one-value-no-group-takes",
        ),
        pytest.param(
            [
                _group("g1", "rest", "otherwise"),
                _group("g2", "high", _rule("max", ">", 180)),
            ],
            "otherwise takes the measured stays",
            id="otherwise-before-a-rule",
        ),
        pytest.param(
            [
                _group("g1", "low", _rule("min", "<", 70, "mg/dL")),
                _group("g2", "high", _rule("max", ">", 10, "mmol/L")),
                _group("g3", "rest", "otherwise"),
            ],
            "exposure_group_units_differ",
            id="units-differ",
        ),
    ],
)
def test_a_grouping_that_cannot_work_as_stated_is_refused(
    groups: list, found: str
) -> None:
    assert found in _refusal(_glucose(groups=groups))


def test_touching_bounds_of_one_summary_leave_one_value() -> None:
    _read(
        _glucose(
            groups=[
                _group("g1", "above 70", _rule("min", ">", 70)),
                _group("g2", "below 70", _rule("min", "<", 70)),
                # A minimum of exactly 70 remains, and it is a value a stay holds.
                _group("g3", "exactly 70", "otherwise"),
            ]
        )
    )


def test_an_ordinal_grouping_numbers_its_groups_along_its_scale() -> None:
    misnumbered = _coarse_bmi(
        groups=[
            _group("g2", "under 18.5", _rule("value", "<", 18.5)),
            _group("g1", "18.5 or more", "otherwise"),
        ],
        reference=None,
        contrast=None,
    )

    assert "g1 lowest" in _refusal(misnumbered)


def test_an_ordinal_grouping_reads_one_summary() -> None:
    assert "group a mixture of summaries as nominal" in _refusal(
        _glucose(scale="ordinal")
    )


@pytest.mark.parametrize(
    ("changes", "found"),
    [
        pytest.param(
            {
                "groups": [
                    _group("g1", "under 18.5", _rule("first", "<", 18.5)),
                    _group("g2", "rest", "otherwise"),
                ]
            },
            "write its rules with summary value",
            id="no-window-but-a-window-summary",
        ),
        pytest.param(
            {"window": _DAY},
            "summary as min, max, mean or first",
            id="a-window-but-a-once-per-stay-value",
        ),
        pytest.param(
            {"reference": "g5"}, "the reference names one of the groups", id="reference"
        ),
        pytest.param(
            {"contrast": "g2"},
            "the contrast names another group than the reference",
            id="contrast-is-the-reference",
        ),
        pytest.param(
            {"reference": None, "contrast": "g4"},
            "the contrast names another group than the reference",
            id="contrast-without-reference",
        ),
        pytest.param(
            {
                "groups": [
                    _group("g1", "same", _rule("value", "<", 18.5)),
                    _group("g2", "Same", "otherwise"),
                ]
            },
            "labels must be distinct",
            id="labels",
        ),
    ],
)
def test_a_grouping_states_its_window_reference_and_labels_consistently(
    changes: dict, found: str
) -> None:
    assert found in _refusal(_coarse_bmi(**changes))


def test_a_study_states_distinct_groupings_and_no_more_than_three() -> None:
    assert "two exposure groupings state the same groups" in _refusal(
        _coarse_bmi(), _coarse_bmi(id="x3")
    )
    assert "ids must be unique" in _refusal(_coarse_bmi(), _fine_bmi() | {"id": "x2"})
    four = [_glucose(id="x1"), _coarse_bmi(), _fine_bmi(), _glucose(id="x2")]
    _refusal(*four)


def test_a_quote_citing_the_study_is_held_to_its_words() -> None:
    question = "Do outcomes differ by GLUCOSE below 70   or above 180 in the first day?"
    stated = _read(_glucose(), _coarse_bmi())

    # Spacing and case are not words; the outline cites no text of the study's.
    assert unquoted_groupings(stated, [question]) == ()
    paraphrased = _read(_glucose(quote="low blood sugar"))
    assert [item.id for item in unquoted_groupings(paraphrased, [question])] == ["x1"]


# -- what the host reads ------------------------------------------------------


def _lab(
    name: str,
    summary: str,
    *,
    unit: str = "mg/dL",
    window: str = "icu_admission[0,24]h",
) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        role=VariableRole.LAB,
        dtype="float64",
        unit=unit,
        source_concept="glu",
        unit_normalization=f"window_numeric_{summary}",
        analysis_window=window,
        valid_range=[0, 1000],
    )


_BMI = ConceptDescriptor(
    name="bmi",
    role=VariableRole.DEMOGRAPHIC,
    dtype="float64",
    unit="kg/m2",
    source_concept="bmi",
    unit_normalization="stay_level_unique_value",
    valid_range=[8, 70],
)


def _context(*extra: ConceptDescriptor, hours: float = 24.0) -> ResearchContext:
    base = _planner_context()
    constraints = json.dumps(
        {
            "materialization_window": {
                "role": "outer_observation_window",
                "anchor": "ICU admission",
                "hours": hours,
            }
        }
    )
    return base.model_copy(
        update={"variables": [*base.variables, *extra], "data_constraints": constraints}
    )


def _compiled(context: ResearchContext, *specs: dict):
    return compile_exposure_groupings(_read(*specs), context).groupings


def test_the_host_reads_every_rule_and_declares_one_variable_per_grouping() -> None:
    context = _context(_lab("glu_min", "min"), _lab("glu_max", "max"), _BMI)

    glucose, coarse, fine = _compiled(context, _glucose(), _coarse_bmi(), _fine_bmi())

    assert (glucose.disposition, glucose.variable) == ("applied", "glu_group_x1")
    assert dict(glucose.source_columns) == {"min": "glu_min", "max": "glu_max"}
    assert (coarse.variable, fine.variable) == ("bmi_group_x2", "bmi_group_x3")
    assert dict(coarse.source_columns) == {"value": "bmi"}
    record = coarse.record()
    assert record["levels"] == ["g1", "g2", "g3", "g4"]
    assert record["reference"] == "g2"
    assert record["derivation"]["unmeasured"] == "exclude"


def test_the_derivation_digest_follows_the_levels_not_the_wording() -> None:
    context = _context(_BMI)
    (stated,) = _compiled(context, _coarse_bmi())
    (requoted,) = _compiled(context, _coarse_bmi(quote="BMI class at admission"))
    moved = _coarse_bmi()
    moved["groups"] = [
        *moved["groups"][:2],
        _group("g3", "26 to 29.9", _rule("value", ">=", 26)),
        moved["groups"][3],
    ]
    (rethresholded,) = _compiled(context, moved)

    assert stated.derivation_sha256() == requoted.derivation_sha256()
    assert stated.derivation_sha256() != rethresholded.derivation_sha256()


@pytest.mark.parametrize(
    ("extra", "hours", "disposition", "reason"),
    [
        pytest.param(
            (_BMI,),
            24.0,
            "requires_extraction",
            "exposure_group_concept_not_in_export",
            id="concept-not-in-export",
        ),
        pytest.param(
            (
                _lab("glu_min", "min", window="icu_admission[0,48]h"),
                _lab("glu_max", "max", window="icu_admission[0,48]h"),
            ),
            48.0,
            "requires_extraction",
            "exposure_group_source_column_unavailable",
            id="summarized-over-another-window",
        ),
        pytest.param(
            (
                _lab("glu_min", "min", unit="mmol/L"),
                _lab("glu_max", "max", unit="mmol/L"),
            ),
            24.0,
            "not_applied",
            "exposure_group_unit_mismatch",
            id="another-unit",
        ),
    ],
)
def test_a_grouping_the_input_cannot_read_waits_or_is_not_applied(
    extra: tuple, hours: float, disposition: str, reason: str
) -> None:
    (glucose,) = _compiled(_context(*extra, hours=hours), _glucose())

    assert (glucose.disposition, glucose.reason) == (disposition, reason)
    assert glucose.variable is None


@pytest.mark.parametrize("threshold", [5.0, 8.0, 70.0])
def test_a_threshold_the_value_cannot_take_on_both_sides_is_not_applied(
    threshold: float,
) -> None:
    spec = _coarse_bmi(
        groups=[
            _group("g1", "low", _rule("value", "<", threshold)),
            _group("g2", "rest", "otherwise"),
        ],
        reference=None,
        contrast=None,
    )

    (grouping,) = _compiled(_context(_BMI), spec)

    assert grouping.reason == "exposure_group_threshold_outside_domain"


def test_a_variable_name_the_input_holds_is_not_taken_again() -> None:
    taken = _BMI.model_copy(update={"name": "bmi_group_x2"})

    (grouping,) = _compiled(_context(_BMI, taken), _coarse_bmi())

    assert grouping.reason == "exposure_group_name_taken"


def test_a_once_per_stay_rule_is_not_read_from_a_windowed_summary() -> None:
    windowless = _glucose(
        window=None,
        scale="ordinal",
        groups=[
            _group("g1", "low", _rule("value", "<", 70, "mg/dL")),
            _group("g2", "rest", "otherwise"),
        ],
    )

    (grouping,) = _compiled(_context(_lab("glu_first", "first")), windowless)

    assert (grouping.disposition, grouping.reason) == (
        "requires_extraction",
        "exposure_group_source_column_unavailable",
    )
