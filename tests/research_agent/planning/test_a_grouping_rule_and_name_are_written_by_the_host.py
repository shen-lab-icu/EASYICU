"""A grouping's rule and its name are written by the host, from its parameters.

A grouped exposure's column holds level codes.  Whoever names a level for a
reader -- the plan a reviewer approves, a figure, the Methods -- reads its
rule and its grouping's name from one owner, the grouping contract, which
writes them in English from the grouping's derivation parameters: the
measure and its window, the threshold, and the groups matched before it, as
the host matches them.  A planned column now carries those parameters and
the study's words for each level, so a context planned on metadata alone
can state them; a column declared before columns carried them is still read,
and states neither.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.research_agent.contracts.exposure_group_rules import (
    ExposureGroupRuleError,
    concept_reader_name,
    grouping_reader_labels,
    grouping_rule_texts,
    planned_group_column,
    read_planned_group_columns,
    reads_as_markup,
)


def _parameters(**changes: Any) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "concept": "glu",
        "window": {"start_hours": 0, "end_hours": 24},
        "scale": "nominal",
        "groups": [
            {
                "id": "g1",
                "rule": {"summary": "min", "op": "<", "value": 70, "unit": "mg/dL"},
            },
            {
                "id": "g2",
                "rule": {"summary": "max", "op": ">=", "value": 180.5},
            },
            {"id": "g3", "rule": "otherwise"},
        ],
        "unmeasured": "own_group",
        "source_columns": {"min": "glu_min", "max": "glu_max"},
    }
    parameters.update(changes)
    return parameters


_WORDS = {
    "g1": "hypoglycaemia",
    "g2": "hyperglycaemia only",
    "g3": "neither",
    "gU": "no glucose on day one",
}


def test_each_levels_rule_is_written_as_the_host_matches_it() -> None:
    measure = "Glucose over 0-24 h after ICU admission"

    assert grouping_rule_texts(_parameters()) == {
        "g1": f"{measure}: minimum < 70 mg/dL",
        "g2": f"{measure}: maximum ≥ 180.5, without minimum < 70 mg/dL",
        "g3": f"{measure}: neither minimum < 70 mg/dL nor maximum ≥ 180.5",
        "gU": f"{measure}: no value recorded",
    }


def test_a_grouping_without_a_window_or_an_unmeasured_level_says_so() -> None:
    texts = grouping_rule_texts(
        _parameters(
            window=None,
            groups=[
                _parameters()["groups"][0],
                {"id": "g2", "rule": "otherwise"},
            ],
            unmeasured="exclude",
            source_columns={"min": "glu_min"},
        )
    )

    assert texts == {
        "g1": "Glucose, recorded once per stay: minimum < 70 mg/dL",
        "g2": "Glucose, recorded once per stay: not minimum < 70 mg/dL",
    }


def test_parameters_that_decide_no_level_name_no_rule() -> None:
    with pytest.raises(ExposureGroupRuleError):
        grouping_rule_texts(_parameters(unmeasured="drop"))


@pytest.mark.parametrize(
    ("text", "markup"),
    [
        ("Glucose: minimum < 70 mg/dL", False),
        ("Glucose: maximum ≥ 180 mmol/L", False),
        ("Glucose: minimum < 70 mg_dL", True),
        ("Glucose: minimum < 70 *", True),
        ("{evidence:research_context}", True),
        ("Glucose `code`", True),
        ("a [link]", True),
        ("two\nlines", True),
    ],
)
def test_text_a_reader_sentence_would_read_as_markup_is_told(
    text: str, markup: bool
) -> None:
    assert reads_as_markup(text) is markup


@pytest.mark.parametrize(
    ("concept", "name"),
    [
        ("glu", "Glucose"),
        ("map", "Mean Arterial Pressure"),
        # The catalog names it with a character a reader sentence may not
        # carry: its id, read as words.
        ("svo2", "svo2"),
        ("not_in_the_catalog", "not in the catalog"),
    ],
)
def test_a_concept_goes_by_its_catalog_name_or_its_id_as_words(
    concept: str, name: str
) -> None:
    assert concept_reader_name(concept) == name


def _window(start: float, end: float) -> dict[str, Any]:
    return {"window": {"start_hours": start, "end_hours": end}}


def test_a_grouping_goes_by_its_concepts_name_then_its_window_then_its_column() -> (
    None
):
    names = grouping_reader_labels(
        [
            {"variable": "lact_group_x1", "concept": "lact"},
            {
                "variable": "glu_group_x2",
                "concept": "glu",
                "parameters": _parameters(**_window(0, 24)),
            },
            {
                "variable": "glu_group_x3",
                "concept": "glu",
                "parameters": _parameters(**_window(24, 48.5)),
            },
            {
                "variable": "map_group_x4",
                "concept": "map",
                "parameters": _parameters(**_window(0, 24)),
            },
            {
                "variable": "map_group_x5",
                "concept": "map",
                "parameters": _parameters(**_window(0, 24)),
            },
            {"variable": "crea_group_x6", "concept": "crea"},
            {"variable": "crea_group_x7", "concept": "crea"},
        ]
    )

    assert names == {
        "lact_group_x1": "Lactate",
        "glu_group_x2": "Glucose 0-24 h",
        "glu_group_x3": "Glucose 24-48.5 h",
        # The same window again, or no parameters to read one from: the
        # column, read as words, is the one name each holds alone.
        "map_group_x4": "map group x4",
        "map_group_x5": "map group x5",
        "crea_group_x6": "crea group x6",
        "crea_group_x7": "crea group x7",
    }


def _planned(**changes: Any) -> dict[str, Any]:
    record = planned_group_column(
        _parameters(),
        variable="glu_group_x1",
        labels=_WORDS,
        compared={"reference": 3, "contrast": 1},
        groupings_record_sha256="a" * 64,
    ).record()
    record.update(changes)
    return record


def test_a_planned_column_carries_its_levels_words_and_parameters() -> None:
    declared = _planned()

    assert declared["labels"] == [
        {"code": 1, "group": "g1", "label": "hypoglycaemia"},
        {"code": 2, "group": "g2", "label": "hyperglycaemia only"},
        {"code": 3, "group": "g3", "label": "neither"},
        {"code": 4, "group": "gU", "label": "no glucose on day one"},
    ]
    assert declared["parameters"] == _parameters()
    (read,) = read_planned_group_columns([declared])
    assert read.record() == declared
    assert read.parameters == _parameters()


def test_a_column_declared_before_columns_carried_them_is_read_as_written() -> None:
    legacy = {
        key: value
        for key, value in _planned().items()
        if key not in ("labels", "parameters")
    }

    (read,) = read_planned_group_columns([legacy])

    assert read.record() == legacy
    assert (read.labels, read.parameters) == (None, None)


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param(
            {k: v for k, v in _planned().items() if k != "parameters"},
            id="words_without_parameters",
        ),
        pytest.param(
            {k: v for k, v in _planned().items() if k != "labels"},
            id="parameters_without_words",
        ),
        pytest.param(_planned(concept="lact"), id="another_concept"),
        pytest.param(
            _planned(window={"start_hours": 0, "end_hours": 48}), id="another_window"
        ),
        pytest.param(
            _planned(transform="exposure_group_ordinal"), id="another_scale"
        ),
        pytest.param(
            _planned(parameters=_parameters(unmeasured="exclude")),
            id="another_level_count",
        ),
        pytest.param(
            _planned(
                labels=[
                    {"code": 1, "group": "g2", "label": "hyperglycaemia only"},
                    {"code": 2, "group": "g1", "label": "hypoglycaemia"},
                    {"code": 3, "group": "g3", "label": "neither"},
                    {"code": 4, "group": "gU", "label": "no glucose on day one"},
                ]
            ),
            id="levels_in_another_order",
        ),
        pytest.param(
            _planned(parameters=_parameters(scale="ordinal")),
            id="parameters_of_another_scale",
        ),
    ],
)
def test_a_planned_column_whose_parameters_form_another_column_is_refused(
    declared: dict[str, Any],
) -> None:
    with pytest.raises(ExposureGroupRuleError):
        read_planned_group_columns([declared])


@pytest.mark.parametrize("code", [True, 1.0, "1"])
def test_a_level_code_in_metadata_is_an_integer(code) -> None:
    record = _planned()
    record["labels"][0]["code"] = code
    with pytest.raises(ExposureGroupRuleError):
        read_planned_group_columns([record])


def test_an_unmeasured_level_is_not_a_planned_contrast() -> None:
    record = _planned(compared={"reference": 4, "contrast": 1})
    with pytest.raises(ExposureGroupRuleError, match="never its unmeasured level"):
        read_planned_group_columns([record])
