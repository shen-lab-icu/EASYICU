from __future__ import annotations

import copy
import json

import pytest

from easyicu.research_agent.research_context.outbound import compact_variable_field_names


def _decode(table: dict) -> list[dict]:
    return [
        {"name": name, **dict(zip(table["column_sets"][index], values))}
        for index, names, values in table["rows"]
        for name in names
    ]


def test_writer_transport_roundtrips_distinct_variable_contracts_and_order():
    variables = [
        {"name": f"column_{i}", "clinical_definition": f"Original definition {i}",
         "analysis_window": {"start_hours": -6, "end_hours": 24},
         "missingness_semantics": "unknown, never a negative finding",
         "observation_semantics": {"source": "recorded", "eligible": None},
         "allowed_aggregations": ["max", "first"], "unit": "mmol/L"}
        for i in range(50)
    ]
    variables[1].pop("unit")
    variables[2]["unit"] = None
    variables[3]["allowed_aggregations"] = []
    payload = {"research_question": "Original question", "variables": variables, "preferences": {"adjust_for": ["age"]}}
    before = copy.deepcopy(payload)
    compact = json.loads(json.dumps(compact_variable_field_names(payload)))
    compact["variables"] = _decode(compact.pop("variables_table"))
    assert compact == before
    assert payload == before
    assert len(json.dumps(compact_variable_field_names(payload))) < len(json.dumps(payload))


def _first_day_panel() -> list[dict]:
    """Each first-day concept summarised four ways, with its count and status."""

    window = {"analysis_window": "icu_admission[0,24]h", "analysis_window_role": "outer_observation_window"}
    rows: list[dict] = []
    for concept, unit, low, high in (
        ("lactate", "mmol/L", 0.0, 30.0),
        ("creatinine", "mg/dL", 0.1, 15.0),
        ("platelets", "10^9/L", 1.0, 1500.0),
    ):
        rows += [
            {"name": f"{concept}_{summary}", "dtype": "float32", "role": "lab", "source_concept": concept,
             "unit": unit, "plausibility_range": [low, high], "allowed_aggregations": ["median_only", "first_value"],
             **window}
            for summary in ("min", "max", "mean", "first")
        ]
        rows += [
            {"name": f"{concept}_n", "dtype": "float64", "role": "meta",
             "materialized_representation": "window_nonnull_count", **window},
            {"name": f"{concept}_measured", "dtype": "int64", "role": "meta",
             "materialized_representation": "window_measurement_status", **window},
        ]
    return rows


def test_variables_that_differ_only_by_name_share_one_row():
    variables = _first_day_panel()
    # A first value drawn before ICU admission is a different variable contract.
    creatinine_first = next(row for row in variables if row["name"] == "creatinine_first")
    creatinine_first["analysis_window"] = "hospital_admission[-24,0]h"
    before = copy.deepcopy(variables)

    table = json.loads(json.dumps(compact_variable_field_names({"variables": variables})))["variables_table"]

    assert variables == before
    assert sorted(_decode(table), key=lambda row: row["name"]) == sorted(before, key=lambda row: row["name"])
    assert [names for _index, names, _values in table["rows"]] == [
        ["lactate_min", "lactate_max", "lactate_mean", "lactate_first"],
        ["lactate_n", "creatinine_n", "platelets_n"],
        ["lactate_measured", "creatinine_measured", "platelets_measured"],
        ["creatinine_min", "creatinine_max", "creatinine_mean"],
        ["creatinine_first"],
        ["platelets_min", "platelets_max", "platelets_mean", "platelets_first"],
    ]


@pytest.mark.parametrize("payload", [{}, {"variables": []}, {"variables": [{"name": "age"}]},
                                     {"variables": [None]}, {"variables": [{"name": "age"}], "variables_table": "reserved"},
                                     {"variables": [{"unit": "mmHg"}] * 40}])
def test_small_or_incompatible_context_stays_unchanged(payload):
    assert compact_variable_field_names(payload) == payload
