from __future__ import annotations

import copy
import json

import pytest

from easyicu.research_agent.research_context.outbound import compact_variable_field_names


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
    table = compact.pop("variables_table")
    compact["variables"] = [dict(zip(table["column_sets"][index], values)) for index, values in table["rows"]]
    assert compact == before
    assert payload == before
    assert len(json.dumps(compact_variable_field_names(payload))) < len(json.dumps(payload))


@pytest.mark.parametrize("payload", [{}, {"variables": []}, {"variables": [{"name": "age"}]},
                                     {"variables": [None]}, {"variables": [{"name": "age"}], "variables_table": "reserved"}])
def test_small_or_incompatible_context_stays_unchanged(payload):
    assert compact_variable_field_names(payload) == payload
