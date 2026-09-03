"""A bound input's descriptor must name the columns it points at.

MEASURED (e3 KDIGO, ``04_stage_stratified_outcome_figure``, 9 of 10 steps
complete): the step consumed ``table:absolute_risk_context``.  The host's own
deterministic runner produces that product as a GROUPED summary -- one row per
group, the stratum carried as a value in ``group_value`` with ``group_type``
naming what was grouped.  The generated script assumed the stratum would be a
column named after the clinical variable, and killed itself::

    RuntimeError: Cannot render a stage-stratified outcome figure: the bound
    typed product lacks the required stage coordinate or fields
    ['aki_stage_max']. The product must contain one row per stage with
    aki_stage_max, outcome_risk, ci_low, ci_high, n, and event_n.

Five of its six required fields were present.  The real column list was one
lookup away in ``EASYICU_RESOLVED_INPUTS_JSON`` -- which the input descriptor
pointed at while saying nothing about what was in it.

These tests lock the descriptor naming the columns, and lock the two ways the
fix could rot: dropping the list, or truncating a long one into something that
still reads as the whole schema.
"""

from __future__ import annotations

import json

from easyicu.research_agent.resources.coder import (
    _MAX_PROJECTED_COLUMNS,
    _data_resources,
)

# The exact contract the host's deterministic runner declared for the measured
# step, columns verbatim.
ABSOLUTE_RISK_COLUMNS = [
    "exposure",
    "group_type",
    "group_value",
    "label",
    "n_denominator",
    "source_measured_column",
    "source_count_column",
    "median",
    "q25",
    "q75",
    "minimum",
    "maximum",
    "estimate_type",
    "n",
    "n_positive",
    "event_n",
    "prevalence",
    "prevalence_pct",
    "outcome_risk",
    "outcome_risk_pct",
    "estimate",
    "ci_low",
    "ci_high",
]


def _binding(columns=None, *, sha: str = "a" * 64):
    binding = {
        "evidence_id": "absolute_risk_context_execute",
        "sha256": sha,
        "identity_row": {"sha256": sha},
    }
    if columns is not None:
        binding["product_contract"] = {"columns": list(columns), "row_count": 6}
    return {"table:absolute_risk_context": binding}


def _projection(bindings):
    resources = _data_resources(bindings)
    assert len(resources) == 1
    return json.loads(resources[0].prompt_projection)


def test_the_descriptor_names_the_bound_columns() -> None:
    payload = _projection(_binding(ABSOLUTE_RISK_COLUMNS))
    assert payload["columns"] == ABSOLUTE_RISK_COLUMNS
    # The measured script invented this name; the real grouping columns are
    # right here and it never had to.
    assert "aki_stage_max" not in payload["columns"]
    assert {"group_type", "group_value"} <= set(payload["columns"])


def test_the_descriptor_still_points_at_the_authoritative_record() -> None:
    payload = _projection(_binding(ABSOLUTE_RISK_COLUMNS))
    # The list is a convenience beside the digest, not a replacement for the
    # record the script verifies.
    assert payload["access"] == "EASYICU_RESOLVED_INPUTS_JSON"
    assert payload["sha256"] == "a" * 64
    assert payload["input_key"] == "table:absolute_risk_context"


def test_an_input_without_a_declared_contract_is_unchanged() -> None:
    payload = _projection(_binding(None))
    assert "columns" not in payload
    assert "column_count" not in payload
    assert set(payload) == {"input_key", "evidence_id", "sha256", "access"}


def test_an_empty_column_list_publishes_nothing() -> None:
    payload = _projection(_binding([]))
    assert "columns" not in payload and "column_count" not in payload


def test_a_wide_product_is_counted_not_truncated() -> None:
    """A partial list would read as the whole schema.

    That is the same failure as publishing an empty list under a sentence
    promising completeness: it answers the agent's question wrongly with the
    host's authority instead of sending it to the record.
    """

    wide = [f"trajectory_sofa_h{i}_{i + 6}" for i in range(_MAX_PROJECTED_COLUMNS + 5)]
    payload = _projection(_binding(wide))
    assert "columns" not in payload
    assert payload["column_count"] == len(wide)
    assert "EASYICU_RESOLVED_INPUTS_JSON" in payload["columns_note"]
    # No subset of the real names may appear, or the agent will read it as the
    # schema and pick from it.
    rendered = json.dumps(payload)
    assert not any(name in rendered for name in wide)
