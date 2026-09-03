"""The gate demanded a remedy and then blocked the remedy.

``mechanical_code_preflight`` refuses "column discovery falls back to an
arbitrary frame-order column after named candidates fail; fail closed on the
missing declared schema field instead."  The defect it names is *frame order*:
whichever column happens to sit first in the DataFrame.

The second detector branch tested that with a name suffix --
``_call_name(node.value).endswith("columns")`` -- which matches every local
list whose identifier merely ends that way.

Measured 2026-08-02 over 2,136 recorded generated scripts: that branch fired 3
times and caught the defect 0 times.  All three were the same shape, and it is
the shape the message asks for -- read a declared list, assert it holds exactly
one entry, raise otherwise, then index it:

    id_columns = cohort_context.get("id_columns")
    if not isinstance(id_columns, list) or len(id_columns) != 1:
        raise ValueError("This step requires one stable cohort key")
    key_column = id_columns[0]

The third instance is the most direct: it filters named candidates against the
frame and raises unless exactly one survives -- literally "fail closed when the
named candidates fail".  The block cost canary42's E3 step 04 two LLM repairs
and then killed it, which took step 05 down with it as a dependency.

Also measured: 0 recorded scripts bind frame order to a local name and then
index that name, so reading the expression instead of the identifier gives up
no reachable coverage.
"""

from __future__ import annotations

import ast

import pytest

from easyicu.research_agent.gates.preflight import (
    _function_arbitrary_column_fallback,
    audit_mechanical_code_contracts,
)
from easyicu.research_agent.schema import AnalysisStep

_REASON = "arbitrary_column_fallback"


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="audit",
        intent="Audit a typed product against its declared contract.",
        inputs=["table:cohort_flow"],
        expected_outputs=["table:cohort_flow"],
        method="descriptive_summary",
    )


def _fires(code: str) -> bool:
    """Ask the host's own detector, at the level a step is actually blocked."""

    return any(
        (finding.detail or {}).get("reason") == _REASON
        for finding in audit_mechanical_code_contracts(code, _step())
    )


def _fires_on_function(code: str) -> bool:
    tree = ast.parse(code)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return _function_arbitrary_column_fallback(function) is not None


# ---------------------------------------------------------------------------
# The three shapes that were really blocked, transcribed from the run records
# ---------------------------------------------------------------------------

#: canary42 e3 steps/04_absolute_risk_context/.quarantine/concept_draft.py:268-272
_RECORDED_ID_COLUMNS = """
def main():
    cohort_context = research_context.get("cohort", {})
    id_columns = cohort_context.get("id_columns")
    if not isinstance(id_columns, list) or len(id_columns) != 1:
        raise ValueError("This step requires one stable cohort key")
    key_column = id_columns[0]
    return key_column
"""

#: canary32 e3 steps/07_ordinal_trend_audit/.quarantine/concept_draft.py:266-274
_RECORDED_KEY_COLUMNS = """
def main():
    key_columns = product_contract.get("key_columns")
    if not isinstance(key_columns, list) or len(key_columns) != 1:
        raise ValueError(
            "Typed cohort product_contract.key_columns must declare exactly one key"
        )
    key_column = key_columns[0]
    if key_column not in typed_frame.columns:
        raise ValueError(f"Typed cohort key is absent: {key_column}")
    return key_column
"""

#: fresh26 e1 steps/05_missingness_event_timing_audit/.quarantine/
#: concept_draft.py:575-597 -- named candidates, then fail closed.
_RECORDED_NAMED_CANDIDATES = """
def summarise(table, label_candidates):
    label_columns = [
        name
        for name in label_candidates
        if name in table.columns
    ]
    if len(label_columns) != 1:
        raise RuntimeError("Binary-event status table has no unique named status column")
    label_column = label_columns[0]
    return label_column
"""

_RECORDED = pytest.mark.parametrize(
    "code",
    [_RECORDED_ID_COLUMNS, _RECORDED_KEY_COLUMNS, _RECORDED_NAMED_CANDIDATES],
    ids=["id_columns", "key_columns", "named_candidates"],
)


@_RECORDED
def test_a_declared_list_guarded_by_exactly_one_is_not_a_fallback(code: str):
    """Every one of these raises when the declared field is missing."""

    assert _fires_on_function(code) is False
    assert _fires(code) is False


@_RECORDED
def test_the_blocked_scripts_really_do_fail_closed(code: str):
    """Guards the premise, not the gate: these transcripts must contain a raise.

    If a future transcription drops the assertion, the shape stops being the
    remedy and this file would be arguing for something it no longer shows.
    """

    tree = ast.parse(code)
    raises = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)]
    assert raises, "the recorded shape is only defensible because it fails closed"


# ---------------------------------------------------------------------------
# The defect itself must still be refused
# ---------------------------------------------------------------------------


def test_the_frames_own_first_column_is_still_refused():
    code = """
def choose(frame):
    return frame.columns[0]
"""
    assert _fires_on_function(code) is True
    assert _fires(code) is True


def test_a_dtype_selection_first_column_is_still_refused():
    code = """
def choose(frame):
    return frame.select_dtypes(include=["number"]).columns[0]
"""
    assert _fires_on_function(code) is True
    assert _fires(code) is True


def test_a_dtype_selection_reached_through_a_list_is_still_refused():
    """``select_dtypes`` survives any amount of chaining after it."""

    code = """
def choose(frame):
    return frame.select_dtypes(include=["number"]).columns.tolist()[0]
"""
    assert _fires_on_function(code) is True


def test_the_named_candidate_loop_fallback_is_still_refused():
    """The first branch, untouched -- named candidates, then frame order."""

    code = """
def find_column(frame, candidates, numeric=False):
    for column in candidates:
        if column in frame.columns:
            return column
    if numeric:
        for column in frame.columns:
            if frame[column].notna().any():
                return column
    return None
"""
    assert _fires_on_function(code) is True
    assert _fires(code) is True


# ---------------------------------------------------------------------------
# What separates them
# ---------------------------------------------------------------------------


def test_the_identifier_alone_decides_nothing():
    """Same identifier, opposite verdicts -- the expression is what is read."""

    declared = """
def choose(contract):
    columns = contract["columns"]
    return columns[0]
"""
    frame_order = """
def choose(frame):
    return frame.columns[0]
"""
    assert _fires_on_function(declared) is False
    assert _fires_on_function(frame_order) is True


def test_indexing_a_plain_list_at_zero_is_not_a_finding():
    """Nothing about ``[0]`` is suspicious on its own."""

    code = """
def choose(contract):
    ordered = contract["ordered_levels"]
    return ordered[0]
"""
    assert _fires_on_function(code) is False
