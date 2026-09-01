"""StepRecord and ProbeSummary schema pin.

These pydantic models centralise the field set of two dicts the
pipeline accumulates across many code paths:

* ``probe_summary`` — the deterministic probe step's output.
* ``per_step_records`` — one entry per planner step, persisted into
  ``manifest.json`` for resume.

Before these schemas were added, the field list lived only as scattered
``step_record[...]`` assignments inside :mod:`pipeline`. The tests below
pin the contract so adding a new field requires either (a) extending
the schema, or (b) flowing the addition through ``extra='allow'``
deliberately.
"""

from __future__ import annotations

import pytest


def test_step_record_minimal_construction(ra):
    rec = ra.StepRecord(step_id="01_table_one", intent="Compute table one.")
    assert rec.step_id == "01_table_one"
    assert rec.intent == "Compute table one."
    assert rec.status is None
    assert rec.evidence_ids == []


def test_step_record_accepts_known_fields(ra):
    rec = ra.StepRecord(
        step_id="02_assoc",
        intent="Primary association.",
        status="ok",
        generation_mode="llm",
        step_summary={"primary_or": 1.42},
        evidence_ids=["primary_association"],
        returncode=0,
        timed_out=False,
        code_repair_attempts=1,
    )
    assert rec.status == "ok"
    assert rec.step_summary["primary_or"] == 1.42


def test_step_record_allows_extra_fields(ra):
    """`extra='allow'` keeps existing dict-style construction sites valid.

    The pipeline accumulates fields incrementally; until we migrate all
    construction sites to typed access, future-added fields must not
    break model_validate on a previously-persisted manifest.
    """
    rec = ra.StepRecord(
        step_id="03",
        intent="x",
        custom_future_field={"k": "v"},
    )
    assert rec.model_dump()["custom_future_field"] == {"k": "v"}


def test_step_record_model_validate_round_trip_from_dict(ra):
    raw = {
        "step_id": "04_outcome",
        "intent": "Outcome incidence.",
        "status": "ok",
        "evidence_ids": ["outcome_rate"],
        "step_summary": {"outcome_rate": 0.18},
    }
    rec = ra.StepRecord.model_validate(raw)
    dumped = rec.model_dump()
    assert dumped["step_id"] == "04_outcome"
    assert dumped["step_summary"] == {"outcome_rate": 0.18}


def test_step_record_requires_step_id_and_intent(ra):
    with pytest.raises(Exception):
        ra.StepRecord(intent="missing step_id")
    with pytest.raises(Exception):
        ra.StepRecord(step_id="missing intent")


def test_probe_summary_required_fields(ra):
    summary = ra.ProbeSummary(n_rows=1234, n_columns=42)
    assert summary.n_rows == 1234
    assert summary.target_outcome is None
    assert summary.top_missing_columns == []
    assert summary.score_completeness == []


def test_probe_summary_round_trip(ra):
    raw = {
        "n_rows": 1234,
        "n_columns": 42,
        "target_outcome": "death",
        "top_missing_columns": [{"variable": "lactate", "fraction_missing": 0.32}],
        "score_completeness": [{"variable": "severity_score"}],
    }
    summary = ra.ProbeSummary.model_validate(raw)
    assert summary.score_completeness[0]["variable"] == "severity_score"
    assert summary.top_missing_columns[0]["variable"] == "lactate"


def test_probe_summary_allows_extra(ra):
    """Probe metrics evolve over time; extra='allow' preserves forward compat."""
    summary = ra.ProbeSummary(
        n_rows=10,
        n_columns=3,
        future_metric={"x": 1},
    )
    assert summary.model_dump()["future_metric"] == {"x": 1}
