"""The simulated statistician finds a measurement audit its owner recorded.

The reviewer looked for the word "missingness" in evidence ids and
descriptions.  An owner such as the signed landmark survival suite counts the
missing values of its sealed columns in its own summary
(``missingness_measurement_audit``) and publishes the table under its own
product name, so every such run was told "A missingness profile is not
registered".  The reviewer now reads the audit from the current,
digest-verified step summary; anything less still leaves the comment.

Synthetic run directories only; no study's values.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.reporting.reviewer import run_reviewer_round

STEP = "primary_outcome_suite"
AUDIT = {
    "source_n": 120,
    "landmark_population_n": 96,
    "source_missing_n_by_column": {"age": 0, "lactate": 14},
}


def _register(run_dir, evidence_id, filename, content, **fields):
    relative = f"evidence/{evidence_id}__{filename}"
    path = run_dir / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return SimpleNamespace(
        evidence_id=evidence_id, relative_path=relative, description=f"Table {filename}.",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), kind="statistic",
        metadata={}, **fields,
    )


def _run(run_dir, summary, *, status="ok", produced_by_step=STEP):
    table = _register(
        run_dir, "table_step_artifact_0001", "outcome_measurement_audit.csv",
        "column,missing_n\nage,0\nlactate,14\n", produced_by_step=STEP,
    )
    record = _register(
        run_dir, "statistic_step_summary_0001", "step_summary.json",
        json.dumps(summary), produced_by_step=produced_by_step,
    )
    step = {
        "step_id": STEP, "status": status,
        "evidence_ids": [table.evidence_id, record.evidence_id],
        "step_summary_evidence_id": record.evidence_id,
    }
    return [table, record], [step]


def _missingness_comments(report):
    return [
        comment
        for critique in report.critiques
        for comment in critique.comments
        if comment.topic == "missingness"
    ]


def test_an_owner_audit_in_its_summary_answers_the_missingness_item(tmp_path):
    records, steps = _run(tmp_path, {"status": "ok", "missingness_measurement_audit": AUDIT})

    report = run_reviewer_round(evidence_records=records, findings=[], per_step_records=steps, run_dir=tmp_path)

    assert _missingness_comments(report) == []


def test_a_summary_without_the_audit_still_draws_the_comment(tmp_path):
    records, steps = _run(tmp_path, {"status": "ok"})

    report = run_reviewer_round(evidence_records=records, findings=[], per_step_records=steps, run_dir=tmp_path)

    assert [comment.severity for comment in _missingness_comments(report)] == ["minor"]


@pytest.mark.parametrize("failure", ["empty_audit", "drifted_bytes", "failed_step", "foreign_summary", "no_run_dir"])
def test_an_unverified_audit_still_draws_the_comment(tmp_path, failure):
    audit = {} if failure == "empty_audit" else AUDIT
    records, steps = _run(
        tmp_path, {"status": "ok", "missingness_measurement_audit": audit},
        status="failed" if failure == "failed_step" else "ok",
        produced_by_step="another_step" if failure == "foreign_summary" else STEP,
    )
    if failure == "drifted_bytes":
        (tmp_path / records[1].relative_path).write_text(
            json.dumps({"status": "ok", "missingness_measurement_audit": {**AUDIT, "source_n": 121}}),
            encoding="utf-8",
        )

    report = run_reviewer_round(
        evidence_records=records, findings=[], per_step_records=steps,
        run_dir=None if failure == "no_run_dir" else tmp_path,
    )

    assert [comment.severity for comment in _missingness_comments(report)] == ["minor"]
