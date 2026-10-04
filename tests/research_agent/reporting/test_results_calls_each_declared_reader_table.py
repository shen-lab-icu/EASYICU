"""Results calls each table an owner declared, by the number the reader prints.

The signed survival suite declares its Table 1 and its risk-set accounting,
and the reader prints them after any plan Table 1.  Only a plan's generic
Table 1 was a registered display, so nothing asked Results to call these
tables and the Writer was never told they existed.  The table owner now
numbers each declared table as the reader does and names its source and its
Results subsection; the Writer's digest lists them, the host restores a
missing callout before binding, and the final audit requires each one.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from typing import Annotated, Union, get_args, get_origin

import pandas as pd
import pytest

from easyicu.research_agent.contracts.manuscript_result_structure import (
    COHORT_RESULT_HEADING,
    PLAN_RESULT_HEADINGS,
)
from easyicu.research_agent.contracts.manuscript_tables import ManuscriptTableDeclaration
from easyicu.research_agent.methods.table_one import build_grouped_table_one
from easyicu.research_agent.reporting import manuscript_tables, write_phase
from easyicu.research_agent.reporting.manuscript_quality import (
    audit_manuscript_quality,
    expected_manuscript_display_labels,
    repair_registered_display_callouts,
)
from easyicu.research_agent.reporting.manuscript_tables import (
    ManuscriptTableProjectionError,
    build_manuscript_tables,
    declared_table_callouts,
    reader_table_digest,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, EvidenceRecord, TableOneSpec
from tests.support.survival_sealed import run_signed_suite, sealed_survival, synthetic_survival_rows

STEP = "primary_survival_suite"


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    _context, authority = sealed_survival(root)
    out = root / "out"
    summary = json.loads(json.dumps(run_signed_suite(authority, synthetic_survival_rows(), out)))
    return authority, summary, out


def _register(run_dir, evidence_id, kind, payload: bytes, name: str, step=STEP):
    target = run_dir / "evidence" / f"{evidence_id}__{name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return EvidenceRecord(
        evidence_id=evidence_id, kind=kind, description=evidence_id,
        relative_path=f"evidence/{target.name}", sha256=hashlib.sha256(payload).hexdigest(),
        produced_by_step=step, producer="runner", generation_mode="deterministic_standard",
    )


def _suite_records(run_dir, summary, out):
    records = [_register(
        run_dir, f"statistic_step_summary_{STEP}", "statistic", json.dumps(summary).encode(), "step_summary.json",
    )]
    for product, name in summary["output_files"].items():
        if product.startswith("table:"):
            records.append(_register(
                run_dir, f"table_{STEP}_{product.split(':', 1)[1]}", "table", (out / name).read_bytes(), name,
            ))
    return records


def _suite_step():
    return AnalysisStep(
        step_id=STEP, intent="Execute the signed survival suite", inputs=["rrt", "mort_90d"],
        expected_outputs=["table:landmark_table_one"], method="signed_landmark_survival_suite",
    )


def _plan(*steps):
    return AnalysisPlan(
        research_question="Is renal replacement therapy associated with 90-day mortality?",
        steps=list(steps) or [_suite_step()],
        display_labels={"age": "Age at admission", "sex": "Sex"},
    )


def _plan_table_one(run_dir):
    """A plan's own grouped Table 1 step and its registered source."""

    spec = TableOneSpec(
        schema_version="easyicu.table_one/2", p_values_required=False,
        p_value_adjustment="not_applicable_repeated_units", group_by="exposure", group_levels=[0, 1],
        variables=[{
            "name": "value", "variable_kind": "continuous", "summary": "median_iqr",
            "test": "none_descriptive_smd_only",
        }],
    )
    frame = pd.DataFrame({"exposure": [0, 0, 1, 1], "value": [1.0, 3.0, 8.0, None]})
    payload = build_grouped_table_one(frame, spec).to_csv(index=False).encode()
    record = _register(run_dir, "baseline_table", "table", payload, "table_one.csv", step="baseline")
    step = AnalysisStep(
        step_id="baseline", intent="Describe baseline values", inputs=["exposure", "value"],
        expected_outputs=["table:table_one"], method="grouped_table_one", table_one_spec=spec,
    )
    return step, record


def _callout_ids(authority):
    return (
        f"table_{STEP}_{authority.table_one_product.split(':', 1)[1]}",
        f"table_{STEP}_{authority.risk_set_product.split(':', 1)[1]}",
    )


def test_a_declared_table_is_called_by_the_number_the_reader_prints(suite, tmp_path):
    authority, summary, out = suite
    records = _suite_records(tmp_path, summary, out)

    callouts = declared_table_callouts(plan=_plan(), evidence_records=records, run_dir=tmp_path)
    tables = build_manuscript_tables(plan=_plan(), evidence_records=records, run_dir=tmp_path)

    assert [callout.label for callout in callouts] == [f"Table {n}" for n in range(1, len(tables) + 1)]
    assert [callout.caption for callout in callouts] == [table.caption for table in tables]
    assert tuple(callout.evidence_id for callout in callouts) == _callout_ids(authority)
    assert {callout.subsection for callout in callouts} == {COHORT_RESULT_HEADING}
    digest = reader_table_digest(callouts)
    for callout in callouts:
        assert (
            f"- {callout.label}: {callout.caption}; subsection={COHORT_RESULT_HEADING}; "
            f"cite={{evidence:{callout.evidence_id}}}"
        ) in digest
    assert reader_table_digest(()) == ""


def test_a_plan_table_one_comes_first_and_keeps_its_own_callout(suite, tmp_path):
    authority, summary, out = suite
    step, table_one = _plan_table_one(tmp_path)
    records = [table_one, *_suite_records(tmp_path, summary, out)]
    plan = _plan(step, _suite_step())

    callouts = declared_table_callouts(plan=plan, evidence_records=records, run_dir=tmp_path)
    tables = build_manuscript_tables(plan=plan, evidence_records=records, run_dir=tmp_path)

    assert tables[0].caption == "Baseline characteristics"
    assert [callout.label for callout in callouts] == ["Table 2", "Table 3"]
    assert [callout.caption for callout in callouts] == [table.caption for table in tables[1:]]
    names = ["table_one", *(record.evidence_id for record in records)]
    assert expected_manuscript_display_labels(names, callouts) == ("Table 1", "Table 2", "Table 3")


def _manuscript(cohort: str) -> str:
    return (
        "## Results\n\n"
        f"### Cohort characteristics\n\n{cohort}\n\n"
        "### Survival results\n\n"
        "Deaths by day 90 are reported by exposure group.\n"
    )


def _uncalled(text, labels):
    audit = audit_manuscript_quality(text, expected_display_labels=labels, require_administrative_sections=False)
    return [finding.excerpts[0] for finding in audit.findings if finding.code == "MANUSCRIPT_DISPLAY_NOT_CALLED_OUT"]


@pytest.mark.parametrize("already_called", [False, True])
def test_results_calls_each_declared_table_once(suite, tmp_path, already_called):
    _authority, summary, out = suite
    records = _suite_records(tmp_path, summary, out)
    callouts = declared_table_callouts(plan=_plan(), evidence_records=records, run_dir=tmp_path)
    table_one, flow = callouts
    labels = expected_manuscript_display_labels([record.evidence_id for record in records], callouts)
    cohort = "The landmark cohort is described by exposure group."
    if already_called:
        cohort += f" Table 1 summarizes it {{evidence:{table_one.evidence_id}}}."
    text = _manuscript(cohort)

    assert labels == ("Table 1", "Table 2")
    assert _uncalled(text, labels) == (["Table 2"] if already_called else ["Table 1", "Table 2"])

    repaired, repairs = repair_registered_display_callouts(
        text, expected_display_labels=labels, analysis_plan=_plan(), reader_tables=callouts,
    )

    assert [repair["label"] for repair in repairs] == (["Table 2"] if already_called else ["Table 1", "Table 2"])
    cohort_section = repaired.split("### Cohort characteristics", 1)[1].split("### Survival results", 1)[0]
    assert cohort_section.count(f"See Table 2 {{evidence:{flow.evidence_id}}}.") == 1
    assert cohort_section.count("Table 1") == 1
    if not already_called:
        assert f"See Table 1 {{evidence:{table_one.evidence_id}}}." in cohort_section
    assert _uncalled(repaired, labels) == []


def test_a_table_the_reader_cannot_print_is_not_called(suite, tmp_path):
    _authority, summary, out = suite
    records = _suite_records(tmp_path, summary, out)
    callouts = declared_table_callouts(plan=_plan(), evidence_records=records, run_dir=tmp_path)
    _table_one, flow = callouts
    # A source that drifted after registration: the reader prints no table.
    drifted = next(tmp_path / record.relative_path for record in records if record.evidence_id == flow.evidence_id)
    drifted.write_bytes(drifted.read_bytes() + b"\n")

    with pytest.raises(ManuscriptTableProjectionError):
        build_manuscript_tables(plan=_plan(), evidence_records=records, run_dir=tmp_path)
    assert declared_table_callouts(plan=_plan(), evidence_records=records, run_dir=tmp_path) == ()
    # A product that is no longer current is not expected either.
    current = [record.evidence_id for record in records if record.evidence_id != flow.evidence_id]
    assert expected_manuscript_display_labels(current, callouts) == ("Table 1",)


def test_every_declared_layout_names_its_results_subsection():
    body = ManuscriptTableDeclaration.model_fields["body"].annotation
    if get_origin(body) is Annotated:
        body = get_args(body)[0]
    members = get_args(body) if get_origin(body) is Union else (body,)
    layouts = {get_args(member.model_fields["layout"].annotation)[0] for member in members}

    assert set(manuscript_tables._RESULTS_SUBSECTION_BY_LAYOUT) == layouts
    assert set(manuscript_tables._RESULTS_SUBSECTION_BY_LAYOUT.values()) <= PLAN_RESULT_HEADINGS


def _calls(function, name):
    tree = ast.parse(inspect.getsource(function).lstrip())
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def test_the_write_phase_carries_the_declared_tables_to_every_callout_gate():
    (draft_call,) = _calls(write_phase._draft_manuscript, "_DraftStageResult")
    assert "reader_tables" in {keyword.arg for keyword in draft_call.keywords}
    assert _calls(write_phase._draft_manuscript, "declared_table_callouts")
    assert _calls(write_phase._draft_manuscript, "reader_table_digest")

    (bind_call,) = _calls(write_phase._draft_bind_and_repair_manuscript, "_bind_and_review_manuscript")
    keyword = next(item for item in bind_call.keywords if item.arg == "reader_tables")
    assert ast.unparse(keyword.value) == "draft.reader_tables"
    loop_labels = _calls(write_phase._draft_bind_and_repair_manuscript, "expected_manuscript_display_labels")
    assert loop_labels and all(
        [ast.unparse(arg) for arg in call.args] == ["draft.current_evidence_names", "draft.reader_tables"]
        for call in loop_labels
    )

    binding_labels = _calls(write_phase._bind_and_review_manuscript, "expected_manuscript_display_labels")
    assert len(binding_labels) == 2 and all(
        [ast.unparse(arg) for arg in call.args] == ["current_evidence_names", "reader_tables"]
        for call in binding_labels
    )
    (repair_call,) = _calls(write_phase._bind_and_review_manuscript, "repair_registered_display_callouts")
    keyword = next(item for item in repair_call.keywords if item.arg == "reader_tables")
    assert ast.unparse(keyword.value) == "reader_tables"
