"""The signed survival suite's Table 1 and risk-set accounting reach the reader.

The suite computes its own Table 1 and its risk-set flow, but only a plan's
generic Table 1 step reached the manuscript's tables, so a survival article
had none: a reader saw neither the groups' characteristics nor how the
landmark cohort was reached.  The suite now declares both products as reader
tables in its step summary, and the reporting owner formats their recorded
cells, computing nothing; Table 1 closes with each group's deaths.  The cells
stay out of the bound text.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import io
import json

import pytest

from easyicu.research_agent.contracts.manuscript_tables import (
    MANUSCRIPT_TABLES_KEY,
    TableGroup,
    validate_manuscript_table_declarations,
)
from easyicu.research_agent.reporting.manuscript_tables import (
    ManuscriptTableProjectionError,
    build_manuscript_tables,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, EvidenceRecord
from tests.support.survival_sealed import run_signed_suite, sealed_survival, synthetic_survival_rows

STEP = "primary_survival_suite"
LABELS = {"age": "Age at admission", "sex": "Sex"}


def _suite(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    out = tmp_path / "out"
    summary = json.loads(json.dumps(run_signed_suite(authority, synthetic_survival_rows(), out)))
    return authority, summary, out


def _register(run_dir, evidence_id, kind, payload: bytes, name: str, *, mode="deterministic_standard"):
    target = run_dir / "evidence" / f"{evidence_id}__{name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return EvidenceRecord(
        evidence_id=evidence_id, kind=kind, description=evidence_id,
        relative_path=f"evidence/{target.name}", sha256=hashlib.sha256(payload).hexdigest(),
        produced_by_step=STEP, producer="runner", generation_mode=mode,
    )


def _records(run_dir, summary, out, *, skip=()):
    records = [_register(
        run_dir, f"statistic_step_summary_{STEP}", "statistic", json.dumps(summary).encode(), "step_summary.json",
    )]
    for product, name in summary["output_files"].items():
        if product.startswith("table:") and product not in skip:
            records.append(_register(
                run_dir, f"table_{STEP}_{product.split(':', 1)[1]}", "table", (out / name).read_bytes(), name,
            ))
    return records


def _plan(labels=LABELS):
    return AnalysisPlan(
        research_question="Is renal replacement therapy associated with 90-day mortality?",
        steps=[AnalysisStep(
            step_id=STEP, intent="Execute the signed survival suite", inputs=["rrt", "mort_90d"],
            expected_outputs=["table:landmark_table_one"], method="signed_landmark_survival_suite",
        )],
        display_labels=labels,
    )


def _csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mean_sd(row, prefix):
    return f"{Decimal(row[f'{prefix}_mean']):.2f} ({Decimal(row[f'{prefix}_sd']):.2f})"


def test_the_suite_declares_its_table_one_and_its_risk_set_flow(tmp_path):
    authority, summary, out = _suite(tmp_path)

    table_one, flow = validate_manuscript_table_declarations(summary[MANUSCRIPT_TABLES_KEY])

    assert (table_one.product, flow.product) == (authority.table_one_product, authority.risk_set_product)
    assert [group.label for group in table_one.body.groups] == [
        authority.comparator_group_label, authority.exposed_group_label,
    ]
    assert sum(group.n for group in table_one.body.groups) == summary["n_landmark_population"]
    # Each group's deaths are the ones its Kaplan-Meier estimate counted.
    km = {int(row["exposure_group"]): int(row["group_events"]) for row in _csv(out / "landmark_km_curve.csv")}
    assert [group.events for group in table_one.body.groups] == [km[0], km[1]]
    assert table_one.body.events_label == "Deaths by day 90, n (%)"


def test_the_reader_tables_copy_the_recorded_cells(tmp_path):
    authority, summary, out = _suite(tmp_path)
    run_dir = tmp_path / "run"
    records = _records(run_dir, summary, out)
    before = {record.relative_path: (run_dir / record.relative_path).read_bytes() for record in records}

    table_one, flow = build_manuscript_tables(plan=_plan(), evidence_records=records, run_dir=run_dir)

    unexposed, exposed = validate_manuscript_table_declarations(summary[MANUSCRIPT_TABLES_KEY])[0].body.groups
    assert table_one.columns == (
        "Characteristic", f"{unexposed.label} (n = {unexposed.n})", f"{exposed.label} (n = {exposed.n})", "SMD",
    )
    age, sex = _csv(out / "landmark_table_one.csv")
    assert table_one.rows[0] == (
        "Age at admission, mean (SD)", _mean_sd(age, "unexposed"), _mean_sd(age, "exposed"),
        f"{Decimal(age['standardized_mean_difference']):.3f}",
    )
    assert table_one.rows[1][0] == "  Median [Q1, Q3]"
    # The synthetic sex column is numeric, so the suite summarized it as such.
    assert table_one.rows[2][:3] == ("Sex, mean (SD)", _mean_sd(sex, "unexposed"), _mean_sd(sex, "exposed"))
    assert table_one.rows[-1] == (
        "Deaths by day 90, n (%)",
        f"{unexposed.events} ({Decimal(repr(unexposed.events_percent)):.1f}%)",
        f"{exposed.events} ({Decimal(repr(exposed.events_percent)):.1f}%)",
        "",
    )

    stages = _csv(out / "landmark_risk_set_flow.csv")
    assert [row[1:] for row in flow.rows] == [
        (stage["count"], "" if index == 0 else stage["excluded_since_prior_stage"])
        for index, stage in enumerate(stages)
    ]
    assert flow.rows[0][0] == "Source cohort" and flow.rows[-1][0] == "Landmark analysis cohort"
    assert any("first recorded as present at or before hour 0" in note for note in flow.notes)
    # Formatting reads the products; it never rewrites them.
    assert before == {record.relative_path: (run_dir / record.relative_path).read_bytes() for record in records}


def test_a_summary_without_declarations_adds_no_table(tmp_path):
    _authority, summary, out = _suite(tmp_path)
    summary.pop(MANUSCRIPT_TABLES_KEY)
    run_dir = tmp_path / "run"

    assert build_manuscript_tables(plan=_plan(), evidence_records=_records(run_dir, summary, out), run_dir=run_dir) == ()


def _earlier_summary(run_dir):
    """A re-executed step's earlier summary, which a store read without its step ledger keeps."""

    return _register(
        run_dir, f"statistic_step_summary_{STEP}_earlier", "statistic",
        json.dumps({"status": "ok", "output_files": {}}).encode(), "step_summary.json",
    )


def test_an_earlier_summary_beside_one_that_declares_nothing_reads_as_before(tmp_path):
    _authority, summary, out = _suite(tmp_path)
    summary.pop(MANUSCRIPT_TABLES_KEY)
    run_dir = tmp_path / "run"
    records = [_earlier_summary(run_dir), *_records(run_dir, summary, out)]

    assert build_manuscript_tables(plan=_plan(), evidence_records=records, run_dir=run_dir) == ()


def test_declared_tables_beside_an_earlier_summary_fail_closed(tmp_path):
    _authority, summary, out = _suite(tmp_path)
    run_dir = tmp_path / "run"
    records = [_earlier_summary(run_dir), *_records(run_dir, summary, out)]

    with pytest.raises(ManuscriptTableProjectionError, match="more than one step summary"):
        build_manuscript_tables(plan=_plan(), evidence_records=records, run_dir=run_dir)


@pytest.mark.parametrize("mutation", ["drift", "unlabelled_stage", "missing_source", "not_an_output"])
def test_a_declared_table_fails_closed_at_its_source(tmp_path, mutation):
    authority, summary, out = _suite(tmp_path)
    run_dir = tmp_path / "run"
    if mutation == "unlabelled_stage":
        summary[MANUSCRIPT_TABLES_KEY][1]["body"]["stage_labels"].pop("source_rows")
    if mutation == "not_an_output":
        summary["output_files"].pop(authority.risk_set_product)
    records = _records(
        run_dir, summary, out, skip=(authority.risk_set_product,) if mutation == "missing_source" else (),
    )
    if mutation == "drift":
        [table_one] = [record for record in records if record.relative_path.endswith("landmark_table_one.csv")]
        (run_dir / table_one.relative_path).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ManuscriptTableProjectionError):
        build_manuscript_tables(plan=_plan(), evidence_records=records, run_dir=run_dir)


def test_a_categorical_variable_reads_by_its_named_levels(tmp_path):
    """The categorical branch of the layout, on a two-row hand-written product."""

    run_dir = tmp_path / "run"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=[
        "variable", "level", "summary_type", "a_n", "a_denominator", "a_percent",
        "b_n", "b_denominator", "b_percent", "standardized_mean_difference",
    ])
    writer.writeheader()
    writer.writerow({"variable": "admission", "level": "elective", "summary_type": "categorical_n_percent",
                     "a_n": "3", "a_denominator": "10", "a_percent": "30.0",
                     "b_n": "5", "b_denominator": "10", "b_percent": "50.0", "standardized_mean_difference": "0.41"})
    writer.writerow({"variable": "admission", "level": "emergency", "summary_type": "categorical_n_percent",
                     "a_n": "7", "a_denominator": "10", "a_percent": "70.0",
                     "b_n": "4", "b_denominator": "10", "b_percent": "40.0", "standardized_mean_difference": ""})
    summary = {
        "status": "ok", "output_files": {"table:demo": "demo.csv"},
        MANUSCRIPT_TABLES_KEY: [{
            "schema_version": "easyicu.manuscript_table/1", "product": "table:demo", "caption": "Demo groups",
            "body": {"layout": "grouped_summary", "groups": [
                {"prefix": "a", "label": "Group A", "n": 10}, {"prefix": "b", "label": "Group B", "n": 10},
            ]},
        }],
    }
    records = [
        _register(run_dir, f"statistic_step_summary_{STEP}", "statistic", json.dumps(summary).encode(), "step_summary.json"),
        _register(run_dir, f"table_{STEP}_demo", "table", stream.getvalue().encode(), "demo.csv"),
    ]

    [table] = build_manuscript_tables(
        plan=_plan({"admission": "Admission type", "admission=elective": "Elective"}),
        evidence_records=records, run_dir=run_dir,
    )

    assert table.caption == "Demo groups"
    assert table.columns == ("Characteristic", "Group A (n = 10)", "Group B (n = 10)", "SMD")
    assert table.rows == (
        ("Admission type, n (%)", "", "", ""),
        ("  Elective", "3 (30.0%)", "5 (50.0%)", "0.410"),
        ("  emergency", "7 (70.0%)", "4 (40.0%)", "N/A"),
    )


@pytest.mark.parametrize(
    "events",
    [{"events": 3}, {"events": 3, "events_percent": 40.0}, {"events": 11, "events_percent": 110.0}],
    ids=["share_missing", "share_contradicts_count", "more_events_than_records"],
)
def test_group_events_must_be_their_recorded_share(events):
    with pytest.raises(ValueError):
        TableGroup(prefix="a", label="Group A", n=10, **events)
