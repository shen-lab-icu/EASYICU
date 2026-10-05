"""Small cells in a run's publication products are listed for sign-off review.

The PhysioNet credentialed data use agreement and the AmsterdamUMCdb licence
set no numeric floor; they ask for reasonable care not to disclose identities.
A report therefore keeps its exact counts, and the run lists every count from
1 to 10 that its reader tables, cited numbers and exported result tables show,
so the person who signs the report off can judge each one.  A source whose
licence requires suppression, or a source with no declared profile, fails
closed, because no product suppresses cells yet.

Synthetic study and seeded synthetic rows (renal replacement therapy and
90-day mortality), plus tiny synthetic tables built for their counts.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.databases.profiles import iter_database_profiles
from easyicu.research_agent.authority.numeric_claim_identity import NumericClaim
from easyicu.research_agent.contracts.manuscript_tables import (
    MANUSCRIPT_TABLES_KEY,
    validate_manuscript_table_declarations,
)
from easyicu.research_agent.gates import publication_disclosure as policy
from easyicu.research_agent.gates.publication_disclosure import (
    PROFILE_UNDECLARED_REASON,
    SUPPRESSION_UNAVAILABLE_REASON,
    PublicationDisclosureProfile as Profile,
    SourceDisclosure,
    is_subject_count_name,
    small_cell_value,
    source_disclosure,
    strictest_profile,
)
from easyicu.research_agent.methods.table_one import build_grouped_table_one
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting.manuscript_tables import (
    build_manuscript_tables,
    manuscript_table_counts,
)
from easyicu.research_agent.reporting.publication_disclosure_review import (
    PUBLICATION_DISCLOSURE_REVIEW_FILENAME,
    persist_publication_disclosure_review,
    review_publication_disclosure,
    study_disclosures,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, EvidenceRecord, TableOneSpec
from tests.support.survival_sealed import run_signed_suite, sealed_survival, synthetic_survival_rows


def _context(database, validation=()):
    return SimpleNamespace(cohort=SimpleNamespace(database=database), cross_database_validation=list(validation))


def _register(run_dir, evidence_id, payload: bytes, name, *, kind="table", step="baseline"):
    target = run_dir / "evidence" / f"{evidence_id}__{name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return EvidenceRecord(
        evidence_id=evidence_id, kind=kind, description=evidence_id,
        relative_path=f"evidence/{target.name}", sha256=hashlib.sha256(payload).hexdigest(),
        produced_by_step=step, producer="runner", generation_mode="deterministic_standard",
    )


def _tiny_table_one(run_dir):
    """Five synthetic rows: every count the Table 1 prints is 5 or less."""

    spec = TableOneSpec(
        schema_version="easyicu.table_one/2", p_values_required=False,
        p_value_adjustment="not_applicable_repeated_units",
        group_by="exposure", group_levels=[0, 1],
        variables=[
            {"name": "ventilated", "variable_kind": "categorical", "summary": "count_percent",
             "test": "none_descriptive_smd_only", "levels": [0, 1]},
            {"name": "value", "variable_kind": "continuous", "summary": "median_iqr",
             "test": "none_descriptive_smd_only"},
        ],
    )
    frame = pd.DataFrame({
        "exposure": [0, 0, 1, 1, 1],
        "ventilated": [0, 1, 1, 1, None],
        "value": [1.0, 3.0, 8.0, None, 5.0],
    })
    table = build_grouped_table_one(frame, spec)
    record = _register(run_dir, "baseline_table", table.to_csv(index=False).encode(), "table_one.csv")
    plan = AnalysisPlan(
        research_question="Describe baseline values by exposure.",
        steps=[AnalysisStep(
            step_id="baseline", intent="Describe baseline values", inputs=["exposure", "ventilated", "value"],
            expected_outputs=["table:table_one"], method="grouped_table_one", table_one_spec=spec,
        )],
    )
    return plan, record


# --- the policy -----------------------------------------------------------------


def test_every_registered_source_follows_a_declared_licence():
    for profile in iter_database_profiles():
        disclosure = source_disclosure(profile.key)
        assert disclosure.profile is Profile.REPORT_WITH_REVIEW, profile.key
        assert disclosure.licence not in {"", "undeclared"}


def test_a_study_resolves_its_sources_through_the_registry():
    sources = study_disclosures(_context("mimiciv", ["eicu-crd", "MIMIC-IV", "sicdb"]))

    assert [item.source for item in sources] == ["miiv", "eicu", "sic"]
    assert {item.licence for item in sources} == {"PhysioNet Credentialed Health Data Use Agreement"}
    assert study_disclosures(_context("amsterdamumc"))[0].licence == "AmsterdamUMCdb end user licence agreement"


@pytest.mark.parametrize("tag", ["synthetic", "test", "mock", "fixture"])
def test_generated_data_needs_no_review(tag, tmp_path):
    assert source_disclosure(tag).profile is Profile.REPORT
    assert review_publication_disclosure(
        context=_context(tag), plan=None, evidence_records=[], run_dir=tmp_path, numeric_binding_map={},
    ) is None


def test_an_unknown_source_has_no_profile():
    assert source_disclosure("local_icu") == SourceDisclosure("local_icu", None, "undeclared")


@pytest.mark.parametrize(("profiles", "expected"), [
    ((Profile.REPORT, Profile.REPORT_WITH_REVIEW), Profile.REPORT_WITH_REVIEW),
    ((Profile.REPORT_WITH_REVIEW, Profile.SUPPRESS_SMALL_CELLS), Profile.SUPPRESS_SMALL_CELLS),
    ((Profile.REPORT_WITH_REVIEW, None), None),
    ((), None),
])
def test_the_strictest_source_decides(profiles, expected):
    disclosures = tuple(SourceDisclosure(f"s{index}", profile, "l") for index, profile in enumerate(profiles))
    assert strictest_profile(disclosures) is expected


@pytest.mark.parametrize(("value", "expected"), [
    (0, None), (1, 1), (10, 10), (11, None), (-3, None), ("7", 7), ("7.0", 7), (7.0, 7),
    (7.5, None), (True, None), (None, None), ("n/a", None), ("1,000", None), ("suppressed", None),
])
def test_a_small_cell_is_a_count_from_one_to_ten(value, expected):
    assert small_cell_value(value) == expected


@pytest.mark.parametrize(("name", "expected"), [
    ("at_risk", True), ("count", True), ("n", True), ("N", True), ("group_events", True),
    ("excluded_since_prior_stage", True), ("n_landmark_population", True), ("unexposed_n", True),
    ("missing_count", True), ("exposed_records", True), ("n events", True), ("denominator", True),
    ("n_bootstrap", False), ("n_clusters", False), ("n_splits", False), ("call_count", False),
    ("token_count", False), ("percent", False), ("events_percent", False), ("hazard_ratio", False),
    ("estimate", False), ("", False),
])
def test_count_names_are_read_wide_and_settings_are_set_aside(name, expected):
    assert is_subject_count_name(name) is expected


def test_the_policy_imports_only_the_standard_library():
    # A kernel owner may read the floor without enlarging the kernel's identity.
    tree = ast.parse(inspect.getsource(policy))
    imported = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported == {"__future__", "re", "dataclasses", "decimal", "enum", "typing"}


# --- the reader-table listing ---------------------------------------------------


def test_the_listing_names_every_count_table_one_prints(tmp_path):
    plan, record = _tiny_table_one(tmp_path)

    [table] = build_manuscript_tables(plan=plan, evidence_records=[record], run_dir=tmp_path)
    counts = manuscript_table_counts(plan=plan, evidence_records=[record], run_dir=tmp_path)

    listed = {(count.row, count.column): count.value for count in counts}
    assert {count.table for count in counts} == {"Table 1"}
    assert {count.evidence_id for count in counts} == {"baseline_table"}
    assert table.rows[0] == ("N", "5", "2", "3", "")
    assert listed == {
        ("N", "Overall"): 5, ("N", "0"): 2, ("N", "1"): 3,
        ("ventilated: 0", "Overall"): 1, ("ventilated: 0", "0"): 1, ("ventilated: 0", "1"): 0,
        ("ventilated: 1", "Overall"): 3, ("ventilated: 1", "0"): 1, ("ventilated: 1", "1"): 2,
        ("ventilated: Missing", "Overall"): 1, ("ventilated: Missing", "0"): 0,
        ("ventilated: Missing", "1"): 1,
        ("value: Missing", "Overall"): 1, ("value: Missing", "0"): 0, ("value: Missing", "1"): 1,
        ("Rows excluded for missing grouping value", ""): 0,
    }
    # Each listed category count is the one its printed cell leads with.
    printed = {row[0].strip(): row[1:4] for row in table.rows}
    assert [int(cell.split(" ")[0]) for cell in printed["1"]] == [3, 1, 2]


def test_the_listing_follows_the_tables_an_owner_declares(tmp_path):
    _context_unused, authority = sealed_survival(tmp_path)
    out = tmp_path / "out"
    summary = json.loads(json.dumps(run_signed_suite(authority, synthetic_survival_rows(), out)))
    run_dir = tmp_path / "run"
    step = "primary_survival_suite"
    records = [_register(
        run_dir, f"statistic_step_summary_{step}", json.dumps(summary).encode(), "step_summary.json",
        kind="statistic", step=step,
    )]
    for product, name in summary["output_files"].items():
        if product.startswith("table:"):
            records.append(_register(
                run_dir, f"table_{step}_{product.split(':', 1)[1]}", (out / name).read_bytes(), name, step=step,
            ))
    plan = AnalysisPlan(
        research_question="Is renal replacement therapy associated with 90-day mortality?",
        steps=[AnalysisStep(
            step_id=step, intent="Execute the signed survival suite", inputs=["rrt", "mort_90d"],
            expected_outputs=["table:landmark_table_one"], method="signed_landmark_survival_suite",
        )],
    )

    table_one, flow = build_manuscript_tables(plan=plan, evidence_records=records, run_dir=run_dir)
    counts = manuscript_table_counts(plan=plan, evidence_records=records, run_dir=run_dir)

    groups = validate_manuscript_table_declarations(summary[MANUSCRIPT_TABLES_KEY])[0].body.groups
    first = [(count.row, count.column, count.value) for count in counts if count.table == "Table 1"]
    assert first == [
        *(("n", group.label, group.n) for group in groups),
        *(("Deaths by day 90, n (%)", group.label, group.events) for group in groups),
    ]
    second = [(count.row, count.column, count.value) for count in counts if count.table == "Table 2"]
    expected = []
    for index, (label, records_cell, excluded_cell) in enumerate(flow.rows):
        expected.append((label, "Records", int(records_cell)))
        if index:
            expected.append((label, "Excluded", int(excluded_cell)))
    assert second == expected
    assert {count.caption for count in counts} == {table_one.caption, flow.caption}


# --- the review -----------------------------------------------------------------


def _claim(field, value, *, derived=()):
    return NumericClaim(
        value=str(value), canonical=float(value), evidence_id="step_summary_primary", step_id="primary",
        source_field=field, derived_from=list(derived),
    )


def test_small_cells_in_every_product_are_listed_and_the_report_keeps_them(tmp_path):
    plan, table_one = _tiny_table_one(tmp_path)
    km = _register(
        tmp_path, "km_curve",
        b"time,survival,at_risk,n_bootstrap\n0,1.0,25,5\n30,0.8,12,5\n60,0.6,9,5\n90,0.5,3,5\n", "km_curve.csv",
    )
    strata = _register(
        tmp_path, "strata",
        json.dumps({"strata": [{"stratum_n": 4, "estimate": 0.3}, {"stratum_n": 40}], "group_events": [12, 2]}).encode(),
        "strata.json",
    )
    records = [table_one, km, strata]
    before = {record.relative_path: (tmp_path / record.relative_path).read_bytes() for record in records}
    cited = {
        "claim_1": _claim("results.n_events", 5),
        "claim_2": _claim("hazard_ratio", 2),
        "claim_3": _claim("n_events", 3, derived=[("primary", "n_events")]),
        "claim_4": _claim("n_complete_case", 240),
    }

    review = review_publication_disclosure(
        context=_context("miiv"), plan=plan, evidence_records=records, run_dir=tmp_path,
        numeric_binding_map=cited,
    )

    payload = review.payload
    assert payload["profile"] == "report_with_review"
    assert payload["sources"] == [{
        "source": "miiv", "profile": "report_with_review",
        "licence": "PhysioNet Credentialed Health Data Use Agreement",
    }]
    reader = {(cell["row"], cell["column"]): cell["value"] for cell in payload["reader_tables"]}
    assert reader[("N", "Overall")] == 5 and reader[("ventilated: 1", "1")] == 2
    assert ("ventilated: 0", "1") not in reader  # a zero is not a small cell
    assert {cell["table"] for cell in payload["reader_tables"]} == {"Table 1"}
    assert payload["cited_numbers"] == [{
        "footnote": "claim_1", "step_id": "primary", "field": "n_events", "value": 5,
        "evidence_id": "step_summary_primary",
    }]
    columns = {(entry["evidence_id"], entry["column"]): entry for entry in payload["result_tables"]}
    assert columns[("km_curve", "at_risk")]["values"] == [3, 9]
    assert columns[("km_curve", "at_risk")]["first_locations"] == ["line 4", "line 5"]
    assert ("km_curve", "n_bootstrap") not in columns
    assert columns[("strata", "stratum_n")]["first_locations"] == ["strata[0].stratum_n"]
    assert columns[("strata", "group_events")]["values"] == [2]
    assert ("baseline_table", "count") in columns  # the exported Table 1 source is read too
    assert payload["unread_result_tables"] == []
    [finding] = review.findings
    assert finding.severity == "warning"
    assert PUBLICATION_DISCLOSURE_REVIEW_FILENAME in finding.message
    assert finding.detail["cited_numbers"] == 1
    # The review reads the products; it never rewrites or suppresses them.
    assert before == {record.relative_path: (tmp_path / record.relative_path).read_bytes() for record in records}


def test_a_study_without_small_cells_is_reviewed_without_a_warning(tmp_path):
    km = _register(tmp_path, "km_curve", b"time,at_risk\n0,40\n90,25\n", "km_curve.csv")

    review = review_publication_disclosure(
        context=_context("eicu"), plan=None, evidence_records=[km], run_dir=tmp_path,
        numeric_binding_map={"claim_1": _claim("n_events", 37)},
    )

    assert review.findings == ()
    assert (review.payload["reader_tables"], review.payload["cited_numbers"], review.payload["result_tables"]) == (
        [], [], [],
    )


def test_a_table_changed_after_registration_is_listed_as_unread(tmp_path):
    km = _register(tmp_path, "km_curve", b"time,at_risk\n0,40\n", "km_curve.csv")
    (tmp_path / km.relative_path).write_bytes(b"time,at_risk\n0,4\n")

    review = review_publication_disclosure(
        context=_context("miiv"), plan=None, evidence_records=[km], run_dir=tmp_path, numeric_binding_map={},
    )

    assert review.payload["unread_result_tables"] == [
        {"evidence_id": "km_curve", "reason": "missing or changed since registration"},
    ]
    [finding] = review.findings
    assert finding.severity == "warning" and "could not be read" in finding.message


def test_an_undeclared_source_fails_closed(tmp_path):
    review = review_publication_disclosure(
        context=_context("miiv", ["local_icu"]), plan=None, evidence_records=[], run_dir=tmp_path,
        numeric_binding_map={},
    )

    assert review.payload["profile"] is None
    [finding] = review.findings
    assert finding.severity == "error"
    assert finding.message.startswith(PROFILE_UNDECLARED_REASON)
    assert finding.detail["sources"] == ["local_icu"]


def test_a_licence_that_requires_suppression_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setitem(
        policy._SOURCE_PROFILES, "local_icu", (Profile.SUPPRESS_SMALL_CELLS, "local licence"),
    )

    review = review_publication_disclosure(
        context=_context("miiv", ["local_icu"]), plan=None, evidence_records=[], run_dir=tmp_path,
        numeric_binding_map={},
    )

    assert review.payload["profile"] == "suppress_small_cells"
    [finding] = review.findings
    assert finding.severity == "error"
    assert finding.detail["reason"] == SUPPRESSION_UNAVAILABLE_REASON


# --- the write phase ------------------------------------------------------------


class _Evidence:
    def __init__(self):
        self.registered = []

    def register_file(self, **kwargs):
        self.registered.append(kwargs)


def test_the_write_phase_persists_the_review_and_its_finding(tmp_path):
    km = _register(tmp_path, "km_curve", b"time,at_risk\n0,40\n90,6\n", "km_curve.csv")
    evidence, findings = _Evidence(), []

    persist_publication_disclosure_review(
        context=_context("hirid"), plan=None, evidence_records=[km], numeric_binding_map={},
        run_dir=tmp_path, evidence=evidence, findings=findings,
    )

    path = tmp_path / PUBLICATION_DISCLOSURE_REVIEW_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["result_tables"][0]["values"] == [6]
    [registered] = evidence.registered
    assert (registered["evidence_id"], registered["kind"], registered["source_path"]) == (
        "publication_disclosure_review", "log", path,
    )
    assert registered["on_sha_change"] == "new_id"
    assert registered["metadata"] == {
        "schema_version": "easyicu.publication_disclosure_review/1", "profile": "report_with_review",
    }
    assert [finding.validator for finding in findings] == ["publication_disclosure"]


def test_generated_data_leaves_no_review_behind(tmp_path):
    evidence, findings = _Evidence(), []

    persist_publication_disclosure_review(
        context=_context("synthetic"), plan=None, evidence_records=[], numeric_binding_map={},
        run_dir=tmp_path, evidence=evidence, findings=findings,
    )

    assert not (tmp_path / PUBLICATION_DISCLOSURE_REVIEW_FILENAME).exists()
    assert (evidence.registered, findings) == ([], [])


def test_the_review_follows_binding_outside_the_writer_probe():
    stage = inspect.getsource(write_phase._bind_and_review_manuscript)
    guard = stage.index("if not writer_probe_mode:\n        _persist_reader_artifacts(")

    assert stage.index("bind_numeric_values(") < guard < stage.index("_review_manuscript_with_fail_safe(")
    assert "numeric_binding_map=numeric_binding_map" in stage[guard:guard + 300]
    assert "persist_publication_disclosure_review(" in inspect.getsource(write_phase._persist_reader_artifacts)
