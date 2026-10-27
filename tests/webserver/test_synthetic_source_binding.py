"""Direct owner tests: no HTTP, TestClient, network, or patient data."""

from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture(params=["hirid", "aumc", "sic"])
def source(tmp_path, monkeypatch, request):
    from easyicu.synthetic_sources import generate_source
    from easyicu.webserver import sources

    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(sources, "_autodiscovery_bases", lambda: [])
    monkeypatch.setattr(sources.settings_store, "load_settings", lambda: {})
    path = tmp_path / "synthetic"
    generate_source(path, request.param, stays=12)
    return path


def test_normal_source_scan_and_registry_keep_synthetic_identity(source):
    from easyicu.databases.profiles import get_database_profile
    from easyicu.synthetic_sources import verify_source
    from easyicu.webserver import dataio, sources

    database = verify_source(source)["database"]
    scan = dataio.scan_path(str(source))
    assert scan["ok"] and scan["ready"]
    assert get_database_profile(scan["db_key"]).key == database and scan["source"] == "prepared"
    assert scan["source_kind"] == "synthetic"
    registry = sources.register_source(str(source), active=True)
    assert registry["ok"]
    bound = next(row for row in registry["sources"] if row["path"] == str(source))
    assert bound["database"] == database
    assert bound["source_kind"] == "synthetic"
    assert bound["provenance"]["contains_patient_data"] is False
    assert "synthetic" in sources.source_reader_label(bound).lower()
    assert sources.registered_source_counts()["by_database"][database] == 0
    with _lease(sources, source) as lease:
        assert lease.sources[0]["source_kind"] == "synthetic"


def _lease(sources, source):
    from contextlib import contextmanager

    @contextmanager
    def leased():
        lease = sources.acquire_path_lease([str(source)])
        try:
            yield lease
        finally:
            lease.release()
    return leased()


def test_only_verified_source_is_non_patient_for_egress_and_scope_resets(source):
    from easyicu.synthetic_sources import verify_source
    from easyicu.research_agent.providers.client_trust import source_requires_clearance
    from easyicu.webserver.demo_source_storage import public_demo_egress_scope

    database = verify_source(source)["database"]
    other = "sic" if database != "sic" else "hirid"
    assert source_requires_clearance(database)
    with public_demo_egress_scope(str(source)):
        assert not source_requires_clearance(database)
        assert source_requires_clearance(other)
    assert source_requires_clearance(database)
    (source / "unexpected.parquet").write_bytes(b"unverified payload")
    with public_demo_egress_scope(str(source)):
        assert source_requires_clearance(database)


def test_research_context_and_reports_disclose_synthetic_results(source):
    from easyicu.synthetic_sources import verify_source
    from easyicu.research_agent.research_context.builder import build_naive_research_context
    from easyicu.research_agent.reporting.manuscript_surface import render_reader_manuscript
    from easyicu.research_agent.reporting.readiness import render_report
    from easyicu.webserver.demo_source_storage import public_demo_egress_scope

    with public_demo_egress_scope(str(source)):
        context = build_naive_research_context(
            cohort=pd.DataFrame({"patientid": [1, 2], "map": [70., 80.]}), cohort_name="fixture",
            research_question="Describe the generated measurements.", database=verify_source(source)["database"])
        assert context.cohort.provenance["source_kind"] == "synthetic"
        assert "SYNTHETIC TEST DATA" in render_reader_manuscript("# Test manuscript\n\nGenerated result.")
    # A persisted context keeps the warning even outside the active source scope.
    report = render_report(context=context, plan=None, findings=[], per_step_records=[],
                           evidence=SimpleNamespace(records=lambda: []))
    assert "SYNTHETIC TEST DATA" in report
    assert "not clinical findings" in report


def test_saved_machine_and_review_reports_disclose_synthetic_results(source, tmp_path):
    from easyicu.research_agent.reporting.completion import run_status_payload
    from easyicu.research_agent.reporting.reviewer import ReviewerReport
    from easyicu.research_agent.reporting.reporting_checklist import ChecklistReport
    from easyicu.research_agent.reporting.manuscript_reader import build_manuscript_reader
    from easyicu.research_agent.reporting.readiness import _render_author_review_note
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.synthetic_source_provenance import write_report
    from easyicu.webserver.demo_source_storage import public_demo_egress_scope

    with public_demo_egress_scope(str(source)):
        status = run_status_payload(gates={"completion_status": "completed"},
                                    research_question="Synthetic example", code_version=None,
                                    stop_after_analysis=True, writer_probe_mode=False,
                                    writer_probe_failed_steps=[], force_diagnostic_only=False)
        reviewer = ReviewerReport()
        checklist = ChecklistReport(name="Synthetic check", version="1", items=[])
        payloads = [status, reviewer.to_json(), checklist.to_json(), build_manuscript_reader(
            manuscript="# Generated example\n\n## Methods\n\nInvented measurements.",
            evidence=EvidenceStore(tmp_path / "evidence"))]
        gates = dict.fromkeys(["execution_complete", "evidence_complete", "numeric_verified",
                              "analysis_validated", "manuscript_ready", "publication_ready"], False)
        prose = [reviewer.to_markdown(), checklist.to_markdown(), _render_author_review_note(
            status="completed", gates=gates, findings=[], per_step_records=[])]
        bound = write_report(tmp_path / "bound.md", "# Generated example")
        assert (tmp_path / "bound.md").read_text() == bound
        prose.append(bound)
    for payload in payloads:
        assert payload["source_kind"] == "synthetic"
        assert payload["contains_patient_data"] is False
        assert "not clinical findings" in payload["notice"]
    assert all("SYNTHETIC TEST DATA" in text for text in prose)


def test_forged_report_disclosure_does_not_grant_egress(source):
    from easyicu.synthetic_source_provenance import mark_report, active_provenance
    from easyicu.research_agent.providers.client_trust import source_requires_clearance
    from easyicu.webserver.demo_source_storage import public_demo_egress_scope

    mark_report("Untrusted report", {"source_kind": "synthetic"})
    assert source_requires_clearance("hirid")
    with pytest.raises(RuntimeError), public_demo_egress_scope(str(source)):
        assert active_provenance()
        raise RuntimeError("failed synthetic job")
    assert active_provenance() == {}


def test_the_normal_extraction_handoff_preserves_synthetic_egress(source, tmp_path, monkeypatch):
    import json
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import verify_source
    from easyicu.synthetic_source_provenance import source_fields
    from easyicu.research_agent.providers.client_trust import source_requires_clearance
    from easyicu.webserver import dataio, sources
    from easyicu.webserver.demo_source_storage import public_demo_egress_scope

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "app"))
    database = verify_source(source)["database"]
    out = tmp_path / "prepared"
    try:
        dataio.make_export_runner(data_path=str(source), database=database,
                                  modules=["demographics"], concepts={"demographics": ["age", "sex"]},
                                  export_format="parquet", out_dir=str(out))(
                                      SimpleNamespace(emit=lambda event: None))
    finally:
        clear_global_loader()
    manifest = json.loads((out / "_manifest.json").read_text())
    assert manifest["source_kind"] == "synthetic"
    assert "not clinical findings" in (out / "README.md").read_text()
    assert source_fields(out)["source_kind"] == "synthetic"
    bound = sources.register_source(str(out), active=True)["sources"][-1]
    assert bound["source_kind"] == "synthetic"
    with public_demo_egress_scope(str(out)):
        assert not source_requires_clearance(database)
    # A copied manifest or edited output has no producer authority.
    import shutil
    clone = tmp_path / "copied"
    shutil.copytree(out, clone)
    assert not source_fields(clone)
    target = out / "demographics.parquet"
    frame = pd.read_parquet(target)
    frame.loc[0, "age"] += 1
    frame.to_parquet(target, index=False)
    assert not source_fields(out)
    with public_demo_egress_scope(str(out)):
        assert source_requires_clearance(database)


def test_a_reader_rebuilt_after_the_job_keeps_its_saved_disclosure(source, tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.manuscript_reader import build_manuscript_reader
    from easyicu.synthetic_source_provenance import write_report
    from easyicu.webserver.demo_source_storage import public_demo_egress_scope
    from easyicu.research_agent.providers.client_trust import source_requires_clearance

    saved = tmp_path / "manuscript.md"
    with public_demo_egress_scope(str(source)):
        write_report(saved, "# Generated example\n\n## Methods\n\nInvented measurements.")
    reader = build_manuscript_reader(manuscript=saved.read_text(), evidence=EvidenceStore(tmp_path / "evidence"))
    assert reader["source_kind"] == "synthetic"
    assert "not clinical findings" in reader["notice"]
    assert source_requires_clearance("hirid")
