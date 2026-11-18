"""Synthetic completed analysis and failed derived manuscript outcomes."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from easyicu.webserver.agent_pipeline_runs import _write_projection
from easyicu.webserver.pi_copilot.projections import project_job, project_run_outcome, project_run_row
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.run_record import RunGate
from easyicu.research_agent.reporting.completion import apply_host_delivery_completion, completion_outcomes
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    complete_study,
)


def _project(tmp_path, *, plan=None, prepare=None, **changes):
    axes = {
        "execution_complete": True,
        "analysis_validated": True,
        "evidence_complete": False,
        "numeric_verified": True,
        "manuscript_ready": False,
        "manuscript_generated": True,
        "missing_evidence_count": 0,
        "evidence_error_count": 1,
        "publication_ready": False,
        "paper_authorized": False,
        **changes,
    }
    run_dir = tmp_path / "wrapper" / "pipeline" / "source"
    run_dir.mkdir(parents=True)
    if plan is not None:
        plan_path = run_dir / "analysis_plan.json"
        raw = plan.model_dump_json().encode("utf-8")
        plan_path.write_bytes(raw)
        (run_dir / "manifest.json").write_text(json.dumps({"current_plan_authority": {
            "relative_path": plan_path.name, "sha256": hashlib.sha256(raw).hexdigest(),
        }}))
    if prepare is not None:
        prepare(run_dir, axes)
    (run_dir / "run_status.json").write_text(json.dumps({"gates": axes}))
    return _write_projection(
        wrapper_dir=tmp_path / "wrapper",
        study=complete_study(),
        provider={"provider": "openai", "model": "synthetic"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
    )


def test_a_completed_analysis_keeps_its_status_when_the_manuscript_fails(tmp_path):
    result = _project(tmp_path)
    gate = result["gate"]
    assert gate["status"] == "analysis_only"
    assert gate["analysis_outcome"] == {"status": "completed", "reason_code": None}
    assert gate["manuscript_outcome"] == {
        "status": "not_passed", "reason_code": "manuscript_checks_not_passed",
    }
    assert gate["reportable"] is False
    assert gate["draft_unlocked"] is False
    checks = {row["id"]: row["passed"] for row in gate["checks"]}
    assert checks["manuscript_ready"] is False
    assert checks["publication_ready"] is False
    assert checks["paper_authorized"] is False
    projection = project_job({
        "id": "synthetic-job", "kind": "agent-run", "status": "done", "result": result,
    })
    assert projection["analysis_results_available"] is True
    assert projection["analysis_outcome"] == gate["analysis_outcome"]
    assert projection["manuscript_outcome"] == gate["manuscript_outcome"]
    durable = project_run_outcome({"ok": True, **result})
    assert durable["analysis_outcome"] == gate["analysis_outcome"]
    assert durable["manuscript_outcome"] == gate["manuscript_outcome"]


@pytest.mark.parametrize("axis", ["execution_complete", "analysis_validated"])
def test_a_failed_analysis_stays_blocked(tmp_path, axis):
    gate = _project(tmp_path, **{axis: False})["gate"]
    assert gate["status"] == "blocked"
    assert gate["analysis_outcome"] == {
        "status": "blocked", "reason_code": f"{axis}_not_satisfied",
    }
    assert gate["manuscript_outcome"]["status"] == "not_passed"


def test_workflow_keeps_the_manuscript_reason_on_its_own_stage(tmp_path):
    gate = _project(tmp_path)["gate"]
    run = {
        "run_id": "synthetic", "run_type": "full", "engine": "easyicu.research_agent.pipeline",
        "gate_status": gate["status"], "gate_reason": gate["reason"],
        "gate_checks": {row["id"]: row["passed"] for row in gate["checks"]},
        "analysis_outcome": gate["analysis_outcome"], "manuscript_outcome": gate["manuscript_outcome"],
        "artifact_names": ["agent_plan.json", "evidence_ledger.json", "source_run_manifest.json", "result_tables.json", "manuscript_draft.json"],
    }
    snapshot = build_research_workflow_snapshot(study={"id": "synthetic", "question": "Describe the aggregate."},
        active_export_present=False, active_job=None, latest_run=run)
    stages = {row.id: row for row in snapshot.stages}
    assert stages["analysis"].status == "complete"
    assert stages["manuscript"].status == "blocked"
    assert stages["manuscript"].reason_code == "manuscript_checks_not_passed"
    assert snapshot.analysis_validation_retry_available is False
    summary = project_run_row(run)
    assert summary["analysis_outcome"] == run["analysis_outcome"]
    assert summary["manuscript_outcome"] == run["manuscript_outcome"]


def test_legacy_record_separates_a_manuscript_stop_without_promoting_any_check():
    payload = {
        "status": "blocked", "reason": "research_agent_pipeline_failed_closed",
        "detail": {"reason_code": "manuscript_checks_not_passed"},
        "reportable": False, "draft_unlocked": False,
        "checks": [{"id": "execution_complete", "passed": True}, {"id": "analysis_validated", "passed": True},
                   {"id": "evidence_complete", "passed": False}, {"id": "numeric_verified", "passed": False}],
    }
    record = RunGate.from_payload(payload)
    assert record.status == "analysis_only"
    assert record.analysis_outcome.status == "completed"
    assert record.manuscript_outcome.reason_code == "manuscript_checks_not_passed"
    assert record.to_dict()["checks"] == payload["checks"]
    assert record.to_dict()["manuscript_detail"] == payload["detail"]
    assert "detail" not in record.to_dict()
    assert payload["status"] == "blocked"


@pytest.mark.parametrize("change,status,reason", [
    ({"manuscript_requested": False}, "not_requested", None),
    ({"evidence_complete": False}, "not_passed", "evidence_complete_not_satisfied"),
    ({"numeric_verified": False}, "not_passed", "numeric_verified_not_satisfied"),
    ({"writer_stop": {"reason_code": "writer_provider_transport_unavailable", "provider_http_status": 503},
      "evidence_complete": False}, "not_passed", "writer_provider_transport_unavailable"),
    ({}, "passed", None),
])
def test_deliverable_verdicts_use_typed_codes_without_changing_analysis(change, status, reason):
    axes = {"execution_complete": True, "analysis_validated": True,
            "evidence_complete": True, "numeric_verified": True, "manuscript_ready": True, **change}
    outcomes = completion_outcomes(axes)
    assert outcomes["analysis_outcome"]["status"] == "completed"
    assert outcomes["manuscript_outcome"] == {"status": status, "reason_code": reason}
    pending = completion_outcomes(axes, pending=True)
    assert pending["analysis_outcome"]["status"] == "pending"
    assert pending["manuscript_outcome"]["status"] == "pending"


@pytest.mark.parametrize("kind, intent", [
    ("manuscript_pdf", "Typeset the overnight collection report"),
    ("article_preview", "Show the collection report in the browser"),
    ("references", "Provide the source bibliography for the collection report"),
])
@pytest.mark.parametrize("has_owner", [True, False])
def test_a_promised_host_deliverable_without_a_current_artifact_fails_closed(tmp_path, kind, intent, has_owner):
    from easyicu.research_agent.contracts.product_files import PlanDeliverables

    step = AnalysisStep(step_id="report_delivery", intent=intent, host_delivery_kind=kind)
    plan = AnalysisPlan(
        research_question=intent, steps=[step] if has_owner else [],
        deliverables=PlanDeliverables(requested=(kind,), confirmed=True, source="user_edit"),
    )
    old_name = {
        "article_preview": "manuscript_provenance.json",
        "manuscript_pdf": "manuscript_pdf_receipt.json",
        "references": "manuscript_scaffold.bib",
    }[kind]
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    (wrapper / old_name).write_text("Old collection report export")
    result = _project(tmp_path, plan=plan, manuscript_ready=True, evidence_complete=True, evidence_error_count=0)
    gate = result["gate"]
    reason = f"research_pipeline_{kind}_delivery_unverified"
    assert gate["status"] == "blocked"
    assert gate["reason"] == reason
    expected_steps = [
        {"step_id": "report_delivery", "kind": kind, "status": "unverified", "reason_code": reason},
    ] if has_owner else []
    assert gate["host_delivery_steps"] == expected_steps
    assert gate["manuscript_outcome"] == {"status": "not_passed", "reason_code": reason}
    assert gate["draft_unlocked"] is False
    assert gate["reportable"] is False
    assert not (wrapper / old_name).exists()


@pytest.mark.parametrize("kind, name, question", [
    ("article_preview", "manuscript_draft.json", "Deliver a browser preview."),
    ("manuscript_pdf", "manuscript_scaffold.pdf", "Export a manuscript PDF."),
    ("references", "manuscript_scaffold.bib", "Include a bibliography."),
])
@pytest.mark.parametrize("change", ["none", "output_name", "missing", "empty", "digest", "run", "plan", "authority"])
def test_the_writer_delivers_only_the_current_run_plan_and_artifact(tmp_path, monkeypatch, kind, name, question, change):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.contracts.product_files import HOST_DELIVERY_RECEIPT, HOST_DELIVERY_SOURCES
    from easyicu.research_agent.literature import CitationRecord, LiteratureBundle, LiteratureScreeningDecision
    from easyicu.research_agent.planning.requirement_typing.delivery import bind_requested_host_delivery
    from easyicu.research_agent.reporting import pdf_render, write_phase
    from easyicu.research_agent.schema import CritiqueReport
    from easyicu.webserver.agent_runs import read_run_artifact_bytes

    plan = bind_requested_host_delivery(AnalysisPlan(research_question=question, steps=[]))
    (step,) = plan.steps

    def fake_pdf(*, tex_path, bib_path, output_dir, draft_watermark):
        pdf = output_dir / "manuscript_scaffold.pdf"
        pdf.write_bytes(b"%PDF-1.5 synthetic shift report")
        receipt = pdf_render._write_receipt(
            output_dir=output_dir, tex_path=tex_path, bib_path=bib_path, pdf_path=pdf,
            engine="synthetic", draft_watermark=draft_watermark,
        )
        return SimpleNamespace(success=True, pdf_path=pdf, receipt_path=receipt, engine="synthetic")

    monkeypatch.setattr(write_phase, "render_pdf_for_run", fake_pdf)
    pipeline = SimpleNamespace(
        _enable_latex=False, _enable_pdf_render=False, _latex_draft_watermark=False,
        _latex_venue_template=None, _enable_causal_audit=False,
        _enable_reporting_checklist=False, _enable_reviewer_round=False,
    )
    bound = "# Shift recovery\n\n## Methods\n\nThe report summarizes the synthetic observation."

    def prepare(run_dir, axes):
        evidence = EvidenceStore(run_dir)
        bound_path = run_dir / "manuscript_scaffold_bound.md"
        bound_path.write_text(bound)
        literature = LiteratureBundle(
            research_question=question,
            citations=[CitationRecord(key="shift_methods", title="Synthetic collection shift methods", year="2022")],
            screening_decisions=[LiteratureScreeningDecision(
                citation_key="shift_methods", source="synthetic", disposition="include",
                evidence_role="method", rationale="Synthetic fixture for the bibliography owner.",
            )],
        )
        write_phase._persist_manuscript_provenance_artifact(
            bound=bound, numeric_binding_map={}, run_dir=run_dir, evidence=evidence, findings=[],
            plan=plan, literature=literature, evidence_records=[],
        )
        write_phase._persist_manuscript_critique(
            critique=CritiqueReport(status="pass", reviewer="synthetic"), run_dir=run_dir,
            evidence=evidence, producer="synthetic",
        )
        write_phase._publish_and_audit_manuscript(
            pipeline, bound=bound, bound_path=bound_path, context=SimpleNamespace(research_question=question),
            plan=plan, current_verified_evidence_records=[], evidence=evidence, findings=[], literature=literature,
            manuscript_authors=None, manuscript_title=None, per_step_records=[], repro_envelope=None,
            run_dir=run_dir, run_id="run-shift-recovery" if change == "output_name" else run_dir.name,
            emit_progress=lambda *args, **kwargs: None,
        )
        receipt = run_dir / HOST_DELIVERY_RECEIPT
        recorded = json.loads(receipt.read_text())
        assert recorded["run_id"] == run_dir.name
        assert recorded["deliveries"][kind]["step_id"] == step.step_id
        if change in {"run", "plan"}:
            altered = json.loads(receipt.read_text())
            field = "run_id" if change == "run" else "plan_sha256"
            altered[field] = "different-run" if change == "run" else "f" * 64
            receipt.write_text(json.dumps(altered))
        source = run_dir / HOST_DELIVERY_SOURCES[kind][-1]
        if change == "missing":
            primary_source = {
                "article_preview": "manuscript_scaffold_bound.md",
                "manuscript_pdf": "manuscript_scaffold.pdf",
                "references": "manuscript_scaffold.bib",
            }[kind]
            (run_dir / primary_source).unlink()
        elif change == "empty":
            source.write_bytes(b"")
        elif change == "digest":
            source.write_bytes(source.read_bytes() + b" ")
        elif change == "authority":
            # A legacy persisted gate can predate per-deliverable status rows.
            # Its receipt identifies the promise, but cannot replace the lost plan.
            (run_dir / "analysis_plan.json").unlink()
            return
        apply_host_delivery_completion(axes, plan=plan, run_dir=run_dir)
        if change not in {"none", "output_name"}:
            assert axes["completion_status"] == "diagnostic_only"
            assert axes["host_delivery_failure"] == f"research_pipeline_{kind}_delivery_unverified"

    result = _project(tmp_path, plan=plan, prepare=prepare, manuscript_ready=True, evidence_complete=True)
    gate = result["gate"]
    wrapper = tmp_path / "wrapper"
    if change not in {"none", "output_name"}:
        assert gate["status"] == "blocked"
        assert gate["reason"] == f"research_pipeline_{kind}_delivery_unverified"
        if kind == "article_preview":
            assert not json.loads((wrapper / name).read_text())["markdown_preview"]
        else:
            assert not (wrapper / name).exists()
    else:
        assert gate["status"] == "analysis_only"
        assert gate["host_delivery_steps"] == [
            {"kind": kind, "step_id": step.step_id, "status": "verified", "reason_code": None},
        ]
        loaded = read_run_artifact_bytes(str(wrapper), name)
        assert loaded["ok"], loaded
        artifacts = {row["name"] for row in result["artifacts"] if row["bytes"]}
        assert name in artifacts
        assert artifacts.intersection({
            "manuscript_draft.json", "manuscript_scaffold.pdf", "manuscript_scaffold.bib",
        }) == {name}


@pytest.mark.parametrize("change, status, manuscript", [
    ({"execution_complete": False, "failed_steps": [{"step_id": "summary", "status": "failed"}]}, "blocked",
     {"status": "not_passed", "reason_code": "execution_complete_not_satisfied"}),
    ({"execution_complete": False, "missing_steps": ["summary"]}, "blocked",
     {"status": "not_passed", "reason_code": "execution_complete_not_satisfied"}),
    ({"analysis_validated": False}, "blocked",
     {"status": "not_passed", "reason_code": "analysis_validated_not_satisfied"}),
    ({"manuscript_requested": False}, "analysis_only", {"status": "not_requested", "reason_code": None}),
])
def test_delivery_checks_follow_execution_analysis_and_writer_policy(tmp_path, change, status, manuscript):
    plan = AnalysisPlan(research_question="Export a PDF of the shift summary.", steps=[
        AnalysisStep(step_id="summary", intent="Summarize recovery by ward"),
        AnalysisStep(step_id="pdf", intent="Typeset the summary", host_delivery_kind="manuscript_pdf"),
    ])

    def prepare(run_dir, axes):
        axes["completion_status"] = status
        apply_host_delivery_completion(axes, plan=plan, run_dir=run_dir)
        assert axes["completion_status"] == status
        assert "host_delivery_failure" not in axes
        assert "host_delivery_steps" not in axes
        # Older persisted gates may already have a stale delivery failure.
        axes["host_delivery_failure"] = "research_pipeline_manuscript_pdf_delivery_unverified"
        assert completion_outcomes(axes)["manuscript_outcome"] == manuscript

    gate = _project(tmp_path, plan=plan, prepare=prepare, **change)["gate"]
    assert gate["status"] == status
    assert gate["manuscript_outcome"] == manuscript
    assert gate["reason"] != "research_pipeline_manuscript_pdf_delivery_unverified"
    assert "host_delivery_steps" not in gate


def test_lost_plan_hides_promised_pdf_even_while_delivery_is_inactive(tmp_path, monkeypatch):
    from easyicu.research_agent.contracts.product_files import HOST_DELIVERY_RECEIPT, HOST_DELIVERY_RECEIPT_SCHEMA
    from easyicu.webserver import agent_pipeline_runs

    def prepare(run_dir, axes):
        (run_dir / HOST_DELIVERY_RECEIPT).write_text(json.dumps({
            "schema_version": HOST_DELIVERY_RECEIPT_SCHEMA, "run_id": run_dir.name,
            "planned_owners": {"manuscript_pdf": "pdf"}, "deliveries": {},
        }))

    def documents(run_dir):
        assert run_dir is None, "the lost plan cannot authorize reading an old PDF"
        return None, []

    monkeypatch.setattr(agent_pipeline_runs, "_validated_manuscript_documents", documents)
    result = _project(tmp_path, prepare=prepare, analysis_validated=False)
    assert result["gate"]["status"] == "blocked"
    assert "manuscript_scaffold.pdf" not in {row["name"] for row in result["artifacts"]}
