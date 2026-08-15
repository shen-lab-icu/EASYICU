from __future__ import annotations

import hashlib
import json

import fitz
import pytest

from easyicu.research_agent.reporting.system_validation_report import (
    SystemValidationReport,
    build_system_validation_receipt,
    projection_payload_sha256,
    render_system_validation_html,
)
from easyicu.webserver import agent_pipeline_runs, agent_runs


def _pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(36, 36, 560, 800), text, fontsize=8)
    raw = document.tobytes()
    document.close()
    return raw


def _bound_pdf_text(report: dict) -> str:
    status_label = (
        "REVIEWER DEMONSTRATION COMPLETE"
        if report["status"] == "engineering_validation_complete"
        else "ENGINEERING VALIDATION INCOMPLETE"
    )
    return "\n".join(
        (
            report["title"],
            report["subtitle"],
            report["executive_summary"],
            report["thesis"],
            f"Run {report['run_id']}",
            "authority=engineering_validation_only",
            "publication_authorized=false",
            status_label,
            "ENGINEERING VALIDATION ONLY · NOT A CLINICAL MANUSCRIPT",
            projection_payload_sha256(report),
        )
    )


def test_manuscript_document_boundary_is_fixed_and_media_typed(tmp_path) -> None:
    (tmp_path / "manuscript_scaffold.pdf").write_bytes(b"%PDF-governed-draft")
    (tmp_path / "system_validation_report.html").write_text(
        "<!doctype html><title>validation</title>", encoding="utf-8"
    )
    (tmp_path / "other.pdf").write_bytes(b"%PDF-not-allowed")

    allowed = agent_runs.read_run_artifact_bytes(
        str(tmp_path), "manuscript_scaffold.pdf"
    )
    rejected = agent_runs.read_run_artifact_bytes(str(tmp_path), "other.pdf")
    validation = agent_runs.read_run_artifact_bytes(
        str(tmp_path), "system_validation_report.html"
    )
    json_reader = agent_runs.read_run_artifact(
        str(tmp_path), "manuscript_scaffold.pdf"
    )

    assert allowed == {
        "ok": True,
        "name": "manuscript_scaffold.pdf",
        "content": b"%PDF-governed-draft",
        "media_type": "application/pdf",
    }
    assert rejected["error"] == "artifact_not_allowed"
    assert validation["ok"] is True
    assert validation["media_type"] == "text/html; charset=utf-8"
    assert json_reader["error"] == "artifact_json_not_allowed"


def test_system_validation_governance_cannot_be_upgraded_to_publication() -> None:
    governance = agent_runs.project_artifact_governance(
        {
            "readiness": {
                "status": "awaiting_human_signoff",
                "signed": True,
                "reportable": True,
            },
            "gate": {"status": "analysis_only"},
            "signoff": None,
        },
        artifact={"name": "system_validation_report.html"},
    )

    assert governance["authority_class"] == "easyicu_system_validation_report"
    assert governance["claim_ceiling"] == "engineering_validation_only"
    assert governance["reportable"] is False
    assert governance["human_signoff"] == "not_signable"


def test_system_validation_pdf_registration_binds_receipt_and_ledger(tmp_path) -> None:
    report = {
        "schema_version": "easyicu.system-validation-report/1",
        "artifact_kind": "system_validation_report",
        "authority_class": "engineering_validation_only",
        "claim_ceiling": "engineering_validation_only",
        "reportable": False,
        "publication_authorized": False,
        "run_id": "run_validation",
        "title": "When Success Must Still Fail Closed",
        "subtitle": "Validation dossier",
        "status": "engineering_validation_complete",
        "executive_summary": "Engineering validation only.",
        "thesis": "Execution and publication authority remain separate.",
        "metrics": [],
        "lifecycle": [],
        "demonstrated": [],
        "not_demonstrated": [],
        "case_study": {
            "role": "bounded_demonstration_case",
            "question": "Bounded case",
            "analysis_type": "descriptive_epidemiology",
            "scientific_claim_ceiling": "descriptive_only",
            "generated_numbers": False,
            "primary_table": None,
            "figures": [],
        },
        "provider_usage": None,
        "scientific_findings": [],
        "source_bindings": [],
        "next_validation_work": [],
    }
    report_bytes = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    report_model = SystemValidationReport.model_validate(report)
    html_bytes = render_system_validation_html(report_model).encode("utf-8")
    (tmp_path / "system_validation_report.json").write_bytes(report_bytes)
    (tmp_path / "system_validation_report.html").write_bytes(html_bytes)
    pdf_bytes = _pdf_bytes(_bound_pdf_text(report))
    (tmp_path / "system_validation_report.pdf").write_bytes(pdf_bytes)
    initial_receipt = build_system_validation_receipt(
        report_payload=report,
        report_bytes=report_bytes,
        html_bytes=html_bytes,
    )
    receipt_bytes = json.dumps(
        initial_receipt, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    (tmp_path / "system_validation_report_receipt.json").write_bytes(receipt_bytes)
    initial_artifacts = [
        {
            "name": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "kind": "json" if name.endswith(".json") else "document",
        }
        for name, raw in (
            ("system_validation_report.json", report_bytes),
            ("system_validation_report_receipt.json", receipt_bytes),
            ("system_validation_report.html", html_bytes),
        )
    ]
    (tmp_path / "evidence_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.web-research-pipeline-ledger/1",
                "artifacts": initial_artifacts,
            }
        ),
        encoding="utf-8",
    )

    registered = agent_pipeline_runs.register_system_validation_pdf(tmp_path)

    assert registered["ok"] is True
    assert registered["claim_ceiling"] == "engineering_validation_only"
    receipt = json.loads(
        (tmp_path / "system_validation_report_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["pdf"]["sha256"] == hashlib.sha256(pdf_bytes).hexdigest()
    ledger = json.loads(
        (tmp_path / "evidence_ledger.json").read_text(encoding="utf-8")
    )
    assert {row["name"] for row in ledger["artifacts"]} == {
        "system_validation_report.json",
        "system_validation_report_receipt.json",
        "system_validation_report.html",
        "system_validation_report.pdf",
    }

    receipt_row = next(
        row
        for row in ledger["artifacts"]
        if row["name"] == "system_validation_report_receipt.json"
    )
    receipt_row["sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    (tmp_path / "evidence_ledger.json").write_text(
        json.dumps(ledger), encoding="utf-8"
    )
    assert agent_pipeline_runs.register_system_validation_pdf(tmp_path)["ok"] is True


def test_system_validation_pdf_registration_rejects_changed_or_unsafe_bytes(
    tmp_path,
) -> None:
    test_system_validation_pdf_registration_binds_receipt_and_ledger(tmp_path)
    report_path = tmp_path / "system_validation_report.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")

    with pytest.raises(
        agent_pipeline_runs.ResearchPipelineRunError,
        match="do not match their receipt",
    ):
        agent_pipeline_runs.register_system_validation_pdf(tmp_path)

    report_path.write_bytes(report_path.read_bytes()[:-1])
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    (tmp_path / "system_validation_report.pdf").write_bytes(
        _pdf_bytes("Unrelated but privacy-safe PDF")
    )
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as unrelated:
        agent_pipeline_runs.register_system_validation_pdf(tmp_path)
    assert unrelated.value.code == "system_validation_pdf_content_binding_mismatch"

    (tmp_path / "system_validation_report.pdf").write_bytes(
        _pdf_bytes(f"{_bound_pdf_text(report_payload)}\nUnsafe source /Users/reviewer/private")
    )
    with pytest.raises(
        agent_pipeline_runs.ResearchPipelineRunError,
        match="browser privacy boundary",
    ):
        agent_pipeline_runs.register_system_validation_pdf(tmp_path)


def test_manuscript_document_boundary_rejects_symlink(tmp_path) -> None:
    outside = tmp_path.parent / "outside-manuscript.pdf"
    outside.write_bytes(b"%PDF-outside")
    (tmp_path / "manuscript_scaffold.pdf").symlink_to(outside)

    result = agent_runs.read_run_artifact_bytes(
        str(tmp_path), "manuscript_scaffold.pdf"
    )

    assert result["error"] == "artifact_path_unsafe"


def test_web_projection_requires_exact_draft_receipt_binding(tmp_path) -> None:
    tex = tmp_path / "manuscript_scaffold.tex"
    pdf = tmp_path / "manuscript_scaffold.pdf"
    tex.write_text("draft source", encoding="utf-8")
    pdf.write_bytes(b"%PDF-bound")
    receipt = {
        "schema_version": "easyicu.manuscript_pdf_receipt.v1",
        "status": "rendered",
        "draft_watermark": True,
        "security": {
            "network_allowed": False,
            "shell_escape_allowed": False,
            "untrusted_input_mode": True,
        },
        "source": {
            "name": tex.name,
            "sha256": hashlib.sha256(tex.read_bytes()).hexdigest(),
        },
        "bibliography": None,
        "pdf": {
            "name": pdf.name,
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "bytes": pdf.stat().st_size,
        },
    }
    (tmp_path / "manuscript_pdf_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    loaded_receipt, documents = agent_pipeline_runs._validated_manuscript_documents(
        tmp_path
    )

    assert loaded_receipt == receipt
    assert [row["name"] for row in documents] == [
        "manuscript_scaffold.pdf",
        "manuscript_scaffold.tex",
    ]

    pdf.write_bytes(b"%PDF-drifted")
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as drifted:
        agent_pipeline_runs._validated_manuscript_documents(tmp_path)
    assert drifted.value.code == "research_pipeline_pdf_document_digest_mismatch"
