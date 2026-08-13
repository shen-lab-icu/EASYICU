from __future__ import annotations

import hashlib
import json

import pytest

from easyicu.webserver import agent_pipeline_runs, agent_runs


def test_manuscript_document_boundary_is_fixed_and_media_typed(tmp_path) -> None:
    (tmp_path / "manuscript_scaffold.pdf").write_bytes(b"%PDF-governed-draft")
    (tmp_path / "other.pdf").write_bytes(b"%PDF-not-allowed")

    allowed = agent_runs.read_run_artifact_bytes(
        str(tmp_path), "manuscript_scaffold.pdf"
    )
    rejected = agent_runs.read_run_artifact_bytes(str(tmp_path), "other.pdf")
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
    assert json_reader["error"] == "artifact_json_not_allowed"


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
