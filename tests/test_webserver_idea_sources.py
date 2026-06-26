from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.ideas import mining as idea_mining
from easyicu.webserver.app import app


def _write_pdf(path: Path, text: str) -> bytes:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    path.write_bytes(pdf_bytes)
    return pdf_bytes


def test_idea_mining_ingests_selected_local_pdf_metadata_only(tmp_path: Path) -> None:
    pdf_bytes = _write_pdf(
        tmp_path / "shock-paper.pdf",
        "Early vasopressors and fluid strategy may affect septic shock mortality.",
    )

    response = TestClient(app).post(
        "/api/ideas/ingest-pdf",
        json={
            "filename": "shock-paper.pdf",
            "content_base64": base64.b64encode(pdf_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "local_pdf_ingest"
    assert payload["source_adapter"]["status"] == "local_pdf_excerpt_ready"
    assert payload["source_adapter"]["network_calls"] == 0
    assert payload["privacy"]["full_text_stored"] is False
    assert payload["privacy"]["patient_rows_returned"] is False
    assert "septic shock mortality" in payload["pdf"]["excerpt"]
    assert payload["suggested_payload"]["source_type"] == "pdf"
    assert payload["suggested_payload"]["source_file_name"] == "shock-paper.pdf"
    assert payload["suggested_payload"]["source_file_sha256"] == payload["pdf"]["sha256"]
    assert "stay_id" not in str(payload)
    assert "subject_id" not in str(payload)
    assert "tableRows" not in str(payload)


def test_idea_mining_scans_local_literature_folder_without_full_text_persistence(tmp_path: Path) -> None:
    literature = tmp_path / "papers"
    literature.mkdir()
    _write_pdf(
        literature / "lactate-review.pdf",
        "Lactate trajectories in adult ICU patients can motivate mortality prediction studies.",
    )

    response = TestClient(app).post(
        "/api/ideas/literature-folder",
        json={"path": str(literature)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "local_literature_folder"
    assert payload["folder"]["pdf_count"] == 1
    assert payload["documents"][0]["filename"] == "lactate-review.pdf"
    assert payload["representative"]["full_text_stored"] is False
    assert "mortality prediction" in payload["representative"]["excerpt"]
    assert payload["source_adapter"]["network_calls"] == 0
    assert payload["privacy"]["full_text_stored"] is False
    assert payload["suggested_payload"]["source_type"] == "pdf"


def test_idea_literature_discovery_blocks_without_network_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network should not be called without opt-in")

    monkeypatch.setattr(idea_mining, "_pubmed_esearch", fail_network)
    response = TestClient(app).post(
        "/api/ideas/discover",
        json={"topic": "septic shock vasopressor fluid mortality", "allow_network": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status"] == "blocked_network_opt_in_required"
    assert payload["search_performed"] is False
    assert payload["privacy"]["network_calls"] == 0
    assert payload["source_candidates"] == []
    assert payload["idea_candidates"] == []
    assert payload["queries_to_run"]


def test_idea_literature_discovery_maps_pubmed_candidates_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_pubmed_esearch", lambda query, limit=5: ["12345"])
    monkeypatch.setattr(
        idea_mining,
        "_pubmed_article_records",
        lambda ids: [
            {
                "pmid": "12345",
                "title": "Vasopressors and Fluids in Early Septic Shock",
                "journal": "New England Journal of Medicine",
                "year": 2026,
                "doi": "10.1056/NEJMoa2516225",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                "abstract_excerpt": (
                    "A trial suggests early vasopressor and fluid-resuscitation strategy "
                    "may affect outcomes in adult septic shock."
                ),
                "evidence_sentence": (
                    "A trial suggests early vasopressor and fluid-resuscitation strategy "
                    "may affect outcomes in adult septic shock."
                ),
                "full_text_stored": False,
            }
        ],
    )

    response = TestClient(app).post(
        "/api/ideas/discover",
        json={
            "topic": "septic shock vasopressor fluid mortality",
            "journal": "New England Journal of Medicine",
            "allow_network": True,
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status"] == "searched"
    assert payload["search_performed"] is True
    assert payload["privacy"]["network_calls"] >= 1
    assert payload["privacy"]["external_llm_calls"] == 0
    assert payload["privacy"]["full_text_stored"] is False
    assert payload["source_candidates"][0]["pmid"] == "12345"
    assert "vasopressor" in payload["source_candidates"][0]["evidence_quote"].lower()
    idea = payload["idea_candidates"][0]["idea"]
    concept_ids = {row["concept_id"] for row in idea["mapped_concepts"]}
    assert {"vaso_ind", "death"} & concept_ids
    assert payload["suggested_payload"]["doi"] == "10.1056/NEJMoa2516225"
    assert "stay_id" not in str(payload)
    assert "subject_id" not in str(payload)
    assert "tableRows" not in str(payload)
