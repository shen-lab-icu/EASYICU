from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import capabilities
from easyicu.webserver import provider_adapter
from easyicu.webserver import settings as settings_store
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


def test_idea_mining_requires_source_seed_before_resolve_or_mine() -> None:
    client = TestClient(app)

    resolve = client.post("/api/ideas/resolve-source", json={})
    mine = client.post("/api/ideas/mine", json={})

    assert resolve.status_code == 400
    assert mine.status_code == 400
    assert resolve.json()["detail"]["error"] == "idea_source_required"
    assert mine.json()["detail"]["error"] == "idea_source_required"


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


def test_idea_url_resolution_falls_back_to_crossref_when_journal_html_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, payload: bytes):
            self.payload = payload

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            return None

        def read(self, limit: int = -1) -> bytes:
            return self.payload if limit < 0 else self.payload[:limit]

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        url = getattr(req, "full_url", str(req))
        calls.append(url)
        if "api.crossref.org" not in url:
            raise OSError("HTTP Error 403: Forbidden")
        return FakeResponse(
            json.dumps(
                {
                    "status": "ok",
                    "message": {
                        "title": ["Vasopressors or Fluids in Early Septic Shock"],
                        "container-title": ["New England Journal of Medicine"],
                        "published-online": {"date-parts": [[2026, 6, 11]]},
                        "DOI": "10.1056/NEJMoa2516225",
                    },
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr(idea_mining.request, "urlopen", fake_urlopen)

    payload = idea_mining.resolve_source(
        {
            "source_type": "url",
            "url": "https://www.nejm.org/doi/full/10.1056/NEJMoa2516225",
            "allow_network": True,
        }
    )

    assert len(calls) == 2
    assert payload["source_adapter"]["status"] == "metadata_fetched"
    assert payload["source_adapter"]["metadata_source"] == "crossref"
    assert payload["source_adapter"]["network_calls"] == 2
    assert payload["suggested_payload"]["title"] == "Vasopressors or Fluids in Early Septic Shock"
    assert payload["suggested_payload"]["topic"] == "Vasopressors or Fluids in Early Septic Shock"
    assert payload["suggested_payload"]["journal"] == "New England Journal of Medicine"
    assert payload["suggested_payload"]["year"] == 2026
    assert payload["suggested_payload"]["doi"] == "10.1056/NEJMoa2516225"
    assert payload["resolved_source"]["title"] == "Vasopressors or Fluids in Early Septic Shock"


def test_idea_mining_maps_resolved_nejm_title_to_vasopressor_fluid_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)

    response = TestClient(app).post(
        "/api/ideas/mine",
        json={
            "source_type": "url",
            "topic": "NEJM idea seed",
            "title": "Vasopressors or Fluids in Early Septic Shock",
            "journal": "New England Journal of Medicine",
            "year": 2026,
            "doi": "10.1056/NEJMoa2516225",
            "url": "https://www.nejm.org/doi/full/10.1056/NEJMoa2516225",
        },
    )

    assert response.status_code == 200
    idea = response.json()["idea_ledger"][0]
    concept_ids = {row["concept_id"] for row in idea["mapped_concepts"]}
    assert {"vaso_ind", "death"} <= concept_ids
    assert concept_ids & {"total_input_ml", "fluid_balance", "fluid_balance_cumulative"}
    assert "death and death" not in idea["rationale"].lower()


def test_idea_mining_accepts_zotero_source_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)

    response = TestClient(app).post(
        "/api/ideas/mine",
        json={
            "source_type": "zotero",
            "source_origin": "zotero_desktop",
            "source_origin_label": "Zotero Desktop",
            "topic": "Early Vasopressors in Septic Shock",
            "title": "Early Vasopressors in Septic Shock",
            "journal": "Intensive Care Medicine",
            "year": 2026,
            "doi": "10.1000/example",
            "citation_key": "smith2026vasopressors",
            "zotero_key": "ABC123",
            "excerpt": "Early vasopressors may define a measurable ICU exposure.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    source = body["source_evidence"][0]
    assert source["source_type"] == "zotero"
    assert source["source_origin"] == "zotero_desktop"
    assert source["source_origin_label"] == "Zotero Desktop"
    assert source["citation_key"] == "smith2026vasopressors"
    assert source["zotero_key"] == "ABC123"
    assert source["source_text_stored"] is False
    assert body["privacy"]["patient_rows_returned"] is False
    assert "stay_id" not in str(body)
    assert "subject_id" not in str(body)


def test_pasted_literature_source_flows_to_agent_project_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(
        capabilities,
        "_AUDIT_PATH",
        tmp_path / "capability_tool_audit.jsonl",
    )
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(
        idea_mining,
        "_AGENT_PROJECTS_ROOT",
        tmp_path / "agent_project_seeds",
    )
    monkeypatch.setattr(
        idea_mining,
        "_AGENT_PROJECTS_PATH",
        tmp_path / "agent_projects.json",
    )
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)
    client = TestClient(app)

    imported = client.post(
        "/api/capabilities/zotero/import",
        json={
            "text": """@article{smith2026shock,
              title={Early Vasopressors in Septic Shock},
              journal={Intensive Care Medicine},
              year={2026},
              doi={10.1000/example},
              abstract={Early vasopressors may define a measurable ICU exposure.}
            }"""
        },
    )
    assert imported.status_code == 200
    source_payload = imported.json()["suggested_payload"]
    assert source_payload["source_origin"] == "pasted_literature"

    mined = client.post("/api/ideas/mine", json=source_payload)
    assert mined.status_code == 200
    run = mined.json()
    source = run["source_evidence"][0]
    idea = run["idea_ledger"][0]
    assert source["source_origin"] == "pasted_literature"
    assert source["source_type"] == "zotero"
    assert source["citation_key"] == "smith2026shock"

    planned = client.post(
        "/api/ideas/plan",
        json={
            "run_id": run["run_id"],
            "idea_id": idea["idea_id"],
            "mode": "plan",
            "plan_edits": "Keep the pasted literature seed as hypothesis-generating.",
        },
    )
    assert planned.status_code == 200

    handoff = client.post(
        "/api/ideas/handoff",
        json={"run_id": run["run_id"], "idea_id": idea["idea_id"]},
    )
    assert handoff.status_code == 200

    created = client.post(
        "/api/ideas/create-agent-project",
        json={"run_id": run["run_id"], "idea_id": idea["idea_id"]},
    )
    assert created.status_code == 200
    project = created.json()["project"]
    assert project["source_run_id"] == run["run_id"]
    assert project["source"]["source_origin"] == "pasted_literature"
    assert project["source"]["source_type"] == "zotero"
    assert project["source"]["citation_key"] == "smith2026shock"
    assert project["source"]["title"] == "Early Vasopressors in Septic Shock"

    listed = client.post("/api/ideas/agent-projects", json={"limit": 10})
    assert listed.status_code == 200
    listed_body = listed.json()
    assert listed_body["privacy"]["patient_rows_returned"] is False
    assert any(row["study_id"] == project["study_id"] for row in listed_body["projects"])
    assert "stay_id" not in str(listed_body)
    assert "subject_id" not in str(listed_body)


def test_idea_plan_stage_precedes_agent_handoff_and_stays_metadata_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)
    client = TestClient(app)

    mined = client.post(
        "/api/ideas/mine",
        json={
            "source_type": "url",
            "topic": "NEJM idea seed",
            "title": "Vasopressors or Fluids in Early Septic Shock",
            "journal": "New England Journal of Medicine",
            "year": 2026,
            "doi": "10.1056/NEJMoa2516225",
            "url": "https://www.nejm.org/doi/full/10.1056/NEJMoa2516225",
        },
    )
    assert mined.status_code == 200
    run = mined.json()
    idea = run["idea_ledger"][0]

    planned = client.post(
        "/api/ideas/plan",
        json={
            "run_id": run["run_id"],
            "idea_id": idea["idea_id"],
            "mode": "plan",
            "plan_edits": "Keep this as an ICU observational feasibility plan before Agent run.",
        },
    )
    assert planned.status_code == 200
    plan_payload = planned.json()
    plan = plan_payload["plan"]

    assert plan_payload["schema_version"] == "easyicu.web_idea_plan/1"
    assert plan_payload["planner"]["stage"] == "idea_mining_plan_before_agent"
    assert plan_payload["planner"]["agent_run_created"] is False
    assert plan_payload["planner"]["draft_unlocked"] is False
    assert plan_payload["privacy"]["patient_rows_returned"] is False
    assert plan_payload["privacy"]["agent_run_created"] is False
    assert plan_payload["privacy"]["draft_unlocked"] is False
    assert plan_payload["privacy"]["reportable"] is False
    assert plan["reference_analysis_patterns"]
    assert plan["clinical_icu_constraints"]
    assert plan["required_user_confirmations"]
    assert "prepare or register a usable EasyICU export" in plan["required_user_confirmations"]
    assert isinstance(plan["analysis_plan"][0], dict)
    assert plan["analysis_plan"][0]["phase"] == "Question"
    assert "Freeze the clinical question" in plan["analysis_plan"][0]["title"]
    assert "Prior work does not automatically block the idea" in str(plan["analysis_plan"])
    assert plan["agent_boundary"]["agent_run_created"] is False
    assert plan["agent_boundary"]["draft_unlocked"] is False
    assert "target-trial-style translation" in str(plan["reference_analysis_patterns"])
    assert "stay_id" not in str(plan_payload)
    assert "subject_id" not in str(plan_payload)
    assert "tableRows" not in str(plan_payload)

    loaded = client.post("/api/ideas/run", json={"run_id": run["run_id"]})
    assert loaded.status_code == 200
    assert loaded.json()["idea_plan"]["schema_version"] == "easyicu.web_idea_plan/1"

    handoff = client.post(
        "/api/ideas/handoff",
        json={
            "run_id": run["run_id"],
            "idea_id": idea["idea_id"],
            "plan_edits": "Use first ICU stay and require explicit mortality horizon.",
        },
    )
    assert handoff.status_code == 200
    frozen = handoff.json()
    assert frozen["handoff_plan"]["reference_analysis_patterns"]
    assert frozen["handoff_plan"]["human_plan_notes"] == (
        "Use first ICU stay and require explicit mortality horizon."
    )
    assert frozen["handoff_plan"]["selection_mode"] == "human_curated_with_text_edits"
    assert frozen["agent_seed"]["requires_human_confirmation"] is True
    assert frozen["agent_seed"]["reportable"] is False
    assert frozen["agent_seed"]["draft_unlocked"] is False


def test_idea_mining_does_not_recommend_mock_export_as_real_feasibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(
        idea_mining,
        "_active_export",
        lambda: (
            {"label": "MOCK", "path": str(tmp_path / "mock_export")},
            {
                "ok": True,
                "label": "MOCK",
                "database": "MOCK",
                "path": str(tmp_path / "mock_export"),
                "summary": {"stays": 10, "modules": 20, "total_rows": 1000},
            },
        ),
    )
    monkeypatch.setattr(
        idea_mining,
        "_export_index",
        lambda export: {
            "concept_to_file": {
                "vaso_ind": {"module": "vasopressors"},
                "total_input_ml": {"module": "vitals"},
                "death": {"module": "outcome"},
            },
            "entity_ids": {"1", "2"},
            "demo_like": True,
        },
    )

    response = TestClient(app).post(
        "/api/ideas/mine",
        json={
            "source_type": "url",
            "title": "Vasopressors or Fluids in Early Septic Shock",
            "excerpt": "Vasopressor and fluid strategy may affect mortality in septic shock.",
        },
    )

    assert response.status_code == 200
    idea = response.json()["idea_ledger"][0]
    assert idea["feasibility"]["tier"] == "demo_only"
    assert idea["go_no_go"] == "hold"
    assert "MOCK/demo" in idea["feasibility"]["reason"]
    assert "real EasyICU export" in idea["next_action"]


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


def test_pubmed_connector_setting_blocks_idea_discovery_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network should not be called when connector is off")

    monkeypatch.setattr(idea_mining, "_pubmed_esearch", fail_network)
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: {**settings_store.DEFAULTS, "connector_pubmed_enabled": False},
    )

    response = TestClient(app).post(
        "/api/ideas/discover",
        json={"topic": "septic shock vasopressor fluid mortality", "allow_network": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status"] == "blocked_network_opt_in_required"
    assert payload["search_performed"] is False
    assert payload["privacy"]["network_calls"] == 0


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


def test_idea_mining_maps_ards_peep_pdf_excerpt_to_respiratory_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)

    response = TestClient(app).post(
        "/api/ideas/mine",
        json={
            "source_type": "pdf",
            "title": "Balancing lung recruitment and venous congestion in ARDS: rethinking PEEP",
            "journal": "Intensive Care Medicine",
            "year": 2026,
            "excerpt": (
                "Positive end-expiratory pressure (PEEP) is a cornerstone of mechanical "
                "ventilation in acute respiratory distress syndrome. Ventilator settings "
                "and oxygenation may be associated with ICU mortality."
            ),
            "allow_network": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    idea = payload["idea_ledger"][0]
    concept_ids = {row["concept_id"] for row in idea["mapped_concepts"]}
    assert {"mech_vent", "peep", "death"} <= concept_ids
    assert idea["prior_art"]["status"] == "not_checked_external_search_required"
    assert payload["pre_experiment"]["status"] == "blocked"
    assert payload["privacy"]["external_llm_calls"] == 0
    assert payload["privacy"]["network_calls"] == 0
    assert "stay_id" not in str(payload)
    assert "subject_id" not in str(payload)


def test_provider_config_route_writes_private_env_without_returning_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / "provider.env"
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(provider_adapter, "_DEFAULT_PROVIDER_ENV_FILE", env_path)
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings_store, "_CONFIG_PATH", settings_path)

    response = TestClient(app).post(
        "/api/agent-runs/provider-config",
        json={
            "provider": "openai",
            "api_key": "sk-test-provider-config",
            "base_url": "http://127.0.0.1:8787/v1",
            "model": "gpt5.4",
            "enable_ai": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["secrets_returned"] is False
    assert payload["provider_status"]["ready"] is True
    assert payload["provider_status"]["credential_present"] is True
    assert payload["provider_status"]["base_url_present"] is True
    assert payload["provider_status"]["model_present"] is True
    assert payload["settings"]["ai_enabled"] is True
    assert env_path.exists()
    assert (env_path.stat().st_mode & 0o777) == 0o600
    assert "sk-test-provider-config" in env_path.read_text(encoding="utf-8")
    assert "sk-test-provider-config" not in str(payload)
    assert "http://127.0.0.1:8787/v1" not in str(payload)
