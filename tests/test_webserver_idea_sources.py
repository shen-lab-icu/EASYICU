from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from easyicu.research_agent.discovery.discovery_handoff import (
    DiscoveryHandoffPacket,
    build_handoff_from_row,
)
from easyicu.webserver import capabilities
from easyicu.webserver import provider_adapter
from easyicu.webserver import settings as settings_store
from easyicu.webserver.ideas.handoff import (
    build_web_handoff_packet,
    map_web_ledger_row,
)
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
    assert (
        payload["suggested_payload"]["source_file_sha256"] == payload["pdf"]["sha256"]
    )
    assert "stay_id" not in str(payload)
    assert "subject_id" not in str(payload)
    assert "tableRows" not in str(payload)


def test_idea_mining_scans_local_literature_folder_without_full_text_persistence(
    tmp_path: Path,
) -> None:
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

    def fake_urlopen(req, timeout=0, **kwargs):  # type: ignore[no-untyped-def]
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

    monkeypatch.setattr(
        idea_mining,
        "_resolve_public_http_target",
        lambda url: SimpleNamespace(url=str(url)),
    )
    monkeypatch.setattr(idea_mining, "_open_public_url", fake_urlopen)
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
    assert (
        payload["suggested_payload"]["title"]
        == "Vasopressors or Fluids in Early Septic Shock"
    )
    assert (
        payload["suggested_payload"]["topic"]
        == "Vasopressors or Fluids in Early Septic Shock"
    )
    assert payload["suggested_payload"]["journal"] == "New England Journal of Medicine"
    assert payload["suggested_payload"]["year"] == 2026
    assert payload["suggested_payload"]["doi"] == "10.1056/NEJMoa2516225"
    assert (
        payload["resolved_source"]["title"]
        == "Vasopressors or Fluids in Early Septic Shock"
    )


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
        json={"text": """@article{smith2026shock,
              title={Early Vasopressors in Septic Shock},
              journal={Intensive Care Medicine},
              year={2026},
              doi={10.1000/example},
              abstract={Early vasopressors may define a measurable ICU exposure.}
            }"""},
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
    assert any(
        row["study_id"] == project["study_id"] for row in listed_body["projects"]
    )
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
    assert (
        "prepare or register a usable EasyICU export"
        in plan["required_user_confirmations"]
    )
    assert isinstance(plan["analysis_plan"][0], dict)
    assert plan["analysis_plan"][0]["phase"] == "Question"
    assert "Freeze the clinical question" in plan["analysis_plan"][0]["title"]
    assert "Prior work does not automatically block the idea" in str(
        plan["analysis_plan"]
    )
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
    canonical = DiscoveryHandoffPacket.model_validate(frozen["canonical_handoff"])
    assert canonical.human_confirmed is False
    assert canonical.analysis_ready is False
    assert frozen["canonical_handoff_path"] == "discovery_handoff.json"
    assert len(frozen["canonical_handoff_sha256"]) == 64
    assert (
        idea_mining._run_dir(run["run_id"]) / "discovery_handoff.json"
    ).is_file()


def test_web_handoff_adapter_matches_canonical_core_semantics(tmp_path: Path) -> None:
    idea = {
        "idea_id": "idea_adapter",
        "idea_title": "Vasopressor exposure and ICU mortality",
        "rationale": "Mapped from a bounded literature seed.",
        "go_no_go": "recommend",
        "go_no_go_reason": "Required concepts are available.",
        "outcome": "In-hospital mortality",
        "analysis_family": "association",
        "mapped_concepts": [
            {"role": "exposure", "concept_id": "vaso_ind", "label": "Vasopressor"},
            {"role": "outcome", "concept_id": "death", "label": "Mortality"},
        ],
        "feasibility": {"tier": "executable"},
        "prior_art": {"novelty_label": "unknown_until_search"},
        "next_action": "Run bounded feasibility review.",
    }
    source = {
        "source_id": "source_adapter",
        "source_type": "url",
        "title": "Bounded source",
        "evidence_quote": "A bounded gap statement.",
        "source_text_sha256": "a" * 64,
    }
    plan = {
        "selection_mode": "human_curated_with_text_edits",
        "research_question": "Is vasopressor exposure associated with ICU mortality?",
        "cohort": {"default": "adult ICU cohort from active EasyICU export"},
        "active_export_contract": {"database": "eicu"},
    }
    pre = {"status": "ready", "source": {"database": "eicu"}}
    mapped = map_web_ledger_row(
        idea=idea,
        source=source,
        plan=plan,
        pre_experiment=pre,
    )
    adapter_packet = build_web_handoff_packet(
        idea=idea,
        source=source,
        plan=plan,
        pre_experiment=pre,
        prior_art_check=None,
        run_dir=tmp_path,
    )
    core_packet = build_handoff_from_row(
        mapped,
        triage_report_path=tmp_path / "idea_mining_run.json",
        selection_mode="human_curated",
        selection_rationale=idea["rationale"],
        target_outcome="death",
        database="eicu",
        research_question=plan["research_question"],
        inclusion_criteria=[plan["cohort"]["default"]],
        human_confirmed=False,
    )
    adapter_payload = adapter_packet.model_dump(mode="json")
    core_payload = core_packet.model_dump(mode="json")
    adapter_payload.pop("created_at")
    core_payload.pop("created_at")
    assert adapter_payload == core_payload


def test_web_handoff_without_database_stays_unspecified(tmp_path: Path) -> None:
    packet = build_web_handoff_packet(
        idea={
            "idea_id": "idea_unknown_db",
            "idea_title": "Unknown-database ICU candidate",
            "go_no_go": "hold",
            "go_no_go_reason": "No active export is selected.",
            "outcome": "In-hospital mortality",
            "mapped_concepts": [],
        },
        source={},
        plan={},
        pre_experiment={},
        prior_art_check=None,
        run_dir=tmp_path,
    )
    assert packet.database == "unspecified"
    assert packet.database != "miiv"


@pytest.mark.parametrize(
    "tamper_target",
    [
        "artifact",
        "artifact_with_replan",
        "envelope",
        "partial_field",
        "legacy_identity",
    ],
)
def test_agent_project_rejects_tampered_canonical_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)
    run = idea_mining.mine_ideas(
        {
            "source_type": "url",
            "title": "Vasopressors and mortality in ICU patients",
            "excerpt": "Vasopressors may be associated with mortality.",
        }
    )
    idea = run["idea_ledger"][0]
    idea_mining.create_handoff(
        {"run_id": run["run_id"], "idea_id": idea["idea_id"]}
    )
    run_dir = idea_mining._run_dir(run["run_id"])
    if tamper_target in {"artifact", "artifact_with_replan"}:
        path = run_dir / "discovery_handoff.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["candidate_topic"] = "tampered artifact"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper_target == "envelope":
        path = run_dir / "idea_handoff.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["canonical_handoff"]["candidate_topic"] = "tampered envelope"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper_target == "partial_field":
        path = run_dir / "idea_handoff.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("canonical_handoff_path")
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path = run_dir / "idea_handoff.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["candidate_topic"] = "mismatched legacy topic"
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(idea_mining.IdeaMiningWebError) as exc_info:
        project_body = {"run_id": run["run_id"], "idea_id": idea["idea_id"]}
        if tamper_target == "artifact_with_replan":
            project_body["plan_edits"] = "A legitimate new plan note."
        idea_mining.create_agent_project(project_body)
    assert exc_info.value.detail["error"] == "canonical_handoff_integrity_error"


def test_legacy_handoff_refreshes_to_locked_unconfirmed_agent_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(
        idea_mining, "_AGENT_PROJECTS_ROOT", tmp_path / "agent_projects"
    )
    monkeypatch.setattr(
        idea_mining, "_AGENT_PROJECTS_PATH", tmp_path / "agent_projects.json"
    )
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)
    run = idea_mining.mine_ideas(
        {
            "source_type": "url",
            "title": "Vasopressors and mortality in ICU patients",
            "excerpt": "Vasopressors may be associated with mortality.",
        }
    )
    idea = run["idea_ledger"][0]
    handoff = idea_mining.create_handoff(
        {"run_id": run["run_id"], "idea_id": idea["idea_id"]}
    )
    run_dir = idea_mining._run_dir(run["run_id"])
    legacy = dict(handoff)
    legacy.pop("canonical_handoff")
    legacy.pop("canonical_handoff_path")
    legacy.pop("canonical_handoff_sha256")
    (run_dir / "idea_handoff.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    (run_dir / "discovery_handoff.json").unlink()

    result = idea_mining.create_agent_project(
        {"run_id": run["run_id"], "idea_id": idea["idea_id"]}
    )
    seed = result["project"]
    packet = DiscoveryHandoffPacket.model_validate(seed["canonical_handoff"])
    assert packet.human_confirmed is False
    assert packet.analysis_ready is False
    assert seed["human_confirmed"] is False
    assert seed["analysis_ready"] is False
    assert seed["requires_human_confirmation"] is True
    assert seed["draft_unlocked"] is False
    assert len(seed["canonical_handoff_sha256"]) == 64


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


def test_idea_literature_discovery_blocks_without_network_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network should not be called without opt-in")

    monkeypatch.setattr(idea_mining, "_pubmed_esearch", fail_network)
    response = TestClient(app).post(
        "/api/ideas/discover",
        json={
            "topic": "septic shock vasopressor fluid mortality",
            "allow_network": False,
        },
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
        json={
            "topic": "septic shock vasopressor fluid mortality",
            "allow_network": True,
        },
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
    monkeypatch.setattr(
        idea_mining, "_pubmed_esearch", lambda query, limit=5: ["12345"]
    )
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


def test_provider_config_route_fails_closed_when_enable_ai_omitted(
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
            "api_key": "sk-test-fail-closed",
            "base_url": "http://127.0.0.1:8787/v1",
            "model": "gpt5.4",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["settings"]["ai_enabled"] is False
    # ``agent_model_mode`` was asserted here too, but it was a write-only
    # mirror of this same gate that nothing ever read; ai_enabled is the gate.
    assert "agent_model_mode" not in payload["settings"]


def test_provider_config_rejects_invalid_enable_ai_before_writing_credentials(
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
            "api_key": "sk-must-not-be-written",
            "enable_ai": "sometimes",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error": "invalid_boolean",
        "field": "enable_ai",
    }
    assert not env_path.exists()
    assert not settings_path.exists()


def test_blocked_prior_art_recheck_does_not_clobber_successful_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A casual re-check without network opt-in must not re-block the seed."""
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path)
    run_id = "priorartkeep"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    successful = {
        "ok": True,
        "run_id": run_id,
        "idea_id": "vasopressor-timing",
        "prior_art": {
            "status": "searched_no_hits",
            "search_performed": True,
            "results": [],
        },
    }
    (run_dir / "prior_art_check.json").write_text(
        json.dumps(successful, ensure_ascii=False), encoding="utf-8"
    )

    blocked = idea_mining.check_prior_art(
        {"run_id": run_id, "idea_title": "vasopressor timing", "allow_network": False}
    )

    assert blocked["prior_art"]["search_performed"] is False
    assert blocked["persisted"] is False
    assert blocked["retained_prior_art_status"] == "searched_no_hits"
    on_disk = json.loads((run_dir / "prior_art_check.json").read_text(encoding="utf-8"))
    assert on_disk["prior_art"]["search_performed"] is True
    assert on_disk["prior_art"]["status"] == "searched_no_hits"


def test_execution_gate_treats_failed_prior_art_search_as_unreviewed() -> None:
    """status=search_failed returned no reviewable metadata; the gate stays closed."""
    idea = {"go_no_go": "recommend"}
    pre_experiment = {"status": "ready", "missing_required_concepts": []}
    failed_check = {"prior_art": {"status": "search_failed", "search_performed": True}}
    gate = idea_mining._execution_gate(idea, pre_experiment, failed_check)
    assert "run prior-art review before Agent execution" in gate["blockers"]

    reviewed_check = {
        "prior_art": {"status": "searched_no_hits", "search_performed": True}
    }
    gate_ok = idea_mining._execution_gate(idea, pre_experiment, reviewed_check)
    assert gate_ok["blockers"] == []
    assert gate_ok["agent_run_ready_after_human_confirmation"] is True
