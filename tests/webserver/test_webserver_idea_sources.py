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
from easyicu.webserver.ideas import direct_evidence_search
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
    assert "vaso_ind" in concept_ids
    # death must no longer be fabricated as the default outcome when the
    # source text does not evidence it.
    assert "death" not in concept_ids
    assert concept_ids & {"total_input_ml", "fluid_balance", "fluid_balance_cumulative"}
    assert "death and death" not in idea["rationale"].lower()
    assert idea["outcome"] is None
    assert idea["go_no_go"] == "hold"
    assert idea["feasibility"]["tier"] == "design_incomplete"
    assert "outcome_not_mapped" in idea["feasibility"]["design_blockers"]


def test_idea_mining_does_not_recommend_an_outcome_without_a_predictor() -> None:
    death = idea_mining._concept_hit("death")
    assert death is not None

    idea = idea_mining._idea_from_source(
        {"source_id": "source_outcome_only", "title": "ICU mortality"},
        "ICU mortality",
        [death],
        {
            "concept_to_file": {"death": {"module": "outcome"}},
            "entity_ids": {"stay-1"},
            "demo_like": False,
        },
    )

    assert idea["outcome"] == death["label"]
    assert idea["go_no_go"] == "hold"
    assert idea["feasibility"]["tier"] == "design_incomplete"
    assert "predictor_or_exposure_not_mapped" in idea["feasibility"]["design_blockers"]
    assert "Confirm the primary exposure" in idea["next_action"]


def test_chinese_fluid_balance_topic_preserves_the_exposure_concept() -> None:
    hits = idea_mining._match_concepts("比较入 ICU 早期液体平衡与机械通气脱机候选设计")
    concept_ids = {row["concept_id"] for row in hits}

    assert "mech_vent" in concept_ids
    assert concept_ids & {"total_input_ml", "fluid_balance", "fluid_balance_cumulative"}


def test_ventilator_liberation_idea_separates_exposure_episode_and_design_support() -> (
    None
):
    text = "比较成人 ICU 入科后早期累积液体平衡与机械通气脱机"
    hits = idea_mining._match_concepts(text)

    idea = idea_mining._idea_from_source(
        {"source_id": "source_liberation", "title": text},
        text,
        hits,
        {"concept_to_file": {}, "entity_ids": set(), "demo_like": False},
    )

    roles = {row["concept_id"]: row["role"] for row in idea["mapped_concepts"]}
    assert idea["exposure_or_predictor"] == "Cumulative Fluid Balance"
    assert roles["fluid_balance_cumulative"] == "exposure"
    assert roles["mech_vent"] == "eligibility_or_episode"
    assert "Mechanical Ventilation Mode" not in idea["exposure_or_predictor"]
    assert idea["outcome"] is None
    assert idea["design_support"]["card_id"] == "mechanical_ventilation_liberation"
    assert "Successful liberation" in idea["design_support"]["outcome_family"]
    assert idea["design_support"]["authority"] == "advisory_design_support_only"
    assert any(
        '"ventilator liberation"[Title/Abstract]' in query
        for query in idea["prior_art"]["queries_to_run"]
    )


def test_one_sentence_chinese_idea_maps_topic_without_inventing_design_details() -> (
    None
):
    text = "我想研究液体平衡和撤机的关系"
    hits = idea_mining._match_concepts(text)

    idea = idea_mining._idea_from_source(
        {"source_id": "source_one_sentence", "title": text},
        text,
        hits,
        {"concept_to_file": {}, "entity_ids": set(), "demo_like": False},
    )

    roles = {row["concept_id"]: row["role"] for row in idea["mapped_concepts"]}
    assert idea["exposure_or_predictor"] == "Cumulative Fluid Balance"
    assert roles["fluid_balance_cumulative"] == "exposure"
    assert roles["mech_vent"] == "eligibility_or_episode"
    assert idea["design_support"]["card_id"] == "mechanical_ventilation_liberation"
    assert idea["design_support"]["study_families"] == [
        "association",
        "time_to_event",
        "prediction",
    ]
    assert (
        "Distinguish extubation from durable liberation"
        in idea["design_support"]["eligibility_candidates"]
    )
    assert "baseline or time-varying" in idea["design_support"]["exposure_family"]
    assert "readiness-to-wean landmark" in idea["design_support"]["time_zero"]
    assert any(
        "Competing-risk or multistate analysis" in value
        for value in idea["design_support"]["recommended_methods"]
    )
    assert (
        "Alternative durable-liberation gap"
        in idea["design_support"]["sensitivity_analyses"]
    )
    assert idea["population"].endswith("(age scope pending)")
    assert idea["outcome"] is None
    assert idea["unresolved_slots"] == ["outcome_not_mapped"]


def test_negative_mortality_instruction_is_not_treated_as_outcome_evidence() -> None:
    text = "比较液体平衡与机械通气脱机，不得默认院内死亡作为主要结局"
    hits = idea_mining._match_concepts(text)

    assert idea_mining._pick_outcome(text, hits) is None


def test_competing_death_handling_is_not_misread_as_the_primary_outcome() -> None:
    text = (
        "优先验证机械通气研究中成功撤机与拔管失败的操作性结局定义，"
        "明确持续成功撤机、再插管、气管切开、死亡、出院和转院的处理。"
    )
    hits = idea_mining._match_concepts(text)

    idea = idea_mining._idea_from_source(
        {"source_id": "source_liberation_refinement", "title": text},
        text,
        hits,
        {"concept_to_file": {}, "entity_ids": set(), "demo_like": False},
    )

    assert idea["outcome"] is None
    assert "In-hospital Mortality" not in idea["idea_title"]
    assert all(
        row["concept_id"] != "death" or row["role"] != "outcome"
        for row in idea["mapped_concepts"]
    )


def test_negative_mortality_instruction_overrides_a_generated_mortality_title() -> None:
    text = "\n".join(
        [
            "比较液体平衡与机械通气脱机，不要默认死亡结局",
            "Cumulative fluid balance and in-hospital mortality in adult ICU patients",
        ]
    )
    hits = idea_mining._match_concepts(text)

    assert idea_mining._pick_outcome(text, hits) is None


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
            "plan_fields": {
                "research_question": "Does 24-hour fluid balance relate to durable liberation?",
                "population": "Ventilated adult ICU patients",
                "exposure": "Cumulative fluid balance during ICU hours 0-24",
                "outcome": "Extubation without reintubation within 48 hours",
                "time_window": "Exposure: ICU hours 0-24; durability: 48 hours",
            },
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
    assert plan["research_question"] == (
        "Does 24-hour fluid balance relate to durable liberation?"
    )
    assert plan["outcome"] == "Extubation without reintubation within 48 hours"
    assert "outcome and time window" not in plan["required_user_confirmations"]
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
    assert (idea_mining._run_dir(run["run_id"]) / "discovery_handoff.json").is_file()


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
    idea_mining.create_handoff({"run_id": run["run_id"], "idea_id": idea["idea_id"]})
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
    (run_dir / "idea_handoff.json").write_text(json.dumps(legacy), encoding="utf-8")
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


def test_create_agent_project_requires_exact_nonempty_identity(
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
    idea_mining.create_handoff({"run_id": run["run_id"], "idea_id": idea["idea_id"]})

    with pytest.raises(idea_mining.IdeaMiningWebError) as exc_info:
        idea_mining.create_agent_project({"run_id": run["run_id"]})

    assert exc_info.value.detail["error"] == "idea_identity_required"


def test_run_scoped_idea_actions_do_not_fall_back_to_first_idea(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)
    run = idea_mining.mine_ideas(
        {
            "source_type": "manual",
            "title": "Lactate and ICU mortality",
            "excerpt": "Lactate may be associated with ICU mortality.",
        }
    )

    assert idea_mining._selected_idea(run, "") is None
    for action in (idea_mining.plan_idea, idea_mining.bounded_sample_feasibility):
        with pytest.raises(idea_mining.IdeaMiningWebError) as exc_info:
            action({"run_id": run["run_id"]})
        assert exc_info.value.detail["error"] == "idea_not_found"


def test_handoff_staleness_compares_the_nested_plan_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path)
    run_id = "nested-plan"
    run_dir = idea_mining._run_dir(run_id)
    run_dir.mkdir(parents=True)
    plan = {
        "human_plan_notes": "Use the first ICU stay.",
        "selection_mode": "human_curated_with_text_edits",
        "plan_status": "planned_requires_final_confirmation",
        "analysis_plan": [{"phase": "Question"}],
        "execution_gate": {"blockers": []},
    }
    (run_dir / "idea_plan.json").write_text(
        json.dumps({"schema_version": "test/1", "plan": plan}), encoding="utf-8"
    )
    handoff = {"handoff_plan": dict(plan)}

    assert not idea_mining._handoff_plan_is_stale(handoff, run_id, {})

    revised = dict(plan)
    revised["human_plan_notes"] = "Use all ICU stays."
    (run_dir / "idea_plan.json").write_text(
        json.dumps({"schema_version": "test/1", "plan": revised}), encoding="utf-8"
    )
    assert idea_mining._handoff_plan_is_stale(handoff, run_id, {})

    (run_dir / "idea_plan.json").unlink()
    assert not idea_mining._handoff_plan_is_stale(handoff, run_id, {})


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


def test_conversational_sepsis_scope_compiles_to_bounded_pubmed_queries() -> None:
    topic = (
        "请帮我从 MIMIC-IV 寻找成人首次 ICU 入住后 24 小时内 Sepsis-3 "
        "与院内死亡的研究机会，并完成数据提取和分析计划。"
    )

    queries = idea_mining._discovery_queries(
        topic,
        "",
        {"exposure": "sep3_sofa1_max", "outcome": "death"},
    )

    assert len(queries) == 6
    assert all('"Sepsis-3"[Title/Abstract]' in query for query in queries)
    assert all(
        '"mortality"[Title/Abstract]' in query
        for index, query in enumerate(queries)
        if index != 1
    )
    assert "definition[Title/Abstract]" in queries[1]
    assert any("observational[Title/Abstract]" in query for query in queries)
    all_years, recent = queries[2:4]
    assert "adult[Title/Abstract]" in all_years
    assert "pediatric[Title/Abstract]" in all_years
    assert "NOT (Review[Publication Type]" in all_years
    assert "association[Title/Abstract]" in all_years
    assert "prevalence[Title/Abstract]" in all_years
    assert "Date - Publication" not in all_years
    assert "2021/01/01" in recent
    assert all("请帮我" not in query for query in queries)


def test_sofa2_execution_concept_uses_retrieval_aliases_without_granting_evidence() -> (
    None
):
    scope = {
        "topic": "adult ICU experimental SOFA-2 Sepsis-3 phenotype and mortality",
        "exposure": "experimental Sepsis-3 phenotype using SOFA-2",
        "outcome": "in-hospital mortality",
        "exposure_concept": "sep3_sofa2",
        "outcome_concept": "death",
        "population": "adult ICU stays",
        "database": "miiv",
        "analysis_family": "association_study",
    }

    queries = idea_mining._discovery_queries(scope["topic"], "", scope)
    fallback = direct_evidence_search.build_query(scope)

    assert len(queries) == 6
    assert all('"SOFA-2"[Title/Abstract]' in query for query in queries)
    assert all(
        '"Sepsis-3"[Title/Abstract]' in query
        for index, query in enumerate(queries)
        if index != 1
    )
    assert all(
        '"SOFA"[Title/Abstract]' in query
        for index, query in enumerate(queries)
        if index != 1
    )
    assert '"SOFA-2"[Title/Abstract]' in queries[1]
    assert '"Sepsis-3"[Title/Abstract]' not in queries[1]
    assert all('"SOFA-2 sepsis"[Title/Abstract]' not in query for query in queries)
    assert '"hospital mortality"[Title/Abstract]' in fallback
    assert '"MIMIC-IV"[Title/Abstract]' in fallback

    # Retrieval aliases increase recall only.  A Sepsis-3/SOFA paper that does
    # not study the exact SOFA-2 exposure remains related context at this stage.
    decision = direct_evidence_search.screen_article(
        {
            "pmid": "123",
            "title": "Sepsis-3 prevalence and hospital mortality in adult ICU stays",
            "design_excerpt": (
                "An observational adult ICU cohort assessed Sepsis-3 prevalence "
                "and hospital mortality using SOFA."
            ),
            "publication_types": ["Observational Study"],
        },
        scope,
    )
    assert decision["disposition"] == "exclude"
    assert decision["evidence_role"] == "related_context"


def test_inferred_concept_search_uses_literature_identity_not_display_label() -> None:
    queries = idea_mining._discovery_queries(
        "adult ICU Sepsis-3 prevalence and in-hospital mortality",
        "",
        {"population": "adult ICU stays"},
    )

    assert all('"Sepsis-3"[Title/Abstract]' in query for query in queries)
    assert all('"mortality"[Title/Abstract]' in query for query in queries)
    assert all("Sepsis-3 (SOFA-1 based)" not in query for query in queries)
    assert all("In-hospital Mortality" not in query for query in queries)


def test_pediatric_literature_scope_does_not_compile_an_adult_filter() -> None:
    queries = idea_mining._discovery_queries(
        "儿童 ICU 脓毒症与院内死亡",
        "",
        {"exposure_concept": "sep3_sofa1", "outcome_concept": "death"},
    )

    direct = queries[1]
    assert "pediatric[Title/Abstract]" in direct
    assert "NOT (adult[Title/Abstract]" in direct
    assert "AND (adult[Title/Abstract]" not in direct


def test_generic_sepsis_maps_to_canonical_sofa1_and_preserves_explicit_adjustment() -> (
    None
):
    text = (
        "纳入成人首次 ICU stay，采用 Sepsis-3 操作定义，主要结局为院内死亡，"
        "同时描述 SOFA、乳酸和尿量，初步按年龄和性别调整。"
    )

    hits = idea_mining._match_concepts(text)
    concept_ids = {row["concept_id"] for row in hits}

    assert "sep3_sofa1" in concept_ids
    assert "sep3_sofa2" not in concept_ids
    assert idea_mining._requested_adjustment_concepts(text, hits) == ["age", "sex"]

    explicit = (
        "主要暴露采用标准 Sepsis-3（传统 SOFA / sep3_sofa1），主要结局为院内死亡；"
        "乳酸和尿量仅用于描述，按年龄和性别调整。"
    )
    explicit_hits = idea_mining._match_concepts(explicit)
    assert idea_mining._requested_exposure_concepts(explicit, explicit_hits) == [
        "sep3_sofa1"
    ]
    idea = idea_mining._idea_from_source(
        {"source_id": "source", "title": "Sepsis study"},
        explicit,
        explicit_hits,
        {
            "concept_to_file": {
                concept_id: {"module": row.get("module")}
                for row in explicit_hits
                if (concept_id := str(row.get("concept_id") or ""))
            },
            "entity_ids": {"1"},
            "demo_like": False,
        },
    )
    assert idea["idea_title"].startswith("Sepsis-3 (SOFA-1 based)")
    assert idea["requested_adjustment_concepts"] == ["age", "sex"]
    assert all(
        '"Sepsis-3"[Title/Abstract]' in query
        for query in idea["prior_art"]["queries_to_run"]
    )
    queries = idea["prior_art"]["queries_to_run"]
    assert '"mortality"[Title/Abstract]' in queries[1]
    assert all("mortality" in query for query in queries)
    assert all(
        '"SOFA Score (Total)"[Title/Abstract]' not in query
        for query in idea["prior_art"]["queries_to_run"]
    )


def test_prior_art_executes_the_queries_prespecified_by_the_mined_idea(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path)
    run_id = "idea_prespecified_query"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    queries = [
        f'("Sepsis-3"[Title/Abstract] AND "mortality"[Title/Abstract]) '
        f'AND ICU[Title/Abstract] AND "stratum {index}"[Title/Abstract]'
        for index in range(5)
    ]
    (run_dir / "idea_mining_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "selected_idea_id": "idea_sepsis",
                "source_evidence": [{"title": "Sepsis idea"}],
                "idea_ledger": [
                    {
                        "idea_id": "idea_sepsis",
                        # The title contains another mapped concept. It must not
                        # be allowed to replace the frozen primary query.
                        "idea_title": "SOFA Score (Total) and mortality",
                        "prior_art": {"queries_to_run": queries},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    checked = idea_mining.check_prior_art(
        {"run_id": run_id, "idea_id": "idea_sepsis", "allow_network": False}
    )

    assert checked["prior_art"]["queries_to_run"] == queries


def test_chinese_conversational_idea_keeps_clinical_action_in_prior_art_queries() -> (
    None
):
    source = {
        "title": "我发现ICU患者夜间发生低血压后，医生的处理差异很大",
        "evidence_quote": "这里面有没有值得研究的问题？",
    }

    queries = idea_mining._prior_art_queries(
        source,
        source["title"],
    )

    assert len(queries) == 5
    assert all('"hypotension"[Title/Abstract]' in query for query in queries)
    assert all(
        '"management"[Title/Abstract]' in query
        or '"treatment"[Title/Abstract]' in query
        for query in queries
    )
    assert queries[0] != queries[1]
    assert '"practice variation"[Title/Abstract]' in queries[0]
    assert '"nighttime"[Title/Abstract]' in queries[1]
    assert '"after-hours"[Title/Abstract]' in queries[1]
    assert all("neonat*[Title/Abstract]" in query for query in queries)
    assert all('"intensive care"[Title/Abstract]' in query for query in queries)
    assert all('"icu"[Title/Abstract]' not in query.lower() for query in queries)


def test_prior_art_search_records_stratified_retrieval_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = idea_mining._prior_art_queries(
        {"title": "Fluid balance and ventilator liberation"},
        "Fluid balance and ventilator liberation",
        exposure="fluid_balance_cumulative",
        outcome="Mechanical ventilation liberation outcomes",
        topic_aliases=(
            "ventilator liberation",
            "extubation failure",
            "reintubation",
        ),
    )
    assert len(queries) == 5
    assert "extubation failure" in queries[0]
    assert '"cumulative fluid balance"[Title/Abstract]' in queries[1]
    assert '"ventilator liberation"[Title/Abstract]' in queries[1]
    assert "pre-admission fluid rows" not in queries[1]
    assert "cohort[Title/Abstract]" in queries[2]
    assert "review[Publication Type]" in queries[3]
    assert "MIMIC[Title/Abstract]" in queries[4]

    query_ids = {
        query: (["100", "200"] if index == 0 else ["100"])
        for index, query in enumerate(queries)
    }
    monkeypatch.setattr(
        idea_mining,
        "_pubmed_esearch",
        lambda query, limit=5: query_ids[query],
    )
    monkeypatch.setattr(
        idea_mining,
        "_pubmed_article_records",
        lambda ids: [
            {
                "pmid": pmid,
                "title": f"Candidate {pmid}",
                "journal": "Critical Care",
                "year": 2025,
            }
            for pmid in ids
        ],
    )

    prior = idea_mining._pubmed_prior_art(queries)

    assert [row["id"] for row in prior["query_strata"]] == [
        "clinical_landscape",
        "candidate_topic",
        "direct_observational_candidates",
        "review_or_guideline",
        "critical_care_database",
    ]
    first = next(row for row in prior["results"] if row["pmid"] == "100")
    assert first["matched_query_strata"] == [
        "clinical_landscape",
        "candidate_topic",
        "direct_observational_candidates",
        "review_or_guideline",
        "critical_care_database",
    ]
    assert len(first["matched_queries"]) == 5


def test_conversational_prior_art_ranks_direct_fit_and_excludes_population_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "title": "我发现 ICU 患者夜间发生低血压后，医生的处理差异很大",
        "evidence_quote": "这里面有没有值得研究的问题？",
    }
    queries = idea_mining._prior_art_queries(source, source["title"])
    ids_by_query = {
        queries[0]: [
            "asthma",
            "neonatal",
            "pediatric_trial",
            "variation",
            "perioperative",
        ],
        queries[1]: ["case", "direct"],
        queries[2]: ["fresh"],
        queries[3]: [],
        queries[4]: [],
    }
    records = {
        "asthma": {
            "title": "Critical Care Management of Severe Asthma Exacerbations",
            "abstract_excerpt": (
                "Severe asthma in the ICU may cause systemic hypotension; "
                "the review describes ventilator management."
            ),
        },
        "neonatal": {
            "title": "Management of Neonatal Hypotension and Shock",
            "abstract_excerpt": "Neonatal intensive care treatment of preterm infants.",
        },
        "pediatric_trial": {
            "title": (
                "Epinephrine vs Norepinephrine as Initial Treatment in Children "
                "With Septic Shock"
            ),
            "abstract_excerpt": (
                "A comparative treatment study in children admitted to a pediatric "
                "intensive care unit with septic shock."
            ),
        },
        "case": {
            "title": "Mitral valve-in-valve implantation: A case report",
            "abstract_excerpt": (
                "One ICU patient developed hypotension during an overnight admission."
            ),
        },
        "direct": {
            "title": (
                "The organizational structure of an intensive care unit influences "
                "treatment of hypotension among critically ill patients"
            ),
            "abstract_excerpt": (
                "A retrospective ICU cohort compared hypotension treatment during "
                "weekday daytime, nighttime, and weekend staffing levels."
            ),
        },
        "variation": {
            "title": (
                "Practice Patterns in the Initiation of Secondary Vasopressors "
                "during Septic Shock"
            ),
            "abstract_excerpt": (
                "A multicenter ICU cohort quantified physician and hospital practice "
                "variation in vasopressor treatment."
            ),
        },
        "fresh": {
            "title": (
                "Fluid Response Evaluation in Sepsis Hypotension and Shock: "
                "A Randomized Clinical Trial"
            ),
            "abstract_excerpt": (
                "Adult ICU admission was anticipated and fluid responsiveness guided "
                "fluid and vasopressor resuscitation."
            ),
        },
        "perioperative": {
            "title": "The Role of Permissive Hypotension in Neuroanesthesia",
            "abstract_excerpt": (
                "A practice variation survey of induced intraoperative hypotension "
                "among neuroanesthesia clinicians in critical care settings."
            ),
        },
    }

    monkeypatch.setattr(
        idea_mining,
        "_pubmed_esearch",
        lambda query, limit=5: ids_by_query[query],
    )
    monkeypatch.setattr(
        idea_mining,
        "_pubmed_article_records",
        lambda ids: [
            {"pmid": pmid, "journal": "Critical Care", **records[pmid]} for pmid in ids
        ],
    )

    prior = idea_mining._pubmed_prior_art(queries, source=source)

    assert [row["pmid"] for row in prior["results"]] == ["direct", "variation"]
    assert prior["retrieved_result_count"] == 8
    assert prior["result_count"] == 2
    assert prior["excluded_result_count"] == 6
    assert {row["pmid"] for row in prior["excluded_results"]} == {
        "asthma",
        "neonatal",
        "pediatric_trial",
        "case",
        "fresh",
        "perioperative",
    }
    assert prior["results"][0]["retrieval_screen"]["fit"] == ("direct_retrieval_fit")


def test_conversational_lactate_aki_screen_requires_both_concepts_and_icu() -> None:
    source = {
        "title": "ICU 患者早期乳酸下降轨迹与后续新发 AKI",
        "evidence_quote": "成人 ICU 乳酸清除速度和急性肾损伤",
    }

    direct = idea_mining._conversational_prior_art_screen(
        {
            "title": "Early lactate clearance and acute kidney injury in ICU patients",
            "abstract_excerpt": "An adult intensive care cohort study.",
            "matched_query_strata": ["candidate_topic"],
        },
        source,
    )
    missing_aki = idea_mining._conversational_prior_art_screen(
        {
            "title": "Metformin-associated lactic acidosis",
            "abstract_excerpt": "A review of lactate in critical care.",
            "matched_query_strata": ["clinical_landscape"],
        },
        source,
    )
    pediatric = idea_mining._conversational_prior_art_screen(
        {
            "title": "Lactate clearance and acute kidney injury in children",
            "abstract_excerpt": "A pediatric intensive care cohort.",
            "matched_query_strata": ["candidate_topic"],
        },
        source,
    )

    assert direct["fit"] == "direct_retrieval_fit"
    assert missing_aki["fit"] == "topic_mismatch"
    assert pediatric["fit"] == "population_mismatch"


def test_conversational_pair_scope_keeps_both_scientific_axes() -> None:
    scope = idea_mining._conversational_literature_scope(
        {
            "title": "ICU患者乳酸清除速度与后续AKI风险",
            "evidence_quote": "乳酸下降轨迹和急性肾损伤",
        }
    )

    assert '"lactate"[Title/Abstract]' in scope
    assert '"acute kidney injury"[Title/Abstract]' in scope


def test_sedation_awakening_scope_and_screen_require_both_axes() -> None:
    source = {
        "title": "ICU 患者镇静药减量后清醒快慢不同",
        "evidence_quote": "有的人很快清醒，有的人持续昏迷",
    }
    scope = idea_mining._conversational_literature_scope(source)
    direct = idea_mining._conversational_prior_art_screen(
        {
            "title": "Delayed awakening after sedation interruption in ICU patients",
            "abstract_excerpt": "An adult intensive care cohort study.",
            "matched_query_strata": ["candidate_topic"],
        },
        source,
    )
    delirium_only = idea_mining._conversational_prior_art_screen(
        {
            "title": "ICU delirium: a diagnostic challenge",
            "abstract_excerpt": "Delirium monitoring in critical care.",
            "matched_query_strata": ["clinical_landscape"],
        },
        source,
    )

    assert '"sedation interruption"[Title/Abstract]' in scope
    assert '"delayed awakening"[Title/Abstract]' in scope
    assert direct["fit"] == "direct_retrieval_fit"
    assert delirium_only["fit"] == "topic_mismatch"


def test_value_difference_does_not_become_practice_variation() -> None:
    source = {
        "title": "ICU 患者乳酸下降快慢差别很大，与后续 AKI 是否有关",
        "evidence_quote": "乳酸下降速度存在差异",
    }
    base_scope = idea_mining._conversational_literature_scope(source)

    assert (
        idea_mining._conversational_candidate_scope(source, base_scope=base_scope)
        == base_scope
    )
    assert (
        idea_mining._conversational_variation_scope(source, base_scope=base_scope)
        == base_scope
    )


def test_prior_art_rejects_untracked_or_mismatched_legacy_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path)
    empty_run = idea_mining._run_dir("empty-retained-dir")
    empty_run.mkdir(parents=True)

    with pytest.raises(idea_mining.IdeaMiningWebError) as exc_info:
        idea_mining.check_prior_art(
            {"run_id": "empty-retained-dir", "idea_id": "idea-a"}
        )
    assert exc_info.value.detail["error"] == "idea_run_not_found"

    legacy_run = idea_mining._run_dir("legacy-prior")
    legacy_run.mkdir(parents=True)
    (legacy_run / "prior_art_check.json").write_text(
        json.dumps(
            {
                "run_id": "legacy-prior",
                "idea_id": "idea-a",
                "prior_art": {
                    "status": "searched_no_hits",
                    "search_performed": True,
                    "queries_to_run": ["prespecified query"],
                    "results": [],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(idea_mining.IdeaMiningWebError) as exc_info:
        idea_mining.check_prior_art({"run_id": "legacy-prior", "idea_id": "idea-b"})
    assert exc_info.value.detail["error"] == "idea_not_found"


def test_url_metadata_failure_does_not_hide_error_behind_doi_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        idea_mining,
        "_resolve_public_http_target",
        lambda url: SimpleNamespace(url=str(url)),
    )

    def fail_html(*_args, **_kwargs):
        raise OSError("journal HTML unavailable")

    monkeypatch.setattr(idea_mining, "_open_public_url", fail_html)
    monkeypatch.setattr(
        idea_mining,
        "_fetch_doi_metadata",
        lambda _doi: {
            "status": "doi_fetch_failed",
            "network_calls": 1,
            "reason": "Crossref unavailable",
        },
    )

    result = idea_mining._fetch_url_metadata(
        "https://example.org/doi/10.1000/unavailable"
    )

    assert result["status"] == "fetch_failed"
    assert "journal HTML unavailable" in result["reason"]


def test_pubmed_esearch_requests_relevance_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            assert limit > 0
            return b'{"esearchresult":{"idlist":["26903338"]}}'

    def fake_urlopen(url: str, *, timeout: float) -> Response:
        assert timeout > 0
        requested.append(url)
        return Response()

    monkeypatch.setattr(idea_mining.request, "urlopen", fake_urlopen)

    assert idea_mining._pubmed_esearch("sepsis mortality", limit=5) == ["26903338"]
    assert "sort=relevance" in requested[0]


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
        lambda ids, *, focus_terms=(): [
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
                "design_excerpt": (
                    "Adult ICU patients were randomized to early vasopressor and "
                    "fluid-resuscitation strategies; mortality was the outcome."
                ),
                "publication_types": ["Randomized Controlled Trial"],
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
    assert payload["source_candidates"][0]["publication_types"] == [
        "Randomized Controlled Trial"
    ]
    assert "Adult ICU patients" in payload["source_candidates"][0]["design_excerpt"]
    idea = payload["idea_candidates"][0]["idea"]
    concept_ids = {row["concept_id"] for row in idea["mapped_concepts"]}
    assert {"vaso_ind", "death"} & concept_ids
    assert payload["suggested_payload"]["doi"] == "10.1056/NEJMoa2516225"
    assert "stay_id" not in str(payload)
    assert "subject_id" not in str(payload)
    assert "tableRows" not in str(payload)


def test_literature_discovery_round_robins_prespecified_query_strata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returned = iter(
        [
            ["100", "101", "102"],
            ["150", "151"],
            ["175", "176"],
            ["200", "201"],
            ["300", "301"],
            ["350", "351"],
        ]
    )
    fetched: list[str] = []

    monkeypatch.setattr(
        idea_mining,
        "_pubmed_esearch",
        lambda query, limit=5: next(returned),
    )

    def article_records(ids: list[str], *, focus_terms=()) -> list[dict[str, object]]:
        fetched.extend(ids)
        return [
            {
                "pmid": pmid,
                "title": f"Sepsis-3 mortality record {pmid}",
                "journal": "Critical Care",
                "year": 2025,
                "abstract_excerpt": "Sepsis-3 and ICU mortality were evaluated.",
                "evidence_sentence": "Sepsis-3 and ICU mortality were evaluated.",
            }
            for pmid in ids
        ]

    monkeypatch.setattr(idea_mining, "_pubmed_article_records", article_records)

    payload = idea_mining.discover_literature(
        {
            "topic": "Sepsis-3 mortality",
            "exposure_concept": "sep3_sofa1",
            "outcome_concept": "death",
            "allow_network": True,
            "limit": 3,
        }
    )

    assert fetched == ["100", "150", "175"]
    assert [row["id"] for row in payload["query_strata"]] == [
        "broad_icu",
        "concept_definition_or_validation",
        "direct_observational_comparator_all_years",
        "direct_observational_comparator_recent",
        "review_or_guideline",
        "critical_care_database",
    ]
    assert [row["retained_count"] for row in payload["query_strata"]] == [
        1,
        1,
        1,
        0,
        0,
        0,
    ]
    assert [row["matched_query_strata"] for row in payload["source_candidates"]] == [
        ["broad_icu"],
        ["concept_definition_or_validation"],
        ["direct_observational_comparator_all_years"],
    ]


def test_literature_discovery_runs_typed_direct_fallback_after_source_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def search(query: str, limit: int = 5) -> list[str]:
        calls.append(query)
        if len(calls) <= 6:
            return [f"10{len(calls)}"]
        return ["900", "901"]

    def article_records(ids: list[str], *, focus_terms=()) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for pmid in ids:
            if pmid == "900":
                rows.append(
                    {
                        "pmid": pmid,
                        "title": (
                            "Sepsis-3 prevalence and hospital mortality among "
                            "adult ICU stays"
                        ),
                        "year": 2024,
                        "design_excerpt": (
                            "An observational cohort of adult ICU stays estimated "
                            "Sepsis-3 prevalence and hospital mortality."
                        ),
                        "abstract_excerpt": (
                            "An observational cohort of adult ICU stays estimated "
                            "Sepsis-3 prevalence and hospital mortality."
                        ),
                        "publication_types": ["Observational Study"],
                    }
                )
            else:
                rows.append(
                    {
                        "pmid": pmid,
                        "title": "Vasopressin timing and mortality in septic shock",
                        "year": 2025,
                        "design_excerpt": (
                            "Adult ICU patients meeting Sepsis-3 criteria received "
                            "vasopressin; timing was associated with mortality."
                        ),
                        "abstract_excerpt": (
                            "Adult ICU patients meeting Sepsis-3 criteria received "
                            "vasopressin; timing was associated with mortality."
                        ),
                        "publication_types": ["Observational Study"],
                    }
                )
        return rows

    monkeypatch.setattr(idea_mining, "_pubmed_esearch", search)
    monkeypatch.setattr(idea_mining, "_pubmed_article_records", article_records)

    payload = idea_mining.discover_literature(
        {
            "topic": "adult ICU Sepsis-3 hospital mortality",
            "exposure_concept": "sep3_sofa1",
            "outcome_concept": "death",
            "outcome": "in-hospital mortality",
            "population": "adult ICU stays",
            "database": "miiv",
            "analysis_family": "descriptive_epidemiology",
            "allow_network": True,
            "limit": 3,
        }
    )

    assert len(calls) == 7
    assert '"Sepsis-3"[Title/Abstract]' in calls[-1]
    assert '"hospital mortality"[Title/Abstract]' in calls[-1]
    assert '"MIMIC-IV"[Title/Abstract]' in calls[-1]
    assert "epidemiolog*[Title/Abstract]" in calls[-1]
    assert payload["query_strata"][-1]["id"] == (
        "typed_direct_observational_comparator"
    )
    assert payload["source_candidates"][0]["pmid"] == "900"
    assert (
        payload["source_candidates"][0]["direct_comparator_screen"]["disposition"]
        == "include"
    )
    unrelated = next(
        row for row in payload["source_candidates"] if row["pmid"] != "900"
    )
    assert unrelated["direct_comparator_screen"]["disposition"] == "exclude"


def test_excluded_direct_fallback_does_not_erase_prespecified_strata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def search(query: str, limit: int = 5) -> list[str]:
        calls.append(query)
        return [f"10{len(calls)}"] if len(calls) <= 6 else ["900", "901"]

    def article_records(ids: list[str], *, focus_terms=()) -> list[dict[str, object]]:
        return [
            {
                "pmid": pmid,
                "title": "Vasopressin timing and mortality in septic shock",
                "year": 2025,
                "design_excerpt": (
                    "Adult ICU patients meeting Sepsis-3 criteria received "
                    "vasopressin; timing was associated with mortality."
                ),
                "publication_types": ["Observational Study"],
            }
            for pmid in ids
        ]

    monkeypatch.setattr(idea_mining, "_pubmed_esearch", search)
    monkeypatch.setattr(idea_mining, "_pubmed_article_records", article_records)

    payload = idea_mining.discover_literature(
        {
            "topic": "adult ICU Sepsis-3 hospital mortality",
            "exposure_concept": "sep3_sofa1",
            "outcome_concept": "death",
            "outcome": "in-hospital mortality",
            "population": "adult ICU stays",
            "database": "miiv",
            "analysis_family": "descriptive_epidemiology",
            "allow_network": True,
            "limit": 3,
        }
    )

    assert len(calls) == 7
    assert [row["pmid"] for row in payload["source_candidates"]] == [
        "101",
        "102",
        "103",
    ]
    assert payload["query_strata"][-1]["id"] == (
        "typed_direct_observational_comparator"
    )
    assert payload["query_strata"][-1]["retained_count"] == 0


def test_literature_discovery_keeps_foundational_and_recent_comparator_strata() -> None:
    queries = idea_mining._discovery_queries(
        "Sepsis-3 mortality",
        "",
        {
            "exposure_concept": "sep3_sofa1",
            "outcome_concept": "death",
            "population": "adult ICU stays",
        },
    )

    assert len(queries) == 6
    all_years = queries[2]
    recent = queries[3]
    assert "cohort[Title/Abstract]" in all_years
    assert "Date - Publication" not in all_years
    assert "cohort[Title/Abstract]" in recent
    assert "Date - Publication" in recent


def test_pubmed_metadata_retains_publication_type_and_design_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml = b"""<?xml version='1.0' encoding='UTF-8'?>
    <PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>12345</PMID><Article>
        <ArticleTitle>Sepsis-3 and mortality in adult ICU patients</ArticleTitle>
        <Abstract><AbstractText>We conducted a retrospective cohort study of adult ICU patients. Sepsis-3 was assessed at admission and hospital mortality was the outcome.</AbstractText></Abstract>
        <Journal><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue><Title>Critical Care</Title></Journal>
        <PublicationTypeList><PublicationType>Observational Study</PublicationType></PublicationTypeList>
      </Article></MedlineCitation><PubmedData><ArticleIdList>
        <ArticleId IdType='doi'>10.1000/example</ArticleId>
        <ArticleId IdType='pmc'>PMC1234567</ArticleId>
      </ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, limit: int = -1) -> bytes:
            return xml if limit < 0 else xml[:limit]

    monkeypatch.setattr(
        idea_mining.request,
        "urlopen",
        lambda url, timeout=0: Response(),
    )

    row = idea_mining._pubmed_article_records(["12345"])[0]
    assert row["publication_types"] == ["Observational Study"]
    assert row["pmcid"] == "PMC1234567"
    assert "retrospective cohort" in row["design_excerpt"]
    assert "hospital mortality" in row["design_excerpt"]


def test_pmc_full_text_review_keeps_only_bounded_section_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml = b"""<?xml version='1.0' encoding='UTF-8'?>
    <article><body>
      <sec><title>Methods</title><p>We enrolled adult ICU patients and aligned measurements to sedation discontinuation.</p></sec>
      <sec><title>Results</title><p>Delayed awakening occurred in a defined subset after discontinuation.</p></sec>
      <sec><title>Discussion</title><p>The findings require confirmation because residual sedation may confound recovery.</p></sec>
    </body></article>"""

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, limit: int = -1) -> bytes:
            return xml if limit < 0 else xml[:limit]

    monkeypatch.setattr(
        idea_mining.request,
        "urlopen",
        lambda url, timeout=0: Response(),
    )

    review = idea_mining._pmc_full_text_evidence("PMC1234567")

    assert review["status"] == "reviewed"
    assert [row["section"] for row in review["evidence_spans"]] == [
        "methods",
        "results",
        "discussion",
    ]
    assert review["full_text_stored"] is False
    assert len(json.dumps(review)) < 4_000


@pytest.mark.parametrize(
    ("publication_types", "expected"),
    [
        (["Observational Study"], "original_research"),
        (["Systematic Review"], "systematic_review"),
        (["Review"], "narrative_review"),
        (["Practice Guideline"], "guideline_consensus"),
        (["Editorial"], "editorial_commentary"),
        (["Clinical Trial Protocol"], "protocol"),
    ],
)
def test_literature_article_kind_changes_interpretation_by_publication_type(
    publication_types: list[str], expected: str
) -> None:
    assert idea_mining._literature_article_kind(publication_types) == expected


def test_literature_article_kind_uses_bounded_design_text_when_pubmed_is_generic() -> (
    None
):
    assert (
        idea_mining._literature_article_kind(
            ["Journal Article"],
            title="Delayed awakening after sedation interruption",
            abstract="We conducted a retrospective cohort study in adult ICU patients.",
        )
        == "original_research"
    )
