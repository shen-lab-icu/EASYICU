"""Discovery handoff source, identity, and version contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.research_agent.discovery.discovery_handoff import (
    DiscoveryHandoffPacket,
    build_handoff_from_row,
)
from easyicu.webserver.app import app
from easyicu.webserver.ideas import mining as idea_mining
from easyicu.webserver.ideas.handoff import (
    build_web_handoff_packet,
    map_web_ledger_row,
)


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
    assert frozen["canonical_handoff_path"] == f"handoffs/{canonical.handoff_sha256}.json"
    assert len(frozen["canonical_handoff_sha256"]) == 64
    assert (idea_mining._run_dir(run["run_id"]) / frozen["canonical_handoff_path"]).is_file()


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
    (tmp_path / "idea_mining_run.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.web_idea_mining/1",
                "idea_ledger": [idea],
                "source_evidence": [source],
                "pre_experiment": pre,
            }
        )
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
        candidate_transformation=adapter_packet.candidate_source.transformation,
    )
    adapter_payload = adapter_packet.model_dump(mode="json")
    core_payload = core_packet.model_dump(mode="json")
    adapter_payload.pop("created_at")
    adapter_payload.pop("handoff_sha256")
    core_payload.pop("handoff_sha256")
    core_payload.pop("created_at")
    assert adapter_payload == core_payload


def test_web_handoff_without_database_stays_unspecified(tmp_path: Path) -> None:
    idea = {
        "idea_id": "idea_unknown_db",
        "idea_title": "Unknown-database ICU candidate",
        "go_no_go": "hold",
        "go_no_go_reason": "No active export is selected.",
        "outcome": "In-hospital mortality",
        "mapped_concepts": [],
    }
    (tmp_path / "idea_mining_run.json").write_text(
        json.dumps(
            {"schema_version": "easyicu.web_idea_mining/1", "idea_ledger": [idea]}
        )
    )
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
    handoff = idea_mining.create_handoff({"run_id": run["run_id"], "idea_id": idea["idea_id"]})
    run_dir = idea_mining._run_dir(run["run_id"])
    if tamper_target in {"artifact", "artifact_with_replan"}:
        path = run_dir / handoff["canonical_handoff_path"]
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
    (run_dir / handoff["canonical_handoff_path"]).unlink()

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
