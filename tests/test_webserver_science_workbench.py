from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from easyicu.webserver import settings as settings_store
from easyicu.webserver.app import app
from easyicu.webserver import science_workbench


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _minimal_run(tmp_path: Path) -> Path:
    run = tmp_path / "projects" / "sepsis" / "run_demo"
    run.mkdir(parents=True)
    _write_json(
        run / "run_context.json",
        {
            "run_id": "run_demo",
            "study_id": "sepsis",
            "mode": "analysis",
            "question": "Does lactate trajectory add prognostic information?",
            "source": {"label": "Fig 2 canonical9 · E1", "database": "benchmark_import"},
            "summary": {"stays": 12, "modules": 4},
            "local_first": {"uploads": 0, "tokens": 0},
        },
    )
    _write_json(
        run / "cohort_summary.json",
        {"summary": {"stays": 12, "modules": 4}, "cohort": {"survived": 8, "deceased": 4}},
    )
    gate = {
        "status": "analysis_only",
        "reportable": False,
        "draft_unlocked": False,
        "reason": "preflight_complete_human_signoff_required",
        "checks": [
            {"id": "source_valid", "label": "Export source valid", "passed": True},
            {"id": "denominator_resolved", "label": "Cohort denominator resolved", "passed": True},
            {"id": "no_patient_rows_persisted", "label": "No patient rows persisted", "passed": True},
            {"id": "human_signoff", "label": "Human sign-off", "passed": False},
        ],
    }
    _write_json(run / "quality_gate.json", {"quality": [], "gate": gate})
    _write_json(
        run / "manuscript_draft.json",
        {
            "run_id": "run_demo",
            "status": "locked_until_human_signoff",
            "claims": [
                {
                    "claim_id": "claim_001",
                    "text": "The active export has 12 stays.",
                    "evidence_ids": ["cohort_summary.json"],
                }
            ],
            "sentences": [],
        },
    )
    _write_json(
        run / "evidence_ledger.json",
        {
            "run_id": "run_demo",
            "run_type": "canonical9_import",
            "status": "analysis_only",
            "artifacts": [],
            "provider": {},
            "strict_evidence_audit": None,
            "numeric_evidence_audit": None,
            "privacy": {
                "patient_rows_persisted": False,
                "artifact_scan": {"passed": True, "scanned_artifacts": 4},
            },
        },
    )
    return run


def _isolate_idea_mining(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        science_workbench.idea_mining_web, "_RUN_ROOT", tmp_path / "idea_runs_empty"
    )
    monkeypatch.setattr(
        science_workbench.idea_mining_web, "_HISTORY_PATH", tmp_path / "history.json"
    )
    monkeypatch.setattr(
        science_workbench.idea_mining_web,
        "_AGENT_PROJECTS_ROOT",
        tmp_path / "agent_projects_empty",
    )
    monkeypatch.setattr(
        science_workbench.idea_mining_web,
        "_AGENT_PROJECTS_PATH",
        tmp_path / "agent_projects.json",
    )


def test_science_workbench_returns_four_claude_science_inspired_objects(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_idea_mining(tmp_path, monkeypatch)
    run = _minimal_run(tmp_path)

    response = TestClient(app).post(
        "/api/agent-runs/science-workbench",
        json={"project_dir": str(run)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "easyicu.science_workbench/1"
    assert payload["workflow_scope"]["id"] == "canonical_benchmark"
    assert payload["workflow_scope"]["supports_fig5"] is False
    assert payload["artifact_history"]["items"]
    assert payload["run_summary"]["denominator"] == 12
    assert payload["run_summary"]["status"] == "review_locked"
    assert payload["run_summary"]["workflow_scope"]["id"] == "canonical_benchmark"
    assert payload["fig5_checklist"]["items"]
    assert payload["fig5_checklist"]["candidate_for_fig5"] is False
    assert payload["fig5_checklist"]["title"] == "Evidence readiness checklist"
    assert payload["fig5_checklist"]["applicable_count"] >= 5
    assert payload["feature_alignment"]["items"]
    assert payload["reviewer_gate"]["checks"]
    assert payload["reusable_protocols"]
    protocol_ids = {row["id"] for row in payload["reusable_protocols"]}
    assert {"nature-figure", "nature-writing"} <= protocol_ids
    assert payload["native_renderers"]
    assert payload["privacy"]["patient_rows_returned"] is False
    assert payload["reviewer_gate"]["reportable"] is False
    assert payload["reviewer_gate"]["draft_unlocked"] is False
    assert "Code" in payload["artifact_history"]["tabs"]
    assert "Review" in payload["artifact_history"]["tabs"]
    assert {row["id"] for row in payload["reviewer_gate"]["checks"]} >= {
        "citation_fidelity",
        "numeric_traceability",
        "figure_source_consistency",
        "denominator_reporting",
        "privacy_scan",
        "conclusion_safety",
    }
    assert {row["id"] for row in payload["fig5_checklist"]["items"]} >= {
        "source_signal",
        "outcome_blind_feasibility",
        "artifact_provenance",
        "reviewer_gate",
        "native_renderers",
        "figure_source_data",
        "workflow_scope",
    }
    assert {row["id"] for row in payload["feature_alignment"]["items"]} >= {
        "artifact_history",
        "reviewer_gate",
        "reusable_protocols",
        "prior_art_gate",
        "outcome_blind_feasibility",
        "icu_native_renderers",
        "workflow_scope",
    }
    assert payload["discovery_pipeline"]["status"] == "waiting_for_idea_run"


def test_science_workbench_empty_state_is_local_and_non_reportable(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_idea_mining(tmp_path, monkeypatch)
    response = TestClient(app).post("/api/agent-runs/science-workbench", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["project_dir"] is None
    assert payload["workflow_scope"]["id"] == "empty_state"
    assert payload["artifact_history"]["items"] == []
    assert payload["run_summary"]["status"] == "review_locked"
    assert payload["run_summary"]["local_only"]["patient_rows_returned"] is False
    assert payload["fig5_checklist"]["candidate_for_fig5"] is False
    assert payload["fig5_checklist"]["passed_count"] == 0
    assert payload["discovery_pipeline"]["status"] == "waiting_for_idea_run"
    assert payload["reviewer_gate"]["reportable"] is False
    assert payload["privacy"]["external_image_loaded_by_api"] is False


def test_science_workbench_capability_policy_controls_protocol_registry(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_idea_mining(tmp_path, monkeypatch)
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: {**settings_store.DEFAULTS, "science_skills_enabled": False},
    )

    response = TestClient(app).post("/api/agent-runs/science-workbench", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["capability_policy"]["settings"]["science_skills_enabled"] is False
    assert payload["reusable_protocols"] == []
    alignment = {row["id"]: row for row in payload["feature_alignment"]["items"]}
    assert alignment["reusable_protocols"]["status"] == "unavailable"
    assert "Settings 已关闭 Skills" in alignment["reusable_protocols"]["evidence"]


def test_science_workbench_filters_individual_publication_skills(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_idea_mining(tmp_path, monkeypatch)
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: {
            **settings_store.DEFAULTS,
            "nature_figure_skill_enabled": True,
            "nature_writing_skill_enabled": False,
        },
    )

    payload = TestClient(app).post(
        "/api/agent-runs/science-workbench", json={}
    ).json()
    protocol_ids = {row["id"] for row in payload["reusable_protocols"]}

    assert "nature-figure" in protocol_ids
    assert "nature-writing" not in protocol_ids


def test_science_workbench_reads_latest_idea_mining_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    idea_root = tmp_path / "idea_runs"
    history_path = tmp_path / "history.json"
    projects_root = tmp_path / "agent_projects"
    projects_path = tmp_path / "agent_projects.json"
    monkeypatch.setattr(science_workbench.idea_mining_web, "_RUN_ROOT", idea_root)
    monkeypatch.setattr(science_workbench.idea_mining_web, "_HISTORY_PATH", history_path)
    monkeypatch.setattr(
        science_workbench.idea_mining_web, "_AGENT_PROJECTS_ROOT", projects_root
    )
    monkeypatch.setattr(
        science_workbench.idea_mining_web, "_AGENT_PROJECTS_PATH", projects_path
    )
    run_id = "ideafixture"
    run_dir = idea_root / run_id
    run_dir.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            [
                {
                    "run_id": run_id,
                    "created_at": "2026-07-02T00:00:00Z",
                    "title": "Vasopressor-fluid resuscitation strategy",
                    "run_dir": str(run_dir),
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "idea_mining_run.json",
        {
            "ok": True,
            "run_id": run_id,
            "created_at": "2026-07-02T00:00:00Z",
            "source_evidence": [
                {
                    "title": "Vasopressors or Fluids in Early Septic Shock",
                    "journal": "New England Journal of Medicine",
                    "year": 2026,
                    "evidence_quote": "Earlier vasopressor use may be measurable.",
                    "source_text_stored": False,
                }
            ],
            "idea_ledger": [
                {
                    "idea_id": "idea_vaso",
                    "idea_title": "Vasopressor-fluid resuscitation strategy",
                    "go_no_go": "hold",
                    "go_no_go_reason": "Needs re-extraction",
                    "mapped_concepts": [
                        {"concept_id": "vaso_ind", "tier": "T1_reextract"},
                        {"concept_id": "death", "tier": "executable"},
                    ],
                    "feasibility": {
                        "tier": "T1_reextract",
                        "label": "Needs re-extraction or extra modules",
                    },
                    "prior_art": {
                        "status": "not_checked_external_search_required",
                        "search_performed": False,
                    },
                }
            ],
            "pre_experiment": {
                "status": "partial",
                "payload_scope": "aggregate_pre_experiment_no_row_payload",
                "reportable": False,
                "cohort": {"entities": 94458},
                "feature_statistics": [{"concept_id": "death"}],
            },
            "handoff_plan": {"analysis_plan": [{"phase": "Question"}]},
            "privacy": {
                "patient_rows_returned": False,
                "network_calls": 0,
                "external_llm_calls": 0,
            },
        },
    )
    _write_json(
        run_dir / "prior_art_check.json",
        {
            "ok": True,
            "run_id": run_id,
            "idea_id": "idea_vaso",
            "prior_art": {
                "status": "blocked_network_opt_in_required",
                "search_performed": False,
                "queries_to_run": ["vasopressor fluid ICU"],
                "results": [],
                "reason": "No request was made.",
            },
            "privacy": {"network_calls": 0, "external_llm_calls": 0},
        },
    )

    response = TestClient(app).post("/api/agent-runs/science-workbench", json={})

    assert response.status_code == 200
    body = response.json()
    pipeline = body["discovery_pipeline"]
    assert pipeline["latest_run_id"] == run_id
    assert pipeline["title"] == "Vasopressor-fluid resuscitation strategy"
    assert pipeline["status"] == "needs_review"
    assert pipeline["fig5_candidate_ready"] is False
    assert pipeline["source_data_review_ready"] is False
    assert pipeline["mapped_concept_count"] == 2
    assert pipeline["cohort_entities"] == 94458
    stages = {row["id"]: row for row in pipeline["stages"]}
    assert stages["source_signal"]["status"] == "passed"
    assert stages["prior_art"]["evidence"] == "blocked_network_opt_in_required"
    assert stages["outcome_blind_feasibility"]["evidence"] == "partial · n=94458"
    assert stages["go_no_go"]["status"] == "needs_review"
    alignment = {row["id"]: row for row in body["feature_alignment"]["items"]}
    assert alignment["prior_art_gate"]["status"] == "needs_review"
    assert alignment["prior_art_gate"]["status_label"] == "needs review / 待审阅"
    assert alignment["outcome_blind_feasibility"]["status"] == "needs_review"
    assert pipeline["privacy"]["patient_rows_returned"] is False
    assert pipeline["privacy"]["external_llm_calls"] == 0


def test_discovery_pipeline_merges_prior_art_privacy_and_blocks_hold_decision(
    tmp_path: Path, monkeypatch
) -> None:
    idea_root = tmp_path / "idea_runs"
    history_path = tmp_path / "history.json"
    projects_root = tmp_path / "agent_projects"
    projects_path = tmp_path / "agent_projects.json"
    monkeypatch.setattr(science_workbench.idea_mining_web, "_RUN_ROOT", idea_root)
    monkeypatch.setattr(science_workbench.idea_mining_web, "_HISTORY_PATH", history_path)
    monkeypatch.setattr(
        science_workbench.idea_mining_web, "_AGENT_PROJECTS_ROOT", projects_root
    )
    monkeypatch.setattr(
        science_workbench.idea_mining_web, "_AGENT_PROJECTS_PATH", projects_path
    )
    run_id = "hold_after_prior_art"
    run_dir = idea_root / run_id
    run_dir.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            [
                {
                    "run_id": run_id,
                    "created_at": "2026-07-02T01:00:00Z",
                    "title": "Hold candidate after prior-art search",
                    "run_dir": str(run_dir),
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "idea_mining_run.json",
        {
            "ok": True,
            "run_id": run_id,
            "source_evidence": [
                {
                    "title": "ICU fluid strategy review",
                    "journal": "Critical Care",
                    "year": 2026,
                    "evidence_quote": "Fluid timing remains uncertain.",
                    "source_text_stored": False,
                }
            ],
            "idea_ledger": [
                {
                    "idea_id": "idea_hold",
                    "idea_title": "Hold candidate after prior-art search",
                    "go_no_go": "hold",
                    "go_no_go_reason": "Prior-art changed the question scope.",
                    "mapped_concepts": [
                        {"concept_id": "fluid_balance", "tier": "executable"},
                        {"concept_id": "death", "tier": "executable"},
                    ],
                }
            ],
            "pre_experiment": {
                "status": "ready",
                "payload_scope": "aggregate_pre_experiment_no_row_payload",
                "reportable": False,
                "cohort": {"entities": 1200},
                "feature_statistics": [
                    {"concept_id": "fluid_balance"},
                    {"concept_id": "death"},
                ],
            },
            "privacy": {
                "patient_rows_returned": False,
                "network_calls": 1,
                "external_llm_calls": 0,
            },
        },
    )
    _write_json(
        run_dir / "prior_art_check.json",
        {
            "ok": True,
            "run_id": run_id,
            "idea_id": "idea_hold",
            "prior_art": {
                "status": "search_complete",
                "search_performed": True,
                "network_calls": 3,
                "results": [{"title": "Prior ICU fluid study"}],
                "reason": "Prior studies partly overlap.",
            },
            "privacy": {
                "network_calls": 3,
                "external_llm_calls": 0,
                "patient_rows_returned": False,
            },
        },
    )
    _write_json(
        run_dir / "idea_plan.json",
        {"plan": {"plan_status": "draft_plan_requires_user_review"}},
    )
    _write_json(
        run_dir / "idea_handoff.json",
        {
            "idea_id": "idea_hold",
            "candidate_topic": "Hold candidate after prior-art search",
            "handoff_plan": {"analysis_plan": [{"phase": "Question"}]},
        },
    )

    response = TestClient(app).post("/api/agent-runs/science-workbench", json={})

    assert response.status_code == 200
    pipeline = response.json()["discovery_pipeline"]
    assert pipeline["status"] == "needs_review"
    assert pipeline["fig5_candidate_ready"] is False
    assert pipeline["source_data_review_ready"] is False
    assert pipeline["privacy"]["network_calls"] == 4
    stages = {row["id"]: row for row in pipeline["stages"]}
    assert stages["prior_art"]["status"] == "passed"
    assert stages["outcome_blind_feasibility"]["status"] == "passed"
    assert stages["plan_replan"]["status"] == "passed"
    assert stages["agent_handoff"]["status"] == "passed"
    assert stages["go_no_go"]["status"] == "needs_review"


def test_discovery_pipeline_reports_unavailable_idea_storage(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_idea_mining(tmp_path, monkeypatch)

    def broken_list_runs(_body):
        raise RuntimeError("local path /private/idea_runs is unreadable")

    monkeypatch.setattr(science_workbench.idea_mining_web, "list_runs", broken_list_runs)

    response = TestClient(app).post("/api/agent-runs/science-workbench", json={})

    assert response.status_code == 200
    pipeline = response.json()["discovery_pipeline"]
    assert pipeline["status"] == "idea_mining_unavailable"
    assert pipeline["fig5_candidate_ready"] is False
    assert pipeline["privacy"]["network_calls"] == 0
    assert "/private/idea_runs" not in json.dumps(pipeline)
